import numpy as np
import pandas as pd
import torch
import math
import seaborn as sns

from sklearn.metrics import accuracy_score
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.nn import Conv2d, MaxPool2d, Linear, ReLU, Softmax, Module, CrossEntropyLoss
from torch.optim import SGD
from torch.nn.init import kaiming_uniform_, xavier_uniform_
from torch.utils.data import Dataset, DataLoader
pi = math.pi
import time

## Defining dataset
class UrbanSoundDataset(Dataset):
    def __init__(self, csv_path, data_path, which_folds, wind, step, load_phase):
        """
        Args:
            csv_path (string): path to file of labels
            data_path (string): directory containing audio files
            which_folds (set of int): folds to load
            wind:
            step:
        """
        self.data_path = data_path
        csv = pd.read_csv(csv_path)
        self.max_length = max(csv.loc[:,"end"]-csv.loc[:,"start"]) # max_length in full set, not necessarily chosen folds
        self.meta = csv.loc[list(map(lambda x: x in which_folds,csv["fold"])),:]
        self.meta.reset_index(drop=True,inplace=True)
        self.n_classes = max(self.meta.loc[:,"classID"]) + 1
        self.class_names = ["" for x in range(self.n_classes)]
        for i in range(self.n_classes):
            self.class_names[i] = self.meta[self.meta["classID"]==i].head(1)["class"].item()
        self.wind = wind
        self.step = step
        self.FS = 22050 # expected samp freq from librosa load
        self.max_timebins = math.floor(self.max_length*self.FS/self.step) - round(self.wind/self.step)+1
        self.load_phase = load_phase


    
    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        """
        Returns: TODO: actually returns PSD and phdiff now
            sound (np.ndarray): time series of sound with FS = 22050
            label (int): label number (0-9)
            fold (int) fold number [1-10]
        """
        fold = self.meta.loc[idx,"fold"]
        label = self.meta.loc[idx,"classID"]
        filename = self.meta.loc[idx,"slice_file_name"][0:-4]
        psds = np.load(self.data_path + "/fold" + str(fold) + "/" + filename + "_psd.npy")

        need_bins = self.max_timebins - np.size(psds,1)
        # half rounded down padded at start, half rounded down at end
        if np.size(psds,1)>341:
            print("stop")
        psds_pad = centerPad2D(psds,need_bins)

        if self.load_phase:
            phdiff = np.load(self.data_path + "/fold" + str(fold) + "/" + filename + "_phdiff.npy")
            phdiff_pad = centerPad2D(phdiff,need_bins+1) # one more bin needed than psd
            channels = np.stack((psds_pad,phdiff_pad)) # psd and phase channels
        else:
            channels = np.expand_dims(psds_pad,0) # single channel, no phase
        return channels, label

## helper functions for data transformation
def centerPad2D(signal, nadd, val=0):
    """Pads second dimension of 2D signal with vals (default=0) s.t. signal is centered (odd padding
    added to end)
    Args:
        signal (2D np array): signal to add rows to
        nadd (int): total number of rows to add
        val: value to pad with (default=0)
    """
    return np.concatenate((val+np.zeros([np.size(signal,0),math.floor(nadd/2)]), signal, val+np.zeros([np.size(signal,0),math.ceil(nadd/2)])), axis=1)


## define network
# modified from https://machinelearningmastery.com/pytorch-tutorial-develop-deep-learning-models/
class CNN(Module):
    # define model elements
    def __init__(self, n_channels, n_t, n_f, n_classes):
        super(CNN, self).__init__()
        kern_conv = 3 # square
        kern_pool = 2
        # input to first hidden layer
        n_out1 = 32
        self.hidden1 = Conv2d(n_channels, n_out1, (kern_conv,kern_conv))
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.act1 = ReLU()
        # first pooling layer
        self.pool1 = MaxPool2d((kern_pool,kern_pool), stride=(kern_pool,kern_pool))
        # second hidden layer
        n_out2 = 32
        self.hidden2 = Conv2d(n_out1, n_out2, (kern_conv,kern_conv))
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act2 = ReLU()
        # second pooling layer
        self.pool2 = MaxPool2d((kern_pool,kern_pool), stride=(kern_pool,kern_pool))
        # fully connected layer
        # compute how big network should be befor elinear layer. floor assumes pool crops, not pads
        size_at_3 = n_out2 * CNN.contract_size(n_t, kern_conv, kern_pool, 2) * CNN.contract_size(n_f, kern_conv, kern_pool, 2)
        n_out3 = 100
        self.hidden3 = Linear(size_at_3, n_out3)
        kaiming_uniform_(self.hidden3.weight, nonlinearity='relu')
        self.act3 = ReLU()
        # output layer
        self.hidden4 = Linear(n_out3, n_classes)
        xavier_uniform_(self.hidden4.weight)
        self.act4 = Softmax(dim=1) # TODO: what's the shape here? why dim 1?

    @staticmethod
    def contract_size(x,kern_conv,kern_pool,n_lay):
        # recursive helper to find length of one dimension of network after several layers
        # assumes all layers have same conv and pool kernel
        if n_lay <= 0:
            return x
        else:
            return math.floor((CNN.contract_size(x,kern_conv,kern_pool,n_lay-1) - kern_conv+1)/kern_pool)
    
    # forward propagate input
    def forward(self, X):
        # input to first hidden layer
        X = self.hidden1(X)
        X = self.act1(X)
        X = self.pool1(X)
        # second hidden layer
        X = self.hidden2(X)
        X = self.act2(X)
        X = self.pool2(X)
        # flatten
        X = X.view(X.size(0),-1)
        # third hidden layer
        X = self.hidden3(X)
        X = self.act3(X)
        # output layer
        X = self.hidden4(X)
        X = self.act4(X)
        return X

# training
def train_model(train_dl, model, n_epochs):
    # define the optimization
    criterion = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    # enumerate epochs
    for epoch in range(n_epochs):
        print("Training epoch #" + str(epoch+1))
        # enumerate mini batches
        for i, (psds, label) in enumerate(train_dl):
            optimizer.zero_grad() # clear the gradients
            # construct inputs
            # if do_phase:
            #     print("phase not implemented")
            #     #TODO: concatenate phase tensors
            # else:
            #     inputs = psds # size: batch_size x freqs x times
            inputs = psds # size: batch_size x freqs x times

            yhat = model(inputs.float())
            loss = criterion(yhat, label)
            loss.backward() # credit assignment
            # update model weights
            optimizer.step()

# evaluate
def evaluate_model(test_dl, model):
    predictions, actuals = list(), list()
    for i, (psds, label) in enumerate(train_dl):
        # construct inputs
        # if do_phase:
        #     print("phase not implemented")
        #     #TODO: concatenate phase tensors
        # else:
        #     inputs = psds # size: batch_size x freqs x times
        inputs = psds # size: batch_size x freqs x times
        # evaluate the model on the test set
        yhat = model(inputs.float())
        # retrieve numpy array
        yhat = yhat.detach().numpy()
        actual = label.numpy()
        # convert to class labels
        yhat = np.argmax(yhat, axis=1)
        # reshape for stacking
        actual = actual.reshape((len(actual), 1))
        yhat = yhat.reshape((len(yhat), 1))
        # store
        predictions.append(yhat)
        actuals.append(actual)
    predictions, actuals = np.vstack(predictions), np.vstack(actuals)
    # calculate accuracy
    acc = accuracy_score(actuals, predictions)
    return acc


## main code
# csv_path = "../input/urbansound8k-meta/UrbanSound8K.csv"
csv_path = "data/UrbanSound8K.csv"
# need already processed data (process_data.py) with correct wind and step
windexp = 10
stepexp = 8
# data_dir = "../input/urbansound8k-processed/processed_wind" + str(windexp) + "_step" + str(stepexp)
data_dir = "processed_wind" + str(windexp) + "_step" + str(stepexp)

wind = 2**windexp
step = 2**stepexp
n_epochs = 10

load_phase = 0
train_set = UrbanSoundDataset(csv_path,data_dir, {x for x in range(9)}, wind, step, load_phase)
test_set = UrbanSoundDataset(csv_path,data_dir, {10}, wind, step, load_phase)

train_dl = DataLoader(train_set,batch_size=64,shuffle=True)
test_dl = DataLoader(test_set,batch_size=64,shuffle=False)

# define the network
model_psd = CNN(load_phase+1, train_set.max_timebins, wind/2+1, train_set.n_classes) # 1 input channel for PSD only
model_psd = model_psd.float()

# train the model
train_start = time.time()
train_model(train_dl, model_psd, n_epochs)
print("Training time: " + str(time.time()-train_start) + " seconds")

# evaluate the model
test_start = time.time()
acc = evaluate_model(test_dl, model_psd)
print("Evaluation time: " + str(time.time()-test_start) + " seconds")
print('Accuracy: %.3f' % acc)