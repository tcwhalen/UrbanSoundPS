UrbansSoundPS
=============

Timothy Whalen, last edited December 2020, tim@timcwhalen.com

A convolutional neural network for classifying cityscape sounds, using the UrbanSound8K dataset:

J. Salamon, C. Jacoby and J. P. Bello, "A Dataset and Taxonomy for Urban Sound Research", 22nd ACM International Conference on Multimedia, Orlando USA, Nov. 2014.
https://urbansounddataset.weebly.com/urbansound8k.html

Specifically, an exploration to see if augmenting the input spectrogram to include a "phase shift-gram" can improve classification accuracy (PS = phase shift). More information on pahse shift can be found in my (otherwise completely unrelated) paper:
Whalen T. C., Willard, A. M., Rubin, J. E., & Gittis, A. H. (2020). Delta oscillations are a robust biomarker of dopamine depletion severity and motor dysfunction in awake mice. Journal of Neurophysiology, 124(2), 312-329. https://journals.physiology.org/doi/abs/10.1152/jn.00158.2020

Very much a work in progress and hardly meant for use by others yet! (as you will see below)

To run
=============

First set up the virtual environment by running these in the project directory:

python3 -m venv venv/
source venv/bin/activate
pip install -r requirements.txt

(currently, the requirements in the .txt are overkill, as some are no longer needed)

Download the UrbanSound8K dataset from the link above. Then, run process_data.py to run pre-processing so the data does not need to be processed on every training epoch. For now, you'll need to adjust arguments and the load and save locations at the top of the script (hence, work in progress)

The urbansound_ph file trains and tests the model. Make sure the csv_path matches the location of the downlaoded UrbanSound8K.csv, and the data_dir and parameters match what you set in process_data so you'll load the correct processed files. Then, run urbansound_ph.
