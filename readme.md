Timothy Whalen, last edited December 2020
tim@timcwhalen.com

A convolutional neural network for classifying cityscape sounds, using the UrbanSound8K dataset
J. Salamon, C. Jacoby and J. P. Bello, "A Dataset and Taxonomy for Urban Sound Research", 22nd ACM International Conference on Multimedia, Orlando USA, Nov. 2014.
https://urbansounddataset.weebly.com/urbansound8k.html

Specifically, an experiment to see if augmenting the input spectrogram to include a "phase shift-gram" can improve classification accuracy. More information on pahse shift can be found in my (otherwise completely unrelated) paper:
Whalen T. C., Willard, A. M., Rubin, J. E., & Gittis, A. H. (2020). Delta oscillations are a robust biomarker of dopamine depletion severity and motor dysfunction in awake mice. Journal of Neurophysiology, 124(2), 312-329. https://journals.physiology.org/doi/abs/10.1152/jn.00158.2020

Very much a work in progress!

To run, first set up the virtual environment by running these in the project directory:

python3 -m venv venv/
source venv/bin/activate
pip install -r requirements.txt

(currently, the requirements in the .txt are overkill, as some are no longer needed)

urbsound_ph.py is the main function to run. Training on a GPU is highly recommended.
