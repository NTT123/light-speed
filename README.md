# Light Speed ⚡

Light Speed ⚡ is an open-source text-to-speech model based on VITS, with some modifications:
- utilizes phoneme duration's ground truth, obtained from an external forced aligner (such as Montreal Forced Aligner), to upsample phoneme information to frame-level information. The result is a more robust model, with a slight trade-off in speech quality.
- employs dilated convolution to expand the Wavenet Flow module's receptive field, enhancing its ability to capture long-range interactions.

<!-- ![network diagram](net.svg) -->

## FAQ

Q: How do I create training data?  
A: See the `./prepare_ljs_tfdata.ipynb` notebook for instructions on preparing the training data.

Q: How can I train the model with 1 GPU?  
A: Run: `python train.py`

Q: How can I train the model with 4 GPUs?  
A: Run: `torchrun --standalone --nnodes=1 --nproc-per-node=4 train.py`

Q: How can I train a model to predict phoneme durations?  
A: See the `./train_duration_model.ipynb` notebook.

Q: How can I generate speech with a trained model?  
A: See the `./inference.ipynb` notebook.


## Credits

- Most of the code in this repository is based on the [VITS official repository](https://github.com/jaywalnut310/vits).
