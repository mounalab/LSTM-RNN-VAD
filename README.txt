*** Project: Voice Activity Detection for Voice Controlled Home Automation 
*** By: Mouna LABIADH


- dataset_utils.py
	Dataset related utilities: One-hot encoding, wav file normalisation, TRS to CSV conversion, JSON to CSV conversion, Youtube wav download for the AudioSet Google corpus, Liblinear library data transformations

- metrics_utils.py
	(NOT FINALISED) Metrics' related utilities for the baseline OpenSAD framework proposed by NIST as NIST Open Speech-Activity-Detection Evaluation 

- feature_extractor.py
	Feature extraction class to extract MFCC, deltas, double deltas, RSE

- VAD_model.py
	LSTM-RNN tensorflow learning models variants
	- v1: simple model with Softmax output function 
	- v2: Sigmoid output function + Batch normalisation technique + Thresholding + Pk metric monitoring
	- v3: Softmax output function + Batch normalisation technique + Pk metric monitoring
	- v4: Linear output function + Batch normalisation technique + Moving Average temporal smoothing + Pk metric monitoring

- multilayer_perceptron.py
	MLP baseline method

- post_process.py
	(NOT FINALISED) Post processing of the per-frame VAD classification partly inspired of the solution proposed in the article:
	Pollák, P., & Rajnoha, J. (2009, June). Long recording segmentation based on simple power voice activity detection with adaptive threshold and post-processing. In Proc. of SPECOM (pp. 55-60).

- __main__.py
	The program's main entry point

- /checkpoint
	Tensorflow checkpoint directory for saving and restoring learning models

- /parameter
	LSTM-RNN learning model hyper-parameters, training parameters, and log/checkpoint directories names

- /notebook
	Jupyter notebooks to test initial VAD prototypes



