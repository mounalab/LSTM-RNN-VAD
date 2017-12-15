# LSTM-RNN Voice Activity Detection


\- dataset_utils.py <br />
	Dataset related utilities: One-hot encoding, wav file normalisation, TRS to CSV conversion, JSON to CSV conversion, Youtube wav download for the AudioSet Google corpus, Liblinear library data transformations

\- metrics_utils.py <br />
	(NOT FINALISED) Metrics' related utilities for the baseline OpenSAD framework proposed by NIST as NIST Open Speech-Activity-Detection Evaluation 

\- feature_extractor.py <br />
	Feature extraction class to extract MFCC, deltas, double deltas, RSE

\- VAD_model.py <br />
	LSTM-RNN tensorflow learning model

\- \__main__.py <br />
	The program's main entry point

\- /checkpoint <br />
	Tensorflow checkpoint directory for saving and restoring learning models

\- /parameter <br />
	LSTM-RNN learning model hyper-parameters, training parameters, and log/checkpoint directories names

\- /notebook <br />
	Jupyter notebooks to test initial VAD prototypes
