from __future__ import unicode_literals

import glob
import os
import csv
import random
import numpy as np
import librosa

from dataset_utils import one_hot_encode


class FeatureExtractor(object):

    def __init__(self, 
        frame_size,
        frame_step,
        n_mfcc,
        sampling_rate,
        audio_ext,
        label_ext,
        **kwargs):

        self.n_mfcc = n_mfcc # number of MFCC features to extract
        self.sampling_rate = sampling_rate
        self.frame_size = frame_size
        self.frame_step = frame_step

        self.audio_ext = audio_ext
        self.label_ext = label_ext


    def get_frames(self,
        data, 
        frame_size, 
        frame_step, 
        sampling_rate):
        """prepares an iterator through the possible batches of the waveform frames
        Args:
            data: audio time series aka the waveform
            batch_size: number of frames contained in one batch
        Returns:
            A `Generator` of all possible batches from the input waveform frames
            and of size `batch_size`
        Raises:
        """
        #data_duration = librosa.core.get_duration(y=data, sr=sampling_rate) # duration in samples
        #n_frames = sound_clip_duration * frame_step

        start = 0
        while start <= len(data):
            yield start, start+frame_size
            start += frame_step
        

        
       def extract_chime_features(self, 
        chime_dir, 
        audio_ext="*.wav",
        n_mfcc = 12, 
        sampling_rate=16000, 
        frame_size=256, 
        frame_step=128):
        """ Extract MFCC features from CHIME corpus audio files
        Args:
            chime_dir: CHIME parent directory name
            audio_ext: (optional) audio file extension
            n_mfcc: (optional) number of MFCCs to extract
            sampling_rate: (optional) sampling rate of the input audio files, default value is 16kHz
            frame_size: (optional) size of the frame,
                    default frame_size is 256 samples ~ 16ms at 16kHz
            frame_step: (optional) number of samples between successive frames, 
                    default frame_step is 128 samples ~ 8ms at 16kHz
        
        Returns:
            A pair on Numpy ndarrays `Features` and `Labels`
        Raises:
        """
    
        features = [] #MFCC features for each frame
        labels = [] #label of each frame
        label = [] #label of each frame
    
        file_list = glob.glob(os.path.join(chime_dir,'normalized', audio_ext))
    
        # Iterate over the channels audio files
        for fn in random.sample(file_list, len(file_list)):
        
            basename = os.path.basename(fn)
        
            # Load the audio time series and its sampling rate
            sound_clip,s = librosa.load(fn, sr=sampling_rate) #sample input files at 16kHz

            # Mel Frequency Cepstral Coefficents
            mfcc = librosa.feature.mfcc(
                y=sound_clip, 
                sr=sampling_rate,
                n_mfcc=n_mfcc, 
                n_fft=frame_size,
                hop_length=frame_step)

            # MFCC deltas
            mfcc_delta = librosa.feature.delta(
                mfcc)

            # MFCC double deltas
            mfcc_delta2 = librosa.feature.delta(
                mfcc, 
                order=2)

            mel_spectogram = librosa.feature.melspectrogram(
                y=sound_clip, 
                sr=sampling_rate,
                n_fft=frame_size,
                hop_length=frame_step)

            # Root Mean Square Energy
            rmse = librosa.feature.rmse(
                S=mel_spectogram, 
                frame_length=frame_size,
                hop_length=frame_step)


            mfcc = np.asarray(mfcc)
            mfcc_delta = np.asarray(mfcc_delta)
            mfcc_delta2 = np.asarray(mfcc_delta2)
            rmse = np.asarray(rmse)
            
            feature = np.concatenate((mfcc, mfcc_delta, mfcc_delta2, rmse), axis=0)
            feature = feature.T

            print(feature.shape)

            feature = np.asarray(feature)
            features = np.asarray(features)
            label = np.asarray(label)
            labels = np.asarray(labels)
        
            if "BGD" in basename:
                label = np.zeros(len(feature)) # non-speech
            else:
                label = np.ones(len(feature)) # speech
            
            
            if features.size == 0 :
                features=feature
            else:
                features = np.concatenate((features, feature))
            
            if labels.size == 0 :
                labels=label
            else:
                labels = np.concatenate((labels, label))
     
    
        features = np.asarray(features)
    
        print("Features size: ",np.array(features).shape)
        print("Labels size [BEFORE one-hot encode]: ",np.array(labels,dtype = np.int).shape)
    
        return np.array(features),np.array(labels,dtype = np.int)

    
    def get_chime(self,
        chime_dir,
        dataset_dir='dataset'):
        """ get the CHIME corpus feature vectors
        :param chime_dir: CHIME corpus parent directory
        :type string
        :returns: -
        :throws: -
        """
        print("CHIME Training data processing ...")

        #chime_dir = 'CHIME'

        X_chime_train, Y_chime_train = self.extract_chime_features(
            chime_dir = chime_dir, 
            audio_ext= self.audio_ext, 
            n_mfcc= self.n_mfcc,
            sampling_rate= self.sampling_rate, 
            frame_size= self.frame_size, 
            frame_step= self.frame_step)

        Y_chime_train_hot = one_hot_encode(Y_chime_train)

        with open(os.path.join(dataset_dir,'X_CHIME_withhot.csv'), 'w') as a:
            wxtrain = csv.writer(a)
            wxtrain.writerows(X_chime_train)

        print("CHIME features saved to ",os.path.join(dataset_dir,'X_CHIME_withhot.csv'))
    
        with open(os.path.join(dataset_dir,'Y_CHIME_withhot.csv'), 'w') as b:
            wytrain = csv.writer(b)
            wytrain.writerows(Y_chime_train_hot)

        print("CHIME labels saved to ",os.path.join(dataset_dir,'Y_CHIME_withhot.csv'))

        with open(os.path.join(dataset_dir,'X_CHIME_nohot.csv'), 'w') as a:
            wxtrain = csv.writer(a)
            wxtrain.writerows(X_chime_train)

        print("CHIME features saved to ",os.path.join(dataset_dir,'X_CHIME_nohot.csv'))
        
    
        with open(os.path.join(dataset_dir,'Y_CHIME_nohot.csv'), 'w') as b:
            wytrain = csv.writer(b)
            for e in Y_chime_train:
                wytrain.writerow([e])

        print("CHIME labels saved to ",os.path.join(dataset_dir,'Y_CHIME_nohot.csv'))

        
if __name__ == "__main__":
    #logger = configure_logging()
    main()



