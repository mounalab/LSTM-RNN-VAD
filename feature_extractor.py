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
        

    
    def extract_sweethome_multimodal_features(self, 
        sweethome_dir,
        sweethome_sub_dirs, 
        audio_ext="*.wav",
        label_ext="*.csv",
        n_mfcc = 28, 
        sampling_rate=16000, 
        frame_size=256, 
        frame_step=128):
        """ Extract MFCC features from AudioSet corpus
        Args:
            sweethome_dir: Sweethome parent directory name
            sweethome_sub_dirs: Sweethome subdirectories names
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

        """
        In our case, we have the following values:
        sweethome_dir = sweethome/corpus-parole-domotique-t2
        sweethome_sub_dirs = ['S01','S02', ..., 'S25']
        """
    
        features = [] # MFCC features to return
        labels = [] # labels to return
        label = [] # label of each frame

        print("extracting SWEETHOME Multimodale features ...")
    
        # Iterate over the speakers
        for l, sub_dir in enumerate(sweethome_sub_dirs):

            # Iterate over the channels audio files
            for fn in glob.glob(os.path.join(sweethome_dir, 'Audio', sub_dir,'normalized', audio_ext)):
            
                file_name = os.path.basename(fn).split('.')[0]
                canal_number = file_name.split('_')[1]

                # Load the audio time series and its sampling rate
                sound_clip,s = librosa.load(fn, sr=sampling_rate) #sample input files at 16kHz

                # Getting the MFCC features of the audio time series
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


                feature = np.asarray(feature)
                features = np.asarray(features)

            
                if features.size == 0 :
                    features=feature
                else:
                    features = np.concatenate((features, feature))

                # Getting the labels
                for csvpath in glob.glob(os.path.join(sweethome_dir, 'CSV', label_ext)):

                    csv_name = os.path.basename(csvpath).split('.')[0]

                    if (sub_dir.lower() in csv_name.lower()):

                        temp_labels=[]
                        nb_labels=0

                        # Iterate through the frame batches
                        nb_frames=0
                        for j,(start,end) in enumerate(self.get_frames(sound_clip,frame_size, frame_step, sampling_rate)):
                            
                            start=int(start)
                            end=int(end)

                            nb_frames=nb_frames+1
     
                            label = 0 # non-speech
                            with open(csvpath, 'r') as f:
                                reader = csv.reader(f)
                                next(reader, None)  # skip the header
                        
                                # Get the corresponding class label
                                for i,row in enumerate(list(reader)):
                                    startspeech = int(float(row[0])*sampling_rate) # convert seconds to samples 
                                    endspeech = int(float(row[1])*sampling_rate)
                                    
                                    if ((start >= startspeech) and (start <= endspeech)):
                                        label = 1 # speech
                                        break
                                    
                                    if ((end >= startspeech) and (end <= endspeech)):
                                        label = 1 # speech
                                        break
                                    
                                    if ((startspeech >= start) and (startspeech <= end)):
                                        label = 1 # speech
                                        break
                                    
                                    if ((endspeech >= start) and (endspeech <= end)):
                                        label = 1 # speech
                                        break
                                    

                            labels.append(label)
                            nb_labels=nb_labels+1

                        break


        print("Features size: ",np.array(features).shape)
        print("Labels size [BEFORE one-hot encode]: ",np.array(labels,dtype = np.int).shape)
    
        return np.array(features),np.array(labels,dtype = np.int)


    def extract_sweethome_parole_features(self,
        sweethome_dir,
        sweethome_sub_dirs, 
        audio_ext="*.wav",
        label_ext="*.csv",
        n_mfcc = 28, 
        sampling_rate=16000, 
        frame_size=256, 
        frame_step=128):
        """ Extract MFCC features from AudioSet corpus
        Args:
            sweethome_dir: Sweethome parent directory name
            sweethome_sub_dirs: Sweethome subdirectories names
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

        """
        In our case, we have the following values:
        sweethome_dir = sweethome/corpus-parole-domotique-t2
        sweethome_sub_dirs = ['S01','S02', ..., 'S25']
        """
    
        features = [] # MFCC features to return
        labels = [] # labels to return
        label = [] # label of each frame

        print("extracting SWEETHOME Parole domotique features ...")
    
        # Iterate over the speakers
        for l, sub_dir in enumerate(sweethome_sub_dirs):

            # Iterate over the channels audio files
            for fn in glob.glob(os.path.join(sweethome_dir, 'Audio', sub_dir,'normalized', audio_ext)):
            
                file_name = os.path.basename(fn).split('.')[0]
                canal_number = file_name.split('_')[1]

                # print(" ",file_name," | ",canal_number)

                # Load the audio time series and its sampling rate
                sound_clip,s = librosa.load(fn, sr=sampling_rate) #sample input files at 16kHz

                # Getting the MFCC features of the audio time series
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

                feature = np.asarray(feature)
                features = np.asarray(features)

                # print(" | getting ",len(feature)," features")
            
                if features.size == 0 :
                    features=feature
                else:
                    features = np.concatenate((features, feature))

                # Getting the labels
                for csvpath in glob.glob(os.path.join(sweethome_dir, 'CSV', label_ext)):

                    csv_name = os.path.basename(csvpath).split('.')[0]

                    if (sub_dir.lower() in csv_name.lower()):

                        # print(" | ",csv_name)
                        temp_labels=[]
                        nb_labels=0

                        # Iterate through the frame batches
                        nb_frames=0
                        for j,(start,end) in enumerate(self.get_frames(sound_clip,frame_size, frame_step, sampling_rate)):
                            
                            start=int(start)
                            end=int(end)

                            nb_frames=nb_frames+1
     
                            label = 0 # non-speech
                            with open(csvpath, 'r') as f:
                                reader = csv.reader(f)
                                next(reader, None)  # skip the header
                        
                                # Get the corresponding class label
                                for i,row in enumerate(list(reader)):
                                    startspeech = int(float(row[0])*sampling_rate) # convert seconds to samples 
                                    endspeech = int(float(row[1])*sampling_rate)
                                    
                                    if ((start >= startspeech) and (start <= endspeech)):
                                        label = 1 # speech
                                        break
                                    
                                    if ((end >= startspeech) and (end <= endspeech)):
                                        label = 1 # speech
                                        break
                                    
                                    if ((startspeech >= start) and (startspeech <= end)):
                                        label = 1 # speech
                                        break
                                    
                                    if ((endspeech >= start) and (endspeech <= end)):
                                        label = 1 # speech
                                        break
                                    

                            labels.append(label)
                            nb_labels=nb_labels+1


                        # print(" | ",nb_frames," frames")
                        # print(" | getting ",len(labels)," labels")
                        # print(" | ",nb_labels," labels")
                        break


        print("Features size: ",np.array(features).shape)
        print("Labels size [BEFORE one-hot encode]: ",np.array(labels,dtype = np.int).shape)
    
        return np.array(features),np.array(labels,dtype = np.int)


    def extract_transcript_features(self,
        transcript_dir,
        transcript_sub_dirs, 
        audio_ext="*.wav",
        label_ext="*.csv",
        n_mfcc = 28, 
        sampling_rate=16000, 
        frame_size=256, 
        frame_step=128):
        """ Extract MFCC features from AudioSet corpus
        Args:
            sweethome_dir: Sweethome parent directory name
            sweethome_sub_dirs: Sweethome subdirectories names
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

        """
        In our case, we have the following values:
        sweethome_dir = sweethome/corpus-parole-domotique-t2
        sweethome_sub_dirs = ['S01','S02', ..., 'S25']
        """
    
        features = [] # MFCC features to return
        labels = [] # labels to return
        label = [] # label of each frame

        print("extracting Lecture Transcript features ...")
    
        # Iterate over the speakers
        for l, sub_dir in enumerate(transcript_sub_dirs):

            # Iterate over the channels audio files
            for fn in glob.glob(os.path.join(transcript_dir, sub_dir,'normalized', audio_ext)):
            
                file_name = os.path.basename(fn).split('.')[0]

                # print(" ",file_name," | ",canal_number)

                # Load the audio time series and its sampling rate
                sound_clip,s = librosa.load(fn, sr=sampling_rate) #sample input files at 16kHz

                # Getting the MFCC features of the audio time series
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

                feature = np.asarray(feature)
                features = np.asarray(features)

                # print(" | getting ",len(feature)," features")
            
                if features.size == 0 :
                    features=feature
                else:
                    features = np.concatenate((features, feature))

                # Getting the labels
                for csvpath in glob.glob(os.path.join(transcript_dir, 'CSV', label_ext)):

                    csv_name = os.path.basename(csvpath).split('.')[0]

                    if (file_name.lower() in csv_name.lower()):

                        temp_labels=[]
                        nb_labels=0

                        # Iterate through the frame batches
                        nb_frames=0
                        for j,(start,end) in enumerate(self.get_frames(sound_clip,frame_size, frame_step, sampling_rate)):
                            
                            start=int(start)
                            end=int(end)

                            nb_frames=nb_frames+1
     
                            label = 0 # non-speech
                            with open(csvpath, 'r') as f:
                                reader = csv.reader(f)
                                next(reader, None)  # skip the header
                        
                                # Get the corresponding class label
                                for i,row in enumerate(list(reader)):
                                    startspeech = int(float(row[0])*sampling_rate) # convert seconds to samples 
                                    endspeech = int(float(row[1])*sampling_rate)
                                    
                                    if ((start >= startspeech) and (start <= endspeech)):
                                        label = 1 # speech
                                        break
                                    
                                    if ((end >= startspeech) and (end <= endspeech)):
                                        label = 1 # speech
                                        break
                                    
                                    if ((startspeech >= start) and (startspeech <= end)):
                                        label = 1 # speech
                                        break
                                    
                                    if ((endspeech >= start) and (endspeech <= end)):
                                        label = 1 # speech
                                        break
                                    

                            labels.append(label)
                            nb_labels=nb_labels+1


                        # print(" | ",nb_frames," frames")
                        # print(" | getting ",len(labels)," labels")
                        # print(" | ",nb_labels," labels")
                        break


        print("Features size: ",np.array(features).shape)
        print("Labels size [BEFORE one-hot encode]: ",np.array(labels,dtype = np.int).shape)
    
        return np.array(features),np.array(labels,dtype = np.int)



    def extract_sweethome_multimodal_features_negative(self, 
        sweethome_dir,
        sweethome_sub_dirs, 
        audio_ext="*.wav",
        label_ext="*.csv",
        n_mfcc = 28, 
        sampling_rate=16000, 
        frame_size=256, 
        frame_step=128):
        """ Extract MFCC features from AudioSet corpus
        Args:
            sweethome_dir: Sweethome parent directory name
            sweethome_sub_dirs: Sweethome subdirectories names
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

        """
        In our case, we have the following values:
        sweethome_dir = sweethome/corpus-parole-domotique-t2
        sweethome_sub_dirs = ['S01','S02', ..., 'S25']
        """
    
        features = [] # MFCC features to return
        labels = [] # labels to return
        label = [] # label of each frame

        print("extracting SWEETHOME Multimodale features ...")
    
        # Iterate over the speakers
        for l, sub_dir in enumerate(sweethome_sub_dirs):

            # Iterate over the channels audio files
            for fn in glob.glob(os.path.join(sweethome_dir, 'Audio', sub_dir, 'normalized', audio_ext)):
            
                file_name = os.path.basename(fn).split('.')[0]
                canal_number = file_name.split('_')[1]

                # Load the audio time series and its sampling rate
                sound_clip,s = librosa.load(fn, sr=sampling_rate) #sample input files at 16kHz

                # Getting the MFCC features of the audio time series
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

            
                if features.size == 0 :
                    features=feature
                else:
                    features = np.concatenate((features, feature))

                # Getting the labels
                for csvpath in glob.glob(os.path.join(sweethome_dir, 'CSV', label_ext)):

                    csv_name = os.path.basename(csvpath).split('.')[0]

                    if (sub_dir.lower() in csv_name.lower()):

                        temp_labels=[]
                        nb_labels=0

                        # Iterate through the frame batches
                        nb_frames=0
                        for j,(start,end) in enumerate(self.get_frames(sound_clip,frame_size, frame_step, sampling_rate)):
                            
                            start=int(start)
                            end=int(end)

                            nb_frames=nb_frames+1
     
                            label = -1 # non-speech
                            with open(csvpath, 'r') as f:
                                reader = csv.reader(f)
                                next(reader, None)  # skip the header
                        
                                # Get the corresponding class label
                                for i,row in enumerate(list(reader)):
                                    startspeech = int(float(row[0])*sampling_rate) # convert seconds to samples 
                                    endspeech = int(float(row[1])*sampling_rate)
                                    
                                    if ((start >= startspeech) and (start <= endspeech)):
                                        label = 1 # speech
                                        break
                                    
                                    if ((end >= startspeech) and (end <= endspeech)):
                                        label = 1 # speech
                                        break
                                    
                                    if ((startspeech >= start) and (startspeech <= end)):
                                        label = 1 # speech
                                        break
                                    
                                    if ((endspeech >= start) and (endspeech <= end)):
                                        label = 1 # speech
                                        break
                                    

                            labels.append(label)
                            nb_labels=nb_labels+1

                        break


        print("Features size: ",np.array(features).shape)
        print("Labels size [BEFORE one-hot encode]: ",np.array(labels,dtype = np.int).shape)
    
        return np.array(features),np.array(labels,dtype = np.int)


    def extract_sweethome_parole_features_negative(self,
        sweethome_dir,
        sweethome_sub_dirs, 
        audio_ext="*.wav",
        label_ext="*.csv",
        n_mfcc = 28, 
        sampling_rate=16000, 
        frame_size=256, 
        frame_step=128):
        """ Extract MFCC features from AudioSet corpus
        Args:
            sweethome_dir: Sweethome parent directory name
            sweethome_sub_dirs: Sweethome subdirectories names
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

        """
        In our case, we have the following values:
        sweethome_dir = sweethome/corpus-parole-domotique-t2
        sweethome_sub_dirs = ['S01','S02', ..., 'S25']
        """
    
        features = [] # MFCC features to return
        labels = [] # labels to return
        label = [] # label of each frame

        print("extracting SWEETHOME Parole domotique features ...")
    
        # Iterate over the speakers
        for l, sub_dir in enumerate(sweethome_sub_dirs):

            # print("Processing ",sub_dir," ...")

            # Iterate over the channels audio files
            for fn in glob.glob(os.path.join(sweethome_dir, 'Audio', sub_dir, 'normalized', audio_ext)):
            
                file_name = os.path.basename(fn).split('.')[0]
                canal_number = file_name.split('_')[1]

                # print(" ",file_name," | ",canal_number)

                # Load the audio time series and its sampling rate
                sound_clip,s = librosa.load(fn, sr=sampling_rate) #sample input files at 16kHz

                # Getting the MFCC features of the audio time series

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

                # print(" | getting ",len(feature)," features")
            
                if features.size == 0 :
                    features=feature
                else:
                    features = np.concatenate((features, feature))

                # Getting the labels
                for csvpath in glob.glob(os.path.join(sweethome_dir, 'CSV', label_ext)):

                    csv_name = os.path.basename(csvpath).split('.')[0]

                    if (sub_dir.lower() in csv_name.lower()):

                        # print(" | ",csv_name)
                        temp_labels=[]
                        nb_labels=0

                        # Iterate through the frame batches
                        nb_frames=0
                        for j,(start,end) in enumerate(self.get_frames(sound_clip,frame_size, frame_step, sampling_rate)):
                            
                            start=int(start)
                            end=int(end)

                            nb_frames=nb_frames+1
     
                            label = -1 # non-speech
                            with open(csvpath, 'r') as f:
                                reader = csv.reader(f)
                                next(reader, None)  # skip the header
                        
                                # Get the corresponding class label
                                for i,row in enumerate(list(reader)):
                                    startspeech = int(float(row[0])*sampling_rate) # convert seconds to samples 
                                    endspeech = int(float(row[1])*sampling_rate)
                                    
                                    if ((start >= startspeech) and (start <= endspeech)):
                                        label = 1 # speech
                                        break
                                    
                                    if ((end >= startspeech) and (end <= endspeech)):
                                        label = 1 # speech
                                        break
                                    
                                    if ((startspeech >= start) and (startspeech <= end)):
                                        label = 1 # speech
                                        break
                                    
                                    if ((endspeech >= start) and (endspeech <= end)):
                                        label = 1 # speech
                                        break
                                    

                            labels.append(label)
                            nb_labels=nb_labels+1


                        # print(" | ",nb_frames," frames")
                        # print(" | getting ",len(labels)," labels")
                        # print(" | ",nb_labels," labels")
                        break


        print("Features size: ",np.array(features).shape)
        print("Labels size [BEFORE one-hot encode]: ",np.array(labels,dtype = np.int).shape)
    
        return np.array(features),np.array(labels,dtype = np.int)


    def extract_timit_features(self,
        timit_dir, 
        audio_ext="*.wav",
        n_mfcc = 28, 
        sampling_rate=16000, 
        frame_size=256, 
        frame_step=128):
        """ Extract MFCC features from TIMIT corpus audio files
        Args:
            timit_dir: TIMIT parent directory name
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
    
        mfccs = [] #MFCC features for each frame
        labels = [] #label of each frame
        label = [] #label of each frame
    
        file_list = glob.glob(os.path.join(timit_dir, audio_ext))
    
        # Iterate over the channels audio files
        for fn in random.sample(file_list, len(file_list)):
        
            basename = os.path.basename(fn)
        
            # Load the audio time series and its sampling rate
            sound_clip,s = librosa.load(fn, sr=sampling_rate) #sample input files at 16kHz

            mfcc = librosa.feature.mfcc(y=sound_clip, sr=sampling_rate,
                                        n_mfcc=n_mfcc, n_fft=frame_size,
                                        hop_length=frame_step).T
            
            mfcc = np.asarray(mfcc)
            mfccs = np.asarray(mfccs)
            label = np.asarray(label)
            labels = np.asarray(labels)
        
            if "BGD" in basename:
                label = np.zeros(len(mfcc)) # non-speech
            else:
                label = np.ones(len(mfcc)) # speech
            
            
            if mfccs.size == 0 :
                mfccs=mfcc
            else:
                mfccs = np.concatenate((mfccs, mfcc))
            
            if labels.size == 0 :
                labels=label
            else:
                labels = np.concatenate((labels, label))
     
    
        features = np.asarray(mfccs)
    
        print("Features size: ",np.array(features).shape)
        print("Labels size [BEFORE one-hot encode]: ",np.array(labels,dtype = np.int).shape)
    
        return np.array(features),np.array(labels,dtype = np.int)



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



    def extract_audioset_features(self,
        audioset_dir, 
        audio_ext="*.wav",
        n_mfcc = 28, 
        sampling_rate=16000, 
        frame_size=256, 
        frame_step=128):
        """ Extract MFCC features from AudioSet corpus
        Args:
            audioset_dir: AudioSet parent directory name
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
    
        mfccs = [] #MFCC features for each frame
        labels = [] #label of each frame
        label = [] #label of each frame
    
        file_list = glob.glob(os.path.join(audioset_dir, 'Audio', audio_ext))
    
        # Iterate over the audio files
        for fn in random.sample(file_list, len(file_list)):
        
            basename = os.path.splitext(os.path.basename(fn))[0]
            label = int(basename.split('__')[2].replace(" ", ""))

            # print(label)
        
            # Load the audio time series and its sampling rate
            if ((label==0) or (label==1)):
                sound_clip,s = librosa.load(fn, sr=sampling_rate) #sample input files at 16kHz

                mfcc = librosa.feature.mfcc(y=sound_clip, sr=sampling_rate,
                                        n_mfcc=n_mfcc, n_fft=frame_size,
                                        hop_length=frame_step).T
                print(np.asarray(mfcc).shape)
            
                mfcc = np.asarray(mfcc)
                mfccs = np.asarray(mfccs)
                label = np.asarray(label)
                labels = np.asarray(labels)
        
                if label == 0:
                    label = np.zeros(len(mfcc)) # Non-speech
                elif label == 1:
                    label = np.ones(len(mfcc)) # Speech
                else:
                    pass # Music
            
                print(np.asarray(label).shape)
            
                if mfccs.size == 0 :
                    mfccs=mfcc
                else:
                    mfccs = np.concatenate((mfccs, mfcc))
            
                if labels.size == 0 :
                    labels=label
                else:
                    labels = np.concatenate((labels, label))
     
    
        features = np.asarray(mfccs)
    
        print("Features size: ",np.array(features).shape)
        print("Labels size [BEFORE one-hot encode]: ",np.array(labels,dtype = np.int).shape)
    
        return np.array(features),np.array(labels,dtype = np.int)


    def extract_bref120_features(self,
        bref120_dir, 
        audio_ext="*.wav",
        n_mfcc = 28, 
        sampling_rate=16000, 
        frame_size=256, 
        frame_step=128):
        """ Extract MFCC features from TIMIT corpus audio files
        Args:
            timit_dir: TIMIT parent directory name
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
        
        file_list = glob.glob(os.path.join(bref120_dir, 'normalized', audio_ext))
    
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


    def get_timit(self,
        timit_dir,
        dataset_dir='dataset'):
        """ get the TIMITcorpus feature vectors
        :param timit_dir: TIMIT corpus parent directory
        :type string
        :returns: -
        :throws: -
        """

        print("TIMIT Training data processing ...")

        #timit_dir = 'TIMIT'

        X_timit_train, Y_timit_train = self.extract_timit_features(
            timit_dir= timit_dir, 
            audio_ext= self.audio_ext, 
            n_mfcc= self.n_mfcc,
            sampling_rate= self.sampling_rate, 
            frame_size= self.frame_size, 
            frame_step= self.frame_step)

        Y_timit_train = one_hot_encode(Y_timit_train)

        with open(os.path.join(dataset_dir,'X_TIMIT_train.csv'), 'w') as a:
            wxtrain = csv.writer(a)
            wxtrain.writerows(X_timit_train)
    
        print("TIMIT features saved to ",os.path.join(dataset_dir,'X_TIMIT_train.csv'))

        with open(os.path.join(dataset_dir,'Y_TIMIT_train.csv'), 'w') as b:
            wytrain = csv.writer(b)
            wytrain.writerows(Y_timit_train)

        print("TIMIT labels saved to ",os.path.join(dataset_dir,'Y_TIMIT_train.csv'))


    def get_sweethome_multimodal(self,
        sweethome_dir,
        sweethome_sub_dirs,
        dataset_dir='dataset'):
        """ get the SWEETHOME feature vectors
        :param root_dir: SWEETHOME root directory
        :type string
        :param root_dir: SWEETHOME parent directory
        :type string
        :param sub_dirs: SWEETHOME subdirectories
        :type list of string
        :param dataset_dir: saving directory
        : type string
        :returns: -
        :throws: -
        """

        print("SWEETHOME Multimodal Test data processing ...")

        #root_dir = 'sweethome' # name of the root directory
        #parent_dir = 'Audio' # parent directory name for the subdirectories
        #sub_dirs = ['S01','S02','S03','S04','S05','S06','S07','S08','S09','S10',
        #'S11','S12','S13','S14','S15','S16','S17','S18','S19','S20',
        #'S21']

        X_sweethome_test,Y_sweethome_test = self.extract_sweethome_multimodal_features(
            sweethome_dir= sweethome_dir, 
            sweethome_sub_dirs= sweethome_sub_dirs,
            audio_ext= self.audio_ext, 
            label_ext= self.label_ext,
            n_mfcc= self.n_mfcc,
            sampling_rate= self.sampling_rate,
            frame_size= self.frame_size, 
            frame_step= self.frame_step)

        Y_sweethome_test_hot = one_hot_encode(Y_sweethome_test)

        with open(os.path.join(dataset_dir,'X_SWEETHOME_Multimodal_withhot.csv'), 'w') as a:
            wxtest = csv.writer(a)
            wxtest.writerows(X_sweethome_test)
    
        print("SWEETHOME Multimodal features saved to ",os.path.join(dataset_dir,'X_SWEETHOME_Multimodal_withhot.csv'))

        with open(os.path.join(dataset_dir,'Y_SWEETHOME_Multimodal_withhot.csv'), 'w') as b:
            wytest = csv.writer(b)
            wytest.writerows(Y_sweethome_test_hot)

        print("SWEETHOME Multimodal labels saved to ",os.path.join(dataset_dir,'Y_SWEETHOME_Multimodal_withhot.csv'))

        with open(os.path.join(dataset_dir,'X_SWEETHOME_Multimodal_nohot.csv'), 'w') as a:
            wxtest = csv.writer(a)
            wxtest.writerows(X_sweethome_test)
    
        print("SWEETHOME Multimodal features saved to ",os.path.join(dataset_dir,'X_SWEETHOME_Multimodal_nohot.csv'))

        with open(os.path.join(dataset_dir,'Y_SWEETHOME_Multimodal_nohot.csv'), 'w') as b:
            wytest = csv.writer(b)
            for e in Y_sweethome_test:
                wytest.writerow([e])

        print("SWEETHOME Multimodal labels saved to ",os.path.join(dataset_dir,'Y_SWEETHOME_Multimodal_nohot.csv'))


    def get_audioset(self, 
        audioset_dir='AudioSet',
        dataset_dir='dataset'):
        """ get the AUDIOSET corpus feature vectors
        :param audioset_dir: AUDIOSET corpus parent directory
        :type string
        :returns: -
        :throws: -
        """

        print("AudioSet Training data processing ...")

        X_audioset_train, Y_audioset_train = self.extract_audioset_features(
            audioset_dir= audioset_dir, 
            audio_ext= self.audio_ext, 
            n_mfcc= self.n_mfcc,
            sampling_rate= self.sampling_rate, 
            frame_size= self.frame_size, 
            frame_step= self.frame_step)

        Y_audioset_train = one_hot_encode(Y_audioset_train)

        with open(os.path.join(dataset_dir,'X_AUDIOSET_train.csv'), 'w') as a:
            wxtrain = csv.writer(a)
            wxtrain.writerows(X_audioset_train)

        print("AUDIOSET features saved to ",os.path.join(dataset_dir,'X_AUDIOSET_train.csv'))
    
        with open(os.path.join(dataset_dir,'Y_AUDIOSET_train.csv'), 'w') as b:
            wytrain = csv.writer(b)
            wytrain.writerows(Y_audioset_train)

        print("AUDIOSET labels saved to ",os.path.join(dataset_dir,'Y_AUDIOSET_train.csv'))



    def get_sweethome_parole(self,
        sweethome_dir,
        sweethome_sub_dirs,
        dataset_dir='dataset'):
        """ get the SWEETHOME feature vectors
        :param root_dir: SWEETHOME root directory
        :type string
        :param root_dir: SWEETHOME parent directory
        :type string
        :param sub_dirs: SWEETHOME subdirectories
        :type list of string
        :param dataset_dir: saving directory
        : type string
        :returns: -
        :throws: -
        """

        print("SWEETHOME Parole domotique Test data processing ...")

        #root_dir = 'sweethome' # name of the root directory
        #parent_dir = 'Audio' # parent directory name for the subdirectories
        #sub_dirs = ['S01','S02','S03','S04','S05','S06','S07','S08','S09','S10',
        #'S11','S12','S13','S14','S15','S16','S17','S18','S19','S20',
        #'S21']

        X_sweethome_parole_test,Y_sweethome_parole_test = self.extract_sweethome_parole_features(
            sweethome_dir= sweethome_dir, 
            sweethome_sub_dirs= sweethome_sub_dirs,
            audio_ext= self.audio_ext, 
            label_ext= self.label_ext,
            n_mfcc= self.n_mfcc,
            sampling_rate= self.sampling_rate,
            frame_size= self.frame_size, 
            frame_step= self.frame_step)

        Y_sweethome_parole_test_hot = one_hot_encode(Y_sweethome_parole_test)

        with open(os.path.join(dataset_dir,'X_SWEETHOME_Parole_withhot.csv'), 'w') as a:
            wxtest = csv.writer(a)
            wxtest.writerows(X_sweethome_parole_test)
    
        print("SWEETHOME Parole features saved to ",os.path.join(dataset_dir,'X_SWEETHOME_Parole_withhot.csv'))

        with open(os.path.join(dataset_dir,'Y_SWEETHOME_Parole_withhot.csv'), 'w') as b:
            wytest = csv.writer(b)
            wytest.writerows(Y_sweethome_parole_test_hot)

        print("SWEETHOME Parole labels saved to ",os.path.join(dataset_dir,'Y_SWEETHOME_Parole_withhot.csv'))

        with open(os.path.join(dataset_dir,'X_SWEETHOME_Parole_nohot.csv'), 'w') as a:
            wxtest = csv.writer(a)
            wxtest.writerows(X_sweethome_parole_test)
    
        print("SWEETHOME Parole features saved to ",os.path.join(dataset_dir,'X_SWEETHOME_Parole_nohot.csv'))

        with open(os.path.join(dataset_dir,'Y_SWEETHOME_Parole_nohot.csv'), 'w') as b:
            wytest = csv.writer(b)
            for e in Y_sweethome_parole_test:
                wytest.writerow([e])

        print("SWEETHOME Parole labels saved to ",os.path.join(dataset_dir,'Y_SWEETHOME_Parole_nohot.csv'))


    def get_transcript(self,
        transcript_dir,
        transcript_sub_dirs,
        dataset_dir='dataset'):
        """ get the SWEETHOME feature vectors
        :param root_dir: SWEETHOME root directory
        :type string
        :param root_dir: SWEETHOME parent directory
        :type string
        :param sub_dirs: SWEETHOME subdirectories
        :type list of string
        :param dataset_dir: saving directory
        : type string
        :returns: -
        :throws: -
        """

        print("Transcript data processing ...")

        #root_dir = 'sweethome' # name of the root directory
        #parent_dir = 'Audio' # parent directory name for the subdirectories
        #sub_dirs = ['S01','S02','S03','S04','S05','S06','S07','S08','S09','S10',
        #'S11','S12','S13','S14','S15','S16','S17','S18','S19','S20',
        #'S21']

        X_transcript,Y_transcript = self.extract_transcript_features(
            transcript_dir= transcript_dir, 
            transcript_sub_dirs= transcript_sub_dirs,
            audio_ext= self.audio_ext, 
            label_ext= self.label_ext,
            n_mfcc= self.n_mfcc,
            sampling_rate= self.sampling_rate,
            frame_size= self.frame_size, 
            frame_step= self.frame_step)

        Y_transcript_hot = one_hot_encode(Y_transcript)

        with open(os.path.join(dataset_dir,'X_Transcript_withhot.csv'), 'w') as a:
            wxtest = csv.writer(a)
            wxtest.writerows(X_transcript)
    
        print("Transcript features saved to ",os.path.join(dataset_dir,'X_Transcript_withhot.csv'))

        with open(os.path.join(dataset_dir,'Y_Transcript_withhot.csv'), 'w') as b:
            wytest = csv.writer(b)
            wytest.writerows(Y_transcript_hot)

        print("Transcript labels saved to ",os.path.join(dataset_dir,'Y_Transcript_withhot.csv'))

        with open(os.path.join(dataset_dir,'X_Transcript_nohot.csv'), 'w') as a:
            wxtest = csv.writer(a)
            wxtest.writerows(X_transcript)
    
        print("Transcript features saved to ",os.path.join(dataset_dir,'X_Transcript_nohot.csv'))

        with open(os.path.join(dataset_dir,'Y_Transcript_nohot.csv'), 'w') as b:
            wytest = csv.writer(b)
            for e in Y_transcript:
                wytest.writerow([e])

        print("Transcript labels saved to ",os.path.join(dataset_dir,'Y_Transcript_nohot.csv'))


    def get_bref120(self,
        bref120_dir,
        dataset_dir='dataset'):
        """ get the SWEETHOME feature vectors
        :param root_dir: SWEETHOME root directory
        :type string
        :param root_dir: SWEETHOME parent directory
        :type string
        :param sub_dirs: SWEETHOME subdirectories
        :type list of string
        :param dataset_dir: saving directory
        : type string
        :returns: -
        :throws: -
        """

        print("BREF120 data processing ...")

        #root_dir = 'sweethome' # name of the root directory
        #parent_dir = 'Audio' # parent directory name for the subdirectories
        #sub_dirs = ['S01','S02','S03','S04','S05','S06','S07','S08','S09','S10',
        #'S11','S12','S13','S14','S15','S16','S17','S18','S19','S20',
        #'S21']

        X_bref120,Y_bref120 = self.extract_bref120_features(
            bref120_dir= bref120_dir, 
            audio_ext= self.audio_ext, 
            n_mfcc= self.n_mfcc,
            sampling_rate= self.sampling_rate,
            frame_size= self.frame_size, 
            frame_step= self.frame_step)

        Y_bref120_hot = one_hot_encode(Y_bref120)

        with open(os.path.join(dataset_dir,'X_BREF120_withhot.csv'), 'w') as a:
            wxtest = csv.writer(a)
            wxtest.writerows(X_bref120)
    
        print("BREF120 features saved to ",os.path.join(dataset_dir,'X_BREF120_withhot.csv'))

        with open(os.path.join(dataset_dir,'Y_BREF120_withhot.csv'), 'w') as b:
            wytest = csv.writer(b)
            wytest.writerows(Y_bref120_hot)

        print("BREF120 labels saved to ",os.path.join(dataset_dir,'Y_BREF120_withhot.csv'))

        with open(os.path.join(dataset_dir,'X_BREF120_nohot.csv'), 'w') as a:
            wxtest = csv.writer(a)
            wxtest.writerows(X_bref120)
    
        print("BREF120 features saved to ",os.path.join(dataset_dir,'X_BREF120_nohot.csv'))

        with open(os.path.join(dataset_dir,'Y_BREF120_nohot.csv'), 'w') as b:
            wytest = csv.writer(b)
            for e in Y_bref120:
                wytest.writerow([e])

        print("BREF120 labels saved to ",os.path.join(dataset_dir,'Y_BREF120_nohot.csv'))


    def get_chime_nohotencoding_dummy(self,
        chime_dir,
        dataset_dir='dataset'):
        """ get the CHIME corpus feature vectors
        :param chime_dir: CHIME corpus parent directory
        :type string
        :returns: -
        :throws: -
        """
        print("CHIME Dummy Training data processing ...")

        #chime_dir = 'CHIME'

        X_chime_train, Y_chime_train = self.extract_chime_features(
            chime_dir = chime_dir, 
            audio_ext= self.audio_ext, 
            n_mfcc= self.n_mfcc,
            sampling_rate= self.sampling_rate, 
            frame_size= self.frame_size, 
            frame_step= self.frame_step)

        X_chime_train=np.asarray(X_chime_train)
        Y_chime_train=np.asarray(Y_chime_train)

        stop = int(len(X_chime_train)*0.1)
        X_train = X_chime_train[:stop,:]
        Y_train = Y_chime_train[:stop]

        
        with open(os.path.join(dataset_dir,'X_CHIME_train_nohot_dummy.csv'), 'w') as a:
            wxtrain = csv.writer(a)
            wxtrain.writerows(X_train)

        print("Dummy CHIME features saved to ",os.path.join(dataset_dir,'X_CHIME_train_nohot_dummy.csv'))
        
    
        with open(os.path.join(dataset_dir,'Y_CHIME_train_nohot_dummy.csv'), 'w') as b:
            wytrain = csv.writer(b)
            for e in Y_train:
                wytrain.writerow([e])

        print("Dummy CHIME labels saved to ",os.path.join(dataset_dir,'Y_CHIME_train_nohot_dummy.csv'))


def main():
    print("Shall we start ??...")

    
    print("\nReading CHIME dataset ...")

    X_chime=[]

    with open('dataset/X_CHIME_dummy_withhot.csv', 'r') as f:
    # with open('dataset/X_CHIME_dummy_nohot.csv', 'r') as f:
        reader = csv.reader(f)
        #next(reader, None)  # skip the headers
        data = list(reader)
        
    for l in data:
        X_chime.append([float(i) for i in l])

    Y_chime=[]
        
    with open('dataset/Y_CHIME_dummy_withhot.csv', 'r') as f:
    # with open('dataset/Y_CHIME_dummy_nohot.csv', 'r') as f:
        reader = csv.reader(f)
        #next(reader, None)  # skip the headers
        data = list(reader)

    for l in data:
        Y_chime.append([int(float(i)) for i in l])

    X_chime=np.asarray(X_chime)
    Y_chime=np.asarray(Y_chime)
    print("CHIME :\n  | Features : ",X_chime.shape,"\n  | Labels : ",Y_chime.shape)

    print("\nReading Transcript dataset ...")

    X_transcript=[]

    with open('dataset/X_Transcript_withhot.csv', 'r') as f:
    # with open('dataset/X_Transcript_nohot.csv', 'r') as f:
        reader = csv.reader(f)
        #next(reader, None)  # skip the headers
        data = list(reader)
        
    for l in data:
        X_transcript.append([float(i) for i in l])

    Y_transcript=[]
        
    with open('dataset/Y_Transcript_withhot.csv', 'r') as f:
    # with open('dataset/Y_Transcript_nohot.csv', 'r') as f:
        reader = csv.reader(f)
        #next(reader, None)  # skip the headers
        data = list(reader)

    for l in data:
        Y_transcript.append([int(float(i)) for i in l])

    X_transcript=np.asarray(X_transcript)
    Y_transcript=np.asarray(Y_transcript)
    print("Transcript :\n  | Features : ",X_transcript.shape,"\n  | Labels : ",Y_transcript.shape)  

    dataset_dir='dataset'

    X_chime_tr = np.concatenate((X_chime, X_transcript))
    Y_chime_tr = np.concatenate((Y_chime, Y_transcript))

    # with open(os.path.join(dataset_dir,'X_CHIME_Transcript_nohot.csv'), 'w') as a:
    #     wxtest = csv.writer(a)
    #     wxtest.writerows(X_chime_tr)

    # with open(os.path.join(dataset_dir,'Y_CHIME_Transcript_nohot.csv'), 'w') as b:
    #     wytest = csv.writer(b)
    #     for e in Y_chime_tr:
    #         wytest.writerow(e)

    with open(os.path.join(dataset_dir,'X_CHIME_Transcript_withhot.csv'), 'w') as a:
      wxtest = csv.writer(a)
      wxtest.writerows(X_chime_tr)

    with open(os.path.join(dataset_dir,'Y_CHIME_Transcript_withhot.csv'), 'w') as b:
      wytest = csv.writer(b)
      wytest.writerows(Y_chime_tr)


    print("And that's it for today ladies and gentlemen!...")

if __name__ == "__main__":
    #logger = configure_logging()
    main()



