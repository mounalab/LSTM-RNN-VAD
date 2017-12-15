from __future__ import unicode_literals

import glob
import os
import random
import librosa
import numpy as np
import tensorflow as tf
import xml.etree.ElementTree as ET
import csv
import time
import youtube_dl
import logging

from pydub import AudioSegment
from datetime import datetime


def opensad_reference(reference_folder, label_folder, subjects_allowed, label_ext="*.csv"):

    print("Getting the ground truth in OpenSAD format ...")

    # Iterate over the channels audio files
    for fn in glob.glob(os.path.join(reference_folder, label_folder, audio_ext)):

        file_name = os.path.basename(fn).split('.')[0]
        subject_id = file_name.split('_')[1]

        csv_data = open(reference_folder+'/OpenSAD/'+file_name+'.csv', 'w')
        # create the csv writer object
            csvwriter = csv.writer(csv_data)

            csv_head = []
            csv_head.append('start')
            csv_head.append('end')
            csv_head.append('utterance')

            csvwriter.writerow(csv_head)

        if (subject_id.lower() in [x.lower() for x in [subjects_allowed]]:



def main():
    print("Shall we start ??...")

    reference_folder = 'sweethome/corpus-parole-domotique-t2'
    label_folder = 'CSV'
    subjects_allowed = ['S01','S02', 'S03', 'S04', 'S05']

    print("And that's it for today ladies and gentlemen!...")

if __name__ == "__main__":
    #logger = configure_logging()
    main()
