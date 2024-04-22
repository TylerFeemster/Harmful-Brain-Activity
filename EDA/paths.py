class paths:
    TRAIN_CSV = "../csv data/train.csv"
    TRAIN_SPECTOGRAMS = "../train_spectrograms/"
    TRAIN_EEGS = "../train_eegs/"
    TRAIN_CLEAN_20 = "../cleaned_train_eegs_20/"
    TRAIN_CLEAN_10 = "../cleaned_train_eegs_10/"
    TRAIN_CLEAN_5 = "../cleaned_train_eegs_5/"
    EEG_LABELS = ['Fp1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'Fz', 'Cz', 
                  'Pz', 'Fp2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2', 'EKG']
    LABEL_SIDES = {'Fp1': 'left', 'F3': 'left', 'C3': 'left', 'P3': 'left', 'F7': 'left', 'T3': 'left', 
            'T5': 'left', 'O1': 'left', 'Fz': 'middle', 'Cz': 'middle', 'Pz': 'middle', 'Fp2': 'right', 
            'F4': 'right', 'C4': 'right', 'P4': 'right', 'F8': 'right', 'T4': 'right', 'T6': 'right', 
            'O2': 'right', 'EKG' : 'none'}
    BRAIN_REGIONS = {'LL' : ['Fp1','F7','T3','T5','O1'],
                     'LP' : ['Fp1','F3','C3','P3','O1'],
                     'RR' : ['Fp2','F8','T4','T6','O2'],
                     'RP' : ['Fp2','F4','C4','P4','O2']}
    BRAIN_REGION_USED = ['Fp1','F7','T3','T5','O1','F3','C3','P3','Fp2','F8','T4','T6','O2','F4','C4','P4']
    HJORTH_10 = "../hjorth_10/"
