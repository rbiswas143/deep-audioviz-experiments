import numpy as np


# Padding mfcc data to make the training/test dimensions divisible by 2 ** num_net_scale_downs
def pad_mfccs(x, num_net_scale_downs, num_tracks, num_segments_per_track, num_mfcc, num_mfcc_frames):
    divisor = 2 ** num_net_scale_downs

    num_pad_mfcc_frames = (
        0 if num_mfcc_frames % divisor == 0 else (int(num_mfcc_frames / divisor) + 1) * divisor - num_mfcc_frames)
    x_pad_frames = np.zeros((num_tracks * num_segments_per_track, num_mfcc, num_pad_mfcc_frames, 1))
    x = np.concatenate((x, x_pad_frames), axis=2)
    num_mfcc_frames_new = x.shape[2]

    num_pad_mfcc = (0 if num_mfcc % divisor == 0 else (int(num_mfcc / divisor) + 1) * divisor - num_mfcc)
    x_pad_mfcc = np.zeros((num_tracks * num_segments_per_track, num_pad_mfcc, num_mfcc_frames_new, 1))
    x = np.concatenate((x, x_pad_mfcc), axis=1)
    num_mfcc_new = x.shape[1]
    print('Data padded', x.shape)
    print('New mfcc dimensions', (num_mfcc_new, num_mfcc_frames_new))

    return x, num_mfcc_new, num_mfcc_frames_new
