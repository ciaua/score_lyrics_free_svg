import os
import sys
import argparse
import random
from pydub import AudioSegment
import numpy as np
import mido
from shutil import copyfile

import torch
import torch.nn as nn
# import torch.nn.functional as F


MIN_MIDI = 21
MAX_MIDI = 108


def parse_midi(path):
    """ Modified from https://github.com/jongwook/onsets-and-frames/blob/master/onsets_and_frames/midi.py#L12

    open midi file and return np.array of (onset, offset, note, velocity) rows
    """
    midi = mido.MidiFile(path)
    duration = midi.length

    time = 0
    sustain = False
    events = []
    for message in midi:
        time += message.time

        if message.type == 'control_change' and message.control == 64 and (message.value >= 64) != sustain:
            # sustain pedal state has just changed
            sustain = message.value >= 64
            event_type = 'sustain_on' if sustain else 'sustain_off'
            event = dict(index=len(events), time=time, type=event_type, note=None, velocity=0)
            events.append(event)

        if 'note' in message.type:
            # MIDI offsets can be either 'note_off' events or 'note_on' with zero velocity
            velocity = message.velocity if message.type == 'note_on' else 0
            event = dict(index=len(events), time=time, type='note', note=message.note, velocity=velocity, sustain=sustain)
            events.append(event)

    notes = []
    for i, onset in enumerate(events):
        if onset['velocity'] == 0:
            continue

        # find the next note_off message
        offset = next(n for n in events[i + 1:] if n['note'] == onset['note'] or n is events[-1])

        if offset['sustain'] and offset is not events[-1]:
            # if the sustain pedal is active at offset, find when the sustain ends
            offset = next(n for n in events[offset['index'] + 1:] if n['type'] == 'sustain_off' or n is events[-1])

        note = (onset['time'], offset['time'], onset['note'], onset['velocity'])
        notes.append(note)

    return np.array(notes), duration


def midi2frame(midi_fp, sr, hop_size):
    notes, duration = parse_midi(midi_fp)

    audio_length = duration * sr
    num_steps = int((audio_length - 1) // hop_size + 1)
    num_keys = MAX_MIDI - MIN_MIDI + 1

    label = np.zeros((num_steps, num_keys), dtype=np.uint8)

    for onset, offset, note, vel in notes:
        left = int(round(onset * sr / hop_size))
        onset_right = min(num_steps, left + 1)
        frame_right = int(round(offset * sr / hop_size))
        frame_right = min(num_steps, frame_right)
        offset_right = min(num_steps, frame_right + 1)

        f = int(note) - MIN_MIDI
        label[left:onset_right, f] = 3
        label[onset_right:frame_right, f] = 2
        label[frame_right:offset_right, f] = 1

    # shape=(num_frames, 88)=(802, 88) if duration=10
    frames = (label > 1).astype('float32')
    frames = np.transpose(frames, (1, 0))

    return frames


def read_lines(file_path):
    with open(file_path, 'r', encoding='utf-8') as opdrf:
        data = [term.strip() for term in opdrf.readlines()]
        return data


def load_params(fp, device_id):
    device = torch.device(device_id)
    params = torch.load(fp, map_location=device)
    return params


def load_model(fp, network, device_id='cpu'):
    obj = load_params(fp, device_id)
    model_state_dict = obj['state_dict.model']
    # optimizer_state_dict = obj['state_dict.optimizer']

    network.load_state_dict(model_state_dict)


def load_vocoder(cuda_id):
    vocoder_model_dir = './trained_parameters/WaveRNN/pretrained_parameters/'
    vocoder_checkpoint_steps = '2000k'
    sys.path.append(vocoder_model_dir)
    import hparams as hp

    sys.path.append('./trained_parameters/WaveRNN')
    from models.fatchord_version import WaveRNN
    vocoder = WaveRNN(
        rnn_dims=hp.voc_rnn_dims,
        fc_dims=hp.voc_fc_dims,
        bits=hp.bits,
        pad=hp.voc_pad,
        upsample_factors=hp.voc_upsample_factors,
        feat_dims=hp.num_mels,
        compute_dims=hp.voc_compute_dims,
        res_out_dims=hp.voc_res_out_dims,
        res_blocks=hp.voc_res_blocks,
        hop_length=hp.hop_length,
        sample_rate=hp.sample_rate,
        mode=hp.voc_mode)

    vocoder_param_fp = os.path.join(vocoder_model_dir, 'checkpoints',
                                    f'checkpoint_{vocoder_checkpoint_steps}_steps.pyt')
    vocoder.load(vocoder_param_fp)

    if cuda_id is not None:
        vocoder = vocoder.cuda(cuda_id)

    return vocoder, hp


def get_raw_filename(path):
    filename = os.path.split(path)[1]
    raw_filename = os.path.splitext(filename)[0]

    return raw_filename


def to_int_if_possible(ss):
    try:
        ss = int(ss)
    except Exception:
        pass

    return ss


def generate_free_singing(num_samples, duration, singer, vocoder, hp, output_path, seed, cuda_id):
    torch.manual_seed(seed)

    sr = hp.sample_rate
    hop_length = hp.hop_length
    num_frames = int(np.ceil(duration * (sr / hop_length)))

    os.makedirs(output_path, exist_ok=True)

    for ii in range(num_samples):
        out_fp_wav = os.path.join(output_path, f'{ii}.wav')
        out_fp_mp3 = os.path.join(output_path, f'{ii}.mp3')

        z = torch.zeros((1, z_dim, num_frames)).normal_(0, 1).float()

        if cuda_id is not None:
            z = z.cuda(cuda_id)

        # Singer
        # shape=(1, num_mels, num_frames)
        melspec_voc = singer(z)

        # Vocoder
        _ = vocoder.generate(melspec_voc, out_fp_wav,
                             hp.voc_gen_batched, hp.voc_target, hp.voc_overlap, hp.mu_law)

        # Convert to mp3
        AudioSegment.from_wav(out_fp_wav).export(out_fp_mp3, format="mp3")
        os.remove(out_fp_wav)


def generate_accompanied_singing(condition, num_samples, singer, vocoder, hp, output_path,
                                 seed, cuda_id):
    if type(condition) == int:
        test_condition_filenames = os.listdir(test_condition_dir)
        random.Random(seed).shuffle(test_condition_filenames)

        condition_files = [os.path.join(test_condition_dir, fn) for fn in test_condition_filenames[:condition]]
    elif os.path.isfile(condition):
        condition_files = [condition]
    elif os.path.isdir(condition):
        condition_files = [os.path.join(condition, fn) for fn in os.listdir(condition)]

    for condition_file in condition_files:
        raw_filename = get_raw_filename(condition_file)

        output_dir = os.path.join(output_path, raw_filename)
        os.makedirs(output_dir, exist_ok=True)

        # ### Copy condition
        cond_audio_fp = os.path.join(condition_audio_dir, f'{raw_filename}.mp3')
        if os.path.exists(cond_audio_fp):
            out_cond_audio_fp = os.path.join(output_dir, '_original.mp3')
            copyfile(cond_audio_fp, out_cond_audio_fp)
        else:
            out_cond_midi_fp = os.path.join(output_dir, '_original.mid')
            copyfile(condition_file, out_cond_midi_fp)

        # ### Load condition ###
        cond = torch.from_numpy(
            midi2frame(condition_file, hp.sample_rate, hp.hop_length)
        ).unsqueeze(dim=0)
        nf = cond.size(2)

        torch.manual_seed(seed)
        for ii in range(num_samples):
            out_fp_wav = os.path.join(output_dir, f'{ii}.wav')
            out_fp_mp3 = os.path.join(output_dir, f'{ii}.mp3')

            # ### Main ###
            z = torch.zeros((1, z_dim, nf)).normal_(0, 1).float()
            zp = torch.cat([z, cond], dim=1)

            if cuda_id is not None:
                zp = zp.cuda(cuda_id)

            # Singer
            # shape=(1, num_mels, num_frames)
            melspec_voc = singer(zp)

            # Vocoder
            _ = vocoder.generate(melspec_voc, out_fp_wav,
                                 hp.voc_gen_batched, hp.voc_target, hp.voc_overlap, hp.mu_law)

            # Convert to mp3
            AudioSegment.from_wav(out_fp_wav).export(out_fp_mp3, format="mp3")
            os.remove(out_fp_wav)


def main(condition, output_path, gender, num_samples, duration, seed, cuda_id):
    # print(type(condition))
    if condition == 0:
        singer_type = 'free_singer'
        print('Free singer')
    else:
        singer_type = 'accompanied_singer'
        print('Accompanied singer')

    singer_name = f'{singer_type}.{gender}'
    param_path = os.path.join(param_dir, f'{singer_name}.torch')

    # ### Load model ###
    Singer = singer_dict[singer_type]
    if singer_type == 'free_singer':
        singer = Singer(num_mels, z_dim)
    elif singer_type == 'accompanied_singer':
        singer = Singer(num_mels, z_dim, freq_dim)
    singer.eval()
    load_model(param_path, singer, device_id='cpu')
    if cuda_id is not None:
        singer = singer.cuda(cuda_id)

    for p in singer.parameters():
        p.requires_grad = False

    # ### Load vocoder ###
    vocoder, hp = load_vocoder(cuda_id)
    vocoder.eval()

    # ### Generate ###
    if singer_type == 'free_singer':

        generate_free_singing(num_samples, duration, singer, vocoder, hp, output_path, seed, cuda_id)

    elif singer_type == 'accompanied_singer':

        generate_accompanied_singing(condition, num_samples, singer, vocoder, hp, output_path,
                                     seed, cuda_id)


def parse_argument():
    parser = argparse.ArgumentParser(description='Singing voice generator')

    parser.add_argument('--condition', dest='condition', default=0,
                        help='integer, midi file, or a folder of midi files. \nThe free singer is selected if condition is 0. The accompanied singer is used otherwise. \nIf a positive integer is given, the given number of conditions are randomly selected from the test set.')
    parser.add_argument('--output_path', dest='output_path', default='./generated',
                        help='Path of the base output folder')
    parser.add_argument('--gender', dest='gender', default='female',
                        help="'female' or 'male'")
    parser.add_argument('--num_samples', dest='num_samples', default=1, type=int,
                        help='The number of samples generated for each condition')
    parser.add_argument('--duration', dest='duration', default=20, type=float,
                        help='The duration of the generated singing for the free singer')
    parser.add_argument('--seed', dest='seed', default=321, type=int,
                        help='Random seed used in the generation process')
    parser.add_argument('--cuda_id', dest='cuda_id', default=None,
                        help='CUDA ID. CPU is used if cuda_id is None.')

    args = parser.parse_args()

    arguments = {
        'condition': to_int_if_possible(args.condition),
        'output_path': args.output_path,
        'gender': args.gender,
        'num_samples': args.num_samples,
        'duration': args.duration,
        'seed': args.seed,
        'cuda_id': int(args.cuda_id) if args.cuda_id is not None else args.cuda_id,
    }
    return arguments


class G3Block(nn.Module):
    def __init__(self, feat_dim, ks, dilation, groups):
        super().__init__()
        # ks = 3  # kernel size
        ksm1 = ks-1
        mfd = feat_dim
        di = dilation
        self.g = groups

        self.relu = nn.LeakyReLU()

        self.rec = nn.GRU(mfd, mfd, num_layers=1, batch_first=True, bidirectional=True)
        self.conv = nn.Conv1d(mfd, mfd, ks, 1, ksm1*di//2, dilation=di, groups=groups)
        self.gn = nn.GroupNorm(groups, mfd)

    def forward(self, x):
        bs, mfd, nf = x.size()

        r = x.transpose(1, 2)
        r, _ = self.rec(r)
        r = r.transpose(1, 2).view(bs, 2, mfd, nf).sum(1)
        c = self.relu(self.gn(self.conv(r)))
        x = x+r+c

        return x


class FreeSinger(nn.Module):
    def __init__(self, feat_dim, z_dim):
        super().__init__()

        ks = 3  # filter size
        mfd = 512
        groups = 4
        self.groups = groups
        self.mfd = mfd

        self.feat_dim = feat_dim
        self.z_dim = z_dim

        # ### Main body ###
        blocks = [
            nn.Conv1d(z_dim, mfd, 3, 1, 1),
            nn.GroupNorm(groups, mfd),
            nn.LeakyReLU(),
            G3Block(mfd, ks, dilation=2, groups=groups),
            G3Block(mfd, ks, dilation=4, groups=groups),
        ]
        self.body = nn.Sequential(*blocks)

        # ### All heads ###
        self.head = nn.Conv1d(mfd, feat_dim, 3, 1, 1)

    def forward(self, z):

        # Body
        x = self.body(z)

        # Head
        # shape=(bs, feat_dim, nf)
        x = torch.sigmoid(self.head(x))

        return x


class AccompaniedSinger(nn.Module):
    def __init__(self, feat_dim, z_dim, freq_dim):
        super().__init__()

        ks = 3  # filter size
        mfd = 512
        groups = 4
        self.groups = groups
        self.mfd = mfd

        self.feat_dim = feat_dim
        self.z_dim = z_dim
        self.freq_dim = freq_dim

        # ### Main body ###
        blocks = [
            nn.Conv1d(z_dim+freq_dim, mfd, 3, 1, 1),
            nn.GroupNorm(groups, mfd),
            nn.LeakyReLU(),
            G3Block(mfd, ks, dilation=2, groups=groups),
            G3Block(mfd, ks, dilation=4, groups=groups),
        ]
        self.body = nn.Sequential(*blocks)

        # ### Head ###
        self.head = nn.Conv1d(mfd, feat_dim, 3, 1, 1)

    def forward(self, z):

        # Body
        x = self.body(z)

        # Head
        # shape=(bs, feat_dim, nf)
        x = torch.sigmoid(self.head(x))

        return x


if __name__ == "__main__":
    z_dim = 20
    num_mels = 80
    freq_dim = 88
    param_dir = './trained_parameters/'
    condition_audio_dir = './data/test_data.jamendo/audios/accompaniment/'
    test_condition_dir = './data/test_data.jamendo/piano_transcription/'

    singer_dict = {
        'free_singer': FreeSinger,
        'accompanied_singer': AccompaniedSinger,
    }

    arguments = parse_argument()

    main(**arguments)
