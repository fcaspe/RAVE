from absl import app, flags, logging
import pdb
import torch, torchaudio, argparse, os, tqdm, re, gin
import cached_conv as cc
import soundfile as sf
from pathlib import Path
from os import path, makedirs, environ
import librosa as li
from einops import rearrange
try:
    import rave
except:
    import sys, os 
    sys.path.append(os.path.abspath('.'))
    import rave


FLAGS = flags.FLAGS
flags.DEFINE_string('model', required=True, default=None, help="model path")
flags.DEFINE_multi_string('input', required=True, default=None, help="model inputs (file or folder)")
flags.DEFINE_string('out_path', 'generations', help="output path")
flags.DEFINE_string('name', None, help="name of the model")
flags.DEFINE_integer('gpu', default=-1, help='GPU to use')
flags.DEFINE_bool('stream', default=False, help='simulates streaming mode')
flags.DEFINE_bool('window', default=True, help='if overlapping and adding, use Hann')
flags.DEFINE_integer('block_size', default=None, help="block size for encoding/decoding")
flags.DEFINE_integer('hop_size', default=None, help="hop size for encoding/decoding")


def get_audio_files(path):
    audio_files = []
    valid_exts = rave.core.get_valid_extensions()
    for root, _, files in os.walk(path):
        valid_files = list(filter(lambda x: os.path.splitext(x)[1] in valid_exts, files))
        audio_files.extend([(path, os.path.join(root, f)) for f in valid_files])
    return audio_files


def main(argv):
    torch.set_float32_matmul_precision('high')
    if(FLAGS.stream is True):
        print(f'[INFO] Using Cached Conv.')
    cc.use_cached_conv(FLAGS.stream)

    model_path = FLAGS.model
    paths = FLAGS.input
    # load model
    logging.info("building rave")
    is_scripted = False
    if not os.path.exists(model_path):
        logging.error('path %s does not seem to exist.'%model_path)
        exit()
    if os.path.splitext(model_path)[1] == ".ts":
        model = torch.jit.load(model_path)
        is_scripted = True
    else:
        config_path = rave.core.search_for_config(model_path)
        print(f'Gin config path is {config_path}')
        if config_path is None:
            logging.error('config not found in folder %s'%model_path)
        gin.parse_config_file(config_path)
        model = rave.RAVE()
        run = rave.core.search_for_run(model_path)
        if run is None:
            logging.error("run not found in folder %s"%model_path)
        model = model.load_from_checkpoint(run)

    # device
    if FLAGS.gpu >= 0:
        device = torch.device('cuda:%d'%FLAGS.gpu)
        model = model.to(device)
    else:
        device = torch.device('cpu')


    # make output directories
    if FLAGS.name is None:
        FLAGS.name = "_".join(os.path.basename(model_path).split('_')[:-1])
    out_path = os.path.join(FLAGS.out_path, FLAGS.name)
    makedirs(out_path, exist_ok=True)

    # parse inputs
    ratio = rave.core.get_minimum_size(model)
    print(f'[INFO] Compression ratio is {ratio} samples')

    audio_files = sum([get_audio_files(f) for f in paths], [])

    progress_bar = tqdm.tqdm(audio_files)
    cc.MAX_BATCH_SIZE = 8

    with torch.no_grad():
        for i, (d, f) in enumerate(progress_bar):

            try:
                x, sr = torchaudio.load(f)
            except: 
                logging.warning('could not open file %s.'%f)
                continue
            # Reset cache
            empty = torch.zeros(1, 1, 2**14).to(device)
            _ = model(empty)

            # LOAD AUDIO TO TENSOR
            x = x.reshape(1, 1, -1).float().to(device)

            # PAD AUDIO
            n_sample = x.shape[-1]
            pad = (ratio - (n_sample % ratio)) % ratio
            x = torch.nn.functional.pad(x, (0, pad))

            # Block-based processing
            block_size = int(FLAGS.block_size) if FLAGS.block_size is not None else x.shape[-1]
            hop_size = int(FLAGS.hop_size) if FLAGS.hop_size is not None else block_size

            # Pad the start of the audio if the block size is larger than the hop size
            # This simulates a causal processing of the audio
            if block_size > hop_size:
                pad = block_size - hop_size
                x = torch.nn.functional.pad(x, (pad, 0))
            elif hop_size > block_size:
                assert hop_size % block_size == 0, "Hop size must be a multiple of the block size."

            x = x.unsqueeze(1)
            blocks = torch.nn.functional.unfold(x, (1, block_size), stride=(1, hop_size))
            blocks = rearrange(blocks, 'b w t -> t b 1 w')

            y = []
            for block in blocks:
                out = model(block)
                y.append(out)
            
            # need to overlap and add
            if hop_size < ratio:
                y = torch.cat(y, dim=1)
                y = rearrange(y, 'b t f -> b f t')
                if FLAGS.window:
                    y = y * torch.hann_window(y.shape[1], device=y.device)[None, :, None]

                out_size = hop_size * (blocks.shape[0] - 1) + block_size
                y = torch.nn.functional.fold(y, (1, out_size), kernel_size=(1, 2048), stride=(1, hop_size))
                y = y.squeeze(2)
            else:
                y = torch.cat(y, dim=-1)

            y = y.squeeze(0)[:,:n_sample]
            print(y.shape)

            # WRITE AUDIO
            out_path = re.sub(d, "", f)
            out_path = os.path.join(str(FLAGS.out_path) +f'/reconstruction_b{block_size}_h{hop_size}'+ f)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            torchaudio.save(out_path, y, sample_rate=model.sr)


if __name__ == "__main__": 
    app.run(main)