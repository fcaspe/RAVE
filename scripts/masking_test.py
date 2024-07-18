from absl import app, flags, logging
import pdb
import torch, torchaudio, argparse, os, tqdm, re, gin
import cached_conv as cc
from rave.core import get_rave_receptive_field
from rave.masker import SpectrogramMasking
import numpy
try:
    import rave
except:
    import sys, os 
    sys.path.append(os.path.abspath('.'))
    import rave


FLAGS = flags.FLAGS
flags.DEFINE_string('model', required=True, default=None, help="model path")
flags.DEFINE_multi_string('input', required=True, default=None, help="model inputs (file or folder)")
flags.DEFINE_string('name', default='curve.csv', help="CSV output file name")
flags.DEFINE_integer('gpu', default=-1, help='GPU to use')

def get_audio_files(path):
    audio_files = []
    valid_exts = rave.core.get_valid_extensions()
    for root, _, files in os.walk(path):
        valid_files = list(filter(lambda x: os.path.splitext(x)[1] in valid_exts, files))
        audio_files.extend([(path, os.path.join(root, f)) for f in valid_files])
    return audio_files


@torch.no_grad()
def main(argv):
    torch.set_float32_matmul_precision('high')
    torch.manual_seed(3402)
    cc.use_cached_conv(False)

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
        model = model.eval()
        pqmf_channels = model.pqmf.forward_conv.weight.shape[0]

    # device
    if FLAGS.gpu >= 0:
        device = torch.device('cuda:%d'%FLAGS.gpu)
        model = model.to(device)
    else:
        device = torch.device('cpu')

    # parse inputs
    audio_files = sum([get_audio_files(f) for f in paths], [])
    ratio = rave.core.get_minimum_size(model)
    print(f'[INFO] Compression ratio is {ratio} samples')

    masker = SpectrogramMasking(target_type='relative_power',
                                pow_times=1,
                                win_length=8192,
                                hop_ratio=8192//4,
                                mask_ratio=0)
    masker = masker.to(device)
    # clean cache
    _ = model(torch.zeros(1,1,2**16).to(device))
    progress_bar = tqdm.tqdm(audio_files)

    audios = []
    for i, (d, f) in enumerate(progress_bar):

        try:
            x, sr = torchaudio.load(f)
        except: 
            logging.warning('could not open file %s.'%f)
            continue

        if sr != model.sr:
            x = torchaudio.functional.resample(x, sr, model.sr)
        if model.n_channels != x.shape[0]:
            if model.n_channels < x.shape[0]:
                x = x[:model.n_channels]
            else:
                print('[Warning] file %s has %d channels, but model has %d channels ; skipping'%(f, model.n_channels))
        
        audios.append(x)

    batches = []
    for a in audios:
        if a.shape[-1] > 131072:
            # Split all but remove the last one that is not of 131072
            batches.extend(torch.split(a,split_size_or_sections=131072,dim=-1)[0:-1])
    batches = torch.cat(batches,dim=0)
    batch_size = 1
    batches = torch.split(batches,split_size_or_sections=batch_size,dim=0)
    if batches[-1].shape[0] != batch_size:
        batches = batches[0:-1]

    thresholds = [0,0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.5,1]
    # Compute at different threshold levels
    # Compute at different threshold levels
    distances = []
    d = [] #batch_wise distances
    for t in thresholds:
        for audio in batches:
            audio = audio.to(device)
            masker.mask_ratio = t
            x = masker(audio)
            #print(f'Masker {x.shape}')
            out = model.forward(x[None])
            #print(f'Output {out.shape}')
            l = model.audio_distance(out,audio.unsqueeze(1))['spectral_distance']
            #print('Distance ',distance)
            d.append(l)
        d = torch.mean(torch.tensor(d))
        print(f'Threshold {t} Distance ',d)
        distances.append(d.item())
        d = []

    distances = torch.tensor(distances)
    curve_csv = torch.stack([torch.tensor(thresholds),distances])
    print(curve_csv)
    curve_csv = curve_csv.numpy()
    numpy.savetxt(FLAGS.name,curve_csv,delimiter = ',')

    # save table of results as file and plot image
    #out_path = re.sub(d, "", f)
    #out_path = os.path.join(FLAGS.out_path, f)
    #os.makedirs(os.path.dirname(out_path), exist_ok=True)
    #torchaudio.save(out_path, out[0].cpu(), sample_rate=model.sr)

if __name__ == "__main__": 
    app.run(main)
