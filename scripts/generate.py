from absl import app, flags, logging
import pdb
import torch, torchaudio, argparse, os, tqdm, re, gin
import cached_conv as cc
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
flags.DEFINE_integer('chunk_size', default=None, help="chunk size for encoding/decoding (default: full file)")


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
    os.makedirs(out_path, exist_ok=True)

    # parse inputs
    audio_files = sum([get_audio_files(f) for f in paths], [])
    ratio = rave.core.get_minimum_size(model)
    print(f'[INFO] Compression ratio is {ratio} samples')

    progress_bar = tqdm.tqdm(audio_files)
    cc.MAX_BATCH_SIZE = 8

    for i, (d, f) in enumerate(progress_bar):
        #TODO reset cache
            
        try:
            x, sr = torchaudio.load(f)
        except: 
            logging.warning('could not open file %s.'%f)
            continue
        progress_bar.set_description(f)

        # load file
        if sr != model.sr:
            x = torchaudio.functional.resample(x, sr, model.sr)
        if model.n_channels != x.shape[0]:
            if model.n_channels < x.shape[0]:
                x = x[:model.n_channels]
            else:
                print('[Warning] file %s has %d channels, but model has %d channels ; skipping'%(f, model.n_channels))
        x = x.to(device)
        if FLAGS.stream:
            if FLAGS.chunk_size:
                #assert FLAGS.chunk_size >= ratio, "chunk_size must be higher than models' compression ratio (here : %s)"%ratio
                if(FLAGS.chunk_size <= ratio): print('[WARNING] RAVE reccomends a chunk size bigger than compression ratio.')
                x = list(x.split(FLAGS.chunk_size, dim=-1))
                print(f'[INFO] Number of Chunks {len(x)} Shape of each one: {x[0].shape}')
                if x[-1].shape[0] < FLAGS.chunk_size:
                    #print(f'[INFO] Last one is {x[-1].shape}')
                    x[-1] = torch.nn.functional.pad(x[-1], (0, FLAGS.chunk_size - x[-1].shape[-1]))
                    #print(f'[INFO] After padding {x[-1].shape}')
                x = torch.stack(x, 0)
                print(f'[INFO] Stacked {x.shape}')
            else:
                x = x[None]
                print(f'[INFO] No chunk size is disabled {x.shape}')
            
            # forward into model
            out = []
            for x_chunk in x:
                x_chunk = x_chunk.to(device)
                out_tmp = model(x_chunk[None])
                out.append(out_tmp)
            out = torch.cat(out, -1)
        else:
            out = model.forward(x[None])

        # save file
        out_path = re.sub(d, "", f)
        out_path = os.path.join(FLAGS.out_path, f)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        torchaudio.save(out_path, out[0].cpu(), sample_rate=model.sr)

if __name__ == "__main__": 
    app.run(main)