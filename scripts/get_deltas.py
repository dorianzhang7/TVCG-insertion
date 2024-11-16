import argparse
import glob
import torch

def main(path, newtoken, gpu_id):
    layers = []
    for files in glob.glob(f'{path}/checkpoints/*'):
        if ('=' in files or '_' in files) and 'delta' not in files:
            print(files)
            if '=' in files:
                epoch_number = files.split('=')[1].split('.ckpt')[0]
            elif '_' in files:
                epoch_number = files.split('/')[-1].split('.ckpt')[0]

            model = torch.load(files, map_location='cuda:'+gpu_id)
            st = model["state_dict"]
            if len(layers) == 0:
                for key in list(st.keys()):
                    if 'attn2.to_k' in key or 'attn2.to_v' in key or 'attn2.to_q' in key or 'attn2.to_out.0' in key or 'cond_stage_model.mapper' in key or\
                        'cond_stage_model.final_ln.weight' == key or 'cond_stage_model.final_ln.bias' == key or\
                        'proj_out.weight' == key or 'proj_out.bias' == key or\
                        'cond_stage_model.prior_encoder.prior_block.1.weight' == key or\
                        'cond_stage_model.prior_encoder.prior_block.1.bias' == key:
                        layers.append(key)
                print(layers)
            param = []
            st_delta = {'state_dict': {}}
            for each in layers:
                st_delta['state_dict'][each] = st[each].clone()
                param.append(st[each].clone())
            print('/'.join(files.split('/')[:-1]) + f'/delta_last.ckpt')

            torch.save(st_delta, '/'.join(files.split('/')[:-1]) + f'/delta_last.ckpt')


def parse_args():
    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('--path', help='path of folder to checkpoints',
                        type=str)
    parser.add_argument('--newtoken', help='number of new tokens in the checkpoint', default=1,
                        type=int)
    parser.add_argument('--gpu_id', default=0, type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.path, args.newtoken, args.gpu_id)
