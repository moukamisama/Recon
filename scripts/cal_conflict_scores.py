import json
import argparse
import os
import os.path as osp
import torch
from collections import OrderedDict

def split_dict(d, n_epoch, lb, hb):
    n_batches = len(list(d.values())[0]) / n_epoch
    lb = int(lb * n_batches)
    hb = int(hb * n_batches)

    for key, value in d.items():
        d[key] = value[lb:hb]

def flooding_lower(dict, flood_level=0.0):
    output = OrderedDict()
    for key, value in dict.items():
        v = torch.stack(value)
        mask = (v < flood_level)
        ans = mask.sum(dim=0)
        output[key] = ans.sum().item()

    output = {k: v for k, v in sorted(output.items(), key=lambda item: item[1], reverse=True)}
    return output

def print_dict(dict):
    for key, value in dict.items():
        print(f'{key}: {value}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculating Conflict Scores')
    parser.add_argument('--path', default='.', type=str, help='The path of the file '
                                                              'that store the layer-wise paired cos similarity')
    # parser.add_argument('--saved_path', default='./saved/MultiFashon+MNIST', type=str, help='The saved path of the file')
    parser.add_argument('--S', default=0.0, type=float, help='The S for calculating the S-conflict scores')
    parser.add_argument('--n_epoch', default=40, type=float, help='The S for calculating the S-conflict scores')

    opt = parser.parse_args()

    # get the base name of the file
    path, file_name = osp.split(opt.path)
    file_name = osp.splitext(file_name)[0]
    device = torch.device('cuda:0')

    dictionary = torch.load(opt.path, map_location=device)

    cos = dictionary['cos']

    # The default setting
    lb = 0
    hb = opt.n_epoch

    split_dict(cos, opt.n_epoch, lb, hb)

    flood_angle = flooding_lower(cos, flood_level=opt.S)

    saved_file = osp.join(path, f'{file_name}_S{opt.S}.json')

    with open(saved_file, "w") as fp:
        json.dump(flood_angle, fp)


    with open(saved_file, "r") as fp:
        angles = json.load(fp)
        print_dict(angles)

    print('finished')