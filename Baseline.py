# -*- coding: utf-8 -*-
# +
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import argparse
import logging
import random
import time
import sys
import gc
import os


import torch.backends.cudnn as cudnn
from attrdict import AttrDict
import torch.optim as optim
import torch.nn as nn
import torch

from sgan.losses import displacement_error, final_displacement_error
from sgan.losses import gan_g_loss, gan_d_loss, l2_loss

from sgan.utils import int_tuple, bool_flag, get_total_norm
from sgan.utils import relative_to_abs, get_dset_path

from sgan.models import TrajectoryGenerator, TrajectoryDiscriminator
from sgan.data.loader import data_loader

import train


# +
# checkpoint = torch.load("models/sgan-models/zara1_8_model.pt")

# for k in sorted(checkpoint['args']):
#     print(k,"\t",checkpoint['args'][k])

# -

def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(seed)
    torch.backends.cudnn.benchmark = True


# +
parser = argparse.ArgumentParser()
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)

# Dataset options
parser.add_argument('--dataset_name', default='zara1', type=str)
parser.add_argument('--delim', default=' ')
parser.add_argument('--loader_num_workers', default=4, type=int)
parser.add_argument('--obs_len', default=8, type=int)
parser.add_argument('--pred_len', default=8, type=int)
parser.add_argument('--skip', default=1, type=int)

# Optimization
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--num_iterations', default=10000, type=int)
parser.add_argument('--num_epochs', default=200, type=int)    

# Model Options
parser.add_argument('--embedding_dim', default=64, type=int)
parser.add_argument('--num_layers', default=1, type=int)
parser.add_argument('--dropout', default=0, type=float)
parser.add_argument('--batch_norm', default=0, type=bool_flag)
parser.add_argument('--mlp_dim', default=1024, type=int)

# Generator Options
parser.add_argument('--encoder_h_dim_g', default=64, type=int)
parser.add_argument('--decoder_h_dim_g', default=128, type=int)
parser.add_argument('--noise_dim', default=None, type=int_tuple)
parser.add_argument('--noise_type', default='gaussian')
parser.add_argument('--noise_mix_type', default='ped')
parser.add_argument('--clipping_threshold_g', default=0, type=float)
parser.add_argument('--g_learning_rate', default=5e-4, type=float)
parser.add_argument('--g_steps', default=1, type=int)    

# Pooling Options
parser.add_argument('--pooling_type', default='pool_net')
parser.add_argument('--pool_every_timestep', default=1, type=bool_flag)    

# Pool Net Option
parser.add_argument('--bottleneck_dim', default=1024, type=int)

# Social Pooling Options
parser.add_argument('--neighborhood_size', default=2.0, type=float)
parser.add_argument('--grid_size', default=8, type=int)    

# Discriminator Options
parser.add_argument('--d_type', default='local', type=str)
parser.add_argument('--encoder_h_dim_d', default=64, type=int)
parser.add_argument('--d_learning_rate', default=5e-4, type=float)
parser.add_argument('--d_steps', default=2, type=int)
parser.add_argument('--clipping_threshold_d', default=0, type=float)

# Loss Options
parser.add_argument('--l2_loss_weight', default=0, type=float)
parser.add_argument('--best_k', default=1, type=int)    

# Output
parser.add_argument('--output_dir', default=os.getcwd())
parser.add_argument('--print_every', default=5, type=int)
parser.add_argument('--checkpoint_every', default=100, type=int)
parser.add_argument('--checkpoint_name', default='checkpoint')
parser.add_argument('--checkpoint_start_from', default=None)
parser.add_argument('--restore_from_checkpoint', default=1, type=int)
parser.add_argument('--num_samples_check', default=5000, type=int)

# Misc
parser.add_argument('--use_gpu', default=0, type=int)
parser.add_argument('--timing', default=0, type=int)
parser.add_argument('--gpu_num', default="0", type=str)



parser.add_argument('--seed', default="0", type=int)
parser.add_argument('--alpha', default="390", type=int, help='LRP weight')
parser.add_argument('--negative', default="1", type=int, help='1 is positive LRP and -1 is negative LRP')

parser.add_argument('--traj_std', default="0.0656", type=float, help='std for randomnoise')
parser.add_argument('--traj_rel_std', default="0.0324", type=float, help='std for randomnoise')

parser.add_argument('--checkpoint_load_path', required=True)
parser.add_argument('--checkpoint_save_path', required=True)
parser.add_argument('--mode', default=""'', type=str, choices=[ 'random_noise', 'lrp'], help = "lrp or random_noise")
                    
parser.add_argument('--l2_loss_distill_weight', default="1", type=float, help = "basic trajectory forecasting loss")
parser.add_argument('--response_distill_loss_weight', default="1", type=float, help = "loss between teacher's output and sutdent's output")
parser.add_argument('--feat_distill_loss_weight', default="1", type=float,  help = "loss between teacher's output and sutdent's feature")
parser.add_argument('--discriminatore_loss_weight', default="1", type=float, help = "discriminator loss")

parser.add_argument('--response_distill_loss', action='store_true')
parser.add_argument('--feat_distill', action='store_true')


# +
def main(args):
    set_seed(args.seed)
    device = 'cuda'
    
    checkpoint_T = torch.load(args.checkpoint_load_path)
    args.output_dir = "./"
    long_dtype, float_dtype = get_dtypes(args)

    generator_T = get_generator(args)
    generator_T.load_state_dict(checkpoint_T['g_state'])
    generator_T = generator_T.to(device)

    generator_S = get_generator(args)
    generator_S.apply(init_weights)
    generator_S.type(float_dtype).train()
    generator_S = generator_S.to(device)

    discriminator_S = get_discriminator(args)
    discriminator_S.apply(init_weights)
    discriminator_S.type(float_dtype).train()
    discriminator_S = discriminator_S.to(device)
    
    generator_T.train()
    generator_S.train()
    discriminator_S.train()

    g_loss_fn = gan_g_loss
    d_loss_fn = gan_d_loss

    optimizer_g = optim.Adam(generator_S.parameters(), lr=args.g_learning_rate)
    optimizer_d = optim.Adam(
        discriminator_S.parameters(), lr=args.d_learning_rate
    )

    #  Data Loader
    train_path = get_dset_path(args.dataset_name, 'train')
    _, train_loader = data_loader(args, train_path)

    val_path = get_dset_path(args.dataset_name, 'val')
    _, val_loader = data_loader(args, val_path)

    checkpoint = {
        'args': args.__dict__,
        'G_losses': defaultdict(list),
        'D_losses': defaultdict(list),
        'losses_ts': [],
        'metrics_val': defaultdict(list),
        'metrics_train': defaultdict(list),
        'sample_ts': [],
        'restore_ts': [],
        'norm_g': [],
        'norm_d': [],
        'counters': {'t': None, 'epoch': None},
        'g_state': None,
        'g_optim_state': None,
        'd_state': None,
        'd_optim_state': None,
        'g_best_state': None,
        'd_best_state': None,
        'best_t': None,
        'g_best_nl_state': None,
        'd_best_state_nl': None,
        'best_t_nl': None,
    }
    
    t = 0
    epoch = 0
    while t < args.num_iterations:
        gc.collect()
        d_steps_left = args.d_steps
        g_steps_left = args.g_steps
        epoch += 1

#         pbar = tqdm(train_loader)
#         for batch in pbar:
        for batch_idx, batch in enumerate(train_loader):
            if d_steps_left > 0:
                step_type = 'd'
                losses_d = discriminator_step(args, batch, generator_S,
                                              discriminator_S, d_loss_fn,
                                              optimizer_d)
                d_steps_left -= 1
            elif g_steps_left > 0:
                step_type = 'g'
                losses_g = generator_step(args, batch, generator_S, generator_T,
                                          discriminator_S, g_loss_fn,
                                          optimizer_g)
                g_steps_left -= 1

            # 여기 밑으로는 그냥 evaluation하고 모델 저장하는 부분
            if d_steps_left > 0 or g_steps_left > 0:
                continue

#             pbar.set_postfix({
#                 "G_l2" : losses_g['G_l2_loss_rel'],
#                 "G_adv" : losses_g['G_discriminator_loss'],
#                 "G_distill" : losses_g['g_distill_loss'],
#                 "G_feat" : losses_g['loss_feat'],
#                 "D" : losses_d['D_total_loss']
#             })

            # Maybe save a checkpoint
            if t > 0 and t % args.checkpoint_every == 0:
                print('Checking stats on val ...')
                metrics_val = train.check_accuracy(
                    args, val_loader, generator_S, discriminator_S, d_loss_fn
                )
                print('Checking stats on train ...')
                metrics_train = train.check_accuracy(
                    args, train_loader, generator_S, discriminator_S,
                    d_loss_fn, limit=True
                )

                for k, v in sorted(metrics_val.items()):
                    print('  [val] {}: {:.3f}'.format(k, v))
                    checkpoint['metrics_val'][k].append(v)
                for k, v in sorted(metrics_train.items()):
                    print('  [train] {}: {:.3f}'.format(k, v))
                    checkpoint['metrics_train'][k].append(v)

                min_ade = min(checkpoint['metrics_val']['ade'])
                min_ade_nl = min(checkpoint['metrics_val']['ade_nl'])

                if metrics_val['ade'] == min_ade:
                    print('New low for avg_disp_error')
                    checkpoint['best_t'] = t
                    checkpoint['g_best_state'] = generator_S.state_dict()
                    checkpoint['d_best_state'] = discriminator_S.state_dict()

                if metrics_val['ade_nl'] == min_ade_nl:
                    print('New low for avg_disp_error_nl')
                    checkpoint['best_t_nl'] = t
                    checkpoint['g_best_nl_state'] = generator_S.state_dict()
                    checkpoint['d_best_nl_state'] = discriminator_S.state_dict()


                checkpoint['g_state'] = generator_S.state_dict()
                checkpoint['g_optim_state'] = optimizer_g.state_dict()

                checkpoint['d_state'] = discriminator_S.state_dict()
                checkpoint['d_optim_state'] = optimizer_d.state_dict()
                
                if feat_distill_loss_weight > 0:
                    os.makedirs(args.checkpoint_save_path, exist_ok=True)
                    checkpoint_path = os.path.join(
                        args.output_dir, f'{args.checkpoint_save_path}/{args.dataset_name}_{args.pred_len}_model.pt')
                else:
                    os.makedirs(args.checkpoint_save_path + "_feat", exist_ok=True)
                    checkpoint_path = os.path.join(
                        args.output_dir, f'{args.checkpoint_save_path}_feat/{args.dataset_name}_{args.pred_len}_model.pt')
                    
                print('Saving checkpoint to {}'.format(checkpoint_path))

                torch.save(checkpoint, checkpoint_path)

            t += 1
            d_steps_left = args.d_steps
            g_steps_left = args.g_steps
            if t >= args.num_iterations:
                break
                
    metrics_val = train.check_accuracy(
        args, val_loader, generator_S, discriminator_S, d_loss_fn
    )
    print('Checking stats on train ...')
    metrics_train = train.check_accuracy(
        args, train_loader, generator_S, discriminator_S,
        d_loss_fn, limit=True
    )

    for k, v in sorted(metrics_val.items()):
        print('  [val] {}: {:.3f}'.format(k, v))
        checkpoint['metrics_val'][k].append(v)
    for k, v in sorted(metrics_train.items()):
        print('  [train] {}: {:.3f}'.format(k, v))
        checkpoint['metrics_train'][k].append(v)


# -

def get_lrp(generator_T, obs_traj, obs_traj_rel, pred_traj_gt_rel, seq_start_end, args):
    generator_T.train()
    
    obs_traj.retain_grad = True
    obs_traj.requires_grad = True
    
    obs_traj_rel.retain_grad = True
    obs_traj_rel.requires_grad = True
    
    pred = generator_T(obs_traj, obs_traj_rel, seq_start_end)

    loss = torch.mean((pred - pred_traj_gt_rel) ** 2)
    loss.backward()

    #  ===================================================================
    if args.pool:
        obs_traj_lrp = obs_traj - (obs_traj.grad * torch.abs(obs_traj) * args.alpha * args.negative)
    else:
        obs_traj_lrp = obs_traj
        
    obs_traj_rel_lrp = obs_traj_rel - (obs_traj_rel.grad * torch.abs(obs_traj_rel) * args.alpha * args.negative)

    return obs_traj_lrp, obs_traj_rel_lrp


def generator_step(args, batch, generator_S, generator_T, discriminator, g_loss_fn, optimizer_g):
    batch = [tensor.cuda() for tensor in batch]
    (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,
     loss_mask, seq_start_end) = batch
    
    losses = {}
    loss = torch.zeros(1).to(pred_traj_gt)
    g_l2_loss_rel = []
    g_distill_loss = []
    
    
    loss_mask = loss_mask[:, args.obs_len:]

    for _ in range(args.best_k):
        generator_out_S, feat_S = generator_S(obs_traj, obs_traj_rel, seq_start_end, is_feat=True)
        
        if args.mode == 'lrp':
            obs_traj_ref, obs_traj_rel_ref = get_lrp(generator_T, obs_traj, obs_traj_rel, pred_traj_gt_rel, seq_start_end, args)
        elif args.mode == 'random_noise':
            obs_traj_ref = obs_traj + (torch.randn_like(obs_traj) * args.traj_std)
            obs_traj_rel_ref = obs_traj_rel + (torch.randn_like(obs_traj_rel) * args.traj_rel_std)
        else:
            assert False, "args.mode is wrong !!!!"
#             obs_traj_ref = obs_traj + (torch.randn_like(obs_traj) * 0.0656)
#             obs_traj_rel_ref = obs_traj_rel + (torch.randn_like(obs_traj_rel) * 0.0324)
            
        generator_out_T, feat_T = generator_T(obs_traj_ref, obs_traj_rel_ref, seq_start_end, is_feat=True)
#         generator_out_S2, feat_S2 = generator_S(obs_traj_ref, obs_traj_rel_ref, seq_start_end, is_feat=True)

        pred_traj_fake_rel_S = generator_out_S
        pred_traj_fake_rel_T = generator_out_T

        pred_traj_fake_S = relative_to_abs(pred_traj_fake_rel_S, obs_traj[-1])
        pred_traj_fake_T = relative_to_abs(pred_traj_fake_rel_T, obs_traj[-1])

        if args.l2_loss_weight > 0:
            g_l2_loss_rel.append(args.l2_loss_weight * l2_loss(
                pred_traj_fake_rel_S,
                pred_traj_gt_rel,
                loss_mask,
                mode='raw'))
            
            if args.response_distill_loss:
                g_distill_loss.append(args.l2_loss_distill_weight * l2_loss(
                    pred_traj_fake_rel_S,
                    pred_traj_fake_rel_T,
                    loss_mask,
                    mode='raw'))
            else:
                g_distill_loss.append(torch.tensor([0]))
            
    g_l2_loss_sum_rel = torch.zeros(1).to(pred_traj_gt)
    g_distill_loss_sum_rel = torch.zeros(1).to(pred_traj_gt)
    
    if args.l2_loss_weight > 0:
        g_l2_loss_rel = torch.stack(g_l2_loss_rel, dim=1)
        for start, end in seq_start_end.data:
            _g_l2_loss_rel = g_l2_loss_rel[start:end]
            _g_l2_loss_rel = torch.sum(_g_l2_loss_rel, dim=0)
            _g_l2_loss_rel = torch.min(_g_l2_loss_rel) / torch.sum(loss_mask[start:end])
            g_l2_loss_sum_rel += _g_l2_loss_rel
            
        losses['G_l2_loss_rel'] = g_l2_loss_sum_rel.item()
        loss += g_l2_loss_sum_rel

        
        g_distill_loss = torch.stack(g_distill_loss, dim=1)
        for start, end in seq_start_end.data:
            _g_distill_loss = g_l2_loss_rel[start:end]
            _g_distill_loss = torch.sum(_g_distill_loss, dim=0)
            _g_distill_loss = torch.min(_g_distill_loss) / torch.sum(loss_mask[start:end])
            g_distill_loss_sum_rel += _g_distill_loss
            
        losses['g_distill_loss'] = g_distill_loss_sum_rel.item()
        loss += g_distill_loss_sum_rel * args.response_distill_loss_weight
        
        if args.feat_distill:
            loss_feat = 0
            for i in range(len(feat_S)):
                if isinstance(feat_S[i], tuple):
                    for j in range(len(feat_S[i])):
                        loss_feat += torch.mean((feat_S[i][j] - feat_T[i][j]) ** 2)
                else:
                    loss_feat += torch.mean((feat_S[i] - feat_T[i]) ** 2)

            losses['loss_feat'] = loss_feat.item()
            loss += loss_feat * args.feat_distill_loss_weight
        else:
            losses['loss_feat'] = 0
        
    traj_fake = torch.cat([obs_traj, pred_traj_fake_S], dim=0)
    traj_fake_rel = torch.cat([obs_traj_rel, pred_traj_fake_rel_S], dim=0)
    
    if discriminator != None:
        scores_fake = discriminator(traj_fake, traj_fake_rel, seq_start_end)
        discriminator_loss = g_loss_fn(scores_fake)
        loss += discriminator_loss * args.discriminatore_loss_weight
        losses['G_discriminator_loss'] = discriminator_loss.item()
    else:
        discriminator_loss = 0
        losses['G_discriminator_loss'] = 0
    
    losses['G_total_loss'] = loss.item()
    
    optimizer_g.zero_grad()
    loss.backward()
    if args.clipping_threshold_g > 0:
        nn.utils.clip_grad_norm_(
            generator_S.parameters(), args.clipping_threshold_g
        )
    optimizer_g.step()

    return losses


def discriminator_step(args, batch, generator, discriminator, d_loss_fn, optimizer_d):
    batch = [tensor.cuda() for tensor in batch]
    (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,
     loss_mask, seq_start_end) = batch
    losses = {}
    loss = torch.zeros(1).to(pred_traj_gt)

    generator_out = generator(obs_traj, obs_traj_rel, seq_start_end)

    pred_traj_fake_rel = generator_out
    pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])

    traj_real = torch.cat([obs_traj, pred_traj_gt], dim=0)
    traj_real_rel = torch.cat([obs_traj_rel, pred_traj_gt_rel], dim=0)
    traj_fake = torch.cat([obs_traj, pred_traj_fake], dim=0)
    traj_fake_rel = torch.cat([obs_traj_rel, pred_traj_fake_rel], dim=0)

    scores_fake = discriminator(traj_fake, traj_fake_rel, seq_start_end)
    scores_real = discriminator(traj_real, traj_real_rel, seq_start_end)
    
    # Compute loss with optional gradient penalty
    data_loss = d_loss_fn(scores_real, scores_fake)
    losses['D_data_loss'] = data_loss.item()
    loss += data_loss
    losses['D_total_loss'] = loss.item()
    
    
    optimizer_d.zero_grad()
    loss.backward()
    if args.clipping_threshold_d > 0:
        nn.utils.clip_grad_norm_(discriminator.parameters(),
                                 args.clipping_threshold_d)
        
    optimizer_d.step()

    return losses


# +
def get_generator(args):
    generator = TrajectoryGenerator(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        embedding_dim=args.embedding_dim,
        encoder_h_dim=args.encoder_h_dim_g,
        decoder_h_dim=args.decoder_h_dim_g,
        mlp_dim=args.mlp_dim,
        num_layers=args.num_layers,
        noise_dim=args.noise_dim,
        noise_type=args.noise_type,
        noise_mix_type=args.noise_mix_type,
        pooling_type=args.pooling_type,
        pool_every_timestep=args.pool_every_timestep,
        dropout=args.dropout,
        bottleneck_dim=args.bottleneck_dim,
        neighborhood_size=args.neighborhood_size,
        grid_size=args.grid_size,
        batch_norm=args.batch_norm)
    
    return generator

def get_discriminator(args):
    discriminator = TrajectoryDiscriminator(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        embedding_dim=args.embedding_dim,
        h_dim=args.encoder_h_dim_d,
        mlp_dim=args.mlp_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        batch_norm=args.batch_norm,
        d_type='local')
#         activation='leakyrelu')
    
    return discriminator


# +
def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight)
        
def get_dtypes(args):
    long_dtype = torch.LongTensor
    float_dtype = torch.FloatTensor
    if args.use_gpu == 1:
        long_dtype = torch.cuda.LongTensor
        float_dtype = torch.cuda.FloatTensor
    return long_dtype, float_dtype




# +
if __name__ == '__main__':
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
    
#     checkpoint = torch.load("./models/sgan-models/hotel_8_model.pt")
#     args = AttrDict(checkpoint['args'])
#     args.seed = 0
#     args.checkpoint_load_path = "./models/sgan-models/hotel_8_model.pt"
#     args.mode = 'lrp'
#     args.alpha = 390
#     args.l2_loss_distill_weight = 1
#     args.response_distill_loss_weight = 1
#     args.feat_distill_loss_weight = 1
#     args.discriminatore_loss_weight = 1
#     args.response_distill_loss = True
#     args.feat_distill = True
#     args.negative = 1

    args.pool = ("sgan-models" not in args.checkpoint_load_path)
    main(args)
