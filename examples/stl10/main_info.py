import os
import time
import argparse

import torchvision
import torch
import torch.nn as nn

from util import AverageMeter, TwoAugUnsupervisedDataset
from encoder import SmallAlexNet
from align_uniform import align_loss, uniform_loss
import json


def parse_option():
    parser = argparse.ArgumentParser('STL-10 Representation Learning with Alignment and Uniformity Losses')

    parser.add_argument('--align_w', type=float, default=1, help='Alignment loss weight')
    parser.add_argument('--unif_w', type=float, default=1, help='Uniformity loss weight')
    parser.add_argument('--align_alpha', type=float, default=2, help='alpha in alignment loss')
    parser.add_argument('--unif_t', type=float, default=2, help='t in uniformity loss')

    parser.add_argument('--batch_size', type=int, default=768, help='Batch size')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate. Default is linear scaling 0.12 per 256 batch size')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='Learning rate decay rate')
    parser.add_argument('--lr_decay_epochs', default=[155, 170, 185], nargs='*', type=int,
                        help='When to decay learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='L2 weight decay')
    parser.add_argument('--feat_dim', type=int, default=128, help='Feature dimensionality')

    parser.add_argument('--num_workers', type=int, default=20, help='Number of data loader workers to use')
    parser.add_argument('--log_interval', type=int, default=40, help='Number of iterations between logs')
    parser.add_argument('--gpus', default=[0], nargs='*', type=int,
                        help='List of GPU indices to use, e.g., --gpus 0 1 2 3')

    parser.add_argument('--data_folder', type=str, default='./data', help='Path to data')
    parser.add_argument('--result_folder', type=str, default='./results', help='Base directory to save model')
    parser.add_argument('--suffix', type=str, default='info', help='Name Suffix')

    opt = parser.parse_args()

    opt.data_folder = '/afs/csail.mit.edu/u/h/hehaodele/radar/Hao/datasets'
    opt.result_folder = '/afs/csail.mit.edu/u/h/hehaodele/radar/Hao/projects/align_uniform/results'

    if opt.lr is None:
        opt.lr = 0.12 * (opt.batch_size / 256)

    print(json.dumps(vars(opt), indent=2, default=lambda o: o.__dict__))

    opt.gpus = list(map(lambda x: torch.device('cuda', x), opt.gpus))

    exp_name = f"align{opt.align_w:g}alpha{opt.align_alpha:g}_unif{opt.unif_w:g}t{opt.unif_t:g}"
    if len(opt.suffix) > 0:
        exp_name += f'_{opt.suffix}'

    opt.save_folder = os.path.join(
        opt.result_folder,
        exp_name,
    )
    os.makedirs(opt.save_folder, exist_ok=True)

    return opt


def get_data_loader(opt):
    from util import RandomResizedCropWithBox, TwoAugUnsupervisedDatasetWithBox
    transform_crop = RandomResizedCropWithBox(64, scale=(0.08, 1))
    transform_others = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
        torchvision.transforms.RandomGrayscale(p=0.2),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            (0.44087801806139126, 0.42790631331699347, 0.3867879370752931),
            (0.26826768628079806, 0.2610450402318512, 0.26866836876860795),
        ),
    ])
    dataset = TwoAugUnsupervisedDatasetWithBox(
        torchvision.datasets.STL10(opt.data_folder, 'train+unlabeled', download=True), transform_crop, transform_others)
    return torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, num_workers=opt.num_workers,
                                       shuffle=True, pin_memory=True)


def get_rate(x):
    return sum(x) / len(x) * 100

def main():
    opt = parse_option()

    print(f'Optimize: {opt.align_w:g} * loss_align(alpha={opt.align_alpha:g}) + {opt.unif_w:g} * loss_uniform(t={opt.unif_t:g})')

    torch.cuda.set_device(opt.gpus[0])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    encoder = nn.DataParallel(SmallAlexNet(feat_dim=opt.feat_dim).to(opt.gpus[0]), opt.gpus)

    optim = torch.optim.SGD(encoder.parameters(), lr=opt.lr,
                            momentum=opt.momentum, weight_decay=opt.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, gamma=opt.lr_decay_rate,
                                                     milestones=opt.lr_decay_epochs)

    loader = get_data_loader(opt)

    align_meter = AverageMeter('align_loss')
    unif_meter = AverageMeter('uniform_loss')
    loss_meter = AverageMeter('total_loss')
    it_time_meter = AverageMeter('iter_time')
    info_rate_meter = AverageMeter('info_rate')
    noni_rate_meter = AverageMeter('noni_rate')

    for epoch in range(opt.epochs):
        align_meter.reset()
        unif_meter.reset()
        loss_meter.reset()
        it_time_meter.reset()
        t0 = time.time()
        for ii, (im_x, info_x, im_y, info_y) in enumerate(loader):
            optim.zero_grad()
            x, y = encoder(torch.cat([im_x.to(opt.gpus[0]), im_y.to(opt.gpus[0])])).chunk(2)
            align_loss_val = align_loss(x, y, alpha=opt.align_alpha)
            unif_loss_val = (uniform_loss(x, t=opt.unif_t) + uniform_loss(y, t=opt.unif_t)) / 2
            loss = align_loss_val * opt.align_w + unif_loss_val * opt.unif_w


            info_x, info_y = info_x.to(opt.gpus[0]), info_y.to(opt.gpus[0])
            info_x_idx, noni_x_idx = info_x > 0.5, info_x < 0.2
            info_y_idx, noni_y_idx = info_y > 0.5, info_y < 0.2

            info_pair_idx = info_x_idx & info_y_idx

            if info_pair_idx.any():
                align_loss_info = align_loss(x[info_pair_idx], y[info_pair_idx], alpha=opt.align_alpha)
            else:
                align_loss_info = 0

            uniform_loss_noninfo = 0
            if noni_x_idx.any():
                uniform_loss_noninfo += uniform_loss(x[noni_x_idx], t=opt.unif_t)
            if noni_y_idx.any():
                uniform_loss_noninfo += uniform_loss(y[noni_y_idx], t=opt.unif_t)
            uniform_loss_noninfo /= 2

            loss_info = align_loss_info * opt.align_w + uniform_loss_noninfo * opt.unif_w

            loss = loss + loss_info

            align_meter.update(align_loss_val, x.shape[0])
            unif_meter.update(unif_loss_val)
            loss_meter.update(loss, x.shape[0])
            info_rate_meter.update((get_rate(info_x_idx)+get_rate(info_y_idx))/2)
            noni_rate_meter.update((get_rate(noni_x_idx)+get_rate(noni_y_idx))/2)

            loss.backward()
            optim.step()
            it_time_meter.update(time.time() - t0)
            if ii % opt.log_interval == 0:
                print(f"Epoch {epoch}/{opt.epochs}\tIt {ii}/{len(loader)}\t" +
                      f"{align_meter}\t{unif_meter}\t{loss_meter}\t{it_time_meter}\t{info_rate_meter}\t{noni_rate_meter}")
            t0 = time.time()
        scheduler.step()

        if epoch % 40 == 0:
            ckpt_file = os.path.join(opt.save_folder, f'encoder-ep{epoch}.pth')
            torch.save(encoder.module.state_dict(), ckpt_file)
    ckpt_file = os.path.join(opt.save_folder, 'encoder.pth')
    torch.save(encoder.module.state_dict(), ckpt_file)
    print(f'Saved to {ckpt_file}')


if __name__ == '__main__':
    main()
