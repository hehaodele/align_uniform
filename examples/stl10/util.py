import torch
import torchvision

class AverageMeter(object):
    r"""
    Computes and stores the average and current value.
    Adapted from
    https://github.com/pytorch/examples/blob/ec10eee2d55379f0b9c87f4b36fcf8d0723f45fc/imagenet/main.py#L359-L380
    """
    def __init__(self, name=None, fmt='.6f'):
        fmtstr = f'{{val:{fmt}}} ({{avg:{fmt}}})'
        if name is not None:
            fmtstr = name + ' ' + fmtstr
        self.fmtstr = fmtstr
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    @property
    def avg(self):
        avg = self.sum / self.count
        if isinstance(avg, torch.Tensor):
            avg = avg.item()
        return avg

    def __str__(self):
        val = self.val
        if isinstance(val, torch.Tensor):
            val = val.item()
        return self.fmtstr.format(val=val, avg=self.avg)


class TwoAugUnsupervisedDataset(torch.utils.data.Dataset):
    r"""Returns two augmentation and no labels."""
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        image, _ = self.dataset[index]
        return self.transform(image), self.transform(image)

    def __len__(self):
        return len(self.dataset)


import torchvision.transforms.functional as F
class RandomResizedCropWithBox(torchvision.transforms.RandomResizedCrop):

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped and resized.

        Returns:
            PIL Image or Tensor: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        return F.resized_crop(img, i, j, h, w, self.size, self.interpolation), (i, j, h, w)

class TwoAugUnsupervisedDatasetWithBox(torch.utils.data.Dataset):
    r"""Returns two augmentation and no labels."""
    def __init__(self, dataset, transform_crop, transform_others):
        self.dataset = dataset
        self.t1 = transform_crop
        self.t2 = transform_others

    def __getitem__(self, index):
        image, _ = self.dataset[index]

        x1, xbox = self.t1(image)
        x2 = self.t2(x1)

        y1, ybox = self.t1(image)
        y2 = self.t2(y1)

        return x2, self.iou(xbox), y2, self.iou(ybox)

    def iou(self, box):
        top, left, height, width = box
        def calc_area(t,d,l,r):
            return (d-t)*(r-l)

        def calc_iou(box1, box2):
            t1,d1,l1,r1 = box1
            t2,d2,l2,r2 = box2
            t3,d3,l3,r3 = max(t1,t2),min(d1,d2),max(l1,l2),min(r1,r2)
            inter = calc_area(t3,d3,l3,r3)
            union = calc_area(t1,d1,l1,r1) + calc_area(t2,d2,l2,r2) - inter
            return inter / union

        return calc_iou((top,top+height, left, left + width),
                        (24, 24 + 48, 24, 24 + 48))

    def irate(self, box):
        top, left, height, width = box
        def calc_area(t,d,l,r):
            return (d-t)*(r-l)

        def calc_irate(box1, box2):
            t1,d1,l1,r1 = box1
            t2,d2,l2,r2 = box2
            t3,d3,l3,r3 = max(t1,t2),min(d1,d2),max(l1,l2),min(r1,r2)
            inter = max(0, calc_area(t3,d3,l3,r3))
            union = calc_area(t2,d2,l2,r2)
            return inter / union

        return calc_irate((top,top+height, left, left + width),
                        (24,24+48,24,24+48))

    def __len__(self):
        return len(self.dataset)


from ISR.models import RDN
from PIL import Image

class TwoAugUnsupervisedDatasetSuper(TwoAugUnsupervisedDataset):
    def __init__(self, dataset, transform):
        super(TwoAugUnsupervisedDatasetSuper, self).__init__(dataset, transform)
        self.rdn = RDN(weights='psnr-small')

    def __getitem__(self, index):
        image, _ = self.dataset[index]
        image = Image.fromarray(self.rdn(np.arrat(image)))
        return self.transform(image), self.transform(image)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    import numpy as np
    from local import data_folder

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
        torchvision.datasets.STL10(data_folder, 'train', download=True),
        transform_crop, transform_others
    )

    values = []
    for i, (x, ix, y, iy) in tqdm(enumerate(dataset)):
        values.extend([ix,iy])
        print(np.percentile(values,10), np.median(values), np.percentile(values, 90))

    plt.hist(values, bins=100, range=(0,1))
    plt.savefig('./hist.png')
    plt.close()