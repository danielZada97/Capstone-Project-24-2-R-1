import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from glob import glob
IMG_SIZE = [512, 512]
IN_CHANS = 15


class Mydataset(Dataset):
    def __init__(self, df, data_path, phase='train', transform=None, ):
        self.df = df
        self.transform = transform
        self.phase = phase
        self.data_path = data_path

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
       # print(f"__getitem__ called with idx={idx}")
        x = np.zeros((IMG_SIZE[0], IMG_SIZE[1], IN_CHANS), dtype=np.uint8)
        t = self.df.iloc[idx]
        st_id = int(t['study_id'])
        label = t[1:].values.astype(np.int64)

        # Sagittal T1
        for i in range(0, 10, 1):
            try:
                p = f'{self.data_path}/{st_id}/Sagittal T1/{i:03d}.png'
                img = Image.open(p).convert('L')
                img = np.array(img)
                x[..., i] = img.astype(np.uint8)
            except:
                print(f'failed to load on {st_id}, Sagittal T1')
                pass

        # Sagittal T2/STIR
        for i in range(0, 10, 1):
            try:
                p = f'{self.data_path}/{st_id}/Sagittal T2_STIR/{i:03d}.png'
                img = Image.open(p).convert('L')
                img = np.array(img)
                x[..., i+10] = img.astype(np.uint8)
            except:
                # print(f'failed to load on {st_id}, Sagittal T2/STIR')
                pass

        # Axial T2
        axt2 = glob(
            f'{self.data_path}/{st_id}/Axial T2/*.png')
        axt2 = sorted(axt2)

        step = len(axt2) / 10.0
        st = len(axt2)/2.0 - 4.0*step
        end = len(axt2)+0.0001

        for i, j in enumerate(np.arange(st, end, step)):
            try:
                p = axt2[max(0, int((j-0.5001).round()))]
                img = Image.open(p).convert('L')
                img = np.array(img)
                x[..., i+20] = img.astype(np.uint8)
            except:
                # print(f'failed to load on {st_id}, Sagittal T2/STIR')
                pass

        assert np.sum(x) > 0

        if self.transform is not None:
            x = self.transform(image=x)['image']

        x = x.transpose(2, 0, 1)

        return x, label
