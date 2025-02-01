import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from glob import glob

# NEW: We only want 15 channels total now (5 T1 + 5 T2_STIR + 5 Axial T2).
IMG_SIZE = [512, 512]
IN_CHANS = 15  # changed from 30


class Mydataset(Dataset):
    def __init__(self, df, data_path, phase='train', transform=None):
        self.df = df
        self.transform = transform
        self.phase = phase
        self.data_path = data_path

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # We'll allocate space for 15 channels now (H, W, C)
        x = np.zeros((IMG_SIZE[0], IMG_SIZE[1], IN_CHANS), dtype=np.uint8)

        row = self.df.iloc[idx]
        st_id = int(row['study_id'])
        # first column is study_id, rest are labels
        label = row[1:].values.astype(np.int64)

        # =====================
        # 1) Sagittal T1 (5 slices)
        # =====================
        # Instead of range(0,10), we only load 5 slices:
        for i in range(5):
            try:
                p = f'{self.data_path}/{st_id}/Sagittal T1/{i:03d}.png'
                img = Image.open(p).convert('L')
                img = np.array(img, dtype=np.uint8)
                x[..., i] = img  # channels 0..4
            except:
                pass

        # =====================
        # 2) Sagittal T2/STIR (5 slices)
        # =====================
        # Again, load only 5 slices
        for i in range(5):
            try:
                p = f'{self.data_path}/{st_id}/Sagittal T2_STIR/{i:03d}.png'
                img = Image.open(p).convert('L')
                img = np.array(img, dtype=np.uint8)
                x[..., i + 5] = img  # channels 5..9
            except:
                pass

        # =====================
        # 3) Axial T2 (5 slices)
        # =====================
        axt2 = glob(f'{self.data_path}/{st_id}/Axial T2/*.png')
        axt2 = sorted(axt2)

        # We want exactly 5 axial slices, spaced across the entire list of Axial T2 slices
        # so define step accordingly:
        num_slices = 5
        total_ax = len(axt2)
        if total_ax > 0:
            # step in float
            step = total_ax / float(num_slices)
            # We'll pick slices around the center region for variety
            # Original code did: st = len(axt2)/2.0 - 4.0*step, end = len(axt2) ...
            # But let's simplify: just pick 5 evenly spaced indices
            for i in range(num_slices):
                try:
                    idx_ax = int(round(i * step))
                    idx_ax = min(idx_ax, total_ax - 1)  # safety
                    p = axt2[idx_ax]
                    img = Image.open(p).convert('L')
                    img = np.array(img, dtype=np.uint8)
                    x[..., i + 10] = img  # channels 10..14
                except:
                    pass

        # Ensure we have non-empty data
        assert np.sum(x) > 0, f"All-zero image for study_id {st_id}"

        # Albumentations transform
        if self.transform is not None:
            # Albumentations expects (H, W, C) in 'image'
            x = self.transform(image=x)['image']

        # Finally, convert to CHW format for PyTorch
        x = x.transpose(2, 0, 1)  # (C, H, W)

        return x, label
