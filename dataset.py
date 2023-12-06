from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
import numpy as np

conditioning_range = int(24 * 7)
prediction_range = int(24)
interval = int(24)


class MyDataset(Dataset):

    def __init__(self, mode):
        super().__init__()
        assert mode in ['train', 'test']
        self.data = pd.read_csv(f'dataset/hour_{mode}.csv').set_index(
            'time').astype(float)

    def __getitem__(self, index):
        offset = np.random.randint(0, conditioning_range)
        index = index * interval + offset
        conditioning = self.data.iloc[index:index +
                                      conditioning_range, :370].to_numpy()
        prediction = self.data.iloc[index + conditioning_range:index +
                                    conditioning_range +
                                    prediction_range, :370].to_numpy()
        # covariates 协变量
        covariates = self.data.iloc[index:index + conditioning_range +
                                    prediction_range, 370:].to_numpy()

        return torch.Tensor(conditioning), torch.Tensor(
            prediction), torch.Tensor(covariates)

    def __len__(self):
        return int(
            (len(self.data) - conditioning_range * 2 - prediction_range + 1) /
            interval)


def MyLoader(mode):
    return DataLoader(MyDataset(mode), 1, True)


if __name__ == '__main__':
    loader = MyLoader('train')
    for z1, z2, xc in loader:
        z1 = z1.permute(2, 1, 0)
        z2 = z2.permute(2, 1, 0)
        xc = xc.repeat(z1.shape[0], 1, 1)
        print(z1.shape, z2.shape, xc.shape)
    print(len(loader))
# idx * interval + conditioning_range + prediction_range = data_len -> idx =
