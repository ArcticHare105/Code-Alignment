import torch
import pandas as pd
import numpy as np
from torch.autograd import Variable
from torch.utils.data import Dataset


class CCDDataset(Dataset):
    def __init__(self, args, run_type):
        self.root = args.root
        self.lang = args.lang
        self.code_type = args.code_type
        self.run_type = run_type
        if self.run_type == 'train':
            train_data_pd = pd.read_pickle(self.root + self.lang + '/train/blocks.pkl').sample(frac=1)
            if self.lang == 'java' and self.code_type in [1, 2, 3, 4, 5]:
                train_data_pos = train_data_pd[train_data_pd['label'].isin([0])]
                train_data_neg = train_data_pd[train_data_pd['label'].isin([self.code_type])]
                if len(train_data_pos) < len(train_data_neg):
                    train_data_neg = train_data_neg.sample(len(train_data_pos), random_state=1)
                else:
                    train_data_pos = train_data_pos.sample(len(train_data_neg), random_state=1)
                train_data_out = pd.concat([train_data_pos, train_data_neg])
                train_data_out.loc[train_data_out['label'] > 0, 'label'] = 1
            elif self.lang == 'java' and self.code_type == 6:
                train_data_pos = train_data_pd[train_data_pd['label'].isin([0])]
                train_datas = [train_data_pos]
                for i in range(5):
                    train_data_neg = train_data_pd[train_data_pd['label'].isin([i+1])]
                    if len(train_data_pos) < len(train_data_neg):
                        train_data_neg = train_data_neg.sample(len(train_data_pos), random_state=1)
                    train_datas.append(train_data_neg)
                train_data_out = pd.concat(train_datas)
                train_data_out.loc[train_data_out['label'] > 0, 'label'] = 1
            else:
                train_data_out = train_data_pd
            self.train_data = [item for _, item in train_data_out.iloc[:].iterrows()]
        elif self.run_type == 'test':
            test_data_pd = pd.read_pickle(self.root + self.lang + '/test/blocks.pkl').sample(frac=1)
            if self.lang == 'java' and self.code_type in [1, 2, 3, 4, 5]:
                test_data_pos = test_data_pd[test_data_pd['label'].isin([0])]
                test_data_neg = test_data_pd[test_data_pd['label'].isin([self.code_type])]
                # if len(test_data_pos) < len(test_data_neg):
                #     test_data_neg = test_data_neg.sample(len(test_data_pos), random_state=1)
                # else:
                #     test_data_pos = test_data_pos.sample(len(test_data_neg), random_state=1)
                test_data_out = pd.concat([test_data_pos, test_data_neg])
                test_data_out.loc[test_data_out['label'] > 0, 'label'] = 1
            elif self.lang == 'java' and self.code_type == 6:
                test_data_pos = test_data_pd[test_data_pd['label'].isin([0])]
                test_datas = [test_data_pos]
                for i in range(5):
                    test_data_neg = test_data_pd[test_data_pd['label'].isin([i+1])]
                    # if len(test_data_pos) < len(test_data_neg):
                    #     test_data_neg = test_data_neg.sample(len(test_data_pos), random_state=1)
                    test_datas.append(test_data_neg)
                test_data_out = pd.concat(test_datas)
                test_data_out.loc[test_data_out['label'] > 0, 'label'] = 1
            else:
                test_data_out = test_data_pd
            self.test_data = [item for _, item in test_data_out.iloc[:].iterrows()]
        else:
            ValueError('Do Not Exist Run Type')

    def __len__(self):
        if self.run_type == 'train':
            return int(len(self.train_data))
        else:
            return int(len(self.test_data))

    def __getitem__(self, idx):
        sample = dict()
        if self.run_type == 'train':
            sample['id_x'] = self.train_data[idx]['id1']
            sample['id_y'] = self.train_data[idx]['id2']
            sample['code_x'] = self.train_data[idx]['code_x']
            sample['code_y'] = self.train_data[idx]['code_y']
            sample['labels'] = self.train_data[idx]['label']
        elif self.run_type == 'test':
            sample['id_x'] = self.test_data[idx]['id1']
            sample['id_y'] = self.test_data[idx]['id2']
            sample['code_x'] = self.test_data[idx]['code_x']
            sample['code_y'] = self.test_data[idx]['code_y']
            sample['labels'] = self.test_data[idx]['label']
        return sample