import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

CLASS_2_NUM = {'back': 0, 'default': 1, 'down': 2, 'front': 3, 'left': 4, 'right': 5, 'up': 6}
IM = np.eye(7)

def get_one_hot_encoding(clazz: str)-> np.ndarray:
    return IM[CLASS_2_NUM[clazz]]

class MyDataset(Dataset):
    def __init__(self, path: str = 'Dataset', frame_length: int = 32):
        # 파일 이름 배열
        self.filenames = []
        # 파일 카테고리
        self.category = []
        # 학습 데이터
        self.data = np.ndarray([0, frame_length, 5])
        # 타겟 라벨
        self.label = np.ndarray([0, 7])
        # 파일 경로 로딩
        for dirname, _, filenames in os.walk(path):
            if len(filenames) == 0: continue

            _, category = os.path.split(dirname)
            self.filenames += [os.path.join(dirname, filename) for filename in filenames]
            self.category += [category] * len(filenames)
        
        # csv 로딩
        for idx, file in enumerate(self.filenames):

            # (file_frame_length, 6)
            frame = pd.read_csv(file)

            # (file_frame_length, 5)
            frame = frame[['thumb', 'index', 'middle', 'ring', 'little']].to_numpy()

            # (file_frame_length + 50 > 160, 5)
            # frame[file_frame_length - 1] 값으로 패딩
            frame = np.pad(frame, ((0, 200), (0, 0)), mode='edge')
            
            batch = []
            for i in range(40, 168 - frame_length):
                batch += [frame[i: i + frame_length]]
                # batch += [frame[i + 32: i: -1]]
            # TODO: 프레임 1마다 한 주기씩 만들기
            # (128 / frame_length, frame_length, 5)
            batch = np.asarray(batch)
            # (total_file * 128 / frame_length, frame_length, 5)
            # print(batch.shape)
            self.data = np.append(self.data, batch, axis=0)
            
            # (total_file * 128 / frame_length, 6)
            self.label = np.append(self.label, [get_one_hot_encoding(self.category[idx])] * len(batch), axis=0)

        print(self.data.shape, self.label.shape)

        # 정규화
        mean = self.data.mean(axis=0)
        std = self.data.std(axis=0)
        
        self.data = (self.data - mean) / std

    def __len__(self)-> int:
        return len(self.data)
    
    def __getitem__(self, idx: int)-> tuple[torch.Tensor, torch.Tensor]:
        x = torch.FloatTensor(self.data[idx])
        y = torch.FloatTensor(self.label[idx])
        return x, y