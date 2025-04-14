import os
import cv2
import sys
import pdb
import six
import glob
import time
import torch
import random
import pandas
import warnings
import time

warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
# import pyarrow as pa
from PIL import Image
import torch.utils.data as data
import matplotlib.pyplot as plt
from utils import video_augmentation,clean_phoenix_2014_trans,clean_phoenix_2014
from torch.utils.data.sampler import Sampler

sys.path.append("..")
global kernel_sizes 

class BaseFeeder(data.Dataset):
    def __init__(self, prefix, gloss_dict, dataset='phoenix2014', drop_ratio=1, num_gloss=-1, mode="train", transform_mode=True,
                 datatype="lmdb", frame_interval=1, image_scale=1.0, kernel_size=1, input_size=224):
        self.mode = mode
        self.ng = num_gloss
        self.prefix = prefix
        self.dict = gloss_dict
        self.data_type = datatype
        self.dataset = dataset
        self.input_size = input_size
        global kernel_sizes 
        kernel_sizes = kernel_size
        self.frame_interval = frame_interval # not implemented for read_features()
        self.image_scale = image_scale # not implemented for read_features()
        self.feat_prefix = f"{prefix}/features/fullFrame-256x256px/{mode}"
        #/data1/gsw/CSL-Daily/sentence/frames_512x512
        self.transform_mode = "train" if transform_mode else "test"
        #self.inputs_list = np.load(f"./preprocess/{dataset}/{mode}_info.npy", allow_pickle=True).item()
        full_inputs_dict = np.load(f"./preprocess/{dataset}/{mode}_info.npy", allow_pickle=True).item()
        self.inputs_list = dict(list(full_inputs_dict.items())[:]) #select length

        print(mode, len(self))
        self.data_aug = self.transform()
        print("")

    def __getitem__(self, idx):
        if self.data_type == "video":
            input_data, label, fi = self.read_video(idx)
            input_data, label = self.normalize(input_data, label)
            # input_data, label = self.normalize(input_data, label, fi['fileid'])
            return input_data, torch.LongTensor(label), fi #self.inputs_list[idx]['original_info']
        elif self.data_type == "lmdb":
            input_data, label, fi = self.read_lmdb(idx)
            input_data, label = self.normalize(input_data, label)
            return input_data, torch.LongTensor(label), self.inputs_list[idx]['original_info']
        else:
            input_data, label = self.read_features(idx)
            return input_data, label, self.inputs_list[idx]['original_info']

    def read_video(self, index):
        fi = self.inputs_list[index]
    
        if 'phoenix' in self.dataset:
#            frame_pattern = os.path.expanduser("~/SignGraph/phoenix2014-release/phoenix-2014-multisigner/features/fullFrame-210x260px/train/01June_2011_Wednesday_heute_default-5/1/*.png")
#            img_list = sorted(glob.glob(frame_pattern))
#            print(img_list)

#            print("[LOG] Using phoenix")

            self.prefix = os.path.expanduser("~/SignGraph/phoenix2014-release/phoenix-2014-multisigner")
            img_folder = os.path.join(self.prefix, "features", "fullFrame-210x260px", fi['folder'])
            # print('phoenix 데이터를 사용함')
            # print(img_folder)
#            print(f"len img_folder:{len(img_folder)}, type: {type(img_folder)}")
#            print(img_folder)
#            img_list = sorted(glob.glob(img_folder))
#            print(f"[DEBUG] Found {len(img_list)} frames")
#            print(len(img_list))
        elif self.dataset == 'CSL-Daily':
            img_folder = os.path.join(self.prefix, "sentence", "frames_512x512", fi['folder'])
    
        img_list = sorted(glob.glob(img_folder))

        # ✅ 여기에 경로 출력 추가
        # print(f"\n[READ_VIDEO] Mode: {self.mode}")
        # print(f"[READ_VIDEO] Folder path: {img_folder}")
        # print(f"[READ_VIDEO] Found {len(img_list)} frames")


        # if len(img_list) > 0:
        #     for i, frame_path in enumerate(img_list[:5]):
        #         print(f"  Frame {i+1}: {frame_path}")
        #     if len(img_list) > 5:
        #         print("  ...")
        
        if len(img_list) == 0:
            print(f"[WARNING] No frames found in: {img_list}")
    
        img_list = img_list[int(torch.randint(0, self.frame_interval, [1]))::self.frame_interval]
    
        label_list = []
        if self.dataset == 'phoenix2014':
            fi['label'] = clean_phoenix_2014(fi['label'])
        elif self.dataset == 'phoenix2014-T':
            fi['label'] = clean_phoenix_2014_trans(fi['label'])
    
        for phase in fi['label'].split(" "):
            if phase and phase in self.dict:
                label_list.append(self.dict[phase][0])
    
        video = [
            cv2.cvtColor(
                cv2.resize(cv2.imread(img_path), (256, 256), interpolation=cv2.INTER_LANCZOS4),
                cv2.COLOR_BGR2RGB
            )   
            for img_path in img_list
        ]
        # print(f"[FRAME COUNT] raw frame count: {len(img_list)}")

        return video, label_list, fi

    def read_features(self, index):
        # load file info
        fi = self.inputs_list[index]
        print('111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111')
        print('111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111')
        print(f"./features/{self.mode}/{fi['fileid']}_features.npy")
        print('111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111')
        print('111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111')
        time.sleep(10)
        data = np.load(f"./features/{self.mode}/{fi['fileid']}_features.npy", allow_pickle=True).item()
        return data['features'], data['label']

    def normalize(self, video, label, file_id=None):
        video, label = self.data_aug(video, label, file_id)
        video = video.float() / 127.5 - 1
        return video, label

    def transform(self):
        if self.transform_mode == "train":
            print("Apply training transform.")
            return video_augmentation.Compose([
                # video_augmentation.CenterCrop(224),
                # video_augmentation.WERAugment('/lustre/wangtao/current_exp/exp/baseline/boundary.npy'),
                video_augmentation.RandomCrop(self.input_size),
                video_augmentation.RandomHorizontalFlip(0.5),
                video_augmentation.Resize(self.image_scale),
                video_augmentation.ToTensor(),
                video_augmentation.TemporalRescale(0.2, self.frame_interval),
            ])
        else:
            print("Apply testing transform.")
            return video_augmentation.Compose([
                video_augmentation.CenterCrop(self.input_size),
                video_augmentation.Resize(self.image_scale),
                video_augmentation.ToTensor(),
            ])

    def byte_to_img(self, byteflow):
        unpacked = pa.deserialize(byteflow)
        imgbuf = unpacked[0]
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')
        return img

    @staticmethod
    def collate_fn(batch):
        # 길이 기준 정렬 (긴 영상 먼저)
        batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)
        videos, labels, infos = zip(*batch)

        global kernel_sizes
        left_pad = 0
        last_stride = 1
        total_stride = 1

        for ks in kernel_sizes:
            if ks[0] == 'K':
                left_pad = left_pad * last_stride
                left_pad += int((int(ks[1]) - 1) / 2)
            elif ks[0] == 'P':
                last_stride = int(ks[1])
                total_stride *= last_stride

        if len(videos[0].shape) > 3:  # [T, C, H, W]
            original_lens = [len(vid) for vid in videos]
            max_len = max(original_lens)
            video_lengths = torch.LongTensor([
                int(np.ceil(l / total_stride) * total_stride + 2 * left_pad)
                for l in original_lens
            ])
            right_pad = int(np.ceil(max_len / total_stride)) * total_stride - max_len + left_pad
            final_len = max_len + left_pad + right_pad

            padded_videos = []
            for vid in videos:
                padded = torch.cat([
                    vid[0].unsqueeze(0).expand(left_pad, -1, -1, -1),
                    vid,
                    vid[-1].unsqueeze(0).expand(final_len - len(vid) - left_pad, -1, -1, -1)
                ], dim=0)
                padded_videos.append(padded)

            padded_videos = torch.stack(padded_videos)  # [B, T, C, H, W]
        else:  # [T, C]
            original_lens = [len(vid) for vid in videos]
            max_len = max(original_lens)
            video_lengths = torch.LongTensor(original_lens)
            padded_videos = []
            for vid in videos:
                pad_len = max_len - len(vid)
                padded = torch.cat([vid, vid[-1].unsqueeze(0).expand(pad_len, -1)], dim=0)
                padded_videos.append(padded)
            padded_videos = torch.stack(padded_videos).permute(0, 2, 1)  # [B, C, T]

        # 라벨 padding
        label_lengths = torch.LongTensor([len(lab) for lab in labels])

        # CTCLoss를 위한 라벨은 1D로 concat된 상태여야 함
        if label_lengths.max() == 0:
            empty_label = torch.LongTensor([])
        else:
            empty_label = torch.cat([torch.LongTensor(lab) for lab in labels], dim=0)

        # 디버깅 로그
        if video_lengths.shape[0] != len(batch):
            print(f"⚠️ Warning: video_lengths({video_lengths.shape[0]}) != batch({len(batch)})")

        return padded_videos, video_lengths, empty_label, label_lengths, infos




    def __len__(self):
        return len(self.inputs_list) - 1

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

# class WordLevelGlossDataset(Dataset):
#     def __init__(self, annotation_list, image_root, transform):
#         self.samples = annotation_list
#         self.image_root = image_root
#         self.transform = transform

#     def __getitem__(self, idx):
#         ann = self.samples[idx]
#         folder = os.path.join(self.image_root, ann["sequence"], "1")
#         frame_paths = sorted(glob.glob(f"{folder}/*.png"))[ann["start"]:ann["end"]+1]

#         frames = [self.transform(Image.open(p).convert("RGB")) for p in frame_paths]
#         video_tensor = torch.stack(frames)  # [T, C, H, W]

#         return video_tensor, ann["gloss_id"]

if __name__ == "__main__":
    feeder = BaseFeeder()
    dataloader = torch.utils.data.DataLoader(
        dataset=feeder,
        batch_size=1,
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )
    for data in dataloader:
        pdb.set_trace()
