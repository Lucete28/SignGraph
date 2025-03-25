import os
import yaml
import torch
import whisper
import utils
import random
import shutil
import inspect
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from modules.resnet import resnet18  # 예제에서는 resnet18 사용
from modules import BiLSTMLayer, TemporalConv

class FeatureExtractor(nn.Module):
    """ 기존 모델에서 CNN + 1D Conv + BiLSTM 부분을 로드하여 특징 추출 """
    def __init__(self, model_path, num_classes, c2d_type="resnet18", conv_type="1d", hidden_size=1024):
        super(FeatureExtractor, self).__init__()

        # 2D CNN Feature Extractor
        self.conv2d = resnet18()  
        self.conv2d.fc = nn.Identity()

        # 1D Temporal Convolution
        self.conv1d = TemporalConv(input_size=512, hidden_size=hidden_size,
                                   conv_type=conv_type, use_bn=False,
                                   num_classes=num_classes)
        self.conv1d.kernel_size = [3, 5, 7]  # 🔥 오류 수정: kernel_size 속성 추가

        # BiLSTM Temporal Model
        self.temporal_model = BiLSTMLayer(rnn_type='LSTM', input_size=hidden_size, 
                                          hidden_size=hidden_size, num_layers=2, 
                                          bidirectional=True)

        # 기존 모델 가중치 로드
        self.load_pretrained_model(model_path)

    def load_pretrained_model(self, model_path):
        """ 기존 모델을 불러와 classifier 이전까지만 사용 """
        print(f"Loading pretrained model from {model_path}")
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        
        # 기존 모델에서 classifier 부분을 제외하고 불러옴
        model_dict = checkpoint['model_state_dict']
        feature_extract_keys = {k: v for k, v in model_dict.items() if "classifier" not in k}

        self.load_state_dict(feature_extract_keys, strict=False)

        # 특징 추출 부분 학습 안되도록 고정 (freeze)
        for param in self.conv2d.parameters():
            param.requires_grad = False
        for param in self.conv1d.parameters():
            param.requires_grad = False
        for param in self.temporal_model.parameters():
            param.requires_grad = False

        print("Feature extractor loaded and frozen.")

    def forward(self, x, len_x):
        with torch.no_grad():
            # CNN 특징 추출
            if len(x.shape) == 5:
                batch, temp, channel, height, width = x.shape
                framewise = self.conv2d(x.permute(0, 2, 1, 3, 4)).view(batch, temp, -1).permute(0, 2, 1)
            else:
                framewise = x

            # 1D Conv 처리
            conv1d_outputs = self.conv1d(framewise, len_x)
            x = conv1d_outputs['visual_feat']
            lgt = conv1d_outputs['feat_len'].cpu()

            # BiLSTM 처리
            tm_outputs = self.temporal_model(x, lgt)

        return tm_outputs['predictions'], lgt


class WhisperSLRModel(nn.Module):
    """ Whisper 디코더를 사용하여 gloss 예측 """
    def __init__(self, num_classes, hidden_size=1024):
        super(WhisperSLRModel, self).__init__()

        # Whisper 디코더 로드
        whisper_model = whisper.load_model("base")  
        self.whisper_decoder = whisper_model.decoder  

        # Gloss 토큰 임베딩 및 분류기
        self.token_embedding = nn.Embedding(num_classes, hidden_size)  
        self.output_proj = nn.Linear(hidden_size, num_classes)  

    def forward(self, feature_inputs, label=None):
        decoder_input = feature_inputs.permute(1, 0, 2)  # Whisper는 (B, T, C) 형식 기대
        
        # Gloss 토큰 임베딩
        if label is not None:
            label_embedding = self.token_embedding(label)  
        else:
            label_embedding = torch.zeros_like(decoder_input)  

        # Whisper 디코더 적용
        decoder_output = self.whisper_decoder(x=decoder_input, xa=label_embedding)

        # 최종 gloss 예측
        gloss_logits = self.output_proj(decoder_output)  
        return gloss_logits


def build_dataloader(args, dataset, mode, train_flag):
    """ Processor의 데이터 로더 로직을 참고하여 데이터 로더 생성 """
    batch_size = args.batch_size if mode == "train" else args.test_batch_size
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train_flag,
        drop_last=train_flag,
        num_workers=args.num_worker,
        collate_fn=dataset.collate_fn,
        pin_memory=False
    )


def train(model, feature_extractor, dataloader, optimizer, criterion, device):
    """ 학습 루프 """
    model.train()
    total_loss = 0

    for batch in dataloader:
        videos, len_x, labels = batch
        videos, len_x, labels = videos.to(device), len_x.to(device), labels.to(device)

        # 기존 모델에서 특징 추출
        feature_inputs, _ = feature_extractor(videos, len_x)

        # Whisper 기반 gloss 예측
        outputs = model(feature_inputs, label=labels)

        # Loss 계산
        loss = criterion(outputs.permute(1, 2, 0), labels)  
        total_loss += loss.item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return total_loss / len(dataloader)


if __name__ == "__main__":
    # YAML에서 설정 불러오기
    with open("configs/baseline.yaml", 'r') as f:
        args = yaml.load(f, Loader=yaml.FullLoader)

    num_classes = 500  
    hidden_size = 1024
    batch_size = 16
    learning_rate = 1e-4
    num_epochs = 10
    model_path = "_best_model.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 기존 모델에서 특징 추출하는 부분만 로드
    feature_extractor = FeatureExtractor(model_path, num_classes)
    feature_extractor.to(device)

    # Whisper 기반 SLR 모델 생성
    slr_model = WhisperSLRModel(num_classes, hidden_size)
    slr_model.to(device)

    # 옵티마이저 & 손실 함수
    optimizer = optim.Adam(slr_model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # 데이터 로더
    dataset = utils.load_dataset(args)  
    train_loader = build_dataloader(args, dataset["train"], "train", True)

    # 학습 시작
    for epoch in range(num_epochs):
        loss = train(slr_model, feature_extractor, train_loader, optimizer, criterion, device)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss:.4f}")

    # 최종 모델 저장
    torch.save(slr_model.state_dict(), "whisper_slr_model.pt")
