import os
import pdb
import torch
import torch.nn as nn


class GpuDataParallel(object):
    def __init__(self):
        self.gpu_list = []
        self.output_device = None

    def set_device(self, device):
        print("🚨 [set_device] device =", device)
        print("🚨 torch.cuda.device_count() =", torch.cuda.device_count())
        print("🚨 os.environ[LOCAL_RANK] =", os.environ.get("LOCAL_RANK"))
        if isinstance(device, int):
            torch.cuda.set_device(device)  # ✅ 가장 확실하게 현재 프로세스 GPU만 선택
            self.gpu_list = [device]
            self.output_device = device
            print(f"✅ Using GPU {device}")
            self.occupy_gpu(device)
        else:
            device = str(device)
            if device.lower() != 'none':
                # os.environ["CUDA_VISIBLE_DEVICES"] = device  # eg: "0,1"
                # DDP에서는 CUDA_VISIBLE_DEVICES 설정 금지
                torch.cuda.set_device(int(device))
                self.gpu_list = [int(device)]
                self.output_device = int(device)

                visible_devices = list(map(int, device.split(',')))
                self.gpu_list = list(range(len(visible_devices)))  # DataParallel expects local device indices
                self.output_device = self.gpu_list[0]
                print(f"✅ Using GPUs: {device} → Local Device IDs: {self.gpu_list}")
                self.occupy_gpu(self.gpu_list)
            else:
                self.gpu_list = []
                self.output_device = "cpu"
        device = int(device)
        if device >= torch.cuda.device_count():
            raise ValueError(f"❌ GPU device {device} is out of range. Available GPUs: {torch.cuda.device_count() - 1}")



    def model_to_device(self, model):
        # model = convert_model(model)
        model = model.to(self.output_device)
        if len(self.gpu_list) > 1:
            model = nn.DataParallel(
                model,
                device_ids=self.gpu_list,
                output_device=self.output_device)
        return model

    def data_to_device(self, data):
        if isinstance(data, torch.FloatTensor):
            return data.to(self.output_device)
        elif isinstance(data, torch.DoubleTensor):
            return data.float().to(self.output_device)
        elif isinstance(data, torch.ByteTensor):
            return data.long().to(self.output_device)
        elif isinstance(data, torch.LongTensor):
            return data.to(self.output_device)
        elif isinstance(data, list) or isinstance(data, tuple):
            return [self.data_to_device(d) for d in data]
        else:
            raise ValueError(data.shape, "Unknown Dtype: {}".format(data.dtype))

    def criterion_to_device(self, loss):
        return loss.to(self.output_device)

    def occupy_gpu(self, gpus=None):
        """
        Make program appear on nvidia-smi.
        """
        if gpus is None:
            gpus = []
        elif isinstance(gpus, int):  # ✅ 정수면 리스트로 감싸기
            gpus = [gpus]

        if len(gpus) == 0:
            torch.zeros(1).cuda()
        else:
            for g in gpus:
                torch.zeros(1).cuda(g)

