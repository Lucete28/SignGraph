class WordLevelGlossDataset(Dataset):
    def __init__(self, annotation_list, image_root, transform):
        self.samples = annotation_list
        self.image_root = image_root
        self.transform = transform

    def __getitem__(self, idx):
        ann = self.samples[idx]
        folder = os.path.join(self.image_root, ann["sequence"], "1")
        frame_paths = sorted(glob.glob(f"{folder}/*.png"))[ann["start"]:ann["end"]+1]

        frames = [self.transform(Image.open(p).convert("RGB")) for p in frame_paths]
        video_tensor = torch.stack(frames)  # [T, C, H, W]

        return video_tensor, ann["gloss_id"]
