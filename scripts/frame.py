#!/usr/bin/env python3

import math
import pandas
import torch
import torch.utils.data
import torchvision
import echonet
import os
import tqdm
import numpy as np
import scipy
import matplotlib.pyplot as plt
import concurrent.futures
import pickle

def main():
    output = "frame"
    block_size = 1024
    os.makedirs(os.path.join(output, "trace"), exist_ok=True)
    os.makedirs(os.path.join(output, "size"), exist_ok=True)
    os.makedirs(os.path.join(output, "lstm"), exist_ok=True)

    if False:
        save_segmentations(output, block_size)
    basename = [os.path.splitext(filename)[0] for filename in sorted(os.listdir(os.path.join(output, "trace")))]

    try:
        with open(os.path.join(output, "size.pkl"), "rb") as f:
            size = pickle.load(f)
    except:
        size = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            for s in tqdm.tqdm(executor.map(get_size, [os.path.join(output, "trace", b + ".npy") for b in  basename]), total=len(basename)):
                size.append(s)
        with open(os.path.join(output, "size.pkl"), "wb") as f:
            pickle.dump(size, f)
    # import collections
    # print(collections.Counter([s.shape[0] for s in size]))
    max_length = max([s.shape[0] for s in size])
    batch_size = 512
    num_workers = 0
    device = torch.device("cuda")

    dataset = echonet.datasets.Echo(split="all")
    systole_r = [dataset.frames[b][0] for b in basename]
    diastole_r = [dataset.frames[b][-1] for b in basename]
    with open(os.path.join(echonet.config.DATA_DIR, "FileList.csv")) as f:
        data = pandas.read_csv(f)
    data["FileName"] = data["FileName"].map(lambda x: os.path.splitext(x)[0])
    split = {split: set(data[data["Split"] == split]["FileName"]) for split in data["Split"].unique()}

    mask = [b in split["TRAIN"] for b in basename]
    size_train = np.concatenate([x for (x, m) in zip(size, mask) if m])
    mean = size_train.mean()
    std = size_train.std()
    dataset = {}
    for s in split:
        mask = [b in split[s] for b in basename]
        dataset[s] = Dataset(
            [x for (x, m) in zip(basename, mask) if m],
            [x for (x, m) in zip(size, mask) if m],
            [x for (x, m) in zip(systole_r, mask) if m],
            [x for (x, m) in zip(diastole_r, mask) if m],
            max_length=max_length,
            mean=mean,
            std=std)

    p = 1 / 50
    model = Sequence()
    model.linear.bias.data[:] = math.log(p)
    model.to(device)
    try:
        raise ValueError()
        model.load_state_dict(torch.load(os.path.join(output, "model.pt")))
    except:
        optim = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=0)
        for epoch in range(100):
            for s in ["TRAIN", "VAL"]:
                print("Epoch #{} {}".format(epoch, s))
                dataloader = torch.utils.data.DataLoader(dataset[s], batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=(device.type == "cuda"), drop_last=True)
                total = 0.
                n = 0
                with tqdm.tqdm(total=len(dataloader)) as pbar:
                    for (_, x, y) in dataloader:
                        x = x.to(device)
                        y = y.to(device)
                        mask = ~torch.logical_or(torch.isnan(torch.stack((x, x), dim=1)), torch.isnan(y))
                        x[torch.isnan(x)] = -10
                        yhat = model(x)
                        loss = torch.nn.functional.binary_cross_entropy_with_logits(yhat[mask], y[mask], reduction="sum")
                        if s == "TRAIN":
                            optim.zero_grad()
                            loss.backward()
                            optim.step()

                        # torch.nn.binary_cross_entropy_with_logits(input, target, weight=None, size_average=None, reduce=None, reduction='mean', pos_weight=None)
                        total += loss.item()
                        n += mask.sum().item()

                        pbar.set_description("Loss: {:.3f}".format(total / n))
                        pbar.update()
        torch.save(model.state_dict(), os.path.join(output, "model.pt"))

    model.eval()
    dataloader = torch.utils.data.DataLoader(dataset["TEST"], batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=(device.type == "cuda"), drop_last=False)
    total = 0.
    n = 0
    systole_p = []
    diastole_p = []
    with torch.no_grad():
        with tqdm.tqdm(total=len(dataloader)) as pbar:
            for (filename, x, y) in dataloader:
                x = x.to(device)
                y = y.to(device)
                yhat = model(x)
                mask = ~torch.logical_or(torch.isnan(torch.stack((x, x), dim=1)), torch.isnan(y))
                yhat = model(x)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(yhat[mask], y[mask], reduction="sum")
                prob = torch.sigmoid(yhat)

                for (f, size, p) in zip(filename, x.cpu(), prob.cpu()):
                    systole_p.append(list(scipy.signal.find_peaks(p[0, :], height=0.1, distance=5)[0]))
                    diastole_p.append(list(scipy.signal.find_peaks(p[1, :], height=0.1, distance=5)[0]))
                    if f == "0X1002E8FBACD08477":
                        asd
                        fig, ax = plt.subplots(3, figsize=(8, 6))

                        ax[0].plot(std * size + mean)

                        ax[1].plot(p[0, :])
                        ylim = ax[1].get_ylim()
                        for x in systole_p[-1]:
                            ax[1].plot([x, x], ylim, linewidth=1, color="k")
                        ax[1].set_ylim(ylim)

                        ax[2].plot(p[1, :])
                        ylim = ax[1].get_ylim()
                        for x in diastole_p[-1]:
                            ax[2].plot([x, x], ylim, linewidth=1, color="k")
                        ax[2].set_ylim(ylim)

                        plt.tight_layout()
                        plt.savefig(os.path.join(output, "lstm", f + ".pdf"))
                        plt.close(fig)
                        print(f)

                # torch.nn.binary_cross_entropy_with_logits(input, target, weight=None, size_average=None, reduce=None, reduction='mean', pos_weight=None)
                total += loss.item()
                n += mask.sum().item()

                pbar.set_description("Loss: {:.3f}".format(total / n))
                pbar.update()

    ans = []
    mask = [b in split["TEST"] for b in basename]
    for (filename, pred, real) in zip([x for (x, m) in zip(basename, mask) if m], systole_p, [x for (x, m) in zip(systole_r, mask) if m]):
        errors = [p - real for p in pred]
        close = [e for e in errors if abs(e) <= 25]
        if len(close) == 1:
            ans.append(abs(close[0]))
        else:
            ans.append(25)
    print("Systole:", sum(ans) / len(ans))

    ans = []
    mask = [b in split["TEST"] for b in basename]
    for (filename, pred, real) in zip([x for (x, m) in zip(basename, mask) if m], diastole_p, [x for (x, m) in zip(diastole_r, mask) if m]):
        errors = [p - real for p in pred]
        close = [e for e in errors if abs(e) <= 25]
        if len(close) == 1:
            ans.append(abs(close[0]))
        else:
            print(filename, len(close))
            ans.append(25)
    print("Diastole:", sum(ans) / len(ans))

    return

    # src_des = [(os.path.join(output, "trace", filename), os.path.join(output, "size", os.path.splitext(filename)[0] + ".pdf")) for filename in os.listdir(os.path.join(output, "trace"))]
    systole_p = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        for s in tqdm.tqdm(executor.map(heuristic, size), total=len(size)):
        # for filename in tqdm.tqdm(map(heuristic, src_des), total=len(src_des)):
            systole_p.append(s)

    ans = []
    ans = []
    for (filename, pred, real, s) in zip(basename, systole_p, systole_r, size):
        errors = [p - real for p in pred]
        close = [e for e in errors if abs(e) <= 25]
        if len(close) == 1:
            ans.append(close[0])
            if abs(close[0]) > 15:
                # print(filename, close[0])
                plot(s, filename, output, real)
        # else:
        #     ans.append(25)
    offset = sum(ans) / len(ans)
    fig = plt.figure(figsize=(3, 3))
    plt.hist(ans, bins=range(-25, 26))
    plt.tight_layout()
    plt.savefig("asd.pdf")
    plt.close(fig)
    print("offset", offset)

    ans = []
    ans_offset = []
    for (filename, pred, real, s) in zip(basename, systole_p, systole_r, size):
        errors = [p - real for p in pred]
        close = [e for e in errors if abs(e) <= 25]
        # print(len(close))
        if len(close) == 1:
            ans.append(abs(close[0]))
            ans_offset.append(abs(close[0] - offset))
        else:
            ans.append(25)
    print("Heuristic:", sum(ans) / len(ans))
    print("Heuristic (offset):", sum(ans_offset) / len(ans_offset))

class Dataset(torch.utils.data.Dataset):
    def __init__(self, filename, size, systole, diastole, max_length, mean=0, std=1, tol=25):
        self.filename = filename
        self.size = size
        self.systole = systole
        self.diastole = diastole
        self.max_length = max_length
        self.mean = mean
        self.std = std
        self.tol = tol

    def __getitem__(self, index):
        x = self.size[index]
        x = x.astype(np.float32)
        x -= self.mean
        x /= self.std
        x = np.concatenate((x, np.full(self.max_length - x.shape[0], math.nan, dtype=np.float32)))

        y = np.full((2, x.shape[0]), math.nan)
        y[0, max(0, self.systole[index] - 25):self.systole[index] + 25] = 0
        y[0, self.systole[index]] = 1
        y[1, max(0, self.diastole[index] - 25):self.diastole[index] + 25] = 0
        y[1, self.diastole[index]] = 1

        return self.filename[index], x, y

    def __len__(self):
        return len(self.size)

class Sequence(torch.nn.Module):
    # Based on https://github.com/pytorch/examples/blob/master/time_sequence_prediction/train.py
    def __init__(self, dim=64):
        super(Sequence, self).__init__()
        self.dim = dim
        self.lstm1 = torch.nn.LSTMCell(1, dim)
        self.lstm2 = torch.nn.LSTMCell(dim, dim)
        self.linear = torch.nn.Linear(dim, 2)
        self.h_t = torch.nn.Parameter(torch.zeros(1, self.dim))
        self.c_t = torch.nn.Parameter(torch.zeros(1, self.dim))
        self.h_t2 = torch.nn.Parameter(torch.zeros(1, self.dim))
        self.c_t2 = torch.nn.Parameter(torch.zeros(1, self.dim))

    def forward(self, input):
        outputs = []
        h_t = self.h_t.repeat(input.size(0), 1)
        c_t = self.c_t.repeat(input.size(0), 1)
        h_t2 = self.h_t2.repeat(input.size(0), 1)
        c_t2 = self.c_t2.repeat(input.size(0), 1)

        for (i, input_t) in enumerate(input.split(1, dim=1)):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        outputs = torch.stack(outputs, dim=2)
        return outputs


def save_segmentations(output, block_size):
    device = torch.device("cuda")
    # Set up Model
    model = torchvision.models.segmentation.deeplabv3_resnet50(aux_loss=False)

    model.classifier[-1] = torch.nn.Conv2d(model.classifier[-1].in_channels, 1, kernel_size=model.classifier[-1].kernel_size)  # change number of outputs to 1
    if device.type == "cuda":
        model = torch.nn.DataParallel(model)
    model.to(device)

    checkpoint = torch.load("deeplabv3_resnet50_random.pt")
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # Compute dataset statistics
    mean, std = echonet.utils.get_mean_and_std(echonet.datasets.Echo(split="train"))

    # Set up dataloader
    def collate_fn(x):
        """Collate function for Pytorch dataloader to merge multiple videos.

        The dataset is expected to return a tuple containing the video, along with some (non-zero) additional features.
        This function returns a 3-tuple:
            - Videos concatenated along the frames dimension
            - The additional features converted into matrix form
            - Lengths of the videos
        """
        x, f = zip(*x)  # Split the input into the videos and the additional features
        f = zip(*f)  # Swap features back into original form (this and the previous line essentially do two transposes)
        i = list(map(lambda t: t.shape[1], x))  # Create a list of videos lengths (in frames)
        x = torch.as_tensor(np.swapaxes(np.concatenate(x, 1), 0, 1))  # Concatenate along frames dimension, and swap to be first dimension
        return x, f, i

    ds = echonet.datasets.Echo(split="all", target_type=["Filename", "LargeIndex", "SmallIndex"], mean=mean, std=std, length=None, max_length=None, period=1)
    done = set([os.path.splitext(x)[0] for x in os.listdir(os.path.join(output, "trace"))])
    mask = [os.path.splitext(x)[0] not in done for x in ds.fnames]
    ds.fnames = [x for (x, m) in zip(ds.fnames, mask) if m]
    ds.outcome = [x for (x, m) in zip(ds.outcome, mask) if m]
    # ds = echonet.datasets.Echo(split="test", target_type=["Filename", "LargeIndex", "SmallIndex"], mean=mean, std=std, length=None, max_length=None, period=1)
    dataloader = torch.utils.data.DataLoader(ds,
                                             batch_size=10, num_workers=5, shuffle=False, pin_memory=False, collate_fn=collate_fn)

    with torch.no_grad():
        for (x, (filenames, large_index, small_index), length) in tqdm.tqdm(dataloader):
            print(x.shape, flush=True)
            # Predict segmentation in blocks (only send subset of frames to gpu at a time)
            y = np.concatenate([model(x[i:(i + block_size), :, :, :].to(device))["out"].detach().cpu().numpy() for i in range(0, x.shape[0], block_size)])
            start = 0
            for (i, (filename, offset)) in enumerate(zip(filenames, length)):
                # Extract one video and segmentation predictions
                logit = y[start:(start + offset), 0, :, :]
                np.save(os.path.join(output, "trace", os.path.splitext(filename)[0] + ".npy"), logit)
                start += offset


def get_size(filename):
    logit = np.load(filename)
    size = (logit > 0).sum((1, 2))
    return size

def heuristic(size):
    # Identify systole frames with peak detection
    trim_min = sorted(size)[round(len(size) ** 0.05)]
    trim_max = sorted(size)[round(len(size) ** 0.95)]
    trim_range = trim_max - trim_min
    systole = set(scipy.signal.find_peaks(-size, distance=20, prominence=(0.50 * trim_range))[0])
    return systole

def plot(size, filename, output, systole):
    # Plot sizes
    fig = plt.figure(num=filename, figsize=(size.shape[0] / 50 * 5, 3))
    plt.scatter(np.arange(size.shape[0]) / 50, size, s=1)
    ylim = plt.ylim()
    s = systole
    plt.plot(np.array([s, s]) / 50, ylim, linewidth=1)
    systole = heuristic(size)
    for s in systole:
        plt.plot(np.array([s, s]) / 50, ylim, "--", linewidth=1)
    plt.ylim(ylim)
    plt.title(filename)
    plt.xlabel("Seconds")
    plt.ylabel("Size (pixels)")
    plt.tight_layout()
    plt.savefig(os.path.join(output, "size", filename + ".pdf"))
    plt.close(fig)

if __name__ == "__main__":
    main()
