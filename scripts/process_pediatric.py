#!/usr/bin/env python3

import collections
import pydicom
import io
import re
import PIL
import click
import cv2
import numpy as np
import os
import tqdm
import skimage.segmentation
import zipfile
import tarfile
import concurrent.futures

import echonet


@click.command()
@click.argument("src", type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument("dest", type=click.Path(file_okay=False))
def main(src, dest):
    os.makedirs(dest, exist_ok=True)
    with zipfile.ZipFile(src) as zf:
        with zf.open("Pediatric Echos/abnormals-deid/deid_coordinates.csv") as f:
            header = f.readline().decode("utf-8").strip().split(",")
            assert header == ['anon_patient_id', 'anon_accession_number', 'instance_number', 'frame_number', 'measurement', 'point_name', 'x', 'y']

            coordinates = collections.defaultdict(list)
            frame = {}
            measurement = {}
            error = set()
            for line in f:
                patient, accession, instance, f, m, point_name, x, y = line.decode("utf-8").strip().split(",")

                instance = int(instance)
                f = int(f)
                point_name = int(point_name)
                x = int(x)
                y = int(y)

                key = (patient, accession, instance, f)

                coordinates[key].append((point_name, x, y))

                if key in measurement:
                    if measurement[key] != m:
                        measurement[key] = "INVALID"
                        error.add(key)
                measurement[key] = m

            for key in coordinates:
                coordinates[key] = sorted(coordinates[key])


        test = collections.defaultdict(set)
        for (p, a, i, f) in measurement:
            test[(p, a, i)].add(measurement[(p, a, i, f)])
        print(collections.Counter(map(lambda x: tuple(sorted(x)), test.values())))

        view = {}
        for (p, a, i, f) in measurement:
            m = measurement[(p, a, i, f)]
            a4c = ("A4C" in m)
            psax = ("psax" in m or "PSAX" in m)

            assert a4c != psax

            if a4c:
                view[(p, a, i)] = "A4C"
            elif psax:
                view[(p, a, i)] = "PSAX"

        instance_of_view = collections.defaultdict(lambda: collections.defaultdict(list))
        for (p, a, i) in view:
            instance_of_view[(p, a)][view[(p, a, i)]].append(i)
        # for (p, a, i) in view:
        #     if view[(p, a, i)] == "A4C":
        #             # print("{}-{}/*/*/

        ef = {}
        with zf.open("Pediatric Echos/abnormals-deid/deid_measurements.csv") as f:
            header = f.readline().decode("utf-8").strip().split(",")
            assert header == ['anon_patient_id', 'anon_accession_number', 'lv_ef_bullet', 'lv_ef_mod_a4c', 'lv_ef_mod_bp', 'lv_area_d_a4c', 'lv_area_s_a4c', 'lv_area_d_psax_pap', 'lv_area_s_psax_pap', 'lv_vol_d_bullet', 'lv_vol_s_bullet']

            for line in f:
                patient, accession, ef_bullet, *_ = line.decode("utf-8").strip().split(",")
                try:
                    ef_bullet = float(ef_bullet)
                    ef[(patient, accession)] = ef_bullet
                except:
                    print(ef_bullet)

        patients = sorted(set(p for (p, a) in ef))
        index = {p: i for (i, p) in enumerate(patients)}
        split = {}
        for p in index:
            i = index[p] % 20
            if i < 14:
                split[p] = "TRAIN"
            elif i < 17:
                split[p] = "VAL"
            else:
                split[p] = "TEST"
        with open(os.path.join(dest, "FileList.csv"), "w") as f:
            f.write("FileName,EF,Split\n")
            for (p, a) in ef:
                for i in instance_of_view[(p, a)]["A4C"]:
                    f.write("{}-{}-{:06d}.avi,{},{}\n".format(p, a, i, ef[(p, a)], split[p]))

        iterables = [(zf, dest, filename) for filename in zf.namelist()]
        with tqdm.tqdm(total=len(iterables)) as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
                for _ in executor.map(save_tgz, iterables):
                # for _ in map(save_tgz, iterables):
                    pbar.update()
                    # save_tgz(zf, filename, dest)


    return
    for filename in tqdm.tqdm(sorted(os.listdir(src))):
        output = os.path.join(dest, os.path.splitext(filename)[0] + ".webm")
        if not os.path.isfile(output):
            capture = cv2.VideoCapture(os.path.join(src, filename))
            fps = capture.get(cv2.CAP_PROP_FPS)
            video = echonet.utils.loadvideo(os.path.join(src, filename))
            video = test(video)
            # video.save(os.path.join(dest, os.path.splitext(filename)[0] + ".png"))
            # continue
            echonet.utils.savevideo(output, video, fps)
        continue
        # if filename in ["VID11216.mp4", "VID11713.mp4", "VID14180.mp4", "VID14278.mp4", "VID14441.mp4", "VID1480.mp4", "VID1481.mp4", "VID16642.mp4", "VID16885.mp4"]:
        #     continue
        # output = os.path.join(dest, os.path.splitext(filename)[0] + ".avi")
        output = os.path.join(dest, filename)
        if not os.path.isfile(output):
            print(filename)
            capture = cv2.VideoCapture(os.path.join(src, filename))
            fps = capture.get(cv2.CAP_PROP_FPS)
            print(fps)
            video = echonet.utils.loadvideo(os.path.join(src, filename))
            print(video.shape)
            video = crop(video)
            video = resize(video)
            video = mask(video)
            echonet.utils.savevideo(output, video, fps)

def save_tgz(x):
    (zf, dest, filename) = x
    m = re.search("Pediatric Echos/abnormals-deid/dicom/(CR[0-9a-z]{7})-(CR[0-9a-z]{7}).tgz", filename)
    
    if m:
        patient, accession = m.groups()
        with zf.open(filename) as f:
            data = f.read()
    
        with tarfile.open(fileobj=io.BytesIO(data)) as tf:
            for dicom in tf.getmembers():
                if dicom.isfile():
                    InstanceNumber, SOPInstanceUID = os.path.basename(dicom.name).split("-")
                    assert len(InstanceNumber) == 6
    
                    # output = os.path.join(dest, "videos", "{}-{}".format(patient, accession), os.path.splitext(dicom.name)[0] + ".avi")
                    # os.makedirs(os.path.dirname(output), exist_ok=True)
                    output = [
                        os.path.join(dest, "videos-full", "{}-{}-{}.avi".format(patient, accession, InstanceNumber)),
                        os.path.join(dest, "videos-crop", "{}-{}-{}.avi".format(patient, accession, InstanceNumber)),
                        os.path.join(dest, "Videos", "{}-{}-{}.avi".format(patient, accession, InstanceNumber)),
                    ]
                    for o in output:
                        os.makedirs(os.path.dirname(o), exist_ok=True)
    
                    if not all(map(os.path.isfile, output)):
                        with tf.extractfile(dicom.name) as f:
                            ds = pydicom.dcmread(f)
    
                        fps = 50
                        try:
                            fps = ds.CineRate  # TODO CineRate frequently missing
                        except:
                            pass
    
                        # breakpoint()
                        try:
                            video = ds.pixel_array.transpose((3, 0, 1, 2))
                            video = video[:, :, ::-1, :]
                            video[1, :, :, :] = video[0, :, :, :]
                            video[2, :, :, :] = video[0, :, :, :]

                            echonet.utils.savevideo(output[0], video, fps=fps)

                            regions = ds.SequenceOfUltrasoundRegions
                            assert len(regions) == 1
                            x0 = regions[0].RegionLocationMinX0
                            y0 = regions[0].RegionLocationMinY0
                            x1 = regions[0].RegionLocationMaxX1
                            y1 = regions[0].RegionLocationMaxY1
                            video = video[:, :, y0:(y1 + 1), x0:(x1 + 1)]
                            echonet.utils.savevideo(output[1], video, fps=fps)

                            _, _, h, w = video.shape
                            video = video[:, :, :, ((w - h) // 2):(h + (w - h) // 2)]
                            video = np.array(list(map(lambda x: cv2.resize(x, (112, 112), interpolation=cv2.INTER_AREA), video.transpose((1, 2, 3, 0))))).transpose((3, 0, 1, 2))
                            echonet.utils.savevideo(output[2], video, fps=fps)
                        except Exception as e:
                            print(filename, dicom.name)
                            print(e, flush=True)
    else:
        assert filename in ["Pediatric Echos/abnormals-deid/README.txt",
                            "Pediatric Echos/abnormals-deid/deid_measurements.csv",
                            "Pediatric Echos/abnormals-deid/deid_coordinates.csv"]

def crop(video):
    h = video.shape[2]
    w = video.shape[3]
    start = round(0.6 * (w - h))
    return video[:, :, :, start:(start + h)]

def resize(video, size=(112, 112)):
    return np.array(list(map(lambda x: cv2.resize(x, size, interpolation=cv2.INTER_AREA), video.transpose((1, 2, 3, 0))))).transpose((3, 0, 1, 2))

def test(video):
    (c, f, h, w) = video.shape
    assert c == 3
    std = video.std(1)
    r = video.max(1) - video.min(1)
    r = r.sum(0)
    r = ((r > 0) * 255).astype(np.uint8)
    # r = skimage.segmentation.flood_fill(r, (h // 2, w // 2), 128)
    mask = skimage.segmentation.flood(r, (h // 2, w // 2))
    video[:, :, ~mask] = 0
    # return PIL.Image.fromarray(((r > 0) * 255).astype(np.uint8))
    return video


def mask(video):
    h = video.shape[2]
    w = video.shape[3]

    target = np.any(video > 10, (0, 1))

    n = h
    x, y = np.meshgrid(np.arange(n), np.arange(n))

    best = None
    n = 25
    for i in range(-n, n + 1, 2):
        for j in range(0, n + 1, 2):
            x_offset = w // 2 + i
            y_offset = j

            mask = np.logical_and(y - y_offset > x - w // 2 - i,
                                  y - y_offset > w - x - w // 2 + i)
            window = np.logical_and.reduce((
                    x_offset - 100 < x,
                    x < x_offset + 100,
                    y_offset < y,
                    y < y_offset + 100))

            score = target[np.logical_and(window, mask)].sum() / np.logical_and(window, mask).sum() + (~target[np.logical_and(window, ~mask)]).sum() / np.logical_and(window, ~mask).sum()
            score = (score, i, j)
            if best is None or score > best:
                best = score
                print(best)

    _, i, j = best
    x_offset = w // 2 + i
    y_offset = j
    mask = np.logical_and(y - y_offset > x - w // 2 - i,
                          y - y_offset > w - x - w // 2 + i)
    window = np.logical_and.reduce((
            x_offset - 100 < x,
            x < x_offset + 100,
            y_offset < y,
            y < y_offset + 100))

    #video[1, :] = (target * 255).astype(np.uint8)
    # video[2, :] = (window * 255).astype(np.uint8)
    # video[0, :, ~mask] = 255

    video[:, :, ~mask] = 0
    return video

if __name__ == "__main__":
    main()
