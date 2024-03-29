from pathlib import Path
import os
import operator
import shutil
import argparse
import math
import multiprocessing as mp

import numpy as np
from tqdm import tqdm

from xlib.DFLIMG.DFLJPG import DFLJPG
from xlib.facelib import LandmarksProcessor
from xlib import joblib
from xlib.interact import interact as io

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


class DataImage:
    def __init__(self, yaw, pitch) -> None:
        self.yaw = yaw
        self.pitch = pitch


def make_dataset(directory: str):
    files = []
    assert os.path.isdir(directory), '%s is not a valid directory' % directory

    for root, _, fnames in os.walk(directory):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                files.append(Path(path))

    return files


def process_yaw_pitch_file(name):
    path = Path(name)
    dflimg = DFLJPG.load(path)
    if dflimg is None or not dflimg.has_data():
        print(f"{path.name} is not a DFL image file. Skipping it...")
        return []
    pitch, yaw, _ = LandmarksProcessor.estimate_pitch_yaw_roll(dflimg.get_landmarks(), size=dflimg.get_shape()[1])
    return [path, yaw, pitch]


class YawPitchComparatorSubprocessor(joblib.Subprocessor):
    class Cli(joblib.Subprocessor.Cli):
        def _round_up(self, n, decimals=0):
            multiplier = 10 ** decimals
            return math.ceil(n * multiplier) / multiplier

        # override
        def on_initialize(self, client_dict):
            self.angle_match = client_dict['angle_match']

        # override
        def process_data(self, data):
            img_list = []

            n = len(data[0])
            for i in range(n):
                yaw_src = data[0][i]
                if yaw_src is not None:
                    yaw_dst = data[1][i]
                    if yaw_dst is not None:
                        for srcimg in yaw_src:
                            for dstimg in yaw_dst:
                                if math.isclose(self._round_up(srcimg[2], 2), self._round_up(dstimg[2], 2),
                                                abs_tol=self.angle_match):
                                    img_list.append(srcimg[0])
                                    break

            return img_list

        # override
        def get_data_name(self, data):
            return "Bunch of images"

    # override
    def __init__(self, src_list, dst_list, angle_match=0.05, cpus=mp.cpu_count()):
        self.src_list = src_list
        self.dst_list = dst_list
        self.src_list_len = len(self.src_list)
        self.angle_match = angle_match
        self.cpus = cpus

        slice_count = self.src_list_len // cpus
        sliced_count = 1 if slice_count == 0 else self.src_list_len // slice_count

        if sliced_count > cpus: sliced_count = cpus

        self.img_chunks_list = []

        grads = 128
        grads_space = np.linspace(-1.2, 1.2, grads)
        yaws_sample_list_src = [None] * grads

        for g in io.progress_bar_generator(range(grads), "Chunking src"):
            yaw = grads_space[g]
            next_yaw = grads_space[g + 1] if g < grads - 1 else yaw

            yaw_samples = []
            for img in self.src_list:
                s_yaw = -img[1]
                if (g == 0 and s_yaw < next_yaw) or \
                        (g < grads - 1 and yaw <= s_yaw < next_yaw) or \
                        (g == grads - 1 and s_yaw >= yaw):
                    yaw_samples += [img]
            if len(yaw_samples) > 0:
                yaws_sample_list_src[g] = yaw_samples

        yaws_sample_list_dst = [None] * grads
        for g in io.progress_bar_generator(range(grads), "Chunking dst"):
            yaw = grads_space[g]
            next_yaw = grads_space[g + 1] if g < grads - 1 else yaw

            yaw_samples = []
            for img in self.dst_list:
                s_yaw = -img[1]
                if (g == 0 and s_yaw < next_yaw) or \
                        (g < grads - 1 and yaw <= s_yaw < next_yaw) or \
                        (g == grads - 1 and s_yaw >= yaw):
                    yaw_samples += [img]
            if len(yaw_samples) > 0:
                yaws_sample_list_dst[g] = yaw_samples

        # SRC
        if sliced_count > 1:

            src_chunks_list = np.array_split(yaws_sample_list_src, sliced_count)
            src_chunks_list = [list(x) for x in src_chunks_list]

            dst_chunks_list = np.array_split(yaws_sample_list_dst, sliced_count)
            dst_chunks_list = [list(x) for x in dst_chunks_list]

            for src_chunk, dst_chunk in zip(src_chunks_list, dst_chunks_list):
                self.img_chunks_list.append([src_chunk, dst_chunk])
        else:
            src_chunks_list = yaws_sample_list_src
            dst_chunks_list = yaws_sample_list_dst

            self.img_chunks_list.append([src_chunks_list, dst_chunks_list])

        self.result = []
        io.log_info("Calculating images match...")
        super().__init__('YawPitchComparator', YawPitchComparatorSubprocessor.Cli, 0)

    # override
    def process_info_generator(self):
        cpu_count = len(self.img_chunks_list)
        print(f"Matching images on {cpu_count} {'threads' if cpu_count > 1 else 'thread'}")
        for i in range(cpu_count):
            yield 'CPU%d' % i, {'i': i}, {'angle_match': self.angle_match}

    # override
    def get_data(self, host_dict):
        if len(self.img_chunks_list) > 0:
            return self.img_chunks_list.pop(0)
        return None

    # override
    def on_data_return(self, host_dict, data):
        raise Exception("Fail to process data. Decrease number of images and try again.")

    # override
    def on_result(self, host_dict, data, result):
        self.result += result
        return 0

    # override
    def get_result(self):
        return self.result


def main():
    # manage input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--src', type=str, dest='src', required=True, help='Folder with source aligned images')
    parser.add_argument('-d', '--dst', type=str, dest='dst', required=True,
                        help='Folder with destination aligned images')
    parser.add_argument('-o', '--output', type=str, dest='output', default='Dataset', help='Folder with final dataset')
    parser.add_argument('-a', '--angle-match', type=float, dest='angle_match', default=0.05,
                        help='Indicates the minimum value difference required for two values to be equal.')
    parser.add_argument('--cpus', type=int, dest='cpus', default=None, help='Number of cpus to use')
    args = parser.parse_args()

    # Create 2 lists with path of each file of the input folders
    srcset = make_dataset(args.src)
    dstset = make_dataset(args.dst)

    # number of cpus to use
    cpus = args.cpus

    if cpus is None:
        cpus = io.input_int('Insert number of CPUs to use',
                            help_message='If the default option is selected it will use all cpu cores and it will slow down pc',
                            default_value=mp.cpu_count())

    # Elaborate srcset
    with mp.Pool(processes=cpus) as p:
        final_srcset = list(tqdm(p.imap_unordered(process_yaw_pitch_file, srcset),
                                 desc=f"Calculating datasrc with {cpus} {'cpus' if cpus > 1 else 'cpu'}",
                                 total=len(srcset), ascii=True))
        final_srcset = [x for x in final_srcset if x]
        io.log_info('Sorting...')
        final_srcset = sorted(final_srcset, key=operator.itemgetter(1), reverse=True)

        # Elaborate dstset
        final_dstset = list(tqdm(p.imap_unordered(process_yaw_pitch_file, dstset),
                                 desc=f"Calculating datadst with {cpus} {'cpus' if cpus > 1 else 'cpu'}",
                                 total=len(dstset), ascii=True))
        final_dstset = [x for x in final_dstset if x]
        io.log_info('Sorting...')
        final_dstset = sorted(final_dstset, key=operator.itemgetter(1), reverse=True)

    # Subprocessor returns a list of image to move in the final dataset
    dataset = YawPitchComparatorSubprocessor(final_srcset, final_dstset, angle_match=args.angle_match, cpus=cpus).run()

    os.makedirs(args.output, exist_ok=True)

    for path in tqdm(dataset, desc='Copying files', ascii=True):
        shutil.copy(path, args.output)


if __name__ == "__main__":
    main()
