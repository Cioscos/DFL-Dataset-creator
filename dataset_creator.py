from pathlib import Path
import os
import shutil
import argparse
import math
import multiprocessing as mp
from tqdm import tqdm

from xlib.DFLIMG.DFLJPG import DFLJPG
from xlib.facelib import LandmarksProcessor
from xlib import joblib
from xlib.interact import interact as io

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG'
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

class DataImage:
    def __init__(self, yaw, pitch) -> None:
        self.yaw = yaw
        self.pitch = pitch

def make_dataset(dir: str) -> list[Path]:
    files = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in os.walk(dir):
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
        return
    pitch, yaw, _ = LandmarksProcessor.estimate_pitch_yaw_roll ( dflimg.get_landmarks(), size=dflimg.get_shape()[1] )
    return [path, DataImage(yaw, pitch)]

def cpu_number(list_size, slice_count=2000):
    sliced_count = list_size // slice_count

    if sliced_count > 12:
        sliced_count = 11.9
        slice_count = int(list_size / sliced_count)
        sliced_count = list_size // slice_count

    return sliced_count if sliced_count != 0 else 1

class YawPitchComparatorSubprocessor(joblib.Subprocessor):
    class Cli(joblib.Subprocessor.Cli):
        def _round_up(self, n, decimals=0):
            multiplier = 10 ** decimals
            return math.ceil(n * multiplier) / multiplier

        #override
        def on_initialize(self, client_dict):
            self.angle_match = client_dict['angle_match']

        #override
        def process_data(self, data):
            img_list = []
            for srcimg in data[0]:
                for dstimg in data[1]:
                    if math.isclose(self._round_up(srcimg[1].yaw, 2), self._round_up(dstimg[1].yaw, 2), abs_tol=self.angle_match) and \
                    math.isclose(self._round_up(srcimg[1].pitch, 2), self._round_up(dstimg[1].pitch, 2), abs_tol=self.angle_match):
                        img_list.append(srcimg[0])
                self.progress_bar_inc(len(data[1]))

            return img_list

        #override
        def get_data_name (self, data):
            return "Bunch of images"

    #override
    def __init__(self, src_list, dst_list, angle_match=0.05, cpus=mp.cpu_count()):
        self.src_list = src_list
        self.dst_list = dst_list
        self.src_list_len = len(self.src_list)
        self.dst_list_len = len(self.dst_list)
        self.angle_match = angle_match
        self.cpus = cpus

        slice_count = self.src_list_len // cpus
        sliced_count = self.src_list_len // slice_count

        if sliced_count > cpus:
            sliced_count = 11.9
            slice_count = int(self.src_list_len / sliced_count)
            sliced_count = self.src_list_len // slice_count

        self.img_chunks_list = []

        # SRC
        src_chunks_list = [ self.src_list[i*slice_count : (i+1)*slice_count] for i in range(sliced_count) ] + \
                            [ self.src_list[sliced_count*slice_count:] ]

        for chunk in src_chunks_list:
            self.img_chunks_list.append( [chunk, self.dst_list] )

        self.result = []
        super().__init__('YawPitchComparator', YawPitchComparatorSubprocessor.Cli, 0)

    #override
    def process_info_generator(self):
        cpu_count = len(self.img_chunks_list)
        print(f"Running on {cpu_count} {'threads' if cpu_count > 1 else 'thread'}")
        for i in range(cpu_count):
            yield 'CPU%d' % (i), {'i':i}, {'angle_match':self.angle_match}

    #override
    def on_clients_initialized(self):
        io.progress_bar ("Calculating data", self.src_list_len*self.dst_list_len)
        io.progress_bar_inc(len(self.img_chunks_list))

    #override
    def on_clients_finalized(self):
        io.progress_bar_close()

    #override
    def get_data(self, host_dict):
        if len (self.img_chunks_list) > 0:
            return self.img_chunks_list.pop(0)
        return None

    #override
    def on_data_return (self, host_dict, data):
        raise Exception("Fail to process data. Decrease number of images and try again.")

    #override
    def on_result (self, host_dict, data, result):
        self.result += result
        return 0

    #override
    def get_result(self):
        return self.result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--src', type=str, dest='src', required=True, help='Folder with source aligned images')
    parser.add_argument('-d', '--dst', type=str, dest='dst', required=True, help='Folder with dstination aligned images')
    parser.add_argument('-o', '--output', type=str, dest='output', default='Dataset', help='Folder with final dataset')
    parser.add_argument('-a', '--angle-match', type=float, dest='angle_match', default=0.05, help='Indicates the minimum value difference required for two values to be equal.')
    args = parser.parse_args()

    srcset = make_dataset(args.src)
    dstset = make_dataset(args.dst)

    cpus = io.input_int('Insert number of CPUs to use',
                    help_message='If the default option is selected it will use all cpu cores and it will slow down pc',
                    default_value=mp.cpu_count())
    # Elaborate srcset
    with mp.Pool(processes=cpus) as p:
        final_srcset = list(tqdm(p.imap_unordered(process_yaw_pitch_file, srcset),desc=f"Calculating data with {cpus} {'cpus' if cpus > 1 else 'cpu'}", total=len(srcset), ascii=True))

    # Elaborate dstset
    with mp.Pool(processes=cpus) as p:
        final_dstset = list(tqdm(p.imap_unordered(process_yaw_pitch_file, dstset),desc=f"Calculating data with {cpus} {'cpus' if cpus > 1 else 'cpu'}", total=len(dstset), ascii=True))

    dataset = YawPitchComparatorSubprocessor(final_srcset, final_dstset, angle_match=args.angle_match, cpus=cpus).run()

    # remove duplicates
    dataset = set(dataset)

    os.makedirs(args.output, exist_ok=True)

    for path in tqdm(dataset, desc='Moving files', ascii=True):
        shutil.copy(path, args.output)

if __name__ == "__main__":
    main()
