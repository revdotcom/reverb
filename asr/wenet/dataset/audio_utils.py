import os
import sys
import torch
import torchaudio
import torchaudio.backend.sox_io_backend as sox
import subprocess
import shlex
import io
import numpy as np
from functools import lru_cache
# for debugging purposes
# import psutil

def bytes_to_list_of_int16(bytes):
    return np.frombuffer(bytes, dtype=np.int16)

# execute a shell command and read all stdout output to a string
def read_cmd_output_old(cmd):
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    out = p.communicate()[0]
    ret = p.returncode
    p = None

    if(ret!= 0):
        return None
    return out

def read_cmd_output(cmd):
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, universal_newlines=False)
    out = p.stdout.read()
    # ret = p.returncode
    p = None

    # print(f"len out = {len(out)}, ret = {ret}")
    # if(ret!= 0):
    #     return None
    return out

# @lru_cache(maxsize=5)
def get_wavdata_and_samplerate(wavscp_filedescriptor):
    """
    Get wav data and sample rate from wavscp line
    """
    wavscp_filedescriptor = wavscp_filedescriptor.strip()
    if wavscp_filedescriptor[-1] == '|': # this is a pipe command
        # here, we assume that the output of the pipe command will be a format torchaudio can load
        tmp_x = read_cmd_output(wavscp_filedescriptor.strip()[:-1])
        ff = io.BytesIO(tmp_x)
        ff.seek(0)
        if ff.getbuffer().nbytes == 0:
            print(f"io.bytesio len {ff.getbuffer().nbytes}--> {wavscp_filedescriptor}")
        if tmp_x is None:
            return None, None

        try:
            wav_data, sample_rate = torchaudio.load(ff)
            # print(f"wav data shape {wav_data.shape}")
        except:
            print(f"Error loading wav file {wavscp_filedescriptor}")
            return None, None
    else:
        try:
            wav_data, sample_rate = torchaudio.load(wavscp_filedescriptor)
            # print(f"wav data shape {wav_data.shape}")
        except:
            print(f"Error loading wav file {wavscp_filedescriptor}")
            return None, None

    # useful for debugging purposes
    # print(f"Memory used: {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2} MB", flush=True)
    return wav_data, sample_rate