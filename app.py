import gradio as gr
import os
import numpy as np
from transfer import run_test
from scipy.io.wavfile import write

def audioCaption(audio):
    text = run_test(audio)
    return text[0]


title = "Audio Caption"
description = "Use CNN-Transformer architecture to generate the description of target audio."

gr.Interface(
    fn=audioCaption,
    inputs=gr.inputs.Audio(type="numpy", label="Input"), 
    outputs="text",
    title=title,
    description=description,
    enable_queue=True
    ).launch(debug=True)

# file_dir = r'/data2/nfs/users/s_zhangyu/multimodal/dcase_CNN/data/test_data/test_0001.npy'
# test_data=np.load(file_dir)
# x=audioCaption(test_data) 
# print(x)