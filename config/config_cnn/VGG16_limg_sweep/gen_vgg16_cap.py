# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 16:39:12 2017

@author: zengh
"""

_NAME = "VGG16_Limg{}.yaml"

_template = """
name: VGG16_Limg_{name}
conv_layers:
  - name: conv1_1
    l_img: {c1_1}
    l_kern: 3
    f_in: 3
    f_out: 64
    stride: 1
    pad: 1
  - name: conv1_2
    l_img: {c1_2}
    l_kern: 3
    f_in: 64
    f_out: 64
    stride: 1
    pad: 1
  - name: conv2_1
    l_img: {c2_1}
    l_kern: 3
    f_in: 64
    f_out: 128
    stride: 1
    pad: 1
  - name: conv2_2
    l_img: {c2_2}
    l_kern: 3
    f_in: 128
    f_out: 128
    stride: 1
    pad: 1
  - name: conv3_1
    l_img: {c3_1}
    l_kern: 3
    f_in: 128
    f_out: 256
    stride: 1
    pad: 1
  - name: conv3_2
    l_img: {c3_2}
    l_kern: 3
    f_in: 256
    f_out: 256
    stride: 1
    pad: 1
  - name: conv3_3
    l_img: {c3_3}
    l_kern: 3
    f_in: 256
    f_out: 256
    stride: 1
    pad: 1
  - name: conv4_1
    l_img: {c4_1}
    l_kern: 3
    f_in: 256
    f_out: 512
    stride: 1
    pad: 1
  - name: conv4_2
    l_img: {c4_2}
    l_kern: 3
    f_in: 512
    f_out: 512
    stride: 1
    pad: 1
  - name: conv4_3
    l_img: {c4_3}
    l_kern: 3
    f_in: 512
    f_out: 512
    stride: 1
    pad: 1
  - name: conv5_1
    l_img: {c5_1}
    l_kern: 3
    f_in: 512
    f_out: 512
    stride: 1
    pad: 1
  - name: conv5_2
    l_img: {c5_2}
    l_kern: 3
    f_in: 512
    f_out: 512
    stride: 1
    pad: 1
  - name: conv5_3
    l_img: {c5_3}
    l_kern: 3
    f_in: 512
    f_out: 512
    stride: 1
    pad: 1
"""

def generate_file(limg_size):
    """
    limg_size is the image size for the last conv layer.
    """
    with open(_NAME.format(limg_size),'w') as f:
        f.write(_template.format(name=limg_size,
                                 c5_3=limg_size,
                                 c5_2=limg_size,
                                 c5_1=limg_size,
                                 c4_3=2*limg_size,
                                 c4_2=2*limg_size,
                                 c4_1=2*limg_size,
                                 c3_3=4*limg_size,
                                 c3_2=4*limg_size,
                                 c3_1=4*limg_size,
                                 c2_2=8*limg_size,
                                 c2_1=8*limg_size,
                                 c1_2=16*limg_size,
                                 c1_1=16*limg_size))
        
def sweep(limg_size_start,limg_size_stride,num_limg_size):
    for i in range(num_limg_size):
        generate_file(limg_size_start+limg_size_stride*i)
        

if __name__ == "__main__":
    sweep(10,1,10)