"""
Verify that the folding technique produce the correct result.
"""

import numpy as np
import zython.logf.filef as filef
from scipy import signal
from math import ceil
from py_util.overlap_add import overlap_add
import pdb

np.random.seed(0)

def conv_spatial(kern, img):
	return signal.convolve2d(img, kern)

def conv_fft_oaa(kern, img, N):
	return overlap_add(kern, img, N)



def gene_img_folded(folding_axis,l,k,array):
	l_large = folding_axis*l + (folding_axis-1)*(k-1)
	padding_row_section = np.zeros((l,k-1))
	padding_col_section = np.zeros((l,l_large+k-1))
	array_large = None
	for i in range(folding_axis):
		array_large_row = []
		for j in range(folding_axis):
			array_large_row += [np.concatenate((array[i][j],padding_row_section),axis=1)]
		array_large_row = np.concatenate(array_large_row,axis=1)
		assert array_large_row.shape[1] == l_large+k-1
		array_large_row = np.concatenate((array_large_row,padding_col_section),axis=0)
		if array_large is None:
			array_large = array_large_row
		else:
			array_large = np.concatenate((array_large,array_large_row),axis=0)
	return array_large[0:l_large,0:l_large]


if __name__ == "__main__":
	file_out = 'verify_folding.out'
	np.set_printoptions(precision=2,linewidth=200)
	folding_axis = 3
	l = 2
	k = 3
	N = 8
	l_large = folding_axis*l + (folding_axis+1)*(k-1)
	imgs = np.random.rand(folding_axis,folding_axis,l,l)
	kern = np.random.rand(k,k)*10
	imgs_large = gene_img_folded(folding_axis,l,k,imgs)
	filef.print_to_file(file_out,'img after folding:\n{}',imgs_large,type=None,separator='=')
	imgs_conv_spatial = conv_spatial(kern,imgs_large)
	imgs_conv_fft_oaa = conv_fft_oaa(kern,imgs_large,N)
	filef.print_to_file(file_out,'img after conv sparial:\n{}',imgs_conv_spatial,type=None,separator='=')
	filef.print_to_file(file_out,'img after conv fft oaa:\n{}',imgs_conv_fft_oaa,type=None,separator='=')
