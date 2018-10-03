"""
I realized one problem: since FFT/Hadamand product is so low (except layer 1), why do we even need FFT on FPGA?
The larger the FFT size, the more the reduction of floating point operations. The more significant FFT/Hadamand ratio is.
* We should probably increase FFT size?
* It is best if we verify that kernel buffer size doesn't have much to do with performance.
* also we can calculate the total memory depending on FFT size.


Recent update:
* Algorithmetic level optimization:
	* Motivation:
		* consider the following case, l_img=13; l_sub=14; padding=1; (16 point FFT).
		  So the image is 13+2*1=15, just 1 pixel wider than l_sub --> # tiles=ceil(15/14)^2=4 --> wasteful.
	* Solution:
		* Unroll along the batch dimension, instead of processing 1 image at a time, we concatenate several to form a larger image.
		* e.g.: Unroll factor of 4 --> large iamge size = 13+3+13+2*1. 
	* Analytic solution:
		* This level of optimization is then trying to find out the optimal unrolling factor and the FFT size. 
*
"""

import numpy as np
from zython.logf.printf import printf
import zython.logf.filef as filef
import math
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import zython.arithmetic as arizhmetic
import sys

_FILE_OUT = 'AlexNet.out'

"""
analysis procedure:

simplification of fft numOp expression & spatial numOp expression.
--> derive the relationship among kernel size, fft size, stride... 
"""

CONST = {'fft':3,'Hadamard':7,'ifft':3}


def profile_ratio(*x_arr, info="", _FILE='debug.out'):
    """
    return the ratio of each element:
        x_arr[i]/sigma_(x_arr)
    """
    filef.print_to_file(_FILE, info, type=None)
    x_arr = np.array(x_arr).reshape(-1)
    _sum = np.sum(x_arr+0.)
    _len = x_arr.shape[0]
    _str = '{:4.3f}%   '*_len
    filef.print_to_file(_FILE, _str, *(x_arr/_sum), type=None)






def op_count_spatial(f_in,f_out,l_img,l_kern,stride,padding):
	#op_per_sliding_win = 
	num_pixel_out = ((l_img-l_kern+2*(padding))/stride+1)**2
	ops_per_pixel = 2*l_kern**2
	return f_in*f_out*num_pixel_out*ops_per_pixel

def operation_FFT_CaP_OaA(f_in,f_out,l_img,l_kern,N,folding):
	""" Corrected version
	folding != 1:
		CaP-OaA
	folding == 1:
		OaA
	N >= folding*l_img+(folding-1)*(l_kern-1):
		native FFT
	all the inputs should be either numpy array or single data
	"""
	# operations per tile
	tile_fft = 2*CONST['fft']*N**2*np.log(N)/np.log(2)*f_in
	tile_ifft = 2*CONST['ifft']*N**2*np.log(N)/np.log(2)*f_out
	tile_Hadamard = CONST['Hadamard']*N**2*f_in*f_out
	# calculate number of tiles
	l_img_fd = folding*l_img + (folding-1)*(l_kern-1)
	num_tiles = np.ceil(l_img_fd/(N-l_kern+1))**2
	return num_tiles*(tile_fft+tile_ifft+tile_Hadamard)/(folding**2)

def operation_min_CaP(f_in,f_out,l_img,l_kern,available_FFT):
	""" Corrected version
	upper bound for achievable operation reduction, by using CaP.
	Return three array of length equal to number of layers.
		array0: number of operations using CaP
		(array1,array2): corresponding (N,fd)
	"""
	opt_fft = np.array([available_FFT]).max()
	assert opt_fft > np.array(l_kern).max()
	opt_fd = (opt_fft-l_kern+1)/arizhmetic.gcd(l_img+l_kern-1,opt_fft-l_kern+1)
	num_layers = len(np.array(f_in))
	return operation_FFT_CaP_OaA(f_in,f_out,l_img,l_kern,opt_fft,opt_fd), \
			(np.array([opt_fft]*num_layers),opt_fd)

def operation_min_OaA(f_in,f_out,l_img,l_kern,available_FFT):
	""" Corrected version
	optimal for using only OaA w/o CaP.
	folding = 1 in this case.
	Return the similar as operation_min_CaP.
	"""
	f_in = np.array(f_in)
	f_out = np.array(f_out)
	l_img = np.array(l_img)
	l_kern = np.array(l_kern)
	available_FFT = np.array(available_FFT)
	num_layers = len(f_in)
	num_FFT = len(available_FFT)
	op_layers_N = np.zeros((num_layers,num_FFT))
	for l in range(num_layers):
		# check for invalid FFT sizes
		available_FFT_l = available_FFT+(available_FFT-l_kern[l]<0)*(sys.maxsize*0.8)
		op_layers_N[l] = operation_FFT_CaP_OaA(f_in[l],f_out[l],l_img[l],l_kern[l],available_FFT_l,1)
	min_idx = np.argmin(op_layers_N,axis=1)
	return op_layers_N[np.arange(num_layers),min_idx], \
			(available_FFT[min_idx],np.array([1]*num_layers))






if __name__ == "__main__":
	"""
	printf("(f_in, f_out, l_img, l_kern, N, stride)")
	for f_in in np.array([128,256,512]):
		for f_out in np.array([128,256,512]):
			if f_in > f_out: continue
			for l_img in np.array([32,64,128,256]):
				for l_kern in np.array([3,5,7,11]):
					for N in np.array([4,8,16]):
						if N <= l_kern: continue
						for stride in np.array([1]):
							op_spatial = op_count_spatial(f_in,f_out,l_img,l_kern,-1,stride)
							op_fft = op_count_fft(f_in,f_out,l_img,l_kern,N,-1)
							printf("({:4d},{:4d},{:4d},{:4d},{:4d},{:4d})-->spatial: {:8.0f}, fft: {:8.0f}-->ratio: {:.3f}", 
								f_in,f_out,l_img,l_kern,N,stride,op_spatial,op_fft,op_fft/op_spatial,separator=None)
	"""
	# param list: [fin, fout, l_img, l_kern, N, stride, padding]
	layers = [[  3, 96,224,11,64,4,0],
			  [ 96,256, 55, 5,64,1,2],
			  [256,384, 27, 3,64,1,1],
			  [384,384, 13, 3,64,1,1],
			  [384,256, 13, 3,64,1,1]]
	folding = -1
	min_tot_op_ratio = float("inf")
	min_folding = -1
	printf("operation count is in unit of Mega")
	_FD_MAX_1D = 9
	_FD_MIN_1D = 1
	#### matplotlib ####
	axis_folding = np.array([])
	axis_layers_op_spatial = np.array([])
	axis_layers_op_oaa = np.array([])
	axis_tot_op_spatial = np.array([])
	axis_tot_op_oaa = np.array([])
	####################
	for fd in range(_FD_MIN_1D,_FD_MAX_1D+1):
		#### matplotlib ####
		axis_folding = np.append(axis_folding,[fd],axis=0)
		####################
		printf("folding factor 1D: {:4d}", fd)
		printf("l   FFT spatial OaA     Ratio ",type=None,separator='-')
		layers_op_spatial = []
		layers_op_oaa = []
		for i,layer in enumerate(layers):
			_op_spatial = int(op_count_spatial(*layer)/1e6)
			_op_oaa = int(op_count_fft(*layer,folding_1D=fd)/1e6)
			layers_op_spatial += [_op_spatial]
			layers_op_oaa += [_op_oaa]
			printf("{:2d}  {:3d} {:7d} {:7d} {:5.4f}",i+1,layer[-3],_op_spatial,_op_oaa,_op_oaa/_op_spatial,type=None)
		tot_op_spatial = np.sum(layers_op_spatial)
		tot_op_oaa = np.sum(layers_op_oaa)
		tot_op_ratio = tot_op_oaa/tot_op_spatial
		#### matplotlib ####
		axis_layers_op_spatial = np.append(axis_layers_op_spatial,layers_op_spatial,axis=0)
		axis_layers_op_oaa = np.append(axis_layers_op_oaa,layers_op_oaa,axis=0)
		axis_tot_op_spatial = np.append(axis_tot_op_spatial,[tot_op_spatial],axis=0)
		axis_tot_op_oaa = np.append(axis_tot_op_oaa,[tot_op_oaa],axis=0)
		####################
		printf("total_spatial   total_oaa   ratio ", type=None, separator='=')
		printf("{:9d}       {:9d}   {:5.4f}", tot_op_spatial, tot_op_oaa, tot_op_ratio, type=None)
		if min_tot_op_ratio>tot_op_ratio:
			min_tot_op_ratio = tot_op_ratio
			min_folding = fd
	printf("min_tot_op_ratio for folding {} to {} is: folding {} --> {:5.4f}",
		_FD_MIN_1D,_FD_MAX_1D,min_folding,min_tot_op_ratio, separator='><')

	#### matplotlib ####
	num_folding = len(axis_folding)
	axis_layers_op_spatial = axis_layers_op_spatial.reshape(num_folding,-1).transpose()
	axis_layers_op_oaa = axis_layers_op_oaa.reshape(num_folding,-1).transpose()
	num_layers = axis_layers_op_spatial.shape[0]

	fig1 = plt.figure(1)
	ax = plt.subplot(111)
	line_spatial, = ax.plot(axis_folding, axis_tot_op_spatial,'-o',label='Spatial')
	line_oaa, = ax.plot(axis_folding, axis_tot_op_oaa,'-o',label='OaA')
	ax.set_title('AlexNet: Effect of Folding Factor\n (64 point FFT for 5 layers)', fontsize=20)
	ax.set_xlabel('Folding Factor',fontsize=16)
	ax.set_ylabel('Num Operations (M)',fontsize=16)
	box = ax.get_position()
	ax.set_position([box.x0,box.y0,box.width*0.8,box.height])
	#ax.legend(loc='upper center',bbox_to_anchor=(0.5,-0.09),fancybox=True,shadow=True,ncol=2)
	#plt.show()
	#fig1.savefig('plots/folding_total.png')

	#fig2 = plt.figure(2)
	bar_width = 0.5
	line_layers = [None]*num_layers
	line_name = [None]*num_layers
	prev_bottom = np.array([0.]*num_folding)

	###########################
	# BAR PLOT ################
	###########################
	cmap = plt.cm.gist_ncar
	colors = [cmap(i) for i in np.linspace(0,1,num_layers+1)]
	for i,l in enumerate(axis_layers_op_oaa):
		line_name[i] = 'layer {}'.format(i+1)
		line_layers[i] = ax.bar(axis_folding, l, bar_width, bottom=prev_bottom,color=colors[i],label='Layer {}'.format(i))
		prev_bottom += l
	#plt.title('folding factor vs. number of operations -- break down')
	#plt.xlabel('folding factor')
	#plt.ylabel('num operations (M)')
	ax.legend(loc='center left', bbox_to_anchor=(1,0.5),fancybox=True,shadow=True,ncol=1)
	#plt.show()
	plt.savefig('plots/folding_layers.png')
	####################


	"""
	layer1 = [3,96,224,11,64,4,0]
	printf("layer1({:2d}FFT): ratio={:.3f}",layer1[-3],op_count_fft(*layer1,info="layer1\n")/op_count_spatial(*layer1),separator=None)
	layer2 = [96,256,55,5,64,1,2]
	printf("layer2({:2d}FFT): ratio={:.3f}",layer2[-3],op_count_fft(*layer2,info="layer2\n")/op_count_spatial(*layer2),separator=None)
	layer3 = [256,384,27,3,8,1,1]
	printf("layer3({:2d}FFT): ratio={:.3f}",layer3[-3],op_count_fft(*layer3,info="layer3\n")/op_count_spatial(*layer3),separator=None)
	layer4 = [384,384,13,3,8,1,1]
	printf("layer4({:2d}FFT): ratio={:.3f}",layer3[-3],op_count_fft(*layer4,info="layer4\n")/op_count_spatial(*layer4),separator=None)
	layer5 = [384,256,13,3,8,1,1]
	printf("layer5({:2d}FFT): ratio={:.3f}",layer3[-3],op_count_fft(*layer5,info="layer5\n")/op_count_spatial(*layer5),separator=None)
	"""
