"""
TODO:
* [FFT64]: finish testing of integer version
* [FFT64]: change to fixed point arithmetic
* [FFT64]: resource estimation
* [ArchOp]:derive first configuration
* [FFT64-2D]
* [MAC]: change to fixed point
* [Chi]: integrate Ren's 64 point FFT
* [Test]: on Vivado
* [Test]: on HARP

-- convLayer/src/afu_user/afu_user.sv
"""
import numpy as np
from zython.logf.printf import printf

HEX_TO_BIN = {	'0':'0000',
				'1':'0001',
				'2':'0010',
				'3':'0011',
				'4':'0100',
				'5':'0101',
				'6':'0110',
				'7':'0111',
				'8':'1000',
				'9':'1001',
				'a':'1010',
				'b':'1011',
				'c':'1100',
				'd':'1101',
				'e':'1110',
				'f':'1111'}

BIN_TO_HEX = {	'0000':'0',
				'0001':'1',
				'0010':'2',
				'0011':'3',
				'0100':'4',
				'0101':'5',
				'0110':'6',
				'0111':'7',
				'1000':'8',
				'1001':'9',
				'1010':'a',
				'1011':'b',
				'1100':'c',
				'1101':'d',
				'1110':'e',
				'1111':'f'}

def fpt_to_decimal(int_bits, tot_bits, ip, format_='h'): 
	"""
	In two's compliment. int_bits include the sign bit.
	convert fixed point represented by int_bits number of integer bits and (tot_bits-int_bits) number of decimal bits
	into decimal points.
	input: str
	output: decimal
	"""
	result = 0
	if format_ == 'h':
		bin_str = ''	# string in binary format_
		for istr in ip:
			bin_str += HEX_TO_BIN[istr]
	else:
		bin_str = ip
	#import pdb; pdb.set_trace()
	for i,bit in enumerate(bin_str):
		sign = (i==0) and -1 or 1		# two's compliment
		result += sign*int(bit)*2**(int_bits-i-1)
	return result


def decimal_to_fpt(int_bits, tot_bits, ip, format_='h'):
	"""
	In two's compliment. int_bits includes the sign bit.
	convert decimal number into fixed point representation: int_bits number of integer bits, and (tot_bits-int_bits) number of decimal points.
	input: decimal
	output: string
	"""
	_max = 0
	for i in range(tot_bits-1):
		_max += 2**(i-tot_bits+int_bits)
	assert ip <= _max and ip >= - 2**(int_bits-1)
	result = ''
	result += (ip<0) and '1' or '0'
	ip += (ip<0) and 2**(int_bits-1) or 0
	for i in range(int_bits-2, int_bits-tot_bits-1, -1):
		if ip >= 2**(i):
			result += '1'
			ip -= 2**(i)
		else:
			result += '0'
	if format_ == 'h':
		result_b = result
		result = ''
		assert tot_bits%4 == 0
		for i in range(0,tot_bits,4):
			result += BIN_TO_HEX[result_b[i:i+4]]
	return result




def fft(N, ip, int_bits,tot_bits,format_='h'):
	ip = list(np.array(ip).flatten())
	ip = ip[0:len(ip)//N*N]
	assert len(ip) >= N
	ip = [fpt_to_decimal(int_bits,tot_bits,x,format_='h') for x in ip]
	ip = np.array(ip).reshape(-1,N)
	op = np.ndarray(shape=ip.shape, dtype=np.complex64)
	op_str = np.ndarray(shape=ip.shape, dtype=(np.str_,16))
	for i,ip_i in enumerate(ip):
		printf(ip_i)
		op[i] = np.fft.fft(ip_i)
	for i,op_i in enumerate(op):
		op_str[i] = np.array(['{} {}'.format(
						decimal_to_fpt(int_bits,tot_bits,o.real,format_=format_),
						decimal_to_fpt(int_bits,tot_bits,o.imag,format_=format_))
							for o in op[i]])
	return op_str




def verify_fft(N, ip_file='Verilog/fft_64/data_file_0.txt', int_bits=16, tot_bits=16, format_='h'):
	with open(ip_file, 'r') as f:
		d = f.readlines()
	d = [x.strip().split(' ') for x in d]
	result = fft(N, d, int_bits, tot_bits, format_=format_)
	return result




def SPN_addr_gen(perm_list, itr):
	"""
	perm_list: list of permutation for P.
	"""
	N = len(perm_list)
	P = np.zeros((N,N))
	temp = np.zeros((N,N))
	i_list = np.arange(N)
	for i in i_list:
		P[i][perm_list[i]] = 1
		temp[i][perm_list[i]] = 1
	for i in range(itr-1):
		temp = np.dot(temp,P)
	return np.where(temp==1)