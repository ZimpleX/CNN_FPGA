## Auto generation of Streaming Permutation Networks (SPN) in `Verilog`

#### Directory structure

* `misc`: some miscellaneous python files for logging and file IO (this directory can be safely ignored).
* `sim_spn`: directory containing python scripts to generate SPN and its testbenchs
	* `SPN.py`: software simulator for the hardware implementation of SPN. This simulation supports stride-S permutation for N x N matrix (for any integer S and N).
	* `matrix_transpose_gen.py`: complete `Verilog` generation script. This script generates hardware that can handle matrix transpose of N x N matrices (essentially, matrix transpose is stride-N permutation of N x N matrices). You can generate SPN for any N and data parallelism P. 
	* `test_bench_gen.py`: auto generation of simple test benches for SPN. 

#### Example

* `$ cd sim_spn`
* `$ python matrix_transpose_gen.py -h` to get the list of supported command line arguments
* `$ python matrix_transpose_gen.py -N 16 -p 4 --gen spn.sv` to generate the SPN hardware for 16 x 16 matrix transpose, with data parallelism of 4 (input 4 data each clock).

#### Requirements

* `Python 3` (my own version is `Python 3.6`)

#### References

* Ren Chen, et. al., Energy and Memory Efficient Mapping of Bitonic Sorting on FPGA (FPGA'15)
* Hanqing Zeng, et. al., A Framework for Generating High Throughput CNN Implementations on FPGAs (FPGA'18)