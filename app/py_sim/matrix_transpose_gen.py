"""
Always remember that folding is done based on the original CLOS.
e.g., if we want to permutate 64 x 64 with data parallelism of 4:
CLOS --
    STAGE 0: 16 sets of 4 x 4 crossbar
    STAGE 1: 4 sets of 16 x 16 crossbar
    STAGE 2: 16 sets of 4 x 4 crossbar
FOLDED --
    STAGE 0: 1 set of 4 x 4 crossbar
    STAGE 1: buffer of size 4 x 16
    STAGE 2: 1 set of 4 x 4 crossbar
ALGORITHM --
    generate full CLOS by straight forward brute force.
    get the control signal for each crossbar directly by STAGE 0/2
    ger the Permutation matrix directly by STAGE 1 (addr_i = P*addr_(i-1))
"""

import numpy as np
import copy
import argparse
import math

block_ram_sv = """module blk_ram(clk,wen,addr,din,dout);

    parameter DATA_WIDTH = 32;
    parameter MEM_ADDR_WIDTH = 6;
    parameter SIZE_RAM = 1<<MEM_ADDR_WIDTH;

    input clk,wen;
    input[MEM_ADDR_WIDTH-1:0] addr;
    input[DATA_WIDTH-1:0] din;
    output reg[DATA_WIDTH-1:0] dout;

    reg[DATA_WIDTH-1:0] ram[SIZE_RAM-1:0];

    always@(posedge clk)
    begin
        if (wen) begin
            ram[addr] <= din;
        end
        dout <= ram[addr];
        
    end
endmodule
"""

spn_verilog_fixed_above = """
    localparam SEL_WIDTH = {SEL_WIDTH};  // data width for one mux sel signal
    localparam MEM_ADDR_WIDTH = {MEM_ADDR_WIDTH};   // log(MEM_DEPTH,2)

    input clk,rst,valid_in;
    input[DATA_WIDTH-1:0] input_stream[PARA-1:0];
    output valid_out;
    output[DATA_WIDTH-1:0] output_stream[PARA-1:0];

    reg[MEM_ADDR_WIDTH:0] flush_counter;
    reg[DATA_WIDTH-1:0] output_stream_reg[PARA-1:0];
    reg[DATA_WIDTH-1:0] input_delay0[PARA-1:0];
    reg[DATA_WIDTH-1:0] mem_delay1[PARA-1:0];
    reg[DATA_WIDTH-1:0] mem_output[PARA-1:0];
    reg[DATA_WIDTH-1:0] output_delay2[PARA-1:0];
    reg[DATA_WIDTH-1:0] mem_input[PARA-1:0];
    
    reg valid_mem_in,valid_perm2_in;
    reg valid_delay0,valid_delay1,valid_out_reg;
    reg valid_out_ahead,valid_out_delay;        // this is for delaying valid_out for 1 clk.
    
    reg[SEL_WIDTH:0] i;
"""

spn_verilog_fixed_middle = """
    genvar g;
    generate 
    for (g=0; g<PARA; g=g+1) begin: mem_chan
      blk_ram #(.DATA_WIDTH(DATA_WIDTH),
            .MEM_ADDR_WIDTH(MEM_ADDR_WIDTH)) ram(clk,valid_delay1,mem_addr[g],mem_delay1[g],mem_output[g]);
    end
    endgenerate
    assign valid_out_ahead = (flush_counter==0)? valid_out_reg:1'b0;
    assign valid_out = valid_out_delay;
    assign output_stream = output_stream_reg;
    always@(posedge clk) begin
        // need to pass in an additional image of all 0 to flush the mem stage.
        // This is ok, since it will also guarantee the 
        if (rst) begin"""


spn_verilog_fixed_below = """
            valid_out_delay <= 0;
            flush_counter = {flush_init};
            valid_mem_in <= 0;
            valid_perm2_in <= 0;
            valid_delay0 <= 0;
            valid_delay1 <= 0;
            valid_out_reg <= 0;
        end else begin
            valid_out_delay <= valid_out_ahead;
            // if in this cycle, the data is valid, then we need to pass-in
            // valid_in to be 1 the cycle before
            valid_delay0 <= valid_in;   // delay for 1 clk
            valid_mem_in <= valid_delay0;
            valid_delay1 <= valid_mem_in;
            valid_perm2_in <= valid_delay1;
            valid_out_reg <= valid_perm2_in;// i will output one wasted image of N x N
                                        // but it is all 0, so it won't affec
                                        
            if (valid_in) begin
            // perm0: get control signal
                counter_perm0 <= counter_perm0+1;
                for (i=0; i<PARA; i=i+1) begin
                    sel_perm0[i] <= control_perm0[i][counter_perm0*SEL_WIDTH+:SEL_WIDTH];
                end
                input_delay0 <= input_stream;
            end
            if (valid_delay0) begin
            // perm0: mux
                for (i=0; i<PARA; i=i+1)
                    mem_input[i] <= input_delay0[sel_perm0[i]];
            end
            if (valid_mem_in) begin
            // mem: get control signal
{control_mem_increment}
{mem_addr_update}
                mem_delay1 <= mem_input;
            end
            if (valid_delay1) begin
            // mem: ram module handles in/out
            end
            if (valid_perm2_in) begin
            // perm2: get control signal
                counter_perm2 <= counter_perm2+1;
                for (i=0; i<PARA; i=i+1) begin
                    sel_perm2[i] <= control_perm2[i][counter_perm2*SEL_WIDTH+:SEL_WIDTH];
            end
            output_delay2 <= mem_output;
            end
            if (valid_out_reg) begin
                if (flush_counter > 0) begin
                    flush_counter <= flush_counter-1;
                end
                for (i=0; i<PARA; i=i+1)
                    output_stream_reg[i] <= output_delay2[sel_perm2[i]];
            end
        end
    end

endmodule
"""

control_mem_increment_template = '                counter_mem_addr_{} <= counter_mem_addr_{}+1;\n'
mem_addr_update_template = '                mem_addr[{}] <= control_mem_{}[counter_mem_addr_{}*MEM_ADDR_WIDTH+:MEM_ADDR_WIDTH];\n'

###########################
# SPN software simulation #
###########################
class SPN:
    def __init__(self, input_idx, output_idx, parallelism, dtype):
        self.dtype = dtype
        self.input_idx = input_idx
        self.output_idx = output_idx
        self.n = len(input_idx)
        self.p = parallelism
        self.mem_buffer = np.zeros(self.n, dtype=self.dtype).reshape(self.p, -1)
        self.counter = 0
        self.conf_stage0 = np.zeros((int(self.n / self.p), int(self.p)), dtype='i4') - 1  # full/fold CLOS
        self.conf_stage1 = np.zeros((int(self.p), int(self.n / self.p)), dtype='i4') - 1  # full CLOS
        self.addr_para = [None] * self.p  # fold CLOS (mem stage)
        self.conf_stage2 = np.zeros((int(self.n / self.p), int(self.p)), dtype='i4') - 1  # full/fold CLOS
        self.resources = {'control bits': 0}

    def clos_full_conf(self):
        """
        index for input and output starts from 0.
        """
        num_mux_stage0 = self.n / self.p
        num_mux_stage1 = self.p
        num_mux_stage2 = self.n / self.p
        for i_i, i in enumerate(self.input_idx):
            found = False
            idx_mux_stage0 = i // self.p
            idx_pin_stage0 = i % self.p
            o = self.output_idx[i_i]
            idx_mux_stage2 = o // self.p
            idx_pin_stage2 = o % self.p
            for r0 in range(self.p):
                if found:
                    break
                if r0 in self.conf_stage0[idx_mux_stage0]:
                    continue
                # go to stage1
                idx_mux_stage1 = r0
                if idx_mux_stage2 in self.conf_stage1[idx_mux_stage1]:
                    continue
                self.conf_stage0[idx_mux_stage0][idx_pin_stage0] = r0
                self.conf_stage1[idx_mux_stage1][idx_mux_stage0] = idx_mux_stage2
                self.conf_stage2[idx_mux_stage2][idx_mux_stage1] = idx_pin_stage2
                found = True

    def clos_fold_conf(self):
        self.clos_full_conf()
        mem_depth = self.conf_stage0.shape[0]
        permutation = np.zeros((self.p, mem_depth, mem_depth))
        for p_i in range(self.p):
            for a_i in range(mem_depth):
                permutation[p_i][a_i][self.conf_stage1[p_i][a_i]] = 1
        power = np.zeros(self.p)
        for p_i in range(self.p):
            _perm = permutation[p_i]
            while True:
                power[p_i] += 1
                _perm = np.dot(_perm, permutation[p_i])
                if (_perm == permutation[p_i]).all():
                    break
        addr = np.arange(mem_depth).reshape(mem_depth, 1)
        for p_i in range(self.p):
            cur_perm = permutation[p_i]
            self.addr_para[p_i] = np.zeros(int(mem_depth * power[p_i]), dtype='i4')
            for c_i in range(int(power[p_i])):
                self.addr_para[p_i][c_i * mem_depth:(c_i + 1) * mem_depth] = np.dot(cur_perm, addr).flatten()
                cur_perm = np.dot(cur_perm, permutation[p_i])

    def clos_fold_gen_control_bits(self, output_file):
        # memory stage
        bits_per_addr = int(math.ceil(math.log(self.n / self.p, 2)))
        strf = '{{:{}b}}'.format(bits_per_addr)
        addr_to_bits = lambda addr, strf: strf.format(addr).replace(' ', '0')
        addr_para_bitstream = [''] * self.p
        for p_i in range(self.p):
            cur_bitstream = ''
            for a_i in self.addr_para[p_i]:
                cur_bitstream = addr_to_bits(a_i, strf) + cur_bitstream
            addr_para_bitstream[p_i] = cur_bitstream
        # permutation stages
        bits_per_mux = int(math.ceil(math.log(self.p, 2)))
        strf = '{{:{}b}}'.format(bits_per_mux)
        stage0_bitstream = [''] * (self.p)
        stage2_bitstream = [''] * (self.p)
        for p_i in range(self.p):
            cur_bitstream0 = ''
            for c_i in self.conf_stage0[:, p_i]:
                cur_bitstream0 = addr_to_bits(c_i, strf) + cur_bitstream0
            stage0_bitstream[p_i] = cur_bitstream0
            cur_bitstream2 = ''
            for c_i in self.conf_stage2[:, p_i]:
                cur_bitstream2 = addr_to_bits(c_i, strf) + cur_bitstream2
            stage2_bitstream[p_i] = cur_bitstream2
        # write to file
        strf = '{}\'b{}'
        output_str = ''
        # meta data
        output_str += 4 * ' ' + '// [BEGIN] VAR DEF\n'
        output_str += 4 * ' ' + '// width for one mem addr, or mux sel signal\n'
        output_str += 4 * ' ' + 'reg[{}:0] mem_addr[{}:0];     // stage1: temp perm\n'.format(bits_per_addr - 1, self.p - 1)
        output_str += 4 * ' ' + 'reg[{}:0] sel_perm0[{}:0];    // stage0: spatial perm\n'.format(bits_per_mux - 1, self.p - 1)
        output_str += 4 * ' ' + 'reg[{}:0] sel_perm2[{}:0];    // stage2: spatial perm\n'.format(bits_per_mux - 1, self.p - 1)
        output_str += '\n\n'
        output_str += 4 * ' ' + '// counter used for traversing control_* (simply add 1 each cycle)\n'
        for p_i in range(self.p):
            bits_per_counter = int(math.ceil(math.log(len(self.addr_para[p_i]), 2)))
            output_str += 4 * ' ' + 'reg[{}:0] counter_mem_addr_{};\n'.format(bits_per_counter - 1, p_i)
        output_str += '\n'
        bits_per_counter = int(math.ceil(math.log(self.n / self.p, 2)))
        output_str += 4 * ' ' + 'reg[{}:0] counter_perm0;\n'.format(bits_per_counter - 1)
        output_str += 4 * ' ' + 'reg[{}:0] counter_perm2;\n'.format(bits_per_counter - 1)
        output_str += '\n\n'
        output_str += 4 * ' ' + '// actual control bits\n'
        for p_i in range(self.p):
            output_str += 4 * ' ' + 'reg[{}:0] control_mem_{};\n'.format(len(addr_para_bitstream[p_i]) - 1, p_i)
        output_str += 4 * ' ' + 'reg[{}:0] control_perm0[{}:0];\n'.format(len(stage0_bitstream[0]) - 1, self.p - 1)
        output_str += 4 * ' ' + 'reg[{}:0] control_perm2[{}:0];\n'.format(len(stage2_bitstream[0]) - 1, self.p - 1)
        output_str += 4 * ' ' + '// [END] VAR DEF\n'

        spn_verilog_str = block_ram_sv + '\n\nmodule spn(clk,rst,input_stream,output_stream,valid_in,valid_out);\n'
        spn_verilog_str += '    parameter PARA = ' + str(self.p) + ';\n'
        spn_verilog_str += '    parameter MEM_DEPTH = ' + str(int(self.n / self.p)) + ';\n'
        spn_verilog_str += '    parameter DATA_WIDTH = ' + str(32) + ';\n'
        
        sel_width = int(math.log(self.p,2))
        mem_addr_width = int(math.log(self.n/self.p,2))
        spn_verilog_str += spn_verilog_fixed_above.format(SEL_WIDTH=sel_width,MEM_ADDR_WIDTH=mem_addr_width) + output_str + spn_verilog_fixed_middle

        output_str = ''
        output_str += '\n'
        output_str += 12 * ' ' + '// [BEGIN] VAR INIT\n'
        output_str += 12 * ' ' + 'counter_perm0 <= 0;\n'
        output_str += 12 * ' ' + 'counter_perm2 <= 0;\n'
        for i in range(self.p):
            output_str += 12 * ' ' + 'counter_mem_addr_{} <= 0;\n'.format(i)
        # mem stage
        for p_i in range(self.p):
            data = strf.format(len(addr_para_bitstream[p_i]), addr_para_bitstream[p_i])
            output_str += 12 * ' ' + 'control_mem_{} <= {};\n'.format(p_i, data)
        # perm stage0
        output_str += '\n'
        for p_i in range(self.p):
            data = strf.format(len(stage0_bitstream[p_i]), stage0_bitstream[p_i])
            output_str += 12 * ' ' + 'control_perm0[{}] <= {};\n'.format(p_i, data)
        # perm stage2
        output_str += '\n'
        for p_i in range(self.p):
            data = strf.format(len(stage2_bitstream[p_i]), stage2_bitstream[p_i])
            output_str += 12 * ' ' + 'control_perm2[{}] <= {};\n'.format(p_i, data)
        output_str += 12 * ' ' + '// [END] VAR INIT'
        output_str += '\n'
        # write
        control_mem_increment = ''
        mem_addr_update = ''
        for i in range(self.p):
            control_mem_increment += control_mem_increment_template.format(i,i)
            mem_addr_update += mem_addr_update_template.format(i,i,i)
        spn_verilog_str += output_str + spn_verilog_fixed_below\
            .format(flush_init=int(self.n/self.p),control_mem_increment=control_mem_increment,\
                mem_addr_update=mem_addr_update)

        with open(output_file, 'w') as f:
            f.write(spn_verilog_str)

    def stride_perm(self, input_stream):
        mem_depth = self.n / self.p
        # stage 0
        mod_stage0 = self.conf_stage0.shape[0]
        cur_conf_stage0 = self.conf_stage0[self.counter % mod_stage0]
        output_stage0 = input_stream[list(cur_conf_stage0)]
        # mem stage
        output_stage1 = []
        for p_i in range(self.p):
            mod_memp = len(self.addr_para[p_i])
            addr_p = self.addr_para[p_i][self.counter % mod_memp]
            output_stage1 += [copy.deepcopy(self.mem_buffer[p_i][addr_p])]
            self.mem_buffer[p_i][addr_p] = copy.deepcopy(output_stage0[p_i])
        output_stage1 = np.array(output_stage1, dtype=self.dtype)
        # stage 2
        mod_stage2 = self.conf_stage2.shape[0]
        cur_conf_stage2 = self.conf_stage2[self.counter % mod_stage2]
        output_stage2 = output_stage1[list(cur_conf_stage2)]
        self.counter += 1
        return output_stage2

    def resource_estimator(self):
        bits_per_addr = math.log(self.n / self.p, 2)
        num_addr = sum([len(addr_seq) for addr_seq in self.addr_para])
        control_bits_mem_addr = bits_per_addr * num_addr
        bits_per_mux = math.ceil(math.log(self.p, 2))
        bits_per_stageX = bits_per_mux * self.p
        control_bits_stage0 = self.conf_stage0.shape[0] * bits_per_stageX
        control_bits_stage2 = self.conf_stage2.shape[0] * bits_per_stageX
        self.resources['control bits'] = control_bits_mem_addr + control_bits_stage0 + control_bits_stage2


############################
# matrix transpose example #
############################

def input_generation(N, p):
    """
    generate total of N x N inputs, with parallelism of p.
    """
    inputs = np.arange(N ** 2)
    count = 0
    cur_input = []
    for r in range(2):  # repeat for 3 times
        for e in inputs:  # p as a group
            count += 1
            cur_input += [(e // N, e % N)]
            if count == p:
                count = 0
                out = np.array(cur_input, dtype=[('x', 'i4'), ('y', 'i4')])
                cur_input = []
                yield out


def matrix_transpose_SPN(N, p, code_gen_output=None):
    """
    transpose a N x N matrix.
    for the SPN, the total data length is n = N x N.
    """
    input_idx = np.arange(N ** 2)
    output_idx = input_idx.reshape(N, N).T.flatten()
    trans_spn = SPN(input_idx, output_idx, p, [('x', 'i4'), ('y', 'i4')])
    trans_spn.clos_fold_conf()
    input_data = []
    output_data = []
    for input_stream in input_generation(N, p):
        output_stream = trans_spn.stride_perm(input_stream)
        input_data += [input_stream]
        output_data += [output_stream]
    trans_spn.resource_estimator()
    if code_gen_output:
        trans_spn.clos_fold_gen_control_bits(code_gen_output)
    return input_data, output_data, trans_spn.resources


def log_streaming_data_SPN(N, p, input_data, output_data, resources):
    row_len = 0
    raw_sequence = {'0: INPUT': input_data, '1: OUTPUT': output_data}
    strf = N // 10 and '{:2s}' or '{:1s}'

    def tostr_tuple(e):
        return (strf + ',' + strf).format(str(e[0]), str(e[1]))

    v_tostr_tuple = np.vectorize(tostr_tuple)
    for k in sorted(raw_sequence.keys()):
        print('{}'.format(k))
        for tup in raw_sequence[k]:
            row_len += len(tup)
            str_tup = ''
            for start_i in range(0,len(tup),N):
                if start_i > 0:
                    str_tup += '\n'*2
                str_tup += '  '.join(v_tostr_tuple(tup[start_i:start_i+N]))
            print(str_tup + ' ',end='')
            if row_len >= N:
                row_len = 0
                print('\n')
    for k in resources.keys():
        print('{}: {}'.format(k, resources[k]))


def parse_args():
    parser = argparse.ArgumentParser('SPN for N x N matrix')
    parser.add_argument('-N', type=int, choices=range(100), required=True,
                        help='transpose N x N matrix')
    parser.add_argument('-p', type=int, required=True,
                        help='data parallelism to SPN')
    parser.add_argument('--gen', type=str, required=False, default=None,
                        help='file name if u want auto Verilog generation')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    input_data, output_data, resources = matrix_transpose_SPN(args.N, args.p, code_gen_output=args.gen)
    log_streaming_data_SPN(args.N, args.p, input_data, output_data, resources)
