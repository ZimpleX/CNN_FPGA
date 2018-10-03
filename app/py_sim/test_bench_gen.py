import argparse

template = """
`timescale 1ns/100ps

module test_bench_{tb_para};
    parameter clk_period = 3.00;
	parameter DATA_WIDTH = 32;
	parameter PARA = {para};
    reg clk,rst;
    reg[DATA_WIDTH-1:0] input_stream[PARA-1:0];
    wire[DATA_WIDTH-1:0] output_stream[PARA-1:0];
    reg valid_in;
    wire valid_out;

	 
    spn #(.DATA_WIDTH(DATA_WIDTH)) UUT(
        .clk(clk),
        .rst(rst),
        .input_stream(input_stream),
        .output_stream(output_stream),
        .valid_in(valid_in),
        .valid_out(valid_out)
    );

    always #(clk_period/2) clk=~clk;

    integer input_file,output_file,scan;
    integer i,j;
    
    initial begin
        clk = 0;
        rst = 1;
        valid_in = 0;
        #(10*clk_period);
        rst = 0;
        #(clk_period);
        $display("start");
        input_file = $fopen("input_file_{ip_para}.txt","r");
        output_file = $fopen("output_file_{op_para}.txt","w");
        if (input_file == 0 || output_file == 0) begin
            $display("in/output file handle was NULL");
            $finish;
        end
        while (!$feof(input_file)) begin
            // NOTE: valid in should be a clk before the real input data
            // ========
            // AUTO GEN
            // ========
            scan = $fscanf(input_file, "{input_regex} - %d\\n",
{input_data_signal}            valid_in);
            #(clk_period);
        end

        for (i=0;i<PARA;i=i+1) begin
            input_stream[i] = 0;
        end
        valid_in = 1;
        #(260*clk_period);
        $fclose(output_file);
        $fclose(input_file);
        $finish;
    end

    always@(negedge clk) begin
        if (output_file != 0) begin
            // ========
            // AUTO GEN
            // ========
            $fwrite(output_file,"{output_regex}    - %d\\n",
{output_data_signal}            valid_out);
	end
    end

endmodule
"""


def parse_args():
    parser = argparse.ArgumentParser(' testbench for SPN (N x N matrix)')
    parser.add_argument('-N', type=int, choices=range(100), required=True,
                        help='transpose N x N matrix')
    parser.add_argument('-p', type=int, required=True,
                        help='data parallelism to SPN')
    parser.add_argument('--gen', type=str, required=True, default=None,
                        help='file name if u want auto Verilog generation')
    return parser.parse_args()

def gen_test_bench(N,p,output_file):
    input_regex = ' '.join(['%d']*p)
    input_data_signal = ''
    for i in range(int(p/2)):
        input_data_signal += '            input_stream[{}],input_stream[{}],\n'.format(2*i,2*i+1)
    output_regex = '    '.join(['%d']*p)
    output_data_signal = ''
    for i in range(int(p/2)):
        output_data_signal += '            output_stream[{}],output_stream[{}],\n'.format(2*i,2*i+1)
    with open(output_file, 'w') as f:
        _t = template.format(tb_para=p,para=p,ip_para=p,op_para=p,input_regex=input_regex,input_data_signal=input_data_signal,\
        					output_regex=output_regex,output_data_signal=output_data_signal)
        f.write(_t)


if __name__ == '__main__':
	args = parse_args()
	gen_test_bench(args.N,args.p,args.gen)
