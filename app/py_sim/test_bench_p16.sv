
`timescale 1ns/100ps

module test_bench_16;
    parameter clk_period = 3.00;
	parameter DATA_WIDTH = 32;
	parameter PARA = 16;
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
        input_file = $fopen("input_file_16.txt","r");
        output_file = $fopen("output_file_16.txt","w");
        if (input_file == 0 || output_file == 0) begin
            $display("in/output file handle was NULL");
            $finish;
        end
        while (!$feof(input_file)) begin
            // NOTE: valid in should be a clk before the real input data
            // ========
            // AUTO GEN
            // ========
            scan = $fscanf(input_file, "%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d - %d\n",
            input_stream[0],input_stream[1],
            input_stream[2],input_stream[3],
            input_stream[4],input_stream[5],
            input_stream[6],input_stream[7],
            input_stream[8],input_stream[9],
            input_stream[10],input_stream[11],
            input_stream[12],input_stream[13],
            input_stream[14],input_stream[15],
            valid_in);
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
            $fwrite(output_file,"%d    %d    %d    %d    %d    %d    %d    %d    %d    %d    %d    %d    %d    %d    %d    %d    - %d\n",
            output_stream[0],output_stream[1],
            output_stream[2],output_stream[3],
            output_stream[4],output_stream[5],
            output_stream[6],output_stream[7],
            output_stream[8],output_stream[9],
            output_stream[10],output_stream[11],
            output_stream[12],output_stream[13],
            output_stream[14],output_stream[15],
            valid_out);
	end
    end

endmodule
