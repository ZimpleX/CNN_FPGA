module blk_ram(clk,wen,addr,din,dout);

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


module spn(clk,rst,input_stream,output_stream,valid_in,valid_out);
    parameter PARA = 16;
    parameter MEM_DEPTH = 16;
    parameter DATA_WIDTH = 32;

    localparam SEL_WIDTH = 4;  // data width for one mux sel signal
    localparam MEM_ADDR_WIDTH = 4;   // log(MEM_DEPTH,2)

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
    // [BEGIN] VAR DEF
    // width for one mem addr, or mux sel signal
    reg[3:0] mem_addr[15:0];     // stage1: temp perm
    reg[3:0] sel_perm0[15:0];    // stage0: spatial perm
    reg[3:0] sel_perm2[15:0];    // stage2: spatial perm


    // counter used for traversing control_* (simply add 1 each cycle)
    reg[3:0] counter_mem_addr_0;
    reg[4:0] counter_mem_addr_1;
    reg[4:0] counter_mem_addr_2;
    reg[4:0] counter_mem_addr_3;
    reg[4:0] counter_mem_addr_4;
    reg[4:0] counter_mem_addr_5;
    reg[4:0] counter_mem_addr_6;
    reg[4:0] counter_mem_addr_7;
    reg[4:0] counter_mem_addr_8;
    reg[4:0] counter_mem_addr_9;
    reg[4:0] counter_mem_addr_10;
    reg[4:0] counter_mem_addr_11;
    reg[4:0] counter_mem_addr_12;
    reg[4:0] counter_mem_addr_13;
    reg[4:0] counter_mem_addr_14;
    reg[4:0] counter_mem_addr_15;

    reg[3:0] counter_perm0;
    reg[3:0] counter_perm2;


    // actual control bits
    reg[63:0] control_mem_0;
    reg[127:0] control_mem_1;
    reg[127:0] control_mem_2;
    reg[127:0] control_mem_3;
    reg[127:0] control_mem_4;
    reg[127:0] control_mem_5;
    reg[127:0] control_mem_6;
    reg[127:0] control_mem_7;
    reg[127:0] control_mem_8;
    reg[127:0] control_mem_9;
    reg[127:0] control_mem_10;
    reg[127:0] control_mem_11;
    reg[127:0] control_mem_12;
    reg[127:0] control_mem_13;
    reg[127:0] control_mem_14;
    reg[127:0] control_mem_15;
    reg[63:0] control_perm0[15:0];
    reg[63:0] control_perm2[15:0];
    // [END] VAR DEF

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
        if (rst) begin
            // [BEGIN] VAR INIT
            counter_perm0 <= 0;
            counter_perm2 <= 0;
            counter_mem_addr_0 <= 0;
            counter_mem_addr_1 <= 0;
            counter_mem_addr_2 <= 0;
            counter_mem_addr_3 <= 0;
            counter_mem_addr_4 <= 0;
            counter_mem_addr_5 <= 0;
            counter_mem_addr_6 <= 0;
            counter_mem_addr_7 <= 0;
            counter_mem_addr_8 <= 0;
            counter_mem_addr_9 <= 0;
            counter_mem_addr_10 <= 0;
            counter_mem_addr_11 <= 0;
            counter_mem_addr_12 <= 0;
            counter_mem_addr_13 <= 0;
            counter_mem_addr_14 <= 0;
            counter_mem_addr_15 <= 0;
            control_mem_0 <= 64'b1111111011011100101110101001100001110110010101000011001000010000;
            control_mem_1 <= 128'b11111110110111001011101010011000011101100101010000110010000100001110111111001101101010111000100101100111010001010010001100000001;
            control_mem_2 <= 128'b11111110110111001011101010011000011101100101010000110010000100001101110011111110100110001011101001010100011101100001000000110010;
            control_mem_3 <= 128'b11111110110111001011101010011000011101100101010000110010000100001100110111101111100010011010101101000101011001110000000100100011;
            control_mem_4 <= 128'b11111110110111001011101010011000011101100101010000110010000100001011101010011000111111101101110000110010000100000111011001010100;
            control_mem_5 <= 128'b11111110110111001011101010011000011101100101010000110010000100001010101110001001111011111100110100100011000000010110011101000101;
            control_mem_6 <= 128'b11111110110111001011101010011000011101100101010000110010000100001001100010111010110111001111111000010000001100100101010001110110;
            control_mem_7 <= 128'b11111110110111001011101010011000011101100101010000110010000100001000100110101011110011011110111100000001001000110100010101100111;
            control_mem_8 <= 128'b11111110110111001011101010011000011101100101010000110010000100000111011001010100001100100001000011111110110111001011101010011000;
            control_mem_9 <= 128'b11111110110111001011101010011000011101100101010000110010000100000110011101000101001000110000000111101111110011011010101110001001;
            control_mem_10 <= 128'b11111110110111001011101010011000011101100101010000110010000100000101010001110110000100000011001011011100111111101001100010111010;
            control_mem_11 <= 128'b11111110110111001011101010011000011101100101010000110010000100000100010101100111000000010010001111001101111011111000100110101011;
            control_mem_12 <= 128'b11111110110111001011101010011000011101100101010000110010000100000011001000010000011101100101010010111010100110001111111011011100;
            control_mem_13 <= 128'b11111110110111001011101010011000011101100101010000110010000100000010001100000001011001110100010110101011100010011110111111001101;
            control_mem_14 <= 128'b11111110110111001011101010011000011101100101010000110010000100000001000000110010010101000111011010011000101110101101110011111110;
            control_mem_15 <= 128'b11111110110111001011101010011000011101100101010000110010000100000000000100100011010001010110011110001001101010111100110111101111;

            control_perm0[0] <= 64'b1111111011011100101110101001100001110110010101000011001000010000;
            control_perm0[1] <= 64'b1110111111001101101010111000100101100111010001010010001100000001;
            control_perm0[2] <= 64'b1101110011111110100110001011101001010100011101100001000000110010;
            control_perm0[3] <= 64'b1100110111101111100010011010101101000101011001110000000100100011;
            control_perm0[4] <= 64'b1011101010011000111111101101110000110010000100000111011001010100;
            control_perm0[5] <= 64'b1010101110001001111011111100110100100011000000010110011101000101;
            control_perm0[6] <= 64'b1001100010111010110111001111111000010000001100100101010001110110;
            control_perm0[7] <= 64'b1000100110101011110011011110111100000001001000110100010101100111;
            control_perm0[8] <= 64'b0111011001010100001100100001000011111110110111001011101010011000;
            control_perm0[9] <= 64'b0110011101000101001000110000000111101111110011011010101110001001;
            control_perm0[10] <= 64'b0101010001110110000100000011001011011100111111101001100010111010;
            control_perm0[11] <= 64'b0100010101100111000000010010001111001101111011111000100110101011;
            control_perm0[12] <= 64'b0011001000010000011101100101010010111010100110001111111011011100;
            control_perm0[13] <= 64'b0010001100000001011001110100010110101011100010011110111111001101;
            control_perm0[14] <= 64'b0001000000110010010101000111011010011000101110101101110011111110;
            control_perm0[15] <= 64'b0000000100100011010001010110011110001001101010111100110111101111;

            control_perm2[0] <= 64'b1111111011011100101110101001100001110110010101000011001000010000;
            control_perm2[1] <= 64'b1110111111001101101010111000100101100111010001010010001100000001;
            control_perm2[2] <= 64'b1101110011111110100110001011101001010100011101100001000000110010;
            control_perm2[3] <= 64'b1100110111101111100010011010101101000101011001110000000100100011;
            control_perm2[4] <= 64'b1011101010011000111111101101110000110010000100000111011001010100;
            control_perm2[5] <= 64'b1010101110001001111011111100110100100011000000010110011101000101;
            control_perm2[6] <= 64'b1001100010111010110111001111111000010000001100100101010001110110;
            control_perm2[7] <= 64'b1000100110101011110011011110111100000001001000110100010101100111;
            control_perm2[8] <= 64'b0111011001010100001100100001000011111110110111001011101010011000;
            control_perm2[9] <= 64'b0110011101000101001000110000000111101111110011011010101110001001;
            control_perm2[10] <= 64'b0101010001110110000100000011001011011100111111101001100010111010;
            control_perm2[11] <= 64'b0100010101100111000000010010001111001101111011111000100110101011;
            control_perm2[12] <= 64'b0011001000010000011101100101010010111010100110001111111011011100;
            control_perm2[13] <= 64'b0010001100000001011001110100010110101011100010011110111111001101;
            control_perm2[14] <= 64'b0001000000110010010101000111011010011000101110101101110011111110;
            control_perm2[15] <= 64'b0000000100100011010001010110011110001001101010111100110111101111;
            // [END] VAR INIT

            valid_out_delay <= 0;
            flush_counter = 16;
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
                counter_mem_addr_0 <= counter_mem_addr_0+1;
                counter_mem_addr_1 <= counter_mem_addr_1+1;
                counter_mem_addr_2 <= counter_mem_addr_2+1;
                counter_mem_addr_3 <= counter_mem_addr_3+1;
                counter_mem_addr_4 <= counter_mem_addr_4+1;
                counter_mem_addr_5 <= counter_mem_addr_5+1;
                counter_mem_addr_6 <= counter_mem_addr_6+1;
                counter_mem_addr_7 <= counter_mem_addr_7+1;
                counter_mem_addr_8 <= counter_mem_addr_8+1;
                counter_mem_addr_9 <= counter_mem_addr_9+1;
                counter_mem_addr_10 <= counter_mem_addr_10+1;
                counter_mem_addr_11 <= counter_mem_addr_11+1;
                counter_mem_addr_12 <= counter_mem_addr_12+1;
                counter_mem_addr_13 <= counter_mem_addr_13+1;
                counter_mem_addr_14 <= counter_mem_addr_14+1;
                counter_mem_addr_15 <= counter_mem_addr_15+1;

                mem_addr[0] <= control_mem_0[counter_mem_addr_0*MEM_ADDR_WIDTH+:MEM_ADDR_WIDTH];
                mem_addr[1] <= control_mem_1[counter_mem_addr_1*MEM_ADDR_WIDTH+:MEM_ADDR_WIDTH];
                mem_addr[2] <= control_mem_2[counter_mem_addr_2*MEM_ADDR_WIDTH+:MEM_ADDR_WIDTH];
                mem_addr[3] <= control_mem_3[counter_mem_addr_3*MEM_ADDR_WIDTH+:MEM_ADDR_WIDTH];
                mem_addr[4] <= control_mem_4[counter_mem_addr_4*MEM_ADDR_WIDTH+:MEM_ADDR_WIDTH];
                mem_addr[5] <= control_mem_5[counter_mem_addr_5*MEM_ADDR_WIDTH+:MEM_ADDR_WIDTH];
                mem_addr[6] <= control_mem_6[counter_mem_addr_6*MEM_ADDR_WIDTH+:MEM_ADDR_WIDTH];
                mem_addr[7] <= control_mem_7[counter_mem_addr_7*MEM_ADDR_WIDTH+:MEM_ADDR_WIDTH];
                mem_addr[8] <= control_mem_8[counter_mem_addr_8*MEM_ADDR_WIDTH+:MEM_ADDR_WIDTH];
                mem_addr[9] <= control_mem_9[counter_mem_addr_9*MEM_ADDR_WIDTH+:MEM_ADDR_WIDTH];
                mem_addr[10] <= control_mem_10[counter_mem_addr_10*MEM_ADDR_WIDTH+:MEM_ADDR_WIDTH];
                mem_addr[11] <= control_mem_11[counter_mem_addr_11*MEM_ADDR_WIDTH+:MEM_ADDR_WIDTH];
                mem_addr[12] <= control_mem_12[counter_mem_addr_12*MEM_ADDR_WIDTH+:MEM_ADDR_WIDTH];
                mem_addr[13] <= control_mem_13[counter_mem_addr_13*MEM_ADDR_WIDTH+:MEM_ADDR_WIDTH];
                mem_addr[14] <= control_mem_14[counter_mem_addr_14*MEM_ADDR_WIDTH+:MEM_ADDR_WIDTH];
                mem_addr[15] <= control_mem_15[counter_mem_addr_15*MEM_ADDR_WIDTH+:MEM_ADDR_WIDTH];

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
