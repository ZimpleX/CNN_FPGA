`include "common.vh"
module HAC (
  input clk,
  input reset,
  input complex_t image [0:15][0:15],
  input complex_t kernel [0:15][0:15],
  output complex out [0:15][0:15],
  input refresh,
  input next,
  output reg next_out,
  output [11:0] count_out 
  );

parameter f=77;
parameter N=16;
reg [10:0] count;
reg [15:0] imreg_r [0:15][0:15];
reg [15:0] imreg_i [0:15][0:15];
reg run;
complex acc_ori [0:15][0:15];
complex acc_out [0:15][0:15];
wire next_out_acc [0:15][0:15];

genvar i,j;
generate
  for(i=0;i<16;i=i+1) begin: acc_row
    for(j=0;j<16;j=j+1) begin: acc_column
      assign acc_ori[i][j].r = imreg_r[i][j];
      assign acc_ori[i][j].i = imreg_i[i][j];
      complexMultAddCanonicalfxp16fxp16 acc_inst(
        .clk(clk),
        .reset(reset),
        .in0(kernel[i][j]),
        .in1(image[i][j]),
        .acc(acc_ori[i][j]),
        .out(acc_out[i][j]),
        .next(next),
        .next_out(next_out_acc[i][j])
        );
      assign out[i][j].r = imreg_r[i][j];
      assign out[i][j].i = imreg_i[i][j];
    end // acc_column
  end // acc_row
endgenerate

always @(posedge clk) begin
    integer i,j;
    for(i=0;i<16;i=i+1) begin
      for(j=0;j<16;j++) begin
        if(next_out_acc[0][0] && (~refresh) && run) begin
          imreg_r[i][j] <= acc_out[i][j].r;
          imreg_i[i][j] <= acc_out[i][j].i;
          count <= (count==(f-1)*N)?0:count+1;
          run <= (count==(f-1)*N)?0:1;
          next_out <= (count==(f-1)*N)?1:0;
        end
        else if(refresh) begin
          imreg_i[i][j] <= 0;
          imreg_r[i][j] <= 0;
          count <= 0;
          next_out <= 0;
          run<=1;
        end
      end
    end
end
assign count_out = count;
endmodule