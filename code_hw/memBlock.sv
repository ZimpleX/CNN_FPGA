
`include "common.vh"

/* a generic dual port ram */
module dual_port_ram # (
  parameter BIT_WIDTH = 11,
  parameter ADDR_WIDTH = 11,
  parameter f=77,
  parameter N=16
) (
  input clk,    // Clock
  input we, // Write Enable
  input complex_t data_in[0:15],
  input [ADDR_WIDTH-1:0] write_address,
  input [ADDR_WIDTH-1:0] read_address,
  output reg [BIT_WIDTH*N*2-1:0] data_out [N*2-1:0]
);

  reg [BIT_WIDTH*N*2-1:0] ram [f*N-1:0];
  wire [BIT_WIDTH*N*2-1:0] wdata_in;
  genvar i,j;
  generate
    for(i=0;i<N;i=i+1) begin:in_loop
      assign wdata_in[(i*2+1)*BIT_WIDTH-1:i*2*BIT_WIDTH] = data_in[i].r;
      assign wdata_in[(i*2+2)*BIT_WIDTH-1:(i*2+1)*BIT_WIDTH] = data_in[i].i;
    end
  endgenerate
  always@(posedge clk) begin
    integer i;
    if (we) begin
      ram[write_address] <=wdata_in;
    end
    for(i=0;i<2*N;i=i+1) begin
      data_out[i] <= (read_address+i<f*N)?ram[read_address+i]:0;
    end
  end

endmodule

module kern_ram # (
  parameter BIT_WIDTH = 11,
  parameter ADDR_WIDTH = 25,
  parameter f=77,
  parameter N=16
) (
  input clk,    // Clock
  input we, // Write Enable
  input data_in,
  input [ADDR_WIDTH-1:0] write_address,
  input [ADDR_WIDTH-1:0] read_address,
  output reg [BIT_WIDTH*N*2-1:0] data_out [N*2-1:0]
);

  reg [BIT_WIDTH*N*2-1:0] ram [f*f*N-1:0];
  wire [BIT_WIDTH*N*2-1:0] wdata_in;
  genvar i,j;
  always@(posedge clk) begin
    integer i;
    if (we) begin
      ram[write_address] <=data_in;
    end
    for(i=0;i<2*N;i=i+1) begin
      data_out[i] <= (read_address+i<f*N)?ram[read_address+i]:0;
    end
  end

endmodule