// SPDX-License-Identifier: MIT
// Boilerplate ERI compressor module for FPGA, based on Wu et al. (2023).

module eri_compressor #(
  parameter integer NBIT        = 16,
  parameter integer ERI_COUNT   = 512,
  parameter integer OUT_WIDTH   = 512
)(
  input  wire                 clk,
  input  wire                 rst_n,
  input  wire                 in_valid,
  output wire                 in_ready,
  input  wire [31:0]          in_eri,
  input  wire                 in_last_quartet,
  output reg                  out_valid,
  input  wire                 out_ready,
  output reg  [OUT_WIDTH-1:0] out_data,
  output reg                  out_last_quartet,
  output reg                  hdr_valid,
  input  wire                 hdr_ready,
  output reg  [31:0]          hdr_bmax,
  output reg  [31:0]          hdr_nvals
);

  localparam integer PER_WORD = OUT_WIDTH / NBIT;

  reg  [31:0] eri_buf   [0:ERI_COUNT-1];
  integer write_idx, read_idx;

  reg [31:0] bmax;
  typedef enum logic [1:0] {S_FILL=2'b00, S_COMPRESS=2'b01, S_FLUSH=2'b10} state_t;
  state_t state, next_state;

  assign in_ready = (state == S_FILL);

  function [31:0] fabs32(input [31:0] f); begin
    fabs32 = {1'b0, f[30:0]};
  end endfunction

  real epsilon_r;
  real bmax_r;

  reg [OUT_WIDTH-1:0] pack_shift;

  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state <= S_FILL;
      write_idx <= 0;
      read_idx  <= 0;
      bmax <= 32'h00000000;
      out_valid <= 1'b0;
      hdr_valid <= 1'b0;
      out_last_quartet <= 1'b0;
      hdr_bmax <= 32'h0;
      hdr_nvals <= 32'h0;
      pack_shift <= {OUT_WIDTH{1'b0}};
    end else begin
      state <= next_state;
      if (state == S_FILL && in_valid && in_ready) begin
        eri_buf[write_idx] <= in_eri;
        write_idx <= write_idx + 1;
        if (fabs32(in_eri) > bmax) bmax <= fabs32(in_eri);
      end

      if (state == S_COMPRESS && hdr_valid == 1'b0) begin
        hdr_valid <= 1'b1;
        hdr_bmax  <= bmax;
        hdr_nvals <= ERI_COUNT;
      end else if (hdr_valid && hdr_ready) begin
        hdr_valid <= 1'b0;
      end

      if (state == S_COMPRESS && (!out_valid || (out_valid && out_ready))) begin
        integer i;
        bmax_r = $bitstoshortreal(bmax);
        if (bmax_r == 0.0) bmax_r = 1.0;
        epsilon_r = bmax_r / ( (1<<NBIT) - 1 );
        pack_shift = '0;
        for (i = 0; i < PER_WORD; i = i + 1) begin
          if (read_idx < ERI_COUNT) begin
            real eri_r = $bitstoshortreal(eri_buf[read_idx]);
            integer q  = $rtoi( (eri_r / epsilon_r) + (eri_r>=0 ? 0.5 : -0.5) );
            pack_shift[i*NBIT +: NBIT] = q[NBIT-1:0];
            read_idx = read_idx + 1;
          end
        end
        out_data  <= pack_shift;
        out_valid <= 1'b1;
        out_last_quartet <= (read_idx >= ERI_COUNT);
      end

      if (out_valid && out_ready) begin
        out_valid <= 1'b0;
      end

      if (state == S_FLUSH && next_state == S_FILL) begin
        write_idx <= 0;
        read_idx  <= 0;
        bmax <= 32'h00000000;
        out_last_quartet <= 1'b0;
      end
    end
  end

  always @(*) begin
    next_state = state;
    case (state)
      S_FILL: begin
        if (in_valid && in_last_quartet && in_ready) next_state = S_COMPRESS;
      end
      S_COMPRESS: begin
        if (out_last_quartet && out_valid && out_ready && !hdr_valid) next_state = S_FLUSH;
      end
      S_FLUSH: begin
        next_state = S_FILL;
      end
    endcase
  end

endmodule
