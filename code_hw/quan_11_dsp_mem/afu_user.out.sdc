## Generated SDC file "afu_user.out.sdc"

## Copyright (C) 2018  Intel Corporation. All rights reserved.
## Your use of Intel Corporation's design tools, logic functions 
## and other software and tools, and its AMPP partner logic 
## functions, and any output files from any of the foregoing 
## (including device programming or simulation files), and any 
## associated documentation or information are expressly subject 
## to the terms and conditions of the Intel Program License 
## Subscription Agreement, the Intel Quartus Prime License Agreement,
## the Intel FPGA IP License Agreement, or other applicable license
## agreement, including, without limitation, that your use is for
## the sole purpose of programming logic devices manufactured by
## Intel and sold by Intel or its authorized distributors.  Please
## refer to the applicable agreement for further details.


## VENDOR  "Altera"
## PROGRAM "Quartus Prime"
## VERSION "Version 18.0.0 Build 614 04/24/2018 SJ Standard Edition"

## DATE    "Sun Aug 12 00:28:29 2018"

##
## DEVICE  "5SGXEA7N3F45I3YY"
##


#**************************************************************
# Time Information
#**************************************************************

set_time_format -unit ns -decimal_places 3



#**************************************************************
# Create Clock
#**************************************************************

create_clock -name {clk_200M} -period 5.000 -waveform { 0.000 2.500 } [get_ports {clk}]


#**************************************************************
# Create Generated Clock
#**************************************************************



#**************************************************************
# Set Clock Latency
#**************************************************************



#**************************************************************
# Set Clock Uncertainty
#**************************************************************

set_clock_uncertainty -rise_from [get_clocks {clk_200M}] -rise_to [get_clocks {clk_200M}] -setup 0.070  
set_clock_uncertainty -rise_from [get_clocks {clk_200M}] -rise_to [get_clocks {clk_200M}] -hold 0.060  
set_clock_uncertainty -rise_from [get_clocks {clk_200M}] -fall_to [get_clocks {clk_200M}] -setup 0.070  
set_clock_uncertainty -rise_from [get_clocks {clk_200M}] -fall_to [get_clocks {clk_200M}] -hold 0.060  
set_clock_uncertainty -fall_from [get_clocks {clk_200M}] -rise_to [get_clocks {clk_200M}] -setup 0.070  
set_clock_uncertainty -fall_from [get_clocks {clk_200M}] -rise_to [get_clocks {clk_200M}] -hold 0.060  
set_clock_uncertainty -fall_from [get_clocks {clk_200M}] -fall_to [get_clocks {clk_200M}] -setup 0.070  
set_clock_uncertainty -fall_from [get_clocks {clk_200M}] -fall_to [get_clocks {clk_200M}] -hold 0.060  


#**************************************************************
# Set Input Delay
#**************************************************************



#**************************************************************
# Set Output Delay
#**************************************************************



#**************************************************************
# Set Clock Groups
#**************************************************************



#**************************************************************
# Set False Path
#**************************************************************



#**************************************************************
# Set Multicycle Path
#**************************************************************



#**************************************************************
# Set Maximum Delay
#**************************************************************



#**************************************************************
# Set Minimum Delay
#**************************************************************



#**************************************************************
# Set Input Transition
#**************************************************************

