@echo off
call C:\Users\ksmer\Downloads\oss-cad-suite\environment.bat
yosys -p "read_verilog npu_params.vh mac_cell.v systolic_array_16x16.v axi_stream_intf.v; prep -top systolic_array_16x16; show -format svg -prefix chip_schematic systolic_array_16x16"
