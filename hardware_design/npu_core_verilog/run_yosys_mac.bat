@echo off
call C:\Users\ksmer\Downloads\oss-cad-suite\environment.bat
yosys -p "read_verilog npu_params.vh mac_cell.v; prep -top mac_cell; show -format svg -prefix mac_cell_schematic"
