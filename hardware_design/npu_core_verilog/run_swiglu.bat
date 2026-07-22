@echo off
call C:\Users\ksmer\Downloads\oss-cad-suite\environment.bat
iverilog -o swiglu.out swiglu_act_v3.v tb_swiglu_v3.v
vvp swiglu.out
copy swiglu.vcd C:\Users\ksmer\Desktop\NPU_Portfolio\swiglu.vcd

yosys -p "read_verilog npu_params.vh swiglu_act_v3.v; prep -top swiglu_act_v3; show -format svg -prefix swiglu_schematic swiglu_act_v3"
dot -Tpng swiglu_schematic.dot -o C:\Users\ksmer\Desktop\NPU_Portfolio\swiglu_schematic.png
copy swiglu_schematic.svg C:\Users\ksmer\Desktop\NPU_Portfolio\swiglu_schematic.svg
