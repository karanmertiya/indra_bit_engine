@echo off
call C:\Users\ksmer\Downloads\oss-cad-suite\environment.bat
iverilog -o moe.out moe_router_v3.v tb_moe_v3.v
vvp moe.out
copy moe.vcd C:\Users\ksmer\Desktop\NPU_Portfolio\moe.vcd

yosys -p "read_verilog npu_params.vh moe_router_v3.v; prep -top moe_router_v3; show -format svg -prefix moe_schematic moe_router_v3"
"C:\Program Files\Graphviz\bin\dot.exe" -Tpng moe_schematic.dot -o C:\Users\ksmer\Desktop\NPU_Portfolio\moe_schematic.png
"C:\Program Files\Graphviz\bin\dot.exe" -Tsvg moe_schematic.dot -o C:\Users\ksmer\Desktop\NPU_Portfolio\moe_schematic.svg
copy moe_schematic.svg C:\Users\ksmer\Desktop\NPU_Portfolio\moe_schematic.svg
