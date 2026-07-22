@echo off
call C:\Users\ksmer\Downloads\oss-cad-suite\environment.bat
iverilog -o mla.out mla_core_v3.v tb_mla_v3.v
vvp mla.out
copy mla.vcd C:\Users\ksmer\Desktop\NPU_Portfolio\mla.vcd

yosys -p "read_verilog npu_params.vh mla_core_v3.v; prep -top mla_core_v3; show -format svg -prefix mla_schematic mla_core_v3"
"C:\Program Files\Graphviz\bin\dot.exe" -Tpng mla_schematic.dot -o C:\Users\ksmer\Desktop\NPU_Portfolio\mla_schematic.png
"C:\Program Files\Graphviz\bin\dot.exe" -Tsvg mla_schematic.dot -o C:\Users\ksmer\Desktop\NPU_Portfolio\mla_schematic.svg
copy mla_schematic.svg C:\Users\ksmer\Desktop\NPU_Portfolio\mla_schematic.svg
