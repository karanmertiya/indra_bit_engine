@echo off
call C:\Users\ksmer\Downloads\oss-cad-suite\environment.bat
iverilog -o rope.out rope_act_v3.v tb_rope_v3.v
vvp rope.out
copy rope.vcd C:\Users\ksmer\Desktop\NPU_Portfolio\rope.vcd

yosys -p "read_verilog npu_params.vh rope_act_v3.v; prep -top rope_act_v3; show -format svg -prefix rope_schematic rope_act_v3"
"C:\Program Files\Graphviz\bin\dot.exe" -Tpng rope_schematic.dot -o C:\Users\ksmer\Desktop\NPU_Portfolio\rope_schematic.png
"C:\Program Files\Graphviz\bin\dot.exe" -Tsvg rope_schematic.dot -o C:\Users\ksmer\Desktop\NPU_Portfolio\rope_schematic.svg
copy rope_schematic.svg C:\Users\ksmer\Desktop\NPU_Portfolio\rope_schematic.svg
