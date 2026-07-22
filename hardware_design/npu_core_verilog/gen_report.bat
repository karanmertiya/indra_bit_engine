@echo off
call C:\Users\ksmer\Downloads\oss-cad-suite\environment.bat
echo === SwiGLU Synthesis Stats === > C:\Users\ksmer\Desktop\NPU_Portfolio\synthesis_report.txt
yosys -p "read_verilog npu_params.vh swiglu_act_v3.v; prep -top swiglu_act_v3; stat" >> C:\Users\ksmer\Desktop\NPU_Portfolio\synthesis_report.txt 2>&1
echo === RoPE Synthesis Stats === >> C:\Users\ksmer\Desktop\NPU_Portfolio\synthesis_report.txt
yosys -p "read_verilog npu_params.vh rope_act_v3.v; prep -top rope_act_v3; stat" >> C:\Users\ksmer\Desktop\NPU_Portfolio\synthesis_report.txt 2>&1
echo === MoE Synthesis Stats === >> C:\Users\ksmer\Desktop\NPU_Portfolio\synthesis_report.txt
yosys -p "read_verilog npu_params.vh moe_router_v3.v; prep -top moe_router_v3; stat" >> C:\Users\ksmer\Desktop\NPU_Portfolio\synthesis_report.txt 2>&1
echo === MLA Synthesis Stats === >> C:\Users\ksmer\Desktop\NPU_Portfolio\synthesis_report.txt
yosys -p "read_verilog npu_params.vh mla_core_v3.v; prep -top mla_core_v3; stat" >> C:\Users\ksmer\Desktop\NPU_Portfolio\synthesis_report.txt 2>&1
