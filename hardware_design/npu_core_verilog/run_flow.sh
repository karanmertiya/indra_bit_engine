#!/bin/bash
set -e
# Synthesis with Yosys (included in the OpenROAD Docker image)
verilog_files=$(ls *.v | grep -v '^tb_' | grep -v '^npu_gate.v' | tr '\n' ' ')
echo "Verilog files: $verilog_files"
yosys -p "read_verilog -sv $verilog_files; hierarchy -top npu_core_top; synth -top npu_core_top; dfflibmap -liberty /foss/pdks/ciel/sky130/versions/7b70722e33c03fcb5dabcf4d479fb0822d9251c9/sky130A/libs.ref/sky130_fd_sc_hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib; abc -liberty /foss/pdks/ciel/sky130/versions/7b70722e33c03fcb5dabcf4d479fb0822d9251c9/sky130A/libs.ref/sky130_fd_sc_hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib; hilomap -hicell sky130_fd_sc_hd__conb_1 HI -locell sky130_fd_sc_hd__conb_1 LO; clean; write_verilog -noattr npu_gate.v"
# OpenROAD flow (also included in the image)
openroad -no_init -exit <<EOF
read_lef /foss/pdks/ciel/sky130/versions/7b70722e33c03fcb5dabcf4d479fb0822d9251c9/sky130A/libs.ref/sky130_fd_sc_hd/techlef/sky130_fd_sc_hd__nom.tlef
read_lef /foss/pdks/ciel/sky130/versions/7b70722e33c03fcb5dabcf4d479fb0822d9251c9/sky130A/libs.ref/sky130_fd_sc_hd/lef/sky130_fd_sc_hd.lef
read_liberty /foss/pdks/ciel/sky130/versions/7b70722e33c03fcb5dabcf4d479fb0822d9251c9/sky130A/libs.ref/sky130_fd_sc_hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib
read_verilog npu_gate.v
link_design npu_core_top
read_sdc constraints.sdc
set_thread_count 8
initialize_floorplan -utilization 30 -aspect_ratio 1.0 -core_space 10.0 -site unithd
make_tracks
place_pins -hor_layer met3 -ver_layer met2
global_placement -density 0.6
detailed_placement
global_route
estimate_parasitics -global_routing
write_def final.def
write_db final.spg
write_spef final.spef
report_design_area
report_timing
report_power
EOF
