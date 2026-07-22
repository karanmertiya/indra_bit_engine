read_verilog llm_top.v
link_design
read_liberty /opt/sky130A/libs.tech/openlane/sky130_fd_sc_hd/sky130_fd_sc_hd__tt_025C_1v80.lib
read_lef /opt/sky130A/libs.tech/sky130_fd_sc_hd/sky130_fd_sc_hd.lef
set_placement_strategy "global"
place_design
global_route
router
write_def final.def
write_gds final.gds
report_design_area
