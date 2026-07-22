create_clock -period 10 [get_ports aclk]
set_input_delay -max 2 -clock [get_clocks aclk] [all_inputs]
set_output_delay -max 2 -clock [get_clocks aclk] [all_outputs]
