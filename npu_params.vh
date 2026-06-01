`ifndef NPU_PARAMS_VH
`define NPU_PARAMS_VH

// Global NPU Configurations
`define ARRAY_SIZE 16

// Data Types
`define INT8_W 8
`define INT32_W 32

// AXI-Stream Data Bus Width (16 elements * 8 bits = 128 bits)
`define AXIS_DATA_W (`ARRAY_SIZE * `INT8_W)

`endif // NPU_PARAMS_VH
