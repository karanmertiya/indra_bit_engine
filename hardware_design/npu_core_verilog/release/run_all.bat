@echo off
setlocal
cd /d "%~dp0"

rem Ensure OSS CAD Suite environment is loaded
call "C:\Users\ksmer\Downloads\oss-cad-suite\environment.bat"
if %errorlevel% neq 0 (
    echo [ERROR] OSS CAD Suite environment failed to load
    exit /b %errorlevel%
)

rem Regenerate stimulus and golden reference hex files from the release-local model
python golden_model_v2.py
if %errorlevel% neq 0 (
    echo [ERROR] Golden model generation failed
    exit /b %errorlevel%
)

rem Remove previous simulation artifacts
if exist tile_sim.out del /f tile_sim.out
if exist integrated_tile.vcd del /f integrated_tile.vcd

rem Compile all RTL and testbench
iverilog -o tile_sim.out -Wall -g2012 npu_params.vh systolic_array_16x16.v mac_cell.v sram_model.v rope_act_v3.v rope_vector_unit.v npu_tile_top.v tb_integrated_tile.v
if %errorlevel% neq 0 (
    echo [ERROR] Compilation failed
    exit /b %errorlevel%
)

rem Run simulation
vvp tile_sim.out
if %errorlevel% neq 0 (
    echo [ERROR] Simulation failed
    exit /b %errorlevel%
)

echo [SUCCESS] Simulation completed
endlocal
