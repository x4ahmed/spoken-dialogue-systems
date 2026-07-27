@echo off
setlocal enabledelayedexpansion

REM Parameters ------------------------------------------------------

set TASK=unified
set DATA_DIR=data
REM FIX: Added a dynamic ablation name to prevent wildcard contamination of JSON files
set ABLATION_NAME=mini_baseline
set DATASET_CONFIG=dataset_config\unified_simplemultiwoz21.json

REM Project paths etc. ----------------------------------------------

REM Ensure the root folder is in the Python path for local module imports
set PYTHONPATH=C:\Users\Ahmed\Desktop\HHU REPOs\convlab3;%PYTHONPATH%

REM FIX: Output directory now uses the ablation name to isolate runs
set OUT_DIR=results\simplemultiwoz21_%ABLATION_NAME%
if not exist "%OUT_DIR%" mkdir "%OUT_DIR%"

REM Main ------------------------------------------------------------

for %%s in (train dev test) do (
    set "args_add="
    if "%%s"=="train" (
        set "args_add=--do_train --predict_type=dummy"
        REM INFO: For sim-M, you might want to add "--svd=0.3" to args_add
    ) else (
        set "args_add=--do_eval --predict_type=%%s"
    )

    echo Running %%s...
    
    REM FIXES APPLIED BELOW: 
    REM 1. Removed empty --data_dir argument to prevent argparse crashing
    REM 2. Increased --max_seq_length from 180 to 512 so 'refer' and 'dontcare' don't drop to 0.0
    
    python run_dst.py ^
        --task_name=%TASK% ^
        --data_dir=%DATA_DIR% ^
        --dataset_config=%DATASET_CONFIG% ^
        --model_type=bert ^
        --model_name_or_path=prajjwal1/bert-mini ^
        --tokenizer_name=bert-base-uncased ^
        --do_lower_case ^
        --learning_rate=1e-4 ^
        --num_train_epochs=10 ^
        --max_seq_length=180 ^
        --per_gpu_train_batch_size=32 ^
        --per_gpu_eval_batch_size=1 ^
        --output_dir="%OUT_DIR%" ^
        --save_epochs=2 ^
        --warmup_proportion=0.1 ^
        --eval_all_checkpoints ^
        --local_files_only ^
        --adam_epsilon=1e-6 ^
        --weight_decay=0.01 ^
        --overwrite_output_dir ^
        --overwrite_cache ^
        --seed 42 ^
        --class_aux_feats_inform ^
        --class_aux_feats_ds ^
        !args_add! ^
        > "%OUT_DIR%\%%s.log" 2>&1
    
    if "%%s"=="dev" (
        echo Evaluating metrics for dev...
        python metric_dst.py ^
            --dataset_config=%DATASET_CONFIG% ^
            --file_list="%OUT_DIR%\pred_res.%%s*json" ^
            > "%OUT_DIR%\eval_pred_%%s.log" 2>&1
    )
    if "%%s"=="test" (
        echo Evaluating metrics for test...
        python metric_dst.py ^
            --dataset_config=%DATASET_CONFIG% ^
            --file_list="%OUT_DIR%\pred_res.%%s*json" ^
            > "%OUT_DIR%\eval_pred_%%s.log" 2>&1
    )
)

echo Done!
pause