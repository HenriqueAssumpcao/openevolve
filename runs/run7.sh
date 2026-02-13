#!/bin/bash

PROB_NAME="codeevolve_benchmarks/autocorrelation_problems/first_autocorr_ineq/"
BASE_DIR="examples/${PROB_NAME}"
CFG_PATH="${BASE_DIR}/configs/qwen_config.yaml"
EVAL_PATH="${BASE_DIR}/evaluate.py"
INIT_PROG_PATH="${BASE_DIR}/init_program.py"
OUT_DIR="experiments/${PROB_NAME}/qwen/ablations_comp/qwen_full_1"
CPU_LIST="60-69"
CKPT_PATH="${OUT_DIR}/checkpoints/checkpoint_1000"

API_KEY=$(python3 -c "
import boto3
import sys
def get_ssm_parameter(parameter_name: str, region_name: str) -> str:
    boto3_session = boto3.Session(region_name=region_name)
    ssm = boto3_session.client('ssm')
    response = ssm.get_parameter(Name=parameter_name, WithDecryption=True)
    return response['Parameter']['Value']
try:
    api_key = get_ssm_parameter('/MIND/PRD/EVOLVE-MIRROR', 'us-east-1')
    print(api_key)
except Exception as e:
    print(f'Error: {e}', file=sys.stderr)
    sys.exit(1)
")
if [ $? -eq 0 ] && [ ! -z "$API_KEY" ]; then
    export OPENAI_API_KEY="$API_KEY"
else
    echo "Failed to retrieve parameter from AWS SSM"
    exit 1
fi


taskset --cpu-list $CPU_LIST python openevolve-run.py $INIT_PROG_PATH $EVAL_PATH --config=$CFG_PATH --output=$OUT_DIR --checkpoint=$CKPT_PATH