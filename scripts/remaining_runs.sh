#!/bin/bash
# List of seeds to run

SEEDS=(0 1 2 3 4 5 6 7 8 9)
NUM_PARALLEL=3
CONFIG_FILES=(
    "Penicillin/MOHOLLM-GP-Botorch-Penicillin-Context-Gemini"
    "VehicleSafety/MOHOLLM-GP-Botorch-VehicleSafety-Context-Gemini"
    "CarSideImpact/MOHOLLM-GP-Botorch-CarSideImpact-Context-Gemini"
    "Penicillin/MOHOLLM-qLogEHVI-Penicillin-Context-Gemini"
    "VehicleSafety/MOHOLLM-qLogEHVI-VehicleSafety-Context-Gemini"
    "CarSideImpact/MOHOLLM-qLogEHVI-CarSideImpact-Context-Gemini"
)

for CONFIG_FILE in "${CONFIG_FILES[@]}"; do
    printf "%s\n" "${SEEDS[@]}" | xargs -I{} -P $NUM_PARALLEL bash -c 'python main.py --config_file="$0" --seed "{}" --seed_id "{}"' "$CONFIG_FILE"
done

