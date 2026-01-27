#!/bin/bash
trials=100
popsize=10
methods=("MOEAD" "CTAEA" "RNSGA2" "UNSGA3" "SMSEMOA" "RVEA" "NSGA2" "NSGA3")
seeds=("42" "6790" "31415927" "1" "2" "3" "4" "5" "6" "7")
NUM_PARALLEL=4


# problems=(
#     "chankong_haimes" "test_function_4" "schaffer_n1" "schaffer_n2" "poloni"
#     "dtlz1" "dtlz2" "vehicle_safety" "toy_robust" "penicillin" "GMM"
#     "car_side_impact" "branin_currin"
# )


problems=(
    "kursawe"
    "dtlz3"
)


# Step 1: Generate all combinations of (seed, problem, method) and print them.
# We now use a single line with a clear separator (space) for each job.
(
    for seed in "${seeds[@]}"; do
        for problem in "${problems[@]}"; do
            for method in "${methods[@]}"; do
                # Print each complete command specification on a single line
                printf "%s %s %s\n" "$seed" "$problem" "$method"
            done
        done
    done
) | \
# Step 2: Pipe the generated list into xargs to run in parallel.
xargs -P $NUM_PARALLEL -L 1 -- sh -c '
    # xargs reads one line at a time (-L 1).
    # The entire line is passed as arguments to `sh -c`.
    # $0 will be `sh`, $1 is the first word (seed), $2 is the second (problem), etc.
    SEED="$1"
    PROBLEM="$2"
    METHOD="$3"

    echo "--- Starting: [Seed=$SEED, Problem=$PROBLEM, Method=$METHOD] ---"

    python run_pymoo_methods.py \
        --method "$METHOD" \
        --seed "$SEED" \
        --trials "'"$trials"'" \
        --pop_size "'"$popsize"'" \
        --problem "$PROBLEM"

    echo "--- Finished: [Seed=$SEED, Problem=$PROBLEM, Method=$METHOD] ---"
' -- # The -- is a good practice to separate xargs options from the command
