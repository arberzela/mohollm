#!/bin/bash

# NOTE: Here we use real trial and not "trials as function evaluations"
python ./plot/plot_utility.py \
    --benchmarks DTLZ1,DTLZ2,DTLZ3,BraninCurrin,GMM,Poloni,Penicillin,CarSideImpact,SchafferN1,SchafferN2,TestFunction4,ToyRobust,VehicleSafety,Kursawe,ChankongHaimes,CarSideImpact \
    --title "Utility Difference" \
    --trials 12 \
    --whitelist "mohollm (Gemini 2.0 Flash),MOHOLLM (Gemini 2.0 Flash) (Context),MOHOLLM (Gemini 2.0 Flash)"\
    --filename "utility_plots"
