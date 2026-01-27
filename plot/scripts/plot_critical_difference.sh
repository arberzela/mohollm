#!/bin/bash

python ./plot/plot_critical_difference_diagram.py \
    --benchmarks DTLZ1,DTLZ2,DTLZ3,BraninCurrin,GMM,Poloni,ChankongHaimes,SchafferN1,SchafferN2,TestFunction4,ToyRobust,Penicillin,CarSideImpact,VehicleSafety,Kursawe \
    --title "Critical Difference" \
    --trials 50 \
    --whitelist "CTAEA,EHVI,GDE3,IBEA,MOEAD,MOHOLLM (Gemini 2.0 Flash),MOHOLLM (Gemini 2.0 Flash) (Context),mohollm (Gemini 2.0 Flash),NSGA2,NSGA3,PESA2,qLogEHVI,RNSGA2,SMSEMOA,SPEA2,UNSGA3" \
    --filename "critical_difference_diagram"

