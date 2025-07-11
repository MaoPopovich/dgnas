#!/bin/bash


job_name=$1
command=$2

mkdir -p logs/log

$command 2>&1|tee -a logs/log/$job_name.log
