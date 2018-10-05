#!/bin/bash

## declare an array variable
declare games=("BreakoutNoFrameskip-v4" "AmidarNoFrameskip-v4")
declare n_options=(1 2 4 8)
declare delib_costs=(0.0 0.005 0.01 0.015 0.02 0.025 0.03)

for game in "${games[@]}"; do
	echo A2C game=$game
	python main.py --env-name $game --algo a2c --num-frames 10000000 --num-processes 8 --no-vis --save-dir ./save_files/ --log-dir-base-path ./save_files/
	# A2OC
	for option in "${n_options[@]}"; do
		if [ "$option" != "1" ]; then
			for delib in "${delib_costs[@]}"; do
				echo A2OC game=$game options=$option delib=$delib
				python main.py --env-name $game --algo a2oc --num-frames 10000000 --num-options $option --delib $delib --num-processes 8 --no-vis --save-dir ./save_files/ --log-dir-base-path ./save_files/
			done
		else
			echo A2OC game=$game options=$option
			python main.py --env-name $game --algo a2oc --num-frames 10000000 --num-options $option --num-processes 8 --no-vis --save-dir ./save_files/ --log-dir-base-path ./save_files/
		fi
	done
done
