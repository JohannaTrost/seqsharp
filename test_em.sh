#!/bin/bash

dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

em_name="test"
em_config_path="configs/em/config.json"

save_path_estim=$(cat $em_config_path | grep -Eo '"save_path"[^}]*' | grep -Eo '[^: ]*$' | cut -d'"' -f2)
save_path_estim_new=$dir/$save_path_estim/$em_name

n_cl=$(cat $em_config_path | grep -Eo '"n_clusters"[^,]*' | grep -Eo '[^: ]*$')
new_n_cl="678"

# new config file (with new path to estimates)
em_config_dir="$(dirname "${em_config_path}")"
em_config_file=$id"$(basename "${em_config_path}")"

mkdir $save_path_estim_new

sed "s+$save_path_estim+$save_path_estim_new+g" $em_config_path > $em_config_dir/$em_config_file
sed "s+$n_cl+$new_n_cl+g" $em_config_path > $em_config_dir/$em_config_file

cat $em_config_path

# AA frequency estimators 
#hEM.py $em_config_dir/$em_config_file


