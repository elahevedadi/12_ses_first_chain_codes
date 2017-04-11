#PBS -N Elahe_fMRI
#PBS -m ae
#PBS -M vadadi_e@ee.sharif.edu
#PBS -l nodes=1:ppn=1
 
#echo "voxel_inds:$voxel_inds"
#echo "inf_method: ${inf_method}"
#echo "session_inds: $session_inds"
#echo "alph: $alph"
#echo "num_iter: ${num_iter}"
#echo "num_storing_sets_of_theta: ${num_storing_sets_of_theta}"
#echo "training_at_random: ${training_at_random}"
#echo "num_train_examp: ${num_train_examp}"

cd ~/NeuralfMRI/12_ses_first_chain_codes

#module load python/2.7.9/gcc-4.4.7

python complete_main_script.py -M $inf_method -F $session_inds -o $voxel_inds -X $num_iter -Y $num_storing_sets_of_theta -R $training_at_random -J $num_train_examp


