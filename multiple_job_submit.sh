nstep=180
sesionstep=8


for n in $(seq 0 ${nstep} 4318); do
    for sess_ind in $(seq 39 ${sesionstep} 45); do
        n1=${n}
        n2=$[${n}+${nstep}]

       # if sess_ind != 41; then
        sess_ind1=${sess_ind}
        sess_ind2=$[${sess_ind}+${sesionstep}]
        
        
        echo "${n1},${n2}"
        export n1 n2
        echo "\"${sess_ind1},${sess_ind2}\""
        export sess_ind1 sess_ind2
      #  fi
        
        qsub -v inf_method="L",alph="1.2",session_inds=""\"${sess_ind1},${sess_ind2}"\"",voxel_inds=""\"${n1},${n2}"\"",num_iter="20",num_storing_sets_of_theta="10",training_at_random="1",num_train_examp="0.9" job_submit.sh
     
        sleep 1 # pause to be kind to the scheduler
   done
done
