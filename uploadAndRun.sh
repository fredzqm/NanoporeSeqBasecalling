rm -rf keras\output
gcloud auth activate-service-account --key-file NanoporeSequence-dab13aa49ef5.json
gcloud compute --project "nanoporesequence" ssh --zone "us-east1-b" "fredzqm@deep-learning-compute" --command "rm -rf keras/*"
gcloud compute --project "nanoporesequence" scp --zone "us-east1-b" keras/*  fredzqm@deep-learning-compute:keras/ --recurse
gcloud compute --project "nanoporesequence" ssh --zone "us-east1-b" "fredzqm@deep-learning-compute" --command "cd keras; gcloud ml-engine local train --module-name trainer.task --package-path trainer -- --train-files train/FMH_15Le080325s_20161103_FNFAB42798_MN17638_sequencing_run_161103_Human5_LSK108R9_4_13493_ch100_read1460_strand.label train/FMH_15Le080325s_20161103_FNFAB42798_MN17638_sequencing_run_161103_Human5_LSK108R9_4_13493_ch100_read1460_strand.signal --eval-files train/FMH_15Le080325s_20161103_FNFAB42798_MN17638_sequencing_run_161103_Human5_LSK108R9_4_13493_ch100_read1460_strand.label train/FMH_15Le080325s_20161103_FNFAB42798_MN17638_sequencing_run_161103_Human5_LSK108R9_4_13493_ch100_read1460_strand.signal --train-steps 100 --train-batch-size 120 --job-dir output --eval-steps 100"
mkdir keras\output
gcloud compute --project "nanoporesequence" scp --zone "us-east1-b" fredzqm@deep-learning-compute:keras/output/* keras/output --recurse 
