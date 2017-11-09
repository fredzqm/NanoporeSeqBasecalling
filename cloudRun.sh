gcloud ml-engine jobs submit training first_job_3 \
	--job-dir gs://chiron-data-fred/output \
	--runtime-version 1.2 \
	--module-name trainer.task \
	--package-path trainer/ \
	--region 'us-east1' \
	-- \
	--train-files train/FMH_15Le080325s_20161103_FNFAB42798_MN17638_sequencing_run_161103_Human5_LSK108R9_4_13493_ch100_read1460_strand.label \
				train/FMH_15Le080325s_20161103_FNFAB42798_MN17638_sequencing_run_161103_Human5_LSK108R9_4_13493_ch100_read1460_strand.signal \
	--eval-files train/FMH_15Le080325s_20161103_FNFAB42798_MN17638_sequencing_run_161103_Human5_LSK108R9_4_13493_ch100_read1460_strand.label \
				train/FMH_15Le080325s_20161103_FNFAB42798_MN17638_sequencing_run_161103_Human5_LSK108R9_4_13493_ch100_read1460_strand.signal \
	--train-steps 1000 \
	--verbosity DEBUG