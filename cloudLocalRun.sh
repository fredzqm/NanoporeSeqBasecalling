gcloud ml-engine local train \
	--job-dir output \
	--module-name trainer.task \
	--package-path keras/trainer/ \
	-- \
	--train-files \
		train/FMH_15Le080325s_20161103_FNFAB42798_MN17638_sequencing_run_161103_Human5_LSK108R9_4_13493_ch100_read1460_strand.label \
		train/FMH_15Le080325s_20161103_FNFAB42798_MN17638_sequencing_run_161103_Human5_LSK108R9_4_13493_ch100_read1460_strand.signal \
		train/FMH_15Le080325s_20161103_FNFAB42798_MN17638_sequencing_run_161103_Human5_LSK108R9_4_13493_ch101_read605_strand.label \
		train/FMH_15Le080325s_20161103_FNFAB42798_MN17638_sequencing_run_161103_Human5_LSK108R9_4_13493_ch101_read605_strand.signal \
		train/FMH_15Le080325s_20161103_FNFAB42798_MN17638_sequencing_run_161103_Human5_LSK108R9_4_13493_ch102_read583_strand.label \
		train/FMH_15Le080325s_20161103_FNFAB42798_MN17638_sequencing_run_161103_Human5_LSK108R9_4_13493_ch102_read583_strand.signal \
	--validate-files \
		train/FMH_15Le080325s_20161103_FNFAB42798_MN17638_sequencing_run_161103_Human5_LSK108R9_4_13493_ch104_read4125_strand.label \
		train/FMH_15Le080325s_20161103_FNFAB42798_MN17638_sequencing_run_161103_Human5_LSK108R9_4_13493_ch104_read4125_strand.signal \
	--eval-files \
		val/FMH_15Le080325s_20161103_FNFAB42798_MN17638_sequencing_run_161103_Human5_LSK108R9_4_13493_ch103_read1667_strand.label \
		val/FMH_15Le080325s_20161103_FNFAB42798_MN17638_sequencing_run_161103_Human5_LSK108R9_4_13493_ch103_read1667_strand.signal \
	--train-steps 50 \
	--num-epochs 20 \
	--early-stop-patience 5 \
	--train-batch-size 50 \
	--checkpoint-epochs 5 \
	--eval-frequency 5 \
	--eval-batch-size 70 \
	--eval-num-epochs 20 \
	--verbosity DEBUG