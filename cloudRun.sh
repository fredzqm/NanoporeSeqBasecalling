gcloud ml-engine jobs submit training job_hp_12_5 ^
	--job-dir gs://chiron-data-fred/output/job_hp_12_5 ^
	--runtime-version 1.2 ^
	--module-name trainer.task ^
	--package-path keras/trainer/ ^
	--region us-east1 ^
	--config config.yaml ^
	-- ^
	--train-files ^
		train/FMH_15Le080325s_20161103_FNFAB42798_MN17638_sequencing_run_161103_Human5_LSK108R9_4_13493_ch100_read1460_strand.label ^
		train/FMH_15Le080325s_20161103_FNFAB42798_MN17638_sequencing_run_161103_Human5_LSK108R9_4_13493_ch100_read1460_strand.signal ^
		train/FMH_15Le080325s_20161103_FNFAB42798_MN17638_sequencing_run_161103_Human5_LSK108R9_4_13493_ch101_read605_strand.label ^
		train/FMH_15Le080325s_20161103_FNFAB42798_MN17638_sequencing_run_161103_Human5_LSK108R9_4_13493_ch101_read605_strand.signal ^
		train/FMH_15Le080325s_20161103_FNFAB42798_MN17638_sequencing_run_161103_Human5_LSK108R9_4_13493_ch102_read583_strand.label ^
		train/FMH_15Le080325s_20161103_FNFAB42798_MN17638_sequencing_run_161103_Human5_LSK108R9_4_13493_ch102_read583_strand.signal ^
		train/FMH_15Le080325s_20161103_FNFAB42798_MN17638_sequencing_run_161103_Human5_LSK108R9_4_13493_ch104_read4125_strand.label ^
		train/FMH_15Le080325s_20161103_FNFAB42798_MN17638_sequencing_run_161103_Human5_LSK108R9_4_13493_ch104_read4125_strand.signal ^
		train/FMH_15Le080325s_20161103_FNFAB42798_MN17638_sequencing_run_161103_Human5_LSK108R9_4_13493_ch106_read5593_strand.label ^
		train/FMH_15Le080325s_20161103_FNFAB42798_MN17638_sequencing_run_161103_Human5_LSK108R9_4_13493_ch106_read5593_strand.signal ^
	--validate-files ^
		train/IMB14_011406_LT_20170118_FNFAF03806_MN17279_sequencing_run_LambdaDNA_control_18012017_99768_ch102_read683_strand.label ^
		train/IMB14_011406_LT_20170118_FNFAF03806_MN17279_sequencing_run_LambdaDNA_control_18012017_99768_ch102_read683_strand.signal ^
	--eval-files ^
		val/FMH_15Le080325s_20161103_FNFAB42798_MN17638_sequencing_run_161103_Human5_LSK108R9_4_13493_ch103_read1667_strand.label ^
		val/FMH_15Le080325s_20161103_FNFAB42798_MN17638_sequencing_run_161103_Human5_LSK108R9_4_13493_ch103_read1667_strand.signal ^
	--train-steps 2000 ^
	--num-epochs 1000 ^
	--checkpoint-epochs 20 ^
	--eval-frequency 20 ^
	--eval-batch-size 500 ^
	--eval-num-epochs 20 ^
	--verbose 2
