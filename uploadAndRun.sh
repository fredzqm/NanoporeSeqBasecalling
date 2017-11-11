gcloud compute --project "nanoporesequence" ssh --zone "us-east1-b" "fredzqm@deep-learning-compute" --command "rm -rf keras/*"
gcloud compute --project "nanoporesequence" scp --zone "us-east1-b" keras/*  fredzqm@deep-learning-compute:keras/ --recurse
gcloud compute --project "nanoporesequence" scp --zone "us-east1-b" cloudLocalRun.sh  fredzqm@deep-learning-compute:cloudLocalRun.sh
gcloud compute --project "nanoporesequence" ssh --zone "us-east1-b" "fredzqm@deep-learning-compute" --command "chmod +x cloudLocalRun.sh; ./cloudLocalRun.sh"


gcloud compute --project "nanoporesequence" scp --zone "us-east1-b" keras/trainer/processInput.py  fredzqm@deep-learning-compute:keras/trainer/processInput.py --recurse && gcloud compute --project "nanoporesequence" ssh --zone "us-east1-b" "fredzqm@deep-learning-compute" --command "python keras/trainer/processInput.py"