gcloud auth activate-service-account --key-file NanoporeSequence-e62a27da5425.json
gcloud compute --project "nanoporesequence" ssh --zone "us-east1-b" "fredzqm@deep-learning-compute" --command "rm -rf keras/*"
gcloud compute --project "nanoporesequence" scp --zone "us-east1-b" keras/*  fredzqm@deep-learning-compute:keras/ --recurse
gcloud compute --project "nanoporesequence" ssh --zone "us-east1-b" "fredzqm@deep-learning-compute" --command "cd keras; gcloud ml-engine local train --module-name trainer.task --package-path trainer -- --train-files data/adult.data.csv --eval-files data/adult.test.csv --train-steps 1000 --job-dir output --eval-steps 100"
gcloud compute --project "nanoporesequence" scp --zone "us-east1-b" fredzqm@deep-learning-compute:keras/* keras/ --recurse
