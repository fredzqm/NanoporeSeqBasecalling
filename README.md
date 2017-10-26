# NanoporeSeqBasecalling
For the deep learning course project.

We are going to build a model that will translate the electronic signals by Nanopore signals to DNA sequence.

# To Start
First install [gcloud SDK](https://cloud.google.com/sdk/)
Clone the project and attempt to run `uploadAndRun.sh`

Try to run the script or just copy past it into terminal/cmd. The script will upload the `keras` folder to my gcloud VM instance, run the execute the model locally, and then download the output locally.

Note that you may want to update the following two lines based on your operation system
```
rm -rf keras\output
mkdir keras\output
```
For windows use `\` as path sepator, for MAXOS and linux, use `/` as path separtor

# Input preprocessing
We start from the sample from [keras example from gcloud ml-samples](https://github.com/fredzqm/cloudml-samples/tree/master/census/keras)

Extract and update the [input preprocessing module] (https://github.com/fredzqm/NanoporeSeqBasecalling/blob/master/keras/trainer/processInput.py)

The sample input is placed at [data folder](https://github.com/fredzqm/NanoporeSeqBasecalling/tree/master/keras/data)

As of 10/26/2017,
We are able to extract information from both `.signal` and `.label` file and run the simple feedforward network on them.

