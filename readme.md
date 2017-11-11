# Classify MNIST dataset using Keras (with a TensorFlow back end)

[git repo](https://jeffmacaluso.visualstudio.com/mnistDemo?parameters=)

This sample uses the popular [TensorFlow](https://www.tensorflow.org/) machine learning library from Google to classify the ageless [MNIST dataset](http://yann.lecun.com/exdb/mnist/) of handwritten digits.

## Logging code

The sample code is primarily copied from the Keras sample code in the documentation on GitHub. The main changes we make is to add some Azure ML specific logging code into the experiment.

Here are the relevant code snippets:

```python
# Reference the Azure ML logging library
from azureml.logging import get_azureml_logger
...
# Initialize the logger
run_logger = get_azureml_logger()
...
# Declare empty lists
metrics = []
losses = []
# During the training session
while step * batch_size < training_iters:
    ...
    # Record accuracy and loss into a list
    metrics.append(float(acc))
    losses.append(float(loss))
    ...

# After the training finishes, log the list of accuracies and losses
run_logger.log("Accuracy", metrics)
run_logger.log("Loss", losses)
```

By adding the above logging code, when the run finishes, you can find the accuracy and loss graphs plotted for you in the run history detail page.

## Instructions for running scripts from CLI window

You can run the scripts from the Workbench app, but it is more interesting to run it from the command-line window so you can watch the feedback in real-time (note: the VS Code Tools for AI extension has a function to auto-generate the command in the command-line).

Open the command-line window by clicking on **File** --> **Open Command Prompt**, then run `train.py` in local Python environment installed by Azure ML Workbench by typing in the following command.

```
# first install tensorflow and keras libraries using pip, you only needed to do this once.
$ pip install tensorflow
$ pip install keras

# submit the experiment to local execution environment
$ az ml experiment submit -c local tf_mnist.py
```

If you have Docker engine running locally, you can run `train.py` in a Docker container.

>Note: this command automatically pulls down a base Docker image so it can take a few minutes before the job is started. But this only happens if you are running it for the first time. The subsequent runs will be much faster.

And you don't need to pip-install the _tensorflow_ or _keras_ libraries, since they are already specified in the `conda_depeendencies.yml` file under `aml_config` folder. The execution engine will automatically install it for as part of the Docker image building process.

```
# submit the experiment to local Docker container for execution
$ az ml experiment submit -c docker tf_mnist.py
```

You can also run `train.py` in a Docker container in a remote machine. Note you need to create/configure myvm.compute.

```
# attach a new compute context
$ az ml computetarget attach --name myvm --address <ip address or FQDN> --username <username> --password <pwd> --type remotedocker

# prepare the Docker image
$ az ml experiment prepare -c myvm

$ az ml experiment submit -c myvm tf_mnist.py
```

## Running it on a VM with GPU

With computationally expensive tasks like training a neural network, you can get a huge performance boost by running it on a GPU-equipped machine.

>Note, if your local machine already has NVidia GPU chips, and you have installed the CUDA libraries and toolkits, you can directly run the script using local compute target. Just be sure to pip-install the _tensorflow-gpu_ Python package. The below instructions are specifically for running script in a remote VM equipped with GPU.

### Step 1. Provision a GPU Linux VM 

Create an Ubuntu-based Data Science Virtual Machine(DSVM) in Azure portal using one of the NC-series VM templates. NC-series VMs are the VMs equipped with GPUs for computation.

### Step 2. Attach the compute context

Run following command to add the GPU VM as a compute target in your current project:

```
$ az ml computetarget attach --name myvm --address <ip address or FQDN> --username <username> --password <pwd> --type remotedocker
```

The above command creates a `myvm.compute` and `myvm.runconfig` file under the `aml_config` folder.

### Step 3. Modify the configuration files under _aml_config_ folder

- You need the TensorFlow library built for GPU:
    
    In `conda_dependencies.yml` file, replace `tensorflow` with `tensorflow-gpu`.

- You need a different base Docker image with CUDA libraries preinstalled:

    In `myvm.compute` file, replace the value of `baseImage` from `microsoft/mmlspark:plus-0.7.91` to  `microsoft/mmlspark:plus-gpu-0.7.91`

- You need to use _NvidiaDocker_ command to start the Docker container as opposed to the regular _docker_ command.

    In `myvm.compute` file, add a line: `nvidiaDocker: true`

- You need to specify the run time framework as _Python_ as opposed to _PySpark_:

    In `myvm.runconfig` file,  change the value of `Framework` from `PySpark` to `Python`.

### Step 4. Run the script

Now you are ready to run the script.
```
$ az ml experiment submit -c myvm tf_mnist.py
```
You should notice the script finishes significantly faster than than if you just use the CPU. And the command-line outputs should indicate that GPU is used for executing this script.

## Deploying the model as a web service

There are a few steps [listed here](https://docs.microsoft.com/en-us/azure/machine-learning/preview/deployment-setup-configuration) for preparing your account and environment, but these steps assume you have already set up the model management account and enabled container services.

### Setting up the local environment

The following command sets up the local environment

```
$ az ml env setup -l [Azure Region, e.g. eastus2] -n [environment name] -g [existing resource group]
```

The following command sets the created environment as the one to be used

```
$ az ml env set -n [environment name] -g [resource group]
```

Lastly, run this command to ensure that we are in local mode

```
$ az ml env local
```

### Setting up the kubernetes cluster

```
$ az ml env setup --cluster -n [cluster name] -l [Azure region, e.g. eastus2] -g [resource group]
```

The following command sets the created environment as the one to be used

```
$ az ml env set -n [environment name] -g [resource group]
```

Lastly, run this command to ensure that we are in cluster mode

```
$ az ml env cluster
```

### Deployment

The following command will deploy the model into either a local environment or a kubernetes cluster depending on which environment was set up prior to running the command.

```
$ az ml service create realtime -n <service name> -v -c aml_config/conda_dependencies.yml -m model.h5 -s schema.json -f score.py -r python
```

## Consuming a deployed model

### Get service information

After the web service has been successfully deployed, use the following command to get the service URL and other details for calling the service endpoint.

```
$ az ml service usage realtime -i <service name>
```

This command will print out the service URL, required request headers, swagger URL, and sample data for calling the service if the service API schema was provided at the deployment time.

You can test the service directly from the CLI without composing an HTTP requst, by entering the sample CLI command with the input data:

```
$ az ml service run realtime -i <service name> -d "Your input data"
```

### Get the service API key

_Note: this is only for models deployed in a cluster_

To get the web service key, use the following command:

```
$ az ml service keys realtime -i <web service id>
```

When creating HTTP request, use the key in the authorization header: "Authorization": "Bearer "

### Get the service Swagger description

If the service API schema was supplied the service endpoint would expose a Swagger document at http://<ip>/api/v1/service/<service name>/swagger.json. The Swagger document can be used to automatically generate the service client and explore the expected input data and other details about the service.
Get service logs

To understand the service behavior and diagnose problems, there are several ways to retrieve the service logs:

- CLI command `az ml service logs realtime -i <service id>`. This command works in both cluster and local modes.
- If the service logging was enabled at deployment, the service logs will also be sent to AppInsight. The CLI command `az ml service usage realtime -i <service id>` shows the AppInsight URL. Note that the AppInsight logs may be delayed by 2-5 mins.
- Cluster logs can be viewed through Kubernetes console that is connected when you set the current cluster environment with az ml env set
- Local docker logs are available through the docker engine logs when the service is running locally.

### Consuming via Python

See the `predict.py` file in this proejct, but it generally follows this structure:

```python
import requests
import json

data = "{\"input_df\": [{\"feature1\": value1, \"feature2\": value2}]}"
body = str.encode(json.dumps(data))

url = 'http://<service ip address>:80/api/v1/service/<service name>/score'
api_key = 'your service key' 
headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key)}

resp = requests.post(url, data, headers=headers)
resp.text
```