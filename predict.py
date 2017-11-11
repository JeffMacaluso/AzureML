"""
Submits a request to the web service and returns a prediction
"""

import json
import requests
import numpy as np
from azureml.api.schema.sampleDefinition import SampleDefinition
from azureml.api.schema.dataTypes import DataTypes

# Generating sample data
sample_input = (np.random.rand(28, 28, 1) * 255)
sample_input = sample_input.reshape(1, 28, 28, 1)  # Reshaping to match training data

# Shaping into the format: "{\"input_df\": [{\"feature1\": value1, \"feature2\": value2}]}"
datalist = sample_input.tolist()  # Converting to a list before converting to JSON
data = json.dumps(datalist, separators=(',', ':'), sort_keys=True, indent=4)  # JSONifying the multi-dimensional array
body = str.encode(data)  # Encoding as a string

# Setting up the request parameters
url = 'http://40.84.6.42:80/api/v1/service/mnistcluster/score'
api_key = 'b91e271f06794fdea8085ab2dc032adb'  # Warning: It's bad practice to share API keys to the public
headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key)}

# Sending the request and printing the response
resp = requests.post(url, data, headers=headers)
print(resp.text)
