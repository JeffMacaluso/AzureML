"""
This script generates the scoring and schema files necessary to operationalize your model
"""

from azureml.api.schema.dataTypes import DataTypes
from azureml.api.schema.sampleDefinition import SampleDefinition
from azureml.api.realtime.services import generate_schema
import numpy as np
# import keras

# Prepare the web service definition by authoring
# init() and run() functions. Test the functions
# before deploying the web service.

model = None

def init():
    from keras.models import load_model
    import h5py
    # Get the path to the model asset
    # local_path = get_local_path('mymodel.model.link')

    # Load model using appropriate library and function
    global model
    # model = model_load_function(local_path)
    model = load_model('model.h5')


# Takes an input and geneartes predictions
def run(input_array):
    import json

    # Predict using appropriate functions
    scores = model.predict(input_array)
    predicted_score = np.max(scores)
    prediction = np.argmax(predicted_score)
    print("Image predicted to be '{}' with score {}.".format(prediction, predicted_score))

    # Creating the JSON-encoded string of the model output
    outDict = {"label": str(prediction), "score": str(predicted_score)}
    outJsonString = json.dumps(outDict)
    print("JSON-encoded detections: " + outJsonString[:120] + "...")
    print("DONE.")

    return str(outJsonString)


def main():
    # Generating random 28x28 pixels to use as sample input
    sample_input = (np.random.rand(28, 28, 1) * 255)#.astype('uint8')
    sample_input = sample_input.reshape(1, 28, 28, 1)  # Reshaping to match training data

    # Calling init() and run()
    init()
    inputs = {"input_array": SampleDefinition(DataTypes.NUMPY, sample_input)}
    result_string = run(sample_input)
    print("resultString = " + str(result_string))

    # Generating the schema
    generate_schema(run_func=run, inputs=inputs, filepath='outputs/schema.json')
    print('Schema generated')


# Implement test code to run in IDE or Azure ML Workbench
if __name__ == '__main__':
    main()
