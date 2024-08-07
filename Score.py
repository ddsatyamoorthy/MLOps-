import joblib
import json
import numpy as np

from azureml.core.model import Model

def init():
    global model_3
    model_3_path = Model.get_model_path(model_name='rf_model_500')
    model_3 = joblib.load(model_3_path)

def run(raw_data):
    try:
        data = json.loads(raw_data)['data']
        data = np.array(data)
        result_1 = model_3.predict(data)
        
        return {"prediction1": result_1.tolist()}
    except Exception as e:
        result = str(e)
        return result

from azureml.core.model import InferenceConfig

inference_config = InferenceConfig(entry_script="score3.py", environment=env)

from azureml.core.webservice import AciWebservice

aci_service_name = "aciservice-modelrfdiabetes"

deployment_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1)

service = Model.deploy(ws, aci_service_name, [model], inference_config, deployment_config, overwrite=True)
service.wait_for_deployment(True)

print(service.state)

datastore= Datastore.get(ws, "workspaceblobstore")
df= Dataset.Tabular.from_delimited_files(path=[(datastore,"preproessed.csv")]).to_pandas_dataframe()
df.head()


5) score.py 
import joblib
import json
import numpy as np

from azureml.core.model import Model

def init():
    global model_3
    model_3_path = Model.get_model_path(model_name='model_estimator_500')
    model_3 = joblib.load(model_3_path)

def run(raw_data):
    try:
        data = json.loads(raw_data)['data']
        data = np.array(data)
        result_1 = model_3.predict(data)
        
        return {"prediction1": result_1.tolist()}
    except Exception as e:
        result = str(e)
        return result
