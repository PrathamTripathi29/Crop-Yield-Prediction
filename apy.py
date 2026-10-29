import requests
import pandas as pd

url = "https://ckandev.indiadataportal.com/api/3/action/datastore_search"
params = {
    "resource_id": "f980409d-49a2-42ae-9eb0-182365005c04",  # APY dataset
    "q": "Madhya Pradesh",
    "limit": 10000,
}
response = requests.get(url, params=params)
records = response.json()["result"]["records"]
df = pd.DataFrame(records)