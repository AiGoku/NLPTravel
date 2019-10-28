import requests
import jsonpath
import json
import requests

api = 'http://map.amap.com/service/subway?_1469083453978&srhdata=1100_drw_beijing.json'
response =requests.get(api)
data = response.text

json_obj = json.loads(data)
print(type(json_obj))

#print(json_obj['l']['ln'])
print(json_obj['l'])
for i in json_obj['l']:

    pass


