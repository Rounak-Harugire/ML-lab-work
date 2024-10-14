import requests
import json

API_key="1146c12e2a02174cf94857dd95e561e3"
city_name=input("Enter City Name : ")
url=f"https://api.openweathermap.org/data/2.5/weather?q={city_name}&appid={API_key}"
response=requests.get(url)
data=response.json()
print(json.dumps(data,indent=4))
with open("my_data.json","w")as file:json.dump(data , file)