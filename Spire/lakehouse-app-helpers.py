# Databricks notebook source
import requests
import json
import time
import numpy as np  
import pandas as pd  
import re  

class LakehouseAppHelper:
    def __init__(self):
        from databricks import sdk
        self.host = sdk.config.Config().host
        

    def get_headers(self):
        from databricks import sdk

        return sdk.config.Config().authenticate()

    def create(self, app_name,endpoint, app_description="This app does something"):
        result = requests.post(
            f"{self.host}/api/2.0/apps",
            headers=self.get_headers(),
            json={"name": app_name, 
                  "description": app_description,
                  "resources" : [{ "name" : "rag-endpoint",
                                  "serving_endpoint": {
                                        "name":endpoint,
                                        "permission": "CAN_QUERY"
      }}]},
        ).json()
        if "error_code" in result:
            if result["error_code"] == "ALREADY_EXISTS":
                print("Application already exists")
            else:
                raise Exception(result)

        # The create API is async, so we need to poll until it's ready.
        for _ in range(20):
            time.sleep(15)
            response = requests.get(
                f"{self.host}/api/2.0/apps/{app_name}",
                headers=self.get_headers(),
            ).json()
            
            if response["compute_status"]["state"] != "STARTING":
                break
        print(response)
        return response

    

    def deploy(self, app_name, source_code_path):
        # Deploy starts the pod, downloads the source code, install necessary dependencies, and starts the app.
        response = requests.post(
            f"{self.host}/api/2.0/apps/{app_name}/deployments",
            headers=self.get_headers(),
            json={"source_code_path": source_code_path},
        ).json()
        deployment_id = response["deployment_id"]

        # wait until app is deployed. We still do not get the real app state from the pod, so even though it will say it is done, it may not be.
        # Especially the first time you deploy. We're working on not restarting the pod on the second deploy.
        # Logs: if you want to see the app logs, go to {app-url}/logz.
        for _ in range(10):
            time.sleep(5)
            response = requests.get(
                f"{self.host}/api/2.0/apps/{app_name}/deployments/{deployment_id}",
                headers=self.get_headers(),
            ).json()
            if response["status"]["state"] != "IN_PROGRESS":
                break
        return response

    def get_app_details(self, app_name):
        url = self.host + f"/api/2.0/apps/{app_name}"
        return requests.get(url, headers=self.get_headers()).json()
    
    def details(self, app_name):
        json = self.get_app_details(app_name)
        # now render it nicely
        df = pd.DataFrame.from_dict(json, orient="index")
        html = df.to_html(header=False)
        html = re.sub(
            r"(<td>)(https://((\w|-|\.)+)\.databricksapps\.com)",
            r'\1<a href="\2">\2</a>',
            html,
        )
        html = (
            "<style>.dataframe tbody td { text-align: left; font-size: 14 } .dataframe th { text-align: left; font-size: 14 }</style>"
            + html
        )
        displayHTML(html)


    def delete(self, app_name):
        url = self.host + f"/api/2.0/apps/{app_name}"
        json = self.get_app_details(app_name)
        if "error_code" in json:
            print(f"App {app_name} doesn't exist {json}")
            return
        print(f"Waiting for the app {app_name} to be deleted...")
        _ = requests.delete(url, headers=self.get_headers()).json()
        while "error_code" not in self.get_app_details(app_name):
            time.sleep(2)
        print(f"App {app_name} successfully deleted")


    def set_permissions(self, app_name,group_name):
        url = f"{self.host}/api/2.0/permissions/apps/{app_name}"
        payload = {

            "access_control_list": [
                    {
                    "group_name": group_name,
                    "permission_level": "CAN_MANAGE"
                    }
                ]
            }
        response = requests.put(
            url,
            headers=self.get_headers(),
            json=payload
        )
        try:
            response.raise_for_status()
            result = response.json()
            print("Permissions set successfully:", result)
        except requests.exceptions.HTTPError as err:
            print("Failed to set permissions:", err)
            print("Response content:", response.text)
