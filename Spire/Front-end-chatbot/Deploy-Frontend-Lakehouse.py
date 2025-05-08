# Databricks notebook source
# MAGIC %pip install --quiet -U mlflow databricks-sdk==0.23.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../00-init

# COMMAND ----------

# DBTITLE 1,load lakehouse helpers
# MAGIC %run ../lakehouse-app-helpers

# COMMAND ----------

# MAGIC %run ../Main_Config

# COMMAND ----------

MODEL_NAME = model_name
endpoint_name = endpoint

yaml_app_config = { "command": ["uvicorn", "main:app", "--workers", "1"],
                    "env": [{"name": "MODEL_SERVING_ENDPOINT", "value": endpoint_name}]
                    }
try:
    with open('chatbot_app/app.yaml', 'w') as f:
        yaml.dump(yaml_app_config, f)
except:
    print('pass to work on build job')

# COMMAND ----------

#dbutils.fs.mkdirs('chatbot_app')

# COMMAND ----------

# MAGIC %%writefile chatbot_app/main.py
# MAGIC from fastapi import FastAPI, Request, Depends
# MAGIC import gradio as gr
# MAGIC import os
# MAGIC from databricks.sdk import WorkspaceClient
# MAGIC from databricks.sdk.service.serving import ChatMessage, ChatMessageRole
# MAGIC import logging
# MAGIC
# MAGIC app = FastAPI()
# MAGIC
# MAGIC # Initialize Databricks client
# MAGIC w = WorkspaceClient()
# MAGIC available_endpoints = [x.name for x in w.serving_endpoints.list()]
# MAGIC
# MAGIC username = None
# MAGIC
# MAGIC def get_request(request: Request):
# MAGIC     global username
# MAGIC     username = request.headers.get("x-forwarded-email")
# MAGIC     return request
# MAGIC
# MAGIC def respond(message, history, dropdown):
# MAGIC     global username
# MAGIC     if username is None:
# MAGIC         return "Error: User email not found in headers"
# MAGIC
# MAGIC     if len(message.strip()) == 0:
# MAGIC         return   "Message can't be empty!"
# MAGIC
# MAGIC     try:
# MAGIC         messages = []
# MAGIC         if history:
# MAGIC             for human, assistant in history:
# MAGIC                 messages.append(ChatMessage(content=human, role=ChatMessageRole.USER))
# MAGIC                 messages.append(ChatMessage(content=assistant, role=ChatMessageRole.ASSISTANT))
# MAGIC
# MAGIC         messages.append(ChatMessage(content=message, role=ChatMessageRole.USER))
# MAGIC
# MAGIC         # Query the model endpoint
# MAGIC         response = w.serving_endpoints.query(
# MAGIC             name=os.environ["MODEL_SERVING_ENDPOINT"],
# MAGIC             messages=messages,
# MAGIC             extra_params={"user_id": username},
# MAGIC             temperature=1.0,
# MAGIC             stream=False,
# MAGIC         )
# MAGIC     except Exception as error:
# MAGIC         return f"ERROR requesting endpoint {dropdown}: {error}"
# MAGIC
# MAGIC     logging.info(f"Response: {response}")
# MAGIC     return response.choices[0].message.content
# MAGIC
# MAGIC theme = gr.themes.Soft(
# MAGIC     text_size=gr.themes.utils.sizes.text_sm,
# MAGIC     radius_size=gr.themes.utils.sizes.radius_sm,
# MAGIC     spacing_size=gr.themes.utils.sizes.spacing_sm,
# MAGIC )
# MAGIC
# MAGIC header_html = gr.HTML(
# MAGIC     '''
# MAGIC     <div style="display: flex; justify-content: space-between; align-items: center; padding: 10px;">
# MAGIC         <img src="https://lirik.io/wp-content/uploads/2024/06/Meta-tag-Logo.jpg" 
# MAGIC              width="40" height="35" />
# MAGIC         <img src="https://spire.com/wp-content/themes/spire2021/img/spire-global-cubesat-satellite-logo.svg" 
# MAGIC              width="60" height="40" />
# MAGIC     </div>
# MAGIC     '''
# MAGIC )
# MAGIC
# MAGIC # Gradio chat interface setup
# MAGIC chat_interface = gr.ChatInterface(
# MAGIC     respond,
# MAGIC     chatbot=gr.Chatbot(show_label=False, container=False, show_copy_button=True, bubble_full_width=True),
# MAGIC     textbox=gr.Textbox(placeholder="What is Spire?", container=False, scale=7),
# MAGIC     title="Databricks Apps Spire - Chat with assistant",
# MAGIC     cache_examples=False,
# MAGIC     theme=theme,
# MAGIC     retry_btn=None,
# MAGIC     undo_btn=None,
# MAGIC     clear_btn="Clear",
# MAGIC )
# MAGIC
# MAGIC # Create the Gradio blocks
# MAGIC demo = gr.Blocks()
# MAGIC with demo:
# MAGIC     header_html.render()
# MAGIC     chat_interface.render()
# MAGIC
# MAGIC # Queue Gradio app and mount it with FastAPI
# MAGIC demo.queue(default_concurrency_limit=100)
# MAGIC
# MAGIC # Pass the request object as a dependency to the Gradio app
# MAGIC app = gr.mount_gradio_app(app, demo, path="/")
# MAGIC
# MAGIC # FastAPI endpoint to handle request object
# MAGIC @app.middleware("http")
# MAGIC async def get_user_email(request: Request, call_next):
# MAGIC     global username
# MAGIC     username = request.headers.get("x-forwarded-email")
# MAGIC     response = await call_next(request)
# MAGIC     return response
# MAGIC

# COMMAND ----------

app_name = application_name

helper = LakehouseAppHelper()

app_details = helper.create(app_name,endpoint_name, app_description="Your Databricks assistant")

# COMMAND ----------

group_name = permission_group_name
helper.set_permissions(app_name ,group_name)

# COMMAND ----------

helper.deploy(app_name, os.path.join(os.getcwd(), 'chatbot_app'))
helper.details(app_name)
