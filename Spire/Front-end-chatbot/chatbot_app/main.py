from fastapi import FastAPI, Request, Depends
import gradio as gr
import os
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole
import logging

app = FastAPI()

# Initialize Databricks client
w = WorkspaceClient()
available_endpoints = [x.name for x in w.serving_endpoints.list()]

username = None

def get_request(request: Request):
    global username
    username = request.headers.get("x-forwarded-email")
    return request

def respond(message, history, dropdown):
    global username
    if username is None:
        return "Error: User email not found in headers"

    if len(message.strip()) == 0:
        return   "Message can't be empty!"

    try:
        messages = []
        if history:
            for human, assistant in history:
                messages.append(ChatMessage(content=human, role=ChatMessageRole.USER))
                messages.append(ChatMessage(content=assistant, role=ChatMessageRole.ASSISTANT))

        messages.append(ChatMessage(content=message, role=ChatMessageRole.USER))

        # Query the model endpoint
        response = w.serving_endpoints.query(
            name=os.environ["MODEL_SERVING_ENDPOINT"],
            messages=messages,
            extra_params={"user_id": username},
            temperature=1.0,
            stream=False,
        )
    except Exception as error:
        return f"ERROR requesting endpoint {dropdown}: {error}"

    logging.info(f"Response: {response}")
    return response.choices[0].message.content

theme = gr.themes.Soft(
    text_size=gr.themes.utils.sizes.text_sm,
    radius_size=gr.themes.utils.sizes.radius_sm,
    spacing_size=gr.themes.utils.sizes.spacing_sm,
)

header_html = gr.HTML(
    '''
    <div style="display: flex; justify-content: space-between; align-items: center; padding: 10px;">
        <img src="https://lirik.io/wp-content/uploads/2024/06/Meta-tag-Logo.jpg" 
             width="40" height="35" />
        <img src="https://spire.com/wp-content/themes/spire2021/img/spire-global-cubesat-satellite-logo.svg" 
             width="60" height="40" />
    </div>
    '''
)

# Gradio chat interface setup
chat_interface = gr.ChatInterface(
    respond,
    chatbot=gr.Chatbot(show_label=False, container=False, show_copy_button=True, bubble_full_width=True),
    textbox=gr.Textbox(placeholder="What is Spire?", container=False, scale=7),
    title="Databricks Apps Spire - Chat with assistant",
    cache_examples=False,
    theme=theme,
    retry_btn=None,
    undo_btn=None,
    clear_btn="Clear",
)

# Create the Gradio blocks
demo = gr.Blocks()
with demo:
    header_html.render()
    chat_interface.render()

# Queue Gradio app and mount it with FastAPI
demo.queue(default_concurrency_limit=100)

# Pass the request object as a dependency to the Gradio app
app = gr.mount_gradio_app(app, demo, path="/")

# FastAPI endpoint to handle request object
@app.middleware("http")
async def get_user_email(request: Request, call_next):
    global username
    username = request.headers.get("x-forwarded-email")
    response = await call_next(request)
    return response
