import os
import time
import logging
import requests
import pandas as pd
import mlflow
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from operator import itemgetter

from databricks.vector_search.client import VectorSearchClient
from databricks import sql, sdk
from databricks.sdk.core import Config, oauth_service_principal

from langchain_community.chat_models import ChatDatabricks
from langchain_community.vectorstores import DatabricksVectorSearch
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from mlflow.models.rag_signatures import ChatCompletionRequest

# ----------------------------
#  Logging
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

mlflow.langchain.autolog()

model_config         = mlflow.models.ModelConfig(development_config="rag_chain_config.yaml")
databricks_resources = model_config.get("databricks_resources")
retriever_config     = model_config.get("retriever_config")
llm_config           = model_config.get("llm_config")
server_host_name    = databricks_resources["server_hostname"]
http_path           = databricks_resources["http_path"]
access_token_value  = databricks_resources["access_token"]
llm_endpoint = databricks_resources["llm_endpoint_name"]

input_example     = model_config.get("input_example")

# ----------------------------
#  Custom Chat Request Schema

# ----------------------------
def connect_to_databricks(server_hostname: str, http_path: str, sp_client_id: str, sp_client_secret:str):

    def credential_provider():
        config = Config(
            host          = f"https://{server_hostname}",
            client_id     = sp_client_id,
            client_secret =  sp_client_secret)
        return oauth_service_principal(config)

    return sql.connect(
        server_hostname=server_hostname,
        http_path=http_path,
        credentials_provider=credential_provider,
    )

def extract_file_id(user_id: str) -> List[str]:
    server = databricks_resources["server_hostname"]
    path   = databricks_resources["http_path"]
    sp_client_id = databricks_resources["sp_client_id"]
    sp_client_secret = databricks_resources["sp_client_secret"]
    conn   = connect_to_databricks(server, path, sp_client_id, sp_client_secret)
    # print(permission_file_query)
    query  = (f"""SELECT file_id FROM spire_catalog.spire_schema.sharepoint_permissions WHERE user_id = '{user_id}'""")
    # query = build_permission_file_query(user_id)

    df = pd.read_sql(query, conn)
    conn.close()
    ids = df["file_id"].drop_duplicates().tolist()
    if not ids:
        logger.warning(f"No file permissions for user {user_id}")
    return ids

# ----------------------------
def combine_all_messages(chat_messages: List[Dict[str, str]]) -> str:
    return extract_user_query_string(chat_messages)

def extract_user_query_string(chat_messages: List[Dict[str, str]]) -> str:
    return chat_messages[-1]["content"] 


def extract_previous_messages(chat_messages_array, assistant_response=None):
    messages = ""
    for msg in chat_messages_array:
        messages += f"\n{msg['role']}: {msg['content']}"
    if assistant_response:
        messages += f"  assistant: {assistant_response}\n"
    return messages


def store_response_with_history(inputs_and_response: dict):
    chat_messages = inputs_and_response["inputs"]["messages"]
    model_response = inputs_and_response["output"]
    
    updated_history = extract_previous_messages(chat_messages, assistant_response=model_response)
    
    return model_response 

def format_context(docs: List[Any]) -> str:
    if not docs:
        return "No relevant documents found."
    template = retriever_config["chunk_template"]
    return "".join(template.format(chunk_text=d.page_content) for d in docs)

def most_recent_message(chat_messages_array):
    if len(chat_messages_array) > 1:
        return chat_messages_array[-2]["content"]
    return chat_messages_array[-1]["content"]

def get_vector_search_index(retries: int = 3, delay: float = 2.0):
    client = VectorSearchClient(
        workspace_url=f"https://{databricks_resources['server_hostname']}",
        service_principal_client_id=databricks_resources['sp_client_id'], 
        service_principal_client_secret=databricks_resources['sp_client_secret'],
        disable_notice=True,
    )
    for i in range(retries):
        try:
            return client.get_index(
                endpoint_name=databricks_resources["vector_search_endpoint_name"],
                index_name=retriever_config["vector_search_index"],
            )
        except Exception as e:
            logger.warning(f"Index fetch failed (attempt {i+1}): {e}")
            time.sleep(delay)
    raise RuntimeError("Could not retrieve vector search index")

vs_index = get_vector_search_index()
schema   = retriever_config["schema"]

# ----------------------------
#  Retriever with embedded user_id
# ----------------------------
def dynamic_retriever_with_user(inputs: Dict[str, Any]) -> List[Any]:
    """
    inputs is a dict with keys:
      - "messages": List[{"role":.., "content":..}]
      - "custom_inputs": {"user_id":.., "filters":{..}}
    """
    chat_messages = inputs["messages"]
    user_id       = inputs["custom_inputs"]["user_id"]

    # lookup file_ids
    file_ids = extract_file_id(user_id)

    # build search kwargs
    params = retriever_config["parameters"].copy()
    params["filters"] = (
        {"file_id": file_ids}
        if file_ids
        else {"file_id": ["__no_permission__"]}
    )

    retr = DatabricksVectorSearch(
        vs_index,
        text_column=schema["chunk_text"],
        columns=[
            schema["primary_key"],
            schema["chunk_text"],
            schema["file_id"],
        ],
    ).as_retriever(search_kwargs=params)

    query = combine_all_messages(chat_messages)
    return retr.get_relevant_documents(query)

runnable_retriever = RunnableLambda(dynamic_retriever_with_user)

prompt = PromptTemplate(
    template=llm_config["llm_prompt_template"],
    input_variables=["chat_history", "context", "question"],
)

model = ChatDatabricks(
    endpoint=databricks_resources["llm_endpoint_name"],
    extra_params=llm_config["llm_parameters"],
)

core_chain = (
    {
        "context": (
            {"messages": itemgetter("messages"), "custom_inputs": itemgetter("extra_params")}
            | runnable_retriever
            | RunnableLambda(format_context)
        ),
        # Extract the question string
        "question": itemgetter("messages") | RunnableLambda(extract_user_query_string),
        # Extract chat history (everything except last message)
        "chat_history": itemgetter("messages") | RunnableLambda(extract_previous_messages),
        "last_question": itemgetter("messages") | RunnableLambda(most_recent_message),
    }
    | prompt
    | model
    | StrOutputParser()
)




chain = RunnableLambda(lambda inputs: {"inputs": inputs, "output": core_chain.invoke(inputs)}) | RunnableLambda(store_response_with_history)


mlflow.models.set_retriever_schema(
    primary_key=schema["primary_key"],
    text_column=schema["chunk_text"],
)
mlflow.models.set_model(model=chain)


