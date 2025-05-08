# Databricks notebook source
# MAGIC %pip install -U -q -r ../requirements.txt 
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../sql_query

# COMMAND ----------

# MAGIC %run ../01-Sharepoint/02-SourceDataFromSharepoint_final

# COMMAND ----------

# MAGIC %run ../00-init $reset_all_data=false

# COMMAND ----------

# MAGIC %run ../Main_Config

# COMMAND ----------

from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import OpenAIGPTTokenizer
from pyspark.sql.types import ArrayType, StringType


# Initialize tokenizer
tokenizer = OpenAIGPTTokenizer.from_pretrained("openai-gpt")

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer, chunk_size=350, chunk_overlap=50, separators=["\n\n","\n", " ", ""]
)

def split_plain_text(text: str, min_chunk_size: int = 20) -> list:
    if not text:
        return []
    
    chunks = text_splitter.split_text(text)

    return [chunk for chunk in chunks if len(tokenizer.encode(chunk)) > min_chunk_size]

# Register the text-splitting function as a Spark UDF
split_text_udf = udf(split_plain_text, ArrayType(StringType()))

# COMMAND ----------

from pyspark.sql.functions import col, explode, udf, coalesce, lit
import pandas as pd

def StoreDataOnChunkedTable(table_name:str, schema:str, catalog:str) -> None:
    # Fully qualified table names
    source_table_fullname = f"{catalog}.{schema}.{table_name}"
    chunked_table_fullname = f"{catalog}.{schema}.{table_name}_chunked"

    create_chunked_table(chunked_table_fullname)

    # Get the last update time from the chunked table
    last_update_time = get_last_modified_datetime(chunked_table_fullname)

    # Read data from the source table while filtering rows with non-null content
    # and newer than the last update time (using coalesce to default to a very old date if needed)
    source_df = (
        spark.table(source_table_fullname)
             .filter(
                 col(col_content).isNotNull() & 
                 (col(col_lastModifiedDate) > coalesce(lit(last_update_time), lit("1900-01-01")))
             )
             .withColumn(col_content, explode(split_text_udf(col(col_content))))
    )

    # Prepare deletion of existing rows from chunked table for files that will be updated
    chunked_to_delete = source_df.select(col_filedID).distinct()
    file_ids = [row.file_id for row in chunked_to_delete.collect()]
    
    if file_ids:
        # Build a comma-separated list of file IDs (wrapped in quotes)
        file_ids_str = ", ".join([f"'{fid}'" for fid in file_ids])
        # spark.sql(f"DELETE FROM {chunked_table_fullname} WHERE file_id IN ({file_ids_str})")
        delete_file_id(chunked_table_fullname, file_ids_str)
        
    # Append the new rows to the chunked table
    source_df.write \
            .mode("append") \
            .option("mergeSchema", "true") \
            .saveAsTable(chunked_table_fullname)

    logger.info(f"Data successfully processed from {source_table_fullname} and stored in {chunked_table_fullname}")

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
vsc = VectorSearchClient(disable_notice=True)

if not endpoint_exists(vsc, VECTOR_SEARCH_ENDPOINT_NAME):
    vsc.create_endpoint(name=VECTOR_SEARCH_ENDPOINT_NAME, endpoint_type="STANDARD")

wait_for_vs_endpoint_to_be_ready(vsc, VECTOR_SEARCH_ENDPOINT_NAME)  
logger.info(f"Endpoint named {VECTOR_SEARCH_ENDPOINT_NAME} is ready.")

# COMMAND ----------

from databricks.sdk import WorkspaceClient
import databricks.sdk.service.catalog as c


from databricks.sdk import WorkspaceClient
import databricks.sdk.service.catalog as c


def CreateIndex(tablename:str, catalog:str, db:str):
    
    source_table_fullname = f"{catalog}.{db}.{tablename}_chunked"

    vs_index_fullname = f"{catalog}.{db}.{tablename}_index"

    if not index_exists(vsc, VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname):
        print(f"Creating index {vs_index_fullname} on endpoint {VECTOR_SEARCH_ENDPOINT_NAME}...")
        vsc.create_delta_sync_index(
            endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME,
            index_name=vs_index_fullname,
            source_table_name=source_table_fullname,
            pipeline_type="TRIGGERED",
            primary_key="id",
            embedding_source_column='content',
            embedding_model_endpoint_name='databricks-gte-large-en',
            columns_to_sync = sink_column
        )
        wait_for_index_to_be_ready(vsc, VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname)
    else:
        logger.info(f"Syncing index {vs_index_fullname} on endpoint {VECTOR_SEARCH_ENDPOINT_NAME}...")
        status = ""
        wait_for_index_to_be_ready(vsc, VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname)
        vsc.get_index(VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname).sync()
        while "ONLINE_NO_PENDING_UPDATE"!=status:
            idx = vsc.get_index(VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname).describe()
            index_status = idx.get('status', idx.get('index_status', {}))
            status = index_status.get('detailed_state', index_status.get('status', 'UNKNOWN')).upper()
            print(f"Status.......: {status}")

    logger.info(f"index {vs_index_fullname} on table {source_table_fullname} is ready")

# COMMAND ----------

tables = [source_table]
for table in tables:
    #Store data on the that table
    StoreDataOnChunkedTable(table, default_schema, default_catalog)

    #Creating index on that table
    CreateIndex(table, default_catalog, default_schema)
    
