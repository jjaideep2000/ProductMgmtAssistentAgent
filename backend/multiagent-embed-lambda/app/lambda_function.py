import json
import boto3
from requests_aws4auth import AWS4Auth
from opensearchpy import RequestsHttpConnection
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_core.documents import Document

# Constants
REGION = "us-east-1"
S3_BUCKET = "multiagents3bucket"
OPEN_SEARCH_HOST = "vpc-multiagentragstorage-rkr5pl76c6hyqf6hdnhk26fu5e.us-east-1.es.amazonaws.com"

print("🧠 Initializing Bedrock Embeddings...")
embedding = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0")

print("🔐 Fetching AWS credentials...")
credentials = boto3.Session().get_credentials()
awsauth = AWS4Auth(
    credentials.access_key,
    credentials.secret_key,
    REGION,
    "es",
    session_token=credentials.token
)

# Load text files from S3
def load_txt_from_s3(bucket, prefix):
    print(f"📥 Loading files from S3 bucket: {bucket}, prefix: {prefix}")
    s3 = boto3.client("s3")
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)

    docs = []
    for obj in response.get("Contents", []):
        key = obj["Key"]
        if key.endswith(".txt"):
            print(f"📄 Found .txt file: {key}")
            content = s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8")
            docs.append(Document(page_content=content, metadata={"source": key}))
    print(f"📦 Loaded {len(docs)} documents from S3.")
    return docs

# Embed and store in OpenSearch
def embed_docs(prefix, index_name):
    try:
        print(f"🚀 Embedding and pushing docs for prefix '{prefix}' to index '{index_name}'...")
        docs = load_txt_from_s3(S3_BUCKET, prefix)
        if not docs:
            print(f"⚠️ No documents found in {prefix}")
            return

        OpenSearchVectorSearch.from_documents(
            documents=docs,
            embedding=embedding,
            index_name=index_name,
            opensearch_url=f"https://{OPEN_SEARCH_HOST}",  # ✅ Correct usage
            http_auth=awsauth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
            timeout=60
        )
        print(f"✅ Embedded {len(docs)} docs to index '{index_name}'")
    except Exception as e:
        print(f"❌ Failed to embed docs for '{prefix}' ➜ {e}")
        raise

# Lambda entry point
def lambda_handler(event, context):
    print("🟢 Lambda handler invoked.")
    try:
        embed_docs("feature_docs/", "feature_index")
        embed_docs("insight_docs/", "insight_index")
        embed_docs("competitive_docs/", "competitive_index")

        return {
            "statusCode": 200,
            "body": json.dumps("✅ All embeddings pushed to OpenSearch")
        }
    except Exception as e:
        print(f"❌ Lambda failed: {str(e)}")
        return {
            "statusCode": 500,
            "error": str(e)
        }