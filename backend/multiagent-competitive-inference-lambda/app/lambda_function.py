import json
import boto3
from requests_aws4auth import AWS4Auth
from opensearchpy import OpenSearch, RequestsHttpConnection
from langchain_community.vectorstores import OpenSearchVectorSearch
#from langchain_community.embeddings import BedrockEmbeddings
from langchain_aws import BedrockEmbeddings
from langchain_core.runnables import RunnableSequence

# Constants
REGION = "us-east-1"
MODEL_ID = "meta.llama3-8b-instruct-v1:0"
OPEN_SEARCH_HOST = "vpc-multiagentragstorage-rkr5pl76c6hyqf6hdnhk26fu5e.us-east-1.es.amazonaws.com"
INDEX_NAME = "competitive_index"

# 🔧 Initialize clients
print("🔧 Initializing Llama 3 Bedrock client...")
bedrock = boto3.client("bedrock-runtime", region_name=REGION)

print("🔐 Fetching AWS credentials...")
credentials = boto3.Session().get_credentials()
awsauth = AWS4Auth(
    credentials.access_key,
    credentials.secret_key,
    REGION,
    "es",
    session_token=credentials.token
)

print("🔗 Setting up OpenSearch vector client...")
opensearch_client = OpenSearch(
    hosts=[{"host": OPEN_SEARCH_HOST, "port": 443}],
    http_auth=awsauth,
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection
)

print("🔍 Setting up embeddings for retriever...")
embedding = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0")

vector_store = OpenSearchVectorSearch(
    index_name=INDEX_NAME,
    embedding_function=embedding,
    opensearch_url=f"https://{OPEN_SEARCH_HOST}",
    http_auth=awsauth,
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection
)

retriever = vector_store.as_retriever()

def build_prompt(query, context):
    return f"""You are a helpful AI assistant for product managers.

Based on the following context, answer the user query in detail.

Context:
{context}

Query:
{query}

Answer:"""

def lambda_handler(event, context):
    print("🟢 Lambda handler invoked.")
    print(f"📥 Incoming event: {event}")

    try:
        body = event.get("body")
        if isinstance(body, str):
            body = json.loads(body)
        elif isinstance(body, dict):
            body = body
        else:
            body = event

        query = body.get("input", "").strip()
        if not query:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "Missing 'input' in request body"})
            }

        print("🔎 Querying OpenSearch for relevant context...")
        docs = retriever.get_relevant_documents(query)
        context_text = "\n".join([doc.page_content for doc in docs])
        print(f"📚 Retrieved {len(docs)} relevant docs.")

        prompt = build_prompt(query, context_text)
        print("🧠 Sending prompt to Llama 3 model...")

        response = bedrock.invoke_model(
            modelId=MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=json.dumps({
                "prompt": prompt,
                "max_gen_len": 512,
                "temperature": 0.5,
                "top_p": 0.9
            })
        )

        print("📦 Raw response from Bedrock:")
        print(response)

        if not response.get("body"):
            raise ValueError("Response body is None. Possible Bedrock error.")

        result = json.loads(response["body"].read())
        print(f"📬 Decoded Bedrock output: {result}")

        # Try getting the response content safely
        answer = result.get("generation") or result.get("output") or ""
        answer = answer.strip()

        if not answer:
            raise ValueError("Model returned empty output.")

        print(f"✅ Generated answer: {answer}")
        return {
            "statusCode": 200,
            "body": json.dumps({"answer": answer})
        }

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }