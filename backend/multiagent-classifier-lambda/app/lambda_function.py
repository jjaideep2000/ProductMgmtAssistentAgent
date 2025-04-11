import json
import boto3
import time

# Constants
BEDROCK_REGION = "us-east-1"
MODEL_ID = "amazon.nova-micro-v1:0"
CLASS_OPTIONS = ["Feature", "Insight", "Competitive"]
LAMBDA_MAP = {
    "Feature": "FeatureInferenceV1",
    "Insight": "InsightInferenceV1",
    "Competitive": "CompetitiveInferenceV1",
    "Unknown": "CompetitiveInferenceV1"
}

# Initialize clients
print("🔧 Initializing Bedrock and Lambda clients...")
bedrock_runtime = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)
lambda_client = boto3.client("lambda", region_name=BEDROCK_REGION)

def build_prompt(user_input):
    return (
        "You are a smart AI classifier for product manager queries. "
        "Classify the following query into one of the following categories: Feature, Insight, or Competitive. "
        "Respond with only the category name.\n"
        f'Input: "{user_input}"\nCategory:'
    )

def lambda_handler(event, context):
    print("🟢 Lambda handler invoked.")
    print(f"📦 Incoming event: {event}")

    try:
        body = {}
        if isinstance(event.get("body"), str):
            body = json.loads(event["body"])
        elif isinstance(event.get("body"), dict):
            body = event["body"]
        elif "input" in event:
            body = event

        user_input = body.get("input", "").strip()
        if not user_input:
            return {
                "statusCode": 400,
                "headers": {
                    "Access-Control-Allow-Origin": "http://multiagent-ui.s3-website-us-east-1.amazonaws.com",
                    "Access-Control-Allow-Headers": "Content-Type",
                    "Access-Control-Allow-Methods": "OPTIONS,POST"
                },
                "body": json.dumps({"error": "Missing 'input' in request body"})
            }

        prompt = build_prompt(user_input)
        print(f"🧠 Prompt sent to Nova Micro:\n{prompt}")

        response = bedrock_runtime.invoke_model(
            modelId=MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=json.dumps({
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"text": prompt}
                        ]
                    }
                ]
            })
        )

        response_body = json.loads(response["body"].read())
        print("📦 Full response from Bedrock:")
        print(json.dumps(response_body, indent=2))

        raw_output = response_body["output"]["message"]["content"][0]["text"].strip()
        print(f"📬 Model raw output: {raw_output}")

        predicted_category = raw_output.split()[0].capitalize()
        if predicted_category not in CLASS_OPTIONS:
            predicted_category = "Unknown"

        print(f"Final Classification: {predicted_category}")

       # if predicted_category == "Unknown":
        #    return {
         #       "statusCode": 200,
          #      "headers": {
           #         "Access-Control-Allow-Origin": "http://multiagent-ui.s3-website-us-east-1.amazonaws.com",
            #        "Access-Control-Allow-Headers": "Content-Type",
             #       "Access-Control-Allow-Methods": "OPTIONS,POST"
              #  },
               # "body": json.dumps({
                #    "category": predicted_category,
                 #   "answer": "Sorry, we couldn't classify your query."
                #})
            #}

        # 🔁 Invoke the corresponding inference Lambda
        target_lambda = LAMBDA_MAP[predicted_category]
        print(f"📡 Invoking inference Lambda: {target_lambda}")
        start = time.time()
        inference_response = lambda_client.invoke(
            FunctionName=target_lambda,
            InvocationType="RequestResponse",
            Payload=json.dumps({"input": user_input})
        )
        print(f"⏱ Inference Lambda took {time.time() - start:.2f} seconds")

        payload = json.loads(inference_response["Payload"].read())
        print(f"📥 Response from {predicted_category} Lambda: {payload}")

        return {
            "statusCode": 200,
            "headers": {
                "Access-Control-Allow-Origin": "http://multiagent-ui.s3-website-us-east-1.amazonaws.com",
                "Access-Control-Allow-Headers": "Content-Type",
                "Access-Control-Allow-Methods": "OPTIONS,POST"
            },
            "body": json.dumps({
                "category": predicted_category,
                "answer": json.loads(payload.get("body", "{}"))["answer"]
            })
        }

    except Exception as e:
        print(f"❌ Error during classification or inference routing: {e}")
        return {
            "statusCode": 500,
            "headers": {
                "Access-Control-Allow-Origin": "http://multiagent-ui.s3-website-us-east-1.amazonaws.com",
                "Access-Control-Allow-Headers": "Content-Type",
                "Access-Control-Allow-Methods": "OPTIONS,POST"
            },
            "body": json.dumps({"error": str(e)})
        }