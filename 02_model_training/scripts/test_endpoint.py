from google.cloud import aiplatform

def test_prediction():
    import os
    print("Initializing Gemini Enterprise Agent Platform SDK...")
    project = os.environ.get("GOOGLE_CLOUD_PROJECT", "<YOUR_GCP_PROJECT_ID>")
    aiplatform.init(project=project, location="us-central1")

    print("Searching for endpoint 'gpt2-serving-endpoint-ddp-8xa100'...")
    endpoints = aiplatform.Endpoint.list(filter='display_name="gpt2-serving-endpoint-ddp-8xa100"')
    if not endpoints:
        print("Error: Endpoint 'gpt2-serving-endpoint-ddp-8xa100' not found.")
        return

    endpoint = endpoints[0]
    print(f"Found endpoint: {endpoint.resource_name}")

    prompt = "Tell me who won the world cup in 2014?"
    print(f"Sending prediction request with prompt: '{prompt}'...")
    response = endpoint.predict(
        instances=[
            {
                "prompt": prompt,
                "max_new_tokens": 64,
                "temperature": 0.85,
                "repetition_penalty": 2
            }
        ]
    )


    print("\nPrediction Response:")
    for pred in response.predictions:
        print(f"Generated text: {pred['generated_text']}")

if __name__ == "__main__":
    test_prediction()
