import time
from xinference.client import Client

# Function to generate data with multiple "Hello" tokens
def gen_hello_token_data(tokens: int):
    return " ".join(["Hello" for _ in range(tokens)])

# Function to calculate latency and token throughput
def benchmark(client, model_uid, prompt, q1, num_requests=1):
    model = client.get_model(model_uid)

    latencies = []
    token_throughputs = []

    for _ in range(num_requests):
        start_time = time.time()
        res = model.generate(prompt=prompt + q1)
        end_time = time.time()

        latency = end_time - start_time
        tokens = len(res['choices'][0]['text'].split())

        latencies.append(latency)
        token_throughputs.append(tokens / latency)  # Tokens per second throughput

    avg_latency = sum(latencies) / len(latencies)
    avg_token_throughput = sum(token_throughputs) / len(token_throughputs)

    return avg_latency, avg_token_throughput

# Setting up the client and model
client = Client("http://192.168.192.55:9997")
model_uid = "qwen2.5"
print("model_uid", model_uid)
model = client.get_model(model_uid)

# Gradually increasing the number of tokens from 1000 to 10000
q1 = "Question: What is the name and country of ID 23? Your answer: The name and country of ID 23 are "
q2 = "Question: What is the name and country of ID 96? Your answer: The name and country of ID 96 are "

for tokens in range(1000, 5000, 1000):  # Testing from 1000, 2000, ..., 10000 tokens
    print(f"Testing with {tokens} tokens of 'Hello'...")

    # Generate prompt with multiple 'Hello' tokens
    LONG_PROMPT = f"{tokens}. " + gen_hello_token_data(tokens)

    # Run benchmark for each data size
    avg_latency, avg_token_throughput = benchmark(client, model_uid, LONG_PROMPT, q1, num_requests=1)  # Run 5 requests for each size

    # Print out the results for each token size
    print(f"Tokens: {tokens} | Average Latency: {avg_latency:.4f} seconds | Average Token Throughput: {avg_token_throughput:.2f} tokens per second")
