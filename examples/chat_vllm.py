from xinference.client import Client

client = Client("http://10.16.1.23:9997")
# The chatglm2 model has the capabilities of "chat" and "embed".
# model_uid = client.launch_model(model_name="qwen2.5",
#                                 model_engine="vllm")
model_uid = "qwen2.5-instruct"
print("model_uid", model_uid)
# model_uid qwen2.5
model = client.get_model(model_uid)

print(model)
messages = [{"role": "user", "content": "What is the largest animal?"}]
# If the model has "generate" capability, then you can call the
# model.generate API.
for i in range(1000):
    res = model.generate(
        prompt="What is the largest animal?",
        generate_config={"max_tokens": 10}
    )
    print(res)