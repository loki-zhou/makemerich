from openai import OpenAI

client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = "xx"
)

# api_keycompletion = client.chat.completions.create(
#   model="meta/llama-3.1-405b-instruct",
#   messages=[{"role":"user","content":"Write a limerick about the wonders of GPU computing."}],
#   temperature=0.2,
#   top_p=0.7,
#   max_tokens=1024,
#   stream=True
# )

completion = client.chat.completions.create(
  model="meta/llama-3.1-405b-instruct",
  messages=[{"role":"user","content":"用中文回答， 3.11和3.9 哪个大？，需要用苏格拉底的方式告诉我"}],
  temperature=0.2,
  top_p=0.7,
  max_tokens=1024,
  stream=True
)

for chunk in completion:
  if chunk.choices[0].delta.content is not None:
    print(chunk.choices[0].delta.content, end="")
