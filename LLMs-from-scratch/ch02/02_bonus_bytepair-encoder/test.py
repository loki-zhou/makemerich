from bpe_openai_gpt2 import get_encoder, download_vocab
orig_tokenizer = get_encoder(model_name="gpt2_model", models_dir=".")
integers = orig_tokenizer.encode("你好")
strings = orig_tokenizer.decode(integers)
print(strings)
