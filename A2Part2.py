import torch
from transformers import AutoTokenizer, GPT2LMHeadModel

tokenizer = AutoTokenizer.from_pretrained("gpt2")
# add the EOS token as PAD token to avoid warnings
model = GPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)

torch.manual_seed(0)

input_ids = tokenizer.encode("I like painting pretty pictures", return_tensors='pt')

beam_output = model.generate(
    input_ids, 
    min_length=250, 
    max_length=500, 
    num_beams=5, 
    no_repeat_ngram_size=2,
    early_stopping=True
)

story = tokenizer.decode(beam_output[0], skip_special_tokens=True)

print("My GPT-2 Story:")
print("---------------")
print(story)

f = open('A2Part2.txt', 'a')
f.write("My GPT-2 Story\n")
f.write("---------------\n")
f.write(story)
f.close()
