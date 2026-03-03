
import transformers as hf
import textwrap as tw


model_path = 'ex/LlamaTracker/GalacticCrait/21'

# This is the base llama3 prompt format (note: this format expects the tokenizer to auto-add <|begin_of_text|>)
format = tw.dedent('''
<|start_header_id|>system<|end_header_id|>

You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id>

{input}<|eot_id|><|start_header_id|>assistant<|end_header_id>

''').lstrip()

def main():
    model = hf.AutoModelForCausalLM.from_pretrained(model_path, load_in_8bit=True)
    tokenizer = hf.AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3.1-8B-Instruct')

    # format of the prompt and dialogue (ALWAYS <= 3 turns, ALWAYS ending with speaker A
    example = tw.dedent('''
        A: I am looking for an attraction in the city center.
        B: Ok, the broughton house gallery is in the centre and admission is free.
        A: Ok can you book me a ticket to the gallery?
        
        Identify the information from the above dialogue:
        request: The type of information Speaker A is asking for [time, duration, location, color]?
    ''').strip()
    # note that the slot is defined as ONE of:

    # {slot_name}: {slot_description}?                                           <- description only, poorer performance
    # {slot_name}: {slot_description} (e.g. example 1, example 2)?               <- 1-5 examples are best
    # {slot_name}: {slot_description} [category 1, category 2]?                  <- 2-10 categories should work

    prompt = tokenizer(format.replace('{input}', example), return_tensors='pt')
    prompt_len = len(prompt['input_ids'][0])
    generated_tokens, = model.generate(**prompt, max_new_tokens=10)
    value = tokenizer.decode(generated_tokens[prompt_len:], skip_special_tokens=True)

    print('Example:', example, '\n', f"Value: {value}", sep='\n')


if __name__ == '__main__':
    main()



'''
Requirments (make sure to use the right version of pytorch for the cuda version)
# torch==2.3.1+cu118
torch==2.3.1+cu121
transformers[deepspeed]==4.43.4
tokenizers==0.19.1
accelerate==0.33.0
peft==0.12.0
bitsandbytes==0.43.3
datasets==2.20.0
'''
