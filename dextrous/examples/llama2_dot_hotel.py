import transformers as hf
import textwrap as tw
import itertools as it
import torch as pt
import peft


# This is the base llama3 prompt format (note: this format expects the tokenizer to auto-add <|begin_of_text|>)
format = tw.dedent('''
[INST] <<SYS>> You are a helpful, respectful, and honest assistant. <</SYS>> {input} [/INST]
''').lstrip()

dialogue = [
    "I am looking for an attraction in the city center.",
    "Ok, the broughton house gallery is in the centre and admission is free.",
    "Ok can you book me a ticket to the gallery?",
    "Sure, I can help you with that. What day would you like to visit the gallery?",
    "Saturday. I need two tickets for me and my friend."
]

slots = {
    "attraction area": "Where the attraction is located [north, south, center, east, west]",
    "attraction name": "The name or title of the attraction (e.g. clare hall, cambridge arts theater, scott polar museum, any)",
    "attraction type": "The type of attraction (e.g. museum, gallery, park, any)",
    "booking date": "The day you would like to visit the attraction",
    "attraction people": "The number of tickets you need for the attraction",
    "attraction price": "The cost of admission to the attraction",
    "attraction address": "The location or address of the attraction",
}


def merge(lora_adapter_path, base_model='meta-llama/Llama-2-13b-chat-hf'):
    base_model = hf.AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-13b-chat-hf',
        torch_dtype=pt.bfloat16, device_map='auto')
    dot_model = peft.PeftModel.from_pretrained(base_model, lora_adapter_path, device_map='auto')
    merged_model = dot_model.merge_and_unload()
    merged_model.save_pretrained('ex/LlamaTracker/Merged',
        safe_serializateion=False, save_peft_format=False)


def main():
    quantization = hf.BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=pt.bfloat16,
        bnb_4bit_use_double_quant=False, bnb_4bit_quant_type='nf4')
    model = hf.AutoModelForCausalLM.from_pretrained('ex/LlamaTracker/Merged',
        quantization_config=quantization, torch_dtype=pt.bfloat16)

    lora = 'ex/LlamaTracker/UntamedCatoNeimoidia/11'
    tokenizer = hf.AutoTokenizer.from_pretrained('meta-llama/Llama-2-13b-chat-hf')
    for next_system_turn_index in range(1, len(dialogue)+1, 2):
        dialogue_context = dialogue[:next_system_turn_index]
        formatted_context = '\n'.join(
            f"{speaker}: {text}" for speaker, text in reversed(list(zip(
                it.cycle('AB'), reversed(dialogue_context[-3:])))))
        print(formatted_context)
        for slot_name, slot_description in slots.items():
            prompt = f'''
                {formatted_context}
                
                Identify the information from the above dialogue:
                {slot_name}: {slot_description}
                '''
            prompt = tokenizer(format.replace('{input}', prompt), return_tensors='pt')
            prompt = {k: v.to('cuda') for k, v in prompt.items()}
            prompt_len = len(prompt['input_ids'][0])
            generated_tokens, = model.generate(**prompt, max_new_tokens=10)
            value = tokenizer.decode(generated_tokens[prompt_len:], skip_special_tokens=True)
            print(f"    {slot_name}: {value}")
        print('\n')


if __name__ == '__main__':
    from dextrous.utils import download
    # main()
    merge('ex/LlamaTracker/DazzlingDengar/21')


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
