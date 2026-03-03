
import transformers as hf
from language_model.llama import Llama
from dextrous.utils import download
import textwrap as tw


model_path = 'ex/LlamaTracker/GalacticCrait/21'

format = tw.dedent('''
<|start_header_id|>system<|end_header_id|>

You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id>

{input}<|eot_id|><|start_header_id|>assistant<|end_header_id>

''').lstrip()


def download_model():
    download('h100', model_path)


def main():

    model = hf.AutoModelForCausalLM.from_pretrained(
        model_path, load_in_8bit=True
    )
    tokenizer = hf.AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3.1-8B-Instruct')

    example = tw.dedent('''
        A: i am looking for a good attraction in the centre .
        B: ok , the broughton house gallery is in the centre and admission is free .
        A: what type of attraction is the broughton house gallery and may i have the address ?
        
        Identify the information from the above dialogue:
        attraction name: The name or title of the attraction (e.g. clare hall, cambridge arts theater, scott polar museum, any)?
    ''').strip() # ONLY up to 3 dialogue turns, ALWAYS end with speaker A last, and may need to lowercase the dialogue text

    prompt = tokenizer(format.replace('{input}', example), return_tensors='pt')
    prompt_len = len(prompt['input_ids'][0])
    generated_tokens, = model.generate(**prompt, max_new_tokens=10)
    value = tokenizer.decode(generated_tokens[prompt_len:], skip_special_tokens=True)

    print('Example:', example, '\n', f"Value: {value}", sep='\n')


if __name__ == '__main__':
    main()


