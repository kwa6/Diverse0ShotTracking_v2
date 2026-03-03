

from language_model.llama import Llama
import textwrap as tw


llama = Llama(
    base='ex/LlamaTracker/LimitlessDromundKaas/6',
    tokenizer_reponame='meta-llama/Meta-Llama-3.1-8B-Instruct',
    format=tw.dedent('''
        <|start_header_id|>system<|end_header_id|>
        
        You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id>
        
        {input}<|eot_id|><|start_header_id|>assistant<|end_header_id>
        
        ''').lstrip() # make sure there's these two  trailing newlines (and no <|begin_of_text|>, oops)
)

example = tw.dedent('''
    A: i am looking for a good attraction in the centre .
    B: ok , the broughton house gallery is in the centre and admission is free .
    A: what type of attraction is the broughton house gallery and may i have the address ?
    
    Identify the information from the above dialogue:
    attraction name: The name or title of the attraction (e.g. clare hall, cambridge arts theater, scott polar museum, any)?
''').strip() # ONLY up to 3 dialogue turns, ALWAYS end with speaker A last, and may need to lowercase the dialogue text


value = llama.generate(example)

print('Example:', example, 'Value:', value, sep='\n')


