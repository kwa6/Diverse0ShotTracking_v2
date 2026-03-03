
from language_model.t5 import T5

t5 = T5(
    base='google/t5-v1_1-xxl',
    quantize='int8',
    param_magnitude='11b',
    format='',
    lora=None,
    train_batch_size=1,
    gen_batch_size=1,
    repetition_penalty=1.2,
    epochs=1,
)

toy_data = [
    ['What is the capital of France?', 'Paris'],
    ['What is the capital of Germany?', 'Berlin'],
    ['What is the capital of Italy?', 'Rome'],
    ['What is the capital of Spain?', 'Madrid'],
    ['What is the capital of Portugal?', 'Lisbon'],
    ['What is the capital of the United Kingdom?', 'London'],
    ['What is the capital of the United States?', 'Washington, D.C.'],
    ['What is the capital of Canada?', 'Ottawa'],
    ['What is the capital of Mexico?', 'Mexico City'],
    ['What is the capital of Brazil?', 'Brasília'],
]

t5.train(toy_data)

print(t5.generate('What is the capital of France?'))

t5.save('ex/Test/T511bInt8')

