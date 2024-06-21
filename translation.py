import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from intel_extension_for_transformers.transformers import RtnConfig, BitsAndBytesConfig
from transformers import TextStreamer

def load_dictionary(file_path):
    dictionary = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for i in range(0, len(lines), 2): 
            english = lines[i].strip()
            if i + 1 < len(lines):
                chinese = lines[i + 1].strip()
                dictionary[english] = chinese
    return dictionary

def max_match_replace(text, dictionary):
    # Create a list of dictionary keys sorted by length in descending order
    sorted_dict_keys = sorted(dictionary.keys(), key=len, reverse=True)
    
    # Traverse the text and replace using the max match principle
    result = ""
    i = 0 
    while i < len(text):
        match_found = False
        for key in sorted_dict_keys:
            if text[i:i+len(key)] == key:
                result += dictionary[key]
                i += len(key)
                match_found = True
                break
        if not match_found:
            result += text[i]
            i += 1
            
    return result

dictionary = load_dictionary("dictionary-example-en2ch.txt")


def run_model(question, history_data):
    device = "cpu"

    model_name = "qwen-7b"
    generate_kwargs = dict(do_sample=False, temperature=0.1, num_beams=1)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    question = max_match_replace(question, dictionary)

    if len(history_data) > 2:
        history_data.pop(0)
    history = '\n'.join(history_data)

    prompt1 = f'''
    身份:
    作为国际工程管理领域的翻译专家,请把给出的英文文献中的内容翻译成中文。

    [任务]
    你的任务是把英文翻译成中文。

    文本：[{question}]

    '''
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt1}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    woq_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")
    woq_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=woq_config,
    )

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    gen_ids = woq_model.generate(
        model_inputs.input_ids,
        max_new_tokens=512,
        streamer=streamer,
        **generate_kwargs,
    )
    gen_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, gen_ids)
    ]
    gen_text = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0]

    new_conversation = f'''
    user:{question}
    assistant:{gen_text}
    '''
    history_data.append(new_conversation)

    return gen_text, history_data


