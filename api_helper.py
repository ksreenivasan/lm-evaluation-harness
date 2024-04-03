import os

from litellm import completion
from tenacity import (retry, stop_after_attempt,  # for exponential backoff
                      wait_random_exponential)

def dep(name):
    return f"https://{name}.inf.hosted-on.mosaicml.hosting/v2/"



os.environ |= {
    # OpenAI
    'OPENAI_API_KEY': 'MY_API_KEY',
    # Mistral Large
    'AZURE_MISTRAL_API_KEY': 'MY_API_KEY', # my key
    'MISTRAL_API_KEY': 'MY_API_KEY', # jose's key
    # Gemini 1.0
    'GEMINI_API_KEY': 'MY_API_KEY',
    # Anthropic
    'ANTHROPIC_API_KEY': 'MY_API_KEY',
    # MCLI
    # 'MCLI_API_KEY': "MY_API_KEY",
    # DATABRICKS
    # 'DATABRICKS_API_KEY': "MY_API_KEY"
}

gemini_block_none = {
    "safety_settings": [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_NONE",
        },
    ]
}

mcli_key = {"api_key": os.environ.get("MCLI_API_KEY")}
db_key = {'api_key': os.environ.get('DATABRICKS_API_KEY')}
models = {
    # OpenAI
    "gpt-4": {"model": "openai/gpt-4-turbo-preview"},
    "gpt-3.5": {"model": "openai/gpt-3.5-turbo"},
    # Mistral
    "mixtral-instruct": {
        "model": "openai/_",
        "api_base": dep("mixtral-8x7b-instruct-at-trtllm-newimg-lorxa5"),
        **mcli_key,
    },
    # gemini
    "gemini-1.0-pro": {"model": "gemini/gemini-1.0-pro-latest"} | gemini_block_none,
    # Claude
    "claude-3-haiku": {"model": "anthropic/claude-3-haiku-20240307"},
    "claude-3-sonnet": {"model": "anthropic/claude-3-sonnet-20240229"},
    "claude-3-opus": {"model": "anthropic/claude-3-opus-20240229"},
    'llama-2-70b-chat-int8': {
        "model": "openai/databricks-llama-2-70b-chat",
        "api_base": 'https://e2-dogfood.staging.cloud.databricks.com/serving-endpoints',
        **db_key,
    },
}


def api_generate(model, prompt, tenacity=False, **kwargs):
    messages = [{"role": "user", "content": prompt}]
    kwargs = kwargs | models[model] | {"messages": messages}

    if kwargs.get('temperature', None) == 0.0 and '.mosaicml.' in kwargs.get('api_base', ''):
        print('Setting top_k=1 for zero temperature')
        kwargs['top_k'] = 1

    def helper(**kwargs):
        return completion(**kwargs)
    if tenacity:
        helper = retry(wait=wait_random_exponential(min=60, max=600), stop=stop_after_attempt(6))(helper)
    
    if kwargs.get('ngrok'):
        from openai import OpenAI
        client = OpenAI(
            base_url=kwargs['api_base'],
            api_key=kwargs['api_key'],
        )
        response = client.chat.completions.create(
            model=kwargs['model'],
            messages=messages
        )
    else:
        try:
            response = helper(**kwargs)
        except Exception as e:
            print(f"Error: {e}")
            return e.message

    # response = completion(**kwargs)
    return response.choices[0].message.content

# testing the api_generate function
def test_model(model="mixtral-instruct", prompt="twinkle twinkle little star", max_tokens=100, system=None):
    if system:
        system_prompt = {'role': 'system', 'content': system}
        messages = [system_prompt, {"role": "user", "content": prompt}]
    else:
        messages = [{"role": "user", "content": prompt}]
    kwargs = {"temperature": 0.0, "max_tokens": max_tokens}
    kwargs = kwargs | models[model] | {"messages": messages}
    response = completion(**kwargs)
    print(f"Prompting Model: {model} with prompt: {prompt}")
    print(response)
    print("\n\n")
    print(response.choices[0].message.content)

# test_model("gpt-4")
# test_model("claude-3-opus")
# test_model("claude-3-sonnet")
# test_model("gemini-1.0-pro", prompt="What is 2+2?")