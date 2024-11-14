# ESTP-LLMRAG


[![Generic badge](https://img.shields.io/badge/python-3.8%7C3.9%7C3.10%7C3.11-blue.svg)](https://pypi.org/project/pypiserver/)

---

## Contents

* [Overview](README#Overview)
* [Features](README#What-Does-Langchain-Chatchat-Offer)
* [Quick Start](README#Quick-Start)
    * [Installation](README#Installation)
* [Acknowledgement](README#Acknowledgement)

## Overview

🤖️ A question-answering application based on local knowledge bases using
the [langchain](https://github.com/langchain-ai/langchain) concept.

✅ This project supports mainstream open-source LLMs, embedding models, and vector databases, allowing full **open-source
** model **offline private deployment**. Additionally, the project supports OpenAI GPT API
calls and will continue to expand access to various models and model APIs.

⛓️ The implementation principle of this project is as shown below, including loading files -> reading text -> text
segmentation -> text vectorization -> question vectorization -> matching the `top k` most similar text vectors with the
question vector -> adding the matched text as context along with the question to the `prompt` -> submitting to the `LLM`
for generating answers.

📀 This project supports video retrieval, based on the video retrieval model Clip4Clip. Giving a brief description of a video and a folder that stores multiple videos,
the app will return the full path to the videos that best fits your description. You could choose the number of returned video paths (1-10).


## Quick Start

### Installation

#### 0. Software and Hardware Requirements

💡 On the software side, this project supports Python 3.8-3.11 environments

💻 It can be used under various hardware conditions such as CPU, GPU, NPU, and MPS.

#### 1. Install Langchain-Chatchat



```shell
pip install langchain-chatchat -U
```

[!Note]
> Since the model deployment framework Xinference requires additional Python dependencies when integrated with
> Langchain-Chatchat, it is recommended to use the following installation method if you want to use it with the
> Xinference
> framework:
> ```shell
> pip install "langchain-chatchat[xinference]" -U
> ```

2. Model Inference Framework and Load Models

It supports integration with mainstream model inference frameworks such
as [Xinference](https://github.com/xorbitsai/inference), [Ollama](https://github.com/ollama/ollama), etc.
Here is an example of Xinference. Please refer to
the [Xinference Document](https://inference.readthedocs.io/zh-cn/latest/getting_started/installation.html) for framework
deployment and model loading.


#### 3. View and Modify Langchain-Chatchat Configuration

The following introduces how to view and modify the configuration.

##### 3.1 View chatchat-config Command Help

Enter the following command to view the optional configuration types:

```shell
chatchat-config --help
```

You will get the following response:

```text 
Usage: chatchat-config [OPTIONS] COMMAND [ARGS]...

  Instruction` chatchat-config` workspace config

Options:
  --help  Show this message and exit.

Commands:
  basic   basic config
  kb      knowledge base config
  model   model config
  server  service config
```

You can choose the required configuration type based on the above commands. For example, to view or
modify `basic configuration`, you can enter the following command to get help information:

```shell
chatchat-config basic --help
```

You will get the following response:

```text
Usage: chatchat-config basic [OPTIONS]

  Basic configs

Options:
  --verbose [true|false]  Enable verbose config
  --data TEXT             init log storage path
  --format TEXT           log format
  --clear                 Clear config
  --show                  List config
  --help                  Show this message and exit.
```

##### 3.2 Use chatchat-config to Modify Corresponding Configuration Parameters

To modify the `default llm` model in `model configuration`, you can execute the following command to view the
configuration item names:

```shell
chatchat-config basic --show
```

If no configuration item modification is made, the default configuration is as follows:

```text 
{
    "log_verbose": false,
    "CHATCHAT_ROOT": "/root/anaconda3/envs/chatchat/lib/python3.11/site-packages/chatchat",
    "DATA_PATH": "/root/anaconda3/envs/chatchat/lib/python3.11/site-packages/chatchat/data",
    "IMG_DIR": "/root/anaconda3/envs/chatchat/lib/python3.11/site-packages/chatchat/img",
    "NLTK_DATA_PATH": "/root/anaconda3/envs/chatchat/lib/python3.11/site-packages/chatchat/data/nltk_data",
    "LOG_FORMAT": "%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s",
    "LOG_PATH": "/root/anaconda3/envs/chatchat/lib/python3.11/site-packages/chatchat/data/logs",
    "MEDIA_PATH": "/root/anaconda3/envs/chatchat/lib/python3.11/site-packages/chatchat/data/media",
    "BASE_TEMP_DIR": "/root/anaconda3/envs/chatchat/lib/python3.11/site-packages/chatchat/data/temp",
    "class_name": "ConfigBasic"
}
```

##### 3.3 Use chatchat-config to Modify Corresponding Configuration Parameters

To modify the default `llm model` in `model configuration`, you can execute the following command to view the
configuration item names:

```shell
chatchat-config model --help
```

You will get:

```text 
Usage: chatchat-config model [OPTIONS]

  Model Configuration

Options:
  --default_llm_model TEXT        Default LLM model
  --default_embedding_model TEXT  Default embedding model
  --agent_model TEXT              Agent model
  --history_len INTEGER           History length
  --max_tokens INTEGER            Maximum tokens
  --temperature FLOAT             Temperature
  --support_agent_models TEXT     Supported agent models
  --set_model_platforms TEXT      Model platform configuration as a JSON string.
  --set_tool_config TEXT          Tool configuration as a JSON string.
  --clear                         Clear configuration
  --show                          Show configuration
  --help                          Show this message and exit.
```

First, view the current `model configuration` parameters:

```shell
chatchat-config model --show
```

You will get:

```text 
{
    "DEFAULT_LLM_MODEL": "glm4-chat",
    "DEFAULT_EMBEDDING_MODEL": "bge-large-zh-v1.5",
    "Agent_MODEL": null,
    "HISTORY_LEN": 3,
    "MAX_TOKENS": null,
    "TEMPERATURE": 0.7,
    ...
    "class_name": "ConfigModel"
}
```

To modify the `default llm` model to `qwen2-instruct`, execute:

```shell
chatchat-config model --default_llm_model qwen2-instruct
```

4. Custom Model Integration Configuration

After completing the above project configuration item viewing and modification, proceed to step 2. Model Inference
Framework and Load Models and select the model inference framework and loaded models. Model inference frameworks include
[Xinference](https://github.com/xorbitsai/inference),[Ollama](https://github.com/ollama/ollama),[LocalAI](https://github.com/mudler/LocalAI),[FastChat](https://github.com/lm-sys/FastChat)
and [One API](https://github.com/songquanpeng/one-api), supporting new multi-language open-source models
like [GLM-4-Chat](https://github.com/THUDM/GLM-4) and [Qwen2-Instruct](https://github.com/QwenLM/Qwen2)

If you already have an address with the capability of an OpenAI endpoint, you can directly configure it in
MODEL_PLATFORMS as follows:

```text
chatchat-config model --set_model_platforms TEXT      Configure model platforms as a JSON string.
```

- `platform_name` can be arbitrarily filled, just ensure it is unique.
- `platform_type` might be used in the future for functional distinctions based on platform types, so it should match
  the platform_name.
- List the models deployed on the framework in the corresponding list. Different frameworks can load models with the
  same name, and the project will automatically balance the load.
- Set up the model

```shell
$ chatchat-config model --set_model_platforms "[{
    \"platform_name\": \"xinference\",
    \"platform_type\": \"xinference\",
    \"api_base_url\": \"http://127.0.0.1:9997/v1\",
    \"api_key\": \"EMPT\",
    \"api_concurrencies\": 5,
    \"llm_models\": [
        \"autodl-tmp-glm-4-9b-chat\"
    ],
    \"embed_models\": [
        \"bge-large-zh-v1.5\"
    ],
    \"image_models\": [],
    \"reranking_models\": [],
    \"speech2text_models\": [],
    \"tts_models\": []
}]"
```

#### 5. Initialize Knowledge Base

> [!WARNING]
> Before initializing the knowledge base, ensure that the model inference framework and corresponding embedding model
> are
> running, and complete the model integration configuration as described in steps 3 and 4.

```shell
cd # Return to the original directory
chatchat-kb -r
```

Specify text-embedding model for initialization (if needed):

```
cd # Return to the original directory
chatchat-kb -r --embed-model=text-embedding-3-smal
```

```

The knowledge base path is in the knowledge_base directory under the path pointed by the *DATA_PATH* variable in
step `3.2`:

```shell
(chatchat) [root@VM-centos ~]#  ls /root/anaconda3/envs/chatchat/lib/python3.11/site-packages/chatchat/data/knowledge_base/samples/vector_store
bge-large-zh-v1.5  text-embedding-3-small
```





If the statement gets stuck and cannot be executed, the following command can be executed:

```shell
pip uninstall python-magic-bin
# check the version of the uninstalled package
pip install 'python-magic-bin=={version}'
```

Then follow the instructions in this section to recreate the knowledge base.

#### 6. Start the Project

```shell
chatchat -a
```

> [!WARNING]
> As the `DEFAULT_BIND_HOST` of the chatchat-config server configuration is set to `127.0.0.1` by default, it cannot be
> accessed through other IPs.
>
> To modify, refer to the following method:
> <details>
> <summary>Instructions</summary>
>
> ```shell
> chatchat-config server --show
> ```
> You will get:
> ```text 
> {
>     "HTTPX_DEFAULT_TIMEOUT": 300.0,
>     "OPEN_CROSS_DOMAIN": true,
>     "DEFAULT_BIND_HOST": "127.0.0.1",
>     "WEBUI_SERVER_PORT": 8501,
>     "API_SERVER_PORT": 7861,
>     "WEBUI_SERVER": {
>         "host": "127.0.0.1",
>         "port": 8501
>     },
>     "API_SERVER": {
>         "host": "127.0.0.1",
>         "port": 7861
>     },
>     "class_name": "ConfigServer"
> }
> ```
> To access via the machine's IP (such as in a Linux system), change the listening address to `0.0.0.0`.
> ```shell
> chatchat-config server --default_bind_host=0.0.0.0
> ```
> You will get:
> ```text 
> {
>     "HTTPX_DEFAULT_TIMEOUT": 300.0,
>     "OPEN_CROSS_DOMAIN": true,
>     "DEFAULT_BIND_HOST": "0.0.0.0",
>     "WEBUI_SERVER_PORT": 8501,
>     "API_SERVER_PORT": 7861,
>     "WEBUI_SERVER": {
>         "host": "0.0.0.0",
>         "port": 8501
>     },
>     "API_SERVER": {
>         "host": "0.0.0.0",
>         "port": 7861
>     },
>     "class_name": "ConfigServer"
> }
> ```
> </details>

## Acknowledgement
This project is based on the open-soruce project https://github.com/chatchat-space/Langchain-Chatchat, with core functionality reserved and video retrieval added.
The code of this project follows the [Apache-2.0](LICENSE) agreement.
