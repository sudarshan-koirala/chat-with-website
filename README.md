# chat-with-website
Simple Streamlit as well as Chainlit app to have interaction with your website URL.

### Chat with your documents 🚀
- [OpenAI model](https://platform.openai.com/docs/models) as Large Language model
- [Ollama](https://ollama.ai/) and `mistral` as Large Language model
- [LangChain](https://python.langchain.com/en/latest/modules/models/llms/integrations/huggingface_hub.html) as a Framework for LLM
- [Streamlit](https://streamlit.io/) as well as [Chainlit](https://docs.chainlit.io/) for deploying.

## System Requirements

You must have Python 3.9 or later installed. Earlier versions of python may not compile.  

---

## Steps to Replicate 

1. Fork this repository and create a codespace in GitHub as I showed you in the youtube video OR Clone it locally.
```
git clone https://github.com/sudarshan-koirala/chat-with-website.git
cd chat-with-website
```

2. Rename example.env to .env with `cp example.env .env`and input the OpenAI API key as follows. Get OpenAI API key from this [URL](https://platform.openai.com/account/api-keys). You need to create an account in OpenAI webiste if you haven't already.
   ```
   OPENAI_API_KEY=your_openai_api_key
   ```

   For langsmith, take the environment variables from [LangSmith](https://smith.langchain.com/) website
   
3. Create a virtualenv and activate it
   ```
   python3 -m venv .venv && source .venv/bin/activate
   ```

4. Run the following command in the terminal to install necessary python packages:
   ```
   pip install -r requirements.txt
   ```

5. Run the following command in your terminal to start the chat UI:
   ```
   streamlit run chat_with_website_openai.py
   streamlit run chat_with_website_ollama.py
   ```

6. For chainlit, use the following command in your terminal.
```
python3 ingest.py #for ingesting
chainlit run main.py #for chainlit ui
```
