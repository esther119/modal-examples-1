#
# # Using the ChatGPT streaming API
#
# This example shows how to stream from the ChatGPT API as the model is generating a completion, instead of
# waiting for the entire completion to finish. This provides a much better user experience, and is what you
# get when playing with ChatGPT on [chat.openai.com](https://chat.openai.com/).

from modal import Image, Secret, Stub, web_endpoint
import os

image = Image.debian_slim().pip_install(
    "langchain",
    "openai",
    "tiktoken==0.3.0",
    "pinecone-client~=2.2.2"
)
stub = Stub(
    name="example-langchain",
    image=image,
    secrets=[Secret.from_name("my-openai-secret"), Secret.from_name("my-pinecone-api")],
)


# # pinecone test
@stub.function()
def pinecone_docsearch(user_input):
    import pinecone
    from langchain.vectorstores import Pinecone
    from langchain.embeddings.openai import OpenAIEmbeddings

    embeddings = OpenAIEmbeddings(disallowed_special=(), openai_api_key=os.environ['OPENAI_API_KEY'])
    print("pinecone start testing")
    pinecone.init(
    api_key= os.environ["pinecone-api"],
    environment='asia-southeast1-gcp-free'   
    )
    index_name =  'tim-urban-test'
    docsearch = Pinecone.from_existing_index(index_name, embeddings)
    print("pinecone end testing")
    docs = docsearch.similarity_search(user_input)
    return docs[0].page_content
    # return docsearch


@stub.function()
def AI_response_messages(user_input: str, store: str) -> str:
    template ='''
    Use the following pieces of context from waitbutwhy to answer the question at the end. 
    If you don't know the answer, just clarify that you are not sure, but this might be how Tim Urban thinks.
    '''
    engineered_user_input = f'Write a blog about {user_input} within 300 words like waitbutwhy using "you" in a casual language'
    from langchain.schema import (
        AIMessage,
        HumanMessage,
        SystemMessage
    )
    from langchain.chat_models import ChatOpenAI
    chat = ChatOpenAI(model_name='gpt-3.5-turbo-16k-0613', openai_api_key=os.environ['OPENAI_API_KEY'])   

    print("engineered_user_input", engineered_user_input)
    messages = [
        SystemMessage(content=template+store),
        HumanMessage(content=engineered_user_input)
    ]
    response=chat(messages)
    return response.content

@stub.function()
def stream_chat(prompt: str):
    import openai

    for chunk in openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    ):
        content = chunk["choices"][0].get("delta", {}).get("content")
        if content is not None:
            yield content


# ## Streaming web endpoint
#
# These four lines are all you need to take that function above and serve it
# over HTTP. It is a single function definition, annotated with decorators to make
# it a Modal function [with a web serving capability](/docs/guide/webhooks).
#
# Notice that the `stream_chat` function is passed into the retuned streaming response.
# This works because the function is a generator and is thus compatible with streaming.
#
# We use the standard Python calling convention `stream_chat(...)` and not the
# Modal-specific calling convention `stream_chat.call(...)`. The latter would still work,
# but it would create a remote function invocation which would unnecessarily involve `stream_chat`
# running in a separate container, sending its results back to the caller over the network.


# @stub.function()
# @web_endpoint()
# def web(prompt: str):
#     from fastapi.responses import StreamingResponse

#     return StreamingResponse(stream_chat(prompt), media_type="text/html")


# ## Try out the web endpoint
#
# Run this example with `modal serve chatgpt_streaming.py` and you'll see an ephemeral web endpoint
# has started serving. Hit this endpoint with a prompt and watch the ChatGPT response streaming back in
# your browser or terminal window.
#
# We've also already deployed this example and so you can try out our deployed web endpoint:
#
# ```bash
# curl --get \
#   --data-urlencode "prompt=Generate a list of 20 great names for sentient cheesecakes that teach SQL" \
#   https://modal-labs--example-chatgpt-stream-web.modal.run
# ```
#
# ## CLI interface
#
# Doing `modal run chatgpt_streaming.py --prompt="Generate a list of the world's most famous people"` also works, and uses the `local_entrypoint` defined below.
default_user_input = "first principles"
default_prompt = (
    "Generate a list of 20 great names for sentient cheesecakes that teach SQL"
)


@stub.local_entrypoint()
def main(prompt: str = default_prompt):    
    
    store = pinecone_docsearch.call(default_user_input)
    print("finished store", store)
    # store = similarity_search(default_user_input, docsearch)
    # print("finished similarity_search")
    response = AI_response_messages.call(default_user_input, store)
    print(response)

    # for part in stream_chat.call(prompt=prompt):
    #     print(part, end="")

@stub.function()
@web_endpoint(method="GET")
def web(user_input: str):
    store = pinecone_docsearch(user_input)
    print("finished store", store)
    response = AI_response_messages.call(user_input, store)
    return response