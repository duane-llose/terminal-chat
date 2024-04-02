from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import MessagesPlaceholder, HumanMessagePromptTemplate, ChatPromptTemplate
# from langchain.memory import ConversationBufferMemory, FileChatMessageHistory
from langchain.memory import ConversationSummaryMemory, FileChatMessageHistory
from dotenv import load_dotenv

load_dotenv()

chat = ChatOpenAI()

memory = ConversationSummaryMemory(
    # chat_memory=FileChatMessageHistory("messages.json"),
    llm=chat,
    memory_key="messages", 
    return_messages=True
)

prompt = ChatPromptTemplate(
    input_variables=["content", "messages"],
    messages=[
        MessagesPlaceholder(variable_name="messages"),
        HumanMessagePromptTemplate.from_template("{content}")
    ]
)

chain = LLMChain(
    llm=chat,
    prompt=prompt,
    memory=memory,
    verbose=True
)

while True:
    content = input(">> ")
    result = chain({"content": content})
    print(result["text"])

# history = []

# while True:
#     content = input(">> ")
#     history.append(content)
#     result = chain({"content": ' '.join(history) })
#     history.append(result["text"])
#     print(result["text"])