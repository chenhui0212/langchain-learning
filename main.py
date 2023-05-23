from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv, find_dotenv
from langchain.chains import ConversationChain
from langchain.chains import SimpleSequentialChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import HNLoader
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.utilities import OpenWeatherMapAPIWrapper
from langchain.agents import load_tools, initialize_agent, AgentType

load_dotenv(find_dotenv())


#%%
llm = OpenAI(temperature=0.9)
prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)
chain = LLMChain(llm=llm, prompt=prompt, verbose=True)
chain("colorful socks")


# %%
chat = ChatOpenAI(temperature=0)
chain = ConversationChain(
    llm=chat,
    memory=ConversationBufferMemory(),
    verbose=True
)
chain.predict(input="Answer briefly. What are the first 3 color of rainbow?")
chain.predict(input="And the next 4?")


# %%
prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)
chain = LLMChain(llm=llm, prompt=prompt, verbose=True)

second_prompt = PromptTemplate(
    input_variables=["company_name"],
    template="Write a catchphrase for the following company: {company_name}"
)
second_chain = LLMChain(llm=llm, prompt=second_prompt, verbose=True)

seq_chain = SimpleSequentialChain( chains=[chain, second_chain], verbose=True)
seq_chain("colorful socks")


# %%
loader = HNLoader("https://news.ycombinator.com/item?id=34422627")
data = loader.load()
print(f"Found {len(data)} comments")
print(f"Here's a sample:\n\n {data[0].page_content}")


# %%
with open('blog.txt') as f:
    fc = f.read()
print(f"You have {len(fc)} characters in your blog")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)
text = text_splitter.create_documents([fc])
print(f"Split into {len(text)} documents\n")

print("Preview:")
print(text[0].page_content, "\n")
print(text[1].page_content)


# %%
loader = TextLoader('blog.txt')
doc = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_documents(doc)

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(texts, embeddings)
retriver = db.as_retriever()

docs = retriver.get_relevant_documents("How to use structured tools?")
print("\n\n".join([x.page_content[:200] for x in docs[:2]]))


#%%
weather = OpenWeatherMapAPIWrapper()
weather_data = weather.run("Shanghai,CN")
print(weather_data)


# %%
llm = OpenAI(temperature=0)
tools = load_tools(["openweathermap-api"], llm)

agent_chain = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)
agent_chain.run("What's the weather like in Shanghai?")


# %%
