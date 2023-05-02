from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv, find_dotenv
from langchain.chains import ConversationChain
from langchain.chains import SimpleSequentialChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

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
