from langchain_groq import ChatGroq


class LLM:
    
    def __init__(self,modelName,api_key,temperature):

        self.modelName = modelName
        self.api_key = api_key
        self.temperature = temperature

        self.llm = ChatGroq(model=modelName,api_key=api_key,temperature=temperature)

    def get_groq_client(self):
        
        return self.llm
        
        
    