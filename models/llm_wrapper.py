import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain_core.tracers import LangChainTracer

#load .env 
load_dotenv()

#steup llm with tracing automatically enabled
llm = ChatGroq(
    temperature=0.2,#creativity
    model="llama3-8b-8192",
    streaming=False
)

def get_llm_response(prompt: str) -> str:
    """
    Calls the Groq LLM and returns the output as a clean string.
    LangSmith tracing is automatically enabled.
    """
    try:
        messages = [HumanMessage(content=prompt)]
        response = llm.invoke(messages)

        # Normalize content
        content = response.content
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            # Flatten list of strings or extract from dicts
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and "text" in item:
                    parts.append(item["text"])
            return " ".join(parts).strip()
        else:
            return str(content)

    except Exception as e:
        print(f"❌ LLM call failed: {e}")
        return "Error: LLM call failed"


