from typing import TypedDict, Annotated, List, Dict, Any, Optional, Literal

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent, tools_condition, ToolNode
from langchain_core.messages import BaseMessage, SystemMessage, AIMessage, HumanMessage, ToolMessage
import re
import operator
from schemas import UserIntent, SessionState, AnswerResponse, SummarizationResponse, CalculationResponse, UpdateMemoryResponse
from prompts import get_intent_classification_prompt, get_chat_prompt_template, MEMORY_SUMMARY_PROMPT


class AgentState(TypedDict):
    """El estado agéntico del objeto"""

    # Conversación actual
    user_input: Optional[str]
    messages: Annotated[List[BaseMessage], add_messages]

    # Intent and routing
    intent: Optional[UserIntent]
    next_step: str

    # Memoria y contexto
    conversation_summary: str
    active_documents: Optional[List[str]]

    # Estado de la tarea actual
    current_response: Optional[Dict[str, Any]]
    tools_used: List[str]

    # Manejo de la sesión
    session_id: Optional[str]
    user_id: Optional[str]

    # Usamos rducers para saber cómo combinar el estado ente los nodos
    # Con operator.add cada nodo puede devolver {"actions_taken":["nombre_nodo"]}
    actions_taken: Annotated[List[str], operator.add]


def invoke_react_agent(response_schema: type[BaseModel], messages: List[BaseMessage], llm, tools):
    llm_with_tools= llm.bind_tools(tools)

    agent= create_react_agent(model=llm_with_tools, tools=tools, response_format= response_schema)
    result= agent.invoke({"messages":messages})
    tools_used= [t.name for t in result.get("messages", []) if isinstance(t, ToolMessage)]

    return result, tools_used

def classify_intent(state: AgentState, config: RunnableConfig)-> AgentState:
    """
    Clasifica la intención del usuario y actualiza el sigiente paso (next_step). También
    guarda esta función ejecutada añadiendo "classify_intent" a actions_taken
    """

    llm= config.get("configurable", {}).get("llm")
    if llm is None:
        raise ValueError("Missing LLM in config['configurable]['llm']")
    
    history= state.get("messages",[])
    user_input=state.get("user_input", "") or ""

    # 1) Forzamos salida estructurada
    structured_llm= llm.with_structured_output(UserIntent)

    # 2) Creamos el prompt (según el README: formar con user_input y conversation_history)
    prompt= get_intent_classification_prompt().format(user_input=user_input, conversation_history= history)

    # 3) Invocamos el modelo
    intent: UserIntent= structured_llm.invoke(prompt)

    # 4) Routing--> next_step
    intent_type= getattr(intent, "intent_type", "unknown")
    if intent_type== "qa":
        next_step= "qa_agent"
    elif intent_type== "summarization":
        next_step= "summarization_agent"
    elif intent_type== "calculation":
        next_step= "calculation_agent"
    else:
        next_step= "qa_agent" # el default

    return{"actions_taken":["classify_intent"], "intent": intent, "next_step": next_step}

def qa_agent(state: AgentState, config: RunnableConfig)-> AgentState:
    """
    Manejan las tareas de Q&A y recordan la acción
    """
    llm= config.get("configurable").get("llm")
    tools= config.get("configurable").get("tools")

    prompt_template= get_chat_prompt_template("qa")

    messages= prompt_template.invoke({"input":state["user_input"], "chat_history":state.get("mesages", [])}).to_messages

    result, tools_used= invoke_react_agent(AnswerResponse, messages, llm, tools)

    return{"messages": result.get("messages", []),
           "actions_taken":["qa_agent"],
           "current_response":result,
           "tools_used":tools_used,
           "next_step": "update_memory"}

def summarization_agent(state: AgentState, config: RunnableConfig)-> AgentState:
    """
    Maneja las tareas de sumarización y guarda la acción
    """

    llm= config.get("configurable", {}).get("llm")
    tools= config.get("configurable", {}).get("tools")
    if llm is None:
        raise ValueError("Missing LLM in config['configurable']['llm']")
    
    prompt_template= get_chat_prompt_template("summarization")
    messages= prompt_template.invoke({
        "input": state.get("user_input"),
        "chat_history": state.get("messages", []),
    }).to_messages()

    result, tools_used= invoke_react_agent(SummarizationResponse, messages= messages, llm= llm, tools= tools)

    return{
        "messages": result.get("messages", []),
        "actions_taken":["sumarization_agent"],
        "current_response": result,
        "tools_used": tools_used,
        "next_step": "update_memory",
    }

def calculation_agent(state: AgentState, config: RunnableConfig)-> AgentState:
    """
    Maenja las tareas de cálculo y guarda la acción 
    """

    llm= config.get("configurable", {}).get("llm")
    tools= config.get("configurable", {}).get("tools")
    if llm is None:
        raise ValueError("Missing LLM in config['configurable']['llm']")
    
    prompt_template= get_chat_prompt_template("calculation")
    messages= prompt_template.invoke({
        "input": state.get("user_input"),
        "chat_history": state.get("messages", []),
    }).to_messages()

    result, tools_used= invoke_react_agent(CalculationResponse, messages= messages, llm= llm, tools= tools)

    return{
        "messages": result.get("messages", []),
        "actions_taken":["calculation_agent"],
        "current_response": result,
        "tools_used": tools_used,
        "next_step": "update_memory",
    }



def update_memory(state: AgentState, config: RunnableConfig)-> AgentState:
    """
    Actualiza la memoria de la conversación y guarda la acción 
    """
    # Necesitamos recuperar los configurables

    llm= config.get("configurable", {}).get("llm")
    if llm is None:
        raise ValueError("Missing LLM in config['configurable']['llm']")
    
    # Prompt: sistema + placeholder del chat history
    prompt_with_history= ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(MEMORY_SUMMARY_PROMPT),
        MessagesPlaceholder("chat_history"),
    ]).invoke({"chat_history":state.get("messages", [])})

    # Structured output con el schema correcto

    structured_llm= llm.with_structured_output(UpdateMemoryResponse)
    response: UpdateMemoryResponse= structured_llm.invoke(prompt_with_history)

    return{
        "actions_taken": ["update_memory"],
        "conversation_summary": response.summary,
        "active_documents": response.document_ids,
        "next_step": "end"
    }
  
def should_continue(state: AgentState) -> str:
    """Router function"""
    return state.get("next_step", "end")

def create_workflow(llm, tools):
    """
    Creamos los agentes de LangGraph
    Compila el workflow con un InMemorySaver chekpointer que hace persistir el estado
    """
    
    workflow= StateGraph(AgentState)

    # 1) Nodos
    workflow.add_node("classify_intent", classify_intent)
    workflow.add_node("qa_agent", qa_agent)
    workflow.add_node("summarization_agent", summarization_agent)
    workflow.add_node("calculation_agent", calculation_agent)
    workflow.add_node("update_memory", update_memory)

    # 2) Entry point
    workflow.set_entry_point("classify_intent")

    # 3) Routing según next_step
    workflow.add_conditional_edges(
        "classify_intent",
        should_continue,
        {
            "qa_agent": "qa_agent",
            "summarization_agent": "summarization_agent",
            "calculation_agent": "calculation_agent",
            "end": END,
        }
    )

    # 4) Cada agente debe acabar en update memory
    workflow.add_edge("qa_agent", "update_memory")
    workflow.add_edge("summarization_agent", "update_memory")
    workflow.add_edge("calculation_agent", "update_memory")

    # 5) update_memory termina
    workflow.add_edge("update_memory", END)

    # 6) Compilación con el checkpointer
    return workflow.compile(checkpointer=InMemorySaver())


