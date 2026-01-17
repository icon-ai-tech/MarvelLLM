import logging
from datetime import datetime
from typing import Annotated, Dict, List, Literal, TypedDict, Union

from langchain_core.messages import BaseMessage, HumanMessage, RemoveMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import Messages, add_messages
from pymongo import MongoClient

from src.config import MONGO_DB_PATH, MONGO_DB_NAME, MAX_MESSAGES_HISTORY
from src.runnables import MarvelRunnables

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("MarvelAssistant")

# MongoDB connection
mongo_client = MongoClient(MONGO_DB_PATH)
db = mongo_client[MONGO_DB_NAME]
logs_collection = db['logs']


def concat_up_to_10(left: Messages, right: Messages) -> Messages:
    """Combine message lists keeping last 10 messages."""
    return add_messages(left=left, right=right)[-MAX_MESSAGES_HISTORY:]


class State(TypedDict):
    """Main state container for the conversation graph."""
    messages: Annotated[List[BaseMessage], concat_up_to_10]
    query: str
    standalone_question: str
    search_result: List[str]
    scenario_result: str
    sql_query_result: str
    final_output: str
    user_id: str
    extracted_programs: List[str]
    closest_matches: List[List[str]]
    clarification_required: List[bool]
    faq_result: Dict[str, Union[str, float]]
    is_faq_confident: bool
    skip_faq: bool
    faq_minus_result: Dict[str, Union[str, float]]
    is_faq_minus_confident: bool
    is_censor_confident: bool
    censorship_result: List[str]


class LogService:
    """Centralized logging service for MongoDB."""

    @staticmethod
    def save(user_id: str, action: str, details: str) -> None:
        """Save log entry to MongoDB."""
        log_entry = {
            "user_id": user_id,
            "action": action,
            "details": details,
            "timestamp": datetime.now(),
        }
        logs_collection.insert_one(log_entry)
        logger.info("Logged action: %s for user: %s", action, user_id)


class GraphBuilder:
    """Helper class for building the state graph."""

    def __init__(self, assistant: "MarvelAssistant"):
        self.assistant = assistant
        self.builder = StateGraph(State)

    def build(self) -> StateGraph:
        """Construct and configure the state graph."""
        self._add_nodes()
        self._configure_edges()
        return self.builder.compile(checkpointer=self.assistant.checkpointer)

    def _add_nodes(self) -> None:
        """Register all graph nodes."""
        nodes = [
            ("start_new_dialog", self.assistant.start_new_dialog),
            ("check_new_dialog", self.assistant.check_new_dialog),
            ("censorship_check", self.assistant.censorship_check),
            ("faq_minus_check", self.assistant.faq_minus_check),
            ("faq_check", self.assistant.faq_check),
            ("scenario", self.assistant.scenario),
            ("sql_gen", self.assistant.sql_gen),
            ("retriever", self.assistant.retriever),
            ("answer", self.assistant.answer),
            ("contextualize_chain", self.assistant.contextualize),
        ]

        for name, handler in nodes:
            self.builder.add_node(name, handler)

    def _configure_edges(self) -> None:
        """Set up graph transitions and conditional edges."""
        
        self.builder.add_edge(START, "check_new_dialog")
        self.builder.add_conditional_edges(
            "check_new_dialog",
            self._route_after_new_dialog_check,
            {
                "new_dialog": "start_new_dialog",
                "censorship_check": "censorship_check",
            },
        )
        self.builder.add_conditional_edges(
            "censorship_check",
            self._route_after_censorship_check,
            {
                "answer": END, 
                "faq_minus_check": "faq_minus_check",
            },
        )
        self.builder.add_conditional_edges(
            "faq_minus_check",
            self._route_after_faq_minus_check,
            {
                "answer": END,
                "faq_check": "faq_check",
            },
        )
        self.builder.add_conditional_edges(
            "faq_check",
            self._route_after_faq_check,
            {
                "answer": END,
                "contextualize_chain": "contextualize_chain",
            },
        )

        self.builder.add_edge("contextualize_chain", "scenario")
        self.builder.add_conditional_edges(
            "scenario",
            self._route_scenario_result,
            {"retriever": "retriever", "sql_gen": "sql_gen", 'answer':'answer'},
        )

        self.builder.add_edge("sql_gen", "retriever")
        self.builder.add_edge("retriever", "answer")

        # Terminal nodes
        self.builder.add_edge("answer", END)
        self.builder.add_edge("start_new_dialog", END)

    def _route_after_new_dialog_check(self, state: State) -> Literal["new_dialog", "censorship_check"]:
        """Determine path after checking for new dialog request."""
        if "начать новый диалог" in state["query"].lower():
            return "new_dialog"
        return "censorship_check"

    def _route_after_censorship_check(self, state: State) -> Literal["answer", "faq_minus_check"]:
        """Determine path after checking for new dialog request."""
        if state["is_censor_confident"]:
            return "answer"
        return "faq_minus_check"

    def _route_after_faq_minus_check(self, state: State) -> Literal["answer", "faq_check"]:
        """Determine path after stop-questions check."""
        if state["is_faq_minus_confident"]:
            return "answer"
        return "faq_check"

    def _route_after_faq_check(self, state: State) -> Literal["answer", "contextualize_chain"]:
        """Determine path after FAQ check."""
        if state["is_faq_confident"]:
            return "answer"
        return "contextualize_chain"

    def _route_scenario_result(self, state: State) -> Literal["retriever", "sql_gen"]:
        """Determine processing path after scenario analysis."""
        scenario = state["scenario_result"].action
        if "SQL" in scenario:
            return "sql_gen"
        elif "RAG" in scenario:
            return "retriever"
        else:
            return "answer"


class MarvelAssistant:
    """Main assistant class handling conversation processing."""

    def __init__(self, runnables: MarvelRunnables, checkpointer: MongoDBSaver):
        self.runnables = runnables
        self.checkpointer = checkpointer
        self.graph = GraphBuilder(self).build()
        logger.info("Conversation graph successfully initialized")

    def _log_node_start(self, state: State, node_name: str):
        """Log node entry with input details."""
        LogService.save(
            state["user_id"],
            "start",
            state.get('query', ''),
        )

    def check_new_dialog(self, state: State) -> State:
        """Check if user requested new dialog."""
        self._log_node_start(state, "check_new_dialog")
        is_new_dialog = "начать новый диалог" in state["query"].lower()
        result = {**state, "is_new_dialog": is_new_dialog}

        LogService.save(
            state["user_id"],
            "new_dialog",
            is_new_dialog,
        )
        return result
    
    def censorship_check(self, state: State, config: RunnableConfig) -> State:
        """Check if query is a censorship."""
        logger.info("Checking censorship for query")
        censorship_response = self.runnables.censorship_chain.invoke({"query": state["query"]})
        
        new_state = {
            **state,
            "censorship_result": censorship_response.result,
            "is_censor_confident": censorship_response.is_confident,
        }
        
        LogService.save(
            state["user_id"],
            "censorship_check",
            (
                f"Input: '{state['query']}' | "
                f"Confidence: {censorship_response.is_confident} | "
                f"Stop Word: {censorship_response.result}"
            ),
        )
        if censorship_response.is_confident:
            new_state["final_output"] = "Я создан для предоставления достоверной и этичной информации об образовательном процессе. Могу помочь с информацией о поступлении, обучении и образовательных программах."
            LogService.save(
                state["user_id"],
                "stop_words_detected",
                f"Used stop-words answer: {new_state['final_output']}",
            )
        else:
            LogService.save(state["user_id"], "no_stop_words", "No stop-words match found")

        return new_state

    def faq_minus_check(self, state: State, config: RunnableConfig) -> State:
        """Check if query is a stop-question."""
        logger.info("Checking stop-questions for query")
        faq_minus_response = self.runnables.faq_minus_chain.invoke({"query": state["query"]})

        new_state = {
            **state,
            "faq_minus_result": faq_minus_response.result,
            "is_faq_minus_confident": faq_minus_response.is_confident,
        }
        LogService.save(
            state["user_id"],
            "faq_minus",
            (
                f"Input: '{state['query']}' | "
                f"Confidence: {faq_minus_response.is_confident} | "
                f"Score: {faq_minus_response.result.get('score', 0):.2f} | "
                f"Stop Theme: {faq_minus_response.result.get('stop_theme', 'N/A')} | "
                f"Stop Answer: {faq_minus_response.result.get('stop_answer', 'N/A')}"
            ),
        )

        if faq_minus_response.is_confident:
            new_state["final_output"] = faq_minus_response.result["stop_answer"]
            LogService.save(
                state["user_id"],
                "stop_question_detected",
                f"Used stop-question answer: {faq_minus_response.result['stop_theme']}",
            )
        else:
            LogService.save(state["user_id"], "no_stop_question", "No stop-question match found")

        return new_state

    def faq_check(self, state: State, config: RunnableConfig) -> State:
        """Check if query can be answered by FAQ."""
        skip_faq = config.get("configurable", {}).get("skip_faq", False)

        if skip_faq:
            logger.info("Skipping FAQ check as requested")
            return {**state, "skip_faq": True, "is_faq_confident": False}

        logger.info("Checking FAQ for query")
        faq_response = self.runnables.faq_chain.invoke({"query": state["query"]})

        LogService.save(
            state["user_id"],
            "faq_check",
            (
                f"Input: '{state['query']}' | "
                f"Confidence: {faq_response.is_confident} | "
                f"Score: {faq_response.result.get('score', 0):.2f} | "
                f"Header: {faq_response.result.get('header', 'N/A')} | "
                f"Text: {faq_response.result.get('text', 'N/A')}..."
            ),
        )

        new_state = {
            **state,
            "messages": state["messages"] + [state["query"]],
            "faq_result": faq_response.result,
            "is_faq_confident": faq_response.is_confident,
            "skip_faq": False,
        }

        if faq_response.is_confident:
            new_state["final_output"] = faq_response.result["text"]
            LogService.save(state["user_id"], "faq_confident_answer", f"Used FAQ: {faq_response.result['header']}")
        else:
            LogService.save(state["user_id"], "faq_no_match", "No confident FAQ match found")

        return new_state

    def contextualize(self, state: State, config: RunnableConfig) -> State:
        """Add conversation context to user query."""
        thread_id = config["metadata"].get("thread_id", "unknown")
        logger.info(f"Contextualizing {state["messages"]} query {state["query"]} for thread: {thread_id}")

        if not state["messages"]:
            return {**state, "standalone_question": state["query"]}

        contextualized = self.runnables.contextualize_chain.invoke(
            {
                "chat_history": str([i.content for i in state["messages"]]),
                "input": state["query"],
            }
        )
        LogService.save(
            state["user_id"],
            "contextualize",
            contextualized.rephrased_query
        )
        logger.info(f"Contextualizing result: {contextualized.rephrased_query}")
        return {**state, "standalone_question": contextualized.rephrased_query}

    def start_new_dialog(self, state: State) -> State:
        """Reset conversation state for new dialog."""
        logger.info("Initializing new dialog for user: %s", state["user_id"])
        LogService.save(state["user_id"], "dialog_reset", f"Clearing {len(state['messages'])} messages from history")
        return {
            **state,
            "messages": [RemoveMessage(id=m.id) for m in state["messages"]],
            "query": "",
            "standalone_question": "",
            "search_result": [],
            "scenario_result": "",
            "sql_query_result": "",
            "final_output": "Начинаем новый диалог!",
            "user_id": state["user_id"],
            "extracted_programs": [],
            "closest_matches": [],
            "clarification_required": [],
            "faq_result": {},
            "is_faq_confident": False,
            "skip_faq": False,
            "faq_minus_result": {},
            "is_faq_minus_confident": False,
            "is_censor_confident": False,
            "censorship_result": [],
        }

    def scenario(self, state: State) -> State:
        """Determine processing scenario for the query."""
        logger.info("Analyzing query scenario")
        query = f'Изначальный запрос: {state["query"]} -> Переформулированный запрос: {state["standalone_question"]}'
        result = self.runnables.scenario_chain.invoke({"query": query})
        print("result_scenario")
        print(result)
        LogService.save(state["user_id"], "scenario", result.action)
        return {**state, "scenario_result": result}

    def sql_gen(self, state: State) -> State:
        """Generate SQL query based on context."""
        logger.info("Generating SQL query")
        sql_result = self.runnables.sql_gen_chain.invoke(
            {
                "sql_shots": [],# "sql_shots": state["sql_shots_list"].example_queries,
                "example_questions": [],#"example_questions": state["sql_shots_list"].example_questions,
                "query": state["standalone_question"],
            }
        )
        LogService.save(state["user_id"], "sql_result", sql_result.generated_sql)
        return {**state, "sql_query_result": sql_result}

    def retriever(self, state: State) -> State:
        """Execute Milvus query."""
        logger.info("Performing Milvus")
        search_results = self.runnables.retriever_chain.invoke({"query": state["query"]})
        LogService.save(state["user_id"], "rag_results", search_results.search_result)
        return {**state, "search_result": search_results.search_result}

    def answer(self, state: State) -> State:
        """Generate final answer based on processing results."""
        if any(state.get("clarification_required", [])):
            return self._handle_clarification(state)
        return self._generate_standard_answer(state)

    def _handle_clarification(self, state: State) -> State:
        """Generate clarification request for ambiguous programs."""
        options = "\n".join(f"{i+1}. {program}" for i, program in enumerate(state["closest_matches"][0]))
        message = f"Пожалуйста, уточните интересующие программы:\n\n{options}"

        LogService.save(state["user_id"], "clarification_required", f"Options presented:\n{options}")
        return {**state, "final_output": message, "messages": state["messages"]}  # + [HumanMessage(content=message)]

    def _generate_standard_answer(self, state: State) -> State:
        """Generate standard answer based on processing results."""
        query = f'Изначальный запрос: {state["query"]} -> Переформулированный запрос: {state["standalone_question"]}'
        context = self._build_answer_context(state)
        answer = self.runnables.answer_chain.invoke(
            {"context": context, "query": query, "scenario": state["scenario_result"]}
        )

        LogService.save(state["user_id"], "answer_context", context)
        LogService.save(state["user_id"], "answer", answer.final_output)
        return {
            **state,
            "final_output": answer.final_output,
            "messages": state["messages"],  # , HumanMessage(content=answer.final_output)
        }

    def _build_answer_context(self, state: State) -> str:
        scenario = state["scenario_result"].action
        history = str([m.content for m in state["messages"]])

        if "SQL" in scenario:
            sql = getattr(state["sql_query_result"], "generated_sql", "")
            rag = state.get("search_result", [])
            return (
                f"{history}\n\n"
                f"SQL INFO:\n{sql}\n\n"
                f"RAG INFO:\n{rag}"
            )

        if "RAG" in scenario:
            return f"{history}\n\nRAG INFO:\n{state.get('search_result', [])}"

        return history