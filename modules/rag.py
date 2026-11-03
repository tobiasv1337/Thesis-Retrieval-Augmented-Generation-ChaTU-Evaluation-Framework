import logging
import os
import re
import hashlib
from pathlib import Path
from typing import Callable, List, Optional, Any

import requests
from bs4 import BeautifulSoup
from pydantic import PrivateAttr

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda, RunnableMap

from .config import Config


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

LegacyTraceSink = Optional[Callable[[str, dict], None]]


def _legacy_document_id(doc: Document) -> str:
    source = str(doc.metadata.get("source", ""))
    content = (doc.page_content or "").strip()
    digest = hashlib.sha256(f"{source}\0{content}".encode("utf-8")).hexdigest()[:32]
    return f"legacy_chunk:{digest}"


def _emit_legacy_trace(trace_sink: LegacyTraceSink, stage: str, payload: dict) -> None:
    if trace_sink is not None:
        trace_sink(stage, payload)


def _trace_document(doc: Document, *, score=None, rank=None) -> dict:
    source = str(doc.metadata.get("source", ""))
    return {
        "legacy_chunk_id": _legacy_document_id(doc),
        "source": source,
        "intent": Path(source).parent.name,
        "score": score,
        "rank": rank,
        "content": (doc.page_content or "").strip(),
    }

text = {
    "instructions": {
        "deutsch": (
            "Du bist ein hilfreicher Assistent und das Chat-Interface der Webseite der TU Berlin. "
            "Sprich die Studierenden immer mit 'du' an (informelle Anrede). "
            "Beantworte Fragen ausschließlich basierend auf dem gefundenen Webseiteninhalt. "
            "Erfinde keine Informationen. Erfinde keine Seiten. Generiere keine Links. Gebe keine E-Mail Adressen oder Telefonnummern aus. Rechne keine Geldsummen aus. "
            "Antworte auf Deutsch. "
            #"Wenn die Frage nicht beantworten kannst und passende Seitentitel angezeigt werden und diese relevant für die Frage klingen, "
            #"schlage vor, auf die entsprechenden Seitentitel-Buttons zu klicken, um weitere Informationen zu erhalten. "
            "Wenn keine der angezeigten Seiten relevant für die Frage ist und du keine Informationen finden konntest, "
            "sage ausdrücklich, dass du keine Informationen finden konntest, um die Frage zu beantworten. "
            "Empfehle dann direkt auf der Webseite nachzusehen und "
            "das Kontaktformular des Studierendensekretariats auf der Webseite zu nutzen "
            "Es ist keine E-Mail Adresse für das Studierendensekretariat verfügbar. "
            "Wenn spezifische Details fehlen, aber allgemeine Informationen vorhanden sind, antworte zuerst mit den allgemeinen Informationen "
            "und erwähne erst am Ende, dass die spezifische Information nicht gefunden wurde. "
            "Antworte strukturiert: zuerst die relevanten Informationen, dann (falls nötig) ein kurzer Hinweis am Ende, "
            "dass die spezifische Information nicht gefunden wurde. "
            "Empfehle das Kontaktformular, wenn keine relevanten Informationen im Webseiteninhalt gefunden wurden. "
        ),
        "english": (
            "You are a helpful assistant and the chat-interface of TU Berlin's website. "
            "Only answer based on the provided website content. "
            "Do not make up information. Do not make up pages. Do not generate links. Do not provide email addresses or phone numbers. Do not make any financial calculations."
            "Respond in English. "
            #"If you can't answer the question and matching pages titles are shown and they seem relevant to the question, "
            #"suggest clicking on the corresponding page-title-buttons to get more information. "
            "If none of the shown pages are relevant to the question and you can't find any information, "
            "explicitly say you could not find information to answer the question. "
            "Then recommend to look for the information on the website directly and "
            "use the website's contact form for the office of student affairs "
            "(use the words \"contact form\"). "
            "There is no e-mail-address for the office of student affairs available. "
            "If specific details are missing but general information is available, answer first with the general information "
            "and only then mention at the end that the specific information was not found. "
            "Answer in a structured way: first the relevant information, then (if needed) a short note at the end "
            "that the specific information was not found. "
            "Only recommend the contact form when no relevant information was found in the website content. "
        ),
    },
    "reminder": {
        "deutsch": (
            "Bitte beantworte die nächste Frage ausschließlich basierend auf dem Webseiteninhalt. "
            "Keine erfundenen Informationen. Keine Links. Keine Rechnungen. Antworte auf Deutsch. "
            #"Wenn die Frage nicht beantwortet werden kann und die Seitentitel-Buttons relevant klingen, schlage Seitentitel-Buttons vor. "
            "Wenn keine Seiten relevant sind, sage, dass du keine Informationen finden konntest, "
            "empfehle das Kontaktformular und nenne das Wort \"Kontaktformular\". "
            "Struktur: zuerst relevante Informationen, dann (falls nötig) kurzer Hinweis zur fehlenden spezifischen Info. "
        ),
        "english": (
            "Please answer the following question only based on the website content. "
            "Do not make up information. No Links. No calculations. Respond in English. "
            #"If you can't find the required information and the page-title-buttons seem relevant, suggest page-title-buttons. "
            "If no pages are relevant, say you could not find information to answer the question, "
            "recommend the contact form and use the words \"contact form\". "
            "Structure: first relevant information, then (if needed) a short note about missing specific info. "
        ),
    },
    "question": {
        "deutsch": "Frage",
        "english": "Question",
    },
    "contact_form_fallback": {
        "deutsch": (
            "Bitte nutze das Kontaktformular des Studierendensekretariats auf der Webseite."
        ),
        "english": (
            "Please use the contact form for the office of student affairs on the website."
        ),
    },
    "context": {
        "deutsch": "Webseiteninhalt",
        "english": "Website content",
    },
    "history": {
        "deutsch": "Gesprächsverlauf",
        "english": "Message history",
    },
    "relevant_pages": {
        "deutsch": "Seitentitel-Buttons",
        "english": "Page-title-buttons"
    }
}


class HTMLCleanerLoader:
    """Turn HTML into clean paragraph-separated text with simple, robust table handling."""

    def __init__(self, file_path: str):
        self.file_path = file_path

    def _table_to_text(self, table: BeautifulSoup) -> str:
        parts = []

        # Optional: Tabellenüberschrift
        caption = table.caption.get_text(" ", strip=True) if table.caption else ""
        if caption:
            parts.append(f"[Tabelle: {caption}]")

        rows = table.find_all("tr")
        if not rows:
            return ""

        # Nur dann Header, wenn wirklich <th> vorhanden sind
        header_row = rows[0]
        header_cells = header_row.find_all("th")
        headers = [th.get_text(" ", strip=True) for th in header_cells] if header_cells else []
        data_rows = rows[1:] if headers else rows  # nur bei echten Headern überspringen

        for tr in data_rows:
            cells = [td.get_text(" ", strip=True) for td in tr.find_all(["td", "th"])]
            cells = [c for c in cells if c]  # leere raus
            if not cells:
                continue

            if headers and len(cells) == len(headers):
                # "Header: Wert" pro Spalte
                row_str = " ; ".join(f"{h}: {c}" for h, c in zip(headers, cells) if c)
            elif len(cells) == 3:
                # typische TU-Geldtabelle: Betrag, Bezeichnung, Rechtsgrundlage
                amount, label, legal = cells
                row_str = f"{label}: {amount} ({legal})"
            elif len(cells) == 2:
                # einfache 2-Spalten-Tabelle
                row_str = f"{cells[0]}: {cells[1]}"
            else:
                # Fallback
                row_str = " | ".join(cells)

            parts.append(row_str)

        return "\n".join(parts)

    def load(self):
        with open(self.file_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f, "html.parser")

        # Tabellen zuerst in Text transformieren
        for table in soup.find_all("table"):
            table_text = self._table_to_text(table)
            table.replace_with(soup.new_string(table_text))

        # Rest wie bisher
        text_ = soup.get_text(separator="\n", strip=True)
        text_ = re.sub(r"[ \t]+", " ", text_)
        text_ = re.sub(r"\n{2,}", "\n", text_)
        text_ = text_.strip()

        return [Document(page_content=text_, metadata={"source": self.file_path})]

    def lazy_load(self):
        yield from self.load()


# === FilteredRetriever using Rasa intent filtering ===
class FilteredRetriever(BaseRetriever):
    _vectorstore: Any = PrivateAttr()
    _legal_intents: List[str] = PrivateAttr()
    _k: int = PrivateAttr()
    _trace_sink: LegacyTraceSink = PrivateAttr()
    _trace_level: str = PrivateAttr()

    def __init__(self, vectorstore, legal_intents: Optional[List[str]] = None, k: int = 5, trace_sink: LegacyTraceSink = None, trace_level: str = "none"):
        super().__init__()
        self._vectorstore = vectorstore
        self._legal_intents = legal_intents or []
        self._k = k
        self._trace_sink = trace_sink
        self._trace_level = trace_level

    def _get_relevant_documents(self, query: str, *, run_manager=None, **kwargs) -> List[Document]:
        #logger.info("_get_relevant_documents")
        # Over-fetch, then filter
        overfetch_k = max(self._k * 50, self._k)
        docs_and_scores = self._vectorstore.similarity_search_with_score(query, k=overfetch_k)
        ranked = [
            (doc, score, rank)
            for rank, (doc, score) in enumerate(docs_and_scores, start=1)
        ]
        vector_top = sorted(ranked, key=lambda item: item[1])[:self._k]

        def intent_of(doc: Document) -> str:
            # Robust across OS/path styles
            return Path(doc.metadata.get("source", "")).parent.name

        # If no legal intents provided, return global top-k
        if not self._legal_intents:
            top = vector_top
            _emit_legacy_trace(
                self._trace_sink,
                "vector_search",
                {
                    "query": query,
                    "input_count": 0,
                    "output_count": len(ranked),
                    "overfetch_k": overfetch_k,
                    "legal_intents": [],
                    "full_candidate_data": self._trace_level == "full",
                    "candidate_prefix_complete": True,
                    "candidates": [
                        _trace_document(doc, score=score, rank=rank)
                        for doc, score, rank in (
                            ranked if self._trace_level == "full" else top
                        )
                    ],
                },
            )
            _emit_legacy_trace(
                self._trace_sink,
                "intent_filter",
                {
                    "query": query,
                    "enabled": False,
                    "input_count": len(ranked),
                    "output_count": len(top),
                    "removed_count": max(0, len(ranked) - len(top)),
                    "candidates": [
                        _trace_document(doc, score=score, rank=rank)
                        for doc, score, rank in top
                    ],
                },
            )
            for doc, score, _ in top:
                doc.metadata["score"] = score
            return [doc for doc, _, _ in top]

        # Else filter by legal intents
        filtered = [
            (doc, score, rank)
            for doc, score, rank in ranked
            if intent_of(doc) in self._legal_intents
        ]
        filtered.sort(key=lambda x: x[1])

        top_filtered = filtered[:self._k]
        removed_count = 0
        removed_sample = []
        for doc, score, rank in ranked:
            if intent_of(doc) in self._legal_intents:
                continue
            removed_count += 1
            if len(removed_sample) < 10:
                removed_sample.append(
                    _trace_document(doc, score=score, rank=rank),
                )
        _emit_legacy_trace(
            self._trace_sink,
            "vector_search",
            {
                "query": query,
                "input_count": 0,
                "output_count": len(ranked),
                "overfetch_k": overfetch_k,
                "legal_intents": list(self._legal_intents),
                "full_candidate_data": self._trace_level == "full",
                "candidate_prefix_complete": True,
                "candidates": [
                    _trace_document(doc, score=score, rank=rank)
                    for doc, score, rank in (
                        ranked if self._trace_level == "full" else vector_top
                    )
                ],
            },
        )
        _emit_legacy_trace(
            self._trace_sink,
            "intent_filter",
            {
                "query": query,
                "enabled": True,
                "input_count": len(ranked),
                "output_count": len(top_filtered),
                "matching_before_top_k": len(filtered),
                "removed_count": removed_count,
                "removed_sample": removed_sample,
                "candidates": [
                    _trace_document(doc, score=score, rank=rank)
                    for doc, score, rank in top_filtered
                ],
            },
        )
        for doc, score, _ in top_filtered:
            doc.metadata["score"] = score
        return [doc for doc, _, _ in top_filtered]


# === Chat LLM (local / OpenAI-compatible) ===
class ChatAILLM(BaseChatModel):
    model: str = "meta-llama-3.1-8b-instruct"
    temperature: float = 0.0

    # Optional override for the HTTP endpoint, so different processes (RAG vs
    # judge/UEQ) can use different backends without changing code.
    endpoint: Optional[str] = None
    api_key: Optional[str] = None

    @staticmethod
    def _resolve_endpoint() -> str:
        cfg = Config()
        url = (
            cfg.get("qa_llm_url")
            or os.getenv("CHATU_QA_LLM_URL")
            or "http://127.0.0.1:5000/generate"
        )
        if url.startswith("http://0.0.0.0:"):
            url = "http://127.0.0.1:" + url.split("http://0.0.0.0:", 1)[1]
        return url.rstrip("/")

    def _call_messages(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs) -> str:
        #logger.info("_call_messages")
        try:
            role_map = {
                "human": "user",
                "ai": "assistant",
                "system": "system",
            }
            max_length = kwargs.get("max_length", 500)
            max_new_tokens = kwargs.get("max_new_tokens", max_length)
            prompt = {
                "max_length": max_length,
                "messages": [{"role": role_map.get(m.type, m.type), "content": m.content} for m in messages],
                "temperature": self.temperature,
                # Some local /generate backends default to do_sample=True which is incompatible with temperature=0.
                "do_sample": False,
            }
            import json
            with open(Config().get("az_data_dir")+"debug_prompt.json", "w") as f:
                f.write(json.dumps(prompt, indent=4))
            # Prefer an explicitly configured endpoint, fall back to the default local backend.
            url = (self.endpoint or self._resolve_endpoint()).rstrip("/")
            timeout_s = float(
                Config().get(
                    "qa_llm_timeout_seconds",
                    os.getenv("CHATU_QA_LLM_TIMEOUT_SECONDS", "20"),
                )
            )


            # If the endpoint looks like an OpenAI-compatible base URL (e.g. .../v1),
            # use /chat/completions; otherwise assume the legacy /generate backend.
            if url.endswith("/v1") or "/v1/" in (url + "/"):
                payload = {
                    "model": Config().get("qa_llm_model", self.model),
                    "messages": prompt["messages"],
                    "temperature": self.temperature,
                }
                headers = {"Content-Type": "application/json"}
                if self.api_key:
                    headers["Authorization"] = f"Bearer {self.api_key}"
                response = requests.post(
                    url=f"{url}/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=timeout_s,
                )
                response.raise_for_status()
                data = response.json()
                try:
                    choices = data.get("choices") or []
                    msg = (choices[0] or {}).get("message") or {}
                    return str(msg.get("content", ""))
                except Exception:
                    return ""

            response = requests.post(
                url=url,
                json=prompt,
                headers={"Content-Type": "application/json"},
                timeout=timeout_s,
            )
            response.raise_for_status()
            data = response.json()
            if isinstance(data, dict):
                if "error" in data and "generate" not in data:
                    return ""
                gen = data.get("generate", "")
                if isinstance(gen, list) and gen:
                    gen_text = str(gen[0])
                else:
                    gen_text = str(gen or "")
                # Guard: some backends return an error message in "generate".
                lowered = gen_text.lower()
                if any(s in lowered for s in [
                    "if do_sample",
                    "internal assert failed",
                    "cudacachingallocator",
                    "cuda driver error",
                    "missing `messages`",
                    "missing messages",
                ]):
                    return ""
                return gen_text
            return ""
        except Exception as e:
            logging.warning(f"LLM request failed: {e}")
            return ""

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        raise NotImplementedError("Use message-based interface with ChatAILLM")

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs):
        #logger.info("_generate")
        output = self._call_messages(messages, stop=stop, **kwargs)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=output))])

    @property
    def _llm_type(self) -> str:
        return "chat-openai-compatible"


# === Helpers ===
def format_docs(docs: List[Document]) -> str:
    #logger.info("format_docs")
    return "\n\n".join(doc.page_content for doc in docs)


def convert_chat_history(history: List[dict]) -> List[BaseMessage]:
    #logger.info("convert_chat_history")
    messages: List[BaseMessage] = []
    for m, msg in enumerate(history):
        role = msg.get("role")
        content = msg.get("content", "")
        if m == len(history)-1 and role == "user": continue
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    return messages


# === ChatQA with Rasa-based Retriever ===
class ChatQA:
    def __init__(
            self,
            language: str = "deutsch",
            recreate_index: bool = False,
            source_dir: Optional[str] = None,
            index_path: Optional[str] = None,
            min_chunk_length: int = 200,
            html_glob: str = "**/content*.html",
            keyword_filter: bool = True,
            use_tfidf_keywords: bool = False,
            keyword_extractor: Optional[Any] = None,
            llm: Optional[BaseChatModel] = None,
    ):
        self.language = language
        self.source_dir = source_dir
        self.index_path = index_path
        self.min_chunk_length = min_chunk_length
        self.html_glob = html_glob
        self.keyword_filter = keyword_filter
        self.use_tfidf_keywords = use_tfidf_keywords
        self.keyword_extractor = keyword_extractor
        self.llm = llm
        self.db = self.load_or_create_faiss(recreate=recreate_index)

    def _ensure_keyword_extractor(self):
        if self.keyword_extractor is None:
            from .keywords import get_shared_keyword_extractor
            self.keyword_extractor = get_shared_keyword_extractor(load_only=True)

    def _append_contact_form_fallback(self, answer_text: str) -> str:
        fallback = text["contact_form_fallback"][self.language]
        marker = "contact form" if self.language == "english" else "kontaktformular"
        if marker in (answer_text or "").lower():
            return answer_text
        if not answer_text:
            return fallback
        return f"{answer_text}\n\n{fallback}"

    def _get_query_keywords(self, text_value: str) -> set:
        query_text = text_value or ""
        if self.use_tfidf_keywords:
            self._ensure_keyword_extractor()
            try:
                kw_sentence = self.keyword_extractor.get_keywords(query_text, language=self.language)
                if isinstance(kw_sentence, str) and kw_sentence.strip():
                    query_text = kw_sentence
            except Exception:
                pass
        tokens = re.findall(r"[a-zA-ZäöüÄÖÜß]+", query_text.lower())
        return {t for t in tokens if len(t) > 3}

    def _get_keyword_filter_details(self, keyword_text: str, docs: List[Document]) -> dict:
        keywords = self._get_query_keywords(keyword_text)
        details = {
            "keywords": sorted(keywords),
            "matched": [],
            "ignored": [],
            "no_keywords": False,
        }
        if not keywords:
            details["no_keywords"] = True
            for doc in docs:
                details["matched"].append({
                    "source": doc.metadata.get("source"),
                    "score": doc.metadata.get("score"),
                    "matched_keywords": [],
                })
            return details
        for doc in docs:
            content = (doc.page_content or "").lower()
            matched = [k for k in keywords if k in content]
            entry = {
                "source": doc.metadata.get("source"),
                "score": doc.metadata.get("score"),
                "matched_keywords": matched,
            }
            if matched:
                details["matched"].append(entry)
            else:
                details["ignored"].append(entry)
        return details

    def _filter_docs_by_query(
            self,
            keyword_text: str,
            docs: List[Document],
            *,
            allow_empty: bool = False,
    ) -> List[Document]:
        keywords = self._get_query_keywords(keyword_text)
        if not keywords:
            return docs
        filtered = [
            doc for doc in docs
            if any(k in (doc.page_content or "").lower() for k in keywords)
        ]
        if filtered:
            return filtered
        return [] if allow_empty else docs

    def _build_keyword_text(self, query: str, chat_history: Optional[List[dict]] = None) -> str:
        history = chat_history or []
        user_turns = [msg.get("content", "") for msg in history if msg.get("role") == "user"]
        parts = [query] + user_turns
        return "\n".join([p for p in parts if p])

    def generate_query_variants(
            self,
            question: str,
            chat_history: Optional[List[dict]] = None,
            max_depth: int = 3,
            separator: str = "\n"
    ) -> List[dict]:
        """
        Build multiple query formulations by combining the current question with
        up to `max_depth` previous user turns.

        Returns a list of dicts containing an identifier and the resulting query text:
        [
            {"id": "current", "query": "<question>"},
            {"id": "last_1", "query": "<prev user>\\n<question>"},
            ...
        ]
        """
        history = chat_history or []
        if max_depth < 1:
            max_depth = 1

        user_turns = [msg["content"] for msg in history if msg.get("role") == "user"]

        # If the latest user turn already equals the current question, avoid duplicates.
        if user_turns and user_turns[-1].strip() == question.strip():
            user_turns = user_turns[:-1]

        variants = [{"id": "current", "query": question}]
        available_depth = min(max_depth, len(user_turns))

        for depth in range(1, available_depth + 1):
            combined_turns = user_turns[-depth:] + [question]
            combined_query = separator.join(turn.strip() for turn in combined_turns if turn.strip())
            if not combined_query:
                continue
            variants.append({
                "id": f"last_{depth}",
                "query": combined_query
            })

        return variants

    def load_or_create_faiss(
            self,
            recreate: bool = False,
            source_dir: Optional[str] = None,
            index_path: Optional[str] = None,
            min_chunk_length: Optional[int] = None,
            html_glob: Optional[str] = None,
    ):
        logger.debug("load_or_create_faiss")
        source_dir = source_dir or self.source_dir
        index_path = index_path or self.index_path or ".cache/rag_faiss_index"
        min_chunk_length = min_chunk_length if min_chunk_length is not None else self.min_chunk_length
        html_glob = html_glob or self.html_glob
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        if not recreate and os.path.exists(os.path.join(index_path, "index.faiss")):
            return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

        padded_docs = self.load_html_documents(
            source_dir=source_dir,
            min_chunk_length=min_chunk_length,
            html_glob=html_glob,
        )

        db = FAISS.from_documents(padded_docs, embeddings)
        os.makedirs(index_path, exist_ok=True)
        db.save_local(index_path)
        print(f"FAISS-Index gespeichert unter '{index_path}'")
        return db

    def load_html_documents(
            self,
            source_dir: Optional[str] = None,
            min_chunk_length: int = 200,
            html_glob: str = "**/content*.html",
    ) -> List[Document]:
        logger.debug("Loading and chunking HTML documents")
        base_dir = source_dir or os.path.join(Config().get("az_data_dir"), "scraper/topics")
        loader = DirectoryLoader(
            base_dir,
            glob=html_glob,
            loader_cls=HTMLCleanerLoader,
            show_progress=True,
            use_multithreading=True,
            max_concurrency=8,
        )
        docs = loader.load()
        if not docs:
            raise ValueError(f"No HTML documents loaded from {base_dir}")

        # Prefer paragraph/line breaks; then sentences; then spaces
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " ", ""],
            keep_separator=False,
        )
        split_docs = splitter.split_documents(docs)
        if not split_docs:
            raise ValueError(f"No chunks produced from {base_dir} (docs={len(docs)})")

        # Only keep chunks with at least min_chunk_length non-space characters
        source_stats = {}
        filtered = []
        for doc in split_docs:
            source = doc.metadata.get("source", "unknown")
            stats = source_stats.setdefault(source, {"kept": 0, "total": 0, "max_len": 0})
            stats["total"] += 1
            content_len = len(doc.page_content.strip())
            if content_len > stats["max_len"]:
                stats["max_len"] = content_len
            if content_len >= min_chunk_length:
                stats["kept"] += 1
                filtered.append(doc)

        excluded_sources = [
            (s, st["max_len"])
            for s, st in source_stats.items()
            if st["kept"] == 0
        ]
        if excluded_sources:
            total_sources = len(source_stats)
            logger.debug(
                "Excluded %d/%d source files with no chunks >= %d chars",
                len(excluded_sources),
                total_sources,
                min_chunk_length,
            )
            excluded_sources.sort(key=lambda item: item[1])
            excluded_str = "; ".join(
                f"{Path(path).name} ({max_len})"
                for path, max_len in excluded_sources
            )
            logger.debug("Excluded sources: %s", excluded_str)

        if not filtered:
            raise ValueError(
                f"No chunks >= {min_chunk_length} chars from {base_dir} "
                f"(docs={len(docs)}, chunks={len(split_docs)})"
            )

        return filtered  # no padding – preserves embedding fidelity

    def get_combined_retriever(self, intent_ranking: Optional[List[str]], trace_sink: LegacyTraceSink = None, trace_level: str = "none"):
        #logger.info("get_combined_retriever")
        legal_intents = intent_ranking or []
        return FilteredRetriever(self.db, legal_intents=legal_intents, k=30, trace_sink=trace_sink, trace_level=trace_level)

    def get_multi_context(
            self,
            question: str,
            intent_ranking: List[str],
            chat_history: Optional[List[dict]] = None,
            max_depth: int = 3,
            keyword_filter_query: Optional[str] = None,
            trace_sink: LegacyTraceSink = None,
            trace_level: str = "none",
    ):
        """
        Retrieve chunks for multiple query variants built from the question and chat history.

        Returns a dict keyed by variant id containing:
        {
            "<variant_id>": {
                "query": "<combined query text>",
                "context": "<joined chunk text>",
                "documents": [
                    {
                        "intent": "<intent dir>",
                        "score": <float>,
                        "source": "<path>",
                        "content": "<chunk text>"
                    },
                    ...
                ]
            }
        }
        """
        #logger.info("get_multi_context")
        variants = self.generate_query_variants(
            question=question,
            chat_history=chat_history,
            max_depth=max_depth,
        )
        _emit_legacy_trace(
            trace_sink,
            "query_variants",
            {
                "question": question,
                "input_count": 1,
                "output_count": len(variants),
                "variants": variants,
            },
        )

        retriever = self.get_combined_retriever(intent_ranking or [], trace_sink=trace_sink, trace_level=trace_level)
        responses = {}

        for variant in variants:
            docs = retriever.invoke(variant["query"])
            if docs is None:
                logger.warning(
                    "Retriever returned None (query=%r, intents=%s, variants=%s)",
                    variant["query"],
                    intent_ranking,
                    [v.get("id") for v in variants],
                )
                docs = []
            keyword_filter_details = None
            docs_for_context = docs
            if self.keyword_filter:
                base_query = keyword_filter_query if keyword_filter_query is not None else variant["query"]
                keyword_text = self._build_keyword_text(base_query, chat_history)
                keyword_filter_details = self._get_keyword_filter_details(keyword_text, docs or [])
                docs_for_context = self._filter_docs_by_query(
                    keyword_text,
                    docs or [],
                    allow_empty=True,
                )
                _emit_legacy_trace(
                    trace_sink,
                    "keyword_filter",
                    {
                        "variant_id": variant["id"],
                        "query": keyword_text,
                        "enabled": True,
                        "input_count": len(docs or []),
                        "output_count": len(docs_for_context or []),
                        "details": keyword_filter_details,
                    },
                )
            else:
                _emit_legacy_trace(
                    trace_sink,
                    "keyword_filter",
                    {
                        "variant_id": variant["id"],
                        "enabled": False,
                        "input_count": len(docs or []),
                        "output_count": len(docs or []),
                    },
                )
            if docs_for_context is None:
                logger.warning(
                    "docs_for_context None after filtering (query=%r, intents=%s)",
                    variant["query"],
                    intent_ranking,
                )
                docs_for_context = []
            before_minimum_length = docs_for_context
            docs_for_context = [
                doc for doc in docs_for_context
                if len((doc.page_content or "").strip()) >= 200
            ]
            removed_short = [
                _trace_document(doc, score=doc.metadata.get("score"))
                for doc in before_minimum_length
                if len((doc.page_content or "").strip()) < 200
            ]
            _emit_legacy_trace(
                trace_sink,
                "minimum_length_filter",
                {
                    "variant_id": variant["id"],
                    "minimum_chars": 200,
                    "input_count": len(before_minimum_length),
                    "output_count": len(docs_for_context),
                    "removed_count": len(removed_short),
                    "removed_sample": removed_short[:10],
                },
            )
            documents = [
                {
                    "intent": doc.metadata.get("source").split("/")[-2],
                    "score": doc.metadata.get("score"),
                    "source": doc.metadata.get("source"),
                    "content": doc.page_content.strip()
                }
                for doc in docs_for_context
            ]
            response_payload = {
                "query": variant["query"],
                "context": format_docs(docs_for_context) if docs_for_context else "",
                "documents": documents
            }
            if keyword_filter_details is not None:
                response_payload["keyword_filter"] = keyword_filter_details
            responses[variant["id"]] = response_payload

            _emit_legacy_trace(
                trace_sink,
                "variant_result",
                {
                    "variant_id": variant["id"],
                    "input_count": len(docs or []),
                    "output_count": len(documents),
                    "documents": documents,
                },
            )

        return responses

    def _retry_empty_context(self, question, intent_ranking, multi_context, trace_sink=None):
        """Capture the intent-filtered retry used by the multi-topic prompt builder."""
        if self._combine_context_sections(multi_context):
            return multi_context
        docs = self.get_combined_retriever(intent_ranking).invoke(question)
        documents = [
            dict(_trace_document(doc, score=doc.metadata.get("score"), rank=rank),
                 query_variant="empty_context_retry")
            for rank, doc in enumerate(docs, start=1)
        ]
        _emit_legacy_trace(trace_sink, "empty_context_retry", {
            "input_count": 0,
            "output_count": len(documents),
            "documents": documents,
            "candidate_prefix_complete": True,
        })
        return dict(multi_context, empty_context_retry={
            "query": question,
            "context": format_docs(docs),
            "documents": documents,
        })

    @staticmethod
    def _select_context_documents(
            multi_context: dict,
            max_docs: int = 15,
            trace_sink: LegacyTraceSink = None,
    ) -> List[dict]:
        """Apply the deployed cross-variant deduplication and score ordering."""
        if not multi_context:
            return []
        if "empty_context_retry" in multi_context:
            # The prompt-time retry keeps the retriever's order and duplicates.
            return multi_context["empty_context_retry"]["documents"][:max_docs]

        all_docs = []
        for variant_id, payload in multi_context.items():
            for d in payload.get("documents", []):
                content = (d.get("content") or "").strip()
                if not content:
                    continue
                enriched = dict(d)
                enriched["query_variant"] = variant_id
                all_docs.append(enriched)

        if not all_docs:
            return []

        # Deduplizieren (source + content)
        seen = set()
        unique_docs = []
        for d in all_docs:
            key = (d.get("source"), d.get("content"))
            if key in seen:
                continue
            seen.add(key)
            unique_docs.append(d)

        # Nach Score sortieren (niedriger = besser, sonst großer Default)
        def score_of(d):
            s = d.get("score")
            return s if isinstance(s, (int, float)) else 1e9

        unique_docs.sort(key=score_of)
        top_docs = unique_docs[:max_docs]

        _emit_legacy_trace(
            trace_sink,
            "result_assembly",
            {
                "input_count": len(all_docs),
                "unique_count": len(unique_docs),
                "output_count": len(top_docs),
                "max_docs": max_docs,
                "documents": top_docs,
                "duplicate_count": len(all_docs) - len(unique_docs),
            },
        )

        logger.debug(
            "combine_context: variants=%s all_docs=%d unique_docs=%d",
            list(multi_context.keys()),
            len(all_docs),
            len(unique_docs),
        )

        return top_docs

    @staticmethod
    def _combine_context_sections(multi_context: dict, max_docs: int = 15, trace_sink: LegacyTraceSink = None) -> str:
        """Kombiniere alle Varianten, dedupliziere Chunks und nimm die besten."""
        top_docs = ChatQA._select_context_documents(multi_context, max_docs=max_docs, trace_sink=trace_sink)
        if not top_docs:
            # Fallback: alte Logik, falls documents nicht befüllt sind
            for payload in (multi_context or {}).values():
                ctx = (payload.get("context") or "").strip()
                if ctx:
                    return ctx
            return ""

        return "\n\n".join(d["content"].strip() for d in top_docs)

    def create_chat_chain_with_messages(self, intent_ranking, context=None, topic_titles=None):
        logger.debug("create_chat_chain_with_messages")
        llm = self.llm or ChatAILLM()

        def get_context(inputs):
            logger.debug("get_context")
            if context:
                # liefert reinen Kontext-String
                return context

            retriever = self.get_combined_retriever(intent_ranking)
            context_str = format_docs(retriever.invoke(inputs["question"]))
            buttons_str = ", ".join(inputs.get("buttons", []))

            parts = []
            if context_str:
                parts.append(context_str)
            if buttons_str:
                parts.append(buttons_str)

            return "\n\n".join(parts)

        def build_messages(inputs):
            logger.debug("build_messages")
            history = convert_chat_history(inputs.get("chat_history", []))
            question = inputs["question"]
            context = inputs["context"]  # reiner Kontext
            system_instruction = inputs.get("instructions")
            reminder = inputs.get("reminder")
            topic_titles_input = inputs.get("topic_titles") or topic_titles or []

            messages: List[BaseMessage] = []

            if system_instruction:
                messages.append(SystemMessage(content=system_instruction))

            # Alte History
            messages.extend(history)

            # Kontextblock bauen (mit Labels)
            context_parts = []
            if context:
                context_parts.append(f"{text['context'][self.language]}:\n{context}")
            if topic_titles_input:
                pages_list = ", ".join(topic_titles_input)
                if pages_list:
                    context_parts.append(f"{text['relevant_pages'][self.language]}: {pages_list}")

            if context_parts:
                messages.append(HumanMessage(content="\n\n".join(context_parts)))
            if reminder:
                messages.append(SystemMessage(content=reminder))

            # aktuelle Frage
            messages.append(HumanMessage(content=f"{text['question'][self.language]}: {question}"))

            return messages

        return (
                RunnableMap({
                    "context": RunnableLambda(get_context),
                    "question": lambda x: x["question"],
                    "chat_history": lambda x: x.get("chat_history", []),
                    "buttons": lambda x: x.get("buttons", []),
                    "instructions": lambda x: x.get("instructions"),
                    "reminder": lambda x: x.get("reminder"),
                    "topic_titles": lambda x: x.get("topic_titles"),
                })
                | RunnableLambda(build_messages)
                | llm.bind()
        )

    def ask_with_direct_context(
            self,
            question: str,
            context: Optional[str] = None,
            intent: Optional[str] = None,
            chat_history: List[dict] = None,
            intent_ranking: List[str] = None,
            max_history_depth: int = 3,
            topic_titles: Optional[List[str]] = None,
            retrieval_only: bool = False,
    ):
        logger.debug("ask_with_direct_context")
        chat_history = chat_history or []

        legal_intents = list(intent_ranking) if intent_ranking else []
        if intent:
            if intent not in legal_intents:
                legal_intents.append(intent)

        multi_context = {}
        combined_context = ""
        if legal_intents:
            multi_context = self.get_multi_context(
                question=question,
                intent_ranking=legal_intents,
                chat_history=chat_history,
                max_depth=max_history_depth,
            )
            combined_context = self._combine_context_sections(multi_context)

        if context:
            final_context = context
            if combined_context:
                final_context = f"{context}\n\n{combined_context}"
        else:
            if combined_context:
                final_context = combined_context
            else:
                final_context = ""
                if intent:
                    logger.warning("No document found for intent '%s'", intent)

            # Add relevant pages if available
        if topic_titles:
            pages_list = ", ".join(topic_titles)
            if pages_list:
                final_context = f"{text['relevant_pages'][self.language]}: {pages_list}\n\n{final_context}"

        results = multi_context if multi_context else {}

        if retrieval_only:
            return "", results

        llm = self.llm or ChatAILLM()

        def build_messages(inputs):
            history = convert_chat_history(inputs.get("chat_history", []))
            messages = [
                SystemMessage(content=text["instructions"][self.language]),
                *history,
                HumanMessage(content=f"{text['context'][self.language]}:\n{inputs['context']}"),
                HumanMessage(content=text["reminder"][self.language]),
                HumanMessage(content=f"{text['question'][self.language]}: {inputs['question']}")
            ]
            return messages

        chain = (
                RunnableLambda(build_messages)
                | llm.bind()
        )

        result = chain.invoke({
            "question": question,
            "context": final_context,
            "chat_history": chat_history,
        })

        # result is an AIMessage
        answer_text = result.content
        if not (final_context or "").strip() and not (topic_titles or []):
            answer_text = self._append_contact_form_fallback(answer_text)
        logger.debug(f"\nAnswer:\n{answer_text}")
        return answer_text, results

    def ask(
            self,
            question: str,
            intent_ranking,
            chat_history: Optional[List[dict]] = None,
            max_history_depth: int = 1,
            topic_titles: Optional[List[str]] = None,
            retrieval_only: bool = False,
    ):
        logger.debug("ask")
        chat_history = chat_history or []
        multi_context = self.get_multi_context(
            question=question,
            intent_ranking=intent_ranking,
            chat_history=chat_history,
            max_depth=max_history_depth,
        )

        combined_context = self._combine_context_sections(multi_context)

        if retrieval_only:
            return "", self._retry_empty_context(question, intent_ranking, multi_context)

        chain = self.create_chat_chain_with_messages(
            intent_ranking,
            context=combined_context,
            topic_titles=topic_titles
        )
        result = chain.invoke({
            "question": question,
            "chat_history": chat_history,
            "instructions": text["instructions"][self.language],
            "reminder": text["reminder"][self.language],
            "topic_titles": topic_titles
        })

        answer_text = result.content
        if not (combined_context or "").strip() and not (topic_titles or []):
            answer_text = self._append_contact_form_fallback(answer_text)
        print("\nAnswer:\n", answer_text)
        return answer_text, multi_context

    def ask_legacy(self, question: str, intent_ranking, chat_history: Optional[List[dict]] = None):
        """
        Original ask implementation that retrieves context only for the current question.
        """
        logger.debug("ask_legacy")
        chain = self.create_chat_chain_with_messages(intent_ranking)
        result = chain.invoke({
            "question": question,
            "chat_history": chat_history or [],
            "instructions": text["instructions"][self.language],
            "reminder": text["reminder"][self.language],
        })

        # Manually collect sources via the retriever
        retriever = self.get_combined_retriever(intent_ranking)
        docs = retriever.invoke(question)
        results = self.get_results_from_docs(docs)

        answer_text = result.content
        if not docs:
            answer_text = self._append_contact_form_fallback(answer_text)
        print("\nAnswer:\n", answer_text)
        return answer_text, results

    def get_results_from_docs(self, docs: List[Document]):
        logger.debug("get_results_from_docs")
        results = {}
        if not docs:
            logger.warning("No source documents were returned.")
            return results

        for i, doc in enumerate(docs, 1):
            intent = Path(doc.metadata.get("source", "")).parent.name
            if intent in results:
                continue
            results[intent] = {
                "rank": i,
                "score": doc.metadata.get("score"),
                "content": doc.page_content.strip(),
            }
        return results


# === Example Usage ===
if __name__ == "__main__":
    qa = ChatQA(language="deutsch", recreate_index=True)  # set True once after changing preprocessing
    history: List[dict] = []
    default_intents: List[str] = []  # empty = allow all intents (no filter)

    while True:
        try:
            query = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not query:
            continue

        answer, sources = qa.ask(
            question=query,
            intent_ranking=default_intents,
            chat_history=history
        )

        print("\n--- Answer ---\n", answer)
        print("\n--- Sources ---")
        for k, v in sources.items():
            score = v["score"]
            score_str = f"{score:.4f}" if isinstance(score, (int, float)) else str(score)
            print(f"[{v['rank']}] {k}  (score={score_str})")

        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": answer})
