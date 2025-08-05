from __future__ import annotations

import queue
from typing import Any, Dict, Iterator, List, Optional, Sequence

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import (
    ChatGeneration,
    ChatGenerationChunk,
    ChatResult,
    LLMResult,
)

from pathlib import Path

import numpy as np
from langchain_core.callbacks import Callbacks
from langchain_core.documents import Document
from langchain_core.documents.compressor import BaseDocumentCompressor
from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, ConfigDict, Field
from genai_helper import ChunkStreamer

DEFAULT_QUERY_INSTRUCTION = "Represent the question for retrieving supporting documents: "
DEFAULT_QUERY_BGE_INSTRUCTION_EN = "Represent this question for searching relevant passages: "
DEFAULT_QUERY_BGE_INSTRUCTION_ZH = "为这个句子生成表示以用于检索相关文章："

DEFAULT_SYSTEM_PROMPT = """You are a helpful, respectful, and honest assistant."""


class OpenVINOLLM(LLM):
    """OpenVINO Pipeline API.

    To use, you should have the ``openvino-genai`` python package installed.

    Example using from_model_path:
        .. code-block:: python

            from langchain_community.llms import OpenVINOLLM
            ov = OpenVINOPipeline.from_model_path(
                model_path="./openvino_model_dir",
                device="CPU",
            )
    Example passing pipeline in directly:
        .. code-block:: python

            import openvino_genai
            pipe = openvino_genai.LLMPipeline("./openvino_model_dir", "CPU")
            config = openvino_genai.GenerationConfig()
            ov = OpenVINOPipeline.from_model_path(
                ov_pipe=pipe,
                config=config,
            )

    """

    ov_pipe: Any = None
    tokenizer: Any = None
    config: Any = None
    streamer: Any = None

    @classmethod
    def from_model_path(
        cls,
        model_path: str,
        device: str = "CPU",
        tokenizer: Any = None,
        **kwargs: Any,
    ) -> OpenVINOLLM:
        """Construct the oepnvino object from model_path"""
        try:
            import openvino_genai

        except ImportError:
            raise ImportError("Could not import OpenVINO GenAI package. " "Please install it with `pip install openvino-genai`.")

        ov_pipe = openvino_genai.LLMPipeline(model_path, device, **kwargs)

        config = ov_pipe.get_generation_config()
        if tokenizer is None:
            tokenizer = ov_pipe.get_tokenizer()
        streamer = ChunkStreamer(tokenizer)

        return cls(
            ov_pipe=ov_pipe,
            tokenizer=tokenizer,
            config=config,
            streamer=streamer,
        )

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call out to OpenVINO's generate request."""
        if stop is not None:
            self.config.stop_strings = set(stop)
        try:
            import openvino as ov
            import openvino_genai

        except ImportError:
            raise ImportError("Could not import OpenVINO GenAI package. " "Please install it with `pip install openvino-genai`.")
        if not isinstance(self.tokenizer, openvino_genai.Tokenizer):
            tokens = self.tokenizer(prompt, add_special_tokens=False, return_tensors="np")
            input_ids = tokens["input_ids"]
            attention_mask = tokens["attention_mask"]
            prompt = openvino_genai.TokenizedInputs(ov.Tensor(input_ids), ov.Tensor(attention_mask))
        output = self.ov_pipe.generate(prompt, self.config, **kwargs)
        if not isinstance(self.tokenizer, openvino_genai.Tokenizer):
            output = self.tokenizer.batch_decode(output.tokens, skip_special_tokens=True)[0]
        return output

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        """Output OpenVINO's generation Stream"""
        from threading import Event, Thread
        import time

        if stop is not None:
            self.config.stop_strings = set(stop)
        try:
            import openvino as ov
            import openvino_genai

        except ImportError:
            raise ImportError("Could not import OpenVINO GenAI package. " "Please install it with `pip install openvino-genai`.")
        if not isinstance(self.tokenizer, openvino_genai.Tokenizer):
            tokens = self.tokenizer(prompt, add_special_tokens=False, return_tensors="np")
            input_ids = tokens["input_ids"]
            attention_mask = tokens["attention_mask"]
            prompt = openvino_genai.TokenizedInputs(ov.Tensor(input_ids), ov.Tensor(attention_mask))
        stream_complete = Event()
        generation_error = None

        def generate_and_signal_complete() -> None:
            """
            Generation function for single thread with error handling
            """
            nonlocal generation_error
            try:
                self.streamer.reset()
                self.ov_pipe.generate(prompt, self.config, self.streamer, **kwargs)
                stream_complete.set()
                self.streamer.end()
            except Exception as e:
                generation_error = e
                stream_complete.set()
                self.streamer.end()

        t1 = Thread(target=generate_and_signal_complete)
        t1.start()

        # Wait for generation to start or fail
        start_time = time.time()
        timeout = 30  # 30 second timeout
        
        try:
            for char in self.streamer:
                # Check if generation thread encountered an error
                if generation_error is not None:
                    # Check if it's a GPU memory error
                    if "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST" in str(generation_error) or \
                       "OpenCL" in str(generation_error) or \
                       "GPU" in str(generation_error):
                        raise RuntimeError(
                            f"GPU memory error detected: {generation_error}\n"
                            # "Suggested solutions:\n"
                            # "  - Switch to CPU device\n"
                            # "  - Use a smaller model (1.5B instead of 3B+)\n"
                            # "  - Reduce max_new_tokens to 256 or less\n"
                            # "  - Close other applications using GPU\n"
                            # "  - Restart the application"
                        )
                    else:
                        raise generation_error
                
                # Check timeout
                if time.time() - start_time > timeout:
                    raise TimeoutError("Generation timed out after 30 seconds")
                
                chunk = GenerationChunk(text=char)
                if run_manager:
                    run_manager.on_llm_new_token(chunk.text, chunk=chunk)

                yield chunk
                
        except Exception as e:
            # Clean up the thread
            if t1.is_alive():
                stream_complete.set()
                t1.join(timeout=5)
            raise e
        
        # Wait for thread to complete
        t1.join()

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters."""
        return {}

    @property
    def _llm_type(self) -> str:
        return "openvino_pipeline"


DEFAULT_SYSTEM_PROMPT = """You are a helpful, respectful, and honest assistant."""


class ChatOpenVINO(BaseChatModel):
    """OpenVINO LLM's as ChatModels.

    Works with `OpenVINOLLM` LLMs.

    See full list of supported init args and their descriptions in the params
    section.

    Instantiate:
        .. code-block:: python

            from langchain_community.llms import OpenVINOLLM
            llm = OpenVINOPipeline.from_model_path(
                model_path="./openvino_model_dir",
                device="CPU",
            )

            chat = ChatOpenVINO(llm=llm, verbose=True)

    Invoke:
        .. code-block:: python

            messages = [
                ("system", "You are a helpful translator. Translate the user
                sentence to French."),
                ("human", "I love programming."),
            ]

            chat(...).invoke(messages)

        .. code-block:: python


    Stream:
        .. code-block:: python

            for chunk in chat.stream(messages):
                print(chunk)

        .. code-block:: python


    """  # noqa: E501

    llm: Any
    """LLM, must be of type OpenVINOLLM"""
    system_message: SystemMessage = SystemMessage(content=DEFAULT_SYSTEM_PROMPT)
    tokenizer: Any = None

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

        if self.tokenizer is None:
            self.tokenizer = self.llm.tokenizer

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        llm_input = self._to_chat_prompt(messages)
        llm_result = self.llm._generate(prompts=[llm_input], stop=stop, run_manager=run_manager, **kwargs)
        return self._to_chat_result(llm_result)

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        request = self._to_chat_prompt(messages)

        for data in self.llm.stream(request, **kwargs):
            delta = data
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=delta))
            if run_manager:
                run_manager.on_llm_new_token(delta, chunk=chunk)
            yield chunk

    def _to_chat_prompt(
        self,
        messages: List[BaseMessage],
    ) -> str:
        """Convert a list of messages into a prompt format expected by wrapped LLM."""
        try:
            import openvino_genai

        except ImportError:
            raise ImportError("Could not import OpenVINO GenAI package. " "Please install it with `pip install openvino-genai`.")
        if not messages:
            raise ValueError("At least one HumanMessage must be provided!")

        if not isinstance(messages[-1], HumanMessage):
            raise ValueError("Last message must be a HumanMessage!")

        messages_dicts = [self._to_chatml_format(m) for m in messages]

        return (
            self.tokenizer.apply_chat_template(messages_dicts, add_generation_prompt=True)
            if isinstance(self.tokenizer, openvino_genai.Tokenizer)
            else self.tokenizer.apply_chat_template(messages_dicts, tokenize=False, add_generation_prompt=True)
        )

    def _to_chatml_format(self, message: BaseMessage) -> dict:
        """Convert LangChain message to ChatML format."""

        if isinstance(message, SystemMessage):
            role = "system"
        elif isinstance(message, AIMessage):
            role = "assistant"
        elif isinstance(message, HumanMessage):
            role = "user"
        else:
            raise ValueError(f"Unknown message type: {type(message)}")

        return {"role": role, "content": message.content}

    @staticmethod
    def _to_chat_result(llm_result: LLMResult) -> ChatResult:
        chat_generations = []

        for g in llm_result.generations[0]:
            chat_generation = ChatGeneration(message=AIMessage(content=g.text), generation_info=g.generation_info)
            chat_generations.append(chat_generation)

        return ChatResult(generations=chat_generations, llm_output=llm_result.llm_output)

    @property
    def _llm_type(self) -> str:
        return "openvino-chat-wrapper"


class OpenVINOEmbeddings(BaseModel, Embeddings):
    """OpenVINO embedding models.

    Example:
        .. code-block:: python

            from langchain_community.embeddings import OpenVINOEmbeddings

            model_name = "sentence-transformers/all-mpnet-base-v2"
            model_kwargs = {'device': 'CPU'}
            encode_kwargs = {'normalize_embeddings': True}
            ov = OpenVINOEmbeddings(
                model_path=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )
    """

    ov_model: Any = None
    """OpenVINO model object."""
    tokenizer: Any = None
    """Tokenizer for embedding model."""
    model_path: str
    """Local model path."""
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Keyword arguments to pass to the model."""
    encode_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Keyword arguments to pass when calling the `encode` method of the model."""
    show_progress: bool = False
    """Whether to show a progress bar."""

    def __init__(self, **kwargs: Any):
        """Initialize the sentence_transformer."""
        super().__init__(**kwargs)

        try:
            import openvino as ov
        except ImportError as e:
            raise ImportError("Could not import openvino python package. " "Please install it with: " "pip install -U 'openvino") from e

        try:
            import openvino_genai
        except ImportError as e:
            raise ImportError("Could not import openvino_genai python package. " "Please install it with: " "pip install -U openvino_genai") from e

        if self.ov_model is None:
            core = ov.Core()
            self.ov_model = core.compile_model(Path(self.model_path) / "openvino_model.xml", **self.model_kwargs)
        self.tokenizer = openvino_genai.Tokenizer(self.model_path)

    def _text_length(self, text: Any) -> int:
        """
        Help function to get the length for the input text. Text can be either
        a list of ints (which means a single text as input), or a tuple of list of ints
        (representing several text inputs to the model).
        """

        if isinstance(text, dict):  # {key: value} case
            return len(next(iter(text.values())))
        elif not hasattr(text, "__len__"):  # Object has no len() method
            return 1
        # Empty string or list of ints
        elif len(text) == 0 or isinstance(text[0], int):
            return len(text)
        else:
            # Sum of length of individual strings
            return sum([len(t) for t in text])

    def encode(
        self,
        sentences: Any,
        batch_size: int = 4,
        show_progress_bar: bool = False,
        mean_pooling: bool = False,
        normalize_embeddings: bool = True,
    ) -> Any:
        """
        Computes sentence embeddings.

        :param sentences: the sentences to embed.
        :param batch_size: the batch size used for the computation.
        :param show_progress_bar: Whether to output a progress bar.
        :param convert_to_numpy: Whether the output should be a list of numpy vectors.
        :param convert_to_tensor: Whether the output should be one large tensor.
        :param mean_pooling: Whether to pool returned vectors.
        :param normalize_embeddings: Whether to normalize returned vectors.

        :return: By default, a 2d numpy array with shape [num_inputs, output_dimension].
        """
        try:
            import numpy as np
        except ImportError as e:
            raise ImportError("Unable to import numpy, please install with `pip install -U numpy`.") from e
        try:
            from tqdm import trange
        except ImportError as e:
            raise ImportError("Unable to import tqdm, please install with `pip install -U tqdm`.") from e

        def run_mean_pooling(model_output: Any, attention_mask: Any) -> Any:
            token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
            input_mask_expanded = np.broadcast_to(np.expand_dims(attention_mask, axis=-1), token_embeddings.size())
            return np.sum(token_embeddings * input_mask_expanded, 1) / np.clip(input_mask_expanded.sum(1), a_min=1e-9)

        input_was_string = False
        if isinstance(sentences, str) or not hasattr(sentences, "__len__"):  # Cast an individual sentence to a list with length 1
            sentences = [sentences]
            input_was_string = True

        all_embeddings: Any = []
        length_sorted_idx = np.argsort([-self._text_length(sen) for sen in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

        for start_index in trange(0, len(sentences), batch_size, desc="Batches", disable=not show_progress_bar):
            sentences_batch = sentences_sorted[start_index : start_index + batch_size]

            length = self.ov_model.inputs[0].get_partial_shape()[1]
            if length.is_dynamic:
                features = self.tokenizer.encode(sentences_batch)
            else:
                features = self.tokenizer.encode(
                    sentences_batch,
                    pad_to_max_length=True,
                    max_length=length.get_length(),
                )
            if "token_type_ids" in (input.any_name for input in self.ov_model.inputs):
                token_type_ids = np.zeros(features.attention_mask.shape)
                model_input = {
                    "input_ids": features.input_ids,
                    "attention_mask": features.attention_mask,
                    "token_type_ids": token_type_ids,
                }
            else:
                model_input = {
                    "input_ids": features.input_ids,
                    "attention_mask": features.attention_mask,
                }
            out_features = self.ov_model(model_input)
            if mean_pooling:
                embeddings = run_mean_pooling(out_features, features["attention_mask"])
            else:
                embeddings = out_features[0][:, 0]
            if normalize_embeddings:
                norm = np.linalg.norm(embeddings, ord=2, axis=1, keepdims=True)
                embeddings = embeddings / norm
            all_embeddings.extend(embeddings)

        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]

        if input_was_string:
            all_embeddings = all_embeddings[0]

        return all_embeddings

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute doc embeddings using a HuggingFace transformer model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """

        texts = list(map(lambda x: x.replace("\n", " "), texts))
        embeddings = self.encode(texts, show_progress_bar=self.show_progress, **self.encode_kwargs)

        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Compute query embeddings using a HuggingFace transformer model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        return self.embed_documents([text])[0]


class OpenVINOBgeEmbeddings(OpenVINOEmbeddings):
    """OpenVNO BGE embedding models.

    Bge Example:
        .. code-block:: python

            from langchain_community.embeddings import OpenVINOBgeEmbeddings

            model_name = "BAAI/bge-large-en-v1.5"
            model_kwargs = {'device': 'CPU'}
            encode_kwargs = {'normalize_embeddings': True}
            ov = OpenVINOBgeEmbeddings(
                model_path=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )
    """

    query_instruction: str = DEFAULT_QUERY_BGE_INSTRUCTION_EN
    """Instruction to use for embedding query."""
    embed_instruction: str = ""
    """Instruction to use for embedding document."""

    def __init__(self, **kwargs: Any):
        """Initialize the sentence_transformer."""
        super().__init__(**kwargs)

        if "-zh" in self.model_path:
            self.query_instruction = DEFAULT_QUERY_BGE_INSTRUCTION_ZH

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute doc embeddings using a HuggingFace transformer model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        texts = [self.embed_instruction + t.replace("\n", " ") for t in texts]
        embeddings = self.encode(texts, **self.encode_kwargs)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Compute query embeddings using a HuggingFace transformer model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        text = text.replace("\n", " ")
        embedding = self.encode(self.query_instruction + text, **self.encode_kwargs)
        return embedding


class RerankRequest:
    """Request for reranking."""

    def __init__(self, query: Any = None, passages: Any = None):
        self.query = query
        self.passages = passages if passages is not None else []


class OpenVINOReranker(BaseDocumentCompressor):
    """
    OpenVINO rerank models.
    """

    ov_model: Any = None
    """OpenVINO model object."""
    tokenizer: Any = None
    """Tokenizer for embedding model."""
    model_path: str
    """Local model path."""
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Keyword arguments passed to the model."""
    top_n: int = 4
    """return Top n texts."""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

        try:
            import openvino as ov
        except ImportError as e:
            raise ImportError("Could not import openvino python package. " "Please install it with: " "pip install -U 'openvino") from e

        try:
            import openvino_genai
        except ImportError as e:
            raise ImportError("Could not import openvino_genai python package. " "Please install it with: " "pip install -U openvino_genai") from e
        if self.ov_model is None:
            core = ov.Core()
            self.ov_model = core.compile_model(Path(self.model_path) / "openvino_model.xml", **self.model_kwargs)
        self.tokenizer = openvino_genai.Tokenizer(self.model_path)

    def rerank(self, request: Any) -> Any:
        query = request.query
        passages = request.passages
        # # openvino tokenizer can only support 1D list
        query_passage_pairs = [query + "</s></s> " + passage["text"] for passage in passages]
        # query_passage_pairs = [[query, passage["text"]] for passage in passages]
        length = self.ov_model.inputs[0].get_partial_shape()[1]
        if length.is_dynamic:
            features = self.tokenizer.encode(query_passage_pairs)
        else:
            features = self.tokenizer.encode(
                query_passage_pairs,
                pad_to_max_length=True,
                max_length=length.get_length(),
            )
        model_input = {
            "input_ids": features.input_ids,
            "attention_mask": features.attention_mask,
        }
        outputs = self.ov_model(model_input)
        if outputs[0].shape[1] > 1:
            scores = outputs[0][:, 1]
        else:
            scores = outputs[0].flatten()

        scores = list(1 / (1 + np.exp(-scores)))

        # Combine scores with passages, including metadata
        for score, passage in zip(scores, passages):
            passage["score"] = score

        # Sort passages based on scores
        passages.sort(key=lambda x: x["score"], reverse=True)

        return passages

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        passages = [{"id": i, "text": doc.page_content} for i, doc in enumerate(documents)]

        rerank_request = RerankRequest(query=query, passages=passages)
        rerank_response = self.rerank(rerank_request)[: self.top_n]
        final_results = []
        for r in rerank_response:
            doc = Document(
                page_content=r["text"],
                metadata={"id": r["id"], "relevance_score": r["score"]},
            )
            final_results.append(doc)
        return final_results


class OpenVINOTextEmbeddings(BaseModel, Embeddings):
    """OpenVINO GenAI Text Embedding Pipeline implementation.
    
    This class uses the native openvino_genai.TextEmbeddingPipeline from the 
    OpenVINO GenAI RAG samples: https://github.com/openvinotoolkit/openvino.genai/tree/master/samples/python/rag
    
    Based on the official example:
    import openvino_genai
    pipeline = openvino_genai.TextEmbeddingPipeline(model_dir, "CPU")
    embeddings = pipeline.embed_documents(["document1", "document2"])

    Example:
        .. code-block:: python

            from ov_langchain_helper import OpenVINOTextEmbeddings

            embeddings = OpenVINOTextEmbeddings(
                model_path="BAAI/bge-small-en-v1.5",
                device="CPU"
            )
    """

    model_path: str
    """Path to the OpenVINO model directory."""
    device: str = "CPU"
    """Device to run the model on (CPU, GPU, AUTO)."""
    batch_size: int = 32
    """Batch size for processing multiple texts."""
    show_progress: bool = False
    """Whether to show progress bar during encoding."""
    
    # Internal attributes
    _pipeline: Any = None

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    def __init__(self, **kwargs: Any):
        """Initialize the OpenVINO GenAI Text Embedding Pipeline."""
        super().__init__(**kwargs)
        
        try:
            import openvino_genai
        except ImportError as e:
            raise ImportError(
                "Could not import openvino_genai python package. "
                "Please install it with: pip install -U openvino-genai"
            ) from e

        try:
            # Initialize the TextEmbeddingPipeline directly
            self._pipeline = openvino_genai.TextEmbeddingPipeline(self.model_path, self.device)
            print(f"✅ OpenVINO GenAI TextEmbeddingPipeline loaded from {self.model_path} on {self.device}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load OpenVINO GenAI TextEmbeddingPipeline: {e}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute embeddings for a list of documents.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        if not texts:
            return []

        # Clean texts (replace newlines with spaces)
        cleaned_texts = [text.replace("\n", " ") for text in texts]
        
        try:
            # Process in batches to manage memory
            all_embeddings = []
            
            for i in range(0, len(cleaned_texts), self.batch_size):
                batch_texts = cleaned_texts[i:i + self.batch_size]
                
                # Use the embed_documents method from TextEmbeddingPipeline
                batch_embeddings = self._pipeline.embed_documents(batch_texts)
                
                # Convert to list of lists if needed
                if hasattr(batch_embeddings, 'tolist'):
                    batch_embeddings = batch_embeddings.tolist()
                elif not isinstance(batch_embeddings[0], list):
                    # Handle case where embeddings are returned as flat array
                    batch_embeddings = [emb.tolist() if hasattr(emb, 'tolist') else emb for emb in batch_embeddings]
                
                all_embeddings.extend(batch_embeddings)
                
                if self.show_progress and len(cleaned_texts) > self.batch_size:
                    print(f"Processed {min(i + self.batch_size, len(cleaned_texts))}/{len(cleaned_texts)} texts")
                    
        except Exception as e:
            print(f"Error during embedding inference: {e}")
            # Return zero embeddings as fallback
            embedding_dim = 384  # Common dimension for bge-small models
            return [[0.0] * embedding_dim] * len(cleaned_texts)

        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """Compute query embeddings.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        return self.embed_documents([text])[0]
