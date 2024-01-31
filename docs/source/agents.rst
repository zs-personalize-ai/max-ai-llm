agents
======

MaxAgentQA
^^^^^^^^^^^^^
A QA Agent for processing and analyzing text data using large language models (LLM).

:param llm_provider: The name of the large language provider you want to use. We support Anthropic, AzureOpenAI, openAI etc.
:type llm_provider: str, optional
:param model_name: The name of the llm model for the given provider.
:type model_name: str, optional
:param model_kwargs: Optional keyword arguments for the llm model, defaults to None
:type model_kwargs: Optional[Dict], optional
:param chunk_size: The size of text chunks for processing, defaults to 2000
:type chunk_size: int, optional
:param chunk_overlap: The overlap size between consecutive text chunks, defaults to 200
:type chunk_overlap: int, optional
:param stream: Flag to indicate if the data should be processed as a stream, defaults to False
:type stream: bool, optional
:param collection: The name of the collection to be used in the vector database, defaults to None
:type collection: str, optional
:param prompt_config: Configuration settings for prompts, defaults to None
:type prompt_config: Dict[str, str], optional
:param metadata_dict: A dictionary to specify which metadata features to include, defaults to a predefined set
:type metadata_dict: Dict[str, bool], optional
:param embedding_model: The name of the embedding model to be used, defaults to "sentence-transformers/all-mpnet-base-v2"
:type embedding_model: str, optional
:param retriever_rank: The ranking method to be used by the retriever, defaults to "LostInMiddle"
:type retriever_rank: str, optional
:param generate_method: The method used for generation tasks, defaults to "stuff"
:type generate_method: str, optional
:param verbose: Flag to indicate if verbose mode is enabled, defaults to "True"
:type verbose: bool, optional
:param vector_store: The type of vector store to be used, defaults to "pgvector"
:type vector_store: str, optional

:raises ValueError: If `prompt_config` is not provided.
:raises ValueError: If `llm_provider` is not provided.

:return: An instance of MaxAgentQA
:rtype: MaxAgentQA

:Example:

>>> from maxaillm.app.agent import MaxAgentQA
>>> agent = MaxAgentQA(llm_provider="anthropic",model_name ="claude-2", chunk_size=1000, stream=True, collection="myCollection", prompt_config=myPromptConfig)