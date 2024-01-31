agents
======

MaxAgentQA
**********
A QA Agent for processing and analyzing text data using large language models (LLM).

Args:
    - ``llm_provider (str, optional)``: The name of the large language provider you want to use. We support Anthropic, AzureOpenAI, openAI etc.
    - ``model_name (str, optional)``: The name of the llm model for the given provider.
    - ``model_kwargs (dict, optional)``: Optional keyword arguments for the llm model, defaults to None.
    - ``chunk_size (int, optional)``: The size of text chunks for processing, defaults to 2000.
    - ``chunk_overlap (int, optional)``: The overlap size between consecutive text chunks, defaults to 200.
    - ``stream (bool, optional)``: Flag to indicate if the data should be processed as a stream, defaults to False.
    - ``collection (str, optional)``: The name of the collection to be used in the vector database, defaults to None.
    - ``prompt_config (dict, optional)``: Configuration settings for prompts, defaults to None.
    - ``metadata_dict (dict, optional)``: A dictionary to specify which metadata features to include, defaults to a predefined set.
    - ``embedding_model (str, optional)``: The name of the embedding model to be used, defaults to "sentence-transformers/all-mpnet-base-v2".
    - ``retriever_rank (str, optional)``: The ranking method to be used by the retriever, defaults to "LostInMiddle".
    - ``generate_method (str, optional)``: The method used for generation tasks, defaults to "stuff".
    - ``verbose (bool, optional)``: Flag to indicate if verbose mode is enabled, defaults to "True".
    - ``vector_store (str, optional)``: The type of vector store to be used, defaults to "pgvector".
    
Attributes:
    - ``agent_type``: The type of the agent.
    - ``llm_provider``: The provider of the LLM.
    - ``model_name``: The name of the model.
    - ``model_kwargs``: Additional keyword arguments for the model.
    - ``chunk_size``: The size of the chunks to be processed.
    - ``chunk_overlap``: The overlap between chunks.
    - ``stream``: Whether to use streaming mode.
    - ``collection``: The collection to use.
    - ``embedding_model``: The embedding model to use.
    - ``metadata_dict``: Metadata configuration.
    - ``retriever_rank``: The rank of the retriever.
    - ``generate_method``: The method to use for generation.
    - ``verbose``: Whether to print verbose output.
    - ``vector_store``: The vector store to use.
    - ``cost_param``: The cost parameters.
    
Methods:
    - ``initialize_llm(provider: str, model_name: str, model_kwargs: dict) -> maxaillm.model.BaseLLM.BaseLLM``: Initializes a large language model (LLM) based on the provided provider, model name, and model arguments.
    
        - ``provider (str)``: The name of the large language provider. Supported providers include 'anthropic', 'openai', 'azureopenai', 'azure', 'bedrock', and 'aws'.
        - ``model_name (str)``: The name of the LLM model for the given provider. If not provided, a default model is used based on the provider.
        - ``model_kwargs (dict)``: A dictionary of keyword arguments for the LLM model. Expected keys are 'temperature' and 'top_p'.
    - ``set_collection(collection: str) -> None``: Sets the collection name and initializes the vector database.
    - ``get_collection(collection: str) -> str``: Gets the collection name.
    - ``init_vector_db() -> None``: Initializes the vector database based on the specified vector store.
    - ``process_file(file: str, doc_metadata: dict) -> list``: Processes a file by extracting text, cleaning it, splitting it into chunks, and adding the chunks to the vector database.
    
        - ``file (str)``: The file to be processed.
        - ``doc_metadata (dict)``: Additional metadata for the document.
    - ``add(files: List[str], default_metadata: List[Dict]) -> bool``: Adds documents to a specified collection from given files.
        - ``files (List[str])``: A list of file paths to be processed and added to the collection.
        - ``default_metadata (List[Dict], optional)``: A list of metadata dictionaries corresponding to each file. Defaults to an empty list.
    - ``query(query: str, k: int, filters: dict, score_threshold: float, prompt_config: dict) -> str``: Queries the collection and generates a response based on the given query.
        - ``query (str, optional)``: The query to be processed. Defaults to an empty string.
        - ``search_type (str, optional)``: The type of search to be performed. Defaults to "mmr".
        - ``k (int, optional)``: The number of top results to return. Defaults to 10.
        - ``filters (dict, optional)``: Filters to apply during the search. Defaults to an empty dictionary.
        - ``score_threshold (float, optional)``: The minimum score threshold for the results. Defaults to 0.05.
        - ``prompt_config (optional)``: Configuration for the prompt. If not provided, the instance's prompt configuration is used.
    - ``aquery(query: str, k: int, filters: dict, search_type: str, score_threshold: float, prompt_config, chat_session, message_id)``: Queries the collection asynchronously and generates a response based on the given query.
        - ``query (str, optional)``: The query to be processed. Defaults to an empty string.
        - ``k (int, optional)``: The number of top results to return. Defaults to 10.
        - ``filters (dict, optional)``: Filters to apply during the search. Defaults to an empty dictionary.
        - ``search_type (str, optional)``: The type of search to be performed. Defaults to "mmr".
        - ``score_threshold (float, optional)``: The minimum score threshold for the results. Defaults to 0.05.
        - ``prompt_config (optional)``: Configuration for the prompt. If not provided, the instance's prompt configuration is used.
        - ``chat_session (optional)``: The chat session to be used for the query. If provided, the chat history is used in the query.
        - ``message_id (optional)``: The ID of the message to be queried.
    - get_sources
        

Raises:
    - ``ValueError``: If `prompt_config` is not provided.
    - ``ValueError``: If `llm_provider` is not provided.

Returns:
    - An instance of MaxAgentQA.

>>> from maxaillm.app.agent import MaxAgentQA
>>> agent = MaxAgentQA(llm_provider="anthropic",model_name ="claude-2", chunk_size=1000, stream=True, collection="myCollection", prompt_config=myPromptConfig)


Methods
^^^^^^^
    
    
get_sources
-----------
Retrieves the sources based on the given query.

Args:
    - ``query (str, optional)``: The query to be processed. Defaults to an empty string.
    - ``search_type (str, optional)``: The type of search to be performed. Defaults to "mmr".
    - ``k (int, optional)``: The number of top results to return. Defaults to 10.
    - ``filters (dict, optional)``: Filters to apply during the search. Defaults to an empty dictionary.
    - ``score_threshold (float, optional)``: The minimum score threshold for the results. Defaults to 0.05.
    - ``top_k (bool, optional)``: If True, returns the top k page contents. If False, returns the sources in a JSON format. Defaults to False.
    - ``chat_session (optional)``: The chat session to be used for the query. If provided, the chat history is used in the query.

Raises:
    - ``Exception``: If any error occurs during the retrieval of sources.

Returns:
    - If ``top_k`` is True, returns a list of page contents. If ``top_k`` is False, returns the sources in a JSON format.
