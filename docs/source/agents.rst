agents
======

MaxAgentQA
**********
A QA Agent for processing and analyzing text data using large language models (LLM).

Args:
    - llm_provider (str, optional): The name of the large language provider you want to use. We support Anthropic, AzureOpenAI, openAI etc.
    - model_name (str, optional): The name of the llm model for the given provider.
    - model_kwargs (dict, optional): Optional keyword arguments for the llm model, defaults to None.
    - chunk_size (int, optional): The size of text chunks for processing, defaults to 2000.
    - chunk_overlap (int, optional): The overlap size between consecutive text chunks, defaults to 200.
    - stream (bool, optional): Flag to indicate if the data should be processed as a stream, defaults to False.
    - collection (str, optional): The name of the collection to be used in the vector database, defaults to None.
    - prompt_config (dict, optional): Configuration settings for prompts, defaults to None.
    - metadata_dict (dict, optional): A dictionary to specify which metadata features to include, defaults to a predefined set.
    - embedding_model (str, optional): The name of the embedding model to be used, defaults to "sentence-transformers/all-mpnet-base-v2".
    - retriever_rank (str, optional): The ranking method to be used by the retriever, defaults to "LostInMiddle".
    - generate_method (str, optional): The method used for generation tasks, defaults to "stuff".
    - verbose (bool, optional): Flag to indicate if verbose mode is enabled, defaults to "True".
    - vector_store (str, optional): The type of vector store to be used, defaults to "pgvector".

Raises:
    - ValueError: If `prompt_config` is not provided.
    - ValueError: If `llm_provider` is not provided.

Returns:
    - MaxAgentQA: An instance of MaxAgentQA.

>>> from maxaillm.app.agent import MaxAgentQA
>>> agent = MaxAgentQA(llm_provider="anthropic",model_name ="claude-2", chunk_size=1000, stream=True, collection="myCollection", prompt_config=myPromptConfig)


Methods
^^^^^^^

initialize_llm
--------------
Initializes a large language model (LLM) based on the provided provider, model name, and model arguments.

Args:
    - ``provider (str)``: The name of the large language provider. Supported providers include 'anthropic', 'openai', 'azureopenai', 'azure', 'bedrock', and 'aws'.
    - ``model_name (str)``: The name of the LLM model for the given provider. If not provided, a default model is used based on the provider.
    - ``model_kwargs (dict)``: A dictionary of keyword arguments for the LLM model. Expected keys are 'temperature' and 'top_p'.

Raises:
    - ``ValueError``: If the provided provider is not recognized.

Returns:
    -  An instance of the initialized LLM model.
    

set_collection
--------------
Sets the collection name and initializes the vector database.

Args:
    - ``collection (str)``: The name of the collection to be used in the vector database.

Returns:
    - None
    
get_collection
--------------
Gets the collection name.

Args:
    - ``collection (str)``: The name of the collection to be retrieved.

Returns:
    - ``str``: The name of the collection.
    
init_vector_db
--------------
Initializes the vector database based on the specified vector store.

Args:
    - None

Raises:
    - None

Returns:
    - None

process_file
------------
Processes a file by extracting text, cleaning it, splitting it into chunks, and adding the chunks to the vector database.

Args:
    - ``file (str)``: The file to be processed.
    - ``doc_metadata (dict)``: Additional metadata for the document.

Raises:
    - ``Exception``: If an error occurs during the processing.

Returns:
    - ``list``: The list of documents added to the vector database.
    
    
Adds documents to a specified collection from given files.

Args:
    - ``files (List[str])``: A list of file paths to be processed and added to the collection.
    - ``default_metadata (List[Dict], optional)``: A list of metadata dictionaries corresponding to each file. Defaults to an empty list.

Raises:
    - ``Exception``: If the collection is not set before adding documents.
    - ``ValueError``: If the files argument is not a list of strings or a single string.

Returns:
    - ``bool``: True if the operation is successful.
    
query
------
Queries the collection and generates a response based on the given query.

Args:
    - ``query (str, optional)``: The query to be processed. Defaults to an empty string.
    - ``search_type (str, optional)``: The type of search to be performed. Defaults to "mmr".
    - ``k (int, optional)``: The number of top results to return. Defaults to 10.
    - ``filters (dict, optional)``: Filters to apply during the search. Defaults to an empty dictionary.
    - ``score_threshold (float, optional)``: The minimum score threshold for the results. Defaults to 0.05.
    - ``prompt_config (optional)``: Configuration for the prompt. If not provided, the instance's prompt configuration is used.

Raises:
    - ``Exception``: If the collection is not set before querying documents.

Returns:
    - The generated response based on the query.