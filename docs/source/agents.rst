agents
======

MaxAgentQA
^^^^^^^^^^^^^
A QA Agent for processing and analyzing text data using large language models (LLM).

Args:
    llm_provider (str, optional): The name of the large language provider you want to use. We support Anthropic, AzureOpenAI, openAI etc.
    model_name (str, optional): The name of the llm model for the given provider.
    model_kwargs (dict, optional): Optional keyword arguments for the llm model, defaults to None.
    chunk_size (int, optional): The size of text chunks for processing, defaults to 2000.
    chunk_overlap (int, optional): The overlap size between consecutive text chunks, defaults to 200.
    stream (bool, optional): Flag to indicate if the data should be processed as a stream, defaults to False.
    collection (str, optional): The name of the collection to be used in the vector database, defaults to None.
    prompt_config (dict, optional): Configuration settings for prompts, defaults to None.
    metadata_dict (dict, optional): A dictionary to specify which metadata features to include, defaults to a predefined set.
    embedding_model (str, optional): The name of the embedding model to be used, defaults to "sentence-transformers/all-mpnet-base-v2".
    retriever_rank (str, optional): The ranking method to be used by the retriever, defaults to "LostInMiddle".
    generate_method (str, optional): The method used for generation tasks, defaults to "stuff".
    verbose (bool, optional): Flag to indicate if verbose mode is enabled, defaults to "True".
    vector_store (str, optional): The type of vector store to be used, defaults to "pgvector".

Raises:
    ValueError: If `prompt_config` is not provided.
    ValueError: If `llm_provider` is not provided.

Returns:
    MaxAgentQA: An instance of MaxAgentQA.

>>> from maxaillm.app.agent import MaxAgentQA
>>> agent = MaxAgentQA(llm_provider="anthropic",model_name ="claude-2", chunk_size=1000, stream=True, collection="myCollection", prompt_config=myPromptConfig)