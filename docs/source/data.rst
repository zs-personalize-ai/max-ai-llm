data
====

chunking
********

MarkdownHeaderSplitter
^^^^^^^^^^^^^^^^^^^^^^^^
MarkdownHeaderSplitter is a class that provides functionality for splitting text based on Markdown headers.

Args:
    - ``splits (List[Tuple[str, str]], optional)``: A list of tuples representing Markdown header patterns to split on. Each tuple contains two strings: the Markdown header prefix (e.g., "#") and the corresponding header name (e.g., "Header 1"). If not provided, default header patterns will be used.

Attributes:
    - ``headers_to_split_on (List[Tuple[str, str]])``: The Markdown header patterns to split on.

Methods:
    - ``split_text(text)``: Splits the given text based on Markdown headers.

        - ``text (str)``: The text to split.

    - ``split_document(document)``: Splits the text of the given document based on Markdown headers.

        - ``document (Document)``: The document to split.

Raises:
    - ``Exception``: If an error occurs while splitting the text.
    

HTMLHeaderSplitter
^^^^^^^^^^^^^^^^^^
HTMLHeaderSplitter is a class that provides functionality for splitting text based on HTML headers.

Args:
    - ``splits (List[Tuple[str, str]], optional)``: A list of tuples representing HTML header tags to split on. Each tuple contains two strings: the HTML header tag (e.g., "h1") and the corresponding header name (e.g., "Header 1"). If not provided, default header tags will be used.

Attributes:
    - ``headers_to_split_on (List[Tuple[str, str]])``: The HTML header tags to split on.

Methods:
    - ``split_text(text)``: Splits the given text based on HTML headers.

        - ``text (str)``: The text to split.

    - ``split_text_from_url(url)``: Fetches a document from the given URL and splits its text based on HTML headers.

        - ``url (str)``: The URL of the document to fetch and split.

    - ``split_document(document)``: Splits the text of the given document based on HTML headers.

        - ``document (Document)``: The document to split.

Raises:
    - ``Exception``: If an error occurs while splitting the text.
    
    
TextSplitter
^^^^^^^^^^^^
TextSplitter is a class that provides functionality for splitting text based on various methods.

Args:
    - ``chunk_size (int, optional)``: The maximum size of each chunk.
    - ``chunk_overlap (int, optional)``: The size of the overlap between chunks.
    - ``length_function (Callable[[str], int], optional)``: The function to use to calculate the length of a chunk.
    - ``separator (str, optional)``: The separator to use between chunks.
    - ``is_separator_regex (bool, optional)``: Whether the separator is a regular expression.
    - ``token_method_params (dict, optional)``: The parameters to use for the token method.

Attributes:
    - ``chunk_size (int)``: The maximum size of each chunk.
    - ``chunk_overlap (int)``: The size of the overlap between chunks.
    - ``length_function (Callable[[str], int])``: The function to use to calculate the length of a chunk.
    - ``separator (str)``: The separator to use between chunks.
    - ``is_separator_regex (bool)``: Whether the separator is a regular expression.
    - ``token_method_params (dict)``: The parameters to use for the token method.

Methods:
    - ``split_text(text, method)``: Splits the given text based on the specified method.

        - ``text (str)``: The text to split.
        - ``method (str)``: The method to use to split the text.

    - ``add_default_metadata_to_document(document, default_metadata)``: Adds default metadata to the given document.

        - ``document (Document)``: The document to add metadata to.
        - ``default_metadata (dict, optional)``: The default metadata to add.

    - ``create_documents(texts, file_metadata, metadata, default_metadata, method)``: Creates documents from the given texts.

        - ``texts (List[str])``: The texts to create documents from.
        - ``file_metadata (Dict[str, str], optional)``: The file metadata to add to the documents.
        - ``metadata (Dict[str, bool], optional)``: The metadata to add to the documents.
        - ``default_metadata (Dict[str, str], optional)``: The default metadata to add to the documents.
        - ``method (str)``: The method to use to split the texts.

    - ``dedup_chunks(chunks)``: Removes duplicate chunks from the given list.

        - ``chunks (List[str])``: The list of chunks to deduplicate.

    - ``split_document(document)``: Splits the text of the given document.

        - ``document (Document)``: The document to split.

    - ``serialize_datetime(obj)``: Serializes the given datetime object.

        - ``obj (datetime)``: The datetime object to serialize.

    - ``extract_metadata_generic(text, metadata_extraction_function, metadata_key)``: Extracts metadata from the given text.

        - ``text (str)``: The text to extract metadata from.
        - ``metadata_extraction_function (Callable[[str], Any])``: The function to use to extract metadata.
        - ``metadata_key (str)``: The key to use for the extracted metadata.

    - ``add_metadata_to_document(document, metadata_extraction_function, metadata_key)``: Adds metadata to the given document.

        - ``document (Document)``: The document to add metadata to.
        - ``metadata_extraction_function (Callable[[str], Any])``: The function to use to extract metadata.
        - ``metadata_key (str)``: The key to use for the extracted metadata.

    - ``add_metadata_to_documents_parallel(documents, metadata_extraction_function, metadata_key, max_workers)``: Adds metadata to the given documents in parallel.

        - ``documents (List[Document])``: The documents to add metadata to.
        - ``metadata_extraction_function (Callable[[str], Any])``: The function to use to extract metadata.
        - ``metadata_key (str)``: The key to use for the extracted metadata.
        - ``max_workers (int, optional)``: The maximum number of workers to use.

Raises:
    - ``Exception``: If an error occurs while splitting the text or adding metadata.
    
.. code-block:: python
    
    from maxaillm.data.chunking.TextSplitter import TextSplitter
    from maxaillm.data.extractor.MaxExtractor import MaxExtractor
    
    # extract the text from the document and clean the text
    me_obj = MaxExtractor()
    text, metadata = me.extract_text_metadata("path/to/file")
    clean_text = me.clean_text(
        text,
        dehyphenate=True, 
        ascii_only=True, 
        remove_isolated_symbols=True, 
        compress_whitespace=True
    )
    
    # define splitter
    splitter = TextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.create_documents(
        [text],
        file_metadata=metadata,
        metadata={
            "default": True,
            "summary": False,
            "entities": False,
            "frequent_keywords": True,
            "links": True,
        },
        default_metadata={"file_name": "file_name"},
    )


embeddings
**********

MaxHuggingFaceEmbeddings
^^^^^^^^^^^^^^^^^^^^^^^^
MaxHuggingFaceEmbeddings is a class that inherits from MaxLangchainEmbeddings and initializes a HuggingFaceEmbeddings model.

Args:
    - ``**kwargs``: Arbitrary keyword arguments for the HuggingFaceEmbeddings model.

Attributes:
    - ``model (MaxEmbeddingsBase)``: The translated MaxEmbeddingsBase model.
    
Methods:
    - ``embed_documents(texts)``: Embeds the given search documents.

        - ``texts (List[str])``: The search documents to embed.

    - ``embed_query(text)``: Embeds the given query text.

        - ``text (str)``: The query text to embed.

    - ``aembed_documents(texts)``: Asynchronously embeds the given search documents.

        - ``texts (List[str])``: The search documents to embed.

    - ``aembed_query(text)``: Asynchronously embeds the given query text.

        - ``text (str)``: The query text to embed.
        
.. code-block:: python

    from maxaillm.data.embeddings.MaxHuggingFaceEmbeddings import MaxHuggingFaceEmbeddings
    
    
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = MaxHuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )


MaxLangchainEmbeddings
^^^^^^^^^^^^^^^^^^^^^^
MaxLangchainEmbeddings is a class that inherits from MaxEmbeddingsBase and provides methods for embedding texts.

Args:
    - ``model``: The model to use for embedding.

Attributes:
    - ``model (MaxEmbeddingsBase)``: The model used for embedding.

Methods:
    - ``embed_many(texts)``: Embeds the given search documents.

        - ``texts (List[str])``: The search documents to embed.

    - ``embed(text)``: Embeds the given query text.

        - ``text (str)``: The query text to embed.

    - ``embed_many_async(texts)``: Asynchronously embeds the given search documents.

        - ``texts (List[str])``: The search documents to embed.

    - ``embed_async(text)``: Asynchronously embeds the given query text.

        - ``text (str)``: The query text to embed.
        
        
extractor
*********

MaxExtractor
^^^^^^^^^^^^
MaxExtractor is a class that inherits from MaxExtractorBase and MaxLLMBase and provides methods for extracting text, pages, details, tables, and metadata from documents.

Args:
    - ``parser_class_map_override (Optional[Dict[str, MaxExtractorBase]])``: A dictionary with a mapping of extensions to parser classes to merge with or override the defaults.

Attributes:
    - ``parser_class_map``: A dictionary with a mapping of extensions to parser classes.
    - ``supported_extensions``: A set of supported extensions.

Methods:
    - ``get_parser(extension)``: Returns the appropriate parser for a file type.

        - ``extension (str)``: The file extension.

    - ``get_extension_from_path(path)``: Gets the file extension from a path.

        - ``path (Union[str, Path])``: The file path.

    - ``get_extension(document, extension)``: Gets the file extension from a path or a named file-like object.

        - ``document (Union[str, Path, bytes, IO])``: The document.
        - ``extension (Optional[str])``: The file extension.

    - ``extract_text(document, extension, ocr, **kwargs)``: Extracts text from a document.

        - ``document (Union[str, Path, bytes, IO])``: The document.
        - ``extension (Optional[str])``: The file extension.
        - ``ocr (bool)``: Whether to use OCR.

    - ``extract_pages(document, extension, ocr, **kwargs)``: Extracts pages from a document.

        - ``document (Union[str, Path, bytes, IO])``: The document.
        - ``extension (Optional[str])``: The file extension.
        - ``ocr (bool)``: Whether to use OCR.

    - ``extract_details(document, extension, ocr)``: Extracts details from a document.

        - ``document (Union[str, Path, bytes, IO])``: The document.
        - ``extension (Optional[str])``: The file extension.
        - ``ocr (bool)``: Whether to use OCR.

    - ``extract_tables(document, extension)``: Extracts tables from a document.

        - ``document (Union[str, Path, bytes, IO])``: The document.
        - ``extension (Optional[str])``: The file extension.

    - ``extract_metadata(document, extension)``: Extracts metadata from a document.

        - ``document (Union[str, Path, bytes, IO])``: The document.
        - ``extension (Optional[str])``: The file extension.

    - ``split_document(document, extension, split_size)``: Splits a document.

        - ``document (Union[str, Path, bytes, IO])``: The document.
        - ``extension (Optional[str])``: The file extension.
        - ``split_size (int)``: The split size.

    - ``to_pdf(document, extension)``: Converts a document to PDF.

        - ``document (Union[str, Path, bytes, IO])``: The document.
        - ``extension (Optional[str])``: The file extension.

Raises:
    - ``Exception``: If the extension must be provided and the document does not refer to a path or a named file-like object.
    - ``ValueError``: If the Azure Storage connection string is not found in environment variables or if the Azure blob path format is invalid.
    
.. code-block:: python

    from maxaillm.data.extractor.MaxExtractor import MaxExtractor
    
    
    me_obj = MaxExtractor()
    text, metadata = me.extract_text_metadata("path/to/file")
    
    # clean the text
    clean_text = me.clean_text(
        text,
        dehyphenate=True, 
        ascii_only=True, 
        remove_isolated_symbols=True, 
        compress_whitespace=True
    )