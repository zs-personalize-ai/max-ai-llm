app
====

generator
************

MaxGenerator
^^^^^^^^^^^^
This class manages multiple generators for response generation and allows switching between different generator implementations.

Initializes the GeneratorManager and sets up available generators.

Args:
    - ``llm (LLM)``: The language model to be used.
    - ``method (str)``: The method to be used for generation.
    - ``prompt_config (dict)``: The configuration for the prompt.
    - ``engine (str, optional)``: The engine to be used for generation. Defaults to "langchain".
    - ``streamable (bool, optional)``: Indicates if the generator supports streaming responses. Defaults to False.
    - ``verbose (bool, optional)``: Indicates if verbose output should be enabled. Defaults to True.

Attributes:
    - ``generators (dict[str, Generator])``: A dictionary mapping generator names to their instances.
    - ``selected_generator (Generator)``: The currently selected generator for response generation.
    
.. code-block:: python
    from maxaillm.app.generator.MaxGenerator import MaxGenerator
    
    
    # define prompt configuration
    p_conf = {'moderations':'', 'task':'', 'identity':''}
    
    # initialize MaxGenerator
    mg = MaxGenerator(llm=llm, method='stuff', prompt_config=p_conf, engine="langchain")
    
    # generate batch response
    mg.generate(query='Explain Reinforcement Learning', context=out)
    
    # to generate 
    mg.generate_stream(query='Explain Reinforcement Learning', context=out)

Methods:
    - ``generate(query, context, conversation)``: Generates a response based on the query and context.

        - ``query (str)``: The query to generate a response for.
        - ``context (List[str])``: The context for the query.
        - ``conversation (List[dict], optional)``: The conversation history.

    - ``generate_stream(query, context, conversation)``: Asynchronously generates a response based on the query and context.

        - ``query (str)``: The query to generate a response for.
        - ``context (List[str])``: The context for the query.
        - ``conversation (List[dict], optional)``: The conversation history.

    - ``set_generator(generator)``: Sets the current generator.

        - ``generator (str)``: The name of the generator to set.

    - ``get_generators()``: Returns the available generators.

    - ``calculate_tokens(prompt_config, context, query, chat)``: Calculates the number of tokens in the formatted response.

        - ``prompt_config (dict)``: The configuration for the prompt.
        - ``context (List[str])``: The context for the query.
        - ``query (str)``: The query to generate a response for.
        - ``chat (str)``: The chat history.