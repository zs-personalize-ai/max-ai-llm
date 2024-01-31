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

Methods:
    - ``generate(query: str, context: List[str], conversation: List[dict] = []) -> str``: Generates an answer based on a list of queries, contexts, and conversational context.
    - ``generate_stream(query: str, context: List[str], conversation: List[dict] = []) -> Iterator[str]``: Generates a stream of responses based on queries, contexts, and conversational context.
    - ``set_generator(generator: str) -> bool``: Sets the currently active generator.
    - ``get_generators() -> Dict[str, Generator]``: Returns the available generators.
    - ``calculate_tokens(prompt_config, context, query, chat)``: Calculates the number of tokens in the formatted response.