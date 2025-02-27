import polars as pl
from smolagents import CodeAgent, HfApiModel, tool


class DataFrame(pl.DataFrame):
    # Class-level variable to store current DataFrame instance
    _current_df = None

    def __init__(
        self,
        df=None,
        schema=None,
        **kwargs,
    ):
        super().__init__(
            df,
            schema,
            **kwargs,
        )
        # Store self as current DataFrame instance
        DataFrame._current_df = self

    @staticmethod
    @tool
    def execute_df_operation(code: str) -> str:
        """
        Executes a DataFrame operation and returns the result as a string.
        The DataFrame is available as 'df'. Use Polars operations to analyze the data.

        Args:
            code: Python code to execute on the DataFrame. Must be a valid Polars operation.
                The DataFrame is available as 'df' in the execution context.

        Returns:
            str: String representation of the operation result.
                For DataFrames, returns shape and preview.
                For scalar values, returns the value directly.
                For errors, returns an error message.

        Examples:
            - df.shape
            - df.columns
            - df.select(pl.col("column_name")).head().to_dict()
            - df.filter(pl.col("column") > 10).shape

        Raises:
            ValueError: If code is empty or not a string
            Exception: If DataFrame operation fails
        """
        if not isinstance(code, str) or not code.strip():
            return "Error: Code must be a non-empty string"

        try:
            # Get current DataFrame instance
            df = DataFrame._current_df
            if df is None:
                return "Error: No DataFrame instance available"

            # Execute in controlled namespace
            namespace = {"df": df, "pl": pl}
            result = eval(code, {"__builtins__": {}}, namespace)

            # Format result based on type
            if isinstance(result, pl.DataFrame):
                return f"Shape: {result.shape}\nPreview:\n{result.head(5)}"
            return str(result)

        except Exception as e:
            error_type = type(e).__name__
            return f"Error executing operation ({error_type}): {str(e)}"

    def chat(self, prompt: str) -> str:
        """Chat with the DataFrame using natural language.

        This method creates a CodeAgent that can analyze and query the DataFrame
        using natural language prompts. The agent has access to the DataFrame
        through a tool that allows executing polars operations.

        Args:
            prompt: The natural language query or instruction about the DataFrame.

        Returns:
            str: The agent's response based on analyzing the DataFrame.
        """

        agent = CodeAgent(
            tools=[DataFrame.execute_df_operation],
            model=HfApiModel("Qwen/Qwen2.5-Coder-32B-Instruct"),
            additional_authorized_imports=["polars"],
        )

        # Run the agent with the user's prompt
        return agent.run(prompt)
