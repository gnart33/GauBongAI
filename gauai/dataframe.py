import polars as pl
from smolagents import CodeAgent, HfApiModel, tool


class DataFrame(pl.DataFrame):
    def __init__(
        self,
        data=None,
        schema=None,
        **kwargs,
    ):
        super().__init__(
            data,
            schema,
            **kwargs,
        )

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

        # Initialize the agent with the DataFrame query tool
        agent = CodeAgent(
            tools=[],
            model=HfApiModel("Qwen/Qwen2.5-Coder-32B-Instruct"),
            additional_authorized_imports=["polars"],
        )

        # Run the agent with the user's prompt
        return agent.run(prompt)
