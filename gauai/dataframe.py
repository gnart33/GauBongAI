import polars as pl


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
        pass
