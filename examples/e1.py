import docker
import os
from typing import Optional
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gauai.dataframe import DataFrame
import polars as pl


def main():
    df = pl.read_csv("examples/data/heart_attack_dataset.csv")
    df_agent = DataFrame(df)
    agent_output = df_agent.chat(
        "What is the average age of the people in the dataset?"
    )
    print("Final output:")
    print(agent_output)


if __name__ == "__main__":
    main()
