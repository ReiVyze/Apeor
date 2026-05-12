"""
Shared pytest fixtures for the OracleE Sentiment Dashboard test suite.
"""
import pytest
import pandas as pd


@pytest.fixture
def sample_comments():
    return [
        "This healthcare policy is excellent and very helpful!",
        "The economic clause lacks clarity and is confusing.",
        "The environmental initiative seems fine to me.",
    ]


@pytest.fixture
def sample_dataframe():
    return pd.DataFrame({
        "text": [
            "Great initiative for the environment!",
            "The funding is insufficient and worrying.",
            "I have no strong opinion about this clause.",
        ],
        "label": ["Positive", "Negative", "Neutral"],
        "score": [0.92, 0.88, 0.55],
        "theme": ["Clause 3: Environment", "Clause 1: Economics", "Clause 2: Healthcare"],
        "region": ["North", "South", "East"],
        "channel": ["Direct Portal", "Email", "Chat"],
    })
