from __future__ import annotations

from sklearn.model_selection import train_test_split


def student_level_split(
    df,
    student_col: str = "user_id",
    test_size: float = 0.20,
    seed: int = 42,
):
    """Split a dataframe by unique students and return train/test frames and IDs."""
    students = df[student_col].unique()
    train_ids, test_ids = train_test_split(students, test_size=test_size, random_state=seed)
    train_df = df[df[student_col].isin(train_ids)].reset_index(drop=True)
    test_df = df[df[student_col].isin(test_ids)].reset_index(drop=True)
    return train_df, test_df, set(train_ids), set(test_ids)
