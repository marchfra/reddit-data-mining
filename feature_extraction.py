import numpy as np
import pandas as pd
from scipy import sparse

DATA_DIR = "data"


def load_csv(filename: str) -> pd.DataFrame:
    """
    Load a CSV file from the data directory.
    """
    ext = filename.split(".")[-1]
    if not ext == "csv":
        raise ValueError(f"Invalid file extension: {ext}. Must be a CSV file.")
    return pd.read_csv(f"{DATA_DIR}/{filename}", encoding="utf-8")


def load_data(
    train_data: str = "train_data.csv",
    train_target: str = "train_target.csv",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the training data and target values.
    """
    train_data = load_csv(train_data)
    target = load_csv(train_target)
    return train_data, target


def create_subreddit_idx(data: pd.DataFrame) -> pd.Series:
    """
    Create a dictionary mapping each subreddit to an integer.
    """
    subreddits = data["subreddit"].unique()
    return pd.Series(index=subreddits, data=np.arange(len(subreddits)))


def extract_subreddits(
    author_data: pd.DataFrame,
    subreddit_idx: pd.Series,
) -> sparse.csr_array:
    """
    This function converts all the subreddits the author has posted in into a sparse
    array of length N (where N is the number of subreddits in the dataset) with 1s in
    the indexes of the subreddits the author has posted in.
    """
    user_subreddits = author_data["subreddit"].to_numpy()

    # idxs is an array with the indexes of the subreddits in subreddits_idx
    idxs = subreddit_idx.loc[user_subreddits].values

    # create a sparse array indicating the subreddits the author has posted in
    v = sparse.dok_array((1, len(subreddit_idx)))  # dok = dictionary of keys
    for idx in idxs:
        v[0, idx] = 1
    return v.tocsr()  # convert to compressed sparse row format


def extract_text(author_data: pd.DataFrame) -> str:
    """
    Concatenates all the posts of an author into a single string.
    """
    group_text = author_data["body"].astype(str).values
    return " ".join(group_text)


def extract_features(
    train_data: pd.DataFrame,
    target: pd.DataFrame,
) -> tuple[sparse.csr_matrix, list[str], pd.Series]:
    # Feature extraction
    subreddit_idx = create_subreddit_idx(train_data)

    subreddits_dict: dict[str, sparse.csr_array] = {}
    for author, group in train_data.groupby("author"):
        subreddits_dict[author] = extract_subreddits(group, subreddit_idx)

    text_dict: dict[str, str] = {}
    for author, group in train_data.groupby("author"):
        text_dict[author] = extract_text(group)

    # Generate a sparse matrix with the labelled authors as rows and the subreddits they
    # have posted in as columns
    X: sparse.csr_matrix = sparse.vstack(
        [subreddits_dict[author] for author in target["author"]]
    )
    author_text: list[str] = [text_dict[author] for author in target["author"]]
    y: pd.Series = target["gender"]

    return X, author_text, y


def main() -> None:
    # Load the training data
    train_data, target = load_data()
    print(f"Number of authors: {len(train_data['author'].unique())}")

    # Extract features
    X, author_text, y = extract_features(train_data, target)


if __name__ == "__main__":
    main()
