import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_and_split(config: dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    try:
        phish = pd.read_csv(config['data']['phishing_path'])
        legit = pd.read_csv(config['data']['legit_path'])
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Dataset not found: {e}\nCreate files in data/raw/") from e

    for df in [phish, legit]:
        if 'URL' in df.columns:
            df.rename(columns={'URL': 'url'}, inplace=True)
        if 'Label' in df.columns:
            df.rename(columns={'Label': 'label'}, inplace=True)

        if 'url' not in df.columns:
            raise ValueError(f"'url' column missing in dataset")
        if 'label' not in df.columns:
            df['label'] = 0

    phish = phish[['url', 'label']].copy()
    legit = legit[['url', 'label']].copy()
    phish['label'] = 1

    df = pd.concat([phish, legit], ignore_index=True).dropna(subset=['url'])

    if len(df) == 0:
        raise ValueError("No valid URLs found in the datasets")

    if config['data'].get('balance_classes', False):
        counts = df['label'].value_counts()
        min_count = counts.min()
        if min_count == 0:
            raise ValueError("One class has zero samples → cannot balance")
        df_phish = df[df['label'] == 1].sample(min_count, random_state=42)
        df_legit = df[df['label'] == 0].sample(min_count, random_state=42)
        df = pd.concat([df_phish, df_legit], ignore_index=True)

    train_df, test_df = train_test_split(
        df,
        test_size=config['data']['test_size'],
        stratify=df['label'],
        random_state=config['model']['random_state']
    )

    return train_df, test_df, train_df['label'], test_df['label']