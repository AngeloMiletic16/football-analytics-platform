from src.data.loaders import load_raw_matches
from src.data.cleaning import clean_matches_v1, save_processed_matches


def main() -> None:
    raw_df = load_raw_matches("matches.csv")
    clean_df = clean_matches_v1(raw_df)

    print("Raw shape:", raw_df.shape)
    print("Clean shape:", clean_df.shape)
    print(clean_df.head())

    save_processed_matches(clean_df)
    print("Saved cleaned dataset to data/processed/matches_v1_clean.csv")


if __name__ == "__main__":
    main()