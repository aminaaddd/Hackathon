import pandas as pd

def clean_data(df):
    """
    Args:
        df (dataframe)

    Returns:
        dataframe: cleaned dataframe
    """

    # Turn dates into real dates
    for c in ["Date of Injury","Date of return"]:
        df[c] = pd.to_datetime(df[c], errors="coerce")
    
    df = df.sort_values(["Name","Date of Injury"]) # sort by player and when they got injured
    df["next_injury_date"] = df.groupby("Name")["Date of Injury"].shift(-1)

    df["days_until_next"] = (df["next_injury_date"] - df["Date of return"]).dt.days # calculate days until next injury
    df["injured_next_time"] = ((df["days_until_next"] > 0) & (df["days_until_next"] <= 180)).astype(int)

    # convert to numeric
    num_cols = [
        "Match1_before_injury_Player_rating","Match2_before_injury_Player_rating","Match3_before_injury_Player_rating",
        "Match1_before_injury_GD","Match2_before_injury_GD","Match3_before_injury_GD", "Team",
        "Age","FIFA rating"
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            
    # average ratings and GDs        
    df["avg_before_rating"] = df[[
        "Match1_before_injury_Player_rating",
        "Match2_before_injury_Player_rating",
        "Match3_before_injury_Player_rating"
    ]].mean(axis=1)

    df["avg_before_gd"] = df[[
        "Match1_before_injury_GD",
        "Match2_before_injury_GD",
        "Match3_before_injury_GD"
    ]].mean(axis=1)

    return df
    
def prepare_features(cleaned_df):
    """
    Args:
        cleaned_df (dataframe)

    Returns:
        X: numeric
        y: 0/1 target
    """
    X = cleaned_df[["Age","FIFA rating","Position","Injury","avg_before_rating","avg_before_gd", "Team Name"]].copy()
    y = cleaned_df["injured_next_time"].copy()
    
    cols_to_encode = [c for c in ["Team Name","Position","Injury"] if c in X.columns]
    if cols_to_encode:
        X = pd.get_dummies(X, columns=cols_to_encode, dummy_na=True)

    X = X.fillna(0)

    mask = y.notna()
    X, y = X[mask], y[mask]
    return X, y
