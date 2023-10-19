def train_val_test_split(df, val_size=0.15, test_size=0.15, random_state=0):
    df = df.reindex(sorted(df.columns), axis=1)
    train_val = df.sample(frac=(1-test_size), random_state=random_state)
    test = df.drop(train_val.index)
    train = train_val.sample(frac=(1-val_size), random_state=random_state)
    val = train_val.drop(train.index)
    return train.reset_index().drop(columns=['index']), val.reset_index().drop(columns=['index']), test.reset_index().drop(columns=['index'])