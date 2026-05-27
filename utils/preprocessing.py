import pandas as pd

def preprocess_for_model(raw_input, location_map):
    df = pd.DataFrame([raw_input])

    df["total_rooms"] = df["bedrooms"] + df["baths"]
    df["location_freq"] = df["location"].map(location_map).fillna(0)

    cols = [
        "property_type",
        "city",
        "baths",
        "bedrooms",
        "Area_in_Marla",
        "total_rooms",
        "location_freq",
        "purpose",
    ]

    return df[cols]
