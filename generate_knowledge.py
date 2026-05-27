import pandas as pd
import numpy as np

# 1. Load your dataset
df = pd.read_csv("house_prices.csv")

# 2. Clean data (Matches your backend training logic)
df = df[(df["Area_in_Marla"] > 0) & (df["bedrooms"] > 0) & (df["baths"] > 0)]
df["price_tier"] = pd.cut(
    df["price"],
    bins=[0, 7_000_000, 22_000_000, np.inf],
    labels=["Low", "Medium", "High"]
)

# 3. Analyze patterns for the RAG Knowledge Base
summary = df.groupby(['city', 'location', 'property_type']).agg(
    avg_price=('price', 'mean'),
    median_area=('Area_in_Marla', 'median'),
    mode_tier=('price_tier', lambda x: x.mode()[0] if not x.mode().empty else "Unknown"),
    count=('price', 'count')
).reset_index().sort_values(by='count', ascending=False)

# 4. Save to market_knowledge.txt
with open("market_knowledge.txt", "w") as f:
    f.write("# Pakistan Real Estate Market Knowledge Base\n\n")
    f.write("## Section 1: Location and Property Type Benchmarks\n")
    for _, row in summary.head(150).iterrows():
        f.write(f"In {row['location']}, {row['city']}, {row['property_type']}s are typically in the '{row['mode_tier']}' price tier. ")
        f.write(f"The historical average price is approximately {row['avg_price']:,.0f} PKR with a median size of {row['median_area']} Marla.\n")
    
    f.write("\n## Section 2: General Market Drivers\n")
    f.write("- Properties in DHA and Bahria Town command a premium due to infrastructure.\n")
    f.write("- 5 Marla houses are common for 'Low/Medium' tiers; 10+ Marla often hit 'High' tiers.\n")

print("✅ Success! 'market_knowledge.txt' has been created in your folder.")