import pickle

# -----------------------------
# Load FP-Growth rules
# -----------------------------
with open("fpgrowth_rules.pkl", "rb") as f:
    rules = pickle.load(f)

# -----------------------------
# Recommendation Function
# -----------------------------
def recommend(user_items):
    results = []

    for _, row in rules.iterrows():
        antecedent = list(row["antecedents"])
        consequent = list(row["consequents"])

        # Check if user’s selected items match LHS
        if all(item in user_items for item in antecedent):
            results.append({
                "recommended_item": consequent[0],
                "confidence": float(row["confidence"]),
                "lift": float(row["lift"])
            })

    return results

# -----------------------------
# Test (optional)
# -----------------------------
if __name__ == "__main__":
    print(recommend(["Samsung", "Android"]))
