# Hitrate@10 evaluation function
import pandas as pd


def hitrate_at_k(true_data: pd.DataFrame,
                 predicted_data: pd.DataFrame,
                 k: int = 10) -> float:
    """
    This function calculates the hitrate at k for the recommendations.
    It assesses how relevant our 10 product recommendations are.
    In other words, it calculates the proportion of recommended products that are actually purchased by the customer.

    Args:
        true_data: a pandas DataFrame containing the true data
            customer_id: the customer identifier
            product_id: the product identifier that was purchased in the test set
        predicted_data: a pandas DataFrame containing the predicted data
            customer_id: the customer identifier
            product_id: the product identifier that was recommended
            rank: the rank of the recommendation. the rank should be between 1 and 10.
        k: the number of recommendations to consider. k should be between 1 and 10.
    
    Returns:
        The hitrate at k
    """
    
    data = pd.merge(left = true_data, right = predicted_data, how = "left", on = ["customer_id", "product_id"])
    df = data[data["rank"] <= k]
    non_null_counts = df.groupby('customer_id')['rank'].apply(lambda x: x.notna().sum()).reset_index(name='non_null_count')
    distinct_products_per_customer = data.groupby('customer_id')['product_id'].nunique().reset_index(name='distinct_product_count')
    df = pd.merge(left = distinct_products_per_customer, right = non_null_counts, how = "left", on = "customer_id")
    df["denominator"] = [min(df.iloc[i].distinct_product_count,k) for i in range(len(df))]
    df = df.fillna(0)
    return (df["non_null_count"]/df["denominator"]).mean()