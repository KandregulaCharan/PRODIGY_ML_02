import pandas as pd
import numpy as np


num_customers = 100  # Number of customers to generate


np.random.seed(42)  


data = {
    'customer_id': np.arange(1, num_customers + 1),
    'total_spent': np.random.uniform(100, 5000, num_customers),  # Total amount spent
    'num_transactions': np.random.randint(1, 50, num_customers),  # Number of transactions
    'avg_transaction_value': np.random.uniform(20, 200, num_customers),  # Average transaction value
    'purchase_frequency': np.random.uniform(1, 12, num_customers)  # Frequency of purchases (per year)
}


df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
csv_file_path = 'customer_purchase_history.csv'
df.to_csv(csv_file_path, index=False)

print(f"CSV file '{csv_file_path}' has been created with {num_customers} customers.")
