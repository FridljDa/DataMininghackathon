# %%
# Simple File to gain some initial insights into the data and create a simple base line solution for the competition

import pandas as pd
import numpy as np


# %%

customers = pd.read_csv("data/01_raw/customer_test.csv", delimiter="\t")
features = pd.read_csv("data/01_raw/features_per_sku.csv", delimiter="\t")
nace_codes = pd.read_csv("data/01_raw/nace_codes.csv", delimiter="\t")
product_line_items = pd.read_csv("data/01_raw/plis_training.csv", delimiter="\t")


features.head()
features.info() 
# %%
# Check which customers where chosen for the test set

# %%
agg_orders = (
    product_line_items
    .groupby("legal_entity_id")
    .agg(
        num_orders=("legal_entity_id", "size"),
        num_order_sets=("set_id", "nunique")
    )
    .reset_index()
)

# %%
customers.merge(agg_orders, on="legal_entity_id")

# %%
cs_cust_agg = customers.merge(agg_orders, on="legal_entity_id", how="left")[customers["task"] == "cold start"]# %%
warm_cust_agg = customers.merge(agg_orders, on="legal_entity_id", how="left")[customers["task"] == "predict future"]

# -> based on this information a validaiton set of 50 warm legal_entity_ids was created to mimic the buying volume distribution of the final test set
# -> See the Snake file

# %%

# Make a first Submission for 1 legal_entity_id to check the submission format -- choose classes based on intuition and human input

sample_entity = 60153637

buyers_items = product_line_items[product_line_items["legal_entity_id"] == sample_entity]
buyers_items.count()

buyers_items["monetary_volume"] = (
    buyers_items["quantityvalue"] * buyers_items["vk_per_item"]
)

# %%



buyers_classes = (
    buyers_items
    .groupby("eclass")
    .agg(
        num_entries=("eclass", "size"),
        total_quantity=("quantityvalue", "sum"),
        total_monetary_volume=("monetary_volume", "sum"),
        buyer_id = ("legal_entity_id", "first")
    )
    .sort_values("num_entries", ascending=False)
    .reset_index()
)

# Data Wrangler shows: For this buyer he bought 918 eclasses
# - about 730 eclasses 0-5 times and 100 eclasses 5-10 times
# %%

random_threshhold = 10

buyers_classes_filtered = buyers_classes[buyers_classes["num_entries"] > random_threshhold]

buyers_classes_filtered[["buyer_id", "eclass"]].to_csv("data/15_sample_solutions/1_buyer.csv", index=False, header=True)
print("Wrote 1_buyer.csv with {} entries".format(len(buyers_classes_filtered)))

# Results:
#   - 1_buyer.csv with 88 entries
#   - Evaluation results: Total Score:€3,890.52 Savings:€4,770.52 Fees:€880.00
#                         Hits:80 Spend Captured:0.19%
#   - meaning: 8 of the items above the threshold were not bought again and and we only captured about 20% of the buyers spend
# %%
# Now the basically the same thing for all buyers!
warm_customers_pli = product_line_items[product_line_items["legal_entity_id"].isin(warm_cust_agg["legal_entity_id"])]

all_buyers_classes = (
    warm_customers_pli
    .groupby(["legal_entity_id", "eclass"])
    .agg(
        num_entries=("eclass", "size"),
        total_quantity=("quantityvalue", "sum"),
        #total_monetary_volume=("monetary_volume", "sum")
    )
    .reset_index()
)

classes_filtered = all_buyers_classes[all_buyers_classes["num_entries"] > random_threshhold]

base_line_submission = classes_filtered[["legal_entity_id", "eclass"]]
base_line_submission.to_csv("data/15_sample_solutions/all_buyers_simple_class_threshold.csv", index=False, header=True)
print("Wrote all_buyers_simple_class_threshold.csv with {} entries".format(len(classes_filtered)))
# %% # Run if you need the RAM
del product_line_items
del base_line_submission
# We now implement a local validation pipeline

# %%
# 1. - Number one the validaiton set of 50 legal_entities:

val_set_cust = pd.read_csv("data/03_customer/customer.csv", delimiter="\t")
val_set_cust = val_set_cust[val_set_cust["task"] == "testing"].reset_index(drop=True)

# %%
    
# 2. - The product line items for the training and validation set

product_line_items_train = pd.read_csv("data/03_training_validation/plis_training.csv", delimiter="\t")
product_line_items_val = pd.read_csv("data/03_training_validation/plis_testing.csv", delimiter="\t")

# 3. - The evaluation function
# %%

def validate_submission(submission, validation_set, savings_rate=0.1, fee=10):

    validation_set["monetary_volume"] = (
        validation_set["quantityvalue"] * validation_set["vk_per_item"]
    )

    # Sanitize submission:
    # Remove duplicates
    print(submission.shape)
    submission = submission.drop_duplicates(subset=["legal_entity_id", "eclass"])
    print(submission.shape)
    # Remove entries with missing values
    submission = submission.dropna(subset=["legal_entity_id", "eclass"])
    print(submission.shape)
    # Remove entries not seen in the validation set
    submission = submission[submission["legal_entity_id"].isin(validation_set["legal_entity_id"])]
    print(submission.shape)


    # Convert to set for faster lookup
    submission = set(zip(submission["legal_entity_id"], submission["eclass"]))


    hitted_entries = set()
    total_savings = 0.0

    for row in validation_set.itertuples():
        legal_entity_id = row.legal_entity_id
        eclass = row.eclass
        monetary_volume = row.monetary_volume

        if (legal_entity_id, eclass) in submission:
            # Hit
            hitted_entries.add((legal_entity_id, eclass))
            
            total_savings += monetary_volume * savings_rate

    total_fees = len(submission) * fee
    net_savings = total_savings - total_fees

    return {
        "Total Score": net_savings,
        "total_savings": total_savings,
        "total_fees": total_fees,
        "Num Hits": len(hitted_entries),
        "Spend Captured": total_savings / validation_set["monetary_volume"].sum()
    }

#%%
# 4 - Now we can validate our simple baseline solution on the validation set


warm_customers_pli = product_line_items_train[product_line_items_train["legal_entity_id"].isin(val_set_cust["legal_entity_id"])]

all_buyers_classes_val = (
    warm_customers_pli
    .groupby(["legal_entity_id", "eclass"])
    .agg(
        num_entries=("eclass", "size"),
        total_quantity=("quantityvalue", "sum"),
        #total_monetary_volume=("monetary_volume", "sum")
    )
    .reset_index()
)

classes_filtered = all_buyers_classes_val[all_buyers_classes_val["num_entries"] > random_threshhold]

base_line_submission = classes_filtered[["legal_entity_id", "eclass"]]

res = validate_submission(base_line_submission, product_line_items_val)

print("Validation Results:")
print(res)  
# %%

# Result Comparison: Criteria,  Baseline Test (Website),    Baseline Val set own implementation
#            Number of lines,   9751,                       6581
#            Total Score,       €655,335.88                 €374'406.88
#            Total Savings,     €752,835.88                 €438'796.88
#            Total Fees,        €97,500.00                  €64'390.00
#            Num Hits,          8'348                       5124
#           Spend Captured,     30.04%                      5.67%

# result look like we can somewhat accurately approximate the testing with our validation
# -> ! what doesn't work correctly is the spend captured metric 
# -> Also the Total Fee and number of lines don't totally match for the VAl -> little research suggests that the plies_val set is not complete or right 