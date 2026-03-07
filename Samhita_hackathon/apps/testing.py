import pandas as pd
import plotly.express as px
import streamlit as st
from core.data import generator as dg

data = [
    {"customer_id": "C399", "customer_name": "Ding Dong", "email": "ra@bba.com", "phone_primary": "450450"},
    {"customer_id": "C699", "customer_name": "Ding Dong", "email": "ra@boba.com", "phone_primary": "450450"},
    {"customer_id": "C699", "customer_name": "Rahul Boby", "email": "rahulboby@what.com", "phone_primary": "090909"},
    


    # Group 1: Exact CID Match (Rule 1)
    {"customer_id": "C001", "customer_name": "John Doe", "email": "john@example.com", "phone_primary": "555-0101"},
    {"customer_id": "C001", "customer_name": "John Do", "email": "j.doe@work.com", "phone_primary": "555-0101"},
    {"customer_id": "C001", "customer_name": "John Doe", "email": "j.doe@work.com", "phone_primary": "555-9999"},
    
    # Group 2: 2/3 Match (Name & Email match, different CID)
    {"customer_id": "C002", "customer_name": "Alice Smith", "email": "alice.s@gmail.com", "phone_primary": "555-0202"},
    {"customer_id": "C099", "customer_name": "Alice Smith", "email": "alice.s@gmail.com", "phone_primary": "555-8888"},
    
    # Group 3: 2/3 Match (Email & Phone match, different Name)
    {"customer_id": "C003", "customer_name": "Robert Brown", "email": "bob@web.com", "phone_primary": "555-0303"},
    {"customer_id": "C004", "customer_name": "Bobby B", "email": "bob@web.com", "phone_primary": "555-0303"},
    
    # Group 4: Chain Match (A matches B on 2/3, B matches C on CID)
    {"customer_id": "C005", "customer_name": "Charlie Day", "email": "c.day@paddy.com", "phone_primary": "555-0505"},
    {"customer_id": "C006", "customer_name": "Charlie Day", "email": "c.day@paddy.com", "phone_primary": "555-0000"}, # Matches C005 on Name/Email
    {"customer_id": "C006", "customer_name": "Charles Day", "email": "charles@paddy.com", "phone_primary": "555-1111"}, # Matches previous on CID
    
    # Group 5: 2/3 Match (Name & Phone match, different Email)
    {"customer_id": "C007", "customer_name": "Diana Prince", "email": "diana@amazon.com", "phone_primary": "555-0707"},
    {"customer_id": "C008", "customer_name": "Diana Prince", "email": "wonder@justice.org", "phone_primary": "555-0707"},
    
    # Group 6: Missing Data Match (CID match with NaNs)
    {"customer_id": "C010", "customer_name": "Edward Nigma", "email": "question@gotham.com", "phone_primary": ""},
    {"customer_id": "C010", "customer_name": "", "email": "", "phone_primary": "555-1010"},

    # Unique Records (Should not merge)
    {"customer_id": "C011", "customer_name": "Frank Castle", "email": "punisher@mail.com", "phone_primary": "555-1100"},
    {"customer_id": "C012", "customer_name": "George Miller", "email": "george@furyroad.com", "phone_primary": "555-1200"},
    {"customer_id": "C013", "customer_name": "Hanna Abbott", "email": "hanna@hogwarts.edu", "phone_primary": "555-1300"},
    {"customer_id": "C014", "customer_name": "Ian Wright", "email": "ian@football.co.uk", "phone_primary": "555-1400"},
    {"customer_id": "C015", "customer_name": "Jenny Forrest", "email": "run.jenny@run.com", "phone_primary": "555-1500"},
    {"customer_id": "C016", "customer_name": "Kevin Hart", "email": "kevin@comedy.com", "phone_primary": "555-1600"},
    {"customer_id": "C017", "customer_name": "Laura Croft", "email": "laura@tomb.com", "phone_primary": "555-1700"},
    {"customer_id": "C018", "customer_name": "Michael Scott", "email": "michael@dundermifflin.com", "phone_primary": "555-1800"},
    {"customer_id": "C019", "customer_name": "Nina Simone", "email": "nina@jazz.com", "phone_primary": "555-1900"},
    {"customer_id": "C020", "customer_name": "Oscar Wilde", "email": "oscar@literature.com", "phone_primary": "555-2000"},
    {"customer_id": "C021", "customer_name": "Peter Parker", "email": "spidey@dailybugle.com", "phone_primary": "555-2100"},
    {"customer_id": "C022", "customer_name": "Quentin Tarantino", "email": "qt@movies.com", "phone_primary": "555-2200"},
    {"customer_id": "C023", "customer_name": "Riley Reid", "email": "riley@example.com", "phone_primary": "555-2300"},
    {"customer_id": "C024", "customer_name": "Steve Rogers", "email": "cap@avengers.com", "phone_primary": "555-2400"},
    {"customer_id": "C025", "customer_name": "Tony Stark", "email": "ironman@stark.com", "phone_primary": "555-2500"},
    {"customer_id": "C026", "customer_name": "Ultron", "email": "extinction@earth.com", "phone_primary": "000-0000"},
    {"customer_id": "C027", "customer_name": "Victor Von Doom", "email": "doom@latveria.gov", "phone_primary": "555-2700"},
    
    # Other records
    {"customer_id": "C027", "customer_name": "Victor Von Doom", "email": "doom@latveria.gov", "phone_primary": "555-2700"}

]

df = pd.DataFrame(data)
# print(df.head(5))

# df = dg.get_data()

cardinality_data = []
for col in df.columns:
    unique_count = df[col].nunique(dropna=False)
    total_count = df.shape[0]
    cardinality_data.append({
        "column": col,
        "unique_count": int(unique_count),
        "total_count": int(total_count),
        "cardinality_ratio": round(unique_count / total_count, 4) if total_count > 0 else 0
    })

cardinality_df = pd.DataFrame(cardinality_data)
filtered_cardinality_df = cardinality_df[cardinality_df["cardinality_ratio"] > 0.9]
df_card = df[cardinality_df['column']]


# 1. Prepare data
sorted_df = df.fillna("").astype(str)
sorted_df = sorted_df.sort_values(by=['customer_id', 'customer_name', 'email', 'phone_primary']).reset_index(drop=True)

index = 0
while index < len(sorted_df) - 1:
    row_curr = sorted_df.iloc[index]
    row_next = sorted_df.iloc[index + 1]
    
    # Rule 1: CID match
    cid_match = (row_curr['customer_id'] == row_next['customer_id'] and row_curr['customer_id'] != "")
    
    # Rule 2: 2 out of 3 match
    n_m = int(row_curr['customer_name'] == row_next['customer_name'] and row_curr['customer_name'] != "")
    e_m = int(row_curr['email'] == row_next['email'] and row_curr['email'] != "")
    p_m = int(row_curr['phone_primary'] == row_next['phone_primary'] and row_curr['phone_primary'] != "")
    
    if cid_match or (n_m + e_m + p_m >= 2):
        # Merge row_next into row_curr
        for col in sorted_df.columns:
            # Only combine if they are actually different to avoid "John/John"
            if row_curr[col] != row_next[col]:
                new_val = f"{row_curr[col]} / {row_next[col]}"
                sorted_df.at[index, col] = new_val
        
        # Drop the next row and DO NOT increment index 
        # (we stay on the same index to check if the NEW merged row matches the NEXT one)
        sorted_df = sorted_df.drop(sorted_df.index[index + 1]).reset_index(drop=True)
    else:
        # No match, move to the next row
        index += 1

print(sorted_df)

