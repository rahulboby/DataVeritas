import numpy as np
import pandas as pd
import random
from faker import Faker
import streamlit as st

@st.cache_data(show_spinner="Generating synthetic dataset...")
def get_data(num_rows=10):
    return generate_base_data(num_rows)

def generate_base_data(num_rows=10):
    fake = Faker()
    # Set seeds for reproducibility
    np.random.seed(42)
    random.seed(42)

    # --- TYPO HELPERS ---
    def transpose_chars(s):
        if len(s) < 2: return s
        i = random.randrange(len(s) - 1)
        return s[:i] + s[i+1] + s[i] + s[i+2:]

    def delete_char(s):
        if len(s) < 2: return s
        i = random.randrange(len(s))
        return s[:i] + s[i+1:]

    def duplicate_char(s):
        if len(s) < 1: return s
        i = random.randrange(len(s))
        return s[:i] + s[i] + s[i:]

    def introduce_typo_name(fullname):
        parts = fullname.split()
        if not parts: return fullname
        first = parts[0]
        method = random.random()
        base = first.lower()
        if method < 0.6: typo = transpose_chars(base)
        elif method < 0.8: typo = delete_char(base)
        else: typo = duplicate_char(base)
        parts[0] = typo.capitalize()
        return " ".join(parts)

    def introduce_typo_email(email):
        if not isinstance(email, str) or '@' not in email: return email
        local, domain = email.split('@', 1)
        segments = local.split('.')
        seg_idx = random.randrange(len(segments))
        seg = segments[seg_idx]
        segments[seg_idx] = transpose_chars(seg) if len(seg) >= 2 else duplicate_char(seg)
        return ".".join(segments) + "@" + domain

    # --- 1. MAPPINGS ---
    vehicle_logic = {
        # --- Trucks & Pickups ---
        'F-150': {'class': 'Truck', 'variants': ['XL', 'XLT', 'Lariat', 'King Ranch', 'Platinum', 'Raptor'], 'base_msrp': 38000},
        'F-250': {'class': 'Heavy Duty Truck', 'variants': ['XL', 'XLT', 'Lariat', 'King Ranch', 'Platinum', 'Limited'], 'base_msrp': 45000},
        'Ranger': {'class': 'Mid-Size Truck', 'variants': ['XL', 'XLT', 'Lariat', 'Raptor'], 'base_msrp': 32000},
        'Maverick': {'class': 'Compact Pickup', 'variants': ['XL', 'XLT', 'Lariat', 'Tremor'], 'base_msrp': 23900},
        
        # --- SUVs & Off-Road ---
        'Explorer': {'class': 'SUV', 'variants': ['Base', 'XLT', 'ST-Line', 'Timberline', 'Limited', 'ST', 'Platinum'], 'base_msrp': 39000},
        'Expedition': {'class': 'Full-Size SUV', 'variants': ['XL STX', 'XLT', 'Limited', 'King Ranch', 'Platinum'], 'base_msrp': 55000},
        'Bronco': {'class': 'SUV (Off-Road)', 'variants': ['Big Bend', 'Black Diamond', 'Outer Banks', 'Badlands', 'Wildtrak', 'Everglades', 'Raptor'], 'base_msrp': 35000},
        'Bronco Sport': {'class': 'Compact SUV', 'variants': ['Big Bend', 'Heritage', 'Free Wheeling', 'Outer Banks', 'Badlands'], 'base_msrp': 29500},
        'Edge': {'class': 'Mid-Size SUV', 'variants': ['SE', 'SEL', 'ST-Line', 'Titanium', 'ST'], 'base_msrp': 38000},
        'Escape': {'class': 'Crossover', 'variants': ['Active', 'ST-Line', 'Platinum', 'PHEV'], 'base_msrp': 29000},
        
        # --- Electric Vehicles ---
        'Mach-E': {'class': 'SUV (EV)', 'variants': ['Select', 'Premium', 'California Route 1', 'GT', 'Rally'], 'base_msrp': 46000},
        'F-150 Lightning': {'class': 'Truck (EV)', 'variants': ['Pro', 'XLT', 'Flash', 'Lariat', 'Platinum'], 'base_msrp': 49900},
        
        # --- Performance & Specialty ---
        'Mustang': {'class': 'Sports Car', 'variants': ['EcoBoost', 'GT', 'Dark Horse', 'GT Premium'], 'base_msrp': 32000},
        'Ford GT': {'class': 'Supercar', 'variants': ['Heritage Edition', 'Liquid Carbon', 'Studio Collection'], 'base_msrp': 500000},
        
        # --- Vans & Commercial ---
        'Transit': {'class': 'Commercial Van', 'variants': ['Cargo Van', 'Passenger Van XL', 'Passenger Van XLT'], 'base_msrp': 46000},
        'Transit Connect': {'class': 'Compact Van', 'variants': ['XL', 'XLT', 'Titanium'], 'base_msrp': 35000},
        'E-Transit': {'class': 'Commercial Van (EV)', 'variants': ['Cargo', 'Chassis Cab', 'Cutaway'], 'base_msrp': 49000}
    }

    models = list(vehicle_logic.keys())

    # --- 2. CUSTOMER & DEALER DATA ---
    names = [fake.name() for _ in range(num_rows)]
    # names = [
    #     "Rahul",
    #     "Boby",
    #     "Maneesh",
    #     "Ancy",
    #     "Sithany",
    #     "Eli",
    #     "Xavier",
    #     "Saji",
    #     "Max",
    #     "Schumacher",
    #     "Hamilton",
    #     "Lewis",
    #     "Nico",
    #     "Rosberg",
    #     "Michael",
    #     "Jacob",
    #     "Charlie",
    #     "Leclerc",
    #     "Jon",
    #     "Jones",
    #     "Maximus",
    #     "Dominic",
    #     "Han",
    #     "Hobs"
    # ]
    emails = [f"{n.replace(' ', '.').lower()}@{fake.free_email_domain()}" for n in names]
    
    # Inject standard typos
    typo_indices = random.sample(range(num_rows), int(num_rows * 0.015))
    for i in typo_indices:
        names[i] = introduce_typo_name(names[i])
        emails[i] = introduce_typo_email(emails[i])

    data = {
        'customer_id': [f"CUST-{i:06d}" for i in range(num_rows)],
        'customer_name': names,
        'email': emails,
        'street_address': [fake.street_address() for _ in range(num_rows)],
        'city': [fake.city() for _ in range(num_rows)],
        'state': [fake.state_abbr() for _ in range(num_rows)],
        'postal_code': [fake.postcode() for _ in range(num_rows)],
        'phone_primary': [fake.phone_number() for _ in range(num_rows)],
        'phone_secondary': [fake.phone_number() if random.random() > 0.7 else None for _ in range(num_rows)],
        'country': np.random.choice(['USA', 'Canada', 'UK', 'Germany', 'Mexico'], num_rows),
        'loyalty_member': np.random.choice([True, False], num_rows, p=[0.3, 0.7]),
        'dealer_name': [f"{fake.last_name()} Ford" for _ in range(num_rows)],
        'dealer_city': [fake.city() for _ in range(num_rows)],
        'dealership_id': [f"DLR-{np.random.randint(100, 999)}" for _ in range(num_rows)]
    }

    # --- 3. VEHICLE SPECS ---
    chosen_models = np.random.choice(models, num_rows)
    engine_cc = np.random.choice([1000, 1400, 1498, 1500, 1890, 1995, 2000, 2134, 2389, 2487, 2500, 2788, 3500, 3800, 3998, 4120, 4389, 4597, 4809, 4993], num_rows)
    hp = (engine_cc / 10) + np.random.normal(20, 15, num_rows) 
    fuel_types = np.random.choice(['Gasoline', 'Diesel', 'Electric', 'Hybrid'], num_rows)

    data.update({
        'vin': [fake.bothify(text='#?#??##???#######').upper() for _ in range(num_rows)],
        'make': 'Ford',
        'model': chosen_models,
        'vehicle_class': [vehicle_logic[m]['class'] for m in chosen_models],
        'variant_name': [random.choice(vehicle_logic[m]['variants']) for m in chosen_models],
        'model_year': np.random.randint(2018, 2026, num_rows),
        'engine_cc': engine_cc,
        'horsepower': hp.astype(int),
        'torque_nm': (hp * 1.2 + 10).astype(int),
        'fuel_type': fuel_types,
        'transmission': np.random.choice(['Automatic', 'Manual', 'CVT'], num_rows),
        'drivetrain': np.random.choice(['AWD', 'FWD', 'RWD', '4WD'], num_rows),
        'exterior_color': [fake.color_name() for _ in range(num_rows)],
        'is_ev': [True if x in ['Electric', 'Hybrid'] else False for x in fuel_types]
    })

    # --- 4. SALES ---
    base_msrps = np.array([vehicle_logic[m]['base_msrp'] for m in chosen_models])
    msrp = base_msrps + (engine_cc * 2) + np.random.randint(2000, 8000, num_rows)
    market_adj = np.random.normal(1000, 2000, num_rows)

    data.update({
        'order_number': [f"ORD-{i:08d}" for i in range(num_rows)],
        'msrp': msrp.astype(float),
        'market_adjustment': market_adj.round(2),
        'discount': np.random.uniform(0, 5000, num_rows).round(2),
        'order_date': [fake.date_between(start_date='-2y', end_date='today') for _ in range(num_rows)],
        'shipping_method': np.random.choice(['Rail', 'Truck', 'Sea'], num_rows),
        'inventory_status': np.random.choice(['In-Stock', 'In-Transit', 'Ordered'], num_rows),
        'warranty_years': np.random.choice([3, 5, 7], num_rows),
    })

    df = pd.DataFrame(data)
    df['order_date'] = pd.to_datetime(df['order_date'])
    df['delivery_date'] = df['order_date'] + pd.to_timedelta(np.random.randint(7, 45, size=num_rows), unit='D')
    df['sale_price'] = df['msrp'] + df['market_adjustment'] - df['discount']

    # --- 5. FIXED SENSOR LOGS ---
    years_old = 2025 - df['model_year']
    df['odometer_km'] = (years_old * np.random.normal(15000, 5000, num_rows)).clip(lower=5).astype(int)
    df['battery_voltage'] = np.random.normal(12.6, 0.5, num_rows).round(2)
    df['tire_pressure_psi'] = np.random.normal(32, 2, num_rows).round(1)
    df['oil_life_pct'] = (100 - (df['odometer_km'] % 12000) / 120).clip(lower=0, upper=100).round(1)
    df['coolant_temp_c'] = np.random.gamma(shape=90, scale=1, size=num_rows).round(1)
    df['avg_fuel_cons_l100k'] = ((df['engine_cc'] / 500) + np.random.uniform(2, 5, num_rows)).round(1)
    df.loc[df['is_ev'] == True, 'avg_fuel_cons_l100k'] = 0.0
    df['brake_wear_pct'] = (100 - (df['odometer_km'] / 1500)).clip(lower=0, upper=100).round(1)
    df['fuel_level_pct'] = np.random.uniform(0.05, 1.0, num_rows).round(2)
    df['engine_load_pct'] = (np.random.beta(a=2, b=5, size=num_rows) * 100).round(1)
    df['last_service_date'] = df['order_date'] + pd.to_timedelta(np.random.randint(180, 500, size=num_rows), unit='D')
    
    df['software_version'] = np.random.choice(['v1.2', 'v1.4.2', 'v2.0.1', 'v2.1'], num_rows)
    df['tire_brand'] = np.random.choice(['Michelin', 'Goodyear', 'Bridgestone', 'Continental'], num_rows)
    df['is_certified_preowned'] = (df['model_year'] < 2023) & (np.random.random(num_rows) > 0.5)

    # --- 6. DATA QUALITY CORRUPTION (Standard) ---
    total_rows = len(df)
    
    # Key Collisions (Order Number)
    coll_idx = np.random.choice(df.index, size=int(total_rows * 0.01), replace=False)
    df.loc[coll_idx, 'order_number'] = df.loc[np.random.choice(df.index, len(coll_idx)), 'order_number'].values

    # Nulls
    df.loc[df.sample(frac=0.05).index, 'email'] = np.nan
    df.loc[df.sample(frac=0.10).index, 'engine_cc'] = np.nan
    df.loc[df.sample(frac=0.02).index, 'vin'] = np.nan
    df.loc[df.sample(frac=0.01).index, 'delivery_date'] = pd.NaT

    # Logical Inconsistencies
    bad_math_idx = np.random.choice(df.index, 200, replace=False)
    df.loc[bad_math_idx, 'sale_price'] = df.loc[bad_math_idx, 'msrp'] + 9999 
    
    bad_date_idx = np.random.choice(df.index, 150, replace=False)
    df.loc[bad_date_idx, 'delivery_date'] = df.loc[bad_date_idx, 'order_date'] - pd.to_timedelta(10, unit='D')

    # Outliers
    df.loc[df.sample(frac=0.002).index, 'horsepower'] *= 10
    df.loc[df.sample(frac=0.001).index, 'msrp'] *= 50

    # # Categorical/Type Corruption
    # df.loc[df.sample(frac=0.01).index, 'transmission'] = 'UNKNOWN'
    # bad_type_idx = np.random.choice(df.index, size=int(total_rows * 0.01), replace=False)
    # df.loc[bad_type_idx, 'engine_cc'] = np.nan

    # String Corruption
    vin_corrupt_idx = np.random.choice(df.index, size=int(total_rows * 0.005), replace=False)
    for i in vin_corrupt_idx:
        if pd.notna(df.at[i, 'vin']): df.at[i, 'vin'] = "ERR!" + str(df.at[i, 'vin'])[4:]

    # Checksum & Swaps
    df['checksum'] = (pd.factorize(df['order_number'])[0] * 31) % 1_000_000
    
    # --- 7. IDENTITY & MULTI-CID INCONSISTENCIES (NEW) ---
    # We will create a "dirty" subset of rows that represent the same person with different IDs
    inconsistency_frames = []

    # Scenario A: Same Name & Phone, but different Email and different CID
    # (Simulates a customer creating a second account with a new email)
    idx_a = df.sample(n=int(num_rows * 0.01)).index
    df_a = df.loc[idx_a].copy()
    df_a['customer_id'] = [f"CUST-NEW-{random.getrandbits(24)}" for _ in range(len(df_a))]
    df_a['email'] = [f"alt.{e}" if pd.notna(e) else e for e in df_a['email']]
    df_a['order_number'] = [f"ORD-ALT-{random.getrandbits(24)}" for _ in range(len(df_a))]
    inconsistency_frames.append(df_a)

    # Scenario B: Same Email & Phone, but slightly different Name and different CID
    # (Simulates missing middle initials or shortened names)
    idx_b = df.sample(n=int(num_rows * 0.01)).index
    df_b = df.loc[idx_b].copy()
    df_b['customer_id'] = [f"CUST-NEW-{random.getrandbits(24)}" for _ in range(len(df_b))]
    
    def strip_initials(name):
        parts = name.split()
        if len(parts) > 2:
            return f"{parts[0]} {parts[-1]}" # Remove middle name/initial
        return parts[0] # Just first name
    
    df_b['customer_name'] = df_b['customer_name'].apply(strip_initials)
    df_b['order_number'] = [f"ORD-ALT-{random.getrandbits(24)}" for _ in range(len(df_b))]
    inconsistency_frames.append(df_b)

    # Scenario C: Same Name & Email, but different Phone and different CID
    idx_c = df.sample(n=int(num_rows * 0.005)).index
    df_c = df.loc[idx_c].copy()
    df_c['customer_id'] = [f"CUST-NEW-{random.getrandbits(24)}" for _ in range(len(df_c))]
    df_c['phone_primary'] = [fake.phone_number() for _ in range(len(df_c))]
    df_c['order_number'] = [f"ORD-ALT-{random.getrandbits(24)}" for _ in range(len(df_c))]
    inconsistency_frames.append(df_c)

    # Append these inconsistent records back to the main dataframe
    df = pd.concat([df] + inconsistency_frames, ignore_index=True)

    # --- 8. DUPLICATION ---
    dup_indices = np.random.choice(df.index, size=int(len(df) * 0.02), replace=False)
    df_dupes = df.loc[np.repeat(dup_indices, np.random.randint(2, 4))].copy()
    df = pd.concat([df, df_dupes], ignore_index=True)

    # Final shuffle
    return df.sample(frac=1).reset_index(drop=True)
