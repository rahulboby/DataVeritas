import numpy as np
import pandas as pd
import random
from faker import Faker


fake = Faker()
Faker.seed(42)
random.seed(42)
np.random.seed(42)


def get_data(num_rows=10):
    """Generate synthetic data for testing. Entry point."""
    return generate_base_data(num_rows)


def generate_base_data(num_rows=10):
    """
    Generate realistic synthetic customer/order data with intentional quality issues.
    Includes typos, near-duplicates, null values, and inconsistencies for testing.
    """

    def transpose_chars(s):
        if len(s) < 2:
            return s
        idx = random.randint(0, len(s) - 2)
        return s[:idx] + s[idx + 1] + s[idx] + s[idx + 2:]

    def delete_char(s):
        if len(s) < 2:
            return s
        idx = random.randint(0, len(s) - 1)
        return s[:idx] + s[idx + 1:]

    def duplicate_char(s):
        idx = random.randint(0, len(s) - 1)
        return s[:idx] + s[idx] + s[idx:]

    def introduce_typo_name(fullname):
        parts = fullname.split()
        i = random.randint(0, len(parts) - 1)
        op = random.choice([transpose_chars, delete_char, duplicate_char])
        parts[i] = op(parts[i])
        return " ".join(parts)

    def introduce_typo_email(email):
        if "@" not in email:
            return email
        local, domain = email.split("@", 1)
        op = random.choice([transpose_chars, delete_char, duplicate_char])
        return f"{op(local)}@{domain}"

    # Vehicle reference data
    vehicle_makes_models = {
        "Toyota": ["Camry", "Corolla", "RAV4", "Highlander", "Tacoma"],
        "Honda": ["Civic", "Accord", "CR-V", "Pilot", "HR-V"],
        "Ford": ["F-150", "Mustang", "Explorer", "Escape", "Bronco"],
        "Chevrolet": ["Silverado", "Malibu", "Equinox", "Traverse", "Camaro"],
        "BMW": ["3 Series", "5 Series", "X3", "X5", "7 Series"],
        "Mercedes": ["C-Class", "E-Class", "GLC", "GLE", "S-Class"],
        "Hyundai": ["Elantra", "Sonata", "Tucson", "Santa Fe", "Palisade"],
        "Kia": ["Forte", "K5", "Sportage", "Sorento", "Telluride"],
        "Tesla": ["Model 3", "Model Y", "Model S", "Model X"],
        "Nissan": ["Altima", "Sentra", "Rogue", "Pathfinder", "Frontier"]
    }

    fuel_types = ["gasoline", "diesel", "electric", "hybrid"]
    transmissions = ["automatic", "manual", "cvt"]
    colors = ["White", "Black", "Silver", "Gray", "Blue", "Red", "Green", "Brown"]
    states = ["CA", "TX", "FL", "NY", "IL", "PA", "OH", "GA", "NC", "MI",
              "NJ", "VA", "WA", "AZ", "MA", "TN", "IN", "MO", "MD", "WI"]

    base_count = int(num_rows * 0.80)
    dup_count = num_rows - base_count

    records = []

    for i in range(base_count):
        cid = f"C{i + 1:05d}"
        name = fake.name()
        email = fake.email()
        phone1 = fake.phone_number()
        phone2 = fake.phone_number() if random.random() > 0.4 else None

        make = random.choice(list(vehicle_makes_models.keys()))
        model = random.choice(vehicle_makes_models[make])
        year = random.randint(2015, 2024)
        fuel = random.choice(fuel_types)
        trans = "automatic" if fuel == "hybrid" else random.choice(transmissions)
        color = random.choice(colors)
        mileage = random.randint(0, 150000)
        price = round(random.uniform(15000, 85000), 2)
        order_num = f"ORD-{i + 1:06d}"

        order_date = fake.date_between(start_date='-3y', end_date='-1y')
        delivery_date = fake.date_between(start_date=order_date, end_date='-6m')
        service_date = fake.date_between(start_date=delivery_date, end_date='today')

        street = fake.street_address()
        city = fake.city()
        state = random.choice(states)
        zipcode = fake.zipcode()

        record = {
            "customer_id": cid,
            "customer_name": name,
            "email": email,
            "phone_primary": phone1,
            "phone_secondary": phone2,
            "street_address": street,
            "city": city,
            "state": state,
            "zipcode": zipcode,
            "vehicle_make": make,
            "vehicle_model": model,
            "vehicle_year": year,
            "vehicle_color": color,
            "fuel_type": fuel,
            "transmission": trans,
            "mileage": mileage,
            "price": price,
            "order_number": order_num,
            "order_date": order_date,
            "delivery_date": delivery_date,
            "last_service_date": service_date
        }

        # Inject some nulls (~5% chance per field)
        for key in record:
            if key != "customer_id" and random.random() < 0.05:
                record[key] = None

        records.append(record)

    # Generate near-duplicates
    for _ in range(dup_count):
        source = random.choice(records[:base_count]).copy()

        # Vary the duplicate
        variation = random.random()
        if variation < 0.3:
            # Same CID, slightly different details
            if source["customer_name"]:
                source["customer_name"] = introduce_typo_name(source["customer_name"])
        elif variation < 0.6:
            # Different CID, same name/email
            source["customer_id"] = f"C{random.randint(base_count + 1, base_count + dup_count + 1000):05d}"
        else:
            # Typo in email
            source["customer_id"] = f"C{random.randint(base_count + 1, base_count + dup_count + 1000):05d}"
            if source["email"]:
                source["email"] = introduce_typo_email(source["email"])

        records.append(source)

    # Inject some date ordering violations (~2%)
    for record in records:
        if random.random() < 0.02:
            record["order_date"], record["delivery_date"] = record["delivery_date"], record["order_date"]

    # Inject hybrid + manual violations (~1%)
    for record in records:
        if record.get("fuel_type") == "hybrid" and random.random() < 0.1:
            record["transmission"] = "manual"

    df = pd.DataFrame(records)
    return df
