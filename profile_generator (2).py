import random
import csv

# Education levels
education_levels = ["middle school", "high school", "university", "postgraduate"]
education_rank = {lvl: i for i, lvl in enumerate(education_levels)}

# Load Occupations from CSV
def load_occupations(file_path="occupations.csv"):
    occupations = []
    with open(file_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            occupations.append({
                "name": row["Occupation Name"],
                "min_age": int(row["Minimum Age"]),
                "min_education": row["Minimum Education"].strip().lower()
            })
    return occupations

# Load Interests from CSV
def load_interests(file_path="interests.csv"):
    with open(file_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return [row["Interests"] for row in reader]

# Load Subreddits from CSV
def load_subreddits(file_path="subreddits.csv"):
    with open(file_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return [row["Subreddits"] for row in reader]


# Load Nationalities from CSV
def load_nationalities(file_path="nationalities.csv"):
    with open(file_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return [row["Countries"] for row in reader]

# STEP 1.Gender selection
def select_gender():
    return random.choices(
        ["male", "female", "non-binary"],
        weights=[61.2, 37.8, 1],
        k=1
    )[0]

# STEP 2. Age selection
age_distribution = {
    (13, 17): 0.01,
    (18, 24): 0.1488,
    (25, 34): 0.2309,
    (35, 44): 0.1934,
    (45, 54): 0.1768,
    (55, 64): 0.1306,
    (65, 90): 1.0 - (0.01 + 0.1488 + 0.2309 + 0.1934 + 0.1768 + 0.1306),
}

def select_age():
    ranges = list(age_distribution.keys())
    weights = list(age_distribution.values())
    selected_range = random.choices(ranges, weights=weights, k=1)[0]
    return random.randint(*selected_range)


# STEP 3. Nationality selection
def select_nationality(countries_list):
    # Explicit country weights
    major_distribution = {
        "USA": 52.5, "UK": 8.4, "Canada": 8.3, "Australia": 4.5,
        "Germany": 2.1, "India": 1.3, "France": 0.9, "Netherlands": 0.7,
        "Brazil": 0.7, "Philippines": 0.7, "Singapore": 0.6, "Italy": 0.5,
        "Sweden": 0.4, "Spain": 0.4, "Ireland": 0.4, "Finland": 0.3,
        "Poland": 0.3, "Norway": 0.3, "Malaysia": 0.3, "South Korea": 0.3,
        "New Zealand": 0.2, "Denmark": 0.2, "Indonesia": 0.2,
        "Belgium": 0.2, "Portugal": 0.2
    }

    total_specified = sum(major_distribution.values())
    remaining_pct = 100.0 - total_specified

    other_countries = [c for c in countries_list if c not in major_distribution]
    equal_share = remaining_pct / len(other_countries) if other_countries else 0

    weights = []
    for country in countries_list:
        if country in major_distribution:
            weights.append(major_distribution[country])
        else:
            weights.append(equal_share)

    return random.choices(countries_list, weights=weights, k=1)[0]


# STEP 4. Marital status selection
def select_marital_status(age):
    if age < 18:
        return "never married"
    return random.choices(["never married", "married"], weights=[45, 55], k=1)[0]



# STEP 5. Education level based on age
def get_possible_education_levels(age):
    if age < 17:
        return ["middle school"]
    elif age < 21:
        return ["middle school", "high school"]
    elif age < 24:
        return ["middle school", "high school", "university"]
    else:
        return education_levels

# STEP 6. Filter occupations by age + education
def get_valid_occupations(age, education, occupations):
    return [
        job["name"] for job in occupations
        if age >= job["min_age"]
        and education_rank[education] >= education_rank[job["min_education"]]
    ]

# STEP 7. Random interests
def select_interests(interests_list):
    return random.sample(interests_list, k=random.randint(3, 10))

# STEP 8. Random subreddit
def select_subreddit(subreddits_list):
    return random.choice(subreddits_list)

# STEP 9. Generate profile
def generate_profile(occupations, interests, subreddits, countries):
    """
    Output: Dictionary with the following keys:
    - {"gender": str}
    - {"age": int}
    - {"education": str}
    - {"occupation": str}
    - {"interests": list[str]}
    - {"subreddit": str}
    - {"nationality": str}
    - {"marital_status": str}
    """
    gender = select_gender()
    age = select_age()
    education = random.choice(get_possible_education_levels(age))
    valid_jobs = get_valid_occupations(age, education, occupations)
    occupation = random.choice(valid_jobs) if valid_jobs else "Unemployed"
    interests_selected = select_interests(interests)
    subreddit = select_subreddit(subreddits)
    nationality = select_nationality(countries)
    marital_status = select_marital_status(age)

    return {
        "gender": gender,
        "age": age,
        "education": education,
        "occupation": occupation,
        "interests": interests_selected,
        "subreddit": subreddit,
        "nationality": nationality,
        "marital_status": marital_status
    }


# Load external data
# occupations_data = load_occupations()
# interests_data = load_interests()
# subreddits_data = load_subreddits()
# countries_data = load_nationalities()

""" EXAMPLE OUTPUT:
{
    'gender': 'non-binary',
    'age': 34,
    'education': 'university',
    'occupation': 'UX Designer',
    'interests': ['coding', 'photography', 'gaming', 'travel'],
    'subreddit': 'r/science',
    'nationality': 'Netherlands',
    'marital_status': 'married'
}
"""