import json
import re
import csv
from collections import defaultdict

INPUT_FILE = "combined_mhc_control.jsonl"   # Adjust to your file
OUTPUT_FILE = "auto_demographics.csv"

# --- Define regex patterns ---
AGE_PATTERNS = [
    (re.compile(r"\bI[' ]?m (\d{1,2})\b", re.IGNORECASE), "I'm {age}"),
    (re.compile(r"\bI am (\d{1,2})\b", re.IGNORECASE), "I am {age}"),
    (re.compile(r"\bI was (\d{1,2})\b", re.IGNORECASE), "I was {age}"),
    (re.compile(r"\b(\d{1,2})\s?[mMfF]\b"), "{age}M/F"),
    (re.compile(r"\bI[' ]?m a (\d{1,2})[- ]?(year[- ]old|yo)\b", re.IGNORECASE), "I'm a {age} year old"),
    (re.compile(r"\bturned (\d{1,2}) (today|last week|this week)\b", re.IGNORECASE), "Turned {age} recently"),
    (re.compile(r"\bjust turned (\d{1,2})\b", re.IGNORECASE), "Just turned {age}"),
    (re.compile(r"\bwhen I was (?:about|around)? ?(\d{1,2})\b", re.IGNORECASE), "When I was {age}"),
    (re.compile(r"\bin (\d{1,2})(st|nd|rd|th)? grade\b", re.IGNORECASE), "In {age}th grade"),
    (re.compile(r"\bstarted high school at (\d{1,2})\b", re.IGNORECASE), "Started high school at {age}"),
    (re.compile(r"\bi[' ]?m (a )?teenager\b", re.IGNORECASE), "Teenager"),
    (re.compile(r"\b(i[' ]?m|being) a minor\b", re.IGNORECASE), "Minor"),
    (re.compile(r"\bi[' ]?m under\s+(\d{2})\b", re.IGNORECASE), "Under {age}"),
]

GENDER_PATTERNS = [
    (re.compile(r"\bI[' ]?m (?:a[n]? )?(guy|boy|man|male|female|woman|girl)\b", re.IGNORECASE), "I am a {gender}"),
    (re.compile(r"\bI[' ]?m (?:a[n]? )?trans(?:gender)? (man|woman|guy|girl)\b", re.IGNORECASE), "I'm a trans {gender}"),
    (re.compile(r"\bmy pronouns are (he\/him|she\/her|they\/them|xe\/xem|ze\/zir)\b", re.IGNORECASE), "Pronouns: {gender}"),
    (re.compile(r"\bi[' ]?m transgender\b", re.IGNORECASE), "Transgender"),
    (re.compile(r"\bi[' ]?m non[- ]?binary\b", re.IGNORECASE), "Nonbinary"),
    (re.compile(r"\bi[' ]?m (cisgender|cis|trans|genderqueer)\b", re.IGNORECASE), "Gender identity: {gender}"),
    (re.compile(r"\bi identify as (?:a[n]? )?(gay|bi|bisexual|lesbian|asexual|pansexual|homoromantic)\b", re.IGNORECASE), "Orientation: {gender}"),
    (re.compile(r"\bI[' ]?m attracted to (men|women|both|everyone)\b", re.IGNORECASE), "Attracted to: {gender}"),
    (re.compile(r"\bwe[' ]?re both (bi|gay|trans|nonbinary)\b", re.IGNORECASE), "We are both {gender}"),
    (re.compile(r"\bi feel like i'?m in between (male|female|nonbinary)\b", re.IGNORECASE), "In between {gender}"),
    (re.compile(r"\bi[' ]?m a queer (teenager|man|woman)\b", re.IGNORECASE), "Queer {gender}"),
    (re.compile(r"\bi[' ]?m not (a )?girl\b", re.IGNORECASE), "Not a girl"),
]

EDUCATION_PATTERNS = [
    (re.compile(r"\b(studying|study|studied) (at|in)? ?.*(college|university|school)\b", re.IGNORECASE), "Studying at {edu}"),
    (re.compile(r"\b(graphic design|psychology|comp[uter]* science|nursing|engineering|[a-z ]+) student\b", re.IGNORECASE), "{edu} student"),
    (re.compile(r"\b(in|at) (high school|college|university)\b", re.IGNORECASE), "Enrolled in {edu}"),
    (re.compile(r"\bgraduated with (?:a|an) (BA|MA|PhD|associate['’]s|bachelor['’]s|master['’]s|[a-z ]+)\b", re.IGNORECASE), "Graduated with {edu}"),
    (re.compile(r"\b(dropped out of|finished) (high school|college|university)\b", re.IGNORECASE), "{edu} status"),
    (re.compile(r"\b(in my )?(freshman|sophomore|junior|senior) year\b", re.IGNORECASE), "{edu} year"),
    (re.compile(r"\bcurrently in (high school|college|university)\b", re.IGNORECASE), "Currently in {edu}"),
    (re.compile(r"\b(taking|took) a gap year\b", re.IGNORECASE), "Gap year"),
    (re.compile(r"\bdoing a (research paper|project|thesis)\b", re.IGNORECASE), "Academic activity: {edu}"),
]

# --- Load and group posts ---
user_posts = defaultdict(lambda: {"group": None, "text": ""})

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        post = json.loads(line)
        uid = post["user_id"]
        group = post.get("group", "Unknown")  # e.g., "MHC" or "Control"
        title = post.get("title", "")
        body = post.get("text", "")
        user_posts[uid]["group"] = group
        user_posts[uid]["text"] += " " + title + " " + body

# --- Extraction logic ---
def extract_demographics(text):
    age, gender, edu = None, None, None
    notes = []

    for pattern, template in AGE_PATTERNS:
        match = pattern.search(text)
        if match:
            age = match.group(1)
            notes.append(f"Age: {match.group(0)}")
            break

    for pattern, template in GENDER_PATTERNS:
        match = pattern.search(text)
        if match:
            gender = match.group(1).capitalize()
            notes.append(f"Gender: {match.group(0)}")
            break

    for pattern, template in EDUCATION_PATTERNS:
        match = pattern.search(text)
        if match:
            edu = match.group(2) if len(match.groups()) > 1 else match.group(1)
            edu = edu.strip().capitalize()
            notes.append(f"Edu: {match.group(0)}")
            break

    return age, gender, edu, " | ".join(notes)

# --- Write output CSV ---
with open(OUTPUT_FILE, "w", newline='', encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["UserID", "Group", "Age", "Gender", "Education Level", "Notes"])

    for uid, data in user_posts.items():
        age, gender, edu, notes = extract_demographics(data["text"])
        writer.writerow([uid, data["group"], age or "", gender or "", edu or "", notes])

print(f"[✓] Demographic detection saved to {OUTPUT_FILE}")
