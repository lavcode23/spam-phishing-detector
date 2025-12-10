import re

def extract_features(text):
    url_count = len(re.findall(r"http[s]?://", text))
    domain_flag = 1 if re.search(r"\.ru|\.cn|\.tk|\.xyz", text.lower()) else 0
    risky_count = len(re.findall(r"(verify|password|click|login|urgent|bank|update)", text.lower()))
    digit_count = sum(c.isdigit() for c in text)
    length = len(text)

    return [url_count, domain_flag, risky_count, digit_count, length]
