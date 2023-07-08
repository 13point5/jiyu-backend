import re


def extract_mention_ids(text):
    pattern = r"<@block:(\w+)>"
    mentions = re.findall(pattern, text)
    return mentions
