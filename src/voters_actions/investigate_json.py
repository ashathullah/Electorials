
import os

filepath = r"e:\Raja_mohaemd\projects\voter-shield-data-cleanup\invalid_voter_sections_json\Tamil Nadu-(S22)_Manachanallur-(AC144)_93_voters.json"

with open(filepath, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for i in range(5):
        print(f"Line {i}: {repr(lines[i])}")
