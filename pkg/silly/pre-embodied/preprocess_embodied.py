import json
import os

# Path to your JSON file
file_path = "embodied.json"

# Read the JSON file
with open(file_path, "r") as f:
    data = json.load(f)

# Round the values in each entry
for item in data:
    # Round embodied emissions to 3 decimal points
    item["average_embodied_emissions"] = round(item["average_embodied_emissions"], 3)
    
    # Round lifetime to 2 decimal points
    item["average_lifetime"] = round(item["average_lifetime"], 2)

# Write the rounded data back to the file
with open(file_path, "w") as f:
    # Use indent for pretty formatting and ensure_ascii=False for proper Unicode handling
    json.dump(data, f, indent=4, ensure_ascii=False)

print(f"Successfully updated aggregate embodfied emissions data in {file_path}")