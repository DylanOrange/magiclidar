import json

# Open the JSON file for reading
with open('OpenSource/final_flickr_mergedGT_train.json', 'r') as file:
    data = json.load(file)

# Now you can work with the parsed data (usually a dictionary or a list)
print(data)
