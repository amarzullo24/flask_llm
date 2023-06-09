import pandas as pd
import requests
import json

# Define the URL of the Flask server
url = "http://localhost:5000/answer"

def query(message):
    # Prepare the message payload
    message_template = "{}" # customize your message
    payload = {"message": message_template.format(message)}

    # Send the POST request
    response = requests.post(url, json=payload)

    # Get the response from the server
    data = response.json()
    generated_response = data["response"]

    # return the generated response
    return generated_response

# Everything begins
message = "EXAMPLE MESSAGE"

response = query(message)
print(response)

print("And that's all Folks!")

