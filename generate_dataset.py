# WORK IN PROGRESS, NEED OPENAI CREDITS TO CONTINUE

import openai

# create a file names 'api_key.txt' and on the first line place your openai API key.
openai.api_key = open('./api_key.txt', "r").read()
print("===== You loaded the {} API key =====".format(openai.api_key))

userinput = "Tell me a story about an ice cream shop, using no less than 100 words."
response = openai.Completion.create(model="text-davinci-003", 
                                    messages=[{"role": "system", "content": "You are a helpful assistant."},
                                              {"role": "user", "content": userinput}])

print(response)