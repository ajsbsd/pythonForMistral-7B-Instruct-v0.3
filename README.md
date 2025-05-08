# pythonForMistral-7B-Instruct-v0.3
HTML BLOCK

<style>
  .flask-widget {
    background-color: #2c3e50;
    padding: 2rem;
    border-radius: 12px;
    max-width: 500px;
    margin: auto;
    color: white;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
  }

  .flask-widget input[type="text"] {
    width: 100%;
    padding: 0.75rem;
    margin-bottom: 1rem;
    border: none;
    border-radius: 8px;
    font-size: 1rem;
  }

  .flask-widget button {
    padding: 0.75rem 1.5rem;
    background-color: #3498db;
    border: none;
    border-radius: 8px;
    color: white;
    font-size: 1rem;
    cursor: pointer;
    transition: background-color 0.3s;
  }

  .flask-widget button:hover {
    background-color: #2980b9;
  }

  #flask-response {
    margin-top: 1rem;
    font-style: italic;
    color: #ecf0f1;
  }
</style>

<div class="flask-widget">
  <form id="flask-form">
    <input type="text" id="user-prompt" placeholder="Enter your prompt" required />
    <button type="submit">Send</button>
  </form>

  <div id="flask-response">Awaiting response...</div>

  <script>
    document.getElementById("flask-form").addEventListener("submit", function (e) {
      e.preventDefault();

      const prompt = document.getElementById("user-prompt").value;
      const responseDiv = document.getElementById("flask-response");
      responseDiv.innerText = "Loading...";

      fetch("https://ajsbsd.net/flask/", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ prompt: prompt })
      })
      .then(response => {
        if (!response.ok) {
          throw new Error("HTTP error " + response.status);
        }
        return response.text();
      })
      .then(data => {
        responseDiv.innerText = data;
      })
      .catch(error => {
        responseDiv.innerText = "API error: " + error;
        console.error("Fetch error:", error);
      });
    });
  </script>
</div>

APACHE PROXY

    ProxyPreserveHost On
    ProxyPass /flask/ http://127.0.0.1:5000/
    ProxyPassReverse /flask/ http://127.0.0.1:5000/


</VirtualHost>
               
source bin/activate
pip install --upgrade transformers
pip install --upgrade pip
pip install --upgrade sentencepiece
pip install --upgrade torch
pip install --upgrade accelerate
cp dot.bashrc ~/.bashrc
pip install --upgrade flask
ssh-keygen -t ed25519
root@n0omw58uom:/home/ai#

poc.py
rom transformers import T5Tokenizer, T5ForConditionalGeneration
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl", device_map="auto")
input_text = "translate English to German: How old are you?"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")
outputs = model.generate(input_ids)
print(tokenizer.decode(outputs[0]))

app.py

rom flask import Flask, request, jsonify
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load your fine-tuned FLAN model (adjust the model path if it's locally stored or on a model hub)
model_name = "google/flan-t5-xl" # Replace this with your FLAN model path or name
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to the FLAN model-based Flask API!"

@app.route('/', methods=['POST'])
def generate():
    # Get the input text from the request
    data = request.json
    prompt = data.get("prompt", "")
   
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    # Tokenize input prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
   
    # Generate response from the model
    outputs = model.generate(input_ids, max_length=150, num_beams=5, early_stopping=True)
   
    # Decode the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return jsonify({"generated_text": generated_text})

if __name__ == '__main__':
    app.run(debug=True)


