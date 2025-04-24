from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
import pandas as pd

app = Flask(__name__)

# Rebuild the model architecture
n_users = 8123   
n_items = 7   


class LightGCN(nn.Module):
    def __init__(self, n_users, n_items, latent_dim=128):
        super(LightGCN, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.latent_dim = latent_dim

        # Initialize user and item embeddings
        self.user_embedding = nn.Embedding(n_users, latent_dim)
        self.item_embedding = nn.Embedding(n_items, latent_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(self, users):
        user_emb = self.user_embedding(users)   
        item_embs = self.item_embedding.weight   

        scores = torch.matmul(user_emb, item_embs.T)   
        return scores


# Load the trained model
model = LightGCN(n_users, n_items)
model.load_state_dict(torch.load(r"C:\Users\hp\Desktop\Last Hope\lightgcn_model.pth", map_location=torch.device('cpu')))
model.eval()   

# Load item names
items = ["1", "2", "3", "4", "5", "6", "7" ]
"""
@app.route("/")
def home():
    return render_template("index1.html")

@app.route("/recommend", methods=["POST"])
def recommend():
    user_id = int(request.form["user_id"])  # Get user input

    # Prepare input tensor
    user_tensor = torch.tensor([user_id], dtype=torch.long)

    # Get model predictions
    with torch.no_grad():
        predictions = model(user_tensor).squeeze()  # Get scores

    # Get top 3 recommendations
    recommended_indices = torch.argsort(predictions, descending=True)[:3]
    recommended_items = [items[i] for i in recommended_indices]

    return jsonify({"recommendations": recommended_items})


if __name__ == "__main__":
    app.run(debug=True)
"
"""

@app.route("/")
def home():
    return render_template("index1.html")

@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        user_id = int(request.form.get("user_id", -1))
        if user_id < 0:
            return render_template("index1.html", error="Invalid User ID")

        # Prepare input tensor
        user_tensor = torch.tensor([user_id], dtype=torch.long)

        # Get model predictions
        with torch.no_grad():
            predictions = model(user_tensor).squeeze()

        # Get top 3 recommendations
        recommended_indices = torch.argsort(predictions, descending=True)[:3].tolist()
        recommended_items = [items[i] for i in recommended_indices]

        return render_template("index1.html", recommendations=recommended_items)

    except Exception as e:
        return render_template("index1.html", error=str(e))

if __name__ == "__main__":
    app.run(debug=True)