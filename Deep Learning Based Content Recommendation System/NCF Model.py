from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
import pickle

app = Flask(__name__)

# Rebuild the model architecture
n_users = 8123   
n_items = 7   


# Define the deep learning model (Neural Collaborative Filtering)
class NCFModel(tf.keras.Model):
    def __init__(self, n_users, n_items, embedding_dim=50, **kwargs):
        super(NCFModel, self).__init__(**kwargs)  # Pass kwargs to parent class
        
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim

        # User and item embedding layers
        self.user_embedding = tf.keras.layers.Embedding(input_dim=n_users, output_dim=embedding_dim)
        self.item_embedding = tf.keras.layers.Embedding(input_dim=n_items, output_dim=embedding_dim)
        
        # Neural network layers
        self.fc1 = tf.keras.layers.Dense(128, activation='relu')
        self.fc2 = tf.keras.layers.Dense(64, activation='relu')
        self.fc3 = tf.keras.layers.Dense(32, activation='relu')
        self.output_layer = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        user_input, item_input = inputs
        user_emb = self.user_embedding(user_input)
        item_emb = self.item_embedding(item_input)
        
        concatenated = tf.concat([user_emb, item_emb], axis=-1)
        x = self.fc1(concatenated)
        x = self.fc2(x)
        x = self.fc3(x)
        output = self.output_layer(x)
        return output

    def get_config(self):
        config = super(NCFModel, self).get_config()   
        config.update({
            "n_users": self.n_users,
            "n_items": self.n_items,
            "embedding_dim": self.embedding_dim
        })
        return config

    @classmethod
    def from_config(cls, config):
        print("Loading model from config:", config)  

        n_users = config.get("n_users", 8123)   
        n_items = config.get("n_items", 7)
        embedding_dim = config.get("embedding_dim", 50)

        if n_users is None or n_items is None:
            raise ValueError(f"Missing parameters: n_users={n_users}, n_items={n_items}")

        return cls(n_users=n_users, n_items=n_items, embedding_dim=embedding_dim)





# Load trained model
model = tf.keras.models.load_model("ncf_model.keras", custom_objects={"NCFModel": NCFModel})

# Load user and course mappings
with open("user_map.pkl", "rb") as f:
    user_map = pickle.load(f)
with open("course_map.pkl", "rb") as f:
    course_map = pickle.load(f)

reverse_course_map = {v: k for k, v in course_map.items()}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/recommend", methods=["POST"])
def recommend():
    user_id = int(request.form["user_id"])
    
    if user_id not in user_map:
        return render_template("index.html", error="User ID not found.")
    
    mapped_user_id = user_map[user_id]
    course_ids = np.array(list(course_map.values()))
    user_ids = np.full_like(course_ids, mapped_user_id)
    predictions = model.predict([user_ids, course_ids])
    
    top_indices = np.argsort(predictions.flatten())[::-1][:3]
    recommended_courses = [reverse_course_map[i] for i in top_indices]
    
    return render_template("index.html", recommendations=recommended_courses)

if __name__ == "__main__":
    app.run(debug=True)
