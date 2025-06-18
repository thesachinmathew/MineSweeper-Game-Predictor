import tkinter as tk
import random
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

GRID_SIZE = 5
TOTAL_TILES = GRID_SIZE * GRID_SIZE
BOMBS = 2
TRAIN_GAMES = 5000
TEST_GAMES = 100

def extract_features(index, grid_size=GRID_SIZE):
    row, col = divmod(index, grid_size)
    center_dist = ((row - 2)**2 + (col - 2)**2)**0.5
    
    neighbors = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            neighbor_row, neighbor_col = row + dx, col + dy
            if 0 <= neighbor_row < grid_size and 0 <= neighbor_col < grid_size:
                neighbor_index = neighbor_row * grid_size + neighbor_col
                neighbors.append(neighbor_index)
                
    # Limit the feature vector size to ensure consistency
    feature_vector = [index, row, col, center_dist]
    feature_vector.extend(neighbors)
    
    # Ensure feature_vector has exactly 13 elements (pad with zeros if necessary)
    while len(feature_vector) < 13:
        feature_vector.append(0)
    feature_vector = feature_vector[:13]  # Ensure we are not exceeding 13 elements
    
    return feature_vector

def generate_board():
    return random.sample(range(TOTAL_TILES), BOMBS)

def create_dataset(num_games):
    X, y = [], []
    for _ in range(num_games):
        bombs = generate_board()
        for i in range(TOTAL_TILES):
            X.append(extract_features(i))
            y.append(1 if i in bombs else 0)
    return np.array(X), np.array(y)

# --- Train the Neural Network ---
def train_model():
    X, y = create_dataset(TRAIN_GAMES)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # MLP Classifier (Feedforward Neural Network)
    model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=200, random_state=42)
    model.fit(X_train, y_train)
    
    # Test the model's accuracy
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions) * 100
    print(f"Model trained. Accuracy on test set: {accuracy:.2f}%")
    return model

# --- GUI Class ---
class NeuralMinesweeper:
    def __init__(self, root, model):
        self.root = root
        self.model = model
        self.frame = tk.Frame(root)
        self.frame.pack()

        self.score = 0
        self.revealed = [False] * TOTAL_TILES
        self.bombs = []
        self.buttons = []
        self.predictions = []

        self.info = tk.Label(root, text="Avoid the bombs! Prediction in orange.")
        self.info.pack()

        self.play_game()

    def play_game(self):
        self.score = 0
        self.revealed = [False] * TOTAL_TILES
        self.bombs = generate_board()
        self.predictions = self.predict_bombs()

        # Reset Frame (Destroy existing buttons)
        for widget in self.frame.winfo_children():
            widget.destroy()

        # Setup new board
        self.setup_board()

    def setup_board(self):
        self.buttons.clear()  # Clear old button references

        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                idx = i * GRID_SIZE + j
                btn = tk.Button(self.frame, text="", width=6, height=3,
                                command=lambda idx=idx: self.reveal(idx))
                btn.grid(row=i, column=j)
                self.buttons.append(btn)

                # Predicted bombs in orange
                if idx in self.predictions:
                    btn.config(bg="orange")

        # Play Again button
        play_again_btn = tk.Button(self.frame, text="Play Again", command=self.play_game)
        play_again_btn.grid(row=GRID_SIZE, column=GRID_SIZE//2, pady=10)

    def predict_bombs(self):
        features = [extract_features(i) for i in range(TOTAL_TILES)]
        probs = self.model.predict_proba(features)[:, 1]
        top_indices = np.argsort(probs)[-BOMBS:]
        return list(top_indices)

    def reveal(self, idx):
        if self.revealed[idx]:
            return
        self.revealed[idx] = True

        if idx in self.bombs:
            self.buttons[idx].config(text="💣", bg="red")
            self.show_bombs()
            self.end_game("Game Over! You hit a bomb.")
        else:
            self.buttons[idx].config(text="1", bg="green")
            self.score += 1
            if self.score == TOTAL_TILES - BOMBS:
                self.show_bombs()
                self.end_game("You Win!")

    def show_bombs(self):
        for i in range(TOTAL_TILES):
            if i in self.bombs:
                self.buttons[i].config(text="💣", bg="gray")

    def end_game(self, message):
        for btn in self.buttons:
            btn.config(state="disabled")
        self.info.config(text=message)

# --- Main Execution ---
if __name__ == "__main__":
    print("Training model with 5000 games...")
    model = train_model()
    
    print("Launching Minesweeper GUI with predictions...")
    root = tk.Tk()
    root.title("Neural Minesweeper Predictor")
    game = NeuralMinesweeper(root, model)
    root.mainloop()
