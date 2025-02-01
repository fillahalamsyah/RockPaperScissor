# main.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import random
import numpy as np

try:
    from src.models.model import train_or_load_model
    from src.data.dataset import choices
except ModuleNotFoundError:
    from models.model import train_or_load_model
    from data.dataset import choices

from sklearn.preprocessing import LabelEncoder

# Load Model
model = train_or_load_model(load_model=True)
le = LabelEncoder()
le.fit(choices)

# Statistik Game
stats = {"win": 0, "draw": 0, "lose": 0}
randomness = 0.3

def play_against_ai(player_choice):
    """Fungsi untuk bermain melawan AI berdasarkan model yang telah dilatih."""
    if player_choice not in choices:
        print("Pilihan tidak valid! Pilih: rock, paper, atau scissors")
        return

    scores = {choice: model.predict(np.array([[le.transform([choice])[0], le.transform([player_choice])[0]]]))[0][0] for choice in choices}
    ai_choice = random.choice(choices) if random.random() < randomness else max(scores, key=scores.get)

    print(f"\nAnda memilih: {player_choice}")
    print(f"AI memilih: {ai_choice} (Confidence: {scores[ai_choice]:.2%})")

    if ai_choice == player_choice:
        print("Hasil SERI!"); stats["draw"] += 1
    elif (player_choice == 'rock' and ai_choice == 'scissors') or \
         (player_choice == 'scissors' and ai_choice == 'paper') or \
         (player_choice == 'paper' and ai_choice == 'rock'):
        print("Anda MENANG!"); stats["win"] += 1
    else:
        print("AI MENANG!"); stats["lose"] += 1

    print(f"Statistik Saat Ini: Menang: {stats['win']}, Seri: {stats['draw']}, Kalah: {stats['lose']}")

# Permainan Interaktif
if __name__ == "__main__":
    while True:
        try:
            player_move = input("\nPilih (rock, paper, scissors) atau 'exit' untuk keluar: ").lower()
            if player_move == 'exit':
                print(f"Statistik Akhir: Menang: {stats['win']}, Seri: {stats['draw']}, Kalah: {stats['lose']}")
                print("Terima kasih telah bermain!")
                break
            play_against_ai(player_move)
        except KeyboardInterrupt:
            print(f"\nStatistik Akhir: Menang: {stats['win']}, Seri: {stats['draw']}, Kalah: {stats['lose']}")
            print("Terima kasih telah bermain!")
            break
        except Exception as e:
            print(f"Error: {str(e)}")
