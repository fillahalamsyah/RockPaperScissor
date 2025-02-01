import streamlit as st
import numpy as np
import pandas as pd
from src.models.model import train_or_load_model
from src.data.dataset import choices
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Batu-Gunting-Kertas", page_icon="🪨", layout="wide")

# Inisialisasi state untuk model dan label encoder
if 'model' not in st.session_state:
    st.session_state.model = train_or_load_model(load_model=True)
    st.session_state.le = LabelEncoder()
    st.session_state.le.fit(choices)

# Inisialisasi state untuk statistik permainan
st.session_state.setdefault('stats', {"win": 0, "draw": 0, "lose": 0})
st.session_state.setdefault('game_played', 0)
st.session_state.setdefault('game_history', [])
st.session_state.setdefault('game_history_df', None)
st.session_state.setdefault('player_move', None)
st.session_state.setdefault('ai_move', None)
st.session_state.setdefault('winratepercent', 0)
st.session_state.setdefault('last_result', None)
st.session_state.setdefault('delta', {"game_played": 0, "win": 0, "draw": 0, "lose": 0, "winrate": 0, "current_win_rate": 0})
st.session_state.setdefault('action_performed', False)  # Flag to track action

icons = {"rock": '🪨', "scissors": '✂️', "paper": '📄'}
randomness = 0.3

st.title("🪨✂️📄 Batu-Gunting-Kertas")

@st.fragment
def game_history():
    game_history_container.empty()
    #game_history_container.header("Game History", anchor="history", divider=True)
    

with st.sidebar:
    st.header("Menu", anchor="menu", divider=True)
    
    if st.button("Reset Statistik"):
        for key in ["stats", "game_played", "game_history", "game_history_df", "winratepercent", "last_result", "player_move", "ai_move", "action_performed"]:
            st.session_state[key] = {"win": 0, "draw": 0, "lose": 0} if key == "stats" else 0 if key in ["game_played", "action_performed"] else None if key.endswith("df") else []
        st.rerun()  # Pastikan state langsung diperbarui dan UI di-refresh

    show_history = st.checkbox("Tampilkan Game History", value=True)
    if show_history:
        with st.expander("Game History", expanded=True):
            game_history_container = st.container()
            game_history()

@st.fragment
def metric_statistic():
    st.markdown("<h2 style='text-align: center;'>Statistik Permainan</h2>", unsafe_allow_html=True)
    metrics = st.columns(5)
    stats_keys = ["game_played", "win", "draw", "lose"]
    labels = ["Total Permainan", "Menang", "Seri", "Kalah", "Win Rate"]
    for i, key in enumerate(stats_keys):
        with metrics[i]:
            st.metric(labels[i], st.session_state[key] if key == "game_played" else st.session_state.stats[key], 
                    delta=st.session_state.delta[key], border=True)
    with metrics[4]:
        st.metric("Win Rate", f"{st.session_state.delta['current_win_rate']:.2%}", 
                f"{st.session_state.delta['winrate']:.2%}", border=True)
metric_statistic()

user_input, ai_input = st.columns(2)
with ai_input:
    ai_input_container = st.container()
    with ai_input_container:
        st.subheader("AI Input", anchor="ai", divider=True)
        st.info("AI akan memilih berdasarkan model yang telah dibuat.")

with user_input:
    st.subheader("User Input", anchor="input", divider=True)
    button_cols = st.columns(len(icons))
    for idx, (choice, icon) in enumerate(icons.items()):
        with button_cols[idx]:
            if st.button(choice.capitalize(), key=choice, icon=icon):
                st.session_state.update({"player_move": choice, "game_played": st.session_state.game_played + 1})
                st.session_state.action_performed = True
                st.session_state.last_result = None
                st.session_state.delta["game_played"] = 1
                st.rerun()  # Rerun to handle AI move and result

    user_container = st.container()

if st.session_state.action_performed:
    st.session_state.action_performed = False  # Reset flag
    
    if st.session_state.player_move:
        
        user_container.markdown(f'<h1 style="text-align: center;font-size: 120px;">{icons[st.session_state.player_move]}</h1>', unsafe_allow_html=True)

        scores = {choice: st.session_state.model.predict(
            np.array([[st.session_state.le.transform([choice])[0], st.session_state.le.transform([st.session_state.player_move])[0]]]))[0][0] for choice in choices}
        ai_choice = np.random.choice(choices) if np.random.random() < randomness else max(scores, key=scores.get)
        st.session_state.ai_move = ai_choice

        ai_input_container.markdown(f'<h1 style="text-align: center;font-size: 120px;">{icons[st.session_state.ai_move]}</h1>', unsafe_allow_html=True)

        game_result = "draw" if st.session_state.ai_move == st.session_state.player_move else "win" if (st.session_state.player_move, st.session_state.ai_move) in [("rock", "scissors"), ("scissors", "paper"), ("paper", "rock")] else "lose"

        result = st.container()
        result_messages = {"draw": "Hasil SERI!", "win": "Anda Menang!", "lose": "Anda Kalah!"}
        icons_messages = {"draw": "🤝", "win": "🎉", "lose": "😢"}
        result.html(f"<h1 style='text-align: center;'>{result_messages[game_result]}</h1>")
        st.toast(result_messages[game_result], icon=icons_messages[game_result])

        st.session_state.stats[game_result] += 1
        st.session_state.last_result = game_result
        st.session_state.game_history.append({"player_move": st.session_state.player_move, "ai_move": st.session_state.ai_move, "result": game_result})
        st.session_state.game_history_df = pd.DataFrame(st.session_state.game_history)
        game_history_container.dataframe(st.session_state.game_history_df, hide_index=True)

        prev_winrate = st.session_state.winratepercent
        st.session_state.winratepercent = st.session_state.stats["win"] / st.session_state.game_played if st.session_state.game_played > 0 else 0

        for key in ["win", "draw", "lose"]:
            st.session_state.delta[key] = 1 if st.session_state.last_result == key else 0

        st.session_state.delta.update({"game_played": 1, "current_win_rate": st.session_state.winratepercent, "winrate": st.session_state.winratepercent - prev_winrate if prev_winrate else 0})

        st.session_state.player_move = None
