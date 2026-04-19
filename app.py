import os
import sys
import sqlite3
from pathlib import Path
import traceback
import torch
from datetime import datetime

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import gradio as gr

# Import existing modules
from modules.safety import (
    is_crisis,
    is_violence_toward_others,
    CRISIS_REPLY,
    VIOLENCE_RISK_REPLY,
    crisis_and_violence_reply,
)

# Import the enhanced pipeline logic from p.py
from modules.pipelines.p import (
    process_chat_request,
    ensure_pipeline_ready,
    infer_emotion_from_text,
    speech_to_text,
)

# ==========================================
# Configuration & Globals
# ==========================================

DISCLAIMER_MD = """
### MindCare — Please Read

- **This service is not medical or psychological counselling.** It does not diagnose or treat any condition and does not replace in-person care from licensed professionals.
- If you have thoughts of **self-harm, suicide, or harming others**, or are in immediate danger, contact **local emergency services** right away (e.g., **911** in US, **999** in UK, **110/120** in China).
- By using this service, you agree that your interactions may be logged for research purposes to improve model quality.
"""

_DB_PATH = os.environ.get("MINDCARE_DB_PATH", "mindcare_logs.db")

def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").lower() in ("1", "true", "yes")

# ==========================================
# Database & Logging
# ==========================================

def _logging_enabled():
    return os.environ.get("MINDCARE_LOG_INTERACTIONS", "").lower() in ("1", "true", "yes")

def init_db():
    conn = sqlite3.connect(_DB_PATH)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS interactions
                 (timestamp TEXT, user_text TEXT, emotion TEXT, strategy TEXT, 
                  reply TEXT)""")
    conn.commit()
    conn.close()

def log_interaction(user_text, emotion, strategy, reply):
    if not _logging_enabled():
        return
    try:
        conn = sqlite3.connect(_DB_PATH)
        c = conn.cursor()
        c.execute(
            "INSERT INTO interactions VALUES (?, ?, ?, ?, ?)",
            (datetime.now().isoformat(), user_text, emotion, strategy, reply[:1000])
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"DB Error: {e}")

# ==========================================
# Mood Selection & Opening Logic
# ==========================================

MOOD_OPTIONS = {
    "sunny_day": {
        "label": "🌸 Sunny Day",
        "desc": "Positive, optimistic, and excited",
        "prompt_hint": "The user feels happy: positive, optimistic, and excited. They need a sense of joy and happiness.",
    },
    "spring_rain": {
        "label": "🌧️ Spring Rain",
        "desc": "Melancholic but hopeful",
        "prompt_hint": "The user feels like a rainy spring day: melancholic but hopeful, perhaps feeling a bit overwhelmed by growth or change.",
    },
    "summer_storm": {
        "label": "⛈️ Summer Storm",
        "desc": "Anxious, angry, or stressed",
        "prompt_hint": "The user feels like a summer storm: intense anxiety, anger, or high stress. They need grounding and calm.",
    },
    "autumn_wind": {
        "label": "🍂 Autumn Wind",
        "desc": "Lonely, lost",
        "prompt_hint": "The user feels like an autumn wind: lonely, lost, or experiencing a sense of loss. They need empathy and companionship.",
    },
    "winter_snow": {
        "label": "❄️ Winter Snow",
        "desc": "Numb, exhausted, or frozen",
        "prompt_hint": "The user feels like winter snow: numb, exhausted, or emotionally frozen. They need gentle warmth and patience.",
    }
}

BUFFERING_ANIMATION_HTML = """
<div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 300px; width: 100%;">
    <style>
        @keyframes breathe {
            0% { transform: scale(0.8); opacity: 0.5; box-shadow: 0 0 20px rgba(100,100,255,0.2); }
            50% { transform: scale(1.1); opacity: 0.9; box-shadow: 0 0 50px rgba(100,100,255,0.6); }
            100% { transform: scale(0.8); opacity: 0.5; box-shadow: 0 0 20px rgba(100,100,255,0.2); }
        }
        .breathing-circle {
            width: 100px;
            height: 100px;
            border-radius: 50%;
            background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
            animation: breathe 4s infinite ease-in-out;
            margin-bottom: 20px;
        }
        .loading-text {
            font-size: 1.2em;
            color: #555;
            font-weight: 300;
            letter-spacing: 1px;
            animation: fadeInOut 3s infinite;
        }
        @keyframes fadeInOut {
            0%, 100% { opacity: 0.6; }
            50% { opacity: 1; }
        }
    </style>
    <div class="breathing-circle"></div>
    <div class="loading-text">Lighting a candle for you... Preparing your safe space...</div>
</div>
"""

def generate_opening_message(mood_key):
    """Generates only the MindCare AI opening message"""
    mood_data = MOOD_OPTIONS.get(mood_key, MOOD_OPTIONS["winter_snow"])
    
    welcomes = {
        "sunny_day": "Hello! I'm so glad to see your sunny disposition. I'm here to share in your joy.",
        "spring_rain": "Hello. I sense a gentle melancholy, like spring rain. I'm here to listen whenever you're ready.",
        "summer_storm": "Hello. I hear the intensity of your feelings. I'm here to help you find some calm.",
        "autumn_wind": "Hello. I sense you might be feeling a bit lost or lonely. I'm here to keep you company.",
        "winter_snow": "Hello. I sense you might be feeling numb or exhausted. I'm here to offer gentle warmth."
    }
    return welcomes.get(mood_key, "Hello. I'm here for you.")

def start_chat_session_ui(mood_key, history):
    if not mood_key:
        return history, gr.update(visible=True), gr.update(visible=False), "", mood_key

    reply_ft = generate_opening_message(mood_key)
    
    new_history = [
        {"role": "user", "content": f"Mood Selected: {MOOD_OPTIONS[mood_key]['label']}"},
        {"role": "assistant", "content": reply_ft}
    ]
    
    return new_history, gr.update(visible=False), gr.update(visible=True), reply_ft, mood_key


# ==========================================
# Normal Chat Logic (Single Model)
# ==========================================

def process_chat_single_model(text, audio_path, history, llm_history_state):
    """
    Handles both text and audio input for the single MindCare AI model.
    """
    # Debug log entry
    print(f"[DEBUG APP] process_chat_single_model entered. Text: {text}, Audio: {audio_path}")

    if not text and not audio_path:
        yield history, llm_history_state, ""
        return

    # 1. Determine User Text & Emotion Concurrently
    user_text = text
    ft_emotion = "unknown"
    original_audio_path = None

    if audio_path and not text:
        original_audio_path = audio_path
        try:
            print(f"[DEBUG APP] Starting concurrent audio processing for: {audio_path}")
            
            # --- CONCURRENT PROCESSING START ---
            from modules.pipelines.p import process_audio_concurrently
            user_text, ft_emotion = process_audio_concurrently(audio_path)
            # --- CONCURRENT PROCESSING END ---

            print(f"[DEBUG APP] Transcribed text: {user_text}, Emotion: {ft_emotion}")
            if not user_text:
                user_text = "[Could not understand audio]"
        except Exception as e:
            print(f"[ERROR APP] Audio Processing Error: {e}")
            traceback.print_exc()
            user_text = f"[Audio Error: {str(e)}]"

    if not user_text:
        yield history, llm_history_state, ""
        return

    # 2. Update UI to show user message and "Thinking..."
    display_history = list(history)
    display_history.append({"role": "user", "content": user_text})
    display_history.append({"role": "assistant", "content": "⏳ MindCare is listening..."})
    
    yield display_history, llm_history_state, "" 

    reply_ft = "Error generating response."
    ft_strategy = "unknown"
    ft_emotion = "unknown"

    try:
        # Ensure pipeline is ready
        ensure_pipeline_ready()
            
        current_llm_history = llm_history_state or []
        
        # --- Generate Response using p.py ---
        # FIX: Pass the pre_computed_emotion (ft_emotion) to avoid re-computing it
        result_ft = process_chat_request(
            user_text=user_text,
            chat_history_list=current_llm_history,
            audio_path=original_audio_path,
            pre_computed_emotion=ft_emotion if ft_emotion != "unknown" else None
        )
        reply_ft = result_ft["reply"]
        ft_strategy = result_ft.get("strategy", "unknown")
        # Update ft_emotion with the one returned from pipeline (in case it was recomputed or refined)
        ft_emotion = result_ft.get("emotion", ft_emotion) 
        
    except Exception as e:
        print(f"Error in generation: {e}")
        traceback.print_exc()
        reply_ft = f"Error: {str(e)[:100]}"

    # 3. Update Display History
    if display_history and display_history[-1]["role"] == "assistant":
        display_history[-1]["content"] = reply_ft
    else:
        display_history.append({"role": "assistant", "content": reply_ft})
    
    # 4. Update LLM Internal History
    new_llm_history = current_llm_history + [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": reply_ft} 
    ]
    
    # 5. Log the interaction
    if _logging_enabled():
        log_interaction(user_text, ft_emotion, ft_strategy, reply_ft)

    # 6. Return updated states
    yield display_history, new_llm_history, reply_ft


# ==========================================
# Gradio UI Construction
# ==========================================

if _logging_enabled():
    init_db()

# Initialize models on startup
print("Starting MindCare Application...")
try:
    ensure_pipeline_ready() 
except Exception as e:
    print(f"Warning: Could not pre-load models: {e}")

with gr.Blocks(theme=gr.themes.Soft()) as app:
    gr.Markdown("# MindCare 🧠")
    gr.Markdown(DISCLAIMER_MD)

    # State variables
    last_reply = gr.State("") # To store the last reply for potential feedback/logging
    current_mood = gr.State("")
    llm_internal_history = gr.State([]) # Clean history for LLM context

    # --- View 1: Mood Selector ---
    with gr.Column(visible=True) as mood_selector_col:
        gr.Markdown("### How is the weather in your heart today?")
        gr.Markdown("Please choose the season that best matches your current feelings.")
        
        with gr.Row():
            btn_sunny = gr.Button(MOOD_OPTIONS["sunny_day"]["label"], variant="secondary", size="lg")
            btn_spring = gr.Button(MOOD_OPTIONS["spring_rain"]["label"], variant="secondary", size="lg")
            btn_summer = gr.Button(MOOD_OPTIONS["summer_storm"]["label"], variant="secondary", size="lg")
            btn_autumn = gr.Button(MOOD_OPTIONS["autumn_wind"]["label"], variant="secondary", size="lg")
            btn_winter = gr.Button(MOOD_OPTIONS["winter_snow"]["label"], variant="secondary", size="lg")
        
        with gr.Row():
            gr.Markdown(f"*{MOOD_OPTIONS['sunny_day']['desc']}*")
            gr.Markdown(f"*{MOOD_OPTIONS['spring_rain']['desc']}*")
            gr.Markdown(f"*{MOOD_OPTIONS['summer_storm']['desc']}*")
            gr.Markdown(f"*{MOOD_OPTIONS['autumn_wind']['desc']}*")
            gr.Markdown(f"*{MOOD_OPTIONS['winter_snow']['desc']}*")

    # --- View 2: Buffering ---
    buffering_display = gr.HTML(value="", visible=False)

    # --- View 3: Chat Interface ---
    with gr.Column(visible=False) as chat_interface_col:
        chatbot = gr.Chatbot(label="Conversation", height=500, type="messages")
        
        with gr.Row():
            with gr.Column(scale=8):
                text_input = gr.Textbox(
                    placeholder="Type your message here...",
                    show_label=False,
                    container=False
                )
            with gr.Column(scale=1):
                # Audio input now directly triggers chat
                audio_input = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Audio", show_label=False)

        submit_btn = gr.Button("Send", variant="primary")

    # ==========================================
    # Event Bindings
    # ==========================================

    def show_buffering():
        return gr.update(visible=False), gr.update(value=BUFFERING_ANIMATION_HTML, visible=True), gr.update(visible=False)

    # Bind Mood Buttons
    for key, btn in [("sunny_day", btn_sunny), ("spring_rain", btn_spring), ("summer_storm", btn_summer), ("autumn_wind", btn_autumn), ("winter_snow", btn_winter)]:
        btn.click(
            fn=show_buffering,
            inputs=None,
            outputs=[mood_selector_col, buffering_display, chat_interface_col]
        ).then(
            fn=start_chat_session_ui,
            inputs=[gr.State(key), chatbot],
            outputs=[chatbot, buffering_display, chat_interface_col, last_reply, current_mood]
        )

    def handle_audio_upload(audio_path, history, llm_hist):
        print(f"[DEBUG GRADIO] Audio upload event triggered. Path: {audio_path}")
        if not audio_path:
            return history, llm_hist, ""
        
        gen = process_chat_single_model("", audio_path, history, llm_hist)
        
        final_result = None
        try:
            for result in gen:
                final_result = result
            return final_result if final_result else (history, llm_hist, "")
        except Exception as e:
            print(f"Error in audio handling: {e}")
            traceback.print_exc()
            return history, llm_hist, f"Error: {str(e)}"

    submit_btn.click(
        fn=process_chat_single_model, 
        inputs=[text_input, gr.State(None), chatbot, llm_internal_history],
        outputs=[chatbot, llm_internal_history, last_reply]
    ).then(
        fn=lambda: "", 
        inputs=None,
        outputs=[text_input]
    )

    audio_input.change(
        fn=handle_audio_upload,
        inputs=[audio_input, chatbot, llm_internal_history],
        outputs=[chatbot, llm_internal_history, last_reply]
    ).then(
        fn=lambda: None, 
        inputs=None,
        outputs=[audio_input]
    )

# Launch Logic
if __name__ == "__main__":
    app.launch(share=True)