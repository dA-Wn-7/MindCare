from peft import PeftModel
import os
import sys
import sqlite3
from pathlib import Path
import html
import traceback
import torch
from datetime import datetime

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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
    build_prompt_messages,
    infer_emotion_from_text,
    generate_llm_response,
    _ensure_llm
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
BASE_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2" 
FINETUNED_MODEL_REPO = os.environ.get("MINDCARE_LLM_REPO", "imnotdawn/mistral7b-qlora-sft-small-v2.1")

# Global variables for models
tokenizer_ft = None
model_ft = None
tokenizer_base = None
model_base = None
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
                  reply_base TEXT, reply_ft TEXT, preferred_model TEXT)""")
    conn.commit()
    conn.close()

def log_preference(user_text, emotion, strategy, reply_base, reply_ft, preferred_model):
    if not _logging_enabled():
        return
    try:
        conn = sqlite3.connect(_DB_PATH)
        c = conn.cursor()
        c.execute(
            "INSERT INTO interactions VALUES (?, ?, ?, ?, ?, ?, ?)",
            (datetime.now().isoformat(), user_text, emotion, strategy, 
             reply_base[:1000], reply_ft[:1000], preferred_model)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"DB Error: {e}")

# ==========================================
# Model Loading Helpers
# ==========================================

def load_finetuned_model():
    """
    Loads the FT model. Note: p.py also loads an LLM instance. 
    To avoid double loading VRAM, we can try to reuse p.py's model if possible,
    but for A/B testing clarity, we often keep separate instances or ensure 
    p.py's instance is used for FT generation.
    
    Here, we will align app.py's FT model with p.py's logic by ensuring 
    p.py is initialized, and then we load Base separately for comparison.
    """
    global tokenizer_ft, model_ft
    
    # We rely on p.py's ensure_pipeline_ready to load the main LLM (which is the FT one usually)
    # But for explicit A/B in app.py, let's load them explicitly if not already done by p.py
    
    if model_ft is None:
        print(f"Loading Fine-tuned Model (LoRA): {FINETUNED_MODEL_REPO}...")
        try:
            tokenizer_ft = AutoTokenizer.from_pretrained(FINETUNED_MODEL_REPO)
            if tokenizer_ft.pad_token_id is None:
                tokenizer_ft.pad_token_id = tokenizer_ft.eos_token_id
            
            use_4bit = _env_flag("MINDCARE_LLM_LOAD_IN_4BIT")
            qconfig = None
            if use_4bit:
                qconfig = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                )

            print("Loading Base Model for LoRA attachment...")
            base_model_for_lora = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL_NAME,
                quantization_config=qconfig,
                device_map="auto",
                torch_dtype=torch.float16,
            )
            
            print("Attaching LoRA Adapter...")
            model_ft = PeftModel.from_pretrained(base_model_for_lora, FINETUNED_MODEL_REPO)
            model_ft.eval()
            print("Fine-tuned Model (LoRA) Loaded Successfully.")
            
        except Exception as e:
            print(f"Failed to load FT model: {e}")
            traceback.print_exc()
            raise e

def load_base_model():
    global tokenizer_base, model_base
    if model_base is None:
        print(f"Loading Base Model: {BASE_MODEL_NAME}...")
        try:
            tokenizer_base = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
            if tokenizer_base.pad_token_id is None:
                tokenizer_base.pad_token_id = tokenizer_base.eos_token_id
                
            use_4bit = _env_flag("MINDCARE_LLM_LOAD_IN_4BIT")
            qconfig = None
            if use_4bit:
                qconfig = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                )

            model_base = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL_NAME,
                quantization_config=qconfig,
                device_map="auto",
                torch_dtype=torch.float16,
            )
            model_base.eval()
            print("Base Model Loaded Successfully.")
        except Exception as e:
            print(f"Failed to load Base model: {e}")
            raise e

def generate_response(model, tokenizer, messages, max_new_tokens=256):
    """Generic generation function for Base Model comparison"""
    try:
        prompt_text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=2048, padding=True)
        inputs = {k: v.to(_DEVICE) for k, v in inputs.items()}
        
        with torch.no_grad():
            output = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
                no_repeat_ngram_size=3,
            )
        
        input_len = inputs["input_ids"].shape[1]
        new_tokens = output[0][input_len:]
        raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        
        if "[/INST]" in raw:
            raw = raw.split("[/INST]")[-1].strip()
        return raw
    except Exception as e:
        print(f"Generation Error: {e}")
        return "Error generating response."

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

def generate_opening_messages(mood_key):
    mood_data = MOOD_OPTIONS.get(mood_key, MOOD_OPTIONS["winter_snow"])
    
    system_content = f"""You are a warm, empathetic mental health companion. 
The user has indicated their current emotional state by choosing a metaphor: "{mood_data['label']}".
Context: {mood_data['prompt_hint']}

Task: 
Generate a VERY SHORT (under 50 words), warm, and inviting opening message. 
Acknowledge their feeling gently. Do NOT ask too many questions yet. Just let them know you are here with them.
"""
    
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": f"I feel like {mood_data['label']}."}
    ]
    
    try:
        load_base_model()
        load_finetuned_model()
        
        print("Generating Base Response...")
        reply_base = generate_response(model_base, tokenizer_base, messages)
        
        print("Generating FT Response...")
        reply_ft = generate_response(model_ft, tokenizer_ft, messages)
        
        return reply_base, reply_ft
    except Exception as e:
        print(f"CRITICAL ERROR in generate_opening_messages: {e}")
        traceback.print_exc()
        return f"Error: {str(e)[:50]}", f"Error: {str(e)[:50]}"

def start_chat_session_ui(mood_key, history):
    if not mood_key:
        return history, history, gr.update(visible=True), gr.update(visible=False), "", "", ""

    reply_base, reply_ft = generate_opening_messages(mood_key)
    
    combined_msg = f"**Option A (Base Model):**\n{reply_base}\n\n---\n\n**Option B (MindCare AI):**\n{reply_ft}"
    
    new_history = [(f"Mood Selected: {MOOD_OPTIONS[mood_key]['label']}", combined_msg)]
    
    return new_history, new_history, gr.update(visible=False), gr.update(visible=True), reply_base, reply_ft, mood_key


# ==========================================
# Normal Chat Logic with A/B Testing (Enhanced)
# ==========================================

def process_normal_chat_ab(text, history, mood_state, llm_history_state):
    """
    Generates responses from BOTH Base and FT models for A/B testing.
    Option B now uses the advanced logic from p.py (process_chat_request).
    """
    if not text:
        return history, history, llm_history_state, "", ""

    # 1. Update UI to show user message and "Thinking..."
    display_history = history + [(text, "⏳ Generating responses from both models...")]
    yield display_history, display_history, llm_history_state, "", "" 

    reply_base = "Error generating base response."
    reply_ft = "Error generating FT response."
    ft_strategy = "unknown"
    ft_emotion = "unknown"

    try:
        # Ensure models are loaded
        if model_base is None: load_base_model()
        if model_ft is None: load_finetuned_model()
        
        # Ensure p.py's pipeline is ready (loads LLM if not already, and other components)
        ensure_pipeline_ready()
            
        current_llm_history = llm_history_state or []
        
        # --- OPTION A: Base Model (Simple Prompt for Baseline) ---
        simple_messages = [{"role": "system", "content": "You are a supportive mental health companion."}]
        simple_messages.extend(current_llm_history)
        simple_messages.append({"role": "user", "content": text})
        
        print("Generating Base Response...")
        reply_base = generate_response(model_base, tokenizer_base, simple_messages, max_new_tokens=256)
        
        # --- OPTION B: FT Model (Advanced Prompt via p.py) ---
        print("Generating FT Response with Advanced Pipeline...")
        # Use the unified interface from p.py
        result_ft = process_chat_request(
            user_text=text,
            chat_history_list=current_llm_history,
            audio_path=None # Text-only mode for now
        )
        reply_ft = result_ft["reply"]
        ft_strategy = result_ft.get("strategy", "unknown")
        ft_emotion = result_ft.get("emotion", "unknown")
        
    except Exception as e:
        print(f"Error in A/B generation: {e}")
        traceback.print_exc()
        reply_base = f"Base Model Error: {str(e)[:50]}"
        reply_ft = f"FT Model Error: {str(e)[:50]}"

    # 2. Format the combined message for Display
    combined_msg = f"**Option A (Base Model):**\n{reply_base}\n\n---\n\n**Option B (MindCare AI):**\n{reply_ft}"
    
    # 3. Update Display History
    display_history[-1] = (text, combined_msg)
    
    # 4. Update LLM Internal History
    # We use the FT response (which follows MI principles) to maintain context consistency
    new_llm_history = current_llm_history + [
        {"role": "user", "content": text},
        {"role": "assistant", "content": reply_ft} 
    ]
    
    # 5. Log the interaction
    if _logging_enabled():
        log_preference(text, ft_emotion, ft_strategy, reply_base, reply_ft, "pending")

    # 6. Return updated states
    yield display_history, display_history, new_llm_history, reply_base, reply_ft


# ==========================================
# Gradio UI Construction
# ==========================================

if _logging_enabled():
    init_db()

# Initialize models on startup to avoid long wait on first click
print("Starting MindCare Application...")
try:
    ensure_pipeline_ready() # Loads p.py models
    load_base_model()       # Loads base model for comparison
    load_finetuned_model()  # Loads FT model explicitly for app control if needed
except Exception as e:
    print(f"Warning: Could not pre-load all models: {e}")

with gr.Blocks(theme=gr.themes.Soft()) as app:
    gr.Markdown("# MindCare 🧠")
    gr.Markdown(DISCLAIMER_MD)

    # State variables
    last_base_reply = gr.State("")
    last_ft_reply = gr.State("")
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
        chatbot = gr.Chatbot(label="Conversation", height=500, bubble_full_width=False)
        state_history = gr.State([]) # Display history

        # Preference Buttons
        with gr.Row():
            prefer_base_btn = gr.Button("👍 Prefer Base Model", variant="secondary")
            prefer_ft_btn = gr.Button("👍 Prefer MindCare AI", variant="primary")
        
        pref_status = gr.Markdown("", visible=False)

        with gr.Row():
            with gr.Column(scale=8):
                text_input = gr.Textbox(
                    placeholder="Type your message here...",
                    show_label=False,
                    container=False
                )
            with gr.Column(scale=1):
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
            inputs=[gr.State(key), state_history],
            outputs=[chatbot, state_history, buffering_display, chat_interface_col, last_base_reply, last_ft_reply, current_mood]
        )

    # Bind Submit Button (A/B Test Chat)
    submit_btn.click(
        fn=process_normal_chat_ab,
        inputs=[text_input, state_history, current_mood, llm_internal_history],
        outputs=[chatbot, state_history, llm_internal_history, last_base_reply, last_ft_reply]
    )

    def process_audio_chat(audio_path, history, mood_state, llm_history_state):
        if not audio_path:
            return history, history, llm_history_state, "", ""

        from modules.pipelines.p import speech_to_text
        try:
            user_text = speech_to_text(audio_path)
        except Exception as e:
            user_text = f"[Audio Error: {str(e)}]"
            
        if not user_text:
            return history, history, llm_history_state, "", ""
        
        gen = process_normal_chat_ab(user_text, history, mood_state, llm_history_state)
        
        step1 = next(gen) 
        step2 = next(gen)
        
        return step2
    def handle_user_interaction(text, audio_path, history, mood_state, llm_history_state):

        user_text = text
    
        if audio_path and not text:
            try:
                from modules.pipelines.p import speech_to_text
                user_text = speech_to_text(audio_path)
            except Exception as e:
                user_text = f"[Audio Processing Error]"

        if not user_text:
            return history, history, llm_history_state, "", ""

        gen = process_normal_chat_ab(user_text, history, mood_state, llm_history_state)

        try:
            next(gen) # Skip thinking state for simplicity in upload, or handle it if you want streaming
            final_state = next(gen)
            return final_state
        except StopIteration:
            return history, history, llm_history_state, "", ""

# 在 Gradio Blocks 中：

    # Bind Submit Button
    submit_btn.click(
        fn=lambda t, h, m, l: handle_user_interaction(t, None, h, m, l),
        inputs=[text_input, state_history, current_mood, llm_internal_history],
        outputs=[chatbot, state_history, llm_internal_history, last_base_reply, last_ft_reply]
    )

    # Bind Audio Upload
    audio_input.upload(
        fn=lambda t, a, h, m, l: handle_user_interaction(t, a, h, m, l),
        inputs=[text_input, audio_input, state_history, current_mood, llm_internal_history],
        outputs=[chatbot, state_history, llm_internal_history, last_base_reply, last_ft_reply]
    )

    # Preference Logging Functions
    def handle_base_preference(base, ft, mood):
        log_preference("Interaction", mood, "ab_test", base, ft, "base")
        return gr.update(value="✅ Recorded: Prefer Base Model", visible=True)

    def handle_ft_preference(base, ft, mood):
        log_preference("Interaction", mood, "ab_test", base, ft, "finetuned")
        return gr.update(value="✅ Recorded: Prefer MindCare AI", visible=True)

    prefer_base_btn.click(
        fn=handle_base_preference,
        inputs=[last_base_reply, last_ft_reply, current_mood],
        outputs=[pref_status]
    )
    
    prefer_ft_btn.click(
        fn=handle_ft_preference,
        inputs=[last_base_reply, last_ft_reply, current_mood],
        outputs=[pref_status]
    )

# Launch Logic
if __name__ == "__main__":
    app.launch(share=True)