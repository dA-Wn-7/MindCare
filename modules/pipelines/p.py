# MindCare Multimodal Pipeline
# Whisper + Wav2Vec2 + Strategy Layer + LLM
# ==============================================

import os
import re
import threading
import torch
import torchaudio
from transformers import (
    WhisperProcessor, WhisperForConditionalGeneration,
    Wav2Vec2Processor, Wav2Vec2ForSequenceClassification,
    AutoModelForCausalLM, AutoTokenizer
)

from modules.safety import (
    is_crisis,
    is_violence_toward_others,
    CRISIS_REPLY,
    VIOLENCE_RISK_REPLY,
    crisis_and_violence_reply,
)

_PIPELINE_LOCK = threading.Lock()
_whisper_processor = whisper_model = None
_wav2_processor = wav2_model = None
tokenizer = llm = None
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_WHISPER_SIZES = frozenset({"tiny", "base", "small", "medium", "large", "large-v2", "large-v3"})


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").lower() in ("1", "true", "yes")


def _whisper_repo_id() -> str:
    raw = (os.environ.get("MINDCARE_WHISPER_SIZE") or "small").strip().lower()
    if raw not in _WHISPER_SIZES:
        raw = "small"
    return f"openai/whisper-{raw}"


def _emotion_mode() -> str:
    """Return model | text | neutral. text/neutral skip Wav2Vec2 to save VRAM."""
    m = (os.environ.get("MINDCARE_EMOTION_MODE") or "model").strip().lower()
    if m not in ("model", "text", "neutral"):
        return "model"
    return m


def _load_whisper():
    global _whisper_processor, whisper_model
    if whisper_model is None:
        repo = _whisper_repo_id()
        _whisper_processor = WhisperProcessor.from_pretrained(repo)
        whisper_model = WhisperForConditionalGeneration.from_pretrained(repo)
        whisper_model.to(_DEVICE)
        whisper_model.eval()


def _load_wav2vec2():
    global _wav2_processor, wav2_model
    if _emotion_mode() != "model":
        return
    if wav2_model is None:
        _wav2_processor = Wav2Vec2Processor.from_pretrained("Dpngtm/wav2vec2-emotion-recognition")
        wav2_model = Wav2Vec2ForSequenceClassification.from_pretrained("Dpngtm/wav2vec2-emotion-recognition")
        wav2_model.to(_DEVICE)
        wav2_model.eval()


def _load_llm():
    global tokenizer, llm
    if llm is None:
        llm_path = os.environ.get("MINDCARE_LLM_REPO", "imnotdawn/mistral7b-qlora-sft-small-v2.1")
        tokenizer = AutoTokenizer.from_pretrained(llm_path)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        use_4bit = _env_flag("MINDCARE_LLM_LOAD_IN_4BIT") and _DEVICE.type == "cuda"
        if use_4bit:
            try:
                from transformers import BitsAndBytesConfig

                qconfig = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                )
                llm = AutoModelForCausalLM.from_pretrained(
                    llm_path,
                    quantization_config=qconfig,
                    device_map="auto",
                )
            except Exception:
                use_4bit = False

        if not use_4bit:
            llm = AutoModelForCausalLM.from_pretrained(llm_path)
            llm.to(_DEVICE)

        llm.eval()


def _ensure_whisper():
    with _PIPELINE_LOCK:
        _load_whisper()


def _ensure_wav2vec2():
    with _PIPELINE_LOCK:
        _load_wav2vec2()


def _ensure_llm():
    with _PIPELINE_LOCK:
        _load_llm()


def ensure_pipeline_ready():
    """Load all models once (thread-safe). Optional warmup before serving."""
    with _PIPELINE_LOCK:
        _load_whisper()
        if _emotion_mode() == "model":
            _load_wav2vec2()
        _load_llm()


def speech_to_text(audio_path):
    _ensure_whisper()
    waveform, sr = torchaudio.load(audio_path)

    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)

    inputs = _whisper_processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")
    inputs = {k: v.to(_DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        generated_ids = whisper_model.generate(**inputs)
    text = _whisper_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text


emotion_map = {
    0: "angry",
    1: "calm",
    2: "disgust",
    3: "fearful",
    4: "happy",
    5: "neutral",
    6: "sad",
    7: "surprised"
}

# English-focused lexicon for text-mode emotion hints (extend with other languages if needed).
_EMOTION_TEXT_SAD = (
    "sad", "depressed", "hopeless", "cry", "crying", "empty", "worthless", "grief",
    "miserable", "devastated", "heartbroken", "despair", "blue", "down",
)
_EMOTION_TEXT_ANGRY = (
    "angry", "mad", "furious", "annoyed", "rage", "hate", "livid", "resentful",
)
_EMOTION_TEXT_FEAR = (
    "fear", "scared", "afraid", "anxious", "panic", "worried", "nervous", "terrified",
    "frightened", "dread",
)
_EMOTION_TEXT_HAPPY = (
    "happy", "glad", "joy", "excited", "great", "wonderful", "cheerful", "delighted",
)


def _kw_hit(text: str, t: str, k: str) -> bool:
    if k.isascii():
        return k in t
    return k in text


def infer_emotion_from_text(text: str) -> str:
    """Lightweight text heuristic; weaker than the acoustic emotion model."""
    if not text:
        return "neutral"
    t = text.lower()
    scores = {
        "sad": sum(1 for k in _EMOTION_TEXT_SAD if _kw_hit(text, t, k)),
        "angry": sum(1 for k in _EMOTION_TEXT_ANGRY if _kw_hit(text, t, k)),
        "fearful": sum(1 for k in _EMOTION_TEXT_FEAR if _kw_hit(text, t, k)),
        "happy": sum(1 for k in _EMOTION_TEXT_HAPPY if _kw_hit(text, t, k)),
    }
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return "neutral"
    return best


def predict_emotion(audio_path, user_text=None):
    mode = _emotion_mode()
    if mode == "neutral":
        return "neutral"
    if mode == "text":
        return infer_emotion_from_text(user_text or "")

    _ensure_wav2vec2()
    if wav2_model is None:
        return infer_emotion_from_text(user_text or "")
    if not audio_path:
        return infer_emotion_from_text(user_text or "")

    waveform, sr = torchaudio.load(audio_path)

    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)

    inputs = _wav2_processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt")
    inputs = {k: v.to(_DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        logits = wav2_model(**inputs).logits
        pred = torch.argmax(logits, dim=-1).item()

    return emotion_map[pred]


def choose_strategy(emotion):
    if emotion in ["sad", "fearful", "angry"]:
        return "supportive_listening"
    if emotion == "neutral":
        return "gentle_exploration"
    if emotion in ["happy", "surprised"]:
        return "light_encouragement"
    return "gentle_exploration"

LOW_MOTIVATION_KEYWORDS = [
    "can't", "cannot", "won't", "don't know", "no point", "nothing helps",
    "too hard", "impossible", "give up", "hopeless", "stuck", "trapped"
]

AMBIVALENT_KEYWORDS = [
    "maybe", "perhaps", "sometimes", "both sides", "not sure", "confused",
    "mixed feelings", "unsure", "doubt", "considering", "thinking about"
]

EMERGING_KEYWORDS = [
    "might", "could try", "possibly", "thinking", "wondering",
    "starting to", "beginning to", "leaning towards", "inclined to"
]

READY_KEYWORDS = [
    "will", "going to", "plan to", "ready", "prepared", "decided",
    "commit", "start", "begin", "do it", "take action", "next step"
]


def detect_motivation_level(text):
    text_lower = text.lower()

    low_count = sum(1 for keyword in LOW_MOTIVATION_KEYWORDS if keyword in text_lower)
    ambivalent_count = sum(1 for keyword in AMBIVALENT_KEYWORDS if keyword in text_lower)
    emerging_count = sum(1 for keyword in EMERGING_KEYWORDS if keyword in text_lower)
    ready_count = sum(1 for keyword in READY_KEYWORDS if keyword in text_lower)

    if ready_count > 0 and ready_count >= max(emerging_count, ambivalent_count, low_count):
        return "ready"
    if emerging_count > 0 and emerging_count >= max(ambivalent_count, low_count):
        return "emerging"
    if ambivalent_count > 0 and ambivalent_count >= low_count:
        return "ambivalent"
    if low_count > 0:
        return "low"
    return "unknown"


def get_strategy_with_motivation(text, emotion):
    strategy = choose_strategy(emotion)
    if text:
        motivation_level = detect_motivation_level(text)
        if motivation_level == "ready":
            return "action_planning"
    return strategy


strategy_instruction_map = {
    "supportive_listening":
        "Reflect the user's emotions with warmth. Ask a gentle, open-ended question. Avoid straight advice.",
    "gentle_exploration":
        "Stay patient. Explore the user's feelings gradually. Avoid straightly giving solutions.",
    "light_encouragement":
        "Acknowledge the user's positive state and gently encourage them.",
    "action_planning":
        "User shows readiness. Help them define small achievable steps without pressure."
}

def build_prompt_messages(user_text, emotion, strategy, chat_history_list):

    final_strategy = get_strategy_with_motivation(user_text, emotion)
    strategy_rule = strategy_instruction_map[final_strategy]

    system_content = f"""You are a mental health support assistant trained in motivational interviewing (MI) and empathetic reflective listening.
Current Strategy: {strategy_rule}
Detected Emotion: {emotion}


CRITICAL RULES FOR OUTPUT:
1. Generate ONLY ONE single response to the user's last message.
2. STOP immediately after your response. Do NOT generate the user's reply.
3. Do NOT simulate a multi-turn conversation.
4. Do NOT use labels like "User:", "Assistant:", or "[INST]" in your output.
5. Keep your response concise (under 100 words if possible).
6. never rush or push the user
7. begin by reflecting the user's emotional experience
8. ask gentle open-ended questions
9. avoid advice unless user shows readiness
10. follow the user's pace


Safety (harm to self or others):
- If the user expresses intent to harm other people (not only themselves), briefly prioritize safety:
  ask whether thoughts are fleeting or persistent, whether there is any concrete plan, and encourage
  contacting local emergency services or a crisis line if risk may be imminent.
- Empathize with pain, anger, or hopelessness; do NOT praise, normalize, or romanticize violence
  toward others. Clearly separate "your feelings matter" from "hurting others is not an acceptable solution."
- State that you are an AI, not a confidential clinician, and cannot manage imminent violence risk alone.
- Offer one short grounding step (e.g. slow breathing, change of space) when intense anger or overwhelm appears.
"""

    messages = [{"role": "system", "content": system_content}]
    
    if chat_history_list:
        messages.extend(chat_history_list)
        
    messages.append({"role": "user", "content": user_text})
    
    return messages

def _sanitize_assistant_output(text: str) -> str:
    if not text:
        return text
    s = text.strip()
    
    s = re.sub(r"\s*\[/INST\]\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*\[INST\]\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*</s>\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*\[/?INST\]\s*", " ", s, flags=re.IGNORECASE)

    stop_patterns = [
        r"\n\s*User\s*:", 
        r"\n\s*Human\s*:", 
        r"\n\s*\[User\]",
        r"\n\s*\[INST\]",
        r"\]\s*I\s+",      # "] I..."
        r"\]\s*My\s+",     # "] My..."
        r"\]\s*It's\s+",   # "] It's..."
        r"\]\s*Not\s+",    # "] Not..."
        r"\]\s*No\s+",     # "] No..."
        r"\]\s*Yes\s+",    # "] Yes..."
        r"\]\s*Well\s+",   # "] Well..."
        r"\]\s*Actually\s+", # "] Actually..."
    ]

    for pattern in stop_patterns:
        match = re.search(pattern, s, re.IGNORECASE)
        if match:
            candidate = s[:match.start()].strip()
            if len(candidate) > 10:
                return candidate

    if s.count('?') > 1 and len(s) > 150:
        first_q_index = s.find('?')
        if first_q_index != -1:
            sentences = re.split(r'(?<=[.!?])\s+', s)
            if len(sentences) >= 1:
                return ' '.join(sentences[:2]).strip()

    s = re.sub(r'^[\]\[]+\s*', '', s)
    s = re.sub(r'\s*[\]\[]+$', '', s)
    
    if len(s) > 300:
        last_period = s.rfind('.', 0, 300)
        if last_period != -1:
            s = s[:last_period+1].strip()
        else:
            s = s[:300].strip() + "..."

    return s

def generate_llm_response(prompt_or_messages):
    _ensure_llm()
    
    if isinstance(prompt_or_messages, str):
        inputs = tokenizer(prompt_or_messages, return_tensors="pt", truncation=True, max_length=1024, padding=True)
    else:
        prompt_text = tokenizer.apply_chat_template(
            prompt_or_messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=1024, padding=True)

    dev = next(llm.parameters()).device
    inputs = {k: v.to(dev) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]
    
    max_new = int(os.environ.get("MINDCARE_MAX_NEW_TOKENS", "100"))
    max_new = max(32, min(max_new, 512))
    
    with torch.no_grad():
        output = llm.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        
    new_tokens = output[0][input_len:]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    
    return _sanitize_assistant_output(raw)


def mindcare_pipeline(audio_path, chat_history=""):
    user_text = speech_to_text(audio_path)
    emotion = predict_emotion(audio_path, user_text=user_text)

    if is_crisis(user_text):
        reply = crisis_and_violence_reply() if is_violence_toward_others(user_text) else CRISIS_REPLY
        tag = "crisis_and_violence_risk" if is_violence_toward_others(user_text) else "crisis_intervention"
        return {"user_text": user_text, "emotion": emotion, "strategy": tag, "reply": reply}
    if is_violence_toward_others(user_text):
        return {
            "user_text": user_text,
            "emotion": emotion,
            "strategy": "violence_risk_intervention",
            "reply": VIOLENCE_RISK_REPLY,
        }

    strategy = choose_strategy(emotion)
    prompt = build_prompt(user_text, emotion, strategy, chat_history)
    final_reply = generate_llm_response(prompt)

    return {
        "user_text": user_text,
        "emotion": emotion,
        "strategy": strategy,
        "reply": final_reply,
    }


if __name__ == "__main__":
    # Local demo only; use a real file path for audio.
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m modules.pipelines.p <path-to-audio.wav>")
        raise SystemExit(1)
    demo_audio = sys.argv[1]
    demo_history = """
[User]: I've been feeling really down lately.
[Assistant]: I hear that you're going through a tough time.
[User]: Yeah, nothing seems to help.
"""
    out = mindcare_pipeline(audio_path=demo_audio, chat_history=demo_history)
    print(out)
