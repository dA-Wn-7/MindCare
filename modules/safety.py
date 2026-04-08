# Keyword interception and fixed safety responses (self-harm + violence toward others).
# Shared by app and pipeline.
# Lists named *_ZH contain Chinese substrings on purpose so messages typed in Chinese still match.

CRISIS_KEYWORDS_EN = [
    "suicide",
    "kill myself",
    "want to die",
    "end it all",
    "end my life",
    "self-harm",
    "cut myself",
    "hang myself",
    "jump off",
    "overdose",
]
CRISIS_KEYWORDS_ZH = [
    "自杀",
    "自殺",
    "不想活",
    "结束生命",
    "了结",
    "割腕",
    "跳楼",
    "服毒",
    "吃安眠药",
    "一死了之",
]

VIOLENCE_TO_OTHERS_KEYWORDS_EN = [
    "kill everyone",
    "kill them all",
    "kill all people",
    "kill all humans",
    "kill the world",
    "kill the whole world",
    "murder everyone",
    "mass shooting",
    "school shooting",
    "shoot everyone",
    "shoot up",
    "go on a rampage",
    "slaughter everyone",
    "wipe out humanity",
    "wipe everyone",
    "bomb everyone",
    "hurt everyone",
    "eliminate everyone",
    "want to kill everyone",
    "kill all of them",
]
VIOLENCE_TO_OTHERS_KEYWORDS_ZH = [
    "想杀死全世界",
    "杀死全世界",
    "杀光所有人",
    "杀死所有人",
    "杀光",
    "报复社会",
    "无差别杀人",
    "无差别伤人",
    "滥杀",
    "屠杀",
    "想杀光",
    "想杀死所有人",
    "杀一个是一个",
    "同归于尽",
    "弄死所有人",
    "杀很多人",
    "随机砍",
]


def is_crisis(text: str) -> bool:
    if not text:
        return False
    t = text.lower()
    if any(k in text for k in CRISIS_KEYWORDS_ZH):
        return True
    return any(k in t for k in CRISIS_KEYWORDS_EN)


def is_violence_toward_others(text: str) -> bool:
    if not text:
        return False
    t = text.lower()
    if any(k in text for k in VIOLENCE_TO_OTHERS_KEYWORDS_ZH):
        return True
    return any(k in t for k in VIOLENCE_TO_OTHERS_KEYWORDS_EN)


CRISIS_REPLY = (
    "We are very concerned for your safety. This tool **cannot** provide crisis intervention. "
    "If you are having thoughts of self-harm or suicide, please **immediately** contact local "
    "emergency services or a professional crisis hotline, or go to the nearest emergency department "
    "with someone you trust. You deserve real human support."
)

VIOLENCE_RISK_REPLY = """What you shared may involve **risk of harm to other people or groups**. I have to put **safety** first. Please read the following.

**1) Boundaries and informed consent**  
I am an AI. I **cannot** perform an in-person risk assessment or crisis response, and I **cannot replace** police, emergency medical services, or licensed mental-health or psychiatric care.  
If you already have a **concrete plan to harm someone** (including timing, target, or means), or easy access to weapons or tools, that is a **public safety and legal** matter. You should **not assume this chat can remain fully confidential**; contacting local services (for example **110 / 120** in mainland China, or your country's police and emergency medical number) may save lives. If you are abroad, dial your **local emergency number**.

**2) Safety assessment (answer only what you feel okay sharing)**  
- Do these thoughts **flash by occasionally**, or **keep returning and feel stronger**?  
- Is there any **fairly specific** plan or method?  
- Is it easy to get hold of **something that could hurt someone**?  
- Is the anger or urge **aimed at a particular person, group, or situation**?

**3) Empathy with limits**  
I understand that someone overwhelmed by pain, anger, or helplessness may think or say extreme things. I **empathize with the pain you are carrying**, but I **do not agree with or romanticize** using violence against others as a solution—it can cause irreversible harm to you and others.

**4) Something you can try right now**  
If your body feels tight or your mind is racing: try **slow breathing** (for example inhale about 4 seconds, brief hold, exhale about 6 seconds) a few times; **move to a safer space**; **clench your fists and slowly release** to ease muscle tension. If you want, you can say whether you are **alone** and **reasonably safe** (no need for private details).

**5) Help resources (examples—verify current local numbers)**  
- Mainland China (examples): police **110**; emergency medical **120**; psychological crisis line (verify current): **010-82951332**  
- Elsewhere: use your **local emergency number** and **crisis hotline**.

If you are willing, you could share in a sentence or two: **Has this urge been getting stronger? Is there someone you trust whom you could contact right now?**

I am an AI and **cannot** promise confidentiality if there is a **concrete plan** to hurt people—contact **emergency services** if danger may be imminent. I **do not endorse violence** toward others. If you can, try **slow breathing**, **changing your environment**, or reaching a **crisis line** or someone you trust.
"""


def crisis_and_violence_reply() -> str:
    return CRISIS_REPLY + "\n\n────────\n\n" + VIOLENCE_RISK_REPLY
