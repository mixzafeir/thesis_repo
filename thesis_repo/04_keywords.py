# 04_keywords.py — Keyword stance detection + framing analysis
# Run: python 04_keywords.py

import os
import re
import pandas as pd
from tqdm import tqdm

PROC_DIR  = "data_processed"
SENT_OUT  = os.path.join(PROC_DIR, "messages_clean_plus_sentiment.parquet")
STANCE_OUT = os.path.join(PROC_DIR, "messages_plus_sentiment_keywords.parquet")


PRO_PAL = {
    # weight 3 — unambiguous pro-Palestinian framing
    "free palestine": 3, "from the river to the sea": 3, "iof": 3,
    "ethnic cleansing": 3, "settler colonialism": 3, "genocide": 3,
    "genocidal": 3, "child killer": 3, "child killers": 3, "baby killer": 3, "baby killers": 3, "open air prison": 3,
    "white phosphorus": 3, "mass grave": 3, "mass graves": 3,
    "genocidal intent": 3, "plausible genocide": 3, "nakba 2": 3,
    "second nakba": 3, "hasbara": 3, "genocide joe": 3, "land theft": 3,
    "forced transfer": 3, "bds": 3, "boycott israel": 3,
    "#freepalestine": 3, "#palestinegenocide": 3, "#gazagenocide": 3,
    "#ceasefirenow": 3, "#savegaza": 3, "#gazaunderattack": 3,
    "#stopthegenocide": 3, "#standwithpalestine": 3, "#endtheoccupation": 3,
    "#palestinewillbefree": 3, "#boycottisrael": 3, "#nakba76": 3,
    "axis of resistance": 3, "death to israel": 3, "zionist regime will fall": 3,
    "wipe israel": 3, "israel will cease to exist": 3, "muqawama": 3,
    "izz ad-din al-qassam": 3,
    # weight 2 — strong signal
    "end the occupation": 2, "apartheid": 2, "apartheid state": 2,
    "apartheid wall": 2, "separation wall": 2, "war crimes": 2,
    "war criminal": 2, "war criminals": 2, "collective punishment": 2, "zionists": 2,
    "zionist regime": 2, "zionist entity": 2, "settler": 2, "settlers": 2,
    "nakba": 2, "al nakba": 2, "#nakba": 2, "intifada": 2,
    "decolonize": 2, "decolonise": 2, "decolonisation": 2, "decolonization": 2,
    "occupation forces": 2, "siege of gaza": 2, "total siege": 2,
    "total blockade": 2, "massacre": 2, "massacres": 2, "unrwa": 2, "resistance": 2,
    "armed resistance": 2, "popular resistance": 2, "pinkwashing": 2,
    "artwashing": 2, "sportwashing": 2, "al-aqsa": 2, "masjid al-aqsa": 2,
    "storming al-aqsa": 2, "raid on al-aqsa": 2, "icc": 2, "icj": 2,
    "world court": 2, "genocide case": 2, "south africa vs israel": 2,
    "provisional measures": 2, "arrest warrant": 2, "netanyahu arrested": 2,
    "illegal settlements": 2, "settlement expansion": 2, "settler violence": 2,
    "colonizer": 2, "colonizers": 2, "colonial project": 2, "colonial state": 2,
    "right of return": 2, "right to return": 2, "1948": 2, "sheikh jarrah": 2,
    "al-shifa": 2, "shifa hospital": 2, "hospital bombing": 2, "hospital attack": 2,
    "targeting civilians": 2, "journalists killed": 2, "targeting journalists": 2,
    "media blackout": 2, "communication blackout": 2, "internet blackout": 2,
    "refugee camp": 2, "refugee camps": 2, "starvation": 2, "famine": 2,
    "aid blocked": 2, "aid convoy": 2, "arms embargo": 2, "weapons to israel": 2,
    "us weapons": 2, "american weapons": 2, "divestment": 2, "academic boycott": 2,
    "cultural boycott": 2, "ban israel": 2, "expel israel": 2, "suspend israel": 2,
    "us veto": 2, "impunity": 2, "al-quds": 2, "occupied jerusalem": 2,
    "east jerusalem": 2, "forced eviction": 2, "home demolition": 2,
    "house demolition": 2, "house demolitions": 2, "home demolitions": 2, "land grab": 2, "desecration": 2,
    "university encampment": 2, "student encampment": 2,
    "solidarity with palestine": 2, "solidarity with gaza": 2,
    "stand with palestine": 2, "jabalia": 2, "khan younis": 2,
    "deir al-balah": 2, "kamal adwan": 2, "red crescent": 2,
    "international humanitarian law": 2, "geneva convention": 2,
    "humanitarian law": 2, "children killed": 2, "civilian deaths": 2,
    "islamic resistance": 2, "resistance front": 2, "resistance forces": 2,
    "ansar allah": 2, "houthis": 2, "houthi": 2, "nasrallah": 2,
    "hassan nasrallah": 2, "soleimani": 2, "qasem soleimani": 2,
    "islamic jihad": 2, "pij": 2, "qassam brigades": 2, "al-qassam": 2,
    "death to america": 2,
    "occupation": 2, "occupied": 2,
    # weight 1 — moderate signal
    "ceasefire": 1, "west bank": 1, "gaza": 1, "gazan": 1, "gazans": 1,
    "palestinians": 1, "palestinian people": 1,
    "rafah": 1, "jenin": 1, "nablus": 1, "ramallah": 1, "bethlehem": 1,
    "hebron": 1, "blockade": 1, "displaced": 1, "displacement": 1,
    "humanitarian crisis": 1, "humanitarian corridor": 1, "civilian casualties": 1,
    "zionist": 1, "liberation": 1, "liberate palestine": 1, "free gaza": 1,
    "pray for palestine": 1, "pray for gaza": 1, "encampment": 1,
    "mass casualty": 1, "beit lahiya": 1, "beit hanoun": 1, "nuseirat": 1,
    "rubble": 1, "aid workers": 1, "world food programme": 1, "malnutrition": 1,
    "electricity cut": 1, "power cut": 1, "water cut": 1, "pre-1967": 1,
    "1967 borders": 1, "security council veto": 1, "indigenous rights": 1,
    "indigenous people of palestine": 1, "summary execution": 1,
    "extrajudicial": 1, "irgc": 1, "khamenei": 1,
    "settlement": 1, "sniper": 1, "snipers": 1, "bombardment": 1, "bombardments": 1,
    "prisoner": 1, "prisoners": 1, "detainee": 1, "detainees": 1,
}

PRO_ISR = {
    # weight 3 — unambiguous pro-Israeli framing
    "stand with israel": 3, "never again": 3, "#bringthemhomenow": 3,
    "#standwithisrael": 3, "#israelunderattack": 3, "#bringthemhome": 3,
    "#hostagesdeals": 3, "#neveragain": 3, "am yisrael chai": 3,
    "october 7 massacre": 3, "supernova massacre": 3, "hamas isis": 3,
    "hamas is isis": 3, "hamas like isis": 3, "hamas atrocities": 3,
    "hamas raped": 3, "raped and murdered": 3, "hamas murderers": 3,
    "hamas rapists": 3, "nova festival massacre": 3, "pogrom": 3,
    "never forget october 7": 3, "remember october 7": 3,
    "anti-zionism is antisemitism": 3, "zionism is not racism": 3,
    "proud zionist": 3, "jewish self-determination": 3,
    "campus antisemitism": 3, "#israelidf": 3, "#october7": 3,
    "iran funds hamas": 3, "iran backs hamas": 3, "iran backed hamas": 3,
    "iran funds hezbollah": 3, "iran backs hezbollah": 3,
    "iran sponsors terror": 3, "iran sponsor of terrorism": 3,
    "death to the mullahs": 3, "irgc terror": 3, "iran must be stopped": 3,
    "iran deal was a mistake": 3, "proxy war against israel": 3,
    # weight 2 — strong signal
    "bring them home": 2, "right to defend": 2, "right to self-defense": 2,
    "israel has the right": 2, "israel's right to exist": 2, "right to exist": 2,
    "antisemitism": 2, "antisemitic": 2, "antisemit": 2, "anti-semit": 2,
    "antisemitic attack": 2, "antisemitic incident": 2, "human shields": 2,
    "human shield": 2, "october 7": 2, "oct 7": 2, "10/7": 2,
    "nova festival": 2, "supernova festival": 2, "kibbutz": 2, "iron dome": 2,
    "sinwar": 2, "yahya sinwar": 2, "hostages": 2, "civilian hostages": 2,
    "child hostages": 2, "hamas kidnapped": 2, "kidnapped israelis": 2,
    "hamas charter": 2, "defend israel": 2, "pro-israel": 2,
    "defend the jewish": 2, "protect jewish": 2, "jewish security": 2,
    "kfar aza": 2, "nir oz": 2, "be'eri": 2, "beeri kibbutz": 2,
    "iran proxy": 2, "iran backed": 2, "iran funding": 2,
    "terror tunnel": 2, "terror tunnels": 2, "hamas tunnel": 2, "hamas tunnels": 2,
    "rocket fire": 2, "rocket attack": 2, "rockets from gaza": 2,
    "hamas rocket": 2, "qassam rocket": 2, "hamas terror": 2,
    "hamas attack": 2, "hamas launched": 2, "terrorist": 2, "terrorists": 2,
    "terrorism": 2, "terror attack": 2, "terror attacks": 2, "terrorist organization": 2,
    "designated terrorist": 2, "antisemite": 2, "jewish homeland": 2, "jewish sovereignty": 2,
    "eretz israel": 2, "land of israel": 2, "harassing jewish": 2,
    "targeting jewish": 2, "jewish students attacked": 2,
    "iranian regime": 2, "iranian proxies": 2, "iran's proxies": 2,
    "state sponsor of terror": 2, "mullahs": 2, "the mullahs": 2,
    "ayatollah regime": 2, "regime in tehran": 2, "tehran regime": 2,
    "irgc designated": 2, "iranian nuclear threat": 2, "stop iran": 2,
    "contain iran": 2, "jcpoa failure": 2, "iranian aggression": 2,
    "iranian expansionism": 2, "iranian imperialism": 2,
    "hezbollah terror": 2, "hezbollah attacks": 2, "hezbollah rockets": 2,
    "northern israel attacked": 2, "houthi attacks": 2,
    "houthi missiles": 2, "houthi drones": 2, "regime": 2,
    # weight 1 — moderate signal
    "idf soldiers": 1, "idf operation": 1, "israeli military": 1,
    "jewish state": 1, "state of israel": 1, "two-state solution": 1,
    "abraham accords": 1, "normalization": 1, "precision strike": 1,
    "targeted strike": 1, "surgical strike": 1, "roof knocking": 1,
    "warned civilians": 1, "evacuation warning": 1, "southern israel": 1,
    "border communities": 1, "israeli communities": 1,
    "weapons cache": 1, "missile stockpile": 1, "temple mount": 1,
    "western wall": 1, "biblical land": 1, "promised land": 1,
    "freed hostages": 1, "hostage deal": 1, "hostage release": 1,
    "hamas holds": 1, "held hostage": 1, "holocaust survivor": 1,
    "6 million jews": 1,
    "conflict": 1, "retaliation": 1, "clashes": 1, "escalation": 1,
}

DEATH_CONTEXT = {
    "killed", "dead", "casualties", "martyr", "martyred", "eliminated",
    "airstrike", "bombing", "massacre", "civilian deaths", "bodies",
    "wounded", "died", "death toll", "fatalities", "slain", "murdered",
    "executions", "corpses", "genocide victims", "body bags", "dead bodies",
}

VICTIM_FRAMING = {
    "civilians", "children", "child", "kids", "babies", "infant",
    "hospitals", "hospital", "refugees", "refugee camp", "displaced",
    "innocent", "families", "women", "elderly", "humanitarian",
    "aid workers", "medics", "doctors", "nurses", "patients",
}

MILITARY_FRAMING = {
    "operation", "strike", "targeted", "eliminated", "neutralized",
    "airstrike", "precision strike", "surgical strike", "ground offensive",
    "troops", "soldiers", "idf", "forces", "destroyed", "dismantled",
    "launched", "fired", "offensive", "combat", "mission",
}

LEGAL_FRAMING = {
    "war crimes", "icc", "icj", "genocide", "international law",
    "humanitarian law", "geneva convention", "war criminal",
    "accountability", "arrest warrant", "violations", "unlawful",
    "illegal", "international court", "human rights", "amnesty",
    "un resolution", "security council", "ceasefire resolution",
}


def _clean(s):
    s = s.lower()
    s = re.sub(r"http\S+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _score(text, terms):
    return sum(w for t, w in terms.items() if t in text)


def run_stance_keywords():
    if not os.path.exists(SENT_OUT):
        raise SystemExit(f"Missing: {SENT_OUT} — run 03_sentiment.py first.")
    df = pd.read_parquet(SENT_OUT).copy()
    labels, scores = [], []
    for txt in tqdm(df["text"].astype(str), desc="Keyword stance"):
        t   = _clean(txt)
        pal = _score(t, PRO_PAL)
        isr = _score(t, PRO_ISR)
        if pal == 0 and isr == 0:
            labels.append("neutral");      scores.append(0.0)
        elif pal > isr:
            labels.append("pro_palestine"); scores.append(round((pal - isr) / max(pal, isr), 4))
        elif isr > pal:
            labels.append("pro_israel");    scores.append(round((isr - pal) / max(pal, isr), 4))
        else:
            labels.append("neutral");      scores.append(0.0)
    df["stance_label"] = labels
    df["stance_score"]  = scores
    os.makedirs(PROC_DIR, exist_ok=True)
    df.to_parquet(STANCE_OUT, index=False)
    print(f"Keyword stance done: {len(df):,} rows  →  {STANCE_OUT}")
    print(df["stance_label"].value_counts())
    return df


def run_framing_analysis():
    if not os.path.exists(STANCE_OUT):
        raise SystemExit(f"Missing: {STANCE_OUT} — run stance keywords first.")
    df = pd.read_parquet(STANCE_OUT).copy()

    def has_any(text, terms):
        t = _clean(str(text))
        return any(term in t for term in terms)

    print("\n── Framing analysis ──")
    for flag, terms, label in [
        ("flag_death_context", DEATH_CONTEXT,    "Death context"),
        ("flag_victim",        VICTIM_FRAMING,   "Victim framing"),
        ("flag_military",      MILITARY_FRAMING, "Military framing"),
        ("flag_legal",         LEGAL_FRAMING,    "Legal/intl law framing"),
    ]:
        tqdm.pandas(desc=label)
        df[flag] = df["text"].progress_apply(lambda x, t=terms: has_any(x, t))
        n = df[flag].sum()
        print(f"  {label}: {n:,} posts ({n / len(df) * 100:.1f}%)")

    df.to_parquet(STANCE_OUT, index=False)
    print(f"Framing flags saved  →  {STANCE_OUT}")
    return df


def main():
    run_stance_keywords()
    run_framing_analysis()
    print("\nDone. Next: python 05_reports.py")


if __name__ == "__main__":
    main()
