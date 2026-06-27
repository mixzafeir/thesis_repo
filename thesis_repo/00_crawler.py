import os
import json
from telethon import TelegramClient
from config import Config

DATA_DIR = 'data_raw'
os.makedirs(DATA_DIR, exist_ok=True)

# ── Target volumes ────────────────────────────────────────────────────────────
TARGET_PER_SIDE = 50_000          # 50k pro-Palestine + 50k pro-Israel = 100k total
MES_CAP         = 20_000          # Middle_East_Spectator fixed cap

# ── Channel lists ─────────────────────────────────────────────────────────────
PRO_PALESTINE = [
    "Middle_East_Spectator",      # capped separately at MES_CAP
    "PalestineResist_Mirror",
    "gazaalanpa",
    "QudsNen",
    "AlHaqNews",
    "DDGeopolitics",
    "LebUpdate",
    "SimurghRes",
]

PRO_ISRAEL = [
    "englishabuali",
    "Israel_Realtime_Updates",
    "TheTimesOfIsrael2022",
    "The_Jerusalem_Post",
    "StandWithUsBreakingNews",
    "HonestReporting",
    "ILtoday",
    "OSINTdefender",
]


# ── Cap calculation ───────────────────────────────────────────────────────────
def compute_caps():
    """
    Middle_East_Spectator → MES_CAP (20k)
    Other pro-Palestine   → remaining 30k split equally
    Pro-Israel            → 50k split equally
    """
    pal_others = [ch for ch in PRO_PALESTINE if ch != "Middle_East_Spectator"]
    cap_pal_other = (TARGET_PER_SIDE - MES_CAP) // len(pal_others)
    cap_isr       = TARGET_PER_SIDE // len(PRO_ISRAEL)

    caps = {"Middle_East_Spectator": MES_CAP}
    for ch in pal_others:
        caps[ch] = cap_pal_other
    for ch in PRO_ISRAEL:
        caps[ch] = cap_isr

    return caps


# ── Client ────────────────────────────────────────────────────────────────────
client = TelegramClient(
    Config['username'],
    Config['api_id'],
    Config['api_hash'],
)


# ── Crawl one channel ─────────────────────────────────────────────────────────
async def crawl_channel(channel, label, limit):
    out_path = os.path.join(DATA_DIR, f"{channel}.json")

    # Resume: skip channels that already have enough messages
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        if len(existing) >= limit:
            print(f"  {channel}: already {len(existing):,} msgs — skipping")
            return len(existing)

    messages = []
    seen_ids = set()

    print(f"  {channel}  [label={label}, target={limit:,}]")
    async for msg in client.iter_messages(channel, limit=limit * 2):
        if not msg.text:
            continue
        if msg.id in seen_ids:
            continue
        seen_ids.add(msg.id)
        messages.append({
            "id":        msg.id,
            "message":   msg.text,
            "timestamp": str(msg.date),
            "label":     label,
        })
        if len(messages) >= limit:
            break

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)

    print(f"  → saved {len(messages):,}  ({out_path})")
    return len(messages)


# ── Main ──────────────────────────────────────────────────────────────────────
async def main():
    await client.start(phone=Config['phone'])

    caps = compute_caps()

    print(f"\n{'='*55}")
    print(f"  Target: {TARGET_PER_SIDE:,} pro-Palestine + {TARGET_PER_SIDE:,} pro-Israel")
    print(f"  Channel caps:")
    for ch in PRO_PALESTINE:
        print(f"    pro_palestine  {ch:<35} {caps[ch]:>6,}")
    for ch in PRO_ISRAEL:
        print(f"    pro_israel     {ch:<35} {caps[ch]:>6,}")
    print(f"{'='*55}\n")

    print("── Pro-Palestine ──")
    pal_total = 0
    for ch in PRO_PALESTINE:
        pal_total += await crawl_channel(ch, "pro_palestine", caps[ch])
    print(f"Pro-Palestine total: {pal_total:,}\n")

    print("── Pro-Israel ──")
    isr_total = 0
    for ch in PRO_ISRAEL:
        isr_total += await crawl_channel(ch, "pro_israel", caps[ch])
    print(f"Pro-Israel total: {isr_total:,}")

    print(f"\nGrand total: {pal_total + isr_total:,} / 100,000")


with client:
    client.loop.run_until_complete(main())
