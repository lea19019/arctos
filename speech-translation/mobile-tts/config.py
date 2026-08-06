from pathlib import Path

# --- Paths ---
DATA_ROOT = Path("/home/vacl2/groups/grp_mtlab/nobackup/autodelete/african_tts/swahili/swahili_cleaned")
METADATA_TRAIN = DATA_ROOT / "metadata_train_clean.csv"
METADATA_TEST  = DATA_ROOT / "metadata_test_clean.csv"
METADATA_EVAL  = DATA_ROOT / "metadata_eval_clean.csv"

PROJECT_ROOT = Path(__file__).parent
OUTPUTS_DIR  = PROJECT_ROOT / "outputs"
DATA_DIR     = PROJECT_ROOT / "data"

# Filtered splits written by prep_swahili.py
SPK7_TRAIN = DATA_DIR / "spk7_train.csv"
SPK7_TEST  = DATA_DIR / "spk7_test.csv"
SPK7_EVAL  = DATA_DIR / "spk7_eval.csv"

# --- Model ---
MODEL_ID      = "facebook/mms-tts-swh"
MODEL_CACHE   = Path("/home/vacl2/groups/grp_mtlab/nobackup/autodelete/african_tts/models/mms-tts-swh")

# --- Data ---
TARGET_SPEAKER = "7"
SAMPLE_RATE    = 16000   # MMS-TTS output sample rate

# --- Fine-tuning ---
FT_OUTPUT_DIR  = Path("/home/vacl2/groups/grp_mtlab/nobackup/autodelete/african_tts/mobile-tts-checkpoints")
FT_BATCH_SIZE  = 16
FT_EPOCHS      = 30
FT_LR          = 2e-4
FT_WARMUP_STEPS = 500
FT_SAVE_STEPS  = 500
FT_EVAL_STEPS  = 500
FT_MAX_AUDIO_LEN = 10.0   # seconds — drop clips longer than this

# --- Evaluation ---
TEST_SENTENCES = [
    "Habari yako leo?",
    "Ninafurahi kukutana nawe.",
    "Watoto wanacheza mpira.",
    "Daktari anasema kwamba unahitaji kupumzika.",
    "Serikali imepanga mkutano mkubwa kesho asubuhi.",
]
