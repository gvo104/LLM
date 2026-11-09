from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"


def load_prompt(name: str) -> str:
    path = (PROMPTS_DIR / name).resolve()
    if PROMPTS_DIR not in path.parents:
        raise ValueError("Prompt должен находиться в папке prompts.")
    if not path.is_file():
        raise FileNotFoundError(f"Prompt не найден: {name}")
    return path.read_text(encoding="utf-8")
