from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run(cmd, cwd: Path) -> None:
    print("[smoke] >", " ".join(str(c) for c in cmd))
    result = subprocess.run(cmd, cwd=str(cwd))
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main() -> None:
    out_weights = PROJECT_ROOT / "out" / "letter_gen.pt"
    out_config = PROJECT_ROOT / "out" / "letter_gen_config.json"
    out_png = PROJECT_ROOT / "out" / "generated" / "smoke_hello_world.png"

    run(
        [
            sys.executable,
            "-m",
            "python_ai.lettergen.train",
            "--epochs",
            "1",
            "--batch-size",
            "32",
            "--beta-warmup-epochs",
            "1",
            "--out-weights",
            str(out_weights),
            "--out-config",
            str(out_config),
        ],
        cwd=PROJECT_ROOT,
    )

    run(
        [
            sys.executable,
            "generate_handwriting_page.py",
            "--text",
            "hello world",
            "--style",
            "ink",
            "--width",
            "960",
            "--height",
            "360",
            "--line-spacing",
            "1.25",
            "--out",
            str(out_png),
            "--use-letter-model",
            "--letter-model-path",
            str(out_weights),
            "--style-strength",
            "1.0",
            "--json",
        ],
        cwd=PROJECT_ROOT,
    )

    print("[smoke] outputs:")
    print("  weights:", out_weights)
    print("  config :", out_config)
    print("  image  :", out_png)


if __name__ == "__main__":
    main()
