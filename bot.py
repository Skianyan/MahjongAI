from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).parent / "src"))

from mahjong_ai.bot_runner import main


if __name__ == "__main__":
    main()