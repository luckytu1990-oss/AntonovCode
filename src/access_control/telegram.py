from __future__ import annotations

from pathlib import Path

import requests

from access_control.config import TelegramConfig


class TelegramNotifier:
    def __init__(self, config: TelegramConfig) -> None:
        self.config = config

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def send_photo(self, image_path: Path, caption: str) -> None:
        if not self.config.enabled:
            return
        if not self.config.bot_token or not self.config.chat_id:
            raise ValueError("Для Telegram нужны bot_token и chat_id")

        url = f"https://api.telegram.org/bot{self.config.bot_token}/sendPhoto"
        with image_path.open("rb") as image_file:
            response = requests.post(
                url,
                data={"chat_id": self.config.chat_id, "caption": caption},
                files={"photo": image_file},
                timeout=20,
            )
        response.raise_for_status()
