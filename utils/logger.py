import logging
import os

def setup_logger():
    # 📁 إنشاء dossier logs
    os.makedirs("logs", exist_ok=True)

    logger = logging.getLogger("ai_project")
    logger.setLevel(logging.INFO)

    # 📝 format ديال logs
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )

    # 📄 log file
    file_handler = logging.FileHandler("logs/app.log")
    file_handler.setFormatter(formatter)

    # 🖥️ terminal output
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # إضافة handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger