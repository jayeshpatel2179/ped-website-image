"""
Telegram bot: send a photo, tap "Go", get back a horizontal (landscape)
editorial news graphic generated with OpenAI's gpt-image-2 model, using a
fixed default prompt.

Usage:
  1. Send a photo to the bot.
  2. The bot replies with "Go" / "Cancel" buttons.
  3. Tap "Go" to generate the editorial graphic (or "Cancel" to discard it).

Setup:
  pip install -r requirements.txt
  copy .env.example .env   # then fill in TELEGRAM_BOT_TOKEN and OPENAI_API_KEY
  python telegram_image_bot.py
"""

import asyncio
import base64
import io
import logging
import os

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

load_dotenv()

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

# Landscape/horizontal output. Must be WIDTHxHEIGHT, both multiples of 16,
# aspect ratio between 1:3 and 3:1.
IMAGE_SIZE = "2048x1024"
IMAGE_MODEL = "gpt-image-2"

DEFAULT_PROMPT = (
    "Create a editorial news graphic** that combines the visual language of a "
    "vintage newspaper front page with modern magazine-style graphic design. Use "
    "the provided image as the main visual. Remove all existing text, logos, "
    "badges, watermarks, score graphics, interface elements, and unwanted "
    "overlays. Isolate the main subject cleanly and position it prominently on "
    "the middle of the composition, allowing parts of the subject to overlap "
    "the paper layers for depth. Build the background with layered ripped "
    "newspaper sheets, torn paper edges, aged cream newsprint, halftone dots, "
    "ink grain, distressed print textures, photocopy marks, folded-paper "
    "shadows, subtle article columns, and abstract editorial shapes. Use a "
    "limited color palette based on the story, brand, country, team, company, "
    "or subject featured in the news. Integrate these colors through painted "
    "paper fragments, ink splashes, geometric blocks, stamps, symbols, maps, "
    "flags, or abstract visual elements relevant to the story. The result "
    "should feel like a premium editorial news poster, newspaper collage, and "
    "modern magazine cover combined. Use strong visual hierarchy, dramatic "
    "scale, balanced spacing, high contrast, realistic paper depth, and "
    "professional graphic design. Keep the composition bold, clean, and "
    "uncluttered. Preserve the original subject's face, body, clothing, pose, "
    "and identity accurately. Do not create duplicate subjects, extra limbs, "
    "distorted faces, random logos, misspelled text, or irrelevant decorative "
    "elements. Output in exact landscape format. Just don't overdo the "
    "designs, please make sure the clutter in the design is less and the "
    "headline is visible."
)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

openai_client = OpenAI(api_key=OPENAI_API_KEY)

# chat_id -> pending photo bytes (waiting for Go/Cancel)
pending_photos: dict[int, bytes] = {}

GO_CANCEL_KEYBOARD = InlineKeyboardMarkup(
    [
        [
            InlineKeyboardButton("Go", callback_data="go"),
            InlineKeyboardButton("Cancel", callback_data="cancel"),
        ]
    ]
)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Send me a photo. I'll turn it into a horizontal editorial news "
        "graphic (vintage newspaper + modern magazine style) using gpt-image-2."
    )


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    photo = update.message.photo[-1]  # highest resolution
    file = await context.bot.get_file(photo.file_id)
    photo_bytes = bytes(await file.download_as_bytearray())

    pending_photos[chat_id] = photo_bytes
    await update.message.reply_text(
        "Got the image. Generate the editorial news graphic?",
        reply_markup=GO_CANCEL_KEYBOARD,
    )


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Please send a photo to get started.")


async def handle_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    chat_id = query.message.chat_id

    if query.data == "cancel":
        pending_photos.pop(chat_id, None)
        await query.edit_message_text("Cancelled. Send a new photo any time.")
        return

    if query.data == "go":
        photo_bytes = pending_photos.get(chat_id)
        if photo_bytes is None:
            await query.edit_message_text(
                "I don't have a pending image anymore. Please send a photo again."
            )
            return

        await query.edit_message_text("Generating your editorial graphic, please wait...")
        try:
            png_bytes = to_png(photo_bytes)
            result_bytes = await asyncio.to_thread(edit_image, png_bytes, DEFAULT_PROMPT)
            await context.bot.send_photo(chat_id=chat_id, photo=io.BytesIO(result_bytes))
            await query.edit_message_text("Done.")
        except Exception:
            logger.exception("Image generation failed")
            await query.edit_message_text(
                "Sorry, something went wrong generating that image. Please try again."
            )
        finally:
            pending_photos.pop(chat_id, None)


def to_png(image_bytes: bytes) -> bytes:
    """gpt-image-2's edit endpoint expects PNG/WEBP input, so normalize."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    buf.name = "input.png"
    return buf.getvalue()


def edit_image(png_bytes: bytes, prompt: str) -> bytes:
    image_file = io.BytesIO(png_bytes)
    image_file.name = "input.png"

    result = openai_client.images.edit(
        model=IMAGE_MODEL,
        image=image_file,
        prompt=prompt,
        size=IMAGE_SIZE,
    )
    b64_data = result.data[0].b64_json
    return base64.b64decode(b64_data)


def main() -> None:
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(CallbackQueryHandler(handle_button))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    logger.info("Bot started. Press Ctrl+C to stop.")
    app.run_polling()


if __name__ == "__main__":
    main()
