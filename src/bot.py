import asyncio
import logging
from datetime import datetime

import aiohttp
import pandas as pd
from pymongo import MongoClient
from telegram import KeyboardButton, ReplyKeyboardMarkup, Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CallbackContext, CommandHandler, MessageHandler, filters, CallbackQueryHandler

import urllib.parse
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

from config import HANDLER_API_URL, MAX_FAQ_MINUS_ATTEMPTS, MONGO_DB_PATH, MONGO_DB_NAME, SECRET_TOKEN, TOKEN

mongo_client = MongoClient(MONGO_DB_PATH)
db = mongo_client[MONGO_DB_NAME]

logs_collection = db["telegram_bot_logs"]
blacklist_collection = db["user_blacklist"]
faq_minus_logs = db["faq_minus_logs"]


class BotLogger:
    """Класс для логирования действий бота в MongoDB"""

    @staticmethod
    def log_action(user_id: int, action: str, details: str = ""):
        """Логирование действия пользователя"""
        log_entry = {"user_id": user_id, "action": action, "details": details, "timestamp": datetime.now()}
        logs_collection.insert_one(log_entry)
        logger.info(f"Logged action: {action} for user {user_id}")


class BlacklistManager:
    """Класс для управления блэк-листом пользователей"""

    MAX_FAQ_MINUS_ATTEMPTS = MAX_FAQ_MINUS_ATTEMPTS

    @classmethod
    def check_blacklist(cls, user_id: int) -> bool:
        """Проверка, находится ли пользователь в блэк-листе"""
        return bool(blacklist_collection.find_one({"user_id": user_id}))

    @classmethod
    def add_to_blacklist(cls, user_id: int, reason: str):
        """Добавление пользователя в блэк-лист"""
        if not cls.check_blacklist(user_id):
            blacklist_collection.insert_one({"user_id": user_id, "reason": reason, "timestamp": datetime.now()})
            logger.warning(f"User {user_id} added to blacklist. Reason: {reason}")

    @classmethod
    def log_faq_minus_attempt(cls, user_id: int, question: str, answer: str):
        """Логирование попытки задать стоп-вопрос"""
        faq_minus_logs.insert_one(
            {"user_id": user_id, "question": question, "answer": answer, "timestamp": datetime.now()}
        )

        attempts = faq_minus_logs.count_documents({"user_id": user_id})
        if attempts >= cls.MAX_FAQ_MINUS_ATTEMPTS:
            cls.add_to_blacklist(user_id=user_id, reason=f"Превышено количество стоп-вопросов ({attempts})")


logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger("AcademicBot")


class BotManager:
    def __init__(self):
        self.main_keyboard = self._create_main_keyboard()
        self.api_url = HANDLER_API_URL
        self.session = aiohttp.ClientSession()
        self.secret_token = SECRET_TOKEN
        self.logger = BotLogger()
        self.blacklist = BlacklistManager()


    @staticmethod
    def _create_main_keyboard() -> ReplyKeyboardMarkup:
        return ReplyKeyboardMarkup(
            [
                [KeyboardButton("🔄 Новый диалог")],
            ],
            resize_keyboard=True,
            one_time_keyboard=False,
        )

    async def _call_api(self, endpoint: str, payload: dict) -> str:
        try:
            async with self.session.post(f"{self.api_url}/{endpoint}", json=payload, timeout=120) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("response", data.get("status", ""))
                return "Попробуйте обновить диалог и переформулировать запрос"
        except Exception as e:
            logger.error(f"API error: {e}")
            return "Сервис временно недоступен"

    async def start(self, update: Update, context: CallbackContext) -> None:
        user_id = update.effective_user.id
        self.logger.log_action(user_id, "start_command")

        if self.blacklist.check_blacklist(user_id):
            await update.message.reply_text(
                "❌ Ваш аккаунт временно ограничен. Пожалуйста, обратитесь в приемную комиссию для выяснения обстоятельств.",
                reply_markup=self.main_keyboard,
            )
            return

        if self.secret_token and not context.user_data.get("authenticated"):
            await update.message.reply_text("🔒 Для доступа к боту введите секретный токен:")
            return

        response = await self._call_api("new_dialog", {"user_id": str(user_id)})
        welcome = self._get_welcome_message()
        await update.message.reply_text(welcome, reply_markup=self.main_keyboard)

    def _get_welcome_message(self) -> str:
        return (
            "Привет! 👋 Я — нейропомощник по вселенной Marvel. "
            "Я могу помочь разобраться в персонажах, командах, способностях, "
            "событиях и других аспектах мира Marvel.\n\n"
            "Ты можешь спросить, кто такой конкретный персонаж, "
            "попросить список героев или злодеев, сравнить персонажей "
            "или узнать факты из комиксной вселенной.\n\n"
            "Я опираюсь на структурированные данные и материалы Марвелпедии, "
            "но, как и любой ИИ, могу иногда ошибаться. "
            "Если вопрос сложный или спорный — я честно скажу, "
            "когда информации недостаточно."
        )


    async def _process_user_input(self, update: Update, context: CallbackContext, user_id: int, text: str) -> None:
        if self.blacklist.check_blacklist(user_id):
            await update.message.reply_text(
                "❌ Ваш аккаунт временно ограничен. Пожалуйста, обратитесь в приемную комиссию для выяснения обстоятельств.",
                reply_markup=self.main_keyboard,
            )
            return

        self.logger.log_action(user_id, "user_message", text)

        processing_msg = await update.message.reply_text("⏳ Обрабатываю ваш запрос...")

        try:
            last_cluster = context.user_data.pop("last_cluster", "")
            full_query = f"{last_cluster} {text}" if last_cluster else text

            async with self.session.post(
                f"{self.api_url}/handle_prompt", json={"user_id": str(user_id), "prompt": full_query}, timeout=120
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    response_text = data["response"]
                    is_faq_confident = data.get("is_faq_confident", False)
                    is_faq_minus_confident = data.get("is_faq_minus_confident", False)

                    if is_faq_minus_confident:
                        is_faq_confident = False
                        self.blacklist.log_faq_minus_attempt(user_id=user_id, question=full_query, answer=response_text)
                        self.logger.log_action(user_id, "blacklist", response_text)

                        if self.blacklist.check_blacklist(user_id):
                            self.logger.log_action(user_id, "blacklist", "❌ Ваш аккаунт временно ограничен. Пожалуйста, обратитесь в приемную комиссию.")
                            await processing_msg.edit_text(
                                "❌ Ваш аккаунт временно ограничен. Пожалуйста, обратитесь в приемную комиссию.",
                                reply_markup=self.main_keyboard,
                            )
                            return

                    if is_faq_confident:
                        await processing_msg.delete()

                        faq_question = data.get("faq_question", full_query)

                        confirm_keyboard = ReplyKeyboardMarkup(
                            [[KeyboardButton("Да"), KeyboardButton("Нет")],
                            [KeyboardButton("🔄 Новый диалог")]],
                            resize_keyboard=True,
                            one_time_keyboard=True,
                        )

                        context.user_data["faq_answer"] = response_text         
                        context.user_data["faq_question"] = faq_question  
                        context.user_data["original_query"] = full_query

                        faq_msg = await update.message.reply_text(
                            f"Вы это имели в виду?\n\n❓ {faq_question}\n\nЕсли не отображаются кнопки да/нет нажмите на ⌘", 
                            reply_markup=confirm_keyboard
                        )
                        context.user_data["faq_message_id"] = faq_msg.message_id
                        self.logger.log_action(user_id, "faq_answer", response_text)

                    else:
                        await processing_msg.edit_text(response_text)
                else:
                    await processing_msg.edit_text("Попробуйте обновить диалог и переформулировать запрос")
        except Exception as e:
            logger.error(f"Error: {e}")
            await processing_msg.edit_text("Попробуйте обновить диалог и переформулировать запрос")

    async def handle_message(self, update: Update, context: CallbackContext) -> None:
        user_input = update.message.text
        user_id = update.effective_user.id
        logger.info(f"Message from {user_id}: {user_input}")

        if self.blacklist.check_blacklist(user_id):
            await update.message.reply_text(
                "❌ Ваш аккаунт временно ограничен. Пожалуйста, обратитесь в приемную комиссию.",
                reply_markup=self.main_keyboard,
            )
            return
        if self.secret_token and not context.user_data.get("authenticated"):
            if user_input == self.secret_token:
                context.user_data["authenticated"] = True
                await update.message.reply_text(
                    "✅ Авторизация успешна! Теперь вы можете использовать бот.", reply_markup=self.main_keyboard
                )
                return
            else:
                await update.message.reply_text("❌ Неверный токен. Пожалуйста, введите секретный токен.")
                return



        if user_input in ["Да", "Нет"] and "faq_answer" in context.user_data:
            try:
                await update.message.delete()
                if "faq_message_id" in context.user_data:
                    await context.bot.delete_message(
                        chat_id=update.effective_chat.id, message_id=context.user_data["faq_message_id"]
                    )
            except Exception as e:
                logger.warning(f"Could not delete messages: {e}")

            if user_input == "Да":
                await update.message.reply_text(context.user_data["faq_answer"], reply_markup=self.main_keyboard)
            else:
                processing_msg = await update.message.reply_text("⏳ Уточняю информацию...")
                try:
                    async with self.session.post(
                        f"{self.api_url}/handle_prompt",
                        json={"user_id": str(user_id), "prompt": context.user_data["original_query"], "skip_faq": True},
                        timeout=120,
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            try:
                                await processing_msg.delete()
                                processing_msg = None
                            except Exception as e:
                                logger.warning(f"Could not delete processing message: {e}")
                                
                            await update.message.reply_text(data["response"], reply_markup=self.main_keyboard)
                            self.logger.log_action(user_id, "response", data["response"])

                        else:
                            if processing_msg:
                                try:
                                    await processing_msg.delete()
                                except Exception as e:
                                    logger.warning(f"Could not delete message: {e}")
                                    
                                await update.message.reply_text("Попробуйте обновить диалог и переформулировать запрос",reply_markup=self.main_keyboard)
                except Exception as e:
                    logger.error(f"Error: {e}")
                    self.logger.log_action(user_id, "Error", e)
                    if processing_msg:
                        try:
                            await processing_msg.edit_text("Попробуйте обновить диалог и переформулировать запрос")
                        except Exception as ex:
                            logger.warning(f"Could not edit message: {ex}")
                            await update.message.reply_text("Попробуйте обновить диалог и переформулировать запрос", reply_markup=self.main_keyboard)

            for key in ["faq_answer", "original_query", "faq_message_id"]:
                if key in context.user_data:
                    del context.user_data[key]
            return

        if user_input != "🔄 Новый диалог":
            context.user_data["original_query"] = user_input


        if user_input == "🔄 Новый диалог":
              await self._handle_new_dialog(update, context, user_id)
        else:
            await self._process_user_input(update, context, user_id, user_input)

    async def _handle_new_dialog(self, update: Update, context: CallbackContext, user_id: int) -> None:
        context.user_data.clear()
        response = await self._call_api("new_dialog", {"user_id": str(user_id)})
        welcome = self._get_welcome_message()
        await update.message.reply_text(welcome, reply_markup=self.main_keyboard)



async def run_bot():
    logger.info("Starting bot...")
    bot_manager = BotManager()

    application = Application.builder().token(TOKEN).build()
    application.add_handler(CommandHandler("start", bot_manager.start))
    application.add_handler(CommandHandler("restart", bot_manager.start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot_manager.handle_message))


    await application.initialize()
    await application.start()

    logger.info("Bot started, polling...")
    await application.updater.start_polling()

    await asyncio.Event().wait()


if __name__ == "__main__":
    try:
        asyncio.run(run_bot())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")