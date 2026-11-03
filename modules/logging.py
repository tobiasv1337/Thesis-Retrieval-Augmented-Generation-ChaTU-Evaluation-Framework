import logging
import json
from datetime import datetime
import os
from pathlib import Path
import pandas as pd
from .encryptor import MessageIDEncryptor
from .config import Config
import re, ast


class _PerMessageFileHandler(logging.FileHandler):
    """Keep conversation log files open only while writing a record."""

    def emit(self, record):
        try:
            super().emit(record)
        finally:
            if self.stream is not None:
                try:
                    self.stream.close()
                finally:
                    self.stream = None


def _append_to_csv(file_path, row, expected_columns):
    """Handles appending to a CSV file while ensuring column consistency."""
    if os.path.exists(file_path):
        # if the header is correct, just append to the csv
        with open(file_path, 'r') as f:
            existing_header = f.readline().strip().split(',')
        if existing_header == expected_columns:
            row[expected_columns].to_csv(file_path, mode='a', header=not os.path.exists(file_path), index=False)
            return

        # otherwise load file
        existing_df = pd.read_csv(
            file_path,
            engine="python",
            on_bad_lines="warn",
        )
        # Ensure all expected columns exist
        for col in expected_columns:
            if col not in existing_df.columns:
                existing_df[col] = None  # Add missing columns
        updated_df = pd.concat([existing_df, row], ignore_index=True)
    else:
        updated_df = row

    updated_df = updated_df[expected_columns] # Maintain correct column order
    updated_df.to_csv(file_path, index=False)


class Logger:

    def __init__(self, msg_args=None, bot_name="root"):
        """Creates a logger with a unique name (only once per server)."""

        self.config = Config()

        self.bot_name = bot_name
        self.msg_args = msg_args
        self.user_id = self.msg_args.tracker.sender_id if msg_args is not None else 0
        self.debug_mode, self.experiment_mode, self.participant_id = self.check_debug_experiment_mode()
        self.log_parent_dir = self.get_log_parent_dir()
        self.log_dir = self.get_log_dir()
        self.dialog_log_file = self.get_log_file()
        self.system_log_file = self.get_log_file(log_type="system")
        self.sys_logging = self.create_system_logger()
        self.update_system_log_file()

        self.expected_columns = [
            "timestamp", "device", "bot", "language", "feedback", "feedback_type", "text", "buttons",
            "intent", "confidence", "bot_intent", "bot_confidence", "debug"
        ]

    def get_log_file(self, timestamp=None, log_type=None):
        """type (str) should be 'system' or 'dialog'"""
        log_dir = self.log_dir if timestamp is None else self.get_log_dir(timestamp)
        file_type = "csv" if log_type != "system" else "log"
        return os.path.join(log_dir, f"{self.user_id}.{file_type}")

    def get_log_dir(self, timestamp=None):

        date = datetime.now() if timestamp is None else datetime.fromtimestamp(timestamp)

        year, month, day = date.strftime("%Y-%m-%d").split("-")
        log_dir = os.path.join(self.log_parent_dir, year, month, day)
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        return log_dir

    def get_log_parent_dir(self):
        """Sets the logging path for the bot. Necessary for experiments"""
        self.log_parent_dir = self.config.get("logging_dir") if not self.experiment_mode else os.path.join(
            self.config.get("experiment_logging_dir"), self.config.get("experiment_password"))
        if self.experiment_mode:
            participant_id_folder = self.participant_id if self.participant_id is not None and len(
                self.participant_id) > 0 else self.config.get("unknown_participant_ID_folder")
            self.log_parent_dir = os.path.join(self.log_parent_dir, participant_id_folder)
        os.makedirs(self.log_parent_dir, exist_ok=True)
        return self.log_parent_dir

    def check_debug_experiment_mode(self):
        if self.msg_args is None: return False, False, None
        # debug passwords
        password, participant_id = self.msg_args.tracker.get_slot('debug'), None
        if type(password) != str: password = "false"
        if self.config.get("separator")+self.config.get("participant_ID_prefix") in password:
                password, participant_id = password.split(self.config.get("separator")+self.config.get("participant_ID_prefix"))
        debug_mode = True if password in [self.config.get("debug_password"), self.config.get("experiment_password")] and self.config.get("debug") else False
        experiment_mode = False if password != self.config.get("experiment_password") else True
        return debug_mode, experiment_mode, participant_id

    def create_system_logger(self):
        """Returns a separate logger for each user, but with the bot name as the printed name."""
        self.sys_logging = logging.getLogger(f"{self.bot_name}_User_{self.user_id}")
        self.sys_logging.setLevel(logging.DEBUG)
        self.sys_logging.propagate = False

        if not self.sys_logging.handlers:
            # Terminal formatter with color
            terminal_formatter = TerminalFormatter(
                f'\033[2m%(asctime)s\033[0m %(levelname)s \t \033[34m{self.bot_name}\033[0m - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(terminal_formatter)
            self.sys_logging.addHandler(stream_handler)

        return self.sys_logging

    def update_system_log_file(self):
        """Dynamically updates the log file without recreating the logger."""

        # Remove existing file handlers
        for handler in self.sys_logging.handlers[:]:
            if isinstance(handler, logging.FileHandler):
                self.sys_logging.removeHandler(handler)
                handler.close()

        # Create new file handler and attach it
        file_handler = _PerMessageFileHandler(self.system_log_file, delay=True)
        file_handler.setFormatter(FileFormatter(
            f'%(asctime)s %(levelname)s \t {self.bot_name.upper()} - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        self.sys_logging.addHandler(file_handler)

    def debug(self, message):
        self.sys_logging.debug(message)

    def info(self, message):
        self.sys_logging.info(message)

    def warning(self, message):
        self.sys_logging.warning(message)

    def error(self, message):
        self.sys_logging.error(message)

    def critical(self, message):
        self.sys_logging.critical(message)

    def update_feedback(self, message_id: str, feedback: str, feedback_type: str):
        """
        Args:
            message_id (str): The encrypted message ID.
            feedback (str): The feedback text to insert.
            feedback_type (str): How was the feedback received ("written" or "thumbs").
        """
        new_row = self.update_message_values(message_id, {"feedback": feedback, "feedback_type": feedback_type})

        # append message to rated_logs file
        self.append_to_global_log_file(f"logs_rated_{feedback}", new_row)

    def update_message_values(self, message_id: str, values: dict):
        """
        Decrypts message_id, finds the correct log file, and updates the column.

        Args:
            message_id (str): The encrypted message ID.
            values (str): The values to insert.
        Returns:
            row: the changed df row
        """

        try:
            sender_id, timestamp = MessageIDEncryptor().decrypt(message_id)
            timestamp = float(timestamp)
            if self.user_id != sender_id:
                raise ValueError("Sender id from message_id does not match sender.")
        except Exception as e:
            logging.warning(f"Decryption failed: {e}")
            return None

        file_path = self.get_log_file(timestamp)
        if not os.path.exists(file_path):
            logging.info(f"Log file not found: {file_path}")
            return None

        try:
            df = pd.read_csv(file_path)

            if "timestamp" not in df.columns or df[df["timestamp"] == timestamp].empty:
                logging.warning("No matching entry found for the timestamp.")
                return None

            for value in values:
                if value not in df.columns: df[value] = None
                df[value] = df[value].astype(type(values[value]))  # Ensure correct dtype
                df.loc[df["timestamp"] == timestamp, value] = values[value]
            df.to_csv(file_path, index=False)
            logging.info(f"Values successfully updated in {file_path} {timestamp}")

            return pd.DataFrame(df.loc[df["timestamp"] == timestamp])
        except Exception as e:
            logging.warning(f"Error processing {file_path}: {e}")
            return None

    def msg_log(self, text, bot=None, ts=None, buttons=None):
        if self.msg_args is None: return None
        if bot is None: bot = self.msg_args.bot
        if buttons is not None and len(buttons) == 0: buttons = None
        if ts is None: ts = datetime.now().timestamp()

        # Create log row
        log_row = pd.DataFrame(
            # TODO: as json
            [[ts, "", bot, self.msg_args.language, "", "", text.replace("\r", "\n").replace("\n", "\\n"), buttons,
              self.msg_args.intent, self.msg_args.confidence, self.msg_args.forwarded_intent,
              self.msg_args.forwarded_confidence, self.debug_mode]],
            columns=self.expected_columns
        )

        # Append to user-specific log
        _append_to_csv(self.dialog_log_file, log_row, self.expected_columns)

        # Append to global logs
        self.append_to_global_log_file("logs", log_row)

        return self.get_message_id(ts)

    def append_to_global_log_file(self, file_name, row):

        date = datetime.fromtimestamp(row["timestamp"].to_list()[0]).strftime("%Y-%m-%d")

        logs_file_path = os.path.join(self.log_parent_dir, f"{file_name}.csv")
        expected_columns = ["date", "id"] + self.expected_columns #list(row.columns)
        row["date"] = date
        row["id"] = self.user_id
        # ensure header consistency without always loading full file
        _append_to_csv(logs_file_path, row, expected_columns)

    def get_message_id(self, ts):
        logging.info(f"Encrypt message: {self.user_id} - {ts}")
        return MessageIDEncryptor().encrypt(self.user_id, ts)

    def get_latest_message_ids(self) -> list:
        """
        Retrieves the latest message_id.
        Returns:
            list: A list of the latest message_id
        """
        latest_file = None

        # Walk through the year/month/day structure to find the latest log file
        for year in sorted(os.listdir(self.log_parent_dir), reverse=True):
            year_path = os.path.join(self.log_parent_dir, year)
            if not os.path.isdir(year_path): continue

            for month in sorted(os.listdir(year_path), reverse=True):
                month_path = os.path.join(year_path, month)
                if not os.path.isdir(month_path): continue

                for day in sorted(os.listdir(month_path), reverse=True):
                    day_path = os.path.join(month_path, day)
                    if not os.path.isdir(day_path): continue

                    file_path = os.path.join(day_path, f"{self.user_id}.csv")
                    if os.path.exists(file_path):
                        latest_file = file_path  # Found the latest sender's log file
                        break  # Stop searching deeper

                if latest_file: break
            if latest_file: break
        if not latest_file: return []  # No log found for this sender

        # Read the CSV and get the latest message_id
        try:
            df = pd.read_csv(latest_file, index_col=False)[:-1]
            # Find the index of the last occurrence of "bot" or "user"
            last_bot_user_index = df[df["bot"] == "user"].index.max()

            if last_bot_user_index is not None and not pd.isna(last_bot_user_index):
                df = df.iloc[last_bot_user_index + 1:]  # Get all rows below the last bot/user row

            if not df.empty:
                message_ids = [MessageIDEncryptor().encrypt(self.user_id, timestamp) for timestamp in df["timestamp"]]
                return message_ids
            else:
                return []
        except Exception as e:
            logging.warning(f"Error reading {latest_file}: {e}")

        return []  # Return None if no valid message_id is found

    def retrieve_feedback_logs(self, feedback_rating=None, feedback_type=None, regenerate=False):
        condition = {
            "feedback": feedback_rating,
            "feedback_type": feedback_type
        }
        file_name = "logs_rated"
        if feedback_rating is not None: file_name += "_" + feedback_rating
        if feedback_type is not None: file_name += "_" + feedback_type
        return self.retrieve_all_logs(file_name, [condition], regenerate=regenerate)

    def retrieve_all_logs(self, file_name="logs", conditions=None, regenerate=False):
        """
        conditions is list full of dicts
        the list is representative of OR conditions while each dict represents an and condition
        if value in dict is list it represents another OR
        None means all values that are not empty, an empty string means all values that are empty
        """
        file_path = os.path.join(self.log_parent_dir, f"{file_name}.csv")
        if os.path.exists(file_path) and not regenerate:
            return pd.read_csv(file_path)
        logs = []
        for year in os.listdir(os.path.join(self.log_parent_dir, "logging")):
            if year.startswith("."): continue
            for month in os.listdir(os.path.join(self.log_parent_dir, "logging", year)):
                if month.startswith("."): continue
                for day in os.listdir(os.path.join(self.log_parent_dir, "logging", year, month)):
                    if day.startswith("."): continue
                    for user_id in os.listdir(os.path.join(self.log_parent_dir, "logging", year, month, day)):
                        if not user_id.endswith(".csv"): continue
                        try:
                            user_logs = pd.read_csv(os.path.join(self.log_parent_dir, "logging", year, month, day, user_id))
                            if type(conditions) is list and type(conditions[0]) is dict:
                                filtered_logs = pd.DataFrame()
                                for condition in conditions:  # each condition is a dict (AND group)
                                    temp_logs = user_logs.copy()
                                    for key, value in condition.items():
                                        if isinstance(value, list):  # OR within a field
                                            temp_logs = temp_logs[temp_logs[key].isin(value)]
                                        elif value is None:  # not empty
                                            temp_logs = temp_logs[temp_logs[key].notna() & (user_logs[key] != "")]
                                        elif value == "":  # empty string
                                            temp_logs = temp_logs[user_logs[key] == ""]
                                        else:
                                            temp_logs = temp_logs[temp_logs[key] == value]
                                    filtered_logs = pd.concat([filtered_logs, temp_logs])
                                user_logs = filtered_logs
                            logs.append(user_logs)
                        except Exception as e:
                            self.warning(f'retrieve_logs {os.path.join(self.log_parent_dir, "logging", year, month, day, user_id)}: {e}')
                            pass
        logs_overview = pd.concat(logs).sort_values(by=["timestamp"])
        logs_overview.to_csv(os.path.join(self.log_parent_dir, f"{file_name}.csv"))
        return logs_overview

    def retrieve_llm_logs(self, file_name="llm_logs", regenerate=False):
        file_path = os.path.join(self.log_parent_dir, f"{file_name}.csv")
        if os.path.exists(file_path) and not regenerate:
            return pd.read_csv(file_path)
        logs = []
        keep = ["user_id", "timestamp", "bot", "turn_number", "language", "text", "button_click", "intent", "bot_intent"]
        base = os.path.join(self.log_parent_dir)
        for year in os.listdir(base):
            if year.startswith(".") or year.endswith(".csv") or year.endswith(".png") or year.endswith(".py"): continue
            for month in os.listdir(os.path.join(base, year)):
                if month.startswith("."): continue
                for day in os.listdir(os.path.join(base, year, month)):
                    if day.startswith("."): continue
                    for user_id in os.listdir(os.path.join(base, year, month, day)):
                        if not user_id.endswith(".csv"): continue
                        p = os.path.join(base, year, month, day, user_id)
                        try:
                            df = pd.read_csv(p)
                            if "text" not in df or "bot" not in df: continue
                            df["button_click"] = False
                            df["user_id"] = user_id.replace(".csv","")
                            df["turn_type"] = 0

                            df = df.sort_values("timestamp").reset_index(drop=True)

                            # parse buttons -> list or NA, then forward-fill
                            df["buttons"] = df["buttons"].apply(
                                lambda x: ast.literal_eval(x) if isinstance(x, str) and x.strip().startswith(
                                    "[") else pd.NA
                            )
                            df["buttons"] = df["buttons"].ffill()

                            if "intent" not in df.columns: df["intent"] = ""
                            if "bot" not in df.columns: df["bot"] = ""
                            df = df[(df["intent"] == "sekretariat") & ((df["bot"] == "user") | (df["text"].astype(str).str.contains("> ") & (~df["text"].astype(str).str.startswith("Preview:") & ~df["text"].astype(str).str.startswith("Vorschau:"))))]

                            df["buttons"] = df["buttons"].apply(lambda x: x if isinstance(x, list) else [])
                            # resolve button clicks (handles CSV-escaped ""button"":"")
                            total_rows = int(len(df))
                            user_rows = 0
                            user_rows_with_click_pattern = 0
                            user_rows_with_buttons = 0
                            payload_regex_match = 0
                            payload_regex_miss = 0
                            payload_value_match = 0
                            payload_value_miss = 0
                            def _fallback_title_from_slash(value):
                                if not isinstance(value, str) or not value.startswith("/"):
                                    return ""
                                core = value[1:]
                                if "{" in core:
                                    core = core.split("{", 1)[0]
                                result = core.replace("-", " ").strip()
                                return result
                            def _normalize_title(value):
                                if isinstance(value, (list, tuple)):
                                    return value[0] if len(value) > 0 else ""
                                if isinstance(value, str) and value.strip().startswith("[") and value.strip().endswith("]"):
                                    try:
                                        parsed = ast.literal_eval(value)
                                        if isinstance(parsed, (list, tuple)) and len(parsed) > 0:
                                            return parsed[0]
                                    except Exception:
                                        return value
                                return value
                            for i, r in df.iterrows():
                                if r["bot"] != "user": continue
                                user_rows += 1
                                text_value = str(r["text"])
                                is_slash_text = text_value.startswith("/")
                                rx = r'/sekretariat\{\s*(?:""|")button(?:""|")\s*:\s*(?:""|")(.+?)(?:""|")\s*\}'
                                m = re.search(rx, text_value)
                                if not m:
                                    if is_slash_text:
                                        df.at[i, "button_click"] = True
                                        fallback_title = _fallback_title_from_slash(text_value)
                                        if fallback_title:
                                            df.at[i, "text"] = fallback_title
                                    continue
                                user_rows_with_click_pattern += 1
                                clicked_inner = m.group(1)
                                if isinstance(r["buttons"], list) and len(r["buttons"]) > 0:
                                    user_rows_with_buttons += 1
                                df.at[i, "button_click"] = True
                                matched = False
                                matched_title = None
                                matched_payload = None
                                for b in (r["buttons"] or []):
                                    payload = str(b.get("payload", ""))
                                    m2 = re.search(rx, payload)
                                    if m2:
                                        payload_regex_match += 1
                                        if m2.group(1) == clicked_inner:
                                            payload_value_match += 1
                                            matched = True
                                            original_title = b.get("title")
                                            matched_title = _normalize_title(original_title)
                                            matched_payload = payload
                                            df.at[i, "text"] = matched_title
                                            df.at[i, "button_click"] = True
                                            break
                                        else:
                                            payload_value_miss += 1
                                    else:
                                        payload_regex_miss += 1
                                if (not matched) and isinstance(clicked_inner, str) and clicked_inner.strip():
                                    if clicked_inner.startswith("/"):
                                        fallback_title = _fallback_title_from_slash(clicked_inner)
                                        if fallback_title:
                                            df.at[i, "text"] = fallback_title
                                    else:
                                        df.at[i, "text"] = clicked_inner.strip()

                            # clean bot quotes
                            bot_mask = df["bot"] != "user"
                            s = df.loc[bot_mask, "text"].astype(str).str.replace(r"\\n", "\n", regex=True)
                            df.loc[bot_mask, "text"] = (
                                s.astype(str).str.split("\n").apply(lambda xs: "\n".join(
                                    l.lstrip()[2:] for l in xs if l.lstrip().startswith("> ")
                                ))
                            )
                            df = df[~(bot_mask & (df["text"].str.len() == 0))]

                            # keep only the last user msg before a response (collapse consecutive user rows)
                            df = df.sort_values("timestamp").reset_index(drop=True)
                            is_user = df["bot"] == "user"
                            next_is_user = is_user.shift(-1, fill_value=False)

                            # drop user rows that are followed by another user row
                            df = df[~(is_user & next_is_user)].copy()
                            is_user = df["bot"] == "user"
                            user_turn = is_user.cumsum()
                            bot_turn = (~is_user).cumsum()
                            df["turn_number"] = user_turn.where(is_user, bot_turn)
                            df["num_usr_turns"] = (df["bot"] == "user").sum()
                            df["num_llm_turns"] = (df["bot"] != "user").sum()

                            # final columns
                            for c in keep:
                                if c not in df: df[c] = pd.NA
                            logs.append(df[keep + ["num_usr_turns", "num_llm_turns"]])

                        except Exception as e:
                            self.warning(f'retrieve_logs {os.path.join(self.log_parent_dir, "logging", year, month, day, user_id)}: {e}')
                            pass
        logs_overview = pd.concat(logs, ignore_index=True)
        # force consistent dtype
        logs_overview["timestamp"] = pd.to_numeric(logs_overview["timestamp"], errors="coerce")
        # drop rows without timestamp (optional)
        logs_overview = logs_overview.dropna(subset=["timestamp"]).sort_values(by=["timestamp"])
        logs_overview = logs_overview

        logs_overview[logs_overview["bot"] == "user"][keep].to_csv(
            os.path.join(self.log_parent_dir, "all_user_logs.csv"), index=False)
        logs_overview[(logs_overview["num_usr_turns"] > 1) & (logs_overview["num_llm_turns"] > 0)][keep].to_csv(
            os.path.join(self.log_parent_dir, f"{file_name}_multi_turn.csv"), index=False)
        logs_overview[(logs_overview["num_usr_turns"] == 1) & (logs_overview["num_llm_turns"] > 0)][keep].to_csv(
            os.path.join(self.log_parent_dir, f"{file_name}_single_turn.csv"), index=False)
        logs_overview[(logs_overview["num_llm_turns"] < 1)][keep].to_csv(
            os.path.join(self.log_parent_dir, f"non_{file_name}_user.csv"), index=False)
        logs_overview[(logs_overview["num_llm_turns"] > 0)][keep].to_csv(
            os.path.join(self.log_parent_dir, f"{file_name}.csv"), index=False)

        return logs_overview

def retrieve_all_logs(log_path, regenerate=False, feedback=False):
    logger = Logger()
    logger.log_parent_dir = log_path
    logger.retrieve_all_logs(regenerate=regenerate)
    if feedback:
        logger.retrieve_feedback_logs("good", regenerate=regenerate)
        logger.retrieve_feedback_logs("bad", regenerate=regenerate)
    logger.retrieve_llm_logs(regenerate=regenerate)

class TerminalFormatter(logging.Formatter):
    COLOR_MAP = {
        logging.INFO:     "\033[1;14m",     # Bright White
        logging.DEBUG:    "\033[1;33m",     # Yellow
        logging.WARNING:  "\033[1;38;5;203m", # Light Red / Orange (256 color)
        logging.ERROR:    "\033[1;31m",     # Bright Red
        logging.CRITICAL: "\033[1;41m",     # Red Background
    }

    RESET = "\033[0m"

    def format(self, record):
        original_levelname = record.levelname
        color = self.COLOR_MAP.get(record.levelno, "")
        record.levelname = f"{color}{original_levelname}{self.RESET}"

        msg = super().format(record)

        record.levelname = original_levelname  # Restore clean

        if record.message:
            lines = msg.splitlines()
            if len(lines) > 1:
                first, rest = lines[0], lines[1:]
                rest = [f"\t\t\t{line}" for line in rest]
                msg = "\n".join([first] + rest)

        return msg

class FileFormatter(logging.Formatter):
    def format(self, record):
        msg = super().format(record)
        if record.message:
            lines = msg.splitlines()
            if len(lines) > 1:
                # Indent every line after the first
                first, rest = lines[0], lines[1:]
                rest = [f"\t\t\t{line}" for line in rest]
                msg = "\n".join([first] + rest)
        return msg
