import base64
import hashlib
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
import os
import pandas as pd
from .config import Config

class MessageIDEncryptor:
    def __init__(self, secret=None):
        if secret is None: secret = Config().get("secret_encryption_key")
        self.key = hashlib.sha256(secret.encode()).digest()
        """Initialize with a secret key (must be 16, 24, or 32 bytes long)."""
        self.key = hashlib.sha256(secret.encode()).digest()  # Ensure a 32-byte key

    def encrypt(self, sender_id: str, timestamp: float) -> str:
        """Encrypt (secret + sender_id + timestamp) to generate a message ID."""
        data = f"{sender_id}|{timestamp}".encode()
        cipher = AES.new(self.key, AES.MODE_CBC)
        encrypted_data = cipher.encrypt(pad(data, AES.block_size))
        return base64.b64encode(cipher.iv + encrypted_data).decode()

    def decrypt(self, encrypted_message_id: str) -> str:
        """Decrypt the message ID back to sender_id and timestamp."""
        encrypted_data = base64.b64decode(encrypted_message_id)
        iv = encrypted_data[:AES.block_size]
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        decrypted_data = unpad(cipher.decrypt(encrypted_data[AES.block_size:]), AES.block_size)
        data = decrypted_data.decode().split("|")
        return data[0], data[1]