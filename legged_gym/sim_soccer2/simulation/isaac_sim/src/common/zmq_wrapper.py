# SPDX-FileCopyrightText: Copyright (c) MOS-Brain Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

import zmq
import json
import logging

class ZMQWrapper:
    def __init__(self, context=None, socket_type=zmq.REP, addr="tcp://*:5555"):
        self.context = context or zmq.Context()
        self.socket = self.context.socket(socket_type)
        self.addr = addr
        self.logger = logging.getLogger("ZMQWrapper")

    def bind(self):
        try:
            self.socket.bind(self.addr)
            self.logger.info(f"Bound to {self.addr}")
        except Exception as e:
            self.logger.error(f"Failed to bind to {self.addr}: {e}")
            raise

    def connect(self):
        try:
            self.socket.connect(self.addr)
            self.logger.info(f"Connected to {self.addr}")
        except Exception as e:
            self.logger.error(f"Failed to connect to {self.addr}: {e}")
            raise

    def send_json(self, data):
        try:
            self.socket.send_json(data)
        except Exception as e:
            self.logger.error(f"Failed to send JSON: {e}")
            raise

    def recv_json(self):
        try:
            return self.socket.recv_json()
        except Exception as e:
            self.logger.error(f"Failed to receive JSON: {e}")
            raise

    def close(self):
        self.socket.close()
        self.context.term()
