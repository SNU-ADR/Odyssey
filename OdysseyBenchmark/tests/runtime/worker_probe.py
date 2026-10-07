import os
import time


class Probe:
    def __init__(self, config, buffer):
        self.buffer = buffer

    def handle(self, m):
        if m["op"] == "error":
            raise ValueError("deliberate probe error")
        if m["op"] == "exit":
            os._exit(3)
        if m["op"] == "sleep":
            time.sleep(4)
        n = m.get("payload_size", 0)
        self.buffer[:n] = self.buffer[:n][::-1]
        return {"size": n}
