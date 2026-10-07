"""Child process entrypoint; all heavy model imports stay in factories."""

import ctypes
import importlib
import mmap
import os
import signal
import socket
import sys
import traceback
from .transport import send, receive


def main():
    socket_fd, buffer_fd, capacity, factory, parent = sys.argv[1:]
    ctypes.CDLL(None).prctl(1, signal.SIGTERM)
    if os.getppid() != int(parent):
        return
    sock = socket.socket(fileno=int(socket_fd))
    buf = mmap.mmap(int(buffer_fd), int(capacity))
    handler = None
    try:
        config = receive(sock)
        module, name = factory.split(":")
        handler = getattr(importlib.import_module(module), name)(config, buf)
        send(sock, {"ready": True, "metadata": getattr(handler, "metadata", {})})
        while True:
            try:
                message = receive(sock)
            except EOFError:
                break
            try:
                result = handler.handle(message)
                send(sock, {"seq": message["seq"], "result": result})
            except BaseException:
                send(sock, {"seq": message["seq"], "error": traceback.format_exc()})
                break
    except BaseException:
        try:
            send(sock, {"ready": False, "error": traceback.format_exc()})
        except (OSError, EOFError):
            pass
        raise
    finally:
        if handler is not None and hasattr(handler, "close"):
            handler.close()
        handler = None
        buf.close()
        sock.close()


if __name__ == "__main__":
    main()
