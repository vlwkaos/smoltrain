"""Unix socket classifier daemon."""
import socket
import threading

from . import config as cfg_mod
from .classify import OnnxClassifier


def run(cfg):
    sock_path = cfg_mod.socket_path(cfg.name)
    classifier = OnnxClassifier(cfg)

    if sock_path.exists():
        sock_path.unlink()

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(sock_path))
    server.listen(32)
    print(f"serving on {sock_path}")

    try:
        while True:
            conn, _ = server.accept()
            threading.Thread(
                target=_handle, args=(conn, classifier), daemon=True
            ).start()
    except KeyboardInterrupt:
        print("\nshutting down")
    finally:
        server.close()
        if sock_path.exists():
            sock_path.unlink()


def _handle(conn, classifier):
    try:
        with conn.makefile("rwb") as f:
            for line in f:
                text = line.decode().strip()
                if text:
                    result = classifier.classify(text)
                    f.write((result + "\n").encode())
                    f.flush()
    except Exception:
        pass
