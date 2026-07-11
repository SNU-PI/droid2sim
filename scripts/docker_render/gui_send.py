"""Send Python code to a running Isaac Sim (jupyter code-editor socket, port 8227)
so the streaming GUI can be driven live. Usage inside the container:

    /isaac-sim/kit/python/bin/python3 gui_send.py <code-file>
    echo 'print("hi")' | /isaac-sim/kit/python/bin/python3 gui_send.py -
"""
import socket, sys, time

TOKEN = "/isaac-sim/exts/isaacsim.code_editor.jupyter/data/launchers/token.txt"
PORT = 8227

code = sys.stdin.read() if sys.argv[1] == "-" else open(sys.argv[1]).read()
tok = open(TOKEN).read().strip()
s = socket.create_connection(("127.0.0.1", PORT), timeout=15)
s.sendall((tok + code).encode())
time.sleep(1.0)
s.settimeout(8); buf = b""
try:
    while True:
        d = s.recv(8192)
        if not d: break
        buf += d
except Exception:
    pass
print(buf.decode(errors="replace"))
s.close()
