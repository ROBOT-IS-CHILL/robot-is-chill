import sqlite3
import config
import json
import asyncio
from http.server import BaseHTTPRequestHandler, HTTPServer
import sys
from pathlib import PurePath as Path
import logging
from logging import info, debug, error, warn, exception
from typing import Optional, Callable, Any
from urllib.parse import urlparse, parse_qs
from src.types import TilingMode
import time
import math

conn = None
endpoint_handlers = {}


def handler(func):
    global endpoint_handlers
    def wrapper(*args, **kwargs):
        func(*args, **kwargs)
    endpoint_handlers[func.__name__] = func
    return func


def root_handler(apih):
    global endpoint_handlers
    return {"status": 200, "data": [*endpoint_handlers.keys()]}


@handler
def macros(apih, query):
    global conn
    command = "SELECT * FROM macros WHERE 1"
    args = []
    if "name" in query:
        command += " AND INSTR(name, ?)"
        args.append(query["name"][0])
    if "creator" in query:
        command += " AND creator == ?"
        try:
            args.append(int(query["creator"][0]))
        except ValueError as e:
            return {"status": 400, "data": 'Invalid query "author": must be an integer'}
    if "data_only" in query:
        command = "SELECT name, value" + command.removeprefix("SELECT *")
    cur = conn.cursor()
    res = cur.execute(command, args)
    rows = res.fetchall()
    return {"status": 200, "data": [
        {key: val for key, val in zip(row.keys(), (*row, ))}
        for row in rows
    ]}


@handler
def tiles(apih, query):
    global conn
    command = "SELECT * FROM tiles WHERE 1"
    args = []
    if "name" in query:
        command += " AND INSTR(name, ?)"
        args.append(query["name"][0])
    if "source" in query:
        command += " AND source == ?"
        args.append(query["source"][0])
    if "tag" in query:
        for t in query["tag"]:
            command += " AND INSTR(tag, ?)"
            args.append(t)
    if "tiling" in query:
        tiling = TilingMode.parse(query["tiling"][0])
        if tiling is None:
            return {"status": 400, "data": 'Invalid query "tiling": not a tiling mode'}
        command += " AND TILING == ?"
        args.append(tiling + 0)
    cur = conn.cursor()
    res = cur.execute(command, args)
    rows = res.fetchall()
    return {"status": 200, "data": [
        {key: val for key, val in zip(row.keys(), (*row, ))}
        for row in rows
    ]}


ratelimit_stuff = {}


class APIHandler(BaseHTTPRequestHandler):
    def handle_endpoint(self) -> Optional[Any]:
        global endpoint_handlers, conn
        url = urlparse(self.path)
        path = Path(url.path)
        if len(path.parents) == 0:
            return (root_handler)(self)
        if path.parent != Path("/"):
            return None
        if path.suffix != ".json":
            return None
        query = parse_qs(url.query)
        return endpoint_handlers.get(path.stem, None)(self, query)

    def handle_ratelimits(self):
        global ratelimit_stuff
        debug(ratelimit_stuff.get(self.addr, None))
        if time.time() - ratelimit_stuff.get(self.addr, 0) < 0.2:
            return True
        ratelimit_stuff[self.addr] = time.time()

    def ratelimit_data(self, data_len):
        global ratelimit_stuff
        ratelimit_stuff[self.addr] = time.time() + max(0, math.log(data_len, 1.5))
        debug(ratelimit_stuff[self.addr] - time.time())

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'X-Requested-With, Content-Type')
        self.end_headers()

    def do_GET(self):
        self.addr = self.client_address[0]
        if self.handle_ratelimits():
            info(f"Too many requests from {self.addr}, returning 429")
            self.send_response(429, f"Wait {ratelimit_stuff[self.addr] - time.time():0.1f} seconds")
            self.end_headers()
            return
        info(f"New request from {self.addr}")
        try:
            response = self.handle_endpoint()
        except Exception as e:
            exception(e)
            self.send_response(500, f"{e.__class__.__name__}: {e}")
            self.end_headers()
            return
        if response is None:
            info(f"Path {self.path} invalid, returning 404")
            self.send_response(404)
            self.end_headers()
            return
        data = json.dumps(response["data"]).encode("utf-8")
        self.send_response(response["status"])
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.ratelimit_data(len(data))
        self.wfile.write(data)



def main():
    global conn
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] [%(module)s:%(funcName)s %(lineno)s] %(message)s"
    )
    info("Starting up...")
    info(f"Connecting to database at {config.db_path}")
    conn = sqlite3.connect(config.db_path)
    conn.row_factory = sqlite3.Row

    info(f"Connected!")

    server_address = ('0.0.0.0', 80)
    httpd = HTTPServer(server_address, APIHandler)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
