import config
import asyncio
import logging
from logging import info, debug, error, warn, exception
import sys
from datetime import timedelta
from urllib.parse import parse_qs

from quart import Quart, jsonify, globals, request
from quart_rate_limiter import RateLimiter, rate_limit, rate_exempt
import asqlite

from src.types import TilingMode


app = Quart(__name__, instance_relative_config=True)
rate_limiter = RateLimiter(app)


@app.before_serving
async def init_globals():
    async with app.app_context():
        logging.basicConfig(
            stream=sys.stderr,
            level=logging.DEBUG,
            format="%(asctime)s [%(levelname)s] [%(module)s:%(funcName)s %(lineno)s] %(message)s"
        )
        info("Starting up...")
        info(f"Connecting to database at {config.db_path}")
        globals.conn = await asqlite.connect(config.db_path)
        info(f"Connected to database!")


@app.route("/", methods = ["GET"])
@rate_exempt
async def root():
    return jsonify(["macros", "tiles", "filters"])


@app.route("/tiles.json", methods = ["GET"])
@rate_limit(1, timedelta(seconds = 3))
@rate_limit(10, timedelta(minutes = 1))
async def tiles():
    query = request.query_string
    query = {
        key.decode("utf-8", errors = "replace"): [val.decode("utf-8", errors = "replace") for val in vals]
        for key, vals in parse_qs(query, keep_blank_values=True, max_num_fields=8).items()
    }

    command = "SELECT name, active_color_x, active_color_y, inactive_color_x, inactive_color_y, source, sprite, tiling, tags FROM tiles WHERE 1"
    args = []
    if "name" in query:
        command += " AND name == ?"
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
            return 'Invalid query "tiling": not a tiling mode', 400
        command += " AND TILING == ?"
        args.append(tiling + 0)
    command += " ORDER BY version DESC"
    async with globals.conn.cursor() as cur:
        res = await cur.execute(command, *args)
        rows = await res.fetchall()
        if "name" in query and len(rows) == 0:
            return b"", 404
        ret = {}
        for row in rows:
            name, active_color_x, active_color_y, inactive_color_x, inactive_color_y, source, sprite, tiling, tags \
                = row
            ret[name] = {
                "active_color": [active_color_x, active_color_y],
                "inactive_color": [inactive_color_x, inactive_color_y],
                "sprite": [source, sprite],
                "tiling": str(TilingMode(tiling)),
                "tags": tags.split("\t") if tags else []
            }
            if "name" in query:
                return ret[name]
        return ret


@app.route("/macros.json", methods = ["GET"])
@rate_limit(1, timedelta(seconds = 3))
@rate_limit(10, timedelta(minutes = 1))
async def macros():
    query = request.query_string
    query = {
        key.decode("utf-8", errors = "replace"): [val.decode("utf-8", errors = "replace") for val in vals]
        for key, vals in parse_qs(query, keep_blank_values=True, max_num_fields=8).items()
    }

    command = "SELECT * FROM macros WHERE 1"
    args = []

    if "name" in query:
        command += " AND name == ?"
        args.append(query["name"][0])
    if "author" in query:
        command += " AND creator == ?"
        try:
            args.append(int(query["author"][0]))
        except ValueError as e:
            return 'Invalid query "author": must be an integer', 400
    if "data_only" in query:
        command = "SELECT name, value" + command.removeprefix("SELECT *")

    async with globals.conn.cursor() as cur:
        res = await cur.execute(command, *args)
        rows = await res.fetchall()
        if "name" in query and len(rows) == 0:
            return b"", 404
        ret = {}
        for row in rows:
            row = {key: val for key, val in zip(row.keys(), (*row, ))}
            if "name" in query:
                return row
            name = row["name"]
            del row["name"]

            ret[name] = row
        return ret


@app.route("/filters.json", methods = ["GET"])
@rate_limit(1, timedelta(seconds = 3))
@rate_limit(10, timedelta(minutes = 1))
async def filters():
    query = request.query_string
    query = {
        key.decode("utf-8", errors = "replace"): [val.decode("utf-8", errors = "replace") for val in vals]
        for key, vals in parse_qs(query, keep_blank_values=True, max_num_fields=8).items()
    }

    command = "SELECT name, author, absolute, upload_time FROM filters WHERE 1"
    args = []

    if "name" in query:
        command += " AND name == ?"
        args.append(query["name"][0])
    if "author" in query:
        command += " AND author == ?"
        try:
            args.append(int(query["author"][0]))
        except ValueError as e:
            return 'Invalid query "author": must be an integer', 400

    async with globals.conn.cursor() as cur:
        res = await cur.execute(command, *args)
        rows = await res.fetchall()
        if "name" in query and len(rows) == 0:
            return b"", 404
        ret = {}
        for row in rows:
            row = {key: val for key, val in zip(row.keys(), (*row, ))}
            if "name" in query:
                return row
            name = row["name"]
            del row["name"]
            ret[name] = row
        return ret


@app.route("/filters/<string:filter_name>.png", methods = ["GET"])
@rate_limit(1, timedelta(seconds = 3))
@rate_limit(10, timedelta(minutes = 1))
async def filter_blob(filter_name):
    async with globals.conn.cursor() as cur:
        res = await cur.execute("SELECT data, absolute FROM filters WHERE name == ?", filter_name)
        row = await res.fetchone()
        if row is None:
            return b"", 404
        return row[0], 200, {"absolute": row[1], "Content-Type": "image/png"}


@app.route("/<path:_>", methods = ["OPTIONS"])
@rate_exempt
async def cors_stuff():
    return b"", 200, {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET, OPTIONS',
        'Access-Control-Allow-Headers': 'X-Requested-With, Content-Type'
    }
