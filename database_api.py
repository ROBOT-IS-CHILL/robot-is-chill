import io
import config
import asyncio
import logging
from logging import info, debug, error, warn, exception
import sys
from datetime import timedelta
from urllib.parse import parse_qs
from functools import wraps

import multiprocessing
from quart import Quart, jsonify, globals, request
from quart_rate_limiter import RateLimiter, rate_limit, rate_exempt
import asqlite

import numpy as np
from PIL import Image

app = Quart(__name__, instance_relative_config=True)
rate_limiter = RateLimiter(app)

CORS_HEADERS = {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'GET, OPTIONS',
    'Access-Control-Allow-Headers': 'X-Requested-With, Content-Type'
}

res_cache = {}
CACHE_MAX_SIZE = 100

with Image.open("data/palettes/vanilla/default.png") as im:
    PALETTE = np.array(im.convert("RGBA"), dtype=np.uint8)

def dumb_cache(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        key = request.full_path
        if key in res_cache:
            return res_cache[key]
        if len(res_cache) > CACHE_MAX_SIZE:
            res_cache.pop(next(iter(res_cache)))
        response = await func(*args, **kwargs)
        res_cache[key] = response
        return response
    return wrapper

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
        info("Connected to database!")


@app.route("/", methods = ["GET"])
@rate_exempt
async def root():
    return jsonify(["macros", "tiles", "filters"]), 200, CORS_HEADERS

@app.route("/tiles.json", methods = ["GET"])
@rate_limit(1, timedelta(seconds = 3))
@rate_limit(10, timedelta(minutes = 1))
async def tiles():
    query = request.query_string
    query = {
        key.decode("utf-8", errors = "replace"): [val.decode("utf-8", errors = "replace") for val in vals]
        for key, vals in parse_qs(query, keep_blank_values=True, max_num_fields=8).items()
    }

    command = """
        SELECT name, active_color_x, active_color_y,
               inactive_color_x, inactive_color_y,
               source, sprite, tiling, tags
       FROM tiles WHERE 1
    """
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
        command += " AND TILING == ?"
        args.append(query)
    command += " ORDER BY version DESC"
    async with globals.conn.cursor() as cur:
        res = await cur.execute(command, *args)
        rows = await res.fetchall()
        if "name" in query and len(rows) == 0:
            return f"No tile by the name {query['name'][0]} exists in the database", 404, CORS_HEADERS
        ret = {}
        for row in rows:
            name, active_color_x, active_color_y, inactive_color_x, inactive_color_y,\
                source, sprite, tiling, tags = row
            ret[name] = {
                "active_color": [active_color_x, active_color_y],
                "inactive_color": [inactive_color_x, inactive_color_y],
                "sprite": [source, sprite],
                "tiling": tiling,
                "tags": tags.split("\t") if tags else []
            }
            if "name" in query:
                return ret[name], CORS_HEADERS
        return ret, CORS_HEADERS

def save_to_gif(buf, frames):
    save_images = []
    for im in frames:
        colors, counts = np.unique(im.reshape(-1, 4), axis=0, return_counts=True)
        sort_indices = np.argsort(counts)
        colors = colors[sort_indices[::-1]] # Sort in descending order
        palette_colors = [0, 0, 0]
        formatted_colors = colors[colors[:, 3] != 0][..., :3]
        formatted_colors = formatted_colors[:255].flatten()
        palette_colors.extend(formatted_colors)
        dummy = Image.new('P', (16, 16))
        dummy.putpalette(palette_colors)
        save_images.append(Image.fromarray(im).convert('RGB').quantize(palette=dummy, dither=Image.Dither.NONE))
    kwargs = {
        'format': "GIF",
        'interlace': True,
        'save_all': True,
        'append_images': save_images[1:],
        'loop': 0,
        'duration': 200,
        'disposal': 2,
        'background': 0,
        'transparency': 0,
        'optimize': False
    }
    save_images[0].save(buf, **kwargs)

@app.route("/tiles/<string:tile_name>.gif", methods = ["GET"])
@rate_limit(100, timedelta(minutes = 1))
@dumb_cache
async def tile_icon(tile_name: str):
    try:
        tile_frame = int(request.args.get('frame', '0'))
    except ValueError:
        return "Query parameter 'frame' must be an integer", 400, CORS_HEADERS
    async with globals.conn.cursor() as cur:
        res = await cur.execute("SELECT source, sprite, active_color_x, active_color_y from tiles WHERE name == ?", tile_name)
        row = await res.fetchone()
        if row is None:
            return "No tile by the specified name exists in the database", 404, CORS_HEADERS
        source, sprite, active_color_x, active_color_y = row
        try:
            palette_color = PALETTE[active_color_y, active_color_x].astype(np.float32) / 255
        except IndexError:
            return "The specified palette index is out of bounds", 400, CORS_HEADERS
        frames = []
        try:
            for i in range(3):
                with Image.open(f"data/sprites/{source}/{sprite}_{tile_frame}_{i+1}.png") as im:
                    frames.append(np.array(im.convert("RGBA"), dtype=np.float32))
        except FileNotFoundError:
            return "One or more image sprites for the tile by the specified name at the specified animation frame does not exist", 404, CORS_HEADERS

        all_frames = np.stack(frames)
        all_frames[all_frames[..., 3] == 0] = 0
        all_frames[..., :] *= palette_color
        all_frames = np.clip(all_frames, 0, 255).astype(np.uint8)

        buf = io.BytesIO()
        save_to_gif(buf, all_frames)

        return buf.getvalue(), 200, {"Content-Type": "image/gif"} | CORS_HEADERS

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
            return 'Invalid query "author": must be an integer', 400, CORS_HEADERS
    if "data_only" in query:
        command = "SELECT name, value" + command.removeprefix("SELECT *")

    async with globals.conn.cursor() as cur:
        res = await cur.execute(command, *args)
        rows = await res.fetchall()
        if "name" in query and len(rows) == 0:
            return f"No macro by the name {query['name'][0]} exists in the database", 404, CORS_HEADERS
        ret = {}
        for row in rows:
            row = {key: val for key, val in zip(row.keys(), (*row, ))}
            name = row["name"]
            del row["name"]
            if "data_only" in query:
                row = row["value"]
                if "name" in query:
                    return row, CORS_HEADERS
            ret[name] = row
        return ret, CORS_HEADERS


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
            return 'Invalid query "author": must be an integer', 400, CORS_HEADERS

    async with globals.conn.cursor() as cur:
        res = await cur.execute(command, *args)
        rows = await res.fetchall()
        if "name" in query and len(rows) == 0:
            return f"No filter by the name {query['name'][0]} exists in the database", 404, CORS_HEADERS
        ret = {}
        for row in rows:
            row = {key: val for key, val in zip(row.keys(), (*row, ))}
            if "name" in query:
                return row, CORS_HEADERS
            name = row["name"]
            del row["name"]
            row["absolute"] = row["absolute"] != 0
            ret[name] = row
        return ret, CORS_HEADERS


@app.route("/filters/<string:filter_name>.png", methods = ["GET"])
@rate_limit(1, timedelta(seconds = 3))
@rate_limit(10, timedelta(minutes = 1))
async def filter_blob(filter_name):
    async with globals.conn.cursor() as cur:
        res = await cur.execute("SELECT data, absolute FROM filters WHERE name == ?", filter_name)
        row = await res.fetchone()
        if row is None:
            return f"No filter by the name {filter_name} exists in the database", 404
        return row[0], 200, {"absolute": row[1], "Content-Type": "image/png"} | CORS_HEADERS



@app.route("/<path:_>", methods = ["OPTIONS"])
@rate_exempt
async def cors_stuff():
    return b"", 200, CORS_HEADERS
