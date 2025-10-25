RIC exposes a REST API for read-only database access at ric-api.sno.mba.

# Endpoints

## `/`
Returns a JSON list of all other valid endpoints.

## `/macros.json`
Returns a JSON map of all macros in the bot's database, in the following schema:

```json
NAME: {
    "creator": AUTHOR_ID <int>,
    "value": VALUE <string>,
    "description" DESCRIPTION <string>
}
```
The following URL query parameters can be used to narrow the search:
- `?name=<string>` - Grabs only the data of the specified macro. Returns a 404 if it doesn't exist.
- `?author=<int>` - Matches macros created by the given Discord user, by ID.
- `?data_only=true` - Returns a string for the `value` instead of a JSON object, for each object in the map.

## `/tiles.json`
Returns a JSON map of all tiles in the bot's database, in the following schema:

```json
NAME: {
	 "active_color": [ACTIVE_COLOR_X <int>, ACTIVE_COLOR_Y <int>],
    "inactive_color": [INACTIVE_COLOR_X <int>, INACTIVE_COLOR_Y <int>],
    "sprite": [SOURCE <string>, SPRITE <string>],
    "tiling": TILING_MODE <string>,
    "tags": TAGS <list[string]>
}
```
`TILING_MODE` may be one of `icon, custom, none, directional, tiling, character, animated_directional, animated, static_character, diagonal_tiling`.

The following URL query parameters can be used to narrow the search:
- `?name=<string>` - Grabs only the data of the specified tile. Returns a 404 if it doesn't exist.
- `?source=<string>` - Matches tiles from the given source.
- `?tag=<string>` - Matches tiles with the given tag. May be specified more than once.
- `?tiling=<string>` - Matches tiles with the given tiling mode. Returns 400 if this is not a valid tiling mode.

## `/filters.json`
Returns a JSON map of all filters in the bot's database, in the following schema:

```json
NAME: {
    "author": AUTHOR_ID <int>,
    "absolute": IS_ABSOLUTE <bool>,
    "upload_time" UPLOAD_TIME <int>
}
```
The following URL query parameters can be used to narrow the search:
- `?name=<string>` - Grabs only the data of the specified macro. Returns a 404 if it doesn't exist.
- `?author=<int>` - Matches macros created by the given Discord user, by ID.

## `/filters/*.png`
Returns the image corresponding to the given filter, or returns 404 if the filter does not exist.
The `absolute` HTTP header will be set to 0 or 1 depending on whether the filter is absolute or not.

# Ratelimits

Each endpoint returning dynamic data is limited to one request per IP every 3 seconds, or 10 requests every minute.
