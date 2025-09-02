RIC exposes a REST API for read-only database access at ric-api.sno.mba.

# Endpoints

## `/`
Returns a JSON list of all other valid endpoints.

## `/macros.json`
Returns a JSON list of all macros in the bot's database.
Does not return builtins.

The following URL query parameters can be used to narrow the search:
- `?name=<string>` - Matches macros with the given string somewhere in their name. Matches using plain text, not regex.
- `?creator=<int>` - Matches macros created by the given Discord user, by ID.
- `?data_only=true` - Does not return the `creator` or `description` fields.

## `/tiles.json`
Returns a JSON list of all tiles in the bot's database.

The following URL query parameters can be used to narrow the search:
- `?name=<string>` - Matches tiles with the given string somewhere in their name. Matches using plain text, not regex.
- `?source=<string>` - Matches tiles from the given source.
- `?tag=<string>` - Matches tiles with the given tag. May be specified more than once.
- `?tiling=<tiling_mode>` - Matches tiles with the given tiling mode.

# Ratelimits

After any request, the bot will block any requests from the requester's IP until
`max(0, log(lastResponseByteLength, base = 1.5)) + 0.2` seconds have passed.
