from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _portal_pairs(g: np.ndarray) -> list[dict]:
    counts = Counter(int(v) for v in g.ravel() if int(v) != BG)
    pairs: list[dict] = []

    for color, count in counts.items():
        if count != 2:
            continue

        pts = sorted((int(r), int(c)) for r, c in np.argwhere(g == int(color)))
        if len(pts) != 2:
            continue

        (r0, c0), (r1, c1) = pts
        if r0 == r1 and c1 - c0 >= 2:
            strip = g[r0, c0 + 1 : c1]
            if strip.size == 0 or not bool(np.all(strip != BG)):
                continue
            strip_colors = {int(v) for v in strip}
            if len(strip_colors) != 1:
                continue
            strip_color = strip_colors.pop()
            if strip_color == int(color):
                continue
            pairs.append(
                {
                    "portal_color": int(color),
                    "orientation": "h",
                    "portals": [(r0, c0), (r1, c1)],
                    "length": int(c1 - c0 - 1),
                }
            )
        elif c0 == c1 and r1 - r0 >= 2:
            strip = g[r0 + 1 : r1, c0]
            if strip.size == 0 or not bool(np.all(strip != BG)):
                continue
            strip_colors = {int(v) for v in strip}
            if len(strip_colors) != 1:
                continue
            strip_color = strip_colors.pop()
            if strip_color == int(color):
                continue
            pairs.append(
                {
                    "portal_color": int(color),
                    "orientation": "v",
                    "portals": [(r0, c0), (r1, c1)],
                    "length": int(r1 - r0 - 1),
                }
            )

    pairs.sort(key=lambda item: item["portals"][0])
    assert pairs, "expected at least one valid portal pair"
    return pairs


def _exit_marker(g: np.ndarray, portal: tuple[int, int], orientation: str) -> tuple[int, int, int, int, int] | None:
    h, w = g.shape
    r, c = portal
    if orientation == "h":
        dirs = [(-1, 0), (1, 0)]
    else:
        dirs = [(0, -1), (0, 1)]

    hits: list[tuple[int, int, int, int, int]] = []
    for dr, dc in dirs:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < h and 0 <= nc < w and int(g[nr, nc]) != BG:
            hits.append((nr, nc, dr, dc, int(g[nr, nc])))

    assert len(hits) <= 1, "each portal may have at most one exit marker"
    return hits[0] if hits else None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Treat each same-color pixel pair as a portal pair and rotate-copy the strip between them through any exit markers.

    Input structure:
    - Background is black.
    - Some colors appear as exactly two isolated portal pixels, aligned on one row or one column.
    - Between the two portals sits a continuous one-cell-wide non-black strip.
    - A portal may also have one perpendicular 4-neighbor marker that indicates an output direction.

    Rule:
    - Pair portals by color.
    - Measure the strip length between the paired portals.
    - For each portal endpoint that has an exit marker, paste a rotated copy of the strip starting at that marker
      and extending away from the portal; the copy uses the exit marker's color.
    - Leave the original portals and original strip unchanged.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for pair in _portal_pairs(g):
        length = int(pair["length"])
        orientation = str(pair["orientation"])
        for portal in pair["portals"]:
            marker = _exit_marker(g, portal, orientation)
            if marker is None:
                continue

            mr, mc, dr, dc, color = marker
            for step in range(length):
                rr = mr + dr * step
                cc = mc + dc * step
                assert _in_bounds(g.shape, rr, cc), "output copy must fit in the grid"
                if step > 0:
                    assert int(g[rr, cc]) == BG, "output corridor must be empty in the input"
                    assert int(out[rr, cc]) == BG, "output copies must not overlap"
                out[rr, cc] = color

    return out


# --------------- generator ---------------

GEN_RETRIES = 400
PAIR_COUNT_RANGE = (2, 4)
STRIP_LEN_RANGE = (3, 6)
BLOCK_PADDING = 1
COLORS = tuple(range(1, 10))
MARKER_PATTERNS = (
    (False, False),
    (True, False),
    (False, True),
    (True, True),
)
GRID_RANGES = {
    2: ((12, 18), (12, 20)),
    3: ((14, 22), (14, 24)),
    4: ((16, 24), (16, 26)),
}


def _portal_positions(spec: dict) -> list[tuple[int, int]]:
    r = int(spec["row"])
    c = int(spec["col"])
    length = int(spec["length"])
    if spec["orientation"] == "h":
        return [(r, c), (r, c + length + 1)]
    return [(r, c), (r + length + 1, c)]


def _draw_input(specs: list[dict], shape: tuple[int, int]) -> np.ndarray:
    g = np.zeros(shape, dtype=int)

    for spec in specs:
        r = int(spec["row"])
        c = int(spec["col"])
        length = int(spec["length"])
        portal_color = int(spec["portal_color"])
        strip_color = int(spec["strip_color"])

        if spec["orientation"] == "h":
            g[r, c] = portal_color
            g[r, c + length + 1] = portal_color
            g[r, c + 1 : c + length + 1] = strip_color
        else:
            g[r, c] = portal_color
            g[r + length + 1, c] = portal_color
            g[r + 1 : r + length + 1, c] = strip_color

        for exit_info in spec["exits"]:
            if exit_info is None:
                continue
            mr, mc = exit_info["cell"]
            g[mr, mc] = int(exit_info["color"])

    return g


def _draw_expected(specs: list[dict], shape: tuple[int, int]) -> np.ndarray:
    out = _draw_input(specs, shape)
    for spec in specs:
        length = int(spec["length"])
        for exit_info in spec["exits"]:
            if exit_info is None:
                continue
            mr, mc = exit_info["cell"]
            dr, dc = exit_info["dir"]
            color = int(exit_info["color"])
            for step in range(length):
                out[mr + dr * step, mc + dc * step] = color
    return out


def _reserve_cells(spec: dict, shape: tuple[int, int]) -> set[tuple[int, int]] | None:
    h, w = shape
    reserved: set[tuple[int, int]] = set()

    for pr, pc in _portal_positions(spec):
        if not _in_bounds(shape, pr, pc):
            return None
        reserved.add((pr, pc))

    r = int(spec["row"])
    c = int(spec["col"])
    length = int(spec["length"])
    if spec["orientation"] == "h":
        for cc in range(c + 1, c + length + 1):
            reserved.add((r, cc))
    else:
        for rr in range(r + 1, r + length + 1):
            reserved.add((rr, c))

    for exit_info in spec["exits"]:
        if exit_info is None:
            continue
        mr, mc = exit_info["cell"]
        dr, dc = exit_info["dir"]
        for step in range(length):
            rr = mr + dr * step
            cc = mc + dc * step
            if not (0 <= rr < h and 0 <= cc < w):
                return None
            reserved.add((rr, cc))

    return reserved


def _can_place(blocked: np.ndarray, reserved: set[tuple[int, int]]) -> bool:
    for r, c in reserved:
        if blocked[r, c]:
            return False
    return True


def _block_region(blocked: np.ndarray, reserved: set[tuple[int, int]]) -> None:
    h, w = blocked.shape
    for r, c in reserved:
        r0 = max(0, r - BLOCK_PADDING)
        r1 = min(h, r + BLOCK_PADDING + 1)
        c0 = max(0, c - BLOCK_PADDING)
        c1 = min(w, c + BLOCK_PADDING + 1)
        blocked[r0:r1, c0:c1] = True


def _sample_exits(
    orientation: str,
    pattern: tuple[bool, bool],
    payload_palette: list[int],
    strip_color: int,
) -> list[dict | None]:
    if orientation == "h":
        options = [(-1, 0), (1, 0)]
    else:
        options = [(0, -1), (0, 1)]

    marker_colors = [color for color in payload_palette if color != strip_color]
    assert marker_colors, "payload palette must leave at least one marker color"

    needed = int(pattern[0]) + int(pattern[1])
    if needed <= len(marker_colors):
        chosen = random.sample(marker_colors, needed)
    else:
        chosen = [random.choice(marker_colors) for _ in range(needed)]

    exits: list[dict | None] = []
    color_index = 0
    for present in pattern:
        if not present:
            exits.append(None)
            continue
        exits.append({"dir": random.choice(options), "color": int(chosen[color_index])})
        color_index += 1
    return exits


def _sample_location(spec: dict, shape: tuple[int, int]) -> tuple[int, int] | None:
    h, w = shape
    length = int(spec["length"])
    exits = spec["exits"]

    if spec["orientation"] == "h":
        any_up = any(exit_info is not None and exit_info["dir"] == (-1, 0) for exit_info in exits)
        any_down = any(exit_info is not None and exit_info["dir"] == (1, 0) for exit_info in exits)
        min_row = length if any_up else 0
        max_row = h - 1 - length if any_down else h - 1
        min_col = 0
        max_col = w - (length + 2)
        if min_row > max_row or min_col > max_col:
            return None
        return random.randint(min_row, max_row), random.randint(min_col, max_col)

    any_left = any(exit_info is not None and exit_info["dir"] == (0, -1) for exit_info in exits)
    any_right = any(exit_info is not None and exit_info["dir"] == (0, 1) for exit_info in exits)
    min_row = 0
    max_row = h - (length + 2)
    min_col = length if any_left else 0
    max_col = w - 1 - length if any_right else w - 1
    if min_row > max_row or min_col > max_col:
        return None
    return random.randint(min_row, max_row), random.randint(min_col, max_col)


def _place_pair(
    portal_color: int,
    pattern: tuple[bool, bool],
    payload_palette: list[int],
    shape: tuple[int, int],
    blocked: np.ndarray,
) -> dict | None:
    for _ in range(120):
        orientation = random.choice(("h", "v"))
        length = random.randint(*STRIP_LEN_RANGE)
        strip_color = random.choice(payload_palette)
        exits = _sample_exits(orientation, pattern, payload_palette, strip_color)

        spec = {
            "portal_color": int(portal_color),
            "orientation": orientation,
            "length": int(length),
            "strip_color": int(strip_color),
            "row": 0,
            "col": 0,
            "exits": exits,
        }

        loc = _sample_location(spec, shape)
        if loc is None:
            continue
        spec["row"], spec["col"] = map(int, loc)

        portals = _portal_positions(spec)
        for exit_info, (pr, pc) in zip(spec["exits"], portals):
            if exit_info is None:
                continue
            dr, dc = exit_info["dir"]
            exit_info["cell"] = (pr + dr, pc + dc)

        reserved = _reserve_cells(spec, shape)
        if reserved is None or not _can_place(blocked, reserved):
            continue

        return {"spec": spec, "reserved": reserved}

    return None


def _sample_shape(pair_count: int) -> tuple[int, int]:
    row_range, col_range = GRID_RANGES[pair_count]
    return random.randint(*row_range), random.randint(*col_range)


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.randint(*PAIR_COUNT_RANGE)
    shape = _sample_shape(pair_count)

    portal_colors = random.sample(list(COLORS), pair_count)
    payload_candidates = [color for color in COLORS if color not in portal_colors]
    payload_size = random.randint(3, min(5, len(payload_candidates)))
    payload_palette = random.sample(payload_candidates, payload_size)

    patterns = [random.choice(MARKER_PATTERNS) for _ in range(pair_count)]
    if not any(any(pattern) for pattern in patterns):
        patterns[random.randrange(pair_count)] = random.choice(MARKER_PATTERNS[1:])

    blocked = np.zeros(shape, dtype=bool)
    specs: list[dict] = []
    for portal_color, pattern in zip(portal_colors, patterns):
        placed = _place_pair(portal_color, pattern, payload_palette, shape, blocked)
        if placed is None:
            return None
        specs.append(placed["spec"])
        _block_region(blocked, placed["reserved"])

    inp = _draw_input(specs, shape)
    expected = _draw_expected(specs, shape)

    try:
        actual = main(inp)
    except AssertionError:
        return None

    if np.array_equal(actual, expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise AssertionError("failed to generate a valid portal-strip input")
