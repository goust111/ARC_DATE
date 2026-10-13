import random
from typing import Iterable

import numpy as np


BG = 0
GREY = 5
PATH_PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)

NORTH = (-1, 0)
EAST = (0, 1)
SOUTH = (1, 0)
WEST = (0, -1)
DIRS = (NORTH, EAST, SOUTH, WEST)
DIR_INDEX = {d: i for i, d in enumerate(DIRS)}


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(grid == color):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        stack = [(r, c)]
        coords: list[tuple[int, int]] = []
        while stack:
            cr, cc = stack.pop()
            coords.append((cr, cc))
            for nr, nc in _neighbors4(cr, cc):
                if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(coords)
    return comps


def _rotate_cw(offset: tuple[int, int]) -> tuple[int, int]:
    dr, dc = offset
    return dc, -dr


def _rotate_offsets(
    offsets: Iterable[tuple[int, int]], src_dir: tuple[int, int], dst_dir: tuple[int, int]
) -> list[tuple[int, int]]:
    turns = (DIR_INDEX[dst_dir] - DIR_INDEX[src_dir]) % 4
    out: list[tuple[int, int]] = []
    for off in offsets:
        cur = off
        for _ in range(turns):
            cur = _rotate_cw(cur)
        out.append(cur)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy each grey endpoint symbol to the far end of another colored path, using the marker color as the lookup key and the receiving path color as the repaint color.

    Input structure:
    - The background is black.
    - Several different-colored paths appear as one-pixel-wide 4-connected lines; each path is monochrome, unbranched, and may include right-angle turns.
    - One endpoint of each path touches a small grey symbol component inside a 3x3 neighborhood.
    - The other endpoint touches a single colored marker pixel whose color matches a different path's color.

    Rule:
    - For each path, identify its two degree-1 endpoints and determine which endpoint touches the grey symbol and which touches the marker pixel.
    - The marker color is an address: find the path whose line color matches that marker color, and take the grey symbol attached to that referenced path.
    - Treat the symbol cell touching its source endpoint as the anchor. Copy that anchored symbol to the current path's marker cell, rotating it so it faces the receiving endpoint in the same way, and recolor the copied symbol to the current path color.
    - Keep the original grey symbols and all paths unchanged; only the marker end gains the recolored copied symbol.

    Key insight:
    - Solving the puzzle requires two linked inferences: first pair each symbol endpoint with its marker endpoint by tracing the path, then interpret marker colors as indirect references to symbols attached to other same-colored paths.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    grey_components = _components_of_color(g, GREY)
    grey_owner: dict[tuple[int, int], int] = {}
    for idx, comp in enumerate(grey_components):
        for cell in comp:
            grey_owner[cell] = idx

    marker_cells: dict[tuple[int, int], int] = {}
    path_records: dict[int, dict[str, object]] = {}

    for color in sorted(int(c) for c in np.unique(g) if int(c) not in (BG, GREY)):
        for comp in _components_of_color(g, color):
            if len(comp) == 1:
                marker_cells[comp[0]] = color
                continue

            comp_set = set(comp)
            degrees = {cell: sum((nbr in comp_set) for nbr in _neighbors4(*cell)) for cell in comp}
            endpoints = [cell for cell, deg in degrees.items() if deg == 1]
            if not (all(deg in (1, 2) for deg in degrees.values()) and len(endpoints) == 2):
                raise AssertionError(f"unexpected non-path component for color {color}")
            if color in path_records:
                raise AssertionError(f"duplicate path color {color}")
            path_records[color] = {
                "cells": comp_set,
                "endpoints": endpoints,
            }

    used_grey: set[int] = set()
    used_markers: set[tuple[int, int]] = set()

    for path_color, rec in path_records.items():
        endpoints = rec["endpoints"]
        symbol_info = None
        marker_info = None

        for endpoint in endpoints:
            adj_grey = {
                grey_owner[(nr, nc)]
                for nr, nc in _neighbors4(*endpoint)
                if (nr, nc) in grey_owner
            }
            adj_markers = [
                (nr, nc)
                for nr, nc in _neighbors4(*endpoint)
                if (nr, nc) in marker_cells and marker_cells[(nr, nc)] != path_color
            ]

            if adj_grey:
                if len(adj_grey) != 1 or adj_markers:
                    raise AssertionError("symbol endpoint must touch exactly one grey component")
                grey_idx = next(iter(adj_grey))
                grey_comp = grey_components[grey_idx]
                anchor_cells = [cell for cell in grey_comp if abs(cell[0] - endpoint[0]) + abs(cell[1] - endpoint[1]) == 1]
                if len(anchor_cells) != 1:
                    raise AssertionError("symbol must touch the path through exactly one anchor cell")
                anchor = anchor_cells[0]
                symbol_info = {
                    "grey_idx": grey_idx,
                    "anchor": anchor,
                    "dir": (anchor[0] - endpoint[0], anchor[1] - endpoint[1]),
                    "offsets": [(r - anchor[0], c - anchor[1]) for r, c in grey_comp],
                }
            elif adj_markers:
                if len(adj_markers) != 1:
                    raise AssertionError("marker endpoint must touch exactly one marker")
                marker = adj_markers[0]
                marker_info = {
                    "pos": marker,
                    "dir": (marker[0] - endpoint[0], marker[1] - endpoint[1]),
                    "color": marker_cells[marker],
                }
            else:
                raise AssertionError("each path endpoint must touch either a symbol or a marker")

        if symbol_info is None or marker_info is None:
            raise AssertionError("each path needs exactly one symbol end and one marker end")

        used_grey.add(int(symbol_info["grey_idx"]))
        used_markers.add(marker_info["pos"])
        rec["symbol"] = symbol_info
        rec["marker"] = marker_info

    if len(used_grey) != len(grey_components):
        raise AssertionError("unused grey component detected")
    if len(used_markers) != len(marker_cells):
        raise AssertionError("unused marker detected")

    out = g.copy()
    for path_color, rec in path_records.items():
        marker = rec["marker"]
        src_color = int(marker["color"])
        if src_color not in path_records:
            raise AssertionError(f"marker color {src_color} does not match any path")
        src_symbol = path_records[src_color]["symbol"]
        rotated = _rotate_offsets(src_symbol["offsets"], src_symbol["dir"], marker["dir"])
        mr, mc = marker["pos"]
        for dr, dc in rotated:
            rr = mr + dr
            cc = mc + dc
            if not (0 <= rr < h and 0 <= cc < w):
                raise AssertionError("copied symbol goes out of bounds")
            out[rr, cc] = path_color
    return out


# --------------- generator ---------------

ROOM = 10
TOP = 3
BOTTOM = 6
LEFT = 3
RIGHT = 6
PATH_COUNTS = (3, 4)
RETRIES = 200

# Canonical symbol offsets assume the anchor is one cell to the right of its path endpoint.
SHAPES: tuple[tuple[tuple[int, int], ...], ...] = (
    ((0, 0), (0, 1), (1, 0)),                            # L triomino
    ((0, 0), (0, 1), (0, 2)),                            # line triomino
    ((0, 0), (0, 1), (1, 0), (1, 1)),                   # square tetromino
    ((0, 0), (-1, 1), (0, 1), (1, 1)),                  # T tetromino
    ((0, 0), (0, 1), (1, 1), (1, 2)),                   # Z tetromino
    ((0, 0), (0, 1), (-1, 1), (1, 1), (0, 2)),          # plus pentomino
    ((0, 0), (0, 1), (1, 0), (1, 1), (0, 2)),           # P pentomino
)


def _segment(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    (r0, c0), (r1, c1) = a, b
    if r0 == r1:
        lo, hi = sorted((c0, c1))
        return [(r0, c) for c in range(lo, hi + 1)]
    if c0 == c1:
        lo, hi = sorted((r0, r1))
        return [(r, c0) for r in range(lo, hi + 1)]
    raise ValueError("segments must be axis-aligned")


def _random_derangement(n: int) -> list[int]:
    perm = list(range(n))
    while True:
        random.shuffle(perm)
        if all(i != perm[i] for i in range(n)):
            return perm[:]


def _shape_cells(
    shape_idx: int, attach_dir: tuple[int, int], anchor: tuple[int, int]
) -> list[tuple[int, int]]:
    rotated = _rotate_offsets(SHAPES[shape_idx], EAST, attach_dir)
    ar, ac = anchor
    return [(ar + dr, ac + dc) for dr, dc in rotated]


def _path_template() -> dict[str, object]:
    kind = random.choice(("straight_h", "straight_v", "l", "zig_h", "zig_v"))

    if kind == "straight_h":
        row = random.randint(TOP, BOTTOM)
        cells = _segment((row, LEFT), (row, RIGHT))
        endpoints = [((row, LEFT), WEST), ((row, RIGHT), EAST)]
    elif kind == "straight_v":
        col = random.randint(LEFT, RIGHT)
        cells = _segment((TOP, col), (BOTTOM, col))
        endpoints = [((TOP, col), NORTH), ((BOTTOM, col), SOUTH)]
    elif kind == "l":
        orient = random.choice(("LT", "TR", "RB", "BL"))
        if orient == "LT":
            r = random.randint(TOP + 1, BOTTOM)
            c = random.randint(LEFT + 1, RIGHT)
            cells = list(set(_segment((r, LEFT), (r, c)) + _segment((TOP, c), (r, c))))
            endpoints = [((r, LEFT), WEST), ((TOP, c), NORTH)]
        elif orient == "TR":
            r = random.randint(TOP + 1, BOTTOM)
            c = random.randint(LEFT, RIGHT - 1)
            cells = list(set(_segment((TOP, c), (r, c)) + _segment((r, c), (r, RIGHT))))
            endpoints = [((TOP, c), NORTH), ((r, RIGHT), EAST)]
        elif orient == "RB":
            r = random.randint(TOP, BOTTOM - 1)
            c = random.randint(LEFT, RIGHT - 1)
            cells = list(set(_segment((r, c), (r, RIGHT)) + _segment((r, c), (BOTTOM, c))))
            endpoints = [((r, RIGHT), EAST), ((BOTTOM, c), SOUTH)]
        else:
            r = random.randint(TOP, BOTTOM - 1)
            c = random.randint(LEFT + 1, RIGHT)
            cells = list(set(_segment((r, LEFT), (r, c)) + _segment((r, c), (BOTTOM, c))))
            endpoints = [((BOTTOM, c), SOUTH), ((r, LEFT), WEST)]
    elif kind == "zig_h":
        r1, r2 = random.sample(range(TOP, BOTTOM + 1), 2)
        cmid = random.randint(LEFT + 1, RIGHT - 1)
        cells = list(
            set(
                _segment((r1, LEFT), (r1, cmid))
                + _segment((r1, cmid), (r2, cmid))
                + _segment((r2, cmid), (r2, RIGHT))
            )
        )
        endpoints = [((r1, LEFT), WEST), ((r2, RIGHT), EAST)]
    else:
        c1, c2 = random.sample(range(LEFT, RIGHT + 1), 2)
        rmid = random.randint(TOP + 1, BOTTOM - 1)
        cells = list(
            set(
                _segment((TOP, c1), (rmid, c1))
                + _segment((rmid, c1), (rmid, c2))
                + _segment((rmid, c2), (BOTTOM, c2))
            )
        )
        endpoints = [((TOP, c1), NORTH), ((BOTTOM, c2), SOUTH)]

    endpoint_index = random.randrange(2)
    symbol_endpoint, symbol_dir = endpoints[endpoint_index]
    marker_endpoint, marker_dir = endpoints[1 - endpoint_index]
    return {
        "cells": cells,
        "symbol_endpoint": symbol_endpoint,
        "symbol_dir": symbol_dir,
        "marker_endpoint": marker_endpoint,
        "marker_dir": marker_dir,
    }


def _symmetry(grid: np.ndarray, code: int) -> np.ndarray:
    if code < 4:
        return np.rot90(grid, code)
    return np.fliplr(np.rot90(grid, code - 4))


def _gen_attempt() -> np.ndarray | None:
    n = random.choice(PATH_COUNTS)
    path_colors = random.sample(PATH_PALETTE, n)
    shape_ids = random.sample(range(len(SHAPES)), n)
    src_for_dest = _random_derangement(n)

    room_specs = []
    for idx in range(n):
        tpl = _path_template()
        room_specs.append(
            {
                "path_color": path_colors[idx],
                "marker_color": path_colors[src_for_dest[idx]],
                "shape_idx": shape_ids[idx],
                **tpl,
            }
        )

    row_gap = random.randint(1, 4)
    col_gap = random.randint(1, 4)
    base_h = 2 * ROOM + row_gap
    base_w = 2 * ROOM + col_gap
    extra_h = random.randint(0, 30 - base_h)
    extra_w = random.randint(0, 30 - base_w)
    top_margin = random.randint(0, extra_h)
    left_margin = random.randint(0, extra_w)
    bottom_margin = extra_h - top_margin
    right_margin = extra_w - left_margin
    height = base_h + top_margin + bottom_margin
    width = base_w + left_margin + right_margin

    row_starts = [top_margin, top_margin + ROOM + row_gap]
    col_starts = [left_margin, left_margin + ROOM + col_gap]
    slots = [(r0, c0) for r0 in row_starts for c0 in col_starts]
    slot_choices = random.sample(slots, n)

    inp = np.full((height, width), BG, dtype=int)
    out = np.full((height, width), BG, dtype=int)

    for spec, (r0, c0) in zip(room_specs, slot_choices):
        spec["room_origin"] = (r0, c0)

        for rr, cc in spec["cells"]:
            inp[r0 + rr, c0 + cc] = spec["path_color"]

        sym_anchor = (
            spec["symbol_endpoint"][0] + spec["symbol_dir"][0],
            spec["symbol_endpoint"][1] + spec["symbol_dir"][1],
        )
        for rr, cc in _shape_cells(spec["shape_idx"], spec["symbol_dir"], sym_anchor):
            if not (0 <= rr < ROOM and 0 <= cc < ROOM):
                return None
            inp[r0 + rr, c0 + cc] = GREY

        mr = spec["marker_endpoint"][0] + spec["marker_dir"][0]
        mc = spec["marker_endpoint"][1] + spec["marker_dir"][1]
        if not (0 <= mr < ROOM and 0 <= mc < ROOM):
            return None
        inp[r0 + mr, c0 + mc] = spec["marker_color"]

    out[:] = inp
    for dest_idx, spec in enumerate(room_specs):
        src_idx = src_for_dest[dest_idx]
        src_shape_idx = room_specs[src_idx]["shape_idx"]
        mr = spec["marker_endpoint"][0] + spec["marker_dir"][0]
        mc = spec["marker_endpoint"][1] + spec["marker_dir"][1]
        for rr, cc in _shape_cells(src_shape_idx, spec["marker_dir"], (mr, mc)):
            if not (0 <= rr < ROOM and 0 <= cc < ROOM):
                return None
            out[spec["room_origin"][0] + rr, spec["room_origin"][1] + cc] = spec["path_color"]

    sym = random.randrange(8)
    inp = _symmetry(inp, sym).copy()
    out = _symmetry(out, sym).copy()

    try:
        if not np.array_equal(main(inp), out):
            return None
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {RETRIES} attempts")
