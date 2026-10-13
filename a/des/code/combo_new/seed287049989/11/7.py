from __future__ import annotations

import random
from typing import Any

import numpy as np


GREY = 5
MAROON = 9
DIR4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]
CORNER_PAD = 3
GAP_MIN = 1
GEN_RETRIES = 400

STATION_TEMPLATES = {
    "top": {
        "seq": [(0, -1), (1, -1), (2, -1), (2, 0), (2, 1), (2, 2), (1, 2), (0, 2)],
        "open": [(-1, 0), (-1, 1)],
    },
    "right": {
        "seq": [(-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (2, -1), (2, 0), (2, 1)],
        "open": [(0, 2), (1, 2)],
    },
    "bottom": {
        "seq": [(1, 2), (0, 2), (-1, 2), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)],
        "open": [(2, 0), (2, 1)],
    },
    "left": {
        "seq": [(2, 0), (2, 1), (2, 2), (1, 2), (0, 2), (-1, 2), (-1, 1), (-1, 0)],
        "open": [(0, -1), (1, -1)],
    },
}

BG_CHOICES = [0, 1, 2, 3, 4, 6, 7, 8]
PATTERN_MASKS = [
    np.array(
        [
            [int((mask >> 0) & 1), int((mask >> 1) & 1)],
            [int((mask >> 2) & 1), int((mask >> 3) & 1)],
        ],
        dtype=int,
    )
    for mask in range(1, 15)
]


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for dr, dc in DIR4]


def _ordered_cycle(path_mask: np.ndarray, forbidden: set[tuple[int, int]]) -> list[tuple[int, int]]:
    h, w = path_mask.shape
    pts = {tuple(map(int, p)) for p in np.argwhere(path_mask)}
    assert pts, "expected a non-empty conveyor path"

    def path_neighbors(cell: tuple[int, int]) -> list[tuple[int, int]]:
        r, c = cell
        return [(nr, nc) for nr, nc in _neighbors4(r, c) if 0 <= nr < h and 0 <= nc < w and (nr, nc) in pts]

    for cell in pts:
        assert len(path_neighbors(cell)) == 2, "conveyor path must be a single 4-connected cycle"

    starts = sorted(cell for cell in pts if cell not in forbidden)
    assert starts, "need at least one non-station path cell to anchor the cycle traversal"
    start = starts[0]
    first = sorted(path_neighbors(start))[0]
    seq = [start, first]
    prev, cur = start, first
    while True:
        nxts = [cell for cell in path_neighbors(cur) if cell != prev]
        assert len(nxts) == 1
        nxt = nxts[0]
        if nxt == start:
            break
        seq.append(nxt)
        prev, cur = cur, nxt
        assert len(seq) <= len(pts), "cycle walk exceeded path size"
    assert len(seq) == len(pts), "path walk did not cover the whole cycle"

    area = 0
    for (r1, c1), (r2, c2) in zip(seq, seq[1:] + seq[:1]):
        area += c1 * r2 - c2 * r1
    if area < 0:
        seq = [seq[0]] + list(reversed(seq[1:]))
    return seq


def _analyze_conveyor(grid: np.ndarray) -> dict[str, Any]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _most_common_color(g)
    path_mask = (g == GREY) | (g == MAROON)
    assert path_mask.any(), "expected grey/maroon conveyor path"

    payload_colors = [int(col) for col in np.unique(g) if int(col) not in (bg, GREY, MAROON)]
    assert len(payload_colors) == 1, "expected exactly one payload color"
    payload = payload_colors[0]

    stations = []
    occupied_pockets: set[tuple[int, int]] = set()

    for r in range(h - 1):
        for c in range(w - 1):
            pocket_cells = {(r + dr, c + dc) for dr in (0, 1) for dc in (0, 1)}
            if pocket_cells & occupied_pockets:
                continue
            block = g[r : r + 2, c : c + 2]
            if np.any((block == GREY) | (block == MAROON)):
                continue
            if np.any((block != bg) & (block != payload)):
                continue

            matches = []
            for kind, tpl in STATION_TEMPLATES.items():
                seq = [(r + dr, c + dc) for dr, dc in tpl["seq"]]
                if not all(_in_bounds(h, w, rr, cc) and path_mask[rr, cc] for rr, cc in seq):
                    continue
                open_cells = [(r + dr, c + dc) for dr, dc in tpl["open"]]
                if not all((not _in_bounds(h, w, rr, cc)) or (not path_mask[rr, cc]) for rr, cc in open_cells):
                    continue
                matches.append((kind, seq))

            if len(matches) != 1:
                continue

            kind, seq = matches[0]
            occupied_pockets |= pocket_cells
            stations.append(
                {
                    "kind": kind,
                    "top_left": (r, c),
                    "seq": seq,
                    "pattern": (block == payload).astype(int),
                }
            )

    assert len(stations) >= 3, "expected at least three conveyor stations"

    station_path_cells = {cell for station in stations for cell in station["seq"]}
    cycle = _ordered_cycle(path_mask, station_path_cells)
    index = {cell: i for i, cell in enumerate(cycle)}

    for station in stations:
        inds = [index[cell] for cell in station["seq"]]
        assert all(inds[i + 1] == inds[i] + 1 for i in range(len(inds) - 1)), "station path cells must be contiguous"
        station["start"] = inds[0]
        station["end"] = inds[-1]

    stations.sort(key=lambda station: station["start"])
    n_cycle = len(cycle)
    rotate_after_station = []
    for i, station in enumerate(stations):
        nxt = stations[(i + 1) % len(stations)]
        segment_cells = []
        cur = (station["end"] + 1) % n_cycle
        while cur != nxt["start"]:
            segment_cells.append(cycle[cur])
            cur = (cur + 1) % n_cycle
        assert segment_cells, "neighboring stations must be separated by at least one path cell"
        rotate_after_station.append(any(g[r, c] == MAROON for r, c in segment_cells))

    return {
        "bg": bg,
        "payload": payload,
        "stations": stations,
        "rotate_after_station": rotate_after_station,
        "cycle": cycle,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Shift each 2x2 payload pattern one station clockwise around the conveyor loop, rotating it on marked segments.

    Input structure:
    - The background is the dominant color.
    - Grey cells form a single-pixel-wide closed orthogonal loop; some loop cells may be maroon markers.
    - Next to the loop are several 2x2 stations: each station is a 2x2 pocket touching the loop on three sides and open on the outer side.
    - Each station contains a small 2x2 pattern made from one shared non-background, non-grey payload color.

    Rule:
    - Read the closed loop in clockwise order and list the stations in that same cyclic order.
    - Move every station's 2x2 payload pattern to the next station, wrapping from the last station back to the first.
    - If the path segment between two consecutive stations contains a maroon loop cell, rotate the moving 2x2 pattern 90 degrees clockwise before placing it.
    - Keep the grey/maroon conveyor path unchanged.

    Key insight:
    The grey loop is not just a border: its ordered cycle defines the station order, and maroon cells on the inter-station segments act as rotation instructions for the payload that crosses them.
    """
    g = np.asarray(grid, dtype=int)
    info = _analyze_conveyor(g)
    bg = int(info["bg"])
    payload = int(info["payload"])
    stations = info["stations"]
    rotate_after_station = info["rotate_after_station"]

    out = g.copy()
    for station in stations:
        r0, c0 = station["top_left"]
        out[r0 : r0 + 2, c0 : c0 + 2] = bg

    n = len(stations)
    for src_idx, station in enumerate(stations):
        dst = stations[(src_idx + 1) % n]
        pattern = station["pattern"]
        if rotate_after_station[src_idx]:
            pattern = np.rot90(pattern, -1)
        r0, c0 = dst["top_left"]
        for dr in range(2):
            for dc in range(2):
                if int(pattern[dr, dc]):
                    out[r0 + dr, c0 + dc] = payload

    return out


def _random_composition(total: int, parts: int, rng: random.Random) -> list[int]:
    if parts == 1:
        return [total]
    bars = sorted(rng.sample(range(total + parts - 1), parts - 1))
    vals = []
    prev = -1
    for bar in bars + [total + parts - 1]:
        vals.append(bar - prev - 1)
        prev = bar
    return vals


def _side_capacity(length: int) -> int:
    usable = length - 2 * CORNER_PAD
    if usable < 4:
        return 0
    return (usable + GAP_MIN) // (4 + GAP_MIN)


def _sample_station_offsets(length: int, count: int, rng: random.Random) -> list[int] | None:
    if count == 0:
        return []
    usable = length - 2 * CORNER_PAD
    slack = usable - (4 * count + GAP_MIN * (count - 1))
    if slack < 0:
        return None
    extra = _random_composition(slack, count + 1, rng)
    starts = []
    pos = CORNER_PAD + extra[0]
    starts.append(pos)
    for i in range(1, count):
        pos += 4 + GAP_MIN + extra[i]
        starts.append(pos)
    return starts


def _append_part(seq: list[tuple[int, int]], part: list[tuple[int, int]]) -> None:
    if not part:
        return
    if not seq:
        seq.extend(part)
        return
    assert seq[-1] == part[0]
    seq.extend(part[1:])


def _build_loop(
    top: int,
    left: int,
    bottom: int,
    right: int,
    offsets: dict[str, list[int]],
) -> tuple[list[tuple[int, int]], list[dict[str, Any]]]:
    seq = [(top, left)]
    stations = []

    cur = left
    for off in offsets["top"]:
        col = left + off
        _append_part(seq, [(top, c) for c in range(cur, col + 1)])
        part = [
            (top, col),
            (top + 1, col),
            (top + 2, col),
            (top + 2, col + 1),
            (top + 2, col + 2),
            (top + 2, col + 3),
            (top + 1, col + 3),
            (top, col + 3),
        ]
        _append_part(seq, part)
        stations.append({"kind": "top", "top_left": (top, col + 1), "seq": part})
        cur = col + 3
    _append_part(seq, [(top, c) for c in range(cur, right + 1)])

    cur = top
    for off in offsets["right"]:
        row = top + off
        _append_part(seq, [(r, right) for r in range(cur, row + 1)])
        part = [
            (row, right),
            (row, right - 1),
            (row, right - 2),
            (row + 1, right - 2),
            (row + 2, right - 2),
            (row + 3, right - 2),
            (row + 3, right - 1),
            (row + 3, right),
        ]
        _append_part(seq, part)
        stations.append({"kind": "right", "top_left": (row + 1, right - 1), "seq": part})
        cur = row + 3
    _append_part(seq, [(r, right) for r in range(cur, bottom + 1)])

    cur = right
    for off in sorted(offsets["bottom"], reverse=True):
        col = left + off
        _append_part(seq, [(bottom, c) for c in range(cur, col + 2, -1)])
        part = [
            (bottom, col + 3),
            (bottom - 1, col + 3),
            (bottom - 2, col + 3),
            (bottom - 2, col + 2),
            (bottom - 2, col + 1),
            (bottom - 2, col),
            (bottom - 1, col),
            (bottom, col),
        ]
        _append_part(seq, part)
        stations.append({"kind": "bottom", "top_left": (bottom - 1, col + 1), "seq": part})
        cur = col
    _append_part(seq, [(bottom, c) for c in range(cur, left - 1, -1)])

    cur = bottom
    for off in sorted(offsets["left"], reverse=True):
        row = top + off
        _append_part(seq, [(r, left) for r in range(cur, row + 2, -1)])
        part = [
            (row + 3, left),
            (row + 3, left + 1),
            (row + 3, left + 2),
            (row + 2, left + 2),
            (row + 1, left + 2),
            (row, left + 2),
            (row, left + 1),
            (row, left),
        ]
        _append_part(seq, part)
        stations.append({"kind": "left", "top_left": (row + 1, left), "seq": part})
        cur = row
    _append_part(seq, [(r, left) for r in range(cur, top - 1, -1)])

    assert seq[-1] == seq[0]
    seq = seq[:-1]
    assert len(seq) == len(set(seq)), "generated loop self-intersects"
    return seq, stations


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    outer_h = rng.randint(12, 24)
    outer_w = rng.randint(12, 24)
    margins = [rng.randint(1, 2) for _ in range(4)]
    top_margin, bottom_margin, left_margin, right_margin = margins[0], margins[1], margins[2], margins[3]
    height = outer_h + top_margin + bottom_margin
    width = outer_w + left_margin + right_margin
    if height > 30 or width > 30:
        return None

    top = top_margin
    left = left_margin
    bottom = top + outer_h - 1
    right = left + outer_w - 1

    caps = {
        "top": _side_capacity(outer_w),
        "right": _side_capacity(outer_h),
        "bottom": _side_capacity(outer_w),
        "left": _side_capacity(outer_h),
    }
    total_cap = sum(caps.values())
    if total_cap < 4:
        return None

    total_stations = rng.randint(4, min(8, total_cap))
    counts = {side: 0 for side in caps}
    positive_sides = [side for side, cap in caps.items() if cap > 0]
    for side in rng.sample(positive_sides, 2):
        counts[side] += 1
    remaining = total_stations - 2
    while remaining > 0:
        side = rng.choice([name for name, cap in caps.items() if counts[name] < cap])
        counts[side] += 1
        remaining -= 1

    offsets = {}
    for side in ("top", "right", "bottom", "left"):
        length = outer_w if side in ("top", "bottom") else outer_h
        offsets[side] = _sample_station_offsets(length, counts[side], rng)
        if offsets[side] is None:
            return None

    path_seq, built_stations = _build_loop(top, left, bottom, right, offsets)

    bg = rng.choice(BG_CHOICES)
    payload = rng.choice([color for color in BG_CHOICES if color != bg])
    grid = np.full((height, width), bg, dtype=int)
    for r, c in path_seq:
        grid[r, c] = GREY

    idx = {cell: i for i, cell in enumerate(path_seq)}
    built_stations.sort(key=lambda station: idx[station["seq"][0]])
    n_stations = len(built_stations)

    marked_segments = [rng.random() < 0.5 for _ in range(n_stations)]
    if all(marked_segments) or not any(marked_segments):
        marked_segments = [False] * n_stations
        for i in rng.sample(range(n_stations), rng.randint(1, n_stations - 1)):
            marked_segments[i] = True

    for station, mask in zip(built_stations, rng.sample(PATTERN_MASKS, n_stations)):
        r0, c0 = station["top_left"]
        for dr in range(2):
            for dc in range(2):
                if int(mask[dr, dc]):
                    grid[r0 + dr, c0 + dc] = payload

    for i, station in enumerate(built_stations):
        nxt = built_stations[(i + 1) % n_stations]
        segment = []
        cur = (idx[station["seq"][-1]] + 1) % len(path_seq)
        while cur != idx[nxt["seq"][0]]:
            segment.append(path_seq[cur])
            cur = (cur + 1) % len(path_seq)
        if not segment:
            return None
        if marked_segments[i]:
            rr, cc = rng.choice(segment)
            grid[rr, cc] = MAROON

    try:
        info = _analyze_conveyor(grid)
    except AssertionError:
        return None

    detected = sorted(station["top_left"] for station in info["stations"])
    expected = sorted(station["top_left"] for station in built_stations)
    if detected != expected:
        return None
    if all(info["rotate_after_station"]) or not any(info["rotate_after_station"]):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted retries")
