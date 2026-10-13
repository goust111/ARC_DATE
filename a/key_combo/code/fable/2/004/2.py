from __future__ import annotations

from collections import Counter
import random

import numpy as np


BBox = tuple[int, int, int, int]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_components(grid: np.ndarray, color: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            coords = np.array(cells, dtype=int)
            comps.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": _bbox_coords(coords),
                    "area": int(len(cells)),
                }
            )
    return comps


def _bg_runs(values: np.ndarray, offset: int, bg: int) -> list[tuple[int, int]]:
    runs = []
    start = None
    for idx, value in enumerate(values):
        if int(value) == int(bg):
            if start is None:
                start = idx
        elif start is not None:
            runs.append((offset + start, offset + idx - 1))
            start = None
    if start is not None:
        runs.append((offset + start, offset + len(values) - 1))
    return runs


def _opening_cells(op: dict) -> tuple[slice, slice]:
    a0, a1 = op["span"]
    wall = op["wall"]
    if op["side"] in ("left", "right"):
        return slice(int(a0), int(a1) + 1), slice(int(wall), int(wall) + 1)
    return slice(int(wall), int(wall) + 1), slice(int(a0), int(a1) + 1)


def _extract_room_openings(grid: np.ndarray, room: dict, bg: int, color: int) -> list[dict]:
    r0, c0, r1, c1 = room["bbox"]
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    h, w = patch.shape
    assert h >= 3 and w >= 3, "rooms must be at least 3x3"
    assert np.all(patch[1:-1, 1:-1] == bg), "room interior must be background"

    border_mask = np.zeros_like(patch, dtype=bool)
    border_mask[0, :] = True
    border_mask[-1, :] = True
    border_mask[:, 0] = True
    border_mask[:, -1] = True
    assert np.all((patch == color) <= border_mask), "foreground must stay on room border"
    assert int(patch[0, 0]) == color and int(patch[0, -1]) == color
    assert int(patch[-1, 0]) == color and int(patch[-1, -1]) == color

    openings = []
    for side, values, offset, wall in (
        ("top", patch[0, 1:-1], c0 + 1, r0),
        ("bottom", patch[-1, 1:-1], c0 + 1, r1),
        ("left", patch[1:-1, 0], r0 + 1, c0),
        ("right", patch[1:-1, -1], r0 + 1, c1),
    ):
        assert set(map(int, values.tolist())) <= {bg, color}, "wall cells must be wall color or openings"
        for span in _bg_runs(values, offset, bg):
            openings.append(
                {
                    "room": int(room["room"]),
                    "bbox": room["bbox"],
                    "side": side,
                    "span": tuple(map(int, span)),
                    "wall": int(wall),
                }
            )
    return openings


def _extract_rooms_and_openings(grid: np.ndarray) -> tuple[int, int, list[dict], list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    colors = sorted(set(map(int, g.ravel())) - {bg})
    assert len(colors) == 1, "expected exactly one room color"
    color = int(colors[0])

    rooms = []
    openings = []
    for room_id, comp in enumerate(_find_components(g, color)):
        room = {
            "room": int(room_id),
            "bbox": comp["bbox"],
        }
        room_openings = _extract_room_openings(g, room, bg, color)
        rooms.append(room)
        for op in room_openings:
            op["id"] = len(openings)
            openings.append(op)
    assert rooms, "expected at least one room"
    return bg, color, rooms, openings


def _pair_signature(grid: np.ndarray, bg: int, a: dict, b: dict) -> tuple[str, dict, dict] | None:
    sides = {a["side"], b["side"]}
    if sides == {"right", "left"}:
        if a["side"] == "right":
            left_room, right_room = a, b
        else:
            left_room, right_room = b, a
        if left_room["wall"] >= right_room["wall"]:
            return None
        if tuple(left_room["span"]) != tuple(right_room["span"]):
            return None
        r0, r1 = left_room["span"]
        if np.all(grid[r0 : r1 + 1, left_room["wall"] : right_room["wall"] + 1] == bg):
            return "h", left_room, right_room
        return None

    if sides == {"bottom", "top"}:
        if a["side"] == "bottom":
            upper_room, lower_room = a, b
        else:
            upper_room, lower_room = b, a
        if upper_room["wall"] >= lower_room["wall"]:
            return None
        if tuple(upper_room["span"]) != tuple(lower_room["span"]):
            return None
        c0, c1 = upper_room["span"]
        if np.all(grid[upper_room["wall"] : lower_room["wall"] + 1, c0 : c1 + 1] == bg):
            return "v", upper_room, lower_room
        return None

    return None


def _pair_geometry(a: dict, b: dict) -> tuple[str, dict, dict]:
    sides = {a["side"], b["side"]}
    if sides == {"right", "left"}:
        if a["side"] == "right":
            first, second = a, b
        else:
            first, second = b, a
        assert first["wall"] < second["wall"]
        assert tuple(first["span"]) == tuple(second["span"])
        return "h", first, second
    assert sides == {"bottom", "top"}
    if a["side"] == "bottom":
        first, second = a, b
    else:
        first, second = b, a
    assert first["wall"] < second["wall"]
    assert tuple(first["span"]) == tuple(second["span"])
    return "v", first, second


def _find_pairs(grid: np.ndarray, bg: int, openings: list[dict]) -> list[tuple[int, int]]:
    pairs = []
    degree = Counter()
    for i in range(len(openings)):
        for j in range(i + 1, len(openings)):
            if _pair_signature(grid, bg, openings[i], openings[j]) is None:
                continue
            degree[i] += 1
            degree[j] += 1
            pairs.append((i, j))
    assert all(v <= 1 for v in degree.values()), "each opening may have at most one partner"
    return pairs


def _seal_opening(out: np.ndarray, op: dict, color: int) -> None:
    rr, cc = _opening_cells(op)
    out[rr, cc] = int(color)


def _draw_corridor(out: np.ndarray, color: int, a: dict, b: dict) -> None:
    kind, first, second = _pair_geometry(a, b)
    if kind == "h":
        r0, r1 = first["span"]
        top = r0 - 1
        bottom = r1 + 1
        c0 = first["wall"]
        c1 = second["wall"]
        out[top, c0 : c1 + 1] = int(color)
        out[bottom, c0 : c1 + 1] = int(color)
    else:
        c0, c1 = first["span"]
        left = c0 - 1
        right = c1 + 1
        r0 = first["wall"]
        r1 = second["wall"]
        out[r0 : r1 + 1, left] = int(color)
        out[r0 : r1 + 1, right] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect only exactly aligned opposite wall openings with corridors, and seal every unmatched opening.

    Input structure:
    - A dominant background surrounds several same-colored hollow rectangular rooms with one-cell-thick walls.
    - Some room walls contain continuous missing segments that act as openings.
    - Openings can face each other horizontally or vertically across clear background.

    Rule:
    - Detect every wall opening as a contiguous missing border segment of a room.
    - Pair two openings only when they face opposite directions, occupy exactly the same row span or column span, and every cell between them is background.
    - For each matched pair, draw a corridor by extending the two wall lines immediately outside the opening span across the gap; keep the corridor interior background.
    - Any opening without a partner is filled back in with the wall color.

    Key insight:
    - Pairing depends on exact interval alignment plus an unobstructed background gap, not on nearest distance; every unpaired opening must be closed rather than left open.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg, color, _rooms, openings = _extract_rooms_and_openings(g)
    pairs = _find_pairs(g, bg, openings)

    out = g.copy()
    paired_ids = {i for pair in pairs for i in pair}
    for op in openings:
        if op["id"] not in paired_ids:
            _seal_opening(out, op, color)
    for i, j in pairs:
        _draw_corridor(out, color, openings[i], openings[j])
    return out


# --------------- generator ---------------

GRID_SIZE = 30
ROOM_HEIGHT_RANGE = (4, 6)
ROOM_WIDTH_RANGE = (4, 6)
PAIR_GAP_RANGE = (2, 5)
BLOCKED_GAP_RANGE = (5, 6)
TOP_OFFSET_RANGE = (0, 2)
FEATURE_GAP = 1
EXTRA_CLOSED_ROOMS = (0, 2)
GENERATE_RETRIES = 400


def _room_box(r0: int, c0: int, h: int, w: int) -> dict:
    return {"bbox": (int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1)), "openings": []}


def _shift_rooms(rooms: list[dict], dr: int, dc: int) -> list[dict]:
    shifted = []
    for room in rooms:
        r0, c0, r1, c1 = room["bbox"]
        new_room = {
            "bbox": (r0 + dr, c0 + dc, r1 + dr, c1 + dc),
            "openings": [],
        }
        for op in room["openings"]:
            a0, a1 = op["span"]
            if op["side"] in ("left", "right"):
                span = (a0 + dr, a1 + dr)
            else:
                span = (a0 + dc, a1 + dc)
            new_room["openings"].append({"side": op["side"], "span": span, "key": op["key"]})
        shifted.append(new_room)
    return shifted


def _feature_bbox(rooms: list[dict]) -> BBox:
    r0 = min(room["bbox"][0] for room in rooms)
    c0 = min(room["bbox"][1] for room in rooms)
    r1 = max(room["bbox"][2] for room in rooms)
    c1 = max(room["bbox"][3] for room in rooms)
    return int(r0), int(c0), int(r1), int(c1)


def _flatten_spec_openings(rooms: list[dict]) -> list[dict]:
    openings = []
    for room_idx, room in enumerate(rooms):
        r0, c0, r1, c1 = room["bbox"]
        for op in room["openings"]:
            side = op["side"]
            if side == "left":
                wall = c0
            elif side == "right":
                wall = c1
            elif side == "top":
                wall = r0
            else:
                wall = r1
            openings.append(
                {
                    "id": len(openings),
                    "room": int(room_idx),
                    "bbox": room["bbox"],
                    "side": side,
                    "span": tuple(map(int, op["span"])),
                    "wall": int(wall),
                    "key": op["key"],
                }
            )
    return openings


def _draw_room(grid: np.ndarray, bbox: BBox, color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = int(color)
    grid[r1, c0 : c1 + 1] = int(color)
    grid[r0 : r1 + 1, c0] = int(color)
    grid[r0 : r1 + 1, c1] = int(color)


def _erase_opening(grid: np.ndarray, bbox: BBox, op: dict, bg: int) -> None:
    a0, a1 = op["span"]
    r0, c0, r1, c1 = bbox
    side = op["side"]
    if side == "left":
        grid[a0 : a1 + 1, c0] = int(bg)
    elif side == "right":
        grid[a0 : a1 + 1, c1] = int(bg)
    elif side == "top":
        grid[r0, a0 : a1 + 1] = int(bg)
    else:
        grid[r1, a0 : a1 + 1] = int(bg)


def _render_input_from_specs(rooms: list[dict], shape: tuple[int, int], bg: int, color: int) -> np.ndarray:
    grid = np.full(shape, int(bg), dtype=int)
    for room in rooms:
        _draw_room(grid, room["bbox"], color)
    for room in rooms:
        for op in room["openings"]:
            _erase_opening(grid, room["bbox"], op, bg)
    return grid


def _intended_pairs_from_specs(rooms: list[dict]) -> set[frozenset[int]]:
    by_key: dict[int, list[int]] = {}
    for op in _flatten_spec_openings(rooms):
        if op["key"] is None:
            continue
        by_key.setdefault(int(op["key"]), []).append(int(op["id"]))
    pairs = set()
    for ids in by_key.values():
        assert len(ids) == 2, "each matched key must appear exactly twice"
        pairs.add(frozenset(ids))
    return pairs


def _actual_pairs_from_specs(rooms: list[dict], grid: np.ndarray, bg: int) -> set[frozenset[int]]:
    openings = _flatten_spec_openings(rooms)
    pairs = set()
    for i in range(len(openings)):
        for j in range(i + 1, len(openings)):
            if _pair_signature(grid, bg, openings[i], openings[j]) is not None:
                pairs.add(frozenset((i, j)))
    return pairs


def _render_output_from_specs(rooms: list[dict], shape: tuple[int, int], bg: int, color: int) -> np.ndarray:
    inp = _render_input_from_specs(rooms, shape, bg, color)
    out = inp.copy()
    openings = _flatten_spec_openings(rooms)
    intended = _intended_pairs_from_specs(rooms)
    paired_ids = {idx for pair in intended for idx in pair}
    for op in openings:
        if op["id"] not in paired_ids:
            _seal_opening(out, op, color)
    for pair in intended:
        i, j = sorted(pair)
        _draw_corridor(out, color, openings[i], openings[j])
    return out


def _bbox_too_close(a: BBox, b: BBox, gap: int) -> bool:
    return not (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _sample_room_size() -> tuple[int, int]:
    return random.randint(*ROOM_HEIGHT_RANGE), random.randint(*ROOM_WIDTH_RANGE)


def _sample_common_span(off_a: int, size_a: int, off_b: int, size_b: int) -> tuple[int, int] | None:
    lo = max(off_a + 1, off_b + 1)
    hi = min(off_a + size_a - 2, off_b + size_b - 2)
    if lo > hi:
        return None
    length = random.randint(1, hi - lo + 1)
    start = random.randint(lo, hi - length + 1)
    return int(start), int(start + length - 1)


def _sample_mismatched_span(base: tuple[int, int], lo: int, hi: int) -> tuple[int, int] | None:
    candidates = []
    for start in range(lo, hi + 1):
        for end in range(start, hi + 1):
            span = (int(start), int(end))
            if span == tuple(base):
                continue
            if not (end < base[0] or base[1] < start):
                candidates.append(span)
    if not candidates:
        return None
    return random.choice(candidates)


def _build_matched_horizontal(key: int) -> list[dict]:
    for _ in range(80):
        h1, w1 = _sample_room_size()
        h2, w2 = _sample_room_size()
        gap = random.randint(*PAIR_GAP_RANGE)
        top2 = random.randint(*TOP_OFFSET_RANGE)
        span = _sample_common_span(0, h1, top2, h2)
        if span is None:
            continue
        left = _room_box(0, 0, h1, w1)
        right = _room_box(top2, w1 + gap, h2, w2)
        left["openings"].append({"side": "right", "span": span, "key": key})
        right["openings"].append({"side": "left", "span": span, "key": key})
        return [left, right]
    raise RuntimeError("failed to build matched horizontal feature")


def _build_matched_vertical(key: int) -> list[dict]:
    for _ in range(80):
        h1, w1 = _sample_room_size()
        h2, w2 = _sample_room_size()
        gap = random.randint(*PAIR_GAP_RANGE)
        left2 = random.randint(*TOP_OFFSET_RANGE)
        span = _sample_common_span(0, w1, left2, w2)
        if span is None:
            continue
        upper = _room_box(0, 0, h1, w1)
        lower = _room_box(h1 + gap, left2, h2, w2)
        upper["openings"].append({"side": "bottom", "span": span, "key": key})
        lower["openings"].append({"side": "top", "span": span, "key": key})
        return [upper, lower]
    raise RuntimeError("failed to build matched vertical feature")


def _build_mismatch_horizontal() -> list[dict]:
    for _ in range(100):
        h1, w1 = _sample_room_size()
        h2, w2 = _sample_room_size()
        gap = random.randint(*PAIR_GAP_RANGE)
        top2 = random.randint(*TOP_OFFSET_RANGE)
        left_span = _sample_common_span(0, h1, top2, h2)
        if left_span is None:
            continue
        right_lo = top2 + 1
        right_hi = top2 + h2 - 2
        right_span = _sample_mismatched_span(left_span, right_lo, right_hi)
        if right_span is None:
            continue
        left = _room_box(0, 0, h1, w1)
        right = _room_box(top2, w1 + gap, h2, w2)
        left["openings"].append({"side": "right", "span": left_span, "key": None})
        right["openings"].append({"side": "left", "span": right_span, "key": None})
        return [left, right]
    raise RuntimeError("failed to build mismatch horizontal feature")


def _build_mismatch_vertical() -> list[dict]:
    for _ in range(100):
        h1, w1 = _sample_room_size()
        h2, w2 = _sample_room_size()
        gap = random.randint(*PAIR_GAP_RANGE)
        left2 = random.randint(*TOP_OFFSET_RANGE)
        upper_span = _sample_common_span(0, w1, left2, w2)
        if upper_span is None:
            continue
        lower_lo = left2 + 1
        lower_hi = left2 + w2 - 2
        lower_span = _sample_mismatched_span(upper_span, lower_lo, lower_hi)
        if lower_span is None:
            continue
        upper = _room_box(0, 0, h1, w1)
        lower = _room_box(h1 + gap, left2, h2, w2)
        upper["openings"].append({"side": "bottom", "span": upper_span, "key": None})
        lower["openings"].append({"side": "top", "span": lower_span, "key": None})
        return [upper, lower]
    raise RuntimeError("failed to build mismatch vertical feature")


def _build_blocked_horizontal() -> list[dict]:
    for _ in range(100):
        h1, w1 = _sample_room_size()
        h2, w2 = _sample_room_size()
        gap = random.randint(*BLOCKED_GAP_RANGE)
        top2 = random.randint(*TOP_OFFSET_RANGE)
        span = _sample_common_span(0, h1, top2, h2)
        if span is None:
            continue
        left = _room_box(0, 0, h1, w1)
        right = _room_box(top2, w1 + gap, h2, w2)
        blocker_w = 3
        blocker_h = 3
        blocker_c0 = w1 + 1
        if blocker_c0 + blocker_w >= w1 + gap:
            continue
        blocker_r0 = max(0, min(span[0] - 1, max(h1 - blocker_h, top2 + h2 - blocker_h)))
        blocker = _room_box(blocker_r0, blocker_c0, blocker_h, blocker_w)
        left["openings"].append({"side": "right", "span": span, "key": None})
        right["openings"].append({"side": "left", "span": span, "key": None})
        return [left, right, blocker]
    raise RuntimeError("failed to build blocked horizontal feature")


def _build_blocked_vertical() -> list[dict]:
    for _ in range(100):
        h1, w1 = _sample_room_size()
        h2, w2 = _sample_room_size()
        gap = random.randint(*BLOCKED_GAP_RANGE)
        left2 = random.randint(*TOP_OFFSET_RANGE)
        span = _sample_common_span(0, w1, left2, w2)
        if span is None:
            continue
        upper = _room_box(0, 0, h1, w1)
        lower = _room_box(h1 + gap, left2, h2, w2)
        blocker_w = 3
        blocker_h = 3
        blocker_r0 = h1 + 1
        if blocker_r0 + blocker_h >= h1 + gap:
            continue
        blocker_c0 = max(0, min(span[0] - 1, max(w1 - blocker_w, left2 + w2 - blocker_w)))
        blocker = _room_box(blocker_r0, blocker_c0, blocker_h, blocker_w)
        upper["openings"].append({"side": "bottom", "span": span, "key": None})
        lower["openings"].append({"side": "top", "span": span, "key": None})
        return [upper, lower, blocker]
    raise RuntimeError("failed to build blocked vertical feature")


def _build_closed_room() -> list[dict]:
    h, w = _sample_room_size()
    return [_room_box(0, 0, h, w)]


def _place_features(features: list[list[dict]]) -> list[dict] | None:
    placed: list[list[dict]] = [None] * len(features)  # type: ignore[assignment]
    taken: list[BBox] = []
    order = sorted(range(len(features)), key=lambda i: (_feature_bbox(features[i])[2] + 1) * (_feature_bbox(features[i])[3] + 1), reverse=True)

    for idx in order:
        group = features[idx]
        _r0, _c0, r1, c1 = _feature_bbox(group)
        height = r1 + 1
        width = c1 + 1
        candidates = []
        for top in range(GRID_SIZE - height + 1):
            for left in range(GRID_SIZE - width + 1):
                bb = (top, left, top + height - 1, left + width - 1)
                if any(_bbox_too_close(bb, other, FEATURE_GAP) for other in taken):
                    continue
                candidates.append((top, left, bb))
        if not candidates:
            return None
        top, left, bb = random.choice(candidates)
        placed[idx] = _shift_rooms(group, top, left)
        taken.append(bb)

    all_rooms = []
    for group in placed:
        assert group is not None
        all_rooms.extend(group)
    return all_rooms


def _non_bg_bbox(grid: np.ndarray, bg: int) -> BBox:
    coords = np.argwhere(grid != bg)
    assert coords.size > 0
    return _bbox_coords(coords)


def _random_crop_pair(inp: np.ndarray, out: np.ndarray, bg: int) -> tuple[np.ndarray, np.ndarray]:
    r0, c0, r1, c1 = _non_bg_bbox(inp, bg)
    top = random.randint(0, r0)
    left = random.randint(0, c0)
    bottom = random.randint(0, inp.shape[0] - 1 - r1)
    right = random.randint(0, inp.shape[1] - 1 - c1)
    rs = slice(r0 - top, r1 + bottom + 1)
    cs = slice(c0 - left, c1 + right + 1)
    return inp[rs, cs].copy(), out[rs, cs].copy()


def _transform_pair(inp: np.ndarray, out: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    transforms = (
        lambda x: x,
        lambda x: np.rot90(x, 1),
        lambda x: np.rot90(x, 2),
        lambda x: np.rot90(x, 3),
        lambda x: np.fliplr(x),
        lambda x: np.flipud(x),
        lambda x: x.T,
        lambda x: np.fliplr(x.T),
    )
    tf = random.choice(transforms)
    return np.asarray(tf(inp), dtype=int).copy(), np.asarray(tf(out), dtype=int).copy()


def _generate_case() -> tuple[np.ndarray, np.ndarray] | None:
    color = random.randint(1, 9)
    next_key = 0

    def take_key() -> int:
        nonlocal next_key
        key = next_key
        next_key += 1
        return key

    features = [
        _build_matched_horizontal(take_key()),
        _build_matched_vertical(take_key()),
        random.choice((_build_mismatch_horizontal, _build_blocked_horizontal))(),
        random.choice((_build_mismatch_vertical, _build_blocked_vertical))(),
    ]
    for _ in range(random.randint(*EXTRA_CLOSED_ROOMS)):
        features.append(_build_closed_room())
    random.shuffle(features)

    rooms = _place_features(features)
    if rooms is None:
        return None

    inp = _render_input_from_specs(rooms, (GRID_SIZE, GRID_SIZE), bg=0, color=color)
    if _actual_pairs_from_specs(rooms, inp, 0) != _intended_pairs_from_specs(rooms):
        return None

    out = _render_output_from_specs(rooms, inp.shape, bg=0, color=color)
    inp, out = _random_crop_pair(inp, out, 0)
    inp, out = _transform_pair(inp, out)

    if inp.ndim != 2 or out.ndim != 2 or max(inp.shape) > 30 or max(out.shape) > 30:
        return None
    if np.array_equal(inp, out):
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        case = _generate_case()
        if case is not None:
            return case[0]
    raise RuntimeError("failed to generate a valid opening-corridor puzzle input")
