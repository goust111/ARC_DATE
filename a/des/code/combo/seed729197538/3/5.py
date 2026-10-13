import random
from collections import defaultdict

import numpy as np


BG = 0
PATTERN_KINDS = ("solid", "hstripe", "vstripe")

GRID_MIN = 18
GRID_MAX = 28
PAIR_ATTEMPTS = 120
ISOLATED_ATTEMPTS = 120
GEN_ATTEMPTS = 300
PADDING = 1


def _clusters(values: list[int], max_gap: int) -> list[list[int]]:
    if not values:
        return []
    out: list[list[int]] = []
    cur = [int(values[0])]
    for value in values[1:]:
        value = int(value)
        if value - cur[-1] <= max_gap:
            cur.append(value)
        else:
            out.append(cur)
            cur = [value]
    out.append(cur)
    return out


def _row_segments(grid: np.ndarray, color: int) -> list[tuple[int, int, int]]:
    h, w = grid.shape
    segments: list[tuple[int, int, int]] = []
    for r in range(h):
        c = 0
        while c < w:
            if int(grid[r, c]) != color:
                c += 1
                continue
            c0 = c
            while c + 1 < w and int(grid[r, c + 1]) == color:
                c += 1
            segments.append((r, c0, c))
            c += 1
    return segments


def _col_segments(grid: np.ndarray, color: int) -> list[tuple[int, int, int]]:
    h, w = grid.shape
    segments: list[tuple[int, int, int]] = []
    for c in range(w):
        r = 0
        while r < h:
            if int(grid[r, c]) != color:
                r += 1
                continue
            r0 = r
            while r + 1 < h and int(grid[r + 1, c]) == color:
                r += 1
            segments.append((c, r0, r))
            r += 1
    return segments


def _detect_pattern(
    grid: np.ndarray, bbox: tuple[int, int, int, int], color: int
) -> tuple[str, int] | None:
    r0, c0, r1, c1 = bbox
    sub = np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int)
    h, w = sub.shape

    if np.all(sub == color):
        return ("solid", 0)

    for phase in (0, 1):
        expected = np.zeros((h, w), dtype=int)
        expected[np.arange(h) % 2 == phase, :] = color
        if np.array_equal(sub, expected):
            return ("hstripe", phase)

    for phase in (0, 1):
        expected = np.zeros((h, w), dtype=int)
        expected[:, np.arange(w) % 2 == phase] = color
        if np.array_equal(sub, expected):
            return ("vstripe", phase)

    return None


def _draw_pattern(
    grid: np.ndarray, bbox: tuple[int, int, int, int], color: int, pattern: tuple[str, int]
) -> None:
    r0, c0, r1, c1 = bbox
    kind, phase = pattern
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    sub[:] = BG

    if kind == "solid":
        sub[:] = color
        return

    if kind == "hstripe":
        rows = np.arange(sub.shape[0]) % 2 == phase
        sub[rows, :] = color
        return

    if kind == "vstripe":
        cols = np.arange(sub.shape[1]) % 2 == phase
        sub[:, cols] = color
        return

    raise ValueError(f"unknown pattern: {kind}")


def _find_rectangles_for_color(grid: np.ndarray, color: int) -> list[dict]:
    candidates: dict[tuple[int, int, int, int], tuple[str, int]] = {}

    row_groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for r, c0, c1 in _row_segments(grid, color):
        if c1 - c0 + 1 >= 2:
            row_groups[(c0, c1)].append(r)

    for (c0, c1), rows in row_groups.items():
        rows.sort()
        for cluster in _clusters(rows, max_gap=4):
            if len(cluster) < 2:
                continue
            bbox = (cluster[0], c0, cluster[-1], c1)
            pattern = _detect_pattern(grid, bbox, color)
            if pattern is not None:
                candidates.setdefault(bbox, pattern)

    col_groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for c, r0, r1 in _col_segments(grid, color):
        if r1 - r0 + 1 >= 2:
            col_groups[(r0, r1)].append(c)

    for (r0, r1), cols in col_groups.items():
        cols.sort()
        for cluster in _clusters(cols, max_gap=4):
            if len(cluster) < 2:
                continue
            bbox = (r0, cluster[0], r1, cluster[-1])
            pattern = _detect_pattern(grid, bbox, color)
            if pattern is not None:
                candidates.setdefault(bbox, pattern)

    rects = [
        {"color": color, "bbox": bbox, "pattern": pattern}
        for bbox, pattern in candidates.items()
    ]
    rects.sort(
        key=lambda rect: (
            rect["bbox"][0],
            rect["bbox"][1],
            rect["bbox"][2] - rect["bbox"][0],
            rect["bbox"][3] - rect["bbox"][1],
        )
    )
    return rects


def _find_bridge(grid: np.ndarray, rect_a: dict, rect_b: dict) -> dict | None:
    color = int(rect_a["color"])
    h, w = grid.shape
    a = rect_a["bbox"]
    b = rect_b["bbox"]

    if a[1] < b[1]:
        left, right = a, b
    else:
        left, right = b, a
    gap = right[1] - left[3] - 1
    if gap >= 1:
        rows = range(max(left[0], right[0]), min(left[2], right[2]) + 1)
        valid_rows: list[int] = []
        for r in rows:
            if int(grid[r, left[3]]) != color or int(grid[r, right[1]]) != color:
                continue
            ok = True
            for c in range(left[3] + 1, right[1]):
                if int(grid[r, c]) != color:
                    ok = False
                    break
                if (r > 0 and int(grid[r - 1, c]) == color) or (
                    r + 1 < h and int(grid[r + 1, c]) == color
                ):
                    ok = False
                    break
            if ok:
                valid_rows.append(r)
        if valid_rows:
            return {
                "orientation": "h",
                "line": valid_rows[0],
                "gap_cells": gap,
                "bridge_length": max(0, gap - 2),
            }

    if a[0] < b[0]:
        top, bottom = a, b
    else:
        top, bottom = b, a
    gap = bottom[0] - top[2] - 1
    if gap >= 1:
        cols = range(max(top[1], bottom[1]), min(top[3], bottom[3]) + 1)
        valid_cols: list[int] = []
        for c in cols:
            if int(grid[top[2], c]) != color or int(grid[bottom[0], c]) != color:
                continue
            ok = True
            for r in range(top[2] + 1, bottom[0]):
                if int(grid[r, c]) != color:
                    ok = False
                    break
                if (c > 0 and int(grid[r, c - 1]) == color) or (
                    c + 1 < w and int(grid[r, c + 1]) == color
                ):
                    ok = False
                    break
            if ok:
                valid_cols.append(c)
        if valid_cols:
            return {
                "orientation": "v",
                "line": valid_cols[0],
                "gap_cells": gap,
                "bridge_length": max(0, gap - 2),
            }

    return None


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    colors = [int(color) for color in np.unique(g) if int(color) != BG]
    pairs: list[tuple[dict, dict, dict]] = []

    for color in colors:
        rects = _find_rectangles_for_color(g, color)
        pair_candidates: list[tuple[int, int, dict]] = []
        for i in range(len(rects)):
            for j in range(i + 1, len(rects)):
                bridge = _find_bridge(g, rects[i], rects[j])
                if bridge is not None:
                    pair_candidates.append((i, j, bridge))

        used: set[int] = set()
        for i, j, bridge in pair_candidates:
            if i in used or j in used:
                continue
            used.add(i)
            used.add(j)
            pairs.append((rects[i], rects[j], bridge))

    for rect_a, rect_b, bridge in pairs:
        color = int(rect_a["color"])
        bbox_a = rect_a["bbox"]
        bbox_b = rect_b["bbox"]

        if int(bridge["bridge_length"]) <= 2:
            r0 = min(bbox_a[0], bbox_b[0])
            c0 = min(bbox_a[1], bbox_b[1])
            r1 = max(bbox_a[2], bbox_b[2])
            c1 = max(bbox_a[3], bbox_b[3])
            out[r0 : r1 + 1, c0 : c1 + 1] = color
            continue

        pattern_a = rect_a["pattern"]
        pattern_b = rect_b["pattern"]
        _draw_pattern(out, bbox_a, color, pattern_b)
        _draw_pattern(out, bbox_b, color, pattern_a)

    return out


def _random_pattern(rng: random.Random) -> tuple[str, int]:
    kind = rng.choice(PATTERN_KINDS)
    if kind == "solid":
        return ("solid", 0)
    return (kind, rng.randrange(2))


def _pair_pattern(
    kind: str,
    orientation: str,
    side: str,
    shape: tuple[int, int],
    bridge_offset: int,
) -> tuple[str, int]:
    height, width = shape
    if kind == "solid":
        return ("solid", 0)
    if orientation == "h":
        if kind == "hstripe":
            return (kind, bridge_offset % 2)
        if side == "left":
            return (kind, (width - 1) % 2)
        return (kind, 0)
    if kind == "vstripe":
        return (kind, bridge_offset % 2)
    if side == "top":
        return (kind, (height - 1) % 2)
    return (kind, 0)


def _can_place(
    reserved: np.ndarray, bbox: tuple[int, int, int, int], pad: int = PADDING
) -> bool:
    h, w = reserved.shape
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    return not bool(np.any(reserved[rr0 : rr1 + 1, cc0 : cc1 + 1]))


def _reserve(reserved: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = bbox
    reserved[r0 : r1 + 1, c0 : c1 + 1] = True


def _sample_pair_kinds(rng: random.Random, branch: str) -> tuple[str, str]:
    kind_a = rng.choice(PATTERN_KINDS)
    kind_b = rng.choice(PATTERN_KINDS)
    if branch == "long":
        while kind_b == kind_a:
            kind_b = rng.choice(PATTERN_KINDS)
    return kind_a, kind_b


def _place_pair(
    rng: random.Random,
    grid: np.ndarray,
    reserved: np.ndarray,
    color: int,
    branch: str,
) -> dict | None:
    h, w = grid.shape

    for _ in range(PAIR_ATTEMPTS):
        orientation = rng.choice(("h", "v"))
        kind_a, kind_b = _sample_pair_kinds(rng, branch)

        if orientation == "h":
            height = rng.randint(5, 8)
            width_a = rng.randint(3, 7)
            width_b = rng.randint(3, 7)
            gap = rng.randint(1, 4) if branch == "short" else rng.randint(5, 8)
            union_h = height
            union_w = width_a + gap + width_b
            if union_h > h or union_w > w:
                continue

            top = rng.randint(0, h - union_h)
            left = rng.randint(0, w - union_w)
            union_bbox = (top, left, top + union_h - 1, left + union_w - 1)
            if not _can_place(reserved, union_bbox):
                continue

            bridge_row = rng.randint(1, height - 2)
            bbox_a = (top, left, top + height - 1, left + width_a - 1)
            bbox_b = (
                top,
                left + width_a + gap,
                top + height - 1,
                left + width_a + gap + width_b - 1,
            )
            pattern_a = _pair_pattern(
                kind_a, "h", "left", (height, width_a), bridge_row
            )
            pattern_b = _pair_pattern(
                kind_b, "h", "right", (height, width_b), bridge_row
            )
            _draw_pattern(grid, bbox_a, color, pattern_a)
            _draw_pattern(grid, bbox_b, color, pattern_b)
            row = top + bridge_row
            c0 = left + width_a
            c1 = left + width_a + gap - 1
            grid[row, c0 : c1 + 1] = color
            _reserve(reserved, union_bbox)
            return {
                "type": "pair",
                "branch": branch,
                "color": color,
                "union_bbox": union_bbox,
                "bridge": {"orientation": "h", "line": row, "start": c0, "stop": c1},
                "rects": [
                    {"bbox": bbox_a, "pattern": pattern_a},
                    {"bbox": bbox_b, "pattern": pattern_b},
                ],
            }

        shared_width = rng.randint(5, 8)
        height_a = rng.randint(3, 7)
        height_b = rng.randint(3, 7)
        gap = rng.randint(1, 4) if branch == "short" else rng.randint(5, 8)
        union_h = height_a + gap + height_b
        union_w = shared_width
        if union_h > h or union_w > w:
            continue

        top = rng.randint(0, h - union_h)
        left = rng.randint(0, w - union_w)
        union_bbox = (top, left, top + union_h - 1, left + union_w - 1)
        if not _can_place(reserved, union_bbox):
            continue

        bridge_col = rng.randint(1, shared_width - 2)
        bbox_a = (top, left, top + height_a - 1, left + shared_width - 1)
        bbox_b = (
            top + height_a + gap,
            left,
            top + height_a + gap + height_b - 1,
            left + shared_width - 1,
        )
        pattern_a = _pair_pattern(
            kind_a, "v", "top", (height_a, shared_width), bridge_col
        )
        pattern_b = _pair_pattern(
            kind_b, "v", "bottom", (height_b, shared_width), bridge_col
        )
        _draw_pattern(grid, bbox_a, color, pattern_a)
        _draw_pattern(grid, bbox_b, color, pattern_b)
        col = left + bridge_col
        r0 = top + height_a
        r1 = top + height_a + gap - 1
        grid[r0 : r1 + 1, col] = color
        _reserve(reserved, union_bbox)
        return {
            "type": "pair",
            "branch": branch,
            "color": color,
            "union_bbox": union_bbox,
            "bridge": {"orientation": "v", "line": col, "start": r0, "stop": r1},
            "rects": [
                {"bbox": bbox_a, "pattern": pattern_a},
                {"bbox": bbox_b, "pattern": pattern_b},
            ],
        }

    return None


def _place_isolated(
    rng: random.Random, grid: np.ndarray, reserved: np.ndarray, color: int
) -> dict | None:
    h, w = grid.shape
    for _ in range(ISOLATED_ATTEMPTS):
        height = rng.randint(3, 7)
        width = rng.randint(3, 7)
        if height > h or width > w:
            continue

        top = rng.randint(0, h - height)
        left = rng.randint(0, w - width)
        bbox = (top, left, top + height - 1, left + width - 1)
        if not _can_place(reserved, bbox):
            continue

        pattern = _random_pattern(rng)
        _draw_pattern(grid, bbox, color, pattern)
        _reserve(reserved, bbox)
        return {
            "type": "isolated",
            "color": color,
            "rects": [{"bbox": bbox, "pattern": pattern}],
            "union_bbox": bbox,
        }
    return None


def _expected_output(grid: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = grid.copy()
    for spec in specs:
        if spec["type"] != "pair":
            continue

        rect_a, rect_b = spec["rects"]
        bbox_a = rect_a["bbox"]
        bbox_b = rect_b["bbox"]
        color = int(spec["color"])
        if spec["branch"] == "short":
            r0 = min(bbox_a[0], bbox_b[0])
            c0 = min(bbox_a[1], bbox_b[1])
            r1 = max(bbox_a[2], bbox_b[2])
            c1 = max(bbox_a[3], bbox_b[3])
            out[r0 : r1 + 1, c0 : c1 + 1] = color
            continue

        _draw_pattern(out, bbox_a, color, rect_b["pattern"])
        _draw_pattern(out, bbox_b, color, rect_a["pattern"])

    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(GRID_MIN, GRID_MAX)
    w = rng.randint(GRID_MIN, GRID_MAX)
    grid = np.zeros((h, w), dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    structure_kinds = ["short", "long"] + ["isolated"] * rng.randint(0, 3)
    rng.shuffle(structure_kinds)
    colors = rng.sample(list(range(1, 10)), len(structure_kinds))

    specs: list[dict] = []
    for color, kind in zip(colors, structure_kinds):
        if kind in {"short", "long"}:
            spec = _place_pair(rng, grid, reserved, color, kind)
        else:
            spec = _place_isolated(rng, grid, reserved, color)
        if spec is None:
            return None
        specs.append(spec)

    expected = _expected_output(grid, specs)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None

    if np.array_equal(expected, grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
