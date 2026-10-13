import random

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))

GRID_H_RANGE = (11, 24)
GRID_W_RANGE = (11, 24)
PAIR_COUNT_SMALL = (1, 2)
PAIR_COUNT_MEDIUM = (1, 3)
PAIR_COUNT_LARGE = (2, 4)
LINE_LENGTH_RANGE = (2, 8)
INTERIOR_SPAN_RANGE = (1, 6)
OUTER_MARGIN = 1
PAIR_PLACEMENT_TRIES = 120
GEN_RETRIES = 300


def _components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append({"color": color, "coords": coords})
    return comps


def _extract_segments(grid: np.ndarray) -> list[dict]:
    segments: list[dict] = []
    for comp in _components(grid):
        rows = [r for r, _ in comp["coords"]]
        cols = [c for _, c in comp["coords"]]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        area = len(comp["coords"])
        if r0 == r1:
            length = c1 - c0 + 1
            assert length >= 2 and area == length, "expected horizontal line segments"
            orientation = "h"
        else:
            assert c0 == c1, "expected width-1 vertical line segments"
            length = r1 - r0 + 1
            assert length >= 2 and area == length, "expected vertical line segments"
            orientation = "v"
        segments.append(
            {
                "color": int(comp["color"]),
                "orientation": orientation,
                "r0": int(r0),
                "r1": int(r1),
                "c0": int(c0),
                "c1": int(c1),
                "length": int(length),
            }
        )
    return segments


def _pair_candidate(a: dict, b: dict, grid: np.ndarray) -> dict | None:
    if a["orientation"] != b["orientation"]:
        return None
    if a["length"] != b["length"]:
        return None
    if a["color"] == b["color"]:
        return None

    if a["orientation"] == "h":
        if a["c0"] != b["c0"] or a["c1"] != b["c1"]:
            return None
        top, bottom = (a, b) if a["r0"] < b["r0"] else (b, a)
        if bottom["r0"] - top["r0"] < 2:
            return None
        interior = grid[top["r0"] + 1 : bottom["r0"], top["c0"] : top["c1"] + 1]
        if not np.all(interior == BG):
            return None
    else:
        if a["r0"] != b["r0"] or a["r1"] != b["r1"]:
            return None
        left, right = (a, b) if a["c0"] < b["c0"] else (b, a)
        if right["c0"] - left["c0"] < 2:
            return None
        interior = grid[left["r0"] : left["r1"] + 1, left["c0"] + 1 : right["c0"]]
        if not np.all(interior == BG):
            return None
        top, bottom = left, right

    return {"orientation": a["orientation"], "first": top, "second": bottom}


def _match_pairs(grid: np.ndarray) -> list[dict]:
    segments = _extract_segments(grid)
    assert segments and len(segments) % 2 == 0, "expected an even number of line segments"

    n = len(segments)
    pair_info: dict[tuple[int, int], dict] = {}
    neighbors: dict[int, list[int]] = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            info = _pair_candidate(segments[i], segments[j], grid)
            if info is None:
                continue
            pair_info[(i, j)] = info
            neighbors[i].append(j)
            neighbors[j].append(i)

    used = [False] * n
    chosen: list[tuple[int, int]] = []

    def search() -> bool:
        remaining = [i for i in range(n) if not used[i]]
        if not remaining:
            return True
        pivot = min(remaining, key=lambda idx: sum(1 for j in neighbors[idx] if not used[j]))
        options = sorted(j for j in neighbors[pivot] if not used[j])
        if not options:
            return False

        used[pivot] = True
        for partner in options:
            used[partner] = True
            chosen.append((min(pivot, partner), max(pivot, partner)))
            if search():
                return True
            chosen.pop()
            used[partner] = False
        used[pivot] = False
        return False

    assert search(), "failed to pair all segments"
    return [pair_info[key] for key in chosen]


def _fill_pair(out: np.ndarray, pair: dict) -> None:
    if pair["orientation"] == "h":
        top = pair["first"]
        bottom = pair["second"]
        start_color = int(top["color"])
        other_color = int(bottom["color"])
        base_r = int(top["r0"] + 1)
        base_c = int(top["c0"])
        for r in range(base_r, int(bottom["r0"])):
            for c in range(base_c, int(top["c1"]) + 1):
                out[r, c] = start_color if (r - base_r + c - base_c) % 2 == 0 else other_color
    else:
        left = pair["first"]
        right = pair["second"]
        start_color = int(left["color"])
        other_color = int(right["color"])
        base_r = int(left["r0"])
        base_c = int(left["c0"] + 1)
        for r in range(base_r, int(left["r1"]) + 1):
            for c in range(base_c, int(right["c0"])):
                out[r, c] = start_color if (r - base_r + c - base_c) % 2 == 0 else other_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair equal-length parallel line segments and fill the rectangle between them
    with a two-color checkerboard whose phase is anchored by the upper/left line.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    for pair in _match_pairs(g):
        _fill_pair(out, pair)
    return out


def _pair_count_for_shape(h: int, w: int) -> int:
    area = h * w
    if area < 180:
        return random.randint(*PAIR_COUNT_SMALL)
    if area < 320:
        return random.randint(*PAIR_COUNT_MEDIUM)
    return random.randint(*PAIR_COUNT_LARGE)


def _fits(occupied: np.ndarray, r0: int, r1: int, c0: int, c1: int) -> bool:
    h, w = occupied.shape
    rr0 = max(0, r0 - 1)
    rr1 = min(h - 1, r1 + 1)
    cc0 = max(0, c0 - 1)
    cc1 = min(w - 1, c1 + 1)
    return not bool(occupied[rr0 : rr1 + 1, cc0 : cc1 + 1].any())


def _reserve(occupied: np.ndarray, r0: int, r1: int, c0: int, c1: int) -> None:
    h, w = occupied.shape
    rr0 = max(0, r0 - 1)
    rr1 = min(h - 1, r1 + 1)
    cc0 = max(0, c0 - 1)
    cc1 = min(w - 1, c1 + 1)
    occupied[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _draw_pair(
    grid: np.ndarray,
    occupied: np.ndarray,
    orientation: str,
    colors: tuple[int, int],
    used_h_spans: set[tuple[int, int]],
    used_v_spans: set[tuple[int, int]],
) -> bool:
    h, w = grid.shape
    for _ in range(PAIR_PLACEMENT_TRIES):
        if orientation == "h":
            max_len = min(LINE_LENGTH_RANGE[1], w - 2 * OUTER_MARGIN)
            if max_len < LINE_LENGTH_RANGE[0]:
                return False
            length = random.randint(LINE_LENGTH_RANGE[0], max_len)
            max_inner = min(INTERIOR_SPAN_RANGE[1], h - 2 * OUTER_MARGIN - 2)
            if max_inner < INTERIOR_SPAN_RANGE[0]:
                return False
            inner = random.randint(INTERIOR_SPAN_RANGE[0], max_inner)
            bbox_h = inner + 2
            bbox_w = length
            if bbox_h + 2 * OUTER_MARGIN > h or bbox_w + 2 * OUTER_MARGIN > w:
                continue
            r0 = random.randint(OUTER_MARGIN, h - OUTER_MARGIN - bbox_h)
            c0 = random.randint(OUTER_MARGIN, w - OUTER_MARGIN - bbox_w)
            r1 = r0 + bbox_h - 1
            c1 = c0 + bbox_w - 1
            span = (c0, c1)
            if span in used_h_spans or not _fits(occupied, r0, r1, c0, c1):
                continue
            top_color, bottom_color = colors
            grid[r0, c0 : c1 + 1] = top_color
            grid[r1, c0 : c1 + 1] = bottom_color
            _reserve(occupied, r0, r1, c0, c1)
            used_h_spans.add(span)
            return True

        max_len = min(LINE_LENGTH_RANGE[1], h - 2 * OUTER_MARGIN)
        if max_len < LINE_LENGTH_RANGE[0]:
            return False
        length = random.randint(LINE_LENGTH_RANGE[0], max_len)
        max_inner = min(INTERIOR_SPAN_RANGE[1], w - 2 * OUTER_MARGIN - 2)
        if max_inner < INTERIOR_SPAN_RANGE[0]:
            return False
        inner = random.randint(INTERIOR_SPAN_RANGE[0], max_inner)
        bbox_h = length
        bbox_w = inner + 2
        if bbox_h + 2 * OUTER_MARGIN > h or bbox_w + 2 * OUTER_MARGIN > w:
            continue
        r0 = random.randint(OUTER_MARGIN, h - OUTER_MARGIN - bbox_h)
        c0 = random.randint(OUTER_MARGIN, w - OUTER_MARGIN - bbox_w)
        r1 = r0 + bbox_h - 1
        c1 = c0 + bbox_w - 1
        span = (r0, r1)
        if span in used_v_spans or not _fits(occupied, r0, r1, c0, c1):
            continue
        left_color, right_color = colors
        grid[r0 : r1 + 1, c0] = left_color
        grid[r0 : r1 + 1, c1] = right_color
        _reserve(occupied, r0, r1, c0, c1)
        used_v_spans.add(span)
        return True
    return False


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    pair_count = _pair_count_for_shape(h, w)
    if 2 * pair_count > len(PALETTE):
        return None

    grid = np.zeros((h, w), dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    colors = random.sample(PALETTE, 2 * pair_count)
    used_h_spans: set[tuple[int, int]] = set()
    used_v_spans: set[tuple[int, int]] = set()

    for i in range(pair_count):
        orientation = random.choice(("h", "v"))
        pair_colors = (int(colors[2 * i]), int(colors[2 * i + 1]))
        if not _draw_pair(grid, occupied, orientation, pair_colors, used_h_spans, used_v_spans):
            return None

    pairs = _match_pairs(grid)
    if len(pairs) != pair_count:
        return None
    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
