from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


TEAL = 8
BACKGROUND = 7
RECT_COLORS = [c for c in range(10) if c not in (BACKGROUND, TEAL)]
SHIFT_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
HEIGHT_RANGE = range(9, 15)
WIDTH_RANGE = range(9, 15)
RECT_COUNT_RANGE = (2, 5)
RECT_SIZE_RANGE = range(2, 6)
AREA_RATIO_RANGE = (0.18, 0.38)
GEN_ATTEMPTS = 400
PLACEMENT_BUDGET = 12000
GAP = 1


def get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def find_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def rectangle_moves(grid: np.ndarray, bg: int) -> list[tuple[int, int, int, int, int, int, int]]:
    g = np.asarray(grid, dtype=int)
    marker_coords = {tuple(map(int, p)) for p in np.argwhere(g == TEAL)}
    assert marker_coords, "expected at least one teal marker"

    moves: list[tuple[int, int, int, int, int, int, int]] = []
    used_markers: set[tuple[int, int]] = set()
    for comp in find_components(g != bg):
        colors = {int(g[r, c]) for r, c in comp}
        assert TEAL in colors and len(colors) == 2, "each object must contain one teal marker and one body color"
        color = next(c for c in colors if c != TEAL)
        rs, cs = comp[:, 0], comp[:, 1]
        r0, r1 = int(rs.min()), int(rs.max())
        c0, c1 = int(cs.min()), int(cs.max())
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        h, w = patch.shape
        teal_locs = np.argwhere(patch == TEAL)
        assert teal_locs.shape == (1, 2), "each rectangle needs exactly one teal corner"
        mr, mc = map(int, teal_locs[0])
        assert mr in (0, h - 1) and mc in (0, w - 1), "teal must sit on a rectangle corner"
        expected = np.full((h, w), color, dtype=int)
        expected[mr, mc] = TEAL
        assert np.array_equal(patch, expected), "object must be a filled rectangle with one replaced corner"

        dr = -1 if mr == 0 else 1
        dc = -1 if mc == 0 else 1
        tr0, tr1 = r0 + dr, r1 + dr
        tc0, tc1 = c0 + dc, c1 + dc
        H, W = g.shape
        assert 0 <= tr0 <= tr1 < H and 0 <= tc0 <= tc1 < W, "shifted rectangle must stay in bounds"
        used_markers.add((r0 + mr, c0 + mc))
        moves.append((r0, c0, r1, c1, color, dr, dc))

    assert used_markers == marker_coords, "every teal pixel must belong to exactly one rectangle"
    return moves


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    moves = rectangle_moves(g, bg)

    out = g.copy()
    for r0, c0, r1, c1, _, _, _ in moves:
        out[r0 : r1 + 1, c0 : c1 + 1] = bg

    occupied = np.zeros_like(g, dtype=bool)
    for r0, c0, r1, c1, color, dr, dc in moves:
        tr0, tr1 = r0 + dr, r1 + dr
        tc0, tc1 = c0 + dc, c1 + dc
        assert not occupied[tr0 : tr1 + 1, tc0 : tc1 + 1].any(), "shifted rectangles must not overlap"
        assert np.all(out[tr0 : tr1 + 1, tc0 : tc1 + 1] == bg), "shift targets must be background after erasure"
        out[tr0 : tr1 + 1, tc0 : tc1 + 1] = color
        occupied[tr0 : tr1 + 1, tc0 : tc1 + 1] = True

    return out


def _bbox_intersects(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _expand_bbox(b: tuple[int, int, int, int], h: int, w: int) -> tuple[int, int, int, int]:
    return (
        max(0, b[0] - GAP),
        max(0, b[1] - GAP),
        min(h - 1, b[2] + GAP),
        min(w - 1, b[3] + GAP),
    )


def _candidate_boxes(h: int, w: int, rh: int, rw: int, dr: int, dc: int) -> list[tuple[tuple[int, int, int, int], tuple[int, int, int, int]]]:
    r0_min = 0 if dr == -1 else 1
    r0_max = h - rh - 1 if dr == -1 else h - rh
    c0_min = 0 if dc == -1 else 1
    c0_max = w - rw - 1 if dc == -1 else w - rw
    boxes = []
    for out_r0 in range(r0_min, r0_max + 1):
        for out_c0 in range(c0_min, c0_max + 1):
            out_bbox = (out_r0, out_c0, out_r0 + rh - 1, out_c0 + rw - 1)
            in_bbox = (out_r0 - dr, out_c0 - dc, out_r0 - dr + rh - 1, out_c0 - dc + rw - 1)
            boxes.append((out_bbox, in_bbox))
    random.shuffle(boxes)
    return boxes


def _overlaps(candidate_out: tuple[int, int, int, int], candidate_in: tuple[int, int, int, int], placed: list[dict], h: int, w: int) -> bool:
    expanded = _expand_bbox(candidate_out, h, w)
    for other in placed:
        if _bbox_intersects(expanded, other["out_bbox"]):
            return True
        if _bbox_intersects(candidate_in, other["in_bbox"]):
            return True
    return False


def _choose_sizes(count: int, height: int, width: int) -> list[tuple[int, int]] | None:
    dims = [(rh, rw, rh * rw) for rh in RECT_SIZE_RANGE for rw in RECT_SIZE_RANGE]
    area = height * width
    lo = int(np.ceil(area * AREA_RATIO_RANGE[0]))
    hi = int(np.floor(area * AREA_RATIO_RANGE[1]))
    sums = [{0}]
    for _ in range(count):
        sums.append({prev + a for prev in sums[-1] for _, _, a in dims})
    targets = [s for s in sums[count] if lo <= s <= hi]
    if not targets:
        return None

    target = random.choice(targets)
    sizes: list[tuple[int, int]] = []
    for remaining in range(count, 0, -1):
        choices = [(rh, rw, a) for rh, rw, a in dims if target - a in sums[remaining - 1]]
        rh, rw, a = random.choice(choices)
        sizes.append((rh, rw))
        target -= a
    random.shuffle(sizes)
    return sizes


def _place_rectangles(rects: list[dict], height: int, width: int) -> list[dict] | None:
    ordered = sorted(rects, key=lambda rect: rect["h"] * rect["w"], reverse=True)
    placed: list[dict] = []
    budget = PLACEMENT_BUDGET

    def rec(idx: int) -> bool:
        nonlocal budget
        if idx == len(ordered):
            return True
        if budget <= 0:
            return False
        budget -= 1
        rect = ordered[idx]
        for out_bbox, in_bbox in _candidate_boxes(height, width, rect["h"], rect["w"], rect["dr"], rect["dc"]):
            if _overlaps(out_bbox, in_bbox, placed, height, width):
                continue
            placed.append({**rect, "out_bbox": out_bbox, "in_bbox": in_bbox})
            if rec(idx + 1):
                return True
            placed.pop()
        return False

    if not rec(0):
        return None
    return placed


def _render_input(rects: list[dict], height: int, width: int) -> np.ndarray:
    grid = np.full((height, width), BACKGROUND, dtype=int)
    for rect in rects:
        color = int(rect["color"])
        dr, dc = int(rect["dr"]), int(rect["dc"])
        r0, c0, r1, c1 = rect["in_bbox"]
        grid[r0 : r1 + 1, c0 : c1 + 1] = color
        grid[r0 + (0 if dr == -1 else rect["h"] - 1), c0 + (0 if dc == -1 else rect["w"] - 1)] = TEAL
    return grid


def _gen_attempt() -> np.ndarray | None:
    height = random.choice(tuple(HEIGHT_RANGE))
    width = random.choice(tuple(WIDTH_RANGE))
    count = random.randint(*RECT_COUNT_RANGE)
    sizes = _choose_sizes(count, height, width)
    if sizes is None:
        return None

    colors = random.sample(RECT_COLORS, count)
    dirs = [random.choice(SHIFT_DIRS) for _ in range(count)]
    if count >= 4 and random.random() < 0.5:
        dirs = SHIFT_DIRS.copy()
        random.shuffle(dirs)
        dirs = dirs[:count]

    rects = []
    for (rh, rw), color, (dr, dc) in zip(sizes, colors, dirs):
        rects.append({"h": rh, "w": rw, "color": color, "dr": dr, "dc": dc})

    placed = _place_rectangles(rects, height, width)
    if placed is None:
        return None

    inp = _render_input(placed, height, width)
    try:
        out = main(inp)
    except AssertionError:
        return None

    expected = np.full((height, width), BACKGROUND, dtype=int)
    for rect in placed:
        r0, c0, r1, c1 = rect["out_bbox"]
        expected[r0 : r1 + 1, c0 : c1 + 1] = int(rect["color"])
    return inp if np.array_equal(out, expected) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
