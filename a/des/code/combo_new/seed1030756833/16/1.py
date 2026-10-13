from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
FRAME = 1
SIDES = ("top", "bottom", "left", "right")
GROWTH = {
    "top": (1, 0),
    "bottom": (-1, 0),
    "left": (0, 1),
    "right": (0, -1),
}

BBox = tuple[int, int, int, int]
Point = tuple[int, int]


def _neighbors4(r: int, c: int) -> tuple[Point, Point, Point, Point]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[Point]]:
    m = np.asarray(mask, dtype=bool)
    h, w = m.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[Point]] = []
    for sr, sc in np.argwhere(m):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        seen[sr, sc] = True
        q: deque[Point] = deque([(sr, sc)])
        comp: list[Point] = []
        while q:
            r, c = q.popleft()
            comp.append((r, c))
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and m[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(comp)
    return comps


def _frame_border_cells(bbox: BBox) -> set[Point]:
    r0, c0, r1, c1 = bbox
    cells: set[Point] = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _side_of_border_cell(cell: Point, bbox: BBox) -> str | None:
    r, c = cell
    r0, c0, r1, c1 = bbox
    sides = []
    if r == r0:
        sides.append("top")
    if r == r1:
        sides.append("bottom")
    if c == c0:
        sides.append("left")
    if c == c1:
        sides.append("right")
    return sides[0] if len(sides) == 1 else None


def _find_frames(g: np.ndarray) -> list[tuple[BBox, Point, str]]:
    frames: list[tuple[BBox, Point, str]] = []
    for comp in _components(g == FRAME):
        rs = [r for r, _ in comp]
        cs = [c for _, c in comp]
        bbox = (min(rs), min(cs), max(rs), max(cs))
        r0, c0, r1, c1 = bbox
        if r1 - r0 + 1 < 4 or c1 - c0 + 1 < 4:
            continue

        expected = _frame_border_cells(bbox)
        comp_set = set(comp)
        if not comp_set <= expected:
            continue
        missing = expected - comp_set
        if len(missing) != 1:
            continue
        gap = next(iter(missing))
        gap_side = _side_of_border_cell(gap, bbox)
        if gap_side is None:
            continue
        frames.append((bbox, gap, gap_side))

    frames.sort(key=lambda item: item[0])
    return frames


def _line_cells(a: Point, b: Point) -> list[Point]:
    ar, ac = a
    br, bc = b
    if ar == br:
        step = 1 if bc >= ac else -1
        return [(ar, c) for c in range(ac, bc + step, step)]
    if ac == bc:
        step = 1 if br >= ar else -1
        return [(r, ac) for r in range(ar, br + step, step)]
    raise AssertionError("path must be horizontal or vertical")


def _inside_any_frame(cell: Point, bboxes: list[BBox]) -> bool:
    r, c = cell
    return any(r0 <= r <= r1 and c0 <= c <= c1 for r0, c0, r1, c1 in bboxes)


def _inside_block(g: np.ndarray, bbox: BBox) -> tuple[int, str, list[Point]]:
    r0, c0, r1, c1 = bbox
    sub = g[r0 + 1 : r1, c0 + 1 : c1]
    comps = _components((sub != BG) & (sub != FRAME))
    assert len(comps) == 1, "each frame must contain one interior color block"

    coords = [(r + r0 + 1, c + c0 + 1) for r, c in comps[0]]
    colors = {int(g[r, c]) for r, c in coords}
    assert len(colors) == 1, "interior block must be monochrome"
    assert 2 <= len(coords) <= 3, "interior block must be a small 2-3 pixel marker"
    color = next(iter(colors))

    touches = []
    if any(r == r0 + 1 for r, _ in coords):
        touches.append("top")
    if any(r == r1 - 1 for r, _ in coords):
        touches.append("bottom")
    if any(c == c0 + 1 for _, c in coords):
        touches.append("left")
    if any(c == c1 - 1 for _, c in coords):
        touches.append("right")
    assert len(touches) == 1, "interior block must touch exactly one inner wall"
    return color, touches[0], coords


def _seed_path(
    g: np.ndarray,
    seed: Point,
    gap: Point,
    gap_side: str,
    bbox: BBox,
    color: int,
) -> list[Point] | None:
    sr, sc = seed
    gr, gc = gap
    r0, c0, r1, c1 = bbox

    if gap_side == "top" and not (sc == gc and sr < r0):
        return None
    if gap_side == "bottom" and not (sc == gc and sr > r1):
        return None
    if gap_side == "left" and not (sr == gr and sc < c0):
        return None
    if gap_side == "right" and not (sr == gr and sc > c1):
        return None

    cells = _line_cells(seed, gap)
    if int(g[seed]) != color or int(g[gap]) != BG:
        return None
    for cell in cells[1:-1]:
        if int(g[cell]) != BG:
            return None
    return cells


def _find_seed_and_path(
    g: np.ndarray,
    color: int,
    bbox: BBox,
    gap: Point,
    gap_side: str,
    frame_bboxes: list[BBox],
) -> tuple[Point, list[Point]]:
    matches: list[tuple[Point, list[Point]]] = []
    for rr, cc in np.argwhere(g == color):
        seed = (int(rr), int(cc))
        if _inside_any_frame(seed, frame_bboxes):
            continue
        path = _seed_path(g, seed, gap, gap_side, bbox, color)
        if path is not None:
            matches.append((seed, path))
    assert len(matches) == 1, "each interior color must have one aligned outside seed"
    return matches[0]


def _entry_cell(gap: Point, gap_side: str, bbox: BBox) -> Point:
    gr, gc = gap
    r0, c0, r1, c1 = bbox
    if gap_side == "top":
        return r0 + 1, gc
    if gap_side == "bottom":
        return r1 - 1, gc
    if gap_side == "left":
        return gr, c0 + 1
    if gap_side == "right":
        return gr, c1 - 1
    raise AssertionError("unknown gap side")


def _internal_line(bbox: BBox, start: Point, direction: Point) -> list[Point]:
    r0, c0, r1, c1 = bbox
    dr, dc = direction
    r, c = start
    cells: list[Point] = []
    while r0 < r < r1 and c0 < c < c1:
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow same-colored paths from outside seeds into gapped blue frames using the interior block's wall contact.

    Input structure:
    - The background is black, and blue pixels form several one-pixel-thick hollow rectangular frames.
    - Each frame border has exactly one missing cell, and each frame contains one small monochrome 2-3 pixel block.
    - The block colors are distinct, and each color has one outside singleton seed aligned with that frame's gap.

    Rule:
    - For each frame, identify the side of the inner wall touched by its small block.
    - The touched side gives the growth direction: top grows downward, bottom upward, left rightward, and right leftward.
    - Draw the seed color from the outside seed to the gap, then from the gap's interior entry cell along the growth direction until the opposite wall is reached.
    - Keep the original block color, the blue frame pixels, and all unrelated background cells unchanged.

    Key insight:
    - The interior block is a directional code, while the matching outside seed and the frame gap provide the entry path; both clues are needed to draw the channel.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    frames = _find_frames(g)
    assert frames, "expected at least one gapped rectangular frame"
    frame_bboxes = [bbox for bbox, _gap, _side in frames]

    out = g.copy()
    for bbox, gap, gap_side in frames:
        color, touched_side, _coords = _inside_block(g, bbox)
        _seed, outside_path = _find_seed_and_path(g, color, bbox, gap, gap_side, frame_bboxes)

        for cell in outside_path:
            assert int(g[cell]) in (BG, color), "outside seed path must be clear"
            out[cell] = color

        entry = _entry_cell(gap, gap_side, bbox)
        inside_path = _internal_line(bbox, entry, GROWTH[touched_side])
        assert inside_path, "entry must lead into the frame interior"
        for cell in inside_path:
            assert int(g[cell]) in (BG, color), "interior channel must be empty except for the same color block"
            out[cell] = color

    return out


# --------------- generator ---------------

GEN_COLORS = tuple(range(2, 10))
GRID_H_RANGE = (16, 30)
GRID_W_RANGE = (16, 30)
FRAME_SIZE_RANGE = (6, 9)
GEN_ATTEMPTS = 500
PLACE_ATTEMPTS = 240


def _rect_free(reserved: np.ndarray, bbox: BBox, pad: int = 0) -> bool:
    h, w = reserved.shape
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    return not bool(reserved[rr0 : rr1 + 1, cc0 : cc1 + 1].any())


def _cells_free(reserved: np.ndarray, cells: list[Point] | set[Point], pad: int = 0) -> bool:
    h, w = reserved.shape
    for r, c in cells:
        rr0 = max(0, r - pad)
        cc0 = max(0, c - pad)
        rr1 = min(h - 1, r + pad)
        cc1 = min(w - 1, c + pad)
        if reserved[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
            return False
    return True


def _mark_rect(reserved: np.ndarray, bbox: BBox, pad: int = 0) -> None:
    h, w = reserved.shape
    r0, c0, r1, c1 = bbox
    reserved[max(0, r0 - pad) : min(h, r1 + pad + 1), max(0, c0 - pad) : min(w, c1 + pad + 1)] = True


def _mark_cells(reserved: np.ndarray, cells: list[Point] | set[Point], pad: int = 0) -> None:
    h, w = reserved.shape
    for r, c in cells:
        reserved[max(0, r - pad) : min(h, r + pad + 1), max(0, c - pad) : min(w, c + pad + 1)] = True


def _block_cells(bbox: BBox, side: str) -> list[Point]:
    r0, c0, r1, c1 = bbox
    make_l = random.random() < 0.35
    length = random.choice([2, 3])

    if side in ("top", "bottom"):
        c = random.randint(c0 + 2, c1 - 2)
        if side == "top":
            cells = [(r0 + 1 + i, c) for i in range(length)]
            side_row = r0 + 1
        else:
            cells = [(r1 - 1 - i, c) for i in range(length)]
            side_row = r1 - 1
        offsets = [off for off in (-1, 1) if c0 + 1 < c + off < c1 - 1]
        if make_l and length == 3 and offsets:
            cells = [cells[0], cells[1], (side_row, c + random.choice(offsets))]
        return cells

    r = random.randint(r0 + 2, r1 - 2)
    if side == "left":
        cells = [(r, c0 + 1 + i) for i in range(length)]
        side_col = c0 + 1
    else:
        cells = [(r, c1 - 1 - i) for i in range(length)]
        side_col = c1 - 1
    offsets = [off for off in (-1, 1) if r0 + 1 < r + off < r1 - 1]
    if make_l and length == 3 and offsets:
        cells = [cells[0], cells[1], (r + random.choice(offsets), side_col)]
    return cells


def _choose_gap(bbox: BBox, side: str, block: list[Point]) -> Point:
    r0, c0, r1, c1 = bbox
    if side in ("top", "bottom"):
        forbidden = {c for _r, c in block}
        candidates = [c for c in range(c0 + 1, c1) if c not in forbidden]
        if not candidates:
            candidates = list(range(c0 + 1, c1))
        c = random.choice(candidates)
        return (r0, c) if side == "top" else (r1, c)

    forbidden = {r for r, _c in block}
    candidates = [r for r in range(r0 + 1, r1) if r not in forbidden]
    if not candidates:
        candidates = list(range(r0 + 1, r1))
    r = random.choice(candidates)
    return (r, c0) if side == "left" else (r, c1)


def _seed_for_gap(h: int, w: int, bbox: BBox, gap: Point, side: str) -> Point:
    gr, gc = gap
    r0, c0, r1, c1 = bbox
    if side == "top":
        return random.randint(0, r0 - 2), gc
    if side == "bottom":
        return random.randint(r1 + 2, h - 1), gc
    if side == "left":
        return gr, random.randint(0, c0 - 2)
    if side == "right":
        return gr, random.randint(c1 + 2, w - 1)
    raise AssertionError("unknown side")


def _sample_bbox(h: int, w: int, side: str) -> BBox | None:
    fh = random.randint(FRAME_SIZE_RANGE[0], min(FRAME_SIZE_RANGE[1], h - 3))
    fw = random.randint(FRAME_SIZE_RANGE[0], min(FRAME_SIZE_RANGE[1], w - 3))

    r_min, r_max = 1, h - fh - 1
    c_min, c_max = 1, w - fw - 1
    if side == "top":
        r_min = max(r_min, 2)
    if side == "bottom":
        r_max = min(r_max, h - fh - 2)
    if side == "left":
        c_min = max(c_min, 2)
    if side == "right":
        c_max = min(c_max, w - fw - 2)
    if r_min > r_max or c_min > c_max:
        return None

    r0 = random.randint(r_min, r_max)
    c0 = random.randint(c_min, c_max)
    return r0, c0, r0 + fh - 1, c0 + fw - 1


def _place_one(grid: np.ndarray, reserved: np.ndarray, color: int) -> dict | None:
    h, w = grid.shape
    for _ in range(PLACE_ATTEMPTS):
        side = random.choice(SIDES)
        bbox = _sample_bbox(h, w, side)
        if bbox is None or not _rect_free(reserved, bbox, pad=1):
            continue

        block = _block_cells(bbox, side)
        gap = _choose_gap(bbox, side, block)
        seed = _seed_for_gap(h, w, bbox, gap, side)
        border = _frame_border_cells(bbox)
        border.remove(gap)
        outside_path = _line_cells(seed, gap)
        inside_path = _internal_line(bbox, _entry_cell(gap, side, bbox), GROWTH[side])

        needed = set(border) | set(block) | {seed} | set(outside_path) | set(inside_path)
        if not _cells_free(reserved, needed):
            continue

        for r, c in border:
            grid[r, c] = FRAME
        for r, c in block:
            grid[r, c] = color
        grid[seed] = color

        _mark_rect(reserved, bbox, pad=1)
        _mark_cells(reserved, outside_path, pad=0)
        return {"color": color, "outside": outside_path, "inside": inside_path}
    return None


def _expected_output(inp: np.ndarray, plans: list[dict]) -> np.ndarray:
    out = inp.copy()
    for plan in plans:
        color = int(plan["color"])
        for r, c in plan["outside"] + plan["inside"]:
            out[r, c] = color
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    grid = np.full((h, w), BG, dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    max_frames = min(4, max(1, (h * w) // 170 + 1), len(GEN_COLORS))
    n_frames = random.randint(1, max_frames)
    colors = random.sample(list(GEN_COLORS), n_frames)

    plans: list[dict] = []
    for color in colors:
        plan = _place_one(grid, reserved, color)
        if plan is None:
            return None
        plans.append(plan)

    expected = _expected_output(grid, plans)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("failed to generate a valid seed1030756833/16/1 input")
