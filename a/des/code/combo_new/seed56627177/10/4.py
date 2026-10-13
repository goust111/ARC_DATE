import random
from typing import Iterable

import numpy as np


BG = 0
START = 4
FRAME_COLORS = [1, 2, 3, 5, 6, 7, 8, 9]
SIDES = ("top", "bottom", "left", "right")
GEN_RETRIES = 400


def _component(start: tuple[int, int], coords: set[tuple[int, int]]) -> set[tuple[int, int]]:
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (nr, nc) in coords and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return seen


def _perimeter_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    pts = set()
    for c in range(c0, c1 + 1):
        pts.add((r0, c))
        pts.add((r1, c))
    for r in range(r0, r1 + 1):
        pts.add((r, c0))
        pts.add((r, c1))
    return pts


def _gap_side(gap: tuple[int, int], bbox: tuple[int, int, int, int]) -> str:
    r, c = gap
    r0, c0, r1, c1 = bbox
    if r == r0:
        return "top"
    if r == r1:
        return "bottom"
    if c == c0:
        return "left"
    if c == c1:
        return "right"
    raise ValueError("gap is not on the perimeter")


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int, gap: tuple[int, int]) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    grid[gap] = BG


def _find_start(grid: np.ndarray) -> tuple[int, int]:
    pts = np.argwhere(grid == START)
    assert len(pts) == 1, "expected exactly one yellow start marker"
    r, c = map(int, pts[0])
    h, w = grid.shape
    assert r in (0, h - 1) or c in (0, w - 1), "start marker must lie on the grid boundary"
    return r, c


def _analyze_frame(grid: np.ndarray, color: int) -> dict | None:
    coords = {(int(r), int(c)) for r, c in np.argwhere(grid == color)}
    if not coords:
        return None
    start = next(iter(coords))
    if _component(start, coords) != coords:
        return None

    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None

    perimeter = _perimeter_cells(r0, c0, r1, c1)
    if any(pt not in perimeter for pt in coords):
        return None

    missing = list(perimeter - coords)
    if len(missing) != 1:
        return None
    gap = missing[0]
    if gap in {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}:
        return None

    return {
        "color": int(color),
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "gap": (int(gap[0]), int(gap[1])),
        "side": _gap_side(gap, (r0, c0, r1, c1)),
    }


def _frames(grid: np.ndarray) -> list[dict]:
    frames = []
    for color in sorted(int(v) for v in np.unique(grid)):
        if color in (BG, START):
            continue
        frame = _analyze_frame(grid, color)
        if frame is not None:
            frames.append(frame)
    assert frames, "expected at least one valid frame"
    return frames


def _start_matches(frame: dict, start: tuple[int, int], shape: tuple[int, int]) -> bool:
    sr, sc = start
    h, w = shape
    r0, c0, r1, c1 = frame["bbox"]
    gr, gc = frame["gap"]
    side = frame["side"]
    if side == "top":
        return sr == 0 and sc == gc and sr < r0
    if side == "bottom":
        return sr == h - 1 and sc == gc and sr > r1
    if side == "left":
        return sc == 0 and sr == gr and sc < c0
    return sc == w - 1 and sr == gr and sc > c1


def _frame_chain(grid: np.ndarray) -> list[dict]:
    start = _find_start(grid)
    frames = _frames(grid)
    by_bbox = {tuple(frame["bbox"]): frame for frame in frames}
    candidates = [frame for frame in frames if _start_matches(frame, start, grid.shape)]
    if not candidates:
        candidates = frames
    candidates.sort(key=lambda frame: (frame["bbox"][2] - frame["bbox"][0]) * (frame["bbox"][3] - frame["bbox"][1]), reverse=True)

    best_chain: list[dict] = []
    for outer in candidates:
        chain = [outer]
        bbox = outer["bbox"]
        while True:
            r0, c0, r1, c1 = bbox
            inner_bbox = (r0 + 2, c0 + 2, r1 - 2, c1 - 2)
            nxt = by_bbox.get(inner_bbox)
            if nxt is None:
                break
            chain.append(nxt)
            bbox = inner_bbox
        if len(chain) > len(best_chain):
            best_chain = chain

    assert best_chain, "expected a nested frame chain"
    return best_chain


def _expected_output_from_chain(grid: np.ndarray, chain: Iterable[dict]) -> np.ndarray:
    chain = list(chain)
    colors = [int(frame["color"]) for frame in chain]
    assert colors, "frame chain must be non-empty"
    out = np.array(grid, dtype=int, copy=True)
    r0, c0, r1, c1 = chain[-1]["bbox"]
    for idx, r in enumerate(range(r0 + 1, r1)):
        out[r, c0 + 1 : c1] = colors[idx % len(colors)]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Follow the nested frame entry order and use the crossed frame colors to stripe the innermost region.

    Input structure:
    - Background is black.
    - Several one-cell-thick rectangular frames are concentrically nested, each in a different color.
    - Every frame is missing exactly one non-corner perimeter cell, forming a one-cell gap on one side.
    - A single yellow marker lies on the outer grid boundary, aligned outside the outermost frame's gap.
    - The area inside the innermost frame is entirely black.

    Rule:
    - Identify the nested chain of rectangular frames reached from the yellow marker through the outermost gap.
    - Traverse the chain from the outer frame to the inner frame; each crossed frame contributes its own color once.
    - Leave the frames and the yellow marker unchanged.
    - Fill the innermost region row by row with the collected color sequence, cycling from the first frame color again when the region has more rows than frames.

    Key insight:
    - The yellow border marker selects the outer frame, and the concentric bboxes reveal the full outer-to-inner crossing order; that ordered color list directly determines the row stripes in the center.
    """
    g = np.asarray(grid, dtype=int)
    chain = _frame_chain(g)
    return _expected_output_from_chain(g, chain)


def _gap_position(rng: random.Random, bbox: tuple[int, int, int, int], side: str) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    if side == "top":
        return r0, rng.randint(c0 + 1, c1 - 1)
    if side == "bottom":
        return r1, rng.randint(c0 + 1, c1 - 1)
    if side == "left":
        return rng.randint(r0 + 1, r1 - 1), c0
    return rng.randint(r0 + 1, r1 - 1), c1


def _place_start(grid: np.ndarray, gap: tuple[int, int], side: str) -> None:
    gr, gc = gap
    h, w = grid.shape
    if side == "top":
        grid[0, gc] = START
    elif side == "bottom":
        grid[h - 1, gc] = START
    elif side == "left":
        grid[gr, 0] = START
    else:
        grid[gr, w - 1] = START


def _attempt_generate(rng: random.Random) -> np.ndarray | None:
    layers = rng.randint(2, 5)
    outer_side = rng.choice(SIDES)

    top = 1 if outer_side == "top" else rng.randint(1, 4)
    bottom = 1 if outer_side == "bottom" else rng.randint(1, 4)
    left = 1 if outer_side == "left" else rng.randint(1, 4)
    right = 1 if outer_side == "right" else rng.randint(1, 4)

    max_inner_h = 30 - top - bottom - (4 * layers - 2)
    max_inner_w = 30 - left - right - (4 * layers - 2)
    if max_inner_h < 1 or max_inner_w < 1:
        return None

    inner_h = rng.randint(1, min(8, max_inner_h))
    inner_w = rng.randint(1, min(10, max_inner_w))
    outer_h = inner_h + 4 * layers - 2
    outer_w = inner_w + 4 * layers - 2

    h = top + outer_h + bottom
    w = left + outer_w + right
    if h > 30 or w > 30:
        return None

    colors = rng.sample(FRAME_COLORS, layers)
    grid = np.full((h, w), BG, dtype=int)
    spec_chain = []

    for depth in range(layers):
        bbox = (top + 2 * depth, left + 2 * depth, top + outer_h - 1 - 2 * depth, left + outer_w - 1 - 2 * depth)
        side = outer_side if depth == 0 else rng.choice(SIDES)
        gap = _gap_position(rng, bbox, side)
        _draw_frame(grid, bbox, colors[depth], gap)
        spec_chain.append({"color": colors[depth], "bbox": bbox, "gap": gap, "side": side})

    _place_start(grid, spec_chain[0]["gap"], spec_chain[0]["side"])

    expected = _expected_output_from_chain(grid, spec_chain)
    try:
        if not np.array_equal(main(grid), expected):
            return None
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _attempt_generate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-frame puzzle")
