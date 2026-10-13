import random
from typing import Iterable

import numpy as np


BG = 0
FRAME = 5
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
SIDES = ("top", "bottom", "left", "right")
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# --------------- solver ---------------


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _rect_border(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    cells = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0 + 1, r1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _segment_cells(start: tuple[int, int], end: tuple[int, int]) -> set[tuple[int, int]]:
    (r0, c0), (r1, c1) = start, end
    cells = set()
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        for c in range(c0, c1 + step, step):
            cells.add((r0, c))
        return cells
    if c0 == c1:
        step = 1 if r1 >= r0 else -1
        for r in range(r0, r1 + step, step):
            cells.add((r, c0))
        return cells
    raise ValueError("segment must be axis-aligned")


def _trunk_cells(seed: tuple[int, int], center: tuple[int, int]) -> set[tuple[int, int]]:
    sr, sc = seed
    cr, cc = center
    cells = _segment_cells((sr, sc), (sr, cc))
    cells |= _segment_cells((sr, cc), (cr, cc))
    return cells


def _branch_cells(center: tuple[int, int], side: str, shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    cr, cc = center
    if side == "top":
        return _segment_cells((cr, cc), (0, cc))
    if side == "bottom":
        return _segment_cells((cr, cc), (h - 1, cc))
    if side == "left":
        return _segment_cells((cr, cc), (cr, 0))
    if side == "right":
        return _segment_cells((cr, cc), (cr, w - 1))
    raise ValueError(f"unknown side: {side}")


def _frame_from_bbox(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> dict:
    assert r1 - r0 >= 4 and c1 - c0 >= 4, "frame bbox too small"
    center = ((r0 + r1) // 2, (c0 + c1) // 2)
    cr, cc = center
    assert r0 < cr < r1 and c0 < cc < c1, "frame must have an interior center cell"

    sides = []
    for side in SIDES:
        gr, gc = _gap_cell(r0, c0, r1, c1, side)
        if grid[gr, gc] != FRAME:
            assert grid[gr, gc] == BG, "gap cells must be black"
            sides.append(side)

    assert 1 <= len(sides) <= 3, "expected 1 to 3 gaps"
    gaps = {_gap_cell(r0, c0, r1, c1, side) for side in sides}

    for rr in range(r0, r1 + 1):
        for cc2 in range(c0, c1 + 1):
            on_border = rr in (r0, r1) or cc2 in (c0, c1)
            if not on_border:
                continue
            if (rr, cc2) == (r0, c0) or (rr, cc2) == (r0, c1) or (rr, cc2) == (r1, c0) or (rr, cc2) == (r1, c1):
                assert grid[rr, cc2] == FRAME, "corners must stay grey"
                continue
            if (rr, cc2) in gaps:
                continue
            assert grid[rr, cc2] == FRAME, "perimeter must be grey outside gaps"

    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    seed_mask = (inner != BG) & (inner != FRAME)
    seed_coords = np.argwhere(seed_mask)
    assert seed_coords.shape == (1, 2), "each frame interior must contain one seed"
    assert bool(np.all(inner[~seed_mask] == BG)), "non-seed interior must be black"
    seed_r, seed_c = map(int, seed_coords[0])
    seed = (r0 + 1 + seed_r, c0 + 1 + seed_c)

    return {
        "bbox": (r0, c0, r1, c1),
        "center": center,
        "seed": seed,
        "color": int(grid[seed]),
        "sides": tuple(sides),
    }


def _find_frame_for_seed(
    grid: np.ndarray,
    seed: tuple[int, int],
    grey_rows: list[int],
    grey_cols: list[int],
) -> dict:
    sr, sc = seed
    top_candidates = [r for r in grey_rows if r < sr]
    bottom_candidates = [r for r in grey_rows if r > sr]
    left_candidates = [c for c in grey_cols if c < sc]
    right_candidates = [c for c in grey_cols if c > sc]

    matches = []
    for r0 in top_candidates:
        for r1 in bottom_candidates:
            if r1 - r0 < 4 or (r0 + r1) % 2 != 0:
                continue
            for c0 in left_candidates:
                for c1 in right_candidates:
                    if c1 - c0 < 4 or (c0 + c1) % 2 != 0:
                        continue
                    if not (r0 < sr < r1 and c0 < sc < c1):
                        continue
                    try:
                        frame = _frame_from_bbox(grid, r0, c0, r1, c1)
                    except AssertionError:
                        continue
                    if frame["seed"] != seed:
                        continue
                    matches.append(frame)

    assert matches, "failed to find frame for seed"
    matches.sort(key=lambda frame: (frame["bbox"][2] - frame["bbox"][0]) * (frame["bbox"][3] - frame["bbox"][1]))
    best = matches[0]
    assert all(frame["bbox"] == best["bbox"] for frame in matches), "seed should determine a unique frame"
    return best


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow a colored trunk from each seed to its frame center, then send one
    branch ray from that center through each frame gap out to the grid edge.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    seed_coords = np.argwhere((g != BG) & (g != FRAME))
    assert seed_coords.size > 0, "expected at least one seed"
    grey_coords = np.argwhere(g == FRAME)
    assert grey_coords.size > 0, "expected grey frames"
    grey_rows = sorted({int(r) for r, _c in grey_coords})
    grey_cols = sorted({int(c) for _r, c in grey_coords})

    frames = [
        _find_frame_for_seed(g, (int(r), int(c)), grey_rows, grey_cols)
        for r, c in seed_coords
    ]
    frames.sort(key=lambda frame: frame["bbox"])

    out = g.copy()
    for frame in frames:
        color = int(frame["color"])
        trunk = _trunk_cells(frame["seed"], frame["center"])
        branches = set()
        for side in frame["sides"]:
            branches |= _branch_cells(frame["center"], side, g.shape)
        for r, c in trunk | branches:
            if g[r, c] != FRAME:
                out[r, c] = color
    return out


# --------------- generator ---------------

GRID_SIDE_RANGE = (18, 28)
FRAME_COUNT_RANGE = (1, 3)
FRAME_SIZE_OPTIONS = (5, 7, 9)
MAX_ATTEMPTS = 400
FRAME_TRIES = 300
HALO_PAD = 1


def _expanded_box_cells(
    r0: int, c0: int, r1: int, c1: int, h: int, w: int, pad: int
) -> set[tuple[int, int]]:
    cells = set()
    for r in range(max(0, r0 - pad), min(h - 1, r1 + pad) + 1):
        for c in range(max(0, c0 - pad), min(w - 1, c1 + pad) + 1):
            cells.add((r, c))
    return cells


def _gap_cell(r0: int, c0: int, r1: int, c1: int, side: str) -> tuple[int, int]:
    cr = (r0 + r1) // 2
    cc = (c0 + c1) // 2
    if side == "top":
        return (r0, cc)
    if side == "bottom":
        return (r1, cc)
    if side == "left":
        return (cr, c0)
    if side == "right":
        return (cr, c1)
    raise ValueError(f"unknown side: {side}")


def _candidate_frame(
    h: int, w: int, color: int, force_off_center: bool
) -> dict | None:
    fh = random.choice(FRAME_SIZE_OPTIONS)
    fw = random.choice(FRAME_SIZE_OPTIONS)
    if fh + 2 >= h or fw + 2 >= w:
        return None

    r0 = random.randint(1, h - fh - 1)
    c0 = random.randint(1, w - fw - 1)
    r1 = r0 + fh - 1
    c1 = c0 + fw - 1
    center = ((r0 + r1) // 2, (c0 + c1) // 2)

    gap_count = random.randint(1, 3)
    sides = tuple(random.sample(SIDES, gap_count))
    border_all = _rect_border(r0, c0, r1, c1)
    gaps = {_gap_cell(r0, c0, r1, c1, side) for side in sides}
    grey_cells = border_all - gaps

    inner_cells = [
        (r, c)
        for r in range(r0 + 1, r1)
        for c in range(c0 + 1, c1)
    ]
    if force_off_center:
        seed_options = [cell for cell in inner_cells if cell != center]
        if not seed_options:
            return None
    else:
        if random.random() < 0.4:
            seed_options = [center]
        else:
            seed_options = [cell for cell in inner_cells if cell != center]
            if not seed_options:
                seed_options = [center]
    seed = random.choice(seed_options)

    trunk = _trunk_cells(seed, center)
    branches = set()
    for side in sides:
        branches |= _branch_cells(center, side, (h, w))
    output_cells = trunk | branches
    if trunk & grey_cells:
        return None
    if branches & grey_cells:
        return None

    return {
        "bbox": (r0, c0, r1, c1),
        "border_all": border_all,
        "grey_cells": grey_cells,
        "seed": seed,
        "color": color,
        "sides": sides,
        "center": center,
        "output_cells": output_cells,
        "halo": _expanded_box_cells(r0, c0, r1, c1, h, w, HALO_PAD),
    }


def _placeable(candidate: dict, state: dict) -> bool:
    input_cells = set(candidate["grey_cells"]) | {candidate["seed"]}
    if candidate["halo"] & state["halos"]:
        return False
    if input_cells & state["input_cells"]:
        return False
    if input_cells & state["output_cells"]:
        return False
    if candidate["border_all"] & state["output_cells"]:
        return False
    if candidate["output_cells"] & state["output_cells"]:
        return False
    if candidate["output_cells"] & state["barriers"]:
        return False
    return True


def _paint_input(h: int, w: int, frames: Iterable[dict]) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    for frame in frames:
        for r, c in frame["grey_cells"]:
            grid[r, c] = FRAME
        sr, sc = frame["seed"]
        grid[sr, sc] = int(frame["color"])
    return grid


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_SIDE_RANGE)
    w = random.randint(*GRID_SIDE_RANGE)
    frame_count = random.randint(*FRAME_COUNT_RANGE)
    if frame_count > len(SEED_COLORS):
        return None

    colors = random.sample(SEED_COLORS, frame_count)
    off_center_index = random.randrange(frame_count)

    state = {
        "halos": set(),
        "input_cells": set(),
        "output_cells": set(),
        "barriers": set(),
    }
    frames = []

    for idx in range(frame_count):
        placed = False
        for _ in range(FRAME_TRIES):
            candidate = _candidate_frame(
                h=h,
                w=w,
                color=colors[idx],
                force_off_center=(idx == off_center_index),
            )
            if candidate is None or not _placeable(candidate, state):
                continue

            frames.append(candidate)
            state["halos"] |= candidate["halo"]
            state["input_cells"] |= set(candidate["grey_cells"]) | {candidate["seed"]}
            state["output_cells"] |= candidate["output_cells"]
            state["barriers"] |= candidate["border_all"]
            placed = True
            break
        if not placed:
            return None

    grid = _paint_input(h, w, frames)

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
