import random
from collections import Counter

import numpy as np


BG = 0
SHADOW = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_RETRIES = 400
DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project repeated grey copies of each arrow's tail rectangle outward from its tip.

    Input structure:
    - The background is the dominant color.
    - Each non-background monochrome object is a single arrow made of a
      1/3/5 triangular head plus an attached rectangular tail block.
    - The triangle points in one of the four cardinal directions, while the
      attached tail is a 5xn or nx5 rectangle of the same color.

    Rule:
    - Keep every original arrow unchanged.
    - For each arrow, split the object into its direction-giving triangle and
      its stamped tail rectangle.
    - Let d be the tail size along the pointing axis. Along the center ray that
      starts at the tip, take sample points separated by d+1 cells. At each
      sample point, place a grey copy of the tail rectangle immediately beyond
      that point in the arrow's pointing direction, keeping the rectangle's
      horizontal/vertical orientation unchanged.
    - Stop when the next copy would leave the grid. If a candidate copy would
      overlap any original arrow or any already stamped grey copy, skip it and
      continue farther along the ray.

    Key insight:
    The triangle only encodes the projection direction; the attached rectangle
    is the reusable template, and its thickness along the motion axis also sets
    the projection stride.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)
    assert int(bg) != SHADOW

    arrows = _detect_arrows(g, int(bg))
    out, _stats = _simulate_shadows(g, arrows=arrows, bg=int(bg))
    return out


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def _build_arrow_spec(direction: str, tail_len: int) -> dict:
    coords = set()
    tail = set()
    if direction == "up":
        rows = {0: [2], 1: [1, 2, 3], 2: [0, 1, 2, 3, 4]}
        for r, cols in rows.items():
            for c in cols:
                coords.add((r, c))
        for r in range(3, 3 + tail_len):
            for c in range(5):
                coords.add((r, c))
                tail.add((r, c))
        tip = (0, 2)
        bbox = (3 + tail_len, 5)
    elif direction == "down":
        for r in range(tail_len):
            for c in range(5):
                coords.add((r, c))
                tail.add((r, c))
        rows = {
            tail_len: [0, 1, 2, 3, 4],
            tail_len + 1: [1, 2, 3],
            tail_len + 2: [2],
        }
        for r, cols in rows.items():
            for c in cols:
                coords.add((r, c))
        tip = (tail_len + 2, 2)
        bbox = (3 + tail_len, 5)
    elif direction == "left":
        cols = {0: [2], 1: [1, 2, 3], 2: [0, 1, 2, 3, 4]}
        for c, rows in cols.items():
            for r in rows:
                coords.add((r, c))
        for c in range(3, 3 + tail_len):
            for r in range(5):
                coords.add((r, c))
                tail.add((r, c))
        tip = (2, 0)
        bbox = (5, 3 + tail_len)
    elif direction == "right":
        for c in range(tail_len):
            for r in range(5):
                coords.add((r, c))
                tail.add((r, c))
        cols = {
            tail_len: [0, 1, 2, 3, 4],
            tail_len + 1: [1, 2, 3],
            tail_len + 2: [2],
        }
        for c, rows in cols.items():
            for r in rows:
                coords.add((r, c))
        tip = (2, tail_len + 2)
        bbox = (5, 3 + tail_len)
    else:
        raise ValueError(direction)
    return {
        "direction": direction,
        "tail_len": int(tail_len),
        "coords": frozenset(coords),
        "tail": frozenset(tail),
        "tip": tip,
        "bbox": bbox,
    }


SPECS = [_build_arrow_spec(direction, tail_len) for direction in DIRS for tail_len in (1, 2, 3)]
SPEC_MAP = {(spec["direction"], spec["tail_len"]): spec for spec in SPECS}


def _detect_arrows(grid: np.ndarray, bg: int) -> list[dict]:
    arrows = []
    for color in sorted(set(map(int, grid.ravel())) - {int(bg)}):
        pts = np.argwhere(grid == int(color))
        assert pts.size > 0
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
        norm = frozenset((int(r - r0), int(c - c0)) for r, c in pts)
        matches = [
            spec
            for spec in SPECS
            if spec["bbox"] == (int(r1 - r0 + 1), int(c1 - c0 + 1)) and norm == spec["coords"]
        ]
        assert len(matches) == 1
        spec = matches[0]
        cells = {(int(r), int(c)) for r, c in pts}
        arrows.append(
            {
                "color": int(color),
                "direction": spec["direction"],
                "tail_len": int(spec["tail_len"]),
                "tip": (int(r0 + spec["tip"][0]), int(c0 + spec["tip"][1])),
                "cells": cells,
            }
        )
    assert arrows
    return arrows


def _stamp_cells(direction: str, tail_len: int, sample_r: int, sample_c: int) -> list[tuple[int, int]]:
    if direction == "up":
        return [(r, c) for r in range(sample_r - tail_len, sample_r) for c in range(sample_c - 2, sample_c + 3)]
    if direction == "down":
        return [(r, c) for r in range(sample_r + 1, sample_r + tail_len + 1) for c in range(sample_c - 2, sample_c + 3)]
    if direction == "left":
        return [(r, c) for r in range(sample_r - 2, sample_r + 3) for c in range(sample_c - tail_len, sample_c)]
    if direction == "right":
        return [(r, c) for r in range(sample_r - 2, sample_r + 3) for c in range(sample_c + 1, sample_c + tail_len + 1)]
    raise ValueError(direction)


def _projection_candidates(arrow: dict, shape: tuple[int, int]) -> list[list[tuple[int, int]]]:
    h, w = shape
    dr, dc = DIRS[arrow["direction"]]
    step = int(arrow["tail_len"]) + 1
    candidates = []
    idx = 0
    while True:
        sample_r = int(arrow["tip"][0] + dr * idx * step)
        sample_c = int(arrow["tip"][1] + dc * idx * step)
        cells = _stamp_cells(arrow["direction"], int(arrow["tail_len"]), sample_r, sample_c)
        if any(r < 0 or r >= h or c < 0 or c >= w for r, c in cells):
            break
        candidates.append(cells)
        idx += 1
    return candidates


def _simulate_shadows(grid: np.ndarray, arrows: list[dict] | None = None, bg: int | None = None) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    if bg is None:
        bg = _get_bg(g)
    if arrows is None:
        arrows = _detect_arrows(g, int(bg))

    out = g.copy()
    arrow_mask = g != int(bg)
    occupied = arrow_mask.copy()
    stats = {
        "painted_copies": 0,
        "painted_cells": 0,
        "blocked_total": 0,
        "blocked_by_arrow": 0,
        "blocked_by_shadow": 0,
    }

    for arrow in sorted(arrows, key=lambda item: (int(item["tip"][0]), int(item["tip"][1]), int(item["color"]))):
        for cells in _projection_candidates(arrow, (h, w)):
            overlap_arrow = any(bool(arrow_mask[r, c]) for r, c in cells)
            overlap_shadow = any(bool(occupied[r, c] and not arrow_mask[r, c]) for r, c in cells)
            if overlap_arrow or overlap_shadow:
                stats["blocked_total"] += 1
                if overlap_arrow:
                    stats["blocked_by_arrow"] += 1
                if overlap_shadow:
                    stats["blocked_by_shadow"] += 1
                continue
            for r, c in cells:
                out[r, c] = SHADOW
                occupied[r, c] = True
            stats["painted_copies"] += 1
            stats["painted_cells"] += len(cells)
    return out, stats


def _placed_arrow(direction: str, tail_len: int, top_left: tuple[int, int], color: int) -> dict:
    spec = SPEC_MAP[(direction, tail_len)]
    r0, c0 = map(int, top_left)
    cells = {(int(r0 + r), int(c0 + c)) for r, c in spec["coords"]}
    return {
        "direction": direction,
        "tail_len": int(tail_len),
        "color": int(color),
        "top_left": (r0, c0),
        "tip": (int(r0 + spec["tip"][0]), int(c0 + spec["tip"][1])),
        "cells": cells,
    }


def _expanded_cells(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    out = set()
    for r, c in cells:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                out.add((int(rr), int(cc)))
    return out


def _can_place(cells: set[tuple[int, int]], h: int, w: int, padded_occupied: set[tuple[int, int]]) -> bool:
    for r, c in cells:
        if r < 0 or r >= h or c < 0 or c >= w:
            return False
        if (int(r), int(c)) in padded_occupied:
            return False
    return True


def _random_arrow(
    rng: random.Random,
    h: int,
    w: int,
    color: int,
    padded_occupied: set[tuple[int, int]],
    min_candidates: int = 1,
) -> dict | None:
    for _ in range(300):
        direction = rng.choice(tuple(DIRS))
        tail_len = rng.randint(1, 3)
        box_h, box_w = SPEC_MAP[(direction, tail_len)]["bbox"]
        r0 = rng.randint(0, h - box_h)
        c0 = rng.randint(0, w - box_w)
        arrow = _placed_arrow(direction, tail_len, (r0, c0), color)
        if not _can_place(arrow["cells"], h, w, padded_occupied):
            continue
        if len(_projection_candidates(arrow, (h, w))) < min_candidates:
            continue
        return arrow
    return None


def _roomy_source_arrow(
    rng: random.Random,
    h: int,
    w: int,
    color: int,
    padded_occupied: set[tuple[int, int]],
) -> tuple[dict, list[list[tuple[int, int]]]] | None:
    for _ in range(400):
        direction = rng.choice(tuple(DIRS))
        tail_len = rng.randint(1, 3)
        box_h, box_w = SPEC_MAP[(direction, tail_len)]["bbox"]
        r0 = rng.randint(0, h - box_h)
        c0 = rng.randint(0, w - box_w)
        arrow = _placed_arrow(direction, tail_len, (r0, c0), color)
        if not _can_place(arrow["cells"], h, w, padded_occupied):
            continue
        candidates = _projection_candidates(arrow, (h, w))
        if len(candidates) < 3:
            continue
        return arrow, candidates
    return None


def _blocking_arrow(
    rng: random.Random,
    h: int,
    w: int,
    color: int,
    padded_occupied: set[tuple[int, int]],
    target_cells: set[tuple[int, int]],
    protected_cells: set[tuple[int, int]],
) -> dict | None:
    for _ in range(500):
        direction = rng.choice(tuple(DIRS))
        tail_len = rng.randint(1, 3)
        box_h, box_w = SPEC_MAP[(direction, tail_len)]["bbox"]
        r0 = rng.randint(0, h - box_h)
        c0 = rng.randint(0, w - box_w)
        arrow = _placed_arrow(direction, tail_len, (r0, c0), color)
        if not _can_place(arrow["cells"], h, w, padded_occupied):
            continue
        if not (arrow["cells"] & target_cells):
            continue
        if arrow["cells"] & protected_cells:
            continue
        if len(_projection_candidates(arrow, (h, w))) < 1:
            continue
        return arrow
    return None


def _paint_input(arrows: list[dict], shape: tuple[int, int]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for arrow in arrows:
        for r, c in arrow["cells"]:
            grid[r, c] = int(arrow["color"])
    return grid


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    require_block = bool(rng.randint(0, 1))
    if require_block:
        h = rng.randint(18, 27)
        w = rng.randint(18, 27)
        arrow_count = rng.randint(3, 4)
    else:
        h = rng.randint(18, 30)
        w = rng.randint(18, 30)
        arrow_count = rng.randint(2, 4)

    colors = rng.sample(PALETTE, arrow_count)
    arrows = []
    occupied = set()
    padded_occupied = set()

    if require_block:
        source_result = _roomy_source_arrow(rng, h, w, colors[0], padded_occupied)
        if source_result is None:
            return None
        source, source_candidates = source_result
        arrows.append(source)
        occupied |= source["cells"]
        padded_occupied |= _expanded_cells(source["cells"])

        protected = set(source_candidates[0]) | set(source_candidates[2])
        blocker = _blocking_arrow(
            rng,
            h,
            w,
            colors[1],
            padded_occupied,
            set(source_candidates[1]),
            protected,
        )
        if blocker is None:
            return None
        arrows.append(blocker)
        occupied |= blocker["cells"]
        padded_occupied |= _expanded_cells(blocker["cells"])
        remaining_colors = colors[2:]
    else:
        remaining_colors = colors

    for color in remaining_colors:
        arrow = _random_arrow(rng, h, w, color, padded_occupied, min_candidates=1)
        if arrow is None:
            return None
        arrows.append(arrow)
        occupied |= arrow["cells"]
        padded_occupied |= _expanded_cells(arrow["cells"])

    if len({arrow["direction"] for arrow in arrows}) == 1 and len(arrows) > 1:
        return None

    grid = _paint_input(arrows, (h, w))
    out, stats = _simulate_shadows(grid, arrows=arrows, bg=BG)

    if stats["painted_copies"] < max(2, len(arrows) - 1):
        return None
    if stats["painted_cells"] < 15:
        return None
    if stats["blocked_by_shadow"] != 0:
        return None
    if require_block:
        if stats["blocked_by_arrow"] < 1:
            return None
    elif stats["blocked_total"] != 0:
        return None

    try:
        if not np.array_equal(main(grid), out):
            return None
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid arrow-projection input")
