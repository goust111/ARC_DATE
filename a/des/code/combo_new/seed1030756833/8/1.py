from __future__ import annotations

from collections import deque
import random

import numpy as np


GREY = 5
DIRECTIONS = ("top", "bottom", "left", "right")
BG_OPTIONS = (0, 7)
SEED_PALETTE = (1, 2, 3, 4, 6, 8, 9)
GEN_RETRIES = 400


def _most_common_color(g: np.ndarray) -> int:
    vals, counts = np.unique(g, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not bool(mask[r, c]):
                continue
            comp: list[tuple[int, int]] = []
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            while q:
                cr, cc = q.popleft()
                comp.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(int(cr), int(cc)):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _rect_border(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    pts: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        pts.add((r0, c))
        pts.add((r1, c))
    for r in range(r0, r1 + 1):
        pts.add((r, c0))
        pts.add((r, c1))
    return pts


def _gap_direction(gap: tuple[int, int], bbox: tuple[int, int, int, int]) -> str:
    r, c = gap
    r0, c0, r1, c1 = bbox
    assert (r, c) not in ((r0, c0), (r0, c1), (r1, c0), (r1, c1)), "gap must not be a corner"
    if r == r0:
        return "top"
    if r == r1:
        return "bottom"
    if c == c0:
        return "left"
    if c == c1:
        return "right"
    raise AssertionError("gap must lie on the rectangle border")


def _frame_color(g: np.ndarray, bg: int) -> int:
    if bool(np.any(g == GREY)):
        return GREY
    vals, counts = np.unique(g[g != bg], return_counts=True)
    assert vals.size, "expected at least one non-background color"
    return int(vals[int(np.argmax(counts))])


def _find_frames(g: np.ndarray, frame_color: int) -> list[dict[str, object]]:
    frames: list[dict[str, object]] = []
    for comp in _components(g == frame_color):
        rows = [p[0] for p in comp]
        cols = [p[1] for p in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        assert r1 - r0 >= 2 and c1 - c0 >= 2, "frame must have an interior"

        border = _rect_border(r0, c0, r1, c1)
        present = set(comp)
        missing = sorted(border - present)
        assert not (present - border), "frame color must lie only on the rectangle border"
        assert len(missing) == 1, "each frame must have exactly one missing border pixel"

        bbox = (int(r0), int(c0), int(r1), int(c1))
        direction = _gap_direction(missing[0], bbox)
        frames.append({"bbox": bbox, "gap": missing[0], "direction": direction})
    frames.sort(key=lambda f: tuple(f["bbox"]))  # deterministic processing order
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each seed into an interior stripe pushed away from its frame gap.

    Input structure:
    - The most common color is the background.
    - Grey pixels form one or more separated one-pixel-wide rectangular frames.
    - Each frame is missing exactly one non-corner border pixel on its top,
      bottom, left, or right side.
    - Inside each frame there is exactly one non-background, non-Grey seed pixel.

    Rule:
    - Detect the side of the missing Grey border pixel for each frame.
    - If the gap is on the top or bottom, fill the seed's whole interior column
      with the seed color.
    - If the gap is on the left or right, fill the seed's whole interior row
      with the seed color.
    - Never recolor Grey frame pixels or cells outside the frame interior.

    Key insight:
    - The gap gives the direction of pressure, so the visible fill axis is chosen
      only after reversing that gap direction relative to the seed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _most_common_color(g)
    frame_color = _frame_color(g, bg)
    frames = _find_frames(g, frame_color)
    assert frames, "expected at least one gapped rectangular frame"

    out = g.copy()
    for frame in frames:
        r0, c0, r1, c1 = map(int, frame["bbox"])
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        seed_locs = np.argwhere((interior != bg) & (interior != frame_color))
        assert seed_locs.shape[0] == 1, "each frame interior must contain exactly one seed"
        sr = int(r0 + 1 + seed_locs[0, 0])
        sc = int(c0 + 1 + seed_locs[0, 1])
        seed_color = int(g[sr, sc])

        if frame["direction"] in ("top", "bottom"):
            out[r0 + 1 : r1, sc] = seed_color
        else:
            out[sr, c0 + 1 : c1] = seed_color

    return out


def _paint_frame(
    grid: np.ndarray,
    bg: int,
    r0: int,
    c0: int,
    oh: int,
    ow: int,
    direction: str,
    seed_color: int,
) -> dict[str, object]:
    r1 = int(r0 + oh - 1)
    c1 = int(c0 + ow - 1)

    for c in range(c0, c1 + 1):
        grid[r0, c] = GREY
        grid[r1, c] = GREY
    for r in range(r0, r1 + 1):
        grid[r, c0] = GREY
        grid[r, c1] = GREY

    if direction == "top":
        gap = (r0, random.randint(c0 + 1, c1 - 1))
        sr = random.randint(r0 + 1, r1 - 2)
        sc = random.randint(c0 + 1, c1 - 1)
    elif direction == "bottom":
        gap = (r1, random.randint(c0 + 1, c1 - 1))
        sr = random.randint(r0 + 2, r1 - 1)
        sc = random.randint(c0 + 1, c1 - 1)
    elif direction == "left":
        gap = (random.randint(r0 + 1, r1 - 1), c0)
        sr = random.randint(r0 + 1, r1 - 1)
        sc = random.randint(c0 + 1, c1 - 2)
    else:
        gap = (random.randint(r0 + 1, r1 - 1), c1)
        sr = random.randint(r0 + 1, r1 - 1)
        sc = random.randint(c0 + 2, c1 - 1)

    grid[gap] = int(bg)
    grid[sr, sc] = int(seed_color)
    return {
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "direction": direction,
        "seed": (int(sr), int(sc)),
        "seed_color": int(seed_color),
    }


def _expected_from_specs(grid: np.ndarray, specs: list[dict[str, object]]) -> np.ndarray:
    out = grid.copy()
    for spec in specs:
        r0, c0, r1, c1 = map(int, spec["bbox"])
        sr, sc = map(int, spec["seed"])
        seed_color = int(spec["seed_color"])
        if spec["direction"] in ("top", "bottom"):
            out[r0 + 1 : r1, sc] = seed_color
        else:
            out[sr, c0 + 1 : c1] = seed_color
    return out


def _max_object_count(h: int, w: int) -> int:
    area = int(h) * int(w)
    if area >= 520:
        return 4
    if area >= 300:
        return 3
    return 2


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(12, 30)
    w = random.randint(12, 30)
    bg = random.choice(BG_OPTIONS)
    grid = np.full((h, w), bg, dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    count = random.randint(1, _max_object_count(h, w))
    directions = list(DIRECTIONS)
    random.shuffle(directions)
    directions = directions[:count]
    seed_colors = random.sample([c for c in SEED_PALETTE if c != bg and c != GREY], count)

    specs: list[dict[str, object]] = []
    for direction, seed_color in zip(directions, seed_colors):
        placed = False
        for _ in range(120):
            oh = random.randint(5, min(9, h - 2))
            ow = random.randint(5, min(9, w - 2))
            r0 = random.randint(1, h - oh - 1)
            c0 = random.randint(1, w - ow - 1)
            r1 = r0 + oh - 1
            c1 = c0 + ow - 1
            rr0, rr1 = max(0, r0 - 1), min(h, r1 + 2)
            cc0, cc1 = max(0, c0 - 1), min(w, c1 + 2)
            if bool(reserved[rr0:rr1, cc0:cc1].any()):
                continue

            spec = _paint_frame(grid, bg, r0, c0, oh, ow, direction, seed_color)
            reserved[rr0:rr1, cc0:cc1] = True
            specs.append(spec)
            placed = True
            break
        if not placed:
            return None

    if _most_common_color(grid) != bg:
        return None

    expected = _expected_from_specs(grid, specs)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(actual, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid seed1030756833/8/1 input")
