from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5

MAX_GENERATION_ATTEMPTS = 500
MIN_BBOX_GAP = 2


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _rect_orientation(sprite: np.ndarray) -> str:
    h, w = sprite.shape
    if h == 3 and w >= 5 and w % 2 == 1:
        return "H"
    if w == 3 and h >= 5 and h % 2 == 1:
        return "V"
    raise ValueError("expected a 3-by-odd or odd-by-3 rectangle")


def _find_rectangles(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    rects = []
    for coords in _components(g == GREY):
        bbox = _bbox(coords)
        sprite = _crop(g, bbox)
        if not np.all(sprite == GREY):
            raise ValueError("grey components must be solid rectangles")
        orientation = _rect_orientation(sprite)
        rects.append(
            {
                "bbox": bbox,
                "coords": coords,
                "sprite": sprite,
                "orientation": orientation,
                "long": int(max(sprite.shape)),
            }
        )
    return rects


def _punch_rectangle(sprite: np.ndarray) -> np.ndarray:
    """
    Apply the task's alternating-center punch in the working orientation.

    Horizontal 3xodd rectangles use the original center-line single-cell holes.
    Vertical oddx3 rectangles turn alternating center-line positions into full
    3-cell cut rows, which makes connectivity depend on the long side length.
    """
    out = np.asarray(sprite, dtype=int).copy()
    orientation = _rect_orientation(out)
    if orientation == "H":
        out[1, 1 : out.shape[1] - 1 : 2] = BLACK
    else:
        out[2 : out.shape[0] - 1 : 2, :] = BLACK
    return out


def _erase_singletons(sprite: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    comps = _components(sprite == GREY)
    kept = [coords for coords in comps if len(coords) > 1]
    cleaned = np.zeros_like(sprite)
    for coords in kept:
        cleaned[coords[:, 0], coords[:, 1]] = GREY
    return cleaned, kept


def _category_color(count: int) -> int:
    if count == 1:
        return BLUE
    if count == 2:
        return RED
    return GREEN


def _solve_rectangle(sprite: np.ndarray) -> tuple[np.ndarray, int]:
    punched = _punch_rectangle(sprite)
    cleaned, comps = _erase_singletons(punched)
    count = len(comps)
    if count < 1:
        raise ValueError("degenerate punched rectangle")
    out = np.zeros_like(sprite)
    out[cleaned == GREY] = _category_color(count)
    return out, count


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    dst[r0 : r0 + h, c0 : c0 + w] = patch


def main(grid: np.ndarray) -> np.ndarray:
    """
    Punch each grey 3-wide rectangle along its center axis and recolor it by how many 4-connected pieces survive.

    Input structure:
    - The grid has a black background and 3 to 5 separated grey solid rectangles.
    - Every rectangle has short side length 3 and odd long side length 5 to 11.
    - Rectangles may be horizontal or vertical, and no two grey rectangles touch.

    Rule:
    - Detect every grey rectangle independently.
    - For a horizontal rectangle, punch alternating single-cell holes along the middle row, starting one cell in from the left end.
    - For a vertical rectangle, punch alternating full-width cut rows across the 3-cell short side, starting two cells in from the top end.
    - Inside each original rectangle area, find the remaining grey 4-connected components, erase any singleton component, and count the surviving components C.
    - Recolor the surviving pixels blue if C=1, red if C=2, and green if C>=3; everything else becomes black.

    Key insight:
    - The alternating punch must be simulated before any recolor decision: rectangles with the same size can end in different classes because orientation changes whether the punch stays local or becomes a through-cut.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("expected a 2D grid")

    out = np.zeros_like(g)
    for rect in _find_rectangles(g):
        r0, c0, _r1, _c1 = rect["bbox"]
        solved, _count = _solve_rectangle(rect["sprite"])
        _paste(out, solved, r0, c0)
    return out


def _bbox_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = MIN_BBOX_GAP) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    rows_overlap = not (ar1 + gap < br0 or br1 + gap < ar0)
    cols_overlap = not (ac1 + gap < bc0 or bc1 + gap < ac0)
    return not (rows_overlap and cols_overlap)


def _place_rectangles(
    shape: tuple[int, int], dims: list[tuple[int, int]], rng: random.Random
) -> list[tuple[int, int, int, int]] | None:
    h, w = shape
    placed: list[tuple[int, int, int, int]] = []
    order = sorted(range(len(dims)), key=lambda i: (max(dims[i]), dims[i][0] * dims[i][1]), reverse=True)
    results: list[tuple[int, int, int, int] | None] = [None] * len(dims)
    for idx in order:
        rh, rw = dims[idx]
        candidates = []
        for r0 in range(h - rh + 1):
            for c0 in range(w - rw + 1):
                bbox = (r0, c0, r0 + rh - 1, c0 + rw - 1)
                if all(_bbox_separated(bbox, other) for other in placed):
                    candidates.append(bbox)
        if not candidates:
            return None
        bbox = rng.choice(candidates)
        placed.append(bbox)
        results[idx] = bbox
    return [bbox for bbox in results if bbox is not None]


def _sample_specs(rng: random.Random, count: int) -> list[dict]:
    specs = [
        {"kind": "blue", "dims": (3, 5)},
        {"kind": "red", "dims": (5, 3)},
        {"kind": "green", "dims": (rng.choice([7, 9, 11]), 3)},
    ]
    while len(specs) < count:
        choice = rng.choice(["blue", "red", "green", "blue-long"])
        if choice == "blue":
            specs.append({"kind": "blue", "dims": (3, 5)})
        elif choice == "blue-long":
            specs.append({"kind": "blue", "dims": (3, rng.choice([7, 9, 11]))})
        elif choice == "red":
            specs.append({"kind": "red", "dims": (5, 3)})
        else:
            specs.append({"kind": "green", "dims": (rng.choice([7, 9, 11]), 3)})
    rng.shuffle(specs)
    return specs


def _build_input(specs: list[dict], rng: random.Random) -> np.ndarray | None:
    max_h = max(spec["dims"][0] for spec in specs)
    max_w = max(spec["dims"][1] for spec in specs)
    height = rng.randint(max(16, max_h + 4), 24)
    width = rng.randint(max(16, max_w + 4), 24)
    bboxes = _place_rectangles((height, width), [spec["dims"] for spec in specs], rng)
    if bboxes is None:
        return None

    grid = np.zeros((height, width), dtype=int)
    for spec, (r0, c0, r1, c1) in zip(specs, bboxes):
        grid[r0 : r1 + 1, c0 : c1 + 1] = GREY
    return grid


def _semantic_validation(grid: np.ndarray) -> bool:
    try:
        rects = _find_rectangles(grid)
    except ValueError:
        return False
    if not (3 <= len(rects) <= 5):
        return False

    buckets = set()
    lengths = []
    vertical_buckets = set()
    has_horizontal_len5 = False
    has_vertical_len5 = False
    has_vertical_long = False

    for rect in rects:
        sprite = rect["sprite"]
        orientation = rect["orientation"]
        long_side = rect["long"]
        solved, count = _solve_rectangle(sprite)
        color_set = set(map(int, np.unique(solved))) - {BLACK}
        if len(color_set) != 1:
            return False
        color = next(iter(color_set))
        if color != _category_color(count):
            return False

        lengths.append(long_side)
        bucket = 1 if count == 1 else 2 if count == 2 else 3
        buckets.add(bucket)
        if orientation == "V":
            vertical_buckets.add(bucket)
        if orientation == "H" and long_side == 5 and count == 1:
            has_horizontal_len5 = True
        if orientation == "V" and long_side == 5 and count == 2:
            has_vertical_len5 = True
        if orientation == "V" and long_side >= 7 and count >= 3:
            has_vertical_long = True

    if not (has_horizontal_len5 and has_vertical_len5 and has_vertical_long):
        return False
    if buckets != {1, 2, 3}:
        return False
    if vertical_buckets != {2, 3}:
        return False
    if 5 not in lengths:
        return False

    out = main(grid)
    out_colors = set(map(int, np.unique(out))) - {BLACK}
    if not {BLUE, RED, GREEN}.issubset(out_colors):
        return False

    seen_same_length_counterexample = False
    for i, rect_a in enumerate(rects):
        solved_a, count_a = _solve_rectangle(rect_a["sprite"])
        for rect_b in rects[i + 1 :]:
            solved_b, count_b = _solve_rectangle(rect_b["sprite"])
            if rect_a["long"] == rect_b["long"] and count_a != count_b:
                seen_same_length_counterexample = True
                break
        if seen_same_length_counterexample:
            break
    if not seen_same_length_counterexample:
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        rng = random.Random()
        specs = _sample_specs(rng, rng.randint(3, 5))
        grid = _build_input(specs, rng)
        if grid is not None and _semantic_validation(grid):
            return grid
    raise RuntimeError("failed to generate a valid input")
