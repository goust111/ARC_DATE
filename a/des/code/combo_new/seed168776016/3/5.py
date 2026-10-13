from __future__ import annotations

import random

import numpy as np


BLACK = 0
RED = 2
GREY = 5
SHAPE_COLORS = (1, 3, 4, 6, 7, 8, 9)
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
NEIGHBORS8 = NEIGHBORS4 + ((1, 1), (1, -1), (-1, 1), (-1, -1))

MAX_GENERATION_ATTEMPTS = 400
MAX_SHAPE_ATTEMPTS = 120


def _find_components(mask: np.ndarray, connectivity: int = 8) -> list[dict[str, object]]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    neighbors = NEIGHBORS8 if connectivity == 8 else NEIGHBORS4
    components: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in neighbors:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            components.append({"coords": arr, "bbox": (int(r0), int(c0), int(r1), int(c1))})

    return components


def _objects_in_side(side: np.ndarray) -> list[dict[str, object]]:
    objects: list[dict[str, object]] = []
    for color in sorted(int(c) for c in np.unique(side) if int(c) not in (BLACK, GREY)):
        for comp in _find_components(side == color, connectivity=8):
            comp["color"] = color
            objects.append(comp)
    objects.sort(key=lambda obj: (obj["bbox"][0], obj["bbox"][1], obj["color"]))
    return objects


def _local_mask(side: np.ndarray, obj: dict[str, object]) -> np.ndarray:
    r0, c0, r1, c1 = obj["bbox"]
    color = int(obj["color"])
    return side[r0 : r1 + 1, c0 : c1 + 1] == color


def main(grid: np.ndarray) -> np.ndarray:
    """Highlight the pixels missing from each right-side shape by comparing it with its left-side twin.

    Input structure:
    - The background is black.
    - One solid grey vertical line splits the grid into left and right halves.
    - Each half contains the same number of monochrome shapes, ordered from top to bottom.
    - In every paired shape, the right shape has the same color and bounding-box size as the left shape but is missing 1 to 3 pixels.

    Rule:
    - Find the unique solid grey separator column and discard it from the output.
    - Read shapes on each side from top to bottom and pair them in that order.
    - For each pair, align their bounding boxes by top-left corner and find the cells present in the left shape but absent in the right shape.
    - Copy the right half into the output and paint those missing cells red at their corresponding coordinates in the right half.

    Key insight:
    - The right half is not repaired; it is preserved and only the left-vs-right pixel differences are highlighted in red.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    separator_cols = np.flatnonzero(np.all(g == GREY, axis=0))
    assert len(separator_cols) == 1, "expected one solid grey separator column"
    sep = int(separator_cols[0])
    assert 0 < sep < g.shape[1] - 1, "separator must leave content on both sides"

    left = g[:, :sep]
    right = g[:, sep + 1 :]

    left_objects = _objects_in_side(left)
    right_objects = _objects_in_side(right)
    assert left_objects and len(left_objects) == len(right_objects), "both sides must contain the same number of shapes"

    out = right.copy()
    for left_obj, right_obj in zip(left_objects, right_objects):
        assert int(left_obj["color"]) == int(right_obj["color"]), "paired shapes must share a color"
        left_mask = _local_mask(left, left_obj)
        right_mask = _local_mask(right, right_obj)
        assert left_mask.shape == right_mask.shape, "paired shapes must share a bounding-box size"
        assert np.all(right_mask <= left_mask), "right shape must be a subset of left shape"

        missing = np.argwhere(left_mask & ~right_mask)
        rr0, rc0, _, _ = right_obj["bbox"]
        for dr, dc in missing:
            out[rr0 + int(dr), rc0 + int(dc)] = RED

    return out


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False

    h, w = mask.shape
    start_r, start_c = map(int, coords[0])
    stack = [(start_r, start_c)]
    seen = {stack[0]}

    while stack:
        r, c = stack.pop()
        for dr, dc in NEIGHBORS4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))

    return len(seen) == len(coords)


def _bbox_preserved(mask: np.ndarray) -> bool:
    return bool(mask.any(axis=0).all() and mask.any(axis=1).all())


def _safe_remove_candidates(mask: np.ndarray) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        trial = mask.copy()
        trial[rr, cc] = False
        if _bbox_preserved(trial) and _is_connected(trial):
            candidates.append((rr, cc))
    return candidates


def _make_shape_pair(rng: random.Random, n_shapes: int, missing_count: int) -> tuple[np.ndarray, np.ndarray] | None:
    max_dim = 5 if n_shapes == 4 else 6

    for _ in range(MAX_SHAPE_ATTEMPTS):
        h = rng.randint(4, max_dim)
        w = rng.randint(4, max_dim)
        left = np.ones((h, w), dtype=bool)

        carve_steps = rng.randint(0, min(5, (h * w) // 3))
        for _ in range(carve_steps):
            candidates = _safe_remove_candidates(left)
            if not candidates:
                break
            rr, cc = rng.choice(candidates)
            left[rr, cc] = False

        if left.sum() < missing_count + 6:
            continue

        right = left.copy()
        removed = 0
        for _ in range(missing_count):
            candidates = _safe_remove_candidates(right)
            if not candidates:
                break
            rr, cc = rng.choice(candidates)
            right[rr, cc] = False
            removed += 1

        if removed == missing_count and not np.array_equal(left, right):
            return left, right

    return None


def _sample_colors(rng: random.Random, n_shapes: int) -> list[int]:
    if n_shapes >= 3 and rng.random() < 0.5:
        colors = [rng.choice(SHAPE_COLORS) for _ in range(n_shapes)]
        if len(set(colors)) == 1:
            replacement = rng.choice([c for c in SHAPE_COLORS if c != colors[0]])
            colors[-1] = replacement
        return colors
    return rng.sample(list(SHAPE_COLORS), n_shapes)


def _paint_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    h, w = mask.shape
    window = grid[top : top + h, left : left + w]
    assert window.shape == mask.shape
    assert np.all(window[mask] == BLACK), "objects must not overlap"
    window[mask] = color


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_shapes = rng.randint(2, 4)
    pairs: list[tuple[np.ndarray, np.ndarray, int]] = []

    for _ in range(n_shapes):
        missing_count = rng.randint(1, 3)
        pair = _make_shape_pair(rng, n_shapes, missing_count)
        if pair is None:
            return None
        left_mask, right_mask = pair
        pairs.append((left_mask, right_mask, missing_count))

    cursor = 1
    row_specs: list[tuple[int, int]] = []
    for idx, (left_mask, _right_mask, _missing_count) in enumerate(pairs):
        h = int(left_mask.shape[0])
        slack = rng.randint(0, 1)
        left_offset = rng.randint(0, slack) if slack else 0
        right_offset = rng.randint(0, slack) if slack else 0
        row_specs.append((cursor + left_offset, cursor + right_offset))
        cursor += h + slack
        if idx + 1 < n_shapes:
            cursor += rng.randint(1, 2)

    height = cursor + 1
    max_w = max(int(left_mask.shape[1]) for left_mask, _right_mask, _missing_count in pairs)
    left_w = max_w + rng.randint(3, 6)
    right_w = max_w + rng.randint(3, 6)
    width = left_w + 1 + right_w

    if height > 30 or width > 30:
        return None

    colors = _sample_colors(rng, n_shapes)
    grid = np.zeros((height, width), dtype=int)
    grid[:, left_w] = GREY

    total_missing = 0
    for (left_mask, right_mask, missing_count), color, (left_r, right_r) in zip(pairs, colors, row_specs):
        h, w = left_mask.shape
        max_left_c = left_w - w - 1
        max_right_c = right_w - w - 1
        if max_left_c < 1 or max_right_c < 1:
            return None

        left_c = rng.randint(1, max_left_c)
        right_c = rng.randint(1, max_right_c)
        _paint_mask(grid, left_mask, left_r, left_c, color)
        _paint_mask(grid, right_mask, right_r, left_w + 1 + right_c, color)
        total_missing += missing_count

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != (height, right_w):
        return None
    if int(np.count_nonzero(out == RED)) != total_missing:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
