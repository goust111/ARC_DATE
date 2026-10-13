from __future__ import annotations

import random
from collections import deque
from typing import Iterable

import numpy as np


BLACK = 0
ORANGE = 7
COLOR_POOL = (1, 2, 3, 4, 5, 6, 8, 9)
CANVAS_SIZE = 30
GEN_RETRIES = 500


def _neighbors4(r: int, c: int, h: int, w: int) -> Iterable[tuple[int, int]]:
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _color_components(grid: np.ndarray, color: int) -> list[dict[str, object]]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []
    coords = np.argwhere(grid == color)

    for r0, c0 in coords:
        r0 = int(r0)
        c0 = int(c0)
        if visited[r0, c0]:
            continue
        queue = deque([(r0, c0)])
        visited[r0, c0] = True
        pts: list[tuple[int, int]] = []

        while queue:
            r, c = queue.popleft()
            pts.append((r, c))
            for nr, nc in _neighbors4(r, c, h, w):
                if not visited[nr, nc] and int(grid[nr, nc]) == color:
                    visited[nr, nc] = True
                    queue.append((nr, nc))

        rows = [r for r, _ in pts]
        cols = [c for _, c in pts]
        br0, br1 = min(rows), max(rows)
        bc0, bc1 = min(cols), max(cols)
        mask = np.zeros((br1 - br0 + 1, bc1 - bc0 + 1), dtype=bool)
        for r, c in pts:
            mask[r - br0, c - bc0] = True
        comps.append({"bbox": (br0, bc0, br1, bc1), "mask": mask})

    return comps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Merge each same-colored fragment pair and mark the mismatching cells in orange.

    Input structure:
    - The background is black.
    - Foreground consists of monochromatic 4-connected objects scattered around the grid.
    - Each foreground color appears in exactly two separate objects, and those two objects have equal bounding-box dimensions.
    - Inside each bounding box, the two same-colored shapes are almost the same but differ on a few filled cells.

    Rule:
    - Pair the two objects of each color.
    - Align the two bounding boxes of a pair by matching their top-left corners, then compare the local fill masks cell by cell.
    - Keep the original color where both objects are filled, paint orange where exactly one object is filled, and leave black where both cells are empty.
    - Output only the merged pair results, laid out horizontally with one black separator column, ordered by the reading-order top-left coordinate of the upper-left representative object from each pair.

    Key insight:
    - Color tells which fragments belong together, but the output is not a plain union: solving requires overlaying each matched pair and turning the symmetric-difference cells into orange markers.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    colors = [int(c) for c in np.unique(g) if int(c) != BLACK]
    assert colors, "expected at least one foreground color"

    merged_pairs: list[tuple[tuple[int, int], np.ndarray]] = []
    for color in colors:
        comps = _color_components(g, color)
        assert len(comps) == 2, f"color {color} must form exactly two objects"

        mask_a = comps[0]["mask"]
        mask_b = comps[1]["mask"]
        assert isinstance(mask_a, np.ndarray) and isinstance(mask_b, np.ndarray)
        assert mask_a.shape == mask_b.shape, f"color {color} must have matching bbox sizes"

        merged = np.zeros(mask_a.shape, dtype=int)
        both = mask_a & mask_b
        diff = mask_a ^ mask_b
        merged[both] = color
        merged[diff] = ORANGE

        box_a = comps[0]["bbox"]
        box_b = comps[1]["bbox"]
        assert isinstance(box_a, tuple) and isinstance(box_b, tuple)
        rep = min((int(box_a[0]), int(box_a[1])), (int(box_b[0]), int(box_b[1])))
        merged_pairs.append((rep, merged))

    merged_pairs.sort(key=lambda item: (item[0][0], item[0][1]))

    out_h = max(int(block.shape[0]) for _, block in merged_pairs)
    out_w = sum(int(block.shape[1]) for _, block in merged_pairs) + len(merged_pairs) - 1
    assert out_h <= CANVAS_SIZE and out_w <= CANVAS_SIZE

    out = np.zeros((out_h, out_w), dtype=int)
    cursor = 0
    for i, (_rep, block) in enumerate(merged_pairs):
        h, w = block.shape
        out[:h, cursor : cursor + w] = block
        cursor += w
        if i + 1 < len(merged_pairs):
            cursor += 1
    return out


def _mask_connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return False
    h, w = mask.shape
    start = tuple(int(x) for x in pts[0])
    seen = {start}
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        for nr, nc in _neighbors4(r, c, h, w):
            if mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                queue.append((nr, nc))
    return len(seen) == int(mask.sum())


def _touches_all_sides(mask: np.ndarray) -> bool:
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _valid_object_mask(mask: np.ndarray) -> bool:
    return bool(mask.sum() > 0 and not mask.all() and _touches_all_sides(mask) and _mask_connected(mask))


def _random_path(
    rng: random.Random, start: tuple[int, int], goal: tuple[int, int]
) -> list[tuple[int, int]]:
    r, c = start
    out = [(r, c)]
    while (r, c) != goal:
        candidates: list[tuple[int, int]] = []
        if r < goal[0]:
            candidates.append((r + 1, c))
        elif r > goal[0]:
            candidates.append((r - 1, c))
        if c < goal[1]:
            candidates.append((r, c + 1))
        elif c > goal[1]:
            candidates.append((r, c - 1))
        r, c = rng.choice(candidates)
        out.append((r, c))
    return out


def _frontier_cells(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    frontier: set[tuple[int, int]] = set()
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        for nr, nc in _neighbors4(r, c, h, w):
            if not mask[nr, nc]:
                frontier.add((nr, nc))
    return list(frontier)


def _removable_cells(mask: np.ndarray) -> list[tuple[int, int]]:
    removable: list[tuple[int, int]] = []
    filled = np.argwhere(mask)
    for r, c in filled:
        r = int(r)
        c = int(c)
        trial = mask.copy()
        trial[r, c] = False
        if _valid_object_mask(trial):
            removable.append((r, c))
    return removable


def _random_base_mask(rng: random.Random, h: int, w: int) -> np.ndarray | None:
    for _ in range(120):
        if h >= 4 and w >= 4 and rng.random() < 0.3:
            mask = np.zeros((h, w), dtype=bool)
            mask[0, :] = True
            mask[-1, :] = True
            mask[:, 0] = True
            mask[:, -1] = True
            extras = rng.randint(0, max(1, (h - 2) * (w - 2) // 3))
            for _ in range(extras):
                frontier = [
                    (r, c)
                    for r in range(1, h - 1)
                    for c in range(1, w - 1)
                    if not mask[r, c]
                    and any(mask[nr, nc] for nr, nc in _neighbors4(r, c, h, w))
                ]
                if not frontier:
                    break
                r, c = rng.choice(frontier)
                mask[r, c] = True
            if _valid_object_mask(mask):
                return mask

        mask = np.zeros((h, w), dtype=bool)
        center = (rng.randrange(h), rng.randrange(w))
        anchors = [
            (0, rng.randrange(w)),
            (h - 1, rng.randrange(w)),
            (rng.randrange(h), 0),
            (rng.randrange(h), w - 1),
        ]
        for anchor in anchors:
            for r, c in _random_path(rng, anchor, center):
                mask[r, c] = True

        extra_budget = max(0, min(h * w - int(mask.sum()) - 1, max(2, (h * w) // 3)))
        for _ in range(rng.randint(0, extra_budget)):
            frontier = _frontier_cells(mask)
            if not frontier:
                break
            r, c = rng.choice(frontier)
            mask[r, c] = True

        if _valid_object_mask(mask):
            return mask
    return None


def _mutate_mask(
    rng: random.Random, base: np.ndarray, allow_identity: bool
) -> np.ndarray | None:
    mask = base.copy()
    max_steps = max(1, min(4, base.size // 4))
    steps = rng.randint(0 if allow_identity else 1, max_steps)
    changed = False

    for _ in range(steps):
        addable = _frontier_cells(mask) if int(mask.sum()) < mask.size - 1 else []
        removable = _removable_cells(mask) if int(mask.sum()) > 4 else []
        actions = []
        if addable:
            actions.append(("add", addable))
        if removable:
            actions.append(("remove", removable))
        if not actions:
            break

        action, cells = rng.choice(actions)
        r, c = rng.choice(cells)
        mask[r, c] = action == "add"
        changed = True

    if not changed and not allow_identity:
        return None
    return mask if _valid_object_mask(mask) else None


def _generate_pair_masks(
    rng: random.Random, h: int, w: int
) -> tuple[np.ndarray, np.ndarray] | None:
    max_diff = max(1, min(6, (h * w) // 2))
    min_overlap = max(3, (h * w) // 3)

    for _ in range(160):
        base = _random_base_mask(rng, h, w)
        if base is None:
            continue

        for _ in range(80):
            first = _mutate_mask(rng, base, allow_identity=True)
            second = _mutate_mask(rng, base, allow_identity=False)
            if first is None or second is None:
                continue
            if np.array_equal(first, second):
                continue

            diff = int(np.count_nonzero(first ^ second))
            overlap = int(np.count_nonzero(first & second))
            if 1 <= diff <= max_diff and overlap >= min_overlap:
                return first, second
    return None


def _place_objects(
    rng: random.Random, objects: list[tuple[int, np.ndarray]]
) -> np.ndarray | None:
    grid = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=int)
    occupied = np.zeros_like(grid, dtype=bool)

    shuffled = list(objects)
    rng.shuffle(shuffled)

    for color, mask in shuffled:
        h, w = mask.shape
        placed = False
        for _ in range(300):
            r0 = rng.randint(0, CANVAS_SIZE - h)
            c0 = rng.randint(0, CANVAS_SIZE - w)
            rr0 = max(0, r0 - 1)
            rr1 = min(CANVAS_SIZE, r0 + h + 1)
            cc0 = max(0, c0 - 1)
            cc1 = min(CANVAS_SIZE, c0 + w + 1)
            if occupied[rr0:rr1, cc0:cc1].any():
                continue

            view = grid[r0 : r0 + h, c0 : c0 + w]
            view[mask] = color
            occupied[rr0:rr1, cc0:cc1] = True
            placed = True
            break
        if not placed:
            return None

    return grid


def _crop_with_margin(rng: random.Random, grid: np.ndarray) -> np.ndarray:
    pts = np.argwhere(grid != BLACK)
    r0, c0 = np.min(pts, axis=0)
    r1, c1 = np.max(pts, axis=0)
    r0 = int(r0)
    c0 = int(c0)
    r1 = int(r1)
    c1 = int(c1)

    top = rng.randint(0, min(4, r0))
    left = rng.randint(0, min(4, c0))
    bottom = rng.randint(0, min(4, CANVAS_SIZE - 1 - r1))
    right = rng.randint(0, min(4, CANVAS_SIZE - 1 - c1))
    return grid[r0 - top : r1 + bottom + 1, c0 - left : c1 + right + 1]


def _has_unambiguous_pair_representatives(grid: np.ndarray) -> bool:
    for color in [int(c) for c in np.unique(grid) if int(c) != BLACK]:
        comps = _color_components(grid, color)
        if len(comps) != 2:
            return False
        a = comps[0]["bbox"]
        b = comps[1]["bbox"]
        assert isinstance(a, tuple) and isinstance(b, tuple)
        ra, ca = int(a[0]), int(a[1])
        rb, cb = int(b[0]), int(b[1])
        if not ((ra <= rb and ca <= cb) or (rb <= ra and cb <= ca)):
            return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    pair_count = rng.randint(2, 4)
    colors = rng.sample(COLOR_POOL, pair_count)

    specs: list[tuple[int, np.ndarray, np.ndarray]] = []
    output_width = pair_count - 1
    max_height = 0

    for color in colors:
        h = rng.randint(3, 6)
        w = rng.randint(3, 6)
        masks = _generate_pair_masks(rng, h, w)
        if masks is None:
            return None
        mask_a, mask_b = masks
        specs.append((color, mask_a, mask_b))
        output_width += w
        max_height = max(max_height, h)

    if output_width > CANVAS_SIZE or max_height > CANVAS_SIZE:
        return None

    objects: list[tuple[int, np.ndarray]] = []
    for color, mask_a, mask_b in specs:
        objects.append((color, mask_a))
        objects.append((color, mask_b))

    grid = _place_objects(rng, objects)
    if grid is None:
        return None

    cropped = _crop_with_margin(rng, grid)
    if not _has_unambiguous_pair_representatives(cropped):
        return None

    try:
        out = main(cropped)
    except AssertionError:
        return None

    if cropped.ndim != 2 or out.ndim != 2:
        return None
    if max(cropped.shape) > CANVAS_SIZE or max(out.shape) > CANVAS_SIZE:
        return None

    return cropped


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted attempts")
