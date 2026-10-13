from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
BODY = 1
MARKER = 2
MAX_ATTEMPTS = 400


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _random_manhattan_path(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = start
    er, ec = end
    path = [(r, c)]
    while (r, c) != (er, ec):
        choices = []
        if r < er:
            choices.append((1, 0))
        elif r > er:
            choices.append((-1, 0))
        if c < ec:
            choices.append((0, 1))
        elif c > ec:
            choices.append((0, -1))
        dr, dc = random.choice(choices)
        r += dr
        c += dc
        path.append((r, c))
    return path


def _rotations_are_distinct(mask: np.ndarray, count: int) -> bool:
    variants = [np.rot90(mask, k=-i) for i in range(count)]
    for i in range(len(variants)):
        for j in range(i + 1, len(variants)):
            if np.array_equal(variants[i], variants[j]):
                return False
    return True


def _build_mask(side: int, target_area: int, copy_count: int) -> np.ndarray | None:
    for _ in range(300):
        hub = (random.randrange(side), random.randrange(side))
        anchors = [
            (0, random.randrange(side)),
            (side - 1, random.randrange(side)),
            (random.randrange(side), 0),
            (random.randrange(side), side - 1),
        ]
        cells = {hub}
        for anchor in anchors:
            cells.update(_random_manhattan_path(anchor, hub))
        if len(cells) > target_area:
            continue

        while len(cells) < target_area:
            frontier = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < side and 0 <= nc < side and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))

        if len(cells) != target_area:
            continue

        mask = np.zeros((side, side), dtype=bool)
        for r, c in cells:
            mask[r, c] = True

        box = _bbox(mask)
        if box != (0, 0, side - 1, side - 1):
            continue
        if len(_components4(mask)) != 1:
            continue
        if not _rotations_are_distinct(mask, copy_count):
            continue
        return mask
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the marked sprite, count its red markers, and concatenate progressively rotated blue copies.
    Input structure:
    - The grid has a black background and exactly one 4-connected sprite made of blue body cells plus red marker cells.
    - The sprite's tight bounding box is an odd-sided square, so 90-degree rotations keep the same size.
    Rule:
    - Crop the sprite to its minimal bounding box and count the red marker cells inside it as K.
    - Make K copies of that crop; copy i is rotated clockwise by i*90 degrees for i=0..K-1.
    - In every copy, repaint all red marker cells to blue, then place the copies left-to-right with one black column between neighbors.
    Key insight:
    - The red markers are disposable control pixels: they encode both how many copies to make and which rotation sequence to use, then disappear in the output.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _most_common_color(g)
    fg_mask = g != bg
    sprite_box = _bbox(fg_mask)
    assert sprite_box is not None, "expected one foreground sprite"
    assert len(_components4(fg_mask)) == 1, "expected a single connected sprite"

    sprite = _crop_bbox(g, sprite_box)
    copy_count = int(np.sum(sprite == MARKER))
    assert copy_count >= 1, "expected at least one marker"

    copies = []
    for i in range(copy_count):
        rotated = np.rot90(sprite, k=-i)
        cleaned = rotated.copy()
        cleaned[cleaned == MARKER] = BODY
        copies.append(cleaned)

    if len(copies) == 1:
        return copies[0]

    separator = np.full((sprite.shape[0], 1), bg, dtype=int)
    out = copies[0]
    for patch in copies[1:]:
        out = np.concatenate((out, separator, patch), axis=1)
    return out


def _sample_canvas(side: int) -> tuple[int, int]:
    min_h = max(8, side + 2)
    max_h = max(min_h, side + 7)
    min_w = max(8, side + 2)
    max_w = max(min_w, side + 7)
    return random.randint(min_h, max_h), random.randint(min_w, max_w)


def _oracle_checks(grid: np.ndarray, copy_count: int, blue_count: int) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        return False
    if max(grid.shape) > 30:
        return False
    if set(np.unique(grid)) - {BG, BODY, MARKER}:
        return False

    fg_mask = grid != BG
    if len(_components4(fg_mask)) != 1:
        return False
    box = _bbox(fg_mask)
    if box is None:
        return False
    r0, c0, r1, c1 = box
    side = r1 - r0 + 1
    if side != c1 - c0 + 1 or side % 2 == 0:
        return False
    if int(np.sum(grid == MARKER)) != copy_count:
        return False
    if int(np.sum(grid == BODY)) != blue_count:
        return False

    out = main(grid)
    if out.ndim != 2 or max(out.shape) > 30:
        return False
    if set(np.unique(out)) - {BG, BODY}:
        return False
    expected_width = side * copy_count + (copy_count - 1)
    return out.shape == (side, expected_width)


def _gen_attempt(forced_copy_count: int | None = None) -> np.ndarray | None:
    copy_count = forced_copy_count if forced_copy_count is not None else random.choice((2, 3))
    blue_count = random.randint(5, 8)
    target_area = blue_count + copy_count

    side_choices = [5]
    if target_area <= 9:
        side_choices.append(3)
    side = random.choice(side_choices)

    mask = _build_mask(side, target_area, copy_count)
    if mask is None:
        return None

    sprite = np.zeros((side, side), dtype=int)
    coords = [tuple(coord) for coord in np.argwhere(mask)]
    marker_coords = set(random.sample(coords, copy_count))
    for r, c in coords:
        sprite[r, c] = MARKER if (r, c) in marker_coords else BODY

    height, width = _sample_canvas(side)
    if max(height, width) > 30:
        return None
    grid = np.zeros((height, width), dtype=int)
    r0 = random.randint(0, height - side)
    c0 = random.randint(0, width - side)
    grid[r0 : r0 + side, c0 : c0 + side] = sprite

    return grid if _oracle_checks(grid, copy_count, blue_count) else None


_NEXT_COPY_COUNT = random.choice((2, 3))


def generate_input() -> np.ndarray:
    global _NEXT_COPY_COUNT
    preferred = _NEXT_COPY_COUNT
    _NEXT_COPY_COUNT = 5 - preferred
    for attempt in range(MAX_ATTEMPTS):
        forced = preferred if attempt < MAX_ATTEMPTS // 2 else None
        grid = _gen_attempt(forced_copy_count=forced)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
