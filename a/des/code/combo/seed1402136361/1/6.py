from __future__ import annotations

from collections import Counter
import random

import numpy as np


_DIRS = {
    "ul": (-1, -1),
    "ur": (-1, 1),
    "dl": (1, -1),
    "dr": (1, 1),
}

_SHAPE_SIZE_RANGE = (4, 6)
_OBJECT_COUNT_RANGE = (4, 5)
_GRID_SIZE_RANGE = (20, 28)
_MAX_GENERATION_ATTEMPTS = 400


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(np.array(coords, dtype=int))
    return comps


def _marker_and_shift(grid: np.ndarray, coords: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    colors = [int(grid[int(r), int(c)]) for r, c in coords]
    counts = Counter(colors)
    marker_pixels = [(int(r), int(c)) for r, c in coords if int(counts[int(grid[int(r), int(c)])]) == 1]
    assert len(marker_pixels) == 1, "each object must contain exactly one singleton marker color"
    marker_r, marker_c = marker_pixels[0]

    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    rel_r = int(marker_r - r0)
    rel_c = int(marker_c - c0)

    top = 2 * rel_r < h - 1
    bottom = 2 * rel_r > h - 1
    left = 2 * rel_c < w - 1
    right = 2 * rel_c > w - 1
    assert top != bottom and left != right, "marker must lie in a strict bbox quadrant"

    shift_r = -1 if top else 1
    shift_c = -1 if left else 1
    return (int(marker_r), int(marker_c)), (int(shift_r), int(shift_c))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each 4-connected foreground object diagonally.

    The object's unique singleton-color pixel indicates the diagonal direction
    by its quadrant inside the object's bounding box, and the translation
    distance equals the total number of pixels in the object.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    comps = _components(g != int(bg))

    out = np.full_like(g, int(bg))
    h, w = g.shape

    for coords in comps:
        area = int(coords.shape[0])
        _marker, (dir_r, dir_c) = _marker_and_shift(g, coords)
        dr = int(dir_r * area)
        dc = int(dir_c * area)
        for r, c in coords:
            rr = int(r + dr)
            cc = int(c + dc)
            if 0 <= rr < h and 0 <= cc < w:
                out[rr, cc] = int(g[int(r), int(c)])
    return out


def _quadrant_cells(height: int, width: int, dir_key: str) -> list[tuple[int, int]]:
    dir_r, dir_c = _DIRS[dir_key]
    cells: list[tuple[int, int]] = []
    for r in range(height):
        for c in range(width):
            top = 2 * r < height - 1
            bottom = 2 * r > height - 1
            left = 2 * c < width - 1
            right = 2 * c > width - 1
            row_ok = top if dir_r < 0 else bottom
            col_ok = left if dir_c < 0 else right
            if row_ok and col_ok:
                cells.append((int(r), int(c)))
    return cells


def _choose_hub(height: int, width: int, dir_key: str) -> tuple[int, int]:
    candidates = _quadrant_cells(height, width, dir_key)
    interior = [(r, c) for r, c in candidates if 0 < r < height - 1 and 0 < c < width - 1]
    pool = interior if interior else candidates
    return random.choice(pool)


def _manhattan_path(a: tuple[int, int], b: tuple[int, int], row_first: bool) -> list[tuple[int, int]]:
    r, c = a
    tr, tc = b
    path = [(int(r), int(c))]
    if row_first:
        step = 1 if tr > r else -1
        while r != tr:
            r += step
            path.append((int(r), int(c)))
        step = 1 if tc > c else -1
        while c != tc:
            c += step
            path.append((int(r), int(c)))
    else:
        step = 1 if tc > c else -1
        while c != tc:
            c += step
            path.append((int(r), int(c)))
        step = 1 if tr > r else -1
        while r != tr:
            r += step
            path.append((int(r), int(c)))
    return path


def _grow_shape(dir_key: str) -> tuple[np.ndarray, tuple[int, int]]:
    height = random.randint(*_SHAPE_SIZE_RANGE)
    width = random.randint(*_SHAPE_SIZE_RANGE)
    hub = _choose_hub(height, width, dir_key)

    anchors = [
        (0, random.randrange(width)),
        (height - 1, random.randrange(width)),
        (random.randrange(height), 0),
        (random.randrange(height), width - 1),
    ]

    cells = {hub}
    for anchor in anchors:
        cells.update(_manhattan_path(hub, anchor, row_first=bool(random.getrandbits(1))))

    extra_budget = min(6, height * width - len(cells))
    target_area = len(cells) + random.randint(0, max(0, int(extra_budget)))
    while len(cells) < target_area:
        frontier: list[tuple[int, int]] = []
        for r, c in cells:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr = int(r + dr)
                nc = int(c + dc)
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    frontier.append((int(nr), int(nc)))
        if not frontier:
            break
        cells.add(random.choice(frontier))

    mask = np.zeros((height, width), dtype=bool)
    for r, c in cells:
        mask[int(r), int(c)] = True
    return mask, hub


def _margin_cells(cells: set[tuple[int, int]], height: int, width: int, margin: int = 1) -> set[tuple[int, int]]:
    padded: set[tuple[int, int]] = set()
    for r, c in cells:
        for rr in range(max(0, int(r - margin)), min(height, int(r + margin + 1))):
            for cc in range(max(0, int(c - margin)), min(width, int(c + margin + 1))):
                padded.add((int(rr), int(cc)))
    return padded


def _biased_coord(limit: int, size: int, shift: int, prefer_clip: bool) -> int:
    max_start = int(limit - size)
    if max_start <= 0:
        return 0
    if not prefer_clip:
        return random.randint(0, max_start)

    mag = abs(int(shift))
    if shift < 0:
        hi = min(max_start, max(0, mag - 1))
        if hi > 0:
            return random.randint(0, hi)
    if shift > 0:
        lo = max(0, max_start - max(0, mag - 1))
        if lo < max_start:
            return random.randint(lo, max_start)
    return random.randint(0, max_start)


def _cells_from_mask(mask: np.ndarray, top: int, left: int) -> set[tuple[int, int]]:
    coords = np.argwhere(mask)
    return {(int(top + r), int(left + c)) for r, c in coords}


def _shifted_visible_cells(
    mask: np.ndarray,
    top: int,
    left: int,
    shift_r: int,
    shift_c: int,
    height: int,
    width: int,
) -> set[tuple[int, int]]:
    coords = np.argwhere(mask)
    visible: set[tuple[int, int]] = set()
    for r, c in coords:
        rr = int(top + r + shift_r)
        cc = int(left + c + shift_c)
        if 0 <= rr < height and 0 <= cc < width:
            visible.add((int(rr), int(cc)))
    return visible


def _validate_scene(grid: np.ndarray, expected_objects: int) -> bool:
    bg = _background_color(grid)
    comps = _components(grid != int(bg))
    if len(comps) != expected_objects:
        return False
    for coords in comps:
        try:
            _marker_and_shift(grid, coords)
        except AssertionError:
            return False
        if int(coords.shape[0]) < 4:
            return False
    out = main(grid)
    out_comps = _components(out != int(bg))
    return len(out_comps) == expected_objects and bool(np.any(out != grid))


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*_GRID_SIZE_RANGE)
    width = random.randint(*_GRID_SIZE_RANGE)
    bg = random.randrange(10)

    object_count = random.randint(*_OBJECT_COUNT_RANGE)
    dir_keys = list(_DIRS)
    extra = [random.choice(dir_keys) for _ in range(max(0, object_count - 4))]
    directions = dir_keys + extra
    random.shuffle(directions)

    palette = [color for color in range(10) if color != bg]
    random.shuffle(palette)
    base_colors = palette[:object_count]

    marker_choices = [color for color in range(10) if color != bg]
    grid = np.full((height, width), int(bg), dtype=int)

    input_reserved: set[tuple[int, int]] = set()
    output_reserved: set[tuple[int, int]] = set()
    clipped_any = False

    order = list(range(object_count))
    random.shuffle(order)
    clip_target = random.choice(order) if bool(random.getrandbits(1)) else None

    objects: list[dict] = [None] * object_count  # type: ignore[list-item]
    for idx in order:
        dir_key = directions[idx]
        base_color = int(base_colors[idx])
        marker_pool = [color for color in marker_choices if color != base_color]
        marker_color = int(random.choice(marker_pool))

        placed = None
        for _ in range(120):
            mask, marker_rel = _grow_shape(dir_key)
            area = int(mask.sum())
            dir_r, dir_c = _DIRS[dir_key]
            shift_r = int(dir_r * area)
            shift_c = int(dir_c * area)

            prefer_clip = idx == clip_target
            min_visible = max(2, area // 3)
            for _ in range(120):
                top = _biased_coord(height, mask.shape[0], shift_r, prefer_clip=prefer_clip)
                left = _biased_coord(width, mask.shape[1], shift_c, prefer_clip=prefer_clip)

                input_cells = _cells_from_mask(mask, top, left)
                if input_cells & input_reserved:
                    continue

                visible_cells = _shifted_visible_cells(mask, top, left, shift_r, shift_c, height, width)
                if len(visible_cells) < min_visible:
                    continue
                clipped = len(visible_cells) < area
                if prefer_clip and not clipped:
                    continue
                if visible_cells & output_reserved:
                    continue

                placed = {
                    "mask": mask,
                    "marker_rel": marker_rel,
                    "top": int(top),
                    "left": int(left),
                    "base_color": base_color,
                    "marker_color": marker_color,
                    "output_cells": visible_cells,
                    "input_cells": input_cells,
                    "clipped": clipped,
                }
                break
            if placed is not None:
                break
        if placed is None:
            return None

        input_reserved |= _margin_cells(placed["input_cells"], height, width, margin=1)
        output_reserved |= _margin_cells(placed["output_cells"], height, width, margin=1)
        clipped_any = clipped_any or bool(placed["clipped"])
        objects[idx] = placed

    if clip_target is not None and not clipped_any:
        return None

    total_foreground = 0
    for obj in objects:
        assert obj is not None
        mask = obj["mask"]
        top = int(obj["top"])
        left = int(obj["left"])
        total_foreground += int(mask.sum())
        grid[top : top + mask.shape[0], left : left + mask.shape[1]][mask] = int(obj["base_color"])
        mr, mc = obj["marker_rel"]
        grid[int(top + mr), int(left + mc)] = int(obj["marker_color"])

    if total_foreground >= (height * width) // 2:
        return None
    if not _validate_scene(grid, object_count):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
