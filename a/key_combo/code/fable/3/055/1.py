from __future__ import annotations

from collections import Counter
import random

import numpy as np


def _background(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _tight_crop(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    return len(_components(mask)) == 1


def _same_mask(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and bool(np.array_equal(a, b))


def _all_and(masks: list[np.ndarray]) -> np.ndarray:
    out = masks[0].copy()
    for mask in masks[1:]:
        out &= mask
    return out


def _all_or(masks: list[np.ndarray]) -> np.ndarray:
    out = masks[0].copy()
    for mask in masks[1:]:
        out |= mask
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the never-drawn eraser shape from the aligned holes of repeated partial copies.

    Input structure:
    - The background is one dominant color, and all foreground pixels share one other color.
    - The foreground consists of several separated 4-connected objects.
    - Each object is the same template inside one shared bbox size, except that one internal region is missing.

    Rule:
    - Crop every foreground object to its tight bbox and align those crops by their top-left corners.
    - OR the aligned crops together to reconstruct the full template mask.
    - Subtract any one cropped object from that template to reveal its missing region.
    - Return the tight bbox crop of that missing-region mask, filled with the foreground color on background.

    Key insight:
    - The answer never appears as drawn pixels in the input; it only exists as the consistent translated absence inside each partial copy.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _background(g)
    fg_colors = [int(color) for color in np.unique(g) if int(color) != bg]
    assert len(fg_colors) == 1
    fg = fg_colors[0]

    pieces = _components(g == fg)
    assert len(pieces) >= 2

    local_masks: list[np.ndarray] = []
    window_shapes = set()
    for coords in pieces:
        r0, c0, r1, c1 = _bbox_from_coords(coords)
        local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        local[coords[:, 0] - r0, coords[:, 1] - c0] = True
        local_masks.append(local)
        window_shapes.add(local.shape)

    assert len(window_shapes) == 1

    template = np.zeros_like(local_masks[0], dtype=bool)
    for local in local_masks:
        template |= local

    missing_shapes: list[np.ndarray] = []
    for local in local_masks:
        missing = template & ~local
        assert missing.any()
        missing_shapes.append(_tight_crop(missing))

    eraser = missing_shapes[0]
    for other in missing_shapes[1:]:
        assert _same_mask(other, eraser)

    out = np.full(eraser.shape, bg, dtype=int)
    out[eraser] = fg
    return out


_COUNT_CHOICES = [2, 3, 3, 4, 4, 5]
_GAP_CHOICES = [1, 2]
_MIN_BOX = 5
_MAX_BOX = 8
_JITTER = 1
_RETRY_LIMIT = 400


def _template_mask(h: int, w: int) -> np.ndarray:
    for _ in range(80):
        mask = np.ones((h, w), dtype=bool)
        for _ in range(random.randint(0, 4)):
            side = random.choice("tblr")
            if side in "tb":
                max_len = max(1, w - 2)
                length = random.randint(1, max_len)
                start = random.randint(0, w - length)
                row = 0 if side == "t" else h - 1
                mask[row, start : start + length] = False
            else:
                max_len = max(1, h - 2)
                length = random.randint(1, max_len)
                start = random.randint(0, h - length)
                col = 0 if side == "l" else w - 1
                mask[start : start + length, col] = False
        if not mask[0].any() or not mask[-1].any():
            continue
        if not mask[:, 0].any() or not mask[:, -1].any():
            continue
        return mask
    raise RuntimeError("failed to build a valid template")


def _grow_shape(box_h: int, box_w: int) -> np.ndarray:
    max_area = min(5, box_h * box_w)
    min_area = 2 if max_area >= 2 else 1
    target_area = random.randint(min_area, max_area)
    start = (random.randrange(box_h), random.randrange(box_w))
    cells = {start}
    while len(cells) < target_area:
        frontier = []
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < box_h and 0 <= nc < box_w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            break
        cells.add(random.choice(frontier))

    mask = np.zeros((box_h, box_w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return _tight_crop(mask)


def _missing_candidates(shape: np.ndarray, h: int, w: int) -> list[np.ndarray]:
    sh, sw = shape.shape
    candidates: list[np.ndarray] = []
    for r0 in range(1, h - sh):
        for c0 in range(1, w - sw):
            mask = np.zeros((h, w), dtype=bool)
            mask[r0 : r0 + sh, c0 : c0 + sw] = shape
            candidates.append(mask)
    return candidates


def _has_pair_overlap(masks: list[np.ndarray]) -> bool:
    for i, left in enumerate(masks):
        for right in masks[i + 1 :]:
            if bool((left & right).any()):
                return True
    return False


def _choose_missing_sets(template: np.ndarray, count: int) -> tuple[np.ndarray, list[np.ndarray]] | None:
    h, w = template.shape
    box_options = []
    for box_h in range(1, min(3, h - 2) + 1):
        for box_w in range(1, min(3, w - 2) + 1):
            if (h - box_h - 1) * (w - box_w - 1) >= count:
                box_options.append((box_h, box_w))
    if not box_options:
        return None

    for _ in range(120):
        shape = _grow_shape(*random.choice(box_options))
        candidates = _missing_candidates(shape, h, w)
        if len(candidates) < count:
            continue

        overlap_possible = False
        for i, left in enumerate(candidates):
            for right in candidates[i + 1 :]:
                if bool((left & right).any()):
                    overlap_possible = True
                    break
            if overlap_possible:
                break

        modes = ["separate", "mixed"]
        if count >= 3 and overlap_possible:
            modes.append("overlap")
        random.shuffle(modes)

        for mode in modes:
            for _ in range(160):
                missing_masks = random.sample(candidates, count)
                common_missing = _all_and(missing_masks)
                if common_missing.any():
                    continue
                pair_overlap = _has_pair_overlap(missing_masks)
                if mode == "separate" and pair_overlap:
                    continue
                if mode == "overlap" and not pair_overlap:
                    continue

                damaged = [template & ~missing for missing in missing_masks]
                if not all(_is_connected(mask) for mask in damaged):
                    continue
                if not bool(np.array_equal(_all_or(damaged), template)):
                    continue
                return shape, damaged
    return None


def _layout(count: int) -> tuple[int, int, int, int, int, int, int] | None:
    choices = []
    for gap in _GAP_CHOICES:
        for rows in range(1, min(3, count) + 1):
            cols = (count + rows - 1) // rows
            max_h = min(_MAX_BOX, (30 - (rows + 1) * gap - rows * _JITTER) // rows)
            max_w = min(_MAX_BOX, (30 - (cols + 1) * gap - cols * _JITTER) // cols)
            if max_h >= _MIN_BOX and max_w >= _MIN_BOX:
                choices.append((rows, cols, gap, max_h, max_w))
    if not choices:
        return None

    rows, cols, gap, max_h, max_w = random.choice(choices)
    h = random.randint(_MIN_BOX, max_h)
    w = random.randint(_MIN_BOX, max_w)
    grid_h = rows * h + (rows + 1) * gap + rows * _JITTER
    grid_w = cols * w + (cols + 1) * gap + cols * _JITTER
    return rows, cols, gap, h, w, grid_h, grid_w


def _place_objects(count: int, h: int, w: int, rows: int, cols: int, gap: int) -> list[tuple[int, int]]:
    slots = []
    for rr in range(rows):
        for cc in range(cols):
            base_r = gap + rr * (h + gap + _JITTER)
            base_c = gap + cc * (w + gap + _JITTER)
            slots.append(
                (
                    base_r + random.randint(0, _JITTER),
                    base_c + random.randint(0, _JITTER),
                )
            )
    random.shuffle(slots)
    return slots[:count]


def _render(mask: np.ndarray, color: int, background: int) -> np.ndarray:
    out = np.full(mask.shape, background, dtype=int)
    out[mask] = color
    return out


def _attempt() -> np.ndarray | None:
    count = random.choice(_COUNT_CHOICES)
    layout = _layout(count)
    if layout is None:
        return None
    rows, cols, gap, h, w, grid_h, grid_w = layout

    template = _template_mask(h, w)
    chosen = _choose_missing_sets(template, count)
    if chosen is None:
        return None
    eraser, damaged_masks = chosen

    fg = random.randint(1, 9)
    grid = np.zeros((grid_h, grid_w), dtype=int)
    positions = _place_objects(count, h, w, rows, cols, gap)
    for (r0, c0), local in zip(positions, damaged_masks):
        patch = grid[r0 : r0 + h, c0 : c0 + w]
        patch[local] = fg

    expected = _render(eraser, fg, 0)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not bool(np.array_equal(actual, expected)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_RETRY_LIMIT):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
