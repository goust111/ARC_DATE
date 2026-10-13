from __future__ import annotations

import random

import numpy as np


GRAY = 5
PATTERN_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
GENERATION_RETRIES = 400
PLACEMENT_RETRIES = 200
TILE_RETRIES = 200


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
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            comps.append(np.array(coords, dtype=int))

    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _shape(box: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = box
    return int(r1 - r0 + 1), int(c1 - c0 + 1)


def _is_hollow_frame(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h, w = _shape((r0, c0, r1, c1))
    if h < 3 or w < 3:
        return False

    patch = np.zeros((h, w), dtype=bool)
    patch[coords[:, 0] - r0, coords[:, 1] - c0] = True

    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return np.array_equal(patch, border)


def _find_special_boxes(grid: np.ndarray) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    comps = _components(grid == GRAY)
    solids: list[tuple[int, int, int, int]] = []
    frames: list[tuple[int, int, int, int]] = []

    for coords in comps:
        box = _bbox(coords)
        h, w = _shape(box)
        area = int(coords.shape[0])
        if area == h * w:
            solids.append(box)
        if _is_hollow_frame(coords):
            frames.append(box)

    pairs = [(solid, frame) for solid in solids for frame in frames if _shape(solid) == _shape(frame)]
    assert len(pairs) == 1, "expected one solid gray rectangle and one same-size hollow frame"
    return pairs[0]


def _detect_axis_period(grid: np.ndarray, ignore_mask: np.ndarray, axis: int) -> int:
    limit = grid.shape[axis]
    best_step = None
    best_support = -1

    for step in range(1, limit):
        if axis == 0:
            left = grid[:-step, :]
            right = grid[step:, :]
            valid = ~ignore_mask[:-step, :] & ~ignore_mask[step:, :]
        else:
            left = grid[:, :-step]
            right = grid[:, step:]
            valid = ~ignore_mask[:, :-step] & ~ignore_mask[:, step:]

        support = int(valid.sum())
        if support == 0:
            continue
        if np.any(left[valid] != right[valid]):
            continue
        if support > best_support or (support == best_support and (best_step is None or step < best_step)):
            best_step = step
            best_support = support

    assert best_step is not None, "failed to infer a translational period"
    return int(best_step)


def _recover_color(
    grid: np.ndarray,
    ignore_mask: np.ndarray,
    r: int,
    c: int,
    row_period: int,
    col_period: int,
) -> int:
    samples: list[tuple[int, int]] = []
    for rr in range(r % row_period, grid.shape[0], row_period):
        for cc in range(c % col_period, grid.shape[1], col_period):
            if ignore_mask[rr, cc]:
                continue
            dist = abs(rr - r) + abs(cc - c)
            samples.append((dist, int(grid[rr, cc])))

    assert samples, "each occluded cell must have a visible periodic counterpart"
    samples.sort(key=lambda item: item[0])
    return samples[0][1]


def _fill_box(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0 : r1 + 1, c0 : c1 + 1] = color


def _draw_frame(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _boxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1) -> bool:
    return (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move the gray solid rectangle into the matching gray frame and repair the exposed hole by periodic inpainting.

    Input structure:
    - The whole canvas is a translationally periodic multicolor pattern with no true background.
    - One gray component is a solid axis-aligned rectangle that occludes the pattern.
    - A second gray component is a same-size one-pixel hollow frame whose interior still shows the pattern.

    Rule:
    - Identify the solid gray rectangle and the same-size hollow frame.
    - Replace the frame and everything inside its bbox with a solid gray rectangle.
    - Remove the original solid rectangle and reconstruct each exposed cell from visible cells in the same translational-period orbit.

    Key insight:
    - The gray block is both the object to move and an occluder: the frame tells where the block must land,
      while the old location can only be restored by inferring the pattern's row and column periods.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    source_box, target_box = _find_special_boxes(g)
    ignore_mask = g == GRAY
    row_period = _detect_axis_period(g, ignore_mask, axis=0)
    col_period = _detect_axis_period(g, ignore_mask, axis=1)

    out = g.copy()
    sr0, sc0, sr1, sc1 = source_box
    for r in range(sr0, sr1 + 1):
        for c in range(sc0, sc1 + 1):
            out[r, c] = _recover_color(g, ignore_mask, r, c, row_period, col_period)

    _fill_box(out, target_box, GRAY)
    return out


def _make_periodic_pattern() -> tuple[np.ndarray, int, int]:
    for _ in range(TILE_RETRIES):
        tile_h = random.randint(2, 5)
        tile_w = random.randint(2, 5)
        rep_r = random.randint(4, max(4, 30 // tile_h))
        rep_c = random.randint(4, max(4, 30 // tile_w))

        height = tile_h * rep_r
        width = tile_w * rep_c
        if height > 30 or width > 30:
            continue

        palette_size = random.randint(3, min(6, len(PATTERN_COLORS)))
        palette = random.sample(PATTERN_COLORS, k=palette_size)
        tile = np.array([[random.choice(palette) for _ in range(tile_w)] for _ in range(tile_h)], dtype=int)
        if len(set(map(int, tile.flatten()))) < 3:
            continue

        pattern = np.tile(tile, (rep_r, rep_c))
        clear = np.zeros_like(pattern, dtype=bool)
        if _detect_axis_period(pattern, clear, axis=0) != tile_h:
            continue
        if _detect_axis_period(pattern, clear, axis=1) != tile_w:
            continue
        return pattern, tile_h, tile_w

    raise RuntimeError("failed to build a non-degenerate periodic pattern")


def _random_box(height: int, width: int, box_h: int, box_w: int) -> tuple[int, int, int, int]:
    r0 = random.randint(0, height - box_h)
    c0 = random.randint(0, width - box_w)
    return r0, c0, r0 + box_h - 1, c0 + box_w - 1


def _choose_boxes(height: int, width: int, tile_h: int, tile_w: int) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]] | None:
    max_h = min(height - tile_h, 2 * tile_h + 3, 10)
    max_w = min(width - tile_w, 2 * tile_w + 3, 10)
    if max_h < 3 or max_w < 3:
        return None

    box_h = random.randint(3, max_h)
    box_w = random.randint(3, max_w)

    for _ in range(PLACEMENT_RETRIES):
        source_box = _random_box(height, width, box_h, box_w)
        target_box = _random_box(height, width, box_h, box_w)
        if _boxes_separated(source_box, target_box, gap=1):
            return source_box, target_box
    return None


def _build_sample() -> tuple[np.ndarray, np.ndarray] | None:
    pattern, tile_h, tile_w = _make_periodic_pattern()
    boxes = _choose_boxes(pattern.shape[0], pattern.shape[1], tile_h, tile_w)
    if boxes is None:
        return None

    source_box, target_box = boxes
    inp = pattern.copy()
    out = pattern.copy()

    _fill_box(inp, source_box, GRAY)
    _draw_frame(inp, target_box, GRAY)
    _fill_box(out, target_box, GRAY)
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        sample = _build_sample()
        if sample is None:
            continue
        inp, out = sample
        if np.array_equal(main(inp), out):
            return inp
    raise RuntimeError("failed to generate a valid input grid")
