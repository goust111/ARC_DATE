from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
ARC_MAX = 30
FOREGROUND_COLORS = tuple(range(1, 10))
MAX_GENERATION_ATTEMPTS = 400
_N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return np.asarray(mask[r0 : r1 + 1, c0 : c1 + 1], dtype=bool)


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue

            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in _N4:
                    nr = rr + dr
                    nc = cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != int(color):
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))

            coords = np.array(cells, dtype=int)
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
            comps.append(mask)

    return comps


def _assembly_signatures(mask_a: np.ndarray, mask_b: np.ndarray) -> tuple[set[tuple[int, int]], set[int]]:
    mask_a = np.asarray(mask_a, dtype=bool)
    mask_b = np.asarray(mask_b, dtype=bool)
    assert mask_a.any() and mask_b.any()

    coords_a = np.argwhere(mask_a)
    area = int(mask_a.sum() + mask_b.sum())
    ha, wa = map(int, mask_a.shape)
    a_cells = {tuple(map(int, cell)) for cell in coords_a}

    dims: set[tuple[int, int]] = set()
    rotations: set[int] = set()
    for rot in range(4):
        rotated_b = np.rot90(mask_b, rot)
        coords_b = np.argwhere(rotated_b)
        hb, wb = map(int, rotated_b.shape)

        for dr in range(-hb, ha + 1):
            for dc in range(-wb, wa + 1):
                min_r = 0
                max_r = ha - 1
                min_c = 0
                max_c = wa - 1
                overlap = False

                for rr, cc in coords_b:
                    nr = int(rr) + int(dr)
                    nc = int(cc) + int(dc)
                    if (nr, nc) in a_cells:
                        overlap = True
                        break
                    if nr < min_r:
                        min_r = nr
                    if nr > max_r:
                        max_r = nr
                    if nc < min_c:
                        min_c = nc
                    if nc > max_c:
                        max_c = nc

                if overlap:
                    continue

                rect_h = max_r - min_r + 1
                rect_w = max_c - min_c + 1
                if rect_h * rect_w == area:
                    dims.add((rect_h, rect_w))
                    rotations.add(rot)

    return dims, rotations


def _assembled_rectangle_dims(mask_a: np.ndarray, mask_b: np.ndarray) -> tuple[int, int]:
    dims, _ = _assembly_signatures(mask_a, mask_b)
    assert len(dims) == 1, "same-color pair must assemble into one unique rectangle size"
    return next(iter(dims))


def _render_output(rects: list[tuple[int, int, int]]) -> np.ndarray:
    rects = sorted(rects, key=lambda item: (item[0] * item[1], item[0], item[1]))
    out_h = max(h for h, _, _ in rects)
    out_w = sum(w for _, w, _ in rects) + len(rects) - 1
    out = np.full((out_h, out_w), BG, dtype=int)

    c0 = 0
    for h, w, color in rects:
        out[:h, c0 : c0 + w] = int(color)
        c0 += w + 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Pair same-colored fragments into solid rectangles, then sort the rectangles by area.

    Input structure:
    - The background color is black.
    - The foreground consists of monochrome 4-connected objects, with exactly two objects per color.
    - For each color, rotating one object by 0/90/180/270 degrees and translating it lets the pair tile one solid rectangle.
    - Different color pairs assemble into different rectangle sizes.

    Rule:
    - For each foreground color, extract its two connected components.
    - Try all four rotations of one component and all relative translations until the two shapes tile a solid rectangle with no overlap.
    - Record that rectangle's height, width, and area.
    - Output one solid rectangle per color, ordered from smallest area to largest area, laid out left-to-right with one black column between neighboring rectangles and all top edges aligned.

    Key insight:
    - The visible fragments are not the final objects: each color first has to be reassembled into its hidden rectangle, and only then can the rectangles be sorted by area.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    colors = [int(color) for color in np.unique(g) if int(color) != int(BG)]
    assert colors

    rects: list[tuple[int, int, int]] = []
    for color in colors:
        comps = _components_of_color(g, color)
        assert len(comps) == 2, "each foreground color must appear in exactly two components"
        rect_h, rect_w = _assembled_rectangle_dims(comps[0], comps[1])
        rects.append((rect_h, rect_w, color))

    return _render_output(rects)


def _step_partition(h: int, w: int, split_row: int, top_w: int, bottom_w: int) -> tuple[np.ndarray, np.ndarray]:
    mask_a = np.zeros((h, w), dtype=bool)
    mask_a[:split_row, :top_w] = True
    mask_a[split_row:, :bottom_w] = True
    mask_b = ~mask_a
    return _crop_mask(mask_a), _crop_mask(mask_b)


def _sample_pair(required_rotation: int) -> tuple[np.ndarray, np.ndarray, tuple[int, int]] | None:
    for _ in range(200):
        h = random.randint(4, 7)
        w = random.randint(4, 7)
        split_row = random.randint(1, h - 1)
        top_w = random.randint(1, w - 1)
        bottom_w = random.randint(1, w - 1)
        if top_w == bottom_w:
            continue

        base_a, base_b = _step_partition(h, w, split_row, top_w, bottom_w)
        if int(base_a.sum()) < 4 or int(base_b.sum()) < 4:
            continue
        if int(base_a.sum()) == int(base_a.size) or int(base_b.sum()) == int(base_b.size):
            continue

        rot_a = random.randrange(4)
        rot_b = (rot_a - required_rotation) % 4
        mask_a = np.rot90(base_a, rot_a)
        mask_b = np.rot90(base_b, rot_b)

        dims_ab, rots_ab = _assembly_signatures(mask_a, mask_b)
        dims_ba, rots_ba = _assembly_signatures(mask_b, mask_a)
        if len(dims_ab) != 1 or len(rots_ab) != 1 or required_rotation not in rots_ab:
            continue
        if len(dims_ba) != 1 or len(rots_ba) != 1:
            continue

        rect_dims = next(iter(dims_ab))
        return np.asarray(mask_a, dtype=bool), np.asarray(mask_b, dtype=bool), rect_dims

    return None


def _dilate(mask: np.ndarray) -> np.ndarray:
    dilated = mask.copy()
    dilated[1:, :] |= mask[:-1, :]
    dilated[:-1, :] |= mask[1:, :]
    dilated[:, 1:] |= mask[:, :-1]
    dilated[:, :-1] |= mask[:, 1:]
    dilated[1:, 1:] |= mask[:-1, :-1]
    dilated[1:, :-1] |= mask[:-1, 1:]
    dilated[:-1, 1:] |= mask[1:, :-1]
    dilated[:-1, :-1] |= mask[1:, 1:]
    return dilated


def _place_patch(grid: np.ndarray, occupied: np.ndarray, mask: np.ndarray, color: int) -> bool:
    h, w = map(int, mask.shape)
    dilated = _dilate(occupied)
    row_choices = list(range(grid.shape[0] - h + 1))
    col_choices = list(range(grid.shape[1] - w + 1))
    random.shuffle(row_choices)
    random.shuffle(col_choices)

    for r0 in row_choices:
        for c0 in col_choices:
            if np.any(dilated[r0 : r0 + h, c0 : c0 + w] & mask):
                continue
            window = grid[r0 : r0 + h, c0 : c0 + w]
            window_occ = occupied[r0 : r0 + h, c0 : c0 + w]
            window[mask] = int(color)
            window_occ[mask] = True
            return True
    return False


def _scatter_patches(patches: list[tuple[np.ndarray, int]], min_h: int, min_w: int) -> np.ndarray | None:
    largest_h = max(mask.shape[0] for mask, _ in patches)
    largest_w = max(mask.shape[1] for mask, _ in patches)
    min_h = max(min_h, largest_h + 2, 12)
    min_w = max(min_w, largest_w + 2, 12)
    if min_h > ARC_MAX or min_w > ARC_MAX:
        return None
    total_fg = sum(int(mask.sum()) for mask, _ in patches)

    for _ in range(120):
        h = random.randint(min_h, ARC_MAX)
        w = random.randint(min_w, ARC_MAX)
        if total_fg * 3 > h * w:
            continue

        grid = np.full((h, w), BG, dtype=int)
        occupied = np.zeros((h, w), dtype=bool)
        order = list(range(len(patches)))
        random.shuffle(order)
        order.sort(key=lambda idx: int(patches[idx][0].sum()), reverse=True)

        ok = True
        for idx in order:
            mask, color = patches[idx]
            if not _place_patch(grid, occupied, mask, color):
                ok = False
                break
        if ok:
            return grid
    return None


def _expected_from_specs(specs: list[tuple[int, int, int]]) -> np.ndarray:
    rects = [(h, w, color) for h, w, color in specs]
    return _render_output(rects)


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.randint(2, 4)
    colors = random.sample(FOREGROUND_COLORS, pair_count)

    pair_specs: list[tuple[int, int, int]] = []
    patches: list[tuple[np.ndarray, int]] = []
    used_areas: set[int] = set()
    used_dims: set[tuple[int, int]] = set()

    for color in colors:
        required_rotation = random.randrange(4)
        sampled = _sample_pair(required_rotation)
        if sampled is None:
            return None

        mask_a, mask_b, (rect_h, rect_w) = sampled
        area = rect_h * rect_w
        if area in used_areas or (rect_h, rect_w) in used_dims:
            return None

        used_areas.add(area)
        used_dims.add((rect_h, rect_w))
        pair_specs.append((rect_h, rect_w, color))
        patches.append((mask_a, color))
        patches.append((mask_b, color))

    expected = _expected_from_specs(pair_specs)
    if max(expected.shape) > ARC_MAX:
        return None

    grid = _scatter_patches(
        patches,
        min_h=14,
        min_w=14,
    )
    if grid is None:
        return None

    try:
        actual = main(grid)
    except Exception:
        return None

    if np.array_equal(actual, expected):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
