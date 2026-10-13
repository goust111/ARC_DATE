from __future__ import annotations

from collections import deque

import numpy as np


BG = 0
COLOR_POOL = tuple(range(1, 10))
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
SIDE_OPTIONS = (3, 4, 5)
OBJECT_COUNT_RANGE = (4, 7)  # 4..6 objects, so every bar code can appear in one sample.
MAX_GENERATION_ATTEMPTS = 300
RNG = np.random.default_rng()


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(coords)
    return components


def _rotate_pixels(
    pixels: list[tuple[int, int, int]], quarter_turns_cw: int
) -> list[tuple[int, int, int]]:
    q = quarter_turns_cw % 4
    if q == 0:
        return list(pixels)

    rows = [r for r, _, _ in pixels]
    cols = [c for _, c, _ in pixels]
    center_r = (min(rows) + max(rows)) / 2.0
    center_c = (min(cols) + max(cols)) / 2.0

    rotated: list[tuple[int, int, int]] = []
    for r, c, color in pixels:
        dr = r - center_r
        dc = c - center_c
        for _ in range(q):
            dr, dc = dc, -dr
        nr = center_r + dr
        nc = center_c + dc
        nr_round = round(nr)
        nc_round = round(nc)
        assert abs(nr - nr_round) < 1e-9 and abs(nc - nc_round) < 1e-9
        rotated.append((int(nr_round), int(nc_round), color))

    positions = {(r, c) for r, c, _ in rotated}
    assert len(positions) == len(rotated)
    return rotated


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore every rotated motif to one shared orientation using the attached bar length.

    Input structure:
    - The background is black.
    - Each foreground connected component consists of one multicolor motif plus a short horizontal monochrome bar attached on its right side.
    - The motifs are rotated copies of one asymmetric template, and each attached bar has length 1 to 4.

    Rule:
    - For every foreground component, isolate the rightmost horizontal run as its code bar.
    - Convert the bar length into a clockwise quarter-turn count: 1 -> 90 degrees, 2 -> 180 degrees, 3 -> 270 degrees, 4 -> no rotation.
    - Erase the bar, rotate the remaining motif around its bounding-box center by that amount, and redraw it in place on the same canvas.

    Key insight:
    - The bar encodes how much clockwise rotation is still needed to return the motif to the common standard orientation, not the motif's current orientation itself.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    bg = BG if np.any(g == BG) else _most_common_color(g)
    out = g.copy()

    for coords in _connected_components(g != bg):
        coord_set = set(coords)
        rightmost_col = max(c for _, c in coords)
        rightmost_pixels = [(r, c) for r, c in coords if c == rightmost_col]
        assert len(rightmost_pixels) == 1
        bar_row, _ = rightmost_pixels[0]
        bar_color = int(g[bar_row, rightmost_col])

        bar: list[tuple[int, int]] = []
        c = rightmost_col
        while (bar_row, c) in coord_set and int(g[bar_row, c]) == bar_color:
            bar.append((bar_row, c))
            c -= 1
        bar_set = set(bar)
        bar_len = len(bar)
        assert 1 <= bar_len <= 4

        motif = [(r, c, int(g[r, c])) for r, c in coords if (r, c) not in bar_set]
        assert motif

        for r, c in coords:
            out[r, c] = bg

        turns_cw = bar_len % 4
        for nr, nc, color in _rotate_pixels(motif, turns_cw):
            out[nr, nc] = color

    return out


def _random_walk_path(mask: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> None:
    r, c = start
    er, ec = end
    mask[r, c] = True
    while (r, c) != (er, ec):
        options: list[tuple[int, int]] = []
        if r != er:
            options.append((r + (1 if er > r else -1), c))
        if c != ec:
            options.append((r, c + (1 if ec > c else -1)))
        r, c = options[int(RNG.integers(len(options)))]
        mask[r, c] = True


def _grow_connected(mask: np.ndarray, target_area: int) -> None:
    while int(mask.sum()) < target_area:
        frontier: list[tuple[int, int]] = []
        for r, c in np.argwhere(mask):
            rr = int(r)
            cc = int(c)
            for dr, dc in NEIGHBORS4:
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and not mask[nr, nc]:
                    frontier.append((nr, nc))
        if not frontier:
            return
        nr, nc = frontier[int(RNG.integers(len(frontier)))]
        mask[nr, nc] = True


def _is_asymmetric(mask: np.ndarray) -> bool:
    return all(not np.array_equal(mask, np.rot90(mask, k)) for k in (1, 2, 3))


def _generate_template_mask(side: int) -> np.ndarray | None:
    min_area = side + 2
    max_area = min(side * side - 1, side * 2 + 4)

    for _ in range(80):
        mask = np.zeros((side, side), dtype=bool)
        anchors = [
            (0, int(RNG.integers(side))),
            (side - 1, int(RNG.integers(side))),
            (int(RNG.integers(side)), 0),
            (int(RNG.integers(side)), side - 1),
        ]

        order = list(RNG.permutation(len(anchors)))
        for idx in range(len(order)):
            a = anchors[order[idx]]
            b = anchors[order[(idx + 1) % len(order)]]
            _random_walk_path(mask, a, b)

        base_area = int(mask.sum())
        if base_area > max_area:
            continue
        target_area = int(RNG.integers(max(min_area, base_area), max_area + 1))
        _grow_connected(mask, target_area)

        if not _is_asymmetric(mask):
            continue
        if any(not mask[edge].any() for edge in (0, -1)) or any(
            not mask[:, edge].any() for edge in (0, -1)
        ):
            continue
        return mask
    return None


def _color_template(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    area = len(coords)
    palette_size = int(RNG.integers(2, min(5, area + 1)))
    colors = RNG.choice(COLOR_POOL, size=palette_size, replace=False)
    assignments = RNG.choice(colors, size=area, replace=True)
    if len(np.unique(assignments)) == 1:
        assignments[0] = colors[1]

    patch = np.zeros(mask.shape, dtype=int)
    for (r, c), color in zip(coords, assignments):
        patch[int(r), int(c)] = int(color)
    return patch


def _attachment_rows(patch: np.ndarray) -> list[int]:
    coords = np.argwhere(patch != BG)
    rightmost = int(coords[:, 1].max())
    rows = sorted({int(r) for r, c in coords if int(c) == rightmost})
    assert rows
    return rows


def _make_pair_patch(object_patch: np.ndarray, bar_len: int, bar_color: int) -> np.ndarray:
    h, w = object_patch.shape
    bar_row = int(RNG.choice(_attachment_rows(object_patch)))
    pair = np.zeros((h, w + bar_len), dtype=int)
    pair[:, :w] = object_patch
    pair[bar_row, w:] = int(bar_color)
    return pair


def _sample_dimensions(side: int, object_count: int) -> tuple[int, int, int, int]:
    slot_cols = 2
    slot_rows = (object_count + slot_cols - 1) // slot_cols
    slot_h = side + 3
    slot_w = side + 6
    base_h = slot_rows * slot_h
    base_w = slot_cols * slot_w
    h = int(RNG.integers(base_h, min(31, base_h + 7)))
    w = int(RNG.integers(base_w, min(31, base_w + 9)))
    return h, w, slot_h, slot_w


def _gen_attempt() -> np.ndarray | None:
    side = int(RNG.choice(SIDE_OPTIONS))
    template_mask = _generate_template_mask(side)
    if template_mask is None:
        return None

    standard_patch = _color_template(template_mask)
    if any(np.array_equal(standard_patch, np.rot90(standard_patch, k)) for k in (1, 2, 3)):
        return None

    used_colors = {int(color) for color in np.unique(standard_patch) if int(color) != BG}
    bar_choices = [color for color in COLOR_POOL if color not in used_colors]
    if not bar_choices:
        return None
    bar_color = int(RNG.choice(bar_choices))

    object_count = int(RNG.integers(*OBJECT_COUNT_RANGE))
    rotations_needed = [0, 1, 2, 3]
    while len(rotations_needed) < object_count:
        rotations_needed.append(int(RNG.integers(4)))
    RNG.shuffle(rotations_needed)

    pair_patches: list[np.ndarray] = []
    expected_patches: list[np.ndarray] = []
    for turns_cw in rotations_needed:
        input_patch = np.rot90(standard_patch, turns_cw)
        bar_len = 4 if turns_cw == 0 else turns_cw
        pair_patches.append(_make_pair_patch(input_patch, bar_len, bar_color))
        expected_patches.append(standard_patch.copy())

    h, w, slot_h, slot_w = _sample_dimensions(side, object_count)
    grid = np.zeros((h, w), dtype=int)
    expected = np.zeros_like(grid)

    slots = [(r * slot_h, c * slot_w) for r in range((object_count + 1) // 2) for c in range(2)]
    if len(slots) < object_count:
        return None
    slot_order = list(RNG.permutation(len(slots)))[:object_count]
    pair_order = list(RNG.permutation(object_count))

    for pair_idx, slot_idx in zip(pair_order, slot_order):
        pair = pair_patches[pair_idx]
        pair_h, pair_w = pair.shape
        slot_r, slot_c = slots[int(slot_idx)]

        max_r_offset = slot_h - pair_h - 1
        max_c_offset = slot_w - pair_w - 1
        if max_r_offset < 1 or max_c_offset < 1:
            return None

        r0 = slot_r + int(RNG.integers(1, max_r_offset + 1))
        c0 = slot_c + int(RNG.integers(1, max_c_offset + 1))
        grid[r0 : r0 + pair_h, c0 : c0 + pair_w][pair != BG] = pair[pair != BG]
        expected[r0 : r0 + side, c0 : c0 + side][expected_patches[pair_idx] != BG] = expected_patches[pair_idx][
            expected_patches[pair_idx] != BG
        ]

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid rotation-bar ARC sample")
