from __future__ import annotations

import random
from collections import Counter

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _largest_enclosed_component_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    best_area = -1
    best_bbox: tuple[int, int, int, int] | None = None

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            area = 0
            touches_border = False
            r0 = r1 = r
            c0 = c1 = c
            while stack:
                rr, cc = stack.pop()
                area += 1
                if rr == 0 or rr == h - 1 or cc == 0 or cc == w - 1:
                    touches_border = True
                r0 = min(r0, rr)
                r1 = max(r1, rr)
                c0 = min(c0, cc)
                c1 = max(c1, cc)
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            if not touches_border and area > best_area:
                best_area = area
                best_bbox = (r0, c0, r1, c1)
    return best_bbox


def _find_frame_bbox(grid: np.ndarray, bg: int) -> tuple[int, int, int, int]:
    inner_bbox = _largest_enclosed_component_bbox(grid == bg)
    assert inner_bbox is not None, "expected one framed background region"
    ir0, ic0, ir1, ic1 = inner_bbox
    fr0, fc0, fr1, fc1 = ir0 - 1, ic0 - 1, ir1 + 1, ic1 + 1
    h, w = grid.shape
    assert 0 <= fr0 < fr1 < h and 0 <= fc0 < fc1 < w, "frame must fit in bounds"

    border = np.concatenate(
        [
            grid[fr0, fc0 : fc1 + 1],
            grid[fr1, fc0 : fc1 + 1],
            grid[fr0 + 1 : fr1, fc0],
            grid[fr0 + 1 : fr1, fc1],
        ]
    )
    assert len(border) > 0 and np.all(border == border[0]) and int(border[0]) != bg, "expected uniform frame border"
    return fr0, fc0, fr1, fc1


def _extract_panels(
    grid: np.ndarray,
    bg: int,
    frame_bbox: tuple[int, int, int, int],
) -> tuple[np.ndarray, np.ndarray, int]:
    fr0, fc0, fr1, fc1 = frame_bbox
    outside = (grid != bg).copy()
    outside[fr0 : fr1 + 1, fc0 : fc1 + 1] = False
    coords = np.argwhere(outside)
    assert coords.size > 0, "expected panel pixels outside the frame"
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    ph, pw = patch.shape

    if pw == 2 * ph + 1:
        k = ph
        divider = patch[:, k]
        assert np.all(divider == divider[0]) and int(divider[0]) != bg, "expected a vertical divider"
        return patch[:, :k], patch[:, k + 1 :], int(k)

    if ph == 2 * pw + 1:
        k = pw
        divider = patch[k, :]
        assert np.all(divider == divider[0]) and int(divider[0]) != bg, "expected a horizontal divider"
        return patch[:k, :], patch[k + 1 :, :], int(k)

    raise AssertionError("failed to recover two aligned k-by-k panels")


def _slot_bounds(
    inner_r0: int,
    inner_c0: int,
    slot_h: int,
    slot_w: int,
    r: int,
    c: int,
) -> tuple[int, int, int, int]:
    top = inner_r0 + r * (slot_h + 1)
    left = inner_c0 + c * (slot_w + 1)
    return top, left, top + slot_h, left + slot_w


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the XOR mask between two aligned panels as coarse coordinates for stamping one sprite across a slot array.

    Input structure:
    - One side of the grid contains two k-by-k panels separated by a single-color divider; each panel shows one foreground color over a shared background.
    - Elsewhere there is a rectangular frame whose interior is partitioned into a k-by-k slot array by background separator bands.
    - Exactly one slot contains a multicolor sprite patch; every other slot is empty background.

    Rule:
    - Compare the two panels cell by cell and keep the positions where exactly one panel has foreground.
    - Clear the framed slot array, then copy the full sprite patch into every slot whose coarse coordinate is selected by that XOR mask.
    - Return only the framed region, including the border and separator bands.

    Key insight:
    - The panel comparison does not directly draw the answer; it produces a coarse k-by-k placement mask that tells where the single observed sprite should be stamped.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _get_bg(g)
    fr0, fc0, fr1, fc1 = _find_frame_bbox(g, bg)
    panel_a, panel_b, k = _extract_panels(g, bg, (fr0, fc0, fr1, fc1))
    xor_mask = (panel_a != bg) ^ (panel_b != bg)
    assert xor_mask.shape == (k, k), "panel comparison must be k-by-k"
    assert xor_mask.any(), "expected at least one selected slot"

    inner_h = (fr1 - fr0 + 1) - 2
    inner_w = (fc1 - fc0 + 1) - 2
    usable_h = inner_h - (k - 1)
    usable_w = inner_w - (k - 1)
    assert usable_h > 0 and usable_w > 0, "slot array must have positive slot size"
    assert usable_h % k == 0 and usable_w % k == 0, "slot lattice must align with k"
    slot_h = usable_h // k
    slot_w = usable_w // k

    inner_r0 = fr0 + 1
    inner_c0 = fc0 + 1
    sprite_slots: list[np.ndarray] = []
    for rr in range(k):
        for cc in range(k):
            r0, c0, r1, c1 = _slot_bounds(inner_r0, inner_c0, slot_h, slot_w, rr, cc)
            patch = g[r0:r1, c0:c1]
            if np.any(patch != bg):
                sprite_slots.append(patch.copy())
    assert len(sprite_slots) == 1, "expected exactly one source sprite slot"
    sprite_patch = sprite_slots[0]

    out = g[fr0 : fr1 + 1, fc0 : fc1 + 1].copy()
    out[1:-1, 1:-1] = bg
    for rr in range(k):
        for cc in range(k):
            if not xor_mask[rr, cc]:
                continue
            r0 = 1 + rr * (slot_h + 1)
            c0 = 1 + cc * (slot_w + 1)
            out[r0 : r0 + slot_h, c0 : c0 + slot_w] = sprite_patch
    return out


def _random_panel_mask(k: int) -> np.ndarray:
    while True:
        mask = np.zeros((k, k), dtype=bool)
        required = {
            (0, random.randrange(k)),
            (k - 1, random.randrange(k)),
            (random.randrange(k), 0),
            (random.randrange(k), k - 1),
        }
        for r, c in required:
            mask[r, c] = True

        density = random.uniform(0.2, 0.65)
        for r in range(k):
            for c in range(k):
                if (r, c) not in required and random.random() < density:
                    mask[r, c] = True

        if mask.any():
            return mask


def _sample_panel_masks(k: int, origin: tuple[int, int], keep_origin: bool) -> tuple[np.ndarray, np.ndarray]:
    max_selected = max(1, (2 * k * k) // 3)
    for _ in range(200):
        a = _random_panel_mask(k)
        b = _random_panel_mask(k)
        xor_mask = a ^ b
        selected = int(xor_mask.sum())
        if 1 <= selected <= max_selected and bool(xor_mask[origin]) == keep_origin and not np.array_equal(a, b):
            return a, b
    raise RuntimeError("failed to sample valid panel masks")


def _random_sprite(slot_size: int, colors: list[int], bg: int) -> np.ndarray:
    for _ in range(200):
        patch = np.full((slot_size, slot_size), bg, dtype=int)
        target = random.randint(2, min(5, slot_size * slot_size - 1))
        start = (random.randrange(slot_size), random.randrange(slot_size))
        cells = {start}
        while len(cells) < target:
            base_r, base_c = random.choice(tuple(cells))
            dr, dc = random.choice(((-1, 0), (1, 0), (0, -1), (0, 1)))
            nr = base_r + dr
            nc = base_c + dc
            if 0 <= nr < slot_size and 0 <= nc < slot_size:
                cells.add((nr, nc))

        if len(cells) < 2:
            continue

        n_used = random.randint(2, min(len(colors), len(cells)))
        used = random.sample(colors, n_used)
        color_pool = used + random.choices(used, k=len(cells) - n_used)
        random.shuffle(color_pool)
        for color, (r, c) in zip(color_pool, cells):
            patch[r, c] = color

        if np.all(np.any(patch == bg, axis=1)) and np.all(np.any(patch == bg, axis=0)):
            return patch
    raise RuntimeError("failed to sample sprite patch")


def _choose_layout(
    frame_h: int,
    frame_w: int,
    cluster_h: int,
    cluster_w: int,
) -> tuple[int, int, tuple[int, int], tuple[int, int]]:
    options: list[tuple[str, int, int, int]] = []
    for gap in range(1, 4):
        min_h = max(frame_h, cluster_h)
        min_w = frame_w + cluster_w + gap
        if min_h <= 30 and min_w <= 30:
            options.append(("horizontal", gap, min_h, min_w))
        min_h = frame_h + cluster_h + gap
        min_w = max(frame_w, cluster_w)
        if min_h <= 30 and min_w <= 30:
            options.append(("vertical", gap, min_h, min_w))
    assert options, "no feasible layout under ARC size limit"

    axis, gap, min_h, min_w = random.choice(options)
    h = random.randint(min_h, 30)
    w = random.randint(min_w, 30)

    if axis == "horizontal":
        left_is_frame = random.choice((False, True))
        left_pad = random.randint(0, w - min_w)
        if left_is_frame:
            frame_x = left_pad
            cluster_x = frame_x + frame_w + gap
        else:
            cluster_x = left_pad
            frame_x = cluster_x + cluster_w + gap
        frame_y = random.randint(0, h - frame_h)
        cluster_y = random.randint(0, h - cluster_h)
        return h, w, (frame_y, frame_x), (cluster_y, cluster_x)

    top_is_frame = random.choice((False, True))
    top_pad = random.randint(0, h - min_h)
    if top_is_frame:
        frame_y = top_pad
        cluster_y = frame_y + frame_h + gap
    else:
        cluster_y = top_pad
        frame_y = cluster_y + cluster_h + gap
    frame_x = random.randint(0, w - frame_w)
    cluster_x = random.randint(0, w - cluster_w)
    return h, w, (frame_y, frame_x), (cluster_y, cluster_x)


def _render_panels(
    canvas: np.ndarray,
    top: int,
    left: int,
    panel_a: np.ndarray,
    panel_b: np.ndarray,
    color_a: int,
    color_b: int,
    divider_color: int,
    orientation: str,
) -> None:
    k = panel_a.shape[0]
    if orientation == "horizontal":
        for r, c in np.argwhere(panel_a):
            canvas[top + int(r), left + int(c)] = color_a
        canvas[top : top + k, left + k] = divider_color
        for r, c in np.argwhere(panel_b):
            canvas[top + int(r), left + k + 1 + int(c)] = color_b
        return

    for r, c in np.argwhere(panel_a):
        canvas[top + int(r), left + int(c)] = color_a
    canvas[top + k, left : left + k] = divider_color
    for r, c in np.argwhere(panel_b):
        canvas[top + k + 1 + int(r), left + int(c)] = color_b


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(range(10))
    remaining = [c for c in range(10) if c != bg]
    k = random.randint(2, 4)
    slot_size = 3 if k == 4 else random.choice((3, 4))
    panel_orientation = random.choice(("horizontal", "vertical"))
    cluster_h = k if panel_orientation == "horizontal" else 2 * k + 1
    cluster_w = 2 * k + 1 if panel_orientation == "horizontal" else k
    frame_h = k * slot_size + (k - 1) + 2
    frame_w = k * slot_size + (k - 1) + 2

    try:
        h, w, frame_pos, cluster_pos = _choose_layout(frame_h, frame_w, cluster_h, cluster_w)
    except AssertionError:
        return None

    sprite_color_count = random.randint(2, 4)
    needed = 4 + sprite_color_count
    if len(remaining) < needed:
        return None
    chosen = random.sample(remaining, needed)
    panel_a_color, panel_b_color, divider_color, frame_color = chosen[:4]
    sprite_colors = chosen[4:]

    origin = (random.randrange(k), random.randrange(k))
    keep_origin = random.choice((False, True))
    try:
        panel_a, panel_b = _sample_panel_masks(k, origin, keep_origin)
        sprite_patch = _random_sprite(slot_size, sprite_colors, bg)
    except RuntimeError:
        return None

    canvas = np.full((h, w), bg, dtype=int)
    fr0, fc0 = frame_pos
    fr1 = fr0 + frame_h - 1
    fc1 = fc0 + frame_w - 1
    canvas[fr0, fc0 : fc1 + 1] = frame_color
    canvas[fr1, fc0 : fc1 + 1] = frame_color
    canvas[fr0 : fr1 + 1, fc0] = frame_color
    canvas[fr0 : fr1 + 1, fc1] = frame_color

    sprite_r, sprite_c = origin
    sr0 = fr0 + 1 + sprite_r * (slot_size + 1)
    sc0 = fc0 + 1 + sprite_c * (slot_size + 1)
    canvas[sr0 : sr0 + slot_size, sc0 : sc0 + slot_size] = sprite_patch

    pr0, pc0 = cluster_pos
    _render_panels(
        canvas,
        pr0,
        pc0,
        panel_a,
        panel_b,
        panel_a_color,
        panel_b_color,
        divider_color,
        panel_orientation,
    )

    if _get_bg(canvas) != bg:
        return None

    try:
        out = main(canvas)
    except Exception:
        return None

    if out.ndim != 2 or max(out.shape) > 30:
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for task 057/2")
