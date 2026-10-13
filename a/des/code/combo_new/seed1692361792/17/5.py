from __future__ import annotations

import random

import numpy as np


BG = 0
COLOR_POOL = (1, 2, 3, 4, 5, 6, 7, 8, 9)
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

GENERATION_ATTEMPTS = 400
SHAPE_ATTEMPTS = 80
PLACEMENT_ATTEMPTS = 120


def _parse_legend(row: np.ndarray) -> dict[int, int]:
    mapping: dict[int, int] = {}
    c = 0
    width = int(row.shape[0])
    while c < width:
        if int(row[c]) == BG:
            c += 1
            continue
        start = c
        while c < width and int(row[c]) != BG:
            c += 1
        run = row[start:c]
        assert len(run) == 2, "each legend entry must be exactly two adjacent non-black pixels"
        src, dst = map(int, run)
        assert src not in mapping, "legend source colors must be unique"
        mapping[src] = dst
    assert mapping, "legend must contain at least one color pair"
    return mapping


def _apply_rule(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape[0] >= 2, "grid must have a legend row and at least one payload row"

    mapping = _parse_legend(g[-1, :])
    body = g[:-1, :]
    colors = set(map(int, np.unique(body))) - {BG}
    assert colors, "expected at least one stamp object"
    assert colors <= set(mapping), "every object color must appear on the legend source side"

    mirrored = np.flipud(body)
    recolored = np.zeros_like(mirrored)
    for src, dst in mapping.items():
        recolored[mirrored == src] = int(dst)

    out = g.copy()
    out[-1, :] = BG
    out_body = out[:-1, :]
    write_mask = (out_body == BG) & (mirrored != BG)
    out_body[write_mask] = recolored[write_mask]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror each upper stamp into the lower half using the bottom color-band legend.

    Input structure:
    - The background is black.
    - The bottom row is a color-band legend made of separated two-pixel entries, where each adjacent pair A B means source color A maps to target color B.
    - Above the legend sit several monochrome 4-connected stamp objects, all using source-side legend colors, while the lower half is otherwise blank and reserved for reflected copies.

    Rule:
    - Read the bottom legend row by grouping every run of two adjacent non-black pixels into one source-to-target mapping.
    - Flip the non-legend part of the grid vertically across its horizontal midpoint.
    - Recolor every mirrored non-black pixel with its mapped target color and write it back only onto black cells, so original upper objects keep priority if a mirrored copy overlaps them.
    - Clear the entire legend row to black and keep the grid size unchanged.

    Key insight:
    - The bottom stripe is a lookup table rather than part of the scene, and the output requires two coupled operations in order: vertical reflection first, then color substitution through that legend.
    """
    return _apply_rule(grid)


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert len(coords) > 0
    (r0, c0), (r1, c1) = coords.min(axis=0), coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _dihedral_variant(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    if random.random() < 0.5:
        out = np.fliplr(out)
    turns = random.randrange(4)
    if turns:
        out = np.rot90(out, turns)
    return _trim_mask(out)


def _random_connected_mask(max_h: int, max_w: int) -> np.ndarray | None:
    for _ in range(SHAPE_ATTEMPTS):
        box_h = random.randint(1, max_h)
        box_w = random.randint(1, max_w)
        area_cap = min(8, box_h * box_w)
        min_area = 1 if area_cap == 1 or random.random() < 0.15 else 2
        if area_cap < min_area:
            continue
        target_area = random.randint(min_area, area_cap)

        start = (random.randrange(box_h), random.randrange(box_w))
        cells = {start}
        while len(cells) < target_area:
            frontier: list[tuple[int, int]] = []
            for r, c in cells:
                for dr, dc in NEIGHBORS4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < box_h and 0 <= nc < box_w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))

        if len(cells) != target_area:
            continue

        mask = np.zeros((box_h, box_w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True

        return _dihedral_variant(_trim_mask(mask))
    return None


def _place_stamp(
    canvas: np.ndarray,
    blocked: np.ndarray,
    mask: np.ndarray,
    color: int,
    upper_limit: int,
    touch_midline: bool,
) -> bool:
    mask_h, mask_w = map(int, mask.shape)
    max_row0 = upper_limit - mask_h + 1
    if max_row0 < 0 or mask_w > canvas.shape[1]:
        return False

    fixed_row0 = max_row0 if touch_midline else None
    for _ in range(PLACEMENT_ATTEMPTS):
        row0 = fixed_row0 if fixed_row0 is not None else random.randint(0, max_row0)
        col0 = random.randint(0, int(canvas.shape[1]) - mask_w)

        br0 = max(0, row0 - 1)
        bc0 = max(0, col0 - 1)
        br1 = min(int(canvas.shape[0]) - 1, row0 + mask_h)
        bc1 = min(int(canvas.shape[1]) - 1, col0 + mask_w)
        if blocked[br0 : br1 + 1, bc0 : bc1 + 1].any():
            continue

        patch = canvas[row0 : row0 + mask_h, col0 : col0 + mask_w]
        if np.any(patch[mask] != BG):
            continue

        patch[mask] = int(color)
        blocked[br0 : br1 + 1, bc0 : bc1 + 1] = True
        return True
    return False


def _legend_row(width: int, pairs: list[tuple[int, int]]) -> np.ndarray:
    row = np.zeros(width, dtype=int)
    band_len = 3 * len(pairs) - 1
    start = random.randint(0, width - band_len)
    col = start
    for idx, (src, dst) in enumerate(pairs):
        row[col] = int(src)
        row[col + 1] = int(dst)
        col += 2
        if idx != len(pairs) - 1:
            col += 1
    return row


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.randint(2, 4)
    palette = random.sample(COLOR_POOL, 2 * pair_count)
    pairs = list(zip(palette[:pair_count], palette[pair_count:]))
    random.shuffle(pairs)
    source_colors = [src for src, _ in pairs]

    band_len = 3 * pair_count - 1
    width = random.randint(max(10, band_len + 1), 30)
    if random.random() < 0.5:
        payload_h = random.choice(tuple(range(7, 24, 2)))
    else:
        payload_h = random.choice(tuple(range(8, 24, 2)))
    upper_limit = (payload_h - 1) // 2

    canvas = np.zeros((payload_h, width), dtype=int)
    blocked = np.zeros_like(canvas, dtype=bool)

    object_count = random.randint(pair_count, min(6, pair_count + 2))
    object_colors = source_colors.copy()
    while len(object_colors) < object_count:
        object_colors.append(random.choice(source_colors))
    random.shuffle(object_colors)

    force_midline = payload_h % 2 == 1 and random.random() < 0.5
    for idx, color in enumerate(object_colors):
        touch_midline = force_midline and idx == 0
        mask = _random_connected_mask(min(4, upper_limit + 1), min(6, width))
        if mask is None:
            return None
        if touch_midline and mask.shape[0] > upper_limit + 1:
            return None
        if not _place_stamp(canvas, blocked, mask, color, upper_limit, touch_midline):
            return None

    grid = np.zeros((payload_h + 1, width), dtype=int)
    grid[:-1, :] = canvas
    grid[-1, :] = _legend_row(width, pairs)

    lower_start = upper_limit + 1
    if np.any(grid[lower_start:-1, :] != BG):
        return None

    out = _apply_rule(grid)
    if not np.all(out[-1, :] == BG):
        return None
    if not np.any(out[lower_start:-1, :] != BG):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirrored-stamp input")
