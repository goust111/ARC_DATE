import random
from collections import deque

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_RETRIES = 300
PLACE_RETRIES = 200
OUTSIDE_PLACE_RETRIES = 300


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(np.min(coords[:, 0])),
        int(np.min(coords[:, 1])),
        int(np.max(coords[:, 0])),
        int(np.max(coords[:, 1])),
    )


def _border_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _find_frame(grid: np.ndarray, bg: int) -> tuple[int, tuple[int, int, int, int]]:
    candidates: list[tuple[int, tuple[int, int, int, int]]] = []
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        for comp in _components(grid == color):
            r0, c0, r1, c1 = _bbox(comp)
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            if h < 3 or w < 3:
                continue
            sub = grid[r0 : r1 + 1, c0 : c1 + 1]
            border = _border_mask(h, w)
            if not np.all(sub[border] == color):
                continue
            if np.any(sub[1:-1, 1:-1] == color):
                continue
            candidates.append((color, (r0, c0, r1, c1)))
    assert len(candidates) == 1, f"expected one frame, found {len(candidates)}"
    return candidates[0]


def _isolated_block(grid: np.ndarray, bg: int, r0: int, c0: int, r1: int, c1: int) -> bool:
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(grid.shape[0] - 1, r1 + 1)
    cc1 = min(grid.shape[1] - 1, c1 + 1)
    for rr in range(rr0, rr1 + 1):
        for cc in range(cc0, cc1 + 1):
            if r0 <= rr <= r1 and c0 <= cc <= c1:
                continue
            if int(grid[rr, cc]) != bg:
                return False
    return True


def _find_legend(
    grid: np.ndarray,
    bg: int,
    frame_bbox: tuple[int, int, int, int],
    placeholder_colors: set[int],
) -> tuple[dict[int, int], tuple[int, int, int, int]]:
    h, w = grid.shape
    rows = len(placeholder_colors)
    fr0, fc0, fr1, fc1 = frame_bbox
    candidates: list[tuple[dict[int, int], tuple[int, int, int, int]]] = []
    for r0 in range(h - rows + 1):
        r1 = r0 + rows - 1
        for c0 in range(w - 1):
            c1 = c0 + 1
            if not (r1 < fr0 or r0 > fr1 or c1 < fc0 or c0 > fc1):
                continue
            block = grid[r0 : r1 + 1, c0 : c1 + 1]
            if np.any(block == bg):
                continue
            left = [int(x) for x in block[:, 0]]
            if len(set(left)) != rows or set(left) != placeholder_colors:
                continue
            if not _isolated_block(grid, bg, r0, c0, r1, c1):
                continue
            mapping = {int(block[i, 0]): int(block[i, 1]) for i in range(rows)}
            if len(mapping) != rows:
                continue
            candidates.append((mapping, (r0, c0, r1, c1)))
    assert len(candidates) == 1, f"expected one legend, found {len(candidates)}"
    return candidates[0]


def _sprite_signature(coords: np.ndarray) -> tuple[tuple[tuple[int, int], ...], int, int]:
    r0, c0, r1, c1 = _bbox(coords)
    rel = tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))
    return rel, r1 - r0 + 1, c1 - c0 + 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find each placeholder's same-color sprite outside the frame, recolor it via the
    two-column legend, stamp it at the placeholder anchor, and clear all outside objects.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    frame_color, frame_bbox = _find_frame(g, bg)
    fr0, fc0, fr1, fc1 = frame_bbox

    inner = g[fr0 + 1 : fr1, fc0 + 1 : fc1]
    placeholder_mask = (inner != bg) & (inner != frame_color)
    placeholder_colors = set(map(int, np.unique(inner[placeholder_mask])))
    assert placeholder_colors

    placeholders: list[tuple[int, int, int]] = []
    for color in sorted(placeholder_colors):
        comps = _components(inner == color)
        assert comps, f"missing placeholder for color {color}"
        for comp in comps:
            assert len(comp) == 1, f"placeholder color {color} must appear as isolated single pixels"
            r, c = map(int, comp[0])
            placeholders.append((color, fr0 + 1 + r, fc0 + 1 + c))

    color_map, legend_bbox = _find_legend(g, bg, frame_bbox, placeholder_colors)
    lr0, lc0, lr1, lc1 = legend_bbox

    legend_mask = np.zeros_like(g, dtype=bool)
    legend_mask[lr0 : lr1 + 1, lc0 : lc1 + 1] = True
    outside_mask = np.ones_like(g, dtype=bool)
    outside_mask[fr0 : fr1 + 1, fc0 : fc1 + 1] = False
    outside_mask &= ~legend_mask

    sprites: dict[int, tuple[tuple[tuple[int, int], ...], int, int]] = {}
    for color in sorted(placeholder_colors):
        comps = _components((g == color) & outside_mask)
        assert len(comps) == 1, f"expected one sprite for color {color}, found {len(comps)}"
        sprites[color] = _sprite_signature(comps[0])

    out = np.full(g.shape, bg, dtype=int)
    out[fr0 : fr1 + 1, fc0 : fc1 + 1] = bg
    out[fr0, fc0 : fc1 + 1] = frame_color
    out[fr1, fc0 : fc1 + 1] = frame_color
    out[fr0 : fr1 + 1, fc0] = frame_color
    out[fr0 : fr1 + 1, fc1] = frame_color

    painted = np.zeros_like(g, dtype=bool)
    for src_color, anchor_r, anchor_c in placeholders:
        target_color = int(color_map[src_color])
        rel_cells, sh, sw = sprites[src_color]
        assert anchor_r + sh - 1 < fr1 and anchor_c + sw - 1 < fc1, "sprite exceeds frame interior"
        for dr, dc in rel_cells:
            rr = anchor_r + dr
            cc = anchor_c + dc
            assert fr0 < rr < fr1 and fc0 < cc < fc1
            assert not painted[rr, cc], "stamped sprites must not overlap"
            painted[rr, cc] = True
            out[rr, cc] = target_color

    return out


def _random_connected_sprite(rng: random.Random) -> np.ndarray:
    box_h = rng.randint(2, 4)
    box_w = rng.randint(2, 4)
    max_area = min(box_h * box_w, 6)
    area = rng.randint(3, max_area)
    cells = {(0, 0)}
    frontier = set()
    for dr, dc in DIR4:
        nr, nc = dr, dc
        if 0 <= nr < box_h and 0 <= nc < box_w:
            frontier.add((nr, nc))
    while len(cells) < area:
        if not frontier:
            return _random_connected_sprite(rng)
        cell = rng.choice(tuple(frontier))
        frontier.remove(cell)
        cells.add(cell)
        for dr, dc in DIR4:
            nr, nc = cell[0] + dr, cell[1] + dc
            if 0 <= nr < box_h and 0 <= nc < box_w and (nr, nc) not in cells:
                frontier.add((nr, nc))
    arr = np.zeros((box_h, box_w), dtype=bool)
    for r, c in cells:
        arr[r, c] = True
    coords = np.argwhere(arr)
    r0, c0, r1, c1 = _bbox(coords)
    arr = arr[r0 : r1 + 1, c0 : c1 + 1]
    if not arr[0, 0]:
        return _random_connected_sprite(rng)
    if arr.shape[0] == 1 and arr.shape[1] == 1:
        return _random_connected_sprite(rng)
    return arr


def _can_place(occ: np.ndarray, r: int, c: int, h: int, w: int, pad: int) -> bool:
    rr0 = r - pad
    cc0 = c - pad
    rr1 = r + h + pad - 1
    cc1 = c + w + pad - 1
    if rr0 < 0 or cc0 < 0 or rr1 >= occ.shape[0] or cc1 >= occ.shape[1]:
        return False
    return not np.any(occ[rr0 : rr1 + 1, cc0 : cc1 + 1])


def _mark_occ(occ: np.ndarray, r: int, c: int, h: int, w: int, pad: int) -> None:
    occ[r - pad : r + h + pad, c - pad : c + w + pad] = True


def _place_inside_frame(
    rng: random.Random,
    inner_h: int,
    inner_w: int,
    items: list[tuple[int, np.ndarray]],
) -> list[tuple[int, int, int]]:
    occ = np.zeros((inner_h, inner_w), dtype=bool)
    order = items[:]
    rng.shuffle(order)
    placed: list[tuple[int, int, int]] = []
    for color, sprite in order:
        h, w = sprite.shape
        success = False
        for _ in range(PLACE_RETRIES):
            r = rng.randint(0, inner_h - h)
            c = rng.randint(0, inner_w - w)
            if _can_place(occ, r, c, h, w, pad=1):
                _mark_occ(occ, r, c, h, w, pad=1)
                placed.append((color, r, c))
                success = True
                break
        if not success:
            raise ValueError("failed to place placeholder anchors")
    return placed


def _place_patch(
    rng: random.Random,
    grid: np.ndarray,
    occ: np.ndarray,
    patch: np.ndarray,
    bg: int,
) -> bool:
    h, w = patch.shape
    for _ in range(OUTSIDE_PLACE_RETRIES):
        r = rng.randint(1, grid.shape[0] - h - 2)
        c = rng.randint(1, grid.shape[1] - w - 2)
        if not _can_place(occ, r, c, h, w, pad=1):
            continue
        sub = grid[r : r + h, c : c + w]
        mask = patch != bg
        sub[mask] = patch[mask]
        _mark_occ(occ, r, c, h, w, pad=1)
        return True
    return False


def _build_legend_patch(
    rng: random.Random,
    placeholder_colors: list[int],
    output_colors: list[int],
) -> np.ndarray:
    rows = list(range(len(placeholder_colors)))
    rng.shuffle(rows)
    patch = np.zeros((len(rows), 2), dtype=int)
    for i, idx in enumerate(rows):
        patch[i, 0] = placeholder_colors[idx]
        patch[i, 1] = output_colors[idx]
    return patch


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    num_colors = rng.randint(2, 4)
    palette = rng.sample(list(range(10)), 2 + 2 * num_colors)
    bg = palette[0]
    frame_color = palette[1]
    placeholder_colors = palette[2 : 2 + num_colors]
    output_colors = palette[2 + num_colors :]

    sprites = {color: _random_connected_sprite(rng) for color in placeholder_colors}

    counts = [1] * num_colors
    for _ in range(rng.randint(0, 2)):
        counts[rng.randrange(num_colors)] += 1

    items: list[tuple[int, np.ndarray]] = []
    for color, count in zip(placeholder_colors, counts):
        for _ in range(count):
            items.append((color, sprites[color]))

    approx_area = sum((sprite.shape[0] + 1) * (sprite.shape[1] + 1) for _, sprite in items)
    side_min = max(
        max(sprite.shape[0] for sprite in sprites.values()) + 2,
        max(sprite.shape[1] for sprite in sprites.values()) + 2,
        int(np.ceil(np.sqrt(approx_area))),
    )
    if side_min > 14:
        return None
    inner_h = rng.randint(side_min, min(14, side_min + 3))
    inner_w = rng.randint(side_min, min(14, side_min + 4))

    try:
        placements = _place_inside_frame(rng, inner_h, inner_w, items)
    except ValueError:
        return None

    frame_h = inner_h + 2
    frame_w = inner_w + 2
    canvas_h = frame_h + rng.randint(8, 13)
    canvas_w = frame_w + rng.randint(8, 13)
    if canvas_h > 30 or canvas_w > 30:
        return None

    if canvas_h - frame_h < 4 or canvas_w - frame_w < 4:
        return None
    fr0 = rng.randint(2, canvas_h - frame_h - 2)
    fc0 = rng.randint(2, canvas_w - frame_w - 2)
    fr1 = fr0 + frame_h - 1
    fc1 = fc0 + frame_w - 1

    grid = np.full((canvas_h, canvas_w), bg, dtype=int)
    grid[fr0, fc0 : fc1 + 1] = frame_color
    grid[fr1, fc0 : fc1 + 1] = frame_color
    grid[fr0 : fr1 + 1, fc0] = frame_color
    grid[fr0 : fr1 + 1, fc1] = frame_color

    for color, r, c in placements:
        grid[fr0 + 1 + r, fc0 + 1 + c] = color

    occ = np.zeros_like(grid, dtype=bool)
    occ[fr0 - 1 : fr1 + 2, fc0 - 1 : fc1 + 2] = True

    legend_patch = _build_legend_patch(rng, placeholder_colors, output_colors)
    if not _place_patch(rng, grid, occ, legend_patch, bg):
        return None

    sprite_items = placeholder_colors[:]
    rng.shuffle(sprite_items)
    for color in sprite_items:
        sprite = np.full(sprites[color].shape, bg, dtype=int)
        sprite[sprites[color]] = color
        if not _place_patch(rng, grid, occ, sprite, bg):
            return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if out.shape != grid.shape:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
