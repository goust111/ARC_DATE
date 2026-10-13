import random
from collections import Counter

import numpy as np


GEOM_OPS = ("hflip", "vflip", "rot180")
GENERATE_ATTEMPTS = 500


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.reshape(-1))).most_common(1)[0][0])


def _bbox(mask: np.ndarray):
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box):
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    stack = [start]
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _find_separator_columns(grid: np.ndarray, bg: int):
    h, w = grid.shape
    candidates = []
    for color in sorted(set(map(int, np.unique(grid))) - {bg}):
        if int(np.count_nonzero(grid == color)) != 2 * h:
            continue
        cols = [c for c in range(w) if np.all(grid[:, c] == color)]
        if len(cols) != 2:
            continue
        candidates.append((color, cols))
    assert len(candidates) == 1, "expected one separator color with two solid columns"
    sep_color, cols = candidates[0]
    panel_w = (w - 2) // 3
    assert 3 * panel_w + 2 == w, "grid width must be 3*panel_w+2"
    assert cols == [panel_w, 2 * panel_w + 1], "separator columns must evenly split three panels"
    return sep_color, cols[0], cols[1], panel_w


def _split_panels(grid: np.ndarray):
    bg = _get_bg(grid)
    _sep_color, s1, _s2, panel_w = _find_separator_columns(grid, bg)
    return bg, [grid[:, 0:panel_w], grid[:, s1 + 1 : s1 + 1 + panel_w], grid[:, -panel_w:]]


def _sprite_box(panel: np.ndarray, bg: int):
    box = _bbox(panel != bg)
    assert box is not None, "each panel must contain one sprite"
    r0, c0, r1, c1 = box
    mask = panel != bg
    assert _connected(mask), "sprite must be 4-connected"
    return r0, c0, r1, c1


def _panel_crop(panel: np.ndarray, bg: int):
    box = _sprite_box(panel, bg)
    return box, _crop_bbox(panel, box)


def _apply_crop_geom(crop: np.ndarray, op_name: str):
    if op_name == "hflip":
        return np.fliplr(crop)
    if op_name == "vflip":
        return np.flipud(crop)
    if op_name == "rot180":
        return np.rot90(crop, 2)
    raise ValueError(op_name)


def _embed_same_box(panel_shape, box, crop: np.ndarray, bg: int):
    out = np.full(panel_shape, bg, dtype=int)
    r0, c0, r1, c1 = box
    assert crop.shape == (r1 - r0 + 1, c1 - c0 + 1)
    out[r0 : r1 + 1, c0 : c1 + 1] = crop
    return out


def _apply_panel_op(panel: np.ndarray, bg: int, op_desc):
    box, crop = _panel_crop(panel, bg)
    kind = op_desc[0]
    if kind in GEOM_OPS:
        new_crop = _apply_crop_geom(crop, kind)
    elif kind == "swap":
        a, b = op_desc[1], op_desc[2]
        new_crop = crop.copy()
        new_crop[crop == a] = b
        new_crop[crop == b] = a
    else:
        raise ValueError(op_desc)
    return _embed_same_box(panel.shape, box, new_crop, bg)


def _infer_swap_op(panel_a: np.ndarray, panel_b: np.ndarray, bg: int):
    mask_a = panel_a != bg
    mask_b = panel_b != bg
    if not np.array_equal(mask_a, mask_b):
        return None
    if not np.any(mask_a):
        return None

    mapping = {}
    for va, vb in zip(panel_a[mask_a], panel_b[mask_b]):
        va = int(va)
        vb = int(vb)
        if va in mapping and mapping[va] != vb:
            return None
        mapping[va] = vb

    changed = [color for color, mapped in mapping.items() if color != mapped]
    if len(changed) != 2:
        return None
    a, b = sorted(changed)
    if mapping.get(a) != b or mapping.get(b) != a:
        return None
    for color, mapped in mapping.items():
        if color not in (a, b) and color != mapped:
            return None
    return ("swap", a, b)


def _candidate_ops(panel_a: np.ndarray, panel_b: np.ndarray, bg: int):
    candidates = []
    for name in GEOM_OPS:
        if np.array_equal(_apply_panel_op(panel_a, bg, (name,)), panel_b):
            candidates.append((name,))
    swap_desc = _infer_swap_op(panel_a, panel_b, bg)
    if swap_desc is not None and np.array_equal(_apply_panel_op(panel_a, bg, swap_desc), panel_b):
        candidates.append(swap_desc)
    return candidates


def _all_ops_for_panel(panel: np.ndarray, bg: int):
    ops = [(name,) for name in GEOM_OPS]
    colors = sorted(set(map(int, np.unique(panel))) - {bg})
    for i, a in enumerate(colors):
        for b in colors[i + 1 :]:
            ops.append(("swap", int(a), int(b)))
    return ops


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer the demonstrated sprite operation from the first two panels, then apply that same operation to the third panel.

    Input structure:
    - Three equal-size panels are arranged left-to-right and separated by two solid separator columns whose color appears nowhere else.
    - Each panel contains exactly one 4-connected multicolor sprite on a dominant background.
    - Panel 1 and panel 2 show the same sprite before and after one hidden operation; panel 3 contains a different sprite.

    Rule:
    - Split the input at the separator columns to recover the three panels.
    - Compare panel 1 with panel 2 and find the unique operation that converts the first sprite into the second one.
    - The allowed demonstrated operations are horizontal mirror, vertical mirror, 180-degree rotation, or swapping one visible pair of sprite colors; geometric operations act inside the sprite's own tight bounding box and keep that box in place.
    - Apply the same inferred operation to panel 3 and output only that transformed third panel.

    Key insight:
    - The global rule is not a fixed transform; each input carries its own training pair, so the solver must read the local example first and then transfer the inferred operation to a new sprite.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg, panels = _split_panels(g)
    panel_a, panel_b, panel_c = panels
    candidates = _candidate_ops(panel_a, panel_b, bg)
    assert len(candidates) == 1, "demonstration must determine exactly one operation"
    return _apply_panel_op(panel_c, bg, candidates[0])


def _random_connected_mask(rng: random.Random, h: int, w: int, area: int):
    cells = {(rng.randrange(h), rng.randrange(w))}
    while len(cells) < area:
        r, c = rng.choice(tuple(cells))
        neighbors = []
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                neighbors.append((nr, nc))
        if neighbors:
            cells.add(rng.choice(neighbors))
        elif len(cells) < h * w:
            frontier = []
            for r0, c0 in cells:
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r0 + dr, c0 + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if frontier:
                cells.add(rng.choice(frontier))
            else:
                break
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    box = _bbox(mask)
    crop = mask if box is None else _crop_bbox(mask.astype(int), box).astype(bool)
    return crop


def _random_sprite_crop(rng: random.Random, palette, *, need_colors=None):
    for _ in range(160):
        h = rng.randint(3, 6)
        w = rng.randint(3, 6)
        area = rng.randint(max(5, (h * w) // 3), min(h * w - 1, max(6, (2 * h * w) // 3)))
        mask = _random_connected_mask(rng, h, w, area)
        box = _bbox(mask)
        if box is None:
            continue
        mask = _crop_bbox(mask.astype(int), box).astype(bool)
        hh, ww = mask.shape
        if hh < 3 or ww < 3:
            continue
        if not _connected(mask):
            continue
        if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue

        cell_count = int(mask.sum())
        color_count = rng.randint(2, min(4, len(palette), cell_count))
        colors = list(rng.sample(list(palette), color_count))
        if need_colors is not None:
            needed = list(need_colors)
            if any(color not in colors for color in needed):
                colors[: len(needed)] = needed
                colors = list(dict.fromkeys(colors))
                if len(colors) < color_count:
                    for color in palette:
                        if color not in colors:
                            colors.append(color)
                        if len(colors) == color_count:
                            break
        rng.shuffle(colors)

        coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
        rng.shuffle(coords)
        sprite = np.zeros(mask.shape, dtype=int)

        required = list(need_colors or [])
        for idx, color in enumerate(required):
            r, c = coords[idx]
            sprite[r, c] = int(color)

        for idx, (r, c) in enumerate(coords[len(required) :], start=len(required)):
            sprite[r, c] = int(colors[idx % len(colors)])

        present = set(map(int, np.unique(sprite))) - {0}
        if need_colors is not None and not set(map(int, need_colors)).issubset(present):
            continue
        if len(present) < 2:
            continue
        return sprite
    return None


def _place_crop(panel_h: int, panel_w: int, crop: np.ndarray, bg: int, rng: random.Random):
    h, w = crop.shape
    r0 = rng.randint(0, panel_h - h)
    c0 = rng.randint(0, panel_w - w)
    panel = np.full((panel_h, panel_w), bg, dtype=int)
    panel[r0 : r0 + h, c0 : c0 + w] = crop
    return panel


def _normalize_op(op_desc):
    if op_desc[0] != "swap":
        return op_desc
    a, b = sorted((int(op_desc[1]), int(op_desc[2])))
    return ("swap", a, b)


def _semantic_ok(panel_a, panel_b, panel_c, chosen_op, bg):
    candidates = [_normalize_op(op) for op in _candidate_ops(panel_a, panel_b, bg)]
    if candidates != [_normalize_op(chosen_op)]:
        return False

    chosen_out = _apply_panel_op(panel_c, bg, chosen_op)
    if np.array_equal(chosen_out, panel_c):
        return False

    for wrong_op in _all_ops_for_panel(panel_c, bg):
        if _normalize_op(wrong_op) == _normalize_op(chosen_op):
            continue
        if np.array_equal(chosen_out, _apply_panel_op(panel_c, bg, wrong_op)):
            return False

    if np.array_equal(chosen_out, panel_a) or np.array_equal(chosen_out, panel_b):
        return False
    return True


def _build_attempt(rng: random.Random):
    panel_h = rng.randint(7, 10)
    panel_w = rng.randint(7, 9)
    bg = 0
    sep_color = rng.choice([1, 2, 3, 4, 5, 6, 7, 8, 9])
    palette = [color for color in range(1, 10) if color != sep_color]

    op_name = rng.choice(["hflip", "vflip", "rot180", "swap"])
    if op_name == "swap":
        swap_a, swap_b = rng.sample(palette, 2)
        sprite_a = _random_sprite_crop(rng, palette, need_colors=(swap_a, swap_b))
        sprite_c = _random_sprite_crop(rng, palette, need_colors=(swap_a, swap_b))
        if sprite_a is None or sprite_c is None:
            return None
        op_desc = ("swap", int(swap_a), int(swap_b))
    else:
        sprite_a = _random_sprite_crop(rng, palette)
        sprite_c = _random_sprite_crop(rng, palette)
        if sprite_a is None or sprite_c is None:
            return None
        op_desc = (op_name,)

    if sprite_a.shape[0] > panel_h or sprite_a.shape[1] > panel_w:
        return None
    if sprite_c.shape[0] > panel_h or sprite_c.shape[1] > panel_w:
        return None

    panel_a = _place_crop(panel_h, panel_w, sprite_a, bg, rng)
    panel_b = _apply_panel_op(panel_a, bg, op_desc)
    panel_c = _place_crop(panel_h, panel_w, sprite_c, bg, rng)

    if not _semantic_ok(panel_a, panel_b, panel_c, op_desc, bg):
        return None

    grid = np.full((panel_h, panel_w * 3 + 2), bg, dtype=int)
    grid[:, :panel_w] = panel_a
    grid[:, panel_w] = sep_color
    grid[:, panel_w + 1 : 2 * panel_w + 1] = panel_b
    grid[:, 2 * panel_w + 1] = sep_color
    grid[:, 2 * panel_w + 2 :] = panel_c

    out = main(grid)
    if not np.array_equal(out, _apply_panel_op(panel_c, bg, op_desc)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_ATTEMPTS):
        rng = random.Random(random.randrange(1 << 30))
        grid = _build_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
