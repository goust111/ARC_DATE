from __future__ import annotations

import random
from typing import Iterable

import numpy as np


WALL_COLOR = 5
USABLE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIR_PRIORITY = (
    ("right", (0, 1)),
    ("down", (1, 0)),
    ("left", (0, -1)),
    ("up", (-1, 0)),
)
DIR_TO_VEC = dict(DIR_PRIORITY)
GEN_RETRIES = 400


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _parse_horizontal_palette(strip: np.ndarray, bg: int) -> tuple[dict[int, np.ndarray], np.ndarray] | None:
    h, w = strip.shape
    if h != 3:
        return None

    mapping: dict[int, np.ndarray] = {}
    mask = np.zeros_like(strip, dtype=bool)
    c = 0
    found = False
    while c < w:
        if np.all(strip[:, c] == bg):
            c += 1
            continue

        if c + 4 > w:
            return None
        block = strip[:, c : c + 4]
        color = int(block[0, 0])
        if color == bg or not np.all(block[:, 0] == color):
            return None
        if not np.all((block[:, 1:] == bg) | (block[:, 1:] == color)):
            return None
        if not np.any(block[:, 1:] == color):
            return None
        if color in mapping:
            return None

        mapping[color] = block[:, 1:].copy()
        mask[:, c : c + 4] |= block != bg
        c += 4
        found = True

        while c < w and np.all(strip[:, c] == bg):
            c += 1

    return (mapping, mask) if found else None


def _parse_vertical_palette(strip: np.ndarray, bg: int) -> tuple[dict[int, np.ndarray], np.ndarray] | None:
    h, w = strip.shape
    if w != 4:
        return None

    mapping: dict[int, np.ndarray] = {}
    mask = np.zeros_like(strip, dtype=bool)
    r = 0
    found = False
    while r < h:
        if np.all(strip[r] == bg):
            r += 1
            continue

        if r + 3 > h:
            return None
        block = strip[r : r + 3, :]
        color = int(block[0, 0])
        if color == bg or not np.all(block[:, 0] == color):
            return None
        if not np.all((block[:, 1:] == bg) | (block[:, 1:] == color)):
            return None
        if not np.any(block[:, 1:] == color):
            return None
        if color in mapping:
            return None

        mapping[color] = block[:, 1:].copy()
        mask[r : r + 3, :] |= block != bg
        r += 3
        found = True

        while r < h and np.all(strip[r] == bg):
            r += 1

    return (mapping, mask) if found else None


def _find_palette(grid: np.ndarray, bg: int) -> tuple[dict[int, np.ndarray], np.ndarray, str]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    candidates: list[tuple[int, dict[int, np.ndarray], np.ndarray, str]] = []

    def try_side(side: str, parsed: tuple[dict[int, np.ndarray], np.ndarray] | None, strip_slice) -> None:
        if parsed is None:
            return
        mapping, local_mask = parsed
        if not mapping:
            return

        palette_mask = np.zeros_like(g, dtype=bool)
        palette_mask[strip_slice] = local_mask
        seed_like = (g != bg) & (g != WALL_COLOR) & ~palette_mask
        if not np.any(seed_like):
            return
        outside_colors = {int(v) for v in g[seed_like].ravel()}
        if not outside_colors.issubset(mapping.keys()):
            return

        score = int(seed_like.sum())
        if side == "top" and h > 3 and np.all(g[3] == bg):
            score += 1000
        if side == "bottom" and h > 3 and np.all(g[h - 4] == bg):
            score += 1000
        if side == "left" and w > 4 and np.all(g[:, 4] == bg):
            score += 1000
        if side == "right" and w > 4 and np.all(g[:, w - 5] == bg):
            score += 1000
        candidates.append((score, mapping, palette_mask, side))

    if h >= 3:
        try_side("top", _parse_horizontal_palette(g[:3, :], bg), np.s_[:3, :])
        try_side("bottom", _parse_horizontal_palette(g[h - 3 :, :], bg), np.s_[h - 3 :, :])
    if w >= 4:
        try_side("left", _parse_vertical_palette(g[:, :4], bg), np.s_[:, :4])
        try_side("right", _parse_vertical_palette(g[:, w - 4 :], bg), np.s_[:, w - 4 :])

    assert candidates, "palette strip not found"
    candidates.sort(key=lambda item: (item[0], len(item[1])), reverse=True)
    _score, mapping, palette_mask, side = candidates[0]
    return mapping, palette_mask, side


def _seed_cells(grid: np.ndarray, bg: int, palette_mask: np.ndarray) -> list[tuple[int, int, int]]:
    mask = (grid != bg) & (grid != WALL_COLOR) & ~palette_mask
    return sorted((int(r), int(c), int(grid[r, c])) for r, c in np.argwhere(mask))


def _ray_length(grid: np.ndarray, bg: int, r: int, c: int, dr: int, dc: int) -> int:
    h, w = grid.shape
    steps = 0
    rr, cc = r + dr, c + dc
    while 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) == bg:
        steps += 1
        rr += dr
        cc += dc
    return steps


def _copy_top_left(r: int, c: int, direction: str, copy_index: int) -> tuple[int, int]:
    offset = 3 * copy_index
    if direction == "right":
        return r - 1, c + 1 + offset
    if direction == "left":
        return r - 1, c - 3 - offset
    if direction == "down":
        return r + 1 + offset, c - 1
    if direction == "up":
        return r - 3 - offset, c - 1
    raise ValueError(direction)


def _footprint_clear(grid: np.ndarray, bg: int, grown_mask: np.ndarray, top: int, left: int) -> bool:
    h, w = grid.shape
    if top < 0 or left < 0 or top + 3 > h or left + 3 > w:
        return False
    patch = grid[top : top + 3, left : left + 3]
    return bool(np.all(patch == bg) and not grown_mask[top : top + 3, left : left + 3].any())


def _render_growth(out: np.ndarray, grown_mask: np.ndarray, template: np.ndarray, top: int, left: int, bg: int) -> None:
    patch = out[top : top + 3, left : left + 3]
    fill = template != bg
    patch[fill] = template[fill]
    grown_mask[top : top + 3, left : left + 3] = True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow repeated 3x3 palette templates outward from colored seeds in the direction with the most free space.

    Input structure:
    - The background is black, and one grid edge contains a palette made of 3x4 blocks.
    - In each palette block, the left column is a solid index color and the right 3x3 area is a two-color template drawn with that index color and the background.
    - Away from the palette, the main area contains single-pixel colored seeds and grey wall cells that act as obstacles.

    Rule:
    - Read the palette as a mapping from seed color to its 3x3 template, then clear the palette in the output.
    - For each seed, measure the contiguous background run lengths in the four cardinal directions and choose the largest, breaking ties with the priority right > down > left > up.
    - Starting next to the seed on that chosen side, place unrotated copies of the seed's 3x3 template in a straight line, one 3x3 block after another, stopping before a copy would hit a wall, a non-background cell, the grid boundary, or another seed's already-grown region.
    - Keep the original seeds and walls unchanged.

    Key insight:
    - The edge palette is both a color lookup table and the source of each template's fixed orientation, while the seed's local free-space comparison chooses the growth direction.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    mapping, palette_mask, _side = _find_palette(g, bg)

    seeds = _seed_cells(g, bg, palette_mask)
    assert seeds, "at least one seed is required"

    out = g.copy()
    out[palette_mask] = bg
    grown_mask = np.zeros_like(g, dtype=bool)

    directions: dict[tuple[int, int], str] = {}
    for r, c, color in seeds:
        assert color in mapping, "every seed color must appear in the palette"
        lengths = {name: _ray_length(g, bg, r, c, dr, dc) for name, (dr, dc) in DIR_PRIORITY}
        best = max(lengths.values())
        for name, _vec in DIR_PRIORITY:
            if lengths[name] == best:
                directions[(r, c)] = name
                break

    for r, c, color in seeds:
        direction = directions[(r, c)]
        template = mapping[color]
        copy_index = 0
        while True:
            top, left = _copy_top_left(r, c, direction, copy_index)
            if not _footprint_clear(g, bg, grown_mask, top, left):
                break
            _render_growth(out, grown_mask, template, top, left, bg)
            copy_index += 1

    return out


def _random_template_masks(count: int) -> list[np.ndarray] | None:
    masks: list[np.ndarray] = []
    seen: set[bytes] = set()
    extras = [(0, 0), (0, 2), (1, 1), (2, 0), (2, 2)]

    for _ in range(count):
        for _attempt in range(200):
            mask = np.zeros((3, 3), dtype=bool)
            mask[0, 1] = True
            mask[1, 0] = True
            mask[1, 2] = True
            mask[2, 1] = True
            extra_count = random.randint(1, 3)
            for cell in random.sample(extras, extra_count):
                mask[cell] = True

            if np.array_equal(mask, np.flipud(mask)) and np.array_equal(mask, np.fliplr(mask)):
                continue
            key = mask.tobytes()
            if key in seen:
                continue
            seen.add(key)
            masks.append(mask)
            break
        else:
            return None

    return masks


def _build_palette(
    grid: np.ndarray,
    side: str,
    colors: list[int],
    masks: dict[int, np.ndarray],
) -> set[tuple[int, int]]:
    h, w = grid.shape
    palette_cells: set[tuple[int, int]] = set()

    if side in ("top", "bottom"):
        total_w = len(colors) * 4 + (len(colors) - 1)
        min_c = 1 if w - total_w >= 2 else 0
        max_c = w - total_w - 1 if w - total_w >= 2 else w - total_w
        start_c = random.randint(min_c, max_c)
        row0 = 0 if side == "top" else h - 3

        for i, color in enumerate(colors):
            c0 = start_c + 5 * i
            block = np.zeros((3, 4), dtype=int)
            block[:, 0] = color
            block[:, 1:] = np.where(masks[color], color, 0)
            grid[row0 : row0 + 3, c0 : c0 + 4] = block
            for rr, cc in np.argwhere(block != 0):
                palette_cells.add((row0 + int(rr), c0 + int(cc)))
    else:
        total_h = len(colors) * 3 + (len(colors) - 1)
        min_r = 1 if h - total_h >= 2 else 0
        max_r = h - total_h - 1 if h - total_h >= 2 else h - total_h
        start_r = random.randint(min_r, max_r)
        col0 = 0 if side == "left" else w - 4

        for i, color in enumerate(colors):
            r0 = start_r + 4 * i
            block = np.zeros((3, 4), dtype=int)
            block[:, 0] = color
            block[:, 1:] = np.where(masks[color], color, 0)
            grid[r0 : r0 + 3, col0 : col0 + 4] = block
            for rr, cc in np.argwhere(block != 0):
                palette_cells.add((r0 + int(rr), col0 + int(cc)))

    return palette_cells


def _main_bounds(side: str, h: int, w: int) -> tuple[int, int, int, int]:
    if side == "top":
        return 5, h - 2, 1, w - 2
    if side == "bottom":
        return 1, h - 6, 1, w - 2
    if side == "left":
        return 1, h - 2, 6, w - 2
    if side == "right":
        return 1, h - 2, 1, w - 7
    raise ValueError(side)


def _distribute_directions(count: int) -> list[str]:
    base = [name for name, _vec in DIR_PRIORITY]
    out: list[str] = []
    while len(out) < count:
        random.shuffle(base)
        out.extend(base)
    random.shuffle(out)
    return out[:count]


def _distribute_colors(colors: list[int], count: int) -> list[int]:
    out = colors[:]
    while len(out) < count:
        out.append(random.choice(colors))
    random.shuffle(out)
    return out[:count]


def _tie_group(direction: str) -> set[str]:
    if direction == "up" or random.random() < 0.6:
        return {direction}
    if direction == "right":
        options = [
            {"right", "down"},
            {"right", "left"},
            {"right", "up"},
            {"right", "down", "left"},
            {"right", "down", "up"},
            {"right", "left", "up"},
            {"right", "down", "left", "up"},
        ]
        return random.choice(options)
    if direction == "down":
        return random.choice([{"down", "left"}, {"down", "up"}, {"down", "left", "up"}])
    if direction == "left":
        return {"left", "up"}
    return {direction}


def _ray_cells(seed: tuple[int, int], direction: str, length: int) -> set[tuple[int, int]]:
    r, c = seed
    dr, dc = DIR_TO_VEC[direction]
    return {(r + dr * step, c + dc * step) for step in range(1, length + 1)}


def _wall_cell(seed: tuple[int, int], direction: str, length: int) -> tuple[int, int]:
    r, c = seed
    dr, dc = DIR_TO_VEC[direction]
    return r + dr * (length + 1), c + dc * (length + 1)


def _stopper_bar(seed: tuple[int, int], direction: str, copies: int) -> set[tuple[int, int]]:
    r, c = seed
    offset = 3 * copies + 1
    if direction == "right":
        return {(r - 1, c + offset), (r, c + offset), (r + 1, c + offset)}
    if direction == "left":
        return {(r - 1, c - offset), (r, c - offset), (r + 1, c - offset)}
    if direction == "down":
        return {(r + offset, c - 1), (r + offset, c), (r + offset, c + 1)}
    if direction == "up":
        return {(r - offset, c - 1), (r - offset, c), (r - offset, c + 1)}
    raise ValueError(direction)


def _footprint_cells(seed: tuple[int, int], direction: str, copies: int) -> set[tuple[int, int]]:
    r, c = seed
    cells: set[tuple[int, int]] = set()
    for copy_index in range(copies):
        top, left = _copy_top_left(r, c, direction, copy_index)
        for rr in range(top, top + 3):
            for cc in range(left, left + 3):
                cells.add((rr, cc))
    return cells


def _expand_cells(cells: Iterable[tuple[int, int]], h: int, w: int, pad: int = 1) -> set[tuple[int, int]]:
    expanded: set[tuple[int, int]] = set()
    for r, c in cells:
        for rr in range(max(0, r - pad), min(h, r + pad + 1)):
            for cc in range(max(0, c - pad), min(w, c + pad + 1)):
                expanded.add((rr, cc))
    return expanded


def _inside_main(cell: tuple[int, int], bounds: tuple[int, int, int, int]) -> bool:
    r, c = cell
    r0, r1, c0, c1 = bounds
    return r0 <= r <= r1 and c0 <= c <= c1


def _sample_seed_spec(
    bounds: tuple[int, int, int, int],
    direction: str,
    color: int,
    reserved_exact: set[tuple[int, int]],
    blocked_pad: set[tuple[int, int]],
) -> dict[str, object] | None:
    r0, r1, c0, c1 = bounds
    copies = random.randint(1, 3)
    main_length = 3 * copies
    tie_group = _tie_group(direction)
    ray_lengths = {
        name: (main_length if name in tie_group else random.randint(0, main_length - 1))
        for name, _vec in DIR_PRIORITY
    }
    ray_lengths[direction] = main_length

    for _attempt in range(400):
        seed = (random.randint(r0 + 1, r1 - 1), random.randint(c0 + 1, c1 - 1))
        footprint = _footprint_cells(seed, direction, copies)
        stopper = _stopper_bar(seed, direction, copies)
        if not footprint or not stopper:
            continue
        if not all(_inside_main(cell, bounds) for cell in footprint | stopper | {seed}):
            continue

        ray_cells = set()
        walls = set(stopper)
        valid = True
        for name, _vec in DIR_PRIORITY:
            corridor = _ray_cells(seed, name, ray_lengths[name])
            wall = _wall_cell(seed, name, ray_lengths[name])
            ray_cells |= corridor
            if name != direction:
                walls.add(wall)
            if not all(_inside_main(cell, bounds) for cell in corridor | {wall}):
                valid = False
                break
        if not valid:
            continue

        exact_cells = footprint | ray_cells | walls | {seed}
        structural_cells = footprint | walls | {seed}
        if exact_cells & reserved_exact:
            continue
        if structural_cells & blocked_pad:
            continue

        return {
            "seed": seed,
            "direction": direction,
            "color": color,
            "copies": copies,
            "walls": walls,
            "footprint": footprint,
            "exact_cells": exact_cells,
            "structural_cells": structural_cells,
        }

    return None


def _render_expected(
    inp: np.ndarray,
    palette_cells: set[tuple[int, int]],
    masks: dict[int, np.ndarray],
    specs: list[dict[str, object]],
) -> np.ndarray:
    out = inp.copy()
    for r, c in palette_cells:
        out[r, c] = 0

    for spec in sorted(specs, key=lambda item: tuple(item["seed"])):
        seed_r, seed_c = spec["seed"]
        direction = str(spec["direction"])
        color = int(spec["color"])
        copies = int(spec["copies"])
        template = np.where(masks[color], color, 0)
        for copy_index in range(copies):
            top, left = _copy_top_left(seed_r, seed_c, direction, copy_index)
            patch = out[top : top + 3, left : left + 3]
            patch[template != 0] = color

    return out


def _gen_attempt() -> np.ndarray | None:
    side = random.choice(["top", "bottom", "left", "right"])
    palette_count = random.randint(2, 4)
    seed_count = random.randint(palette_count, min(5, palette_count + 2))

    if side in ("top", "bottom"):
        min_w = max(18, palette_count * 4 + (palette_count - 1) + 2)
        h = random.randint(18, 28)
        w = random.randint(min_w, 30)
    else:
        min_h = max(18, palette_count * 3 + (palette_count - 1) + 2)
        h = random.randint(min_h, 30)
        w = random.randint(18, 28)

    bounds = _main_bounds(side, h, w)
    if bounds[1] - bounds[0] + 1 < 9 or bounds[3] - bounds[2] + 1 < 9:
        return None

    colors = random.sample(list(USABLE_COLORS), palette_count)
    template_masks_list = _random_template_masks(palette_count)
    if template_masks_list is None:
        return None
    masks = {color: mask for color, mask in zip(colors, template_masks_list)}

    grid = np.zeros((h, w), dtype=int)
    palette_cells = _build_palette(grid, side, colors, masks)

    reserved_exact = set(palette_cells)
    blocked_pad = _expand_cells(palette_cells, h, w, pad=1)

    directions = _distribute_directions(seed_count)
    seed_colors = _distribute_colors(colors, seed_count)
    specs: list[dict[str, object]] = []

    for direction, color in zip(directions, seed_colors):
        spec = _sample_seed_spec(bounds, direction, color, reserved_exact, blocked_pad)
        if spec is None:
            return None
        specs.append(spec)
        reserved_exact |= set(spec["exact_cells"])
        blocked_pad |= _expand_cells(spec["structural_cells"], h, w, pad=1)

    for spec in specs:
        seed_r, seed_c = spec["seed"]
        grid[seed_r, seed_c] = int(spec["color"])
        for wr, wc in spec["walls"]:
            grid[wr, wc] = WALL_COLOR

    expected = _render_expected(grid, palette_cells, masks, specs)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _attempt in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
