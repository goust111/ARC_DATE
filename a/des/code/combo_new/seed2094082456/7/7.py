from __future__ import annotations

import random
import numpy as np


LEGEND_BLOCK_WIDTH = 5  # 1 key column + up to 4 template columns
BG = 0
COLOR_POOL = [1, 2, 3, 4, 5, 6, 7, 8, 9]
GEN_RETRIES = 400


def _background_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[counts.argmax()])


def _row_spans(mask: np.ndarray) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    r = 0
    while r < len(mask):
        if not mask[r]:
            r += 1
            continue
        r0 = r
        while r < len(mask) and mask[r]:
            r += 1
        spans.append((r0, r - 1))
    return spans


def _parse_legend(grid: np.ndarray, bg: int) -> tuple[dict[int, list[tuple[int, int]]], int]:
    cols = np.flatnonzero(np.any(grid != bg, axis=0))
    assert cols.size, "grid must contain a left legend"
    key_c = int(cols[0])
    block = grid[:, key_c : min(grid.shape[1], key_c + LEGEND_BLOCK_WIDTH)]
    active_rows = np.any(block != bg, axis=1)
    spans = _row_spans(active_rows)
    assert spans, "legend must contain at least one entry"

    templates: dict[int, list[tuple[int, int]]] = {}
    for r0, r1 in spans:
        key_rows = np.flatnonzero(grid[r0 : r1 + 1, key_c] != bg)
        assert key_rows.size == 1, "each legend entry must have exactly one key pixel"
        key_r = r0 + int(key_rows[0])
        color = int(grid[key_r, key_c])

        offsets: set[tuple[int, int]] = set()
        for rr in range(r0, r1 + 1):
            for cc in range(key_c + 1, min(grid.shape[1], key_c + LEGEND_BLOCK_WIDTH)):
                if int(grid[rr, cc]) == color:
                    offsets.add((int(rr - key_r), int(cc - key_c)))
        assert offsets, "each legend entry must provide a non-empty template"
        templates[color] = sorted(offsets)

    main_c0 = min(grid.shape[1], key_c + LEGEND_BLOCK_WIDTH)
    return templates, main_c0


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each colored seed by stamping its legend template into the main area.

    Input structure:
    - Background cells are black.
    - A vertical legend sits on the left: each entry has one color key pixel and, immediately to its right, a same-color 3x3 or 4x4 template patch.
    - Entries are separated by one background row, while the right side contains scattered single-pixel seeds whose colors match legend keys.

    Rule:
    - Read every legend entry as a set of relative offsets, using the key pixel as the origin.
    - Remove the legend area and keep only the right-side main area as the output canvas.
    - Process seeds in row-major order; for each seed, stamp its template at the seed position.
    - A stamped cell is written only when the target cell is background; existing seeds and earlier stamps block later growth.

    Key insight:
    - The left legend is not a recoloring table but a per-color shape library: each seed expands by the exact offset pattern shown beside its key, and row-major blocking determines the final visible growth.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    templates, main_c0 = _parse_legend(g, bg)
    assert main_c0 < g.shape[1], "main area must exist to the right of the legend"

    out = g[:, main_c0:].copy()
    seeds = sorted(map(tuple, np.argwhere(out != bg)))
    for r, c in seeds:
        color = int(out[r, c])
        assert color in templates, "every seed color must appear in the legend"
        for dr, dc in templates[color]:
            rr = int(r + dr)
            cc = int(c + dc)
            if 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1] and int(out[rr, cc]) == bg:
                out[rr, cc] = color
    return out


def _is_connected(cells: set[tuple[int, int]]) -> bool:
    start = next(iter(cells))
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (r + dr, c + dc)
            if nxt in cells and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == len(cells)


def _sample_template(size: int) -> tuple[int, tuple[tuple[int, int], ...]]:
    anchor_r = 1 if size == 3 else random.choice([1, 2])
    max_cells = min(size * size - 1, size + 4)

    for _ in range(200):
        target = random.randint(size + 1, max_cells)
        cells = {(anchor_r, 0)}
        while len(cells) < target:
            frontier: list[tuple[int, int]] = []
            for r, c in cells:
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    rr = r + dr
                    cc = c + dc
                    if 0 <= rr < size and 0 <= cc < size and (rr, cc) not in cells:
                        frontier.append((rr, cc))
            if not frontier:
                break
            cells.add(random.choice(frontier))

        if len(cells) != target or not _is_connected(cells):
            continue
        if any(not any(rr == r for rr, _ in cells) for r in range(size)):
            continue
        if not any(c == 0 for _, c in cells):
            continue
        if len({c for _, c in cells}) == 1:
            continue

        offsets = tuple(sorted((r - anchor_r, c + 1) for r, c in cells))
        return anchor_r, offsets

    raise RuntimeError("failed to sample a valid legend template")


def _has_interaction(
    seeds: list[tuple[int, tuple[int, int]]],
    templates: dict[int, tuple[tuple[int, int], ...]],
    shape: tuple[int, int],
) -> bool:
    h, w = shape
    seed_positions = {pos for _color, pos in seeds}
    targets: dict[tuple[int, int], int] = {}
    for color, (r, c) in seeds:
        for dr, dc in templates[color]:
            rr = r + dr
            cc = c + dc
            if not (0 <= rr < h and 0 <= cc < w):
                continue
            pos = (rr, cc)
            if pos in seed_positions:
                return True
            targets[pos] = targets.get(pos, 0) + 1
            if targets[pos] > 1:
                return True
    return False


def _place_interacting_pair(
    colors: list[int],
    templates: dict[int, tuple[tuple[int, int], ...]],
    h: int,
    w: int,
) -> list[tuple[int, tuple[int, int]]] | None:
    for _ in range(200):
        color1 = random.choice(colors)
        offsets = list(templates[color1])
        random.shuffle(offsets)
        for dr, dc in offsets:
            r_min = max(0, -dr)
            r_max = h - 1 - max(0, dr)
            c_min = max(0, -dc)
            c_max = w - 1 - max(0, dc)
            if r_min > r_max or c_min > c_max:
                continue
            r1 = random.randint(r_min, r_max)
            c1 = random.randint(c_min, c_max)
            r2 = r1 + dr
            c2 = c1 + dc
            if (r1, c1) == (r2, c2):
                continue
            color2 = random.choice(colors)
            return [(color1, (r1, c1)), (color2, (r2, c2))]
    return None


def _random_free_pos(used: set[tuple[int, int]], h: int, w: int) -> tuple[int, int] | None:
    spots = [(r, c) for r in range(h) for c in range(w) if (r, c) not in used]
    if not spots:
        return None
    return random.choice(spots)


def _sample_seed_positions(
    colors: list[int],
    templates: dict[int, tuple[tuple[int, int], ...]],
    h: int,
    w: int,
    force_interaction: bool,
) -> list[tuple[int, tuple[int, int]]] | None:
    if h * w < len(colors):
        return None

    seed_colors = colors.copy()
    total = random.randint(max(4, len(colors)), min(8, h * w))
    while len(seed_colors) < total:
        seed_colors.append(random.choice(colors))
    random.shuffle(seed_colors)

    seeds: list[tuple[int, tuple[int, int]]] = []
    used: set[tuple[int, int]] = set()

    if force_interaction:
        pair = _place_interacting_pair(colors, templates, h, w)
        if pair is None:
            return None
        seeds.extend(pair)
        used.update(pos for _color, pos in pair)
        remaining = seed_colors[:-2] if len(seed_colors) >= 2 else []
    else:
        remaining = seed_colors

    for color in remaining:
        pos = _random_free_pos(used, h, w)
        if pos is None:
            return None
        seeds.append((color, pos))
        used.add(pos)

    if not force_interaction and _has_interaction(seeds, templates, (h, w)):
        return None
    if force_interaction and not _has_interaction(seeds, templates, (h, w)):
        return None
    return seeds


def _build_case() -> np.ndarray | None:
    entry_count = random.randint(2, 4)
    left_pad = random.randint(0, 2)

    if entry_count >= 2 and random.random() < 0.5:
        sizes = [3, 4]
        while len(sizes) < entry_count:
            sizes.append(random.choice([3, 4]))
        random.shuffle(sizes)
    else:
        sizes = [random.choice([3, 4]) for _ in range(entry_count)]

    top_pad = random.randint(0, 2)
    bottom_pad = random.randint(0, 4)
    height = top_pad + bottom_pad + sum(sizes) + (entry_count - 1)
    if height < 8:
        bottom_pad += 8 - height
        height = 8
    if height > 30:
        return None

    max_main_w = 30 - left_pad - LEGEND_BLOCK_WIDTH
    if max_main_w < 8:
        return None
    main_w = random.randint(8, min(20, max_main_w))
    width = left_pad + LEGEND_BLOCK_WIDTH + main_w

    colors = random.sample(COLOR_POOL, entry_count)
    templates: dict[int, tuple[tuple[int, int], ...]] = {}
    render_info: dict[int, tuple[int, int, set[tuple[int, int]]]] = {}
    signatures: set[tuple[tuple[int, int], ...]] = set()
    for color, size in zip(colors, sizes, strict=False):
        for _ in range(100):
            anchor_r, offsets = _sample_template(size)
            if offsets in signatures:
                continue
            signatures.add(offsets)
            templates[color] = offsets
            local_cells = {(dr + anchor_r, dc - 1) for dr, dc in offsets}
            render_info[color] = (size, anchor_r, local_cells)
            break
        else:
            return None

    force_interaction = random.random() < 0.5
    seeds = _sample_seed_positions(colors, templates, height, main_w, force_interaction)
    if seeds is None:
        return None

    grid = np.full((height, width), BG, dtype=int)
    key_c = left_pad
    main_c0 = key_c + LEGEND_BLOCK_WIDTH

    row = top_pad
    for i, color in enumerate(colors):
        size, anchor_r, local_cells = render_info[color]
        key_r = row + anchor_r
        grid[key_r, key_c] = color
        for rr, cc in local_cells:
            grid[row + rr, key_c + 1 + cc] = color
        row += size
        if i != len(colors) - 1:
            row += 1

    for color, (r, c) in seeds:
        grid[r, main_c0 + c] = color

    out = main(grid)
    if out.shape != (height, main_w):
        return None
    if len({color for color, _pos in seeds}) < 2:
        return None
    if np.count_nonzero(out != BG) <= len(seeds):
        return None
    if force_interaction != _has_interaction(seeds, templates, out.shape):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _build_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid seed-template growth puzzle")
