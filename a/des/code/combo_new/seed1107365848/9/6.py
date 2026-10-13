from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
AZURE = 8
BROWN = 9

ROTATION_BY_COLOR = {
    GREY: 0,
    RED: 1,
    BLUE: 2,
    GREEN: 3,
}
RING_COLORS = tuple(ROTATION_BY_COLOR.keys())
SEED_PALETTE = (YELLOW, MAGENTA, ORANGE, AZURE, BROWN)
MAX_GENERATION_ATTEMPTS = 500


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected at least one foreground pixel"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _border_values(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> set[int]:
    values = set(map(int, grid[r0, c0 : c1 + 1]))
    if r1 > r0:
        values.update(map(int, grid[r1, c0 : c1 + 1]))
    if c1 > c0:
        values.update(map(int, grid[r0 + 1 : r1, c0]))
        values.update(map(int, grid[r0 + 1 : r1, c1]))
    return values


def _ring_mask(shape: tuple[int, int], offset: int) -> np.ndarray:
    h, w = shape
    r0, c0 = offset, offset
    r1, c1 = h - 1 - offset, w - 1 - offset
    mask = np.zeros(shape, dtype=bool)
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    return mask


def _spacing_mask(shape: tuple[int, int], ring_offset: int) -> np.ndarray:
    return _ring_mask(shape, ring_offset + 1)


def _parse_structure(grid: np.ndarray) -> tuple[tuple[int, int], np.ndarray, list[int], np.ndarray]:
    r0, c0, r1, c1 = _bbox(grid != BLACK)
    structure = np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int)
    h, w = structure.shape

    ring_colors: list[int] = []
    offset = 0
    while h - 2 * offset >= 5 and w - 2 * offset >= 5:
        color_values = _border_values(structure, offset, offset, h - 1 - offset, w - 1 - offset)
        if len(color_values) != 1:
            break
        ring_color = next(iter(color_values))
        if ring_color not in ROTATION_BY_COLOR:
            break
        gap_values = _border_values(structure, offset + 1, offset + 1, h - 2 - offset, w - 2 - offset)
        if gap_values != {BLACK}:
            break
        ring_colors.append(int(ring_color))
        offset += 2

    assert ring_colors, "expected at least one concentric ring"
    seed = structure[offset : h - offset, offset : w - offset].copy()
    assert seed.size > 0 and np.any(seed != BLACK), "expected a non-empty center seed"
    return (int(r0), int(c0)), structure, ring_colors, seed


def _paint_seed_into_spacing(out: np.ndarray, ring_offset: int, seed: np.ndarray) -> None:
    h, w = out.shape
    band_r0 = ring_offset + 1
    band_c0 = ring_offset + 1
    band_r1 = h - 2 - ring_offset
    band_c1 = w - 2 - ring_offset
    band_h = band_r1 - band_r0 + 1
    band_w = band_c1 - band_c0 + 1

    ph, pw = seed.shape
    for rr in range(min(ph, band_h)):
        for cc in range(min(pw, band_w)):
            value = int(seed[rr, cc])
            if value == BLACK:
                continue
            tr = band_r0 + rr
            tc = band_c0 + cc
            if tr in (band_r0, band_r1) or tc in (band_c0, band_c1):
                out[tr, tc] = value


def _transform_structure(structure: np.ndarray, ring_colors: list[int], seed: np.ndarray) -> np.ndarray:
    out = structure.copy()
    cumulative_turns = 0
    total_rings = len(ring_colors)
    for inner_index, ring_color in enumerate(reversed(ring_colors)):
        cumulative_turns = (cumulative_turns + ROTATION_BY_COLOR[int(ring_color)]) % 4
        rotated_seed = np.rot90(seed, -cumulative_turns)
        ring_offset = 2 * (total_rings - 1 - inner_index)
        _paint_seed_into_spacing(out, ring_offset, rotated_seed)
    return out


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            out.append((int(rr), int(cc)))
    return out


def _connected(mask: np.ndarray) -> bool:
    cells = np.argwhere(mask)
    if cells.size == 0:
        return False
    h, w = mask.shape
    start = tuple(map(int, cells[0]))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for rr, cc in _neighbors4(r, c, h, w):
            if mask[rr, cc] and (rr, cc) not in seen:
                seen.add((rr, cc))
                stack.append((rr, cc))
    return len(seen) == int(mask.sum())


def _visible_signature(arr: np.ndarray) -> tuple[tuple[int, int], tuple[int, ...], tuple[int, ...]]:
    return (
        tuple(map(int, arr.shape)),
        tuple(map(int, arr[0, :])),
        tuple(map(int, arr[:, 0])),
    )


def _make_seed(h: int, w: int) -> np.ndarray | None:
    area = h * w
    lo = max(4, (area + 1) // 2)
    hi = max(lo, area - 1)
    n_cells = random.randint(lo, hi)

    start = (random.randrange(h), random.randrange(w))
    cells = {start}
    while len(cells) < n_cells:
        frontier = set()
        for r, c in cells:
            frontier.update((rr, cc) for rr, cc in _neighbors4(r, c, h, w) if (rr, cc) not in cells)
        if not frontier:
            return None
        cells.add(random.choice(tuple(frontier)))

    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True

    if not _connected(mask):
        return None
    if not mask[0, :].any() or not mask[-1, :].any() or not mask[:, 0].any() or not mask[:, -1].any():
        return None

    palette_size = random.randint(1, 3)
    palette = random.sample(SEED_PALETTE, palette_size)
    seed = np.zeros((h, w), dtype=int)
    shuffled = list(cells)
    random.shuffle(shuffled)
    for i, (r, c) in enumerate(shuffled):
        if i < len(palette):
            color = palette[i]
        else:
            color = random.choice(palette)
        seed[r, c] = int(color)

    rotations = [np.rot90(seed, -k) for k in range(4)]
    full_signatures = {(tuple(rot.shape), rot.tobytes()) for rot in rotations}
    visible_signatures = {_visible_signature(rot) for rot in rotations}
    if len(full_signatures) != 4 or len(visible_signatures) != 4:
        return None
    return seed


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate the central seed into each black spacing ring using the nested ring colors as cumulative rotation commands.

    Input structure:
    - Background is black.
    - A single central structure contains 3-4 one-pixel-thick concentric rectangular rings, with a one-pixel black spacing ring just inside every colored ring.
    - Ring colors are chosen from red, blue, green, and grey, encoding clockwise quarter-turn counts.
    - The innermost remaining rectangle is a small non-symmetric seed pattern that may contain black background cells.

    Rule:
    - Read the colored rings from the innermost one outward.
    - Convert each ring color to a clockwise rotation amount: grey -> 0, red -> 90, blue -> 180, green -> 270.
    - Keep a cumulative rotation total; for each ring, rotate the original seed by that total.
    - Place the rotated seed at the top-left of the black spacing ring just inside that colored ring, clipping to the spacing ring's cells so only pixels that land on that one-cell border are drawn.
    - Preserve the original seed, all colored rings, and the outer black background.

    Key insight:
    - The ring colors are instructions rather than decoration, and the rotations accumulate from inner to outer layers; each filled spacing ring shows a clipped copy of the same seed under the current cumulative orientation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    (top, left), structure, ring_colors, seed = _parse_structure(g)
    transformed = _transform_structure(structure, ring_colors, seed)

    out = g.copy()
    h, w = structure.shape
    out[top : top + h, left : left + w] = transformed
    return out


def _build_input(seed: np.ndarray, ring_colors: list[int], extra_h: int, extra_w: int) -> np.ndarray:
    ring_count = len(ring_colors)
    seed_h, seed_w = seed.shape
    structure_h = seed_h + 4 * ring_count
    structure_w = seed_w + 4 * ring_count
    structure = np.zeros((structure_h, structure_w), dtype=int)

    seed_offset = 2 * ring_count
    structure[seed_offset : seed_offset + seed_h, seed_offset : seed_offset + seed_w] = seed
    for i, color in enumerate(ring_colors):
        structure[_ring_mask(structure.shape, 2 * i)] = int(color)

    top = extra_h // 2
    left = extra_w // 2
    canvas = np.zeros((structure_h + extra_h, structure_w + extra_w), dtype=int)
    canvas[top : top + structure_h, left : left + structure_w] = structure
    return canvas


def _filled_spacing_counts(inp: np.ndarray, out: np.ndarray) -> list[int]:
    (top, left), structure, ring_colors, _seed = _parse_structure(inp)
    h, w = structure.shape
    sub = out[top : top + h, left : left + w]
    counts = []
    for i in range(len(ring_colors)):
        ring_offset = 2 * i
        mask = _spacing_mask(sub.shape, ring_offset)
        counts.append(int(np.count_nonzero(sub[mask] != BLACK)))
    return counts


def _gen_attempt() -> np.ndarray | None:
    ring_count = random.choice((3, 4))
    seed_h = random.randint(2, 8)
    seed_w = random.randint(2, 8)
    seed = _make_seed(seed_h, seed_w)
    if seed is None:
        return None

    ring_colors = [random.choice(RING_COLORS) for _ in range(ring_count)]
    cumulative = []
    total = 0
    for color in reversed(ring_colors):
        total = (total + ROTATION_BY_COLOR[int(color)]) % 4
        cumulative.append(int(total))
    if len(set(cumulative)) < 2:
        return None

    structure_h = seed_h + 4 * ring_count
    structure_w = seed_w + 4 * ring_count
    max_extra_h = min(8, 30 - structure_h)
    max_extra_w = min(8, 30 - structure_w)
    if max_extra_h < 0 or max_extra_w < 0:
        return None
    extra_h = random.randint(0, max_extra_h)
    extra_w = random.randint(0, max_extra_w)

    inp = _build_input(seed, ring_colors, extra_h, extra_w)
    out = main(inp)
    if np.array_equal(inp, out):
        return None

    counts = _filled_spacing_counts(inp, out)
    if min(counts) < 2:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid concentric-ring rotation puzzle")
