from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 8
BLOCK = 3
MOTIF = 9
RING_SLOTS = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0), (1, 0)]
CENTER_SOURCE_COORDS = [(2, 2), (2, 4), (2, 6), (4, 6), (6, 6), (6, 4), (6, 2), (4, 2)]
CENTER_TARGET_COORDS = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0), (1, 0)]
MAIN_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]

GRID_H_RANGE = (12, 20)
GRID_W_RANGE = (17, 24)
MARKER_GAP_RANGE = (1, 3)
MARKER_SPAN = 15
OWN_COUNT_PATTERNS = {
    5: [
        [8, 7, 6, 5, 3, 2, 1, 0],
        [8, 6, 6, 5, 4, 2, 1, 0],
        [7, 6, 5, 5, 4, 3, 1, 1],
    ],
    6: [
        [5, 4, 4, 3, 3, 2, 2, 1],
        [6, 4, 4, 3, 3, 2, 1, 1],
    ],
}
GEN_ATTEMPTS = 600


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not bool(mask[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _dominant_color(block: np.ndarray, bg: int) -> int:
    values, counts = np.unique(block[block != bg], return_counts=True)
    assert len(values) > 0, "expected a non-empty block"
    top = int(np.argmax(counts))
    assert int(np.sum(counts == counts[top])) == 1, "expected a unique dominant color"
    return int(values[top])


def _extract_motif_bbox(grid: np.ndarray, bg: int) -> tuple[int, int, int, int]:
    comps = _components(grid != bg)
    assert comps, "expected foreground"
    largest = max(comps, key=len)
    r0, c0, r1, c1 = _bbox(largest)
    assert (r1 - r0 + 1, c1 - c0 + 1) == (MOTIF, MOTIF), "expected one 9x9 motif"
    return r0, c0, r1, c1


def _marker_colors(grid: np.ndarray, motif_bbox: tuple[int, int, int, int], bg: int) -> list[int]:
    r0, c0, r1, c1 = motif_bbox
    cells: list[tuple[int, int, int]] = []
    for r, c in np.argwhere(grid != bg):
        rr = int(r)
        cc = int(c)
        if r0 <= rr <= r1 and c0 <= cc <= c1:
            continue
        cells.append((rr, cc, int(grid[rr, cc])))
    assert len(cells) == 8, "expected eight marker pixels"
    rows = {r for r, _, _ in cells}
    assert len(rows) == 1, "expected one horizontal marker row"
    ordered = sorted(cells, key=lambda item: item[1])
    cols = [c for _, c, _ in ordered]
    assert len(cols) == len(set(cols)), "marker columns must be unique"
    return [color for _, _, color in ordered]


def _ring_blocks(motif: np.ndarray) -> list[np.ndarray]:
    blocks: list[np.ndarray] = []
    for br, bc in RING_SLOTS:
        r0 = br * BLOCK
        c0 = bc * BLOCK
        blocks.append(motif[r0 : r0 + BLOCK, c0 : c0 + BLOCK].copy())
    return blocks


def _assemble_ring(blocks: list[np.ndarray], bg: int) -> np.ndarray:
    motif = np.full((MOTIF, MOTIF), bg, dtype=int)
    for block, (br, bc) in zip(blocks, RING_SLOTS, strict=True):
        r0 = br * BLOCK
        c0 = bc * BLOCK
        motif[r0 : r0 + BLOCK, c0 : c0 + BLOCK] = block
    return motif


def _decode_center(motif: np.ndarray, bg: int) -> np.ndarray:
    center = np.full((BLOCK, BLOCK), bg, dtype=int)
    for (sr, sc), (tr, tc) in zip(CENTER_SOURCE_COORDS, CENTER_TARGET_COORDS, strict=True):
        center[tr, tc] = int(motif[sr, sc])
    colors = motif[motif != bg]
    assert len(colors) > 0, "expected non-background motif pixels"
    majority = Counter(map(int, colors.tolist())).most_common(1)[0][0]
    center[1, 1] = int(majority)
    return center


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder the scrambled 3x3 block ring by global dominant-color frequency, then decode the missing center from the restored inner border.

    Input structure:
    - A teal background contains one 9x9 foreground motif made of a 3x3 array of 3x3 blocks.
    - The middle 3x3 block is entirely teal, while the other eight blocks each have a unique dominant non-teal color plus a few auxiliary pixels.
    - Outside the motif there is one horizontal row of eight isolated marker pixels whose colors match the eight dominant block colors.

    Rule:
    - Identify the dominant color of each surrounding 3x3 block.
    - For each dominant color, count how many times that color appears anywhere in the 9x9 motif, including auxiliary appearances inside other blocks.
    - Sort the blocks by descending total count, breaking ties by the left-to-right order of the external marker row, and place the whole 3x3 blocks intact into the clockwise ring slots starting at the top-left corner.
    - After reordering, read the eight pixels adjacent to the central hole, copy them to the corresponding border cells of the missing center block, set the center cell to the motif's most frequent non-teal color, and erase the marker row back to teal.

    Key insight:
    - The marker row does not directly say where each block is now; it only resolves the ordering induced by global color-frequency counts, and only the correctly restored block arrangement exposes the right eight border pixels for center reconstruction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = int(Counter(map(int, g.flatten().tolist())).most_common(1)[0][0])
    motif_bbox = _extract_motif_bbox(g, bg)
    marker_order = _marker_colors(g, motif_bbox, bg)
    marker_rank = {color: idx for idx, color in enumerate(marker_order)}

    r0, c0, r1, c1 = motif_bbox
    motif = g[r0 : r1 + 1, c0 : c1 + 1].copy()
    ring_items = []
    for block in _ring_blocks(motif):
        main_color = _dominant_color(block, bg)
        ring_items.append({"block": block, "main": main_color})

    freq = {color: int(np.count_nonzero(motif == color)) for color in marker_order}
    ordered_items = sorted(ring_items, key=lambda item: (-freq[item["main"]], marker_rank[item["main"]]))
    restored = _assemble_ring([item["block"] for item in ordered_items], bg)
    restored[BLOCK : 2 * BLOCK, BLOCK : 2 * BLOCK] = _decode_center(restored, bg)

    out = g.copy()
    for r, c in np.argwhere(g != bg):
        rr = int(r)
        cc = int(c)
        if not (r0 <= rr <= r1 and c0 <= cc <= c1):
            out[rr, cc] = bg
    out[r0 : r1 + 1, c0 : c1 + 1] = restored
    return out


def _assign_aux_tokens(colors: list[int], own_count: int, extras: dict[int, int]) -> list[list[int]] | None:
    aux_per_block = BLOCK * BLOCK - own_count
    tokens: list[int] = []
    for color in colors:
        tokens.extend([color] * extras[color])

    for _ in range(300):
        assignments = [[] for _ in colors]
        random.shuffle(tokens)
        ok = True
        for color in tokens:
            candidates = [i for i, own in enumerate(colors) if own != color and len(assignments[i]) < aux_per_block]
            if not candidates:
                ok = False
                break
            random.shuffle(candidates)
            candidates.sort(key=lambda idx: (assignments[idx].count(color), -len(assignments[idx])))
            assignments[candidates[0]].append(color)
        if ok and all(len(block) == aux_per_block for block in assignments):
            return assignments
    return None


def _make_block(main_color: int, aux_colors: list[int], own_count: int) -> np.ndarray:
    block = np.full((BLOCK, BLOCK), main_color, dtype=int)
    positions = [(r, c) for r in range(BLOCK) for c in range(BLOCK)]
    chosen = random.sample(positions, BLOCK * BLOCK - own_count)
    random.shuffle(aux_colors)
    for (r, c), color in zip(chosen, aux_colors, strict=True):
        block[r, c] = int(color)
    return block


def _build_ranked_blocks() -> tuple[list[int], list[np.ndarray], dict[int, int], int]:
    own_count = random.choices([5, 6], weights=[3, 2])[0]
    extras_pattern = random.choice(OWN_COUNT_PATTERNS[own_count])
    ranked_colors = random.sample(MAIN_COLORS, len(MAIN_COLORS))
    extras = {color: extra for color, extra in zip(ranked_colors, extras_pattern, strict=True)}

    assignments = _assign_aux_tokens(ranked_colors, own_count, extras)
    if assignments is None:
        raise RuntimeError("failed to distribute auxiliary colors")

    blocks = []
    totals = {color: own_count + extras[color] for color in ranked_colors}
    for color, aux_colors in zip(ranked_colors, assignments, strict=True):
        blocks.append(_make_block(color, aux_colors, own_count))
    return ranked_colors, blocks, totals, own_count


def _place_marker_row(grid: np.ndarray, colors: list[int], row: int, start_col: int) -> None:
    for idx, color in enumerate(colors):
        grid[row, start_col + 2 * idx] = int(color)


def _render_input_grid(
    shape: tuple[int, int],
    motif_rc: tuple[int, int],
    marker_row: int,
    marker_col0: int,
    scrambled_blocks: list[np.ndarray],
    ranked_colors: list[int],
) -> np.ndarray:
    h, w = shape
    r0, c0 = motif_rc
    grid = np.full((h, w), BG, dtype=int)
    grid[r0 : r0 + MOTIF, c0 : c0 + MOTIF] = _assemble_ring(scrambled_blocks, BG)
    _place_marker_row(grid, ranked_colors, marker_row, marker_col0)
    return grid


def _expected_output(
    shape: tuple[int, int],
    motif_rc: tuple[int, int],
    ordered_blocks: list[np.ndarray],
) -> np.ndarray:
    h, w = shape
    r0, c0 = motif_rc
    grid = np.full((h, w), BG, dtype=int)
    motif = _assemble_ring(ordered_blocks, BG)
    motif[BLOCK : 2 * BLOCK, BLOCK : 2 * BLOCK] = _decode_center(motif, BG)
    grid[r0 : r0 + MOTIF, c0 : c0 + MOTIF] = motif
    return grid


def _center_signature(motif: np.ndarray) -> tuple[int, ...]:
    center = _decode_center(motif, BG)
    return tuple(int(x) for x in center.flatten())


def _choose_layout() -> tuple[tuple[int, int], tuple[int, int], int, int] | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    side = random.choice(["above", "below"])
    gap = random.randint(*MARKER_GAP_RANGE)

    if w < MARKER_SPAN:
        return None

    marker_c0 = random.randint(0, w - MARKER_SPAN)
    c0 = random.randint(0, w - MOTIF)

    if side == "above":
        r0_min = gap + 1
        r0_max = h - MOTIF
        if r0_min > r0_max:
            return None
        r0 = random.randint(r0_min, r0_max)
        marker_row = r0 - gap - 1
    else:
        r0_min = 0
        r0_max = h - MOTIF - gap - 1
        if r0_min > r0_max:
            return None
        r0 = random.randint(r0_min, r0_max)
        marker_row = r0 + MOTIF + gap

    return (h, w), (r0, c0), marker_row, marker_c0


def _gen_attempt() -> np.ndarray | None:
    layout = _choose_layout()
    if layout is None:
        return None
    shape, motif_rc, marker_row, marker_c0 = layout

    ranked_colors, ordered_blocks, totals, own_count = _build_ranked_blocks()
    if own_count == 5:
        has_tie = len(set(totals.values())) < len(totals)
    else:
        has_tie = True

    scramble = list(range(8))
    random.shuffle(scramble)
    if scramble == list(range(8)):
        return None
    scrambled_blocks = [ordered_blocks[idx] for idx in scramble]

    input_motif = _assemble_ring(scrambled_blocks, BG)
    ordered_motif = _assemble_ring(ordered_blocks, BG)
    if _center_signature(input_motif) == _center_signature(ordered_motif):
        return None

    inp = _render_input_grid(shape, motif_rc, marker_row, marker_c0, scrambled_blocks, ranked_colors)
    expected = _expected_output(shape, motif_rc, ordered_blocks)

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    colors = [int(color) for color in ranked_colors]
    freqs = [totals[color] for color in colors]
    if freqs[0] == freqs[1]:
        return None
    if has_tie and all(freqs[i] != freqs[i + 1] for i in range(len(freqs) - 1)):
        return None
    if not has_tie and any(freqs[i] == freqs[i + 1] for i in range(len(freqs) - 1)):
        return None
    if np.array_equal(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
