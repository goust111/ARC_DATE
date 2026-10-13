from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)

# Pairing-table geometry.
TABLE_INNER_H = 3
SEQ_BOX_W = 5
SHAPE_BOX_H = 3
SHAPE_BOX_W = 3
TABLE_H = TABLE_INNER_H + 2
ENTRY_W = 1 + SEQ_BOX_W + 1 + SHAPE_BOX_W + 1

# Generator ranges.
ENTRY_COUNT_RANGE = (3, 5)
EXTRA_CHROMOSOME_RANGE = (0, 2)
TOP_H_RANGE = (8, 14)
TABLE_GAP_RANGE = (1, 3)
BOTTOM_PAD_RANGE = (0, 2)
SIDE_PAD_RANGE = (1, 5)
WIDTH_EXTRA_RANGE = (0, 8)
GENERATION_ATTEMPTS = 500
PLACEMENT_ATTEMPTS = 300


SHAPE_LIBRARY = (
    np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]], dtype=int),
    np.array([[0, 0, 1], [0, 0, 1], [1, 1, 1]], dtype=int),
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=int),
    np.array([[0, 1, 0], [0, 1, 0], [1, 1, 1]], dtype=int),
    np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=int),
    np.array([[0, 1, 1], [0, 1, 0], [1, 1, 0]], dtype=int),
    np.array([[1, 1, 1], [1, 0, 0], [1, 1, 1]], dtype=int),
    np.array([[1, 1, 1], [0, 0, 1], [1, 1, 1]], dtype=int),
)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _runs(values: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i, value in enumerate(values):
        if bool(value):
            if start is None:
                start = int(i)
        elif start is not None:
            runs.append((start, int(i - 1)))
            start = None
    if start is not None:
        runs.append((start, int(len(values) - 1)))
    return runs


def _crop_nonblack(patch: np.ndarray) -> np.ndarray:
    coords = np.argwhere(patch != BLACK)
    assert coords.size > 0, "expected non-empty patch"
    r0, c0, r1, c1 = _bbox(coords)
    return patch[r0 : r1 + 1, c0 : c1 + 1]


def _rotate_cw(patch: np.ndarray) -> np.ndarray:
    return np.rot90(patch, -1)


def _compose_output(pieces: list[np.ndarray]) -> np.ndarray:
    assert pieces, "expected at least one output piece"
    out_h = max(int(piece.shape[0]) for piece in pieces)
    out_w = sum(int(piece.shape[1]) for piece in pieces) + len(pieces) - 1
    out = np.full((out_h, out_w), BLACK, dtype=int)
    col = 0
    for piece in pieces:
        h, w = map(int, piece.shape)
        region = out[:h, col : col + w]
        region[piece != BLACK] = piece[piece != BLACK]
        col += w + 1
    return out


def _extract_chromosomes(top: np.ndarray) -> list[dict]:
    comps = _components((top != BLACK) & (top != GREY))
    assert comps, "expected chromosomes above the lookup table"

    chromosomes: list[dict] = []
    for coords in comps:
        r0, c0, r1, c1 = _bbox(coords)
        if r0 == r1:
            seq = tuple(int(top[r0, c]) for c in range(c0, c1 + 1))
            assert int(coords.shape[0]) == c1 - c0 + 1
            orient = "h"
        else:
            assert c0 == c1
            seq = tuple(int(top[r, c0]) for r in range(r0, r1 + 1))
            assert int(coords.shape[0]) == r1 - r0 + 1
            orient = "v"
        assert 3 <= len(seq) <= 5
        chromosomes.append({"seq": seq, "orient": orient, "r0": r0, "c0": c0})

    chromosomes.sort(key=lambda item: (int(item["c0"]), int(item["r0"])))
    return chromosomes


def _extract_lookup(table: np.ndarray) -> dict[tuple[int, ...], np.ndarray]:
    assert np.all(table[0, :] == GREY) and np.all(table[-1, :] == GREY)
    col_runs = _runs(np.all(table == GREY, axis=0))
    assert len(col_runs) >= 3 and len(col_runs) % 2 == 1

    entries: dict[tuple[int, ...], np.ndarray] = {}
    for i in range((len(col_runs) - 1) // 2):
        left_border = col_runs[2 * i]
        divider = col_runs[2 * i + 1]
        right_border = col_runs[2 * i + 2]

        seq_box = table[1:-1, left_border[1] + 1 : divider[0]]
        shape_box = table[1:-1, divider[1] + 1 : right_border[0]]
        assert seq_box.shape[1] >= 3
        assert shape_box.shape[0] >= 1 and shape_box.shape[1] >= 1

        seq_rows = [r for r in range(seq_box.shape[0]) if np.any(seq_box[r] != BLACK)]
        assert len(seq_rows) == 1
        row = seq_box[seq_rows[0]]
        color_cols = np.flatnonzero(row != BLACK)
        assert color_cols.size >= 3
        c0, c1 = int(color_cols[0]), int(color_cols[-1])
        assert np.all(row[c0 : c1 + 1] != BLACK)
        seq = tuple(int(v) for v in row[c0 : c1 + 1])

        shape = _crop_nonblack(shape_box)
        colors = set(map(int, np.unique(shape))) - {BLACK}
        assert len(colors) == 1
        entries[seq] = shape.copy()

    assert entries, "lookup table must contain at least one entry"
    return entries


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read each upper straight multicolor chromosome as an ordered color sequence,
    match that sequence against the lower grey lookup table, rotate the matched
    sprite clockwise for vertical chromosomes, then concatenate the resulting
    sprites from left to right using chromosome position order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    grey_coords = np.argwhere(g == GREY)
    assert grey_coords.size > 0, "expected a grey lookup table"
    tr0, tc0, tr1, tc1 = _bbox(grey_coords)
    top = g[:tr0, :]
    table = g[tr0 : tr1 + 1, tc0 : tc1 + 1]

    lookup = _extract_lookup(table)
    chromosomes = _extract_chromosomes(top)

    pieces: list[np.ndarray] = []
    for chrom in chromosomes:
        seq = chrom["seq"]
        assert seq in lookup, "every chromosome must appear in the lookup table"
        piece = lookup[seq]
        if chrom["orient"] == "v":
            piece = _rotate_cw(piece)
        pieces.append(piece)

    return _compose_output(pieces)


def _unique_sequences(count: int) -> list[tuple[int, ...]]:
    sequences: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    while len(sequences) < count:
        length = random.randint(3, 5)
        seq = tuple(random.choice(COLOR_POOL) for _ in range(length))
        if len(set(seq)) < 2:
            continue
        if seq in seen:
            continue
        seen.add(seq)
        sequences.append(seq)
    return sequences


def _shape_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.full(mask.shape, BLACK, dtype=int)
    patch[mask.astype(bool)] = int(color)
    return patch


def _expected_output(chromosomes: list[dict], mapping: dict[tuple[int, ...], np.ndarray]) -> np.ndarray:
    ordered = sorted(chromosomes, key=lambda item: (int(item["c0"]), int(item["r0"])))
    pieces: list[np.ndarray] = []
    for chrom in ordered:
        piece = mapping[chrom["seq"]]
        if chrom["orient"] == "v":
            piece = _rotate_cw(piece)
        pieces.append(piece)
    return _compose_output(pieces)


def _table_patch(entries: list[dict]) -> np.ndarray:
    width = 1 + len(entries) * (ENTRY_W - 1)
    table = np.full((TABLE_H, width), BLACK, dtype=int)
    table[0, :] = GREY
    table[-1, :] = GREY

    for i, entry in enumerate(entries):
        left = i * (ENTRY_W - 1)
        divider = left + 1 + SEQ_BOX_W
        right = left + ENTRY_W - 1
        table[:, left] = GREY
        table[:, divider] = GREY
        table[:, right] = GREY

        seq = entry["seq"]
        row = random.randint(1, TABLE_H - 2)
        start = (SEQ_BOX_W - len(seq)) // 2
        for j, color in enumerate(seq):
            table[row, left + 1 + start + j] = int(color)

        patch = entry["shape"]
        table[1 : 1 + patch.shape[0], divider + 1 : divider + 1 + patch.shape[1]] = patch

    return table


def _can_place(occupied: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    h_grid, w_grid = occupied.shape
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    if r0 < 0 or c0 < 0 or r1 >= h_grid or c1 >= w_grid:
        return False
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(h_grid - 1, r1 + 1)
    cc1 = min(w_grid - 1, c1 + 1)
    return not bool(occupied[rr0 : rr1 + 1, cc0 : cc1 + 1].any())


def _mark_line(occupied: np.ndarray, r0: int, c0: int, orient: str, length: int) -> None:
    if orient == "h":
        occupied[r0, c0 : c0 + length] = True
    else:
        occupied[r0 : r0 + length, c0] = True


def _place_chromosomes(
    grid: np.ndarray,
    top_h: int,
    specs: list[dict],
) -> list[dict] | None:
    occupied = np.zeros(grid.shape, dtype=bool)
    placed: list[dict] = []
    order = list(range(len(specs)))
    random.shuffle(order)
    order.sort(key=lambda i: len(specs[i]["seq"]), reverse=True)

    for idx in order:
        spec = specs[idx]
        seq = spec["seq"]
        orient = spec["orient"]
        length = len(seq)
        h = 1 if orient == "h" else length
        w = length if orient == "h" else 1
        success = False
        for _ in range(PLACEMENT_ATTEMPTS):
            r0 = random.randint(0, top_h - h)
            c0 = random.randint(0, grid.shape[1] - w)
            if not _can_place(occupied, r0, c0, h, w):
                continue
            if orient == "h":
                for j, color in enumerate(seq):
                    grid[r0, c0 + j] = int(color)
            else:
                for j, color in enumerate(seq):
                    grid[r0 + j, c0] = int(color)
            _mark_line(occupied, r0, c0, orient, length)
            placed.append({"seq": seq, "orient": orient, "r0": r0, "c0": c0})
            success = True
            break
        if not success:
            return None
    return placed


def _orientation_list(count: int) -> list[str]:
    orientations = [random.choice(("h", "v")) for _ in range(count)]
    if count >= 2 and len(set(orientations)) == 1:
        flip = random.randrange(count)
        orientations[flip] = "v" if orientations[flip] == "h" else "h"
    return orientations


def _gen_attempt() -> np.ndarray | None:
    entry_count = random.randint(*ENTRY_COUNT_RANGE)
    sequences = _unique_sequences(entry_count)

    masks = random.sample(list(SHAPE_LIBRARY), entry_count)
    shape_colors = random.sample(list(COLOR_POOL), entry_count)
    entries = [
        {"seq": seq, "shape": _shape_patch(mask, color)}
        for seq, mask, color in zip(sequences, masks, shape_colors)
    ]
    mapping = {entry["seq"]: entry["shape"] for entry in entries}

    extra = random.randint(*EXTRA_CHROMOSOME_RANGE)
    chromosome_seqs = sequences.copy()
    for _ in range(extra):
        chromosome_seqs.append(random.choice(sequences))
    random.shuffle(chromosome_seqs)
    orientations = _orientation_list(len(chromosome_seqs))
    chromosome_specs = [
        {"seq": seq, "orient": orient}
        for seq, orient in zip(chromosome_seqs, orientations)
    ]

    table = _table_patch(entries)
    top_h = random.randint(*TOP_H_RANGE)
    gap = random.randint(*TABLE_GAP_RANGE)
    bottom_pad = random.randint(*BOTTOM_PAD_RANGE)
    side_pad = random.randint(*SIDE_PAD_RANGE)
    min_width = max(table.shape[1] + 2 * side_pad, 6 * len(chromosome_specs) + 6)
    width = min_width + random.randint(*WIDTH_EXTRA_RANGE)
    height = top_h + gap + table.shape[0] + bottom_pad
    grid = np.full((height, width), BLACK, dtype=int)

    table_col = random.randint(side_pad, width - side_pad - table.shape[1])
    table_row = top_h + gap
    grid[table_row : table_row + table.shape[0], table_col : table_col + table.shape[1]] = table

    placed = _place_chromosomes(grid, top_h, chromosome_specs)
    if placed is None:
        return None

    expected = _expected_output(placed, mapping)
    produced = main(grid)
    if not np.array_equal(produced, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC input")
