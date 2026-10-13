from __future__ import annotations

from collections import defaultdict

import numpy as np


DIRS = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # E, S, W, N in clockwise order
NONZERO_COLORS = tuple(range(1, 10))
TURN_FLIP = {"L": "R", "R": "L"}
RNG = np.random.default_rng()

# --------------- solver ---------------


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _classify_marker(coords: list[tuple[int, int]]) -> tuple[int, int, str] | None:
    if len(coords) != 3:
        return None
    r0, c0, r1, c1 = _bbox(coords)
    if (r1 - r0 + 1, c1 - c0 + 1) != (2, 2):
        return None
    rel = {(r - r0, c - c0) for r, c in coords}
    if len(rel) != 3:
        return None
    col_counts = [sum(dc == 0 for _, dc in rel), sum(dc == 1 for _, dc in rel)]
    row_counts = [sum(dr == 0 for dr, _ in rel), sum(dr == 1 for dr, _ in rel)]
    if sorted(col_counts) != [1, 2] or sorted(row_counts) != [1, 2]:
        return None
    turn = "R" if col_counts[0] == 2 else "L"
    return r0, c0, turn


def _classify_block(grid: np.ndarray, coords: list[tuple[int, int]]) -> tuple[int, int, int] | None:
    if len(coords) != 4:
        return None
    r0, c0, r1, c1 = _bbox(coords)
    if (r1 - r0 + 1, c1 - c0 + 1) != (2, 2):
        return None
    rel = {(r - r0, c - c0) for r, c in coords}
    if rel != {(0, 0), (0, 1), (1, 0), (1, 1)}:
        return None
    colors = {int(grid[r, c]) for r, c in coords}
    if len(colors) != 1:
        return None
    return r0, c0, next(iter(colors))


def _parse_legend(grid: np.ndarray) -> tuple[dict[str, int], np.ndarray]:
    marker_rows: dict[int, list[tuple[int, str, list[tuple[int, int]]]]] = defaultdict(list)
    block_rows: dict[int, list[tuple[int, int, list[tuple[int, int]]]]] = defaultdict(list)

    for comp in _components(grid != 0):
        marker = _classify_marker(comp)
        if marker is not None:
            r0, c0, turn = marker
            marker_rows[r0].append((c0, turn, comp))
            continue
        block = _classify_block(grid, comp)
        if block is not None:
            r0, c0, color = block
            block_rows[r0].append((c0, color, comp))

    mapping: dict[str, int] = {}
    table_mask = np.zeros_like(grid, dtype=bool)

    for r0, blocks in block_rows.items():
        if len(blocks) != 1:
            continue
        block_c0, out_color, block_coords = blocks[0]
        markers = [(c0, turn, coords) for c0, turn, coords in marker_rows.get(r0, []) if c0 < block_c0]
        if not (2 <= len(markers) <= 4):
            continue

        coords_set = set(block_coords)
        for _, _, coords in markers:
            coords_set.update(coords)

        c_lo = min(c for _, c in coords_set)
        c_hi = max(c for _, c in coords_set)
        band = np.argwhere(grid[r0 : r0 + 2, c_lo : c_hi + 1] != 0)
        if any((r0 + int(dr), c_lo + int(dc)) not in coords_set for dr, dc in band):
            continue

        markers.sort(key=lambda item: item[0])
        seq = "".join(turn for _, turn, _ in markers)
        mapping[seq] = int(out_color)
        for r, c in coords_set:
            table_mask[r, c] = True

    assert mapping, "legend not found"
    return mapping, table_mask


def _order_path(comp: list[tuple[int, int]]) -> list[tuple[int, int]]:
    pts = set(comp)
    degrees = {pt: sum(nei in pts for nei in _neighbors4(*pt)) for pt in pts}
    endpoints = [pt for pt, deg in degrees.items() if deg == 1]
    assert len(endpoints) == 2, "pipe must have exactly two endpoints"
    assert all(deg in (1, 2) for deg in degrees.values()), "pipe cannot branch"

    start = min(endpoints)
    ordered = [start]
    prev = None
    cur = start
    while len(ordered) < len(comp):
        nxts = [nei for nei in _neighbors4(*cur) if nei in pts and nei != prev]
        assert len(nxts) == 1, "pipe walk must stay unambiguous"
        nxt = nxts[0]
        ordered.append(nxt)
        prev, cur = cur, nxt
    return ordered


def _turn_sequence(path: list[tuple[int, int]]) -> str:
    steps = [(r1 - r0, c1 - c0) for (r0, c0), (r1, c1) in zip(path, path[1:])]
    turns: list[str] = []
    for prev, cur in zip(steps, steps[1:]):
        if prev == cur:
            continue
        cross = prev[0] * cur[1] - prev[1] * cur[0]
        assert cross != 0, "path turn must be 90 degrees"
        turns.append("L" if cross > 0 else "R")
    return "".join(turns)


def _canonical_turns(seq: str) -> str:
    mirror = "".join(TURN_FLIP[ch] for ch in reversed(seq))
    return min(seq, mirror)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each pipe by matching its turn sequence against the edge legend.

    Input structure:
    - Black background.
    - Several disconnected monochrome 4-connected paths with no branches.
    - A legend near the grid edge: each row has 2-4 L-shaped 3-pixel markers
      followed by a 2x2 color block.

    Rule:
    - For each pipe, trace it from both endpoints and turn the geometry into an
      L/R turn string.
    - Use the lexicographically smaller of the two endpoint readings as the key.
    - Find the matching legend row and repaint the whole pipe with the legend's
      color block color.
    - Leave the legend unchanged.
    """
    g = np.asarray(grid, dtype=int)
    mapping, table_mask = _parse_legend(g)

    out = g.copy()
    pipe_mask = (g != 0) & ~table_mask
    for comp in _components(pipe_mask):
        ordered = _order_path(comp)
        key = _canonical_turns(_turn_sequence(ordered))
        assert key in mapping, f"missing legend entry for {key}"
        out_color = mapping[key]
        for r, c in comp:
            out[r, c] = out_color
    return out


# --------------- generator ---------------

GRID_H_MIN = 18
GRID_H_MAX = 30
GRID_W_MIN = 18
GRID_W_MAX = 30
ENTRY_COUNT_MIN = 3
ENTRY_COUNT_MAX = 5
EXTRA_PIPE_MAX = 2
PIPE_AREA_MIN_H = 8
SEGMENT_LEN_MIN = 2
SEGMENT_LEN_MAX = 5
PATH_GEN_RETRIES = 300
PLACEMENT_RETRIES = 300
GENERATE_RETRIES = 300


def _inverse_reverse(seq: str) -> str:
    return "".join(TURN_FLIP[ch] for ch in reversed(seq))


def _all_canonical_sequences() -> dict[int, list[str]]:
    by_len: dict[int, set[str]] = defaultdict(set)
    for length in range(2, 5):
        for bits in range(1 << length):
            seq = "".join("R" if (bits >> i) & 1 else "L" for i in range(length))
            by_len[length].add(_canonical_turns(seq))
    return {length: sorted(values) for length, values in by_len.items()}


CANONICAL_BY_LEN = _all_canonical_sequences()
CANONICAL_ALL = sorted(
    {seq for values in CANONICAL_BY_LEN.values() for seq in values},
    key=lambda item: (len(item), item),
)


def _turn_dir(dir_idx: int, turn: str) -> int:
    return (dir_idx - 1) % 4 if turn == "L" else (dir_idx + 1) % 4


def _build_path(seq: str, start_dir: int, lengths: list[int]) -> list[tuple[int, int]]:
    dirs = [start_dir]
    for turn in seq:
        dirs.append(_turn_dir(dirs[-1], turn))
    assert len(dirs) == len(lengths)

    r = 0
    c = 0
    path = [(r, c)]
    for dir_idx, steps in zip(dirs, lengths):
        dr, dc = DIRS[dir_idx]
        for _ in range(steps):
            r += dr
            c += dc
            path.append((r, c))
    return path


def _normalize(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    r0 = min(r for r, _ in coords)
    c0 = min(c for _, c in coords)
    return [(r - r0, c - c0) for r, c in coords]


def _is_simple_path(coords: list[tuple[int, int]]) -> bool:
    if len(set(coords)) != len(coords):
        return False
    pts = set(coords)
    degrees = [sum(nei in pts for nei in _neighbors4(*pt)) for pt in pts]
    return degrees.count(1) == 2 and all(deg in (1, 2) for deg in degrees)


def _path_bbox(coords: list[tuple[int, int]]) -> tuple[int, int]:
    return max(r for r, _ in coords) + 1, max(c for _, c in coords) + 1


def _make_path_for_key(key: str, max_h: int, max_w: int) -> list[tuple[int, int]] | None:
    if max_h < 3 or max_w < 3:
        return None
    for _ in range(PATH_GEN_RETRIES):
        seq = key if RNG.integers(2) == 0 else _inverse_reverse(key)
        start_dir = int(RNG.integers(4))
        lengths = [int(RNG.integers(SEGMENT_LEN_MIN, SEGMENT_LEN_MAX + 1)) for _ in range(len(seq) + 1)]
        coords = _build_path(seq, start_dir, lengths)
        if not _is_simple_path(coords):
            continue
        coords = _normalize(coords)
        h, w = _path_bbox(coords)
        if h > max_h or w > max_w:
            continue
        if _canonical_turns(_turn_sequence(coords)) != key:
            continue
        return coords
    return None


def _marker_shape(turn: str, variant: int) -> list[tuple[int, int]]:
    if turn == "R":
        return [(0, 0), (1, 0), (0, 1)] if variant == 0 else [(0, 0), (1, 0), (1, 1)]
    return [(0, 1), (1, 1), (0, 0)] if variant == 0 else [(0, 1), (1, 1), (1, 0)]


def _draw_legend_row(grid: np.ndarray, r0: int, c0: int, seq: str, marker_color: int, out_color: int) -> None:
    x = c0
    for turn in seq:
        for dr, dc in _marker_shape(turn, int(RNG.integers(2))):
            grid[r0 + dr, x + dc] = marker_color
        x += 3
    grid[r0 : r0 + 2, x : x + 2] = out_color


def _mark_blocked(blocked: np.ndarray, coords: list[tuple[int, int]], pad: int = 1) -> None:
    h, w = blocked.shape
    for r, c in coords:
        for dr in range(-pad, pad + 1):
            for dc in range(-pad, pad + 1):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    blocked[nr, nc] = True


def _sample_sequences(count: int) -> list[str]:
    chosen: list[str] = []
    for length in (2, 3, 4):
        chosen.append(str(RNG.choice(CANONICAL_BY_LEN[length])))
    remaining = [seq for seq in CANONICAL_ALL if seq not in chosen]
    RNG.shuffle(remaining)
    chosen.extend(remaining[: count - 3])
    RNG.shuffle(chosen)
    return chosen[:count]


def _gen_attempt() -> np.ndarray | None:
    entry_count = int(RNG.integers(ENTRY_COUNT_MIN, ENTRY_COUNT_MAX + 1))
    seqs = _sample_sequences(entry_count)
    legend_height = 3 * entry_count
    max_legend_width = max(3 * len(seq) + 2 for seq in seqs)

    min_h = max(GRID_H_MIN, legend_height + PIPE_AREA_MIN_H)
    min_w = max(GRID_W_MIN, max_legend_width + 2)
    if min_h > GRID_H_MAX or min_w > GRID_W_MAX:
        return None

    h = int(RNG.integers(min_h, GRID_H_MAX + 1))
    w = int(RNG.integers(min_w, GRID_W_MAX + 1))
    legend_top = bool(RNG.integers(2))

    grid = np.zeros((h, w), dtype=int)
    output_colors = [int(v) for v in RNG.choice(NONZERO_COLORS, size=entry_count, replace=False)]
    expected_mapping: dict[str, int] = {}

    for i, (seq, out_color) in enumerate(zip(seqs, output_colors)):
        row_r = 3 * i if legend_top else h - legend_height + 3 * i
        row_w = 3 * len(seq) + 2
        row_c = int(RNG.integers(0, w - row_w + 1))
        marker_choices = [color for color in NONZERO_COLORS if color != out_color]
        marker_color = int(RNG.choice(marker_choices))
        _draw_legend_row(grid, row_r, row_c, seq, marker_color, out_color)
        expected_mapping[seq] = out_color

    blocked = np.zeros_like(grid, dtype=bool)
    if legend_top:
        blocked[:legend_height, :] = True
        pipe_r0, pipe_r1 = legend_height, h
    else:
        blocked[h - legend_height :, :] = True
        pipe_r0, pipe_r1 = 0, h - legend_height
    if pipe_r1 - pipe_r0 < PIPE_AREA_MIN_H:
        return None

    pipe_count = entry_count + int(RNG.integers(EXTRA_PIPE_MAX + 1))
    assignment = list(range(entry_count))
    assignment.extend(int(v) for v in RNG.integers(0, entry_count, size=pipe_count - entry_count))
    RNG.shuffle(assignment)

    placed: list[tuple[list[tuple[int, int]], int]] = []
    max_path_h = pipe_r1 - pipe_r0
    max_path_w = w
    for seq_idx in assignment:
        key = seqs[seq_idx]
        coords = _make_path_for_key(key, max_path_h, max_path_w)
        if coords is None:
            return None
        ph, pw = _path_bbox(coords)

        placed_here = False
        for _ in range(PLACEMENT_RETRIES):
            r_off = int(RNG.integers(pipe_r0, pipe_r1 - ph + 1))
            c_off = int(RNG.integers(0, w - pw + 1))
            shifted = [(r + r_off, c + c_off) for r, c in coords]
            if any(blocked[r, c] for r, c in shifted):
                continue
            pipe_color_choices = [color for color in NONZERO_COLORS if color != expected_mapping[key]]
            pipe_color = int(RNG.choice(pipe_color_choices))
            for r, c in shifted:
                grid[r, c] = pipe_color
            _mark_blocked(blocked, shifted, pad=1)
            placed.append((shifted, expected_mapping[key]))
            placed_here = True
            break
        if not placed_here:
            return None

    parsed_mapping, table_mask = _parse_legend(grid)
    if parsed_mapping != expected_mapping:
        return None
    if len(_components((grid != 0) & ~table_mask)) != pipe_count:
        return None

    expected = grid.copy()
    for coords, color in placed:
        for r, c in coords:
            expected[r, c] = color
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
