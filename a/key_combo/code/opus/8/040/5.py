from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
TOKEN_COLORS = (1, 2, 3, 4)
FILL = 2
WALL = 5
SEP = 7

GRID_H_RANGE = (11, 15)
TOKEN_COUNT_RANGE = (4, 6)
GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _find_separator(grid: np.ndarray) -> tuple[int, int]:
    candidates = []
    for r in range(1, grid.shape[0] - 1):
        color = int(grid[r, 0])
        if color != BG and np.all(grid[r, :] == color):
            candidates.append((r, color))
    assert len(candidates) == 1, f"expected one full separator row, got {candidates}"
    return candidates[0]


def _find_source_row(grid: np.ndarray, sep_row: int, sep_color: int) -> int:
    candidates = []
    for r in range(sep_row + 1, grid.shape[0]):
        row = grid[r]
        mask = (row != BG) & (row != WALL) & (row != sep_color)
        if np.any(mask):
            candidates.append(r)
    assert len(candidates) == 1, f"expected one source row, got {candidates}"
    return int(candidates[0])


def _fill_closed_regions(grid: np.ndarray, sep_row: int) -> np.ndarray:
    out = grid.copy()
    H, W = out.shape
    visited = np.zeros((H, W), dtype=bool)

    for r0 in range(sep_row + 1, H):
        for c0 in range(W):
            if int(out[r0, c0]) != BG or visited[r0, c0]:
                continue

            comp = []
            q = deque([(r0, c0)])
            visited[r0, c0] = True
            closed = True

            while q:
                r, c = q.popleft()
                comp.append((r, c))

                for nr, nc in _neighbors4(r, c):
                    if not (0 <= nr < H and 0 <= nc < W):
                        closed = False
                        continue
                    if nr <= sep_row:
                        if int(out[nr, nc]) == BG:
                            closed = False
                        continue
                    if int(out[nr, nc]) == BG and not visited[nr, nc]:
                        visited[nr, nc] = True
                        q.append((nr, nc))

            if closed:
                for r, c in comp:
                    out[r, c] = FILL

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide aligned source pixels by color match, then fill lower-panel background regions closed by the moved pixels and wall segments.

    Input structure:
    - A black grid is split by one full orange separator row into a narrow upper reference panel and a lower source panel.
    - The row just above the separator contains several single reference pixels, each in its own column.
    - One lower row contains aligned source pixels in those same columns, while grey wall pixels form short horizontal or vertical segments elsewhere in the lower panel.

    Rule:
    - Compare each source pixel with the reference pixel in the same column.
    - Matching source pixels move to the row immediately below the separator; non-matching source pixels move to the bottom row; all original source cells are erased to black.
    - After all moves, examine only the lower panel and find every 4-connected black region whose entire 4-neighbor boundary is non-black or the separator row.
    - Fill each such closed region with red, leaving the separator, references, walls, and moved source pixels unchanged.

    Key insight:
    - Color equality decides the new source-pixel layout, and that layout completes or fails to complete wall-bounded cavities that then determine which background cells become filled.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    H, W = g.shape
    sep_row, sep_color = _find_separator(g)
    ref_row = sep_row - 1
    src_row = _find_source_row(g, sep_row, sep_color)

    ref_cols = [int(c) for c in np.where((g[ref_row] != BG) & (g[ref_row] != sep_color))[0]]
    src_cols = [int(c) for c in np.where((g[src_row] != BG) & (g[src_row] != WALL) & (g[src_row] != sep_color))[0]]
    assert ref_cols, "expected reference pixels"
    assert src_cols, "expected source pixels"
    assert set(src_cols).issubset(set(ref_cols)), "every source pixel must align with a reference pixel"

    out = g.copy()
    for c in src_cols:
        source_color = int(g[src_row, c])
        ref_color = int(g[ref_row, c])
        assert ref_color not in (BG, sep_color), "source columns must have aligned reference pixels"
        out[src_row, c] = BG
        dest_r = sep_row + 1 if source_color == ref_color else H - 1
        assert int(out[dest_r, c]) == BG, "destination cells must be empty before sliding"
        out[dest_r, c] = source_color

    return _fill_closed_regions(out, sep_row)


def _token_groups(n_tokens: int) -> tuple[list[dict], list[int]]:
    max_pairs = min(2, (n_tokens - 1) // 2)
    pair_count = 1 if max_pairs == 1 else random.choices([1, 2], weights=[3, 2])[0]

    groups = [("pair", idx) for idx in range(pair_count)] + [("single", idx) for idx in range(n_tokens - 2 * pair_count)]
    random.shuffle(groups)

    tokens: list[dict] = []
    chamber_ids: list[int] = []
    for kind, idx in groups:
        if kind == "pair":
            chamber_ids.append(idx)
            tokens.append({"matched": True, "chamber": idx, "side": "L"})
            tokens.append({"matched": True, "chamber": idx, "side": "R"})
        else:
            tokens.append({"matched": False, "chamber": None, "side": None})
    return tokens, chamber_ids


def _build_columns(tokens: list[dict]) -> tuple[list[int], list[tuple[int, int]]]:
    cols = [random.randint(1, 3)]
    chambers: dict[int, tuple[int, int]] = {}

    for i in range(1, len(tokens)):
        prev = tokens[i - 1]
        cur = tokens[i]
        if prev["chamber"] is not None and prev["chamber"] == cur["chamber"]:
            gap = random.randint(3, 5)
        else:
            gap = random.randint(2, 4)
        cols.append(cols[-1] + gap)

    for i, token in enumerate(tokens):
        chamber = token["chamber"]
        if chamber is None:
            continue
        if token["side"] == "L":
            chambers[chamber] = (cols[i], -1)
        else:
            left, _ = chambers[chamber]
            chambers[chamber] = (left, cols[i])

    return cols, [chambers[idx] for idx in sorted(chambers)]


def _place_extra_walls(grid: np.ndarray, sep_row: int, token_cols: set[int], reserved: set[tuple[int, int]]) -> None:
    H, W = grid.shape
    extra_count = random.randint(0, 2)
    placed = 0

    for _ in range(80):
        if placed >= extra_count:
            return

        orient = random.choice(("h", "v"))
        length = random.randint(2, 4)

        if orient == "h":
            if W - length - 1 <= 1 or H - 2 <= sep_row + 4:
                continue
            r = random.randint(sep_row + 4, H - 2)
            c0 = random.randint(1, W - length - 1)
            cells = [(r, c) for c in range(c0, c0 + length)]
            if any(c in token_cols for _, c in cells):
                continue
        else:
            if H - length - 2 <= sep_row + 3:
                continue
            c = random.randint(1, W - 2)
            if c in token_cols:
                continue
            r0 = random.randint(sep_row + 3, H - length - 1)
            cells = [(r, c) for r in range(r0, r0 + length)]

        if any((r, c) in reserved or int(grid[r, c]) != BG for r, c in cells):
            continue
        if any(any((nr, nc) in reserved for nr, nc in _neighbors4(r, c)) for r, c in cells):
            continue

        for r, c in cells:
            grid[r, c] = WALL
            reserved.add((r, c))
        placed += 1


def _expected_output(
    inp: np.ndarray,
    sep_row: int,
    src_row: int,
    cols: list[int],
    ref_colors: list[int],
    src_colors: list[int],
    chambers: list[tuple[int, int]],
) -> np.ndarray:
    out = inp.copy()
    H = out.shape[0]

    for c, ref_color, src_color in zip(cols, ref_colors, src_colors):
        out[src_row, c] = BG
        out[sep_row + 1 if src_color == ref_color else H - 1, c] = src_color

    for left, right in chambers:
        out[sep_row + 1, left + 1 : right] = FILL

    return out


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    sep_row = random.randint(3, max(3, H // 3 + 1))
    src_row = random.choice([H - 3, H - 2])
    if src_row <= sep_row + 2:
        return None

    n_tokens = random.randint(*TOKEN_COUNT_RANGE)
    tokens, _ = _token_groups(n_tokens)
    cols, chambers = _build_columns(tokens)

    W = max(13, cols[-1] + random.randint(3, 5))
    if W > 30:
        return None

    grid = np.full((H, W), BG, dtype=int)
    grid[sep_row, :] = SEP

    ref_colors = []
    src_colors = []
    for token in tokens:
        if token["matched"]:
            ref_color = random.choice((1, 3, 4))
            src_color = ref_color
        else:
            ref_color = random.choice(TOKEN_COLORS)
            src_color = random.choice([c for c in TOKEN_COLORS if c != ref_color])
        ref_colors.append(ref_color)
        src_colors.append(src_color)

    for c, ref_color, src_color in zip(cols, ref_colors, src_colors):
        grid[sep_row - 1, c] = ref_color
        grid[src_row, c] = src_color

    reserved = {(sep_row - 1, c) for c in cols} | {(src_row, c) for c in cols}
    reserved |= {(sep_row, c) for c in range(W)}

    for left, right in chambers:
        wall_row = sep_row + 2
        if wall_row >= src_row:
            return None
        for c in range(left + 1, right):
            grid[wall_row, c] = WALL
            reserved.add((wall_row, c))

    _place_extra_walls(grid, sep_row, set(cols), reserved)

    expected = _expected_output(grid, sep_row, src_row, cols, ref_colors, src_colors, chambers)
    actual = main(grid)

    if not np.array_equal(actual, expected):
        return None

    if np.count_nonzero(actual == FILL) < sum(right - left - 1 for left, right in chambers):
        return None

    if not any(src == ref for src, ref in zip(src_colors, ref_colors)):
        return None
    if not any(src != ref for src, ref in zip(src_colors, ref_colors)):
        return None

    token_cols = set(cols)
    wall_cols = {int(c) for _, c in np.argwhere(grid == WALL)}
    if token_cols & wall_cols:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
