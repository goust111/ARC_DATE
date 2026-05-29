# Combo variant of 2b83f449 (index 23)
# combo_spec: D5/0+D11/2
# Name: 门对称分布+单侧令牌
#
# Design analysis
# a) Concrete behavior:
#    Use a 19-column grid so there is a visible center column (col 9). Gate bars
#    always appear in mirrored pairs, e.g. centers [6,12], [2,16], [6,12]. Pipe
#    rows adjacent to those paired gates get a central hole at col 9, which splits
#    the symmetric structure into two mirrored single-exit halves. Tokens appear
#    only on the left edge (col 0) of each non-last pipe row. In the original task
#    both edges were populated; here the geometry stays mirror-symmetric but only
#    the left half is functionally activated. A representative left token can fall
#    through a 3-wall zigzag path 0 -> 5 -> 3 -> 5, while the right half remains
#    empty because there are no right-edge tokens.
# b) Complexity calibration:
#    The variant changes two things that are both easy to see: every active wall
#    row has a left/right mirror gate pair, and every source row has exactly one
#    token on the same side. The generator keeps only a small family of 3-row
#    mirrored spines so the pattern repeats across examples instead of exploding
#    into too many cases.
# c) Implicit assumptions in main():
#    - Gate inference still assumes a 1x3 horizontal bar color exists; generator
#      keeps color 7 bars unchanged.
#    - Token inference still assumes tokens are the rare non-wall, non-gate color;
#      generator keeps only color 3 as tokens.
#    - Falling code asserts <=1 exit and <=1 entry per segment; mirrored gate rows
#      would violate this without a central separator, so generator inserts the
#      center hole on every pipe row touching the mirrored spine.
#    - Packing code works for any token multiplicity; no branch assumes both sides
#      are populated, so single-sided tokens are safe.
# d) Placement legality:
#    Mirrored centers are always interior (2..16) and use pairs [left, 18-left].
#    Active rows are three consecutive wall rows; center holes are only placed on
#    the four adjacent pipe rows, never on the outer boundary columns.
# e) Visual observability:
#    Humans can see the bilateral gate symmetry immediately, and can also see that
#    inputs break that symmetry because tokens only exist on the left edge. Outputs
#    preserve the symmetric pipe skeleton but show asymmetric token settling.
# f) Cleanliness:
#    The central hole is the only extra separator, so the grid remains readable:
#    wall rows contain only mirrored bars, pipe rows contain bg plus at most one
#    centered hole, and tokens stay on edges / settled pipe ends.
# g) Non-degeneracy:
#    The generator later checks a counterfactual "add mirrored right-side tokens"
#    input and rejects samples whose output would stay unchanged. This ensures the
#    single-sided source choice actually affects the resulting token layout.

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from arc_utils import label_mask, paste

WALL = 0
TOKEN = 3
OPEN = 6
GATE = 7
BG = 8

WIDTH = 19
CENTER = WIDTH // 2


def _mirror_col(c: int, width: int = WIDTH) -> int:
    return width - 1 - c


def _find_gate_color(grid: np.ndarray) -> int:
    colors = sorted(set(map(int, grid.flatten())))
    for col in colors:
        if col == WALL:
            continue
        labels, n = label_mask(grid == col, connectivity=4)
        for k in range(1, int(n) + 1):
            pts = np.argwhere(labels == k)
            if int(pts.shape[0]) != 3:
                continue
            (r0, c0), (r1, c1) = pts.min(axis=0), pts.max(axis=0)
            if int(r1 - r0 + 1) == 1 and int(c1 - c0 + 1) == 3:
                return int(col)
    raise AssertionError("gate color not found")


def _segments_in_row(grid: np.ndarray, r: int, bg: int) -> list[tuple[int, int]]:
    width = grid.shape[1]
    segs: list[tuple[int, int]] = []
    c = 0
    while c < width:
        if int(grid[r, c]) != bg:
            c += 1
            continue
        start = c
        while c + 1 < width and int(grid[r, c + 1]) == bg:
            c += 1
        segs.append((start, c))
        c += 1
    return segs


def _find_segment(grid: np.ndarray, r: int, c: int, bg: int) -> tuple[int, int] | None:
    for seg in _segments_in_row(grid, r, bg):
        if seg[0] <= c <= seg[1]:
            return seg
    return None


def _exits_below(grid: np.ndarray, r: int, seg: tuple[int, int], bg: int) -> list[int]:
    if r + 2 >= grid.shape[0]:
        return []
    a, b = seg
    return [c for c in range(a, b + 1) if int(grid[r + 1, c]) == bg and int(grid[r + 2, c]) == bg]


def _entries_above(grid: np.ndarray, r: int, seg: tuple[int, int], bg: int) -> list[int]:
    if r < 2:
        return []
    a, b = seg
    return [c for c in range(a, b + 1) if int(grid[r - 1, c]) == bg and int(grid[r - 2, c]) == bg]


def _open_gates(grid: np.ndarray, gate: int, bg: int, open_col: int) -> np.ndarray:
    opened = np.array(grid, dtype=int)
    labels, n = label_mask(opened == gate, connectivity=4)
    height = opened.shape[0]
    for k in range(1, int(n) + 1):
        pts = np.argwhere(labels == k)
        if int(pts.shape[0]) != 3:
            continue
        (r0, c0), (r1, c1) = pts.min(axis=0), pts.max(axis=0)
        if int(r1 - r0 + 1) != 1 or int(c1 - c0 + 1) != 3:
            continue
        r = int(r0)
        c = int(c0 + 1)
        opened[r, c - 1 : c + 2] = bg
        for rr in (r - 1, r, r + 1):
            if 0 <= rr < height:
                opened[rr, c] = open_col
    return opened


def _simulate_tokens(
    opened_with_tokens: np.ndarray,
    token_pts: list[tuple[int, int]],
    token: int,
    bg: int,
) -> tuple[np.ndarray, list[int]]:
    opened = np.array(opened_with_tokens, dtype=int)
    opened[opened == token] = bg

    finals: list[tuple[int, int]] = []
    drop_counts: list[int] = []

    for start_r, start_c in token_pts:
        r, c = int(start_r), int(start_c)
        drops = 0
        if int(opened[r, c]) != bg:
            finals.append((r, c))
            drop_counts.append(drops)
            continue
        while True:
            seg = _find_segment(opened, r, c, bg)
            if seg is None:
                break
            exits = _exits_below(opened, r, seg, bg)
            assert len(exits) <= 1, f"segment {seg} at row {r} has exits {exits}"
            if not exits:
                break
            r += 2
            c = int(exits[0])
            drops += 1
            if r >= opened.shape[0]:
                break
        finals.append((r, c))
        drop_counts.append(drops)

    out = opened.copy()
    buckets: dict[tuple[int, tuple[int, int], int], int] = defaultdict(int)
    for r, c in finals:
        seg = _find_segment(opened, r, c, bg)
        if seg is None:
            continue
        a, b = seg
        entries = _entries_above(opened, r, seg, bg)
        assert len(entries) <= 1, f"segment {seg} at row {r} has entries {entries}"
        if entries:
            entry = int(entries[0])
            pack_end = b if (entry - a) <= (b - entry) else a
        else:
            pack_end = a if c <= a else b
        buckets[(r, seg, pack_end)] += 1

    for (r, (a, b), pack_end), count in buckets.items():
        if pack_end == a:
            for i in range(count):
                if a + i <= b:
                    out[r, a + i] = token
        else:
            for i in range(count):
                if b - i >= a:
                    out[r, b - i] = token

    return out, drop_counts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Open all horizontal gates into vertical posts, then let the actually present
    tokens fall through single-exit pipe segments and pack at segment ends.

    In this combo family the gate network is mirror-symmetric, but tokens may
    appear on only one side. The solver therefore must process the real token
    set rather than assuming both mirrored sides are populated.
    """
    g = np.array(grid, dtype=int)
    colors = set(map(int, g.flatten()))

    gate = _find_gate_color(g)
    counts = Counter(map(int, g.flatten()))
    token_candidates = [c for c in colors if int(c) not in (WALL, gate)]
    assert token_candidates
    token = int(min(token_candidates, key=lambda c: (int(counts[int(c)]), int(c))))
    bg_candidates = [c for c in colors if int(c) not in (WALL, gate, token)]
    assert bg_candidates
    bg = int(max(bg_candidates, key=lambda c: (int(counts[int(c)]), -int(c))))
    open_col = OPEN if gate == GATE and OPEN not in colors else (gate - 1 if gate > 0 else OPEN)

    opened = _open_gates(g, gate, bg, open_col)
    token_pts = [(int(r), int(c)) for r, c in np.argwhere(g == token)]
    out, _drop_counts = _simulate_tokens(opened, token_pts, token, bg)
    return out


def _build_opened_grid(gates: list[list[int]], hole_rows: set[int], width: int) -> np.ndarray:
    n_walls = len(gates)
    grid = np.full((2 * n_walls + 1, width), WALL, dtype=int)
    for pipe_idx in range(n_walls + 1):
        r = 2 * pipe_idx
        grid[r, :] = BG
        if pipe_idx in hole_rows:
            grid[r, CENTER] = WALL
    for wall_idx, centers in enumerate(gates):
        wr = 2 * wall_idx + 1
        for c in centers:
            grid[wr - 1, c] = OPEN
            grid[wr, c - 1] = BG
            grid[wr, c] = OPEN
            grid[wr, c + 1] = BG
            grid[wr + 1, c] = OPEN
    return grid


def _build_input(gates: list[list[int]], hole_rows: set[int], width: int) -> np.ndarray:
    n_walls = len(gates)
    grid = np.full((2 * n_walls + 1, width), WALL, dtype=int)
    gate_patch = np.full((1, 3), GATE, dtype=int)
    for pipe_idx in range(n_walls + 1):
        r = 2 * pipe_idx
        grid[r, :] = BG
        if pipe_idx in hole_rows:
            grid[r, CENTER] = WALL
    for wall_idx, centers in enumerate(gates):
        wr = 2 * wall_idx + 1
        for c in centers:
            paste(grid, gate_patch, wr, c - 1)
    for pipe_idx in range(n_walls):
        grid[2 * pipe_idx, 0] = TOKEN
    return grid


def _find_issues(opened: np.ndarray) -> list[tuple[int, tuple[int, int], list[int], list[int]]]:
    issues: list[tuple[int, tuple[int, int], list[int], list[int]]] = []
    for r in range(0, opened.shape[0], 2):
        for seg in _segments_in_row(opened, r, BG):
            above = _entries_above(opened, r, seg, BG)
            below = _exits_below(opened, r, seg, BG)
            if len(above) > 1 or len(below) > 1:
                issues.append((r, seg, above, below))
    return issues


def _simulate_output(opened: np.ndarray) -> tuple[np.ndarray, list[int]]:
    n_walls = (opened.shape[0] - 1) // 2
    token_pts = [(2 * pipe_idx, 0) for pipe_idx in range(n_walls)]
    seeded = opened.copy()
    for r, c in token_pts:
        seeded[r, c] = TOKEN
    return _simulate_tokens(seeded, token_pts, TOKEN, BG)


def _is_mirror_symmetric_structure(gates: list[list[int]], hole_rows: set[int], width: int) -> bool:
    if width != WIDTH:
        return False
    for row in gates:
        if sorted(_mirror_col(c, width) for c in row) != sorted(row):
            return False
    for pipe_idx in hole_rows:
        if CENTER != _mirror_col(CENTER, width):
            return False
        if pipe_idx < 0:
            return False
    return True


def _counterfactual_with_right_tokens(inp: np.ndarray) -> np.ndarray:
    alt = np.array(inp, dtype=int)
    n_walls = (alt.shape[0] - 1) // 2
    for pipe_idx in range(n_walls):
        alt[2 * pipe_idx, alt.shape[1] - 1] = TOKEN
    return alt


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    n_walls = int(rng.integers(5, 10))
    spine_start = int(rng.integers(0, n_walls - 2))
    outer = int(rng.integers(4, 7))
    inner = int(rng.integers(2, outer - 1))
    left_centers = [outer, inner, outer]

    gates: list[list[int]] = [[] for _ in range(n_walls)]
    for offset, left in enumerate(left_centers):
        row = spine_start + offset
        gates[row] = [left, _mirror_col(left)]

    hole_rows = set(range(spine_start, spine_start + 4))
    if not _is_mirror_symmetric_structure(gates, hole_rows, WIDTH):
        return None

    opened = _build_opened_grid(gates, hole_rows, WIDTH)
    if _find_issues(opened):
        return None

    output, drop_counts = _simulate_output(opened)
    if max(drop_counts, default=0) < 3:
        return None

    inp = _build_input(gates, hole_rows, WIDTH)
    solved = main(inp)
    if not np.array_equal(solved, output):
        return None

    counterfactual = _counterfactual_with_right_tokens(inp)
    if np.array_equal(main(counterfactual), output):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(200):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate combo input for 2b83f449/23")
