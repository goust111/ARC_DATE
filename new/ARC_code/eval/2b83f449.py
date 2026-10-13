from __future__ import annotations
from collections import Counter, defaultdict
import numpy as np
from arc_utils import get_bg, label_mask
import itertools
from arc_utils import paste
from collections import defaultdict

def main(grid: np.ndarray) -> np.ndarray:
    """
    Simulate tokens falling through a pipe network after gates rotate open.

    Input structure:
    - Wall (0), bg (pipe fill), gate (horizontal 1×3 bars), token (small edge pixels).

    Rule:
    - Rotate each gate into a vertical blocker; gate cells become bg.
    - Tokens drop downward through bg, choosing the extreme column for their side (left→min, right→max).
    - Tokens settle at true pipe ends (adjacent to wall/boundary), packing inward.

    Key insight: Tokens select drop columns by entry side and land at wall-bounded endpoints, ignoring blocker-bounded ones.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    colors = set(map(int, g.flatten()))

    # --- infer gate color: any color forming 4-ccs of size 3 with bbox 1x3 (horizontal) ---
    gate = None
    for col in sorted(colors):
        if int(col) == 0:
            continue
        lab, n = label_mask(g == int(col), connectivity=4)
        found = False
        for k in range(1, int(n) + 1):
            pts = np.argwhere(lab == int(k))
            if int(pts.shape[0]) != 3:
                continue
            r0, c0 = pts.min(axis=0)
            r1, c1 = pts.max(axis=0)
            if int(r1 - r0 + 1) == 1 and int(c1 - c0 + 1) == 3:
                found = True
                break
        if found:
            gate = int(col)
            break
    assert not (gate is None)

    # --- token/bg inference ---
    cnt_all = Counter(map(int, g.flatten()))
    cand_token = [c for c in colors if int(c) not in (0, int(gate))]
    assert not (not cand_token)
    token = int(min(cand_token, key=lambda c: (int(cnt_all[int(c)]), int(c))))
    cand_bg = [c for c in colors if int(c) not in (0, int(gate), int(token))]
    assert cand_bg
    bg = int(max(cand_bg, key=lambda c: (int(cnt_all[int(c)]), -int(c))))

    # --- opened gate color (vertical post) ---
    open_col = 6 if int(gate) == 7 and 6 not in colors else int(gate - 1) if int(gate) > 0 else 6

    # --- rotate gates (gates open into bg, center becomes vertical open_col) ---
    out = g.copy()
    lab, n = label_mask(out == int(gate), connectivity=4)
    for k in range(1, int(n) + 1):
        pts = np.argwhere(lab == int(k))
        if int(pts.shape[0]) != 3:
            continue
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        if int(r1 - r0 + 1) != 1 or int(c1 - c0 + 1) != 3:
            continue
        r = int(r0)
        c0 = int(c0)
        center_c = int(c0 + 1)
        out[int(r), int(c0) : int(c0) + 3] = int(bg)
        for rr in (int(r - 1), int(r), int(r + 1)):
            if 0 <= int(rr) < H:
                out[int(rr), int(center_c)] = int(open_col)

    # helpers
    def segments_in_row(r: int) -> list[tuple[int, int]]:
        segs: list[tuple[int, int]] = []
        c = 0
        while c < W:
            if int(out[r, c]) != int(bg):
                c += 1; continue
            c0 = c
            while c + 1 < W and int(out[r, c + 1]) == int(bg):
                c += 1
            segs.append((c0, c)); c += 1
        return segs

    def find_seg(segs: list[tuple[int, int]], c: int) -> tuple[int, int] | None:
        for a, b in segs:
            if a <= c <= b: return (a, b)
        return None

    def seg_exit(r: int, seg: tuple[int, int]) -> int | None:
        """Return the single exit column, or None. Assert if >1 exit."""
        a, b = seg
        if r + 2 >= H: return None
        drops = [c for c in range(a, b + 1)
                 if int(out[r + 1, c]) == int(bg) and int(out[r + 2, c]) == int(bg)]
        assert len(drops) <= 1, f"segment ({a},{b}) at row {r} has {len(drops)} exits: {drops}"
        return drops[0] if drops else None

    def entry_from_above(r: int, seg: tuple[int, int]) -> int | None:
        """Return the single entry column from the wall above, or None. Assert if >1."""
        a, b = seg
        if r < 2: return None
        entries = [c for c in range(a, b + 1)
                   if int(out[r - 1, c]) == int(bg) and int(out[r - 2, c]) == int(bg)]
        assert len(entries) <= 1, f"segment ({a},{b}) at row {r} has {len(entries)} entries from above: {entries}"
        return entries[0] if entries else None

    # Remove tokens; simulate; re-place.
    token_pts = [(int(r), int(c)) for r, c in np.argwhere(out == int(token))]
    out[out == int(token)] = int(bg)

    # Phase 1: each token drops through single exits until stuck.
    finals: list[tuple[int, int]] = []  # (row, col) of final position
    for r, c in token_pts:
        cur_r, cur_c = r, c
        if int(out[cur_r, cur_c]) != int(bg):
            finals.append((cur_r, cur_c)); continue
        while True:
            seg = find_seg(segments_in_row(cur_r), cur_c)
            if seg is None: break
            ex = seg_exit(cur_r, seg)
            if ex is None: break
            cur_r += 2; cur_c = ex
            if cur_r >= H: break
        finals.append((cur_r, cur_c))

    # Phase 2: pack tokens at opposite end from entry.
    buckets: dict[tuple[int, tuple[int, int], int], int] = defaultdict(int)
    for r, c in finals:
        segs = segments_in_row(r)
        seg = find_seg(segs, c)
        if seg is None: continue
        a, b = seg
        entry = entry_from_above(r, seg)
        if entry is not None:
            # Pack to opposite end from entry
            pack_end = b if (entry - a) <= (b - entry) else a
        else:
            # No entry from above: token at starting position (grid edge), stay put
            pack_end = a if c <= a else b
        buckets[(r, seg, pack_end)] += 1

    for (r, (a, b), pack_end), k in buckets.items():
        if pack_end == a:
            for i in range(k):
                if a + i <= b: out[r, a + i] = int(token)
        else:
            for i in range(k):
                if b - i >= a: out[r, b - i] = int(token)

    return out

# --------------- generator (Codex, output-first with spine) ---------------

WALL = 0
TOKEN = 3
OPEN = 6
GATE = 7
BG = 8


def _build_opened_grid(gates: list[list[int]], holes: list[list[int]], width: int) -> np.ndarray:
    n_walls = len(gates)
    height = 2 * n_walls + 1
    grid = np.full((height, width), WALL, dtype=int)

    for pipe_idx in range(n_walls + 1):
        r = 2 * pipe_idx
        grid[r, :] = BG
        for c in holes[pipe_idx]:
            grid[r, c] = WALL

    for wall_idx, centers in enumerate(gates):
        r = 2 * wall_idx + 1
        for c in centers:
            grid[r, c - 1] = BG
            grid[r, c] = OPEN
            grid[r, c + 1] = BG
            grid[r - 1, c] = OPEN
            grid[r + 1, c] = OPEN

    return grid


def _build_input(gates: list[list[int]], holes: list[list[int]], width: int) -> np.ndarray:
    n_walls = len(gates)
    height = 2 * n_walls + 1
    grid = np.full((height, width), WALL, dtype=int)
    gate_patch = np.full((1, 3), GATE, dtype=int)

    for pipe_idx in range(n_walls + 1):
        r = 2 * pipe_idx
        grid[r, :] = BG
        for c in holes[pipe_idx]:
            grid[r, c] = WALL

    for wall_idx, centers in enumerate(gates):
        r = 2 * wall_idx + 1
        for c in centers:
            paste(grid, gate_patch, r, c - 1)

    for pipe_idx in range(n_walls):
        r = 2 * pipe_idx
        grid[r, 0] = TOKEN
        grid[r, width - 1] = TOKEN

    return grid


def _segments_in_row(grid: np.ndarray, r: int) -> list[tuple[int, int]]:
    width = grid.shape[1]
    segments: list[tuple[int, int]] = []
    c = 0
    while c < width:
        if grid[r, c] != BG:
            c += 1
            continue
        start = c
        while c + 1 < width and grid[r, c + 1] == BG:
            c += 1
        segments.append((start, c))
        c += 1
    return segments


def _find_segment(grid: np.ndarray, r: int, c: int) -> tuple[int, int] | None:
    for seg in _segments_in_row(grid, r):
        if seg[0] <= c <= seg[1]:
            return seg
    return None


def _exits_below(grid: np.ndarray, r: int, seg: tuple[int, int]) -> list[int]:
    height = grid.shape[0]
    if r + 2 >= height:
        return []
    a, b = seg
    return [c for c in range(a, b + 1) if grid[r + 1, c] == BG and grid[r + 2, c] == BG]


def _entries_above(grid: np.ndarray, r: int, seg: tuple[int, int]) -> list[int]:
    if r < 2:
        return []
    a, b = seg
    return [c for c in range(a, b + 1) if grid[r - 1, c] == BG and grid[r - 2, c] == BG]


def _find_issues(grid: np.ndarray) -> list[tuple[int, tuple[int, int], list[int], list[int]]]:
    height = grid.shape[0]
    issues: list[tuple[int, tuple[int, int], list[int], list[int]]] = []
    for r in range(0, height, 2):
        for seg in _segments_in_row(grid, r):
            above = _entries_above(grid, r, seg)
            below = _exits_below(grid, r, seg)
            if len(above) > 1 or len(below) > 1:
                issues.append((r, seg, above, below))
    return issues


def _separator_column(grid: np.ndarray, r: int, left: int, right: int) -> int | None:
    choices = [c for c in range(left + 1, right) if grid[r, c] == BG]
    if not choices:
        return None
    return int(choices[len(choices) // 2])


def _repair_holes(
    gates: list[list[int]], holes: list[list[int]], width: int
) -> list[list[int]] | None:
    hole_sets = [set(row) for row in holes]

    while True:
        grid = _build_opened_grid(gates, [sorted(row) for row in hole_sets], width)
        issues = _find_issues(grid)
        if not issues:
            return [sorted(row) for row in hole_sets]

        r, _seg, above, below = issues[0]
        pipe_idx = r // 2
        if len(hole_sets[pipe_idx]) >= 2:
            return None

        events = sorted([(c, "up") for c in above] + [(c, "down") for c in below])
        seen_up = False
        seen_down = False
        prev_col: int | None = None
        new_cols: list[int] = []

        for col, kind in events:
            duplicate = (kind == "up" and seen_up) or (kind == "down" and seen_down)
            if duplicate:
                assert prev_col is not None
                sep = _separator_column(grid, r, prev_col, col)
                if sep is None:
                    return None
                new_cols.append(sep)
                seen_up = False
                seen_down = False

            if kind == "up":
                seen_up = True
            else:
                seen_down = True
            prev_col = col

        if not new_cols:
            return None

        for c in new_cols:
            hole_sets[pipe_idx].add(c)
            if len(hole_sets[pipe_idx]) > 2:
                return None

        if sum(len(row) for row in hole_sets) > 4:
            return None


def _candidate_centers(gates: list[list[int]], row: int, width: int) -> list[int]:
    candidates: list[int] = []
    for c in range(2, width - 2):
        if c in gates[row]:
            continue
        if any(abs(c - other) < 5 for other in gates[row]):
            continue

        ok = True
        for adj in (row - 1, row + 1):
            if 0 <= adj < len(gates) and any(abs(c - other) < 2 for other in gates[adj]):
                ok = False
                break
        if ok:
            candidates.append(c)
    return candidates


def _simulate_output(opened: np.ndarray) -> tuple[np.ndarray, list[int]]:
    height, width = opened.shape
    n_walls = (height - 1) // 2

    finals: list[tuple[int, int]] = []
    drop_counts: list[int] = []

    for pipe_idx in range(n_walls):
        for start_c in (0, width - 1):
            r = 2 * pipe_idx
            c = start_c
            drops = 0
            while True:
                seg = _find_segment(opened, r, c)
                if seg is None:
                    break
                exits = _exits_below(opened, r, seg)
                if len(exits) != 1:
                    break
                r += 2
                c = exits[0]
                drops += 1
            finals.append((r, c))
            drop_counts.append(drops)

    out = opened.copy()
    buckets: dict[tuple[int, tuple[int, int], int], int] = defaultdict(int)

    for r, c in finals:
        seg = _find_segment(opened, r, c)
        if seg is None:
            continue
        a, b = seg
        entries = _entries_above(opened, r, seg)
        if len(entries) == 1:
            entry = entries[0]
            pack_end = b if (entry - a) <= (b - entry) else a
        else:
            pack_end = a if c <= a else b
        buckets[(r, seg, pack_end)] += 1

    for (r, (a, b), pack_end), count in buckets.items():
        if pack_end == a:
            for i in range(count):
                if a + i <= b:
                    out[r, a + i] = TOKEN
        else:
            for i in range(count):
                if b - i >= a:
                    out[r, b - i] = TOKEN

    return out, drop_counts


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    n_walls = int(rng.integers(5, 10))
    width = int(rng.integers(16, 20))

    gates: list[list[int]] = [[] for _ in range(n_walls)]
    holes: list[list[int]] = [[] for _ in range(n_walls + 1)]

    spine_len = int(rng.integers(3, min(6, n_walls) + 1))
    spine_start = int(rng.integers(0, n_walls - spine_len + 1))

    prev_center: int | None = None
    go_left = bool(rng.integers(2))  # alternate direction each step
    for row in range(spine_start, spine_start + spine_len):
        if prev_center is None:
            options = list(range(3, width - 3))
        elif go_left:
            options = [c for c in range(3, prev_center - 1)]  # c <= prev - 2
        else:
            options = [c for c in range(prev_center + 2, width - 2)]  # c >= prev + 2
        if not options:
            return None
        center = int(rng.choice(options))
        gates[row] = [center]
        prev_center = center
        go_left = not go_left

    holes = _repair_holes(gates, holes, width)
    if holes is None:
        return None

    total_gates = sum(len(row) for row in gates)
    target_total = int(rng.integers(max(6, total_gates), 13))

    while total_gates < target_total:
        candidates: list[tuple[int, int]] = []
        for row in range(n_walls):
            if len(gates[row]) >= 3:
                continue
            for center in _candidate_centers(gates, row, width):
                candidates.append((row, center))

        if not candidates:
            break

        order = rng.permutation(len(candidates))
        added = False
        for idx in order:
            row, center = candidates[int(idx)]
            new_gates = [list(cols) for cols in gates]
            new_gates[row] = sorted(new_gates[row] + [center])
            new_holes = _repair_holes(new_gates, holes, width)
            if new_holes is None:
                continue
            gates = new_gates
            holes = new_holes
            total_gates += 1
            added = True
            break

        if not added:
            break

    if not (6 <= total_gates <= 12):
        return None

    opened = _build_opened_grid(gates, holes, width)
    if _find_issues(opened):
        return None

    output, drop_counts = _simulate_output(opened)
    if max(drop_counts, default=0) < 3:
        return None

    inp = _build_input(gates, holes, width)
    if not np.array_equal(main(inp), output):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate input for task 2b83f449")
