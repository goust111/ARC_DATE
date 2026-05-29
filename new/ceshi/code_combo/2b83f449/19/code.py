# Combo variant of 2b83f449 (index 19)
# combo_spec: D12/2+D13/2
# Name: 全行令牌+路径交叉检测
#
# Design analysis for this combo variant
# -------------------------------------
# a) Concrete behavior trace:
# - Keep the original geometry shell: even rows are pipe rows, odd rows are
#   walls, and each 1x3 horizontal gate opens into a 3-cell vertical post. Pipe
#   rows may also contain 0-valued holes that split one long pipe into segments.
# - D13/2 changes the input only: every pipe row, including the bottommost one,
#   has a left token at col 0 and a right token at col W-1. The old task omitted
#   the bottom row; now those two bottom tokens visibly exist and never move
#   because they start in the last pipe row and no lower exit exists.
# - D12/2 changes the falling simulation. Example: suppose row 0 has one wide
#   segment with a single exit into row 2 segment [3,12]. The left token enters
#   row 2 first, so segment key (2,3,12) becomes "visited". The right token from
#   the same row then attempts to enter that same row 2 segment; instead of
#   following the first token, it is bounced back and stays in row 0. Packing then
#   happens on different rows, so the output differs clearly from the base task.
# - To make such crossings physically possible, the combo intentionally relaxes
#   one hidden base-task invariant: lower pipe segments may now have multiple
#   entries from above, as long as they still have at most one exit below. Those
#   shared lower segments are exactly where the collision rule becomes visible.
# - If a token reaches row 4 and the next segment below has already been used by
#   a token from a higher row, that later token stops in row 4 instead. This
#   means collisions are checked at every downward step, not only at the first.
#
# b) Complexity calibration:
# - This combo is easy to trivialize if most rows are already split by holes,
#   because then left/right tokens start in different segments and never compete.
#   The generator therefore requires at least two bounce events and at least one
#   token with drop depth >= 3, so the output exhibits both deeper routing and
#   collision-induced truncation.
# - A second degeneracy would be "combo output == original output". The generator
#   compares the collision-aware result against the base no-collision result and
#   rejects samples where they match, ensuring D12/2 is visually active.
# - D13/2 stays visually obvious because the bottom input row always contains two
#   extra tokens. Their unchanged positions in the output reinforce the rule
#   "tokens with no exit remain where they are".
#
# c) Implicit-assumption audit of the original solver:
# - Gate inference still relies on a unique horizontal 1x3 component color. The
#   generator keeps exactly that gate shape.
# - The original fall loop assumed token paths are independent. That assumption is
#   broken here, so the new solver adds a global `visited_segments` set keyed by
#   (row, seg_start, seg_end) and consults it before each descent.
# - The original code assumed the bottom row contains no tokens. That no longer
#   holds; the packing logic must tolerate source tokens on the last row. It does,
#   because "no exit" already means "stay in current segment", and with no entry
#   from above those tokens pack back to their own edge.
# - The original generator enforced both <=1 exit below and <=1 entry above for
#   every segment. That makes true crossings almost impossible with edge tokens,
#   so the combo keeps only the <=1-exit rule and allows controlled multi-entry
#   segments. The solver therefore cannot rely on a unique entry-from-above.
# - Because a final segment may now have more than one possible upper entry, the
#   old packing heuristic ("inspect the segment globally to find the one entry")
#   is broken. The combo solver tracks each token's actual last successful entry
#   column and packs away from that concrete entry instead.
# - Token priority is now a real assumption: "later" means later in deterministic
#   row-major source order (`np.argwhere` order, top-to-bottom then left-to-right).
#   The generator uses the same order when validating outputs.
#
# d) Placement legality:
# - Gate centers remain interior (`2..W-3`), same-row centers remain >= 5 apart,
#   adjacent-wall centers remain >= 2 apart, and hole repair still preserves <= 2
#   holes per pipe row and <= 4 total holes. The only relaxed legality rule is
#   that some segments may have multiple entries above; no overlap or out-of-
#   bounds placement is introduced by that change.
#
# e) Visual observability:
# - Across 2-4 samples a human can see three repeated cues:
#   1. bottom-row tokens exist in the input and also remain in the output;
#   2. some upper-row tokens stop earlier than others even though the geometry is
#      unchanged, because a previous token already used the lower segment;
#   3. compared with the base task, several tokens accumulate one row higher.
# - The generator rejects any sample where cue (2) would be absent.
#
# f) Visual cleanliness:
# - No new colors are introduced; the combo only changes where the existing green
#   tokens settle. This keeps the grids visually close to the original task and
#   avoids noisy overlays.
#
# g) New solver assumptions checked by the generator:
# - Every token cell lies on a bg pipe cell after removing tokens.
# - Collision checks only matter when the next row really contains a valid segment
#   at the chosen exit column.
# - Because packing is still end-based, destination segments must have enough room
#   for the number of tokens assigned there; the generator validates that by
#   calling `main(inp)` before accepting each sample.

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
ARC_UTILS_DIR = ROOT / "solvers"
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import label_mask, paste


WALL = 0
TOKEN = 3
OPEN = 6
GATE = 7
BG = 8


def _infer_colors(grid: np.ndarray) -> tuple[int, int, int, int]:
    colors = set(map(int, grid.flatten()))
    gate = None
    for col in sorted(colors):
        if col == WALL:
            continue
        lab, n = label_mask(grid == col, connectivity=4)
        found = False
        for k in range(1, int(n) + 1):
            pts = np.argwhere(lab == k)
            if int(pts.shape[0]) != 3:
                continue
            r0, c0 = pts.min(axis=0)
            r1, c1 = pts.max(axis=0)
            if int(r1 - r0 + 1) == 1 and int(c1 - c0 + 1) == 3:
                found = True
                break
        if found:
            gate = col
            break
    assert gate is not None

    cnt_all = Counter(map(int, grid.flatten()))
    token_candidates = [c for c in colors if c not in (WALL, gate)]
    assert token_candidates
    token = int(min(token_candidates, key=lambda c: (int(cnt_all[c]), int(c))))
    bg_candidates = [c for c in colors if c not in (WALL, gate, token)]
    assert bg_candidates
    bg = int(max(bg_candidates, key=lambda c: (int(cnt_all[c]), -int(c))))
    open_col = 6 if gate == 7 and 6 not in colors else int(gate - 1) if gate > 0 else 6
    return int(gate), int(token), int(bg), int(open_col)


def _open_input(grid: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]], int, int]:
    g = np.array(grid, dtype=int)
    h, _w = g.shape
    gate, token, bg, open_col = _infer_colors(g)

    out = g.copy()
    lab, n = label_mask(out == gate, connectivity=4)
    for k in range(1, int(n) + 1):
        pts = np.argwhere(lab == k)
        if int(pts.shape[0]) != 3:
            continue
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        if int(r1 - r0 + 1) != 1 or int(c1 - c0 + 1) != 3:
            continue
        r = int(r0)
        c = int(c0 + 1)
        out[r, c - 1 : c + 2] = bg
        for rr in (r - 1, r, r + 1):
            if 0 <= rr < h:
                out[rr, c] = open_col

    token_pts = [(int(r), int(c)) for r, c in np.argwhere(out == token)]
    out[out == token] = bg
    return out, token_pts, token, bg


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


def _find_seg(segs: list[tuple[int, int]], c: int) -> tuple[int, int] | None:
    for a, b in segs:
        if a <= c <= b:
            return (a, b)
    return None


def _seg_exit(grid: np.ndarray, r: int, seg: tuple[int, int], bg: int) -> int | None:
    if r + 2 >= grid.shape[0]:
        return None
    a, b = seg
    exits = [c for c in range(a, b + 1) if int(grid[r + 1, c]) == bg and int(grid[r + 2, c]) == bg]
    assert len(exits) <= 1, f"segment ({a},{b}) at row {r} has {len(exits)} exits: {exits}"
    return exits[0] if exits else None


def _entries_from_above(grid: np.ndarray, r: int, seg: tuple[int, int], bg: int) -> list[int]:
    if r < 2:
        return []
    a, b = seg
    return [c for c in range(a, b + 1) if int(grid[r - 1, c]) == bg and int(grid[r - 2, c]) == bg]


def _settle_tokens(
    opened: np.ndarray,
    token_pts: list[tuple[int, int]],
    token: int,
    bg: int,
    *,
    detect_collision: bool,
) -> tuple[np.ndarray, dict[str, int]]:
    out = np.array(opened, dtype=int)
    h, _w = out.shape
    visited_segments: set[tuple[int, int, int]] = set()
    records: list[tuple[int, tuple[int, int], int]] = []
    drop_counts: list[int] = []
    collision_count = 0

    for start_r, start_c in token_pts:
        cur_r, cur_c = int(start_r), int(start_c)
        final_entry: int | None = None
        drops = 0
        if int(out[cur_r, cur_c]) != bg:
            seg = _find_seg(_segments_in_row(out, cur_r, bg), cur_c)
            if seg is not None:
                a, b = seg
                pack_end = a if cur_c <= a else b
                records.append((cur_r, seg, pack_end))
            drop_counts.append(0)
            continue

        while True:
            seg = _find_seg(_segments_in_row(out, cur_r, bg), cur_c)
            if seg is None:
                break
            exit_col = _seg_exit(out, cur_r, seg, bg)
            if exit_col is None:
                break
            next_r = cur_r + 2
            if next_r >= h:
                break
            next_seg = _find_seg(_segments_in_row(out, next_r, bg), exit_col)
            if next_seg is None:
                break
            if detect_collision:
                key = (next_r, next_seg[0], next_seg[1])
                if key in visited_segments:
                    collision_count += 1
                    break
                visited_segments.add(key)
            cur_r, cur_c = next_r, exit_col
            final_entry = exit_col
            drops += 1

        seg = _find_seg(_segments_in_row(out, cur_r, bg), cur_c)
        if seg is not None:
            a, b = seg
            if final_entry is not None:
                pack_end = b if (final_entry - a) <= (b - final_entry) else a
            else:
                pack_end = a if cur_c <= a else b
            records.append((cur_r, seg, pack_end))
        drop_counts.append(drops)

    buckets: dict[tuple[int, tuple[int, int], int], int] = defaultdict(int)
    for r, seg, pack_end in records:
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

    stats = {
        "collisions": int(collision_count),
        "max_drop": int(max(drop_counts, default=0)),
        "token_count": int(len(token_pts)),
    }
    return out, stats


def main(grid: np.ndarray) -> np.ndarray:
    opened, token_pts, token, bg = _open_input(np.array(grid, dtype=int))
    out, _stats = _settle_tokens(opened, token_pts, token, bg, detect_collision=True)
    return out


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

    for pipe_idx in range(n_walls + 1):
        r = 2 * pipe_idx
        grid[r, 0] = TOKEN
        grid[r, width - 1] = TOKEN

    return grid


def _exits_below(grid: np.ndarray, r: int, seg: tuple[int, int]) -> list[int]:
    if r + 2 >= grid.shape[0]:
        return []
    a, b = seg
    return [c for c in range(a, b + 1) if int(grid[r + 1, c]) == BG and int(grid[r + 2, c]) == BG]


def _entries_above(grid: np.ndarray, r: int, seg: tuple[int, int]) -> list[int]:
    if r < 2:
        return []
    a, b = seg
    return [c for c in range(a, b + 1) if int(grid[r - 1, c]) == BG and int(grid[r - 2, c]) == BG]


def _find_exit_issues(grid: np.ndarray) -> list[tuple[int, tuple[int, int], list[int]]]:
    issues: list[tuple[int, tuple[int, int], list[int]]] = []
    for r in range(0, grid.shape[0], 2):
        for seg in _segments_in_row(grid, r, BG):
            below = _exits_below(grid, r, seg)
            if len(below) > 1:
                issues.append((r, seg, below))
    return issues


def _count_multi_entry_segments(grid: np.ndarray) -> int:
    count = 0
    for r in range(0, grid.shape[0], 2):
        for seg in _segments_in_row(grid, r, BG):
            if len(_entries_above(grid, r, seg)) > 1:
                count += 1
    return count


def _separator_column(grid: np.ndarray, r: int, left: int, right: int) -> int | None:
    choices = [c for c in range(left + 1, right) if int(grid[r, c]) == BG]
    if not choices:
        return None
    return int(choices[len(choices) // 2])


def _repair_holes(gates: list[list[int]], holes: list[list[int]], width: int) -> list[list[int]] | None:
    hole_sets = [set(row) for row in holes]

    while True:
        grid = _build_opened_grid(gates, [sorted(row) for row in hole_sets], width)
        issues = _find_exit_issues(grid)
        if not issues:
            return [sorted(row) for row in hole_sets]

        r, _seg, below = issues[0]
        pipe_idx = r // 2
        if len(hole_sets[pipe_idx]) >= 2:
            return None

        new_cols: list[int] = []
        for left, right in zip(below, below[1:]):
            sep = _separator_column(grid, r, left, right)
            if sep is not None:
                new_cols.append(sep)
                break

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


def _simulate_fixed(opened: np.ndarray, *, detect_collision: bool) -> tuple[np.ndarray, dict[str, int]]:
    n_walls = (opened.shape[0] - 1) // 2
    token_pts = [(2 * pipe_idx, c) for pipe_idx in range(n_walls + 1) for c in (0, opened.shape[1] - 1)]
    return _settle_tokens(opened, token_pts, TOKEN, BG, detect_collision=detect_collision)


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    n_walls = int(rng.integers(5, 10))
    width = int(rng.integers(16, 20))

    gates: list[list[int]] = [[] for _ in range(n_walls)]
    holes: list[list[int]] = [[] for _ in range(n_walls + 1)]

    spine_len = int(rng.integers(3, min(6, n_walls) + 1))
    spine_start = int(rng.integers(0, n_walls - spine_len + 1))

    prev_center: int | None = None
    go_left = bool(rng.integers(2))
    for row in range(spine_start, spine_start + spine_len):
        if prev_center is None:
            options = list(range(3, width - 3))
        elif go_left:
            options = [c for c in range(3, prev_center - 1)]
        else:
            options = [c for c in range(prev_center + 2, width - 2)]
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
    if _find_exit_issues(opened):
        return None

    combo_out, combo_stats = _simulate_fixed(opened, detect_collision=True)
    base_out, _base_stats = _simulate_fixed(opened, detect_collision=False)
    if combo_stats["max_drop"] < 3:
        return None
    if combo_stats["collisions"] < 2:
        return None
    if _count_multi_entry_segments(opened) < 1:
        return None
    if np.array_equal(combo_out, base_out):
        return None

    inp = _build_input(gates, holes, width)
    if not np.array_equal(main(inp), combo_out):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(1000):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate input for combo variant 2b83f449/19")
