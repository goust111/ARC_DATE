# Combo variant of 7666fa5d
# Combo: D3/0 + D15/0
# Name: 虚线段+孤立噪声像素

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import get_bg as _bg_color


H = W = 16
BG = 8
FILL = 2
LINE_COLORS = [1, 3, 4, 5, 6, 7, 9]

K_LO, K_HI = 4, 26
GAP_VALUES = (4, 6, 8)
NUM_KS_CHOICES = [4, 5]

SEG_LEN_CHOICES = (5, 7, 9)  # odd lengths keep both dashed endpoints visible
OVERLAP_MIN_B = 4

NOISE_COUNT_CHOICES = (2, 3, 4)

SEG_RETRY = 50
GEN_RETRY = 400
MIN_FILL_CELLS = 6
MIN_DASH_POINTS = 3


def _b_bounds_for_k(k: int) -> tuple[int, int]:
    b_lo = max(-k, k - 2 * (W - 1))
    b_hi = min(2 * (H - 1) - k, k)
    return b_lo, b_hi


def _solid_cells(k: int, b_min: int, b_max: int) -> list[tuple[int, int]]:
    return [((k + b) // 2, (k - b) // 2) for b in range(b_min, b_max + 1, 2)]


def _dash_cells(k: int, b_min: int, b_max: int) -> list[tuple[int, int]]:
    return [((k + b) // 2, (k - b) // 2) for b in range(b_min, b_max + 1, 4)]


def _apply_fill(
    grid: np.ndarray,
    segs: list[tuple[int, int, int]],
    *,
    bg: int,
    fill: int,
) -> np.ndarray:
    out = grid.copy()
    for b in range(-(W - 1), H):
        active = sorted([s for s in segs if s[1] <= b <= s[2]], key=lambda t: t[0])
        if len(active) < 2:
            continue
        r_min = max(0, b)
        r_max = min(H - 1, (W - 1) + b)
        for left, right in zip(active, active[1:]):
            k_left = left[0]
            k_right = right[0]
            for r in range(r_min, r_max + 1):
                c = r - b
                if k_left < r + c < k_right and out[r, c] == bg:
                    out[r, c] = fill
    return out


def _step4_runs(values: list[int]) -> list[list[int]]:
    if not values:
        return []
    runs: list[list[int]] = []
    run = [values[0]]
    for v in values[1:]:
        if v - run[-1] == 4:
            run.append(v)
        else:
            runs.append(run)
            run = [v]
    runs.append(run)
    return runs


def _candidate_dashed_segments(points: list[tuple[int, int]]) -> list[tuple[int, int, int, int]]:
    by_k: dict[int, list[int]] = {}
    for r, c in points:
        k = int(r + c)
        b = int(r - c)
        by_k.setdefault(k, []).append(b)

    candidates: list[tuple[int, int, int, int]] = []
    for k, bs in by_k.items():
        runs = [run for run in _step4_runs(sorted(set(bs))) if len(run) >= MIN_DASH_POINTS]
        if not runs:
            continue
        best = max(runs, key=lambda run: (len(run), run[-1] - run[0]))
        candidates.append((int(k), int(best[0]), int(best[-1]), len(best)))
    candidates.sort(key=lambda t: t[0])
    return candidates


def _overlap_b(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    return min(a[2], b[2]) - max(a[1], b[1])


def _choose_segment_chain(cands: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int]]:
    if not cands:
        return []

    n = len(cands)
    dp = [(1, cands[i][3]) for i in range(n)]
    prev = [-1] * n

    for i in range(n):
        for j in range(i):
            if cands[i][0] % 2 != cands[j][0] % 2:
                continue
            if _overlap_b(cands[j], cands[i]) < OVERLAP_MIN_B:
                continue
            candidate = (dp[j][0] + 1, dp[j][1] + cands[i][3])
            if candidate > dp[i]:
                dp[i] = candidate
                prev[i] = j

    best_idx = max(range(n), key=lambda i: dp[i])
    chain: list[tuple[int, int, int]] = []
    cur = best_idx
    while cur != -1:
        k, b_min, b_max, _score = cands[cur]
        chain.append((k, b_min, b_max))
        cur = prev[cur]
    chain.reverse()
    return chain


def _extract_segments(grid: np.ndarray, line: int) -> list[tuple[int, int, int]]:
    rows, cols = np.where(grid == line)
    points = [(int(r), int(c)) for r, c in zip(rows, cols)]
    cands = _candidate_dashed_segments(points)
    chain = _choose_segment_chain(cands)
    if len(chain) >= 2:
        return chain
    return [(k, b_min, b_max) for k, b_min, b_max, _n in cands]


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = int(_bg_color(g))
    non_bg = [c for c in sorted(set(map(int, g.ravel()))) if c != bg]
    assert non_bg, "grid must contain a dashed line color"

    line = max(non_bg, key=lambda c: int(np.sum(g == c)))
    segs = _extract_segments(g, int(line))
    assert len(segs) >= 2, "failed to recover dashed segments"

    return _apply_fill(g, segs, bg=bg, fill=FILL)


def _choose_k_values(parity: int) -> list[int] | None:
    num_ks = random.choice(NUM_KS_CHOICES)
    for _ in range(80):
        gaps = [random.choice(GAP_VALUES) for _ in range(num_ks - 1)]
        span = sum(gaps)
        starts = [k for k in range(K_LO, K_HI - span + 1) if k % 2 == parity]
        if not starts:
            continue
        k0 = random.choice(starts)
        ks = [k0]
        for gap in gaps:
            ks.append(ks[-1] + gap)
        if ks[-1] <= K_HI:
            return ks
    return None


def _random_segment(
    k: int,
    prev_range: tuple[int, int] | None,
) -> tuple[int, int] | None:
    b_lo, b_hi = _b_bounds_for_k(k)
    if b_lo % 2 != k % 2:
        b_lo += 1
    if b_hi % 2 != k % 2:
        b_hi -= 1
    if b_lo > b_hi:
        return None

    for _ in range(SEG_RETRY):
        length = random.choice(SEG_LEN_CHOICES)
        step = 2 * (length - 1)
        if b_lo + step > b_hi:
            continue
        candidates = list(range(b_lo, b_hi - step + 1, 2))
        if prev_range is not None:
            prev_lo, prev_hi = prev_range
            candidates = [
                start
                for start in candidates
                if min(start + step, prev_hi) - max(start, prev_lo) >= OVERLAP_MIN_B
            ]
            if not candidates:
                continue
        b_min = random.choice(candidates)
        b_max = b_min + step
        return b_min, b_max
    return None


def _paint_dashed_segments(
    grid: np.ndarray,
    segs: list[tuple[int, int, int]],
    line: int,
) -> set[tuple[int, int]]:
    colored: set[tuple[int, int]] = set()
    for k, b_min, b_max in segs:
        for cell in _dash_cells(k, b_min, b_max):
            grid[cell] = line
            colored.add(cell)
    return colored


def _is_isolated(cell: tuple[int, int], colored: set[tuple[int, int]]) -> bool:
    r0, c0 = cell
    for r, c in colored:
        if max(abs(r - r0), abs(c - c0)) <= 1:
            return False
    return True


def _noise_candidates(
    grid: np.ndarray,
    forbidden_fill: set[tuple[int, int]],
    used_ks: set[int],
    colored: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    for r in range(H):
        for c in range(W):
            if grid[r, c] != BG:
                continue
            if (r, c) in forbidden_fill:
                continue
            if (r + c) in used_ks:
                continue
            if not _is_isolated((r, c), colored):
                continue
            candidates.append((r, c))
    random.shuffle(candidates)
    return candidates


def _pick_isolated_noise(
    grid: np.ndarray,
    forbidden_fill: set[tuple[int, int]],
    used_ks: set[int],
    colored: set[tuple[int, int]],
    noise_count: int,
) -> list[tuple[int, int]] | None:
    choices = _noise_candidates(grid, forbidden_fill, used_ks, colored)
    picked: list[tuple[int, int]] = []
    occupied = set(colored)
    used_noise_ks: set[int] = set()
    for cell in choices:
        k = int(cell[0] + cell[1])
        if k in used_noise_ks:
            continue
        if not _is_isolated(cell, occupied):
            continue
        picked.append(cell)
        occupied.add(cell)
        used_noise_ks.add(k)
        if len(picked) == noise_count:
            return picked
    return None


def _generate_case() -> tuple[np.ndarray, np.ndarray]:
    line = random.choice(LINE_COLORS)
    parity = random.randint(0, 1)
    segs: list[tuple[int, int, int]] | None = None

    for _ in range(40):
        ks = _choose_k_values(parity)
        if ks is None:
            continue

        built: list[tuple[int, int, int]] = []
        first = _random_segment(ks[0], None)
        if first is None:
            continue
        built.append((ks[0], first[0], first[1]))

        ok = True
        for k in ks[1:]:
            nxt = _random_segment(k, (built[-1][1], built[-1][2]))
            if nxt is None:
                ok = False
                break
            built.append((k, nxt[0], nxt[1]))

        if ok:
            segs = built
            break

    if segs is None:
        raise RuntimeError("failed to place dashed segment chain")

    clean = np.full((H, W), BG, dtype=int)
    colored = _paint_dashed_segments(clean, segs, line)
    output = _apply_fill(clean, segs, bg=BG, fill=FILL)

    if int(np.sum(output == FILL)) < MIN_FILL_CELLS:
        raise RuntimeError("too little fill")

    forbidden_fill = {
        (int(r), int(c))
        for r, c in zip(*np.where(output == FILL))
    }
    used_ks = {int(k) for k, _b0, _b1 in segs}

    noise_count = random.choice(NOISE_COUNT_CHOICES)
    noise = _pick_isolated_noise(clean, forbidden_fill, used_ks, colored, noise_count)
    if noise is None:
        raise RuntimeError("not enough noise slots")

    inp = clean.copy()
    for cell in noise:
        inp[cell] = line
        colored.add(cell)

    expected = _apply_fill(inp, segs, bg=BG, fill=FILL)
    if not np.array_equal(main(inp), expected):
        raise RuntimeError("self-check failed")
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY):
        try:
            inp, _out = _generate_case()
            return inp
        except RuntimeError:
            continue
    raise RuntimeError("Failed to generate input")
