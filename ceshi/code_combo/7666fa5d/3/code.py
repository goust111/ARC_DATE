# Combo variant of 7666fa5d (index 3)
# combo_spec: D3/1+D15/2
# Name: 端点标记线段与异色干扰线段

import os
import random
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
SOLVER_DIR = os.path.join(ROOT, "solvers")
if SOLVER_DIR not in sys.path:
    sys.path.append(SOLVER_DIR)

from arc_utils import components_mask, get_bg as _bg_color


# Design analysis for the fusion:
# - Original task: one-color anti-diagonal segments define fill regions between consecutive k=r+c lines.
# - D3/1 changes the source object itself: each valid segment is now two-colored, with matching endpoint
#   markers and a different body color. This breaks the original "one connected color == one segment"
#   assumption, so the solver must reason over 8-connected mixed-color components instead of a single color.
# - D15/2 adds another diagonal family in a different color. If these distractors touched the fill zone,
#   the puzzle would become visually muddy, so the generator keeps them on main diagonals and outside the
#   active/fill envelope. That preserves human readability while still forcing color-based filtering.
# - The new observable rule is therefore structural, not numeric: "yellow-tipped anti-diagonal chains are
#   active; plain off-direction diagonals are noise." This is visible in 2-4 examples without hidden state.


def _cc8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the regions between marked anti-diagonal segments while ignoring off-direction noise lines.

    Input structure:
    - bg is the most common color.
    - Valid boundary objects are 8-connected anti-diagonal chains on one k=r+c line.
    - Each valid chain has matching endpoint-marker cells and a different single body color.
    - Extra diagonal segments of another color may appear elsewhere, but they do not use the endpoint pattern.

    Rule:
    - Detect the anti-diagonal components whose two ends share one color and whose interior cells share another color.
    - Treat those marked components as the true segments, extract each segment's k=r+c and b=r-c span.
    - For every main diagonal b, sort the active segments crossing that b by k and fill background cells strictly between consecutive segments with red (2).

    Key insight:
    - The puzzle now requires two filters at once: geometry selects anti-diagonal candidates, and endpoint markers distinguish real segments from same-grid distractors.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _bg_color(g)
    fill = 2

    candidates: list[tuple[int, int, int, int, int]] = []
    for pts in _cc8(g != bg):
        if len(pts) < 3:
            continue
        ks = {r + c for r, c in pts}
        if len(ks) != 1:
            continue

        ordered = sorted(pts, key=lambda rc: (rc[0] - rc[1], rc[0]))
        colors = [int(g[r, c]) for r, c in ordered]
        endpoint = colors[0]
        if endpoint != colors[-1]:
            continue

        interior = colors[1:-1]
        if not interior:
            continue
        body = interior[0]
        if body == endpoint or any(c != body for c in interior):
            continue

        bs = [r - c for r, c in ordered]
        candidates.append((int(next(iter(ks))), int(min(bs)), int(max(bs)), endpoint, body))

    if not candidates:
        return g.copy()

    groups: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
    for k, b0, b1, endpoint, body in candidates:
        groups.setdefault((endpoint, body), []).append((k, b0, b1))

    # Pick the marker/body pair that yields the richest consistent family of segments.
    active = max(groups.values(), key=lambda segs: (len(segs), sum(b1 - b0 for _k, b0, b1 in segs)))
    out = g.copy()

    for b in range(-(w - 1), h):
        crossing = [seg for seg in active if seg[1] <= b <= seg[2]]
        if len(crossing) < 2:
            continue
        crossing.sort(key=lambda seg: seg[0])
        r_min = max(0, b)
        r_max = min(h - 1, (w - 1) + b)
        for (k_left, _a0, _a1), (k_right, _b0, _b1) in zip(crossing, crossing[1:]):
            for r in range(r_min, r_max + 1):
                c = r - b
                k = r + c
                if k_left < k < k_right and out[r, c] == bg:
                    out[r, c] = fill

    return out


H = W = 16
BG = 8
FILL = 2
MARKER = 4
ACTIVE_BODY_COLORS = [1, 3, 5, 6, 7, 9]
DISTRACTOR_COLORS = [1, 3, 5, 6, 7, 9]

NUM_SEGMENTS_CHOICES = [4, 5]
K_LO, K_HI = 4, 26
K_GAPS = [4, 6]
SEG_LEN_MIN, SEG_LEN_MAX = 4, 8
OVERLAP_MIN_B = 4
MIN_FILL_CELLS = 6
DISTRACTOR_COUNT_CHOICES = [2, 3]
DISTRACTOR_LEN_MIN, DISTRACTOR_LEN_MAX = 3, 5
GEN_RETRY = 500


def _b_bounds_for_k(k: int) -> tuple[int, int]:
    b_lo = max(-k, k - 2 * (W - 1))
    b_hi = min(2 * (H - 1) - k, k)
    return b_lo, b_hi


def _anti_cells(k: int, b_min: int, b_max: int) -> list[tuple[int, int]]:
    return [((k + b) // 2, (k - b) // 2) for b in range(b_min, b_max + 1, 2)]


def _fill_from_segments(grid: np.ndarray, segs: list[tuple[int, int, int]]) -> np.ndarray:
    out = grid.copy()
    for b in range(-(W - 1), H):
        active = sorted([seg for seg in segs if seg[1] <= b <= seg[2]], key=lambda seg: seg[0])
        if len(active) < 2:
            continue
        r_min = max(0, b)
        r_max = min(H - 1, (W - 1) + b)
        for (k_left, _a0, _a1), (k_right, _b0, _b1) in zip(active, active[1:]):
            for r in range(r_min, r_max + 1):
                c = r - b
                if k_left < r + c < k_right and out[r, c] == BG:
                    out[r, c] = FILL
    return out


def _choose_k_values(parity: int) -> list[int] | None:
    num = random.choice(NUM_SEGMENTS_CHOICES)
    gaps = [random.choice(K_GAPS) for _ in range(num - 1)]
    span = sum(gaps)
    starts = [k for k in range(K_LO, K_HI - span + 1) if k % 2 == parity]
    if not starts:
        return None
    k0 = random.choice(starts)
    ks = [k0]
    for gap in gaps:
        ks.append(ks[-1] + gap)
    return ks


def _random_segment(k: int, prev_range: tuple[int, int] | None) -> tuple[int, int] | None:
    b_lo, b_hi = _b_bounds_for_k(k)
    if b_lo % 2 != k % 2:
        b_lo += 1
    if b_hi % 2 != k % 2:
        b_hi -= 1
    if b_lo > b_hi:
        return None

    choices: list[tuple[int, int]] = []
    for length in range(SEG_LEN_MIN, SEG_LEN_MAX + 1):
        span = 2 * (length - 1)
        for b_min in range(b_lo, b_hi - span + 1, 2):
            b_max = b_min + span
            if prev_range is not None:
                overlap = min(b_max, prev_range[1]) - max(b_min, prev_range[0]) + 2
                if overlap < OVERLAP_MIN_B:
                    continue
            choices.append((b_min, b_max))
    if not choices:
        return None
    return random.choice(choices)


def _expand_mask(mask: np.ndarray) -> np.ndarray:
    expanded = mask.copy()
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < H and 0 <= cc < W:
                    expanded[rr, cc] = True
    return expanded


def _main_diag_segment_cells(b: int, start_r: int, length: int) -> list[tuple[int, int]]:
    cells = []
    for i in range(length):
        r = start_r + i
        c = r - b
        cells.append((r, c))
    return cells


def _place_distractors(inp: np.ndarray, out: np.ndarray, color: int) -> bool:
    forbidden = _expand_mask(out != BG)
    placed: list[list[tuple[int, int]]] = []
    candidates: list[list[tuple[int, int]]] = []
    for b in range(-(W - 4), H - 2):
        r_min = max(0, b)
        r_max = min(H - 1, (W - 1) + b)
        for length in range(DISTRACTOR_LEN_MIN, DISTRACTOR_LEN_MAX + 1):
            for start_r in range(r_min, r_max - length + 2):
                cells = _main_diag_segment_cells(b, start_r, length)
                if any(forbidden[r, c] for r, c in cells):
                    continue
                candidates.append(cells)
    random.shuffle(candidates)

    need = random.choice(DISTRACTOR_COUNT_CHOICES)
    for cells in candidates:
        local_mask = np.zeros((H, W), dtype=bool)
        for r, c in cells:
            local_mask[r, c] = True
        if np.any(_expand_mask(local_mask) & forbidden):
            continue
        if any(np.any(_expand_mask(local_mask) & _expand_mask(_cells_to_mask(prev))) for prev in placed):
            continue
        placed.append(cells)
        forbidden |= _expand_mask(local_mask)
        if len(placed) == need:
            break

    if len(placed) < need:
        return False

    for cells in placed:
        for r, c in cells:
            inp[r, c] = color
            out[r, c] = color
    return True


def _cells_to_mask(cells: list[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros((H, W), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _gen_attempt() -> np.ndarray | None:
    parity = random.randint(0, 1)
    ks = _choose_k_values(parity)
    if ks is None:
        return None

    active_body = random.choice(ACTIVE_BODY_COLORS)
    distractor_choices = [c for c in DISTRACTOR_COLORS if c != active_body]
    distractor_color = random.choice(distractor_choices)

    segs: list[tuple[int, int, int]] = []
    prev_range = None
    for k in ks:
        seg = _random_segment(k, prev_range)
        if seg is None:
            return None
        segs.append((k, seg[0], seg[1]))
        prev_range = seg

    inp = np.full((H, W), BG, dtype=int)
    for k, b_min, b_max in segs:
        cells = _anti_cells(k, b_min, b_max)
        if len(cells) < 3:
            return None
        for idx, (r, c) in enumerate(cells):
            inp[r, c] = MARKER if idx in (0, len(cells) - 1) else active_body

    out = _fill_from_segments(inp, segs)
    if int(np.sum(out == FILL)) < MIN_FILL_CELLS:
        return None

    fill_mask = out == FILL
    inp[fill_mask] = BG

    if not _place_distractors(inp, out, distractor_color):
        return None

    # Fusion sanity checks: marked active family and monochrome off-direction distractors must both exist.
    if sum(1 for pts in _cc8(inp != BG) if len({r + c for r, c in pts}) == 1) < len(segs):
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("Failed to generate input")
