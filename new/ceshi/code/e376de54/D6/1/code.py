# Variant of e376de54
# Dimension: D6 — 线段之间的空间间距 (meta)
# Change: 平行线段之间保持固定间距（通常1-3像素），间距在变换后不变... → 间距按长度排序重排

import random

import numpy as np

from arc_utils import get_bg

# Design analysis for the D6 variant
# a) Concrete behavior:
#    The input still contains 3 or 5 parallel monochrome segments on distinct parallel lines.
#    They keep one shared endpoint directionally, and the median-by-offset segment still provides
#    the target length. The change is in the destination slots: the parallel lines are treated as
#    ordered slots from low s to high s, and after normalizing every segment to the reference
#    length, the colors are reassigned to those slots by INPUT length order (longest -> shortest).
#    Small trace: if three horizontal segments at rows 2/5/8 have lengths [4, 7, 5], original
#    logic would leave colors on rows [2,5,8] and only change all lengths to 7. The variant still
#    outputs length 7 everywhere, but row 2 receives the color of length-7, row 5 receives the
#    color of length-5, and row 8 receives the color of length-4.
# b) Complexity calibration:
#    Ordering becomes ambiguous if equal lengths are allowed, so the generator uses pairwise
#    distinct lengths. The slot count stays at 3 or 5 to keep the ranking human-readable.
# c) Old assumptions that break:
#    - Original `dt = ref_t0 - s["t_start"]` only works because each segment stayed on its own
#      perpendicular line. Once lines are reassigned by rank, source identity and destination slot
#      must be decoupled.
#    - Original output painting reused each segment's own start cell; the variant must instead
#      reconstruct cells from `(ref_t0, destination_s)`.
#    - Original irregular offsets were harmless because s order only selected the reference. Here
#      the slot order is semantically important, so the generator uses evenly spaced offsets.
# d) Placement legality:
#    The generator first chooses evenly spaced slot offsets, then finds a common reference extent
#    that fits on every slot. Input segments are derived by shared-start or shared-end anchoring,
#    all stay in bounds, and output uses the same slot set so segments never overlap.
# e) Visual observability:
#    Colors are unique per segment so the permutation is visible without relying on hidden IDs.
#    The generator rejects identity rank orders, so output differs from the original "keep slots"
#    behavior on every accepted sample.
# f) Visual cleanliness:
#    Only one family of thin parallel lines is present, with uniform slot spacing and no extra
#    markers. This keeps the permutation easy to read.
# g) Color logic:
#    Colors do not transform; they only move with their source segment. The sortable attribute is
#    visible geometric length, never the color id itself.
# New solver assumptions audited against the generator:
#    - segment count is odd, so the median slot is unique;
#    - perpendicular offsets are unique and ordered, so slot positions are well-defined;
#    - input lengths are distinct, so the longest->shortest ranking is unique;
#    - all output cells for `(ref_t0, slot_s, ref_len)` are in bounds because the generator
#      validates the common output extent before building the input.

_DIR_CANDIDATES = [(0, 1), (1, 0), (1, 1), (1, -1)]  # horiz, vert, diag \, diag /


def _pick_main_dir(g: np.ndarray, bg: int) -> tuple[int, int]:
    counts = []
    h, w = g.shape
    for dr, dc in _DIR_CANDIDATES:
        cnt = 0
        for r in range(h):
            for c in range(w):
                v = int(g[r, c])
                if v == bg:
                    continue
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and int(g[rr, cc]) == v:
                    cnt += 1
        counts.append(((dr, dc), cnt))

    best_cnt = max(cnt for _, cnt in counts)
    assert best_cnt > 0, "main direction must be detectable"
    best_dirs = [d for d, cnt in counts if cnt == best_cnt]
    assert len(best_dirs) == 1, "main direction must be unique"
    return best_dirs[0]


def _dot(p: tuple[int, int], d: tuple[int, int]) -> int:
    return p[0] * d[0] + p[1] * d[1]


def _perp_dot(p: tuple[int, int], d: tuple[int, int]) -> int:
    return p[0] * d[1] - p[1] * d[0]


def _step(d: tuple[int, int]) -> int:
    dr, dc = d
    return dr * dr + dc * dc


def _cell_from_ts(shape: tuple[int, int], d: tuple[int, int], t: int, s: int) -> tuple[int, int] | None:
    h, w = shape
    if d == (0, 1):
        r, c = s, t
    elif d == (1, 0):
        r, c = t, -s
    elif d == (1, 1):
        if (t + s) % 2 or (t - s) % 2:
            return None
        r, c = (t + s) // 2, (t - s) // 2
    else:
        if (t - s) % 2 or (-s - t) % 2:
            return None
        r, c = (t - s) // 2, (-s - t) // 2
    if 0 <= r < h and 0 <= c < w:
        return int(r), int(c)
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Normalize parallel colored segments to the reference length, then reorder which slot each color occupies by input length.

    Input structure:
    - The background is the most common color.
    - Several multi-cell segments of solid colors lie on distinct parallel lines, all sharing one dominant direction.
    - The parallel lines act as ordered slots, and segment lengths are intentionally different so their rank is visible.

    Rule:
    - Detect the shared direction and extract every multi-cell segment on it.
    - Use the segment at the median perpendicular offset as the reference; its start position and length define the common output extent.
    - Rank the input segments from longest to shortest, rank the slot positions by perpendicular offset, and place the i-th longest color onto the i-th slot.
    - Draw every output segment with the reference start and reference length; isolated pixels remain untouched.

    Key insight: The geometry is still normalized to the median slot's extent, but the segment identities no longer stay on their original lines; slot order is reassigned by visible length rank.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    d = _pick_main_dir(g, bg)
    dr, dc = d
    delta_t = _step(d)

    segments = []
    colors = [int(c) for c in np.unique(g) if int(c) != bg]
    for col in colors:
        mask = g == col
        for r in range(h):
            for c in range(w):
                if not mask[r, c]:
                    continue
                pr, pc = r - dr, c - dc
                if 0 <= pr < h and 0 <= pc < w and mask[pr, pc]:
                    continue
                cells = []
                rr, cc = r, c
                while 0 <= rr < h and 0 <= cc < w and mask[rr, cc]:
                    cells.append((rr, cc))
                    rr += dr
                    cc += dc
                if len(cells) < 2:
                    continue
                start = cells[0]
                end = cells[-1]
                segments.append(
                    {
                        "color": col,
                        "cells": cells,
                        "start": start,
                        "end": end,
                        "t_start": _dot(start, d),
                        "t_end": _dot(end, d),
                        "s": _perp_dot(start, d),
                        "len": len(cells),
                    }
                )

    assert segments, "at least one multi-cell segment must exist"

    s_vals = sorted(s["s"] for s in segments)
    if delta_t == 2:
        for i in range(len(s_vals) - 1):
            assert s_vals[i + 1] - s_vals[i] >= 4
    assert len(s_vals) % 2 == 1, "segment count must have a single median"
    assert len(s_vals) == len(set(s_vals)), "segment line coordinates must be unique"

    ref_s = s_vals[len(s_vals) // 2]
    ref_cands = [s for s in segments if s["s"] == ref_s]
    assert len(ref_cands) == 1, "median reference segment must be unique"
    ref = ref_cands[0]
    ref_t0 = int(ref["t_start"])
    ref_t1 = int(ref["t_end"])
    target_len = int(ref["len"])
    assert ref_t1 >= ref_t0, "traced segment coordinates must increase"
    assert (ref_t1 - ref_t0) % delta_t == 0, "reference segment length must align with delta_t"

    out = g.copy()
    for s in segments:
        for r, c in s["cells"]:
            out[r, c] = bg

    slots = sorted(segments, key=lambda s: (int(s["s"]), int(s["t_start"]), int(s["color"])))
    ranked = sorted(segments, key=lambda s: (-int(s["len"]), int(s["s"]), int(s["color"])))

    for slot, src in zip(slots, ranked):
        start_cell = _cell_from_ts((h, w), d, ref_t0, int(slot["s"]))
        assert start_cell is not None, "destination slot must admit the reference start"
        rr, cc = start_cell
        for _ in range(target_len):
            assert 0 <= rr < h and 0 <= cc < w, "reordered segment must stay in bounds"
            out[rr, cc] = int(src["color"])
            rr += dr
            cc += dc

    return out


# --------------- generator (D6 variant) ---------------

_H, _W = 16, 16
_BG = 7

_DIRS = [(0, 1), (1, 0), (1, 1), (1, -1)]
_COLORS = [1, 2, 3, 5, 6, 8, 9]

_SEGMENT_COUNTS = [3, 5]
_REF_LEN_RANGE = (3, 8)
_MAX_SEG_LEN = 12

_PERP_GAP_ORTHO = [2, 3, 4, 5]
_PERP_GAP_DIAG = [4, 6]

_T_SCAN_RANGE = (-30, 31)
_GEN_RETRIES = 400


def _segment_cells(d: tuple[int, int], t0: int, s: int, length: int) -> list[tuple[int, int]] | None:
    delta = _step(d)
    cells = []
    for k in range(length):
        cell = _cell_from_ts((_H, _W), d, t0 + k * delta, s)
        if cell is None:
            return None
        cells.append(cell)
    return cells


def _max_len_from(d: tuple[int, int], t0: int, s: int) -> int:
    delta = _step(d)
    length = 0
    while _cell_from_ts((_H, _W), d, t0 + length * delta, s) is not None:
        length += 1
    return length


def _max_len_to(d: tuple[int, int], t1: int, s: int) -> int:
    delta = _step(d)
    length = 0
    while _cell_from_ts((_H, _W), d, t1 - length * delta, s) is not None:
        length += 1
    return length


def _s_limits(d: tuple[int, int]) -> tuple[int, int]:
    if d == (0, 1):
        return 0, 15
    if d == (1, 0):
        return -15, 0
    if d == (1, 1):
        return -15, 15
    return -30, 0


def _choose_offsets(d: tuple[int, int], n: int) -> tuple[int, ...]:
    lo, hi = _s_limits(d)
    allowed = _PERP_GAP_DIAG if d in [(1, 1), (1, -1)] else _PERP_GAP_ORTHO
    feasible = [gap for gap in allowed if lo + (n - 1) * gap <= hi]
    gap = random.choice(feasible)
    s0 = random.randint(lo, hi - (n - 1) * gap)
    return tuple(s0 + i * gap for i in range(n))


def _feasible_placements(d: tuple[int, int], s_vals: tuple[int, ...], ref_len: int):
    delta = _step(d)
    ref_idx = len(s_vals) // 2
    placements = []
    for t0 in range(*_T_SCAN_RANGE):
        if any(_segment_cells(d, t0, s, ref_len) is None for s in s_vals):
            continue
        t1 = t0 + (ref_len - 1) * delta
        start_caps = [_max_len_from(d, t0, s) for s in s_vals]
        if any(start_caps[i] > ref_len for i in range(len(s_vals)) if i != ref_idx):
            placements.append((t0, "start", start_caps))
        end_caps = [_max_len_to(d, t1, s) for s in s_vals]
        if any(end_caps[i] > ref_len for i in range(len(s_vals)) if i != ref_idx):
            placements.append((t0, "end", end_caps))
    return placements


def _rank_indices_by_length(lengths: list[int], s_vals: tuple[int, ...]) -> list[int]:
    return sorted(range(len(lengths)), key=lambda i: (-int(lengths[i]), int(s_vals[i]), i))


def _choose_colors(n: int) -> tuple[int, ...]:
    return tuple(random.sample(_COLORS, n))


def _choose_lengths(ref_len: int, caps: list[int], n: int) -> list[int] | None:
    ref_idx = n // 2
    max_lens = [min(_MAX_SEG_LEN, int(cap)) for cap in caps]
    if max_lens[ref_idx] < ref_len:
        return None

    domains = {}
    for i in range(n):
        if i == ref_idx:
            domains[i] = [ref_len]
        else:
            domains[i] = [length for length in range(2, max_lens[i] + 1) if length != ref_len]
            if not domains[i]:
                return None

    order = [ref_idx] + [i for i in range(n) if i != ref_idx]
    for _ in range(120):
        lengths = [None] * n
        used = set()

        def backtrack(pos: int) -> bool:
            if pos == len(order):
                return True
            idx = order[pos]
            candidates = domains[idx][:]
            random.shuffle(candidates)
            for length in candidates:
                if length in used:
                    continue
                lengths[idx] = length
                used.add(length)
                if backtrack(pos + 1):
                    return True
                used.remove(length)
                lengths[idx] = None
            return False

        if not backtrack(0):
            continue
        if not any(length < ref_len for length in lengths if length is not None):
            continue
        if not any(length > ref_len for length in lengths if length is not None):
            continue
        if _rank_indices_by_length(lengths, tuple(range(n))) == list(range(n)):
            continue
        return [int(length) for length in lengths]
    return None


def _paint(grid: np.ndarray, d: tuple[int, int], s_vals: tuple[int, ...], starts: list[int], lengths: list[int], colors: tuple[int, ...]) -> bool:
    for s, t0, length, color in zip(s_vals, starts, lengths, colors):
        cells = _segment_cells(d, t0, s, length)
        if cells is None:
            return False
        for r, c in cells:
            if grid[r, c] != _BG:
                return False
            grid[r, c] = color
    return True


def _gen_attempt() -> np.ndarray | None:
    d = random.choice(_DIRS)
    n = random.choice(_SEGMENT_COUNTS)
    ref_len = random.randint(*_REF_LEN_RANGE)
    s_vals = _choose_offsets(d, n)
    placements = _feasible_placements(d, s_vals, ref_len)
    if not placements:
        return None

    t0, anchor, caps = random.choice(placements)
    lengths = _choose_lengths(ref_len, caps, n)
    if lengths is None:
        return None

    order = _rank_indices_by_length(lengths, s_vals)
    if order == list(range(n)):
        return None

    colors = _choose_colors(n)
    out_colors = tuple(colors[i] for i in order)
    out = np.full((_H, _W), _BG, dtype=int)
    if not _paint(out, d, s_vals, [t0] * n, [ref_len] * n, out_colors):
        return None

    delta = _step(d)
    t1 = t0 + (ref_len - 1) * delta
    if anchor == "start":
        starts = [t0] * n
    else:
        starts = [t1 - (length - 1) * delta for length in lengths]

    inp = np.full((_H, _W), _BG, dtype=int)
    if not _paint(inp, d, s_vals, starts, lengths, colors):
        return None

    try:
        if np.array_equal(main(inp), out):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
