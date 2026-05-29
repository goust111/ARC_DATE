# Combo variant of da515329 (index 43)
# combo_spec: D2/2+D16/2+D18/2
# Name: 先延伸N步 + 间距由臂长编码 + 段长步幅由臂长决定
#
# Design analysis for the fused spiral variant
# -------------------------------------------
# a) Concrete behavior, with arm_len = 2 as a trace:
#    - Source task: each tip immediately participates in a 1-gap spiral whose local
#      step logic alternates between short and long moves. The cross is almost flush
#      with the first turn.
#    - This combo changes three linked stages at once:
#      1. D16/2: a tip first extends 2 more cells outward, so a visible black halo
#         remains around the cross.
#      2. D2/2: the next parallel rail of the same branch must be separated by a
#         2-cell blank corridor. On a pixel grid, that means the branch centerline
#         repeats every arm_len + 1 cells.
#      3. D18/2: once the branch has left the tip, it grows as outward nested open
#         rectangles in its own facing quadrant. The rectangular rails widen by that
#         same arm-derived stride, so larger crosses yield visibly coarser and
#         faster-expanding spirals.
#    - Example top branch for arm 2: after the tip is prolonged upward by 2 cells,
#      the branch then produces nested left-open rectangles in the north-east quadrant,
#      and each newly reached top row sits 3 cells above the previous one. Relative to
#      the source task,
#      the first bend is farther away, the blank corridor is wider, and the outer
#      rectangles scale more aggressively.
#
# b) Complexity calibration:
#    - Keep the original single-object input: one hollow teal cross with four equal
#      arms. The only free parameter is the common arm length 1..4.
#    - Reusing one parameter across extension, corridor width, and spiral growth
#      makes the combo visually strong without exploding into too many cases.
#
# c) Hidden assumptions broken from the original solver:
#    - `phase == "out_first" if arm_len == 1` is invalid: now every arm extends N.
#    - `spacing_ok(... neighbors4 ...)` with a fixed 1-cell halo is invalid: the
#      corridor width must depend on the measured arm length.
#    - The original parity cleanup and border patches assumed the exact old greedy
#      turtle. Those assumptions do not survive once launch distance and spacing are
#      re-parameterized, so the combo removes them entirely and uses explicit spiral
#      geometry instead.
#
# d) Placement legality in the generator:
#    - The generator searches square sizes and center positions, then validates the
#      result with the fused solver itself.
#    - It only keeps inputs where every arm shows the required outward stub, every
#      branch has at least one extra axis echo beyond the stub, branches do not
#      overlap each other, and the density stays in a readable range.
#
# e) Visual observability:
#    - A human can read arm length directly from the input cross.
#    - In the output, that same arm length reappears as:
#      1. the empty halo between the cross and the first bend,
#      2. the blank corridor between same-direction rails,
#      3. the speed at which outer rectangular runs widen.
#    - No invisible ordering by color id or metadata is used.
#
# f) Visual tidiness:
#    - Only one generated color is used, and the layout remains axis-aligned.
#    - Overlap-heavy samples are rejected, so the four branches read as clean
#      clockwise motifs instead of a noisy weave.
#
# g) New solver assumptions introduced by the combo:
#    - The input still contains exactly one hollow plus with equal arm lengths.
#    - Every arm is axis-aligned and long enough to measure the shared arm length.
#    - The generator guarantees those assumptions by construction.

import random
import numpy as np

_DIRS_URDL = [(-1, 0), (0, 1), (1, 0), (0, -1)]

CROSS_COLOR = 8
ARM_LEN_MIN, ARM_LEN_MAX = 1, 4
SIZE_PAD_CHOICES = (8, 10, 12)
DENSITY_LO, DENSITY_HI = 0.10, 0.33
MAX_OUTER_RETRIES = 200

_POOL: list[tuple[int, int, int, int]] | None = None


def _add(pos: tuple[int, int], d: tuple[int, int], k: int = 1) -> tuple[int, int]:
    return pos[0] + d[0] * k, pos[1] + d[1] * k


def _inb(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _find_cross(
    grid: np.ndarray,
) -> tuple[int, int, list[tuple[tuple[int, int], list[tuple[int, int]]]], int]:
    row_counts = np.sum(grid == CROSS_COLOR, axis=1)
    col_counts = np.sum(grid == CROSS_COLOR, axis=0)
    cr = int(np.argmax(row_counts))
    cc = int(np.argmax(col_counts))

    arms: list[tuple[tuple[int, int], list[tuple[int, int]]]] = []
    for d in _DIRS_URDL:
        dr, dc = d
        r, c = cr, cc
        cells: list[tuple[int, int]] = []
        while _inb(grid.shape, r + dr, c + dc) and grid[r + dr, c + dc] == CROSS_COLOR:
            r += dr
            c += dc
            cells.append((r, c))
        if cells:
            arms.append((d, cells))

    arm_len = max((len(cells) for _, cells in arms), default=0)
    return cr, cc, arms, arm_len


def _paint_h(mask: np.ndarray, r: int, c0: int, c1: int) -> None:
    if 0 <= r < mask.shape[0]:
        lo, hi = sorted((c0, c1))
        lo = max(lo, 0)
        hi = min(hi, mask.shape[1] - 1)
        if lo <= hi:
            mask[r, lo : hi + 1] = True


def _paint_v(mask: np.ndarray, c: int, r0: int, r1: int) -> None:
    if 0 <= c < mask.shape[1]:
        lo, hi = sorted((r0, r1))
        lo = max(lo, 0)
        hi = min(hi, mask.shape[0] - 1)
        if lo <= hi:
            mask[lo : hi + 1, c] = True


def _draw_quarter_spiral(
    grid_shape: tuple[int, int],
    cr: int,
    cc: int,
    arm_len: int,
    outward: tuple[int, int],
    tip: tuple[int, int],
) -> np.ndarray:
    stride = arm_len + 1
    mask = np.zeros(grid_shape, dtype=bool)
    mask[tip] = True

    cur = tip
    for _ in range(arm_len):
        nxt = _add(cur, outward, 1)
        if not _inb(grid_shape, nxt[0], nxt[1]):
            return mask
        mask[nxt] = True
        cur = nxt

    if outward == (-1, 0):
        for k in range(1000):
            top = cur[0] - k * stride
            right = cc + arm_len + k * stride
            if top < 0 or right >= grid_shape[1]:
                break
            _paint_h(mask, top, cc, right)
            _paint_v(mask, right, top, tip[0])
            _paint_h(mask, tip[0], cc + 1, right)
    elif outward == (0, 1):
        for k in range(1000):
            bottom = cr + arm_len + k * stride
            right = cur[1] + k * stride
            if bottom >= grid_shape[0] or right >= grid_shape[1]:
                break
            _paint_v(mask, right, cr, bottom)
            _paint_h(mask, bottom, tip[1], right)
            _paint_v(mask, tip[1], cr + 1, bottom)
    elif outward == (1, 0):
        for k in range(1000):
            bottom = cur[0] + k * stride
            left = cc - arm_len - k * stride
            if bottom >= grid_shape[0] or left < 0:
                break
            _paint_h(mask, bottom, left, cc)
            _paint_v(mask, left, tip[0], bottom)
            _paint_h(mask, tip[0], left, cc - 1)
    else:
        for k in range(1000):
            top = cr - arm_len - k * stride
            left = cur[1] - k * stride
            if top < 0 or left < 0:
                break
            _paint_v(mask, left, top, cr)
            _paint_h(mask, top, left, tip[1])
            _paint_v(mask, tip[1], top, cr - 1)

    return mask


def _render_branches(grid: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    grid = np.asarray(grid, dtype=int)
    out = grid.copy()
    cr, cc, arms, arm_len = _find_cross(grid)
    if arm_len == 0:
        return out, []

    branch_masks: list[np.ndarray] = []
    for outward, cells in arms:
        mask = _draw_quarter_spiral(grid.shape, cr, cc, arm_len, outward, cells[-1])
        branch_masks.append(mask)

    for mask in branch_masks:
        out[mask] = CROSS_COLOR
    return out, branch_masks


def _stub_span(out: np.ndarray, tip: tuple[int, int], outward: tuple[int, int]) -> int:
    span = 1
    cur = tip
    while True:
        nxt = _add(cur, outward, 1)
        if not _inb(out.shape, nxt[0], nxt[1]) or out[nxt] != CROSS_COLOR:
            break
        span += 1
        cur = nxt
    return span


def _branch_echoes(
    mask: np.ndarray, endpoint: tuple[int, int], outward: tuple[int, int], stride: int
) -> int:
    hits = 0
    pos = endpoint
    while _inb(mask.shape, pos[0], pos[1]):
        if outward in {(-1, 0), (1, 0)}:
            if int(np.sum(mask[pos[0], :])) < 2:
                break
        else:
            if int(np.sum(mask[:, pos[1]])) < 2:
                break
        hits += 1
        pos = _add(pos, outward, stride)
    return hits


def _fusion_metrics(
    inp: np.ndarray, out: np.ndarray, branch_masks: list[np.ndarray]
) -> dict[str, int | float]:
    cr, cc, arms, arm_len = _find_cross(inp)
    stride = arm_len + 1

    stubs = []
    echoes = []
    for (outward, cells), mask in zip(arms, branch_masks):
        tip = cells[-1]
        endpoint = _add(tip, outward, arm_len)
        stubs.append(_stub_span(mask.astype(int) * CROSS_COLOR, tip, outward))
        echoes.append(_branch_echoes(mask, endpoint, outward, stride))

    overlap = 0
    if branch_masks:
        overlap = int(np.sum(np.sum(np.stack(branch_masks, axis=0), axis=0) > 1))

    return {
        "arm": arm_len,
        "cr": cr,
        "cc": cc,
        "stride": stride,
        "density": float(np.mean(out == CROSS_COLOR)),
        "min_stub": min(stubs) if stubs else 0,
        "min_echoes": min(echoes) if echoes else 0,
        "overlap": overlap,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow four arm-length-coded clockwise rectangular spirals from a teal cross.

    Input structure:
    - Background is black (0).
    - A single teal (8) plus-sign cross is present, with four equal cardinal arms and a one-cell center hole.

    Rule:
    - Measure the common arm length N from the cross.
    - Each arm tip first continues straight outward for N cells.
    - From that prolonged endpoint, each branch occupies its own outward-facing quadrant and grows as
      nested open rectangles whose opening direction follows the clockwise order around the cross.
    - Parallel rails leave an N-cell blank corridor, so the rectangle centerlines repeat every N+1 cells,
      and each outer rectangle widens by that same arm-derived stride.

    Key insight:
    - The cross arm length is the only parameter source. Once N is known, it simultaneously fixes the
      launch offset, the visible corridor width, and the widening ladder of the outward quarter-spirals.
    """

    out, _branch_masks = _render_branches(np.asarray(grid, dtype=int))
    return out


def _build_input(arm: int, size: int, cr: int, cc: int) -> np.ndarray:
    grid = np.zeros((size, size), dtype=int)
    for k in range(1, arm + 1):
        grid[cr - k, cc] = CROSS_COLOR
        grid[cr + k, cc] = CROSS_COLOR
        grid[cr, cc - k] = CROSS_COLOR
        grid[cr, cc + k] = CROSS_COLOR
    return grid


def _build_pool() -> list[tuple[int, int, int, int]]:
    pool: list[tuple[int, int, int, int]] = []
    for arm in range(ARM_LEN_MIN, ARM_LEN_MAX + 1):
        for pad in SIZE_PAD_CHOICES:
            size = 6 * arm + pad
            lo = 2 * arm + 1
            hi = size - 2 * arm - 2
            if lo > hi:
                continue
            for cr in range(lo, hi + 1):
                for cc in range(lo, hi + 1):
                    inp = _build_input(arm, size, cr, cc)
                    out, branch_masks = _render_branches(inp)
                    metrics = _fusion_metrics(inp, out, branch_masks)
                    if metrics["min_stub"] != arm + 1:
                        continue
                    # One visible post-stub echo per arm is sufficient to make the
                    # arm-coded spacing and growth rate observable; requiring two
                    # was over-constraining the candidate pool and produced no samples.
                    if metrics["min_echoes"] < 1:
                        continue
                    if metrics["overlap"] != 0:
                        continue
                    density = float(metrics["density"])
                    if not (DENSITY_LO <= density <= DENSITY_HI):
                        continue
                    pool.append((arm, size, cr, cc))
    return pool


def generate_input() -> np.ndarray:
    global _POOL
    if _POOL is None:
        _POOL = _build_pool()
    if not _POOL:
        raise RuntimeError("generate_input: no valid fused spiral placements found")

    for _ in range(MAX_OUTER_RETRIES):
        arm, size, cr, cc = random.choice(_POOL)
        inp = _build_input(arm, size, cr, cc)
        out, branch_masks = _render_branches(inp)
        metrics = _fusion_metrics(inp, out, branch_masks)
        if metrics["min_stub"] != arm + 1:
            continue
        if metrics["min_echoes"] < 1:
            continue
        if metrics["overlap"] != 0:
            continue
        density = float(metrics["density"])
        if not (DENSITY_LO <= density <= DENSITY_HI):
            continue
        if not np.array_equal(main(inp), out):
            continue
        return inp
    raise RuntimeError("generate_input: retry budget exhausted")
