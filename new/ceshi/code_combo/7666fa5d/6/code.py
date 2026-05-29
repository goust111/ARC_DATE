# Combo variant of 7666fa5d (index 6)
# combo_spec: D6/0+D7/0
# Name: 纯填充掩码紧凑裁剪

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import components_mask, get_bg as _bg_color


# Design analysis for the fused variant
# a) Concrete behavior trace:
#    Suppose two anti-diagonal segments lie on k=6 and k=10 and both cover b=-2,0,2.
#    The original task keeps those two diagonal boundaries and paints the strict in-between
#    anti-diagonal k=8 red. This variant first infers the same red cells, then deletes the
#    source segments, so the output contains only the red interior. Finally the output is
#    cropped to the bounding box of those red cells alone. With more than two segments the
#    same scan over b=r-c yields several filled bands, but only the union of red pixels
#    survives to the output.
# b) Complexity calibration:
#    The generator preserves the original 4-5 anti-diagonal structure so the task still has
#    repeated evidence for the fill rule. To make the fusion visually observable, it rejects
#    cases where the fill-only crop matches the bbox of the line-preserving output; that
#    forces line removal to affect the crop, not just the colors.
# c) Hidden-assumption review of the original solver:
#    - "Largest non-bg color is the line color" still holds because inputs remain line-only.
#    - "Each 8-connected component sits on one constant k" still holds because generation
#      places each segment on a single anti-diagonal and keeps different ks separated.
#    - "Filling writes only over background" changes: the logical fill is still computed from
#      background cells, but the returned grid no longer preserves the source segments.
#    - "Output shape equals input shape" is removed; the returned grid is a tight crop of the
#      fill mask, so any downstream code must not assume 16x16 output.
# d) Placement legality:
#    Segment coordinates are generated from valid (k,b) ranges, so every cell is in-bounds.
#    Split segments on the same k are separated by a b-gap that breaks 8-connectivity.
#    Different ks are spaced by >=4, preventing overlaps between distinct anti-diagonals.
# e) Observability for a human solver:
#    The input still exposes only parallel diagonal segments; the output shows only compact
#    red parallelogram interiors. Across examples, the common pattern is that red appears
#    where the gaps between neighboring segments would have been, while the diagonal borders
#    themselves disappear and the canvas shrinks around the remaining red mask.
# f) Visual cleanliness:
#    The output uses only bg and red, with no overprinted line colors. Tight cropping removes
#    irrelevant empty margins, so the transformed object is a clean compact mask.
# g) New solver assumptions and generator guarantees:
#    The new main() assumes there is at least one fill cell after applying the diagonal scan.
#    The generator enforces that by requiring a minimum fill area and by rejecting any sample
#    whose fill-only bbox is not strictly smaller than the bbox of the line-preserving result.


H = W = 16
BG = 8
FILL = 2
LINE_COLORS = [1, 3, 4, 5, 6, 7, 9]

K_LO, K_HI = 4, 26
GAP_VALUES = (4, 6, 8)
GAP_TARGET_SPANS = (14, 16)
NUM_KS_CHOICES = [4, 5]
NUM_KS_WEIGHTS = [1, 1]

SEG_LEN_MIN, SEG_LEN_MAX = 2, 9
OVERLAP_MIN_B = 2

SPLIT_PROB = 0.4
SPLIT_MIN_LEN = 4
SPLIT_B_GAP = 4
SPLIT_PIECE2_LEN_MIN = 2
SPLIT_PIECE2_LEN_MAX = 6

SEG_RETRY = 40
SPLIT_RETRY = 10
GEN_RETRY = 500

MIN_FILL_CELLS = 6
MIN_CROP_SIDE = 2
MIN_LINE_CELLS = 8


def _cc8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8)]


def _bbox_of_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_fill_mask(fill_mask: np.ndarray, bg: int) -> np.ndarray:
    box = _bbox_of_mask(fill_mask)
    if box is None:
        return np.full((1, 1), int(bg), dtype=int)
    r0, c0, r1, c1 = box
    out = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
    out[fill_mask[r0 : r1 + 1, c0 : c1 + 1]] = FILL
    return out


def _line_cells_inside_fill_crop(grid: np.ndarray, fill_mask: np.ndarray, line: int) -> int:
    box = _bbox_of_mask(fill_mask)
    if box is None:
        return 0
    r0, c0, r1, c1 = box
    return int(np.sum(grid[r0 : r1 + 1, c0 : c1 + 1] == line))


def _extract_segments(grid: np.ndarray, line: int) -> list[tuple[int, int, int]]:
    segs: list[tuple[int, int, int]] = []
    for pts in _cc8(grid == line):
        ks = {r + c for r, c in pts}
        if len(ks) != 1:
            continue
        k = int(next(iter(ks)))
        bs = [r - c for r, c in pts]
        segs.append((k, int(min(bs)), int(max(bs))))
    return segs


def _fill_mask_from_segments(grid: np.ndarray, bg: int, segs: list[tuple[int, int, int]]) -> np.ndarray:
    h, w = grid.shape
    fill_mask = np.zeros((h, w), dtype=bool)

    for b in range(-(w - 1), h):
        active = [s for s in segs if s[1] <= b <= s[2]]
        if len(active) < 2:
            continue
        active.sort(key=lambda t: t[0])

        r_min = max(0, b)
        r_max = min(h - 1, (w - 1) + b)
        for left_seg, right_seg in zip(active, active[1:]):
            k_left, k_right = int(left_seg[0]), int(right_seg[0])
            for r in range(r_min, r_max + 1):
                c = r - b
                k = r + c
                if k_left < k < k_right and grid[r, c] == bg:
                    fill_mask[r, c] = True
    return fill_mask


def _render_full_output(grid: np.ndarray, bg: int, segs: list[tuple[int, int, int]]) -> tuple[np.ndarray, np.ndarray]:
    fill_mask = _fill_mask_from_segments(grid, bg, segs)
    out = grid.copy()
    out[fill_mask] = FILL
    return out, fill_mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer the red fill between parallel anti-diagonal segments, then keep only that fill as a compact mask.

    Input structure:
    - The input is still a 16x16 grid with one non-background line color on several anti-diagonals (r+c = const).
    - Each 8-connected component is a short contiguous segment on one anti-diagonal.
    - No red fill is present yet; the only clues are the segment positions and overlaps in b=r-c space.

    Rule:
    - Reconstruct the same fill as the base task: for each main diagonal b=r-c, collect segments covering that b,
      sort them by k=r+c, and mark background cells strictly between consecutive active segments as red.
    - After the fill is inferred, discard every original line pixel instead of preserving it.
    - Return the tight crop around the remaining red fill cells only.

    Key insight:
    The diagonal reasoning is unchanged from the base task, but the visible answer is not the bounded frame.
    The true output is the interior fill mask alone, trimmed to its minimal bounding rectangle.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    non_bg = [c for c in sorted(set(map(int, g.ravel()))) if c != bg]
    assert non_bg, "grid must contain at least one non-background color"

    line = max(non_bg, key=lambda c: int(np.sum(g == c)))
    segs = _extract_segments(g, line)
    fill_mask = _fill_mask_from_segments(g, bg, segs)
    return _crop_fill_mask(fill_mask, bg)


def _b_bounds_for_k(k: int) -> tuple[int, int]:
    b_lo = max(-k, k - 2 * (W - 1))
    b_hi = min(2 * (H - 1) - k, k)
    return b_lo, b_hi


def _seg_cells(k: int, b_min: int, b_max: int) -> list[tuple[int, int]]:
    cells = []
    for b in range(b_min, b_max + 1, 2):
        cells.append(((k + b) // 2, (k - b) // 2))
    return cells


def _choose_k_values(parity: int) -> list[int] | None:
    num_ks = random.choices(NUM_KS_CHOICES, weights=NUM_KS_WEIGHTS)[0]
    if num_ks == 4:
        valid_combos = [
            (a, b, c)
            for a in GAP_VALUES
            for b in GAP_VALUES
            for c in GAP_VALUES
            if a + b + c in GAP_TARGET_SPANS
        ]
        gaps = list(random.choice(valid_combos))
    else:
        gaps = [4, 4, 4, 4]

    total_span = sum(gaps)
    k0_candidates = [k for k in range(K_LO, K_HI - total_span + 1) if k % 2 == parity]
    if not k0_candidates:
        return None

    ks = [random.choice(k0_candidates)]
    for gap in gaps:
        ks.append(ks[-1] + gap)
    if ks[-1] > K_HI or len(set(ks)) != len(ks):
        return None
    return ks


def _random_seg(
    k: int,
    constrain_lo: int | None = None,
    constrain_hi: int | None = None,
) -> tuple[int, int] | None:
    b_lo, b_hi = _b_bounds_for_k(k)
    if b_lo % 2 != k % 2:
        b_lo += 1
    if b_hi % 2 != k % 2:
        b_hi -= 1
    if b_lo > b_hi:
        return None

    for _ in range(SEG_RETRY):
        length = random.randint(SEG_LEN_MIN, SEG_LEN_MAX)
        step = 2 * (length - 1)
        if b_lo + step > b_hi:
            continue
        candidates = list(range(b_lo, b_hi - step + 1, 2))
        if constrain_lo is not None and constrain_hi is not None:
            candidates = [
                bm
                for bm in candidates
                if bm + step >= constrain_lo + OVERLAP_MIN_B and bm <= constrain_hi - OVERLAP_MIN_B
            ]
            if not candidates:
                continue
        b_min = random.choice(candidates)
        return b_min, b_min + step
    return None


def _try_split_segment(segs: list[tuple[int, int, int]], idx: int) -> None:
    k, bmin, bmax = segs[idx]
    length = (bmax - bmin) // 2 + 1
    if length < SPLIT_MIN_LEN:
        return

    _, b_hi = _b_bounds_for_k(k)
    for _ in range(SPLIT_RETRY):
        first_len = random.randint(2, length - 2)
        piece1_max = bmin + 2 * (first_len - 1)
        p2_candidates = [b for b in range(piece1_max + SPLIT_B_GAP, b_hi - 1, 2) if b % 2 == k % 2]
        if not p2_candidates:
            continue
        p2_min = random.choice(p2_candidates)
        second_len = random.randint(SPLIT_PIECE2_LEN_MIN, SPLIT_PIECE2_LEN_MAX)
        p2_max = p2_min + 2 * (second_len - 1)
        if p2_max > b_hi:
            continue
        segs[idx] = (k, bmin, piece1_max)
        segs.append((k, p2_min, p2_max))
        return


def _variant_is_observable(full_output: np.ndarray, fill_mask: np.ndarray, line: int) -> bool:
    fill_box = _bbox_of_mask(fill_mask)
    full_box = _bbox_of_mask(full_output != BG)
    if fill_box is None or full_box is None:
        return False
    if fill_box == full_box:
        return False
    cropped = _crop_fill_mask(fill_mask, BG)
    if cropped.shape == (H, W):
        return False
    if cropped.shape[0] < MIN_CROP_SIDE or cropped.shape[1] < MIN_CROP_SIDE:
        return False
    if int(np.sum(full_output == line)) < MIN_LINE_CELLS:
        return False
    # D6 must be visible inside the cropped frame itself; otherwise line removal could be
    # mistaken for an artifact of cropping away all source pixels.
    if _line_cells_inside_fill_crop(full_output, fill_mask, line) == 0:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    line = random.choice(LINE_COLORS)
    parity = random.randint(0, 1)
    ks = _choose_k_values(parity)
    if ks is None:
        return None

    segs: list[tuple[int, int, int]] = []
    first = _random_seg(ks[0])
    if first is None:
        return None
    segs.append((ks[0], first[0], first[1]))

    for k in ks[1:]:
        prev_bmin, prev_bmax = segs[-1][1], segs[-1][2]
        seg = _random_seg(k, prev_bmin, prev_bmax)
        if seg is None:
            return None
        segs.append((k, seg[0], seg[1]))

    if len(segs) <= 5 and random.random() < SPLIT_PROB:
        _try_split_segment(segs, random.randrange(len(segs)))

    grid = np.full((H, W), BG, dtype=int)
    for k, bmin, bmax in segs:
        for r, c in _seg_cells(k, bmin, bmax):
            if not (0 <= r < H and 0 <= c < W):
                return None
            if grid[r, c] != BG:
                return None
            grid[r, c] = line

    full_output, fill_mask = _render_full_output(grid, BG, segs)
    if int(np.sum(fill_mask)) < MIN_FILL_CELLS:
        return None
    if not _variant_is_observable(full_output, fill_mask, line):
        return None

    expected = _crop_fill_mask(fill_mask, BG)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("Failed to generate input")
