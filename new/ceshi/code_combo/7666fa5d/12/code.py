# Combo variant of 7666fa5d (index 12)
# combo_spec: D1/3+D2/3
# Name: 镜像对称配对+混合方向线段

from __future__ import annotations

import os
import random
import sys

import numpy as np

ROOT = "/home/lixinyu/snap/new"
for _path in (ROOT, os.path.join(ROOT, "solvers")):
    if _path not in sys.path:
        sys.path.append(_path)

from arc_utils import components_mask, get_bg as _bg_color


"""
Fusion rule for 7666fa5d, combo D1/3 + D2/3.

- Input contains both anti-diagonal segments (constant k=r+c) and main-diagonal
  segments (constant b=r-c).
- Every visible segment belongs to a 180-degree symmetric pair around the grid
  center: (r, c) <-> (H-1-r, W-1-c).
- Each same-direction symmetric pair defines one filled band, using the original
  "fill between the two parallel segments" idea in that direction.
- Where an anti-diagonal band and a main-diagonal band overlap, the overlap is
  recolored to a special color.
"""


H = W = 16
BG = 8
FILL = 2
CROSS = 1
LINE_COLORS = [3, 4, 5, 6, 7, 9]
ANTI_K_CHOICES = (4, 6, 8)
MAIN_B_CHOICES = (4, 6, 8)
GEN_RETRY = 400


def _cc8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8)]


def _anti_bounds(k: int) -> tuple[int, int]:
    b_lo = max(-k, k - 2 * (W - 1))
    b_hi = min(2 * (H - 1) - k, k)
    return b_lo, b_hi


def _main_bounds(b: int) -> tuple[int, int]:
    r_lo = max(0, b)
    r_hi = min(H - 1, (W - 1) + b)
    k_lo = 2 * r_lo - b
    k_hi = 2 * r_hi - b
    return k_lo, k_hi


def _anti_cells(k: int, b_min: int, b_max: int) -> list[tuple[int, int]]:
    return [((k + b) // 2, (k - b) // 2) for b in range(b_min, b_max + 1, 2)]


def _main_cells(b: int, k_min: int, k_max: int) -> list[tuple[int, int]]:
    return [((k + b) // 2, (k - b) // 2) for k in range(k_min, k_max + 1, 2)]


def _classify_segment(pts: list[tuple[int, int]]) -> tuple[str, int, int, int] | None:
    ks = {r + c for r, c in pts}
    if len(ks) == 1:
        k = int(next(iter(ks)))
        bs = [r - c for r, c in pts]
        return ("anti", k, int(min(bs)), int(max(bs)))

    bs = {r - c for r, c in pts}
    if len(bs) == 1:
        b = int(next(iter(bs)))
        ks2 = [r + c for r, c in pts]
        return ("main", b, int(min(ks2)), int(max(ks2)))

    return None


def _extract_segments(mask: np.ndarray) -> list[tuple[str, int, int, int]]:
    segs: list[tuple[str, int, int, int]] = []
    for pts in _cc8(mask):
        seg = _classify_segment(pts)
        if seg is not None:
            segs.append(seg)
    segs.sort()
    return segs


def _partner_signature(seg: tuple[str, int, int, int], h: int, w: int) -> tuple[str, int, int, int]:
    orient, axis, lo, hi = seg
    s = h + w - 2
    if orient == "anti":
        return ("anti", s - axis, -hi, -lo)
    return ("main", -axis, s - hi, s - lo)


def _pair_segments(
    segs: list[tuple[str, int, int, int]], h: int, w: int
) -> list[tuple[tuple[str, int, int, int], tuple[str, int, int, int]]]:
    index = {seg: i for i, seg in enumerate(segs)}
    used: set[int] = set()
    pairs: list[tuple[tuple[str, int, int, int], tuple[str, int, int, int]]] = []

    for i, seg in enumerate(segs):
        if i in used:
            continue
        partner = _partner_signature(seg, h, w)
        j = index.get(partner)
        if j is None or j == i or j in used:
            continue
        used.add(i)
        used.add(j)
        if seg <= partner:
            pairs.append((seg, partner))
        else:
            pairs.append((partner, seg))

    pairs.sort()
    return pairs


def _build_fill_masks(
    shape: tuple[int, int],
    pairs: list[tuple[tuple[str, int, int, int], tuple[str, int, int, int]]],
) -> tuple[np.ndarray, np.ndarray]:
    h, w = shape
    rr, cc = np.indices((h, w))
    kk = rr + cc
    bb = rr - cc

    anti_mask = np.zeros((h, w), dtype=bool)
    main_mask = np.zeros((h, w), dtype=bool)

    for left, right in pairs:
        orient = left[0]
        if orient == "anti":
            _, k1, b1_lo, b1_hi = left
            _, k2, b2_lo, b2_hi = right
            b_lo = max(b1_lo, b2_lo)
            b_hi = min(b1_hi, b2_hi)
            if b_lo <= b_hi:
                anti_mask |= (bb >= b_lo) & (bb <= b_hi) & (kk > min(k1, k2)) & (kk < max(k1, k2))
        else:
            _, b1, k1_lo, k1_hi = left
            _, b2, k2_lo, k2_hi = right
            k_lo = max(k1_lo, k2_lo)
            k_hi = min(k1_hi, k2_hi)
            if k_lo <= k_hi:
                main_mask |= (kk >= k_lo) & (kk <= k_hi) & (bb > min(b1, b2)) & (bb < max(b1, b2))

    return anti_mask, main_mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover symmetric diagonal segment pairs of both orientations and fill their bands.

    Input structure:
    - One non-background color draws short diagonal segments.
    - Some components lie on anti-diagonals r+c=k; others lie on main diagonals r-c=b.
    - Every component has a 180-degree symmetric partner around the grid center.

    Rule:
    - Pair each segment with its center-symmetric partner, keeping orientation fixed.
    - Each anti-diagonal pair fills the band between the two segments on the b=r-c axis.
    - Each main-diagonal pair fills the band between the two segments on the k=r+c axis.
    - Cells filled by both direction-families become the special overlap color.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    bg = _bg_color(g)
    non_bg = [c for c in sorted(set(map(int, g.ravel()))) if c != bg]
    assert non_bg, "grid must contain one line color"

    line = max(non_bg, key=lambda c: int(np.sum(g == c)))
    segs = _extract_segments(g == line)
    pairs = _pair_segments(segs, h, w)

    anti_mask, main_mask = _build_fill_masks((h, w), pairs)
    fill_only = (anti_mask ^ main_mask) & (g == bg)
    cross_only = anti_mask & main_mask & (g == bg)

    out = g.copy()
    out[fill_only] = FILL
    out[cross_only] = CROSS
    return out


def _mirror_cells(cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return sorted((H - 1 - r, W - 1 - c) for r, c in cells)


def _touches_existing(cells: list[tuple[int, int]], occupied: set[tuple[int, int]]) -> bool:
    for r, c in cells:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if (rr, cc) in occupied:
                    return True
    return False


def _sample_anti_segment() -> tuple[int, int, int]:
    k = random.choice(ANTI_K_CHOICES)
    b_lo, b_hi = _anti_bounds(k)
    parity = k % 2
    candidates: list[tuple[int, int]] = []

    for b_min in range(b_lo, min(-1, b_hi) + 1):
        if b_min % 2 != parity:
            continue
        for b_max in range(max(1, b_min + 4), b_hi + 1):
            if b_max % 2 != parity:
                continue
            if b_max - b_min < 6:
                continue
            if -b_max < b_lo or -b_min > b_hi:
                continue
            overlap_lo = max(b_min, -b_max)
            overlap_hi = min(b_max, -b_min)
            if overlap_hi - overlap_lo < 3:
                continue
            candidates.append((b_min, b_max))

    b_min, b_max = random.choice(candidates)
    return k, b_min, b_max


def _sample_main_segment() -> tuple[int, int, int]:
    b = random.choice(MAIN_B_CHOICES)
    k_lo, k_hi = _main_bounds(b)
    parity = b % 2
    candidates: list[tuple[int, int]] = []

    for k_min in range(k_lo, 15):
        if k_min % 2 != parity:
            continue
        for k_max in range(max(16, k_min + 6), k_hi + 1):
            if k_max % 2 != parity:
                continue
            if 30 - k_max < k_lo or 30 - k_min > k_hi:
                continue
            overlap_lo = max(k_min, 30 - k_max)
            overlap_hi = min(k_max, 30 - k_min)
            if overlap_hi - overlap_lo < 3:
                continue
            candidates.append((k_min, k_max))

    k_min, k_max = random.choice(candidates)
    return b, k_min, k_max


def _segments_to_grid(
    line: int,
    anti: tuple[int, int, int],
    main_seg: tuple[int, int, int],
) -> tuple[np.ndarray, list[tuple[str, int, int, int]]] | None:
    anti_k, anti_b_min, anti_b_max = anti
    main_b, main_k_min, main_k_max = main_seg

    anti_cells_a = _anti_cells(anti_k, anti_b_min, anti_b_max)
    anti_cells_b = _mirror_cells(anti_cells_a)
    main_cells_a = _main_cells(main_b, main_k_min, main_k_max)
    main_cells_b = _mirror_cells(main_cells_a)

    groups = [anti_cells_a, anti_cells_b, main_cells_a, main_cells_b]
    occupied: set[tuple[int, int]] = set()
    for cells in groups:
        for r, c in cells:
            if not (0 <= r < H and 0 <= c < W):
                return None
        if _touches_existing(cells, occupied):
            return None
        occupied.update(cells)

    grid = np.full((H, W), BG, dtype=int)
    for r, c in occupied:
        grid[r, c] = line

    segs = [
        ("anti", anti_k, anti_b_min, anti_b_max),
        ("anti", 30 - anti_k, -anti_b_max, -anti_b_min),
        ("main", main_b, main_k_min, main_k_max),
        ("main", -main_b, 30 - main_k_max, 30 - main_k_min),
    ]
    segs.sort()
    return grid, segs


def _render_output(
    grid: np.ndarray,
    segs: list[tuple[str, int, int, int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pairs = _pair_segments(segs, H, W)
    anti_mask, main_mask = _build_fill_masks((H, W), pairs)
    out = grid.copy()
    out[(anti_mask ^ main_mask) & (grid == BG)] = FILL
    out[(anti_mask & main_mask) & (grid == BG)] = CROSS
    return out, anti_mask, main_mask


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY):
        line = random.choice(LINE_COLORS)
        anti = _sample_anti_segment()
        main_seg = _sample_main_segment()

        built = _segments_to_grid(line, anti, main_seg)
        if built is None:
            continue
        grid, segs = built
        expected, anti_mask, main_mask = _render_output(grid, segs)

        anti_only = int(np.sum(anti_mask & ~main_mask & (grid == BG)))
        main_only = int(np.sum(main_mask & ~anti_mask & (grid == BG)))
        cross = int(np.sum(anti_mask & main_mask & (grid == BG)))
        if anti_only < 10 or main_only < 10 or cross < 4:
            continue
        if cross > anti_only + main_only:
            continue

        anti_only_out = grid.copy()
        anti_only_out[anti_mask & (grid == BG)] = FILL
        main_only_out = grid.copy()
        main_only_out[main_mask & (grid == BG)] = FILL
        if np.array_equal(expected, anti_only_out) or np.array_equal(expected, main_only_out):
            continue

        inp = grid.copy()
        solved = main(inp)
        if not np.array_equal(solved, expected):
            continue
        return inp

    raise RuntimeError("failed to generate a valid D1/3+D2/3 sample")
