# Combo variant of 7666fa5d (index 8)
# combo_spec: D5/1+D8/2
# Name: 距离优先处理+交替色填充

"""
Design analysis for the combo variant.

1. Concrete behavior trace
   - Preserve the original object class: one line color draws anti-diagonal segments on r+c=k, and fills sweep
     along main diagonals r-c=b.
   - Original task: on one b-line with active k values [8, 12, 20, 24], only consecutive active gaps matter, so
     the output becomes red in (8,12), red in (12,20), red in (20,24).
   - D5 alone would recolor those three visible gaps by left-to-right order: red, blue, red.
   - This combo instead keeps D8's staged processing over all active pairs:
     distance 4 first, then 8, then 12, then 16; the stage colors alternate by round index:
     4 -> red, 8 -> blue, 12 -> red, 16 -> blue.
   - Difference from both parents:
     on the central b-lines the output can still look red/blue/red, but on three-segment b-lines both visible gaps
     become red/red, and on the outermost two-segment b-lines the only visible gap becomes blue because color comes
     from distance round rather than left-to-right position.

2. Complexity calibration
   - Use exactly four anti-diagonals with gaps [4, 8, 4]. This yields four visible distance stages
     {4, 8, 12, 16} while keeping the scene close to the source task.
   - Restrict the fill palette to two colors [2, 1]. That is enough to show alternation without creating an
     arbitrary distance->many-colors lookup.
   - Stagger the two inner segment lengths so all four stages appear in one sample:
     center lines show 4/8, near-edge lines show 12, and edge lines show 16.

3. Implicit-assumption review
   - Connected-component extraction still assumes each visible object is one contiguous anti-diagonal segment.
     The generator preserves that.
   - The old solver assumption "adjacent active pairs are sufficient" is false now. The combo must enumerate every
     active pair on a b-line, then process them by increasing k-distance.
   - The old solver assumption "fill color is always 2" is also false. Color now depends on the global distance
     stage order, alternating red/blue/red/blue...
   - The write-on-background-only condition remains necessary; later, wider stages must not overwrite cells claimed
     by earlier, narrower stages.
   - New solver assumption: the number of overlapping distance stages is small and stable. The generator fixes this
     to four ordered distances so the alternation is learnable.

4. Placement legality
   - All k values stay within [4, 26], same parity, and all b ranges are clipped to legal coordinates derived from
     (r, c) = ((k+b)/2, (k-b)/2).
   - The four segments never overlap because distinct k values map to distinct anti-diagonals.
   - The inner segments are shortened by 2 cells on opposite ends, which preserves legality while exposing different
     active sets across b.

5. Visual observability and cleanliness
   - The output stays tidy: one boundary color plus two fill colors, all arranged in diagonal parallelogram bands.
   - The generator rejects degenerate samples where the combo output matches either the original all-red fill or the
     D5 left-to-right alternation, so the multi-step distance logic is visible from pixels alone.
   - Both fill colors must appear, and every distance stage {4, 8, 12, 16} must paint at least one cell.

6. New solver assumptions guaranteed by the generator
   - Inputs contain exactly one non-background line color, never 1 or 2.
   - There is at least one b-line with four active segments, at least one with three, and at least one with only
     the two outer segments active, so every processing round is observable.
   - Distance-stage parity, not line color or left-right gap order, determines fill color.
"""

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


H = W = 16
BG = 8
FILL_CYCLE = (2, 1)
LINE_COLORS = [3, 4, 5, 6, 7, 9]
K_LO, K_HI = 4, 26
K_GAPS = (4, 8, 4)
TOTAL_K_SPAN = sum(K_GAPS)
MASTER_SPANS = (12, 14)
GEN_RETRY = 400
MIN_FILL_CELLS = 8
EXPECTED_STAGE_DISTANCES = (4, 8, 12, 16)


def _cc8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8)]


def _b_bounds_for_k(k: int) -> tuple[int, int]:
    b_lo = max(-k, k - 2 * (W - 1))
    b_hi = min(2 * (H - 1) - k, k)
    return b_lo, b_hi


def _seg_cells(k: int, b_min: int, b_max: int) -> list[tuple[int, int]]:
    return [((k + b) // 2, (k - b) // 2) for b in range(b_min, b_max + 1, 2)]


def _extract_segments(mask: np.ndarray) -> list[tuple[int, int, int]]:
    segs: list[tuple[int, int, int]] = []
    for pts in _cc8(mask):
        ks = {r + c for r, c in pts}
        if len(ks) != 1:
            continue
        k = int(next(iter(ks)))
        bs = [r - c for r, c in pts]
        segs.append((k, int(min(bs)), int(max(bs))))
    segs.sort(key=lambda t: (t[0], t[1], t[2]))
    return segs


def _distance_palette(segs: list[tuple[int, int, int]]) -> dict[int, int]:
    distances = sorted(
        {
            segs[j][0] - segs[i][0]
            for i in range(len(segs))
            for j in range(i + 1, len(segs))
            if max(segs[i][1], segs[j][1]) <= min(segs[i][2], segs[j][2])
        }
    )
    assert distances, "expected at least one overlapping segment pair"
    return {dist: FILL_CYCLE[idx % len(FILL_CYCLE)] for idx, dist in enumerate(distances)}


def _apply_combo_fill(
    grid: np.ndarray, segs: list[tuple[int, int, int]]
) -> tuple[np.ndarray, dict[int, int], dict[int, int]]:
    out = grid.copy()
    dist_to_color = _distance_palette(segs)
    stage_counts = {dist: 0 for dist in dist_to_color}
    color_counts = {color: 0 for color in FILL_CYCLE}

    for b in range(-(W - 1), H):
        active = sorted([s for s in segs if s[1] <= b <= s[2]], key=lambda t: t[0])
        if len(active) < 2:
            continue

        pairs_by_dist: dict[int, list[tuple[int, int]]] = {}
        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                dist = active[j][0] - active[i][0]
                pairs_by_dist.setdefault(dist, []).append((active[i][0], active[j][0]))

        r_min = max(0, b)
        r_max = min(H - 1, (W - 1) + b)
        for dist in sorted(pairs_by_dist):
            color = dist_to_color[dist]
            for k_left, k_right in pairs_by_dist[dist]:
                for r in range(r_min, r_max + 1):
                    c = r - b
                    k = r + c
                    if k_left < k < k_right and out[r, c] == BG:
                        out[r, c] = color
                        stage_counts[dist] += 1
                        color_counts[color] += 1

    return out, stage_counts, color_counts


def _apply_original_fill(grid: np.ndarray, segs: list[tuple[int, int, int]]) -> np.ndarray:
    out = grid.copy()
    for b in range(-(W - 1), H):
        active = sorted([s for s in segs if s[1] <= b <= s[2]], key=lambda t: t[0])
        if len(active) < 2:
            continue
        ks = [t[0] for t in active]
        r_min = max(0, b)
        r_max = min(H - 1, (W - 1) + b)
        for k_left, k_right in zip(ks, ks[1:]):
            for r in range(r_min, r_max + 1):
                c = r - b
                k = r + c
                if k_left < k < k_right and out[r, c] == BG:
                    out[r, c] = 2
    return out


def _apply_gap_alternation_fill(grid: np.ndarray, segs: list[tuple[int, int, int]]) -> np.ndarray:
    out = grid.copy()
    for b in range(-(W - 1), H):
        active = sorted([s for s in segs if s[1] <= b <= s[2]], key=lambda t: t[0])
        if len(active) < 2:
            continue
        ks = [t[0] for t in active]
        r_min = max(0, b)
        r_max = min(H - 1, (W - 1) + b)
        for idx, (k_left, k_right) in enumerate(zip(ks, ks[1:])):
            color = FILL_CYCLE[idx % len(FILL_CYCLE)]
            for r in range(r_min, r_max + 1):
                c = r - b
                k = r + c
                if k_left < k < k_right and out[r, c] == BG:
                    out[r, c] = color
    return out


def _line_gap_color_sequences_by_active_count(
    segs: list[tuple[int, int, int]]
) -> dict[int, set[tuple[int, ...]]]:
    dist_to_color = _distance_palette(segs)
    signatures: dict[int, set[tuple[int, ...]]] = {}
    for b in range(-(W - 1), H):
        active = sorted([s for s in segs if s[1] <= b <= s[2]], key=lambda t: t[0])
        if len(active) < 2:
            continue
        signature = tuple(dist_to_color[active[i + 1][0] - active[i][0]] for i in range(len(active) - 1))
        signatures.setdefault(len(active), set()).add(signature)
    return signatures


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the diagonal bands between anti-diagonal segments in distance-ordered rounds.

    Input structure:
    - Background is the most common color.
    - One other color forms 8-connected segments, each confined to a single anti-diagonal r+c=k.
    - Several such segments occupy parallel anti-diagonals, with staggered lengths so different pair distances can
      appear on different main diagonals r-c=b.

    Rule:
    - Recover every segment as (k, b_min, b_max). For each b-line, collect all active segments and consider every
      active segment pair, not just neighboring pairs.
    - Group those pairs by their anti-diagonal distance delta_k, process smaller distances first, and paint only
      cells that are still background.
    - The color alternates by distance round over the whole puzzle: nearest distance stage red (2), next stage
      blue (1), then red again, then blue again.

    Key insight:
    - The output is controlled by two coupled patterns: nesting order comes from segment distance, while color comes
      from the parity of that distance-processing round. A farther visible gap can therefore flip color relative to
      a nearer gap even when they appear left-to-right on the same diagonal.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    non_bg = [c for c in sorted(set(map(int, g.ravel()))) if c != bg]
    assert non_bg, "grid must contain at least one non-background color"

    line = max(non_bg, key=lambda c: int(np.sum(g == c)))
    segs = _extract_segments(g == line)
    assert segs, "expected at least one anti-diagonal segment"

    out, _, _ = _apply_combo_fill(g, segs)
    return out


def _choose_layout() -> tuple[list[int], int, int] | None:
    parity = random.randint(0, 1)
    candidates: list[tuple[list[int], int, int]] = []

    for k0 in range(K_LO, K_HI - TOTAL_K_SPAN + 1):
        if k0 % 2 != parity:
            continue
        ks = [k0]
        for gap in K_GAPS:
            ks.append(ks[-1] + gap)

        common_lo = max(_b_bounds_for_k(k)[0] for k in ks)
        common_hi = min(_b_bounds_for_k(k)[1] for k in ks)
        if common_lo % 2 != parity:
            common_lo += 1
        if common_hi % 2 != parity:
            common_hi -= 1
        if common_lo > common_hi:
            continue

        for master_span in MASTER_SPANS:
            start_max = common_hi - master_span
            if start_max < common_lo:
                continue
            for master_lo in range(common_lo, start_max + 1, 2):
                candidates.append((ks, master_lo, master_span))

    if not candidates:
        return None
    return random.choice(candidates)


def _build_segments(ks: list[int], master_lo: int, master_span: int) -> list[tuple[int, int, int]]:
    hi = master_lo + master_span
    shorter_low = (master_lo + 2, hi - 4)
    shorter_high = (master_lo + 4, hi - 2)
    if random.random() < 0.5:
        inner_ranges = (shorter_low, shorter_high)
    else:
        inner_ranges = (shorter_high, shorter_low)

    return [
        (ks[0], master_lo, hi),
        (ks[1], inner_ranges[0][0], inner_ranges[0][1]),
        (ks[2], inner_ranges[1][0], inner_ranges[1][1]),
        (ks[3], master_lo, hi),
    ]


def _valid_segments(segs: list[tuple[int, int, int]]) -> bool:
    for k, b_min, b_max in segs:
        if b_min > b_max:
            return False
        lo, hi = _b_bounds_for_k(k)
        if b_min < lo or b_max > hi:
            return False
        if (b_min - k) % 2 or (b_max - k) % 2:
            return False
        if len(_seg_cells(k, b_min, b_max)) < 2:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    layout = _choose_layout()
    if layout is None:
        return None

    ks, master_lo, master_span = layout
    segs = _build_segments(ks, master_lo, master_span)
    if not _valid_segments(segs):
        return None

    line = random.choice(LINE_COLORS)
    grid = np.full((H, W), BG, dtype=int)
    for k, b_min, b_max in segs:
        for r, c in _seg_cells(k, b_min, b_max):
            if not (0 <= r < H and 0 <= c < W):
                return None
            if grid[r, c] != BG:
                return None
            grid[r, c] = line

    combo_out, stage_counts, color_counts = _apply_combo_fill(grid, segs)
    if sum(color_counts.values()) < MIN_FILL_CELLS:
        return None
    if tuple(sorted(stage_counts)) != EXPECTED_STAGE_DISTANCES:
        return None
    if any(stage_counts[dist] == 0 for dist in EXPECTED_STAGE_DISTANCES):
        return None
    if any(color_counts[color] == 0 for color in FILL_CYCLE):
        return None

    line_signatures = _line_gap_color_sequences_by_active_count(segs)
    if (2, 1, 2) not in line_signatures.get(4, set()):
        return None
    if (2, 2) not in line_signatures.get(3, set()):
        return None
    if (1,) not in line_signatures.get(2, set()):
        return None

    original_out = _apply_original_fill(grid, segs)
    d5_out = _apply_gap_alternation_fill(grid, segs)
    if np.array_equal(combo_out, original_out):
        return None
    if np.array_equal(combo_out, d5_out):
        return None

    if not np.array_equal(main(grid), combo_out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY):
        res = _gen_attempt()
        if res is not None:
            return res
    raise RuntimeError("Failed to generate combo variant input")
