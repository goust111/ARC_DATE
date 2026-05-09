# Variant of 1818057f
# Dimension: D4 — 前景色数量与背景色关系 (meta)
# Change: 恰好1种前景色(黄色4)在1种背景色(红色2)上，前景密度26-41%... → 多种背景色分区

from __future__ import annotations

from collections import Counter

import numpy as np


# Design analysis for the D4 variant
# a) Concrete behavior and trace:
#    - The single red background is replaced by 2 or 3 vertical background bands.
#      Each band is a solid color chosen from a small fixed palette, while the
#      foreground is still a single yellow color.
#    - '+' detection stays local and unchanged: a center pixel is valid iff its
#      4 orthogonal neighbors are also yellow.
#    - The new behavior happens at recoloring time. Example:
#        Original: a yellow plus on red background -> all 5 plus pixels become teal.
#        Variant:  the same yellow plus shape inside a blue band -> the 5 pixels
#        become green instead; inside a grey band -> orange instead.
#      The difference is not in the geometry of the plus, but in the recovered
#      background partition under that plus.
# b) Complexity calibration:
#    - Background colors are limited to 2 or 3 repeated bands, not arbitrary
#      patchwork. This keeps the partition visually obvious in 2-4 examples.
#    - Target colors are limited to the same small repeated mapping:
#        blue(1)->green(3), red(2)->teal(8), grey(5)->orange(7)
#      so the color rule is learnable rather than combinatorial.
#    - Every region gets at least one plus, so the new control variable
#      (background region) is always exercised.
# c) Implicit-assumption review of the original main():
#    - `bg = get_bg(g)` assumed exactly one global background color. That is
#      invalid now because several background colors coexist.
#    - `colors = ... - {bg}; fg = colors[0]` assumed every non-bg color is
#      foreground. That also breaks now because other non-fg colors are just
#      other background bands.
#    - `out[...] = 8` assumed one universal output color. That is the main line
#      that must change: the target color now depends on the recovered band color.
#    - The interior scan and the local 4-neighbor plus test remain valid.
# d) Placement legality:
#    - Bands cover the full grid width without overlap.
#    - Plus centers are kept at least one cell away from band edges, and at least
#      two rows/columns away from the outer grid border, so the 5-cell cross stays
#      inside one band and inside the grid.
#    - Centers stay L1-distance >= 4 apart, so plus footprints do not overlap.
# e) Visual observability:
#    - A human can first notice the large vertical color bands, then notice that
#      identical yellow pluses change to different colors depending on the band.
#    - Because each sample uses multiple bands and multiple recolored pluses, the
#      mapping is supported by repeated evidence rather than a one-off accident.
# f) Visual cleanliness:
#    - Only wide rectangular bands are used; no nested partitions or mixed tilings.
#    - Foreground density stays in the original 26-41% range, but the majority of
#      cells remain background, so the bands are still easy to read.
# g) Color-change inferability:
#    - The mapping is tied to visible region color, not to hidden ids or ordering.
#    - Target colors are disjoint from the background palette, so recolored pluses
#      remain salient instead of blending into a neighboring band.
# 9) New main() assumptions and generator guarantees:
#    - main() assumes the partition is vertical, meaning each column belongs to a
#      single background band once yellow foreground cells are ignored.
#    - main() assumes the only non-band input color is the shared foreground color.
#    - generate_input() guarantees both by using only vertical bands from a fixed
#      palette and keeping the foreground color fixed to yellow(4).

FG = 4
PLUS_OFFSETS = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
BG_TO_OUTPUT = {
    1: 3,  # blue band -> green plus
    2: 8,  # red band -> teal plus (preserves the original branch)
    5: 7,  # grey band -> orange plus
}


def _infer_foreground(colors: list[int]) -> int | None:
    candidates = [c for c in colors if c not in BG_TO_OUTPUT]
    if FG in candidates:
        return FG
    if len(candidates) == 1:
        return int(candidates[0])
    return None


def _infer_column_backgrounds(grid: np.ndarray, fg: int) -> np.ndarray:
    """Recover the background band color for each column by ignoring foreground."""
    H, W = grid.shape
    col_bg = np.full(W, -1, dtype=int)
    for c in range(W):
        vals = [int(grid[r, c]) for r in range(H) if int(grid[r, c]) != fg]
        if vals:
            col_bg[c] = Counter(vals).most_common(1)[0][0]

    known = np.where(col_bg != -1)[0]
    if len(known) == 0:
        raise ValueError("no background columns available")

    first = int(known[0])
    col_bg[:first] = col_bg[first]
    for left, right in zip(known, known[1:]):
        left = int(left)
        right = int(right)
        if right - left <= 1:
            continue
        mid = (left + right) // 2
        col_bg[left + 1 : mid + 1] = col_bg[left]
        col_bg[mid + 1 : right] = col_bg[right]
    last = int(known[-1])
    col_bg[last + 1 :] = col_bg[last]
    return col_bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect thin '+' shapes in a shared foreground color laid over several background bands, and recolor each one by its band.

    Input structure:
    - The background is partitioned into 2 or 3 vertical color bands rather than a single global bg.
    - One foreground color (yellow) forms mixed shapes on top of those bands, and some local 5-cell '+' motifs are embedded inside the foreground.

    Rule:
    - A '+' center is any foreground pixel whose 4 orthogonal neighbors are also foreground.
    - Infer the background color of the center column by ignoring foreground cells, then map that band color to its output color.
    - Recolor the center and its 4 orthogonal neighbors with that band-specific target color; leave everything else unchanged.

    Key insight: The local '+' geometry is unchanged from the original task; the new challenge is recovering which background region each plus belongs to before choosing the recolor.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()

    colors = sorted(set(map(int, g.flatten())))
    fg = _infer_foreground(colors)
    if fg is None:
        return out

    col_bg = _infer_column_backgrounds(g, fg)

    H, W = g.shape
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            if int(g[r, c]) != fg:
                continue
            if all(int(g[r + dr, c + dc]) == fg for dr, dc in PLUS_OFFSETS[1:]):
                target = BG_TO_OUTPUT.get(int(col_bg[c]))
                if target is None:
                    continue
                for dr, dc in PLUS_OFFSETS:
                    out[r + dr, c + dc] = target
    return out


def _would_create_plus(grid: np.ndarray, r: int, c: int, fg: int) -> bool:
    """Check if setting (r,c) to fg would create any new plus center at (r,c) or its neighbors."""
    H, W = grid.shape
    candidates = [(r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
    for cr, cc in candidates:
        if cr < 1 or cr >= H - 1 or cc < 1 or cc >= W - 1:
            continue
        all_fg = True
        for dr, dc in PLUS_OFFSETS:
            nr, nc = cr + dr, cc + dc
            if nr < 0 or nr >= H or nc < 0 or nc >= W:
                all_fg = False
                break
            val = fg if (nr, nc) == (r, c) else int(grid[nr, nc])
            if val != fg:
                all_fg = False
                break
        if all_fg:
            return True
    return False


def _build_regions(rng: np.random.Generator, width: int) -> list[tuple[int, int, int]]:
    n_regions = 2
    if width >= 14 and rng.random() < 0.6:
        n_regions = 3

    widths = [4] * n_regions
    for _ in range(width - 4 * n_regions):
        widths[int(rng.integers(0, n_regions))] += 1

    colors = list(map(int, rng.choice(list(BG_TO_OUTPUT.keys()), size=n_regions, replace=False)))
    regions = []
    c0 = 0
    for band_w, bg in zip(widths, colors):
        c1 = c0 + int(band_w) - 1
        regions.append((c0, c1, int(bg)))
        c0 = c1 + 1
    return regions


def _region_candidates(H: int, W: int, c0: int, c1: int) -> list[tuple[int, int]]:
    low_c = max(c0 + 1, 2)
    high_c = min(c1 - 1, W - 3)
    if low_c > high_c:
        return []
    return [(r, c) for r in range(2, H - 2) for c in range(low_c, high_c + 1)]


def _find_region(regions: list[tuple[int, int, int]], c: int) -> tuple[int, int, int]:
    for region in regions:
        c0, c1, _ = region
        if c0 <= c <= c1:
            return region
    raise ValueError(f"column {c} is outside every region")


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    H = int(rng.integers(10, 23))
    W = int(rng.integers(10, 23))

    regions = _build_regions(rng, W)
    grid = np.zeros((H, W), dtype=int)
    for c0, c1, bg in regions:
        grid[:, c0 : c1 + 1] = bg

    target_centers = int(rng.integers(len(regions), min(10, len(regions) * 3 + 1)))
    centers: list[tuple[int, int]] = []
    center_bgs: dict[tuple[int, int], int] = {}

    for c0, c1, bg in regions:
        candidates = _region_candidates(H, W, c0, c1)
        if not candidates:
            return None
        rng.shuffle(candidates)
        placed = False
        for cr, cc in candidates:
            if all(abs(cr - er) + abs(cc - ec) >= 4 for er, ec in centers):
                centers.append((cr, cc))
                center_bgs[(cr, cc)] = bg
                placed = True
                break
        if not placed:
            return None

    all_candidates: list[tuple[int, int, int]] = []
    for c0, c1, bg in regions:
        for cr, cc in _region_candidates(H, W, c0, c1):
            all_candidates.append((cr, cc, bg))
    rng.shuffle(all_candidates)
    for cr, cc, bg in all_candidates:
        if len(centers) >= target_centers:
            break
        if (cr, cc) in center_bgs:
            continue
        if all(abs(cr - er) + abs(cc - ec) >= 4 for er, ec in centers):
            centers.append((cr, cc))
            center_bgs[(cr, cc)] = bg

    if len(centers) < len(regions):
        return None

    for cr, cc in centers:
        for dr, dc in PLUS_OFFSETS:
            grid[cr + dr, cc + dc] = FG

    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    for cr, cc in centers:
        c0, c1, _ = _find_region(regions, cc)
        n_ext = int(rng.integers(1, 3))
        chosen = list(map(int, rng.choice(4, size=n_ext, replace=False)))
        for di in chosen:
            dr, dc = dirs[di]
            ext_len = int(rng.integers(1, 4))
            tip_r, tip_c = cr + dr, cc + dc
            for step in range(1, ext_len + 1):
                nr = tip_r + dr * step
                nc = tip_c + dc * step
                if nr < 0 or nr >= H or nc < 0 or nc >= W:
                    break
                if nc < c0 or nc > c1:
                    break
                if grid[nr, nc] == FG:
                    break
                grid[nr, nc] = FG

    total = H * W
    target_density = float(rng.uniform(0.26, 0.41))
    target_fg = int(total * target_density)
    current_fg = int(np.sum(grid == FG))

    bg_positions = list(zip(*np.where(grid != FG)))
    rng.shuffle(bg_positions)
    for r, c in bg_positions:
        if current_fg >= target_fg:
            break
        if not _would_create_plus(grid, int(r), int(c), FG):
            grid[int(r), int(c)] = FG
            current_fg += 1

    expected = grid.copy()
    used_targets = set()
    for (cr, cc), bg in center_bgs.items():
        target = BG_TO_OUTPUT[bg]
        used_targets.add(target)
        for dr, dc in PLUS_OFFSETS:
            expected[cr + dr, cc + dc] = target

    if len(used_targets) < 2:
        return None

    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 400 attempts")
