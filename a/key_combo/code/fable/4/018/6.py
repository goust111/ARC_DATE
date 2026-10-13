from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path
from typing import Literal

import numpy as np

if str(Path("/home/lixinyu/snap/new/solvers")) not in sys.path:
    sys.path.append(str(Path("/home/lixinyu/snap/new/solvers")))

from arc_utils import Color, get_bg  # type: ignore  # noqa: E402

PairMode = Literal["translate", "mirror"]
SplitAxis = Literal["vertical", "horizontal"]

BACKGROUND = Color.BLACK
PALETTE = [int(c) for c in Color.NOT_BLACK]

GEN_ATTEMPTS = 500
MIN_HALF = 4
MAX_HALF = 12
MIN_AXIS_DIM = 4
MAX_AXIS_DIM = 22
MAX_DAMAGE = 4


def _pair_iterator(
    H: int,
    W: int,
    orientation: SplitAxis,
    mode: PairMode,
    source_is_first: bool,
) -> list[tuple[int, int, int, int]]:
    """
    Enumerate (src_r, src_c, dst_r, dst_c) coordinate pairs for a given split config.
    source_is_first means left/top half is the source side.
    """
    pairs: list[tuple[int, int, int, int]] = []
    if orientation == "vertical":
        assert W % 2 == 0
        half_w = W // 2
        for r in range(H):
            for lc in range(half_w):
                src_c = lc if source_is_first else lc + half_w
                src_r = r
                if mode == "translate":
                    dst_c = lc + half_w if source_is_first else lc
                else:
                    dst_c = W - 1 - src_c
                dst_r = r
                pairs.append((src_r, src_c, dst_r, dst_c))
        return pairs

    assert orientation == "horizontal"
    assert H % 2 == 0
    half_h = H // 2
    for lr in range(half_h):
        for c in range(W):
            src_r = lr if source_is_first else lr + half_h
            src_c = c
            if mode == "translate":
                dst_r = lr + half_h if source_is_first else lr
            else:
                dst_r = H - 1 - src_r
            dst_c = c
            pairs.append((src_r, src_c, dst_r, dst_c))
    return pairs


def _evaluate_candidate(
    g: np.ndarray,
    orientation: SplitAxis,
    mode: PairMode,
    source_is_first: bool,
    marker_color: int,
) -> tuple[int, int, bool, bool]:
    """
    Return (mismatches, compared_cells, src_has_marker, target_has_marker) for one hypothesis.
    Comparisons at source marker cells are skipped, as they are repaired backward.
    """
    H, W = g.shape
    mismatches = 0
    compared = 0
    src_marker_count = 0
    target_marker_count = 0

    for sr, sc, tr, tc in _pair_iterator(H, W, orientation, mode, source_is_first):
        src_v = int(g[sr, sc])
        dst_v = int(g[tr, tc])
        if src_v == marker_color:
            src_marker_count += 1
            if dst_v == marker_color:
                target_marker_count += 1
            continue
        if dst_v == marker_color:
            target_marker_count += 1
        compared += 1
        if src_v != dst_v:
            mismatches += 1

    return mismatches, compared, src_marker_count == 1, target_marker_count == 0


def _infer_transform(
    g: np.ndarray,
) -> tuple[SplitAxis, PairMode, bool, int] | None:
    """
    Recover the unique split axis + copy mode + source side + marker color.
    """
    g = np.asarray(g, dtype=int)
    bg = get_bg(g)
    counts = Counter(map(int, g.ravel()))
    marker_candidates = [color for color, cnt in counts.items() if color != bg and cnt == 1]
    if len(marker_candidates) != 1:
        return None
    marker_color = int(marker_candidates[0])

    candidates: list[tuple[int, SplitAxis, PairMode, bool]] = []
    for orientation in ("vertical", "horizontal"):
        if orientation == "vertical" and g.shape[1] % 2 != 0:
            continue
        if orientation == "horizontal" and g.shape[0] % 2 != 0:
            continue
        half_area = g.shape[0] * (g.shape[1] // 2) if orientation == "vertical" else (g.shape[0] // 2) * g.shape[1]
        max_mismatch = max(3, half_area // 4 + 1)

        for mode in ("translate", "mirror"):
            for source_is_first in (True, False):
                mismatches, compared, src_has_marker, target_has_marker = _evaluate_candidate(
                    g, orientation, mode, source_is_first, marker_color
                )
                if not src_has_marker:
                    continue
                if target_has_marker and mismatches <= max_mismatch:
                    candidates.append((mismatches, orientation, mode, source_is_first))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0])
    if len(candidates) > 1 and candidates[0][0] == candidates[1][0]:
        # ambiguous hypothesis
        return None
    _, orientation, mode, source_is_first = candidates[0]
    return orientation, mode, source_is_first, marker_color


def _repair_grid(g: np.ndarray, cfg: tuple[SplitAxis, PairMode, bool, int]) -> np.ndarray:
    orientation, mode, source_is_first, marker_color = cfg
    H, W = g.shape
    out = g.copy()
    for sr, sc, tr, tc in _pair_iterator(H, W, orientation, mode, source_is_first):
        src_v = int(g[sr, sc])
        dst_v = int(g[tr, tc])
        if src_v == marker_color:
            out[sr, sc] = dst_v
        else:
            out[tr, tc] = src_v
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore the damaged half of a doubled pattern by copying from the trusted marker half.

    Input structure
    - The grid is two equal halves (split vertically or horizontally) of one pattern.
    - One half is a trusted copy containing one unique marker pixel whose color does not
      appear in the original pattern.
    - The opposite half is the same pattern transformed by translation or mirror and then
      corrupted at a few cells.

    Rule:
    - Detect the split axis and copy mode that makes both halves mostly correspond when
      ignoring the single marker and corrupted pixels.
    - Identify the marker-containing half as the trusted source.
    - Rewrite every opposite-half cell with the transformed source cell color.
    - Reverse the information flow only at the marker location itself and paint that source
      marker cell from its opposite-side counterpart.

    Key insight:
    - The marker is both the hint selecting the trusted source half and the only location that
      should not be overwritten with a direct one-way copy.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) > 0

    cfg = _infer_transform(g)
    assert cfg is not None, "input must contain a unique marker and a consistent mirrored/translated pairing"
    return _repair_grid(g, cfg)


def _random_pattern(height: int, width: int, bg: int) -> np.ndarray:
    pattern = np.full((height, width), bg, dtype=int)
    num_colors = random.randint(1, 2)
    colors = random.sample(PALETTE, num_colors)

    # Large base block.
    bh = random.randint(max(2, height // 2), max(2, height - 1))
    bw = random.randint(max(2, width // 2), max(2, width - 1))
    r0 = random.randint(0, max(0, height - bh))
    c0 = random.randint(0, max(0, width - bw))
    pattern[r0 : r0 + bh, c0 : c0 + bw] = colors[0]

    # Extra asymmetrical fragments.
    for _ in range(random.randint(1, 3)):
        col = random.choice(colors)
        if random.random() < 0.5:
            r = random.randint(0, height - 1)
            if width >= 2:
                l = random.randint(2, width)
            else:
                l = 1
            c = random.randint(0, width - l)
            pattern[r, c : c + l] = col
        else:
            c = random.randint(0, width - 1)
            if height >= 2:
                l = random.randint(2, height)
            else:
                l = 1
            r = random.randint(0, height - l)
            pattern[r : r + l, c] = col

    # Ensure every chosen color appears with more than one cell.
    for color in colors:
        if np.count_nonzero(pattern == color) < 2:
            rr = random.randint(0, height - 1)
            cc = random.randint(0, width - 1)
            pattern[rr, cc] = color
            if width >= 2:
                cc = (cc + 1) % width
                pattern[rr, cc] = color
            else:
                rr = (rr + 1) % height
                pattern[rr, cc] = color

    return pattern


def _apply_damage(
    grid: np.ndarray,
    cfg: tuple[SplitAxis, PairMode, bool],
    marker_color: int,
    marker_r: int,
    marker_c: int,
) -> np.ndarray:
    H, W = grid.shape
    orientation, mode, source_is_first = cfg
    pairs = _pair_iterator(H, W, orientation, mode, source_is_first)
    marker_target = None
    target_positions: list[tuple[int, int, int, int]] = []
    for sr, sc, tr, tc in pairs:
        if sr == marker_r and sc == marker_c:
            marker_target = (tr, tc)
        else:
            target_positions.append((tr, tc, sr, sc))

    assert marker_target is not None
    max_damage = min(MAX_DAMAGE, len(target_positions))
    num_damage = random.randint(1, max_damage)
    selected = random.sample(target_positions, num_damage)

    source_colors = {int(v) for _, _, sr, sc in pairs for v in [grid[sr, sc]] if int(v) != marker_color}
    for tr, tc, sr, sc in selected:
        source_value = int(grid[sr, sc])
        noise_choices = [c for c in source_colors if c != source_value and c != marker_color]
        if not noise_choices:
            noise_choices = [BACKGROUND] if BACKGROUND != source_value and BACKGROUND != marker_color else []
            if not noise_choices:
                noise_choices = [c for c in PALETTE if c != source_value and c != marker_color]
        grid[tr, tc] = random.choice(noise_choices)

    return grid


def _select_marker_color(pattern: np.ndarray, bg: int) -> int:
    used = {int(v) for v in pattern.ravel() if int(v) != bg}
    avail = [c for c in PALETTE if c not in used]
    if not avail:
        raise RuntimeError("marker color unavailable")
    return random.choice(avail)


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        orientation = random.choice(("vertical", "horizontal"))
        mode = random.choice(("translate", "mirror"))
        source_is_first = random.choice((True, False))
        if orientation == "vertical":
            half_w = random.randint(MIN_HALF, MAX_HALF)
            height = random.randint(MIN_AXIS_DIM, MAX_AXIS_DIM)
            width = half_w * 2
            H, W = height, width
            base_pattern = _random_pattern(H, half_w, BACKGROUND)
            target = base_pattern if mode == "translate" else np.fliplr(base_pattern)
            if source_is_first:
                source = base_pattern.copy()
                dest = target.copy()
                grid = np.hstack([source, dest])
            else:
                source = target.copy()
                dest = base_pattern.copy()
                grid = np.hstack([dest, source])
        else:
            half_h = random.randint(MIN_HALF, MAX_HALF)
            width = random.randint(MIN_AXIS_DIM, MAX_AXIS_DIM)
            height = half_h * 2
            H, W = height, width
            base_pattern = _random_pattern(half_h, W, BACKGROUND)
            target = base_pattern if mode == "translate" else np.flipud(base_pattern)
            if source_is_first:
                source = base_pattern.copy()
                dest = target.copy()
                grid = np.vstack([source, dest])
            else:
                source = target.copy()
                dest = base_pattern.copy()
                grid = np.vstack([dest, source])

        marker_color = _select_marker_color(source, BACKGROUND)
        source_pixels = np.argwhere(source != BACKGROUND)
        if len(source_pixels) == 0:
            continue
        sr_local, sc_local = random.choice(source_pixels)
        if source_is_first:
            if orientation == "vertical":
                sr, sc = int(sr_local), int(sc_local)
            else:
                sr, sc = int(sr_local), int(sc_local)
        else:
            if orientation == "vertical":
                sr, sc = int(sr_local), int(sc_local) + W // 2
            else:
                sr, sc = int(sr_local) + H // 2, int(sc_local)

        if orientation == "vertical":
            source_half_col = sc < W // 2
        else:
            source_half_col = sr < H // 2
        if source_half_col != source_is_first:
            continue

        grid[sr, sc] = marker_color

        grid = _apply_damage(grid, (orientation, mode, source_is_first), marker_color, sr, sc)

        # Force the marker color remain unique and outside pattern semantics.
        if np.count_nonzero(grid == marker_color) != 1:
            continue

        cfg = _infer_transform(grid)
        if cfg != (orientation, mode, source_is_first, marker_color):
            continue
        return grid

    raise RuntimeError("failed to generate a valid 6 input")
