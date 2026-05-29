# Variant of 269e22fb
# Dimension: D1 — 匹配条件：裁剪窗口唯一性匹配方式 (match_cond)
# Change: 通过穷举8个二面体变体x2种前景/背景极性x所有偏移位置，找到唯一匹配裁剪窗口的(变体, 偏移, 极性)组合... → 颜色编码极性提示

from __future__ import annotations

import numpy as np

from arc_utils import Color, colors_present


# Design analysis
# ----------------
# a) Concrete behavior and trace:
# - The 20x20 binary mother pattern and its 8 dihedral orientations stay unchanged.
# - The input is still a rectangular crop from one oriented pattern, but it now uses
#   three colors: one color marks cells that are known foreground, one color marks
#   cells that are known background, and one color marks cells hidden behind an
#   occluder. The output is the full 20x20 oriented pattern painted only with the
#   foreground/background colors from the input.
# - Small trace: in the original task, a 12x10 crop had no hidden cells, so matching
#   required exact equality with one candidate window. In this variant, a 12x10 crop
#   may have a 4x3 hidden rectangle inside it. The solver must ignore those 12 cells,
#   only demand that visible foreground cells land on pattern 1s and visible background
#   cells land on pattern 0s, then fill the missing region after the unique orientation
#   and offset are identified.
#
# b) Complexity calibration:
# - If the hidden region were too small, the task would visually collapse back toward
#   the original exact-crop matching problem.
# - If it were too large, multiple orientations/offsets could satisfy the visible cells.
# - The generator therefore keeps crop sizes fairly large (10-16 by 10-12) and uses a
#   single rectangular occluder whose area is roughly 18%-35% of the crop. This keeps
#   the partial-information behavior obvious while preserving uniqueness after retries.
#
# c) Implicit assumptions in the original main():
# - `assert len(colors) == 2` no longer holds; the input now has exactly 3 colors.
# - `visible = g == fg` assumed every non-fg pixel was known background. That breaks
#   once some cells are "unknown", so the new solver keeps separate masks for known
#   foreground and known background and ignores unknown cells during matching.
# - `np.array_equal(window, visible)` assumed a fully observed crop. The new matching
#   condition is partial consistency on known pixels only.
# - The original code searched both polarities because the two colors were unlabeled.
#   Here the colors encode roles on visible cells; the solver still tries role
#   assignments for robustness, but uniqueness is now driven by partial evidence rather
#   than exact two-color equality.
#
# d) Placement legality:
# - The crop is always chosen inside the 20x20 pattern bounds.
# - The occluder is an axis-aligned rectangle fully inside the crop, so there are no
#   overlap or out-of-bounds issues beyond the intended overwrite by the unknown color.
# - The generator rejects masks that hide everything, reveal only one known class, or
#   fail to cover both underlying pattern states.
#
# e) Visual observability:
# - Across 2-4 examples, a human sees the same type of irregular 20x20 motif completion
#   task, but now every input has a solid third-color patch inside the crop. The output
#   removes that patch and completes the full motif using the two non-occluder colors.
# - Because the hidden rectangle moves and changes size, the "unknown" interpretation is
#   visible as a rule and not a fixed decorative artifact.
#
# f) Visual cleanliness:
# - Using one rectangular occluder keeps the crop readable: there are only three layers
#   of meaning in the input, and the hidden region is contiguous rather than noisy.
#
# g) Color inference:
# - Only three colors are present. Two of them reappear in the output and keep their
#   roles as motif foreground/background; the third exists only in the input and serves
#   as the unknown marker. This is a small, repeatable color change that can be inferred
#   from examples without relying on color ids.
#
# h) New solver assumptions and generator guarantees:
# - main() assumes there is exactly one `(orientation, offset, color-role assignment)`
#   consistent with all known cells. generate_input() enforces this by rejection.
# - main() assumes the unknown color is actually used. generate_input() always places a
#   non-empty occluder.
# - main() assumes both known foreground and known background evidence exist. The
#   generator rejects any crop where one of those visible classes disappears.


_BASE_PATTERN = (
    "11000000000000011000",
    "11111100000000011000",
    "11111110000000011100",
    "11000111000000011100",
    "00000011100000111110",
    "00000001110000100110",
    "00000000110011100110",
    "00000000010010100110",
    "00000000011110100110",
    "01111111110110100110",
    "01000000011110100110",
    "01111111110010100110",
    "01010000010011100110",
    "11101111110000100110",
    "01011000010000111110",
    "11101000010001101100",
    "01011000010001000110",
    "11101000010001100011",
    "01011000101000101110",
    "11100001000100111000",
)


def _pattern_mask() -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in _BASE_PATTERN], dtype=bool)


def _dihedral_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    for k in range(4):
        rotated = np.rot90(mask, k)
        variants.append(rotated)
        variants.append(np.fliplr(rotated))
    return variants


def _role_candidates(colors: list[int]) -> list[tuple[int, int, int]]:
    candidates: list[tuple[int, int, int]] = []
    for unk in colors:
        remain = [c for c in colors if c != unk]
        candidates.append((int(remain[0]), int(remain[1]), int(unk)))
        candidates.append((int(remain[1]), int(remain[0]), int(unk)))
    return candidates


def _find_matches(grid: np.ndarray) -> list[tuple[int, int, int, int, int, np.ndarray]]:
    g = np.asarray(grid, dtype=int)
    H, W = map(int, g.shape)
    colors = colors_present(g)
    assert len(colors) == 3

    base = _pattern_mask()
    matches: list[tuple[int, int, int, int, int, np.ndarray]] = []
    for fg, bg, unk in _role_candidates(colors):
        known_fg = g == fg
        known_bg = g == bg
        unknown = g == unk
        if not known_fg.any() or not known_bg.any() or not unknown.any():
            continue
        for idx, pattern in enumerate(_dihedral_variants(base)):
            for r in range(20 - H + 1):
                for c in range(20 - W + 1):
                    window = pattern[r : r + H, c : c + W]
                    if np.all(window[known_fg]) and np.all(~window[known_bg]):
                        matches.append((fg, bg, idx, r, c, pattern))
    return matches


def _random_occlusion_mask(window: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    H, W = map(int, window.shape)
    area = H * W
    min_hidden = max(6, area * 18 // 100)
    max_hidden = max(min_hidden, area * 35 // 100)

    for _ in range(200):
        rh_low = max(2, H // 3)
        rh_high = max(rh_low, min(H - 1, (2 * H) // 3))
        rw_low = max(2, W // 3)
        rw_high = max(rw_low, min(W - 1, (2 * W) // 3))
        rh = int(rng.integers(rh_low, rh_high + 1))
        rw = int(rng.integers(rw_low, rw_high + 1))
        hidden_area = rh * rw
        if not (min_hidden <= hidden_area <= max_hidden):
            continue
        r0 = int(rng.integers(0, H - rh + 1))
        c0 = int(rng.integers(0, W - rw + 1))
        mask = np.zeros((H, W), dtype=bool)
        mask[r0 : r0 + rh, c0 : c0 + rw] = True
        hidden_fg = int(np.sum(window & mask))
        hidden_bg = int(np.sum((~window) & mask))
        visible_fg = int(np.sum(window & ~mask))
        visible_bg = int(np.sum((~window) & ~mask))
        if hidden_fg > 0 and hidden_bg > 0 and visible_fg >= 6 and visible_bg >= 6:
            return mask
    raise RuntimeError("failed to sample a valid occlusion mask")


def _generate_instance(
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    base = _pattern_mask()
    variants = _dihedral_variants(base)
    palette = list(Color.NOT_BLACK)

    for _ in range(500):
        fg, bg, unk = map(int, rng.choice(palette, size=3, replace=False))
        pattern_idx = int(rng.integers(0, len(variants)))
        pattern = variants[pattern_idx]
        output = np.where(pattern, fg, bg).astype(int)

        H = int(rng.integers(10, 17))
        W = int(rng.integers(10, 13))
        r = int(rng.integers(0, 20 - H + 1))
        c = int(rng.integers(0, 20 - W + 1))
        window = pattern[r : r + H, c : c + W]
        crop = output[r : r + H, c : c + W].copy()
        hidden = _random_occlusion_mask(window, rng)
        crop[hidden] = unk

        matches = _find_matches(crop)
        if len(matches) != 1:
            continue
        m_fg, m_bg, m_idx, m_r, m_c, m_pattern = matches[0]
        if not (m_fg == fg and m_bg == bg and m_idx == pattern_idx and m_r == r and m_c == c):
            continue
        if not np.array_equal(np.where(m_pattern, m_fg, m_bg).astype(int), output):
            continue

        meta = {
            "H": H,
            "W": W,
            "hidden_cells": int(np.sum(hidden)),
            "hidden_fg": int(np.sum(window & hidden)),
            "hidden_bg": int(np.sum((~window) & hidden)),
            "visible_fg": int(np.sum(window & ~hidden)),
            "visible_bg": int(np.sum((~window) & ~hidden)),
        }
        return crop, output, meta

    raise RuntimeError("failed to generate a unique partially hidden crop")


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    grid, output, _ = _generate_instance(rng)
    assert np.array_equal(main(grid), output)
    return grid


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the full 20x20 motif from a partially hidden crop with color-coded evidence.

    Input structure:
    - The grid is a rectangular crop taken from one rotation/reflection of the shared 20x20 motif.
    - Exactly three colors appear: one marks known motif cells, one marks known background cells,
      and the third marks cells inside the crop whose true value is hidden.

    Rule:
    - Test the possible color-role assignments and compare the crop against every window in every
      dihedral orientation, but only on the cells whose value is known.
    - Ignore the hidden-color cells during matching, keep the unique consistent orientation, and
      repaint the full 20x20 motif with the matched foreground/background colors.

    Key insight:
    - The crop is not fully observed anymore; its visible foreground/background evidence still
      singles out one orientation and offset, and the missing area can then be completed from the
      shared canonical motif.
    """
    g = np.asarray(grid, dtype=int)
    H, W = map(int, g.shape)
    assert H <= 20 and W <= 20

    matches = _find_matches(g)
    assert len(matches) == 1
    fg, bg, _, _, _, pattern = matches[0]
    return np.where(pattern, int(fg), int(bg)).astype(int)
