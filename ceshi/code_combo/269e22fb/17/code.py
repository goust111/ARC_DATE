# Combo variant of 269e22fb (index 17)
# combo_spec: D1/2+D2/3
# Name: 三色部分可见遮挡匹配

from __future__ import annotations

import numpy as np

from arc_utils import colors_present


# Design analysis
# ---------------
# a) Concrete behavior and trace:
# - The shared 20x20 mother pattern and its 8 dihedral orientations stay unchanged.
# - The input is still one rectangular crop from a chosen orientation, but the crop now
#   uses exactly 3 colors: known foreground, known background, and unknown.
# - The unknown color appears in two visibly different ways at once:
#   1) one axis-aligned rectangular hidden block,
#   2) extra isolated hidden pixels scattered outside that block.
# - Trace on a 12x10 crop:
#   original task: every cell is visible, so the solver compares the whole 12x10 binary
#   window against every candidate offset/orientation and requires exact equality.
#   fused variant: suppose a 4x3 middle block is hidden and 6 isolated outside cells are
#   also hidden. The solver must keep only the remaining visible foreground/background
#   evidence, ignore all unknown cells regardless of their shape, find the unique
#   orientation/offset/color-role assignment, then repaint the entire 20x20 pattern.
#
# b) Complexity calibration:
# - If the rectangle were the only missing-data device, this would collapse toward the
#   single-occluder variant. If the scatter were too sparse, humans could ignore it;
#   if it were too dense, the crop would look noisy and uniqueness would drop.
# - The generator therefore keeps crop sizes inside the original 80-156 area envelope,
#   hides about 14%-28% of the crop with one rectangle, then adds 4-10 isolated unknown
#   pixels outside that rectangle. This makes the two-part occlusion visually explicit
#   while preserving enough known evidence for a unique match after rejection sampling.
#
# c) Implicit assumptions review:
# - Original `assert len(colors) == 2` is invalid: the fused input has exactly 3 colors.
# - Original `visible = g == fg` assumed every non-foreground cell was known background.
#   Now the solver must split the crop into `known_fg`, `known_bg`, and `unknown`.
# - Original `np.array_equal(window, visible)` assumed full observability. The fused
#   solver instead checks only that candidate windows agree on known foreground cells
#   and known background cells, while ignoring unknown cells entirely.
# - Original polarity inference relied on the two colors being unlabeled. Here the crop
#   still does not tell us which color is unknown a priori, so solver-side role search
#   stays necessary across `(fg, bg, unk)` assignments.
#
# d) Placement legality:
# - Crop sizes remain within the original spec envelope and offsets stay inside 20x20.
# - The hidden rectangle is fully contained in the crop.
# - Scatter cells are chosen strictly outside that rectangle, so the two occlusion modes
#   coexist without overlap ambiguity.
# - The generator rejects any sample where hidden/visible regions fail to contain both
#   underlying binary states or where the solver finds anything other than one match.
#
# e) Visual observability:
# - Across a few examples, a human will repeatedly see the same 20x20 completion task,
#   but every input now contains one clean block of missing data plus several detached
#   unknown dots. The output always removes the third color and completes the motif
#   using the other two colors only.
# - Because the block changes position/size and the scattered dots move independently,
#   the unknown color is clearly interpretable as "missing information" rather than as
#   part of the motif itself.
#
# f) Visual cleanliness:
# - The unknown overlay is limited to one contiguous rectangle and a small number of
#   isolated points. That keeps the crop legible: there are only three semantic layers,
#   and the extra unknown pixels do not clutter into large irregular blobs.
#
# g) Change inferability:
# - Color roles are recoverable from structure rather than from numeric color ids: two
#   colors persist into the output, while the third disappears after completion.
# - The same third color marks both the hidden block and the hidden scattered cells, so
#   the solver rule is a single consistent idea: ignore unknown cells during matching.
#
# h) New solver assumptions and generator guarantees:
# - main() assumes the input has exactly one consistent `(fg, bg, unk, orientation,
#   offset)` combination; `_generate_instance()` enforces this by rejection.
# - main() assumes all three colors are present and both known classes have evidence;
#   `_generate_instance()` rejects samples lacking visible foreground/background.
# - main() treats all unknown cells identically and does not detect the rectangle or the
#   scatter explicitly; `_generate_instance()` is responsible for ensuring both forms are
#   present so the fusion behavior is genuinely manifested in the data.


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

_VALID_SIZES = [
    (h, w)
    for h in range(8, 17)
    for w in range(8, 13)
    if 80 <= h * w <= 156
]
_ARC_COLORS = tuple(range(10))


def _pattern_mask() -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in _BASE_PATTERN], dtype=bool)


def _dihedral_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    for k in range(4):
        rotated = np.rot90(mask, k)
        variants.append(rotated)
        variants.append(np.fliplr(rotated))
    return variants


_BASE_MASK = _pattern_mask()
_VARIANTS = _dihedral_variants(_BASE_MASK)


def _role_candidates(colors: list[int]) -> list[tuple[int, int, int]]:
    candidates: list[tuple[int, int, int]] = []
    for unk in colors:
        remain = [int(c) for c in colors if int(c) != int(unk)]
        candidates.append((remain[0], remain[1], int(unk)))
        candidates.append((remain[1], remain[0], int(unk)))
    return candidates


def _find_matches(grid: np.ndarray) -> list[tuple[int, int, int, int, int, np.ndarray]]:
    g = np.asarray(grid, dtype=int)
    H, W = map(int, g.shape)
    colors = colors_present(g)
    assert len(colors) == 3

    matches: list[tuple[int, int, int, int, int, np.ndarray]] = []
    for fg, bg, unk in _role_candidates(colors):
        known_fg = g == int(fg)
        known_bg = g == int(bg)
        unknown = g == int(unk)
        if not known_fg.any() or not known_bg.any() or not unknown.any():
            continue
        for idx, pattern in enumerate(_VARIANTS):
            for r in range(20 - H + 1):
                for c in range(20 - W + 1):
                    window = pattern[r : r + H, c : c + W]
                    if np.all(window[known_fg]) and np.all(~window[known_bg]):
                        matches.append((int(fg), int(bg), idx, int(r), int(c), pattern))
    return matches


def _sample_isolated_points(
    allowed: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray | None:
    H, W = map(int, allowed.shape)
    coords = np.argwhere(allowed)
    if len(coords) < count:
        return None

    order = rng.permutation(len(coords))
    chosen: list[tuple[int, int]] = []
    for idx in order:
        r, c = map(int, coords[idx])
        if any(abs(r - rr) <= 1 and abs(c - cc) <= 1 for rr, cc in chosen):
            continue
        chosen.append((r, c))
        if len(chosen) == count:
            mask = np.zeros((H, W), dtype=bool)
            for rr, cc in chosen:
                mask[rr, cc] = True
            return mask
    return None


def _sample_unknown_masks(
    window: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray] | None:
    H, W = map(int, window.shape)
    area = H * W
    min_rect = max(12, area * 14 // 100)
    max_rect = max(min_rect, area * 28 // 100)
    scatter_min = max(4, area * 4 // 100)
    scatter_max = max(scatter_min, min(10, area * 9 // 100))

    for _ in range(300):
        rh_low = max(3, H // 3)
        rh_high = max(rh_low, min(H - 2, (2 * H) // 3))
        rw_low = max(3, W // 3)
        rw_high = max(rw_low, min(W - 2, (2 * W) // 3))
        rh = int(rng.integers(rh_low, rh_high + 1))
        rw = int(rng.integers(rw_low, rw_high + 1))
        rect_area = rh * rw
        if not (min_rect <= rect_area <= max_rect):
            continue

        r0 = int(rng.integers(0, H - rh + 1))
        c0 = int(rng.integers(0, W - rw + 1))
        rect_mask = np.zeros((H, W), dtype=bool)
        rect_mask[r0 : r0 + rh, c0 : c0 + rw] = True

        rect_fg = int(np.sum(window & rect_mask))
        rect_bg = int(np.sum((~window) & rect_mask))
        if rect_fg == 0 or rect_bg == 0:
            continue

        allowed_scatter = ~rect_mask
        scatter_count = int(rng.integers(scatter_min, scatter_max + 1))
        scatter_mask = _sample_isolated_points(allowed_scatter, scatter_count, rng)
        if scatter_mask is None:
            continue

        scatter_fg = int(np.sum(window & scatter_mask))
        scatter_bg = int(np.sum((~window) & scatter_mask))
        if scatter_fg == 0 or scatter_bg == 0:
            continue

        unknown_mask = rect_mask | scatter_mask
        visible_mask = ~unknown_mask
        visible_fg = int(np.sum(window & visible_mask))
        visible_bg = int(np.sum((~window) & visible_mask))
        if visible_fg < 8 or visible_bg < 8:
            continue

        return rect_mask, scatter_mask
    return None


def _generate_instance(
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict[str, int | float]]:
    for _ in range(800):
        fg, bg, unk = map(int, rng.choice(_ARC_COLORS, size=3, replace=False))
        pattern_idx = int(rng.integers(0, len(_VARIANTS)))
        pattern = _VARIANTS[pattern_idx]
        output = np.where(pattern, int(fg), int(bg)).astype(int)

        H, W = _VALID_SIZES[int(rng.integers(0, len(_VALID_SIZES)))]
        r = int(rng.integers(0, 20 - H + 1))
        c = int(rng.integers(0, 20 - W + 1))
        window = pattern[r : r + H, c : c + W]

        masks = _sample_unknown_masks(window, rng)
        if masks is None:
            continue
        rect_mask, scatter_mask = masks
        unknown_mask = rect_mask | scatter_mask

        crop = output[r : r + H, c : c + W].copy()
        crop[unknown_mask] = int(unk)

        matches = _find_matches(crop)
        if len(matches) != 1:
            continue
        m_fg, m_bg, m_idx, m_r, m_c, m_pattern = matches[0]
        if not (m_fg == fg and m_bg == bg and m_idx == pattern_idx and m_r == r and m_c == c):
            continue
        if not np.array_equal(np.where(m_pattern, m_fg, m_bg).astype(int), output):
            continue

        meta: dict[str, int | float] = {
            "H": H,
            "W": W,
            "rect_hidden": int(np.sum(rect_mask)),
            "scatter_hidden": int(np.sum(scatter_mask)),
            "total_hidden": int(np.sum(unknown_mask)),
            "unknown_ratio": float(np.mean(unknown_mask)),
            "rect_fg": int(np.sum(window & rect_mask)),
            "rect_bg": int(np.sum((~window) & rect_mask)),
            "scatter_fg": int(np.sum(window & scatter_mask)),
            "scatter_bg": int(np.sum((~window) & scatter_mask)),
            "visible_fg": int(np.sum(window & ~unknown_mask)),
            "visible_bg": int(np.sum((~window) & ~unknown_mask)),
        }
        return crop, output, meta

    raise RuntimeError("failed to generate a unique mixed-occlusion crop")


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    grid, output, _ = _generate_instance(rng)
    assert np.array_equal(main(grid), output)
    return grid


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the full 20x20 motif from a partially known crop with mixed unknown regions.

    Input structure:
    - The grid is a rectangular crop from one rotation/reflection of the shared 20x20 motif.
    - Exactly three colors appear: one marks known motif foreground, one marks known motif
      background, and one marks cells whose true value is unknown.
    - The unknown color may cover both a solid block and extra isolated cells inside the crop.

    Rule:
    - Enumerate the possible assignments of the three input colors to foreground, background,
      and unknown roles.
    - For every dihedral orientation and crop offset, compare the candidate motif window only
      on known cells, ignoring every unknown-colored position.
    - Keep the unique consistent match and repaint the full 20x20 motif with the recovered
      foreground/background colors; the unknown color never appears in the output.

    Key insight:
    - The third color encodes missing information rather than motif content. Even though the
      hidden cells come in both contiguous and scattered forms, the visible foreground/background
      evidence still identifies one orientation and offset of the canonical motif.
    """
    g = np.asarray(grid, dtype=int)
    H, W = map(int, g.shape)
    assert H <= 20 and W <= 20

    matches = _find_matches(g)
    assert len(matches) == 1
    fg, bg, _idx, _r, _c, pattern = matches[0]
    return np.where(pattern, int(fg), int(bg)).astype(int)
