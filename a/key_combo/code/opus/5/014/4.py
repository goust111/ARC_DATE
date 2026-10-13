from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, bbox, crop_bbox, get_bg, paste


BG = Color.BLACK
PALETTE = tuple(Color.NOT_BLACK)
K_CHOICES = (2, 3, 4, 5)
MAX_FOREGROUND = 5
GEN_RETRIES = 300


def _is_l_legend(mask: np.ndarray) -> tuple[bool, int]:
    box = bbox(mask)
    if box is None:
        return False, -1
    r0, c0, r1, c1 = box
    if r0 != 0 or c0 != 0 or r1 != c1 or r1 < 2:
        return False, -1
    k = int(r1)
    expected = np.zeros_like(mask, dtype=bool)
    expected[k, : k + 1] = True
    expected[: k + 1, k] = True
    return bool(np.array_equal(mask, expected)), k


def _extract_spec(grid: np.ndarray) -> tuple[np.ndarray, int, int, int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] == g.shape[1], "expected a square 2D grid"
    bg = int(get_bg(g))
    assert bg == BG, "background must be black"

    colors = sorted(int(c) for c in np.unique(g) if int(c) != bg)
    assert len(colors) == 2, "expected exactly two non-background colors"

    legend_candidates: list[tuple[int, int]] = []
    for color in colors:
        ok, k = _is_l_legend(g == color)
        if ok:
            legend_candidates.append((color, k))
    assert len(legend_candidates) == 1, "expected a unique L-shaped border legend"

    secondary, k = legend_candidates[0]
    primary = colors[0] if colors[1] == secondary else colors[1]
    n = int(g.shape[0])
    assert n == k * k, "grid side must equal the seed side squared"

    legend_box = crop_bbox(g, (0, 0, k, k))
    assert np.all(legend_box[k, :] == secondary), "legend bottom row must be solid"
    assert np.all(legend_box[:, k] == secondary), "legend right column must be solid"
    assert np.all(g[k + 1 :, :] == bg), "cells below the legend must be background"
    assert np.all(g[: k + 1, k + 1 :] == bg), "cells right of the legend must be background"

    seed = legend_box[:k, :k]
    assert np.all((seed == bg) | (seed == primary)), "seed must use only black and the primary color"
    seed_mask = seed == primary
    assert seed_mask.any() and (~seed_mask).any(), "seed must contain both foreground and background"
    return seed_mask, primary, secondary, k


def _expand(seed_mask: np.ndarray, primary: int, secondary: int, *, parity_flip: bool) -> np.ndarray:
    k = int(seed_mask.shape[0])
    out = np.zeros((k * k, k * k), dtype=int)
    for r in range(k):
        for c in range(k):
            if parity_flip:
                use_primary = bool(seed_mask[r, c]) if (r + c) % 2 == 0 else not bool(seed_mask[r, c])
            else:
                use_primary = bool(seed_mask[r, c])
            color = primary if use_primary else secondary
            block = np.where(seed_mask, color, BG)
            paste(out, block, r * k, c * k)
    return out


def _parity_only(seed_mask: np.ndarray, primary: int, secondary: int) -> np.ndarray:
    k = int(seed_mask.shape[0])
    out = np.zeros((k * k, k * k), dtype=int)
    for r in range(k):
        for c in range(k):
            color = primary if (r + c) % 2 == 0 else secondary
            block = np.where(seed_mask, color, BG)
            paste(out, block, r * k, c * k)
    return out


def _has_adjacent_pair(mask: np.ndarray, value: bool) -> bool:
    target = mask == value
    return bool(np.any(target[:, :-1] & target[:, 1:]) or np.any(target[:-1, :] & target[1:, :]))


def _seed_is_non_degenerate(seed_mask: np.ndarray) -> bool:
    if seed_mask.shape[0] != seed_mask.shape[1]:
        return False
    k = int(seed_mask.shape[0])
    fg = int(seed_mask.sum())
    if fg < 2 or fg > min(MAX_FOREGROUND, k * k - 2):
        return False

    parities = (np.indices(seed_mask.shape).sum(axis=0) % 2).astype(bool)
    for parity in (False, True):
        parity_cells = parities == parity
        if not np.any(seed_mask & parity_cells):
            return False
        if not np.any((~seed_mask) & parity_cells):
            return False

    if not _has_adjacent_pair(seed_mask, True):
        return False
    if not _has_adjacent_pair(seed_mask, False):
        return False
    if np.array_equal(seed_mask, np.flipud(seed_mask)) and np.array_equal(seed_mask, np.fliplr(seed_mask)):
        return False
    return True


def _random_seed_mask(k: int) -> np.ndarray:
    cells = [(r, c) for r in range(k) for c in range(k)]
    min_fg = 2
    max_fg = min(MAX_FOREGROUND, k * k - 2)
    for _ in range(200):
        fg_count = random.randint(min_fg, max_fg)
        chosen = set(random.sample(cells, fg_count))
        seed_mask = np.array(
            [[(r, c) in chosen for c in range(k)] for r in range(k)],
            dtype=bool,
        )
        if _seed_is_non_degenerate(seed_mask):
            return seed_mask
    raise RuntimeError("failed to generate a non-degenerate seed mask")


def _render_input(seed_mask: np.ndarray, primary: int, secondary: int) -> np.ndarray:
    k = int(seed_mask.shape[0])
    n = k * k
    grid = np.zeros((n, n), dtype=int)
    grid[:k, :k] = np.where(seed_mask, primary, BG)
    grid[k, : k + 1] = secondary
    grid[: k + 1, k] = secondary
    return grid


def _validate_semantics(grid: np.ndarray, out: np.ndarray) -> None:
    seed_mask, primary, secondary, k = _extract_spec(grid)
    expected = _expand(seed_mask, primary, secondary, parity_flip=True)
    assert np.array_equal(out, expected), "main() must implement the parity-flipped recursive expansion"

    vanilla = _expand(seed_mask, primary, secondary, parity_flip=False)
    assert not np.array_equal(out, vanilla), "parity flipping must change the result"

    parity_only = _parity_only(seed_mask, primary, secondary)
    assert not np.array_equal(out, parity_only), "the seed foreground/background roles must affect the result"

    assert out.shape == (k * k, k * k)
    colors = set(int(c) for c in np.unique(out))
    assert colors == {BG, primary, secondary}, "output must contain black plus both legend colors"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand the top-left seed legend into a self-similar tiling whose block color flips on odd coarse-grid parity.

    Input structure:
    - The background is black.
    - The only non-background content is a top-left square legend whose last row and last column form a solid L shape.
    - Removing that L border leaves a square seed that uses black plus one primary color.

    Rule:
    - Read the seed as a binary mask M where primary-color cells are foreground and black cells are background.
    - Let the seed side be k; the full grid side is k^2, so the output is tiled by a k x k coarse grid of k x k blocks.
    - For coarse position (i, j), stamp a copy of the seed mask in one color: on even parity blocks use the primary color for seed-foreground cells and the secondary color for seed-background cells; on odd parity blocks swap those two roles.
    - Inside every stamped block, background cells of the seed mask remain black.

    Key insight:
    - The L border supplies the secondary color, while the final block color depends jointly on the seed bit at that coarse position and on the parity of that coarse position.
    """
    seed_mask, primary, secondary, _k = _extract_spec(np.asarray(grid, dtype=int))
    return _expand(seed_mask, primary, secondary, parity_flip=True)


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        k = random.choice(K_CHOICES)
        seed_mask = _random_seed_mask(k)
        primary, secondary = random.sample(PALETTE, 2)
        grid = _render_input(seed_mask, primary, secondary)
        out = main(grid)
        _validate_semantics(grid, out)
        return grid
    raise RuntimeError("failed to generate a valid input")
