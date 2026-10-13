"""ARC task 7: complete the missing panel by foreground parity."""

from __future__ import annotations

import random

import numpy as np


RED = 2
ARC_COLORS = tuple(range(10))
MAX_GENERATION_ATTEMPTS = 200
PANEL_SIDE_RANGE = range(4, 11)


def _split_panels(grid: np.ndarray) -> tuple[list[np.ndarray], int]:
    """Validate and split a one-line-separated 2x2 panel grid."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height >= 3 and width >= 3, "panels must be nonempty"
    assert height % 2 == 1 and width % 2 == 1, "expected equal panels around one-pixel separators"

    split_r, split_c = height // 2, width // 2
    separator = int(g[split_r, split_c])
    assert np.all(g[split_r, :] == separator), "expected a solid central horizontal separator"
    assert np.all(g[:, split_c] == separator), "expected a matching solid central vertical separator"

    panels = [
        g[:split_r, :split_c],
        g[:split_r, split_c + 1 :],
        g[split_r + 1 :, :split_c],
        g[split_r + 1 :, split_c + 1 :],
    ]
    assert all(panel.shape == panels[0].shape for panel in panels), "all four panels must align"
    return panels, separator


def main(grid: np.ndarray) -> np.ndarray:
    """Complete the blank fourth panel so every aligned position has even foreground parity.

    Input structure:
    - A solid horizontal line and a solid vertical line partition the grid into a 2x2
      array of four equal panels in the shared row-column coordinate system.
    - The first three panels are binary masks with a shared background color and one
      monochromatic foreground each; the bottom-right panel is entirely background.

    Rule:
    - At each aligned panel coordinate, count foreground occurrences in the first
      three panels. Return a bottom-right-sized background panel with red exactly
      where that count is odd.

    Key insight:
    - The missing panel is the three-mask parity complement: both one and three
      foreground votes produce red, while zero and two votes remain background.
    """
    panels, separator = _split_panels(grid)
    top_left, top_right, bottom_left, missing = panels

    missing_colors = np.unique(missing)
    assert missing_colors.size == 1, "the bottom-right panel must be blank"
    background = int(missing_colors[0])
    assert background != RED, "red must remain a visible output marker, not the background"
    assert separator != background, "separator and shared background must be visibly distinct"

    masks: list[np.ndarray] = []
    for panel in (top_left, top_right, bottom_left):
        colors = set(map(int, np.unique(panel)))
        assert background in colors, "every filled panel must use the shared background"
        foreground_colors = colors - {background}
        assert len(foreground_colors) == 1, "each filled panel must have one foreground color"
        masks.append(panel != background)

    odd = np.logical_xor(np.logical_xor(masks[0], masks[1]), masks[2])
    out = np.full(missing.shape, background, dtype=int)
    out[odd] = RED
    return out


def _sample_masks(height: int, width: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample masks with witnesses for every overlap count and every panel."""
    cell_count = height * width
    # Categories are three-bit membership codes.  Forcing all seven nonzero
    # codes supplies singleton, pair-overlap, and triple-overlap counterexamples.
    categories = np.zeros(cell_count, dtype=np.uint8)
    order = np.random.permutation(cell_count)
    categories[order[:7]] = np.random.permutation(np.arange(1, 8, dtype=np.uint8))

    remaining = order[7:]
    if remaining.size:
        # Blank cells remain most common; all overlap branches still vary freely.
        categories[remaining] = np.random.choice(
            np.arange(8, dtype=np.uint8),
            size=remaining.size,
            p=(0.42, 0.09, 0.09, 0.08, 0.09, 0.08, 0.08, 0.07),
        )

    masks = tuple(((categories & bit) != 0).reshape(height, width) for bit in (1, 2, 4))
    return masks  # type: ignore[return-value]


def _expected_from_masks(masks: tuple[np.ndarray, np.ndarray, np.ndarray], background: int) -> np.ndarray:
    odd = np.logical_xor(np.logical_xor(masks[0], masks[1]), masks[2])
    return np.where(odd, RED, background).astype(int)


def _gen_attempt() -> np.ndarray | None:
    panel_h = random.choice(tuple(PANEL_SIDE_RANGE))
    panel_w = random.choice(tuple(PANEL_SIDE_RANGE))
    masks = _sample_masks(panel_h, panel_w)

    role_colors = random.sample([color for color in ARC_COLORS if color != RED], 5)
    background, separator, fg0, fg1, fg2 = role_colors
    panels = []
    for mask, foreground in zip(masks, (fg0, fg1, fg2)):
        panel = np.full((panel_h, panel_w), background, dtype=int)
        panel[mask] = foreground
        panels.append(panel)
    blank = np.full((panel_h, panel_w), background, dtype=int)

    grid = np.full((2 * panel_h + 1, 2 * panel_w + 1), separator, dtype=int)
    grid[:panel_h, :panel_w] = panels[0]
    grid[:panel_h, panel_w + 1 :] = panels[1]
    grid[panel_h + 1 :, :panel_w] = panels[2]
    grid[panel_h + 1 :, panel_w + 1 :] = blank

    expected = _expected_from_masks(masks, background)
    if not np.array_equal(main(grid), expected):
        return None

    counts = masks[0].astype(int) + masks[1] + masks[2]
    # Semantic checks: all four parity counts occur, every source is necessary,
    # and no two-panel XOR or union is an accidental complete explanation.
    if set(map(int, np.unique(counts))) != {0, 1, 2, 3}:
        return None
    parity = expected == RED
    if any(np.array_equal(parity, np.logical_xor(masks[i], masks[j])) for i, j in ((0, 1), (0, 2), (1, 2))):
        return None
    if np.array_equal(parity, masks[0] | masks[1] | masks[2]):
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate a varied, semantically non-degenerate parity-panel input."""
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
