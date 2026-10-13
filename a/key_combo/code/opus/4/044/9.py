from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, bbox_coords, colors_present, components_mask, get_bg, paste, random_free_location


PALETTE = [Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.ORANGE]
MIN_FRAMES = 3
MAX_FRAMES = 5
MIN_GRID = 20
MAX_GRID = 28
MAX_ATTEMPTS = 400


def _render_nested(colors: list[int]) -> np.ndarray:
    size = 2 * len(colors) - 1
    out = np.zeros((size, size), dtype=int)
    for layer, color in enumerate(colors):
        end = size - 1 - layer
        out[layer : end + 1, layer] = color
        out[layer : end + 1, end] = color
        out[layer, layer : end + 1] = color
        out[end, layer : end + 1] = color
    return out


def _extract_infos(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    infos: list[dict] = []

    for color in colors_present(g, exclude={bg}):
        coords = np.argwhere(g == color)
        assert coords.size > 0, "each foreground color must appear"
        r0, c0, r1, c1 = bbox_coords(coords)
        outer_h = int(r1 - r0 + 1)
        outer_w = int(c1 - c0 + 1)
        assert outer_h >= 5 and outer_w >= 5, "each object must be a sufficiently large frame"

        assert np.all(g[r0, c0 : c1 + 1] == color)
        assert np.all(g[r1, c0 : c1 + 1] == color)
        assert np.all(g[r0 : r1 + 1, c0] == color)
        assert np.all(g[r0 : r1 + 1, c1] == color)

        inner_mask = (g[r0 + 1 : r1, c0 + 1 : c1] == color)
        inner_h, inner_w = inner_mask.shape
        comps = components_mask(inner_mask, connectivity=4)
        assert len(comps) == 1, "each frame must contain exactly one inner bar"
        bar = comps[0]

        rows = np.unique(bar[:, 0])
        cols = np.unique(bar[:, 1])
        assert len(rows) == 1, "the inner mark must be horizontal"
        assert int(bar.shape[0]) == int(cols.max() - cols.min() + 1), "the bar must be contiguous"
        assert rows[0] > 0 and rows[0] < inner_h - 1, "bar must not touch the frame vertically"
        assert cols.min() > 0 and cols.max() < inner_w - 1, "bar must not touch the frame horizontally"

        infos.append(
            {
                "color": int(color),
                "length": int(bar.shape[0]),
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "outer_h": outer_h,
                "outer_w": outer_w,
                "area": outer_h * outer_w,
                "center_r": r0 + r1,
                "center_c": c0 + c1,
            }
        )

    assert MIN_FRAMES <= len(infos) <= MAX_FRAMES, "expected 3 to 5 colored frames"
    lengths = [info["length"] for info in infos]
    assert len(set(lengths)) == len(lengths), "bar lengths must be unique"
    return infos


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort scattered frames by the length of their enclosed horizontal bars and redraw them as concentric square rings.

    Input structure:
    - Black background with 3 to 5 one-pixel-thick colored rectangular frames.
    - Each frame color is unique, and each frame contains exactly one same-color horizontal solid bar.
    - The bar lies strictly inside the frame and does not touch the frame border.

    Rule:
    - For each colored frame, locate the enclosed horizontal bar and measure its pixel length.
    - Sort all frame colors by that bar length from shortest to longest.
    - Create a square output of side `2K-1`, where `K` is the number of frames.
    - Redraw the sorted colors as concentric square rings from outermost to innermost; the longest-bar color becomes the center pixel.

    Key insight:
    - The visible frames are not ordered by their own size or position; the hidden ordering key is the length of the same-color bar inside each frame, and that bar disappears in the output.
    """
    infos = _extract_infos(np.asarray(grid, dtype=int))
    ordered_colors = [info["color"] for info in sorted(infos, key=lambda info: info["length"])]
    return _render_nested(ordered_colors)


def _frame_patch(color: int, outer_h: int, outer_w: int, bar_len: int, bar_row: int, bar_col: int) -> np.ndarray:
    patch = np.zeros((outer_h, outer_w), dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    patch[1 + bar_row, 1 + bar_col : 1 + bar_col + bar_len] = color
    return patch


def _order_tuple(infos: list[dict], key_fn, reverse: bool = False) -> tuple[int, ...]:
    return tuple(info["color"] for info in sorted(infos, key=lambda info: (key_fn(info), info["color"]), reverse=reverse))


def _passes_shortcut_checks(grid: np.ndarray) -> bool:
    infos = _extract_infos(grid)
    intended = tuple(info["color"] for info in sorted(infos, key=lambda info: info["length"]))
    proxy_keys = [
        lambda info: info["outer_w"],
        lambda info: info["outer_h"],
        lambda info: info["area"],
        lambda info: info["bbox"][0],
        lambda info: info["bbox"][1],
        lambda info: (info["bbox"][0], info["bbox"][1]),
        lambda info: (info["center_r"], info["center_c"]),
    ]
    for key_fn in proxy_keys:
        if _order_tuple(infos, key_fn) == intended:
            return False
        if _order_tuple(infos, key_fn, reverse=True) == intended:
            return False
    return True


def _attempt_generate() -> np.ndarray | None:
    rng = np.random.default_rng()
    n_frames = int(rng.integers(MIN_FRAMES, MAX_FRAMES + 1))
    height = int(rng.integers(MIN_GRID, MAX_GRID + 1))
    width = int(rng.integers(MIN_GRID, MAX_GRID + 1))

    grid = np.zeros((height, width), dtype=int)
    occupied = np.zeros_like(grid)
    colors = [int(x) for x in rng.choice(PALETTE, size=n_frames, replace=False)]

    # One-cell margins inside the frame imply bar_len <= outer_w - 4.
    # Allowing width up to 10 keeps five distinct lengths realizable.
    candidate_lengths = np.arange(1, 7)
    bar_lengths = [int(x) for x in rng.choice(candidate_lengths, size=n_frames, replace=False)]
    rng.shuffle(bar_lengths)

    for color, bar_len in zip(colors, bar_lengths):
        min_outer_w = max(5, bar_len + 4)
        outer_h = int(rng.integers(5, 9))
        outer_w = int(rng.integers(min_outer_w, 11))
        inner_h = outer_h - 2
        inner_w = outer_w - 2
        if inner_h < 3 or inner_w < bar_len + 2:
            return None

        bar_row = int(rng.integers(1, inner_h - 1))
        bar_col = int(rng.integers(1, inner_w - bar_len))
        patch = _frame_patch(color, outer_h, outer_w, bar_len, bar_row, bar_col)

        full_box = np.ones((outer_h, outer_w), dtype=int)
        try:
            r0, c0 = random_free_location(
                occupied,
                full_box,
                transparent=0,
                padding=1,
                padding_connectivity=8,
            )
        except ValueError:
            return None

        paste(occupied, full_box, r0, c0)
        paste(grid, patch, r0, c0)

    try:
        infos = _extract_infos(grid)
    except AssertionError:
        return None

    if len(infos) != n_frames:
        return None
    if not _passes_shortcut_checks(grid):
        return None

    out = main(grid)
    if out.shape != (2 * n_frames - 1, 2 * n_frames - 1):
        return None

    expected = _render_nested([info["color"] for info in sorted(infos, key=lambda info: info["length"])])
    if not np.array_equal(out, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_ATTEMPTS} attempts")
