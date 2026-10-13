from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, bbox


MASK_H = 5
MASK_W_MIN = 7
MASK_W_MAX = 9
SPECIAL_COLORS = {Color.BLACK, Color.RED, Color.YELLOW}
PALETTE_POOL = [Color.BLUE, Color.GREEN, Color.GREY, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON]
MAX_GENERATION_ATTEMPTS = 500


def _bbox_center(box: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = map(int, box)
    return (r0 + r1) // 2, (c0 + c1) // 2


def _paint_concentric(mask: np.ndarray, palette: list[int], metric: str = "chebyshev") -> np.ndarray:
    box = bbox(mask)
    assert box is not None, "expected a non-empty visible region"
    center_r, center_c = _bbox_center(box)
    out = np.zeros(mask.shape, dtype=int)
    for r, c in np.argwhere(mask):
        rr, cc = int(r), int(c)
        if metric == "chebyshev":
            dist = max(abs(rr - center_r), abs(cc - center_c))
        elif metric == "manhattan":
            dist = abs(rr - center_r) + abs(cc - center_c)
        else:
            raise ValueError(f"unknown metric: {metric}")
        out[rr, cc] = int(palette[min(dist, len(palette) - 1)])
    return out


def _find_separator_and_panels(grid: np.ndarray) -> tuple[int, int, int, np.ndarray, np.ndarray]:
    sep_rows = np.where(np.any(grid == Color.YELLOW, axis=1))[0]
    assert sep_rows.size == 1, "expected exactly one separator row"
    sep_r = int(sep_rows[0])
    sep_cols = np.where(grid[sep_r] == Color.YELLOW)[0]
    assert sep_cols.size >= MASK_W_MIN, "separator row is too short"
    assert np.all(np.diff(sep_cols) == 1), "separator row must be contiguous"
    assert np.count_nonzero(grid == Color.YELLOW) == sep_cols.size, "yellow only belongs to the separator row"

    c0, c1 = int(sep_cols[0]), int(sep_cols[-1])
    assert sep_r >= MASK_H and sep_r + MASK_H < grid.shape[0], "separator row must have two 5-row panels"

    top = grid[sep_r - MASK_H : sep_r, c0 : c1 + 1].copy()
    bottom = grid[sep_r + 1 : sep_r + 1 + MASK_H, c0 : c1 + 1].copy()
    valid_panel = {Color.BLACK, Color.RED}
    assert set(map(int, np.unique(top))) <= valid_panel
    assert set(map(int, np.unique(bottom))) <= valid_panel
    return sep_r, c0, c1, top, bottom


def _read_palette(grid: np.ndarray, panel_right_col: int) -> list[int]:
    markers: list[tuple[int, int, int]] = []
    for color in sorted(set(map(int, np.unique(grid))) - SPECIAL_COLORS):
        coords = np.argwhere(grid == color)
        assert coords.shape == (1, 2), "each palette color must appear exactly once"
        r, c = map(int, coords[0])
        assert c > panel_right_col, "palette markers must sit in the blank area to the right of the panels"
        markers.append((r, c, color))

    assert len(markers) >= 2, "expected at least two palette markers"
    markers.sort(key=lambda item: (item[0], item[1]))
    for (r0, c0, _), (r1, c1, _) in zip(markers, markers[1:]):
        assert (r1, c1) == (r0 + 1, c0 + 1), "palette markers must form a contiguous down-right diagonal"
    return [color for _r, _c, color in markers]


def main(grid: np.ndarray) -> np.ndarray:
    """Use the XOR-visible region of two red/black panels as a mask for concentric square-ring recoloring.

    Input structure:
    - The grid contains a 5-row top red/black mask, a one-row yellow separator, and a 5-row bottom red/black mask.
    - The separator's contiguous yellow run gives the shared panel width; cells outside that band are background.
    - Additional singleton colors appear only as a contiguous down-right diagonal in the blank area to the right.

    Rule:
    - Split the input at the yellow separator and compare the two 5xN panels cell by cell.
    - A panel cell is visible in the output iff exactly one panel has red there; this XOR-visible set defines the output mask.
    - Compute the XOR mask's bounding box, take its floor-centered box center, read the diagonal marker colors from upper-left to lower-right, and color each visible cell by its Chebyshev distance from that center.

    Key insight:
    - The two red/black panels determine only where color may appear, while the diagonal markers determine which concentric square-ring color each visible XOR cell receives.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    _sep_r, _c0, c1, top, bottom = _find_separator_and_panels(g)
    palette = _read_palette(g, c1)

    xor_mask = (top == Color.RED) ^ (bottom == Color.RED)
    assert np.any(xor_mask), "expected a non-empty XOR-visible region"
    return _paint_concentric(xor_mask, palette, metric="chebyshev")


def _sample_bbox(mask_w: int) -> tuple[int, int, int, int]:
    height = random.choice([3, 4, 5])
    width_choices = [w for w in range(3, mask_w + 1) if w >= 3]
    width = random.choice(width_choices)
    r0 = random.randint(0, MASK_H - height)
    c0 = random.randint(0, mask_w - width)
    return r0, c0, r0 + height - 1, c0 + width - 1


def _sample_xor_mask(mask_w: int) -> tuple[np.ndarray, tuple[int, int, int, int], int]:
    for _ in range(300):
        box = _sample_bbox(mask_w)
        r0, c0, r1, c1 = box
        center_r, center_c = _bbox_center(box)
        dmax = max(
            max(abs(r - center_r), abs(c - center_c))
            for r, c in ((r0, c0), (r0, c1), (r1, c0), (r1, c1))
        )
        cells = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
        diag_cells = [(r, c) for r, c in cells if abs(r - center_r) == abs(c - center_c) and (r, c) != (center_r, center_c)]
        inner_cells = [(r, c) for r, c in cells if max(abs(r - center_r), abs(c - center_c)) < dmax]
        if not diag_cells or not inner_cells:
            continue

        active: set[tuple[int, int]] = set()
        edge_groups = [
            [(r0, c) for c in range(c0, c1 + 1)],
            [(r1, c) for c in range(c0, c1 + 1)],
            [(r, c0) for r in range(r0, r1 + 1)],
            [(r, c1) for r in range(r0, r1 + 1)],
        ]
        for group in edge_groups:
            active.add(random.choice(group))

        active.add(random.choice(diag_cells))
        active.add(random.choice(inner_cells))
        if random.random() < 0.8:
            active.add((center_r, center_c))

        box_area = len(cells)
        min_area = max(8, len(active))
        max_area = box_area - 1
        if min_area > max_area:
            continue

        target_area = random.randint(min_area, max_area)
        remaining = [cell for cell in cells if cell not in active]
        random.shuffle(remaining)
        for cell in remaining:
            if len(active) >= target_area:
                break
            if random.random() < 0.55:
                active.add(cell)
        if len(active) < target_area:
            active.update(remaining[: target_area - len(active)])

        mask = np.zeros((MASK_H, mask_w), dtype=bool)
        for r, c in active:
            mask[r, c] = True

        if bbox(mask) != box:
            continue
        if np.count_nonzero(mask) < 8:
            continue
        if np.all(mask[r0 : r1 + 1, c0 : c1 + 1]):
            continue
        if len({max(abs(r - center_r), abs(c - center_c)) for r, c in active}) < 2:
            continue
        if not any(abs(r - center_r) == abs(c - center_c) and max(abs(r - center_r), abs(c - center_c)) > 0 for r, c in active):
            continue
        return mask, box, dmax
    raise RuntimeError("failed to sample a valid XOR mask")


def _decompose_xor_mask(xor_mask: np.ndarray, box: tuple[int, int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    active = [tuple(map(int, coord)) for coord in np.argwhere(xor_mask)]
    inactive = [tuple(map(int, coord)) for coord in np.argwhere(~xor_mask)]
    r0, c0, r1, c1 = box
    inactive_inside = [(r, c) for r, c in inactive if r0 <= r <= r1 and c0 <= c <= c1]
    assert inactive_inside, "generator expects at least one masked-out cell inside the XOR bbox"

    for _ in range(300):
        top = np.zeros_like(xor_mask, dtype=bool)
        bottom = np.zeros_like(xor_mask, dtype=bool)

        choices = [random.randint(0, 1) for _ in active]
        if all(choice == 0 for choice in choices) or all(choice == 1 for choice in choices):
            continue
        for (r, c), choice in zip(active, choices):
            if choice == 0:
                top[r, c] = True
            else:
                bottom[r, c] = True

        overlap_inside = set(random.sample(inactive_inside, k=1))
        other_inactive = [coord for coord in inactive if coord not in overlap_inside]
        extra_overlap_limit = min(len(other_inactive), max(0, len(inactive) // 6))
        extra_overlap_count = random.randint(0, extra_overlap_limit)
        overlap = overlap_inside | set(random.sample(other_inactive, k=extra_overlap_count))
        for r, c in overlap:
            top[r, c] = True
            bottom[r, c] = True

        if not np.any(top) or not np.any(bottom):
            continue
        if np.array_equal(top, xor_mask) or np.array_equal(bottom, xor_mask):
            continue
        if np.array_equal(top | bottom, xor_mask):
            continue
        if np.count_nonzero(top) < 4 or np.count_nonzero(bottom) < 4:
            continue
        return top, bottom
    raise RuntimeError("failed to decompose XOR mask into two informative panels")


def _render_input(top: np.ndarray, bottom: np.ndarray, palette: list[int]) -> np.ndarray:
    mask_w = top.shape[1]
    right_gap = random.randint(1, 3)
    extra_right = random.randint(0, 2)
    base_rows = 2 * MASK_H + 1
    extra_bottom = random.randint(0, 3)
    height = base_rows + extra_bottom
    width = mask_w + right_gap + len(palette) + extra_right
    start_r = random.randint(0, height - len(palette))
    start_c = mask_w + right_gap

    grid = np.full((height, width), Color.BLACK, dtype=int)
    grid[:MASK_H, :mask_w] = np.where(top, Color.RED, Color.BLACK)
    grid[MASK_H, :mask_w] = Color.YELLOW
    grid[MASK_H + 1 : 2 * MASK_H + 1, :mask_w] = np.where(bottom, Color.RED, Color.BLACK)
    for i, color in enumerate(palette):
        grid[start_r + i, start_c + i] = int(color)
    return grid


def _bbox_fill_mask(mask: np.ndarray) -> np.ndarray:
    box = bbox(mask)
    assert box is not None
    r0, c0, r1, c1 = box
    filled = np.zeros_like(mask, dtype=bool)
    filled[r0 : r1 + 1, c0 : c1 + 1] = True
    return filled


def _semantic_instance_ok(grid: np.ndarray, xor_mask: np.ndarray, top: np.ndarray, bottom: np.ndarray, palette: list[int]) -> bool:
    expected = _paint_concentric(xor_mask, palette, metric="chebyshev")
    try:
        actual = main(grid)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False

    visible_colors = {int(color) for color in np.unique(expected) if int(color) != Color.BLACK}
    if len(visible_colors) < 2:
        return False

    alternatives = [
        _paint_concentric(xor_mask, list(reversed(palette)), metric="chebyshev"),
        _paint_concentric(xor_mask, palette, metric="manhattan"),
        _paint_concentric(top | bottom, palette, metric="chebyshev"),
        _paint_concentric(_bbox_fill_mask(xor_mask), palette, metric="chebyshev"),
        _paint_concentric(top, palette, metric="chebyshev"),
        _paint_concentric(bottom, palette, metric="chebyshev"),
    ]
    if any(np.array_equal(expected, alt) for alt in alternatives):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    mask_w = random.randint(MASK_W_MIN, MASK_W_MAX)
    xor_mask, box, dmax = _sample_xor_mask(mask_w)
    palette_len = dmax + 1 + random.randint(0, 1)
    if palette_len > len(PALETTE_POOL):
        return None
    palette = random.sample(PALETTE_POOL, k=palette_len)
    top, bottom = _decompose_xor_mask(xor_mask, box)
    grid = _render_input(top, bottom, palette)
    if not _semantic_instance_ok(grid, xor_mask, top, bottom, palette):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid XOR-masked concentric palette task")
