from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, bbox, bbox_coords, components_mask, crop_bbox, paste


SPRITE_CATALOG = [
    np.array(
        [
            [1, 1, 0],
            [0, 1, 0],
            [0, 1, 1],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [0, 1, 0],
            [1, 1, 1],
            [1, 0, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 1],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [0, 1, 1],
            [1, 1, 0],
            [1, 0, 0],
        ],
        dtype=bool,
    ),
]

GEN_RETRY_LIMIT = 400


def _all_symmetries(mask: np.ndarray) -> list[np.ndarray]:
    out = []
    for k in range(4):
        rot = np.rot90(mask, k)
        out.append(rot)
        out.append(np.fliplr(rot))
    return out


def _canonical_mask(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    reps = []
    for alt in _all_symmetries(mask):
        reps.append(tuple(tuple(int(v) for v in row) for row in alt.tolist()))
    return min(reps)


def _find_separator(grid: np.ndarray) -> tuple[int, int]:
    grey_components = components_mask(grid == Color.GREY, connectivity=4)
    candidates = []
    for coords in grey_components:
        r0, c0, r1, c1 = bbox_coords(coords)
        if c0 != c1:
            continue
        if int(coords.shape[0]) != r1 - r0 + 1:
            continue
        candidates.append((int(coords.shape[0]), r0, c0, r1))
    assert candidates, "expected a vertical grey separator"
    candidates.sort(reverse=True)
    best = candidates[0]
    if len(candidates) > 1:
        assert best[0] > candidates[1][0], "separator line must be unique"
    height, r0, sep_col, r1 = best
    assert r0 == 0, "panel block must start at the top edge"
    assert height == r1 - r0 + 1
    return sep_col, height


def _extract_sprite(lower: np.ndarray, color: int) -> np.ndarray:
    components = components_mask(lower == color, connectivity=4)
    area_components = [coords for coords in components if 3 <= int(coords.shape[0]) <= 5]
    assert area_components, f"no sprite component found for color {color}"
    area_components.sort(key=lambda coords: int(coords.shape[0]), reverse=True)
    best = area_components[0]
    if len(area_components) > 1:
        assert int(best.shape[0]) > int(area_components[1].shape[0]), f"ambiguous sprite for color {color}"
    r0, c0, r1, c1 = bbox_coords(best)
    comp_mask = np.zeros_like(lower, dtype=bool)
    for r, c in best:
        comp_mask[int(r), int(c)] = True
    sprite = crop_bbox(comp_mask.astype(int), (r0, c0, r1, c1)).astype(bool)
    assert sprite.shape == (3, 3), f"sprite for color {color} must occupy a 3x3 bbox"
    return sprite


def _pair_is_isolated(grid: np.ndarray, r: int, c: int) -> bool:
    H, W = grid.shape
    cells = {(r, c), (r, c + 1)}
    for rr, cc in cells:
        for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
            if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in cells:
                if int(grid[nr, nc]) != Color.BLACK:
                    return False
    return True


def _extract_color_map(lower: np.ndarray) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for r in range(lower.shape[0]):
        for c in range(lower.shape[1] - 1):
            left = int(lower[r, c])
            right = int(lower[r, c + 1])
            if left in (Color.RED, Color.BLUE) and right in (Color.GREEN, Color.YELLOW) and left != right:
                if _pair_is_isolated(lower, r, c):
                    assert left not in mapping, f"duplicate legend mapping for color {left}"
                    mapping[left] = right
    assert mapping == {Color.RED: Color.GREEN, Color.BLUE: Color.YELLOW}, "legend must define both recolorings"
    return mapping


def _stamp_mask(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = int(color)
    return patch


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compare the two top panels and stamp recolored sprites at XOR-only positions.

    Input structure:
    - The background is black, and the top block spans the full grid width as a left red panel, a single gray separator column, and a right blue panel.
    - Below the panels sits a dictionary area containing one red 3x3 sprite shape, one blue 3x3 sprite shape, and two isolated 1x2 legend pairs.
    - Each panel is a binary mask in its own color; some cells are shared by both masks and 3-5 cells are XOR-only, with at least one XOR cell from each side.

    Rule:
    - Find the gray separator to split the aligned left and right panels.
    - Build an output grid the size of one panel; paint every shared foreground cell gray.
    - Extract the red and blue sprites from the lower dictionary and extract the legend mappings red->green and blue->yellow from the isolated horizontal pairs.
    - Scan XOR-only panel cells in row-major order; if the left panel owns a difference cell, stamp the red sprite with the mapped green color at that cell as the sprite bbox top-left, and if the right panel owns it, stamp the blue sprite with the mapped yellow color.

    Key insight:
    - The XOR comparison gives the anchor positions, but the side that owns each XOR cell chooses which sprite to stamp, and the legend chooses the final stamp color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep_col, panel_h = _find_separator(g)
    left = g[:panel_h, :sep_col]
    right = g[:panel_h, sep_col + 1 :]
    assert left.shape == right.shape and left.size > 0, "panels must be aligned and non-empty"

    lower = g[panel_h:, :]
    red_sprite = _extract_sprite(lower, Color.RED)
    blue_sprite = _extract_sprite(lower, Color.BLUE)
    color_map = _extract_color_map(lower)

    out = np.full(left.shape, Color.BLACK, dtype=int)
    common = (left != Color.BLACK) & (right != Color.BLACK)
    out[common] = Color.GREY

    diffs = np.argwhere((left != Color.BLACK) ^ (right != Color.BLACK))
    red_patch = _stamp_mask(red_sprite, color_map[Color.RED])
    blue_patch = _stamp_mask(blue_sprite, color_map[Color.BLUE])
    for r_raw, c_raw in diffs:
        r = int(r_raw)
        c = int(c_raw)
        if int(left[r, c]) != Color.BLACK:
            paste(out, red_patch, r, c, transparent=Color.BLACK)
        else:
            paste(out, blue_patch, r, c, transparent=Color.BLACK)

    return out


def _choose_spaced_anchors(panel_h: int, panel_w: int, count: int) -> list[tuple[int, int]]:
    cells = [(r, c) for r in range(panel_h - 2) for c in range(panel_w - 2)]
    random.shuffle(cells)
    anchors: list[tuple[int, int]] = []
    for r, c in cells:
        if all(max(abs(r - rr), abs(c - cc)) >= 4 for rr, cc in anchors):
            anchors.append((r, c))
            if len(anchors) == count:
                return anchors
    raise RuntimeError("failed to place spaced anchors")


def _sample_common_mask(panel_h: int, panel_w: int, anchors: list[tuple[int, int]]) -> np.ndarray:
    reserved = np.zeros((panel_h, panel_w), dtype=bool)
    for r, c in anchors:
        reserved[r : r + 3, c : c + 3] = True

    available = [(r, c) for r in range(panel_h) for c in range(panel_w) if not reserved[r, c]]
    random.shuffle(available)
    min_common = 2
    max_common = max(min_common, min(len(available), panel_h + panel_w))
    count = random.randint(min_common, max_common)
    chosen = available[:count]
    mask = np.zeros((panel_h, panel_w), dtype=bool)
    for r, c in chosen:
        mask[r, c] = True
    assert int(mask.sum()) >= min_common
    return mask


def _sample_sprite_pair() -> tuple[np.ndarray, np.ndarray]:
    for _ in range(40):
        red_base, blue_base = random.sample(SPRITE_CATALOG, 2)
        red_sprite = random.choice(_all_symmetries(red_base)).copy()
        blue_sprite = random.choice(_all_symmetries(blue_base)).copy()
        if _canonical_mask(red_sprite) != _canonical_mask(blue_sprite):
            assert red_sprite.shape == (3, 3) and blue_sprite.shape == (3, 3)
            return red_sprite, blue_sprite
    raise RuntimeError("failed to sample two distinct sprite classes")


def _independent_expected(
    left_mask: np.ndarray,
    right_mask: np.ndarray,
    red_sprite: np.ndarray,
    blue_sprite: np.ndarray,
) -> np.ndarray:
    out = np.full(left_mask.shape, Color.BLACK, dtype=int)
    out[left_mask & right_mask] = Color.GREY
    for r_raw, c_raw in np.argwhere(left_mask ^ right_mask):
        r = int(r_raw)
        c = int(c_raw)
        if left_mask[r, c]:
            patch = np.zeros((3, 3), dtype=int)
            patch[red_sprite] = Color.GREEN
        else:
            patch = np.zeros((3, 3), dtype=int)
            patch[blue_sprite] = Color.YELLOW
        paste(out, patch, r, c, transparent=Color.BLACK)
    return out


def _render_input(
    panel_h: int,
    panel_w: int,
    gap_rows: int,
    left_mask: np.ndarray,
    right_mask: np.ndarray,
    red_sprite: np.ndarray,
    blue_sprite: np.ndarray,
) -> np.ndarray:
    grid_h = panel_h + gap_rows + 3
    grid_w = 2 * panel_w + 1
    grid = np.full((grid_h, grid_w), Color.BLACK, dtype=int)
    grid[:panel_h, panel_w] = Color.GREY
    grid[:panel_h, :panel_w][left_mask] = Color.RED
    grid[:panel_h, panel_w + 1 :][right_mask] = Color.BLUE

    dict_top = panel_h + gap_rows
    red_c = random.randint(0, 1)
    blue_c = random.randint(red_c + 5, red_c + 7)
    legend_c = random.randint(grid_w - 5, grid_w - 3)
    assert blue_c + 3 < legend_c, "dictionary layout must separate sprites from legend"

    grid[dict_top : dict_top + 3, red_c : red_c + 3][red_sprite] = Color.RED
    grid[dict_top : dict_top + 3, blue_c : blue_c + 3][blue_sprite] = Color.BLUE

    legend_rows = random.sample([0, 1, 2], 2)
    grid[dict_top + legend_rows[0], legend_c] = Color.RED
    grid[dict_top + legend_rows[0], legend_c + 1] = Color.GREEN
    grid[dict_top + legend_rows[1], legend_c] = Color.BLUE
    grid[dict_top + legend_rows[1], legend_c + 1] = Color.YELLOW
    return grid


def _gen_attempt() -> np.ndarray | None:
    panel_h = random.randint(7, 10)
    panel_w = random.randint(8, 11)
    diff_count = random.randint(3, 5)
    gap_rows = random.randint(1, 2)

    try:
        anchors = _choose_spaced_anchors(panel_h, panel_w, diff_count)
    except RuntimeError:
        return None

    owners = ["L", "R"]
    while len(owners) < diff_count:
        owners.append(random.choice(["L", "R"]))
    random.shuffle(owners)

    left_mask = _sample_common_mask(panel_h, panel_w, anchors)
    right_mask = left_mask.copy()
    left_only = 0
    right_only = 0
    for (r, c), owner in zip(anchors, owners):
        if owner == "L":
            left_mask[r, c] = True
            right_only += 0
            left_only += 1
        else:
            right_mask[r, c] = True
            right_only += 1
    if left_only == 0 or right_only == 0:
        return None

    red_sprite, blue_sprite = _sample_sprite_pair()
    grid = _render_input(panel_h, panel_w, gap_rows, left_mask, right_mask, red_sprite, blue_sprite)
    expected = _independent_expected(left_mask, right_mask, red_sprite, blue_sprite)

    if grid.shape[0] > 30 or grid.shape[1] > 30 or expected.shape[0] > 30 or expected.shape[1] > 30:
        return None

    if int(np.count_nonzero(expected == Color.GREY)) < 2:
        return None
    if not np.any(expected == Color.GREEN) or not np.any(expected == Color.YELLOW):
        return None

    diff_positions = np.argwhere(left_mask ^ right_mask)
    if diff_positions.shape[0] != diff_count:
        return None
    for i in range(diff_positions.shape[0]):
        r0, c0 = map(int, diff_positions[i])
        if r0 + 3 > panel_h or c0 + 3 > panel_w:
            return None
        for j in range(i + 1, diff_positions.shape[0]):
            r1, c1 = map(int, diff_positions[j])
            if max(abs(r0 - r1), abs(c0 - c1)) < 4:
                return None

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    lower = grid[panel_h:, :]
    if _extract_color_map(lower) != {Color.RED: Color.GREEN, Color.BLUE: Color.YELLOW}:
        return None
    if _extract_sprite(lower, Color.RED).shape != (3, 3) or _extract_sprite(lower, Color.BLUE).shape != (3, 3):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid XOR sprite-stamp input")
