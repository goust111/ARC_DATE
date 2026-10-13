from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np


for _parent in Path(__file__).resolve().parents:
    _arc_utils_dir = _parent / "new" / "solvers"
    if (_arc_utils_dir / "arc_utils.py").exists():
        if str(_arc_utils_dir) not in sys.path:
            sys.path.append(str(_arc_utils_dir))
        break
else:
    raise ImportError("could not locate new/solvers/arc_utils.py")

from arc_utils import Color, bbox, colors_present, crop_bbox, find_components, get_bg, paste


GEN_ATTEMPTS = 400
PANEL_COUNT_RANGE = (3, 4)
PANEL_GAP_RANGE = (1, 2)
REGION_GAP_RANGE = (2, 4)
PANEL_H_RANGE = (4, 6)
PANEL_W_RANGE = (4, 6)
FRAME_H_RANGE = (6, 9)
FRAME_W_RANGE = (6, 9)


def _perimeter_mask(shape: tuple[int, int], box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = map(int, box)
    mask = np.zeros(shape, dtype=bool)
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    return mask


def _paint_perimeter(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = map(int, box)
    value = int(color)
    grid[r0, c0 : c1 + 1] = value
    grid[r1, c0 : c1 + 1] = value
    grid[r0 : r1 + 1, c0] = value
    grid[r0 : r1 + 1, c1] = value


def _overlay_panels(panel_by_color: dict[int, np.ndarray], order: list[int], bg: int) -> np.ndarray:
    shape = next(iter(panel_by_color.values())).shape
    out = np.full(shape, int(bg), dtype=int)
    for color in order:
        patch = panel_by_color[int(color)]
        mask = patch != int(bg)
        out[mask] = patch[mask]
    return out


def _extract_regions(grid: np.ndarray, bg: int) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    comps = find_components(grid, bg=bg, connectivity=4, monochromatic=False)
    frame_patch = None
    panel_patches: dict[int, np.ndarray] = {}
    panel_shape = None

    for comp in comps:
        patch = crop_bbox(grid, comp["bbox"])
        cols = colors_present(patch, exclude={bg})
        if len(cols) == 1:
            color = int(cols[0])
            if panel_shape is None:
                panel_shape = patch.shape
            assert patch.shape == panel_shape
            assert color not in panel_patches
            panel_patches[color] = patch
        else:
            assert frame_patch is None
            frame_patch = patch

    assert frame_patch is not None
    assert len(panel_patches) >= 2
    return frame_patch, panel_patches


def _recover_layer_order(frame_patch: np.ndarray, bg: int) -> list[int]:
    frame_colors = colors_present(frame_patch, exclude={bg})
    perimeters: dict[int, np.ndarray] = {}
    for color in frame_colors:
        box = bbox(frame_patch == int(color))
        assert box is not None
        perimeters[int(color)] = _perimeter_mask(frame_patch.shape, box)

    edges = {int(color): set() for color in frame_colors}
    indeg = {int(color): 0 for color in frame_colors}

    for i, a in enumerate(frame_colors):
        a = int(a)
        for b in frame_colors[i + 1 :]:
            b = int(b)
            inter = perimeters[a] & perimeters[b]
            if not np.any(inter):
                continue
            visible = set(map(int, np.unique(frame_patch[inter]))) - {int(bg)}
            assert len(visible) == 1
            top = next(iter(visible))
            assert top in {a, b}
            bottom = b if top == a else a
            if top not in edges[bottom]:
                edges[bottom].add(top)
                indeg[top] += 1

    order: list[int] = []
    frontier = sorted(color for color, deg in indeg.items() if deg == 0)
    while frontier:
        assert len(frontier) == 1
        cur = frontier.pop()
        order.append(int(cur))
        for nxt in sorted(edges[cur]):
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                frontier.append(nxt)
                frontier.sort()

    assert len(order) == len(frame_colors)
    return order


def _make_panel_mask(height: int, width: int) -> np.ndarray:
    stripe_w = random.randint(1, min(2, width - 1))
    stripe_h = random.randint(1, min(2, height - 1))
    stripe_c = random.randint(0, width - stripe_w)
    stripe_r = random.randint(0, height - stripe_h)

    mask = np.zeros((height, width), dtype=bool)
    mask[:, stripe_c : stripe_c + stripe_w] = True
    mask[stripe_r : stripe_r + stripe_h, :] = True
    return mask


def _make_frame_boxes(count: int) -> tuple[list[tuple[int, int, int, int]], int, int]:
    fh_max = FRAME_H_RANGE[1] if count == 3 else min(FRAME_H_RANGE[1], 8)
    fw_max = FRAME_W_RANGE[1] if count == 3 else min(FRAME_W_RANGE[1], 8)
    frame_h = random.randint(FRAME_H_RANGE[0], fh_max)
    frame_w = random.randint(FRAME_W_RANGE[0], fw_max)
    step_r = random.randint(1, min(2, frame_h - 3))
    step_c = random.randint(1, min(2, frame_w - 3))
    sign_r = random.choice([-1, 1])
    sign_c = random.choice([-1, 1])

    rows = [k * step_r if sign_r > 0 else (count - 1 - k) * step_r for k in range(count)]
    cols = [k * step_c if sign_c > 0 else (count - 1 - k) * step_c for k in range(count)]
    boxes = [(r, c, r + frame_h - 1, c + frame_w - 1) for r, c in zip(rows, cols, strict=False)]
    region_h = frame_h + (count - 1) * step_r
    region_w = frame_w + (count - 1) * step_c
    return boxes, region_h, region_w


def _build_instance() -> tuple[
    np.ndarray,
    np.ndarray,
    list[int],
    dict[int, tuple[int, int, int, int]],
    list[int],
    int,
]:
    count = random.randint(*PANEL_COUNT_RANGE)
    bg = random.choice(Color.ALL_COLORS)
    colors = random.sample([c for c in Color.ALL_COLORS if c != bg], count)
    bottom_to_top = random.sample(colors, len(colors))

    panel_h_max = PANEL_H_RANGE[1] if count == 3 else min(PANEL_H_RANGE[1], 5)
    panel_w_max = PANEL_W_RANGE[1] if count == 3 else min(PANEL_W_RANGE[1], 5)
    panel_h = random.randint(PANEL_H_RANGE[0], panel_h_max)
    panel_w = random.randint(PANEL_W_RANGE[0], panel_w_max)

    panel_masks = []
    seen_masks = set()
    for _ in range(count):
        for _ in range(80):
            mask = _make_panel_mask(panel_h, panel_w)
            key = tuple(mask.astype(int).ravel().tolist())
            if key not in seen_masks:
                seen_masks.add(key)
                panel_masks.append(mask)
                break
        else:
            raise ValueError("failed to sample distinct panel masks")

    panel_by_color: dict[int, np.ndarray] = {}
    for color, mask in zip(colors, panel_masks, strict=False):
        patch = np.full((panel_h, panel_w), int(bg), dtype=int)
        patch[mask] = int(color)
        panel_by_color[int(color)] = patch

    coverage = np.zeros((panel_h, panel_w), dtype=int)
    for patch in panel_by_color.values():
        coverage += (patch != int(bg)).astype(int)
    if int(coverage.max()) < 2 or not bool(np.any(coverage == 1)):
        raise ValueError("panel masks do not create a useful overlap pattern")

    intended = _overlay_panels(panel_by_color, bottom_to_top, bg)
    if len(colors_present(intended, exclude={bg})) < 2:
        raise ValueError("overlay collapsed to a single visible color")

    for _ in range(80):
        panel_order = random.sample(colors, len(colors))
        if panel_order == bottom_to_top:
            continue
        naive = _overlay_panels(panel_by_color, panel_order, bg)
        if not np.array_equal(naive, intended):
            break
    else:
        raise ValueError("failed to find a misleading panel order")

    boxes, frame_region_h, frame_region_w = _make_frame_boxes(count)
    box_by_color = dict(zip(bottom_to_top, boxes, strict=False))

    frame_patch = np.full((frame_region_h, frame_region_w), int(bg), dtype=int)
    for color in bottom_to_top:
        _paint_perimeter(frame_patch, box_by_color[int(color)], int(color))

    panel_gap = random.randint(*PANEL_GAP_RANGE)
    panel_strip_w = count * panel_w + (count - 1) * panel_gap
    region_gap = random.randint(*REGION_GAP_RANGE)
    left_margin = random.randint(1, 2)
    right_margin = random.randint(1, 2)
    width = left_margin + frame_region_w + region_gap + panel_strip_w + right_margin
    if width > 30:
        raise ValueError("grid too wide")

    top_margin = random.randint(1, 3)
    bottom_margin = random.randint(1, 3)
    height = max(frame_region_h, panel_h) + top_margin + bottom_margin + random.randint(0, 2)
    if height > 30:
        raise ValueError("grid too tall")

    panel_strip = np.full((panel_h, panel_strip_w), int(bg), dtype=int)
    x = 0
    for color in panel_order:
        paste(panel_strip, panel_by_color[int(color)], 0, x, transparent=int(bg))
        x += panel_w + panel_gap

    grid = np.full((height, width), int(bg), dtype=int)
    frame_top = random.randint(0, height - frame_region_h)
    panel_top = random.randint(0, height - panel_h)
    frames_left = random.choice([False, True])
    if frames_left:
        frame_left = left_margin
        panel_left = left_margin + frame_region_w + region_gap
    else:
        panel_left = left_margin
        frame_left = left_margin + panel_strip_w + region_gap

    paste(grid, frame_patch, frame_top, frame_left, transparent=int(bg))
    paste(grid, panel_strip, panel_top, panel_left, transparent=int(bg))
    return grid, intended, bottom_to_top, box_by_color, panel_order, int(bg)


def _validate_instance(
    grid: np.ndarray,
    intended: np.ndarray,
    bottom_to_top: list[int],
    box_by_color: dict[int, tuple[int, int, int, int]],
    panel_order: list[int],
    bg: int,
) -> bool:
    try:
        if int(get_bg(grid)) != int(bg):
            return False

        frame_patch, panel_patches = _extract_regions(grid, bg)
        if set(panel_patches) != set(bottom_to_top):
            return False
        if _recover_layer_order(frame_patch, bg) != [int(c) for c in bottom_to_top]:
            return False

        for color, box in box_by_color.items():
            if bbox(frame_patch == int(color)) != tuple(map(int, box)):
                return False

        naive = _overlay_panels(panel_patches, [int(c) for c in panel_order], bg)
        if np.array_equal(naive, intended):
            return False
        if not np.array_equal(_overlay_panels(panel_patches, bottom_to_top, bg), intended):
            return False
        if not np.array_equal(main(grid), intended):
            return False
        return True
    except AssertionError:
        return False


def main(grid: np.ndarray) -> np.ndarray:
    """Read the frame occlusion order and use it as the transparent-overlay priority for the color panels.

    Input structure:
    - The background is the most common color.
    - One side of the grid contains several 1-cell-thick monochrome rectangular frames that partially overlap.
    - The other side contains separated equal-size rectangular panels; each panel uses only the background and one
      foreground color, and the panel colors match the frame colors one-to-one.

    Rule:
    - Recover each frame's full perimeter from the bbox of its visible color.
    - At every reconstructed perimeter crossing, the visible color is the upper frame; these pairwise cover relations
      form a unique bottom-to-top total order over the frame colors.
    - Treat the matching panels as transparent layers and overlay them in that recovered order; the highest-priority
      non-background color wins at each pixel.

    Key insight:
    - The left-to-right panel arrangement is only a placement decoy; the real overlay order is written elsewhere by
      the frame cluster's crossing visibility.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    frame_patch, panel_patches = _extract_regions(g, bg)
    order = _recover_layer_order(frame_patch, bg)
    assert set(order) == set(panel_patches)
    return _overlay_panels(panel_patches, order, bg)


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        try:
            grid, intended, bottom_to_top, box_by_color, panel_order, bg = _build_instance()
        except (AssertionError, ValueError):
            continue
        if _validate_instance(grid, intended, bottom_to_top, box_by_color, panel_order, bg):
            return grid
    raise RuntimeError("failed to generate a valid frame-layer overlay task")
