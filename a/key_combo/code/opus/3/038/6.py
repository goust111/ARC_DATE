from __future__ import annotations

import random
from importlib.machinery import SourceFileLoader

import numpy as np


_ARC = SourceFileLoader(
    "arc_utils_shared",
    "/home/lixinyu/snap/new/solvers/arc_utils.py",
).load_module()

Color = _ARC.Color
crop_bbox = _ARC.crop_bbox
find_components = _ARC.find_components
get_bg = _ARC.get_bg
paste = _ARC.paste


_BG = Color.BLACK
_GEN_ATTEMPTS = 500
_PALETTE = [c for c in Color.NOT_BLACK if c != Color.RED]
_N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _cropped_component_mask(shape: tuple[int, int], component: dict) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    coords = component["coords"]
    mask[coords[:, 0], coords[:, 1]] = True
    return crop_bbox(mask.astype(np.uint8), component["bbox"]).astype(bool)


def _find_axis_col(grid: np.ndarray, bg: int) -> int:
    red_components = find_components(
        grid,
        bg=bg,
        colors=[Color.RED],
        connectivity=4,
        monochromatic=True,
    )
    assert len(red_components) == 1, "expected one red axis segment"
    r0, c0, r1, c1 = map(int, red_components[0]["bbox"])
    assert c0 == c1, "red segment must be vertical"
    assert r1 - r0 + 1 >= 3, "red segment must be at least three cells tall"
    return c0


def _find_motif(grid: np.ndarray, axis_col: int, bg: int) -> tuple[int, int, int, np.ndarray]:
    components = find_components(grid, bg=bg, connectivity=4, monochromatic=True)
    left_components = [
        comp
        for comp in components
        if int(comp["color"]) not in {bg, Color.RED}
        and int(comp["area"]) >= 3
        and int(comp["bbox"][3]) < axis_col
    ]

    matches: list[tuple[int, int, int, np.ndarray]] = []
    for i in range(len(left_components)):
        for j in range(i + 1, len(left_components)):
            comp_a = left_components[i]
            comp_b = left_components[j]
            if int(comp_a["color"]) != int(comp_b["color"]):
                continue

            a_r0, a_c0, a_r1, _a_c1 = map(int, comp_a["bbox"])
            b_r0, b_c0, b_r1, _b_c1 = map(int, comp_b["bbox"])
            if (a_r0, a_r1) != (b_r0, b_r1):
                continue

            mask_a = _cropped_component_mask(grid.shape, comp_a)
            mask_b = _cropped_component_mask(grid.shape, comp_b)
            if not np.array_equal(mask_a, mask_b):
                continue

            left_c0, right_c0 = sorted((a_c0, b_c0))
            matches.append((a_r0, left_c0, right_c0 - left_c0, mask_a))

    assert len(matches) == 1, "expected one repeated left-side motif pair"
    row0, _left_col, period, motif_mask = matches[0]
    assert period >= 3
    motif_color = int(left_components[0]["color"])
    return row0, period, motif_color, motif_mask


def _find_marker_color(grid: np.ndarray, axis_col: int, bg: int, motif_color: int) -> int:
    right_side = np.zeros_like(grid, dtype=bool)
    right_side[:, axis_col + 1 :] = True
    marker_coords = np.argwhere(right_side & (grid != bg) & (grid != Color.RED))
    assert marker_coords.shape[0] == 1, "expected one right-side marker pixel"
    r, c = map(int, marker_coords[0])
    marker_color = int(grid[r, c])
    assert marker_color != motif_color
    return marker_color


def _extend_pattern(
    grid: np.ndarray,
    row0: int,
    axis_col: int,
    period: int,
    motif_mask: np.ndarray,
    fill_color: int,
    bg: int,
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    patch = np.zeros(motif_mask.shape, dtype=int)
    patch[np.fliplr(motif_mask)] = int(fill_color)
    for c0 in range(axis_col + 1, out.shape[1], period):
        paste(out, patch, row0, c0, transparent=bg, clip=True)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Continue the left-side motif period past the red axis using mirrored copies painted with the right-side marker color.

    Input structure:
    - The grid has a black background, two identical small 4-connected motifs of one color on the left, one vertical red axis segment, and one isolated marker pixel on the right.
    - The two motif copies share the same row span and their left-edge offset defines a horizontal period.
    - The marker pixel uses a non-black, non-red color different from the motif color.

    Rule:
    - Detect the repeated left motif, crop it to its bounding box, and read the period from the two motifs' left-edge columns.
    - Starting at the first column to the right of the red axis, place copies of the motif every period columns on the original motif rows.
    - Flip each new copy left-right and recolor all motif pixels to the marker pixel's color.
    - Keep the original motifs, red axis, and marker unchanged, clipping any new copy that crosses the right border.

    Key insight:
    - The two left copies tell you how often the pattern repeats, while the red axis changes the continuation rule and the lone right-side marker supplies the post-axis color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    bg = get_bg(g)
    axis_col = _find_axis_col(g, bg)
    row0, period, motif_color, motif_mask = _find_motif(g, axis_col, bg)
    marker_color = _find_marker_color(g, axis_col, bg, motif_color)
    return _extend_pattern(g, row0, axis_col, period, motif_mask, marker_color, bg)


def _sample_motif() -> np.ndarray:
    for _ in range(200):
        target_size = random.randint(3, 5)
        cells = {(0, 0)}
        while len(cells) < target_size:
            frontier = []
            for r, c in cells:
                for dr, dc in _N4:
                    rr, cc = r + dr, c + dc
                    if -2 <= rr <= 2 and -2 <= cc <= 2 and (rr, cc) not in cells:
                        frontier.append((rr, cc))
            cells.add(random.choice(frontier))

        coords = np.array(sorted(cells), dtype=int)
        coords -= coords.min(axis=0)
        h = int(coords[:, 0].max()) + 1
        w = int(coords[:, 1].max()) + 1
        if not (2 <= h <= 4 and 2 <= w <= 4):
            continue

        mask = np.zeros((h, w), dtype=bool)
        mask[coords[:, 0], coords[:, 1]] = True
        if np.array_equal(mask, np.fliplr(mask)):
            continue
        return mask

    raise RuntimeError("failed to sample an asymmetric connected motif")


def _future_copy_mask(
    shape: tuple[int, int],
    row0: int,
    axis_col: int,
    period: int,
    motif_mask: np.ndarray,
) -> np.ndarray:
    h, w = motif_mask.shape
    H, W = shape
    flipped = np.fliplr(motif_mask)
    out = np.zeros(shape, dtype=bool)
    for c0 in range(axis_col + 1, W, period):
        for rr, cc in np.argwhere(flipped):
            r = row0 + int(rr)
            c = c0 + int(cc)
            if 0 <= r < H and 0 <= c < W:
                out[r, c] = True
    return out


def _marker_pools(
    grid: np.ndarray,
    axis_col: int,
    future_mask: np.ndarray,
    row0: int,
    motif_h: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    H, W = grid.shape
    occupied = grid != _BG
    inside: list[tuple[int, int]] = []
    outside: list[tuple[int, int]] = []

    for r in range(H):
        for c in range(axis_col + 1, W):
            if occupied[r, c] or future_mask[r, c]:
                continue

            clear = True
            for rr in range(max(0, r - 1), min(H, r + 2)):
                for cc in range(max(0, c - 1), min(W, c + 2)):
                    if occupied[rr, cc] or future_mask[rr, cc]:
                        clear = False
                        break
                if not clear:
                    break
            if not clear:
                continue

            if row0 <= r < row0 + motif_h:
                inside.append((r, c))
            else:
                outside.append((r, c))

    return inside, outside


def _gen_attempt() -> np.ndarray | None:
    motif_mask = _sample_motif()
    motif_h, motif_w = motif_mask.shape
    period = random.randint(max(3, motif_w + 1), motif_w + 5)
    first_col = random.randint(1, 4)
    gap_before_axis = random.randint(1, 4)
    second_col = first_col + period
    axis_col = second_col + motif_w + gap_before_axis

    two_only_widths = list(range(axis_col + period + motif_w + 1, axis_col + 1 + 2 * period + 1))
    partial_third_widths = list(
        range(axis_col + 1 + 2 * period + 1, axis_col + 1 + 2 * period + motif_w)
    )
    full_third_widths = list(
        range(axis_col + 1 + 2 * period + motif_w, axis_col + 1 + 3 * period + motif_w + 1)
    )
    width_options = {
        "two_only": [w for w in two_only_widths if 16 <= w <= 30],
        "third_partial": [w for w in partial_third_widths if 16 <= w <= 30],
        "third_full": [w for w in full_third_widths if 16 <= w <= 30],
    }
    mode = random.choices(
        ["two_only", "third_partial", "third_full"],
        weights=[2, 4, 3],
        k=1,
    )[0]
    if not width_options[mode]:
        usable = [name for name, values in width_options.items() if values]
        if not usable:
            return None
        mode = random.choice(usable)
    width = random.choice(width_options[mode])

    min_height = max(10, motif_h + 5)
    max_height = min(20, width - 1)
    if min_height > max_height:
        return None
    height = random.randint(min_height, max_height)
    row0 = random.randint(1, height - motif_h - 2)

    top_slack = random.randint(0, min(2, row0))
    bottom_slack = random.randint(0, min(2, height - (row0 + motif_h)))
    axis_r0 = row0 - top_slack
    axis_r1 = row0 + motif_h - 1 + bottom_slack
    if axis_r1 - axis_r0 + 1 < 3:
        if axis_r1 + 1 < height:
            axis_r1 += 1
        elif axis_r0 > 0:
            axis_r0 -= 1
        else:
            return None

    motif_color = random.choice(_PALETTE)
    marker_color = random.choice([c for c in _PALETTE if c != motif_color])
    motif_patch = np.zeros((motif_h, motif_w), dtype=int)
    motif_patch[motif_mask] = motif_color

    grid = np.zeros((height, width), dtype=int)
    paste(grid, motif_patch, row0, first_col, transparent=_BG)
    paste(grid, motif_patch, row0, second_col, transparent=_BG)
    grid[axis_r0 : axis_r1 + 1, axis_col] = Color.RED

    future_mask = _future_copy_mask(grid.shape, row0, axis_col, period, motif_mask)
    inside_pool, outside_pool = _marker_pools(grid, axis_col, future_mask, row0, motif_h)
    prefer_inside = random.random() < 0.5
    if prefer_inside and inside_pool:
        marker_r, marker_c = random.choice(inside_pool)
    elif outside_pool:
        marker_r, marker_c = random.choice(outside_pool)
    elif inside_pool:
        marker_r, marker_c = random.choice(inside_pool)
    else:
        return None
    grid[marker_r, marker_c] = marker_color

    expected = _extend_pattern(grid, row0, axis_col, period, motif_mask, marker_color, _BG)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(actual, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid periodic-mirror ARC input")
