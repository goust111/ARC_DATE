from __future__ import annotations

import itertools
import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, bbox_h, bbox_w, crop_bbox, find_components, get_bg, paste, scale_sprite


SCALE_BY_RANK = (3, 2, 1)
GRID_MIN = 18
GRID_MAX = 24
SPRITE_MARGIN = 2
GEN_RETRIES = 800
SPRITE_COLORS = [Color.GREEN, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON]

# 4-connected monochrome sprites with bbox exactly 2x2 or 2x3.
SHAPE_LIBRARY = [
    np.array([[1, 1], [1, 0]], dtype=int),
    np.array([[1, 1], [0, 1]], dtype=int),
    np.array([[1, 0], [1, 1]], dtype=int),
    np.array([[0, 1], [1, 1]], dtype=int),
    np.array([[1, 1], [1, 1]], dtype=int),
    np.array([[1, 1, 1], [1, 0, 0]], dtype=int),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=int),
    np.array([[1, 1, 1], [0, 0, 1]], dtype=int),
    np.array([[1, 1, 1], [1, 1, 0]], dtype=int),
    np.array([[1, 1, 1], [1, 0, 1]], dtype=int),
    np.array([[1, 1, 1], [0, 1, 1]], dtype=int),
    np.array([[1, 0, 0], [1, 1, 1]], dtype=int),
    np.array([[0, 0, 1], [1, 1, 1]], dtype=int),
    np.array([[1, 1, 1], [1, 1, 1]], dtype=int),
]


def _corner_name_to_coord(name: str, shape: tuple[int, int]) -> tuple[int, int]:
    h, w = map(int, shape)
    if name == "tl":
        return 0, 0
    if name == "tr":
        return 0, w - 1
    if name == "bl":
        return h - 1, 0
    if name == "br":
        return h - 1, w - 1
    raise ValueError(f"unknown corner {name!r}")


def _locate_axes(grid: np.ndarray) -> tuple[tuple[int, int], bool]:
    yellow = np.argwhere(grid == Color.YELLOW)
    assert yellow.shape == (1, 2), "expected a unique yellow axis corner"
    corner = tuple(map(int, yellow[0]))
    h, w = map(int, grid.shape)
    cr, cc = corner
    assert cr in (0, h - 1) and cc in (0, w - 1), "yellow corner must lie on a grid corner"

    hvals = {int(grid[cr, c]) for c in range(w) if c != cc}
    vvals = {int(grid[r, cc]) for r in range(h) if r != cr}
    assert len(hvals) == 1 and len(vvals) == 1, "expected one uniform horizontal axis and one uniform vertical axis"

    hcol = next(iter(hvals))
    vcol = next(iter(vvals))
    assert {hcol, vcol} == {Color.BLUE, Color.RED}, "axis lines must use blue and red"
    return corner, vcol == Color.BLUE


def _sprite_components(grid: np.ndarray, bg: int) -> list[dict]:
    sprite_colors = [
        int(color)
        for color in np.unique(grid)
        if int(color) not in {int(bg), Color.BLUE, Color.RED, Color.YELLOW}
    ]
    assert len(sprite_colors) == 3, "expected exactly three sprite colors"

    comps = find_components(
        grid,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        colors=sprite_colors,
    )
    assert len(comps) == 3, "expected exactly three detached sprites"

    sprites: list[dict] = []
    for comp in comps:
        box = tuple(map(int, comp["bbox"]))
        color = int(comp["color"])
        patch = crop_bbox(grid, box)
        patch = np.where(patch == color, color, bg).astype(int)
        assert bbox_h(box) == 2 and bbox_w(box) in (2, 3), "sprite bbox must be 2x2 or 2x3"
        assert 3 <= int(comp["area"]) <= 6, "sprite size must stay within the puzzle range"
        sprites.append({"bbox": box, "patch": patch, "color": color})

    assert len({sprite["color"] for sprite in sprites}) == 3, "sprite colors must be unique"
    return sprites


def _distance_metrics(box: tuple[int, int, int, int], corner: tuple[int, int]) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = map(int, box)
    cr, cc = map(int, corner)
    row_metric = abs((r0 + r1) - 2 * cr)
    col_metric = abs((c0 + c1) - 2 * cc)
    center_metric = row_metric + col_metric
    top_left_metric = abs(r0 - cr) + abs(c0 - cc)
    return row_metric, col_metric, center_metric, top_left_metric


def _stack_patches(patches: list[np.ndarray], bg: int) -> np.ndarray:
    out_h = sum(int(patch.shape[0]) for patch in patches)
    out_w = max(int(patch.shape[1]) for patch in patches)
    out = np.full((out_h, out_w), int(bg), dtype=int)
    r0 = 0
    for patch in patches:
        paste(out, patch, r0, 0, transparent=bg)
        r0 += int(patch.shape[0])
    return out


def _build_output(
    sprites: list[dict],
    corner: tuple[int, int],
    blue_vertical: bool,
    bg: int,
) -> tuple[np.ndarray, tuple[int, int, int]]:
    ranked = []
    for idx, sprite in enumerate(sprites):
        row_metric, col_metric, center_metric, top_left_metric = _distance_metrics(sprite["bbox"], corner)
        ranked.append(
            {
                "index": idx,
                "patch": sprite["patch"],
                "bbox": sprite["bbox"],
                "row_metric": row_metric,
                "col_metric": col_metric,
                "center_metric": center_metric,
                "top_left_metric": top_left_metric,
            }
        )

    ranked.sort(key=lambda item: (item["center_metric"], item["row_metric"], item["col_metric"], item["index"]))
    order = tuple(int(item["index"]) for item in ranked)
    patches: list[np.ndarray] = []
    for item, scale in zip(ranked, SCALE_BY_RANK):
        patch = scale_sprite(item["patch"], scale)
        if blue_vertical:
            patch = np.rot90(patch, 1)
        patches.append(patch.astype(int))
    return _stack_patches(patches, bg), order


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort the three interior sprites by distance to the marked axis corner, magnify them, optionally rotate them, and stack them.

    Input structure:
    - The background is black.
    - A blue edge line and a red edge line lie on one horizontal border and one vertical border, meeting at a yellow corner marker.
    - Three detached monochrome sprites sit in the interior away from the axes, each with a 2x2 or 2x3 bounding box.

    Rule:
    - Find the yellow corner and determine whether the blue axis is horizontal or vertical.
    - For each sprite, compute the Manhattan distance from its bounding-box center to the yellow corner, then sort the sprites from nearest to farthest.
    - Magnify the nearest, middle, and farthest sprites by 3x, 2x, and 1x respectively; rotate every magnified sprite 90 degrees counterclockwise iff the blue axis is vertical; finally stack the three results top-to-bottom with no blank rows and left alignment.

    Key insight:
    - The corner-distance ranking controls two output decisions at once: stack order and magnification, while the blue-axis orientation independently selects whether the whole stack uses the rotated view.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = int(get_bg(g))
    assert bg == Color.BLACK, "this puzzle uses black background"
    corner, blue_vertical = _locate_axes(g)
    sprites = _sprite_components(g, bg)
    out, _order = _build_output(sprites, corner, blue_vertical, bg)
    return out


def _box_from_origin(r0: int, c0: int, width: int) -> tuple[int, int, int, int]:
    return int(r0), int(c0), int(r0 + 1), int(c0 + width - 1)


def _boxes_clear(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1) -> bool:
    return (
        int(a[2]) + gap < int(b[0])
        or int(b[2]) + gap < int(a[0])
        or int(a[3]) + gap < int(b[1])
        or int(b[3]) + gap < int(a[1])
    )


def _make_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask.astype(bool)] = int(color)
    return patch


def _all_candidate_origins(height: int, width: int, sprite_width: int) -> list[tuple[int, int]]:
    row_stop = height - SPRITE_MARGIN - 2
    col_stop = width - SPRITE_MARGIN - sprite_width
    return [
        (r0, c0)
        for r0 in range(SPRITE_MARGIN, row_stop + 1)
        for c0 in range(SPRITE_MARGIN, col_stop + 1)
    ]


def _sample_specs(height: int, width: int, corner: tuple[int, int]) -> list[dict] | None:
    for _ in range(80):
        masks = random.sample(SHAPE_LIBRARY, 3)
        widths = [int(mask.shape[1]) for mask in masks]
        if set(widths) != {2, 3}:
            continue

        colors = random.sample(SPRITE_COLORS, 3)
        order = list(range(3))
        random.shuffle(order)
        placed: list[dict | None] = [None, None, None]
        boxes: list[tuple[int, int, int, int]] = []

        ok = True
        for idx in order:
            candidates = _all_candidate_origins(height, width, widths[idx])
            random.shuffle(candidates)
            chosen = None
            for r0, c0 in candidates:
                box = _box_from_origin(r0, c0, widths[idx])
                if all(_boxes_clear(box, other) for other in boxes):
                    chosen = (r0, c0, box)
                    break
            if chosen is None:
                ok = False
                break
            r0, c0, box = chosen
            patch = _make_patch(masks[idx], colors[idx])
            placed[idx] = {
                "bbox": box,
                "patch": patch,
                "color": int(colors[idx]),
                "width": int(widths[idx]),
            }
            boxes.append(box)

        if not ok:
            continue

        specs = [item for item in placed if item is not None]
        if len(specs) != 3:
            continue

        metrics = [_distance_metrics(spec["bbox"], corner) for spec in specs]
        row_metrics = [metric[0] for metric in metrics]
        col_metrics = [metric[1] for metric in metrics]
        center_metrics = [metric[2] for metric in metrics]
        top_left_metrics = [metric[3] for metric in metrics]

        if len(set(row_metrics)) != 3 or len(set(col_metrics)) != 3 or len(set(center_metrics)) != 3:
            continue

        center_order = tuple(sorted(range(3), key=lambda i: center_metrics[i]))
        row_order = tuple(sorted(range(3), key=lambda i: row_metrics[i]))
        col_order = tuple(sorted(range(3), key=lambda i: col_metrics[i]))
        if center_order == row_order or center_order == col_order:
            continue

        tied_top_left_pair = any(
            top_left_metrics[i] == top_left_metrics[j] and specs[i]["width"] != specs[j]["width"]
            for i, j in itertools.combinations(range(3), 2)
        )
        if not tied_top_left_pair:
            continue

        return specs
    return None


def _render_input(
    height: int,
    width: int,
    corner: tuple[int, int],
    blue_vertical: bool,
    specs: list[dict],
) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    cr, cc = corner
    hcol = Color.RED if blue_vertical else Color.BLUE
    vcol = Color.BLUE if blue_vertical else Color.RED
    grid[cr, :] = int(hcol)
    grid[:, cc] = int(vcol)
    grid[cr, cc] = Color.YELLOW

    for spec in specs:
        r0, c0, _r1, _c1 = map(int, spec["bbox"])
        paste(grid, spec["patch"], r0, c0, transparent=Color.BLACK)
    return grid


def _config_output(
    specs: list[dict],
    order: tuple[int, int, int],
    scales: tuple[int, int, int],
    rotate: bool,
) -> np.ndarray:
    patches: list[np.ndarray] = []
    for sprite_idx, scale in zip(order, scales):
        patch = scale_sprite(specs[sprite_idx]["patch"], scale)
        if rotate:
            patch = np.rot90(patch, 1)
        patches.append(patch.astype(int))
    return _stack_patches(patches, Color.BLACK)


def _semantic_checks(
    specs: list[dict],
    corner: tuple[int, int],
    blue_vertical: bool,
    expected: np.ndarray,
) -> bool:
    metrics = [_distance_metrics(spec["bbox"], corner) for spec in specs]
    center_metrics = [metric[2] for metric in metrics]
    correct_order = tuple(sorted(range(3), key=lambda i: center_metrics[i]))

    # Distinguish the intended rule from alternative order/scale/rotation choices.
    alt_scales = list(itertools.permutations(SCALE_BY_RANK))
    alt_scales.extend([(1, 1, 1), (2, 2, 2), (3, 3, 3)])
    for order in itertools.permutations(range(3)):
        for scales in alt_scales:
            for rotate in (False, True):
                if order == correct_order and tuple(scales) == SCALE_BY_RANK and rotate == blue_vertical:
                    continue
                candidate = _config_output(specs, tuple(order), tuple(scales), rotate)
                if candidate.shape == expected.shape and np.array_equal(candidate, expected):
                    return False

    # Rotation must matter whenever the blue axis is vertical.
    if blue_vertical:
        unrotated = _config_output(specs, correct_order, SCALE_BY_RANK, False)
        if unrotated.shape == expected.shape and np.array_equal(unrotated, expected):
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(GRID_MIN, GRID_MAX)
    width = random.randint(GRID_MIN, GRID_MAX)
    blue_vertical = bool(random.getrandbits(1))
    corner = _corner_name_to_coord(random.choice(("tl", "tr", "bl", "br")), (height, width))

    specs = _sample_specs(height, width, corner)
    if specs is None:
        return None

    inp = _render_input(height, width, corner, blue_vertical, specs)
    expected, order = _build_output(specs, corner, blue_vertical, Color.BLACK)

    if expected.ndim != 2 or max(expected.shape) > 30:
        return None
    if tuple(order) != tuple(sorted(range(3), key=lambda i: _distance_metrics(specs[i]["bbox"], corner)[2])):
        return None
    if not _semantic_checks(specs, corner, blue_vertical, expected):
        return None

    try:
        solved = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid distance-ordered scaling input")
