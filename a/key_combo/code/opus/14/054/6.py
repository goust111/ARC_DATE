from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import crop_bbox, find_components, get_bg


BG = 0
FRAME_COLORS = (4, 5, 8)
SPRITE_COLORS = (1, 2, 3, 6, 7, 9)
WIDTH_TRIPLES = [
    widths
    for widths in (
        (3, 4, 5),
        (3, 4, 6),
        (3, 4, 7),
        (3, 4, 8),
        (3, 5, 6),
        (3, 5, 7),
        (3, 5, 8),
        (3, 6, 7),
        (3, 6, 8),
        (4, 5, 6),
        (4, 5, 7),
        (4, 5, 8),
        (4, 6, 7),
        (4, 6, 8),
        (5, 6, 7),
    )
]
AREA_CHOICES = (3, 4, 5, 6)
HEIGHT_CHOICES = (8, 9)
MAX_ATTEMPTS = 400

_SEEN_OUTPUTS: set[bytes] = set()


def _u_cells(bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = map(int, bbox)
    cells = {(r, c0) for r in range(r0, r1 + 1)}
    cells |= {(r, c1) for r in range(r0, r1 + 1)}
    cells |= {(r1, c) for c in range(c0, c1 + 1)}
    return cells


def _is_u_bracket(component: dict) -> bool:
    r0, c0, r1, c1 = map(int, component["bbox"])
    if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 5:
        return False
    coords = {tuple(map(int, rc)) for rc in component["coords"]}
    return coords == _u_cells(component["bbox"])


def _extract_bracket_instances(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    by_color: dict[int, list[dict]] = {}
    bracket_candidates: list[tuple[int, list[dict]]] = []

    for color in sorted(set(map(int, np.unique(g))) - {bg}):
        comps = find_components(
            g,
            bg=bg,
            colors=[color],
            connectivity=4,
            monochromatic=True,
            return_format="dict",
        )
        by_color[int(color)] = comps
        if len(comps) >= 3 and all(_is_u_bracket(comp) for comp in comps):
            heights = {int(comp["bbox"][2] - comp["bbox"][0] + 1) for comp in comps}
            inner_widths = {int(comp["bbox"][3] - comp["bbox"][1] - 1) for comp in comps}
            if len(heights) == 1 and len(inner_widths) == len(comps):
                bracket_candidates.append((int(color), comps))

    assert len(bracket_candidates) == 1, "expected one bracket color with multiple U-frames"
    frame_color, brackets = bracket_candidates[0]

    sprite_components = [
        comp
        for color, comps in by_color.items()
        if color != frame_color
        for comp in comps
    ]
    assert len(sprite_components) == len(brackets), "expected one sprite per bracket"

    instances = []
    used = set()
    for bracket in brackets:
        r0, c0, r1, c1 = map(int, bracket["bbox"])
        inside = []
        for idx, sprite in enumerate(sprite_components):
            if idx in used:
                continue
            coords = sprite["coords"]
            if np.all((coords[:, 0] >= r0) & (coords[:, 0] < r1) & (coords[:, 1] > c0) & (coords[:, 1] < c1)):
                inside.append((idx, sprite))
        assert len(inside) == 1, "each bracket must contain exactly one sprite"
        idx, sprite = inside[0]
        used.add(idx)
        instances.append(
            {
                "frame_color": frame_color,
                "frame_bbox": tuple(map(int, bracket["bbox"])),
                "inner_width": int(c1 - c0 - 1),
                "sprite_color": int(sprite["color"]),
                "sprite_area": int(sprite["area"]),
                "sprite_bbox": tuple(map(int, sprite["bbox"])),
                "sprite_patch": crop_bbox(g, sprite["bbox"]),
            }
        )

    assert len(used) == len(sprite_components)
    return instances


def _nested_rings(colors: list[int]) -> np.ndarray:
    size = 2 * len(colors) - 1
    out = np.full((size, size), BG, dtype=int)
    for layer, color in enumerate(colors):
        out[layer : size - layer, layer : size - layer] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Measure each U-bracket's inner width, sort the enclosed sprite colors by that width, and compress them into nested square rings.

    Input structure:
    - The grid has a black background and several same-colored 1-pixel-thick U-shaped brackets open upward.
    - The brackets share one height, have pairwise different inner widths, and are separated horizontally.
    - Each bracket contains exactly one small monochrome 4-connected sprite in its interior.

    Rule:
    - Detect the U-shaped brackets and measure each bracket's inner width between its two vertical walls.
    - Crop the sprite inside each bracket to its own bounding box, but keep only its color as the payload.
    - Sort the enclosed sprites by their bracket inner widths from largest to smallest.
    - Build a square output of size (2N-1)x(2N-1), where N is the number of brackets, and fill concentric square rings from outside to inside with the sorted sprite colors.

    Key insight:
    - The bracket is only a geometric ruler: its inner width determines ring order, while the sprite contributes only its color and not its original shape.
    """
    instances = _extract_bracket_instances(np.asarray(grid, dtype=int))
    ordered = sorted(instances, key=lambda item: item["inner_width"], reverse=True)
    return _nested_rings([int(item["sprite_color"]) for item in ordered])


def _random_connected_shape(area: int, max_h: int, max_w: int) -> np.ndarray:
    for _ in range(200):
        h = random.randint(max(1, (area + max_w - 1) // max_w), max_h)
        w = random.randint(max(1, (area + h - 1) // h), max_w)
        start = (random.randrange(h), random.randrange(w))
        cells = {start}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in cells:
                        frontier.append((rr, cc))
            if not frontier:
                break
            cells.add(random.choice(frontier))
        if len(cells) != area:
            continue
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        r0, c0 = min(rows), min(cols)
        norm = {(r - r0, c - c0) for r, c in cells}
        hh = max(r for r, _ in norm) + 1
        ww = max(c for _, c in norm) + 1
        shape = np.zeros((hh, ww), dtype=int)
        for r, c in norm:
            shape[r, c] = 1
        return shape
    raise RuntimeError("failed to build connected sprite")


def _paint_bracket(grid: np.ndarray, top: int, left: int, height: int, inner_width: int, color: int) -> tuple[int, int, int, int]:
    total_w = inner_width + 2
    bottom = top + height - 1
    right = left + total_w - 1
    grid[top : bottom + 1, left] = color
    grid[top : bottom + 1, right] = color
    grid[bottom, left : right + 1] = color
    return top, left, bottom, right


def _place_sprite_in_bracket(
    grid: np.ndarray,
    bracket_bbox: tuple[int, int, int, int],
    sprite_color: int,
    target_area: int,
) -> tuple[int, int]:
    r0, c0, r1, c1 = map(int, bracket_bbox)
    usable_h = r1 - r0 - 1
    usable_w = c1 - c0 - 3
    assert usable_h >= 3 and usable_w >= 1
    sprite = _random_connected_shape(target_area, usable_h, usable_w)
    sh, sw = sprite.shape
    bottom = r1 - 2
    top = bottom - sh + 1
    left = c0 + 2 + random.randint(0, usable_w - sw)
    patch = np.full(sprite.shape, BG, dtype=int)
    patch[sprite == 1] = int(sprite_color)
    grid[top : top + sh, left : left + sw][sprite == 1] = int(sprite_color)
    return int(np.count_nonzero(sprite)), int(sw)


def _distribute(total: int, slots: int) -> list[int]:
    cuts = sorted(random.sample(range(total + slots - 1), slots - 1))
    prev = -1
    parts = []
    for cut in cuts + [total + slots - 1]:
        parts.append(cut - prev - 1)
        prev = cut
    return parts


def _layout_positions(inner_widths: list[int], grid_w: int) -> list[int]:
    total_widths = [w + 2 for w in inner_widths]
    min_used = sum(total_widths) + 2 * (len(total_widths) - 1) + 2
    assert min_used <= grid_w
    extra = grid_w - min_used
    left_extra, gap1_extra, gap2_extra, right_extra = _distribute(extra, 4)
    left = 1 + left_extra
    gap1 = 2 + gap1_extra
    gap2 = 2 + gap2_extra
    positions = [left]
    positions.append(positions[-1] + total_widths[0] + gap1)
    positions.append(positions[-1] + total_widths[1] + gap2)
    assert positions[-1] + total_widths[2] + 1 + right_extra == grid_w
    return positions


def _semantic_checks(grid: np.ndarray, meta: list[dict]) -> None:
    instances = _extract_bracket_instances(grid)
    got = sorted(
        [(item["inner_width"], item["sprite_color"], item["sprite_area"]) for item in instances],
        reverse=True,
    )
    want = sorted(
        [(item["inner_width"], item["sprite_color"], item["sprite_area"]) for item in meta],
        reverse=True,
    )
    assert got == want

    widths = [int(item["inner_width"]) for item in meta]
    left_order = [int(item["left"]) for item in meta]
    areas = [int(item["sprite_area"]) for item in meta]

    width_rank = sorted(range(len(meta)), key=lambda i: widths[i], reverse=True)
    left_rank = sorted(range(len(meta)), key=lambda i: left_order[i])
    area_rank = sorted(range(len(meta)), key=lambda i: areas[i], reverse=True)
    assert width_rank != left_rank, "generator must defeat left-to-right sorting"
    assert width_rank != area_rank, "generator must defeat sprite-area sorting"

    out = main(grid)
    expected_colors = [meta[i]["sprite_color"] for i in width_rank]
    assert np.array_equal(out, _nested_rings(expected_colors))


def _generate_one() -> np.ndarray | None:
    frame_color = random.choice(FRAME_COLORS)
    sprite_colors = random.sample(SPRITE_COLORS, 3)
    inner_widths = list(random.choice(WIDTH_TRIPLES))
    permuted_widths = list(random.sample(inner_widths, len(inner_widths)))
    while permuted_widths == sorted(inner_widths, reverse=True) or permuted_widths == sorted(inner_widths):
        permuted_widths = list(random.sample(inner_widths, len(inner_widths)))

    bracket_h = random.choice(HEIGHT_CHOICES)
    total_min_w = sum(w + 2 for w in permuted_widths) + 2 * (len(permuted_widths) - 1) + 2
    grid_w = random.randint(total_min_w, 30)
    grid_h = random.randint(bracket_h + 3, min(18, 30))
    top = random.randint(1, grid_h - bracket_h - 1)

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    lefts = _layout_positions(permuted_widths, grid_w)

    width_desc = sorted(range(3), key=lambda i: permuted_widths[i], reverse=True)
    areas = random.sample(AREA_CHOICES, 3)
    while sorted(range(3), key=lambda i: areas[i], reverse=True) == width_desc:
        areas = random.sample(AREA_CHOICES, 3)

    meta = []
    for idx, (inner_w, left, sprite_color, area) in enumerate(zip(permuted_widths, lefts, sprite_colors, areas)):
        bbox = _paint_bracket(grid, top, left, bracket_h, inner_w, frame_color)
        actual_area, sprite_bbox_w = _place_sprite_in_bracket(grid, bbox, sprite_color, area)
        meta.append(
            {
                "left": left,
                "inner_width": inner_w,
                "sprite_color": sprite_color,
                "sprite_area": actual_area,
                "sprite_bbox_w": sprite_bbox_w,
            }
        )

    if len({item["sprite_color"] for item in meta}) != 3:
        return None

    try:
        _semantic_checks(grid, meta)
    except AssertionError:
        return None

    out = main(grid)
    out_sig = out.tobytes() + str(out.shape).encode()
    if out_sig in _SEEN_OUTPUTS:
        return None
    _SEEN_OUTPUTS.add(out_sig)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _generate_one()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate puzzle 6 input")
