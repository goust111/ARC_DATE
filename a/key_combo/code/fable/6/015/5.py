from __future__ import annotations

import itertools
import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, find_components, get_bg, in_bounds, is_contiguous


KINDS = ("vertical", "horizontal", "point")
MASK_BBOXES = ((2, 2), (2, 3), (3, 2), (3, 3), (2, 4), (4, 2))
MAX_GENERATION_ATTEMPTS = 500
MAX_OBJECT_PLACEMENT_ATTEMPTS = 250
MAX_MARKER_ATTEMPTS = 250


def _edge_covered(mask: np.ndarray) -> bool:
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _enumerate_masks() -> tuple[np.ndarray, ...]:
    masks: list[np.ndarray] = []
    for h, w in MASK_BBOXES:
        upper_area = min(6, h * w - 1)
        for area in range(3, upper_area + 1):
            for cells in itertools.combinations(range(h * w), area):
                mask = np.zeros((h, w), dtype=bool)
                for cell in cells:
                    mask[cell // w, cell % w] = True
                if not _edge_covered(mask):
                    continue
                if not is_contiguous(mask, connectivity=8):
                    continue
                masks.append(mask)
    return tuple(masks)


MASKS = _enumerate_masks()


def _mask_coords(mask: np.ndarray, r0: int, c0: int) -> tuple[tuple[int, int], ...]:
    return tuple((int(r0 + r), int(c0 + c)) for r, c in np.argwhere(mask))


def _marker_rule(markers: tuple[tuple[int, int], tuple[int, int]]) -> tuple[str, int | tuple[int, int]]:
    (r0, c0), (r1, c1) = markers
    if r0 == r1:
        assert c0 != c1 and abs(c0 - c1) % 2 == 0
        return "vertical", (c0 + c1) // 2
    if c0 == c1:
        assert r0 != r1 and abs(r0 - r1) % 2 == 0
        return "horizontal", (r0 + r1) // 2
    assert abs(r0 - r1) % 2 == 0 and abs(c0 - c1) % 2 == 0
    return "point", ((r0 + r1) // 2, (c0 + c1) // 2)


def _transform_coords(
    coords: tuple[tuple[int, int], ...],
    kind: str,
    param: int | tuple[int, int],
) -> tuple[tuple[int, int], ...]:
    if kind == "vertical":
        axis_c = int(param)
        return tuple((int(r), 2 * axis_c - int(c)) for r, c in coords)
    if kind == "horizontal":
        axis_r = int(param)
        return tuple((2 * axis_r - int(r), int(c)) for r, c in coords)
    center_r, center_c = map(int, param)
    return tuple((2 * center_r - int(r), 2 * center_c - int(c)) for r, c in coords)


def _expand_cells(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    H, W = map(int, shape)
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < H and 0 <= cc < W:
                    out.add((rr, cc))
    return out


def _parse_objects(grid: np.ndarray) -> tuple[int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    bg = int(get_bg(g))
    comps = find_components(g, bg=bg, connectivity=8, monochromatic=True)

    by_color: dict[int, list[dict]] = {}
    for comp in comps:
        by_color.setdefault(int(comp["color"]), []).append(comp)

    objects: list[dict] = []
    for color, color_comps in sorted(by_color.items()):
        singletons = [comp for comp in color_comps if int(comp["area"]) == 1]
        sprites = [comp for comp in color_comps if int(comp["area"]) > 1]
        assert len(singletons) == 2, "each color must have exactly two marker pixels"
        assert len(sprites) == 1, "each color must have exactly one sprite body"

        marker_coords = sorted(tuple(map(int, comp["coords"][0])) for comp in singletons)
        markers = (marker_coords[0], marker_coords[1])
        kind, param = _marker_rule(markers)
        sprite_coords = tuple(sorted((int(r), int(c)) for r, c in sprites[0]["coords"]))
        objects.append(
            {
                "color": int(color),
                "markers": markers,
                "kind": kind,
                "param": param,
                "sprite": sprite_coords,
            }
        )

    assert objects, "expected at least one colored object"
    return bg, objects


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Mirror each monochrome sprite using the implicit axis encoded by its same-color marker pair.

    Input structure:
    - The background is the most common color.
    - Each non-background color has one sprite body and exactly two isolated same-color marker pixels.
    - If the markers share a row, their midpoint column defines a vertical mirror axis.
    - If the markers share a column, their midpoint row defines a horizontal mirror axis.
    - Otherwise their midpoint defines a point reflection center.

    Rule:
    - Keep the original sprite.
    - Draw its transformed copy in the same color.
    - Remove the two marker pixels.
    """
    g = np.asarray(input_grid, dtype=int)
    bg, objects = _parse_objects(g)

    out = g.copy()
    for obj in objects:
        for r, c in obj["markers"]:
            out[r, c] = bg

    H, W = map(int, g.shape)
    for obj in objects:
        for r, c in _transform_coords(obj["sprite"], obj["kind"], obj["param"]):
            assert in_bounds((H, W), r, c), "transformed copy left the grid"
            out[r, c] = int(obj["color"])
    return out


def _choose_kinds(count: int) -> list[str]:
    kinds = list(KINDS[: min(count, len(KINDS))])
    random.shuffle(kinds)
    while len(kinds) < count:
        kinds.append(random.choice(KINDS))
    random.shuffle(kinds)
    return kinds


def _vertical_col_range(W: int, w: int, axis_c: int, side: str) -> tuple[int, int] | None:
    if side == "left":
        low = max(0, 2 * axis_c - (W - 1))
        high = min(W - w, axis_c - w)
    else:
        low = max(0, axis_c + 1)
        high = min(W - w, 2 * axis_c)
    if low > high:
        return None
    return low, high


def _horizontal_row_range(H: int, h: int, axis_r: int, side: str) -> tuple[int, int] | None:
    if side == "above":
        low = max(0, 2 * axis_r - (H - 1))
        high = min(H - h, axis_r - h)
    else:
        low = max(0, axis_r + 1)
        high = min(H - h, 2 * axis_r)
    if low > high:
        return None
    return low, high


def _point_range(limit: int, span: int, center: int, forward: bool) -> tuple[int, int] | None:
    if forward:
        low = max(0, center + 1)
        high = min(limit - span, 2 * center)
    else:
        low = max(0, 2 * center - limit + 1)
        high = min(limit - span, center - span)
    if low > high:
        return None
    return low, high


def _sample_object(
    H: int,
    W: int,
    mask: np.ndarray,
    kind: str,
    input_occ: set[tuple[int, int]],
    output_occ: set[tuple[int, int]],
) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...], tuple[tuple[int, int], tuple[int, int]]] | None:
    h, w = map(int, mask.shape)
    input_halo = _expand_cells(input_occ, (H, W))

    for _ in range(MAX_OBJECT_PLACEMENT_ATTEMPTS):
        if kind == "vertical":
            axis_c = random.randint(0, W - 1)
            side = random.choice(("left", "right"))
            col_range = _vertical_col_range(W, w, axis_c, side)
            if col_range is None:
                continue
            r0 = random.randint(0, H - h)
            c0 = random.randint(col_range[0], col_range[1])
            sprite = _mask_coords(mask, r0, c0)
            copied = _transform_coords(sprite, kind, axis_c)
            param = axis_c
        elif kind == "horizontal":
            axis_r = random.randint(0, H - 1)
            side = random.choice(("above", "below"))
            row_range = _horizontal_row_range(H, h, axis_r, side)
            if row_range is None:
                continue
            r0 = random.randint(row_range[0], row_range[1])
            c0 = random.randint(0, W - w)
            sprite = _mask_coords(mask, r0, c0)
            copied = _transform_coords(sprite, kind, axis_r)
            param = axis_r
        else:
            center_r = random.randint(0, H - 1)
            center_c = random.randint(0, W - 1)
            row_forward = random.choice((False, True))
            col_forward = random.choice((False, True))
            row_range = _point_range(H, h, center_r, row_forward)
            col_range = _point_range(W, w, center_c, col_forward)
            if row_range is None or col_range is None:
                continue
            r0 = random.randint(row_range[0], row_range[1])
            c0 = random.randint(col_range[0], col_range[1])
            sprite = _mask_coords(mask, r0, c0)
            copied = _transform_coords(sprite, kind, (center_r, center_c))
            param = (center_r, center_c)

        sprite_set = set(sprite)
        copied_set = set(copied)
        if sprite_set & copied_set:
            continue
        if any(not in_bounds((H, W), r, c) for r, c in copied):
            continue
        if sprite_set & input_halo:
            continue
        if (sprite_set | copied_set) & output_occ:
            continue

        marker_block = input_occ | sprite_set
        marker_halo = _expand_cells(marker_block, (H, W))
        reserved = marker_block | output_occ | copied_set
        markers = _sample_markers(H, W, kind, param, marker_halo, reserved)
        if markers is None:
            continue
        return sprite, copied, markers

    return None


def _sample_markers(
    H: int,
    W: int,
    kind: str,
    param: int | tuple[int, int],
    marker_halo: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    for _ in range(MAX_MARKER_ATTEMPTS):
        if kind == "vertical":
            axis_c = int(param)
            reach = min(axis_c, W - 1 - axis_c)
            if reach < 1:
                return None
            d = random.randint(1, reach)
            r = random.randint(0, H - 1)
            markers = ((r, axis_c - d), (r, axis_c + d))
        elif kind == "horizontal":
            axis_r = int(param)
            reach = min(axis_r, H - 1 - axis_r)
            if reach < 1:
                return None
            d = random.randint(1, reach)
            c = random.randint(0, W - 1)
            markers = ((axis_r - d, c), (axis_r + d, c))
        else:
            center_r, center_c = map(int, param)
            reach_r = min(center_r, H - 1 - center_r)
            reach_c = min(center_c, W - 1 - center_c)
            if reach_r < 1 or reach_c < 1:
                return None
            dr = random.randint(1, reach_r)
            dc = random.randint(1, reach_c)
            markers = ((center_r - dr, center_c - dc), (center_r + dr, center_c + dc))

        if any(cell in marker_halo or cell in reserved for cell in markers):
            continue
        return markers
    return None


def _paint_cells(grid: np.ndarray, cells: tuple[tuple[int, int], ...], color: int) -> None:
    for r, c in cells:
        grid[r, c] = int(color)


def _generate_case() -> np.ndarray | None:
    H = random.randint(10, 18)
    W = random.randint(10, 18)
    bg = random.choice(Color.ALL_COLORS)
    palette = [color for color in Color.ALL_COLORS if color != bg]
    count = random.randint(1, min(4, len(palette)))
    colors = random.sample(palette, count)
    kinds = _choose_kinds(count)

    inp = np.full((H, W), bg, dtype=int)
    expected = np.full((H, W), bg, dtype=int)
    input_occ: set[tuple[int, int]] = set()
    output_occ: set[tuple[int, int]] = set()

    for color, kind in zip(colors, kinds):
        mask = random.choice(MASKS)
        placed = _sample_object(H, W, mask, kind, input_occ, output_occ)
        if placed is None:
            return None
        sprite, copied, markers = placed

        _paint_cells(inp, sprite, color)
        _paint_cells(inp, markers, color)
        _paint_cells(expected, sprite, color)
        _paint_cells(expected, copied, color)

        input_occ.update(sprite)
        input_occ.update(markers)
        output_occ.update(sprite)
        output_occ.update(copied)

    if np.array_equal(main(inp), expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid marker-mirror ARC input")
