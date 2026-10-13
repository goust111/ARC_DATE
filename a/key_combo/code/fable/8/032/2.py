from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, colors_present, find_components


CORNER_NAMES = ("tl", "tr", "bl", "br")
MIN_RECT_SIDE = 5
MAX_GRID_SIDE = 30
MIN_GRID_SIDE_BY_COUNT = {2: 16, 3: 20, 4: 24}
MAX_RECT_SIDE_BY_COUNT = {2: 12, 3: 10, 4: 8}
GROUP_COUNT_CHOICES = (2, 3, 4)
GEN_RETRIES = 400
PLACEMENT_RETRIES = 120


def _missing_name_from_corner(
    r0: int, c0: int, r1: int, c1: int, missing: tuple[int, int]
) -> str:
    if missing == (r0, c0):
        return "tl"
    if missing == (r0, c1):
        return "tr"
    if missing == (r1, c0):
        return "bl"
    if missing == (r1, c1):
        return "br"
    raise AssertionError("missing corner must be one of the rectangle corners")


def _missing_corner(
    r0: int, c0: int, r1: int, c1: int, name: str
) -> tuple[int, int]:
    if name == "tl":
        return r0, c0
    if name == "tr":
        return r0, c1
    if name == "bl":
        return r1, c0
    if name == "br":
        return r1, c1
    raise AssertionError("unknown corner name")


def _top_left_phase_diff(height: int, width: int, missing_name: str) -> int:
    if missing_name == "tl":
        return 0
    if missing_name == "tr":
        return (width - 3) % 2
    if missing_name == "bl":
        return (height - 3) % 2
    return (height + width) % 2


def _boxes_do_not_touch(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return (
        ar1 + 1 < br0
        or br1 + 1 < ar0
        or ac1 + 1 < bc0
        or bc1 + 1 < ac0
    )


def _extract_groups(grid: np.ndarray) -> list[dict]:
    groups: list[dict] = []
    for color in colors_present(grid, exclude={Color.BLACK}):
        components = find_components(
            grid,
            bg=Color.BLACK,
            connectivity=4,
            monochromatic=True,
            colors=[color],
        )
        assert len(components) == 3, "each color group must contain exactly three isolated pixels"
        coords = []
        for component in components:
            assert int(component["area"]) == 1, "each corner marker must be a single pixel"
            coords.append(tuple(map(int, component["coords"][0])))

        rows = sorted({r for r, _ in coords})
        cols = sorted({c for _, c in coords})
        assert len(rows) == 2 and len(cols) == 2, "three markers must imply one axis-aligned rectangle"
        r0, r1 = rows
        c0, c1 = cols
        assert r1 - r0 + 1 >= MIN_RECT_SIDE and c1 - c0 + 1 >= MIN_RECT_SIDE

        corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
        assert set(coords) <= corners and len(set(coords)) == 3
        missing = next(iter(corners - set(coords)))
        groups.append(
            {
                "color": int(color),
                "bbox": (r0, c0, r1, c1),
                "missing": missing,
                "missing_name": _missing_name_from_corner(r0, c0, r1, c1, missing),
            }
        )

    assert groups, "expected at least one color group"
    for i, first in enumerate(groups):
        for second in groups[i + 1 :]:
            assert _boxes_do_not_touch(first["bbox"], second["bbox"]), "rectangles must not touch"
    return groups


def _draw_group(
    out: np.ndarray,
    group: dict,
    *,
    anchor_mode: str = "inferred",
    draw_fill: bool = True,
) -> None:
    color = int(group["color"])
    r0, c0, r1, c1 = group["bbox"]
    out[r0 : r1 + 1, c0] = color
    out[r0 : r1 + 1, c1] = color
    out[r0, c0 : c1 + 1] = color
    out[r1, c0 : c1 + 1] = color

    if not draw_fill:
        return

    if anchor_mode == "inferred":
        mr, mc = group["missing"]
        anchor_r = r0 + 1 if mr == r0 else r1 - 1
        anchor_c = c0 + 1 if mc == c0 else c1 - 1
    elif anchor_mode == "top_left":
        anchor_r, anchor_c = r0 + 1, c0 + 1
    else:
        raise AssertionError("unsupported anchor mode")

    interior = out[r0 + 1 : r1, c0 + 1 : c1]
    rows, cols = np.indices(interior.shape)
    mask = (
        (rows + (r0 + 1 - anchor_r) + cols + (c0 + 1 - anchor_c)) % 2 == 0
    )
    interior[mask] = color


def _render_output(
    shape: tuple[int, int],
    groups: list[dict],
    *,
    anchor_mode: str = "inferred",
    draw_fill: bool = True,
) -> np.ndarray:
    out = np.full(shape, Color.BLACK, dtype=int)
    for group in groups:
        _draw_group(out, group, anchor_mode=anchor_mode, draw_fill=draw_fill)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each implied rectangle from three corner markers and checker-fill its interior from the missing corner's phase.

    Input structure:
    - A black background contains several color groups.
    - Each group has exactly three same-color isolated pixels, located on three corners of one axis-aligned rectangle.
    - Group colors are distinct, every implied rectangle is at least 5x5, and different rectangles do not touch.

    Rule:
    - For each group, infer the missing fourth corner and draw the full one-pixel-thick rectangle border in that group color.
    - Fill the rectangle interior with a checkerboard whose anchor is the interior cell diagonally adjacent to the missing corner.
    - Interior cells whose row difference plus column difference from that anchor is even take the group color; the other interior cells stay black.

    Key insight:
    - The checkerboard phase is not fixed to the rectangle's top-left interior cell; it is determined by the unseen fourth corner implied by the three visible markers.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and 0 < g.shape[0] <= MAX_GRID_SIDE and 0 < g.shape[1] <= MAX_GRID_SIDE
    groups = _extract_groups(g)
    out = g.copy()
    for group in groups:
        _draw_group(out, group, anchor_mode="inferred", draw_fill=True)
    return out


def _sample_rect_size(
    max_h: int,
    max_w: int,
    missing_name: str,
    *,
    require_phase_witness: bool,
) -> tuple[int, int] | None:
    candidates = [
        (height, width)
        for height in range(MIN_RECT_SIDE, max_h + 1)
        for width in range(MIN_RECT_SIDE, max_w + 1)
        if (not require_phase_witness)
        or _top_left_phase_diff(height, width, missing_name) == 1
    ]
    if not candidates:
        return None
    return random.choice(candidates)


def _build_group_specs(
    height: int, width: int, group_count: int
) -> list[dict] | None:
    max_side = min(MAX_RECT_SIDE_BY_COUNT[group_count], height - 2, width - 2)
    if max_side < MIN_RECT_SIDE:
        return None

    colors = random.sample(list(range(1, 10)), group_count)
    missing_names = random.sample(list(CORNER_NAMES), group_count)
    witness_options = [name for name in missing_names if name != "tl"]
    witness_name = random.choice(witness_options)

    raw_groups = []
    for color, missing_name in zip(colors, missing_names):
        size = _sample_rect_size(
            max_side,
            max_side,
            missing_name,
            require_phase_witness=missing_name == witness_name,
        )
        if size is None:
            return None
        rect_h, rect_w = size
        raw_groups.append(
            {
                "color": int(color),
                "missing_name": missing_name,
                "rect_h": rect_h,
                "rect_w": rect_w,
            }
        )
    raw_groups.sort(key=lambda group: group["rect_h"] * group["rect_w"], reverse=True)

    placed: list[dict] = []
    for group in raw_groups:
        rect_h = int(group["rect_h"])
        rect_w = int(group["rect_w"])
        spots = [
            (r0, c0)
            for r0 in range(0, height - rect_h + 1)
            for c0 in range(0, width - rect_w + 1)
            if all(
                _boxes_do_not_touch(
                    (r0, c0, r0 + rect_h - 1, c0 + rect_w - 1),
                    other["bbox"],
                )
                for other in placed
            )
        ]
        if not spots:
            return None

        r0, c0 = random.choice(spots)
        r1 = r0 + rect_h - 1
        c1 = c0 + rect_w - 1
        missing = _missing_corner(r0, c0, r1, c1, group["missing_name"])
        placed.append(
            {
                "color": group["color"],
                "bbox": (r0, c0, r1, c1),
                "missing": missing,
                "missing_name": group["missing_name"],
            }
        )

    return placed


def _render_input(shape: tuple[int, int], groups: list[dict]) -> np.ndarray:
    grid = np.full(shape, Color.BLACK, dtype=int)
    for group in groups:
        color = int(group["color"])
        r0, c0, r1, c1 = group["bbox"]
        mr, mc = group["missing"]
        for coord in ((r0, c0), (r0, c1), (r1, c0), (r1, c1)):
            if coord != (mr, mc):
                grid[coord] = color
    return grid


def _valid_example(grid: np.ndarray, groups: list[dict]) -> bool:
    parsed_groups = _extract_groups(grid)
    expected = _render_output(grid.shape, groups, anchor_mode="inferred", draw_fill=True)
    if not np.array_equal(expected, _render_output(grid.shape, parsed_groups, anchor_mode="inferred", draw_fill=True)):
        return False

    border_only = _render_output(grid.shape, groups, anchor_mode="inferred", draw_fill=False)
    naive_top_left = _render_output(grid.shape, groups, anchor_mode="top_left", draw_fill=True)
    if np.array_equal(expected, border_only):
        return False
    if np.array_equal(expected, naive_top_left):
        return False
    if len({group["missing_name"] for group in groups}) < 2:
        return False

    try:
        actual = main(grid)
    except AssertionError:
        return False
    return np.array_equal(actual, expected)


def _gen_attempt() -> np.ndarray | None:
    group_count = random.choice(GROUP_COUNT_CHOICES)
    min_side = MIN_GRID_SIDE_BY_COUNT[group_count]
    height = random.randint(min_side, MAX_GRID_SIDE)
    width = random.randint(min_side, MAX_GRID_SIDE)

    for _ in range(PLACEMENT_RETRIES):
        groups = _build_group_specs(height, width, group_count)
        if groups is None:
            continue
        grid = _render_input((height, width), groups)
        if _valid_example(grid, groups):
            return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a non-degenerate example")
