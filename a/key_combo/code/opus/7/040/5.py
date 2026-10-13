from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, crop_bbox, find_components, get_bg


BACKGROUND = Color.BLACK
CORNER_COLOR = Color.RED
TEMPLATE_COLOR = Color.GREEN
CONTROL_COLOR = Color.YELLOW

_CORNERS = ("TL", "TR", "BL", "BR")
_GEN_ATTEMPTS = 400
_GRID_SIDE_RANGE = (12, 20)


def _corner_to_coord(corner: str, shape: tuple[int, int]) -> tuple[int, int]:
    h, w = map(int, shape)
    return {
        "TL": (0, 0),
        "TR": (0, w - 1),
        "BL": (h - 1, 0),
        "BR": (h - 1, w - 1),
    }[corner]


def _corner_from_grid(grid: np.ndarray) -> str:
    h, w = map(int, grid.shape)
    corners = {
        "TL": (0, 0),
        "TR": (0, w - 1),
        "BL": (h - 1, 0),
        "BR": (h - 1, w - 1),
    }
    marked = [name for name, (r, c) in corners.items() if int(grid[r, c]) == CORNER_COLOR]
    assert len(marked) == 1, "expected exactly one red corner marker"
    return marked[0]


def _triangle_local_coords(
    shape: tuple[int, int], corner: str
) -> tuple[np.ndarray, np.ndarray]:
    h, w = map(int, shape)
    rr, cc = np.indices((h, w))
    if corner == "TL":
        return rr, cc
    if corner == "TR":
        return rr, (w - 1 - cc)
    if corner == "BL":
        return (h - 1 - rr), cc
    return (h - 1 - rr), (w - 1 - cc)


def _triangle_mask(shape: tuple[int, int], corner: str, side: int) -> np.ndarray:
    local_r, local_c = _triangle_local_coords(shape, corner)
    return local_r + local_c < int(side)


def _render_output(
    shape: tuple[int, int], corner: str, template: np.ndarray, control_count: int, bg: int = BACKGROUND
) -> np.ndarray:
    template = np.asarray(template, dtype=bool)
    th, tw = map(int, template.shape)
    side = int(control_count) * max(th, tw)
    local_r, local_c = _triangle_local_coords(shape, corner)
    tri = local_r + local_c < side
    tiled = template[local_r % th, local_c % tw]
    out = np.full(shape, int(bg), dtype=int)
    out[tri & tiled] = TEMPLATE_COLOR
    return out


def _parse_patch(lines: tuple[str, ...]) -> np.ndarray:
    return np.array([[ch == "#" for ch in row] for row in lines], dtype=bool)


def _patch_variants(lines: tuple[str, ...]) -> list[np.ndarray]:
    base = _parse_patch(lines)
    variants: list[np.ndarray] = []
    seen: set[tuple[tuple[int, ...], ...]] = set()
    for k in range(4):
        rot = np.rot90(base, k)
        for variant in (rot, np.fliplr(rot)):
            key = tuple(tuple(int(v) for v in row) for row in variant)
            if key not in seen:
                seen.add(key)
                variants.append(variant.copy())
    return variants


_TEMPLATE_PATCHES: list[np.ndarray] = []
for _lines in (
    ("##", "##"),
    ("###", "###"),
    ("##.", "###", ".##"),
    (".#.", "###", ".#."),
    ("###", "#.#", "###"),
    ("##.", "#..", "###"),
    ("###", "##.", "#.."),
):
    _TEMPLATE_PATCHES.extend(_patch_variants(_lines))


def _template_positions(
    shape: tuple[int, int], template: np.ndarray, triangle: np.ndarray, corner: str
) -> dict[str, list[tuple[int, int]]]:
    h, w = map(int, shape)
    th, tw = map(int, template.shape)
    groups = {"inside": [], "outside": [], "mixed": []}
    corner_r, corner_c = _corner_to_coord(corner, shape)
    rel_coords = [tuple(map(int, rc)) for rc in np.argwhere(template)]
    for r0 in range(1, h - th):
        for c0 in range(1, w - tw):
            abs_coords = [(r0 + dr, c0 + dc) for dr, dc in rel_coords]
            if any(max(abs(r - corner_r), abs(c - corner_c)) <= 1 for r, c in abs_coords):
                continue
            inside_flags = [bool(triangle[r, c]) for r, c in abs_coords]
            if all(inside_flags):
                groups["inside"].append((r0, c0))
            elif not any(inside_flags):
                groups["outside"].append((r0, c0))
            else:
                groups["mixed"].append((r0, c0))
    return groups


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    return out


def _place_controls(
    grid: np.ndarray, template_mask: np.ndarray, corner: str, count: int
) -> bool:
    h, w = map(int, grid.shape)
    forbidden = _dilate8(template_mask)
    corner_mask = np.zeros_like(grid, dtype=bool)
    corner_mask[_corner_to_coord(corner, grid.shape)] = True
    forbidden |= _dilate8(corner_mask)

    candidates = [
        (r, c)
        for r in range(1, h - 1)
        for c in range(1, w - 1)
        if not forbidden[r, c] and int(grid[r, c]) == BACKGROUND
    ]
    random.shuffle(candidates)

    picked: list[tuple[int, int]] = []
    for r, c in candidates:
        if all(max(abs(r - rr), abs(c - cc)) > 1 for rr, cc in picked):
            picked.append((r, c))
            if len(picked) == int(count):
                break

    if len(picked) != int(count):
        return False

    for r, c in picked:
        grid[r, c] = CONTROL_COLOR
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill a corner triangle by periodically tiling the detected template and clipping it to the triangle boundary.

    Input structure:
    - The background is black.
    - Exactly one red pixel sits in one of the four grid corners and chooses the triangle corner.
    - One compact green 4-connected template object lies away from the border; its bounding box is the tiling period and may contain black holes.
    - Several isolated yellow singleton pixels act as count controls.

    Rule:
    - Read the red corner to determine which corner owns the right-isosceles triangle.
    - Count the yellow singleton pixels; if that count is N and the template bbox is h x w, the triangle side is N * max(h, w).
    - Tile the green template bbox periodically across the whole grid, anchored at the marked corner, and keep only green tile pixels whose local row-distance plus local column-distance from that corner is less than the triangle side.
    - Erase the original red marker, all yellow controls, and any original template pixels outside the retained tiled triangle.

    Key insight:
    - The output is fixed only after combining four signals: corner gives orientation, yellow count gives scale, the template bbox gives the period, and the triangle mask clips the repeated template.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = int(get_bg(g))
    assert bg == BACKGROUND, "expected black background"
    assert int(np.sum(g == CORNER_COLOR)) == 1, "expected a unique red corner marker"
    corner = _corner_from_grid(g)

    green_components = find_components(
        g, bg=bg, connectivity=4, monochromatic=True, colors=[TEMPLATE_COLOR]
    )
    assert len(green_components) == 1, "expected exactly one green template component"
    template_component = green_components[0]
    assert 4 <= int(template_component["area"]) <= 9, "template area out of range"
    r0, c0, r1, c1 = map(int, template_component["bbox"])
    assert r0 > 0 and c0 > 0 and r1 < g.shape[0] - 1 and c1 < g.shape[1] - 1, "template must not touch the border"
    template = crop_bbox(g == TEMPLATE_COLOR, template_component["bbox"]).astype(bool)

    yellow_components = find_components(
        g, bg=bg, connectivity=8, monochromatic=True, colors=[CONTROL_COLOR]
    )
    assert yellow_components, "expected yellow control pixels"
    assert all(int(comp["area"]) == 1 for comp in yellow_components), "yellow controls must be isolated singletons"
    control_count = len(yellow_components)
    assert 3 <= control_count <= 6, "yellow control count out of range"

    return _render_output(g.shape, corner, template, control_count, bg=bg)


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*_GRID_SIDE_RANGE)
    w = random.randint(*_GRID_SIDE_RANGE)
    template = random.choice(_TEMPLATE_PATCHES)
    th, tw = map(int, template.shape)
    maxdim = max(th, tw)

    max_controls = min(6, min(h, w) // maxdim)
    if max_controls < 3:
        return None

    control_count = random.randint(3, max_controls)
    corner = random.choice(_CORNERS)
    triangle = _triangle_mask((h, w), corner, control_count * maxdim)
    pos_groups = _template_positions((h, w), template, triangle, corner)

    preferred_mode = random.choices(
        ("inside", "outside", "mixed"), weights=(4, 4, 2), k=1
    )[0]
    choices = pos_groups[preferred_mode] or pos_groups["inside"] or pos_groups["outside"] or pos_groups["mixed"]
    if not choices:
        return None
    r0, c0 = random.choice(choices)

    grid = np.full((h, w), BACKGROUND, dtype=int)
    grid[_corner_to_coord(corner, grid.shape)] = CORNER_COLOR

    template_mask = np.zeros((h, w), dtype=bool)
    for dr, dc in np.argwhere(template):
        rr = int(r0 + dr)
        cc = int(c0 + dc)
        template_mask[rr, cc] = True
        grid[rr, cc] = TEMPLATE_COLOR

    if not _place_controls(grid, template_mask, corner, control_count):
        return None

    expected = _render_output(grid.shape, corner, template, control_count, bg=BACKGROUND)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for puzzle 5")
