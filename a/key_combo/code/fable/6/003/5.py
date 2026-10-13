from __future__ import annotations

from collections import Counter
import random

import numpy as np


Coord = tuple[int, int]


def get_bg(grid: np.ndarray) -> int:
    """Return the most common color in the grid."""
    counts = Counter(map(int, np.asarray(grid).reshape(-1)))
    return int(counts.most_common(1)[0][0])


def colors_present(grid: np.ndarray, *, exclude: set[int] | None = None) -> list[int]:
    """Return sorted unique colors, optionally excluding some ids."""
    colors = set(map(int, np.asarray(grid).reshape(-1)))
    if exclude is not None:
        colors -= set(map(int, exclude))
    return sorted(colors)


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    """Return an inclusive bbox (r0, c0, r1, c1) for an (N,2) coord array."""
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _diamond_vertices(center_r: int, center_c: int, radius: int) -> list[Coord]:
    return [
        (int(center_r - radius), int(center_c)),
        (int(center_r + radius), int(center_c)),
        (int(center_r), int(center_c - radius)),
        (int(center_r), int(center_c + radius)),
    ]


def _diamond_from_coords(coords: np.ndarray) -> tuple[int, int, int]:
    """
    Recover (center_r, center_c, radius) from four diamond vertex pixels.

    The four points must be exactly the N/S/W/E vertices of a Manhattan diamond.
    """
    assert coords.shape == (4, 2), "diamond vertex color must occupy exactly four cells"
    r0, c0, r1, c1 = bbox_coords(coords)
    height = int(r1 - r0 + 1)
    width = int(c1 - c0 + 1)
    assert height == width and height % 2 == 1, "diamond bbox must be an odd square"
    radius = height // 2
    center_r = int(r0 + radius)
    center_c = int(c0 + radius)
    expected = set(_diamond_vertices(center_r, center_c, radius))
    actual = {(int(r), int(c)) for r, c in coords}
    assert actual == expected, "four pixels must be the exact diamond vertices"
    return center_r, center_c, radius


def _find_diamonds(grid: np.ndarray, bg: int) -> dict[str, int]:
    candidates: list[tuple[int, int, int, int]] = []
    for color in colors_present(grid, exclude={bg}):
        coords = np.argwhere(grid == int(color))
        if coords.shape[0] != 4:
            continue
        try:
            center_r, center_c, radius = _diamond_from_coords(coords)
        except AssertionError:
            continue
        candidates.append((int(color), int(center_r), int(center_c), int(radius)))

    assert len(candidates) == 2, "expected exactly two four-vertex diamonds"
    (color_a, center_r_a, center_c_a, radius_a), (color_b, center_r_b, center_c_b, radius_b) = candidates
    assert (center_r_a, center_c_a) == (center_r_b, center_c_b), "diamonds must be concentric"

    if radius_a < radius_b:
        inner = (color_a, radius_a)
        outer = (color_b, radius_b)
    else:
        inner = (color_b, radius_b)
        outer = (color_a, radius_a)

    assert inner[1] < outer[1], "inner diamond must be strictly smaller"
    return {
        "center_r": int(center_r_a),
        "center_c": int(center_c_a),
        "inner_color": int(inner[0]),
        "inner_radius": int(inner[1]),
        "outer_color": int(outer[0]),
        "outer_radius": int(outer[1]),
    }


_SIDE_DIRS: dict[str, Coord] = {
    "top": (1, 0),
    "bottom": (-1, 0),
    "left": (0, 1),
    "right": (0, -1),
}


def _marker_side(r: int, c: int, shape: tuple[int, int]) -> str:
    height, width = map(int, shape)
    sides = []
    if r == 0:
        sides.append("top")
    if r == height - 1:
        sides.append("bottom")
    if c == 0:
        sides.append("left")
    if c == width - 1:
        sides.append("right")
    assert len(sides) == 1, "marker must lie on exactly one grid border"
    return sides[0]


def _marker_coord(
    shape: tuple[int, int],
    center_r: int,
    center_c: int,
    side: str,
    offset: int,
) -> Coord:
    height, width = map(int, shape)
    if side == "top":
        return 0, int(center_c + offset)
    if side == "bottom":
        return height - 1, int(center_c + offset)
    if side == "left":
        return int(center_r + offset), 0
    if side == "right":
        return int(center_r + offset), width - 1
    raise AssertionError("unknown side")


def _marker_offset(center_r: int, center_c: int, side: str, r: int, c: int) -> int:
    if side in ("top", "bottom"):
        return int(c - center_c)
    return int(r - center_r)


def _ray_cells(
    shape: tuple[int, int],
    center_r: int,
    center_c: int,
    inner_radius: int,
    outer_radius: int,
    side: str,
    offset: int,
) -> tuple[list[Coord], bool]:
    """Return the correctly painted cells for one marker ray."""
    marker_r, marker_c = _marker_coord(shape, center_r, center_c, side, offset)
    dr, dc = _SIDE_DIRS[side]
    rr = int(marker_r + dr)
    cc = int(marker_c + dc)
    height, width = map(int, shape)

    cells: list[Coord] = []
    hit_inner = False
    while 0 <= rr < height and 0 <= cc < width:
        dist = abs(int(rr - center_r)) + abs(int(cc - center_c))
        if dist == int(inner_radius):
            hit_inner = True
            break
        if int(inner_radius) < dist < int(outer_radius):
            cells.append((int(rr), int(cc)))
        rr += dr
        cc += dc
    return cells, bool(hit_inner)


def _paint_from_specs(
    grid: np.ndarray,
    *,
    center_r: int,
    center_c: int,
    inner_radius: int,
    outer_radius: int,
    markers: list[tuple[str, int, int]],
) -> tuple[np.ndarray, list[list[Coord]]]:
    """Paint the correct virtual-annulus rays from known generation specs."""
    bg = int(get_bg(grid))
    out = np.asarray(grid, dtype=int).copy()
    painted_by_marker: list[list[Coord]] = []

    for side, offset, color in markers:
        cells, hit_inner = _ray_cells(
            out.shape,
            center_r,
            center_c,
            inner_radius,
            outer_radius,
            side,
            offset,
        )
        assert hit_inner, "every generated marker must reach the inner diamond boundary"

        current_cells: list[Coord] = []
        for rr, cc in cells:
            assert int(out[rr, cc]) == bg, "generated rays must not overlap visible objects"
            out[rr, cc] = int(color)
            current_cells.append((int(rr), int(cc)))
        painted_by_marker.append(current_cells)

    return out, painted_by_marker


def _paint_wrong_ignore_inner(
    grid: np.ndarray,
    *,
    center_r: int,
    center_c: int,
    outer_radius: int,
    markers: list[tuple[str, int, int]],
) -> np.ndarray:
    """Shortcut model: ignore the inner blocker and fill the full outer interior."""
    bg = int(get_bg(grid))
    out = np.asarray(grid, dtype=int).copy()
    height, width = out.shape

    for side, offset, color in markers:
        marker_r, marker_c = _marker_coord(out.shape, center_r, center_c, side, offset)
        dr, dc = _SIDE_DIRS[side]
        rr = int(marker_r + dr)
        cc = int(marker_c + dc)
        while 0 <= rr < height and 0 <= cc < width:
            dist = abs(int(rr - center_r)) + abs(int(cc - center_c))
            if dist < int(outer_radius) and int(out[rr, cc]) == bg:
                out[rr, cc] = int(color)
            rr += dr
            cc += dc
    return out


def _paint_wrong_rectangles(
    grid: np.ndarray,
    *,
    center_r: int,
    center_c: int,
    inner_radius: int,
    outer_radius: int,
    markers: list[tuple[str, int, int]],
) -> np.ndarray:
    """Shortcut model: treat both diamonds as axis-aligned rectangles."""
    bg = int(get_bg(grid))
    out = np.asarray(grid, dtype=int).copy()

    outer_top = int(center_r - outer_radius)
    outer_bottom = int(center_r + outer_radius)
    outer_left = int(center_c - outer_radius)
    outer_right = int(center_c + outer_radius)
    inner_top = int(center_r - inner_radius)
    inner_bottom = int(center_r + inner_radius)
    inner_left = int(center_c - inner_radius)
    inner_right = int(center_c + inner_radius)

    for side, offset, color in markers:
        if side == "top":
            column = int(center_c + offset)
            for rr in range(outer_top + 1, inner_top):
                if int(out[rr, column]) == bg:
                    out[rr, column] = int(color)
        elif side == "bottom":
            column = int(center_c + offset)
            for rr in range(inner_bottom + 1, outer_bottom):
                if int(out[rr, column]) == bg:
                    out[rr, column] = int(color)
        elif side == "left":
            row = int(center_r + offset)
            for cc in range(outer_left + 1, inner_left):
                if int(out[row, cc]) == bg:
                    out[row, cc] = int(color)
        elif side == "right":
            row = int(center_r + offset)
            for cc in range(inner_right + 1, outer_right):
                if int(out[row, cc]) == bg:
                    out[row, cc] = int(color)
        else:
            raise AssertionError("unknown side")

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint side-marker rays through the implied annulus between two invisible concentric diamonds.

    Input structure:
    - The background contains exactly two colors that each mark four isolated
      diamond vertices: one outer set and one inner set, with both diamonds
      concentric and otherwise unpainted.
    - Additional colored single-pixel markers lie on the top, bottom, left,
      or right grid border.

    Rule:
    - Reconstruct the two virtual Manhattan-diamond boundaries from their four
      vertex pixels.
    - Each border marker emits a straight ray inward along its row or column.
    - Cells outside the outer diamond stay background; cells with
      `inner_radius < distance < outer_radius` are recolored to the marker
      color; reaching the inner diamond boundary stops the ray immediately.
    - Keep all original vertex pixels and marker pixels unchanged.

    Key insight:
    - Neither clipping boundary is drawn. The start and end of every painted
      segment come only from recovering the shared center and radii of the two
      implied diamonds, not from any visible line pixels.
    """
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    bg = int(get_bg(g))
    diamonds = _find_diamonds(g, bg)
    center_r = int(diamonds["center_r"])
    center_c = int(diamonds["center_c"])
    inner_radius = int(diamonds["inner_radius"])
    outer_radius = int(diamonds["outer_radius"])
    vertex_colors = {int(diamonds["inner_color"]), int(diamonds["outer_color"])}

    marker_mask = (g != bg) & (~np.isin(g, list(vertex_colors)))
    marker_coords = np.argwhere(marker_mask)
    assert marker_coords.size > 0, "expected at least one border marker"

    out = g.copy()
    for r_raw, c_raw in marker_coords:
        r = int(r_raw)
        c = int(c_raw)
        side = _marker_side(r, c, (height, width))
        offset = _marker_offset(center_r, center_c, side, r, c)
        if side in ("top", "bottom"):
            assert abs(offset) <= inner_radius, "generated vertical rays must reach the inner diamond"
        else:
            assert abs(offset) <= inner_radius, "generated horizontal rays must reach the inner diamond"

        dr, dc = _SIDE_DIRS[side]
        rr = int(r + dr)
        cc = int(c + dc)
        hit_inner = False
        while 0 <= rr < height and 0 <= cc < width:
            dist = abs(int(rr - center_r)) + abs(int(cc - center_c))
            if dist == inner_radius:
                hit_inner = True
                break
            if inner_radius < dist < outer_radius and int(out[rr, cc]) == bg:
                out[rr, cc] = int(g[r, c])
            rr += dr
            cc += dc

        assert hit_inner, "every ray must stop at the implied inner diamond"

    return out


# --------------- generator ---------------

_RNG = random.Random()

MAX_GENERATION_ATTEMPTS = 400
OUTER_RADIUS_RANGE = (5, 8)
GRID_EXTRA_RANGE = (3, 11)
MARKER_COUNT_RANGE = (4, 7)


def _choose_side_offsets(offset_limit: int) -> tuple[int, int]:
    options = list(range(-offset_limit, offset_limit + 1))
    first = int(_RNG.choice(options))
    second = int(_RNG.choice(options))
    if first == 0 and second == 0:
        second = int(_RNG.choice([value for value in options if value != 0]))
    return first, second


def _build_marker_list(
    inner_radius: int,
    marker_colors: list[int],
) -> list[tuple[str, int, int]]:
    assert len(marker_colors) >= 4
    vertical_band = int(_RNG.randint(1, inner_radius - 1))
    horizontal_band = int(_RNG.randint(1, inner_radius - vertical_band))

    top_offset, bottom_offset = _choose_side_offsets(vertical_band)
    left_offset, right_offset = _choose_side_offsets(horizontal_band)
    if left_offset == 0 and right_offset == 0:
        right_offset = int(_RNG.choice([value for value in range(-horizontal_band, horizontal_band + 1) if value != 0]))

    markers: list[tuple[str, int, int]] = [
        ("top", top_offset, int(marker_colors[0])),
        ("bottom", bottom_offset, int(marker_colors[1])),
        ("left", left_offset, int(marker_colors[2])),
        ("right", right_offset, int(marker_colors[3])),
    ]

    used_pairs = {(side, offset) for side, offset, _color in markers}
    all_candidates: list[tuple[str, int]] = []
    for side in ("top", "bottom"):
        for offset in range(-vertical_band, vertical_band + 1):
            all_candidates.append((side, int(offset)))
    for side in ("left", "right"):
        for offset in range(-horizontal_band, horizontal_band + 1):
            all_candidates.append((side, int(offset)))

    extras = [(side, offset) for side, offset in all_candidates if (side, offset) not in used_pairs]
    _RNG.shuffle(extras)
    for color, (side, offset) in zip(marker_colors[4:], extras, strict=False):
        markers.append((side, int(offset), int(color)))

    return markers


def _render_input_from_spec(spec: dict[str, object]) -> np.ndarray:
    shape = tuple(spec["shape"])
    height, width = map(int, shape)
    bg = int(spec["bg"])
    center_r, center_c = map(int, spec["center"])
    outer_radius = int(spec["outer_radius"])
    inner_radius = int(spec["inner_radius"])
    outer_color = int(spec["outer_color"])
    inner_color = int(spec["inner_color"])
    markers: list[tuple[str, int, int]] = list(spec["markers"])

    grid = np.full((height, width), bg, dtype=int)
    for rr, cc in _diamond_vertices(center_r, center_c, outer_radius):
        grid[rr, cc] = outer_color
    for rr, cc in _diamond_vertices(center_r, center_c, inner_radius):
        grid[rr, cc] = inner_color
    for side, offset, color in markers:
        rr, cc = _marker_coord((height, width), center_r, center_c, side, offset)
        assert int(grid[rr, cc]) == bg, "marker placement must not overlap a vertex"
        grid[rr, cc] = int(color)
    return grid


def _validate_generated(spec: dict[str, object], grid: np.ndarray, expected: np.ndarray) -> bool:
    shape = tuple(spec["shape"])
    center_r, center_c = map(int, spec["center"])
    inner_radius = int(spec["inner_radius"])
    outer_radius = int(spec["outer_radius"])
    markers: list[tuple[str, int, int]] = list(spec["markers"])
    bg = int(spec["bg"])
    outer_color = int(spec["outer_color"])
    inner_color = int(spec["inner_color"])

    if grid.shape != shape or expected.shape != shape:
        return False
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if np.array_equal(grid, expected):
        return False

    non_bg = colors_present(grid, exclude={bg})
    if len(non_bg) != 2 + len(markers):
        return False

    counts = Counter(map(int, grid.reshape(-1)))
    if counts[int(outer_color)] != 4 or counts[int(inner_color)] != 4:
        return False
    if any(counts[int(color)] != 1 for _side, _offset, color in markers):
        return False

    sides = {side for side, _offset, _color in markers}
    if sides != {"top", "bottom", "left", "right"}:
        return False

    vertical_offsets = [abs(int(offset)) for side, offset, _color in markers if side in ("top", "bottom")]
    horizontal_offsets = [abs(int(offset)) for side, offset, _color in markers if side in ("left", "right")]
    if not any(value > 0 for value in vertical_offsets):
        return False
    if not any(value > 0 for value in horizontal_offsets):
        return False

    # The generator guarantees non-overlap by keeping vertical and horizontal
    # marker bands within the inner radius. Check that the painted rays really
    # stay disjoint and each contributes visible output.
    correct_from_specs, painted_by_marker = _paint_from_specs(
        grid,
        center_r=center_r,
        center_c=center_c,
        inner_radius=inner_radius,
        outer_radius=outer_radius,
        markers=markers,
    )
    if not np.array_equal(correct_from_specs, expected):
        return False
    if any(len(cells) == 0 for cells in painted_by_marker):
        return False

    union: set[Coord] = set()
    for cells in painted_by_marker:
        cell_set = set(cells)
        if union & cell_set:
            return False
        union |= cell_set
    if len(union) < 8:
        return False

    # Semantic anti-shortcut checks:
    # 1) ignoring the inner blocker must fail;
    # 2) clipping by rectangles instead of diamonds must fail.
    if np.array_equal(
        expected,
        _paint_wrong_ignore_inner(
            grid,
            center_r=center_r,
            center_c=center_c,
            outer_radius=outer_radius,
            markers=markers,
        ),
    ):
        return False
    if np.array_equal(
        expected,
        _paint_wrong_rectangles(
            grid,
            center_r=center_r,
            center_c=center_c,
            inner_radius=inner_radius,
            outer_radius=outer_radius,
            markers=markers,
        ),
    ):
        return False

    try:
        solved = main(grid)
    except Exception:
        return False
    if not np.array_equal(solved, expected):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    outer_radius = int(_RNG.randint(*OUTER_RADIUS_RANGE))
    inner_radius = int(_RNG.randint(2, outer_radius - 2))

    min_side = 2 * outer_radius + 3
    max_side = min(30, 2 * outer_radius + GRID_EXTRA_RANGE[1])
    height = int(_RNG.randint(min_side, max_side))
    width = int(_RNG.randint(min_side, max_side))
    center_r = int(_RNG.randint(outer_radius + 1, height - outer_radius - 2))
    center_c = int(_RNG.randint(outer_radius + 1, width - outer_radius - 2))

    marker_count = int(_RNG.randint(*MARKER_COUNT_RANGE))
    palette = _RNG.sample(list(range(10)), 3 + marker_count)
    bg = int(palette[0])
    outer_color = int(palette[1])
    inner_color = int(palette[2])
    marker_colors = [int(color) for color in palette[3:]]
    markers = _build_marker_list(inner_radius, marker_colors)

    spec: dict[str, object] = {
        "shape": (height, width),
        "bg": bg,
        "center": (center_r, center_c),
        "outer_radius": outer_radius,
        "inner_radius": inner_radius,
        "outer_color": outer_color,
        "inner_color": inner_color,
        "markers": markers,
    }

    grid = _render_input_from_spec(spec)
    expected, _painted_by_marker = _paint_from_specs(
        grid,
        center_r=center_r,
        center_c=center_c,
        inner_radius=inner_radius,
        outer_radius=outer_radius,
        markers=markers,
    )
    if not _validate_generated(spec, grid, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid virtual-annulus ray puzzle")
