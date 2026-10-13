from __future__ import annotations

import random
from typing import TypedDict

import numpy as np


BACKGROUND = 7
PAIR_MAP = {
    8: 2,
    3: 1,
    5: 4,
}
ASYMMETRIC_DEFAULT = 5
ASYMMETRIC_FROM_GREY = 9

GRID_SIZE_RANGE = (18, 22)
RADIUS_RANGE = (3, 6)
MAX_GENERATION_ATTEMPTS = 500
PLACEMENT_ATTEMPTS = 160

MANDATORY_ROLE_COLORS = (
    ("full", 8),
    ("repair", 3),
    ("persist", 5),
    ("side", 6),
)
EXTRA_COLORS = (1, 2, 4, 9)
EXTRA_ROLE_FAMILIES = ("full", "repair", "persist", "side")


Coord = tuple[int, int]


class ObjectSpec(TypedDict):
    role: str
    color: int
    cr: int
    cc: int
    radius: int


def _neighbors8(r: int, c: int) -> list[Coord]:
    cells: list[Coord] = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            cells.append((r + dr, c + dc))
    return cells


def _background_color(grid: np.ndarray) -> int:
    counts = np.bincount(np.asarray(grid, dtype=int).ravel(), minlength=10)
    return int(np.argmax(counts))


def _diamond_outline(shape: tuple[int, int], cr: int, cc: int, radius: int) -> set[Coord]:
    h, w = shape
    if radius == 0:
        if 0 <= cr < h and 0 <= cc < w:
            return {(cr, cc)}
        return set()

    cells: set[Coord] = set()
    row_start = max(0, cr - radius)
    row_stop = min(h - 1, cr + radius)
    for r in range(row_start, row_stop + 1):
        dc = radius - abs(r - cr)
        left = cc - dc
        right = cc + dc
        if 0 <= left < w:
            cells.add((r, left))
        if 0 <= right < w:
            cells.add((r, right))
    return cells


def _outline_size(radius: int) -> int:
    return 1 if radius == 0 else 4 * radius


def _split_horizontal_symmetry(cells: set[Coord], cr: int) -> tuple[set[Coord], set[Coord]]:
    symmetric: set[Coord] = set()
    asymmetric: set[Coord] = set()
    for r, c in cells:
        mirror = (2 * cr - r, c)
        if mirror in cells:
            symmetric.add((r, c))
        else:
            asymmetric.add((r, c))
    return symmetric, asymmetric


def _find_components(grid: np.ndarray, bg: int) -> list[tuple[int, np.ndarray]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    visited = np.zeros((h, w), dtype=bool)
    components: list[tuple[int, np.ndarray]] = []

    for r in range(h):
        for c in range(w):
            if visited[r, c] or int(g[r, c]) == bg:
                continue
            color = int(g[r, c])
            stack = [(r, c)]
            visited[r, c] = True
            coords: list[Coord] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors8(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(g[nr, nc]) == color:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            components.append((color, np.array(coords, dtype=int)))

    return components


def _fit_diamond(coords: np.ndarray, shape: tuple[int, int]) -> tuple[int, int, int]:
    pts = {tuple(map(int, rc)) for rc in coords}
    h, w = shape
    fits: list[tuple[int, int, int]] = []

    for cr in range(h):
        for cc in range(w):
            radii = {abs(r - cr) + abs(c - cc) for r, c in pts}
            if len(radii) != 1:
                continue
            radius = radii.pop()
            if _diamond_outline(shape, cr, cc, radius) == pts:
                fits.append((cr, cc, radius))

    if not fits:
        raise AssertionError("component is not a valid clipped diamond outline")

    min_radius = min(radius for _cr, _cc, radius in fits)
    minimal = [fit for fit in fits if fit[2] == min_radius]
    if len(minimal) != 1:
        raise AssertionError("diamond fit is ambiguous")
    return minimal[0]


def _paint(out: np.ndarray, cells: set[Coord], color: int, bg: int) -> None:
    for r, c in cells:
        current = int(out[r, c])
        if current != bg and current != color:
            raise AssertionError("overlapping output objects")
        out[r, c] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Shrink every colored diamond outline by one layer, then recolor each shrunk outline by row-reflection symmetry about its own center row.

    Input structure:
    - The background is orange, and every foreground object is a one-cell-thick Manhattan-diamond outline.
    - Each diamond has a single original color and may be clipped by the grid boundary, but it still belongs to one full diamond with a fixed center and radius.
    - Different diamonds are separated from each other and must be processed independently.

    Rule:
    - Recover each diamond's full center and radius from its visible clipped outline, keeping the same center while reducing the radius by one.
    - Redraw only the visible part of that shrunk outline, then test each shrunk pixel against its reflection across the diamond's center row.
    - Recolor symmetric shrunk pixels with the paired mapping teal->red, green->blue, grey->yellow, leaving unmapped colors unchanged; recolor asymmetric shrunk pixels to grey, except original grey diamonds use maroon for those asymmetric pixels.

    Key insight:
    - Clipping matters twice: it does not change the hidden center used for shrinking, and after shrinking the remaining visible outline must be classified by top-bottom reflection around that same recovered center row.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _background_color(g)
    out = np.full(g.shape, bg, dtype=int)

    for color, coords in _find_components(g, bg):
        cr, cc, radius = _fit_diamond(coords, g.shape)
        shrunk = _diamond_outline(g.shape, cr, cc, max(radius - 1, 0))
        symmetric, asymmetric = _split_horizontal_symmetry(shrunk, cr)
        symmetric_color = PAIR_MAP.get(color, color)
        asymmetric_color = ASYMMETRIC_FROM_GREY if color == 5 else ASYMMETRIC_DEFAULT
        _paint(out, symmetric, symmetric_color, bg)
        _paint(out, asymmetric, asymmetric_color, bg)

    return out


def _sample_grid_shape() -> tuple[int, int]:
    side = random.randint(*GRID_SIZE_RANGE)
    if random.random() < 0.55:
        return side, side
    delta = random.randint(0, 2)
    other = max(GRID_SIZE_RANGE[0], min(GRID_SIZE_RANGE[1], side + random.choice((-delta, delta))))
    return side, other


def _random_radius(shape: tuple[int, int], min_radius: int = RADIUS_RANGE[0]) -> int:
    h, w = shape
    limit = min(RADIUS_RANGE[1], h - 2, w - 2)
    if limit < min_radius:
        raise AssertionError("shape too small for requested radius")
    return random.randint(min_radius, limit)


def _make_full_spec(shape: tuple[int, int], color: int) -> ObjectSpec:
    h, w = shape
    radius = _random_radius(shape)
    return {
        "role": "full",
        "color": color,
        "cr": random.randint(radius, h - 1 - radius),
        "cc": random.randint(radius, w - 1 - radius),
        "radius": radius,
    }


def _make_repair_spec(shape: tuple[int, int], color: int) -> ObjectSpec:
    h, w = shape
    radius = _random_radius(shape)
    orientation = random.choice(("top", "bottom"))
    cr = radius - 1 if orientation == "top" else h - radius
    return {
        "role": f"repair_{orientation}",
        "color": color,
        "cr": cr,
        "cc": random.randint(radius, w - 1 - radius),
        "radius": radius,
    }


def _make_persist_spec(shape: tuple[int, int], color: int) -> ObjectSpec:
    h, w = shape
    radius = _random_radius(shape, min_radius=4)
    orientation = random.choice(("top", "bottom"))
    distance = random.randint(0, radius - 2)
    cr = distance if orientation == "top" else h - 1 - distance
    return {
        "role": f"persist_{orientation}",
        "color": color,
        "cr": cr,
        "cc": random.randint(radius, w - 1 - radius),
        "radius": radius,
    }


def _make_side_spec(shape: tuple[int, int], color: int) -> ObjectSpec:
    h, w = shape
    radius = _random_radius(shape, min_radius=4)
    orientation = random.choice(("left", "right"))
    distance = random.randint(0, radius - 2)
    cc = distance if orientation == "left" else w - 1 - distance
    return {
        "role": f"side_{orientation}",
        "color": color,
        "cr": random.randint(radius, h - 1 - radius),
        "cc": cc,
        "radius": radius,
    }


def _sample_spec(shape: tuple[int, int], family: str, color: int) -> ObjectSpec:
    if family == "full":
        return _make_full_spec(shape, color)
    if family == "repair":
        return _make_repair_spec(shape, color)
    if family == "persist":
        return _make_persist_spec(shape, color)
    if family == "side":
        return _make_side_spec(shape, color)
    raise AssertionError("unknown role family")


def _expanded_cells(shape: tuple[int, int], cells: set[Coord]) -> set[Coord]:
    h, w = shape
    expanded: set[Coord] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    expanded.add((nr, nc))
    return expanded


def _spec_input_cells(shape: tuple[int, int], spec: ObjectSpec) -> set[Coord]:
    return _diamond_outline(shape, spec["cr"], spec["cc"], spec["radius"])


def _spec_output_cells(shape: tuple[int, int], spec: ObjectSpec) -> set[Coord]:
    return _diamond_outline(shape, spec["cr"], spec["cc"], max(spec["radius"] - 1, 0))


def _spec_is_semantically_valid(shape: tuple[int, int], spec: ObjectSpec) -> bool:
    input_cells = _spec_input_cells(shape, spec)
    output_cells = _spec_output_cells(shape, spec)
    if len(input_cells) < 5 or len(output_cells) < 4:
        return False

    recovered = _fit_diamond(np.array(sorted(input_cells), dtype=int), shape)
    if recovered != (spec["cr"], spec["cc"], spec["radius"]):
        return False

    symmetric, asymmetric = _split_horizontal_symmetry(output_cells, spec["cr"])
    input_clipped = len(input_cells) < _outline_size(spec["radius"])
    output_clipped = len(output_cells) < _outline_size(spec["radius"] - 1)

    role = spec["role"]
    if role == "full":
        return not input_clipped and not asymmetric
    if role.startswith("repair_"):
        return input_clipped and not output_clipped and not asymmetric
    if role.startswith("persist_"):
        return input_clipped and bool(symmetric) and bool(asymmetric)
    if role.startswith("side_"):
        return input_clipped and output_clipped and not asymmetric
    return False


def _place_specs(shape: tuple[int, int], families_and_colors: list[tuple[str, int]]) -> list[ObjectSpec] | None:
    specs: list[ObjectSpec] = []
    reserved: set[Coord] = set()

    for family, color in families_and_colors:
        placed = None
        for _ in range(PLACEMENT_ATTEMPTS):
            spec = _sample_spec(shape, family, color)
            if not _spec_is_semantically_valid(shape, spec):
                continue
            cells = _spec_input_cells(shape, spec) | _spec_output_cells(shape, spec)
            if reserved & cells:
                continue
            placed = spec
            reserved |= _expanded_cells(shape, cells)
            break
        if placed is None:
            return None
        specs.append(placed)

    return specs


def _render_input(shape: tuple[int, int], specs: list[ObjectSpec]) -> np.ndarray:
    grid = np.full(shape, BACKGROUND, dtype=int)
    for spec in specs:
        _paint(grid, _spec_input_cells(shape, spec), spec["color"], BACKGROUND)
    return grid


def _render_expected_output(shape: tuple[int, int], specs: list[ObjectSpec]) -> np.ndarray:
    out = np.full(shape, BACKGROUND, dtype=int)
    for spec in specs:
        color = spec["color"]
        shrunk = _spec_output_cells(shape, spec)
        symmetric, asymmetric = _split_horizontal_symmetry(shrunk, spec["cr"])
        _paint(out, symmetric, PAIR_MAP.get(color, color), BACKGROUND)
        _paint(out, asymmetric, ASYMMETRIC_FROM_GREY if color == 5 else ASYMMETRIC_DEFAULT, BACKGROUND)
    return out


def _same_specs_recovered(grid: np.ndarray, specs: list[ObjectSpec]) -> bool:
    recovered = sorted(
        (color, *_fit_diamond(coords, grid.shape))
        for color, coords in _find_components(grid, BACKGROUND)
    )
    expected = sorted((spec["color"], spec["cr"], spec["cc"], spec["radius"]) for spec in specs)
    return recovered == expected


def _covers_required_counterexamples(shape: tuple[int, int], specs: list[ObjectSpec]) -> bool:
    if len(specs) < 4:
        return False

    families = {spec["role"].split("_", 1)[0] for spec in specs}
    if not {"full", "repair", "persist", "side"}.issubset(families):
        return False

    grey_persist = [spec for spec in specs if spec["color"] == 5 and spec["role"].startswith("persist_")]
    if len(grey_persist) != 1:
        return False

    shrunk = _spec_output_cells(shape, grey_persist[0])
    symmetric, asymmetric = _split_horizontal_symmetry(shrunk, grey_persist[0]["cr"])
    if not symmetric or not asymmetric:
        return False

    return True


def _sample_object_plan() -> list[tuple[str, int]]:
    plan = list(MANDATORY_ROLE_COLORS)
    if random.random() < 0.65:
        extra_color = random.choice(EXTRA_COLORS)
        extra_family = random.choice(EXTRA_ROLE_FAMILIES)
        plan.append((extra_family, extra_color))
    random.shuffle(plan)
    return plan


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        shape = _sample_grid_shape()
        plan = _sample_object_plan()
        specs = _place_specs(shape, plan)
        if specs is None:
            continue
        if not _covers_required_counterexamples(shape, specs):
            continue

        inp = _render_input(shape, specs)
        expected = _render_expected_output(shape, specs)

        if not _same_specs_recovered(inp, specs):
            continue

        try:
            actual = main(inp)
        except Exception:
            continue
        if not np.array_equal(actual, expected):
            continue

        return inp

    raise RuntimeError("failed to generate a valid diamond-symmetry recoloring input")
