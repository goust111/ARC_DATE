from __future__ import annotations

import random

import numpy as np


# The eight symmetries of a square, expressed in the panel's own (row, column)
# coordinates.  The generator deliberately excludes identity so that geometry is
# a necessary part of every generated puzzle.
_GEOMETRIES = (
    "identity",
    "rotate_90_cw",
    "rotate_180",
    "rotate_270_cw",
    "flip_left_right",
    "flip_up_down",
    "transpose",
    "anti_transpose",
)

_INVERSE_GEOMETRY = {
    "identity": "identity",
    "rotate_90_cw": "rotate_270_cw",
    "rotate_180": "rotate_180",
    "rotate_270_cw": "rotate_90_cw",
    "flip_left_right": "flip_left_right",
    "flip_up_down": "flip_up_down",
    "transpose": "transpose",
    "anti_transpose": "anti_transpose",
}


def _transform(a: np.ndarray, geometry: str) -> np.ndarray:
    """Apply a D4 transform within the square array's own coordinates."""
    if geometry == "identity":
        return a.copy()
    if geometry == "rotate_90_cw":
        return np.rot90(a, -1).copy()
    if geometry == "rotate_180":
        return np.rot90(a, 2).copy()
    if geometry == "rotate_270_cw":
        return np.rot90(a, 1).copy()
    if geometry == "flip_left_right":
        return np.fliplr(a).copy()
    if geometry == "flip_up_down":
        return np.flipud(a).copy()
    if geometry == "transpose":
        return a.T.copy()
    if geometry == "anti_transpose":
        return np.rot90(a, 2).T.copy()
    raise ValueError(f"unknown geometry: {geometry}")


def _recolor(a: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    """Apply a color map without allowing newly written values to cascade."""
    out = a.copy()
    for source, target in mapping.items():
        out[a == int(source)] = int(target)
    return out


def _rect_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Return the inclusive bbox and assert that mask is one filled rectangle."""
    coords = np.argwhere(mask)
    assert len(coords) > 0, "each panel must contain a black rectangular hole"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    box = np.zeros_like(mask, dtype=bool)
    box[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = True
    assert np.array_equal(mask, box), "black cells in a panel must form one filled rectangle"
    return int(r0), int(c0), int(r1), int(c1)


def _candidate_mapping(
    left: np.ndarray,
    right: np.ndarray,
    geometry: str,
) -> dict[int, int] | None:
    """Read the only possible bijection for one proposed geometric alignment."""
    aligned_left = _transform(left, geometry)
    visible = (aligned_left != 0) & (right != 0)
    if not np.any(visible):
        return None

    left_colors = set(map(int, np.unique(left[left != 0])))
    right_colors = set(map(int, np.unique(right[right != 0])))
    if len(left_colors) != len(right_colors):
        return None

    forward: dict[int, int] = {}
    reverse: dict[int, int] = {}
    for source_raw, target_raw in zip(aligned_left[visible], right[visible]):
        source, target = int(source_raw), int(target_raw)
        if source in forward and forward[source] != target:
            return None
        if target in reverse and reverse[target] != source:
            return None
        forward[source] = target
        reverse[target] = source

    # Every palette entry has to be witnessed outside both holes.  This makes the
    # permutation fully determined rather than guessed from unused colors.
    if set(forward) != left_colors or set(forward.values()) != right_colors:
        return None
    return forward


def _infer_composite(left: np.ndarray, right: np.ndarray) -> tuple[str, dict[int, int]]:
    candidates = []
    for geometry in _GEOMETRIES:
        mapping = _candidate_mapping(left, right, geometry)
        if mapping is not None:
            candidates.append((geometry, mapping))
    assert len(candidates) == 1, "visible structure must determine one geometry and one color bijection"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer a geometric transform and color permutation between twin panels, then repair both complementary holes.

    Input structure:
    - A uniform vertical separator divides the grid into two equally sized square panels.
    - The right panel is a recolored D4 transform of the left panel; their nonblack palettes are disjoint.
    - Each panel contains exactly one filled black rectangular hole, and the two holes do not correspond under the transform.

    Rule:
    - On cells visible in both panels, test square rotations and reflections and retain the unique alignment admitting a full color bijection.
    - Fill the right hole from the geometrically transformed left panel and then apply the inferred color permutation.
    - Fill the left hole by inverse-recoloring the right panel and applying the inverse geometry; preserve all visible cells and the separator.

    Key insight:
    - Geometry must be inferred up to color renaming before the color map can be read, and repairing the source-side hole requires inverting both layers.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert width == 2 * height + 1, "expected two side-by-side square panels and a one-cell separator"
    assert max(g.shape) <= 30, "ARC grids may not exceed 30 in either dimension"

    n = height
    left = g[:, :n].copy()
    divider = g[:, n]
    right = g[:, n + 1 :].copy()
    assert np.all(divider == divider[0]), "separator must be a uniform vertical line"
    divider_color = int(divider[0])
    assert divider_color != 0, "black is reserved for the two holes"

    _rect_bbox(left == 0)
    _rect_bbox(right == 0)
    left_colors = set(map(int, np.unique(left[left != 0])))
    right_colors = set(map(int, np.unique(right[right != 0])))
    assert len(left_colors) >= 3, "expected a multi-color structural pattern"
    assert left_colors.isdisjoint(right_colors), "the two panels must use different palettes"
    assert divider_color not in left_colors | right_colors, "separator color must be visually distinct"

    geometry, color_map = _infer_composite(left, right)
    inverse_map = {target: source for source, target in color_map.items()}
    assert len(inverse_map) == len(color_map), "color relation must be bijective"

    predicted_right = _recolor(_transform(left, geometry), color_map)
    right_hole = right == 0
    assert np.all(predicted_right[right_hole] != 0), "right hole must correspond to visible source cells"
    repaired_right = right.copy()
    repaired_right[right_hole] = predicted_right[right_hole]

    inverse_recolored_right = _recolor(right, inverse_map)
    predicted_left = _transform(inverse_recolored_right, _INVERSE_GEOMETRY[geometry])
    left_hole = left == 0
    assert np.all(predicted_left[left_hole] != 0), "left hole must correspond to visible image cells"
    repaired_left = left.copy()
    repaired_left[left_hole] = predicted_left[left_hole]

    out = g.copy()
    out[:, :n] = repaired_left
    out[:, n + 1 :] = repaired_right
    return out


# ------------------------------- generator ---------------------------------

_GENERATION_ATTEMPTS = 500


def _seed_positions(n: int, count: int) -> list[tuple[int, int]] | None:
    """Choose well-separated region seeds for a legible Manhattan mosaic."""
    positions: list[tuple[int, int]] = []
    candidates = [(r, c) for r in range(n) for c in range(n)]
    random.shuffle(candidates)
    for r, c in candidates:
        if all(abs(r - rr) + abs(c - cc) >= max(3, n // 3) for rr, cc in positions):
            positions.append((r, c))
            if len(positions) == count:
                return positions
    return None


def _make_mosaic(n: int, colors: list[int]) -> np.ndarray | None:
    """Make large, visible color regions, with small stamps breaking symmetries."""
    seeds = _seed_positions(n, len(colors))
    if seeds is None:
        return None

    # Manhattan Voronoi regions form crisp axis-aligned color blocks.  A random
    # priority resolves ties without using color ids as semantic quantities.
    priorities = list(range(len(colors)))
    random.shuffle(priorities)
    rank = {seed_i: priorities[seed_i] for seed_i in range(len(seeds))}
    panel = np.empty((n, n), dtype=int)
    for r in range(n):
        for c in range(n):
            owner = min(
                range(len(seeds)),
                key=lambda i: (abs(r - seeds[i][0]) + abs(c - seeds[i][1]), rank[i]),
            )
            panel[r, c] = colors[owner]

    # Rectangular stamps create additional corners and adjacency signatures while
    # keeping the scene much clearer than independent pixel noise.
    for _ in range(random.randint(2, 4)):
        h = random.randint(1, 2)
        w = random.randint(2, 3) if h == 1 else random.randint(1, 2)
        r0 = random.randint(0, n - h)
        c0 = random.randint(0, n - w)
        current = int(panel[r0 + h // 2, c0 + w // 2])
        choices = [color for color in colors if color != current]
        panel[r0 : r0 + h, c0 : c0 + w] = random.choice(choices)

    counts = {color: int(np.sum(panel == color)) for color in colors}
    if min(counts.values()) < 5:
        return None
    return panel


def _rect_mask(n: int, *, favor_edge: bool = False) -> np.ndarray:
    h = random.randint(2, 3)
    w = random.randint(2, 3)
    if favor_edge:
        side = random.choice(("top", "bottom", "left", "right"))
        r0 = random.randint(0, n - h)
        c0 = random.randint(0, n - w)
        if side == "top":
            r0 = 0
        elif side == "bottom":
            r0 = n - h
        elif side == "left":
            c0 = 0
        else:
            c0 = n - w
    else:
        r0 = random.randint(0, n - h)
        c0 = random.randint(0, n - w)
    mask = np.zeros((n, n), dtype=bool)
    mask[r0 : r0 + h, c0 : c0 + w] = True
    return mask


def _semantic_accept(
    inp: np.ndarray,
    expected: np.ndarray,
    full_left: np.ndarray,
    full_right: np.ndarray,
    left_hole: np.ndarray,
    right_hole: np.ndarray,
    geometry: str,
    color_map: dict[int, int],
) -> bool:
    """Check the stated semantics and reject several common shortcut datasets."""
    n = full_left.shape[0]
    damaged_left = inp[:, :n]
    damaged_right = inp[:, n + 1 :]

    # Complementary holes: each missing region's counterpart is fully visible.
    image_of_left_hole = _transform(left_hole, geometry)
    if np.any(image_of_left_hole & right_hole):
        return False
    if len(np.unique(full_left[left_hole])) < 2 or len(np.unique(full_right[right_hole])) < 2:
        return False

    # Every mapping pair must actually be readable from jointly visible cells.
    inferred = _candidate_mapping(damaged_left, damaged_right, geometry)
    if inferred != color_map:
        return False
    valid = [name for name in _GEOMETRIES if _candidate_mapping(damaged_left, damaged_right, name) is not None]
    if valid != [geometry]:
        return False

    # Both layers are indispensable: recoloring without geometry and geometry
    # without recoloring must each fail on the complete panels.
    if geometry == "identity":
        return False
    recolor_only = _recolor(full_left, color_map)
    geometry_only = _transform(full_left, geometry)
    if np.array_equal(recolor_only, full_right) or np.array_equal(geometry_only, full_right):
        return False

    # The inverse-side repair cannot be obtained by copying raw right-palette
    # values, and both holes must be the only changed input cells.
    raw_inverse = _transform(full_right, _INVERSE_GEOMETRY[geometry])
    if np.array_equal(raw_inverse[left_hole], full_left[left_hole]):
        return False
    changed = inp != expected
    expected_changed = np.zeros_like(inp, dtype=bool)
    expected_changed[:, :n] = left_hole
    expected_changed[:, n + 1 :] = right_hole
    if not np.array_equal(changed, expected_changed):
        return False

    try:
        actual = main(inp)
    except (AssertionError, ValueError):
        return False
    return bool(np.array_equal(actual, expected))


def _generation_attempt() -> np.ndarray | None:
    n = random.randint(8, 12)
    palette_size = random.choice((3, 4))
    palette = list(range(1, 10))
    random.shuffle(palette)
    left_colors = palette[:palette_size]
    right_colors = palette[palette_size : 2 * palette_size]
    divider_color = palette[2 * palette_size]

    full_left = _make_mosaic(n, left_colors)
    if full_left is None:
        return None

    geometry = random.choice(_GEOMETRIES[1:])
    shuffled_right = right_colors.copy()
    random.shuffle(shuffled_right)
    color_map = dict(zip(left_colors, shuffled_right))
    full_right = _recolor(_transform(full_left, geometry), color_map)

    # Edge-touching holes are deliberately common counterexamples to solvers that
    # only recognize interior black components.
    left_hole = _rect_mask(n, favor_edge=random.random() < 0.45)
    right_hole = _rect_mask(n, favor_edge=random.random() < 0.45)
    if np.any(_transform(left_hole, geometry) & right_hole):
        return None
    if len(np.unique(full_left[left_hole])) < 2:
        return None
    if len(np.unique(full_right[right_hole])) < 2:
        return None

    damaged_left = full_left.copy()
    damaged_right = full_right.copy()
    damaged_left[left_hole] = 0
    damaged_right[right_hole] = 0
    if set(map(int, np.unique(damaged_left[damaged_left != 0]))) != set(left_colors):
        return None
    if set(map(int, np.unique(damaged_right[damaged_right != 0]))) != set(right_colors):
        return None

    inp = np.full((n, 2 * n + 1), divider_color, dtype=int)
    inp[:, :n] = damaged_left
    inp[:, n + 1 :] = damaged_right
    expected = np.full_like(inp, divider_color)
    expected[:, :n] = full_left
    expected[:, n + 1 :] = full_right

    if not _semantic_accept(
        inp,
        expected,
        full_left,
        full_right,
        left_hole,
        right_hole,
        geometry,
        color_map,
    ):
        return None
    return inp


def generate_input() -> np.ndarray:
    """Generate a diverse, uniquely solvable twin-panel completion puzzle."""
    for _ in range(_GENERATION_ATTEMPTS):
        result = _generation_attempt()
        if result is not None:
            return result
    raise RuntimeError("failed to generate a unique composite-transform puzzle")
