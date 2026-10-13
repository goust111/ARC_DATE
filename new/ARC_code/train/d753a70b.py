import numpy as np

from arc_utils import find_components, get_bg
import random


RED = 2
GRAY = 5


def _diamond_outline(shape: tuple[int, int], cr: int, cc: int, radius: int) -> set[tuple[int, int]]:
    assert radius >= 0
    h, w = shape
    return {
        (r, c)
        for r in range(h)
        for c in range(w)
        if abs(r - cr) + abs(c - cc) == radius
    }


def _diamond_fits(coords: np.ndarray, shape: tuple[int, int]) -> list[tuple[int, int, int]]:
    h, w = shape
    pts = {tuple(map(int, p)) for p in coords}
    margin = h + w
    fits: list[tuple[int, int, int]] = []
    for cr in range(-margin, h + margin + 1):
        for cc in range(-margin, w + margin + 1):
            radii = {abs(r - cr) + abs(c - cc) for r, c in pts}
            if len(radii) != 1:
                continue
            radius = radii.pop()
            if _diamond_outline(shape, cr, cc, radius) == pts:
                fits.append((cr, cc, radius))
    return fits


def _minimal_diamond_fit(coords: np.ndarray, shape: tuple[int, int]) -> tuple[int, int, int, bool]:
    fits = _diamond_fits(coords, shape)
    assert fits
    min_radius = min(radius for _cr, _cc, radius in fits)
    minimal = [(cr, cc, radius) for cr, cc, radius in fits if radius == min_radius]
    assert len(minimal) == 1
    cr, cc, radius = minimal[0]
    return cr, cc, radius, len(fits) > 1


def _gray_outline(coords: np.ndarray, shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    cr, cc, radius, edge_ambiguous = _minimal_diamond_fit(coords, shape)

    if edge_ambiguous:
        # Continue an edge-clipped growing arc away from the nearest perpendicular edge.
        if cr in (0, h - 1):
            assert cc * 2 != w - 1
            cc += 1 if cc * 2 < w - 1 else -1
        elif cc in (0, w - 1):
            assert cr * 2 != h - 1
            cr += 1 if cr * 2 < h - 1 else -1
        else:
            raise AssertionError("ambiguous diamond fit without edge clipping")

    return _diamond_outline(shape, cr, cc, radius + 1)


def _paint_checked(
    out: np.ndarray,
    coords: set[tuple[int, int]],
    color: int,
    protected: np.ndarray,
    painted: np.ndarray,
) -> None:
    for r, c in coords:
        assert not protected[r, c]
        assert not painted[r, c] or int(out[r, c]) == color
        out[r, c] = color
        painted[r, c] = True


def main(grid: np.ndarray) -> np.ndarray:
    """Red diamond outlines contract by one layer and gray diamond outlines expand by one layer while other objects stay fixed.

    Input structure
    - A 16x16 orange-background grid contains one-cell-thick Manhattan-diamond outlines.
    - Red and gray outlines are the changing objects; other colored diamond outlines are static.
    - Some outlines may be clipped by the grid boundary, but their visible pixels still fit a clipped Manhattan-distance contour.
    Rule
    - Clear all red and gray pixels.
    - Red objects are redrawn with the same diamond center and radius minus one.
    - Gray objects are redrawn with radius plus one; an ambiguous edge-clipped growing arc continues away from the nearest perpendicular edge.
    Key insight
    - The task is not about moving whole objects: each changing object is a contour of a diamond, and the visible operation is one step of contour contraction or expansion.
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape == (16, 16)
    bg = get_bg(g)
    assert bg not in (RED, GRAY)

    colors = set(map(int, g.flatten()))
    assert RED in colors and GRAY in colors

    out = g.copy()
    changing = np.isin(g, [RED, GRAY])
    protected = (g != bg) & ~changing
    out[changing] = bg
    painted = np.zeros(g.shape, dtype=bool)

    for comp in find_components(g, bg=bg, connectivity=8, colors=[RED], return_format="dict"):
        cr, cc, radius, _ = _minimal_diamond_fit(comp["coords"], g.shape)
        assert radius >= 1
        _paint_checked(out, _diamond_outline(g.shape, cr, cc, radius - 1), RED, protected, painted)

    for comp in find_components(g, bg=bg, connectivity=8, colors=[GRAY], return_format="dict"):
        _paint_checked(out, _gray_outline(comp["coords"], g.shape), GRAY, protected, painted)

    return out

# --------------- generator (att1) ---------------

# Canvas and palette.
GEN_SHAPE = (16, 16)
BG = 7
_gen_RED = RED
_gen_GRAY = GRAY
STATIC_COLORS = [8, 9]

# Object counts and radius ranges.
COUNT_RANGE = (1, 2)
RED_OUT_RADIUS_RANGE = (0, 5)
GRAY_OUT_RADIUS_RANGE = (1, 5)
STATIC_RADIUS_RANGE = (0, 2)

# Placement retries and clipping controls.
PLACE_TRIES = 50
GEN_ATTEMPTS = 300
CLIP_PROB = 0.3
MIN_CLIPPED_RADIUS = 3
EDGES = ["top", "bottom", "left", "right"]


def _gen_diamond_pixels(shape, cr, cc, radius):
    """Get visible pixels of a diamond outline clipped to grid."""
    h, w = shape
    if radius == 0:
        if 0 <= cr < h and 0 <= cc < w:
            return {(cr, cc)}
        return set()
    pts = set()
    for i in range(radius):
        for r, c in [(cr - radius + i, cc + i), (cr + i, cc + radius - i),
                     (cr + radius - i, cc - i), (cr - i, cc - radius + i)]:
            if 0 <= r < h and 0 <= c < w:
                pts.add((r, c))
    return pts


def _choose_counts():
    return (
        random.randint(*COUNT_RANGE),
        random.randint(*COUNT_RANGE),
        random.randint(*COUNT_RANGE),
    )


def _choose_static_colors(n_static):
    if n_static == 1:
        return [random.choice(STATIC_COLORS)]
    return random.sample(STATIC_COLORS, 2)


def _random_edge_center(shape, max_r):
    """Sample a center just outside one edge, matching the original clipped placement."""
    h, w = shape
    edge = random.choice(EDGES)
    if edge == "top":
        return (
            random.randint(max(-max_r + 1, -max_r), min(max_r - 2, 0)),
            random.randint(max_r, w - 1 - max_r),
        )
    if edge == "bottom":
        return (
            random.randint(
                max(h - 1, h - max_r + 1),
                min(h + max_r - 2, h + max_r - 1),
            ),
            random.randint(max_r, w - 1 - max_r),
        )
    if edge == "left":
        return (
            random.randint(max_r, h - 1 - max_r),
            random.randint(max(-max_r + 1, -max_r), min(max_r - 2, 0)),
        )
    return (
        random.randint(max_r, h - 1 - max_r),
        random.randint(
            max(w - 1, w - max_r + 1),
            min(w + max_r - 2, w + max_r - 1),
        ),
    )


def _random_center(shape, max_r, allow_clip):
    h, w = shape
    if allow_clip and random.random() < CLIP_PROB and max_r >= MIN_CLIPPED_RADIUS:
        return _random_edge_center(shape, max_r)
    return random.randint(0, h - 1), random.randint(0, w - 1)


def _place_object(shape, occupied, out_r, in_r, allow_clip=False):
    """Return a non-overlapping visible center, or None."""
    max_r = max(out_r, in_r)
    for _ in range(PLACE_TRIES):
        cr, cc = _random_center(shape, max_r, allow_clip)
        out_pixels = _gen_diamond_pixels(shape, cr, cc, out_r)
        in_pixels = _gen_diamond_pixels(shape, cr, cc, in_r)
        all_pixels = out_pixels | in_pixels
        if out_pixels and in_pixels and not (all_pixels & occupied):
            occupied.update(all_pixels)
            return cr, cc
    return None


def _append_object(objects, occupied, color, out_r, in_r, allow_clip=False):
    placed = _place_object(GEN_SHAPE, occupied, out_r, in_r, allow_clip)
    if placed is None:
        return False
    cr, cc = placed
    objects.append((color, cr, cc, out_r, in_r))
    return True


def _plan_objects():
    n_red, n_gray, n_static = _choose_counts()
    static_cols = _choose_static_colors(n_static)
    objects = []
    occupied = set()

    for _ in range(n_red):
        out_r = random.randint(*RED_OUT_RADIUS_RANGE)
        if not _append_object(objects, occupied, _gen_RED, out_r, out_r + 1, allow_clip=True):
            return None

    for _ in range(n_gray):
        out_r = random.randint(*GRAY_OUT_RADIUS_RANGE)
        if not _append_object(objects, occupied, _gen_GRAY, out_r, out_r - 1):
            return None

    for i in range(n_static):
        radius = random.randint(*STATIC_RADIUS_RANGE)
        if not _append_object(objects, occupied, static_cols[i], radius, radius):
            return None

    return objects


def _render_pair(objects):
    input_grid = np.full(GEN_SHAPE, BG, dtype=int)
    output_grid = np.full(GEN_SHAPE, BG, dtype=int)

    for color, cr, cc, out_r, in_r in objects:
        for r, c in _gen_diamond_pixels(GEN_SHAPE, cr, cc, in_r):
            input_grid[r, c] = color
        for r, c in _gen_diamond_pixels(GEN_SHAPE, cr, cc, out_r):
            output_grid[r, c] = color

    return input_grid, output_grid


def _passes_oracle(input_grid, output_grid):
    try:
        return np.array_equal(main(input_grid), output_grid)
    except Exception:
        return False


def _gen_attempt():
    objects = _plan_objects()
    if objects is None:
        return None

    input_grid, output_grid = _render_pair(objects)
    if _passes_oracle(input_grid, output_grid):
        return input_grid
    return None


def generate_input():
    for _ in range(GEN_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_ATTEMPTS} attempts")
