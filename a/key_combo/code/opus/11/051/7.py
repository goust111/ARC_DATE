import random
from collections import Counter, deque

import numpy as np


BASE_COLORS = (1, 4, 5)
SQUARE_COLORS = (2, 3, 7)
SIDE_CYCLE = (1, 3, 5, 7, 9)
GRID_SIZES = (13, 15, 17, 19, 21)
MAX_GENERATION_ATTEMPTS = 400


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    """BBox of an (N,2) coordinate array using the ARC (r,c) convention."""
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    """Connected components of a boolean mask as (N,2) coordinate arrays."""
    assert connectivity == 4
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.asarray(coords, dtype=int))
    return out


def find_monochrome_components(grid: np.ndarray, colors: tuple[int, ...]) -> list[dict]:
    """Minimal local variant of arc_utils.find_components(..., monochromatic=True)."""
    out: list[dict] = []
    for color in colors:
        comps = components_mask(grid == color, connectivity=4)
        for coords in comps:
            out.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": bbox_coords(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return out


def d4_orbit(r: int, c: int, size: int) -> tuple[tuple[int, int], ...]:
    """D4 orbit in a square size x size grid."""
    last = int(size - 1)
    pts = {
        (r, c),
        (r, last - c),
        (last - r, c),
        (last - r, last - c),
        (c, r),
        (c, last - r),
        (last - c, r),
        (last - c, last - r),
    }
    return tuple(sorted(pts))


def square_bbox(center: tuple[int, int], side: int) -> tuple[int, int, int, int]:
    """Inclusive bbox for an odd-sided square with the given center."""
    cr, cc = map(int, center)
    half = int(side // 2)
    return cr - half, cc - half, cr + half, cc + half


def next_side(side: int) -> int:
    """Odd side cycle 1->9, otherwise n->n-2."""
    assert side in SIDE_CYCLE
    return 9 if side == 1 else side - 2


def paint_square(grid: np.ndarray, center: tuple[int, int], side: int, color: int) -> None:
    """Paint a solid odd square in-place."""
    r0, c0, r1, c1 = square_bbox(center, side)
    grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)


def render_squares(shape: tuple[int, int], specs: list[dict], *, use_new_side: bool) -> np.ndarray:
    """Render a boolean mask of all squares in either input or output state."""
    mask = np.zeros(shape, dtype=bool)
    for spec in specs:
        side = int(spec["new_side"] if use_new_side else spec["side"])
        r0, c0, r1, c1 = square_bbox(spec["center"], side)
        mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


def canonical_pattern(size: int, rng: np.random.Generator) -> np.ndarray:
    """Generate an irregular but exact D4-symmetric base pattern on {1,4,5}."""
    grid = np.full((size, size), -1, dtype=int)
    reps: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    last = size - 1

    def rep_key(pos: tuple[int, int]) -> tuple[int, int, int]:
        r, c = pos
        edge = min(r, c, last - r, last - c)
        diag = min(abs(r - c), abs((r + c) - last))
        return int(edge), int(diag), int(r + c)

    for r in range(size):
        for c in range(size):
            rep = min(d4_orbit(r, c, size))
            if rep not in seen:
                seen.add(rep)
                reps.append(rep)
    reps.sort(key=rep_key)

    for rep in reps:
        orbit = d4_orbit(rep[0], rep[1], size)
        neighbor_colors: list[int] = []
        for rr, cc in orbit:
            for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                if 0 <= nr < size and 0 <= nc < size and grid[nr, nc] != -1:
                    neighbor_colors.append(int(grid[nr, nc]))
        if neighbor_colors and rng.random() < 0.72:
            counts = Counter(neighbor_colors)
            best = max(counts.values())
            choices = [color for color, count in counts.items() if count == best]
            color = int(rng.choice(choices))
        else:
            color = int(rng.choice(BASE_COLORS))
        for rr, cc in orbit:
            grid[rr, cc] = color

    assert set(np.unique(grid)) <= set(BASE_COLORS)
    return grid


def infer_exposed_colors(grid: np.ndarray, specs: list[dict]) -> np.ndarray:
    """Recover the newly exposed pixels by reading visible cells from D4 orbits."""
    out = np.asarray(grid, dtype=int).copy()
    old_mask = render_squares(grid.shape, specs, use_new_side=False)
    new_mask = render_squares(grid.shape, specs, use_new_side=True)
    blocked = old_mask | new_mask
    size = int(grid.shape[0])

    for spec in specs:
        color = int(spec["color"])
        old_side = int(spec["side"])
        new_side = int(spec["new_side"])
        old_r0, old_c0, old_r1, old_c1 = square_bbox(spec["center"], old_side)
        nr0, nc0, nr1, nc1 = square_bbox(spec["center"], new_side)
        for r in range(old_r0, old_r1 + 1):
            for c in range(old_c0, old_c1 + 1):
                if nr0 <= r <= nr1 and nc0 <= c <= nc1:
                    continue
                values = {
                    int(grid[rr, cc])
                    for rr, cc in d4_orbit(r, c, size)
                    if not blocked[rr, cc]
                }
                assert len(values) == 1, "each exposed pixel must have one visible D4 color"
                out[r, c] = values.pop()
        paint_square(out, spec["center"], new_side, color)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Shrink each odd square by the side-length cycle and refill the exposed ring from the hidden D4-symmetric base pattern.

    Input structure:
    - The grid is square and contains a full-grid base pattern using only blue, yellow, and grey.
    - The base pattern is globally D4-symmetric, but some cells are covered by 2-4 solid odd-sided squares.
    - The covering squares use only red, green, and orange, are 4-connected, and have side lengths in {1,3,5,7,9}.

    Rule:
    - Detect every monochrome square object, keep its center and color, and cycle its side length by n->n-2 and 1->9.
    - Pixels from the old square that are no longer covered by the new square become exposed.
    - For each exposed pixel, inspect its D4 orbit over horizontal, vertical, and diagonal reflections and copy the unique color seen at orbit mates that are covered by neither an old square nor a new square.
    - Paint every transformed square at its new size on top of the recovered base pattern.

    Key insight:
    - The resize step determines exactly which ring becomes visible, and the revealed colors are not background defaults but the corresponding values of a global D4-symmetric layer hidden underneath.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] == g.shape[1], "expected a square 2D grid"
    assert max(g.shape) <= 30, "ARC grids must fit within 30x30"
    assert set(map(int, np.unique(g))) <= set(BASE_COLORS) | set(SQUARE_COLORS)

    specs: list[dict] = []
    seen_centers: set[tuple[int, int]] = set()
    for obj in find_monochrome_components(g, SQUARE_COLORS):
        color = int(obj["color"])
        r0, c0, r1, c1 = obj["bbox"]
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        assert h == w and h in SIDE_CYCLE, "foreground objects must be odd-sided squares in the resize cycle"
        assert int(obj["area"]) == h * h, "foreground squares must be solid"
        assert np.all(g[r0 : r1 + 1, c0 : c1 + 1] == color), "square bboxes must be monochrome and filled"
        center = ((r0 + r1) // 2, (c0 + c1) // 2)
        assert center not in seen_centers, "square centers must be distinct"
        seen_centers.add(center)
        specs.append(
            {
                "color": color,
                "center": center,
                "side": h,
                "new_side": next_side(h),
            }
        )

    assert 2 <= len(specs) <= 4, "expected 2-4 square objects"
    old_mask = render_squares(g.shape, specs, use_new_side=False)
    new_mask = render_squares(g.shape, specs, use_new_side=True)
    size = int(g.shape[0])
    for spec in specs:
        for side in (int(spec["side"]), int(spec["new_side"])):
            r0, c0, r1, c1 = square_bbox(spec["center"], side)
            assert 0 <= r0 <= r1 < size and 0 <= c0 <= c1 < size, "all old/new squares must stay in bounds"

    for color in SQUARE_COLORS:
        comps = components_mask(g == color, connectivity=4)
        assert len(comps) <= 1, "generator uses at most one square per square color"

    out = infer_exposed_colors(g, specs)
    for spec in specs:
        color = int(spec["color"])
        new_side = int(spec["new_side"])
        paint_square(out, spec["center"], new_side, color)
        r0, c0, r1, c1 = square_bbox(spec["center"], new_side)
        assert np.all(out[r0 : r1 + 1, c0 : c1 + 1] == color), "each transformed square must stay solid"
    return out


def expected_output(base: np.ndarray, specs: list[dict]) -> np.ndarray:
    """Overlay the transformed squares onto the unobscured base pattern."""
    out = np.asarray(base, dtype=int).copy()
    for spec in specs:
        paint_square(out, spec["center"], int(spec["new_side"]), int(spec["color"]))
    return out


def input_from_base(base: np.ndarray, specs: list[dict]) -> np.ndarray:
    """Overlay the original squares onto the base pattern."""
    inp = np.asarray(base, dtype=int).copy()
    for spec in specs:
        paint_square(inp, spec["center"], int(spec["side"]), int(spec["color"]))
    return inp


def visible_orbit_sources(shape: tuple[int, int], specs: list[dict], cell: tuple[int, int]) -> list[tuple[int, int]]:
    """Visible D4-mate cells outside every old/new square."""
    old_mask = render_squares(shape, specs, use_new_side=False)
    new_mask = render_squares(shape, specs, use_new_side=True)
    blocked = old_mask | new_mask
    return [pos for pos in d4_orbit(cell[0], cell[1], shape[0]) if not blocked[pos]]


def nonlocal_exposed_cell(out: np.ndarray, specs: list[dict]) -> bool:
    """At least one exposed cell should not be recoverable from adjacent visible neighbors alone."""
    old_mask = render_squares(out.shape, specs, use_new_side=False)
    new_mask = render_squares(out.shape, specs, use_new_side=True)
    exposed = old_mask & ~new_mask
    h, w = out.shape
    for r, c in np.argwhere(exposed):
        visible_neighbors = []
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and not old_mask[nr, nc]:
                visible_neighbors.append(int(out[nr, nc]))
        if visible_neighbors and all(color != int(out[r, c]) for color in visible_neighbors):
            return True
    return False


def square_specs_ok(size: int, specs: list[dict]) -> bool:
    """Validate placement constraints and orbit-source solvability before painting the pattern."""
    occupied = np.zeros((size, size), dtype=bool)
    for spec in specs:
        eff_side = max(int(spec["side"]), int(spec["new_side"]))
        r0, c0, r1, c1 = square_bbox(spec["center"], eff_side)
        if not (0 <= r0 <= r1 < size and 0 <= c0 <= c1 < size):
            return False
        rr0 = max(0, r0 - 1)
        cc0 = max(0, c0 - 1)
        rr1 = min(size, r1 + 2)
        cc1 = min(size, c1 + 2)
        if occupied[rr0:rr1, cc0:cc1].any():
            return False
        occupied[r0 : r1 + 1, c0 : c1 + 1] = True

    old_mask = render_squares((size, size), specs, use_new_side=False)
    new_mask = render_squares((size, size), specs, use_new_side=True)
    exposed = old_mask & ~new_mask
    if exposed.sum() == 0:
        return False

    blocked = old_mask | new_mask
    has_multi_hidden_orbit = False
    for r, c in np.argwhere(exposed):
        orbit = d4_orbit(int(r), int(c), size)
        if not any(not blocked[rr, cc] for rr, cc in orbit):
            return False
        if sum(bool(blocked[rr, cc]) for rr, cc in orbit) >= 2:
            has_multi_hidden_orbit = True
    return has_multi_hidden_orbit


def choose_specs(size: int, rng: np.random.Generator) -> list[dict] | None:
    """Randomly place 2-3 colored odd squares while keeping both old and new states valid."""
    count = int(rng.integers(2, 4))
    colors = [int(c) for c in rng.choice(SQUARE_COLORS, size=count, replace=False)]
    include_wrap = bool(rng.random() < 0.35)

    sides: list[int] = []
    if include_wrap:
        sides.append(1)
    sides.append(int(rng.choice((5, 7, 9))))
    while len(sides) < count:
        sides.append(int(rng.choice((3, 5, 7, 9))))
    rng.shuffle(sides)

    paired = sorted(
        zip(colors, sides),
        key=lambda item: max(item[1], next_side(item[1])),
        reverse=True,
    )
    specs: list[dict] = []
    occupied = np.zeros((size, size), dtype=bool)

    for color, side in paired:
        new_side = next_side(side)
        eff_side = max(side, new_side)
        half = eff_side // 2
        candidates: list[tuple[int, int]] = []
        for cr in range(half, size - half):
            for cc in range(half, size - half):
                r0, c0, r1, c1 = square_bbox((cr, cc), eff_side)
                rr0 = max(0, r0 - 1)
                cc0 = max(0, c0 - 1)
                rr1 = min(size, r1 + 2)
                cc1 = min(size, c1 + 2)
                if not occupied[rr0:rr1, cc0:cc1].any():
                    candidates.append((cr, cc))
        if not candidates:
            return None
        center = candidates[int(rng.integers(len(candidates)))]
        r0, c0, r1, c1 = square_bbox(center, eff_side)
        occupied[r0 : r1 + 1, c0 : c1 + 1] = True
        specs.append(
            {
                "color": int(color),
                "center": center,
                "side": int(side),
                "new_side": int(new_side),
            }
        )

    if not square_specs_ok(size, specs):
        return None
    return specs


def pattern_is_good(base: np.ndarray, specs: list[dict]) -> bool:
    """Reject degenerate patterns that make the exposed repair visually trivial."""
    out = expected_output(base, specs)
    inp = input_from_base(base, specs)
    old_mask = render_squares(base.shape, specs, use_new_side=False)
    new_mask = render_squares(base.shape, specs, use_new_side=True)
    exposed = old_mask & ~new_mask
    if len(set(map(int, np.unique(base)))) < 3:
        return False
    if len(set(map(int, np.unique(out[exposed])))) < 2:
        return False
    if not nonlocal_exposed_cell(out, specs):
        return False
    if np.array_equal(inp, out):
        return False
    return True


def gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    """One randomized construction attempt."""
    size = int(rng.choice(GRID_SIZES))
    specs = choose_specs(size, rng)
    if specs is None:
        return None

    for _ in range(20):
        base = canonical_pattern(size, rng)
        if not pattern_is_good(base, specs):
            continue
        inp = input_from_base(base, specs)
        out = expected_output(base, specs)
        if not np.array_equal(main(inp), out):
            continue

        old_mask = render_squares(base.shape, specs, use_new_side=False)
        new_mask = render_squares(base.shape, specs, use_new_side=True)
        exposed = old_mask & ~new_mask
        for cell in map(tuple, np.argwhere(exposed)):
            values = {int(base[pos]) for pos in visible_orbit_sources(base.shape, specs, cell)}
            if len(values) != 1:
                return None
        return inp
    return None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng(random.randrange(1 << 30))
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
