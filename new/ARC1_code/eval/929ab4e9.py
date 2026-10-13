import numpy as np
from arc_utils import paste


def _symmetry_orbit(n: int, r: int, c: int) -> tuple[tuple[int, int], ...]:
    last = int(n) - 1
    return tuple(
        sorted(
            {
                (r, c),
                (c, r),
                (r, last - c),
                (last - r, c),
                (last - r, last - c),
                (last - c, last - r),
                (c, last - r),
                (last - c, r),
            }
        )
    )


def _all_symmetry_orbits(n: int):
    seen = set()
    for r in range(int(n)):
        for c in range(int(n)):
            orbit = _symmetry_orbit(n, r, c)
            if orbit not in seen:
                seen.add(orbit)
                yield orbit


def _mask_color(g: np.ndarray) -> int:
    n, w = g.shape
    assert n == w, "expected a square symmetry pattern"
    candidates = []

    for color in np.unique(g):
        color = int(color)
        ok = True
        for orbit in _all_symmetry_orbits(n):
            values = [int(g[r, c]) for r, c in orbit]
            visible = {v for v in values if v != color}
            if len(visible) > 1:
                ok = False
                break
            if color in values and len(visible) != 1:
                ok = False
                break
        if ok:
            candidates.append(color)

    assert len(candidates) == 1, "expected exactly one symmetry-mask color"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """Restore masked cells by completing the full mirror-and-rotation symmetry of the square pattern.

    Input structure
    - A square grid whose visible non-mask pixels are invariant under all eight symmetries of the square.
    - Exactly one color acts as a mask: ignoring it leaves each symmetry orbit with one visible color.

    Rule
    - Copy the input and replace each mask pixel with the unique visible color from its eight-way symmetry orbit.
    - Preserve every non-mask pixel.

    Key insight
    - Some masked cells hide all horizontal/vertical counterparts, so the needed orbit also includes diagonal reflections and quarter-turn rotations.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    n, w = g.shape
    assert n == w, "expected a square grid"

    mask_color = _mask_color(g)
    out = g.copy()

    for orbit in _all_symmetry_orbits(n):
        values = [int(g[r, c]) for r, c in orbit]
        visible = {v for v in values if v != mask_color}
        assert len(visible) == 1, "each symmetry orbit must expose one color"
        fill_color = visible.pop()
        for r, c in orbit:
            if int(g[r, c]) == mask_color:
                out[r, c] = fill_color
            else:
                assert int(g[r, c]) == fill_color, "visible cells must already match the symmetry"

    assert not np.any(out == mask_color), "all mask pixels should be restored"
    return out

# --------------- generator (o5x) ---------------

# Canvas and color vocabulary.
_GEN_N = 24
_GEN_MASK_COLOR = 2
_GEN_COLORS = np.array([1, 3, 4, 5, 6, 7, 8, 9], dtype=int)
_GEN_PALETTE_SIZE_MIN = 7
_GEN_PALETTE_SIZE_MAX = 8

# Rectangle mask geometry.
_GEN_RECT_COUNT_MIN = 2
_GEN_RECT_COUNT_MAX = 4
_GEN_RECT_HEIGHT_MIN = 2
_GEN_RECT_HEIGHT_MAX = 6
_GEN_RECT_WIDTH_MIN = 2
_GEN_RECT_WIDTH_MAX = 6
_GEN_MASK_AREA_MIN = 18
_GEN_MASK_AREA_MAX = 96

# Per-orbit mask cap: at least half of every D4 orbit remains visible.
_GEN_ORBIT_MASK_LIMIT_DIVISOR = 2

# Retry budget.
_GEN_MAX_ATTEMPTS = 300

_GEN_RECT_DIMS = tuple(
    (h, w)
    for h in range(_GEN_RECT_HEIGHT_MIN, _GEN_RECT_HEIGHT_MAX + 1)
    for w in range(_GEN_RECT_WIDTH_MIN, _GEN_RECT_WIDTH_MAX + 1)
)
_GEN_RECT_AREAS = tuple(sorted({h * w for h, w in _GEN_RECT_DIMS}))
_GEN_SUMS_BY_COUNT = None
_GEN_ORBITS = None
_GEN_ORBIT_INDEX = None
_GEN_ORBIT_LIMITS = None
_GEN_AREA_BAG = []


def _gen_symmetry_data():
    """Cache D4 orbits, cell-to-orbit ids, and each orbit's mask cap."""
    global _GEN_ORBITS, _GEN_ORBIT_INDEX, _GEN_ORBIT_LIMITS
    if _GEN_ORBITS is not None:
        return _GEN_ORBITS, _GEN_ORBIT_INDEX, _GEN_ORBIT_LIMITS

    seen = set()
    orbits = []
    orbit_index = np.full((_GEN_N, _GEN_N), -1, dtype=int)
    for r in range(_GEN_N):
        for c in range(_GEN_N):
            orbit = _symmetry_orbit(_GEN_N, r, c)
            if orbit in seen:
                continue
            seen.add(orbit)
            orbit_id = len(orbits)
            orbits.append(orbit)
            for rr, cc in orbit:
                orbit_index[rr, cc] = orbit_id

    limits = np.array(
        [len(orbit) // _GEN_ORBIT_MASK_LIMIT_DIVISOR for orbit in orbits],
        dtype=int,
    )
    _GEN_ORBITS = tuple(orbits)
    _GEN_ORBIT_INDEX = orbit_index
    _GEN_ORBIT_LIMITS = limits
    return _GEN_ORBITS, _GEN_ORBIT_INDEX, _GEN_ORBIT_LIMITS


def _gen_possible_sums():
    """Return reachable rectangle-area sums keyed by rectangle count."""
    global _GEN_SUMS_BY_COUNT
    if _GEN_SUMS_BY_COUNT is not None:
        return _GEN_SUMS_BY_COUNT

    sums_by_count = {0: {0}}
    for count in range(1, _GEN_RECT_COUNT_MAX + 1):
        sums_by_count[count] = {
            prev + area
            for prev in sums_by_count[count - 1]
            for area in _GEN_RECT_AREAS
        }
    _GEN_SUMS_BY_COUNT = sums_by_count
    return sums_by_count


def _gen_next_mask_area(rng: np.random.Generator) -> int:
    """Draw each target area once per shuffled bag cycle."""
    if not _GEN_AREA_BAG:
        _GEN_AREA_BAG.extend(range(_GEN_MASK_AREA_MIN, _GEN_MASK_AREA_MAX + 1))
        rng.shuffle(_GEN_AREA_BAG)
    return int(_GEN_AREA_BAG.pop())


def _gen_rectangles(rng: np.random.Generator) -> list[tuple[int, int]]:
    """Choose rectangle dimensions whose raw areas sum to the target mask area."""
    sums_by_count = _gen_possible_sums()
    target = _gen_next_mask_area(rng)
    feasible_counts = [
        count
        for count in range(_GEN_RECT_COUNT_MIN, _GEN_RECT_COUNT_MAX + 1)
        if target in sums_by_count[count]
    ]
    count = int(feasible_counts[int(rng.integers(0, len(feasible_counts)))])

    dims = []
    remaining = target
    for idx in range(count):
        left = count - idx - 1
        valid_dims = [
            (h, w)
            for h, w in _GEN_RECT_DIMS
            if remaining - h * w in sums_by_count[left]
        ]
        h, w = valid_dims[int(rng.integers(0, len(valid_dims)))]
        dims.append((h, w))
        remaining -= h * w

    rng.shuffle(dims)
    return dims


def _gen_symmetric_output(rng: np.random.Generator) -> np.ndarray:
    """Create the fully D4-symmetric answer grid before masking."""
    orbits, _, _ = _gen_symmetry_data()
    palette_size = int(
        rng.integers(_GEN_PALETTE_SIZE_MIN, _GEN_PALETTE_SIZE_MAX + 1)
    )
    palette = rng.choice(_GEN_COLORS, size=palette_size, replace=False)

    assignments = []
    repeats, extra = divmod(len(orbits), palette_size)
    for color in palette:
        assignments.extend([int(color)] * repeats)
    if extra:
        assignments.extend(int(c) for c in rng.choice(palette, size=extra, replace=True))
    rng.shuffle(assignments)

    out = np.empty((_GEN_N, _GEN_N), dtype=int)
    for orbit, color in zip(orbits, assignments):
        for r, c in orbit:
            out[r, c] = color
    return out


def _gen_touches_existing(mask: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    """Reject overlap and orthogonal contact so rectangles remain separate blobs."""
    if np.any(mask[r0 : r0 + h, c0 : c0 + w]):
        return True
    if r0 > 0 and np.any(mask[r0 - 1, c0 : c0 + w]):
        return True
    if r0 + h < _GEN_N and np.any(mask[r0 + h, c0 : c0 + w]):
        return True
    if c0 > 0 and np.any(mask[r0 : r0 + h, c0 - 1]):
        return True
    if c0 + w < _GEN_N and np.any(mask[r0 : r0 + h, c0 + w]):
        return True
    return False


def _gen_rect_orbit_delta(
    orbit_index: np.ndarray,
    r0: int,
    c0: int,
    h: int,
    w: int,
) -> tuple[np.ndarray, np.ndarray]:
    ids = orbit_index[r0 : r0 + h, c0 : c0 + w].ravel()
    return np.unique(ids, return_counts=True)


def _gen_valid_locations(
    mask: np.ndarray,
    orbit_counts: np.ndarray,
    h: int,
    w: int,
    orbit_index: np.ndarray,
    orbit_limits: np.ndarray,
) -> list[tuple[int, int]]:
    """Return placements that keep every intersected orbit below its mask cap."""
    locations = []
    for r0 in range(_GEN_N - h + 1):
        for c0 in range(_GEN_N - w + 1):
            if _gen_touches_existing(mask, r0, c0, h, w):
                continue
            unique_ids, added = _gen_rect_orbit_delta(orbit_index, r0, c0, h, w)
            if np.any(orbit_counts[unique_ids] + added > orbit_limits[unique_ids]):
                continue
            locations.append((r0, c0))
    return locations


def _gen_mask(rng: np.random.Generator) -> np.ndarray | None:
    """Build a separated rectangle mask that still exposes every D4 orbit."""
    _, orbit_index, orbit_limits = _gen_symmetry_data()
    mask = np.zeros((_GEN_N, _GEN_N), dtype=int)
    orbit_counts = np.zeros(len(orbit_limits), dtype=int)

    for h, w in _gen_rectangles(rng):
        locations = _gen_valid_locations(mask, orbit_counts, h, w, orbit_index, orbit_limits)
        if not locations:
            return None
        r0, c0 = locations[int(rng.integers(0, len(locations)))]
        unique_ids, added = _gen_rect_orbit_delta(orbit_index, r0, c0, h, w)
        orbit_counts[unique_ids] += added
        paste(mask, np.ones((h, w), dtype=int), r0, c0, transparent=0)

    area = int(mask.sum())
    if not _GEN_MASK_AREA_MIN <= area <= _GEN_MASK_AREA_MAX:
        return None
    return mask.astype(bool)


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    out = _gen_symmetric_output(rng)
    mask = _gen_mask(rng)
    if mask is None:
        return None

    inp = out.copy()
    inp[mask] = _GEN_MASK_COLOR
    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid 929ab4e9 input")
