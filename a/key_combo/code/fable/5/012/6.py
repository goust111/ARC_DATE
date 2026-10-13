from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
PALETTE = tuple(range(1, 10))
SIDES = ("top", "bottom", "left", "right")
GENERATION_ATTEMPTS = 400


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components as (N, 2) arrays of (row, column)."""
    seen = np.zeros(mask.shape, dtype=bool)
    components: list[np.ndarray] = []
    height, width = mask.shape

    for start_r, start_c in np.argwhere(mask):
        sr, sc = int(start_r), int(start_c)
        if seen[sr, sc]:
            continue
        seen[sr, sc] = True
        stack = [(sr, sc)]
        cells: list[tuple[int, int]] = []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        components.append(np.asarray(cells, dtype=int))
    return components


def _bar_candidates(grid: np.ndarray) -> list[tuple[str, int]]:
    """Find non-black colors that fill an entire outer edge."""
    edges = (
        ("top", grid[0, :]),
        ("bottom", grid[-1, :]),
        ("left", grid[:, 0]),
        ("right", grid[:, -1]),
    )
    candidates: list[tuple[str, int]] = []
    for side, edge in edges:
        color = int(edge[0])
        if color != BACKGROUND and np.all(edge == color):
            candidates.append((side, color))
    return candidates


def _bar_mask(shape: tuple[int, int], side: str) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    if side == "top":
        mask[0, :] = True
    elif side == "bottom":
        mask[-1, :] = True
    elif side == "left":
        mask[:, 0] = True
    else:
        mask[:, -1] = True
    return mask


def _objects(grid: np.ndarray, bar_color: int, bar: np.ndarray) -> list[dict]:
    """Extract monochrome 4-connected objects after excluding the fixed bar."""
    objects: list[dict] = []
    colors = [int(c) for c in np.unique(grid) if int(c) not in (BACKGROUND, bar_color)]
    for color in colors:
        object_mask = (grid == color) & ~bar
        for coords in _components4(object_mask):
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            objects.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(coords.shape[0]),
                }
            )
    return objects


def _shift_coords(coords: np.ndarray, side: str, amount: int, shape: tuple[int, int]) -> np.ndarray:
    """Translate in the bar-defined direction on the playable cyclic track."""
    height, width = shape
    moved = coords.copy()
    if side == "top":
        track = height - 1
        moved[:, 0] = 1 + (coords[:, 0] - 1 - amount) % track
    elif side == "bottom":
        track = height - 1
        moved[:, 0] = (coords[:, 0] + amount) % track
    elif side == "left":
        track = width - 1
        moved[:, 1] = 1 + (coords[:, 1] - 1 - amount) % track
    else:
        track = width - 1
        moved[:, 1] = (coords[:, 1] + amount) % track
    return moved


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move every object toward the full edge bar by its own area on a cyclic lane.

    Input structure:
    - The background is black, and one non-black color fills exactly one
      one-cell-thick outer edge bar.
    - All remaining foreground components are monochrome 4-connected objects
      whose row bands (for a side bar) or column bands (for a top/bottom bar)
      are pairwise disjoint.

    Rule:
    - Keep the full edge bar fixed and clear every object's original cells.
    - Translate each object rigidly toward the bar by its number of pixels.
      The playable rows or columns exclude the bar and form a ring, so pixels
      passing the bar-adjacent end reappear at the opposite outer edge.

    Key insight:
    - The bar simultaneously supplies the direction and joins each lane's two
      ends, while each object's visible area supplies its own displacement.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and min(g.shape) >= 3, "expected a two-dimensional ARC grid"

    candidates = _bar_candidates(g)
    assert len(candidates) == 1, "expected exactly one full non-black outer-edge bar"
    side, bar_color = candidates[0]
    bar = _bar_mask(g.shape, side)
    assert np.all(g[bar] == bar_color), "bar must be uniform"
    assert not np.any((g == bar_color) & ~bar), "bar color must be reserved for the bar"

    objects = _objects(g, bar_color, bar)
    assert objects, "expected at least one movable object"
    foreground = (g != BACKGROUND) & ~bar
    assert sum(obj["area"] for obj in objects) == int(foreground.sum()), (
        "all non-bar foreground cells must belong to monochrome objects"
    )

    # Side bars define horizontal motion and therefore row lanes.  Horizontal
    # bars define vertical motion and therefore column lanes.
    intervals: list[tuple[int, int]] = []
    for obj in objects:
        r0, c0, r1, c1 = obj["bbox"]
        interval = (r0, r1) if side in ("left", "right") else (c0, c1)
        for lo, hi in intervals:
            assert interval[1] < lo or hi < interval[0], "object lane bands must be disjoint"
        intervals.append(interval)

    out = g.copy()
    out[foreground] = BACKGROUND
    occupied = bar.copy()
    for obj in objects:
        moved = _shift_coords(obj["coords"], side, obj["area"], g.shape)
        rr, cc = moved[:, 0], moved[:, 1]
        assert not np.any(occupied[rr, cc]), "cyclic object destinations must not overlap"
        out[rr, cc] = obj["color"]
        occupied[rr, cc] = True

    assert np.array_equal(out[bar], g[bar]), "the full edge bar must remain unchanged"
    return out


def _random_polyomino(area: int, require_axis_span: bool) -> np.ndarray | None:
    """Grow an area-exact polyomino in (movement-axis, perpendicular) coordinates."""
    dimension_choices = [
        (axis_size, lane_size)
        for axis_size in range(1, min(4, area) + 1)
        for lane_size in range(1, min(3, area) + 1)
        if axis_size * lane_size >= area and (not require_axis_span or axis_size >= 2)
    ]
    if not dimension_choices:
        return None

    for _ in range(40):
        axis_size, lane_size = random.choice(dimension_choices)
        seed = (random.randrange(axis_size), random.randrange(lane_size))
        cells = {seed}
        while len(cells) < area:
            frontier: set[tuple[int, int]] = set()
            for axis, lane in cells:
                for da, dl in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    na, nl = axis + da, lane + dl
                    if 0 <= na < axis_size and 0 <= nl < lane_size and (na, nl) not in cells:
                        frontier.add((na, nl))
            if not frontier:
                break
            cells.add(random.choice(tuple(frontier)))
        if len(cells) != area:
            continue

        coords = np.asarray(sorted(cells), dtype=int)
        coords -= coords.min(axis=0)
        if require_axis_span and int(np.ptp(coords[:, 0])) < 1:
            continue
        return coords
    return None


def _distance_to_grid(axis_distance: int, lane: int, side: str, shape: tuple[int, int]) -> tuple[int, int]:
    """Map distance from the bar and an absolute lane coordinate to (r, c)."""
    height, width = shape
    if side == "top":
        return 1 + axis_distance, lane
    if side == "bottom":
        return height - 2 - axis_distance, lane
    if side == "left":
        return lane, 1 + axis_distance
    return lane, width - 2 - axis_distance


def _distribute_extra(total: int, slots: int) -> list[int]:
    shares = [0] * slots
    for _ in range(total):
        shares[random.randrange(slots)] += 1
    return shares


def _independent_expected(
    grid: np.ndarray,
    descriptors: list[dict],
    side: str,
    *,
    reverse: bool = False,
    fixed_shift: int | None = None,
) -> np.ndarray:
    """Render an oracle from generator metadata, independently of object parsing."""
    out = grid.copy()
    for desc in descriptors:
        coords = desc["grid_coords"]
        out[coords[:, 0], coords[:, 1]] = BACKGROUND

    track = (grid.shape[0] - 1) if side in ("top", "bottom") else (grid.shape[1] - 1)
    for desc in descriptors:
        amount = int(desc["area"] if fixed_shift is None else fixed_shift)
        if reverse:
            amount = -amount
        color = int(desc["color"])
        for axis, lane_offset in desc["local"]:
            old_distance = int(desc["start"]) + int(axis)
            new_distance = (old_distance - amount) % track
            r, c = _distance_to_grid(new_distance, int(desc["lane_start"]) + int(lane_offset), side, grid.shape)
            assert out[r, c] == BACKGROUND
            out[r, c] = color
    return out


def _semantic_candidate_ok(grid: np.ndarray, descriptors: list[dict], side: str) -> bool:
    """Reject ambiguous cases and cases that admit the main tempting shortcuts."""
    candidates = _bar_candidates(grid)
    if len(candidates) != 1 or candidates[0][0] != side:
        return False

    try:
        actual = main(grid)
        expected = _independent_expected(grid, descriptors, side)
    except (AssertionError, ValueError):
        return False
    if not np.array_equal(actual, expected) or np.array_equal(actual, grid):
        return False

    # Parsing must recover precisely the intended objects, not merely an output
    # that happens to agree with the metadata-based oracle.
    bar_color = candidates[0][1]
    parsed = _objects(grid, bar_color, _bar_mask(grid.shape, side))
    intended = {frozenset(map(tuple, desc["grid_coords"])) for desc in descriptors}
    recovered = {frozenset(map(tuple, obj["coords"])) for obj in parsed}
    if intended != recovered:
        return False

    areas = [int(desc["area"]) for desc in descriptors]
    if len(set(areas)) != len(areas):
        return False
    if not any(bool(desc["wraps"]) for desc in descriptors):
        return False
    if not any(not bool(desc["wraps"]) for desc in descriptors):
        return False

    # A one-cell/fixed motion and motion away from the bar must both disagree.
    if np.array_equal(actual, _independent_expected(grid, descriptors, side, fixed_shift=1)):
        return False
    if np.array_equal(actual, _independent_expected(grid, descriptors, side, reverse=True)):
        return False

    # Every individual area's exact value is necessary: perturbing any one
    # displacement by one cell must change the target grid.
    for index, desc in enumerate(descriptors):
        altered = [dict(item) for item in descriptors]
        altered[index]["area"] = int(desc["area"]) + 1
        if np.array_equal(actual, _independent_expected(grid, altered, side)):
            return False
    return True


def _generation_attempt() -> np.ndarray | None:
    height = random.randint(11, 18)
    width = random.randint(11, 18)
    side = random.choice(SIDES)
    track = (height - 1) if side in ("top", "bottom") else (width - 1)
    lane_count = width if side in ("top", "bottom") else height
    object_count = random.randint(2, 4)

    possible_areas = list(range(2, min(8, track - 2) + 1))
    if len(possible_areas) < object_count:
        return None
    areas = random.sample(possible_areas, object_count)
    wrap_flags = [True, False] + [bool(random.getrandbits(1)) for _ in range(object_count - 2)]
    random.shuffle(wrap_flags)

    bar_color, *object_colors = random.sample(PALETTE, object_count + 1)
    descriptors: list[dict] = []
    for area, wraps, color in zip(areas, wrap_flags, object_colors):
        local = _random_polyomino(area, require_axis_span=wraps)
        if local is None:
            return None
        descriptors.append({"area": area, "wraps": wraps, "color": color, "local": local})

    # At least one genuinely two-dimensional shape prevents all objects from
    # looking like mere counters or lane markers.
    if not any(int(np.ptp(desc["local"][:, 1])) >= 1 for desc in descriptors):
        return None

    random.shuffle(descriptors)
    lane_widths = [int(desc["local"][:, 1].max()) + 1 for desc in descriptors]
    base_width = sum(lane_widths) + object_count - 1
    if base_width > lane_count:
        return None
    gaps = _distribute_extra(lane_count - base_width, object_count + 1)
    lane_cursor = gaps[0]
    for index, (desc, lane_width) in enumerate(zip(descriptors, lane_widths)):
        desc["lane_start"] = lane_cursor
        lane_cursor += lane_width
        if index < object_count - 1:
            lane_cursor += 1 + gaps[index + 1]

    grid = np.full((height, width), BACKGROUND, dtype=int)
    if side == "top":
        grid[0, :] = bar_color
    elif side == "bottom":
        grid[-1, :] = bar_color
    elif side == "left":
        grid[:, 0] = bar_color
    else:
        grid[:, -1] = bar_color

    for desc in descriptors:
        local = desc["local"]
        area = int(desc["area"])
        axis_span = int(local[:, 0].max()) + 1
        if desc["wraps"]:
            low = max(0, area - axis_span + 1)
            high = min(area - 1, track - axis_span)
        else:
            low = area
            high = track - axis_span
        if low > high:
            return None
        start = random.randint(low, high)
        desc["start"] = start

        distances = start + local[:, 0]
        crosses = (distances < area)
        if bool(desc["wraps"]) != bool(np.any(crosses)):
            return None
        if desc["wraps"] and not np.any(~crosses):
            return None

        coords = np.asarray(
            [
                _distance_to_grid(int(distance), int(desc["lane_start"]) + int(lane_offset), side, grid.shape)
                for distance, lane_offset in zip(distances, local[:, 1])
            ],
            dtype=int,
        )
        desc["grid_coords"] = coords
        if np.any(grid[coords[:, 0], coords[:, 1]] != BACKGROUND):
            return None
        grid[coords[:, 0], coords[:, 1]] = int(desc["color"])

    background_count = int(np.count_nonzero(grid == BACKGROUND))
    if background_count <= grid.size // 2:
        return None
    return grid if _semantic_candidate_ok(grid, descriptors, side) else None


def generate_input() -> np.ndarray:
    """Generate a diverse, oracle-checked instance with both wrap branches."""
    for _ in range(GENERATION_ATTEMPTS):
        grid = _generation_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GENERATION_ATTEMPTS} attempts")
