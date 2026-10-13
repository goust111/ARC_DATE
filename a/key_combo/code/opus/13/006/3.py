import numpy as np


GRID_SIZE = 11
CENTER = GRID_SIZE // 2
WALL = 9
BG = 0
YELLOW = 4
GREY = 5
MARKER_COLORS = (1, 2, 3, 6, 7)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox_area(coords: np.ndarray) -> int:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r1 - r0 + 1) * int(c1 - c0 + 1)


def _chebyshev_distance(coords: np.ndarray) -> int:
    dr = np.abs(coords[:, 0] - CENTER)
    dc = np.abs(coords[:, 1] - CENTER)
    return int(np.min(np.maximum(dr, dc)))


def _analyze_regions(grid: np.ndarray) -> list[dict]:
    regions = []
    for coords in _components(grid != WALL):
        marker_coords = []
        for r, c in coords:
            value = int(grid[r, c])
            if value not in (BG, WALL):
                marker_coords.append((int(r), int(c), value))
        assert len(marker_coords) <= 1, "each region may contain at most one marker pixel"
        marker_color = marker_coords[0][2] if marker_coords else None
        regions.append(
            {
                "coords": coords,
                "distance": _chebyshev_distance(coords),
                "complete": int(coords.shape[0]) == _bbox_area(coords),
                "marker_color": marker_color,
            }
        )
    return regions


def main(grid: np.ndarray) -> np.ndarray:
    """Encode each region's integrity/marker state as a centered square-ring color palette.

    Input structure:
    - The input is an 11x11 grid on a black background with maroon wall pixels forming a broken rectilinear lattice.
    - Non-wall cells form exactly six 4-connected regions; some are perfect rectangles, others are merged non-rectangular regions created by missing wall pixels.
    - A region may contain one isolated marker pixel in blue/red/green/magenta/orange, and the center cell belongs to a complete marked region.

    Rule:
    - Ignore the maroon lattice itself and treat every non-wall connected region as one interval.
    - For each region, compare its area with its inclusive bounding-box area to decide whether it is complete or defective, and map the region to the minimum Chebyshev distance of any of its cells from the grid center.
    - Build an 11x11 concentric square-ring output: a complete marked region contributes its marker color to its ring, a defective region forces its ring to yellow, and a complete unmarked region makes its ring grey.

    Key insight:
    - The lattice is only a hidden spatial codebook: its intact versus broken compartments determine the ring colors after regions are re-indexed by Chebyshev distance from the center.
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape == (GRID_SIZE, GRID_SIZE), "expected an 11x11 grid"

    regions = _analyze_regions(g)
    assert len(regions) == 6, "expected exactly six regions"

    ring_colors: dict[int, int] = {}
    for region in regions:
        dist = region["distance"]
        assert dist not in ring_colors, "each Chebyshev distance must map to one region"
        if region["complete"] and region["marker_color"] is not None:
            ring_colors[dist] = int(region["marker_color"])
        elif not region["complete"]:
            ring_colors[dist] = YELLOW
        else:
            ring_colors[dist] = GREY

    assert set(ring_colors) == set(range(CENTER + 1)), "missing ring distances"
    assert ring_colors[0] not in (YELLOW, GREY), "center region must be complete and marked"

    rr = np.arange(GRID_SIZE)[:, None]
    cc = np.arange(GRID_SIZE)[None, :]
    distances = np.maximum(np.abs(rr - CENTER), np.abs(cc - CENTER))
    out = np.empty_like(g)
    for dist, color in ring_colors.items():
        out[distances == dist] = int(color)
    return out


_SHAPE_LIBRARY = {
    0: {
        "complete": (
            ((5, 5),),
        ),
        "defective": (),
    },
    1: {
        "complete": (
            ((4, 4),),
            ((4, 3), (4, 4)),
            ((3, 4), (4, 4)),
            ((3, 3), (3, 4), (4, 3), (4, 4)),
        ),
        "defective": (
            ((3, 3), (3, 4), (4, 4)),
            ((3, 4), (4, 3), (4, 4)),
            ((3, 3), (4, 3), (4, 4)),
        ),
    },
    2: {
        "complete": (
            ((4, 7),),
            ((4, 7), (4, 8)),
            ((3, 7), (4, 7)),
            ((3, 7), (3, 8), (4, 7), (4, 8)),
        ),
        "defective": (
            ((3, 7), (3, 8), (4, 8)),
            ((3, 7), (4, 7), (4, 8)),
            ((3, 8), (4, 7), (4, 8)),
        ),
    },
    3: {
        "complete": (
            ((8, 4),),
            ((8, 4), (8, 5), (8, 6)),
            ((8, 4), (9, 4)),
            ((8, 5), (9, 5)),
            ((8, 4), (8, 5), (9, 4), (9, 5)),
        ),
        "defective": (
            ((8, 4), (8, 5), (9, 5)),
            ((8, 5), (8, 6), (9, 6)),
            ((8, 4), (8, 5), (8, 6), (9, 6)),
        ),
    },
    4: {
        "complete": (
            ((1, 1),),
            ((1, 1), (1, 2), (1, 3)),
            ((0, 2), (1, 2)),
            ((0, 2), (0, 3), (1, 2), (1, 3)),
        ),
        "defective": (
            ((0, 1), (1, 1), (1, 2)),
            ((0, 2), (1, 2), (1, 3)),
            ((0, 1), (0, 2), (1, 2)),
        ),
    },
    5: {
        "complete": (
            ((0, 10),),
            ((0, 9), (0, 10)),
            ((0, 10), (1, 10)),
            ((0, 8), (0, 9), (0, 10)),
        ),
        "defective": (
            ((0, 9), (0, 10), (1, 10)),
        ),
    },
}


_STATUS_PATTERNS = (
    ("complete_marked", "complete_unmarked", "defective", "complete_marked", "defective"),
    ("complete_marked", "defective", "complete_unmarked", "defective", "complete_marked"),
    ("complete_unmarked", "complete_marked", "defective", "complete_unmarked", "defective"),
    ("defective", "complete_marked", "complete_unmarked", "defective", "complete_marked"),
)


def _choose_statuses(rng: np.random.Generator) -> dict[int, tuple[str, bool]]:
    pattern = _STATUS_PATTERNS[int(rng.integers(len(_STATUS_PATTERNS)))]
    statuses = {0: ("complete", True)}
    defective_dists = []
    marked_count = 1
    for dist, code in zip(range(1, 6), pattern):
        if code == "complete_marked":
            statuses[dist] = ("complete", True)
            marked_count += 1
        elif code == "complete_unmarked":
            statuses[dist] = ("complete", False)
        else:
            statuses[dist] = ("defective", False)
            defective_dists.append(dist)

    extra_slots = min(2, len(defective_dists), len(MARKER_COLORS) - marked_count)
    extra_marked = rng.choice(defective_dists, size=int(rng.integers(extra_slots + 1)), replace=False)
    for dist in np.atleast_1d(extra_marked):
        statuses[int(dist)] = ("defective", True)
    return statuses


def _paint_region(grid: np.ndarray, coords: tuple[tuple[int, int], ...], marker_color: int | None, rng: np.random.Generator) -> None:
    for r, c in coords:
        grid[r, c] = BG
    if marker_color is not None:
        marker_r, marker_c = coords[int(rng.integers(len(coords)))]
        grid[marker_r, marker_c] = int(marker_color)


def _apply_transform(grid: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    transformed = np.rot90(grid, int(rng.integers(4)))
    if int(rng.integers(2)):
        transformed = np.fliplr(transformed)
    return transformed.copy()


def _build_candidate(rng: np.random.Generator) -> np.ndarray:
    grid = np.full((GRID_SIZE, GRID_SIZE), WALL, dtype=int)
    statuses = _choose_statuses(rng)

    marked_dists = [dist for dist in range(6) if statuses[dist][1]]
    palette = list(MARKER_COLORS)
    rng.shuffle(palette)
    marker_assignments = {dist: palette[i] for i, dist in enumerate(marked_dists)}

    for dist in range(6):
        shape_kind, marked = statuses[dist]
        choices = _SHAPE_LIBRARY[dist][shape_kind]
        coords = choices[int(rng.integers(len(choices)))]
        _paint_region(grid, coords, marker_assignments.get(dist) if marked else None, rng)

    return _apply_transform(grid, rng)


def _valid_candidate(grid: np.ndarray) -> bool:
    try:
        regions = _analyze_regions(grid)
        if len(regions) != 6:
            return False
        distances = sorted(region["distance"] for region in regions)
        if distances != list(range(6)):
            return False
        if not any(region["complete"] and region["marker_color"] is None for region in regions):
            return False
        if not any(not region["complete"] for region in regions):
            return False
        center_region = next(region for region in regions if region["distance"] == 0)
        if not center_region["complete"] or center_region["marker_color"] is None:
            return False
        out = main(grid)
        return out.shape == (GRID_SIZE, GRID_SIZE) and not np.any(out == WALL)
    except Exception:
        return False


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(300):
        grid = _build_candidate(rng)
        if _valid_candidate(grid):
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")
