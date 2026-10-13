import numpy as np


BLACK = 0
GREY = 5
PALETTE = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)
BRANCHES = ("wide", "tall", "square")

GRID_H_RANGE = (18, 31)
GRID_W_RANGE = (18, 31)
FRAME_COUNT_RANGE = (3, 6)
INNER_SIZE_RANGE = (3, 8)
GEN_ATTEMPTS = 400

_RNG = np.random.default_rng()


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _grey_components(grid: np.ndarray) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != GREY:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == GREY:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _frame_kind(inner_h: int, inner_w: int) -> str:
    if inner_w > inner_h:
        return "wide"
    if inner_h > inner_w:
        return "tall"
    return "square"


def _frame_records(grid: np.ndarray) -> list[dict[str, int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    frames: list[dict[str, int]] = []
    for coords in _grey_components(g):
        rows = coords[:, 0]
        cols = coords[:, 1]
        r0 = int(rows.min())
        r1 = int(rows.max())
        c0 = int(cols.min())
        c1 = int(cols.max())
        outer_h = int(r1 - r0 + 1)
        outer_w = int(c1 - c0 + 1)
        assert outer_h >= 3 and outer_w >= 3, "frame must have non-empty interior"

        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        border = np.zeros((outer_h, outer_w), dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True

        assert int(coords.shape[0]) == int(border.sum()), "grey component must be a 1-pixel rectangle border"
        assert np.all(patch[border] == GREY), "frame perimeter must be grey"
        assert not np.any(patch[~border] == GREY), "frame interior must not contain grey"

        interior = patch[1:-1, 1:-1]
        seed_coords = np.argwhere((interior != BLACK) & (interior != GREY))
        assert int(seed_coords.shape[0]) == 1, "each frame must contain exactly one seed"
        seed_r, seed_c = map(int, seed_coords[0])
        seed_color = int(interior[seed_r, seed_c])

        frames.append(
            {
                "r0": r0,
                "c0": c0,
                "r1": r1,
                "c1": c1,
                "inner_h": outer_h - 2,
                "inner_w": outer_w - 2,
                "seed_r": seed_r,
                "seed_c": seed_c,
                "seed_color": seed_color,
            }
        )

    assert frames, "grid must contain at least one grey frame"
    colored_pixels = int(np.count_nonzero((g != BLACK) & (g != GREY)))
    assert colored_pixels == len(frames), "the only non-black, non-grey pixels must be seeds"
    frames.sort(key=lambda item: (item["r0"], item["c0"]))
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow striped patterns from a colored seed inside each grey rectangular frame.

    Input structure:
    - Background is black.
    - Grey pixels form 1-pixel-wide hollow rectangles.
    - Each rectangle contains exactly one colored seed pixel.

    Rule:
    - If the frame interior is wider than tall, fill every other row through the
      seed row with the seed color.
    - If the frame interior is taller than wide, fill every other column through
      the seed column with the seed color.
    - If the interior is square, fill the seed row and seed column to make a cross.
    - Keep the grey border unchanged.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for frame in _frame_records(g):
        r0 = frame["r0"]
        c0 = frame["c0"]
        r1 = frame["r1"]
        c1 = frame["c1"]
        inner_h = frame["inner_h"]
        inner_w = frame["inner_w"]
        seed_r = frame["seed_r"]
        seed_c = frame["seed_c"]
        seed_color = frame["seed_color"]

        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        interior[:, :] = BLACK

        if inner_w > inner_h:
            rows = np.arange(inner_h, dtype=int)
            fill_rows = np.abs(rows - seed_r) % 2 == 0
            interior[fill_rows, :] = seed_color
        elif inner_h > inner_w:
            cols = np.arange(inner_w, dtype=int)
            fill_cols = np.abs(cols - seed_c) % 2 == 0
            interior[:, fill_cols] = seed_color
        else:
            interior[seed_r, :] = seed_color
            interior[:, seed_c] = seed_color

    return out


def _balanced_branch_types(count: int) -> list[str]:
    base = count // len(BRANCHES)
    remainder = count % len(BRANCHES)
    branch_types = [branch for branch in BRANCHES for _ in range(base)]
    extras = list(_RNG.permutation(np.array(BRANCHES, dtype=object))[:remainder])
    branch_types.extend(map(str, extras))
    _RNG.shuffle(branch_types)
    return branch_types


def _sample_inner_dims(branch: str) -> tuple[int, int]:
    lo, hi = INNER_SIZE_RANGE
    if branch == "wide":
        inner_h = int(_RNG.integers(lo, hi))
        inner_w = int(_RNG.integers(inner_h + 1, min(inner_h + 5, hi + 2)))
        return inner_h, inner_w
    if branch == "tall":
        inner_w = int(_RNG.integers(lo, hi))
        inner_h = int(_RNG.integers(inner_w + 1, min(inner_w + 5, hi + 2)))
        return inner_h, inner_w
    side = int(_RNG.integers(lo, hi + 1))
    return side, side


def _candidate_positions(occupied: np.ndarray, outer_h: int, outer_w: int) -> list[tuple[int, int]]:
    h, w = occupied.shape
    positions: list[tuple[int, int]] = []
    for r0 in range(1, h - outer_h):
        for c0 in range(1, w - outer_w):
            rr0 = max(0, r0 - 1)
            rr1 = min(h, r0 + outer_h + 1)
            cc0 = max(0, c0 - 1)
            cc1 = min(w, c0 + outer_w + 1)
            if not occupied[rr0:rr1, cc0:cc1].any():
                positions.append((r0, c0))
    return positions


def _build_grid(specs: list[dict[str, int]]) -> np.ndarray | None:
    max_outer_h = max(spec["inner_h"] + 2 for spec in specs)
    max_outer_w = max(spec["inner_w"] + 2 for spec in specs)
    area_budget = sum((spec["inner_h"] + 4) * (spec["inner_w"] + 4) for spec in specs)

    grid_h_lo = max(GRID_H_RANGE[0], max_outer_h + 3, int(np.ceil(np.sqrt(area_budget))) - 1)
    grid_w_lo = max(GRID_W_RANGE[0], max_outer_w + 3, int(np.ceil(np.sqrt(area_budget))) - 1)
    if grid_h_lo >= GRID_H_RANGE[1] or grid_w_lo >= GRID_W_RANGE[1]:
        return None

    grid_h = int(_RNG.integers(grid_h_lo, GRID_H_RANGE[1]))
    grid_w = int(_RNG.integers(grid_w_lo, GRID_W_RANGE[1]))
    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    placements: list[dict[str, int]] = []

    for spec in sorted(specs, key=lambda item: (item["inner_h"] + 2) * (item["inner_w"] + 2), reverse=True):
        outer_h = spec["inner_h"] + 2
        outer_w = spec["inner_w"] + 2
        positions = _candidate_positions(occupied, outer_h, outer_w)
        if not positions:
            return None
        r0, c0 = positions[int(_RNG.integers(0, len(positions)))]
        occupied[r0 : r0 + outer_h, c0 : c0 + outer_w] = True
        placed = spec.copy()
        placed["r0"] = r0
        placed["c0"] = c0
        placements.append(placed)

    grid = np.full((grid_h, grid_w), BLACK, dtype=int)
    for spec in placements:
        r0 = spec["r0"]
        c0 = spec["c0"]
        inner_h = spec["inner_h"]
        inner_w = spec["inner_w"]
        outer_h = inner_h + 2
        outer_w = inner_w + 2
        r1 = r0 + outer_h - 1
        c1 = c0 + outer_w - 1
        grid[r0, c0 : c1 + 1] = GREY
        grid[r1, c0 : c1 + 1] = GREY
        grid[r0 : r1 + 1, c0] = GREY
        grid[r0 : r1 + 1, c1] = GREY
        seed_abs_r = r0 + 1 + spec["seed_r"]
        seed_abs_c = c0 + 1 + spec["seed_c"]
        grid[seed_abs_r, seed_abs_c] = spec["seed_color"]

    return grid


def _validate_generated_grid(grid: np.ndarray, expected_frames: int) -> bool:
    try:
        frames = _frame_records(grid)
    except AssertionError:
        return False
    if len(frames) != expected_frames:
        return False
    if { _frame_kind(frame["inner_h"], frame["inner_w"]) for frame in frames } != set(BRANCHES):
        return False
    out = main(grid)
    if np.array_equal(out, grid):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    frame_count = int(_RNG.integers(FRAME_COUNT_RANGE[0], FRAME_COUNT_RANGE[1]))
    branch_types = _balanced_branch_types(frame_count)

    palette_size = int(_RNG.integers(1, min(5, len(PALETTE)) + 1))
    palette_subset = _RNG.choice(PALETTE, size=palette_size, replace=False)

    specs: list[dict[str, int]] = []
    for branch in branch_types:
        inner_h, inner_w = _sample_inner_dims(branch)
        seed_color = int(_RNG.choice(palette_subset))
        seed_r = int(_RNG.integers(0, inner_h))
        seed_c = int(_RNG.integers(0, inner_w))
        specs.append(
            {
                "inner_h": inner_h,
                "inner_w": inner_w,
                "seed_r": seed_r,
                "seed_c": seed_c,
                "seed_color": seed_color,
            }
        )

    grid = _build_grid(specs)
    if grid is None:
        return None
    if not _validate_generated_grid(grid, len(specs)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
