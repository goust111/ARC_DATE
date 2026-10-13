from __future__ import annotations

import numpy as np


Coord = tuple[int, int]
CenterSums = tuple[int, int]
Orbit = tuple[Coord, Coord]


GRID_H_RANGE = (16, 24)
GRID_W_RANGE = (16, 24)
TOTAL_COLORS_RANGE = (3, 5)  # anchor + 2..4 other colors
ANCHOR_PAIR_RANGE = (2, 4)   # 4..8 anchor pixels
OTHER_POINT_RANGE = (3, 5)
MAX_GENERATION_ATTEMPTS = 500
INNER_RETRIES = 60


def _color_coords(grid: np.ndarray, color: int) -> list[Coord]:
    return [tuple(map(int, rc)) for rc in np.argwhere(grid == int(color))]


def _bbox_center_sums(coords: list[Coord]) -> CenterSums:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return int(min(rows) + max(rows)), int(min(cols) + max(cols))


def _central_symmetry_center(coords: list[Coord]) -> CenterSums | None:
    if not coords:
        return None
    sr, sc = _bbox_center_sums(coords)
    coord_set = set(coords)
    for r, c in coords:
        if (sr - r, sc - c) not in coord_set:
            return None
    return sr, sc


def _rotate180(cell: Coord, center_sums: CenterSums) -> Coord:
    r, c = cell
    sr, sc = center_sums
    return int(sr - r), int(sc - c)


def _in_bounds(shape: tuple[int, int], cell: Coord) -> bool:
    r, c = cell
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _find_anchor(grid: np.ndarray) -> tuple[int, CenterSums]:
    colors = [int(c) for c in np.unique(grid) if int(c) != 0]
    candidates: list[tuple[int, CenterSums]] = []
    for color in colors:
        coords = _color_coords(grid, color)
        if len(coords) < 4:
            continue
        center = _central_symmetry_center(coords)
        if center is None:
            continue
        # The borrowed center is never explicitly drawn.
        if any(2 * r == center[0] and 2 * c == center[1] for r, c in coords):
            continue
        candidates.append((color, center))
    assert len(candidates) == 1, "expected exactly one self-symmetric anchor color"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Borrow the hidden 180-degree symmetry center from the one self-symmetric color and complete every other color around it.

    Input structure:
    - A black background contains several colors of isolated foreground pixels.
    - Exactly one color forms a set of at least four pixels that is already 180-degree
      point-symmetric about an unmarked center.
    - Every other color has at least three pixels and is not itself 180-degree
      point-symmetric about any center.

    Rule:
    - Identify the unique self-symmetric anchor color and infer its hidden center.
    - Leave all anchor-color pixels unchanged.
    - For every pixel of every other color, paint its 180-degree rotated counterpart
      around the borrowed center in the same color.

    Key insight:
    - The center is never drawn directly; it is recoverable only from the one color
      whose pixels already complete their own half-turn symmetry.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    anchor_color, center_sums = _find_anchor(g)
    h, w = g.shape
    out = g.copy()

    for color in [int(c) for c in np.unique(g) if int(c) not in (0, anchor_color)]:
        for cell in _color_coords(g, color):
            mate = _rotate180(cell, center_sums)
            assert _in_bounds((h, w), mate), "rotated partner must stay inside the grid"
            assert int(g[mate]) == 0, "non-anchor pixels must point to empty mirror locations"
            out[mate] = color

    return out


def _choose_center_sums(h: int, w: int, rng: np.random.Generator) -> CenterSums:
    row_choices = [2 * r for r in range(2, h - 2)] + [2 * r + 1 for r in range(1, h - 3)]
    col_choices = [2 * c for c in range(2, w - 2)] + [2 * c + 1 for c in range(1, w - 3)]
    row_choices = [int(x) for x in row_choices]
    col_choices = [int(x) for x in col_choices]
    rng.shuffle(row_choices)
    rng.shuffle(col_choices)
    for sr in row_choices:
        for sc in col_choices:
            if (sr, sc) != (h - 1, w - 1):
                return int(sr), int(sc)
    raise RuntimeError("failed to choose a non-central borrowed center")


def _all_canonical_orbits(shape: tuple[int, int], center_sums: CenterSums) -> list[Orbit]:
    h, w = shape
    sr, sc = center_sums
    orbits: list[Orbit] = []
    for r in range(h):
        for c in range(w):
            mate = (sr - r, sc - c)
            if not _in_bounds(shape, mate):
                continue
            cell = (r, c)
            if cell >= mate:
                continue
            if max(abs(cell[0] - mate[0]), abs(cell[1] - mate[1])) <= 1:
                continue
            orbits.append((cell, mate))
    return orbits


def _neighbor_cells(cell: Coord, shape: tuple[int, int]) -> list[Coord]:
    h, w = shape
    r, c = cell
    out: list[Coord] = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w:
                out.append((rr, cc))
    return out


def _orbit_is_free(orbit: Orbit, blocked: set[Coord]) -> bool:
    return orbit[0] not in blocked and orbit[1] not in blocked


def _block_orbit(blocked: set[Coord], orbit: Orbit, shape: tuple[int, int]) -> None:
    for cell in orbit:
        blocked.update(_neighbor_cells(cell, shape))


def _valid_anchor_pairs(anchor_pairs: list[Orbit], center_sums: CenterSums) -> bool:
    coords = [cell for pair in anchor_pairs for cell in pair]
    if len(coords) < 4:
        return False
    if _central_symmetry_center(coords) != center_sums:
        return False
    if len({r for r, _ in coords}) < 3 or len({c for _, c in coords}) < 3:
        return False
    offsets = {(abs(2 * r - center_sums[0]), abs(2 * c - center_sums[1])) for r, c in coords}
    return len(offsets) >= 3


def _sample_anchor_pairs(
    shape: tuple[int, int],
    center_sums: CenterSums,
    orbits: list[Orbit],
    rng: np.random.Generator,
) -> tuple[list[Orbit], set[Coord]] | None:
    target_pairs = int(rng.integers(ANCHOR_PAIR_RANGE[0], ANCHOR_PAIR_RANGE[1] + 1))
    for _ in range(INNER_RETRIES):
        blocked: set[Coord] = set()
        anchor_pairs: list[Orbit] = []
        shuffled = list(orbits)
        rng.shuffle(shuffled)
        for orbit in shuffled:
            if not _orbit_is_free(orbit, blocked):
                continue
            anchor_pairs.append(orbit)
            _block_orbit(blocked, orbit, shape)
            if len(anchor_pairs) == target_pairs:
                break
        if len(anchor_pairs) == target_pairs and _valid_anchor_pairs(anchor_pairs, center_sums):
            return anchor_pairs, blocked
    return None


def _sample_other_points(
    shape: tuple[int, int],
    center_sums: CenterSums,
    orbits: list[Orbit],
    blocked: set[Coord],
    count: int,
    rng: np.random.Generator,
) -> tuple[list[Coord], list[Orbit], set[Coord]] | None:
    for _ in range(INNER_RETRIES):
        trial_blocked = set(blocked)
        chosen_points: list[Coord] = []
        chosen_orbits: list[Orbit] = []
        shuffled = list(orbits)
        rng.shuffle(shuffled)
        for orbit in shuffled:
            if not _orbit_is_free(orbit, trial_blocked):
                continue
            # Always expose the lexicographically smaller endpoint so generator-side
            # completed outputs do not collapse multiple distinct visible halves.
            chosen_points.append(orbit[0])
            chosen_orbits.append(orbit)
            _block_orbit(trial_blocked, orbit, shape)
            if len(chosen_points) == count:
                break
        if len(chosen_points) != count:
            continue
        if _central_symmetry_center(chosen_points) is not None:
            continue
        if _bbox_center_sums(chosen_points) == center_sums:
            continue
        if len({r for r, _ in chosen_points}) < 2 or len({c for _, c in chosen_points}) < 2:
            continue
        return chosen_points, chosen_orbits, trial_blocked
    return None


def _expected_output(grid: np.ndarray, anchor_color: int, center_sums: CenterSums) -> np.ndarray:
    out = grid.copy()
    for color in [int(c) for c in np.unique(grid) if int(c) not in (0, anchor_color)]:
        for cell in _color_coords(grid, color):
            out[_rotate180(cell, center_sums)] = color
    return out


def _validate_generated_case(grid: np.ndarray, anchor_color: int, center_sums: CenterSums) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if tuple(_find_anchor(grid)) != (anchor_color, center_sums):
        return False

    fg_coords = [tuple(map(int, rc)) for rc in np.argwhere(grid != 0)]
    if _bbox_center_sums(fg_coords) == center_sums:
        return False

    out = _expected_output(grid, anchor_color, center_sums)
    if np.array_equal(grid, out):
        return False

    for color in [int(c) for c in np.unique(grid) if int(c) != 0]:
        coords = _color_coords(grid, color)
        if color == anchor_color:
            if len(coords) < 4 or _central_symmetry_center(coords) != center_sums:
                return False
            if _color_coords(out, color) != coords:
                return False
            continue
        if len(coords) < 3:
            return False
        if _central_symmetry_center(coords) is not None:
            return False
        if _bbox_center_sums(coords) == center_sums:
            return False
        for cell in coords:
            mate = _rotate180(cell, center_sums)
            if not _in_bounds(grid.shape, mate):
                return False
            if int(grid[mate]) != 0:
                return False
            if int(out[mate]) != color:
                return False
        if len(_color_coords(out, color)) != 2 * len(coords):
            return False

    try:
        pred = main(grid)
    except Exception:
        return False
    return np.array_equal(pred, out)


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(GRID_H_RANGE[0], GRID_H_RANGE[1] + 1))
    w = int(rng.integers(GRID_W_RANGE[0], GRID_W_RANGE[1] + 1))
    center_sums = _choose_center_sums(h, w, rng)
    orbits = _all_canonical_orbits((h, w), center_sums)
    if len(orbits) < 24:
        return None

    anchor_sample = _sample_anchor_pairs((h, w), center_sums, orbits, rng)
    if anchor_sample is None:
        return None
    anchor_pairs, blocked = anchor_sample

    total_colors = int(rng.integers(TOTAL_COLORS_RANGE[0], TOTAL_COLORS_RANGE[1] + 1))
    colors = rng.choice(np.arange(1, 10), size=total_colors, replace=False).astype(int).tolist()
    anchor_color = int(colors[0])

    placements: dict[int, list[Coord]] = {
        anchor_color: [cell for pair in anchor_pairs for cell in pair]
    }

    for color in colors[1:]:
        count = int(rng.integers(OTHER_POINT_RANGE[0], OTHER_POINT_RANGE[1] + 1))
        sample = _sample_other_points((h, w), center_sums, orbits, blocked, count, rng)
        if sample is None:
            return None
        coords, _chosen_orbits, blocked = sample
        placements[int(color)] = coords

    grid = np.zeros((h, w), dtype=int)
    for color, coords in placements.items():
        for r, c in coords:
            grid[r, c] = int(color)

    if not _validate_generated_case(grid, anchor_color, center_sums):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid hidden-center symmetry-completion input")
