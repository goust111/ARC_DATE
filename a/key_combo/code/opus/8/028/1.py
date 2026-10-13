from __future__ import annotations

from collections import Counter, defaultdict
from importlib.machinery import SourceFileLoader
import random

import numpy as np


_ARC_UTILS = SourceFileLoader(
    "_arc_utils_028_1",
    "/home/lixinyu/snap/new/solvers/arc_utils.py",
).load_module()

Color = _ARC_UTILS.Color
components_mask = _ARC_UTILS.components_mask
neighbors4 = _ARC_UTILS.neighbors4


BACKGROUND = Color.BLACK
WALL = Color.GREY
MARKER_COLORS = (Color.BLUE, Color.RED, Color.GREEN)
GRID_SIZES = (15, 17, 19)
MAX_ATTEMPTS = 400

ARM_CONFIGS = (
    ("N",),
    ("E",),
    ("S",),
    ("W",),
    ("N", "S"),
    ("E", "W"),
    ("N", "E"),
    ("E", "S"),
    ("S", "W"),
    ("W", "N"),
    ("N", "E", "S"),
    ("E", "S", "W"),
    ("S", "W", "N"),
    ("W", "N", "E"),
)

SIDE_CONFIGS = (
    ("T",),
    ("R",),
    ("B",),
    ("L",),
    ("T", "R"),
    ("R", "B"),
    ("B", "L"),
    ("L", "T"),
    ("T", "B"),
    ("L", "R"),
    ("T", "R", "B"),
    ("R", "B", "L"),
    ("B", "L", "T"),
    ("L", "T", "R"),
)


def _complete_wall_mask(mask: np.ndarray) -> np.ndarray:
    closed = mask.astype(bool).copy()
    for k in range(1, 4):
        closed |= np.rot90(mask, k)
    return closed


def _full_wall_mask(size: int, radius: int) -> np.ndarray:
    mid = size // 2
    wall = np.zeros((size, size), dtype=bool)
    wall[mid, :] = True
    wall[:, mid] = True

    lo = mid - radius
    hi = mid + radius
    wall[lo, lo : hi + 1] = True
    wall[hi, lo : hi + 1] = True
    wall[lo : hi + 1, lo] = True
    wall[lo : hi + 1, hi] = True
    return wall


def _add_arm(mask: np.ndarray, arm: str) -> None:
    size = mask.shape[0]
    mid = size // 2
    if arm == "N":
        mask[: mid + 1, mid] = True
    elif arm == "E":
        mask[mid, mid:] = True
    elif arm == "S":
        mask[mid:, mid] = True
    elif arm == "W":
        mask[mid, : mid + 1] = True
    else:
        raise ValueError(f"unknown arm {arm}")


def _add_side(mask: np.ndarray, radius: int, side: str) -> None:
    mid = mask.shape[0] // 2
    lo = mid - radius
    hi = mid + radius
    if side == "T":
        mask[lo, lo : hi + 1] = True
    elif side == "R":
        mask[lo : hi + 1, hi] = True
    elif side == "B":
        mask[hi, lo : hi + 1] = True
    elif side == "L":
        mask[lo : hi + 1, lo] = True
    else:
        raise ValueError(f"unknown side {side}")


def _partial_wall_mask(size: int, radius: int) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    for arm in random.choice(ARM_CONFIGS):
        _add_arm(mask, arm)
    for side in random.choice(SIDE_CONFIGS):
        _add_side(mask, radius, side)
    if np.array_equal(mask, _complete_wall_mask(mask)):
        raise AssertionError("partial wall must actually be incomplete")
    return mask


def _unique_majority(counter: Counter[int]) -> int:
    top = max(counter.values())
    winners = [color for color, count in counter.items() if count == top]
    assert len(winners) == 1, "expected a unique majority marker color"
    return int(winners[0])


def _adjacent_marker_counts(grid: np.ndarray, coords: np.ndarray, marker_colors: set[int]) -> Counter[int]:
    height, width = grid.shape
    seen: dict[tuple[int, int], int] = {}
    for r0, c0 in coords:
        for r, c in neighbors4(int(r0), int(c0)):
            if 0 <= r < height and 0 <= c < width and int(grid[r, c]) in marker_colors:
                seen[(int(r), int(c))] = int(grid[r, c])
    assert seen, "every chamber must touch at least one marker"
    return Counter(seen.values())


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the grey walls by quarter-turn symmetry, then flood-fill each chamber with its majority adjacent marker color.

    Input structure:
    - A square odd-sized grid has black background, grey wall pixels, and single-cell blue/red/green markers.
    - The grey walls are an incomplete quarter-turn-symmetric divider centered on the grid center.
    - Markers never lie on walls and each marker is orthogonally adjacent to at least one background chamber cell.

    Rule:
    - Rotate every grey wall pixel by 90, 180, and 270 degrees about the grid center and add the union of those positions as new grey wall pixels.
    - Treat the completed grey walls as barriers and find all 4-connected black background chambers.
    - For each chamber, collect the distinct orthogonally adjacent marker pixels and take the unique majority marker color.
    - Fill that chamber with its majority color while preserving every grey wall and every marker pixel.

    Key insight:
    - The missing grey segments are not decorative: completing the wall first changes which black cells belong to the same chamber, which in turn changes the marker votes that each filled region receives.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] == g.shape[1], "expected a square grid"
    assert g.shape[0] % 2 == 1, "expected an odd side length"

    wall_mask = g == WALL
    completed_wall = _complete_wall_mask(wall_mask)
    marker_mask = (g != BACKGROUND) & (g != WALL)
    assert not np.any(completed_wall & marker_mask & ~wall_mask), "markers cannot occupy completed wall cells"

    completed = g.copy()
    completed[completed_wall] = WALL

    marker_colors = set(map(int, np.unique(completed[marker_mask])))
    assert marker_colors, "expected at least one marker color"

    out = completed.copy()
    chambers = components_mask(completed == BACKGROUND, connectivity=4)
    assert chambers, "expected at least one chamber"
    for coords in chambers:
        counts = _adjacent_marker_counts(completed, coords, marker_colors)
        fill = _unique_majority(counts)
        out[coords[:, 0], coords[:, 1]] = fill
    return out


def _chamber_key(coords: np.ndarray, mid: int, radius: int) -> tuple[int, int, int]:
    r, c = map(int, coords[0])
    inside = 1 if max(abs(r - mid), abs(c - mid)) < radius else 0
    row_sign = -1 if r < mid else 1
    col_sign = -1 if c < mid else 1
    return inside, row_sign, col_sign


def _full_chambers(size: int, radius: int) -> dict[tuple[int, int, int], np.ndarray]:
    grid = np.full((size, size), BACKGROUND, dtype=int)
    grid[_full_wall_mask(size, radius)] = WALL
    chambers = components_mask(grid == BACKGROUND, connectivity=4)
    by_key = {_chamber_key(coords, size // 2, radius): coords for coords in chambers}
    assert len(by_key) == 8, "cross-plus-ring should create eight chambers"
    return by_key


def _good_marker_cells(coords: np.ndarray) -> list[tuple[int, int]]:
    cells = {tuple(map(int, rc)) for rc in coords}
    candidates = []
    for r, c in cells:
        degree = sum((nr, nc) in cells for nr, nc in neighbors4(r, c))
        if degree >= 2:
            candidates.append((r, c))
    return candidates or list(cells)


def _pick_cells(coords: np.ndarray, count: int) -> list[tuple[int, int]]:
    candidates = _good_marker_cells(coords)
    random.shuffle(candidates)
    chosen: list[tuple[int, int]] = []
    for r, c in candidates:
        if all(abs(r - rr) + abs(c - cc) >= 2 for rr, cc in chosen):
            chosen.append((r, c))
        if len(chosen) == count:
            return chosen
    for cell in candidates:
        if cell not in chosen:
            chosen.append(cell)
        if len(chosen) == count:
            return chosen
    raise AssertionError("not enough cells for markers")


def _random_targets(keys: list[tuple[int, int, int]]) -> dict[tuple[int, int, int], int]:
    return {key: random.choice(MARKER_COLORS) for key in keys}


def _place_markers(
    size: int,
    full_wall: np.ndarray,
    chambers: dict[tuple[int, int, int], np.ndarray],
    targets: dict[tuple[int, int, int], int],
) -> tuple[np.ndarray, dict[tuple[int, int, int], Counter[int]]]:
    grid = np.full((size, size), BACKGROUND, dtype=int)
    grid[full_wall] = WALL

    keys = list(chambers)
    random.shuffle(keys)
    mixed_keys = set(keys[: random.randint(2, 5)])
    counts_by_key: dict[tuple[int, int, int], Counter[int]] = {}

    for key in keys:
        coords = chambers[key]
        majority = targets[key]
        if key in mixed_keys:
            minority = random.choice([c for c in MARKER_COLORS if c != majority])
            colors = [majority, majority, minority]
        else:
            colors = [majority, majority]
        cells = _pick_cells(coords, len(colors))
        for (r, c), color in zip(cells, colors):
            grid[r, c] = color
        counts_by_key[key] = Counter(colors)

    return grid, counts_by_key


def _majorities_from_grid(grid: np.ndarray, size: int, radius: int) -> dict[tuple[int, int, int], int]:
    mid = size // 2
    chambers = components_mask(grid == BACKGROUND, connectivity=4)
    result: dict[tuple[int, int, int], int] = {}
    marker_colors = set(MARKER_COLORS)
    for coords in chambers:
        key = _chamber_key(coords, mid, radius)
        counts = _adjacent_marker_counts(grid, coords, marker_colors)
        result[key] = _unique_majority(counts)
    return result


def _split_condition(
    inp: np.ndarray,
    full_grid: np.ndarray,
    radius: int,
    targets: dict[tuple[int, int, int], int],
) -> bool:
    final_chambers = components_mask(full_grid == BACKGROUND, connectivity=4)
    partial_chambers = components_mask(inp == BACKGROUND, connectivity=4)
    partial_id: dict[tuple[int, int], int] = {}
    for idx, coords in enumerate(partial_chambers):
        for r, c in coords:
            partial_id[(int(r), int(c))] = idx

    grouped_keys: defaultdict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for coords in final_chambers:
        key = _chamber_key(coords, inp.shape[0] // 2, radius)
        pid = partial_id[tuple(map(int, coords[0]))]
        grouped_keys[pid].append(key)

    for keys in grouped_keys.values():
        if len(keys) < 2:
            continue
        colors = {targets[key] for key in keys}
        if len(colors) >= 2:
            return True
    return False


def _verify_case(
    inp: np.ndarray,
    radius: int,
    full_wall: np.ndarray,
    targets: dict[tuple[int, int, int], int],
) -> None:
    assert np.array_equal(_complete_wall_mask(inp == WALL), full_wall)

    full_grid = inp.copy()
    full_grid[full_wall] = WALL

    final_chambers = components_mask(full_grid == BACKGROUND, connectivity=4)
    assert len(final_chambers) == 8
    assert len(components_mask(inp == BACKGROUND, connectivity=4)) < len(final_chambers)

    observed = _majorities_from_grid(full_grid, inp.shape[0], radius)
    assert observed == targets
    assert _split_condition(inp, full_grid, radius, targets)

    out = main(inp)
    assert np.all(out[full_wall] == WALL)
    for coords in final_chambers:
        key = _chamber_key(coords, inp.shape[0] // 2, radius)
        assert np.all(out[coords[:, 0], coords[:, 1]] == targets[key])


def _attempt() -> np.ndarray | None:
    size = random.choice(GRID_SIZES)
    mid = size // 2
    radius = random.randint(4, mid - 2)

    full_wall = _full_wall_mask(size, radius)
    partial_wall = _partial_wall_mask(size, radius)
    if not np.array_equal(_complete_wall_mask(partial_wall), full_wall):
        return None

    chambers = _full_chambers(size, radius)
    targets = _random_targets(list(chambers))

    grid, _counts = _place_markers(size, full_wall, chambers, targets)
    if len(components_mask(grid == BACKGROUND, connectivity=4)) != 8:
        return None

    inp = grid.copy()
    inp[full_wall] = BACKGROUND
    inp[partial_wall] = WALL

    try:
        _verify_case(inp, radius, full_wall, targets)
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid symmetric-wall chamber instance")
