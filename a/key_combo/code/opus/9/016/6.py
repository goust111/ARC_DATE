import random
from typing import Optional

import numpy as np


class Color:
    BLACK = 0
    BLUE = 1
    RED = 2
    GREEN = 3
    GREY = 5


GEN_ATTEMPTS = 600
TILE_SHAPES = [
    (1, 3),
    (1, 4),
    (2, 2),
    (2, 3),
    (2, 4),
    (3, 1),
    (3, 2),
    (3, 3),
    (3, 4),
    (4, 1),
    (4, 2),
    (4, 3),
    (4, 4),
]


def _find_divider_row(grid: np.ndarray) -> int:
    divider_rows = np.where(np.all(grid == Color.GREY, axis=1))[0]
    assert divider_rows.size == 1, "expected exactly one full grey divider row"
    divider = int(divider_rows[0])
    assert divider > 0 and divider < grid.shape[0] - 1, "divider must split two nonempty panels"
    assert divider == grid.shape[0] - divider - 1, "top and bottom panels must have equal height"
    return divider


def _overlay_panels(grid: np.ndarray) -> np.ndarray:
    divider = _find_divider_row(grid)
    top = grid[:divider]
    bottom = grid[divider + 1 :]
    assert top.shape == bottom.shape

    top_fg = top != Color.BLACK
    bottom_fg = bottom != Color.BLACK

    overlay = np.zeros_like(top)
    overlay[top_fg & bottom_fg] = Color.GREEN
    overlay[top_fg & ~bottom_fg] = Color.BLUE
    overlay[~top_fg & bottom_fg] = Color.RED
    return overlay


def _candidate_tile(sample: np.ndarray, tile_h: int, tile_w: int) -> Optional[np.ndarray]:
    tile = np.full((tile_h, tile_w), -1, dtype=int)
    seen = np.zeros((tile_h, tile_w), dtype=bool)

    coords = np.argwhere(sample != Color.BLACK)
    if coords.size == 0:
        return None

    for r, c in coords:
        rr = int(r) % tile_h
        cc = int(c) % tile_w
        color = int(sample[r, c])
        if seen[rr, cc] and int(tile[rr, cc]) != color:
            return None
        tile[rr, cc] = color
        seen[rr, cc] = True

    if not bool(seen.all()):
        return None
    return tile


def _valid_tile_candidates(sample: np.ndarray) -> list[tuple[int, int, int, np.ndarray]]:
    height, width = sample.shape
    candidates = []
    for tile_h in range(1, height + 1):
        for tile_w in range(1, width + 1):
            tile = _candidate_tile(sample, tile_h, tile_w)
            if tile is None:
                continue
            candidates.append((tile_h * tile_w, tile_h, tile_w, tile))
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    return candidates


def _unique_min_tile(sample: np.ndarray) -> Optional[tuple[int, int, np.ndarray]]:
    candidates = _valid_tile_candidates(sample)
    if not candidates:
        return None
    best_area = candidates[0][0]
    best = [item for item in candidates if item[0] == best_area]
    if len(best) != 1:
        return None
    _, tile_h, tile_w, tile = best[0]
    return tile_h, tile_w, tile


def _repeat_tile(tile: np.ndarray, height: int, width: int) -> np.ndarray:
    row_ids = np.arange(height) % tile.shape[0]
    col_ids = np.arange(width) % tile.shape[1]
    return tile[row_ids[:, None], col_ids[None, :]]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Overlay the two panels, then repair the black gaps by inferring the smallest periodic color tile.

    Input structure:
    - The grid has a black background with one full grey divider row.
    - Above the divider is a blue-on-black panel; below it is an equally sized red-on-black panel.
    - Foreground cells in the two panels partially overlap when aligned cell by cell.

    Rule:
    - Remove the grey divider and align the top and bottom panels.
    - Convert each aligned cell into an overlay sample: both foreground -> green, top only -> blue,
      bottom only -> red, neither -> black gap.
    - Among all rectangular period tiles consistent with the non-black samples, choose the unique
      smallest-area tile whose every residue class is actually sampled by at least one non-black cell.
    - Fill every black gap with the color from that tile at the same row/column residue.

    Key insight:
    - The panel overlay creates a three-color partial sampling of a periodic pattern; the missing
      cells are recovered only by inferring the minimal periodic tile from those sampled residues.
    """
    overlay = _overlay_panels(np.asarray(grid, dtype=int))
    inferred = _unique_min_tile(overlay)
    assert inferred is not None, "expected a unique smallest periodic tile"
    tile_h, tile_w, tile = inferred

    out = overlay.copy()
    gap_coords = np.argwhere(out == Color.BLACK)
    for r, c in gap_coords:
        out[r, c] = int(tile[int(r) % tile_h, int(c) % tile_w])

    assert bool(np.all(out != Color.BLACK)), "repaired output must not contain black gaps"
    return out


def _random_tile(tile_h: int, tile_w: int) -> Optional[np.ndarray]:
    for _ in range(300):
        tile = np.random.choice([Color.BLUE, Color.RED, Color.GREEN], size=(tile_h, tile_w))
        unique_colors, counts = np.unique(tile, return_counts=True)
        if set(map(int, unique_colors)) != {Color.BLUE, Color.RED, Color.GREEN}:
            continue
        if int(counts.max()) > max(2, (tile_h * tile_w * 2 + 2) // 3):
            continue
        inferred = _unique_min_tile(tile)
        if inferred is None or inferred[0:2] != (tile_h, tile_w):
            continue
        return tile
    return None


def _make_observed_mask(height: int, width: int, tile_h: int, tile_w: int) -> Optional[np.ndarray]:
    total = height * width
    mask = np.ones((height, width), dtype=bool)
    groups: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r in range(height):
        for c in range(width):
            groups.setdefault((r % tile_h, c % tile_w), []).append((r, c))

    target_holes = random.randint(max(3, total // 5), min(total - tile_h * tile_w, (total * 2) // 5))
    remaining = {key: len(cells) for key, cells in groups.items()}

    removable = [(r, c) for cells in groups.values() for r, c in cells]
    random.shuffle(removable)

    holes = 0
    for r, c in removable:
        if holes >= target_holes:
            break
        key = (r % tile_h, c % tile_w)
        if remaining[key] <= 1:
            continue
        mask[r, c] = False
        remaining[key] -= 1
        holes += 1

    if holes < 3:
        return None
    if float(mask.mean()) < 0.60:
        return None
    return mask


def _build_input_from_output(full_output: np.ndarray, observed: np.ndarray) -> np.ndarray:
    top = np.zeros_like(full_output)
    bottom = np.zeros_like(full_output)

    top[observed & np.isin(full_output, [Color.BLUE, Color.GREEN])] = Color.BLUE
    bottom[observed & np.isin(full_output, [Color.RED, Color.GREEN])] = Color.RED

    divider = np.full((1, full_output.shape[1]), Color.GREY, dtype=int)
    return np.vstack([top, divider, bottom])


def _semantic_checks(inp: np.ndarray, expected: np.ndarray, tile_shape: tuple[int, int]) -> bool:
    overlay = _overlay_panels(inp)
    if np.count_nonzero(overlay == Color.BLACK) < 3:
        return False
    if float(np.count_nonzero(overlay != Color.BLACK)) / overlay.size < 0.60:
        return False

    inferred = _unique_min_tile(overlay)
    if inferred is None or inferred[0:2] != tile_shape:
        return False

    expected_inferred = _unique_min_tile(expected)
    if expected_inferred is None or expected_inferred[0:2] != tile_shape:
        return False

    if set(map(int, np.unique(expected))) != {Color.BLUE, Color.RED, Color.GREEN}:
        return False
    if len(set(map(int, np.unique(expected[overlay == Color.BLACK])))) < 2:
        return False

    try:
        solved = main(inp)
    except AssertionError:
        return False
    return bool(np.array_equal(solved, expected))


def _generate_case() -> Optional[np.ndarray]:
    tile_h, tile_w = random.choice(TILE_SHAPES)
    tile = _random_tile(tile_h, tile_w)
    if tile is None:
        return None

    min_height = max(7, tile_h * 2)
    min_width = max(8, tile_w * 2)
    height = random.randint(min_height, min(10, 30))
    width = random.randint(min_width, min(15, 30))

    full_output = _repeat_tile(tile, height, width)
    observed = _make_observed_mask(height, width, tile_h, tile_w)
    if observed is None:
        return None

    if len(set(map(int, np.unique(full_output[~observed])))) < 2:
        return None

    inp = _build_input_from_output(full_output, observed)
    if not _semantic_checks(inp, full_output, (tile_h, tile_w)):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
