from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np

try:
    from arc_utils import crop_bbox, get_bg, tile_fill
except ModuleNotFoundError:
    toolbox_dir = Path("/home/lixinyu/snap/new/solvers")
    if str(toolbox_dir) not in sys.path:
        sys.path.append(str(toolbox_dir))
    from arc_utils import crop_bbox, get_bg, tile_fill


BACKGROUND = 0
RECT_COLOR = 1
NOISE_COLORS = (2, 3, 4, 6)
MAX_GENERATION_ATTEMPTS = 500
ORTHO_NEIGHBORS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _prefix(mask: np.ndarray) -> np.ndarray:
    return mask.astype(np.int16).cumsum(axis=0).cumsum(axis=1)


def _rect_sum(prefix: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> int:
    total = int(prefix[r1, c1])
    if r0 > 0:
        total -= int(prefix[r0 - 1, c1])
    if c0 > 0:
        total -= int(prefix[r1, c0 - 1])
    if r0 > 0 and c0 > 0:
        total += int(prefix[r0 - 1, c0 - 1])
    return total


def _largest_rectangle(grid: np.ndarray, *, max_impurities: int) -> dict:
    g = np.asarray(grid, dtype=int)
    height, width = map(int, g.shape)
    bg = int(get_bg(g))
    colors = [int(color) for color in np.unique(g) if int(color) != bg]
    bg_prefix = _prefix(g == bg)
    color_prefix = {color: _prefix(g == color) for color in colors}

    dims = [(h, w) for h in range(2, height + 1) for w in range(2, width + 1)]
    dims.sort(key=lambda hw: hw[0] * hw[1], reverse=True)

    for area, group in itertools.groupby(dims, key=lambda hw: hw[0] * hw[1]):
        hits = []
        for h, w in group:
            border_cells = 2 * h + 2 * w - 4
            for r0 in range(height - h + 1):
                r1 = r0 + h - 1
                for c0 in range(width - w + 1):
                    c1 = c0 + w - 1
                    if _rect_sum(bg_prefix, r0, c0, r1, c1) != 0:
                        continue
                    for color in colors:
                        total_color = _rect_sum(color_prefix[color], r0, c0, r1, c1)
                        if area - total_color > int(max_impurities):
                            continue
                        if h > 2 and w > 2:
                            inner_color = _rect_sum(color_prefix[color], r0 + 1, c0 + 1, r1 - 1, c1 - 1)
                        else:
                            inner_color = 0
                        border_color = total_color - inner_color
                        if border_color != border_cells:
                            continue
                        hits.append(
                            {
                                "color": int(color),
                                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                                "shape": (int(h), int(w)),
                                "area": int(area),
                                "impurities": int(area - total_color),
                            }
                        )
        if hits:
            assert len(hits) == 1, "expected a unique largest rectangle candidate"
            return hits[0]
    raise AssertionError("failed to locate the seed rectangle")


def _strict_largest_rectangle(grid: np.ndarray) -> dict | None:
    try:
        return _largest_rectangle(grid, max_impurities=0)
    except AssertionError:
        return None


def _seed_patch(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    return crop_bbox(np.asarray(grid, dtype=int), bbox)


def _orthogonally_isolated(grid: np.ndarray, r: int, c: int) -> bool:
    height, width = map(int, grid.shape)
    if int(grid[r, c]) != BACKGROUND:
        return False
    for dr, dc in ORTHO_NEIGHBORS:
        rr = int(r + dr)
        cc = int(c + dc)
        if 0 <= rr < height and 0 <= cc < width and int(grid[rr, cc]) != BACKGROUND:
            return False
    return True


def _expanded_bbox_cells(shape: tuple[int, int], bbox: tuple[int, int, int, int], pad: int) -> set[tuple[int, int]]:
    height, width = map(int, shape)
    r0, c0, r1, c1 = map(int, bbox)
    blocked = set()
    for r in range(max(0, r0 - pad), min(height, r1 + pad + 1)):
        for c in range(max(0, c0 - pad), min(width, c1 + pad + 1)):
            blocked.add((int(r), int(c)))
    return blocked


def _random_positions(
    rng: np.random.Generator,
    grid: np.ndarray,
    *,
    count: int,
    blocked: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    candidates = [(int(r), int(c)) for r in range(grid.shape[0]) for c in range(grid.shape[1]) if (int(r), int(c)) not in blocked]
    rng.shuffle(candidates)
    chosen: list[tuple[int, int]] = []
    for r, c in candidates:
        if not _orthogonally_isolated(grid, r, c):
            continue
        grid[r, c] = 9
        chosen.append((int(r), int(c)))
        if len(chosen) == count:
            break
    for r, c in chosen:
        grid[r, c] = BACKGROUND
    return chosen if len(chosen) == count else None


def _semantic_checks(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    candidate = _largest_rectangle(g, max_impurities=2)
    strict = _strict_largest_rectangle(g)

    if candidate["color"] != RECT_COLOR or candidate["shape"] not in {(3, 4), (4, 3)}:
        return False

    seed = _seed_patch(g, candidate["bbox"])
    if np.any(seed == BACKGROUND):
        return False
    if candidate["impurities"] not in (1, 2):
        return False
    if np.any(seed[0, :] != RECT_COLOR) or np.any(seed[-1, :] != RECT_COLOR):
        return False
    if np.any(seed[:, 0] != RECT_COLOR) or np.any(seed[:, -1] != RECT_COLOR):
        return False

    impurity_count = int(np.count_nonzero(seed != RECT_COLOR))
    if impurity_count not in (1, 2):
        return False
    if not set(map(int, seed[seed != RECT_COLOR].tolist())).issubset(set(NOISE_COLORS)):
        return False

    r0, c0, r1, c1 = candidate["bbox"]
    outside_blue = g.copy()
    outside_blue[r0 : r1 + 1, c0 : c1 + 1] = BACKGROUND
    if not 3 <= int(np.count_nonzero(outside_blue == RECT_COLOR)) <= 5:
        return False

    expected = tile_fill(g.shape, seed)
    if not np.array_equal(main(g), expected):
        return False

    pure_seed = np.full(seed.shape, RECT_COLOR, dtype=int)
    if np.array_equal(expected, tile_fill(g.shape, pure_seed)):
        return False

    if strict is not None and strict["bbox"] == candidate["bbox"]:
        return False

    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find the noisy maximal rectangle, keep its embedded impurities, and tile that seed across the whole grid.

    Input structure:
    - The background is black, with many scattered single-pixel noise marks in several non-black colors.
    - One color forms the unique largest axis-aligned rectangle candidate; its border is intact, but 1-2 interior cells are replaced by other noise colors.
    - A few extra pixels of that rectangle color also appear outside the rectangle as unrelated noise.

    Rule:
    - Find the largest rectangle whose border is entirely one color, whose two side lengths are at least 2, and whose bounding box contains no background cells.
    - Allow 1-2 interior impurities inside that box, then crop the full bounding box from the original grid without cleaning those impurities away.
    - Repeat that cropped seed from the top-left corner with its own height and width as row and column periods until the full canvas is filled.

    Key insight:
    - The rectangle is not extracted as a pure monochrome block: the interior off-color pixels are part of the seed and become the repeated global pattern.
    """
    g = np.asarray(grid, dtype=int)
    candidate = _largest_rectangle(g, max_impurities=2)
    seed = _seed_patch(g, candidate["bbox"])
    return tile_fill(g.shape, seed)


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    seed_h, seed_w = ((3, 4), (4, 3))[int(rng.integers(0, 2))]
    row_mult = int(rng.integers(4, min(8, 30 // seed_h + 1)))
    col_mult = int(rng.integers(4, min(8, 30 // seed_w + 1)))
    height = seed_h * row_mult
    width = seed_w * col_mult

    grid = np.full((height, width), BACKGROUND, dtype=int)

    r0 = int(rng.integers(1, height - seed_h))
    c0 = int(rng.integers(1, width - seed_w))
    r1 = r0 + seed_h - 1
    c1 = c0 + seed_w - 1
    bbox = (int(r0), int(c0), int(r1), int(c1))
    grid[r0 : r1 + 1, c0 : c1 + 1] = RECT_COLOR

    interior = [(r0 + dr, c0 + dc) for dr in range(1, seed_h - 1) for dc in range(1, seed_w - 1)]
    impurity_count = int(rng.integers(1, 3))
    if impurity_count > len(interior):
        impurity_count = len(interior)
    if impurity_count == 0:
        return None
    rng.shuffle(interior)
    for r, c in interior[:impurity_count]:
        grid[r, c] = int(rng.choice(NOISE_COLORS))

    blocked = _expanded_bbox_cells(grid.shape, bbox, pad=1)

    blue_noise_count = int(rng.integers(3, 6))
    blue_positions = _random_positions(rng, grid, count=blue_noise_count, blocked=blocked)
    if blue_positions is None:
        return None
    for r, c in blue_positions:
        grid[r, c] = RECT_COLOR

    colored_noise_total = int(rng.integers(15, 26))
    extra_noise_count = colored_noise_total - impurity_count
    extra_positions = _random_positions(rng, grid, count=extra_noise_count, blocked=blocked | set(blue_positions))
    if extra_positions is None:
        return None
    for r, c in extra_positions:
        grid[r, c] = int(rng.choice(NOISE_COLORS))

    return grid if _semantic_checks(grid) else None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
