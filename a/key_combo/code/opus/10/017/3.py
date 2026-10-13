from __future__ import annotations

import random

import numpy as np


BLACK = 0
RED = 2
GREEN = 3

PLUS_OFFSETS = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

GEN_HEIGHT_RANGE = (15, 18)
GEN_WIDTH_RANGE = (16, 20)
GEN_CLEAR_DISTANCE_RANGE = (3, 5)
GEN_BLOCK_DISTANCE_RANGE = (3, 5)
GEN_ATTEMPTS = 500
GEN_INCLUDE_CLIPPED_PROB = 0.5


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(comp)

    return components


def _plus_cells(center: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = center
    return {(r + dr, c + dc) for dr, dc in PLUS_OFFSETS}


def _extract_pluses(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert np.all((g == BLACK) | (g == GREEN)), "input should contain only black and green"

    pluses: list[dict[str, object]] = []
    for comp in _components(g == GREEN):
        coords = set(comp)
        assert len(coords) == 5, "every foreground object must be a 5-pixel plus"

        centers = []
        for r, c in coords:
            expected = _plus_cells((r, c))
            if coords == expected:
                centers.append((r, c))
        assert len(centers) == 1, "every object must be an isolated plus"

        pluses.append({"center": centers[0], "coords": tuple(sorted(coords))})

    assert 4 <= len(pluses) <= 6, "new puzzle instances use 4-6 pluses"
    return pluses


def _global_bottom(pluses: list[dict[str, object]]) -> int:
    return max(r for plus in pluses for r, _ in plus["coords"])  # type: ignore[index]


def _shear_plus(plus: dict[str, object], bottom: int, width: int) -> dict[str, object] | None:
    center_r, center_c = plus["center"]  # type: ignore[misc]
    shifted_center_c = center_c - (bottom - center_r)
    if shifted_center_c < 0 or shifted_center_c >= width:
        return None

    shifted_cells = set()
    for r, c in plus["coords"]:  # type: ignore[misc]
        new_c = c - (bottom - r)
        if 0 <= new_c < width:
            shifted_cells.add((r, new_c))

    return {"center": (center_r, shifted_center_c), "coords": tuple(sorted(shifted_cells))}


def _gap_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]] | None:
    dr = b[0] - a[0]
    dc = b[1] - a[1]
    if dr == 0 or abs(dr) != abs(dc):
        return None
    sr = 1 if dr > 0 else -1
    sc = 1 if dc > 0 else -1
    return [(a[0] + k * sr, a[1] + k * sc) for k in range(1, abs(dr))]


def _input_diagonal_pair_count(centers: list[tuple[int, int]]) -> int:
    count = 0
    for i, (r1, c1) in enumerate(centers):
        for r2, c2 in centers[i + 1 :]:
            if abs(r2 - r1) == abs(c2 - c1):
                count += 1
    return count


def _analyze_grid(grid: np.ndarray) -> dict[str, object]:
    pluses = _extract_pluses(grid)
    bottom = _global_bottom(pluses)
    h, w = grid.shape

    survivors: list[dict[str, object]] = []
    erased_count = 0
    base = np.full((h, w), BLACK, dtype=int)
    total_survivor_pixels = 0
    for plus in pluses:
        shifted = _shear_plus(plus, bottom, w)
        if shifted is None:
            erased_count += 1
            continue
        survivors.append(shifted)
        total_survivor_pixels += len(shifted["coords"])  # type: ignore[arg-type]
        for r, c in shifted["coords"]:  # type: ignore[misc]
            base[r, c] = GREEN

    pairs: list[dict[str, object]] = []
    clear_pair_count = 0
    ascending_clear_pairs = 0
    descending_blocked_pairs = 0
    for i, first in enumerate(survivors):
        center_a = first["center"]  # type: ignore[assignment]
        for second in survivors[i + 1 :]:
            center_b = second["center"]  # type: ignore[assignment]
            gap = _gap_cells(center_a, center_b)
            if gap is None:
                continue
            clear = all(base[r, c] == BLACK for r, c in gap)
            slope = 1 if (center_b[1] - center_a[1]) == (center_b[0] - center_a[0]) else -1
            if clear:
                clear_pair_count += 1
                if slope == -1:
                    ascending_clear_pairs += 1
            elif slope == 1:
                descending_blocked_pairs += 1
            pairs.append(
                {
                    "a": center_a,
                    "b": center_b,
                    "gap": gap,
                    "clear": clear,
                    "slope": slope,
                }
            )

    return {
        "input_centers": [plus["center"] for plus in pluses],
        "input_diagonal_pairs": _input_diagonal_pair_count([plus["center"] for plus in pluses]),
        "bottom": bottom,
        "survivors": survivors,
        "survivor_count": len(survivors),
        "erased_count": erased_count,
        "base_grid": base,
        "pairs": pairs,
        "clear_pair_count": clear_pair_count,
        "ascending_clear_pairs": ascending_clear_pairs,
        "descending_blocked_pairs": descending_blocked_pairs,
        "green_overlap_free": total_survivor_pixels == int(np.count_nonzero(base == GREEN)),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Shear all green pluses left from the global bottom anchor, then draw red connectors only where that shear creates a clear 45-degree diagonal between surviving centers.

    Input structure:
    - The grid has a black background and 4-6 separate green 5-pixel plus objects.
    - No pair of input plus centers is already on a 45-degree diagonal.
    - At least one plus reaches the global bottom anchor row through its lower arm.

    Rule:
    - Find the bottommost row among all plus pixels and use it as the shared shear anchor.
    - Move every plus pixel at (r, c) to (r, c - (bottom - r)); clip pixels that fall past the left edge.
    - If a plus loses its center pixel under this shear, erase that whole plus instead of keeping any clipped remainder.
    - Among the surviving sheared centers, connect each pair whose centers lie on a 45-degree diagonal and whose interior diagonal cells are all black by painting those interior cells red.

    Key insight:
    - The relevant diagonal alignment does not exist in the input; it emerges only after the bottom-anchored shear, so the solver must shear first and test center-to-center visibility second.
    """
    g = np.asarray(grid, dtype=int)
    info = _analyze_grid(g)
    out = info["base_grid"].copy()  # type: ignore[assignment]
    for pair in info["pairs"]:  # type: ignore[assignment]
        if not pair["clear"]:
            continue
        for r, c in pair["gap"]:
            out[r, c] = RED
    return out


def _center_in_bounds(center: tuple[int, int], h: int, w: int) -> bool:
    r, c = center
    return 1 <= r < h - 1 and 1 <= c < w - 1


def _centers_are_separate(centers: list[tuple[int, int]]) -> bool:
    for i, (r1, c1) in enumerate(centers):
        for r2, c2 in centers[i + 1 :]:
            if max(abs(r2 - r1), abs(c2 - c1)) < 3:
                return False
    return True


def _valid_input_centers(centers: list[tuple[int, int]], h: int, w: int) -> bool:
    if len(set(centers)) != len(centers):
        return False
    if not all(_center_in_bounds(center, h, w) for center in centers):
        return False
    if not _centers_are_separate(centers):
        return False
    return _input_diagonal_pair_count(centers) == 0


def _render_input_grid(h: int, w: int, centers: list[tuple[int, int]]) -> np.ndarray:
    grid = np.full((h, w), BLACK, dtype=int)
    for center in centers:
        for r, c in _plus_cells(center):
            grid[r, c] = GREEN
    return grid


def _sample_clear_pair(h: int, w: int) -> list[tuple[int, int]] | None:
    max_d = min(GEN_CLEAR_DISTANCE_RANGE[1], h - 3, (w - 4) // 2)
    if max_d < GEN_CLEAR_DISTANCE_RANGE[0]:
        return None
    d = random.randint(GEN_CLEAR_DISTANCE_RANGE[0], max_d)
    max_sheared_lower_c = w - 3 - 2 * d
    if max_sheared_lower_c < 1:
        return None

    low = max(1, max_sheared_lower_c - 3)
    sheared_lower_c = random.randint(low, max_sheared_lower_c)
    lower = (h - 2, sheared_lower_c + 1)
    upper = (h - 2 - d, sheared_lower_c + 2 * d + 1)
    return [upper, lower]


def _sample_blocked_pair(h: int, w: int) -> list[tuple[int, int]] | None:
    lower_row_choices = [row for row in (h - 2, h - 3, h - 4) if row >= 4]
    if not lower_row_choices:
        return None
    lower_row = random.choice(lower_row_choices)
    max_d = min(GEN_BLOCK_DISTANCE_RANGE[1], lower_row - 1)
    if max_d < GEN_BLOCK_DISTANCE_RANGE[0]:
        return None
    d = random.randint(GEN_BLOCK_DISTANCE_RANGE[0], max_d)
    shift_upper = h - 1 - (lower_row - d)
    min_col = shift_upper
    if min_col > w - 2:
        return None
    col = random.randint(min_col, w - 2)
    return [(lower_row - d, col), (lower_row, col)]


def _sample_clipped_center(h: int, w: int) -> tuple[int, int] | None:
    row_choices = [r for r in range(1, h - 3) if h - 2 - r >= 1]
    if not row_choices:
        return None
    row = random.choice(row_choices)
    shift = h - 1 - row
    max_col = min(w - 2, shift - 1)
    if max_col < 1:
        return None
    return row, random.randint(1, max_col)


def _validate_generated_grid(grid: np.ndarray, require_clipped: bool) -> bool:
    info = _analyze_grid(grid)
    if info["input_diagonal_pairs"] != 0:
        return False
    if not info["green_overlap_free"]:
        return False
    if info["clear_pair_count"] != 1:
        return False
    if info["ascending_clear_pairs"] != 1:
        return False
    if info["descending_blocked_pairs"] < 1:
        return False
    if require_clipped and info["erased_count"] < 1:
        return False
    out = main(grid)
    if out.shape != grid.shape or max(out.shape) > 30:
        return False
    if np.array_equal(out, grid):
        return False
    if int(np.count_nonzero(out == RED)) < 2:
        return False
    return True


def _attempt_generate(require_clipped: bool) -> np.ndarray | None:
    h = random.randint(*GEN_HEIGHT_RANGE)
    w = random.randint(*GEN_WIDTH_RANGE)

    clear_pair = _sample_clear_pair(h, w)
    blocked_pair = _sample_blocked_pair(h, w)
    if clear_pair is None or blocked_pair is None:
        return None

    centers = clear_pair[:]
    for _ in range(120):
        trial_centers = centers + blocked_pair
        if _valid_input_centers(trial_centers, h, w):
            centers = trial_centers
            break
        blocked_pair = _sample_blocked_pair(h, w)
        if blocked_pair is None:
            return None
    else:
        return None

    if require_clipped:
        for _ in range(120):
            clipped = _sample_clipped_center(h, w)
            if clipped is None:
                return None
            trial_centers = centers + [clipped]
            if _valid_input_centers(trial_centers, h, w):
                centers = trial_centers
                break
        else:
            return None

    grid = _render_input_grid(h, w, centers)
    if not _validate_generated_grid(grid, require_clipped=require_clipped):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        require_clipped = random.random() < GEN_INCLUDE_CLIPPED_PROB
        try:
            grid = _attempt_generate(require_clipped=require_clipped)
        except AssertionError:
            grid = None
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid shear-then-diagonal instance")
