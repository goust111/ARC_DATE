from __future__ import annotations

import numpy as np


BG = 0
BLUE = 1
RED = 2
YELLOW = 4

NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    height, width = shape
    return 0 <= r < height and 0 <= c < width


def _find_blue_components(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    height, width = grid.shape
    seen = np.zeros((height, width), dtype=bool)
    components: list[list[tuple[int, int]]] = []

    for r in range(height):
        for c in range(width):
            if seen[r, c] or int(grid[r, c]) != BLUE:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            component: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                component.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if _in_bounds((height, width), nr, nc) and not seen[nr, nc] and int(grid[nr, nc]) == BLUE:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(component)
    return components


def _is_two_cell_bar(component: list[tuple[int, int]]) -> bool:
    if len(component) != 2:
        return False
    (r0, c0), (r1, c1) = component
    return (r0 == r1 and abs(c0 - c1) == 1) or (c0 == c1 and abs(r0 - r1) == 1)


def _find_implied_plus_centers(grid: np.ndarray) -> list[tuple[int, int]]:
    height, width = grid.shape
    centers: list[tuple[int, int]] = []
    for r in range(3, height - 3):
        for c in range(3, width - 3):
            if int(grid[r, c]) != BG:
                continue
            if int(grid[r, c - 1]) != BG or int(grid[r, c + 1]) != BG:
                continue
            if int(grid[r - 1, c]) != BG or int(grid[r + 1, c]) != BG:
                continue
            if (
                int(grid[r, c - 3]) == BLUE
                and int(grid[r, c - 2]) == BLUE
                and int(grid[r, c + 2]) == BLUE
                and int(grid[r, c + 3]) == BLUE
                and int(grid[r - 3, c]) == BLUE
                and int(grid[r - 2, c]) == BLUE
                and int(grid[r + 2, c]) == BLUE
                and int(grid[r + 3, c]) == BLUE
            ):
                centers.append((r, c))
    return centers


def _mark_centers(grid: np.ndarray, centers: list[tuple[int, int]]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for r, c in centers:
        out[r, c] = YELLOW
    return out


def _extract_controls(grid: np.ndarray) -> list[tuple[int, int]]:
    height, width = grid.shape
    controls: list[tuple[int, int]] = []
    for r, c in np.argwhere(grid == RED):
        rr = int(r)
        cc = int(c)
        if rr in (0, height - 1) or cc in (0, width - 1):
            controls.append((rr, cc))
    return controls


def _ray_metadata(grid: np.ndarray) -> tuple[list[tuple[int, int]], set[int], set[int], set[str]]:
    height, width = grid.shape
    controls = _extract_controls(grid)
    split_rows: set[int] = set()
    split_cols: set[int] = set()
    edges: set[str] = set()

    for r, c in controls:
        if r == 0:
            split_cols.add(c)
            edges.add("top")
        elif r == height - 1:
            split_cols.add(c)
            edges.add("bottom")
        elif c == 0:
            split_rows.add(r)
            edges.add("left")
        elif c == width - 1:
            split_rows.add(r)
            edges.add("right")

    return controls, split_rows, split_cols, edges


def _draw_rays(marked: np.ndarray) -> np.ndarray:
    out = np.array(marked, dtype=int, copy=True)
    height, width = out.shape
    controls = _extract_controls(out)

    for r, c in controls:
        dr = dc = 0
        if r == 0:
            dr = 1
        elif r == height - 1:
            dr = -1
        elif c == 0:
            dc = 1
        elif c == width - 1:
            dc = -1

        rr, cc = r, c
        while 0 <= rr < height and 0 <= cc < width:
            if int(out[rr, cc]) == BG:
                out[rr, cc] = RED
            rr += dr
            cc += dc
    return out


def _keep_mask(shape: tuple[int, int], split_rows: set[int], split_cols: set[int], edges: set[str]) -> np.ndarray:
    height, width = shape
    keep = np.zeros((height, width), dtype=bool)

    for row in split_rows:
        keep[row, :] = True
    for col in split_cols:
        keep[:, col] = True

    row_bounds = [-1] + sorted(split_rows) + [height]
    col_bounds = [-1] + sorted(split_cols) + [width]

    for row_idx in range(len(row_bounds) - 1):
        r0 = row_bounds[row_idx] + 1
        r1 = row_bounds[row_idx + 1] - 1
        if r0 > r1:
            continue
        for col_idx in range(len(col_bounds) - 1):
            c0 = col_bounds[col_idx] + 1
            c1 = col_bounds[col_idx + 1] - 1
            if c0 > c1:
                continue
            touches: set[str] = set()
            if r0 == 0:
                touches.add("top")
            if r1 == height - 1:
                touches.add("bottom")
            if c0 == 0:
                touches.add("left")
            if c1 == width - 1:
                touches.add("right")
            if touches & edges:
                keep[r0 : r1 + 1, c0 : c1 + 1] = True

    return keep


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect implied-plus centers, draw border-controlled rays, then keep only the control-edge-side regions.

    Input structure:
    - The grid has a black background with many isolated blue two-cell bars.
    - Some groups of four bars form implied pluses around an empty center, and red control pixels sit on the border.
    - Each border control defines a full inward row or column cut, while blue bars remain in place and can create visible gaps in the painted ray.

    Rule:
    - First mark every black cell whose left/right arms are horizontal blue bars at offsets 2-3 and whose up/down arms are vertical blue bars at offsets 2-3 with yellow.
    - Next extend each border control straight inward to the opposite border, painting only black cells red and skipping over non-black cells without stopping.
    - Treat the control-aligned ray rows and columns as partition lines, keep the regions that touch at least one control edge together with the pixels on those partition lines, and clear every other region to black.

    Key insight:
    - The same blue bars serve two incompatible roles: before the cut they certify missing plus centers, and during the cut they behave like obstacles that punch holes in otherwise straight red rays.
    """
    g = np.asarray(grid, dtype=int)
    centers = _find_implied_plus_centers(g)
    marked = _mark_centers(g, centers)
    with_rays = _draw_rays(marked)
    _controls, split_rows, split_cols, edges = _ray_metadata(g)
    if not edges:
        return with_rays
    keep = _keep_mask(g.shape, split_rows, split_cols, edges)
    out = with_rays.copy()
    out[~keep] = BG
    return out


GEN_ATTEMPTS = 500
GRID_MIN = 15
GRID_MAX = 19
BAR_TARGET_MIN = 15
BAR_TARGET_MAX = 25


def _region_bounds(
    height: int, width: int, r_split: int, c_split: int
) -> dict[str, tuple[int, int, int, int]]:
    return {
        "TL": (0, r_split - 1, 0, c_split - 1),
        "TR": (0, r_split - 1, c_split + 1, width - 1),
        "BL": (r_split + 1, height - 1, 0, c_split - 1),
        "BR": (r_split + 1, height - 1, c_split + 1, width - 1),
    }


def _kept_regions(vertical_edge: str, horizontal_edge: str) -> set[str]:
    kept: set[str] = set()
    if vertical_edge == "top":
        kept.update({"TL", "TR"})
    else:
        kept.update({"BL", "BR"})
    if horizontal_edge == "left":
        kept.update({"TL", "BL"})
    else:
        kept.update({"TR", "BR"})
    return kept


def _clear_region(vertical_edge: str, horizontal_edge: str) -> str:
    if vertical_edge == "top" and horizontal_edge == "left":
        return "BR"
    if vertical_edge == "top" and horizontal_edge == "right":
        return "BL"
    if vertical_edge == "bottom" and horizontal_edge == "left":
        return "TR"
    return "TL"


def _center_candidates(bounds: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, r1, c0, c1 = bounds
    if r1 - r0 + 1 < 7 or c1 - c0 + 1 < 7:
        return []
    return [
        (r, c)
        for r in range(r0 + 3, r1 - 2)
        for c in range(c0 + 3, c1 - 2)
    ]


def _cells_are_empty(grid: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    height, width = grid.shape
    for r, c in cells:
        if not _in_bounds((height, width), r, c):
            return False
        if int(grid[r, c]) != BG:
            return False
    return True


def _bar_neighbors_are_clear(grid: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    height, width = grid.shape
    own = set(cells)
    for r, c in cells:
        for dr, dc in NEIGHBORS4:
            nr, nc = r + dr, c + dc
            if not _in_bounds((height, width), nr, nc) or (nr, nc) in own:
                continue
            if int(grid[nr, nc]) == BLUE:
                return False
    return True


def _place_bar(grid: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    if not _cells_are_empty(grid, cells):
        return False
    if not _bar_neighbors_are_clear(grid, cells):
        return False
    for r, c in cells:
        grid[r, c] = BLUE
    return True


def _bar_cells_from_anchor(anchor: tuple[int, int], horizontal: bool) -> list[tuple[int, int]]:
    r, c = anchor
    if horizontal:
        return [(r, c), (r, c + 1)]
    return [(r, c), (r + 1, c)]


def _place_cross(grid: np.ndarray, center: tuple[int, int]) -> bool:
    return _place_bar(grid, [(center[0], center[1] - 3), (center[0], center[1] - 2)]) and _place_bar(
        grid, [(center[0], center[1] + 2), (center[0], center[1] + 3)]
    ) and _place_bar(grid, [(center[0] - 3, center[1]), (center[0] - 2, center[1])]) and _place_bar(
        grid, [(center[0] + 2, center[1]), (center[0] + 3, center[1])]
    )


def _bar_count(grid: np.ndarray) -> int:
    return int(np.count_nonzero(grid == BLUE) // 2)


def _candidate_bar_anchors(height: int, width: int) -> list[tuple[int, int, bool]]:
    anchors: list[tuple[int, int, bool]] = []
    for r in range(height):
        for c in range(width - 1):
            anchors.append((r, c, True))
    for r in range(height - 1):
        for c in range(width):
            anchors.append((r, c, False))
    return anchors


def _forced_split_bar_candidates(
    height: int, width: int, r_split: int, c_split: int, kind: str
) -> list[list[tuple[int, int]]]:
    candidates: list[list[tuple[int, int]]] = []
    if kind in {"vertical", "both"}:
        for r in range(1, height - 1):
            if c_split - 1 >= 1:
                candidates.append([(r, c_split - 1), (r, c_split)])
            if c_split + 1 <= width - 2:
                candidates.append([(r, c_split), (r, c_split + 1)])
    if kind in {"horizontal", "both"}:
        for c in range(1, width - 1):
            if r_split - 1 >= 1:
                candidates.append([(r_split - 1, c), (r_split, c)])
            if r_split + 1 <= height - 2:
                candidates.append([(r_split, c), (r_split + 1, c)])
    return candidates


def _semantic_region_mask(
    height: int,
    width: int,
    vertical_edge: str,
    horizontal_edge: str,
    r_split: int,
    c_split: int,
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    bounds = _region_bounds(height, width, r_split, c_split)
    for name, (r0, r1, c0, c1) in bounds.items():
        if name in _kept_regions(vertical_edge, horizontal_edge):
            mask[r0 : r1 + 1, c0 : c1 + 1] = True
    mask[r_split, :] = True
    mask[:, c_split] = True
    return mask


def _check_blue_components(grid: np.ndarray) -> bool:
    components = _find_blue_components(grid)
    return bool(components) and all(_is_two_cell_bar(component) for component in components)


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    height = int(rng.integers(GRID_MIN, GRID_MAX))
    width = int(rng.integers(GRID_MIN, GRID_MAX))
    if height < 15 or width < 15:
        return None

    min_r_split = 7
    max_r_split = height - 8
    min_c_split = 7
    max_c_split = width - 8
    if min_r_split > max_r_split or min_c_split > max_c_split:
        return None

    r_split = int(rng.integers(min_r_split, max_r_split + 1))
    c_split = int(rng.integers(min_c_split, max_c_split + 1))
    vertical_edge = str(rng.choice(["top", "bottom"]))
    horizontal_edge = str(rng.choice(["left", "right"]))

    clear_name = _clear_region(vertical_edge, horizontal_edge)
    kept_names = sorted(_kept_regions(vertical_edge, horizontal_edge))
    bounds = _region_bounds(height, width, r_split, c_split)

    cross_count = int(rng.integers(2, 4))
    region_choices = [clear_name]
    remaining_kept = kept_names[:]
    rng.shuffle(remaining_kept)
    while len(region_choices) < cross_count and remaining_kept:
        region_choices.append(remaining_kept.pop())
    if len(region_choices) != cross_count:
        return None

    grid = np.zeros((height, width), dtype=int)
    chosen_centers: list[tuple[int, int]] = []
    intended_centers: list[tuple[int, int]] = []
    for region_name in region_choices:
        candidates = _center_candidates(bounds[region_name])
        rng.shuffle(candidates)
        center = None
        for candidate in candidates:
            if candidate in chosen_centers:
                continue
            center = candidate
            break
        if center is None:
            return None
        chosen_centers.append(center)
        intended_centers.append(center)

    for center in intended_centers:
        if not _place_cross(grid, center):
            return None

    total_bars_target = int(rng.integers(BAR_TARGET_MIN, BAR_TARGET_MAX + 1))
    target_distractors = total_bars_target - 4 * cross_count
    if target_distractors < 3:
        return None

    forced_kind = str(rng.choice(["none", "vertical", "horizontal", "both"], p=[0.2, 0.3, 0.3, 0.2]))
    forced_placed = 0
    if forced_kind != "none":
        candidates = _forced_split_bar_candidates(height, width, r_split, c_split, forced_kind)
        rng.shuffle(candidates)
        desired_forced = 1 if forced_kind in {"vertical", "horizontal"} else 2
        for cells in candidates:
            if forced_placed >= min(desired_forced, target_distractors):
                break
            if _place_bar(grid, cells):
                forced_placed += 1

    anchors = _candidate_bar_anchors(height, width)
    rng.shuffle(anchors)
    placed_distractors = forced_placed
    for r, c, horizontal in anchors:
        if placed_distractors >= target_distractors:
            break
        if r in (0, height - 1) or c in (0, width - 1):
            continue
        cells = _bar_cells_from_anchor((r, c), horizontal)
        if _place_bar(grid, cells):
            placed_distractors += 1
    if placed_distractors != target_distractors:
        return None

    detected_centers = _find_implied_plus_centers(grid)
    if sorted(detected_centers) != sorted(intended_centers):
        return None
    if not _check_blue_components(grid):
        return None
    if _bar_count(grid) != total_bars_target:
        return None

    if vertical_edge == "top":
        grid[0, c_split] = RED
    else:
        grid[height - 1, c_split] = RED
    if horizontal_edge == "left":
        grid[r_split, 0] = RED
    else:
        grid[r_split, width - 1] = RED

    out = main(grid)
    if out.shape != grid.shape:
        return None

    keep_mask = _semantic_region_mask(height, width, vertical_edge, horizontal_edge, r_split, c_split)
    kept_centers = [center for center in intended_centers if keep_mask[center]]
    cleared_centers = [center for center in intended_centers if not keep_mask[center]]
    if not kept_centers or not cleared_centers:
        return None

    for r, c in kept_centers:
        if int(out[r, c]) != YELLOW:
            return None
    for r, c in cleared_centers:
        if int(out[r, c]) != BG:
            return None

    if np.array_equal(out, grid):
        return None

    if not np.any(out == RED):
        return None
    if not np.any(out == YELLOW):
        return None

    if forced_kind in {"vertical", "both"} and not np.any(grid[:, c_split] == BLUE):
        return None
    if forced_kind in {"horizontal", "both"} and not np.any(grid[r_split, :] == BLUE):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid implied-plus ray-partition input")
