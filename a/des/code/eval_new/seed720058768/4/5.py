from __future__ import annotations

from collections import deque
import numpy as np


BG = 0
DIVIDER = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIRS8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 0),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)
GEN_RETRIES = 300
SHAPE_RETRIES = 120
SPLIT_RETRIES = 300
PLACEMENT_RETRIES = 160


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_separator(grid: np.ndarray, bg: int) -> tuple[str, int, int]:
    h, w = grid.shape
    candidates: list[tuple[str, int, int]] = []

    if h % 2 == 1:
        mid_r = h // 2
        if np.all(grid[mid_r, :] == grid[mid_r, 0]) and int(grid[mid_r, 0]) != bg:
            candidates.append(("horizontal", mid_r, int(grid[mid_r, 0])))

    if w % 2 == 1:
        mid_c = w // 2
        if np.all(grid[:, mid_c] == grid[0, mid_c]) and int(grid[0, mid_c]) != bg:
            candidates.append(("vertical", mid_c, int(grid[0, mid_c])))

    assert len(candidates) == 1, "expected exactly one central separator"
    return candidates[0]


def _mirror_point(orientation: str, axis: int, r: int, c: int) -> tuple[int, int]:
    if orientation == "vertical":
        return r, 2 * axis - c
    return 2 * axis - r, c


def _extract_components(grid: np.ndarray, valid_mask: np.ndarray) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros_like(valid_mask, dtype=bool)
    components: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            if not valid_mask[r, c] or seen[r, c]:
                continue

            color = int(grid[r, c])
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: set[tuple[int, int]] = set()

            while queue:
                rr, cc = queue.popleft()
                coords.add((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and valid_mask[nr, nc] and not seen[nr, nc]:
                        if int(grid[nr, nc]) == color:
                            seen[nr, nc] = True
                            queue.append((nr, nc))

            components.append({"color": color, "coords": coords})

    components.sort(key=lambda obj: min(obj["coords"]))  # type: ignore[arg-type]
    return components


def _pair_components(
    side_a: list[dict[str, object]],
    side_b: list[dict[str, object]],
    orientation: str,
    axis: int,
) -> list[tuple[int, int]]:
    n = len(side_a)
    assert n == len(side_b) and n > 0, "both sides must have the same positive object count"

    scores = [[0] * n for _ in range(n)]
    for i, obj_a in enumerate(side_a):
        mirrored_a = {
            _mirror_point(orientation, axis, r, c)
            for r, c in obj_a["coords"]  # type: ignore[assignment]
        }
        for j, obj_b in enumerate(side_b):
            overlap = len(mirrored_a & obj_b["coords"])  # type: ignore[operator]
            scores[i][j] = overlap

    order = sorted(range(n), key=lambda i: sum(score > 0 for score in scores[i]))
    best_assignment: dict[int, int] | None = None
    best_score = -1

    def dfs(k: int, used_mask: int, total_score: int, assignment: dict[int, int]) -> None:
        nonlocal best_assignment, best_score
        if k == n:
            if total_score > best_score:
                best_score = total_score
                best_assignment = assignment.copy()
            return

        i = order[k]
        choices = [j for j in range(n) if not (used_mask & (1 << j)) and scores[i][j] > 0]
        choices.sort(key=lambda j: scores[i][j], reverse=True)
        for j in choices:
            assignment[i] = j
            dfs(k + 1, used_mask | (1 << j), total_score + scores[i][j], assignment)
            del assignment[i]

    dfs(0, 0, 0, {})
    assert best_assignment is not None, "failed to find an overlap-based pairing"
    return sorted(best_assignment.items())


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each mirrored fragment using the missing pixels from its opposite-side partner.

    Input structure:
    - Background (bg) is black.
    - A single grey separator line runs through the exact middle of the grid, either horizontally or vertically.
    - Both sides of the separator contain several colored 4-connected fragment objects.
    - Every fragment has exactly one mirrored partner on the opposite side; the two partner shapes overlap after reflection but each is missing some pixels.

    Rule:
    - Detect whether the central separator is horizontal or vertical.
    - Split the foreground objects by separator side and mirror each object across the separator.
    - Pair objects whose reflected shapes overlap on the opposite side.
    - For each pair, fill every pixel missing from one fragment but present in the mirrored partner using the partner's color.
    - Keep the original pixels and the separator line unchanged.

    Key insight:
    - Each object is only a partial version of a larger mirrored shape, and the opposite-side partner supplies exactly the missing geometry while also donating the fill color for those missing cells.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _most_common_color(g)
    orientation, axis, divider = _find_separator(g, bg)

    valid = (g != bg) & (g != divider)
    if orientation == "vertical":
        side_a = _extract_components(g, valid & (np.arange(g.shape[1])[None, :] < axis))
        side_b = _extract_components(g, valid & (np.arange(g.shape[1])[None, :] > axis))
    else:
        side_a = _extract_components(g, valid & (np.arange(g.shape[0])[:, None] < axis))
        side_b = _extract_components(g, valid & (np.arange(g.shape[0])[:, None] > axis))

    pairs = _pair_components(side_a, side_b, orientation, axis)

    out = g.copy()
    for i, j in pairs:
        obj_a = side_a[i]
        obj_b = side_b[j]
        coords_a: set[tuple[int, int]] = obj_a["coords"]  # type: ignore[assignment]
        coords_b: set[tuple[int, int]] = obj_b["coords"]  # type: ignore[assignment]
        color_a = int(obj_a["color"])
        color_b = int(obj_b["color"])

        mirrored_a = {_mirror_point(orientation, axis, r, c) for r, c in coords_a}
        mirrored_b = {_mirror_point(orientation, axis, r, c) for r, c in coords_b}

        assert coords_a & mirrored_b, "paired fragments must overlap on side A"
        assert coords_b & mirrored_a, "paired fragments must overlap on side B"
        assert coords_a != mirrored_b and coords_b != mirrored_a, "both fragments must be incomplete"

        for r, c in mirrored_b - coords_a:
            assert int(out[r, c]) == bg, "missing side-A pixels must be empty before filling"
            out[r, c] = color_b

        for r, c in mirrored_a - coords_b:
            assert int(out[r, c]) == bg, "missing side-B pixels must be empty before filling"
            out[r, c] = color_a

    return out


def _randint(rng: np.random.Generator, lo: int, hi: int) -> int:
    return int(rng.integers(lo, hi + 1))


def _choice(rng: np.random.Generator, items: list[tuple[int, int]]) -> tuple[int, int]:
    return items[int(rng.integers(0, len(items)))]


def _normalize(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return {(r - min_r, c - min_c) for r, c in cells}


def _bbox_size(cells: set[tuple[int, int]]) -> tuple[int, int]:
    return max(r for r, _ in cells) + 1, max(c for _, c in cells) + 1


def _is_connected(cells: set[tuple[int, int]]) -> bool:
    if not cells:
        return False
    start = next(iter(cells))
    queue = deque([start])
    seen = {start}
    while queue:
        r, c = queue.popleft()
        for dr, dc in DIRS4:
            nbr = (r + dr, c + dc)
            if nbr in cells and nbr not in seen:
                seen.add(nbr)
                queue.append(nbr)
    return seen == cells


def _random_connected_shape(
    rng: np.random.Generator,
    max_h: int,
    max_w: int,
) -> set[tuple[int, int]] | None:
    if min(max_h, max_w) < 3:
        return None

    for _ in range(SHAPE_RETRIES):
        box_h = _randint(rng, 3, min(6, max_h))
        box_w = _randint(rng, 3, min(6, max_w))
        max_area = min(box_h * box_w - 1, 12)
        if max_area < 5:
            continue
        target_area = _randint(rng, 5, max_area)

        start = (_randint(rng, 0, box_h - 1), _randint(rng, 0, box_w - 1))
        cells = {start}
        while len(cells) < target_area:
            frontier: set[tuple[int, int]] = set()
            for r, c in cells:
                for dr, dc in DIRS4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < box_h and 0 <= nc < box_w and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            cells.add(_choice(rng, list(frontier)))

        cells = _normalize(cells)
        rows = {r for r, _ in cells}
        cols = {c for _, c in cells}
        if len(cells) < 5 or len(rows) < 2 or len(cols) < 2:
            continue
        if len(cells) == box_h * box_w:
            continue
        return cells

    return None


def _random_connected_subset(
    rng: np.random.Generator,
    cells: set[tuple[int, int]],
    size: int,
) -> set[tuple[int, int]] | None:
    if size <= 0 or size > len(cells):
        return None

    cell_list = list(cells)
    subset = {cell_list[int(rng.integers(0, len(cell_list)))]}
    while len(subset) < size:
        frontier = [
            cell
            for cell in cells - subset
            if any((cell[0] + dr, cell[1] + dc) in subset for dr, dc in DIRS4)
        ]
        if not frontier:
            return None
        subset.add(_choice(rng, frontier))
    return subset


def _split_shape(
    rng: np.random.Generator,
    shape: set[tuple[int, int]],
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]] | None:
    n = len(shape)
    low = max(2, (3 * n) // 5)
    for _ in range(SPLIT_RETRIES):
        size_a = _randint(rng, low, n - 1)
        size_b = _randint(rng, low, n - 1)
        frag_a = _random_connected_subset(rng, shape, size_a)
        frag_b = _random_connected_subset(rng, shape, size_b)
        if frag_a is None or frag_b is None:
            continue
        if frag_a | frag_b != shape:
            continue
        if not (frag_a & frag_b):
            continue
        if not (frag_a - frag_b) or not (frag_b - frag_a):
            continue
        return frag_a, frag_b
    return None


def _halo(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    halo: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr, dc in DIRS8:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                halo.add((nr, nc))
    return halo


def _paint_fragment(
    grid: np.ndarray,
    orientation: str,
    fragment: set[tuple[int, int]],
    color: int,
) -> None:
    h, w = grid.shape
    if orientation == "vertical":
        for r, c in fragment:
            grid[r, w - 1 - c] = color
    else:
        for r, c in fragment:
            grid[h - 1 - r, c] = color


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    orientation = "vertical" if float(rng.random()) < 0.5 else "horizontal"
    pair_count = _randint(rng, 1, 4)

    if orientation == "vertical":
        h = _randint(rng, max(13, 4 * pair_count + 5), 25)
        half_w = _randint(rng, 6, 11)
        w = 2 * half_w + 1
        side_h, side_w = h, half_w
    else:
        half_h = _randint(rng, 6, 11)
        h = 2 * half_h + 1
        w = _randint(rng, max(13, 4 * pair_count + 5), 25)
        side_h, side_w = half_h, w

    if max(h, w) > 30:
        return None

    grid = np.full((h, w), BG, dtype=int)
    if orientation == "vertical":
        axis = w // 2
        grid[:, axis] = DIVIDER
    else:
        axis = h // 2
        grid[axis, :] = DIVIDER

    color_order = list(rng.permutation(np.array(COLOR_POOL, dtype=int)))
    reserved: set[tuple[int, int]] = set()
    planned: list[tuple[set[tuple[int, int]], set[tuple[int, int]], int, int, int, int]] = []

    for pair_idx in range(pair_count):
        pair_colors = (int(color_order[2 * pair_idx]), int(color_order[2 * pair_idx + 1]))
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            shape = _random_connected_shape(rng, side_h, side_w)
            if shape is None:
                continue
            split = _split_shape(rng, shape)
            if split is None:
                continue

            frag_a, frag_b = split
            if float(rng.random()) < 0.5:
                frag_a, frag_b = frag_b, frag_a

            shape_h, shape_w = _bbox_size(shape)
            max_r0 = side_h - shape_h
            max_c0 = side_w - shape_w
            if max_r0 < 0 or max_c0 < 0:
                continue

            r0 = _randint(rng, 0, max_r0)
            c0 = _randint(rng, 0, max_c0)
            placed_shape = {(r0 + r, c0 + c) for r, c in shape}
            if reserved & _halo(placed_shape, side_h, side_w):
                continue

            reserved |= _halo(placed_shape, side_h, side_w)
            placed_a = {(r0 + r, c0 + c) for r, c in frag_a}
            placed_b = {(r0 + r, c0 + c) for r, c in frag_b}
            planned.append((placed_a, placed_b, pair_colors[0], pair_colors[1], r0, c0))
            placed = True
            break

        if not placed:
            return None

    for frag_a, frag_b, color_a, color_b, _r0, _c0 in planned:
        for r, c in frag_a:
            grid[r, c] = color_a
        _paint_fragment(grid, orientation, frag_b, color_b)

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None

    valid = (grid != BG) & (grid != DIVIDER)
    if orientation == "vertical":
        left = _extract_components(grid, valid & (np.arange(w)[None, :] < axis))
        right = _extract_components(grid, valid & (np.arange(w)[None, :] > axis))
    else:
        top = _extract_components(grid, valid & (np.arange(h)[:, None] < axis))
        right = _extract_components(grid, valid & (np.arange(h)[:, None] > axis))
        left = top
    if len(left) != pair_count or len(right) != pair_count:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirrored-fragment input")
