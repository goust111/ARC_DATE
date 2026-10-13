import random
from collections import Counter

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
BG_COLOR = 0
PALETTE = tuple(range(1, 10))
GEN_ATTEMPTS = 400


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    """BBox of an (N,2) coordinate array using ARC's (row, col) convention."""
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _run_starts(grid: np.ndarray, axis: int) -> list[int]:
    size = int(grid.shape[axis])
    keep = [0]
    for idx in range(1, size):
        cur = grid[idx, :] if axis == 0 else grid[:, idx]
        prev = grid[idx - 1, :] if axis == 0 else grid[:, idx - 1]
        if not np.array_equal(cur, prev):
            keep.append(idx)
    return keep


def compress_grid(grid: np.ndarray) -> np.ndarray:
    row_ids = _run_starts(grid, axis=0)
    col_ids = _run_starts(grid, axis=1)
    return np.asarray(grid, dtype=int)[np.ix_(row_ids, col_ids)].copy()


def find_components(
    grid: np.ndarray, bg: int, *, connectivity: int = 4
) -> list[dict[str, np.ndarray | int | tuple[int, int, int, int]]]:
    assert connectivity == 4
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, np.ndarray | int | tuple[int, int, int, int]]] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == bg or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            arr = np.array(coords, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": bbox_coords(arr),
                    "area": int(arr.shape[0]),
                }
            )
    return comps


def _select_largest_component(grid: np.ndarray) -> tuple[int, int, np.ndarray, tuple[int, int, int, int]]:
    bg = get_bg(grid)
    comps = find_components(grid, bg, connectivity=4)
    assert comps, "expected foreground components after compression"
    areas = [int(comp["area"]) for comp in comps]
    max_area = max(areas)
    largest = [comp for comp in comps if int(comp["area"]) == max_area]
    assert len(largest) == 1, "largest component must be unique"
    comp = largest[0]
    return bg, int(comp["color"]), comp["coords"], comp["bbox"]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Collapse repeated neighboring rows and columns, then crop the unique largest connected object.

    Input structure:
    - The visible grid is a run-length-expanded image: each original row and column may be repeated
      1-4 times, so adjacent rows or columns can be exactly identical.
    - After compression, the recovered small grid has a dominant background color and several
      non-background monochrome 4-connected components.
    - Exactly one non-background component has the largest area.

    Rule:
    - Keep only the first row of each consecutive run of identical rows, and the first column of each
      consecutive run of identical columns.
    - On the compressed grid, take the most frequent color as background and find all monochrome
      4-connected non-background components.
    - Select the unique largest component, crop to its bounding box, and replace every other color
      inside that crop with the background while preserving the selected component and any holes.

    Key insight:
    - The apparent noise is just uneven row/column stretching; once those duplicate vectors are merged,
      the hidden objects separate cleanly and the target is identified by connected area.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    compressed = compress_grid(g)
    bg, color, coords, bbox = _select_largest_component(compressed)

    r0, c0, r1, c1 = bbox
    out = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    out[coords[:, 0] - r0, coords[:, 1] - c0] = color
    return out


def _bounded_composition(total: int, parts: int, lo: int, hi: int) -> list[int]:
    vals: list[int] = []
    remaining = total
    for idx in range(parts):
        left = parts - idx - 1
        low = max(lo, remaining - hi * left)
        high = min(hi, remaining - lo * left)
        vals.append(random.randint(low, high))
        remaining -= vals[-1]
    random.shuffle(vals)
    return vals


def _random_blob(height: int, width: int, area: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    seed = (random.randrange(height), random.randrange(width))
    mask[seed] = True

    while int(mask.sum()) < area:
        frontier: set[tuple[int, int]] = set()
        for r, c in np.argwhere(mask):
            for dr, dc in DIR4:
                nr, nc = int(r) + dr, int(c) + dc
                if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]:
                    frontier.add((nr, nc))
        if not frontier:
            return _random_blob(height, width, area)
        rr, cc = random.choice(tuple(frontier))
        mask[rr, cc] = True
    return mask


def _hole_shape(area: int) -> np.ndarray | None:
    choices = {
        8: [(3, 3)],
        10: [(3, 4), (4, 3)],
        12: [(4, 4)],
        14: [(4, 5), (5, 4)],
    }.get(area, [])
    if not choices:
        return None
    h, w = random.choice(choices)
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _make_main_patch(color: int) -> np.ndarray:
    area = random.randint(8, 15)
    use_hole = random.random() < 0.35 and area in {8, 10, 12, 14}
    mask = _hole_shape(area) if use_hole else None
    if mask is None:
        while True:
            h = random.randint(3, 5)
            w = random.randint(3, 5)
            if area <= h * w:
                break
        mask = _random_blob(h, w, area)
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = color
    return patch


def _small_patch(area: int, color: int) -> np.ndarray:
    if area == 1:
        return np.array([[color]], dtype=int)
    if area == 2:
        return np.array([[color, color]], dtype=int) if random.random() < 0.5 else np.array([[color], [color]], dtype=int)
    if random.random() < 0.5:
        return np.array([[color, color, color]], dtype=int)
    return np.array([[color, color], [color, 0]], dtype=int)


def _touches_existing(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> bool:
    h, w = patch.shape
    for pr in range(h):
        for pc in range(w):
            if int(patch[pr, pc]) == BG_COLOR:
                continue
            gr, gc = top + pr, left + pc
            if int(grid[gr, gc]) != BG_COLOR:
                return True
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr, nc = gr + dr, gc + dc
                    if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                        if (dr != 0 or dc != 0) and int(grid[nr, nc]) != BG_COLOR:
                            return True
    return False


def _paste_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    sub = grid[top : top + h, left : left + w]
    mask = patch != BG_COLOR
    sub[mask] = patch[mask]


def _count_nonzero_rows_cols(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fg = grid != BG_COLOR
    return fg.any(axis=1), fg.any(axis=0)


def _seed_is_valid(seed: np.ndarray, main_color: int) -> bool:
    if seed.shape[0] < 7 or seed.shape[0] > 9 or seed.shape[1] < 7 or seed.shape[1] > 9:
        return False
    if not np.array_equal(compress_grid(seed), seed):
        return False
    if get_bg(seed) != BG_COLOR:
        return False

    row_fg, col_fg = _count_nonzero_rows_cols(seed)
    if int(row_fg.sum()) < seed.shape[0] - 1 or int(col_fg.sum()) < seed.shape[1] - 1:
        return False

    comps = find_components(seed, BG_COLOR, connectivity=4)
    if not (3 <= len(comps) <= 5):
        return False
    comps = sorted(comps, key=lambda comp: int(comp["area"]), reverse=True)
    largest = int(comps[0]["area"])
    second = int(comps[1]["area"])
    if not (8 <= largest <= 15):
        return False
    if largest <= 2 * second:
        return False
    if int(comps[0]["color"]) != main_color:
        return False
    if sum(1 for comp in comps if int(comp["area"]) == largest) != 1:
        return False
    if any(not (1 <= int(comp["area"]) <= 3) for comp in comps[1:]):
        return False
    return True


def _build_seed() -> np.ndarray | None:
    seed_h = random.randint(7, 9)
    seed_w = random.randint(7, 9)
    seed = np.zeros((seed_h, seed_w), dtype=int)

    main_color = random.choice(PALETTE)
    main_patch = _make_main_patch(main_color)
    if main_patch.shape[0] > seed_h or main_patch.shape[1] > seed_w:
        return None
    main_r = random.randint(0, seed_h - main_patch.shape[0])
    main_c = random.randint(0, seed_w - main_patch.shape[1])
    _paste_patch(seed, main_patch, main_r, main_c)

    comp_count = random.randint(2, 4)
    same_color_slots = 1 if random.random() < 0.5 else 0
    areas = [random.choices([1, 2, 3], weights=[5, 3, 2], k=1)[0] for _ in range(comp_count)]
    random.shuffle(areas)

    other_colors = [c for c in PALETTE if c != main_color]
    for idx, area in enumerate(areas):
        color = main_color if idx < same_color_slots else random.choice(other_colors)
        patch = _small_patch(area, color)
        if random.random() < 0.5:
            patch = np.rot90(patch, random.randint(0, 3))
        positions = [
            (r, c)
            for r in range(seed_h - patch.shape[0] + 1)
            for c in range(seed_w - patch.shape[1] + 1)
        ]
        random.shuffle(positions)
        placed = False
        for r, c in positions:
            if not _touches_existing(seed, patch, r, c):
                _paste_patch(seed, patch, r, c)
                placed = True
                break
        if not placed:
            return None

    if not _seed_is_valid(seed, main_color):
        return None
    return seed


def _expand_seed(seed: np.ndarray) -> np.ndarray | None:
    row_total = random.randint(18, 22)
    col_total = random.randint(18, 22)
    if not (seed.shape[0] <= row_total <= 4 * seed.shape[0]):
        return None
    if not (seed.shape[1] <= col_total <= 4 * seed.shape[1]):
        return None

    row_runs = _bounded_composition(row_total, seed.shape[0], 1, 4)
    col_runs = _bounded_composition(col_total, seed.shape[1], 1, 4)
    if max(row_runs) == 1 or max(col_runs) == 1:
        return None

    expanded = np.repeat(np.repeat(seed, row_runs, axis=0), col_runs, axis=1)
    if not np.array_equal(compress_grid(expanded), seed):
        return None
    return expanded


def _attempt() -> np.ndarray | None:
    seed = _build_seed()
    if seed is None:
        return None
    expanded = _expand_seed(seed)
    if expanded is None:
        return None

    out = main(expanded)
    if out.ndim != 2 or max(out.shape) > 30:
        return None
    return expanded


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
