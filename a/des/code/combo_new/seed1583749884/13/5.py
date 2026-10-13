from __future__ import annotations

import random

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))

DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIRS8 = DIRS4 + ((-1, -1), (-1, 1), (1, -1), (1, 1))

GRID_MIN = 18
GRID_MAX = 30
OBJECT_COUNT_RANGE = (3, 5)
PATCH_ATTEMPTS = 80
PLACE_ATTEMPTS = 200
GEN_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _components(mask: np.ndarray, connectivity: int) -> list[np.ndarray]:
    dirs = DIRS4 if connectivity == 4 else DIRS8
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(mask.shape[0]):
        for c in range(mask.shape[1]):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in dirs:
                    nr, nc = cr + dr, cc + dc
                    if (
                        _in_bounds(mask.shape, nr, nc)
                        and bool(mask[nr, nc])
                        and not bool(seen[nr, nc])
                    ):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _tight_crop_mask(mask: np.ndarray) -> np.ndarray:
    rows, cols = np.where(mask)
    assert rows.size > 0, "expected a nonempty mask"
    return mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]


def _unique_largest_color(values: np.ndarray) -> int:
    colors, counts = np.unique(values[values != BG], return_counts=True)
    assert colors.size > 0, "expected at least one foreground color"
    best = int(counts.max())
    winners = colors[counts == best]
    assert winners.size == 1, "expected a unique largest color"
    return int(winners[0])


def _two_color_trace(patch: np.ndarray) -> np.ndarray:
    values = patch[patch != BG]
    colors = np.unique(values)
    assert colors.size == 2, "expected exactly two colors"
    major = _unique_largest_color(values)

    out = np.zeros_like(patch)
    keep = np.zeros_like(patch, dtype=bool)
    for r, c in np.argwhere(patch == major):
        for dr, dc in DIRS4:
            nr, nc = int(r + dr), int(c + dc)
            if _in_bounds(patch.shape, nr, nc) and int(patch[nr, nc]) not in (BG, major):
                keep[r, c] = True
                break
    assert bool(keep.any()), "expected a visible interface trace"
    out[keep] = major
    return out


def _simplify_patch(patch: np.ndarray) -> np.ndarray:
    out = np.zeros_like(patch)
    values = patch[patch != BG]
    colors = np.unique(values)

    if colors.size == 1:
        return patch.copy()
    if colors.size == 2:
        return _two_color_trace(patch)

    major = _unique_largest_color(values)
    out[patch == major] = major
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Simplify each foreground object according to how many colors it contains.

    Input structure:
    - The background is black.
    - The grid contains several 8-connected foreground objects separated by background.
    - Each object contains one, two, or three non-background colors.

    Rule:
    - For each object, count its distinct non-background colors.
    - If the object is monochrome, keep it unchanged.
    - If it has exactly two colors, keep only the larger color's side of the 4-neighbor interface between the two regions.
    - If it has three or more colors, keep only the pixels of the unique largest color and erase the rest.

    Key insight:
    - The same geometric object is processed by branching on internal color diversity, so color variety decides which simplification rule applies.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = np.zeros_like(g)
    for coords in _components(g != BG, connectivity=8):
        rows = coords[:, 0]
        cols = coords[:, 1]
        r0, r1 = int(rows.min()), int(rows.max())
        c0, c1 = int(cols.min()), int(cols.max())

        patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
        patch[rows - r0, cols - c0] = g[rows, cols]
        simplified = _simplify_patch(patch)

        keep = simplified != BG
        window = out[r0 : r1 + 1, c0 : c1 + 1]
        window[keep] = simplified[keep]

    return out


def _choose_colors(k: int) -> list[int]:
    return random.sample(PALETTE, k)


def _halo(coords: set[tuple[int, int]], radius: int = 1) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in coords:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                out.add((r + dr, c + dc))
    return out


def _grow_blob_mask(height: int, width: int, target_area: int) -> np.ndarray:
    seed = (random.randrange(height), random.randrange(width))
    mask = np.zeros((height, width), dtype=bool)
    mask[seed] = True
    frontier = {
        (seed[0] + dr, seed[1] + dc)
        for dr, dc in DIRS8
        if _in_bounds(mask.shape, seed[0] + dr, seed[1] + dc)
    }

    while int(mask.sum()) < target_area:
        options = list(frontier)
        weights = []
        for r, c in options:
            support = sum(
                _in_bounds(mask.shape, r + dr, c + dc) and bool(mask[r + dr, c + dc])
                for dr, dc in DIRS8
            )
            dist = abs(r - seed[0]) + abs(c - seed[1])
            weights.append((support + 1) ** 2 / (dist + 1))
        chosen = random.choices(options, weights=weights, k=1)[0]
        frontier.remove(chosen)
        mask[chosen] = True
        for dr, dc in DIRS8:
            nr, nc = chosen[0] + dr, chosen[1] + dc
            if _in_bounds(mask.shape, nr, nc) and not bool(mask[nr, nc]):
                frontier.add((nr, nc))
    return _tight_crop_mask(mask)


def _grow_region(
    mask: np.ndarray,
    blocked: set[tuple[int, int]],
    size: int,
    connectivity: int = 4,
) -> set[tuple[int, int]] | None:
    dirs = DIRS4 if connectivity == 4 else DIRS8
    allowed = {(int(r), int(c)) for r, c in np.argwhere(mask) if (int(r), int(c)) not in blocked}
    if len(allowed) < size:
        return None

    for _ in range(PATCH_ATTEMPTS):
        seed = random.choice(tuple(allowed))
        region = {seed}
        frontier = {
            (seed[0] + dr, seed[1] + dc)
            for dr, dc in dirs
            if (seed[0] + dr, seed[1] + dc) in allowed
        }
        while len(region) < size and frontier:
            options = tuple(frontier)
            weights = [
                1 + sum((r + dr, c + dc) in region for dr, dc in dirs)
                for r, c in options
            ]
            chosen = random.choices(options, weights=weights, k=1)[0]
            frontier.remove(chosen)
            region.add(chosen)
            for dr, dc in dirs:
                nxt = (chosen[0] + dr, chosen[1] + dc)
                if nxt in allowed and nxt not in region:
                    frontier.add(nxt)
        if len(region) == size:
            return region
    return None


def _monochrome_patch() -> tuple[np.ndarray, np.ndarray]:
    for _ in range(PATCH_ATTEMPTS):
        height = random.randint(4, 7)
        width = random.randint(4, 7)
        area = random.randint(max(5, (height * width + 1) // 2), height * width)
        mask = _grow_blob_mask(height, width, area)
        if min(mask.shape) < 2:
            continue
        color = _choose_colors(1)[0]
        patch = np.zeros(mask.shape, dtype=int)
        patch[mask] = color
        return patch, patch.copy()
    raise RuntimeError("failed to build a monochrome object")


def _bicolor_patch() -> tuple[np.ndarray, np.ndarray]:
    for _ in range(PATCH_ATTEMPTS):
        height = random.randint(5, 8)
        width = random.randint(5, 8)
        major, minor = _choose_colors(2)
        patch = np.zeros((height, width), dtype=int)
        orientation = random.choice(("V", "H"))

        if orientation == "V":
            max_minor = max(1, min(3, (width - 1) // 2))
            side = random.choice(("L", "R"))
            widths = [random.randint(1, max_minor)]
            for _row in range(1, height):
                lo = max(1, widths[-1] - 1)
                hi = min(max_minor, widths[-1] + 1)
                widths.append(random.randint(lo, hi))
            for r, minor_width in enumerate(widths):
                if side == "L":
                    patch[r, :minor_width] = minor
                    patch[r, minor_width:] = major
                else:
                    split = width - minor_width
                    patch[r, :split] = major
                    patch[r, split:] = minor
        else:
            max_minor = max(1, min(3, (height - 1) // 2))
            side = random.choice(("T", "B"))
            heights = [random.randint(1, max_minor)]
            for _col in range(1, width):
                lo = max(1, heights[-1] - 1)
                hi = min(max_minor, heights[-1] + 1)
                heights.append(random.randint(lo, hi))
            for c, minor_height in enumerate(heights):
                if side == "T":
                    patch[:minor_height, c] = minor
                    patch[minor_height:, c] = major
                else:
                    split = height - minor_height
                    patch[:split, c] = major
                    patch[split:, c] = minor

        out = _two_color_trace(patch)
        if int((patch == minor).sum()) >= 3 and int(out.sum() > 0):
            return patch, out
    raise RuntimeError("failed to build a bicolor object")


def _tricolor_patch() -> tuple[np.ndarray, np.ndarray]:
    for _ in range(PATCH_ATTEMPTS):
        height = random.randint(5, 8)
        width = random.randint(5, 8)
        area = random.randint(max(10, (height * width + 1) // 2), height * width)
        mask = _grow_blob_mask(height, width, area)
        area = int(mask.sum())
        if area < 9 or min(mask.shape) < 2:
            continue

        accent_a = random.randint(1, min(3, max(1, area // 4)))
        accent_b = random.randint(1, min(3, max(1, area // 4)))
        if area - accent_a - accent_b <= max(accent_a, accent_b):
            continue

        region_a = _grow_region(mask, set(), accent_a, connectivity=4)
        if region_a is None:
            continue
        region_b = _grow_region(mask, _halo(region_a, radius=1), accent_b, connectivity=4)
        if region_b is None:
            continue

        base, color_a, color_b = _choose_colors(3)
        patch = np.zeros(mask.shape, dtype=int)
        patch[mask] = base
        for r, c in region_a:
            patch[r, c] = color_a
        for r, c in region_b:
            patch[r, c] = color_b

        out = np.zeros_like(patch)
        out[patch == base] = base
        return patch, out
    raise RuntimeError("failed to build a tricolor object")


def _build_patch(kind: str) -> tuple[np.ndarray, np.ndarray]:
    if kind == "mono":
        return _monochrome_patch()
    if kind == "bi":
        return _bicolor_patch()
    return _tricolor_patch()


def _place_patch(
    grid: np.ndarray,
    expected: np.ndarray,
    occupied: np.ndarray,
    patch: np.ndarray,
    out_patch: np.ndarray,
) -> bool:
    h, w = grid.shape
    ph, pw = patch.shape
    if ph > h or pw > w:
        return False

    occ_coords = {(int(r), int(c)) for r, c in np.argwhere(patch != BG)}
    halo = _halo(occ_coords, radius=1)
    row_lo = 1 if h - ph >= 2 else 0
    col_lo = 1 if w - pw >= 2 else 0
    row_hi = h - ph - 1 if h - ph >= 2 else h - ph
    col_hi = w - pw - 1 if w - pw >= 2 else w - pw

    for _ in range(PLACE_ATTEMPTS):
        r0 = random.randint(row_lo, row_hi)
        c0 = random.randint(col_lo, col_hi)
        if any(
            _in_bounds(grid.shape, r0 + dr, c0 + dc) and bool(occupied[r0 + dr, c0 + dc])
            for dr, dc in halo
        ):
            continue

        patch_mask = patch != BG
        grid_window = grid[r0 : r0 + ph, c0 : c0 + pw]
        grid_window[patch_mask] = patch[patch_mask]

        out_mask = out_patch != BG
        out_window = expected[r0 : r0 + ph, c0 : c0 + pw]
        out_window[out_mask] = out_patch[out_mask]

        for r, c in occ_coords:
            occupied[r0 + r, c0 + c] = True
        return True
    return False


def _scene_branch_kinds(grid: np.ndarray) -> list[str]:
    kinds: list[str] = []
    for coords in _components(grid != BG, connectivity=8):
        colors = np.unique(grid[coords[:, 0], coords[:, 1]])
        if colors.size == 1:
            kinds.append("mono")
        elif colors.size == 2:
            kinds.append("bi")
        else:
            kinds.append("tri")
    return kinds


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(GRID_MIN, GRID_MAX)
    width = random.randint(GRID_MIN, GRID_MAX)
    grid = np.zeros((height, width), dtype=int)
    expected = np.zeros_like(grid)
    occupied = np.zeros_like(grid, dtype=bool)

    n_objects = random.randint(*OBJECT_COUNT_RANGE)
    kinds = ["mono", "bi", "tri"]
    if n_objects > 3:
        kinds.extend(random.choices(("mono", "bi", "tri"), k=n_objects - 3))

    patches = []
    for kind in kinds:
        patch, out_patch = _build_patch(kind)
        patches.append((kind, patch, out_patch, int(np.count_nonzero(patch))))
    patches.sort(key=lambda item: item[3], reverse=True)

    for _kind, patch, out_patch, _area in patches:
        if not _place_patch(grid, expected, occupied, patch, out_patch):
            return None

    actual_kinds = _scene_branch_kinds(grid)
    if len(actual_kinds) != len(kinds):
        return None
    if not {"mono", "bi", "tri"}.issubset(set(actual_kinds)):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
