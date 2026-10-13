from collections import deque
import random

import numpy as np


BLACK = 0
GREY = 5
SEED_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]


def _neighbors(r: int, c: int, h: int, w: int, dirs: list[tuple[int, int]]):
    for dr, dc in dirs:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _components(mask: np.ndarray, connectivity: int = 4) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    dirs = DIR4 if connectivity == 4 else DIR8
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors(cr, cc, h, w, dirs):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(comp)
    return comps


def _connected_from(
    mask: np.ndarray,
    starts: set[tuple[int, int]],
    connectivity: int = 4,
) -> list[tuple[int, int]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    dirs = DIR4 if connectivity == 4 else DIR8
    queue = deque()
    for r, c in starts:
        if mask[r, c] and not seen[r, c]:
            seen[r, c] = True
            queue.append((r, c))

    comp: list[tuple[int, int]] = []
    while queue:
        cr, cc = queue.popleft()
        comp.append((cr, cc))
        for nr, nc in _neighbors(cr, cc, h, w, dirs):
            if mask[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                queue.append((nr, nc))
    return comp


def _touches_border(cells: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in cells)


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    assert rows.size and cols.size
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _dilate8(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    padded = np.pad(mask, 1, constant_values=False)
    out = np.zeros((h, w), dtype=bool)
    for dr in range(3):
        for dc in range(3):
            out |= padded[dr : dr + h, dc : dc + w]
    return out


def _make_rectangle_mask(rng: random.Random) -> np.ndarray:
    height = rng.randint(2, 6)
    width = rng.randint(2, 6)
    return np.ones((height, width), dtype=bool)


def _make_diamond_mask(rng: random.Random) -> np.ndarray:
    radius = rng.randint(1, 3)
    size = 2 * radius + 1
    mask = np.zeros((size, size), dtype=bool)
    center = radius
    for r in range(size):
        for c in range(size):
            if abs(r - center) + abs(c - center) <= radius:
                mask[r, c] = True
    return mask


def _make_blob_mask(rng: random.Random) -> np.ndarray:
    box_h = rng.randint(3, 7)
    box_w = rng.randint(3, 7)
    target = rng.randint(max(4, (box_h + box_w) // 2), min(box_h * box_w, 12))

    start = (rng.randint(0, box_h - 1), rng.randint(0, box_w - 1))
    cells = {start}
    frontier = [start]

    while frontier and len(cells) < target:
        cr, cc = frontier[rng.randrange(len(frontier))]
        frontier.remove((cr, cc))
        nbrs = []
        for nr, nc in _neighbors(cr, cc, box_h, box_w, DIR4):
            if (nr, nc) not in cells:
                nbrs.append((nr, nc))
        rng.shuffle(nbrs)
        for nr, nc in nbrs[: rng.randint(1, len(nbrs)) if nbrs else 0]:
            if len(cells) >= target:
                break
            if (nr, nc) not in cells:
                cells.add((nr, nc))
                frontier.append((nr, nc))

    mask = np.zeros((box_h, box_w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return _trim_mask(mask)


def _build_inner_mask(rng: random.Random) -> np.ndarray:
    builder = rng.choice((_make_rectangle_mask, _make_diamond_mask, _make_blob_mask))
    return builder(rng)


def _make_region_patch(rng: random.Random, seed_color: int) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(40):
        inner = _build_inner_mask(rng)
        inner_patch = np.zeros((inner.shape[0] + 2, inner.shape[1] + 2), dtype=bool)
        inner_patch[1:-1, 1:-1] = inner
        shell = _dilate8(inner_patch) & ~inner_patch
        if len(_components(shell, connectivity=4)) != 1:
            continue

        patch_in = np.full(inner_patch.shape, BLACK, dtype=int)
        patch_out = np.full(inner_patch.shape, BLACK, dtype=int)
        patch_in[shell] = GREY
        patch_out[shell] = seed_color

        seed_cells = np.argwhere(inner_patch)
        seed_r, seed_c = map(int, seed_cells[rng.randrange(len(seed_cells))])
        patch_in[seed_r, seed_c] = seed_color
        patch_out[seed_r, seed_c] = GREY
        return patch_in, patch_out
    raise RuntimeError("failed to build a valid region patch")


def _paint_patch(canvas: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    view = canvas[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]]
    mask = patch != BLACK
    view[mask] = patch[mask]


def _can_place(occupied: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> bool:
    h, w = occupied.shape
    ph, pw = patch.shape
    rr0 = max(0, r0 - 1)
    rr1 = min(h, r0 + ph + 1)
    cc0 = max(0, c0 - 1)
    cc1 = min(w, c0 + pw + 1)
    return not occupied[rr0:rr1, cc0:cc1].any()


def _reserve(occupied: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = occupied.shape
    ph, pw = patch.shape
    rr0 = max(0, r0 - 1)
    rr1 = min(h, r0 + ph + 1)
    cc0 = max(0, c0 - 1)
    cc1 = min(w, c0 + pw + 1)
    occupied[rr0:rr1, cc0:cc1] = True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap each enclosed region's seed color with its grey boundary.

    Input structure:
    - The background is black.
    - One or more closed grey loops each surround a separate interior region.
    - Inside every loop, exactly one non-black, non-grey seed pixel appears, and all
      other interior pixels are black.

    Rule:
    - For each enclosed region, find its unique colored seed pixel.
    - Recolor the entire surrounding grey boundary loop to the seed's color.
    - Recolor the seed pixel itself to grey, leaving all other pixels unchanged.

    Key insight:
    - The transformation is a local color exchange: each seed identifies its own
      enclosing loop, and the seed color and the loop's grey color swap roles.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape

    grey_mask = g == GREY
    non_grey = ~grey_mask
    planned: dict[tuple[int, int], int] = {}
    interior_count = 0

    for comp in _components(non_grey, connectivity=4):
        if _touches_border(comp, h, w):
            continue

        seed_cells = [(r, c) for r, c in comp if int(g[r, c]) not in (BLACK, GREY)]
        assert len(seed_cells) == 1, "each enclosed region must contain exactly one seed pixel"
        seed_r, seed_c = seed_cells[0]
        seed_color = int(g[seed_r, seed_c])

        boundary_starts: set[tuple[int, int]] = set()
        for r, c in comp:
            for nr, nc in _neighbors(r, c, h, w, DIR8):
                if grey_mask[nr, nc]:
                    boundary_starts.add((nr, nc))

        assert boundary_starts, "each enclosed region must touch a grey boundary"
        boundary = _connected_from(grey_mask, boundary_starts, connectivity=4)
        assert boundary, "failed to recover the enclosing boundary"

        interior_count += 1
        for br, bc in boundary:
            previous = planned.get((br, bc))
            assert previous is None or previous == seed_color, "conflicting boundary recolors"
            planned[(br, bc)] = seed_color

        previous = planned.get((seed_r, seed_c))
        assert previous is None or previous == GREY, "conflicting seed recolors"
        planned[(seed_r, seed_c)] = GREY

    assert interior_count > 0, "expected at least one enclosed seed region"

    out = g.copy()
    for (r, c), color in planned.items():
        out[r, c] = color
    return out


GRID_H_RANGE = (12, 30)
GRID_W_RANGE = (12, 30)
MAX_REGIONS = 5
PLACEMENT_TRIES = 120
GENERATION_ATTEMPTS = 400


def _expected_region_count(h: int, w: int, rng: random.Random) -> int:
    area_cap = max(1, min(MAX_REGIONS, (h * w) // 60))
    return rng.randint(1, min(area_cap, len(SEED_COLORS)))


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    grid = np.full((h, w), BLACK, dtype=int)
    expected = np.full((h, w), BLACK, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    region_count = _expected_region_count(h, w, rng)
    seed_colors = rng.sample(SEED_COLORS, region_count)

    for seed_color in seed_colors:
        patch_in, patch_out = _make_region_patch(rng, seed_color)
        ph, pw = patch_in.shape
        if ph + 2 > h or pw + 2 > w:
            return None

        placed = False
        for _ in range(PLACEMENT_TRIES):
            r0 = rng.randint(1, h - ph - 1)
            c0 = rng.randint(1, w - pw - 1)
            if not _can_place(occupied, patch_in, r0, c0):
                continue
            _paint_patch(grid, patch_in, r0, c0)
            _paint_patch(expected, patch_out, r0, c0)
            _reserve(occupied, patch_in, r0, c0)
            placed = True
            break
        if not placed:
            return None

    try:
        actual = main(grid)
    except Exception:
        return None
    if np.array_equal(actual, expected) and not np.array_equal(grid, expected):
        return grid
    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GENERATION_ATTEMPTS} attempts")
