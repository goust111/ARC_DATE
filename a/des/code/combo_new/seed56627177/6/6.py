import random
from collections import Counter, deque

import numpy as np


_DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
_ROTATE_TO_UP = {"up": 0, "right": 1, "down": 2, "left": 3}
_SLOT_OFFSET = {"up": (0, 0), "right": (0, 1), "down": (1, 1), "left": (1, 0)}
_GEN_RETRIES = 400
_GEN_SIDES = (3, 5)


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            pts = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for dr, dc in _DIRS4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _perimeter_coords(h: int, w: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    for c in range(w):
        coords.append((0, c))
        if h > 1:
            coords.append((h - 1, c))
    for r in range(1, h - 1):
        coords.append((r, 0))
        if w > 1:
            coords.append((r, w - 1))
    return coords


def _window_direction(local_r: int, local_c: int, h: int, w: int) -> str | None:
    if local_r == 0 and 0 < local_c < w - 1:
        return "up"
    if local_r == h - 1 and 0 < local_c < w - 1:
        return "down"
    if local_c == 0 and 0 < local_r < h - 1:
        return "left"
    if local_c == w - 1 and 0 < local_r < h - 1:
        return "right"
    return None


def _extract_regions(grid: np.ndarray) -> tuple[int, list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    regions: list[dict[str, object]] = []
    for pts in _components(g != bg):
        r0, c0, r1, c1 = _bbox(pts)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        if np.any(patch == bg):
            continue
        ph, pw = patch.shape
        perimeter = _perimeter_coords(ph, pw)
        perimeter_vals = [int(patch[rr, cc]) for rr, cc in perimeter]
        border_color, _ = Counter(perimeter_vals).most_common(1)[0]
        marker_cells = [(rr, cc) for rr, cc in perimeter if int(patch[rr, cc]) != border_color]
        if len(marker_cells) != 1:
            continue
        mr, mc = marker_cells[0]
        direction = _window_direction(mr, mc, ph, pw)
        if direction is None:
            continue
        inner = patch[1:-1, 1:-1]
        if inner.size == 0:
            continue
        upright = np.rot90(inner, _ROTATE_TO_UP[direction])
        regions.append(
            {
                "direction": direction,
                "inner": upright.copy(),
                "bbox": (r0, c0, r1, c1),
            }
        )
    return bg, regions


def _assemble_output(bg: int, by_direction: dict[str, np.ndarray]) -> np.ndarray:
    slot_h = max((patch.shape[0] for patch in by_direction.values()), default=0)
    slot_w = max((patch.shape[1] for patch in by_direction.values()), default=0)
    out = np.full((2 * slot_h, 2 * slot_w), bg, dtype=int)
    for direction, patch in by_direction.items():
        slot_r, slot_c = _SLOT_OFFSET[direction]
        rr0 = slot_r * slot_h
        cc0 = slot_c * slot_w
        ph, pw = patch.shape
        out[rr0 : rr0 + ph, cc0 : cc0 + pw] = patch
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract window-marked framed regions, rotate each interior to a shared upright orientation, and place them into direction-coded quadrants.

    Input structure:
    - The background is the most common color.
    - Several separated solid rectangles appear on the background; each rectangle is a framed region whose perimeter is mostly one border color.
    - Exactly one perimeter cell of each framed region is replaced by a special window color on the top, right, bottom, or left side.
    - The interior of each framed region is the meaningful content; borders and window markers are only control structure.

    Rule:
    - Find every non-background rectangle whose perimeter has a single off-border window cell.
    - The window side determines both the destination quadrant and the rotation needed to make that window point upward.
    - Remove the border and window, rotate the interior patch to the upright orientation, and place it into the 2x2 output layout: up -> top-left, right -> top-right, down -> bottom-right, left -> bottom-left.
    - Any missing quadrant stays filled with the background color.

    Key insight: The lone perimeter window is a coupled control signal: its side simultaneously selects the output position and the inverse rotation required to normalize the interior.
    """
    bg, regions = _extract_regions(np.asarray(grid, dtype=int))
    assert regions, "expected at least one window-marked region"
    by_direction: dict[str, np.ndarray] = {}
    for region in regions:
        direction = str(region["direction"])
        if direction not in by_direction:
            by_direction[direction] = np.asarray(region["inner"], dtype=int)
    return _assemble_output(bg, by_direction)


def _random_patch(rng: random.Random, side: int, colors: list[int]) -> np.ndarray:
    for _ in range(200):
        chosen = rng.sample(colors, k=min(len(colors), rng.randint(3, min(5, len(colors)))))
        base = chosen[0]
        accents = chosen[1:]
        patch = np.full((side, side), base, dtype=int)

        row = rng.randrange(side)
        row_start = rng.randrange(side - 1)
        row_len = rng.randint(1, side - row_start)
        patch[row, row_start : row_start + row_len] = accents[0]

        col = rng.randrange(side)
        col_start = rng.randrange(side - 1)
        col_len = rng.randint(1, side - col_start)
        patch[col_start : col_start + col_len, col] = accents[min(1, len(accents) - 1)]

        if rng.random() < 0.8:
            color = accents[min(2, len(accents) - 1)]
            r0 = rng.choice((0, side - 2))
            c0 = rng.choice((0, side - 2))
            patch[r0 : r0 + 2, c0 : c0 + 2] = color

        patch[rng.randrange(side), rng.randrange(side)] = accents[-1]
        if len(np.unique(patch)) < 2:
            continue
        if any(np.array_equal(patch, np.rot90(patch, k)) for k in (1, 2, 3)):
            continue
        return np.rot90(patch, rng.randrange(4))
    raise RuntimeError("failed to build asymmetric interior patch")


def _make_region(inner_upright: np.ndarray, border: int, window: int, direction: str) -> np.ndarray:
    side_h, side_w = inner_upright.shape
    inverse_k = (-_ROTATE_TO_UP[direction]) % 4
    inner_input = np.rot90(inner_upright, inverse_k)
    region = np.full((side_h + 2, side_w + 2), border, dtype=int)
    region[1:-1, 1:-1] = inner_input
    mid_r = side_h // 2 + 1
    mid_c = side_w // 2 + 1
    if direction == "up":
        region[0, mid_c] = window
    elif direction == "right":
        region[mid_r, -1] = window
    elif direction == "down":
        region[-1, mid_c] = window
    else:
        region[mid_r, 0] = window
    return region


def _sample_grid_shape(rng: random.Random, region_count: int, outer_side: int) -> tuple[int, int] | None:
    border_cells = region_count * (4 * (outer_side - 2) + 4)
    full_cells = region_count * (outer_side * outer_side)
    min_area = full_cells + border_cells + 24
    min_dim = max(outer_side + 6, int(np.ceil(np.sqrt(min_area))))
    if min_dim > 30:
        return None
    h = rng.randint(min_dim, 30)
    w_min = max(min_dim, int(np.ceil(min_area / h)))
    if w_min > 30:
        return None
    w = rng.randint(w_min, 30)
    if rng.random() < 0.5:
        return h, w
    return w, h


def _all_free_positions(occupied: np.ndarray, rh: int, rw: int, pad: int) -> list[tuple[int, int]]:
    h, w = occupied.shape
    out: list[tuple[int, int]] = []
    for r0 in range(h - rh + 1):
        for c0 in range(w - rw + 1):
            rr0 = max(0, r0 - pad)
            cc0 = max(0, c0 - pad)
            rr1 = min(h, r0 + rh + pad)
            cc1 = min(w, c0 + rw + pad)
            if not occupied[rr0:rr1, cc0:cc1].any():
                out.append((r0, c0))
    return out


def _expected_output(bg: int, side: int, upright_by_direction: dict[str, np.ndarray]) -> np.ndarray:
    out = np.full((2 * side, 2 * side), bg, dtype=int)
    for direction, patch in upright_by_direction.items():
        slot_r, slot_c = _SLOT_OFFSET[direction]
        rr0 = slot_r * side
        cc0 = slot_c * side
        out[rr0 : rr0 + side, cc0 : cc0 + side] = patch
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    side = rng.choice(_GEN_SIDES)
    directions = rng.sample(("up", "right", "down", "left"), k=rng.randint(1, 4))
    shape = _sample_grid_shape(rng, len(directions), side + 2)
    if shape is None:
        return None
    grid_h, grid_w = shape

    colors = list(range(10))
    bg = rng.choice(colors)
    colors.remove(bg)
    border = rng.choice(colors)
    colors.remove(border)
    window = rng.choice(colors)
    colors.remove(window)
    palette = rng.sample(colors, k=min(len(colors), rng.randint(4, 6)))

    upright_by_direction: dict[str, np.ndarray] = {}
    used_signatures: set[bytes] = set()
    for direction in directions:
        for _ in range(100):
            patch = _random_patch(rng, side, palette)
            sig = patch.tobytes()
            if sig in used_signatures:
                continue
            used_signatures.add(sig)
            upright_by_direction[direction] = patch
            break
        else:
            return None

    region_arrays = {direction: _make_region(patch, border, window, direction) for direction, patch in upright_by_direction.items()}

    grid = np.full((grid_h, grid_w), bg, dtype=int)
    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    pad = rng.randint(1, 3)
    order = directions[:]
    rng.shuffle(order)
    for direction in order:
        region = region_arrays[direction]
        rh, rw = region.shape
        positions = _all_free_positions(occupied, rh, rw, pad)
        if not positions:
            return None
        r0, c0 = rng.choice(positions)
        grid[r0 : r0 + rh, c0 : c0 + rw] = region
        rr0 = max(0, r0 - pad)
        cc0 = max(0, c0 - pad)
        rr1 = min(grid_h, r0 + rh + pad)
        cc1 = min(grid_w, c0 + rw + pad)
        occupied[rr0:rr1, cc0:cc1] = True

    expected = _expected_output(bg, side, upright_by_direction)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted retries")
