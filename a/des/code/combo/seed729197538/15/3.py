from collections import deque
import random

import numpy as np


BLACK = 0
WALL = 5
LEGEND_SIZE = 3
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)

GRID_H_MIN, GRID_H_MAX = 17, 25
GRID_W_MIN, GRID_W_MAX = 20, 30
ROOM_COUNT_MIN, ROOM_COUNT_MAX = 2, 4
ROOM_PATCH_RETRIES = 120
ROOM_PLACEMENT_RETRIES = 120
GENERATION_RETRIES = 300


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _boundary_reachable(passable: np.ndarray) -> np.ndarray:
    passable = np.asarray(passable, dtype=bool)
    h, w = passable.shape
    seen = np.zeros_like(passable, dtype=bool)
    q = deque()

    for r in range(h):
        for c in (0, w - 1):
            if passable[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if passable[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c, h, w):
            if passable[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
    return seen


def _analyze_grid(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    passable = g != WALL
    outside = _boundary_reachable(passable)
    enclosed = passable & ~outside

    regions = []
    seed_colors = []
    for coords in _components(enclosed):
        vals = g[coords[:, 0], coords[:, 1]]
        seed_idx = np.flatnonzero((vals != BLACK) & (vals != WALL))
        assert len(seed_idx) == 1, "each enclosed region must contain exactly one seed pixel"
        seed_r, seed_c = map(int, coords[seed_idx[0]])
        color = int(g[seed_r, seed_c])
        regions.append(
            {
                "coords": coords,
                "seed": (seed_r, seed_c),
                "color": color,
            }
        )
        seed_colors.append(color)

    assert regions, "expected at least one enclosed region"
    assert len(set(seed_colors)) == len(seed_colors), "seed colors must be unique"

    legend_cells = outside & (g != BLACK) & (g != WALL)
    legend_colors = set(map(int, np.unique(g[legend_cells])))
    assert legend_colors == set(seed_colors), "legend colors must match seed colors"

    templates: dict[int, np.ndarray] = {}
    legend_boxes: dict[int, tuple[int, int]] = {}
    for color in seed_colors:
        coords = np.argwhere(legend_cells & (g == color))
        assert len(coords) > 0, "every seed color must appear in the legend"
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        assert r1 - r0 < LEGEND_SIZE and c1 - c0 < LEGEND_SIZE
        assert r0 + LEGEND_SIZE <= h and c0 + LEGEND_SIZE <= w

        patch = g[r0 : r0 + LEGEND_SIZE, c0 : c0 + LEGEND_SIZE]
        assert patch.shape == (LEGEND_SIZE, LEGEND_SIZE)
        assert int(patch[0, 0]) == color, "legend patch must be keyed by its top-left pixel"
        assert np.all((patch == BLACK) | (patch == color)), "legend patch must be binary in its own color"
        assert np.all(coords[:, 0] < r0 + LEGEND_SIZE) and np.all(coords[:, 1] < c0 + LEGEND_SIZE)

        templates[color] = patch == color
        legend_boxes[color] = (int(r0), int(c0))

    return g, regions, templates, legend_boxes


def main(grid: np.ndarray) -> np.ndarray:
    g, regions, templates, legend_boxes = _analyze_grid(grid)
    out = g.copy()

    for r0, c0 in legend_boxes.values():
        out[r0 : r0 + LEGEND_SIZE, c0 : c0 + LEGEND_SIZE] = BLACK

    for region in regions:
        coords = region["coords"]
        seed_r, seed_c = region["seed"]
        color = region["color"]
        template = templates[color]
        for r, c in coords:
            tr = (int(r) - seed_r) % LEGEND_SIZE
            tc = (int(c) - seed_c) % LEGEND_SIZE
            out[r, c] = color if template[tr, tc] else BLACK

    return out


def _dilate8(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    coords = np.argwhere(mask)
    for r, c in coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = int(r) + dr
                nc = int(c) + dc
                if 0 <= nr < h and 0 <= nc < w:
                    out[nr, nc] = True
    return out


def _translation_invariant(mask: np.ndarray) -> bool:
    for dr in range(LEGEND_SIZE):
        for dc in range(LEGEND_SIZE):
            if dr == 0 and dc == 0:
                continue
            if np.array_equal(mask, np.roll(mask, shift=(dr, dc), axis=(0, 1))):
                return True
    return False


def _random_template(rng: random.Random, used: set[tuple[int, ...]]) -> np.ndarray:
    for _ in range(200):
        density = rng.uniform(0.35, 0.7)
        mask = np.array(
            [[rng.random() < density for _ in range(LEGEND_SIZE)] for _ in range(LEGEND_SIZE)],
            dtype=bool,
        )
        mask[0, 0] = True
        filled = int(mask.sum())
        if filled < 3 or filled > 7:
            continue
        if _translation_invariant(mask):
            continue
        key = tuple(int(v) for v in mask.ravel())
        if key in used:
            continue
        used.add(key)
        return mask
    raise RuntimeError("failed to sample a non-degenerate legend template")


def _random_room_patch(rng: random.Random):
    for _ in range(ROOM_PATCH_RETRIES):
        inner_h = rng.randint(4, 7)
        inner_w = rng.randint(4, 7)
        canvas_h = inner_h + 2
        canvas_w = inner_w + 2
        target_lo = max(9, (inner_h * inner_w) // 3)
        target_hi = min(inner_h * inner_w - 1, (2 * inner_h * inner_w) // 3 + 2)
        if target_lo > target_hi:
            continue
        target_area = rng.randint(target_lo, target_hi)

        start = (rng.randint(1, inner_h), rng.randint(1, inner_w))
        cells = {start}
        frontier = set()
        for nr, nc in _neighbors4(start[0], start[1], canvas_h, canvas_w):
            if 1 <= nr <= inner_h and 1 <= nc <= inner_w:
                frontier.add((nr, nc))

        while frontier and len(cells) < target_area:
            choices = list(frontier)
            weights = []
            for r, c in choices:
                touch = sum((nr, nc) in cells for nr, nc in _neighbors4(r, c, canvas_h, canvas_w))
                weights.append(1 + touch)
            pick = rng.choices(choices, weights=weights, k=1)[0]
            frontier.remove(pick)
            cells.add(pick)
            for nr, nc in _neighbors4(pick[0], pick[1], canvas_h, canvas_w):
                if 1 <= nr <= inner_h and 1 <= nc <= inner_w and (nr, nc) not in cells:
                    frontier.add((nr, nc))

        if len(cells) < 9:
            continue

        interior = np.zeros((canvas_h, canvas_w), dtype=bool)
        for r, c in cells:
            interior[r, c] = True

        if int(interior.any(axis=0).sum()) < 4 or int(interior.any(axis=1).sum()) < 4:
            continue

        wall = _dilate8(interior) & ~interior
        total = wall | interior
        coords = np.argwhere(total)
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        wall = wall[r0 : r1 + 1, c0 : c1 + 1]
        interior = interior[r0 : r1 + 1, c0 : c1 + 1]
        interior_coords = np.argwhere(interior)
        if len(interior_coords) < 9:
            continue

        seed_r, seed_c = map(int, interior_coords[rng.randrange(len(interior_coords))])
        return {"wall": wall, "interior": interior, "seed": (seed_r, seed_c)}

    return None


def _place_rooms(
    rng: random.Random, grid: np.ndarray, rooms: list[dict], legend_top: int
) -> bool:
    h, w = grid.shape
    reserved = np.zeros((h, w), dtype=bool)
    reserved[legend_top - 1 :, :] = True

    for room in sorted(rooms, key=lambda item: item["wall"].shape[0] * item["wall"].shape[1], reverse=True):
        ph, pw = room["wall"].shape
        max_r0 = legend_top - ph - 2
        max_c0 = w - pw - 2
        if max_r0 < 1 or max_c0 < 1:
            return False

        placed = False
        for _ in range(ROOM_PLACEMENT_RETRIES):
            r0 = rng.randint(1, max_r0)
            c0 = rng.randint(1, max_c0)
            rs = r0 - 1
            re = r0 + ph + 1
            cs = c0 - 1
            ce = c0 + pw + 1
            if reserved[rs:re, cs:ce].any():
                continue

            reserved[rs:re, cs:ce] = True
            wall = room["wall"]
            grid[r0 : r0 + ph, c0 : c0 + pw][wall] = WALL
            seed_r, seed_c = room["seed"]
            grid[r0 + seed_r, c0 + seed_c] = room["color"]
            placed = True
            break

        if not placed:
            return False

    return True


def _instance_is_nontrivial(grid: np.ndarray, templates: dict[int, np.ndarray]) -> bool:
    try:
        _, regions, parsed_templates, legend_boxes = _analyze_grid(grid)
        out = main(grid)
    except AssertionError:
        return False

    if set(parsed_templates) != set(templates):
        return False
    for color, template in templates.items():
        if not np.array_equal(parsed_templates[color], template):
            return False

    if len(legend_boxes) != len(templates):
        return False

    changed_regions = 0
    for region in regions:
        coords = region["coords"]
        before = grid[coords[:, 0], coords[:, 1]]
        after = out[coords[:, 0], coords[:, 1]]
        color = region["color"]
        if np.any(before != after):
            changed_regions += 1
        if np.count_nonzero(after == color) < 2:
            return False
        if np.count_nonzero(after == BLACK) < 2:
            return False

    if changed_regions != len(regions):
        return False
    if np.count_nonzero(out != grid) < 10:
        return False
    return True


def _gen_attempt(rng: random.Random):
    n_rooms = rng.randint(ROOM_COUNT_MIN, ROOM_COUNT_MAX)
    colors = rng.sample(COLOR_POOL, n_rooms)

    template_used: set[tuple[int, ...]] = set()
    templates = {color: _random_template(rng, template_used) for color in colors}

    legend_order = colors[:]
    rng.shuffle(legend_order)
    gaps = [rng.choice((1, 2)) for _ in range(n_rooms - 1)]
    legend_width = LEGEND_SIZE * n_rooms + sum(gaps)

    rooms = []
    for color in colors:
        room = _random_room_patch(rng)
        if room is None:
            return None
        room["color"] = color
        rooms.append(room)

    min_w = max(GRID_W_MIN, legend_width + 2)
    if min_w > GRID_W_MAX:
        return None

    h = rng.randint(GRID_H_MIN, GRID_H_MAX)
    w = rng.randint(min_w, GRID_W_MAX)
    legend_top = h - LEGEND_SIZE
    grid = np.zeros((h, w), dtype=int)

    if not _place_rooms(rng, grid, rooms, legend_top):
        return None

    legend_left = rng.randint(1, w - legend_width - 1)
    col = legend_left
    for i, color in enumerate(legend_order):
        template = templates[color]
        patch = grid[legend_top : legend_top + LEGEND_SIZE, col : col + LEGEND_SIZE]
        patch[template] = color
        if i < len(gaps):
            col += LEGEND_SIZE + gaps[i]

    if not _instance_is_nontrivial(grid, templates):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
