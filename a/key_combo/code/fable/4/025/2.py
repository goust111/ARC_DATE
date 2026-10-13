from __future__ import annotations

import random

import numpy as np


BG = 0
MIRROR = 5

N = (-1, 0)
S = (1, 0)
W = (0, -1)
E = (0, 1)
DIR4 = (N, S, W, E)

PLUS_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
MODULE_COUNT_CHOICES = (1, 2)
MODULE_HV_RANGE = (2, 7)
GRID_SIZE_MIN = 18
GRID_SIZE_MAX = 30
PLACEMENT_MARGIN = 1
PLACEMENT_ATTEMPTS = 200
GEN_RETRIES = 400


Coord = tuple[int, int]


def _add(cell: Coord, direction: Coord, steps: int = 1) -> Coord:
    return cell[0] + direction[0] * steps, cell[1] + direction[1] * steps


def _in_bounds(shape: tuple[int, int], cell: Coord) -> bool:
    r, c = cell
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _rot90_dir(direction: Coord, k: int) -> Coord:
    dr, dc = direction
    k %= 4
    for _ in range(k):
        dr, dc = dc, -dr
    return dr, dc


def _rotate_patch(patch: np.ndarray, k: int) -> np.ndarray:
    return np.rot90(patch, k)


def _paint_plus(grid: np.ndarray, center: Coord, color: int, missing_dir: Coord) -> None:
    r, c = center
    grid[r, c] = color
    for direction in DIR4:
        if direction == missing_dir:
            continue
        rr, cc = _add(center, direction)
        grid[rr, cc] = color


def _components_of_color(grid: np.ndarray, color: int) -> list[list[Coord]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[Coord]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[Coord] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            comps.append(coords)
    return comps


def _parse_plus(coords: list[Coord]) -> tuple[Coord, Coord]:
    pts = set(coords)
    assert len(pts) == 4, "expected each colored plus component to have exactly 4 cells"

    centers = []
    for cell in pts:
        count = sum(_add(cell, direction) in pts for direction in DIR4)
        if count == 3:
            centers.append(cell)
    assert len(centers) == 1, "expected a unique three-neighbor plus center"

    center = centers[0]
    missing = [direction for direction in DIR4 if _add(center, direction) not in pts]
    assert len(missing) == 1, "expected exactly one missing arm"
    return center, missing[0]


def _find_pluses(grid: np.ndarray) -> list[dict]:
    pluses: list[dict] = []
    for color in PLUS_COLORS:
        comps = _components_of_color(grid, color)
        for coords in comps:
            center, missing_dir = _parse_plus(coords)
            pluses.append(
                {
                    "color": int(color),
                    "center": center,
                    "missing_dir": missing_dir,
                    "slot": _add(center, missing_dir),
                }
            )
    assert pluses, "expected at least one incomplete plus"
    return pluses


def _find_grey_components(grid: np.ndarray) -> list[list[Coord]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[Coord]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != MIRROR:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[Coord] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = rr + dr, cc + dc
                        if not (0 <= nr < h and 0 <= nc < w):
                            continue
                        if seen[nr, nc] or int(grid[nr, nc]) != MIRROR:
                            continue
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _build_turn_map(grid: np.ndarray) -> dict[Coord, dict[Coord, Coord]]:
    turn_map: dict[Coord, dict[Coord, Coord]] = {}
    for comp in _find_grey_components(grid):
        pts = sorted(comp)
        assert len(pts) == 2, "each mirror must be a 2-pixel component"
        (r0, c0), (r1, c1) = pts
        assert abs(r1 - r0) == 1 and abs(c1 - c0) == 1, "mirror pixels must be diagonally adjacent"

        if (r0 + 1, c0 + 1) == (r1, c1):
            # '\' mirror
            entries = {
                (r0, c0 + 1): {W: N, S: E},
                (r1, c0): {E: S, N: W},
            }
        else:
            # '/' mirror
            assert (r0 + 1, c0 - 1) == (r1, c1), "mirror must be '/' or '\\'"
            entries = {
                (r0, c0 - 1): {E: N, S: W},
                (r1, c0): {W: S, N: E},
            }

        for cell, mapping in entries.items():
            turn_map.setdefault(cell, {})
            for incoming, outgoing in mapping.items():
                assert incoming not in turn_map[cell], "turn cell must not have conflicting mirrors"
                turn_map[cell][incoming] = outgoing
    return turn_map


def _find_launch_start(grid: np.ndarray, center: Coord, missing_dir: Coord) -> Coord:
    slot = _add(center, missing_dir)
    current = slot
    seen_support = False

    while True:
        current = _add(current, missing_dir)
        if not _in_bounds(grid.shape, current):
            return _add(slot, missing_dir)
        value = int(grid[current])
        if value == BG:
            if seen_support:
                return current
        else:
            seen_support = True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each missing-arm plus by tracing same-colored beams through diagonal mirror pairs.

    Input structure:
    - Black background contains several monochrome 4-cell plus shapes, each missing exactly one arm.
    - Grey foreground appears only as 2-pixel diagonal mirror pairs shaped like "/" or "\\".
    - The empty corridor outside every missing arm uniquely determines one nonintersecting beam route.

    Rule:
    - For each plus, read the missing-arm direction and emit a beam outward along that corridor.
    - When the beam reaches an empty corner cell adjacent to a grey diagonal pair, reflect it by 90 degrees according to that mirror's slash orientation.
    - Paint every visited background cell with the source plus color until the next step would enter a non-background cell; that last painted cell is the destination plus's missing arm slot.
    - Keep all original plus pixels and all grey mirror pixels unchanged.

    Key insight:
    - The incomplete arm is both the directional clue and the target docking gap: grey diagonal pairs encode the only legal turns, and the beam stops precisely when its next step would hit the destination plus body.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    pluses = _find_pluses(g)
    turn_map = _build_turn_map(g)
    out = g.copy()

    max_steps = g.size * 8
    for plus in pluses:
        color = int(plus["color"])
        center = plus["center"]
        direction = plus["missing_dir"]
        current = _find_launch_start(g, center, direction)
        assert _in_bounds(g.shape, current) and int(g[current]) == BG, "beam must launch from background"

        seen_states: set[tuple[Coord, Coord]] = set()
        for _ in range(max_steps):
            state = (current, direction)
            assert state not in seen_states, "beam route must not loop"
            seen_states.add(state)

            out[current] = color
            if current in turn_map and direction in turn_map[current]:
                direction = turn_map[current][direction]

            nxt = _add(current, direction)
            if not _in_bounds(g.shape, nxt) or int(g[nxt]) != BG:
                break
            current = nxt
        else:
            raise AssertionError("beam simulation exceeded step budget")

    return out


def _make_module(h: int, v: int, colors: list[int]) -> np.ndarray:
    patch = np.zeros((v + 9, h + 9), dtype=int)

    left = (3, 1)
    top = (1, h + 5)
    right = (v + 5, h + 7)
    bottom = (v + 7, 3)

    _paint_plus(patch, left, colors[0], E)
    _paint_plus(patch, top, colors[1], S)
    _paint_plus(patch, right, colors[2], W)
    _paint_plus(patch, bottom, colors[3], N)

    grey_cells = [
        (2, 3),
        (3, 4),
        (3, h + 6),
        (4, h + 5),
        (v + 5, h + 4),
        (v + 6, h + 5),
        (v + 4, 3),
        (v + 5, 2),
    ]
    for cell in grey_cells:
        patch[cell] = MIRROR

    return patch


def _module_bbox_size(module: np.ndarray) -> tuple[int, int]:
    coords = np.argwhere(module != BG)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r1 - r0 + 1), int(c1 - c0 + 1)


def _try_place_modules(modules: list[np.ndarray], height: int, width: int) -> np.ndarray | None:
    grid = np.zeros((height, width), dtype=int)
    occupied = np.zeros((height, width), dtype=bool)

    for module in modules:
        mh, mw = module.shape
        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            r0 = random.randint(0, height - mh)
            c0 = random.randint(0, width - mw)

            rr0 = max(0, r0 - PLACEMENT_MARGIN)
            cc0 = max(0, c0 - PLACEMENT_MARGIN)
            rr1 = min(height, r0 + mh + PLACEMENT_MARGIN)
            cc1 = min(width, c0 + mw + PLACEMENT_MARGIN)
            if occupied[rr0:rr1, cc0:cc1].any():
                continue

            mask = module != BG
            patch = grid[r0 : r0 + mh, c0 : c0 + mw]
            if np.any(mask & (patch != BG)):
                continue
            patch[mask] = module[mask]
            occupied[r0 : r0 + mh, c0 : c0 + mw] |= mask
            placed = True
            break
        if not placed:
            return None

    return grid


def _gen_attempt() -> np.ndarray | None:
    module_count = random.choice(MODULE_COUNT_CHOICES)
    palette = random.sample(PLUS_COLORS, 4 * module_count)

    modules: list[np.ndarray] = []
    for idx in range(module_count):
        h = random.randint(*MODULE_HV_RANGE)
        v = random.randint(*MODULE_HV_RANGE)
        module = _make_module(h, v, palette[4 * idx : 4 * idx + 4])
        module = _rotate_patch(module, random.randrange(4))
        modules.append(module)

    max_h = max(module.shape[0] for module in modules)
    max_w = max(module.shape[1] for module in modules)
    total_area = sum(int(module.shape[0] * module.shape[1]) for module in modules)

    min_side = max(GRID_SIZE_MIN, max(max_h, max_w), int(np.ceil(total_area ** 0.5)) + 4)
    if min_side > GRID_SIZE_MAX:
        return None

    height = random.randint(min_side, GRID_SIZE_MAX)
    width = random.randint(min_side, GRID_SIZE_MAX)
    grid = _try_place_modules(modules, height, width)
    if grid is None:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    if np.count_nonzero(out != grid) < 12 * module_count:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirror-plus input")
