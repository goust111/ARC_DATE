import random
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


BLACK = 0
GREY = 5
ORANGE = 7
LEGEND_HEIGHT = 3
PLAY_TOP = 4
PALETTE_COLORS = (1, 2, 3, 4, 6, 8, 9)
CARDINAL_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> Iterable[Tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> List[List[Tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: List[List[Tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: List[Tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _perpendiculars(direction: Tuple[int, int]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    dr, dc = direction
    if dr == 0:
        return (-1, 0), (1, 0)
    return (0, -1), (0, 1)


def _parse_legend(grid: np.ndarray) -> Dict[int, Tuple[int, int]]:
    band = grid[:LEGEND_HEIGHT, :]
    comps = _components(band != BLACK)
    mapping: Dict[int, Tuple[int, int]] = {}
    for comp in comps:
        assert len(comp) == 2, "each legend group must contain exactly two cells"
        values = [int(band[r, c]) for r, c in comp]
        assert values.count(GREY) == 1, "each legend group must contain exactly one grey cell"
        grey_idx = values.index(GREY)
        color_idx = 1 - grey_idx
        gr, gc = comp[grey_idx]
        cr, cc = comp[color_idx]
        color = int(values[color_idx])
        assert color not in (BLACK, GREY, ORANGE), "legend colors must be distinct from black/grey/orange"
        direction = (gr - cr, gc - cc)
        assert abs(direction[0]) + abs(direction[1]) == 1, "legend pair must be orthogonally adjacent"
        assert color not in mapping, "each legend color must appear once"
        mapping[color] = direction
    assert mapping, "expected at least one legend group"
    return mapping


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each colored seed through the arena using a color-specific asymmetric growth rule.

    The top band is a direction legend. Each group contains one colored pixel and one
    grey pixel; the grey offset from the colored pixel is that color's main growth
    direction. In each synchronous step, a territory grows one cell forward in its main
    direction and one cell in each perpendicular direction, never backward. Orange cells
    are walls. When multiple colors reach the same empty cell in the same step, a unique
    main-direction claim wins; otherwise that cell stays black as a boundary.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h > LEGEND_HEIGHT

    direction_by_color = _parse_legend(g)

    owner = np.zeros((h, w), dtype=int)
    for color in direction_by_color:
        owner[g == color] = color
    owner[:LEGEND_HEIGHT, :] = BLACK

    blocked = np.zeros((h, w), dtype=bool)
    blocked[:LEGEND_HEIGHT, :] = True
    blocked[g == ORANGE] = True

    while True:
        claims: Dict[Tuple[int, int], Dict[int, int]] = {}
        for color, main_dir in direction_by_color.items():
            cells = np.argwhere(owner == color)
            if cells.size == 0:
                continue
            moves = (main_dir,) + _perpendiculars(main_dir)
            for r, c in cells:
                r = int(r)
                c = int(c)
                for move_idx, (dr, dc) in enumerate(moves):
                    nr = r + dr
                    nc = c + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if blocked[nr, nc] or owner[nr, nc] != BLACK:
                        continue
                    state = 2 if move_idx == 0 else 1
                    cell_claims = claims.setdefault((nr, nc), {})
                    cell_claims[color] = cell_claims.get(color, 0) | state

        new_cells: List[Tuple[int, int, int]] = []
        new_boundaries: List[Tuple[int, int]] = []
        for (r, c), cell_claims in claims.items():
            if len(cell_claims) == 1:
                color = next(iter(cell_claims))
                new_cells.append((r, c, color))
                continue
            main_colors = [color for color, state in cell_claims.items() if state & 2]
            if len(main_colors) == 1:
                new_cells.append((r, c, main_colors[0]))
            else:
                new_boundaries.append((r, c))

        if not new_cells and not new_boundaries:
            break

        for r, c in new_boundaries:
            blocked[r, c] = True
        for r, c, color in new_cells:
            owner[r, c] = color

    out = g.copy()
    clear_mask = np.ones((h, w), dtype=bool)
    clear_mask[:LEGEND_HEIGHT, :] = False
    clear_mask &= g != ORANGE
    out[clear_mask] = BLACK
    out[owner != BLACK] = owner[owner != BLACK]
    return out


def _legend_group_width(direction: Tuple[int, int]) -> int:
    return 2 if direction[1] != 0 else 1


def _draw_legend(grid: np.ndarray, colors: Sequence[int], directions: Sequence[Tuple[int, int]], rng: random.Random) -> None:
    widths = [_legend_group_width(direction) for direction in directions]
    min_width = sum(widths) + 2 * (len(colors) - 1)
    slack = grid.shape[1] - min_width
    assert slack >= 0
    col = rng.randint(0, slack) if slack > 0 else 0
    for idx, (color, direction) in enumerate(zip(colors, directions)):
        dr, dc = direction
        if direction == (-1, 0):
            grid[0, col] = GREY
            grid[1, col] = color
        elif direction == (1, 0):
            grid[1, col] = color
            grid[2, col] = GREY
        elif direction == (0, -1):
            grid[1, col] = GREY
            grid[1, col + 1] = color
        else:
            grid[1, col] = color
            grid[1, col + 1] = GREY
        col += _legend_group_width(direction)
        if idx < len(colors) - 1:
            col += rng.randint(2, 4)


def _wall_shape(h: int, w: int, rng: random.Random) -> List[Tuple[int, int]]:
    mode = rng.choice(("h", "v", "L"))
    if mode == "h":
        max_len = min(7, w)
        length = rng.randint(2, max_len)
        r = rng.randint(PLAY_TOP, h - 1)
        c0 = rng.randint(0, w - length)
        return [(r, c0 + dc) for dc in range(length)]

    if mode == "v":
        max_len = min(7, h - PLAY_TOP)
        length = rng.randint(2, max_len)
        r0 = rng.randint(PLAY_TOP, h - length)
        c = rng.randint(0, w - 1)
        return [(r0 + dr, c) for dr in range(length)]

    arm_h = rng.randint(2, min(5, w))
    arm_v = rng.randint(2, min(5, h - PLAY_TOP))
    box_r = rng.randint(PLAY_TOP, h - arm_v)
    box_c = rng.randint(0, w - arm_h)
    corner = rng.choice(("tl", "tr", "bl", "br"))
    cells: set[Tuple[int, int]] = set()
    if corner == "tl":
        cells.update((box_r, box_c + dc) for dc in range(arm_h))
        cells.update((box_r + dr, box_c) for dr in range(arm_v))
    elif corner == "tr":
        cells.update((box_r, box_c + dc) for dc in range(arm_h))
        cells.update((box_r + dr, box_c + arm_h - 1) for dr in range(arm_v))
    elif corner == "bl":
        cells.update((box_r + arm_v - 1, box_c + dc) for dc in range(arm_h))
        cells.update((box_r + dr, box_c) for dr in range(arm_v))
    else:
        cells.update((box_r + arm_v - 1, box_c + dc) for dc in range(arm_h))
        cells.update((box_r + dr, box_c + arm_h - 1) for dr in range(arm_v))
    return sorted(cells)


def _can_place_wall(grid: np.ndarray, cells: Sequence[Tuple[int, int]]) -> bool:
    cell_set = set(cells)
    h, w = grid.shape
    for r, c in cells:
        if grid[r, c] != BLACK:
            return False
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if not (0 <= rr < h and 0 <= cc < w):
                    continue
                if (rr, cc) in cell_set:
                    continue
                if grid[rr, cc] == ORANGE:
                    return False
    return True


def _place_walls(grid: np.ndarray, rng: random.Random) -> int:
    wall_target = rng.choice((0, 1, 2, 3))
    placed = 0
    if grid.shape[0] - PLAY_TOP < 2:
        return 0
    for _ in range(40):
        if placed >= wall_target:
            break
        cells = _wall_shape(*grid.shape, rng)
        if not _can_place_wall(grid, cells):
            continue
        for r, c in cells:
            grid[r, c] = ORANGE
        placed += 1
    return placed


def _seed_ok(grid: np.ndarray, r: int, c: int, direction: Tuple[int, int]) -> bool:
    h, w = grid.shape
    if grid[r, c] != BLACK:
        return False
    for rr in range(r - 1, r + 2):
        for cc in range(c - 1, c + 2):
            if 0 <= rr < h and 0 <= cc < w and grid[rr, cc] != BLACK:
                return False
    moves = (direction,) + _perpendiculars(direction)
    for dr, dc in moves:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == BLACK:
            return True
    return False


def _place_seeds(
    grid: np.ndarray,
    colors: Sequence[int],
    directions: Sequence[Tuple[int, int]],
    rng: random.Random,
) -> Dict[int, List[Tuple[int, int]]]:
    seed_positions: Dict[int, List[Tuple[int, int]]] = {}
    counts = [rng.randint(1, 3) for _ in colors]
    if len(colors) > 1 and max(counts) == 1 and rng.random() < 0.6:
        counts[rng.randrange(len(counts))] = 2

    for color, direction, count in zip(colors, directions, counts):
        seed_positions[color] = []
        for _ in range(count):
            candidates: List[Tuple[int, int]] = []
            for r in range(PLAY_TOP, grid.shape[0]):
                for c in range(grid.shape[1]):
                    if _seed_ok(grid, r, c, direction):
                        candidates.append((r, c))
            if not candidates:
                raise ValueError("no valid seed position")
            r, c = rng.choice(candidates)
            grid[r, c] = color
            seed_positions[color].append((r, c))
    return seed_positions


def _seed_count(grid: np.ndarray, colors: Sequence[int]) -> int:
    total = 0
    for color in colors:
        total += int(np.sum(grid == color))
    return total


def _nonlegend_black_count(grid: np.ndarray) -> int:
    return int(np.sum(grid[LEGEND_HEIGHT:, :] == BLACK))


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    color_count = rng.randint(2, 4)
    colors = rng.sample(PALETTE_COLORS, color_count)
    directions = [rng.choice(CARDINAL_DIRS) for _ in colors]
    if len(colors) > 1 and len(set(directions)) == 1 and rng.random() < 0.8:
        idx = rng.randrange(len(directions))
        alternatives = [direction for direction in CARDINAL_DIRS if direction != directions[idx]]
        directions[idx] = rng.choice(alternatives)

    min_width = sum(_legend_group_width(direction) for direction in directions) + 2 * (len(colors) - 1)
    width_lo = max(12, min_width)
    width_hi = max(width_lo, 20)
    width = rng.randint(width_lo, width_hi)
    height = rng.randint(12, 18)
    if height <= PLAY_TOP + 1:
        return None

    grid = np.zeros((height, width), dtype=int)
    _draw_legend(grid, colors, directions, rng)
    wall_count = _place_walls(grid, rng)
    try:
        _place_seeds(grid, colors, directions, rng)
    except ValueError:
        return None

    output = main(grid)
    input_seed_total = _seed_count(grid, colors)
    output_area_total = _seed_count(output, colors)
    if output_area_total <= input_seed_total + len(colors):
        return None
    if wall_count > 0 and int(np.sum(output == ORANGE)) == 0:
        return None
    if np.array_equal(output, grid):
        return None
    if _nonlegend_black_count(output) == 0 and wall_count == 0:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        try:
            grid = _gen_attempt(rng)
        except Exception:
            continue
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
