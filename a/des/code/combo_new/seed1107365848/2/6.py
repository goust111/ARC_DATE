import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5

# Canonical templates: depth grows from row 0 (near the pot) to row 1 (farther away).
TEMPLATES = {
    1: np.array([[1, 0, 1], [0, 1, 0]], dtype=bool),
    2: np.array([[0, 1, 0], [1, 1, 1]], dtype=bool),
    3: np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
}

# Generator configuration.
GRID_H_RANGE = (18, 30)
GRID_W_RANGE = (18, 30)
POT_COUNT_RANGE = (2, 4)
POT_WIDTHS = (3, 5, 7)
GAP_RANGE = (1, 5)
GEN_RETRIES = 400
PLACEMENT_RETRIES = 300
DIRS = ("up", "down", "left", "right")
LEGEND_CELL = 3


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in points]
    cs = [c for _, c in points]
    return min(rs), min(cs), max(rs), max(cs)


def _extract_legend(grid: np.ndarray) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    grey_components = _components(grid == GREY)
    h, w = grid.shape
    corners = {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}
    candidates = [comp for comp in grey_components if any(pt in corners for pt in comp)]
    assert len(candidates) == 1, "expected exactly one corner legend"

    r0, c0, r1, c1 = _bbox(candidates[0])
    region = grid[r0 : r1 + 1, c0 : c1 + 1]
    row_lines = [rr for rr in range(region.shape[0]) if np.all(region[rr, :] == GREY)]
    col_lines = [cc for cc in range(region.shape[1]) if np.all(region[:, cc] == GREY)]
    assert len(row_lines) >= 2 and len(col_lines) >= 2, "legend must have a grey frame"

    row_spans = [
        (row_lines[i] + 1, row_lines[i + 1])
        for i in range(len(row_lines) - 1)
        if row_lines[i + 1] - row_lines[i] > 1
    ]
    col_spans = [
        (col_lines[i] + 1, col_lines[i + 1])
        for i in range(len(col_lines) - 1)
        if col_lines[i + 1] - col_lines[i] > 1
    ]

    mapping: dict[int, np.ndarray] = {}
    for rs, re in row_spans:
        for cs, ce in col_spans:
            cell = region[rs:re, cs:ce]
            colors = set(map(int, np.unique(cell))) - {BLACK, GREY}
            if not colors:
                continue
            assert len(colors) == 1, "each legend cell should define a single color"
            color = colors.pop()
            pts = np.argwhere(cell == color)
            rr0, cc0 = pts.min(axis=0)
            rr1, cc1 = pts.max(axis=0)
            trimmed = (cell[rr0 : rr1 + 1, cc0 : cc1 + 1] == color)
            if color in mapping:
                assert np.array_equal(mapping[color], trimmed)
            mapping[color] = trimmed

    legend_mask = np.zeros_like(grid, dtype=bool)
    legend_mask[r0 : r1 + 1, c0 : c1 + 1] = True
    assert mapping, "legend mapping must be non-empty"
    return legend_mask, mapping


def _find_pots(grid: np.ndarray, legend_mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    pot_mask = (grid == GREY) & (~legend_mask)
    pots: list[tuple[int, int, int, int]] = []
    for comp in _components(pot_mask):
        r0, c0, r1, c1 = _bbox(comp)
        sub = pot_mask[r0 : r1 + 1, c0 : c1 + 1]
        assert sub.all(), "pots must be solid grey rectangles"
        assert r1 - r0 + 1 == 2, "pots must be exactly two rows tall"
        pots.append((r0, c0, r1, c1))
    return pots


def _find_seeds(grid: np.ndarray, legend_mask: np.ndarray) -> list[tuple[int, int, int]]:
    seed_mask = (grid != BLACK) & (grid != GREY) & (~legend_mask)
    seeds: list[tuple[int, int, int]] = []
    for comp in _components(seed_mask):
        assert len(comp) == 1, "seeds must be isolated single pixels"
        r, c = comp[0]
        seeds.append((r, c, int(grid[r, c])))
    return seeds


def _pair_seed(
    grid: np.ndarray, seed: tuple[int, int, int], pots: list[tuple[int, int, int, int]]
) -> tuple[int, str, int]:
    sr, sc, _ = seed
    matches: list[tuple[int, str, int]] = []
    for idx, (r0, c0, r1, c1) in enumerate(pots):
        if c0 <= sc <= c1:
            if sr < r0 and np.all(grid[sr + 1 : r0, sc] == BLACK):
                matches.append((r0 - sr - 1, "up", idx))
            if sr > r1 and np.all(grid[r1 + 1 : sr, sc] == BLACK):
                matches.append((sr - r1 - 1, "down", idx))
        if r0 <= sr <= r1:
            if sc < c0 and np.all(grid[sr, sc + 1 : c0] == BLACK):
                matches.append((c0 - sc - 1, "left", idx))
            if sc > c1 and np.all(grid[sr, c1 + 1 : sc] == BLACK):
                matches.append((sc - c1 - 1, "right", idx))
    assert matches, "each seed must align with at least one pot"
    matches.sort()
    best = matches[0]
    assert len(matches) == 1 or matches[1][0] != best[0], "nearest pot must be unique"
    return best


def _template_points(template: np.ndarray) -> list[tuple[int, int]]:
    h, w = template.shape
    mid = w // 2
    return [
        (r, c - mid)
        for r in range(h)
        for c in range(w)
        if bool(template[r, c])
    ]


def _growth_cells(
    pot: tuple[int, int, int, int], direction: str, gap: int, template: np.ndarray
) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = pot
    center_col = (c0 + c1) // 2
    center_row = (r0 + r1) // 2
    cells: set[tuple[int, int]] = set()
    for layer in range(gap):
        for depth, cross in _template_points(template):
            if direction == "up":
                rr = r0 - 1 - layer - depth
                cc = center_col + cross
            elif direction == "down":
                rr = r1 + 1 + layer + depth
                cc = center_col + cross
            elif direction == "left":
                rr = center_row + cross
                cc = c0 - 1 - layer - depth
            else:
                rr = center_row + cross
                cc = c1 + 1 + layer + depth
            cells.add((rr, cc))
    return cells


def _seed_position(pot: tuple[int, int, int, int], direction: str, gap: int) -> tuple[int, int]:
    r0, c0, r1, c1 = pot
    center_col = (c0 + c1) // 2
    center_row = (r0 + r1) // 2
    if direction == "up":
        return r0 - gap - 1, center_col
    if direction == "down":
        return r1 + gap + 1, center_col
    if direction == "left":
        return center_row, c0 - gap - 1
    return center_row, c1 + gap + 1


def _render_output(grid: np.ndarray) -> np.ndarray:
    legend_mask, mapping = _extract_legend(grid)
    pots = _find_pots(grid, legend_mask)
    seeds = _find_seeds(grid, legend_mask)
    assert len(seeds) == len(pots), "every pot should have exactly one seed"

    out = grid.copy()
    used_pots: set[int] = set()
    for seed in seeds:
        gap, direction, pot_idx = _pair_seed(grid, seed, pots)
        assert pot_idx not in used_pots, "pots must not receive multiple seeds"
        used_pots.add(pot_idx)
        color = seed[2]
        assert color in mapping, "seed color must appear in the legend"
        for rr, cc in _growth_cells(pots[pot_idx], direction, gap, mapping[color]):
            if 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1] and out[rr, cc] != GREY:
                out[rr, cc] = color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow repeated gene-pattern plants from grey pots toward aligned colored seeds.

    Input structure:
    - The background is black.
    - One corner contains a grey framed legend divided into small cells; each cell
      shows one colored template that defines that color's growth pattern.
    - Elsewhere, several solid grey rectangles of height 2 act as pots, and each
      pot has exactly one isolated colored seed pixel aligned directly above,
      below, left, or right with black space in between.

    Rule:
    - Read the corner legend to build a mapping from seed color to template.
    - Pair each seed with the nearest pot lying on the same row or column line,
      with only black cells between them.
    - Let N be the number of black rows or columns between the seed and the pot.
      Starting from the pot edge facing the seed, place N overlapping copies of
      the template, shifting each successive copy by one cell toward the seed.
    - Vertical growth keeps the template horizontally centered on the pot; horizontal
      growth keeps it vertically centered. Pots and the legend remain unchanged.

    Key insight:
    The corner legend is a color-to-template lookup table, while the seed's side
    relative to its pot determines the growth direction and the gap length gives
    the number of repeated layers.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    return _render_output(g)


def _legend_shape(horizontal: bool) -> tuple[int, int]:
    if horizontal:
        return LEGEND_CELL + 2, len(TEMPLATES) * LEGEND_CELL + len(TEMPLATES) + 1
    return len(TEMPLATES) * LEGEND_CELL + len(TEMPLATES) + 1, LEGEND_CELL + 2


def _paint_legend(grid: np.ndarray, rng: random.Random) -> np.ndarray:
    horizontal = rng.choice([True, False])
    corner = rng.choice(["tl", "tr", "bl", "br"])
    h, w = _legend_shape(horizontal)
    gh, gw = grid.shape
    if corner == "tl":
        r0, c0 = 0, 0
    elif corner == "tr":
        r0, c0 = 0, gw - w
    elif corner == "bl":
        r0, c0 = gh - h, 0
    else:
        r0, c0 = gh - h, gw - w

    legend_mask = np.zeros_like(grid, dtype=bool)
    legend_mask[r0 : r0 + h, c0 : c0 + w] = True

    if horizontal:
        rows, cols = 1, len(TEMPLATES)
    else:
        rows, cols = len(TEMPLATES), 1

    grid[r0 : r0 + h, c0 : c0 + w] = BLACK
    for rr in range(rows + 1):
        gr = r0 + rr * (LEGEND_CELL + 1)
        grid[gr, c0 : c0 + w] = GREY
    for cc in range(cols + 1):
        gc = c0 + cc * (LEGEND_CELL + 1)
        grid[r0 : r0 + h, gc] = GREY

    colors = list(TEMPLATES.keys())
    rng.shuffle(colors)
    idx = 0
    for rr in range(rows):
        for cc in range(cols):
            color = colors[idx]
            idx += 1
            cell_r = r0 + 1 + rr * (LEGEND_CELL + 1)
            cell_c = c0 + 1 + cc * (LEGEND_CELL + 1)
            template = TEMPLATES[color]
            for tr in range(template.shape[0]):
                for tc in range(template.shape[1]):
                    if template[tr, tc]:
                        grid[cell_r + tr, cell_c + tc] = color
    return legend_mask


def _expanded(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    out.add((nr, nc))
    return out


def _intended_specs_match(grid: np.ndarray, specs: list[tuple[tuple[int, int, int, int], str, int, int]]) -> bool:
    try:
        legend_mask, mapping = _extract_legend(grid)
        if set(mapping) != set(TEMPLATES):
            return False
        pots = _find_pots(grid, legend_mask)
        seeds = _find_seeds(grid, legend_mask)
        if len(pots) != len(specs) or len(seeds) != len(specs):
            return False

        expected = {}
        for pot, direction, gap, color in specs:
            expected[_seed_position(pot, direction, gap)] = (pot, direction, gap, color)

        for sr, sc, color in seeds:
            if (sr, sc) not in expected:
                return False
            pot, direction, gap, exp_color = expected[(sr, sc)]
            if color != exp_color:
                return False
            got_gap, got_direction, pot_idx = _pair_seed(grid, (sr, sc, color), pots)
            if pots[pot_idx] != pot or got_gap != gap or got_direction != direction:
                return False
        return True
    except AssertionError:
        return False


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    grid = np.full((h, w), BLACK, dtype=int)
    legend_mask = _paint_legend(grid, rng)
    reserved = set(zip(*np.where(legend_mask)))

    pot_count = rng.randint(*POT_COUNT_RANGE)
    specs: list[tuple[tuple[int, int, int, int], str, int, int]] = []
    direction_pool = list(DIRS)
    rng.shuffle(direction_pool)

    for i in range(pot_count):
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            direction = direction_pool[i % len(direction_pool)] if i < len(DIRS) else rng.choice(DIRS)
            if i >= len(DIRS):
                direction = rng.choice(DIRS)
            color = rng.choice(list(TEMPLATES.keys()))
            width = rng.choice(POT_WIDTHS)
            gap = rng.randint(*GAP_RANGE)
            r0 = rng.randint(1, h - 3)
            c0 = rng.randint(1, w - width - 1)
            pot = (r0, c0, r0 + 1, c0 + width - 1)

            seed = _seed_position(pot, direction, gap)
            plant = _growth_cells(pot, direction, gap, TEMPLATES[color])
            all_cells = set(plant)
            all_cells.add(seed)
            for rr in range(r0, r0 + 2):
                for cc in range(c0, c0 + width):
                    all_cells.add((rr, cc))

            if not all(0 <= rr < h and 0 <= cc < w for rr, cc in all_cells):
                continue
            if any((rr, cc) in reserved for rr, cc in all_cells):
                continue

            grown = _expanded(all_cells, h, w)
            if any(cell in reserved for cell in grown):
                continue

            # Reject placements that align a seed with a different pot or legend lane.
            trial = grid.copy()
            for rr in range(r0, r0 + 2):
                trial[rr, c0 : c0 + width] = GREY
            trial[seed] = color
            trial_specs = specs + [(pot, direction, gap, color)]
            if not _intended_specs_match(trial, trial_specs):
                continue

            grid = trial
            reserved |= grown
            specs.append((pot, direction, gap, color))
            placed = True
            break
        if not placed:
            return None

    if not _intended_specs_match(grid, specs):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        rng = random.Random()
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a valid input")
