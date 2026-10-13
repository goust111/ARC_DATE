from __future__ import annotations

import random
from collections import Counter

import numpy as np


BACKGROUND = 0
FRAME = 5
DIVIDER = 7
PATTERN_POOL = (1, 2, 3, 4, 6, 8, 9)
GEN_MAX_ATTEMPTS = 400

_MINORITY_TEMPLATES = (
    (1, 0, 0, 0),
    (0, 1, 0, 0),
    (0, 0, 1, 0),
    (0, 0, 0, 1),
)
_TIE_TEMPLATES = (
    (1, 1, 0, 0),
    (0, 0, 1, 1),
    (1, 0, 1, 0),
    (0, 1, 0, 1),
    (1, 0, 0, 1),
    (0, 1, 1, 0),
)
_MAJORITY_TEMPLATES = tuple(
    tuple(1 - cell for cell in tpl) for tpl in _MINORITY_TEMPLATES
)
_CATEGORY_TEMPLATES = {
    "minority": _MINORITY_TEMPLATES,
    "tie": _TIE_TEMPLATES,
    "majority": _MAJORITY_TEMPLATES,
}


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _orbit_coords(height: int, width: int, r: int, c: int) -> list[tuple[int, int]]:
    return [
        (r, c),
        (r, width - 1 - c),
        (height - 1 - r, c),
        (height - 1 - r, width - 1 - c),
    ]


def _find_divider_column(grid: np.ndarray) -> int:
    height, width = grid.shape
    candidates = []
    for c in range(width):
        color = int(grid[0, c])
        if np.all(grid[:, c] == color) and int(np.count_nonzero(grid == color)) == height:
            candidates.append(c)
    assert len(candidates) == 1, "expected one unique full-height divider column"
    return candidates[0]


def _parse_legend(side_panel: np.ndarray) -> dict[frozenset[int], int]:
    gray_mask = side_panel == FRAME
    box = _bbox(gray_mask)
    assert box is not None, "expected a gray legend box"
    r0, c0, r1, c1 = box

    expected = np.zeros_like(gray_mask, dtype=bool)
    expected[r0, c0 : c1 + 1] = True
    expected[r1, c0 : c1 + 1] = True
    expected[r0 : r1 + 1, c0] = True
    expected[r0 : r1 + 1, c1] = True
    assert np.array_equal(gray_mask, expected), "gray pixels must form one hollow rectangle"

    mapping: dict[frozenset[int], int] = {}
    colored_rows: list[tuple[int, int]] = []
    for r in range(r0 + 1, r1):
        vals = {
            int(v)
            for v in side_panel[r, c0 + 1 : c1]
            if int(v) not in {BACKGROUND, FRAME}
        }
        if not vals:
            continue
        assert len(vals) == 1, "each legend row should contain one color"
        colored_rows.append((r, next(iter(vals))))

    assert len(colored_rows) >= 4 and len(colored_rows) % 2 == 0, "expected stacked 2-cell legend pairs"

    idx = 0
    while idx < len(colored_rows):
        top_r, top_color = colored_rows[idx]
        bottom_r, bottom_color = colored_rows[idx + 1]
        assert bottom_r == top_r + 1, "legend pair rows must be consecutive"
        key = frozenset({top_color, bottom_color})
        assert len(key) == 2 and key not in mapping, "legend pairs must be distinct two-color relations"
        mapping[key] = top_color
        idx += 2
        if idx < len(colored_rows):
            assert colored_rows[idx][0] >= bottom_r + 2, "legend pairs must be vertically separated"

    return mapping


def _extract_panels(grid: np.ndarray) -> tuple[np.ndarray, dict[frozenset[int], int]]:
    divider_c = _find_divider_column(grid)
    left = grid[:, :divider_c]
    right = grid[:, divider_c + 1 :]
    assert left.size > 0 and right.size > 0, "divider must separate two nonempty panels"

    if left.shape[1] < right.shape[1]:
        side_panel, main_panel = left, right
    else:
        side_panel, main_panel = right, left

    legend = _parse_legend(side_panel)
    assert main_panel.shape[0] % 2 == 0 and main_panel.shape[1] % 2 == 0, "main panel must have even dimensions"
    return main_panel.copy(), legend


def _resolve_main_panel(
    main_panel: np.ndarray, legend: dict[frozenset[int], int]
) -> np.ndarray:
    height, width = main_panel.shape
    out = main_panel.copy()

    for r in range(height // 2):
        for c in range(width // 2):
            orbit = _orbit_coords(height, width, r, c)
            colors = [int(main_panel[rr, cc]) for rr, cc in orbit]
            uniq = set(colors)
            if len(uniq) == 1:
                continue
            assert len(uniq) == 2, "each conflict orbit must contain exactly two colors"
            winner = legend[frozenset(uniq)]
            for rr, cc in orbit:
                out[rr, cc] = winner

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair a biaxially symmetric main panel by resolving each two-color mirror orbit with the side-panel legend.

    Input structure:
    - One full-height divider splits the grid into a large main panel and a narrower side panel.
    - The side panel contains one gray rectangular legend box whose interior stacks several vertical 2-cell color pairs.
    - The main panel should obey horizontal and vertical reflection symmetry, except some 4-cell reflection orbits contain exactly two colors.
    - Every conflicting orbit's unordered color pair appears exactly once in the legend.

    Rule:
    - Ignore the divider and side panel in the output; only the repaired main panel is returned.
    - Partition the main panel into 4-cell orbits under horizontal and vertical reflection.
    - Uniform orbits stay unchanged.
    - For each two-color orbit, find the matching legend pair and repaint the whole orbit with the pair's top color.

    Key insight:
    - The side panel is not extra artwork: the vertical order inside each legend pair encodes which color dominates when a symmetric orbit is disputed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"
    main_panel, legend = _extract_panels(g)
    return _resolve_main_panel(main_panel, legend)


def _choose_pairs(active_colors: list[int], n_pairs: int) -> list[tuple[int, int]]:
    if n_pairs == 2:
        a, b, c = random.sample(active_colors, 3)
        return [(a, b), (c, a)]

    seq = random.sample(active_colors, n_pairs)
    return [(seq[i], seq[(i + 1) % n_pairs]) for i in range(n_pairs)]


def _paint_orbit(panel: np.ndarray, r: int, c: int, color: int) -> None:
    for rr, cc in _orbit_coords(*panel.shape, r, c):
        panel[rr, cc] = color


def _build_symmetric_output(
    height: int,
    width: int,
    active_colors: list[int],
    pairs: list[tuple[int, int]],
    conflict_specs: list[tuple[int, str, tuple[int, ...]]],
) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    reps = [(r, c) for r in range(height // 2) for c in range(width // 2)]
    random.shuffle(reps)

    n_conflicts = len(conflict_specs)
    needed = n_conflicts + len(active_colors)
    assert needed <= len(reps), "not enough orbits for conflicts and color seeding"

    conflict_positions = reps[:n_conflicts]
    seed_positions = reps[n_conflicts : needed]
    free_positions = reps[needed:]

    output = np.empty((height, width), dtype=int)
    for pos, color in zip(seed_positions, active_colors):
        _paint_orbit(output, pos[0], pos[1], color)

    for r, c in free_positions:
        _paint_orbit(output, r, c, random.choice(active_colors))

    for (pair_idx, _category, _template), (r, c) in zip(conflict_specs, conflict_positions):
        winner, _loser = pairs[pair_idx]
        _paint_orbit(output, r, c, winner)

    return output, conflict_positions, seed_positions


def _inject_conflicts(
    output: np.ndarray,
    pairs: list[tuple[int, int]],
    conflict_specs: list[tuple[int, str, tuple[int, ...]]],
    conflict_positions: list[tuple[int, int]],
) -> np.ndarray:
    inp = output.copy()
    for (pair_idx, _category, template), (r, c) in zip(conflict_specs, conflict_positions):
        winner, loser = pairs[pair_idx]
        for (rr, cc), use_winner in zip(_orbit_coords(*output.shape, r, c), template):
            inp[rr, cc] = winner if use_winner else loser
    return inp


def _make_conflict_specs(n_pairs: int, orbit_budget: int) -> list[tuple[int, str, tuple[int, ...]]]:
    n_conflicts = random.randint(max(3, n_pairs), min(max(3, n_pairs) + 3, orbit_budget))
    categories = ["minority", "tie", "majority"][:n_conflicts]
    while len(categories) < n_conflicts:
        categories.append(random.choice(["minority", "tie", "majority"]))
    random.shuffle(categories)

    pair_indices = list(range(n_pairs))
    while len(pair_indices) < n_conflicts:
        pair_indices.append(random.randrange(n_pairs))
    random.shuffle(pair_indices)

    return [
        (pair_idx, category, random.choice(_CATEGORY_TEMPLATES[category]))
        for pair_idx, category in zip(pair_indices, categories)
    ]


def _build_legend_panel(
    height: int, side_width: int, pairs: list[tuple[int, int]]
) -> np.ndarray:
    panel = np.zeros((height, side_width), dtype=int)
    box_width = random.choice([w for w in (5, 7) if w <= side_width - 2] or [5])
    box_height = 3 * len(pairs) + 1
    max_r0 = height - box_height
    r0 = random.randint(0, max_r0)
    c0_choices = [c for c in range(1, side_width - box_width) ] or [1]
    c0 = random.choice(c0_choices)
    r1 = r0 + box_height - 1
    c1 = c0 + box_width - 1

    panel[r0, c0 : c1 + 1] = FRAME
    panel[r1, c0 : c1 + 1] = FRAME
    panel[r0 : r1 + 1, c0] = FRAME
    panel[r0 : r1 + 1, c1] = FRAME

    center_col = c0 + box_width // 2
    for idx, (winner, loser) in enumerate(pairs):
        top_r = r0 + 1 + 3 * idx
        panel[top_r, center_col] = winner
        panel[top_r + 1, center_col] = loser

    return panel


def _compose_input(main_panel: np.ndarray, side_panel: np.ndarray, side_on_left: bool) -> np.ndarray:
    height = main_panel.shape[0]
    divider = np.full((height, 1), DIVIDER, dtype=int)
    if side_on_left:
        return np.hstack([side_panel, divider, main_panel])
    return np.hstack([main_panel, divider, side_panel])


def _solve_with_majority_shortcut(panel: np.ndarray) -> np.ndarray:
    out = panel.copy()
    height, width = panel.shape
    for r in range(height // 2):
        for c in range(width // 2):
            orbit = _orbit_coords(height, width, r, c)
            colors = [int(panel[rr, cc]) for rr, cc in orbit]
            counts = Counter(colors)
            if len(counts) == 1:
                continue
            fill = max(counts.items(), key=lambda item: (item[1], -colors.index(item[0])))[0]
            for rr, cc in orbit:
                out[rr, cc] = fill
    return out


def _solve_with_topleft_shortcut(panel: np.ndarray) -> np.ndarray:
    out = panel.copy()
    height, width = panel.shape
    for r in range(height // 2):
        for c in range(width // 2):
            fill = int(panel[r, c])
            for rr, cc in _orbit_coords(height, width, r, c):
                out[rr, cc] = fill
    return out


def _validate_instance(
    grid: np.ndarray,
    expected_main: np.ndarray,
    pairs: list[tuple[int, int]],
    conflict_specs: list[tuple[int, str, tuple[int, ...]]],
) -> bool:
    try:
        actual = main(grid)
    except Exception:
        return False
    if not np.array_equal(actual, expected_main):
        return False

    if len({winner for winner, _loser in pairs}) < 2:
        return False
    if not any(color in {winner for winner, _ in pairs} and color in {loser for _, loser in pairs} for color in PATTERN_POOL):
        return False
    if not any(category == "minority" for _idx, category, _tpl in conflict_specs):
        return False
    if not any(category == "tie" for _idx, category, _tpl in conflict_specs):
        return False
    if not any(category == "majority" for _idx, category, _tpl in conflict_specs):
        return False
    if not any(template[0] == 1 for _idx, _category, template in conflict_specs):
        return False
    if not any(template[0] == 0 for _idx, _category, template in conflict_specs):
        return False

    divider_c = _find_divider_column(grid)
    full_unique_columns = [
        c
        for c in range(grid.shape[1])
        if np.all(grid[:, c] == grid[0, c]) and int(np.count_nonzero(grid == int(grid[0, c]))) == grid.shape[0]
    ]
    if full_unique_columns != [divider_c]:
        return False

    main_panel, legend = _extract_panels(grid)
    if set(legend.keys()) != {frozenset(pair) for pair in pairs}:
        return False
    if np.array_equal(_solve_with_majority_shortcut(main_panel), expected_main):
        return False
    if np.array_equal(_solve_with_topleft_shortcut(main_panel), expected_main):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    main_height = random.choice((12, 14, 16, 18))
    max_pairs = min(4, (main_height - 1) // 3)
    n_pairs = random.randint(2, max_pairs)

    side_width = random.choice((7, 8, 9))
    main_width_options = [
        w
        for w in (10, 12, 14, 16, 18)
        if w > side_width and w + side_width + 1 <= 30
    ]
    main_width = random.choice(main_width_options)

    n_active = random.randint(max(4, n_pairs), min(6, len(PATTERN_POOL)))
    active_colors = random.sample(list(PATTERN_POOL), n_active)
    pairs = _choose_pairs(active_colors, n_pairs)

    orbit_budget = (main_height // 2) * (main_width // 2) - len(active_colors)
    if orbit_budget < max(3, n_pairs):
        return None
    conflict_specs = _make_conflict_specs(n_pairs, orbit_budget)

    output_main, conflict_positions, _seed_positions = _build_symmetric_output(
        main_height, main_width, active_colors, pairs, conflict_specs
    )
    input_main = _inject_conflicts(output_main, pairs, conflict_specs, conflict_positions)
    side_panel = _build_legend_panel(main_height, side_width, pairs)
    grid = _compose_input(input_main, side_panel, side_on_left=random.choice([False, True]))

    if not _validate_instance(grid, output_main, pairs, conflict_specs):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid symmetry-arbitration input")
