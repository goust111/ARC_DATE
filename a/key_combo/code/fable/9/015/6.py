import hashlib
from collections import Counter, deque

import numpy as np


BACKGROUND = 0
PALETTE = tuple(range(1, 10))
QUADRANT_HEIGHT_RANGE = (5, 12)
QUADRANT_WIDTH_RANGE = (5, 12)
GENERATION_RETRIES = 500

_RNG = np.random.default_rng()
_RECENT_INPUT_SIGNATURES: deque[str] = deque(maxlen=64)
_RECENT_OUTPUT_SIGNATURES: deque[str] = deque(maxlen=64)


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, grid.flat))
    if exclude is not None:
        colors -= {int(c) for c in exclude}
    return sorted(colors)


def _find_separator_cross(grid: np.ndarray, bg: int) -> tuple[int, int, int]:
    h, w = grid.shape
    row_candidates = [
        r for r in range(h) if int(grid[r, 0]) != bg and np.all(grid[r, :] == grid[r, 0])
    ]
    col_candidates = [
        c for c in range(w) if int(grid[0, c]) != bg and np.all(grid[:, c] == grid[0, c])
    ]

    matches: list[tuple[int, int, int]] = []
    for r in row_candidates:
        for c in col_candidates:
            color = int(grid[r, c])
            if color == bg:
                continue
            if int(grid[r, 0]) == color and int(grid[0, c]) == color:
                matches.append((r, c, color))

    assert len(matches) == 1, "expected exactly one solid separator cross"
    r, c, color = matches[0]
    assert r == h - r - 1 and c == w - c - 1, "expected equal-sized quadrants"
    return r, c, color


def _split_quadrants(grid: np.ndarray, cross_row: int, cross_col: int) -> list[np.ndarray]:
    return [
        grid[:cross_row, :cross_col],
        grid[:cross_row, cross_col + 1 :],
        grid[cross_row + 1 :, :cross_col],
        grid[cross_row + 1 :, cross_col + 1 :],
    ]


def _quadrant_slices(cross_row: int, cross_col: int) -> list[tuple[slice, slice]]:
    return [
        (slice(0, cross_row), slice(0, cross_col)),
        (slice(0, cross_row), slice(cross_col + 1, None)),
        (slice(cross_row + 1, None), slice(0, cross_col)),
        (slice(cross_row + 1, None), slice(cross_col + 1, None)),
    ]


def _repair_from_majority(
    quadrants: list[np.ndarray], pattern_color: int, divider_color: int, bg: int
) -> tuple[np.ndarray, list[np.ndarray]]:
    masks = [quadrant == pattern_color for quadrant in quadrants]
    majority_mask = np.sum(masks, axis=0) >= 3

    repaired = []
    for quadrant, mask in zip(quadrants, masks):
        out = quadrant.copy()
        out[~majority_mask & mask] = bg
        out[majority_mask & ~mask] = divider_color
        repaired.append(out)
    return majority_mask, repaired


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair four damaged quadrant copies by strict majority vote across corresponding cells.

    Input structure:
    - One non-black color forms a full solid row and a full solid column whose
      intersection splits the canvas into four equal-sized quadrants.
    - Aside from the divider cross, the only other non-black color is a single
      pattern color shared by all four quadrants.
    - Each quadrant is a damaged copy of the same pattern on a black background:
      some true pattern cells are missing, and some background cells are polluted
      with extra pattern-color pixels.

    Rule:
    - Align the four quadrants by corresponding local coordinates.
    - A local cell belongs to the reconstructed pattern iff at least three of the
      four quadrants show the pattern color there; otherwise it is background.
    - For reconstructed pattern cells, any quadrant that is missing the pattern
      gets the divider color at that cell. For non-pattern cells, any quadrant's
      extra pattern-color pixel is erased to black. The divider cross stays fixed.

    Key insight:
    - No quadrant is a clean template, so the only reliable reference is the
      4-way strict majority mask in the quadrant coordinate frame.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    cross_row, cross_col, divider_color = _find_separator_cross(g, bg)
    quadrants = _split_quadrants(g, cross_row, cross_col)

    pattern_colors = _colors_present(g, exclude={bg, divider_color})
    assert len(pattern_colors) == 1, "expected exactly one pattern color"
    pattern_color = int(pattern_colors[0])

    _, repaired = _repair_from_majority(quadrants, pattern_color, divider_color, bg)
    out = g.copy()
    for (rs, cs), quadrant in zip(_quadrant_slices(cross_row, cross_col), repaired):
        out[rs, cs] = quadrant
    return out


def _array_signature(arr: np.ndarray) -> str:
    return hashlib.md5(arr.tobytes() + str(arr.shape).encode()).hexdigest()


def _border_anchor(height: int, width: int, side: str) -> tuple[int, int]:
    if side == "top":
        return 0, int(_RNG.integers(1, width - 1))
    if side == "bottom":
        return height - 1, int(_RNG.integers(1, width - 1))
    if side == "left":
        return int(_RNG.integers(1, height - 1)), 0
    return int(_RNG.integers(1, height - 1)), width - 1


def _walk_path(mask: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> None:
    r, c = start
    tr, tc = end
    mask[r, c] = True
    while (r, c) != (tr, tc):
        moves = []
        if r < tr:
            moves.append((1, 0))
        elif r > tr:
            moves.append((-1, 0))
        if c < tc:
            moves.append((0, 1))
        elif c > tc:
            moves.append((0, -1))
        dr, dc = moves[int(_RNG.integers(len(moves)))]
        r += dr
        c += dc
        mask[r, c] = True


def _grow_template(height: int, width: int) -> np.ndarray:
    total = height * width
    target = int(_RNG.integers(max(8, total // 4), max(9, (total * 11) // 20) + 1))

    for _ in range(200):
        mask = np.zeros((height, width), dtype=bool)
        anchors = [
            _border_anchor(height, width, "top"),
            _border_anchor(height, width, "right"),
            _border_anchor(height, width, "bottom"),
            _border_anchor(height, width, "left"),
        ]
        order = list(_RNG.permutation(len(anchors)))
        for i in range(len(order) - 1):
            _walk_path(mask, anchors[order[i]], anchors[order[i + 1]])
        if _RNG.random() < 0.5:
            _walk_path(mask, anchors[order[-1]], anchors[order[0]])

        while int(mask.sum()) < target:
            frontier = []
            for r, c in np.argwhere(mask):
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = int(r + dr), int(c + dc)
                    if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]:
                        frontier.append((nr, nc))
            if not frontier:
                break
            nr, nc = frontier[int(_RNG.integers(len(frontier)))]
            mask[nr, nc] = True

        border_constraints = (
            mask[0].any(),
            mask[-1].any(),
            mask[:, 0].any(),
            mask[:, -1].any(),
            (~mask[0]).any(),
            (~mask[-1]).any(),
            (~mask[:, 0]).any(),
            (~mask[:, -1]).any(),
        )
        if not all(border_constraints):
            continue
        if int(mask.sum()) < 8 or int((~mask).sum()) < 8:
            continue
        if np.array_equal(mask, np.flipud(mask)):
            continue
        if np.array_equal(mask, np.fliplr(mask)):
            continue
        if np.array_equal(mask, np.rot90(mask, 2)):
            continue
        return mask

    raise RuntimeError("failed to build a non-degenerate template mask")


def _balanced_assignment(coords: np.ndarray, n_quadrants: int) -> list[list[tuple[int, int]]]:
    shuffled = coords[_RNG.permutation(len(coords))]
    assignments: list[list[tuple[int, int]]] = [[] for _ in range(n_quadrants)]
    for quadrant, coord in enumerate(shuffled[:n_quadrants]):
        assignments[quadrant].append((int(coord[0]), int(coord[1])))
    for coord in shuffled[n_quadrants:]:
        counts = np.array([len(items) for items in assignments], dtype=int)
        candidates = np.flatnonzero(counts == counts.min())
        quadrant = int(candidates[int(_RNG.integers(len(candidates)))])
        assignments[quadrant].append((int(coord[0]), int(coord[1])))
    return assignments


def _corrupt_template(template_mask: np.ndarray) -> list[np.ndarray]:
    true_coords = np.argwhere(template_mask)
    false_coords = np.argwhere(~template_mask)
    assert len(true_coords) >= 8 and len(false_coords) >= 8

    n_true = len(true_coords)
    n_false = len(false_coords)
    omit_count = int(_RNG.integers(4, min(n_true - 1, max(5, n_true // 2)) + 1))
    single_add_count = int(_RNG.integers(4, min(n_false - 2, max(5, n_false // 3 + 1)) + 1))
    extra_add_count = int(_RNG.integers(1, min(n_false - single_add_count, max(2, n_false // 4)) + 1))

    omit_positions = true_coords[_RNG.permutation(n_true)[:omit_count]]
    single_add_positions = false_coords[_RNG.permutation(n_false)[:single_add_count]]

    remaining_false = [tuple(map(int, coord)) for coord in false_coords]
    chosen_false = {tuple(map(int, coord)) for coord in single_add_positions}
    remaining_false = [coord for coord in remaining_false if coord not in chosen_false]
    pair_positions = np.array(remaining_false[:extra_add_count], dtype=int)
    _RNG.shuffle(pair_positions)

    quadrants = [template_mask.copy() for _ in range(4)]

    for q, coords in enumerate(_balanced_assignment(omit_positions, 4)):
        for r, c in coords:
            quadrants[q][r, c] = False

    for q, coords in enumerate(_balanced_assignment(single_add_positions, 4)):
        for r, c in coords:
            quadrants[q][r, c] = True

    for coord in pair_positions:
        r, c = int(coord[0]), int(coord[1])
        chosen = _RNG.choice(4, size=2, replace=False)
        for q in chosen:
            quadrants[int(q)][r, c] = True

    return quadrants


def _render_input(
    quadrants: list[np.ndarray], pattern_color: int, divider_color: int
) -> np.ndarray:
    qh, qw = quadrants[0].shape
    grid = np.zeros((2 * qh + 1, 2 * qw + 1), dtype=int)
    grid[qh, :] = divider_color
    grid[:, qw] = divider_color
    for (rs, cs), mask in zip(_quadrant_slices(qh, qw), quadrants):
        grid[rs, cs][mask] = pattern_color
    return grid


def _expected_output(
    template_mask: np.ndarray,
    quadrants: list[np.ndarray],
    pattern_color: int,
    divider_color: int,
) -> np.ndarray:
    qh, qw = template_mask.shape
    out = np.zeros((2 * qh + 1, 2 * qw + 1), dtype=int)
    out[qh, :] = divider_color
    out[:, qw] = divider_color
    for (rs, cs), quadrant in zip(_quadrant_slices(qh, qw), quadrants):
        region = np.zeros_like(template_mask, dtype=int)
        region[template_mask & quadrant] = pattern_color
        region[template_mask & ~quadrant] = divider_color
        out[rs, cs] = region
    return out


def _semantic_checks(
    grid: np.ndarray,
    template_mask: np.ndarray,
    quadrants: list[np.ndarray],
    expected: np.ndarray,
    pattern_color: int,
    divider_color: int,
) -> bool:
    if np.array_equal(grid, expected):
        return False

    counts = np.sum(quadrants, axis=0)
    if not np.array_equal(counts >= 3, template_mask):
        return False
    if not np.any(counts[template_mask] == 3):
        return False
    if not np.any(counts[~template_mask] == 2):
        return False

    quadrant_keys = set()
    for quadrant in quadrants:
        if np.array_equal(quadrant, template_mask):
            return False
        if not np.any(template_mask & ~quadrant):
            return False
        if not np.any(~template_mask & quadrant):
            return False
        quadrant_keys.add(quadrant.tobytes())
    if len(quadrant_keys) < 4:
        return False

    if np.any(grid == divider_color) and len(_colors_present(grid, exclude={BACKGROUND})) != 2:
        return False
    try:
        solved = main(grid)
    except Exception:
        return False
    if not np.array_equal(solved, expected):
        return False
    if np.array_equal(solved, grid):
        return False

    filled_marks = solved == divider_color
    cross_row, cross_col = template_mask.shape
    interior_divider = filled_marks.copy()
    interior_divider[cross_row, :] = False
    interior_divider[:, cross_col] = False
    if not interior_divider.any():
        return False
    return True


def _attempt() -> np.ndarray | None:
    qh = int(_RNG.integers(*QUADRANT_HEIGHT_RANGE))
    qw = int(_RNG.integers(*QUADRANT_WIDTH_RANGE))

    template_mask = _grow_template(qh, qw)
    quadrants = _corrupt_template(template_mask)

    colors = list(PALETTE)
    _RNG.shuffle(colors)
    pattern_color, divider_color = int(colors[0]), int(colors[1])

    grid = _render_input(quadrants, pattern_color, divider_color)
    expected = _expected_output(template_mask, quadrants, pattern_color, divider_color)

    if not _semantic_checks(grid, template_mask, quadrants, expected, pattern_color, divider_color):
        return None

    inp_sig = _array_signature(grid)
    out_sig = _array_signature(expected)
    if inp_sig in _RECENT_INPUT_SIGNATURES or out_sig in _RECENT_OUTPUT_SIGNATURES:
        return None
    _RECENT_INPUT_SIGNATURES.append(inp_sig)
    _RECENT_OUTPUT_SIGNATURES.append(out_sig)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid majority-vote quadrant repair input")
