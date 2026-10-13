from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
YELLOW = 4
TEAL = 8

CARDINAL_PATTERN = (TEAL, TEAL, YELLOW)

DIRECTION_STEPS = {
    "N": (-1, 0),
    "S": (1, 0),
    "W": (0, -1),
    "E": (0, 1),
    "NW": (-1, -1),
    "NE": (-1, 1),
    "SW": (1, -1),
    "SE": (1, 1),
}
ALL_DIRECTIONS = ("N", "S", "W", "E", "NW", "NE", "SW", "SE")
CARDINAL_DIRECTIONS = ("N", "S", "W", "E")
DIAGONAL_DIRECTIONS = ("NW", "NE", "SW", "SE")
OPPOSITE_PAIRS = (("N", "S"), ("W", "E"), ("NW", "SE"), ("NE", "SW"))


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _components(mask: np.ndarray, connectivity: int) -> list[np.ndarray]:
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    if connectivity == 4:
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        steps = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

    out: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if visited[r, c]:
            continue
        stack = [(r, c)]
        visited[r, c] = True
        coords: list[tuple[int, int]] = []
        while stack:
            cr, cc = stack.pop()
            coords.append((cr, cc))
            for dr, dc in steps:
                nr = cr + dr
                nc = cc + dc
                if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not visited[nr, nc]:
                    visited[nr, nc] = True
                    stack.append((nr, nc))
        out.append(np.array(coords, dtype=int))
    return out


def _classify_blue(coords: np.ndarray) -> tuple[str, int, np.ndarray]:
    assert len(coords) >= 2, "diagonal fragments must have visible orientation"
    diffs = coords[:, 0] - coords[:, 1]
    sums = coords[:, 0] + coords[:, 1]
    if np.all(diffs == diffs[0]):
        ordered = coords[np.argsort(coords[:, 0])]
        assert np.all(np.diff(ordered[:, 0]) == 1)
        assert np.all(np.diff(ordered[:, 1]) == 1)
        return "d1", int(diffs[0]), ordered
    if np.all(sums == sums[0]):
        ordered = coords[np.argsort(coords[:, 0])]
        assert np.all(np.diff(ordered[:, 0]) == 1)
        assert np.all(np.diff(ordered[:, 1]) == -1)
        return "d2", int(sums[0]), ordered
    raise AssertionError("blue fragments must be straight diagonals")


def _classify_cardinal(coords: np.ndarray, grid: np.ndarray) -> tuple[str, int, np.ndarray, np.ndarray]:
    assert len(coords) >= 2, "cardinal fragments must have visible orientation"
    rows = coords[:, 0]
    cols = coords[:, 1]
    if np.all(rows == rows[0]):
        ordered = coords[np.argsort(cols)]
        assert np.all(np.diff(ordered[:, 1]) == 1)
        colors = grid[ordered[:, 0], ordered[:, 1]].astype(int)
        return "h", int(rows[0]), ordered, colors
    if np.all(cols == cols[0]):
        ordered = coords[np.argsort(rows)]
        assert np.all(np.diff(ordered[:, 0]) == 1)
        colors = grid[ordered[:, 0], ordered[:, 1]].astype(int)
        return "v", int(cols[0]), ordered, colors
    raise AssertionError("cardinal fragments must be straight rows or columns")


def _extract_fragments(grid: np.ndarray) -> list[dict]:
    colors = set(map(int, np.unique(grid)))
    assert colors <= {BLACK, BLUE, YELLOW, TEAL}, "unexpected input colors"

    fragments: list[dict] = []
    for coords in _components(grid == BLUE, connectivity=8):
        kind, const, ordered = _classify_blue(coords)
        fragments.append(
            {
                "kind": kind,
                "const": const,
                "coords": ordered,
                "colors": np.full(len(ordered), BLUE, dtype=int),
            }
        )

    card_mask = (grid == TEAL) | (grid == YELLOW)
    for coords in _components(card_mask, connectivity=4):
        kind, const, ordered, colors_arr = _classify_cardinal(coords, grid)
        fragments.append(
            {
                "kind": kind,
                "const": const,
                "coords": ordered,
                "colors": colors_arr,
            }
        )

    assert len(fragments) >= 2, "expected several residual fragments"
    return fragments


def _intersect_lines(a: dict, b: dict) -> tuple[int, int] | None:
    ka = a["kind"]
    kb = b["kind"]
    va = int(a["const"])
    vb = int(b["const"])
    if ka == kb:
        return None

    kinds = {ka, kb}
    if kinds == {"h", "v"}:
        row = va if ka == "h" else vb
        col = va if ka == "v" else vb
        return int(row), int(col)
    if kinds == {"h", "d1"}:
        row = va if ka == "h" else vb
        diff = va if ka == "d1" else vb
        return int(row), int(row - diff)
    if kinds == {"h", "d2"}:
        row = va if ka == "h" else vb
        total = va if ka == "d2" else vb
        return int(row), int(total - row)
    if kinds == {"v", "d1"}:
        col = va if ka == "v" else vb
        diff = va if ka == "d1" else vb
        return int(diff + col), int(col)
    if kinds == {"v", "d2"}:
        col = va if ka == "v" else vb
        total = va if ka == "d2" else vb
        return int(total - col), int(col)
    if kinds == {"d1", "d2"}:
        diff = va if ka == "d1" else vb
        total = va if ka == "d2" else vb
        if (diff + total) % 2 != 0 or (total - diff) % 2 != 0:
            return None
        return int((diff + total) // 2), int((total - diff) // 2)
    return None


def _fragment_direction(fragment: dict, center: tuple[int, int]) -> str:
    cr, cc = center
    coords = fragment["coords"]
    rows = coords[:, 0]
    cols = coords[:, 1]
    kind = fragment["kind"]

    if kind == "h":
        assert np.all(rows == cr)
        if np.all(cols < cc):
            return "W"
        if np.all(cols > cc):
            return "E"
    elif kind == "v":
        assert np.all(cols == cc)
        if np.all(rows < cr):
            return "N"
        if np.all(rows > cr):
            return "S"
    elif kind == "d1":
        assert np.all(rows - cols == cr - cc)
        if np.all((rows < cr) & (cols < cc)):
            return "NW"
        if np.all((rows > cr) & (cols > cc)):
            return "SE"
    elif kind == "d2":
        assert np.all(rows + cols == cr + cc)
        if np.all((rows < cr) & (cols > cc)):
            return "NE"
        if np.all((rows > cr) & (cols < cc)):
            return "SW"
    raise AssertionError("each fragment must lie on one outward ray from the hidden center")


def _validate_fragment(fragment: dict, center: tuple[int, int]) -> tuple[str, tuple[int, int]]:
    cr, cc = center
    direction = _fragment_direction(fragment, center)
    coords = fragment["coords"]
    colors = fragment["colors"]
    steps = np.array([max(abs(int(r) - cr), abs(int(c) - cc)) for r, c in coords], dtype=int)
    order = np.argsort(steps)
    steps = steps[order]
    colors = colors[order]
    expected_steps = np.arange(int(steps[0]), int(steps[-1]) + 1)
    assert np.array_equal(steps, expected_steps), "fragments must be contiguous ray segments"

    if direction in DIAGONAL_DIRECTIONS:
        assert int(steps[0]) >= 1
        assert np.all(colors == BLUE), "diagonal fragments must be blue"
    else:
        assert int(steps[0]) >= 2
        expected_colors = np.array([CARDINAL_PATTERN[(int(step) - 2) % 3] for step in steps], dtype=int)
        assert np.array_equal(colors, expected_colors), "cardinal phase must align with the center"
    return direction, (int(steps[0]), int(steps[-1]))


def _solve_center(grid: np.ndarray, fragments: list[dict] | None = None) -> tuple[int, int]:
    if fragments is None:
        fragments = _extract_fragments(grid)

    candidates: set[tuple[int, int]] = set()
    for i in range(len(fragments)):
        for j in range(i + 1, len(fragments)):
            point = _intersect_lines(fragments[i], fragments[j])
            if point is not None:
                candidates.add(point)

    assert len(candidates) == 1, "fragment extensions must meet at one unique center"
    center = next(iter(candidates))
    cr, cc = center
    h, w = grid.shape
    assert 1 <= cr < h - 1 and 1 <= cc < w - 1, "red plus must fit inside the canvas"

    seen_dirs: set[str] = set()
    for fragment in fragments:
        direction, _span = _validate_fragment(fragment, center)
        assert direction not in seen_dirs, "each ray contributes at most one visible fragment"
        seen_dirs.add(direction)
    return center


def _ray_cells(shape: tuple[int, int], center: tuple[int, int], direction: str, start_step: int) -> list[tuple[int, int]]:
    cr, cc = center
    dr, dc = DIRECTION_STEPS[direction]
    cells: list[tuple[int, int]] = []
    r = cr + dr * start_step
    c = cc + dc * start_step
    while _in_bounds(shape, r, c):
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def _paint_full_star(shape: tuple[int, int], center: tuple[int, int]) -> np.ndarray:
    out = np.full(shape, BLACK, dtype=int)
    cr, cc = center
    plus = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
    for dr, dc in plus:
        out[cr + dr, cc + dc] = RED

    for direction in DIAGONAL_DIRECTIONS:
        for r, c in _ray_cells(shape, center, direction, start_step=1):
            out[r, c] = BLUE

    for direction in CARDINAL_DIRECTIONS:
        for idx, (r, c) in enumerate(_ray_cells(shape, center, direction, start_step=2)):
            out[r, c] = CARDINAL_PATTERN[idx % len(CARDINAL_PATTERN)]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Reconstruct the full eight-direction star by backtracking residual ray fragments to their hidden emitter.

    Input structure:
    - The background is black.
    - The only visible marks are short outer ray fragments from one unseen center.
    - Diagonal fragments are blue, while horizontal and vertical fragments use the repeating teal-teal-yellow phase.

    Rule:
    - Reverse-extend every fragment along its supporting line and find their unique common intersection.
    - Treat that intersection as the hidden emitter and paint a red five-cell plus there.
    - Complete all eight rays to the canvas edge: diagonals stay blue, and cardinal rays start just beyond the red arms with the teal-teal-yellow cycle aligned to every visible fragment.

    Key insight:
    - The emitter itself is absent from the input; only the shared geometric intersection, plus the cardinal color phase, reveals the correct center.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    fragments = _extract_fragments(g)
    center = _solve_center(g, fragments)
    return _paint_full_star(g.shape, center)


GRID_MIN = 11
GRID_MAX = 30
CENTER_MARGIN = 4
GEN_RETRY_LIMIT = 500
FRAGMENT_MAX_LEN = 5


def _direction_family(direction: str) -> str:
    if direction in ("N", "S"):
        return "v"
    if direction in ("W", "E"):
        return "h"
    if direction in ("NW", "SE"):
        return "d1"
    return "d2"


def _sample_group(group: tuple[str, ...], count: int) -> list[str]:
    assert 0 < count <= len(group)
    chosen: list[str] = []
    pair_options = [pair for pair in OPPOSITE_PAIRS if set(pair).issubset(set(group))]
    if count >= 2 and random.random() < 0.55:
        pair = random.choice(pair_options)
        chosen.extend(pair)
    remaining = [direction for direction in group if direction not in chosen]
    if len(chosen) < count:
        chosen.extend(random.sample(remaining, count - len(chosen)))
    return chosen


def _sample_visible_directions() -> list[str]:
    mode = random.choice(("full", "mixed", "card_heavy", "diag_heavy"))
    if mode == "full":
        n_card = random.randint(3, 4)
        n_diag = random.randint(3, 4)
    elif mode == "mixed":
        n_card = random.randint(2, 3)
        n_diag = random.randint(2, 3)
    elif mode == "card_heavy":
        n_card = random.randint(3, 4)
        n_diag = random.randint(1, 2)
    else:
        n_card = random.randint(1, 2)
        n_diag = random.randint(3, 4)

    chosen = set(_sample_group(CARDINAL_DIRECTIONS, n_card))
    chosen.update(_sample_group(DIAGONAL_DIRECTIONS, n_diag))
    assert chosen & set(CARDINAL_DIRECTIONS)
    assert chosen & set(DIAGONAL_DIRECTIONS)
    assert len({_direction_family(direction) for direction in chosen}) >= 2
    return [direction for direction in ALL_DIRECTIONS if direction in chosen]


def _sample_fragment_span(ray_len: int) -> tuple[int, int]:
    assert ray_len >= 3
    hidden = random.randint(1, min(3, ray_len - 2))
    max_len = min(FRAGMENT_MAX_LEN, ray_len - hidden)
    frag_len = random.randint(2, max_len)
    max_trim = min(2, ray_len - hidden - frag_len)
    trim = random.randint(0, max_trim)
    start_idx = ray_len - trim - frag_len
    end_idx = start_idx + frag_len
    assert hidden <= start_idx and end_idx <= ray_len
    return start_idx, end_idx


def _paint_fragment(inp: np.ndarray, center: tuple[int, int], direction: str) -> tuple[int, int]:
    base_step = 1 if direction in DIAGONAL_DIRECTIONS else 2
    cells = _ray_cells(inp.shape, center, direction, start_step=base_step)
    start_idx, end_idx = _sample_fragment_span(len(cells))
    for idx in range(start_idx, end_idx):
        r, c = cells[idx]
        if direction in DIAGONAL_DIRECTIONS:
            inp[r, c] = BLUE
        else:
            inp[r, c] = CARDINAL_PATTERN[idx % len(CARDINAL_PATTERN)]
    return base_step + start_idx, base_step + end_idx - 1


def _semantic_check(inp: np.ndarray, center: tuple[int, int], spans: dict[str, tuple[int, int]]) -> bool:
    try:
        fragments = _extract_fragments(inp)
        solved_center = _solve_center(inp, fragments)
        if solved_center != center:
            return False

        seen: dict[str, tuple[int, int]] = {}
        for fragment in fragments:
            direction, span = _validate_fragment(fragment, center)
            if direction in seen:
                return False
            seen[direction] = span
        if seen != spans:
            return False

        expected = _paint_full_star(inp.shape, center)
        actual = main(inp)
        if not np.array_equal(actual, expected):
            return False
    except Exception:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    cr = random.randint(CENTER_MARGIN, h - CENTER_MARGIN - 1)
    cc = random.randint(CENTER_MARGIN, w - CENTER_MARGIN - 1)
    center = (cr, cc)

    visible = _sample_visible_directions()
    inp = np.full((h, w), BLACK, dtype=int)
    spans: dict[str, tuple[int, int]] = {}
    for direction in visible:
        base_step = 1 if direction in DIAGONAL_DIRECTIONS else 2
        ray_len = len(_ray_cells(inp.shape, center, direction, start_step=base_step))
        if ray_len < 3:
            return None
        spans[direction] = _paint_fragment(inp, center, direction)

    if not _semantic_check(inp, center, spans):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid convergence-fragment input")
