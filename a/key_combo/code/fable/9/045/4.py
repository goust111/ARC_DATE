from __future__ import annotations

import random
from collections import Counter, defaultdict, deque

import numpy as np


BG = 0
PATH_COLOR = 5
OVERLAP_COLOR = 6
PALETTE = (1, 2, 3, 4, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 400

Coord = tuple[int, int]
Instruction = tuple[int, int]


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(grid: np.ndarray, bg: int) -> list[list[Coord]]:
    height, width = grid.shape
    seen = np.zeros_like(grid, dtype=bool)
    comps: list[list[Coord]] = []
    for r in range(height):
        for c in range(width):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[Coord] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < height and 0 <= nc < width and not seen[nr, nc] and int(grid[nr, nc]) != bg:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(coords)
    return comps


def _parse_input(grid: np.ndarray) -> tuple[int, dict[int, Coord], list[Instruction], list[tuple[Coord, Coord]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _background_color(g)
    dots: dict[int, Coord] = {}
    instructions: list[Instruction] = []
    dominoes: list[tuple[Coord, Coord]] = []

    for comp in _components(g, bg):
        if len(comp) == 1:
            r, c = comp[0]
            color = int(g[r, c])
            assert color not in dots, "each dot color must be unique"
            dots[color] = (r, c)
            continue

        assert len(comp) == 2, "foreground components must be singletons or horizontal dominoes"
        ordered = sorted(comp, key=lambda rc: (rc[0], rc[1]))
        (r1, c1), (r2, c2) = ordered
        assert r1 == r2 and c2 == c1 + 1, "instruction blocks must be horizontal dominoes"
        left = int(g[r1, c1])
        right = int(g[r2, c2])
        assert left != right, "domino halves must use different colors"
        instructions.append((left, right))
        dominoes.append(((r1, c1), (r2, c2)))

    assert instructions, "expected at least one instruction domino"
    for start_color, end_color in instructions:
        assert start_color in dots and end_color in dots, "domino colors must refer to visible dots"
    return bg, dots, instructions, dominoes


def _path_cells(start: Coord, end: Coord) -> list[Coord]:
    sr, sc = start
    er, ec = end
    path: list[Coord] = []
    cur_r, cur_c = sr, sc

    if cur_c != ec:
        step_c = 1 if ec > cur_c else -1
        while cur_c != ec:
            cur_c += step_c
            if (cur_r, cur_c) != end:
                path.append((cur_r, cur_c))

    if cur_r != er:
        step_r = 1 if er > cur_r else -1
        while cur_r != er:
            cur_r += step_r
            if (cur_r, cur_c) != end:
                path.append((cur_r, cur_c))

    return path


def _render_output(shape: tuple[int, int], bg: int, dots: dict[int, Coord], instructions: list[Instruction]) -> np.ndarray:
    claims: defaultdict[Coord, int] = defaultdict(int)
    for start_color, end_color in instructions:
        for cell in _path_cells(dots[start_color], dots[end_color]):
            claims[cell] += 1

    out = np.full(shape, int(bg), dtype=int)
    for cell, count in claims.items():
        r, c = cell
        out[r, c] = OVERLAP_COLOR if count >= 2 else PATH_COLOR
    for color, (r, c) in dots.items():
        out[r, c] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read each horizontal two-cell block as a connection instruction, draw its Manhattan path between matching dots, and recolor shared path cells pink.

    Input structure:
    - A black background contains several isolated single-pixel colored dots, with at most one dot per color.
    - Separate foreground components of size two are horizontal dominoes whose left and right cells have different colors.
    - Every color appearing in a domino also appears on exactly one singleton dot somewhere else in the grid.

    Rule:
    - Delete all instruction dominoes.
    - For each domino [A|B], start from the dot of color A, move horizontally along A's row until reaching B's column, then move vertically along that column to the dot of color B.
    - Paint visited non-endpoint cells grey; any cell visited by two or more such paths becomes pink instead.
    - Keep all singleton dots unchanged.

    Key insight:
    - The dominoes are not objects to preserve: they are ordered color instructions whose left-right order determines which dot provides the starting row and which dot provides the destination column.
    """
    bg, dots, instructions, _ = _parse_input(np.asarray(grid, dtype=int))
    return _render_output(np.asarray(grid, dtype=int).shape, bg, dots, instructions)


def _transform(coord: Coord, height: int, width: int, flip_r: bool, flip_c: bool) -> Coord:
    r, c = coord
    if flip_r:
        r = height - 1 - r
    if flip_c:
        c = width - 1 - c
    return int(r), int(c)


def _sample_range(lo: int, hi: int) -> int | None:
    if lo > hi:
        return None
    return random.randint(lo, hi)


def _build_cross(height: int, width: int) -> tuple[dict[str, Coord], list[tuple[str, str]]] | None:
    qr = _sample_range(4, height - 5)
    qc = _sample_range(4, width - 5)
    if qr is None or qc is None:
        return None

    d1 = _sample_range(2, qc - 2)
    d2 = _sample_range(2, height - 1 - qr - 2)
    d3 = _sample_range(2, width - 1 - qc - 2)
    d4 = _sample_range(2, qr - 2)
    d5 = _sample_range(2, width - 1 - qc - 2)
    d6 = _sample_range(2, height - 1 - qr - 2)
    if None in (d1, d2, d3, d4, d5, d6):
        return None

    dots = {
        "a": (qr, qc - d1),
        "b": (qr + d2, qc + d3),
        "c": (qr - d4, qc + d5),
        "d": (qr + d6, qc),
    }
    instructions = [("a", "b"), ("c", "d")]
    return dots, instructions


def _build_shared_source(height: int, width: int) -> tuple[dict[str, Coord], list[tuple[str, str]]] | None:
    hr = _sample_range(4, height - 5)
    hc = _sample_range(4, width - 5)
    if hr is None or hc is None:
        return None

    d1 = _sample_range(2, hc - 2)
    d2 = _sample_range(2, hr - 2)
    d3 = _sample_range(2, height - 1 - hr - 2)
    if None in (d1, d2, d3):
        return None

    dots = {
        "a": (hr, hc - d1),
        "b": (hr - d2, hc),
        "c": (hr + d3, hc),
    }
    instructions = [("a", "b"), ("a", "c")]
    return dots, instructions


def _build_shared_target(height: int, width: int) -> tuple[dict[str, Coord], list[tuple[str, str]]] | None:
    target_r = _sample_range(5, height - 4)
    target_c = _sample_range(4, width - 5)
    if target_r is None or target_c is None:
        return None

    n_sources = random.choice((2, 3))
    max_above = target_r - 2
    if max_above < n_sources + 1:
        return None
    row_gaps = sorted(random.sample(range(2, max_above + 1), n_sources))

    side_choices = []
    if n_sources == 2:
        side_choices = [-1, 1]
    else:
        side_choices = [-1, 1, random.choice((-1, 1))]
        random.shuffle(side_choices)

    dots = {"t": (target_r, target_c)}
    instructions: list[tuple[str, str]] = []
    for idx, (gap, side) in enumerate(zip(row_gaps, side_choices)):
        if side < 0:
            max_offset = target_c - 2
        else:
            max_offset = width - 1 - target_c - 2
        offset = _sample_range(2, min(5, max_offset))
        if offset is None:
            return None
        key = chr(ord("a") + idx)
        dots[key] = (target_r - gap, target_c + side * offset)
        instructions.append((key, "t"))
    return dots, instructions


def _build_layout(height: int, width: int) -> tuple[dict[Coord, int], list[Instruction]] | None:
    builders = [_build_cross, _build_shared_source, _build_shared_target]
    builder = random.choice(builders)
    base = builder(height, width)
    if base is None:
        return None
    name_coords, symbolic_instructions = base

    flip_r = random.choice((False, True))
    flip_c = random.choice((False, True))
    transformed = {
        name: _transform(coord, height, width, flip_r=flip_r, flip_c=flip_c)
        for name, coord in name_coords.items()
    }

    color_order = list(transformed)
    colors = random.sample(PALETTE, len(color_order))
    name_to_color = {name: color for name, color in zip(color_order, colors)}
    dots = {transformed[name]: name_to_color[name] for name in color_order}
    instructions = [(name_to_color[a], name_to_color[b]) for a, b in symbolic_instructions]
    return dots, instructions


def _claim_sets(dots_by_color: dict[int, Coord], instructions: list[Instruction]) -> list[set[Coord]]:
    return [set(_path_cells(dots_by_color[a], dots_by_color[b])) for a, b in instructions]


def _expand_mask(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    if radius <= 0:
        return mask.copy()
    height, width = mask.shape
    out = mask.copy()
    for _ in range(radius):
        grown = out.copy()
        for r, c in np.argwhere(out):
            for rr in range(r - 1, r + 2):
                for cc in range(c - 1, c + 2):
                    if 0 <= rr < height and 0 <= cc < width:
                        grown[rr, cc] = True
        out = grown
    return out


def _place_extra_dot(
    dots: dict[Coord, int],
    instructions: list[Instruction],
    height: int,
    width: int,
) -> None:
    if random.random() >= 0.4:
        return

    used_colors = set(dots.values())
    available = [color for color in PALETTE if color not in used_colors]
    if not available:
        return

    by_color = {color: coord for coord, color in dots.items()}
    path_cells = set().union(*_claim_sets(by_color, instructions))
    blocked = set(dots)
    blocked |= path_cells
    for r, c in list(blocked):
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if 0 <= rr < height and 0 <= cc < width:
                    blocked.add((rr, cc))

    candidates = [(r, c) for r in range(height) for c in range(width) if (r, c) not in blocked]
    if not candidates:
        return
    dots[random.choice(candidates)] = random.choice(available)


def _place_dominoes(
    height: int,
    width: int,
    dots: dict[Coord, int],
    instructions: list[Instruction],
) -> np.ndarray | None:
    grid = np.full((height, width), BG, dtype=int)
    for (r, c), color in dots.items():
        grid[r, c] = int(color)

    by_color = {color: coord for coord, color in dots.items()}
    protected = np.zeros((height, width), dtype=bool)
    for coord in dots:
        protected[coord] = True
    for path in _claim_sets(by_color, instructions):
        for coord in path:
            protected[coord] = True
    protected = _expand_mask(protected, radius=1)

    dominoes = instructions.copy()
    random.shuffle(dominoes)
    for left_color, right_color in dominoes:
        candidates: list[Coord] = []
        for r in range(height):
            for c in range(width - 1):
                if grid[r, c] != BG or grid[r, c + 1] != BG:
                    continue
                window_has_fg = False
                for rr in range(r - 1, r + 2):
                    for cc in range(c - 1, c + 3):
                        if 0 <= rr < height and 0 <= cc < width and grid[rr, cc] != BG:
                            window_has_fg = True
                if window_has_fg:
                    continue
                if protected[r, c] or protected[r, c + 1]:
                    continue
                candidates.append((r, c))
        if not candidates:
            return None
        r, c = random.choice(candidates)
        grid[r, c] = int(left_color)
        grid[r, c + 1] = int(right_color)
    return grid


def _validate_semantics(grid: np.ndarray, expected_dots: dict[Coord, int], expected_instructions: list[Instruction]) -> bool:
    try:
        bg, dots_by_color, instructions, dominoes = _parse_input(grid)
    except AssertionError:
        return False

    if len(dominoes) != len(expected_instructions):
        return False

    parsed_dots = {coord: color for color, coord in dots_by_color.items()}
    if parsed_dots != expected_dots:
        return False
    if Counter(instructions) != Counter(expected_instructions):
        return False

    paths = _claim_sets(dots_by_color, instructions)
    if len(paths) < 2:
        return False
    union = set().union(*paths)
    if not union:
        return False

    claim_count: Counter[Coord] = Counter()
    for path in paths:
        for cell in path:
            claim_count[cell] += 1
    if not any(count >= 2 for count in claim_count.values()):
        return False
    if not any(count == 1 for count in claim_count.values()):
        return False

    dot_positions = set(parsed_dots)
    if union & dot_positions:
        return False
    if any(not (set(path) - (union - set(path))) for path in paths):
        return False

    expected = _render_output(grid.shape, bg, dots_by_color, instructions)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False

    swapped = _render_output(grid.shape, bg, dots_by_color, [(b, a) for a, b in instructions])
    if np.array_equal(swapped, expected):
        return False

    if np.array_equal(grid, expected):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        height = random.randint(13, 20)
        width = random.randint(13, 20)
        built = _build_layout(height, width)
        if built is None:
            continue

        dots, instructions = built
        _place_extra_dot(dots, instructions, height, width)
        grid = _place_dominoes(height, width, dots, instructions)
        if grid is None:
            continue
        if _validate_semantics(grid, dots, instructions):
            return grid
    raise RuntimeError("failed to generate a valid instruction-overlap input")
