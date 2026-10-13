from __future__ import annotations

import itertools
import random
from collections import deque
from dataclasses import dataclass

import numpy as np


BLACK = 0
RED = 2
BODY_COLOR_CHOICES = (1, 3, 4, 5, 6, 7, 8, 9)

UP = "U"
DOWN = "D"
LEFT = "L"
RIGHT = "R"
HORIZONTAL_DIRS = (LEFT, RIGHT)
VERTICAL_DIRS = (UP, DOWN)

COUNT_CHOICES = ((1, 2), (2, 1), (1, 3), (3, 1), (2, 2), (2, 3), (3, 2))
GRID_MIN = 14
GRID_MAX = 24
NOZZLE_MARGIN = 2
POSITION_GAP = 3
GENERATION_RETRIES = 500


@dataclass(frozen=True)
class Sprite:
    nozzle_r: int
    nozzle_c: int
    direction: str
    cells: frozenset[tuple[int, int]]


@dataclass(frozen=True)
class SpriteSpec:
    nozzle_r: int
    nozzle_c: int
    direction: str
    variant: int


SPRITE_BODY_OFFSETS = {
    (UP, 0): ((1, 0), (1, 1), (2, 0), (2, 1)),
    (UP, 1): ((1, -1), (1, 0), (2, -1), (2, 0)),
    (DOWN, 0): ((-2, 0), (-2, 1), (-1, 0), (-1, 1)),
    (DOWN, 1): ((-2, -1), (-2, 0), (-1, -1), (-1, 0)),
    (LEFT, 0): ((0, 1), (0, 2), (1, 1), (1, 2)),
    (LEFT, 1): ((-1, 1), (-1, 2), (0, 1), (0, 2)),
    (RIGHT, 0): ((0, -2), (0, -1), (1, -2), (1, -1)),
    (RIGHT, 1): ((-1, -2), (-1, -1), (0, -2), (0, -1)),
}


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _sprite_cells_from_spec(spec: SpriteSpec) -> frozenset[tuple[int, int]]:
    cells = {(spec.nozzle_r, spec.nozzle_c)}
    for dr, dc in SPRITE_BODY_OFFSETS[(spec.direction, spec.variant)]:
        cells.add((spec.nozzle_r + dr, spec.nozzle_c + dc))
    return frozenset(cells)


def _extract_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            cells = []
            while queue:
                rr, cc = queue.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(cells)
    return components


def _decode_sprite(component: list[tuple[int, int]]) -> Sprite:
    cells = set((int(r), int(c)) for r, c in component)
    assert len(cells) == 5, "each sprite must have exactly five pixels"
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]

    matches: list[tuple[int, int, str]] = []
    for r0 in range(min(rs) - 1, max(rs) + 1):
        for c0 in range(min(cs) - 1, max(cs) + 1):
            block = {(r0, c0), (r0, c0 + 1), (r0 + 1, c0), (r0 + 1, c0 + 1)}
            if not block.issubset(cells):
                continue
            remainder = cells - block
            if len(remainder) != 1:
                continue
            nozzle_r, nozzle_c = next(iter(remainder))
            if nozzle_r == r0 - 1 and nozzle_c in (c0, c0 + 1):
                matches.append((nozzle_r, nozzle_c, UP))
            elif nozzle_r == r0 + 2 and nozzle_c in (c0, c0 + 1):
                matches.append((nozzle_r, nozzle_c, DOWN))
            elif nozzle_c == c0 - 1 and nozzle_r in (r0, r0 + 1):
                matches.append((nozzle_r, nozzle_c, LEFT))
            elif nozzle_c == c0 + 2 and nozzle_r in (r0, r0 + 1):
                matches.append((nozzle_r, nozzle_c, RIGHT))

    assert len(matches) == 1, "each component must be one 2x2 block plus one side nozzle"
    nozzle_r, nozzle_c, direction = matches[0]
    return Sprite(nozzle_r, nozzle_c, direction, frozenset(cells))


def _extract_sprites(grid: np.ndarray) -> tuple[int, list[Sprite]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be 2D"
    colors = sorted(set(map(int, np.unique(g))) - {BLACK})
    assert len(colors) == 1, "input must contain exactly one non-black sprite color"
    body_color = colors[0]
    components = _extract_components(g == body_color)
    assert components, "expected at least one sprite"
    sprites = [_decode_sprite(component) for component in components]
    return body_color, sprites


def _ray_reaches(direction: str, start_r: int, start_c: int, target_r: int, target_c: int) -> bool:
    if direction == LEFT:
        return target_r == start_r and target_c <= start_c
    if direction == RIGHT:
        return target_r == start_r and target_c >= start_c
    if direction == UP:
        return target_c == start_c and target_r <= start_r
    if direction == DOWN:
        return target_c == start_c and target_r >= start_r
    raise AssertionError("unknown direction")


def _pairwise_intersections(
    shape: tuple[int, int],
    sprites: list[Sprite | SpriteSpec],
    sprite_cells: set[tuple[int, int]],
) -> tuple[set[tuple[int, int]], dict[int, int], int] | None:
    intersections: set[tuple[int, int]] = set()
    degrees = {idx: 0 for idx in range(len(sprites))}
    non_intersections = 0

    for i, first in enumerate(sprites):
        for j in range(i + 1, len(sprites)):
            second = sprites[j]
            if first.direction in HORIZONTAL_DIRS and second.direction in VERTICAL_DIRS:
                h_sprite, h_idx = first, i
                v_sprite, v_idx = second, j
            elif first.direction in VERTICAL_DIRS and second.direction in HORIZONTAL_DIRS:
                h_sprite, h_idx = second, j
                v_sprite, v_idx = first, i
            else:
                continue

            cell = (h_sprite.nozzle_r, v_sprite.nozzle_c)
            assert 0 <= cell[0] < shape[0] and 0 <= cell[1] < shape[1]
            hits = (
                _ray_reaches(h_sprite.direction, h_sprite.nozzle_r, h_sprite.nozzle_c, cell[0], cell[1])
                and _ray_reaches(v_sprite.direction, v_sprite.nozzle_r, v_sprite.nozzle_c, cell[0], cell[1])
            )
            if hits:
                if cell in sprite_cells:
                    return None
                intersections.add(cell)
                degrees[h_idx] += 1
                degrees[v_idx] += 1
            elif cell not in sprite_cells:
                non_intersections += 1

    return intersections, degrees, non_intersections


def _oracle_from_sprites(grid: np.ndarray, sprites: list[Sprite]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    sprite_cells = set().union(*(sprite.cells for sprite in sprites))
    pairwise = _pairwise_intersections(out.shape, sprites, sprite_cells)
    assert pairwise is not None
    intersections, _degrees, _non_intersections = pairwise
    for r, c in intersections:
        assert out[r, c] == BLACK
        out[r, c] = RED
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Light only the cells where two invisible nozzle rays cross.

    Input structure:
    - A black background contains several monochrome five-pixel sprites.
    - Each sprite is one 2x2 square plus one extra nozzle pixel attached on the
      top, bottom, left, or right side of that square.
    - Different sprites may face different cardinal directions, and no two
      sprites lie on the same full ray line.

    Rule:
    - Treat the extra pixel of each sprite as a nozzle and extend an invisible
      ray from that pixel to the grid edge in the nozzle's facing direction.
    - Keep the input unchanged, then paint red only on background cells that are
      reached by two rays at once.
    - Other ray cells stay invisible, and rays passing through sprite bodies do
      not recolor those sprite pixels.

    Key insight:
    - The visible sprite is only a ray generator: the answer depends on the
      exact nozzle cell and facing direction, not on drawing the whole ray path.
    """
    g = np.asarray(grid, dtype=int)
    _, sprites = _extract_sprites(g)
    return _oracle_from_sprites(g, sprites)


def _sample_spaced_positions(low: int, high: int, count: int, gap: int) -> list[int] | None:
    candidates = list(range(low, high + 1))
    if count == 0:
        return []
    for _ in range(200):
        random.shuffle(candidates)
        chosen: list[int] = []
        for value in candidates:
            if all(abs(value - prev) >= gap for prev in chosen):
                chosen.append(value)
                if len(chosen) == count:
                    return sorted(chosen)
    return None


def _choose_horizontal_col(v_cols: list[int], width: int, direction: str) -> int | None:
    low, high = NOZZLE_MARGIN, width - 1 - NOZZLE_MARGIN
    if direction == RIGHT:
        allowed = [c for c in range(low, min(high, max(v_cols)) + 1) if c not in v_cols]
        preferred = [c for c in allowed if c > min(v_cols)]
    else:
        allowed = [c for c in range(max(low, min(v_cols)), high + 1) if c not in v_cols]
        preferred = [c for c in allowed if c < max(v_cols)]
    if not allowed:
        return None
    pool = preferred if preferred and random.random() < 0.7 else allowed
    return random.choice(pool)


def _choose_vertical_row(h_rows: list[int], height: int, direction: str) -> int | None:
    low, high = NOZZLE_MARGIN, height - 1 - NOZZLE_MARGIN
    if direction == DOWN:
        allowed = [r for r in range(low, min(high, max(h_rows)) + 1) if r not in h_rows]
        preferred = [r for r in allowed if r > min(h_rows)]
    else:
        allowed = [r for r in range(max(low, min(h_rows)), high + 1) if r not in h_rows]
        preferred = [r for r in allowed if r < max(h_rows)]
    if not allowed:
        return None
    pool = preferred if preferred and random.random() < 0.7 else allowed
    return random.choice(pool)


def _specs_are_separated(specs: list[SpriteSpec]) -> bool:
    expanded: set[tuple[int, int]] = set()
    for spec in specs:
        cells = _sprite_cells_from_spec(spec)
        for r, c in cells:
            if (r, c) in expanded:
                return False
        for r, c in cells:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    expanded.add((r + dr, c + dc))
    return True


def _ray_cells(shape: tuple[int, int], spec: SpriteSpec) -> list[tuple[int, int]]:
    h, w = shape
    r, c = spec.nozzle_r, spec.nozzle_c
    dr, dc = {
        UP: (-1, 0),
        DOWN: (1, 0),
        LEFT: (0, -1),
        RIGHT: (0, 1),
    }[spec.direction]
    cells: list[tuple[int, int]] = []
    rr, cc = r + dr, c + dc
    while 0 <= rr < h and 0 <= cc < w:
        cells.append((rr, cc))
        rr += dr
        cc += dc
    return cells


def _has_pass_through(shape: tuple[int, int], specs: list[SpriteSpec], intersections: set[tuple[int, int]]) -> bool:
    body_sets = [set(_sprite_cells_from_spec(spec)) - {(spec.nozzle_r, spec.nozzle_c)} for spec in specs]
    for i, spec in enumerate(specs):
        for cell in _ray_cells(shape, spec):
            if cell in intersections:
                continue
            for j, body in enumerate(body_sets):
                if i != j and cell in body:
                    return True
    return False


def _render_input(shape: tuple[int, int], body_color: int, specs: list[SpriteSpec]) -> np.ndarray:
    grid = np.full(shape, BLACK, dtype=int)
    for spec in specs:
        for r, c in _sprite_cells_from_spec(spec):
            grid[r, c] = body_color
    return grid


def _validate_generated(grid: np.ndarray, specs: list[SpriteSpec], want_pass_through: bool) -> bool:
    try:
        body_color, extracted = _extract_sprites(grid)
    except AssertionError:
        return False
    if body_color == RED:
        return False

    expected_specs = sorted(
        (
            spec.nozzle_r,
            spec.nozzle_c,
            spec.direction,
            tuple(sorted(_sprite_cells_from_spec(spec))),
        )
        for spec in specs
    )
    actual_specs = sorted(
        (
            sprite.nozzle_r,
            sprite.nozzle_c,
            sprite.direction,
            tuple(sorted(sprite.cells)),
        )
        for sprite in extracted
    )
    if actual_specs != expected_specs:
        return False

    sprite_cells = set().union(*(_sprite_cells_from_spec(spec) for spec in specs))
    pairwise = _pairwise_intersections(grid.shape, specs, sprite_cells)
    if pairwise is None:
        return False
    intersections, degrees, non_intersections = pairwise
    if not intersections or non_intersections == 0:
        return False
    if min(degrees.values()) <= 0:
        return False
    if len(intersections) > 6:
        return False
    if want_pass_through and not _has_pass_through(grid.shape, specs, intersections):
        return False

    oracle = grid.copy()
    for r, c in intersections:
        if oracle[r, c] != BLACK:
            return False
        oracle[r, c] = RED

    try:
        actual = main(grid)
    except AssertionError:
        return False
    if not np.array_equal(actual, oracle):
        return False

    for r, c in intersections:
        if grid[r, c] != BLACK or actual[r, c] != RED:
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(GRID_MIN, GRID_MAX)
    width = random.randint(GRID_MIN, GRID_MAX)
    body_color = random.choice(BODY_COLOR_CHOICES)
    n_h, n_v = random.choice(COUNT_CHOICES)
    want_pass_through = random.random() < 0.4

    h_rows = _sample_spaced_positions(NOZZLE_MARGIN, height - 1 - NOZZLE_MARGIN, n_h, POSITION_GAP)
    v_cols = _sample_spaced_positions(NOZZLE_MARGIN, width - 1 - NOZZLE_MARGIN, n_v, POSITION_GAP)
    if h_rows is None or v_cols is None:
        return None

    h_dirs = [random.choice(HORIZONTAL_DIRS) for _ in range(n_h)]
    v_dirs = [random.choice(VERTICAL_DIRS) for _ in range(n_v)]

    h_cols = []
    for direction in h_dirs:
        col = _choose_horizontal_col(v_cols, width, direction)
        if col is None:
            return None
        h_cols.append(col)

    v_rows = []
    for direction in v_dirs:
        row = _choose_vertical_row(h_rows, height, direction)
        if row is None:
            return None
        v_rows.append(row)

    base_specs = [
        (h_rows[i], h_cols[i], h_dirs[i]) for i in range(n_h)
    ] + [
        (v_rows[i], v_cols[i], v_dirs[i]) for i in range(n_v)
    ]

    variant_assignments = list(itertools.product((0, 1), repeat=len(base_specs)))
    random.shuffle(variant_assignments)
    for variants in variant_assignments:
        specs = [
            SpriteSpec(r, c, direction, variant)
            for (r, c, direction), variant in zip(base_specs, variants)
        ]
        if not _specs_are_separated(specs):
            continue
        grid = _render_input((height, width), body_color, specs)
        if _validate_generated(grid, specs, want_pass_through):
            return grid

    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid invisible-ray intersection input")
