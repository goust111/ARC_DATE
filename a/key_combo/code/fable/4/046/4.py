from __future__ import annotations

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
TEMPLATE_COLORS = np.array([3, 4, 5, 6, 7, 8, 9], dtype=int)

GRID_MIN = 18
GRID_MAX = 30
PATH_LEN_RANGE = (4, 7)
DISTRACTOR_CHOICES = (0, 1, 2, 3)
DISTRACTOR_WEIGHTS = np.array([0.2, 0.25, 0.35, 0.2], dtype=float)
STEP_MIN = 4
STEP_MAX = 7
FIELD_MARGIN = 5
GEN_RETRIES = 400


Coord = tuple[int, int]


def _bg_color(grid: np.ndarray) -> int:
    values = np.asarray(grid, dtype=int).ravel()
    return int(np.bincount(values).argmax())


def _rotate_clockwise(sprite: np.ndarray, quarter_turns: int) -> np.ndarray:
    return np.rot90(sprite, -(quarter_turns % 4))


def _find_template(grid: np.ndarray, bg: int) -> tuple[np.ndarray, tuple[int, int]]:
    H, W = grid.shape
    assert H >= 5 and W >= 5, "grid too small for a corner template"

    candidates: list[tuple[np.ndarray, tuple[int, int]]] = []

    def consider(r0: int, c0: int, sep_row: int, row_cols: slice, sep_col: int, col_rows: slice) -> None:
        block = grid[r0 : r0 + 3, c0 : c0 + 3]
        if block.shape != (3, 3) or np.any(block == bg):
            return
        if np.all(grid[sep_row, row_cols] == bg) and np.all(grid[col_rows, sep_col] == bg):
            candidates.append((block.copy(), (r0, c0)))

    consider(0, 0, 3, slice(0, 4), 3, slice(0, 4))
    consider(0, W - 3, 3, slice(W - 4, W), W - 4, slice(0, 4))
    consider(H - 3, 0, H - 4, slice(0, 4), 3, slice(H - 4, H))
    consider(H - 3, W - 3, H - 4, slice(W - 4, W), W - 4, slice(H - 4, H))

    assert len(candidates) == 1, "expected exactly one corner template"
    return candidates[0]


def _visible(grid: np.ndarray, bg: int, a: Coord, b: Coord) -> int | None:
    ar, ac = a
    br, bc = b
    if ar == br:
        step = 1 if bc > ac else -1
        for c in range(ac + step, bc, step):
            if int(grid[ar, c]) != bg:
                return None
        return abs(bc - ac)
    if ac == bc:
        step = 1 if br > ar else -1
        for r in range(ar + step, br, step):
            if int(grid[r, ac]) != bg:
                return None
        return abs(br - ar)
    return None


def _walk_targets(grid: np.ndarray, bg: int, start: Coord, targets: list[Coord]) -> list[Coord]:
    current = start
    remaining = list(targets)
    visited: list[Coord] = []

    while True:
        candidates = []
        for target in remaining:
            distance = _visible(grid, bg, current, target)
            if distance is not None:
                candidates.append((distance, target))
        if not candidates:
            break
        best = min(distance for distance, _target in candidates)
        nearest = [target for distance, target in candidates if distance == best]
        assert len(nearest) == 1, "expected a unique nearest visible target"
        chosen = nearest[0]
        visited.append(chosen)
        remaining.remove(chosen)
        current = chosen

    return visited


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild the nearest-visible walk from the red start and stamp rotated template copies onto the visited blue markers.

    Input structure:
    - The background is black.
    - One corner contains a separated 3x3 multicolor asymmetric template sprite.
    - The playfield contains one red start pixel and several blue target pixels.

    Rule:
    - Identify the corner template, the red start, and all blue targets outside the template.
    - From the start, repeatedly move to the unique nearest unvisited blue target visible in the same row or column through only black cells, using visibility from the input layout.
    - At the k-th visited target, replace that marker with the template centered there and rotated clockwise by (k mod 4) quarter-turns.

    Key insight:
    - Template orientation does not come from target position; it encodes the hidden visit order of a greedy line-of-sight walk, and unreachable blue markers remain untouched.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    assert bg == BLACK, "this puzzle uses black background"

    template, (tr0, tc0) = _find_template(g, bg)
    non_template = np.ones_like(g, dtype=bool)
    non_template[tr0 : tr0 + 3, tc0 : tc0 + 3] = False

    starts = np.argwhere((g == RED) & non_template)
    targets = np.argwhere((g == BLUE) & non_template)
    assert starts.shape == (1, 2), "expected one red start marker"
    assert targets.shape[0] >= 1, "expected blue targets"

    start = tuple(map(int, starts[0]))
    target_list = [tuple(map(int, rc)) for rc in targets]
    visited = _walk_targets(g, bg, start, target_list)

    out = g.copy()
    template_mask = template != bg
    center_mask = template_mask.copy()
    center_mask[1, 1] = False

    for k, (r, c) in enumerate(visited, start=1):
        stamp = _rotate_clockwise(template, k)
        r0 = r - 1
        c0 = c - 1
        assert 0 <= r0 and r0 + 3 <= g.shape[0]
        assert 0 <= c0 and c0 + 3 <= g.shape[1]
        base = g[r0 : r0 + 3, c0 : c0 + 3]
        assert base.shape == (3, 3)
        assert np.all(base[center_mask] == bg), "stamp halo must land on background"
        assert int(base[1, 1]) == BLUE, "visited cell must be a blue marker"
        window = out[r0 : r0 + 3, c0 : c0 + 3]
        window[template_mask] = stamp[template_mask]

    return out


def _field_bounds(shape: tuple[int, int], corner: str) -> tuple[int, int, int, int]:
    H, W = shape
    r_min, r_max = 1, H - 2
    c_min, c_max = 1, W - 2
    if corner[0] == "t":
        r_min = max(r_min, FIELD_MARGIN)
    else:
        r_max = min(r_max, H - FIELD_MARGIN - 1)
    if corner[1] == "l":
        c_min = max(c_min, FIELD_MARGIN)
    else:
        c_max = min(c_max, W - FIELD_MARGIN - 1)
    return r_min, r_max, c_min, c_max


def _chebyshev_clear(point: Coord, others: list[Coord], radius: int = 2) -> bool:
    pr, pc = point
    return all(max(abs(pr - orow), abs(pc - ocol)) > radius for orow, ocol in others)


def _random_template(rng: np.random.Generator) -> np.ndarray:
    for _ in range(200):
        block = rng.choice(TEMPLATE_COLORS, size=(3, 3), replace=True).astype(int)
        if len(set(map(int, block.ravel()))) < 3:
            continue
        if any(np.array_equal(block, np.rot90(block, turns)) for turns in range(1, 4)):
            continue
        return block
    raise RuntimeError("failed to build an asymmetric template")


def _build_path(
    rng: np.random.Generator,
    shape: tuple[int, int],
    corner: str,
    path_len: int,
) -> list[Coord] | None:
    r_min, r_max, c_min, c_max = _field_bounds(shape, corner)
    rows = list(range(r_min, r_max + 1))
    cols = list(range(c_min, c_max + 1))
    if len(rows) < 4 or len(cols) < 4:
        return None

    start_axis = str(rng.choice(np.array(["h", "v"])))
    for _ in range(200):
        start = (int(rng.choice(rows)), int(rng.choice(cols)))
        path = [start]
        used_rows = {start[0]}
        used_cols = {start[1]}
        axis = start_axis

        for _step in range(path_len):
            current = path[-1]
            candidates: list[Coord] = []
            if axis == "h":
                for new_c in cols:
                    if new_c in used_cols:
                        continue
                    if not (STEP_MIN <= abs(new_c - current[1]) <= STEP_MAX):
                        continue
                    candidate = (current[0], new_c)
                    if _chebyshev_clear(candidate, path):
                        candidates.append(candidate)
            else:
                for new_r in rows:
                    if new_r in used_rows:
                        continue
                    if not (STEP_MIN <= abs(new_r - current[0]) <= STEP_MAX):
                        continue
                    candidate = (new_r, current[1])
                    if _chebyshev_clear(candidate, path):
                        candidates.append(candidate)

            if not candidates:
                break

            chosen = candidates[int(rng.integers(len(candidates)))]
            path.append(chosen)
            used_rows.add(chosen[0])
            used_cols.add(chosen[1])
            axis = "v" if axis == "h" else "h"
        else:
            return path

    return None


def _place_distractors(
    rng: np.random.Generator,
    shape: tuple[int, int],
    corner: str,
    blocked_rows: set[int],
    blocked_cols: set[int],
    occupied: list[Coord],
    count: int,
) -> list[Coord] | None:
    r_min, r_max, c_min, c_max = _field_bounds(shape, corner)
    rows = [r for r in range(r_min, r_max + 1) if r not in blocked_rows]
    cols = [c for c in range(c_min, c_max + 1) if c not in blocked_cols]
    distractors: list[Coord] = []
    used_rows = set(blocked_rows)
    used_cols = set(blocked_cols)

    for _ in range(count):
        candidates = [
            (r, c)
            for r in rows
            for c in cols
            if r not in used_rows
            and c not in used_cols
            and _chebyshev_clear((r, c), occupied + distractors)
        ]
        if not candidates:
            return None
        chosen = candidates[int(rng.integers(len(candidates)))]
        distractors.append(chosen)
        used_rows.add(chosen[0])
        used_cols.add(chosen[1])

    return distractors


def _paint_template(grid: np.ndarray, template: np.ndarray, corner: str) -> None:
    H, W = grid.shape
    if corner == "tl":
        r0, c0 = 0, 0
    elif corner == "tr":
        r0, c0 = 0, W - 3
    elif corner == "bl":
        r0, c0 = H - 3, 0
    else:
        r0, c0 = H - 3, W - 3
    grid[r0 : r0 + 3, c0 : c0 + 3] = template


def _attempt_generate() -> np.ndarray | None:
    rng = np.random.default_rng()
    H = int(rng.integers(GRID_MIN, GRID_MAX + 1))
    W = int(rng.integers(GRID_MIN, GRID_MAX + 1))
    corner = str(rng.choice(np.array(["tl", "tr", "bl", "br"])))
    path_len = int(rng.integers(PATH_LEN_RANGE[0], PATH_LEN_RANGE[1] + 1))

    path = _build_path(rng, (H, W), corner, path_len)
    if path is None:
        return None

    distractor_count = int(rng.choice(np.arange(len(DISTRACTOR_CHOICES)), p=DISTRACTOR_WEIGHTS))
    distractors = _place_distractors(
        rng,
        (H, W),
        corner,
        {r for r, _c in path},
        {c for _r, c in path},
        path,
        DISTRACTOR_CHOICES[distractor_count],
    )
    if distractors is None:
        return None

    grid = np.zeros((H, W), dtype=int)
    template = _random_template(rng)
    _paint_template(grid, template, corner)

    start = path[0]
    grid[start] = RED
    for point in path[1:] + distractors:
        grid[point] = BLUE

    visited = _walk_targets(grid, BLACK, start, path[1:] + distractors)
    if visited != path[1:]:
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    if out.shape != grid.shape or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid walking-compass input")
