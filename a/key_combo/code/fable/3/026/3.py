from __future__ import annotations

from collections import deque

import numpy as np


BLACK = 0
YELLOW = 4
ORANGE = 7
PORTAL_COLORS = (1, 2, 3, 5, 6, 8, 9)
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

MAX_GENERATION_ATTEMPTS = 500


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    H, W = shape
    return 0 <= int(r) < H and 0 <= int(c) < W


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield int(r) + dr, int(c) + dc


def _components(mask: np.ndarray) -> list[np.ndarray]:
    H, W = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(H):
        for c in range(W):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_portals(g: np.ndarray) -> tuple[list[dict], dict[int, int], np.ndarray]:
    rectangles: list[dict] = []
    by_color: dict[int, list[int]] = {}

    for color in sorted(set(map(int, np.unique(g))) - {BLACK, YELLOW, ORANGE}):
        assert color in PORTAL_COLORS, "portal colors must not be seed colors"
        comps = _components(g == color)
        assert len(comps) == 2, "each portal color must appear as exactly two rectangles"
        dims: list[tuple[int, int]] = []
        indices: list[int] = []
        for coords in comps:
            r0, c0, r1, c1 = _bbox(coords)
            h, w = r1 - r0 + 1, c1 - c0 + 1
            assert int(coords.shape[0]) == h * w, "portal must be a solid rectangle"
            assert np.all(g[r0 : r1 + 1, c0 : c1 + 1] == color), "portal rectangle must be solid"
            indices.append(len(rectangles))
            dims.append((h, w))
            rectangles.append(
                {
                    "color": int(color),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "height": int(h),
                    "width": int(w),
                }
            )
        assert dims[0] == dims[1], "same-color portal rectangles must have the same size"
        by_color[int(color)] = indices

    assert rectangles, "expected at least one portal pair"

    rect_at = np.full(g.shape, -1, dtype=int)
    for idx, rect in enumerate(rectangles):
        r0, c0, r1, c1 = rect["bbox"]
        assert np.all(rect_at[r0 : r1 + 1, c0 : c1 + 1] == -1), "portal rectangles must not overlap"
        rect_at[r0 : r1 + 1, c0 : c1 + 1] = idx

    partner: dict[int, int] = {}
    for indices in by_color.values():
        a, b = indices
        partner[a] = b
        partner[b] = a
    return rectangles, partner, rect_at


def _find_seeds(g: np.ndarray) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    heads = [tuple(map(int, rc)) for rc in np.argwhere(g == YELLOW)]
    tails = {tuple(map(int, rc)) for rc in np.argwhere(g == ORANGE)}
    assert heads, "expected at least one yellow seed head"
    assert tails, "expected orange seed tails"

    used_tails: dict[tuple[int, int], tuple[int, int]] = {}
    seeds: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for head in heads:
        adjacent_tails = [p for p in _neighbors4(*head) if p in tails]
        assert len(adjacent_tails) == 1, "each yellow head must touch one orange tail"
        tail = adjacent_tails[0]
        assert tail not in used_tails, "each orange tail belongs to one seed"
        used_tails[tail] = head
        dr, dc = head[0] - tail[0], head[1] - tail[1]
        assert abs(dr) + abs(dc) == 1, "seed head and tail must be adjacent"
        seeds.append((head, tail))

    assert set(used_tails) == tails, "every orange cell must be a seed tail"
    return seeds


def _draw_rays(g: np.ndarray, rectangles: list[dict], partner: dict[int, int], rect_at: np.ndarray) -> np.ndarray:
    out = g.copy()
    seeds = _find_seeds(g)

    for head, tail in seeds:
        dr, dc = head[0] - tail[0], head[1] - tail[1]
        current = head
        seen_states: set[tuple[int, int, int, int]] = set()

        while True:
            state = (int(current[0]), int(current[1]), int(dr), int(dc))
            assert state not in seen_states, "ray must terminate instead of cycling"
            seen_states.add(state)

            nr, nc = current[0] + dr, current[1] + dc
            if not _in_bounds(g.shape, nr, nc):
                break

            rect_idx = int(rect_at[nr, nc])
            if rect_idx != -1:
                rect = rectangles[rect_idx]
                r0, c0, _r1, _c1 = rect["bbox"]
                off_r, off_c = nr - r0, nc - c0
                exit_rect = rectangles[partner[rect_idx]]
                er0, ec0, er1, ec1 = exit_rect["bbox"]
                tr, tc = er0 + off_r, ec0 + off_c
                assert er0 <= tr <= er1 and ec0 <= tc <= ec1

                current = (int(tr), int(tc))
                while True:
                    sr, sc = current[0] + dr, current[1] + dc
                    if not _in_bounds(g.shape, sr, sc) or int(rect_at[sr, sc]) != partner[rect_idx]:
                        break
                    current = (int(sr), int(sc))
                continue

            if int(g[nr, nc]) == BLACK:
                out[nr, nc] = YELLOW
            else:
                assert int(g[nr, nc]) in (YELLOW, ORANGE), "rays may only cross seed cells outside portals"
            current = (int(nr), int(nc))

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Draw yellow ray paths that teleport between paired same-color rectangles.

    Input structure:
    - The grid has a black background with several pairs of same-color solid
      rectangles. Each non-seed color appears in exactly two equal-size
      rectangles, and that color uniquely identifies a portal pair.
    - One or more two-cell seeds are present: a yellow head touching an orange
      tail. The ray direction is from the orange tail toward the yellow head.

    Rule:
    - From every seed head, advance one cell at a time in its direction and draw
      yellow on each black cell visited.
    - When the ray enters a portal rectangle, do not draw inside it. Instead,
      move to the same relative offset inside the rectangle's same-color partner,
      keep the direction unchanged, skip through that partner's interior, and
      continue drawing after the ray exits it.
    - Stop at the canvas edge. Existing rectangles and seed cells remain
      unchanged.

    Key insight:
    The rectangles are paired portals rather than obstacles: the entry cell's
    relative offset within one rectangle determines the exit offset in its
    same-color partner, so separated yellow line segments are one continuous ray.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must fit within 30x30"
    rectangles, partner, rect_at = _find_portals(g)
    return _draw_rays(g, rectangles, partner, rect_at)


def _paint_rect(grid: np.ndarray, r0: int, c0: int, h: int, w: int, color: int) -> bool:
    if r0 < 0 or c0 < 0 or r0 + h > grid.shape[0] or c0 + w > grid.shape[1]:
        return False
    if np.any(grid[r0 : r0 + h, c0 : c0 + w] != BLACK):
        return False
    grid[r0 : r0 + h, c0 : c0 + w] = int(color)
    return True


def _line_cells_right(row: int, c0: int, c1: int) -> list[tuple[int, int]]:
    if c0 > c1:
        return []
    return [(int(row), int(c)) for c in range(int(c0), int(c1) + 1)]


def _canonical_scene(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    n_pairs = 3 if float(rng.random()) < 0.40 else 2
    H = int(rng.integers(14, 27))
    min_w = 20 if n_pairs == 2 else 28
    W = int(rng.integers(min_w, 31))
    grid = np.zeros((H, W), dtype=int)
    path: set[tuple[int, int]] = set()
    colors = [int(c) for c in rng.choice(np.array(PORTAL_COLORS), size=n_pairs, replace=False)]

    start_row = int(rng.integers(3, H - 3))
    head_col = int(rng.integers(1, 3))
    tail = (start_row, head_col - 1)
    head = (start_row, head_col)
    grid[tail] = ORANGE
    grid[head] = YELLOW

    current_r, current_c = head
    used_rows = {current_r}
    interior_offset_seen = False
    row_change_seen = False

    for j, color in enumerate(colors):
        options: list[tuple[int, int, int]] = []
        max_w_exclusive = 4 if n_pairs == 3 else 5
        for h in range(1, 5):
            for w in range(2, max_w_exclusive):
                for off in range(h):
                    r0 = current_r - off
                    if 0 <= r0 <= H - h:
                        if j == 0 and not (h >= 3 and 0 < off < h - 1):
                            continue
                        options.append((h, w, off))
        if not options:
            return None
        h, w, off = options[int(rng.integers(len(options)))]
        interior_offset_seen = interior_offset_seen or (h >= 3 and 0 < off < h - 1)

        entry_gap = int(rng.integers(2, 4 if n_pairs == 3 else 5))
        teleport_gap = int(rng.integers(2, 4 if n_pairs == 3 else 7))
        cA = current_c + entry_gap
        cB = cA + w + teleport_gap
        new_current_c = cB + w - 1
        remaining = n_pairs - j - 1
        min_future_span = remaining * 7 + 1
        if new_current_c + min_future_span >= W:
            return None

        rA = current_r - off
        possible_rB = [
            r
            for r in range(0, H - h + 1)
            if r + off != current_r and abs((r + off) - current_r) >= 2
        ]
        if not possible_rB:
            return None
        rB = int(possible_rB[int(rng.integers(len(possible_rB)))])
        next_r = rB + off
        row_change_seen = row_change_seen or (next_r != current_r)
        used_rows.add(int(next_r))

        segment = _line_cells_right(current_r, current_c + 1, cA - 1)
        if any(grid[r, c] != BLACK for r, c in segment):
            return None
        path.update(segment)

        if not _paint_rect(grid, rA, cA, h, w, color):
            return None
        if not _paint_rect(grid, rB, cB, h, w, color):
            return None

        current_r = int(next_r)
        current_c = int(new_current_c)

    final_segment = _line_cells_right(current_r, current_c + 1, W - 1)
    if len(final_segment) < 2 or any(grid[r, c] != BLACK for r, c in final_segment):
        return None
    path.update(final_segment)

    _add_extra_straight_seeds(rng, grid, path)

    if not interior_offset_seen or not row_change_seen or len(used_rows) < 2:
        return None
    if any(grid[r, c] != BLACK for r, c in path):
        return None

    expected = grid.copy()
    for r, c in path:
        expected[r, c] = YELLOW
    return grid, expected


def _add_extra_straight_seeds(rng: np.random.Generator, grid: np.ndarray, path: set[tuple[int, int]]) -> None:
    H, W = grid.shape
    target = int(rng.choice(np.array([0, 1, 1, 2])))
    rows = [r for r in range(1, H - 1) if np.all(grid[r, :] == BLACK) and not any(rr == r for rr, _c in path)]
    rng.shuffle(rows)

    added = 0
    for row in rows:
        if added >= target:
            break
        go_right = bool(rng.integers(2))
        if go_right:
            head_c = int(rng.integers(1, min(5, W - 1)))
            tail_c = head_c - 1
            ray = [(row, c) for c in range(head_c + 1, W)]
        else:
            head_c = int(rng.integers(max(1, W - 5), W - 1))
            tail_c = head_c + 1
            if tail_c >= W:
                continue
            ray = [(row, c) for c in range(0, head_c)]

        cells = [(row, head_c), (row, tail_c), *ray]
        if any(grid[r, c] != BLACK for r, c in cells):
            continue
        if any((r, c) in path for r, c in cells):
            continue
        grid[row, tail_c] = ORANGE
        grid[row, head_c] = YELLOW
        path.update(ray)
        added += 1


def _transform_grid(g: np.ndarray, kind: int) -> np.ndarray:
    if kind == 0:
        out = g
    elif kind == 1:
        out = np.rot90(g, 1)
    elif kind == 2:
        out = np.rot90(g, 2)
    elif kind == 3:
        out = np.rot90(g, 3)
    elif kind == 4:
        out = np.fliplr(g)
    elif kind == 5:
        out = np.flipud(g)
    elif kind == 6:
        out = g.T
    else:
        out = np.fliplr(np.flipud(g)).T
    return out.copy()


def _straight_no_teleport_output(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    seeds = _find_seeds(g)
    for head, tail in seeds:
        dr, dc = head[0] - tail[0], head[1] - tail[1]
        r, c = head
        while True:
            r += dr
            c += dc
            if not _in_bounds(g.shape, r, c):
                break
            if int(g[r, c]) == BLACK:
                out[r, c] = YELLOW
    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    scene = _canonical_scene(rng)
    if scene is None:
        return None
    inp, expected = scene

    transform = int(rng.integers(0, 8))
    inp = _transform_grid(inp, transform)
    expected = _transform_grid(expected, transform)

    try:
        actual = main(inp)
        straight = _straight_no_teleport_output(inp)
        _find_seeds(inp)
        rectangles, partner, rect_at = _find_portals(inp)
    except AssertionError:
        return None

    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(straight, expected):
        return None
    if len(rectangles) < 4 or len(partner) != len(rectangles):
        return None
    if np.any((inp != BLACK) & (actual != inp)):
        return None
    if not np.any((inp == BLACK) & (actual == YELLOW)):
        return None

    # At least one portal interior must be skipped in the output, proving that
    # rectangles are not repainted as ordinary ray cells.
    for rect in rectangles:
        r0, c0, r1, c1 = rect["bbox"]
        if np.any(rect_at[r0 : r1 + 1, c0 : c1 + 1] == -1):
            return None
        if np.any(actual[r0 : r1 + 1, c0 : c1 + 1] != inp[r0 : r1 + 1, c0 : c1 + 1]):
            return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid portal-ray input")
