from __future__ import annotations

import random

import numpy as np


BG = 0
GRAY = 5
TEAL = 8
SPRITE_COLORS = [1, 2, 3, 4, 6, 7, 9]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_GEN_COUNTER = 0


def _neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _ordered_path(component: set[tuple[int, int]], start: tuple[int, int]) -> list[tuple[int, int]]:
    degree = {
        cell: sum((nr, nc) in component for nr, nc in _neighbors4(cell[0], cell[1]))
        for cell in component
    }
    endpoints = [cell for cell, deg in degree.items() if deg == 1]
    assert len(endpoints) == 2, "wire must be a simple path with two endpoints"
    assert start in endpoints, "ordered traversal must start from an endpoint"

    path = [start]
    prev: tuple[int, int] | None = None
    cur = start
    while True:
        nxts = [nb for nb in _neighbors4(cur[0], cur[1]) if nb in component and nb != prev]
        if not nxts:
            break
        assert len(nxts) == 1, "wire path must be non-branching"
        prev, cur = cur, nxts[0]
        path.append(cur)
    assert len(path) == len(component), "wire traversal must cover the whole path"
    return path


def _extract_sprite(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sprite_mask = ~np.isin(grid, [BG, GRAY, TEAL])
    assert sprite_mask.any(), "expected one colored sprite"
    comps = _components(sprite_mask)
    assert len(comps) == 1, "expected one sprite component"
    coords = np.argwhere(sprite_mask).astype(int)
    colors = grid[sprite_mask].astype(int)
    return sprite_mask, coords, colors


def _find_wire_component(grid: np.ndarray) -> set[tuple[int, int]]:
    wire_mask = np.isin(grid, [GRAY, TEAL])
    comps = _components(wire_mask)
    wire_comps = [comp for comp in comps if any(grid[r, c] == TEAL for r, c in comp)]
    assert len(wire_comps) == 1, "expected exactly one wire component containing teal"
    return set(wire_comps[0])


def _reflect(coords: np.ndarray, pivot: tuple[int, int]) -> np.ndarray:
    pr, pc = pivot
    pivot_vec = np.array([pr, pc], dtype=int)
    return 2 * pivot_vec - coords


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reflect the multicolor sprite through each gray pivot along the teal wire, keeping only the final image.

    Input structure:
    - A black background contains one connected multicolor sprite.
    - A single-pixel teal wire starts near the sprite and includes several gray pivot pixels in order.
    - Additional gray pixels may appear away from the wire as distractors and do not belong to the pivot sequence.

    Rule:
    - Find the teal-and-gray wire component, choose the endpoint nearest the sprite, and traverse the wire from that end.
    - Whenever a traversed wire cell is gray, reflect the entire sprite by 180 degrees about that pivot cell, using the previous reflected image as the next input.
    - Clear the original sprite location and draw only the final reflected sprite, while leaving the wire, pivots, and stray gray distractors unchanged.

    Key insight:
    - The gray pivots matter only in the order induced by the teal path; gray cells outside that connected wire are distractors and must be ignored.
    """
    g = np.asarray(grid, dtype=int)
    sprite_mask, sprite_coords, sprite_colors = _extract_sprite(g)
    wire_component = _find_wire_component(g)
    degree = {
        cell: sum((nr, nc) in wire_component for nr, nc in _neighbors4(cell[0], cell[1]))
        for cell in wire_component
    }
    endpoints = [cell for cell, deg in degree.items() if deg == 1]
    assert len(endpoints) == 2, "wire must have two endpoints"

    sprite_cells = [tuple(map(int, cell)) for cell in sprite_coords.tolist()]
    def endpoint_distance(ep: tuple[int, int]) -> int:
        return min(abs(ep[0] - r) + abs(ep[1] - c) for r, c in sprite_cells)

    start = min(endpoints, key=endpoint_distance)
    path = _ordered_path(wire_component, start)
    pivots = [cell for cell in path if int(g[cell]) == GRAY]
    assert pivots, "wire must contain at least one pivot"

    final_coords = sprite_coords.copy()
    for pivot in pivots:
        final_coords = _reflect(final_coords, pivot)

    h, w = g.shape
    out = g.copy()
    out[sprite_mask] = BG
    for (r, c), color in zip(final_coords.tolist(), sprite_colors.tolist()):
        rr = int(r)
        cc = int(c)
        assert 0 <= rr < h and 0 <= cc < w, "final sprite left the canvas"
        assert out[rr, cc] == BG, "final sprite must not overwrite wire or distractors"
        out[rr, cc] = int(color)
    return out


def _tight_crop(cells: dict[tuple[int, int], int]) -> np.ndarray:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for (r, c), color in cells.items():
        patch[r - r0, c - c0] = color
    return patch


def _make_sprite() -> dict[tuple[int, int], int] | None:
    h = random.randint(2, 5)
    w = random.randint(2, 5)
    max_cells = min(h * w - 1, 8)
    if max_cells < 3:
        return None
    target = random.randint(3, max_cells)
    cells = {(random.randrange(h), random.randrange(w))}
    while len(cells) < target:
        frontier = []
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        cells.add(random.choice(frontier))

    color_count = random.randint(2, min(4, len(cells)))
    palette = random.sample(SPRITE_COLORS, color_count)
    colored = {cell: random.choice(palette) for cell in cells}

    patch = _tight_crop(colored)
    if np.array_equal(patch, np.rot90(patch, 2)):
        return None
    if len(set(colored.values())) < 2:
        return None
    return colored


def _boundary_contact(
    sprite: dict[tuple[int, int], int], direction: tuple[int, int]
) -> tuple[int, int]:
    rows = [r for r, _ in sprite]
    cols = [c for _, c in sprite]
    dr, dc = direction
    if dr == 1:
        edge = min(rows)
        candidates = [cell for cell in sprite if cell[0] == edge]
    elif dr == -1:
        edge = max(rows)
        candidates = [cell for cell in sprite if cell[0] == edge]
    elif dc == 1:
        edge = min(cols)
        candidates = [cell for cell in sprite if cell[1] == edge]
    else:
        edge = max(cols)
        candidates = [cell for cell in sprite if cell[1] == edge]
    return random.choice(candidates)


def _build_path() -> list[tuple[int, int]] | None:
    segment_count = random.choices([1, 2, 3, 4], weights=[1, 3, 3, 2])[0]
    direction = random.choice(DIR4)
    directions = [direction]
    for _ in range(segment_count - 1):
        prev = directions[-1]
        options = [d for d in DIR4 if d != prev and d != (-prev[0], -prev[1])]
        directions.append(random.choice(options))

    cells = [(0, 0)]
    seen = {(0, 0)}
    cur = (0, 0)
    for idx, (dr, dc) in enumerate(directions):
        length = random.randint(3 if idx == 0 else 2, 6)
        for _ in range(length):
            cur = (cur[0] + dr, cur[1] + dc)
            if cur in seen:
                return None
            seen.add(cur)
            cells.append(cur)
    return cells


def _translate_cells(
    cells: dict[tuple[int, int], int], shift: tuple[int, int]
) -> dict[tuple[int, int], int]:
    sr, sc = shift
    return {(r + sr, c + sc): color for (r, c), color in cells.items()}


def _reflect_dict(
    cells: dict[tuple[int, int], int], pivots: list[tuple[int, int]]
) -> dict[tuple[int, int], int]:
    cur = dict(cells)
    for pr, pc in pivots:
        cur = {(2 * pr - r, 2 * pc - c): color for (r, c), color in cur.items()}
    return cur


def _min_manhattan(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> int:
    return min(abs(r1 - r2) + abs(c1 - c2) for r1, c1 in a for r2, c2 in b)


def _bbox(cells: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _build_case() -> tuple[np.ndarray, np.ndarray] | None:
    global _GEN_COUNTER

    path = _build_path()
    if path is None or len(path) < 5:
        return None
    initial_dir = (path[1][0] - path[0][0], path[1][1] - path[0][1])

    sprite = _make_sprite()
    if sprite is None:
        return None
    contact = _boundary_contact(sprite, initial_dir)
    desired_contact = (-initial_dir[0], -initial_dir[1])
    sprite = _translate_cells(sprite, (desired_contact[0] - contact[0], desired_contact[1] - contact[1]))

    sprite_positions = set(sprite)
    path_positions = set(path)
    if sprite_positions & path_positions:
        return None
    if _min_manhattan(sprite_positions, {path[0]}) != 1:
        return None
    if _min_manhattan(sprite_positions, path_positions - {path[0]}) < 2:
        return None
    if _min_manhattan(sprite_positions, {path[-1]}) <= 2:
        return None

    _GEN_COUNTER += 1
    parity = _GEN_COUNTER % 2
    pivot_choices = [1, 3] if parity == 0 else [2, 4]
    pivot_choices = [k for k in pivot_choices if k <= len(path) - 2]
    if not pivot_choices:
        return None
    pivot_count = random.choice(pivot_choices)
    pivot_indices = sorted(random.sample(range(1, len(path) - 1), pivot_count))
    pivots = [path[idx] for idx in pivot_indices]

    final_sprite = _reflect_dict(sprite, pivots)
    final_positions = set(final_sprite)
    if len(final_positions) != len(sprite):
        return None
    if final_positions & path_positions:
        return None
    if final_positions & sprite_positions:
        return None
    if _min_manhattan(final_positions, path_positions) < 2:
        return None

    all_positions = sprite_positions | final_positions | path_positions
    r0, c0, r1, c1 = _bbox(all_positions)
    span_h = r1 - r0 + 1
    span_w = c1 - c0 + 1
    if span_h > 24 or span_w > 24:
        return None

    h_lo, h_hi = span_h + 2, min(30, span_h + 7)
    w_lo, w_hi = span_w + 2, min(30, span_w + 7)
    if h_lo > h_hi or w_lo > w_hi:
        return None
    h = random.randint(h_lo, h_hi)
    w = random.randint(w_lo, w_hi)
    shift_r = random.randint(1 - r0, h - 2 - r1)
    shift_c = random.randint(1 - c0, w - 2 - c1)
    shift = (shift_r, shift_c)

    abs_sprite = _translate_cells(sprite, shift)
    abs_final = _translate_cells(final_sprite, shift)
    abs_path = [(r + shift_r, c + shift_c) for r, c in path]
    abs_pivot_set = {(r + shift_r, c + shift_c) for r, c in pivots}
    occupied = set(abs_sprite) | set(abs_final) | set(abs_path)

    grid = np.full((h, w), BG, dtype=int)
    expected = np.full((h, w), BG, dtype=int)
    for (r, c), color in abs_sprite.items():
        grid[r, c] = color
    for r, c in abs_path:
        grid[r, c] = GRAY if (r, c) in abs_pivot_set else TEAL
        expected[r, c] = grid[r, c]
    for (r, c), color in abs_final.items():
        expected[r, c] = color

    stray_target = random.randint(1, 4)
    stray_positions: list[tuple[int, int]] = []
    attempts = 0
    while len(stray_positions) < stray_target and attempts < 200:
        attempts += 1
        cand = (random.randrange(h), random.randrange(w))
        if cand in occupied or cand in stray_positions:
            continue
        if any(max(abs(cand[0] - r), abs(cand[1] - c)) <= 1 for r, c in occupied):
            continue
        stray_positions.append(cand)
    if len(stray_positions) != stray_target:
        return None
    for r, c in stray_positions:
        grid[r, c] = GRAY
        expected[r, c] = GRAY

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(400):
        case = _build_case()
        if case is not None:
            return case[0]
    raise RuntimeError("failed to generate valid input for fable/8/019/2")
