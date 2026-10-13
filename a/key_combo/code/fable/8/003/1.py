from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
SEED = 2
ANCHOR = 3
BODY = 8
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_GENERATION_ATTEMPTS = 600


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= int(r) < int(h) and 0 <= int(c) < int(w)


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in DIRS4:
                    nr = rr + dr
                    nc = cc + dc
                    if in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def find_components(
    grid: np.ndarray,
    *,
    bg: int | None = None,
    monochromatic: bool = True,
) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    if bg is None:
        bg = get_bg(g)
    bg = int(bg)
    out: list[dict] = []
    if monochromatic:
        for color in sorted(set(map(int, g.flat)) - {bg}):
            for coords in _components_from_mask(g == color):
                out.append(
                    {
                        "color": int(color),
                        "coords": coords,
                        "bbox": bbox_coords(coords),
                        "area": int(len(coords)),
                    }
                )
    else:
        for coords in _components_from_mask(g != bg):
            out.append(
                {
                    "color": None,
                    "coords": coords,
                    "bbox": bbox_coords(coords),
                    "area": int(len(coords)),
                }
            )
    return out


def _stamp_cells(
    out: np.ndarray,
    rel_cells: list[tuple[int, int, int]],
    anchor_rc: tuple[int, int],
) -> None:
    ar, ac = anchor_rc
    for dr, dc, color in rel_cells:
        out[ar + dr, ac + dc] = int(color)


def _fits(shape: tuple[int, int], rel_cells: list[tuple[int, int, int]], anchor_rc: tuple[int, int]) -> bool:
    ar, ac = anchor_rc
    return all(in_bounds(shape, ar + dr, ac + dc) for dr, dc, _ in rel_cells)


def _chain_length(shape: tuple[int, int], rel_cells: list[tuple[int, int, int]], start: tuple[int, int], step: tuple[int, int]) -> int:
    sr, sc = start
    dr, dc = step
    count = 0
    while _fits(shape, rel_cells, (sr, sc)):
        count += 1
        sr += dr
        sc += dc
        if count > 12:
            break
    return count


def _solve_from_spec(
    grid: np.ndarray,
    rel_cells: list[tuple[int, int, int]],
    start: tuple[int, int],
    step: tuple[int, int],
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    cur = tuple(map(int, start))
    while _fits(out.shape, rel_cells, cur):
        _stamp_cells(out, rel_cells, cur)
        cur = (cur[0] + int(step[0]), cur[1] + int(step[1]))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Chain-stamp the marked sprite from the isolated red seed until the next copy would leave the canvas.

    Input structure:
    - Black cells are background.
    - One 4-connected multicolor sprite contains cyan body cells together with exactly one green anchor pixel and one red seed pixel replacing two body cells.
    - Elsewhere there is exactly one isolated red pixel, separate from the sprite.

    Rule:
    - Identify the unique sprite component containing the green anchor.
    - Record every sprite cell as a relative offset from the green anchor, keeping each cell's color.
    - Paste one full copy with the anchor aligned to the isolated red pixel, then repeatedly paste another copy with its anchor aligned to the previous copy's red seed.
    - Stop before the first copy whose any cell would go out of bounds; keep the original sprite and every completed copy.

    Key insight:
    - The green pixel says where to attach each copy, while the red pixel says where the next attachment point will appear; their relative displacement is the fixed step vector of the whole chain.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = get_bg(g)
    assert int(np.count_nonzero(g == ANCHOR)) == 1
    assert int(np.count_nonzero(g == SEED)) == 2

    components = find_components(g, bg=bg, monochromatic=False)
    sprite_candidates = []
    for comp in components:
        coords = comp["coords"]
        colors = set(map(int, g[coords[:, 0], coords[:, 1]]))
        if ANCHOR in colors:
            sprite_candidates.append(comp)
    assert len(sprite_candidates) == 1
    sprite = sprite_candidates[0]

    sprite_coords = sprite["coords"]
    sprite_set = {tuple(map(int, cell)) for cell in sprite_coords}
    sprite_colors = set(map(int, g[sprite_coords[:, 0], sprite_coords[:, 1]]))
    assert sprite["area"] >= 3
    assert sprite_colors <= {BODY, ANCHOR, SEED}
    assert BODY in sprite_colors

    anchor_coords = sprite_coords[g[sprite_coords[:, 0], sprite_coords[:, 1]] == ANCHOR]
    seed_coords = sprite_coords[g[sprite_coords[:, 0], sprite_coords[:, 1]] == SEED]
    assert anchor_coords.shape == (1, 2)
    assert seed_coords.shape == (1, 2)

    anchor_r, anchor_c = map(int, anchor_coords[0])
    seed_r, seed_c = map(int, seed_coords[0])
    step = (seed_r - anchor_r, seed_c - anchor_c)
    assert step != (0, 0)

    outside_red = [tuple(map(int, rc)) for rc in np.argwhere(g == SEED) if tuple(map(int, rc)) not in sprite_set]
    assert len(outside_red) == 1
    start = outside_red[0]
    for dr, dc in DIRS4:
        nr = start[0] + dr
        nc = start[1] + dc
        if in_bounds(g.shape, nr, nc):
            assert g[nr, nc] == bg

    rel_cells = [(int(r - anchor_r), int(c - anchor_c), int(g[r, c])) for r, c in sprite_coords]
    return _solve_from_spec(g, rel_cells, start, step)


def _random_connected_shape() -> list[tuple[int, int]]:
    for _ in range(200):
        h = random.randint(3, 6)
        w = random.randint(3, 6)
        target = random.randint(5, min(h * w - 1, 10))
        start = (random.randrange(h), random.randrange(w))
        cells = {start}
        frontier = set()
        for dr, dc in DIRS4:
            nr = start[0] + dr
            nc = start[1] + dc
            if 0 <= nr < h and 0 <= nc < w:
                frontier.add((nr, nc))
        while len(cells) < target and frontier:
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            cells.add(cell)
            for dr, dc in DIRS4:
                nr = cell[0] + dr
                nc = cell[1] + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        if len(set(rows)) < 2 or len(set(cols)) < 2:
            continue
        return sorted(cells)
    raise RuntimeError("failed to sample connected shape")


def _choose_controls(shape_cells: list[tuple[int, int]]) -> tuple[list[tuple[int, int, int]], tuple[int, int]] | None:
    ordered_pairs = [(a, b) for a in shape_cells for b in shape_cells if a != b]
    random.shuffle(ordered_pairs)
    cell_set = set(shape_cells)
    for anchor, seed in ordered_pairs:
        ar, ac = anchor
        sr, sc = seed
        step = (sr - ar, sc - ac)
        rel_set = {(r - ar, c - ac) for r, c in cell_set}
        shifted_once = {(r + step[0], c + step[1]) for r, c in rel_set}
        if rel_set & shifted_once != {step}:
            continue
        if any(rel_set & {(r + k * step[0], c + k * step[1]) for r, c in rel_set} for k in range(2, 5)):
            continue
        rel_cells: list[tuple[int, int, int]] = []
        for r, c in shape_cells:
            rel = (r - ar, c - ac)
            if (r, c) == anchor:
                color = ANCHOR
            elif (r, c) == seed:
                color = SEED
            else:
                color = BODY
            rel_cells.append((int(rel[0]), int(rel[1]), int(color)))
        return rel_cells, step
    return None


def _sample_sprite() -> tuple[list[tuple[int, int, int]], tuple[int, int]]:
    for _ in range(300):
        chosen = _choose_controls(_random_connected_shape())
        if chosen is not None:
            return chosen
    raise RuntimeError("failed to sample sprite controls")


def _candidate_starts(
    shape: tuple[int, int],
    rel_cells: list[tuple[int, int, int]],
    step: tuple[int, int],
    desired_copies: int,
) -> list[tuple[int, int]]:
    rows = [dr for dr, _, _ in rel_cells]
    cols = [dc for _, dc, _ in rel_cells]
    r_lo = -min(rows)
    r_hi = shape[0] - 1 - max(rows)
    c_lo = -min(cols)
    c_hi = shape[1] - 1 - max(cols)
    starts = []
    for r in range(r_lo, r_hi + 1):
        for c in range(c_lo, c_hi + 1):
            if _chain_length(shape, rel_cells, (r, c), step) == desired_copies:
                starts.append((r, c))
    return starts


def _copy_footprint(
    shape: tuple[int, int],
    rel_cells: list[tuple[int, int, int]],
    start: tuple[int, int],
    step: tuple[int, int],
    copies: int,
) -> np.ndarray:
    occ = np.zeros(shape, dtype=bool)
    for k in range(copies):
        ar = start[0] + k * step[0]
        ac = start[1] + k * step[1]
        for dr, dc, _ in rel_cells:
            occ[ar + dr, ac + dc] = True
    return occ


def _dilate4(mask: np.ndarray, steps: int = 1) -> np.ndarray:
    out = mask.copy()
    for _ in range(steps):
        grown = out.copy()
        for r, c in np.argwhere(out):
            for dr, dc in DIRS4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                nr = int(r + dr)
                nc = int(c + dc)
                if in_bounds(out.shape, nr, nc):
                    grown[nr, nc] = True
        out = grown
    return out


def _original_anchor_candidates(
    shape: tuple[int, int],
    rel_cells: list[tuple[int, int, int]],
    forbidden: np.ndarray,
) -> list[tuple[int, int]]:
    rows = [dr for dr, _, _ in rel_cells]
    cols = [dc for _, dc, _ in rel_cells]
    r_lo = -min(rows)
    r_hi = shape[0] - 1 - max(rows)
    c_lo = -min(cols)
    c_hi = shape[1] - 1 - max(cols)
    candidates = []
    for r in range(r_lo, r_hi + 1):
        for c in range(c_lo, c_hi + 1):
            cells = [(r + dr, c + dc) for dr, dc, _ in rel_cells]
            if any(forbidden[rr, cc] for rr, cc in cells):
                continue
            candidates.append((r, c))
    return candidates


def _render_input(rel_cells: list[tuple[int, int, int]], original_anchor: tuple[int, int], chain_start: tuple[int, int], shape: tuple[int, int]) -> np.ndarray:
    inp = np.full(shape, BG, dtype=int)
    _stamp_cells(inp, rel_cells, original_anchor)
    inp[chain_start] = SEED
    return inp


def _validate_semantics(
    inp: np.ndarray,
    expected: np.ndarray,
    rel_cells: list[tuple[int, int, int]],
    chain_start: tuple[int, int],
    step: tuple[int, int],
    copies: int,
) -> bool:
    try:
        actual = main(inp)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False

    bg = get_bg(inp)
    comps = find_components(inp, bg=bg, monochromatic=False)
    if len(comps) != 2:
        return False
    if int(np.count_nonzero(inp == ANCHOR)) != 1 or int(np.count_nonzero(inp == SEED)) != 2:
        return False

    if copies > 1:
        one_copy = inp.copy()
        _stamp_cells(one_copy, rel_cells, chain_start)
        if np.array_equal(one_copy, expected):
            return False

    wrong_start = (chain_start[0] - step[0], chain_start[1] - step[1])
    if _fits(inp.shape, rel_cells, wrong_start):
        wrong = _solve_from_spec(inp, rel_cells, wrong_start, step)
        if np.array_equal(wrong, expected):
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    rel_cells, step = _sample_sprite()
    desired_copies = 1 if random.random() < 0.45 else random.randint(2, 4)

    for _ in range(80):
        h = random.randint(11, 26)
        w = random.randint(11, 26)
        starts = _candidate_starts((h, w), rel_cells, step, desired_copies)
        if not starts:
            continue

        chain_start = random.choice(starts)
        chain_occ = _copy_footprint((h, w), rel_cells, chain_start, step, desired_copies)
        forbidden = _dilate4(chain_occ, steps=1)
        candidates = _original_anchor_candidates((h, w), rel_cells, forbidden)
        if not candidates:
            continue

        original_anchor = random.choice(candidates)
        inp = _render_input(rel_cells, original_anchor, chain_start, (h, w))
        if np.count_nonzero(inp == BG) <= inp.size // 2:
            continue

        expected = _solve_from_spec(inp, rel_cells, chain_start, step)
        if not _validate_semantics(inp, expected, rel_cells, chain_start, step, desired_copies):
            continue
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
