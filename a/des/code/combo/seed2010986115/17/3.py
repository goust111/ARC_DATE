from __future__ import annotations

import random
from collections import deque

import numpy as np


Coord = tuple[int, int]

DIR4: tuple[Coord, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIR8: tuple[Coord, ...] = tuple(
    (dr, dc)
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
    if not (dr == 0 and dc == 0)
)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _components(mask: np.ndarray) -> list[list[Coord]]:
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    parts: list[list[Coord]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[Coord] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if not _in_bounds((h, w), nr, nc):
                        continue
                    if not bool(mask[nr, nc]) or bool(seen[nr, nc]):
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
            parts.append(comp)
    return parts


def _enclosed_components(mask: np.ndarray) -> list[list[Coord]]:
    h, w = map(int, mask.shape)
    enclosed = []
    for comp in _components(mask):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
            continue
        enclosed.append(comp)
    return enclosed


def _border_background(grid: np.ndarray) -> int:
    g = np.asarray(grid, dtype=int)
    border = np.concatenate((g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]))
    values, counts = np.unique(border, return_counts=True)
    return int(values[np.argmax(counts)])


def _detect_contour_color(grid: np.ndarray, bg: int) -> tuple[int, list[list[Coord]]]:
    candidates: list[tuple[int, list[list[Coord]]]] = []
    for color in sorted(set(map(int, np.unique(grid))) - {bg}):
        enclosed = _enclosed_components(grid != color)
        if enclosed:
            candidates.append((int(color), enclosed))
    assert len(candidates) == 1, "expected exactly one contour color"
    return candidates[0]


def _trace_clockwise_wall_cycle(region_mask: np.ndarray) -> list[Coord]:
    h, w = map(int, region_mask.shape)
    outgoing: dict[Coord, tuple[Coord, Coord]] = {}
    indegree: dict[Coord, int] = {}

    def add_edge(start: Coord, end: Coord, cell: Coord) -> None:
        assert start not in outgoing, "boundary should have one outgoing edge per vertex"
        outgoing[start] = (end, cell)
        indegree[end] = indegree.get(end, 0) + 1

    for r in range(h):
        for c in range(w):
            if not bool(region_mask[r, c]):
                continue
            if r == 0 or not bool(region_mask[r - 1, c]):
                add_edge((r, c), (r, c + 1), (r, c))
            if c == w - 1 or not bool(region_mask[r, c + 1]):
                add_edge((r, c + 1), (r + 1, c + 1), (r, c))
            if r == h - 1 or not bool(region_mask[r + 1, c]):
                add_edge((r + 1, c + 1), (r + 1, c), (r, c))
            if c == 0 or not bool(region_mask[r, c - 1]):
                add_edge((r + 1, c), (r, c), (r, c))

    assert outgoing, "region must have a boundary"
    for vertex in outgoing:
        assert indegree.get(vertex, 0) == 1, "boundary should be a single simple cycle"

    start = min(outgoing)
    ordered_cells: list[Coord] = []
    seen: set[Coord] = set()
    cur = start
    while cur not in seen:
        seen.add(cur)
        nxt, cell = outgoing[cur]
        ordered_cells.append(cell)
        cur = nxt

    assert cur == start and len(seen) == len(outgoing), "boundary trace must close once"

    cycle: list[Coord] = []
    for cell in ordered_cells:
        if not cycle or cycle[-1] != cell:
            cycle.append(cell)
    if cycle and cycle[0] == cycle[-1]:
        cycle.pop()
    assert cycle and len(cycle) == len(set(cycle)), "wall cycle must visit unique cells once"
    return cycle


def _unique_nearest_wall_index(seed: Coord, wall_cycle: list[Coord]) -> int:
    distances = [abs(seed[0] - wr) + abs(seed[1] - wc) for wr, wc in wall_cycle]
    best = min(distances)
    winners = [i for i, dist in enumerate(distances) if dist == best]
    assert len(winners) == 1, "seed must have a unique nearest wall pixel"
    return int(winners[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each seed along the clockwise inner wall of its enclosing contour for an equal share of that wall.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and min(g.shape) >= 5

    bg = _border_background(g)
    contour_color, regions = _detect_contour_color(g, bg)

    out = g.copy()
    for region in regions:
        seeds = [(r, c) for r, c in region if int(g[r, c]) not in {bg, contour_color}]
        assert seeds, "every enclosed region needs at least one seed"

        rows = [r for r, _ in region]
        cols = [c for _, c in region]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)

        local_mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in region:
            local_mask[r - r0, c - c0] = True

        local_cycle = _trace_clockwise_wall_cycle(local_mask)
        wall_cycle = [(r0 + r, c0 + c) for r, c in local_cycle]
        assert all(int(g[r, c]) == bg for r, c in wall_cycle), "wall cells must be background"

        perimeter = len(wall_cycle)
        n_seeds = len(seeds)
        assert perimeter % n_seeds == 0, "wall length must divide evenly among seeds"
        share = perimeter // n_seeds

        for seed in seeds:
            out[seed] = bg

        painted: dict[Coord, int] = {}
        start_indices = [_unique_nearest_wall_index(seed, wall_cycle) for seed in seeds]
        assert len(set(start_indices)) == len(start_indices), "seed starts must be distinct"

        for seed, start in zip(seeds, start_indices):
            color = int(g[seed])
            for offset in range(share):
                cell = wall_cycle[(start + offset) % perimeter]
                assert cell not in painted, "seed segments must not overlap"
                painted[cell] = color

        assert len(painted) == perimeter, "seed segments should exactly partition the wall"
        for (r, c), color in painted.items():
            out[r, c] = color

    return out


COLOR_POOL = tuple(range(10))
GRID_SIDE_RANGE = (20, 30)
REGION_COUNT_RANGE = (1, 3)
REGION_HEIGHT_RANGE = (5, 9)
REGION_WIDTH_RANGE = (5, 10)
MAX_SEEDS_PER_REGION = 4
MAX_PATCH_PLACEMENT_TRIES = 100
MAX_SHAPE_TRIES = 80
GENERATION_RETRIES = 400


def _region_is_connected(mask: np.ndarray) -> bool:
    comps = _components(mask)
    return len(comps) == 1


def _region_mask_attempt(height: int, width: int) -> np.ndarray:
    mask = np.ones((height, width), dtype=bool)
    choices = ["rect"]
    if height >= 5 and width >= 5:
        choices.append("corner_notch")
    if (height >= 5 and width >= 6) or (height >= 6 and width >= 5):
        choices.append("side_notch")
    shape = random.choice(choices)
    if shape == "rect":
        return mask

    if shape == "corner_notch":
        notch_h = random.randint(2, height - 3)
        notch_w = random.randint(2, width - 3)
        corner = random.choice(("tl", "tr", "bl", "br"))
        if corner == "tl":
            mask[:notch_h, :notch_w] = False
        elif corner == "tr":
            mask[:notch_h, width - notch_w :] = False
        elif corner == "bl":
            mask[height - notch_h :, :notch_w] = False
        else:
            mask[height - notch_h :, width - notch_w :] = False
        return mask

    side_choices = []
    if height >= 5 and width >= 6:
        side_choices.extend(("top", "bottom"))
    if height >= 6 and width >= 5:
        side_choices.extend(("left", "right"))
    side = random.choice(side_choices)
    if side in {"top", "bottom"}:
        notch_h = random.randint(2, height - 3)
        notch_w = random.randint(2, width - 4)
        c0 = random.randint(1, width - notch_w - 1)
        rows = slice(0, notch_h) if side == "top" else slice(height - notch_h, height)
        mask[rows, c0 : c0 + notch_w] = False
    else:
        notch_w = random.randint(2, width - 3)
        notch_h = random.randint(2, height - 4)
        r0 = random.randint(1, height - notch_h - 1)
        cols = slice(0, notch_w) if side == "left" else slice(width - notch_w, width)
        mask[r0 : r0 + notch_h, cols] = False
    return mask


def _build_patch_from_region(region: np.ndarray, bg: int, contour_color: int) -> tuple[np.ndarray, list[Coord]]:
    h, w = map(int, region.shape)
    patch = np.full((h + 2, w + 2), bg, dtype=int)
    padded_region = np.zeros((h + 2, w + 2), dtype=bool)
    padded_region[1 : h + 1, 1 : w + 1] = region

    contour = np.zeros_like(padded_region)
    for r in range(h + 2):
        for c in range(w + 2):
            if not bool(padded_region[r, c]):
                continue
            for dr, dc in DIR8:
                nr = r + dr
                nc = c + dc
                if _in_bounds(padded_region.shape, nr, nc) and not bool(padded_region[nr, nc]):
                    contour[nr, nc] = True

    patch[contour] = contour_color
    enclosed = _enclosed_components(patch != contour_color)
    expected_region = {(r, c) for r, c in np.argwhere(padded_region)}
    assert len(enclosed) == 1 and set(enclosed[0]) == expected_region

    cycle = _trace_clockwise_wall_cycle(padded_region)
    assert all(int(patch[r, c]) == bg for r, c in cycle)
    return patch, cycle


def _seed_start_candidates(region: np.ndarray, wall_cycle: list[Coord]) -> dict[int, list[Coord]]:
    wall_set = set(wall_cycle)
    candidates: dict[int, list[Coord]] = {}
    for r, c in np.argwhere(region):
        cell = (int(r), int(c))
        if cell in wall_set:
            continue
        distances = [abs(cell[0] - wr) + abs(cell[1] - wc) for wr, wc in wall_cycle]
        best = min(distances)
        winners = [i for i, dist in enumerate(distances) if dist == best]
        if len(winners) != 1:
            continue
        candidates.setdefault(int(winners[0]), []).append(cell)
    return candidates


def _sample_seed_colors(n: int, bg: int, contour_color: int) -> list[int]:
    palette = [c for c in COLOR_POOL if c not in {bg, contour_color}]
    if n == 1:
        return [random.choice(palette)]
    distinct = random.randint(1, min(len(palette), n))
    chosen = random.sample(palette, distinct)
    return [random.choice(chosen) for _ in range(n)]


def _build_region_patch(bg: int, contour_color: int) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(MAX_SHAPE_TRIES):
        height = random.randint(*REGION_HEIGHT_RANGE)
        width = random.randint(*REGION_WIDTH_RANGE)
        region = _region_mask_attempt(height, width)
        if not _region_is_connected(region):
            continue

        try:
            patch_in, wall_cycle = _build_patch_from_region(region, bg, contour_color)
        except AssertionError:
            continue
        start_candidates = _seed_start_candidates(region, [(r - 1, c - 1) for r, c in wall_cycle])
        perimeter = len(wall_cycle)

        options: list[tuple[int, int, int]] = []
        for n_seeds in range(1, min(MAX_SEEDS_PER_REGION, region.sum()) + 1):
            if perimeter % n_seeds != 0:
                continue
            share = perimeter // n_seeds
            for offset in range(perimeter):
                starts = [(offset + i * share) % perimeter for i in range(n_seeds)]
                if all(start in start_candidates for start in starts):
                    options.append((n_seeds, share, offset))
        if not options:
            continue

        n_seeds, share, offset = random.choice(options)
        starts = [(offset + i * share) % perimeter for i in range(n_seeds)]
        seed_positions_local = [random.choice(start_candidates[start]) for start in starts]
        seed_colors = _sample_seed_colors(n_seeds, bg, contour_color)

        patch_out = patch_in.copy()
        for (rr, cc), color in zip(seed_positions_local, seed_colors):
            patch_in[rr + 1, cc + 1] = color

        for rr, cc in seed_positions_local:
            patch_out[rr + 1, cc + 1] = bg

        for start, color in zip(starts, seed_colors):
            for delta in range(share):
                wr, wc = wall_cycle[(start + delta) % perimeter]
                patch_out[wr, wc] = color

        return patch_in, patch_out

    raise RuntimeError("failed to build a valid region patch")


def _sample_patch_location(occupied: np.ndarray, patch_shape: tuple[int, int]) -> Coord | None:
    h, w = map(int, occupied.shape)
    ph, pw = map(int, patch_shape)
    if ph + 2 > h or pw + 2 > w:
        return None

    for _ in range(MAX_PATCH_PLACEMENT_TRIES):
        r0 = random.randint(1, h - ph - 1)
        c0 = random.randint(1, w - pw - 1)
        window = occupied[r0 - 1 : r0 + ph + 1, c0 - 1 : c0 + pw + 1]
        if not bool(window.any()):
            return (r0, c0)
    return None


def _paste_patch(canvas: np.ndarray, patch: np.ndarray, top_left: Coord) -> None:
    r0, c0 = top_left
    ph, pw = map(int, patch.shape)
    canvas[r0 : r0 + ph, c0 : c0 + pw] = patch


def _place_patch(canvas: np.ndarray, patch: np.ndarray, occupied: np.ndarray) -> Coord | None:
    h, w = map(int, canvas.shape)
    ph, pw = map(int, patch.shape)
    if ph + 2 > h or pw + 2 > w:
        return None

    top_left = _sample_patch_location(occupied, patch.shape)
    if top_left is None:
        return None
    r0, c0 = top_left
    _paste_patch(canvas, patch, top_left)
    occupied[r0 : r0 + ph, c0 : c0 + pw] = True
    return top_left


def _generate_pair() -> tuple[np.ndarray, np.ndarray]:
    bg = random.choice(COLOR_POOL)
    contour_color = random.choice([c for c in COLOR_POOL if c != bg])
    n_regions = random.randint(*REGION_COUNT_RANGE)

    patches_in: list[np.ndarray] = []
    patches_out: list[np.ndarray] = []
    for _ in range(n_regions):
        patch_in, patch_out = _build_region_patch(bg, contour_color)
        patches_in.append(patch_in)
        patches_out.append(patch_out)

    max_h = max(p.shape[0] for p in patches_in)
    max_w = max(p.shape[1] for p in patches_in)
    total_area = sum(int(p.shape[0] * p.shape[1]) for p in patches_in)
    min_side = max(max_h + 4, max_w + 4, int(np.ceil(np.sqrt(total_area * 2.6))))
    low = max(GRID_SIDE_RANGE[0], min_side)
    high = max(low, GRID_SIDE_RANGE[1])
    grid_h = random.randint(low, high)
    grid_w = random.randint(low, high)

    inp = np.full((grid_h, grid_w), bg, dtype=int)
    out = np.full((grid_h, grid_w), bg, dtype=int)
    occupied = np.zeros((grid_h, grid_w), dtype=bool)

    order = list(range(n_regions))
    random.shuffle(order)
    for idx in order:
        top_left = _place_patch(inp, patches_in[idx], occupied)
        if top_left is None:
            raise RuntimeError("failed to place input patch")
        _paste_patch(out, patches_out[idx], top_left)

    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        try:
            inp, expected = _generate_pair()
        except RuntimeError:
            continue
        try:
            actual = main(inp)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return inp
    raise RuntimeError("failed to generate a valid input")
