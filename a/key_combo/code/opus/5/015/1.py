import itertools
import random
from typing import Iterable

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]
MAX_GENERATION_ATTEMPTS = 600


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    for dr, dc in DIR4:
        yield r + dr, c + dc


def bbox_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def components_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
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
                for nr, nc in neighbors4(rr, cc):
                    if in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _is_solid_rectangle(grid: np.ndarray, coords: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = bbox_coords(coords)
    if r1 - r0 + 1 < 2 or c1 - c0 + 1 < 2:
        return False
    if len(coords) != (r1 - r0 + 1) * (c1 - c0 + 1):
        return False
    patch = crop_bbox(grid, (r0, c0, r1, c1))
    colors = set(map(int, patch.ravel()))
    return len(colors) == 1 and 0 not in colors


def _trace_path(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    coord_set = set(coords)
    adj = {
        cell: [nbr for nbr in neighbors4(*cell) if nbr in coord_set]
        for cell in coord_set
    }
    endpoints = [cell for cell, nbrs in adj.items() if len(nbrs) == 1]
    assert len(endpoints) == 2, f"snake must have two endpoints, got {len(endpoints)}"
    start = min(endpoints)
    order = [start]
    prev = None
    cur = start
    while len(order) < len(coords):
        nxts = [nbr for nbr in adj[cur] if nbr != prev]
        assert len(nxts) == 1, "snake path must not branch"
        prev, cur = cur, nxts[0]
        order.append(cur)
    return order


def _extract_objects(grid: np.ndarray) -> tuple[list[dict], list[dict]]:
    rects: list[dict] = []
    snakes: list[dict] = []
    for coords in components_mask(grid != 0):
        bbox = bbox_coords(coords)
        if _is_solid_rectangle(grid, coords):
            color = int(grid[coords[0]])
            rects.append({"coords": coords, "bbox": bbox, "color": color})
            continue
        path = _trace_path(coords)
        colors = [int(grid[r, c]) for r, c in path]
        snakes.append(
            {
                "coords": coords,
                "bbox": bbox,
                "path": path,
                "colors": colors,
                "endpoints": (path[0], path[-1]),
                "endpoint_colors": (colors[0], colors[-1]),
            }
        )
    rects.sort(key=lambda obj: (obj["bbox"][0], obj["bbox"][1]))
    return rects, snakes


def _chain_candidates(snakes: list[dict]) -> list[dict]:
    n = len(snakes)
    variants = []
    for snake in snakes:
        variants.append(
            [
                {
                    "start_color": snake["colors"][0],
                    "end_color": snake["colors"][-1],
                    "colors": snake["colors"],
                    "path": snake["path"],
                    "start_coord": snake["path"][0],
                    "end_coord": snake["path"][-1],
                },
                {
                    "start_color": snake["colors"][-1],
                    "end_color": snake["colors"][0],
                    "colors": list(reversed(snake["colors"])),
                    "path": list(reversed(snake["path"])),
                    "start_coord": snake["path"][-1],
                    "end_coord": snake["path"][0],
                },
            ]
        )

    sols: list[dict] = []
    for perm in itertools.permutations(range(n)):
        for bits in itertools.product((0, 1), repeat=n):
            oriented = [variants[idx][bit] for idx, bit in zip(perm, bits)]
            if any(
                oriented[i]["end_color"] != oriented[i + 1]["start_color"]
                for i in range(n - 1)
            ):
                continue
            sols.append(
                {
                    "order": perm,
                    "bits": bits,
                    "snakes": oriented,
                    "start_coord": oriented[0]["start_coord"],
                    "sequence": [col for part in oriented for col in part["colors"]],
                }
            )
    return sols


def _canonical_chain_key(sol: dict) -> tuple[tuple[int, ...], tuple[int, ...]]:
    order = tuple(int(x) for x in sol["order"])
    bits = tuple(int(x) for x in sol["bits"])
    rev_order = tuple(reversed(order))
    rev_bits = tuple(1 - b for b in reversed(bits))
    return min((order, bits), (rev_order, rev_bits))


def _pick_chain(candidates: list[dict]) -> dict:
    assert candidates, "no legal snake chain found"
    canonical = {_canonical_chain_key(sol) for sol in candidates}
    assert len(canonical) == 1, "snake links are ambiguous"
    return min(candidates, key=lambda sol: tuple(sol["start_coord"]))


def _render_with_chain(grid: np.ndarray, rects: list[dict], snakes: list[dict], chain: dict) -> np.ndarray:
    out = grid.copy()
    for snake in snakes:
        for r, c in snake["coords"]:
            out[r, c] = 0
    seq = chain["sequence"]
    for i, rect in enumerate(rects):
        color = int(seq[i % len(seq)])
        for r, c in rect["coords"]:
            out[r, c] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Link scattered snakes by equal endpoint colors, then use the traversed color stream to repaint rectangles.

    Input structure:
    A black background contains several disconnected multicolor 4-connected snakes and several disconnected solid monochrome rectangles. Each snake is a simple path with exactly two endpoints; each rectangle is a filled 2D box whose bbox is fully occupied by one color.

    Rule:
    Partition the non-black pixels into objects, separate solid rectangles from snakes, and for every snake read its two endpoint colors and pixel order along the 4-connected path. Chain all snakes so consecutive snakes meet at equal endpoint colors, using every snake exactly once. The chain is unique up to reversal; to fix the traversal head, start from the free endpoint with smaller `(row, col)`. Traverse the resulting super-snake pixel by pixel to obtain a color sequence. Sort the rectangles by their bbox top-left corner in row-major order, repaint rectangle `i` with `sequence[i mod len(sequence)]`, and erase all snake pixels to black.

    Key insight:
    The snakes are not output objects: endpoint-color matching determines a global snake order, and only the ordered path colors survive by cyclically recoloring the position-sorted rectangles.
    """
    grid = np.asarray(grid, dtype=int)
    rects, snakes = _extract_objects(grid)
    assert 3 <= len(rects) <= 5
    assert 3 <= len(snakes) <= 5
    candidates = _chain_candidates(snakes)
    chain = _pick_chain(candidates)
    return _render_with_chain(grid, rects, snakes, chain)


def _transform_coords(
    coords: list[tuple[int, int]],
    mode: int,
) -> list[tuple[int, int]]:
    if not coords:
        return []
    arr = np.array(coords, dtype=int)
    r0, c0 = arr.min(axis=0)
    arr = arr - np.array([r0, c0])
    h = int(arr[:, 0].max()) + 1
    w = int(arr[:, 1].max()) + 1
    transformed: list[tuple[int, int]] = []
    for r, c in arr:
        if mode == 0:
            rr, cc = r, c
        elif mode == 1:
            rr, cc = c, h - 1 - r
        elif mode == 2:
            rr, cc = h - 1 - r, w - 1 - c
        elif mode == 3:
            rr, cc = w - 1 - c, r
        elif mode == 4:
            rr, cc = r, w - 1 - c
        elif mode == 5:
            rr, cc = h - 1 - r, c
        elif mode == 6:
            rr, cc = c, r
        else:
            rr, cc = w - 1 - c, h - 1 - r
        transformed.append((int(rr), int(cc)))
    mins = np.min(np.array(transformed, dtype=int), axis=0)
    return [(r - int(mins[0]), c - int(mins[1])) for r, c in transformed]


def _random_snake_path(length: int, rng: random.Random) -> list[tuple[int, int]]:
    for _ in range(400):
        path = [(0, 0)]
        occupied = {path[0]}
        while len(path) < length:
            options: list[tuple[str, tuple[int, int]]] = []
            for side, endpoint in (("head", path[0]), ("tail", path[-1])):
                for nbr in neighbors4(*endpoint):
                    if nbr in occupied:
                        continue
                    degree = sum(1 for adj in neighbors4(*nbr) if adj in occupied)
                    if degree != 1:
                        continue
                    options.append((side, nbr))
            if not options:
                break
            side, cell = rng.choice(options)
            if side == "head":
                path.insert(0, cell)
            else:
                path.append(cell)
            occupied.add(cell)
        if len(path) != length:
            continue
        mode = rng.randrange(8)
        transformed = _transform_coords(path, mode)
        if rng.random() < 0.5:
            transformed = list(reversed(transformed))
        return transformed
    raise RuntimeError("failed to sample snake path")


def _cell_is_free(blocked: np.ndarray, r: int, c: int) -> bool:
    return in_bounds(blocked.shape, r, c) and not blocked[r, c]


def _place_object(
    grid: np.ndarray,
    blocked: np.ndarray,
    cells: list[tuple[int, int]],
    colors: list[int],
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    arr = np.array(cells, dtype=int)
    h = int(arr[:, 0].max()) + 1
    w = int(arr[:, 1].max()) + 1
    placements = [
        (r0, c0)
        for r0 in range(1, grid.shape[0] - h)
        for c0 in range(1, grid.shape[1] - w)
    ]
    rng.shuffle(placements)
    for r0, c0 in placements:
        abs_cells = [(r0 + r, c0 + c) for r, c in cells]
        ok = True
        for rr, cc in abs_cells:
            if not _cell_is_free(blocked, rr, cc):
                ok = False
                break
        if not ok:
            continue
        for (rr, cc), col in zip(abs_cells, colors):
            grid[rr, cc] = int(col)
        for rr, cc in abs_cells:
            for dr, dc in DIR8 + [(0, 0)]:
                nr, nc = rr + dr, cc + dc
                if in_bounds(blocked.shape, nr, nc):
                    blocked[nr, nc] = True
        return abs_cells
    return None


def _choose_distinct_rectangle_colors(
    n_rects: int,
    target_colors: list[int],
    rng: random.Random,
) -> list[int] | None:
    palette = list(range(1, 10))
    for _ in range(200):
        rng.shuffle(palette)
        chosen = palette[:n_rects]
        if len(set(chosen)) != n_rects:
            continue
        if any(chosen[i] == target_colors[i] for i in range(n_rects)):
            continue
        return chosen
    return None


def _row_major_rect_order(rects: list[dict]) -> list[int]:
    return sorted(range(len(rects)), key=lambda i: (rects[i]["bbox"][0], rects[i]["bbox"][1]))


def _col_major_rect_order(rects: list[dict]) -> list[int]:
    return sorted(range(len(rects)), key=lambda i: (rects[i]["bbox"][1], rects[i]["bbox"][0]))


def _spatial_snake_chain(snakes: list[dict]) -> dict:
    order = sorted(range(len(snakes)), key=lambda i: (snakes[i]["bbox"][0], snakes[i]["bbox"][1]))
    parts = []
    for idx in order:
        snake = snakes[idx]
        forward = snake["path"][0] <= snake["path"][-1]
        parts.append(
            snake["colors"] if forward else list(reversed(snake["colors"]))
        )
    return {"sequence": [col for part in parts for col in part]}


def _validate_instance(grid: np.ndarray) -> bool:
    rects, snakes = _extract_objects(grid)
    if not (4 <= len(rects) <= 5 and 3 <= len(snakes) <= 5):
        return False
    candidates = _chain_candidates(snakes)
    if not candidates:
        return False
    if len({_canonical_chain_key(sol) for sol in candidates}) != 1:
        return False
    chain = _pick_chain(candidates)
    seq = chain["sequence"]
    if len(set(seq[: len(rects)])) < 3:
        return False
    if len(chain["snakes"][0]["colors"]) >= len(rects):
        return False
    out = _render_with_chain(grid, rects, snakes, chain)
    if np.array_equal(out, grid):
        return False
    for snake in snakes:
        for r, c in snake["coords"]:
            if out[r, c] != 0:
                return False

    row_order = _row_major_rect_order(rects)
    col_order = _col_major_rect_order(rects)
    if row_order == col_order:
        return False
    row_colors = [seq[i] for i in range(len(rects))]
    col_colors = row_colors[:]
    alt = grid.copy()
    for snake in snakes:
        for r, c in snake["coords"]:
            alt[r, c] = 0
    for i, rect_idx in enumerate(col_order):
        for r, c in rects[rect_idx]["coords"]:
            alt[r, c] = int(col_colors[i])
    if np.array_equal(out, alt):
        return False

    first_only = chain["snakes"][0]["colors"]
    alt2 = grid.copy()
    for snake in snakes:
        for r, c in snake["coords"]:
            alt2[r, c] = 0
    for i, rect in enumerate(rects):
        fill = int(first_only[i % len(first_only)])
        for r, c in rect["coords"]:
            alt2[r, c] = fill
    if np.array_equal(out, alt2):
        return False

    spatial = _spatial_snake_chain(snakes)
    alt3 = _render_with_chain(grid, rects, snakes, spatial)
    if np.array_equal(out, alt3):
        return False
    return True


def _generate_one(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(15, 19)
    w = rng.randint(15, 19)
    n_snakes = rng.randint(3, 5)
    n_rects = rng.randint(4, 5)
    grid = np.zeros((h, w), dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    lengths = [rng.randint(3, 6) for _ in range(n_snakes)]
    snake_shapes = [_random_snake_path(length, rng) for length in lengths]
    placements: list[list[tuple[int, int]]] = []
    for shape in snake_shapes:
        placed = _place_object(grid, blocked, shape, [1] * len(shape), rng)
        if placed is None:
            return None
        placements.append(placed)

    order = list(range(n_snakes))
    rng.shuffle(order)
    bits = [rng.randint(0, 1) for _ in range(n_snakes)]
    oriented_paths = []
    for idx, bit in zip(order, bits):
        path = placements[idx]
        oriented_paths.append(path if bit == 0 else list(reversed(path)))
    if oriented_paths[0][0] > oriented_paths[-1][-1]:
        order = list(reversed(order))
        bits = [1 - bit for bit in reversed(bits)]
        oriented_paths = []
        for idx, bit in zip(order, bits):
            path = placements[idx]
            oriented_paths.append(path if bit == 0 else list(reversed(path)))

    if len(oriented_paths[0]) >= n_rects:
        return None

    chain_palette = rng.sample(range(1, 10), n_snakes + 1)
    interior_palette = [c for c in range(1, 10) if c not in chain_palette]
    if not interior_palette:
        interior_palette = chain_palette[:]

    snake_color_lists: dict[int, list[int]] = {}
    chain_sequences: list[list[int]] = []
    for pos, idx in enumerate(order):
        path_len = len(placements[idx])
        colors = [chain_palette[pos]]
        while len(colors) < path_len - 1:
            colors.append(rng.choice(interior_palette))
        colors.append(chain_palette[pos + 1])
        snake_color_lists[idx] = colors if bits[pos] == 0 else list(reversed(colors))
        chain_sequences.append(colors)

    grid.fill(0)
    for idx, cells in enumerate(placements):
        for (r, c), col in zip(cells, snake_color_lists[idx]):
            grid[r, c] = int(col)

    true_sequence = [col for seq in chain_sequences for col in seq]
    target_rect_colors = true_sequence[:n_rects]
    rect_colors = _choose_distinct_rectangle_colors(n_rects, target_rect_colors, rng)
    if rect_colors is None:
        return None

    rect_shapes = []
    for _ in range(n_rects):
        rh = rng.randint(2, 3)
        rw = rng.randint(2, 4)
        rect_shapes.append([(r, c) for r in range(rh) for c in range(rw)])

    for shape, color in zip(rect_shapes, rect_colors):
        placed = _place_object(grid, blocked, shape, [color] * len(shape), rng)
        if placed is None:
            return None

    if not _validate_instance(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_one(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC instance")
