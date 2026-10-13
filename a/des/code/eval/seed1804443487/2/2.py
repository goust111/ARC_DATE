import random

import numpy as np


BG = 0
RED = 2
GREY = 5
LAYER_COLOR_POOL = (1, 3, 4, 6, 7, 8, 9)
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR_TO_K = {
    (-1, 0): 0,  # up
    (0, -1): 1,  # left
    (1, 0): 2,   # down
    (0, 1): 3,   # right
}

GEN_RETRIES = 400
TEMPLATE_RETRIES = 120
LAYER_RETRIES = 80
PLACEMENT_ATTEMPTS = 500


def _in_bbox(r: int, c: int, bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    return r0 <= r <= r1 and c0 <= c <= c1


def _rotated_point(r: int, c: int, h: int, w: int, k: int) -> tuple[int, int]:
    k %= 4
    if k == 0:
        return r, c
    if k == 1:
        return w - 1 - c, r
    if k == 2:
        return h - 1 - r, w - 1 - c
    return c, h - 1 - r


def _frame_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    h, w = shape
    r0, c0, r1, c1 = bbox
    mask = np.zeros((h, w), dtype=bool)
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    return mask


def _find_template_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    grey_pts = np.argwhere(grid == GREY)
    assert len(grey_pts) >= 8, "expected a grey template frame"
    r0, c0 = map(int, grey_pts.min(axis=0))
    r1, c1 = map(int, grey_pts.max(axis=0))
    assert r1 - r0 >= 2 and c1 - c0 >= 2, "template frame must enclose an interior"
    expected = _frame_mask(grid.shape, (r0, c0, r1, c1))
    assert np.array_equal(grid == GREY, expected), "grey cells must form exactly one rectangular frame"
    return r0, c0, r1, c1


def _find_seed_groups(
    grid: np.ndarray, template_bbox: tuple[int, int, int, int]
) -> list[tuple[int, int, int, int, int]]:
    h, w = grid.shape
    groups: list[tuple[int, int, int, int, int]] = []
    used_selectors: set[tuple[int, int]] = set()
    for rr, cc in map(tuple, np.argwhere(grid == RED)):
        rr = int(rr)
        cc = int(cc)
        if _in_bbox(rr, cc, template_bbox):
            continue
        neighbors = []
        for dr, dc in DIRS4:
            nr = rr + dr
            nc = cc + dc
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if _in_bbox(nr, nc, template_bbox):
                continue
            val = int(grid[nr, nc])
            if val in (BG, RED, GREY):
                continue
            neighbors.append((nr, nc, val))
        assert len(neighbors) == 1, "each red marker must touch exactly one selector pixel"
        sr, sc, color = neighbors[0]
        assert (sr, sc) not in used_selectors, "selector pixels must be unique"
        used_selectors.add((sr, sc))
        groups.append((sr, sc, rr, cc, color))
    groups.sort()
    return groups


def _layer_offsets(layer: np.ndarray, direction: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = layer.shape
    anchor = (h - 1, w // 2)
    k = DIR_TO_K[direction]
    rotated = np.rot90(layer, k)
    ar, ac = _rotated_point(anchor[0], anchor[1], h, w, k)
    return {(int(r) - ar, int(c) - ac) for r, c in np.argwhere(rotated)}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow the selected template layer from each seed after rotating it to the seed direction.

    The template is the unique grey rectangular frame in the upper-left area. Each seed is a
    selector pixel adjacent to a red direction pixel; the red selector offset is the growth
    direction. The chosen color filters the template to one layer, the full template canvas is
    rotated, and its canonical bottom-center anchor is aligned to the selector pixel.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    r0, c0, r1, c1 = _find_template_bbox(g)
    interior = g[r0 + 1 : r1, c0 + 1 : c1]
    assert interior.size > 0, "template interior must be non-empty"
    layer_colors = {int(color) for color in np.unique(interior) if int(color) != BG}
    assert layer_colors, "template must contain at least one colored layer"

    groups = _find_seed_groups(g, (r0, c0, r1, c1))
    out = g.copy()
    for _, _, rr, cc, _ in groups:
        out[rr, cc] = BG

    for sr, sc, rr, cc, color in groups:
        assert color in layer_colors, "selector color must exist in the template"
        direction = (rr - sr, cc - sc)
        assert direction in DIR_TO_K, "seed direction must be cardinal"
        offsets = _layer_offsets(interior == color, direction)
        for dr, dc in offsets:
            tr = sr + dr
            tc = sc + dc
            assert 0 <= tr < g.shape[0] and 0 <= tc < g.shape[1], "grown layer must stay in bounds"
            if out[tr, tc] == BG:
                out[tr, tc] = color

    return out


def _points_from_mask(mask: np.ndarray) -> set[tuple[int, int]]:
    return {(int(r), int(c)) for r, c in np.argwhere(mask)}


def _dilate(points: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in points:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    out.add((nr, nc))
    return out


def _make_layer_mask(
    th: int, tw: int, occupied: set[tuple[int, int]], start: tuple[int, int], rng: random.Random
) -> np.ndarray | None:
    for _ in range(LAYER_RETRIES):
        target_size = rng.randint(4, min(8, th + 2))
        cells = {start}
        frontier = [start]

        while frontier and len(cells) < target_size:
            base = rng.choice(frontier if rng.random() < 0.65 else list(cells))
            options: list[tuple[int, int]] = []
            for dr, dc, weight in [(-1, 0, 7), (0, -1, 3), (0, 1, 3), (1, 0, 1)]:
                nr = base[0] + dr
                nc = base[1] + dc
                if not (0 <= nr < th and 0 <= nc < tw):
                    continue
                if (nr, nc) in occupied or (nr, nc) in cells:
                    continue
                if nr > start[0]:
                    continue
                if nr == start[0] and abs(nc - start[1]) > 2:
                    continue
                options.extend([(nr, nc)] * weight)
            if not options:
                frontier = [pt for pt in frontier if pt != base]
                continue
            nxt = rng.choice(options)
            cells.add(nxt)
            frontier.append(nxt)

        for _ in range(rng.randint(0, 2)):
            bases = list(cells)
            rng.shuffle(bases)
            added = False
            for br, bc in bases:
                spur_options: list[tuple[int, int]] = []
                for dr, dc, weight in [(-1, 0, 6), (0, -1, 2), (0, 1, 2)]:
                    nr = br + dr
                    nc = bc + dc
                    if not (0 <= nr < th and 0 <= nc < tw):
                        continue
                    if (nr, nc) in occupied or (nr, nc) in cells:
                        continue
                    if nr > start[0]:
                        continue
                    spur_options.extend([(nr, nc)] * weight)
                if not spur_options:
                    continue
                cells.add(rng.choice(spur_options))
                added = True
                break
            if not added:
                break

        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        if len(cells) < 4:
            continue
        if max(rows) - min(rows) < 2:
            continue
        if min(rows) > th - 3:
            continue
        if max(cols) == min(cols) and len(cells) < 5:
            continue

        mask = np.zeros((th, tw), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        return mask
    return None


def _make_template(th: int, tw: int, colors: list[int], rng: random.Random) -> np.ndarray | None:
    center = tw // 2
    start_band = list(range(max(0, center - 2), min(tw, center + 3)))
    for _ in range(TEMPLATE_RETRIES):
        interior = np.zeros((th, tw), dtype=int)
        occupied: set[tuple[int, int]] = set()
        starts: list[tuple[int, int]] = []
        available = [(row, col) for row in (th - 1, th - 2) for col in start_band]
        rng.shuffle(available)

        for _color in colors:
            pick = None
            for row, col in available:
                if (row, col) in occupied:
                    continue
                if any(abs(col - prev_col) <= 0 for _, prev_col in starts) and len(start_band) >= len(colors):
                    continue
                pick = (row, col)
                break
            if pick is None:
                break
            starts.append(pick)
            occupied.add(pick)
        if len(starts) != len(colors):
            continue

        occupied.clear()
        ok = True
        for color, start in zip(colors, starts):
            mask = _make_layer_mask(th, tw, occupied, start, rng)
            if mask is None:
                ok = False
                break
            interior[mask] = color
            occupied |= _points_from_mask(mask)
        if not ok:
            continue

        if np.count_nonzero(interior) >= th * tw - 2:
            continue
        if np.array_equal(interior, np.rot90(interior, 1)):
            continue
        if np.array_equal(interior, np.rot90(interior, 2)):
            continue

        widthful = False
        for color in colors:
            pts = np.argwhere(interior == color)
            assert len(pts) > 0
            if int(pts[:, 1].max()) > int(pts[:, 1].min()):
                widthful = True
                break
        if not widthful:
            continue
        return interior
    return None


def _template_cells(
    top: int, left: int, th: int, tw: int
) -> set[tuple[int, int]]:
    return {(r, c) for r in range(top, top + th + 2) for c in range(left, left + tw + 2)}


def _place_template(
    grid: np.ndarray, top: int, left: int, interior: np.ndarray
) -> tuple[int, int, int, int]:
    th, tw = interior.shape
    r0, c0 = top, left
    r1, c1 = top + th + 1, left + tw + 1
    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY
    grid[r0 + 1 : r1, c0 + 1 : c1] = interior
    return r0, c0, r1, c1


def _seed_specs(colors: list[int], rng: random.Random) -> list[tuple[int, tuple[int, int]]]:
    n_seeds = rng.randint(max(4, len(colors)), 6)
    seed_colors = colors[:] + [rng.choice(colors) for _ in range(n_seeds - len(colors))]
    rng.shuffle(seed_colors)

    base_dirs = list(DIR_TO_K)
    if n_seeds <= 4:
        seed_dirs = rng.sample(base_dirs, n_seeds)
    else:
        seed_dirs = base_dirs + [rng.choice(base_dirs) for _ in range(n_seeds - 4)]
        rng.shuffle(seed_dirs)

    specs = list(zip(seed_colors, seed_dirs))
    rng.shuffle(specs)
    return specs


def _place_seeds(
    grid: np.ndarray,
    interior: np.ndarray,
    template_bbox: tuple[int, int, int, int],
    specs: list[tuple[int, tuple[int, int]]],
    rng: random.Random,
) -> bool:
    h, w = grid.shape
    template_block = _template_cells(template_bbox[0], template_bbox[1], interior.shape[0], interior.shape[1])
    input_forbidden = _dilate(template_block, h, w)
    output_forbidden = _dilate(template_block, h, w)

    offsets_cache = {
        (color, direction): _layer_offsets(interior == color, direction)
        for color, direction in specs
    }

    for color, direction in specs:
        offsets = offsets_cache[(color, direction)]
        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            sr = rng.randrange(h)
            sc = rng.randrange(w)
            rr = sr + direction[0]
            cc = sc + direction[1]
            if not (0 <= rr < h and 0 <= cc < w):
                continue
            pair = {(sr, sc), (rr, cc)}
            if any(grid[r, c] != BG for r, c in pair):
                continue
            if any(cell in input_forbidden or cell in output_forbidden for cell in pair):
                continue

            targets: set[tuple[int, int]] = set()
            ok = True
            for dr, dc in offsets:
                tr = sr + dr
                tc = sc + dc
                if not (0 <= tr < h and 0 <= tc < w):
                    ok = False
                    break
                targets.add((tr, tc))
            if not ok:
                continue
            if any((tr, tc) in output_forbidden for tr, tc in targets if (tr, tc) != (sr, sc)):
                continue

            grid[sr, sc] = color
            grid[rr, cc] = RED
            input_forbidden |= _dilate(pair, h, w)
            output_forbidden |= _dilate(targets | {(sr, sc)}, h, w)
            placed = True
            break
        if not placed:
            return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_colors = rng.choice([2, 3])
    colors = rng.sample(list(LAYER_COLOR_POOL), n_colors)
    th = rng.randint(5, 8)
    tw = rng.choice([5, 7, 9])

    min_h = max(20, th + 10)
    min_w = max(20, tw + 12)
    h = rng.randint(min_h, min_h + 6)
    w = rng.randint(min_w, min_w + 7)
    grid = np.full((h, w), BG, dtype=int)

    top = rng.randint(0, min(2, h - (th + 2)))
    left = rng.randint(0, min(2, w - (tw + 2)))

    interior = _make_template(th, tw, colors, rng)
    if interior is None:
        return None
    template_bbox = _place_template(grid, top, left, interior)

    specs = _seed_specs(colors, rng)
    if not _place_seeds(grid, interior, template_bbox, specs, rng):
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    if np.any(out == RED):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a valid puzzle instance")
