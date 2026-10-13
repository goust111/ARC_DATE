from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

RADII = (1, 3, 4, 5)
RADIUS_WEIGHTS = (2, 3, 3, 2)
MAX_ATTEMPTS = 400

SHAPES: dict[int, tuple[tuple[tuple[int, int], ...], ...]] = {
    1: (((0, 0),),),
    3: (
        ((0, -1), (0, 0), (0, 1)),
        ((-1, 0), (0, 0), (1, 0)),
    ),
    4: (
        ((-1, 0), (0, -1), (0, 0), (1, 0)),
        ((-1, 0), (0, 0), (0, 1), (1, 0)),
        ((-1, 0), (0, -1), (0, 0), (0, 1)),
        ((0, -1), (0, 0), (0, 1), (1, 0)),
    ),
    5: (
        ((-1, 0), (0, -1), (0, 0), (0, 1), (1, 0)),
        ((0, -2), (0, -1), (0, 0), (0, 1), (0, 2)),
        ((-2, 0), (-1, 0), (0, 0), (1, 0), (2, 0)),
    ),
}


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _same_color_components(grid: np.ndarray) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color in {BLACK, GREY} or bool(seen[r, c]):
                continue

            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not bool(seen[nr, nc])
                        and int(grid[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))

            comps.append({"color": color, "coords": coords})

    return comps


def _visual_center(coords: list[tuple[int, int]]) -> tuple[int, int]:
    cells = set(coords)
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    n = len(coords)
    sum_r = sum(rows)
    sum_c = sum(cols)

    best_cell = (r0, c0)
    best_key: tuple[int, int, int, int, int, int] | None = None
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            dist = sum(abs(r - rr) + abs(c - cc) for rr, cc in coords)
            in_comp = (r, c) in cells
            degree = (
                sum((r + dr, c + dc) in cells for dr, dc in DIR4)
                if in_comp
                else -1
            )
            mean_delta = abs(n * r - sum_r) + abs(n * c - sum_c)
            key = (
                dist,
                0 if in_comp else 1,
                -degree,
                mean_delta,
                r,
                c,
            )
            if best_key is None or key < best_key:
                best_key = key
                best_cell = (int(r), int(c))

    return best_cell


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor grey cells inside each bomb's Manhattan blast, with smaller bombs
    taking priority on overlaps.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    h, w = g.shape
    grey_mask = g == GREY
    out = g.copy()
    best_radius = np.full((h, w), 10**9, dtype=int)

    bombs: list[dict[str, object]] = []
    for comp in _same_color_components(g):
        coords = comp["coords"]
        bombs.append(
            {
                "color": int(comp["color"]),
                "radius": int(len(coords)),
                "center": _visual_center(coords),
            }
        )

    bombs.sort(
        key=lambda bomb: (
            int(bomb["radius"]),
            int(bomb["center"][0]),
            int(bomb["center"][1]),
            int(bomb["color"]),
        )
    )

    for bomb in bombs:
        radius = int(bomb["radius"])
        color = int(bomb["color"])
        cr, cc = bomb["center"]
        for rr in range(max(0, cr - radius), min(h, cr + radius + 1)):
            rem = radius - abs(rr - cr)
            c0 = max(0, cc - rem)
            c1 = min(w, cc + rem + 1)

            grey_slice = grey_mask[rr, c0:c1]
            if not bool(grey_slice.any()):
                continue

            best_slice = best_radius[rr, c0:c1]
            out_slice = out[rr, c0:c1]
            better = grey_slice & (best_slice > radius)
            if bool(better.any()):
                best_slice[better] = radius
                out_slice[better] = color

    return out


def _shift(cells: set[tuple[int, int]], dr: int, dc: int) -> set[tuple[int, int]]:
    return {(r + dr, c + dc) for r, c in cells}


def _bbox(cells: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _diamond(center: tuple[int, int], radius: int) -> set[tuple[int, int]]:
    cr, cc = center
    cells: set[tuple[int, int]] = set()
    for dr in range(-radius, radius + 1):
        rem = radius - abs(dr)
        for dc in range(-rem, rem + 1):
            cells.add((int(cr + dr), int(cc + dc)))
    return cells


def _path(a: tuple[int, int], b: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = a
    rr, cc = b
    cells = {(int(r), int(c))}
    if random.random() < 0.5:
        while c != cc:
            c += 1 if cc > c else -1
            cells.add((int(r), int(c)))
        while r != rr:
            r += 1 if rr > r else -1
            cells.add((int(r), int(c)))
    else:
        while r != rr:
            r += 1 if rr > r else -1
            cells.add((int(r), int(c)))
        while c != cc:
            c += 1 if cc > c else -1
            cells.add((int(r), int(c)))
    return cells


def _grow_region(
    h: int,
    w: int,
    support: set[tuple[int, int]],
    target_size: int,
) -> set[tuple[int, int]]:
    region = set(support)
    frontier: set[tuple[int, int]] = set()

    for r, c in region:
        for dr, dc in DIR4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in region:
                frontier.add((int(nr), int(nc)))

    while len(region) < target_size and frontier:
        choices = list(frontier)
        weights = []
        for r, c in choices:
            degree = sum((r + dr, c + dc) in region for dr, dc in DIR4)
            weights.append(degree * degree + 1)

        chosen = random.choices(choices, weights=weights, k=1)[0]
        frontier.remove(chosen)
        region.add(chosen)
        r, c = chosen
        for dr, dc in DIR4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in region:
                frontier.add((int(nr), int(nc)))

    return region


def _touches4(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> bool:
    for r, c in a:
        for dr, dc in DIR4:
            if (r + dr, c + dc) in b:
                return True
    return False


def _crop_cluster(
    arr: np.ndarray,
    bombs: list[dict[str, object]],
) -> dict[str, object]:
    pts = np.argwhere(arr != BLACK)
    assert pts.size > 0
    r0 = max(0, int(pts[:, 0].min()) - 1)
    c0 = max(0, int(pts[:, 1].min()) - 1)
    r1 = min(arr.shape[0] - 1, int(pts[:, 0].max()) + 1)
    c1 = min(arr.shape[1] - 1, int(pts[:, 1].max()) + 1)

    cropped = arr[r0 : r1 + 1, c0 : c1 + 1].copy()
    shifted_bombs: list[dict[str, object]] = []
    for bomb in bombs:
        br, bc = bomb["center"]
        shifted_bombs.append(
            {
                "center": (int(br - r0), int(bc - c0)),
                "radius": int(bomb["radius"]),
                "color": int(bomb["color"]),
            }
        )

    return {"grid": cropped, "bombs": shifted_bombs}


def _pick_radius() -> int:
    return int(random.choices(RADII, weights=RADIUS_WEIGHTS, k=1)[0])


def _pick_shape(radius: int) -> set[tuple[int, int]]:
    return set(random.choice(SHAPES[int(radius)]))


def _make_single_cluster(color: int) -> dict[str, object]:
    radius = _pick_radius()
    bomb_cells = _pick_shape(radius)
    halo = random.randint(1, 3)
    support = _diamond((0, 0), radius + halo)

    all_cells = set(support) | set(bomb_cells)
    sr0, sc0, sr1, sc1 = _bbox(all_cells)
    pad = random.randint(2, 4)
    h = sr1 - sr0 + 1 + 2 * pad
    w = sc1 - sc0 + 1 + 2 * pad
    shift_r = pad - sr0
    shift_c = pad - sc0

    support2 = _shift(support, shift_r, shift_c)
    bomb2 = _shift(bomb_cells, shift_r, shift_c)
    target = min(h * w, len(support2) + random.randint(6, 14))
    region = _grow_region(h, w, support2, target)

    arr = np.full((h, w), BLACK, dtype=int)
    for r, c in region:
        arr[r, c] = GREY
    for r, c in bomb2:
        arr[r, c] = color

    return _crop_cluster(
        arr,
        [{"center": (shift_r, shift_c), "radius": radius, "color": color}],
    )


def _make_overlap_cluster(colors: tuple[int, int]) -> dict[str, object] | None:
    radius_a = _pick_radius()
    radius_b = _pick_radius()
    for _ in range(20):
        if radius_b != radius_a:
            break
        radius_b = _pick_radius()
    if radius_a == radius_b:
        return None
    if radius_a > radius_b:
        radius_a, radius_b = radius_b, radius_a
        colors = (colors[1], colors[0])

    cells_a = _pick_shape(radius_a)
    cells_b = _pick_shape(radius_b)
    axis = random.choice(["h", "v"])
    distance_candidates: list[int] = []
    max_d = radius_a + radius_b
    for d in range(2, max_d + 1):
        shift_r, shift_c = (0, d) if axis == "h" else (d, 0)
        moved_b = _shift(cells_b, shift_r, shift_c)
        if cells_a & moved_b:
            continue
        if _touches4(cells_a, moved_b):
            continue
        distance_candidates.append(d)

    if not distance_candidates:
        return None

    distance = random.choice(distance_candidates)
    center_a = (0, 0)
    center_b = (0, distance) if axis == "h" else (distance, 0)
    bomb_a = cells_a
    bomb_b = _shift(cells_b, center_b[0], center_b[1])

    support = _diamond(center_a, radius_a + random.randint(1, 2))
    support |= _diamond(center_b, radius_b + random.randint(1, 2))
    support |= _path(center_a, center_b)

    all_cells = support | bomb_a | bomb_b
    sr0, sc0, sr1, sc1 = _bbox(all_cells)
    pad = random.randint(2, 4)
    h = sr1 - sr0 + 1 + 2 * pad
    w = sc1 - sc0 + 1 + 2 * pad
    shift_r = pad - sr0
    shift_c = pad - sc0

    support2 = _shift(support, shift_r, shift_c)
    bomb_a2 = _shift(bomb_a, shift_r, shift_c)
    bomb_b2 = _shift(bomb_b, shift_r, shift_c)
    center_a2 = (shift_r + center_a[0], shift_c + center_a[1])
    center_b2 = (shift_r + center_b[0], shift_c + center_b[1])

    target = min(h * w, len(support2) + random.randint(8, 18))
    region = _grow_region(h, w, support2, target)

    arr = np.full((h, w), BLACK, dtype=int)
    for r, c in region:
        arr[r, c] = GREY
    for r, c in bomb_a2:
        arr[r, c] = colors[0]
    for r, c in bomb_b2:
        arr[r, c] = colors[1]

    return _crop_cluster(
        arr,
        [
            {"center": center_a2, "radius": radius_a, "color": colors[0]},
            {"center": center_b2, "radius": radius_b, "color": colors[1]},
        ],
    )


def _assemble(clusters: list[dict[str, object]]) -> tuple[np.ndarray, list[dict[str, object]]]:
    shuffled = clusters[:]
    random.shuffle(shuffled)

    if len(shuffled) == 1:
        rows = [shuffled]
    elif len(shuffled) == 2 and random.random() < 0.5:
        rows = [shuffled]
    else:
        split = (len(shuffled) + 1) // 2
        rows = [shuffled[:split], shuffled[split:]]

    row_gap = random.randint(2, 4)
    col_gap = random.randint(2, 4)
    left_pad = random.randint(1, 4)
    right_pad = random.randint(1, 4)
    top_pad = random.randint(1, 4)
    bottom_pad = random.randint(1, 4)

    row_heights: list[int] = []
    row_widths: list[int] = []
    for row in rows:
        height = max(item["grid"].shape[0] for item in row)
        width = sum(item["grid"].shape[1] for item in row) + col_gap * (len(row) - 1)
        row_heights.append(int(height))
        row_widths.append(int(width))

    h = top_pad + sum(row_heights) + row_gap * (len(rows) - 1) + bottom_pad
    w = left_pad + max(row_widths) + right_pad
    grid = np.full((h, w), BLACK, dtype=int)
    bombs: list[dict[str, object]] = []

    y = top_pad
    for row, row_h, row_w in zip(rows, row_heights, row_widths):
        x = left_pad + random.randint(0, max(0, max(row_widths) - row_w))
        for item in row:
            cluster = item["grid"]
            ch, cw = cluster.shape
            dy = y + random.randint(0, max(0, row_h - ch))
            mask = cluster != BLACK
            grid[dy : dy + ch, x : x + cw][mask] = cluster[mask]
            for bomb in item["bombs"]:
                br, bc = bomb["center"]
                bombs.append(
                    {
                        "center": (int(dy + br), int(x + bc)),
                        "radius": int(bomb["radius"]),
                        "color": int(bomb["color"]),
                    }
                )
            x += cw + col_gap
        y += row_h + row_gap

    return grid, bombs


def _coverage_mask(
    shape: tuple[int, int],
    center: tuple[int, int],
    radius: int,
) -> np.ndarray:
    h, w = shape
    cr, cc = center
    cover = np.zeros((h, w), dtype=bool)
    for rr in range(max(0, cr - radius), min(h, cr + radius + 1)):
        rem = radius - abs(rr - cr)
        c0 = max(0, cc - rem)
        c1 = min(w, cc + rem + 1)
        cover[rr, c0:c1] = True
    return cover


def _valid_instance(
    grid: np.ndarray,
    bombs: list[dict[str, object]],
    require_overlap: bool,
) -> bool:
    if len(bombs) < 2:
        return False
    if not any(int(bomb["radius"]) > 1 for bomb in bombs):
        return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False

    grey = grid == GREY
    changed = grey & (out != GREY)
    unchanged = grey & (out == GREY)
    if not bool(changed.any()) or not bool(unchanged.any()):
        return False

    best_radius = np.full(grid.shape, 10**9, dtype=int)
    best_color = np.full(grid.shape, -1, dtype=int)
    cover_count = np.zeros(grid.shape, dtype=int)
    tie_mask = np.zeros(grid.shape, dtype=bool)

    for bomb in bombs:
        radius = int(bomb["radius"])
        color = int(bomb["color"])
        cover = grey & _coverage_mask(grid.shape, bomb["center"], radius)
        if not bool(cover.any()):
            return False

        cover_count += cover.astype(int)
        tie_mask |= cover & (best_radius == radius) & (best_color != -1) & (best_color != color)

        smaller = cover & (radius < best_radius)
        best_radius[smaller] = radius
        best_color[smaller] = color

    if bool(tie_mask.any()):
        return False

    contested = cover_count >= 2
    if require_overlap:
        if not bool(contested.any()):
            return False
        if not bool((contested & changed).any()):
            return False
    else:
        if bool(contested.any()):
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    require_overlap = random.random() < 0.6
    bomb_count = random.randint(2, 4)
    colors = random.sample(PALETTE, bomb_count)

    clusters: list[dict[str, object]] = []

    if require_overlap:
        overlap_cluster = _make_overlap_cluster((colors[0], colors[1]))
        if overlap_cluster is None:
            return None
        clusters.append(overlap_cluster)
        for color in colors[2:]:
            clusters.append(_make_single_cluster(color))
    else:
        for color in colors:
            clusters.append(_make_single_cluster(color))

    grid, bombs = _assemble(clusters)
    if grid.shape[0] > 30 or grid.shape[1] > 30:
        return None
    if not _valid_instance(grid, bombs, require_overlap=require_overlap):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
