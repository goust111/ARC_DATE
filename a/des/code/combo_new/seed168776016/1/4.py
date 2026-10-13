import itertools
import random
from collections import deque

import numpy as np


BG = 0
MOLD_COLOR = 5
MARKER_COLOR = 9
BAR_COLORS = (1, 2, 3, 4, 6, 7, 8)
DIRECTIONS = ("left", "right", "up", "down")
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_RETRIES = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(coords)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _intervals_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(a0, b0) <= min(a1, b1)


def _point_rect_distance(r: int, c: int, bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    dr = 0 if r0 <= r <= r1 else min(abs(r - r0), abs(r - r1))
    dc = 0 if c0 <= c <= c1 else min(abs(c - c0), abs(c - c1))
    return dr + dc


def _bar_mold_distance(bar: dict, mold: dict) -> int:
    return min(_point_rect_distance(r, c, mold["bbox"]) for r, c in bar["coords"])


def _best_assignment(
    molds: list[dict], bars: list[dict]
) -> tuple[tuple[int, ...], bool]:
    n = len(molds)
    assert n == len(bars) and n > 0, "expected one material bar per mold"
    best_perm: tuple[int, ...] | None = None
    best_cost: int | None = None
    unique = True
    for perm in itertools.permutations(range(n)):
        cost = sum(_bar_mold_distance(bars[perm[i]], molds[i]) for i in range(n))
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_perm = perm
            unique = True
        elif cost == best_cost:
            unique = False
    assert best_perm is not None
    return best_perm, unique


def _ordered_holes(holes: np.ndarray, orientation: str) -> list[tuple[int, int]]:
    coords = [(r, c) for r in range(holes.shape[0]) for c in range(holes.shape[1]) if holes[r, c]]
    if orientation == "left":
        coords.sort(key=lambda p: (p[0], -p[1]))
    elif orientation == "right":
        coords.sort(key=lambda p: (p[0], p[1]))
    elif orientation == "up":
        coords.sort(key=lambda p: (-p[0], p[1]))
    else:
        coords.sort(key=lambda p: (p[0], p[1]))
    return coords


def _parse_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    molds: list[dict] = []
    arrows: list[dict] = []
    bars: list[dict] = []

    for coords in _components(g == MOLD_COLOR):
        r0, c0, r1, c1 = _bbox(coords)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 4 or w < 4:
            continue
        local = np.zeros((h, w), dtype=bool)
        for r, c in coords:
            local[r - r0, c - c0] = True
        if not (np.all(local[0, :]) and np.all(local[-1, :]) and np.all(local[:, 0]) and np.all(local[:, -1])):
            continue
        holes = ~local
        if not np.any(holes):
            continue
        molds.append(
            {
                "bbox": (r0, c0, r1, c1),
                "coords": np.array(coords, dtype=int),
                "solid": local,
                "holes": holes,
            }
        )

    special_mask = (g != BG) & (g != MOLD_COLOR)
    for coords in _components(special_mask):
        vals = {int(g[r, c]) for r, c in coords}
        r0, c0, r1, c1 = _bbox(coords)
        rows = sorted({r for r, _ in coords})
        cols = sorted({c for _, c in coords})
        if vals == {MARKER_COLOR}:
            if len(coords) == 3 and (r1 - r0 + 1) == 2 and (c1 - c0 + 1) == 2:
                arrows.append({"bbox": (r0, c0, r1, c1), "coords": coords})
            continue
        if MARKER_COLOR not in vals or len(vals) != 2:
            continue
        marker_pts = [(r, c) for r, c in coords if int(g[r, c]) == MARKER_COLOR]
        if len(marker_pts) != 1 or len(rows) != 1:
            continue
        if cols[-1] - cols[0] + 1 != len(coords):
            continue
        marker_r, marker_c = marker_pts[0]
        if marker_c not in (cols[0], cols[-1]):
            continue
        color = next(iter(vals - {MARKER_COLOR}))
        material = [(r, c) for r, c in coords if int(g[r, c]) == color]
        if len(material) != len(coords) - 1:
            continue
        if marker_c == cols[0]:
            material.sort(key=lambda p: p[1])
            marker_side = "left"
        else:
            material.sort(key=lambda p: -p[1])
            marker_side = "right"
        bars.append(
            {
                "bbox": (r0, c0, r1, c1),
                "coords": coords,
                "marker": (marker_r, marker_c),
                "marker_side": marker_side,
                "material_coords": material,
                "color": int(color),
            }
        )

    molds.sort(key=lambda item: item["bbox"])
    arrows.sort(key=lambda item: item["bbox"])
    bars.sort(key=lambda item: item["bbox"])

    used_arrows: set[int] = set()
    for mold in molds:
        r0, c0, r1, c1 = mold["bbox"]
        matches: list[tuple[str, int]] = []
        for idx, arrow in enumerate(arrows):
            ar0, ac0, ar1, ac1 = arrow["bbox"]
            if ac1 == c0 - 1 and _intervals_overlap(ar0, ar1, r0, r1):
                matches.append(("left", idx))
            if ac0 == c1 + 1 and _intervals_overlap(ar0, ar1, r0, r1):
                matches.append(("right", idx))
            if ar1 == r0 - 1 and _intervals_overlap(ac0, ac1, c0, c1):
                matches.append(("up", idx))
            if ar0 == r1 + 1 and _intervals_overlap(ac0, ac1, c0, c1):
                matches.append(("down", idx))
        assert len(matches) == 1, "each mold must have exactly one adjacent arrow"
        orient, arrow_idx = matches[0]
        assert arrow_idx not in used_arrows, "an arrow cannot label multiple molds"
        used_arrows.add(arrow_idx)
        mold["orientation"] = orient

    assert len(molds) == len(bars) == len(arrows) and molds, "scene must have matched molds, arrows, and bars"
    return {"molds": molds, "bars": bars, "arrows": arrows}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each gray mold's interior holes using its nearest material strip and keep only the stamped molds.

    Input structure:
    - The background is black.
    - Gray 4-connected molds are rectangular blocks whose outer border is solid gray and whose internal black cells form hole patterns.
    - Each mold has one adjacent maroon 3-pixel L marker just outside one side, and each horizontal material strip is a single-color run with one maroon endpoint marker.

    Rule:
    - Detect every gray mold, its hole mask, and the side on which its maroon L marker sits.
    - Pair each material strip with the nearest mold, then order the mold's hole pixels by the arrow-implied scan: left = right-to-left row scan, right = left-to-right row scan, up = bottom-to-top column-preserving scan, down = top-to-bottom row scan.
    - Fill the ordered hole pixels with the paired strip's color so the hole pattern becomes a colored stamp inside the original mold.
    - Return a grid of the same size that keeps only the gray molds with their holes filled; erase all strips and maroon markers.

    Key insight:
    - The gray object is a stencil, the nearby maroon L fixes the fill traversal direction, and the separate horizontal strip contributes the fill color for the nearest stencil.
    """
    g = np.asarray(grid, dtype=int)
    scene = _parse_scene(g)
    molds = scene["molds"]
    bars = scene["bars"]
    assignment, _ = _best_assignment(molds, bars)

    out = np.full_like(g, BG)
    for mold in molds:
        coords = mold["coords"]
        out[coords[:, 0], coords[:, 1]] = MOLD_COLOR

    for mold_idx, mold in enumerate(molds):
        bar = bars[assignment[mold_idx]]
        hole_coords = _ordered_holes(mold["holes"], mold["orientation"])
        values = [int(g[r, c]) for r, c in bar["material_coords"]]
        repeats = (len(hole_coords) + len(values) - 1) // len(values)
        fill_values = (values * repeats)[: len(hole_coords)]
        r0, c0, _, _ = mold["bbox"]
        for (rr, cc), value in zip(hole_coords, fill_values):
            out[r0 + rr, c0 + cc] = value

    return out


def _rect_is_clear(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> bool:
    h, w = grid.shape
    if r0 < 0 or c0 < 0 or r1 >= h or c1 >= w:
        return False
    return bool(np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == BG))


def _component_can_fit(grid: np.ndarray, cells: list[tuple[int, int]], gap: int) -> bool:
    h, w = grid.shape
    cell_set = set(cells)
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if int(grid[r, c]) != BG:
            return False
    for r, c in cells:
        for rr in range(r - gap, r + gap + 1):
            for cc in range(c - gap, c + gap + 1):
                if not (0 <= rr < h and 0 <= cc < w):
                    continue
                if (rr, cc) in cell_set:
                    continue
                if int(grid[rr, cc]) != BG:
                    return False
    return True


def _make_mold_mask(height: int, width: int, rng: random.Random) -> tuple[np.ndarray, int]:
    inner_h = height - 2
    inner_w = width - 2
    hole_cap = min(10, inner_h * inner_w - 1)
    target = rng.randint(3, hole_cap)
    holes = np.zeros((inner_h, inner_w), dtype=bool)
    frontier: list[tuple[int, int]] = []

    seed_count = rng.randint(1, min(3, target))
    for _ in range(seed_count):
        while True:
            rr = rng.randrange(inner_h)
            cc = rng.randrange(inner_w)
            if not holes[rr, cc]:
                holes[rr, cc] = True
                frontier.append((rr, cc))
                break

    while int(holes.sum()) < target:
        grown = False
        if frontier and rng.random() < 0.75:
            br, bc = frontier[rng.randrange(len(frontier))]
            neighbors = []
            for dr, dc in DIRS4:
                nr, nc = br + dr, bc + dc
                if 0 <= nr < inner_h and 0 <= nc < inner_w and not holes[nr, nc]:
                    neighbors.append((nr, nc))
            if neighbors:
                nr, nc = neighbors[rng.randrange(len(neighbors))]
                holes[nr, nc] = True
                frontier.append((nr, nc))
                grown = True
        if grown:
            continue
        empties = np.argwhere(~holes)
        rr, cc = map(int, empties[rng.randrange(len(empties))])
        holes[rr, cc] = True
        frontier.append((rr, cc))

    mask = np.ones((height, width), dtype=bool)
    mask[1:-1, 1:-1] = ~holes
    return mask, int(holes.sum())


def _arrow_cells(
    bbox: tuple[int, int, int, int], orientation: str, rng: random.Random
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    if orientation == "left":
        top = rng.randint(r0, r1 - 1)
        base_r, base_c = top, c0 - 2
        offsets = [(0, 1), (1, 0), (1, 1)]
    elif orientation == "right":
        top = rng.randint(r0, r1 - 1)
        base_r, base_c = top, c1 + 1
        offsets = [(0, 0), (1, 0), (1, 1)]
    elif orientation == "up":
        left = rng.randint(c0, c1 - 1)
        base_r, base_c = r0 - 2, left
        offsets = [(0, 0), (0, 1), (1, 0)]
    else:
        left = rng.randint(c0, c1 - 1)
        base_r, base_c = r1 + 1, left
        offsets = [(0, 1), (1, 0), (1, 1)]
    return [(base_r + dr, base_c + dc) for dr, dc in offsets]


def _bar_candidate(
    grid_shape: tuple[int, int],
    bbox: tuple[int, int, int, int],
    total_len: int,
    side: str,
    marker_left: bool,
    rng: random.Random,
) -> list[tuple[int, int]]:
    h, w = grid_shape
    r0, c0, r1, c1 = bbox
    mid_r = (r0 + r1) // 2
    if side == "up":
        row_lo, row_hi = max(0, r0 - 6), r0 - 2
        col_lo, col_hi = max(0, c0 - total_len), min(w - total_len, c1)
    elif side == "down":
        row_lo, row_hi = r1 + 2, min(h - 1, r1 + 6)
        col_lo, col_hi = max(0, c0 - total_len), min(w - total_len, c1)
    elif side == "left":
        row_lo, row_hi = max(0, mid_r - 2), min(h - 1, mid_r + 2)
        col_lo, col_hi = max(0, c0 - total_len - 6), c0 - total_len - 2
    else:
        row_lo, row_hi = max(0, mid_r - 2), min(h - 1, mid_r + 2)
        col_lo, col_hi = c1 + 2, min(w - total_len, c1 + 6)
    if row_lo > row_hi or col_lo > col_hi:
        return []

    row = rng.randint(row_lo, row_hi)
    start = rng.randint(col_lo, col_hi)
    if marker_left:
        return [(row, start + i) for i in range(total_len)]
    return [(row, start + i) for i in range(total_len)]


def _paint_bar(
    grid: np.ndarray,
    cells: list[tuple[int, int]],
    color: int,
    marker_left: bool,
) -> None:
    ordered = sorted(cells, key=lambda p: p[1])
    marker = ordered[0] if marker_left else ordered[-1]
    for r, c in ordered:
        grid[r, c] = color
    grid[marker] = MARKER_COLOR


def _latent_bar_distance(cells: list[tuple[int, int]], bbox: tuple[int, int, int, int]) -> int:
    return min(_point_rect_distance(r, c, bbox) for r, c in cells)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(18, 30)
    w = rng.randint(18, 30)
    mold_count = rng.choices([1, 2, 3], weights=[2, 4, 3])[0]
    grid = np.zeros((h, w), dtype=int)

    orientations = list(DIRECTIONS)
    rng.shuffle(orientations)
    molds_meta: list[dict] = []

    for idx in range(mold_count):
        placed = False
        for _ in range(80):
            mh = rng.randint(5, 8 if mold_count > 1 else 9)
            mw = rng.randint(5, 8 if mold_count > 1 else 9)
            if h - mh - 4 <= 2 or w - mw - 4 <= 2:
                continue
            r0 = rng.randint(2, h - mh - 2)
            c0 = rng.randint(2, w - mw - 2)
            r1 = r0 + mh - 1
            c1 = c0 + mw - 1
            if not _rect_is_clear(grid, r0 - 2, c0 - 2, r1 + 2, c1 + 2):
                continue

            mask, hole_count = _make_mold_mask(mh, mw, rng)
            for rr in range(mh):
                for cc in range(mw):
                    if mask[rr, cc]:
                        grid[r0 + rr, c0 + cc] = MOLD_COLOR

            bbox = (r0, c0, r1, c1)
            orientation = orientations[idx]
            arrow = _arrow_cells(bbox, orientation, rng)
            if not _component_can_fit(grid, arrow, gap=0):
                for rr in range(mh):
                    for cc in range(mw):
                        if mask[rr, cc]:
                            grid[r0 + rr, c0 + cc] = BG
                continue
            for ar, ac in arrow:
                grid[ar, ac] = MARKER_COLOR

            molds_meta.append(
                {
                    "bbox": bbox,
                    "orientation": orientation,
                    "hole_count": hole_count,
                    "arrow": arrow,
                }
            )
            placed = True
            break
        if not placed:
            return None

    colors = list(BAR_COLORS)
    rng.shuffle(colors)
    bars_meta: list[dict] = []
    for idx, mold in enumerate(molds_meta):
        color = colors[idx]
        total_len = mold["hole_count"] + 1
        placed = False
        for _ in range(120):
            side = rng.choice(DIRECTIONS)
            marker_left = rng.random() < 0.5
            cells = _bar_candidate(grid.shape, mold["bbox"], total_len, side, marker_left, rng)
            if not cells or not _component_can_fit(grid, cells, gap=1):
                continue
            _paint_bar(grid, cells, color, marker_left)
            bars_meta.append({"coords": cells, "target": idx, "color": color})
            placed = True
            break
        if not placed:
            return None

    mold_bboxes = [mold["bbox"] for mold in molds_meta]
    for bar in bars_meta:
        distances = [_latent_bar_distance(bar["coords"], bbox) for bbox in mold_bboxes]
        best = min(distances)
        if distances[bar["target"]] != best or distances.count(best) != 1:
            return None

    try:
        scene = _parse_scene(grid)
        if len(scene["molds"]) != mold_count or len(scene["bars"]) != mold_count:
            return None
        _, unique = _best_assignment(scene["molds"], scene["bars"])
        if not unique:
            return None
        out = main(grid)
        if np.array_equal(out, grid):
            return None
    except AssertionError:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC input")
