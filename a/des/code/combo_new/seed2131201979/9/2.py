import random
from collections import Counter, defaultdict, deque

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 0), (0, 1),
    (1, -1), (1, 0), (1, 1),
]


GRID_MIN = 22
GRID_MAX = 30
PAIR_COUNT_OPTIONS = (1, 2, 3, 4)
MAX_SHIFT = 6
PLACEMENT_GAP = 1
GEN_ATTEMPTS = 400
MASK_ATTEMPTS = 300


def _get_bg(grid: np.ndarray) -> int:
    counts = Counter(map(int, grid.flat))
    return max(counts.items(), key=lambda kv: (kv[1], -kv[0]))[0]


def _component_bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == int(bg):
                continue
            color = int(grid[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
            coords = np.array(cells, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _component_bbox(coords),
                    "area": int(len(cells)),
                }
            )
    return comps


def _component_mask(comp: dict) -> np.ndarray:
    r0, c0, r1, c1 = comp["bbox"]
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel = comp["coords"] - np.array([r0, c0], dtype=int)
    mask[rel[:, 0], rel[:, 1]] = True
    return mask


def _crop_non_bg(grid: np.ndarray, bg: int) -> np.ndarray:
    pts = np.argwhere(grid != int(bg))
    assert pts.size > 0
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _center_aligned_cells(comp: dict, anchor_r2: int, anchor_c2: int) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = comp["bbox"]
    cr2 = int(r0 + r1)
    cc2 = int(c0 + c1)
    cells = []
    for r, c in comp["coords"]:
        nr2 = int(anchor_r2 + 2 * int(r) - cr2)
        nc2 = int(anchor_c2 + 2 * int(c) - cc2)
        assert nr2 % 2 == 0 and nc2 % 2 == 0
        cells.append((nr2 // 2, nc2 // 2))
    return cells


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = int(_get_bg(g))
    comps = _find_components(g, bg)
    assert comps

    by_color: dict[int, list[dict]] = defaultdict(list)
    for comp in comps:
        by_color[int(comp["color"])].append(comp)

    painted = []
    for color, pair in by_color.items():
        assert len(pair) == 2
        mask0 = _component_mask(pair[0])
        mask1 = _component_mask(pair[1])
        assert mask0.shape == mask1.shape
        assert np.array_equal(mask0, np.rot90(mask1, 2))

        b0 = pair[0]["bbox"]
        b1 = pair[1]["bbox"]
        anchor_r2 = (int(b0[0] + b0[2]) + int(b1[0] + b1[2])) // 2
        anchor_c2 = (int(b0[1] + b0[3]) + int(b1[1] + b1[3])) // 2
        for comp in pair:
            for r, c in _center_aligned_cells(comp, anchor_r2, anchor_c2):
                painted.append((int(r), int(c), int(color)))

    min_r = min(r for r, _c, _color in painted)
    min_c = min(c for _r, c, _color in painted)
    max_r = max(r for r, _c, _color in painted)
    max_c = max(c for _r, c, _color in painted)
    out = np.full((max_r - min_r + 1, max_c - min_c + 1), int(bg), dtype=int)
    for r, c, color in painted:
        rr = int(r - min_r)
        cc = int(c - min_c)
        if out[rr, cc] not in (int(bg), int(color)):
            raise AssertionError("merged pairs overlap with different colors")
        out[rr, cc] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Merge each same-color fragment pair into one centrally aligned composite, then tight-crop.

    Input structure:
    - The background is the dominant color.
    - The foreground consists of monochrome 4-connected sprite fragments.
    - Each non-background color appears in exactly two separate sprites.
    - Within each color pair, the two sprites have the same tight bbox size and are 180-degree rotations of each other.

    Rule:
    - Group the sprites by color to recover the unique fragment pairs.
    - For each pair, compute the midpoint between the two bbox centers as the merge anchor.
    - Re-express both sprites around that shared anchor while keeping their local pixels, so the pair collapses into one overlaid composite whose filled cells are the union of the two opposite fragments.
    - Erase the scattered originals and output the tight crop containing all merged composites.

    Key insight:
    - Color identifies which fragments belong together, while the 180-degree relation tells you that both pieces should be brought into one shared center frame before taking their union.
    """
    return _solve(grid)


def _mask_connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return False
    start = tuple(map(int, pts[0]))
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc]:
                cell = (int(nr), int(nc))
                if cell not in seen:
                    seen.add(cell)
                    q.append(cell)
    return len(seen) == int(mask.sum())


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _random_fragment_mask() -> np.ndarray:
    for _ in range(MASK_ATTEMPTS):
        h = random.randint(2, 5)
        w = random.randint(2, 5)
        target = random.randint(3, h * w - 1)
        mask = np.zeros((h, w), dtype=bool)
        start = (random.randrange(h), random.randrange(w))
        mask[start] = True
        while int(mask.sum()) < target:
            frontier = set()
            for r, c in np.argwhere(mask):
                for dr, dc in DIR4:
                    nr, nc = int(r) + dr, int(c) + dc
                    if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                        frontier.add((nr, nc))
            if not frontier:
                break
            nr, nc = random.choice(list(frontier))
            mask[nr, nc] = True
        if int(mask.sum()) != target:
            continue

        mask = _trim_mask(mask)
        if min(mask.shape) < 2:
            continue
        if not _mask_connected(mask):
            continue

        rot = np.rot90(mask, 2)
        if np.array_equal(mask, rot):
            continue

        merged = mask | rot
        if int(merged.sum()) < int(mask.sum()) + 2:
            continue
        if not _mask_connected(merged):
            continue
        return mask
    raise RuntimeError("failed to generate a valid fragment mask")


def _placed_cells(mask: np.ndarray, top: int, left: int) -> list[tuple[int, int]]:
    pts = np.argwhere(mask)
    return [(int(top + r), int(left + c)) for r, c in pts]


def _cells_fit(cells: list[tuple[int, int]], h: int, w: int) -> bool:
    return all(0 <= r < h and 0 <= c < w for r, c in cells)


def _touch_or_overlap(cells_a: list[tuple[int, int]], cells_b: list[tuple[int, int]]) -> bool:
    occupied_b = set(cells_b)
    expanded_b = set()
    for r, c in occupied_b:
        for dr, dc in DIR8:
            expanded_b.add((int(r + dr), int(c + dc)))
    return any((int(r), int(c)) in expanded_b for r, c in cells_a)


def _can_place(cells: list[tuple[int, int]], forbid: np.ndarray) -> bool:
    h, w = forbid.shape
    if not _cells_fit(cells, h, w):
        return False
    return all(not forbid[r, c] for r, c in cells)


def _mark_forbid(forbid: np.ndarray, cells: list[tuple[int, int]]) -> None:
    h, w = forbid.shape
    for r, c in cells:
        for dr, dc in DIR8:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                forbid[nr, nc] = True


def _candidate_shifts() -> list[tuple[int, int]]:
    shifts = []
    for d in range(2, MAX_SHIFT + 1):
        shifts.extend(
            [
                (d, 0), (-d, 0), (0, d), (0, -d),
                (d, d), (d, -d), (-d, d), (-d, -d),
                (d, 1), (-d, 1), (1, d), (1, -d),
                (d, -1), (-d, -1), (-1, d), (-1, -d),
            ]
        )
    random.shuffle(shifts)
    return shifts


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    bg = random.randrange(10)
    pair_count = random.choice(PAIR_COUNT_OPTIONS)
    colors = random.sample([c for c in range(10) if c != bg], pair_count)

    specs = []
    for color in colors:
        mask_a = _random_fragment_mask()
        mask_b = np.rot90(mask_a, 2)
        merged = mask_a | mask_b
        specs.append(
            {
                "color": int(color),
                "mask_a": mask_a,
                "mask_b": mask_b,
                "merged": merged,
                "shape": merged.shape,
            }
        )

    out_forbid = np.zeros((h, w), dtype=bool)
    out_canvas = np.full((h, w), int(bg), dtype=int)
    placed_specs = []
    for spec in sorted(specs, key=lambda item: int(item["merged"].sum()), reverse=True):
        mh, mw = spec["shape"]
        row_lo = MAX_SHIFT
        row_hi = h - mh - MAX_SHIFT
        col_lo = MAX_SHIFT
        col_hi = w - mw - MAX_SHIFT
        if row_lo > row_hi or col_lo > col_hi:
            return None

        success = False
        for _ in range(200):
            top = random.randint(row_lo, row_hi)
            left = random.randint(col_lo, col_hi)
            cells = _placed_cells(spec["merged"], top, left)
            if not _can_place(cells, out_forbid):
                continue
            for r, c in cells:
                out_canvas[r, c] = int(spec["color"])
            _mark_forbid(out_forbid, cells)
            placed_specs.append({**spec, "out_top": int(top), "out_left": int(left)})
            success = True
            break
        if not success:
            return None

    in_forbid = np.zeros((h, w), dtype=bool)
    grid = np.full((h, w), int(bg), dtype=int)
    for spec in placed_specs:
        mh, mw = spec["shape"]
        shift_ok = False
        for dr, dc in _candidate_shifts():
            top_a = int(spec["out_top"] + dr)
            left_a = int(spec["out_left"] + dc)
            top_b = int(spec["out_top"] - dr)
            left_b = int(spec["out_left"] - dc)

            cells_a = _placed_cells(spec["mask_a"], top_a, left_a)
            cells_b = _placed_cells(spec["mask_b"], top_b, left_b)
            if not _cells_fit(cells_a, h, w) or not _cells_fit(cells_b, h, w):
                continue
            if _touch_or_overlap(cells_a, cells_b):
                continue
            if not _can_place(cells_a, in_forbid) or not _can_place(cells_b, in_forbid):
                continue

            for r, c in cells_a:
                grid[r, c] = int(spec["color"])
            for r, c in cells_b:
                grid[r, c] = int(spec["color"])
            _mark_forbid(in_forbid, cells_a)
            _mark_forbid(in_forbid, cells_b)
            shift_ok = True
            break
        if not shift_ok:
            return None

    expected = _crop_non_bg(out_canvas, int(bg))
    actual = _solve(grid)
    if not np.array_equal(actual, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for ARC pair-merge puzzle")
