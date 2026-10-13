import random
from collections import Counter, deque

import numpy as np


BG = 8
COLORS = tuple(range(10))
GEN_ATTEMPTS = 400

GRID_H_RANGE = (17, 30)
GRID_W_RANGE = (17, 30)
ACTIVE_COUNT_CHOICES = [1, 2, 3]
ACTIVE_COUNT_WEIGHTS = [4, 4, 1]
PASSIVE_COUNT_CHOICES = [0, 1, 2]
PASSIVE_COUNT_WEIGHTS = [2, 3, 2]
HOLE_COUNT_RANGE = (1, 4)

HOLE_H_RANGE = (2, 3)
HOLE_W_RANGE = (2, 3)
SOLID_H_RANGE = (3, 8)
SOLID_W_RANGE = (3, 8)
MAX_PROTRUSIONS = 2
MAX_PROTRUSION_LEN = 3
MAX_PROTRUSION_THICK = 3
MIN_SPACING = 1


def neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not bool(mask[r, c]):
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and bool(mask[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _outside_non_object(obj_mask: np.ndarray) -> np.ndarray:
    h, w = obj_mask.shape
    outside = np.zeros((h, w), dtype=bool)
    q = deque()

    for r in range(h):
        for c in (0, w - 1):
            if not bool(obj_mask[r, c]) and not bool(outside[r, c]):
                outside[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not bool(obj_mask[r, c]) and not bool(outside[r, c]):
                outside[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and not bool(obj_mask[nr, nc]) and not bool(outside[nr, nc]):
                outside[nr, nc] = True
                q.append((nr, nc))
    return outside


def _hole_infos(grid: np.ndarray, obj_mask: np.ndarray, obj_color: int, bg: int) -> list[tuple[list[tuple[int, int]], int]]:
    coords = [(int(r), int(c)) for r, c in np.argwhere(obj_mask)]
    r0, c0, r1, c1 = _bbox(coords)
    region_mask = obj_mask[r0 : r1 + 1, c0 : c1 + 1]
    inv = ~region_mask
    infos: list[tuple[list[tuple[int, int]], int]] = []

    for comp in _components(inv):
        if any(r == 0 or c == 0 or r == inv.shape[0] - 1 or c == inv.shape[1] - 1 for r, c in comp):
            continue

        cells = [(r0 + r, c0 + c) for r, c in comp]
        markers = [
            int(grid[gr, gc])
            for gr, gc in cells
            if int(grid[gr, gc]) not in (bg, obj_color)
        ]
        if not markers:
            continue
        marker_set = set(markers)
        if len(marker_set) != 1:
            continue
        infos.append((cells, markers[0]))
    return infos


def _expansion_cells(grid: np.ndarray, obj_mask: np.ndarray, steps: int, bg: int) -> set[tuple[int, int]]:
    if steps <= 0:
        return set()

    h, w = grid.shape
    outside = _outside_non_object(obj_mask)
    dist = -np.ones((h, w), dtype=int)
    q = deque()

    for r, c in np.argwhere(outside & (grid == bg)):
        rr = int(r)
        cc = int(c)
        for nr, nc in neighbors4(rr, cc):
            if 0 <= nr < h and 0 <= nc < w and bool(obj_mask[nr, nc]):
                dist[rr, cc] = 1
                q.append((rr, cc))
                break

    while q:
        r, c = q.popleft()
        if dist[r, c] >= steps:
            continue
        for nr, nc in neighbors4(r, c):
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if dist[nr, nc] != -1 or int(grid[nr, nc]) != bg:
                continue
            dist[nr, nc] = dist[r, c] + 1
            q.append((nr, nc))

    out = set()
    for r, c in np.argwhere((dist >= 1) & (dist <= steps)):
        out.add((int(r), int(c)))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand outward from each hollow object by as many layers as its hole count, using the hole-marker color.

    Input structure:
    - The background is azure.
    - Several 4-connected foreground objects with distinct body colors are scattered on the grid.
    - Some objects contain enclosed holes: interior non-body regions surrounded by the object.
    - Each hole contains exactly one marker pixel, and all holes of the same object share one marker color.

    Rule:
    - For every foreground object that encloses one or more holes, count its holes.
    - Use the shared marker color inside those holes to paint an outward 4-neighbor expansion from the object's outer contour.
    - Paint exactly N layers, where N is the hole count, and only over background pixels; other foreground pixels stay untouched.
    - Preserve the original object colors, the hole backgrounds, and the marker pixels themselves.

    Key insight:
    - The number of enclosed holes determines the expansion distance, while the marker color inside the holes determines the expansion color.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    bg = BG if int(BG) in set(map(int, np.unique(g))) else int(Counter(map(int, g.ravel())).most_common(1)[0][0])

    planned: dict[tuple[int, int], int] = {}

    for color in sorted(int(c) for c in np.unique(g) if int(c) != bg):
        for comp in _components(g == color):
            obj_mask = np.zeros_like(g, dtype=bool)
            for r, c in comp:
                obj_mask[r, c] = True

            holes = _hole_infos(g, obj_mask, color, bg)
            if not holes:
                continue

            marker_colors = {marker for _cells, marker in holes}
            if not marker_colors:
                continue
            marker_color = Counter(marker for _cells, marker in holes).most_common(1)[0][0]
            if len(marker_colors) != 1:
                marker_color = min(marker_colors)

            for r, c in _expansion_cells(g, obj_mask, len(holes), bg):
                planned.setdefault((r, c), int(marker_color))

    for (r, c), color in planned.items():
        if int(out[r, c]) == bg:
            out[r, c] = int(color)
    return out


def _crop_patch(patch: np.ndarray) -> np.ndarray:
    rows, cols = np.where(patch != BG)
    if len(rows) == 0:
        raise ValueError("empty patch")
    r0, r1 = int(rows.min()), int(rows.max())
    c0, c1 = int(cols.min()), int(cols.max())
    return patch[r0 : r1 + 1, c0 : c1 + 1]


def _choose_layout(hole_count: int) -> tuple[int, int]:
    if hole_count == 1:
        return 1, 1
    if hole_count == 2:
        return random.choice([(1, 2), (2, 1)])
    if hole_count == 3:
        return random.choice([(1, 3), (3, 1), (2, 2)])
    return 2, 2


def _attach_protrusions(mask: np.ndarray, base_r0: int, base_c0: int, base_h: int, base_w: int) -> None:
    n = random.randint(0, MAX_PROTRUSIONS)
    h, w = mask.shape
    for _ in range(n):
        side = random.choice("TBLR")
        length = random.randint(1, MAX_PROTRUSION_LEN)
        thick = random.randint(1, MAX_PROTRUSION_THICK)

        if side in "TB":
            width = random.randint(1, max(1, min(base_w, 4)))
            c0 = base_c0 + random.randint(0, base_w - width)
            if side == "T":
                r1 = base_r0
                r0 = max(0, r1 - length)
            else:
                r0 = base_r0 + base_h
                r1 = min(h, r0 + length)
            mid = base_r0 if side == "T" else base_r0 + base_h - 1
            c1 = min(w, c0 + width)
            mask[r0:r1, c0:c1] = True
            mask[mid : mid + 1, c0 : min(w, c0 + max(width, thick))] = True
        else:
            height = random.randint(1, max(1, min(base_h, 4)))
            r0 = base_r0 + random.randint(0, base_h - height)
            if side == "L":
                c1 = base_c0
                c0 = max(0, c1 - length)
            else:
                c0 = base_c0 + base_w
                c1 = min(w, c0 + length)
            mid = base_c0 if side == "L" else base_c0 + base_w - 1
            r1 = min(h, r0 + height)
            mask[r0:r1, c0:c1] = True
            mask[r0 : min(h, r0 + max(height, thick)), mid : mid + 1] = True


def _object_patch(body_color: int, marker_color: int | None, hole_count: int) -> np.ndarray:
    if hole_count == 0:
        base_h = random.randint(*SOLID_H_RANGE)
        base_w = random.randint(*SOLID_W_RANGE)
        margin = MAX_PROTRUSION_LEN + 1
        mask = np.zeros((base_h + 2 * margin, base_w + 2 * margin), dtype=bool)
        base_r0 = base_c0 = margin
        mask[base_r0 : base_r0 + base_h, base_c0 : base_c0 + base_w] = True
        _attach_protrusions(mask, base_r0, base_c0, base_h, base_w)
        patch = np.full(mask.shape, BG, dtype=int)
        patch[mask] = int(body_color)
        return _crop_patch(patch)

    layout_rows, layout_cols = _choose_layout(hole_count)
    hole_sizes = [
        (random.randint(*HOLE_H_RANGE), random.randint(*HOLE_W_RANGE))
        for _ in range(hole_count)
    ]
    row_heights = [0] * layout_rows
    col_widths = [0] * layout_cols
    for idx, (hh, ww) in enumerate(hole_sizes):
        rr = idx // layout_cols
        cc = idx % layout_cols
        row_heights[rr] = max(row_heights[rr], hh)
        col_widths[cc] = max(col_widths[cc], ww)

    base_h = 2 + sum(row_heights) + max(0, layout_rows - 1)
    base_w = 2 + sum(col_widths) + max(0, layout_cols - 1)
    margin = MAX_PROTRUSION_LEN + 1
    mask = np.zeros((base_h + 2 * margin, base_w + 2 * margin), dtype=bool)
    base_r0 = base_c0 = margin
    mask[base_r0 : base_r0 + base_h, base_c0 : base_c0 + base_w] = True
    _attach_protrusions(mask, base_r0, base_c0, base_h, base_w)

    patch = np.full(mask.shape, BG, dtype=int)
    patch[mask] = int(body_color)

    idx = 0
    cur_r = base_r0 + 1
    for rr in range(layout_rows):
        cur_c = base_c0 + 1
        for cc in range(layout_cols):
            if idx >= hole_count:
                break
            hh, ww = hole_sizes[idx]
            extra_r = row_heights[rr] - hh
            extra_c = col_widths[cc] - ww
            hole_r0 = cur_r + random.randint(0, extra_r)
            hole_c0 = cur_c + random.randint(0, extra_c)
            patch[hole_r0 : hole_r0 + hh, hole_c0 : hole_c0 + ww] = BG
            mr = hole_r0 + random.randrange(hh)
            mc = hole_c0 + random.randrange(ww)
            patch[mr, mc] = int(marker_color)
            idx += 1
            cur_c += col_widths[cc] + 1
        cur_r += row_heights[rr] + 1

    return _crop_patch(patch)


def _reserve_box(top: int, left: int, shape: tuple[int, int], pad: int, grid_shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = shape
    gh, gw = grid_shape
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(gh, top + h + pad)
    c1 = min(gw, left + w + pad)
    return r0, c0, r1, c1


def _place_specs(specs: list[dict], grid_shape: tuple[int, int]) -> list[dict] | None:
    gh, gw = grid_shape
    reserved = np.zeros((gh, gw), dtype=bool)
    placed: list[dict] = []

    ordered = sorted(
        specs,
        key=lambda spec: (spec["patch"].shape[0] + 2 * spec["pad"]) * (spec["patch"].shape[1] + 2 * spec["pad"]),
        reverse=True,
    )

    for spec in ordered:
        patch = spec["patch"]
        ph, pw = patch.shape
        if ph > gh or pw > gw:
            return None
        candidates = []
        for top in range(gh - ph + 1):
            for left in range(gw - pw + 1):
                r0, c0, r1, c1 = _reserve_box(top, left, patch.shape, spec["pad"], grid_shape)
                if bool(reserved[r0:r1, c0:c1].any()):
                    continue
                candidates.append((top, left))
        if not candidates:
            return None

        top, left = random.choice(candidates)
        r0, c0, r1, c1 = _reserve_box(top, left, patch.shape, spec["pad"], grid_shape)
        reserved[r0:r1, c0:c1] = True
        placed.append({**spec, "top": top, "left": left})
    return placed


def _render_grid(h: int, w: int, placed: list[dict]) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    for spec in placed:
        patch = spec["patch"]
        top = spec["top"]
        left = spec["left"]
        view = grid[top : top + patch.shape[0], left : left + patch.shape[1]]
        mask = patch != BG
        view[mask] = patch[mask]
    return grid


def _build_specs() -> list[dict]:
    active_count = random.choices(ACTIVE_COUNT_CHOICES, weights=ACTIVE_COUNT_WEIGHTS, k=1)[0]
    passive_count = random.choices(PASSIVE_COUNT_CHOICES, weights=PASSIVE_COUNT_WEIGHTS, k=1)[0]
    total = active_count + passive_count

    object_colors = random.sample([c for c in COLORS if c != BG], total)
    marker_pool = [c for c in COLORS if c != BG]

    specs: list[dict] = []
    for idx in range(active_count):
        body_color = object_colors[idx]
        marker_choices = [c for c in marker_pool if c != body_color]
        marker_color = random.choice(marker_choices)
        holes = random.randint(*HOLE_COUNT_RANGE)
        patch = _object_patch(body_color, marker_color, holes)
        specs.append(
            {
                "patch": patch,
                "hole_count": holes,
                "pad": holes + MIN_SPACING,
                "body_color": body_color,
                "marker_color": marker_color,
            }
        )

    for idx in range(active_count, total):
        body_color = object_colors[idx]
        patch = _object_patch(body_color, None, 0)
        specs.append(
            {
                "patch": patch,
                "hole_count": 0,
                "pad": MIN_SPACING,
                "body_color": body_color,
                "marker_color": None,
            }
        )

    random.shuffle(specs)
    return specs


def _has_hollow_and_solid(specs: list[dict]) -> bool:
    holes = [spec["hole_count"] for spec in specs]
    return any(h > 0 for h in holes) and any(h == 0 for h in holes)


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    specs = _build_specs()

    placed = _place_specs(specs, (h, w))
    if placed is None:
        return None

    inp = _render_grid(h, w, placed)
    out = main(inp)

    if np.array_equal(inp, out):
        return None
    if max(inp.shape) > 30 or max(out.shape) > 30:
        return None
    if not _has_hollow_and_solid(specs) and random.random() < 0.5:
        return None

    changed = int(np.count_nonzero(inp != out))
    if changed < 3:
        return None

    active_holes = [spec["hole_count"] for spec in specs if spec["hole_count"] > 0]
    if not active_holes:
        return None

    if max(active_holes) >= min(h, w):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
