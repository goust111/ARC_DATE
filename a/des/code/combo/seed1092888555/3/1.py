import numpy as np


BG = 0
ORTHO_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_RETRIES = 500


def _find_components(mask: np.ndarray):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in ORTHO_DIRS:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _bbox(coords):
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _legend_mapping(grid: np.ndarray):
    assert grid.ndim == 2
    h, w = grid.shape
    assert h >= 3 and w >= 2
    src_row = grid[h - 2]
    dst_row = grid[h - 1]
    cols = [c for c in range(w) if int(src_row[c]) != BG and int(dst_row[c]) != BG]
    assert cols, "legend not found"
    assert cols == list(range(cols[0], cols[-1] + 1)), "legend columns must be contiguous"
    mapping = {}
    for c in cols:
        src = int(src_row[c])
        dst = int(dst_row[c])
        assert src not in mapping or mapping[src] == dst, "ambiguous legend mapping"
        mapping[src] = dst
    return mapping, cols[0], cols[-1]


def main(grid: np.ndarray) -> np.ndarray:
    """Flip each object according to bbox aspect ratio and recolor via the bottom legend."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    h, w = g.shape
    mapping, _legend_c0, _legend_c1 = _legend_mapping(g)

    work = g.copy()
    work[h - 2 :, :] = BG
    fg_mask = work != BG
    comps = _find_components(fg_mask)

    out = np.zeros_like(g)
    for coords in comps:
        color = int(work[coords[0]])
        assert all(int(work[r, c]) == color for r, c in coords), "objects must be monochrome"
        assert color in mapping, "object color missing from legend"

        r0, c0, r1, c1 = _bbox(coords)
        local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in coords:
            local[r - r0, c - c0] = True

        bh, bw = local.shape
        if bw > bh:
            local = np.fliplr(local)
        elif bh > bw:
            local = np.flipud(local)

        dst_color = mapping[color]
        rr, cc = np.where(local)
        out[r0 + rr, c0 + cc] = dst_color

    return out


def _random_connected_mask(height: int, width: int, rng: np.random.Generator):
    area_min = max(2, (height * width) // 3)
    area_max = max(area_min, height * width - 1)
    target = int(rng.integers(area_min, area_max + 1))

    for _ in range(120):
        cells = {(int(rng.integers(height)), int(rng.integers(width)))}
        frontier = set(cells)
        while len(cells) < target and frontier:
            cr, cc = frontier.pop()
            neighbors = []
            for dr, dc in ORTHO_DIRS:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    neighbors.append((nr, nc))
            rng.shuffle(neighbors)
            added = 0
            for nr, nc in neighbors:
                if len(cells) >= target:
                    break
                if rng.random() < 0.72 or added == 0:
                    cells.add((nr, nc))
                    frontier.add((nr, nc))
                    added += 1
            if not frontier and len(cells) < target:
                frontier = set(cells)

        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        if min(rows) != 0 or max(rows) != height - 1 or min(cols) != 0 or max(cols) != width - 1:
            continue
        mask = np.zeros((height, width), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        if len(_find_components(mask)) != 1:
            continue
        if np.all(mask) or np.count_nonzero(mask) < 2:
            continue
        return mask
    return None


def _sample_dims(kind: str, rng: np.random.Generator):
    if kind == "horizontal":
        h = int(rng.integers(2, 5))
        w = int(rng.integers(h + 1, min(8, h + 4) + 1))
    elif kind == "vertical":
        w = int(rng.integers(2, 5))
        h = int(rng.integers(w + 1, min(8, w + 4) + 1))
    else:
        side = int(rng.integers(2, 6))
        h = side
        w = side
    return h, w


def _category(mask: np.ndarray):
    h, w = mask.shape
    if w > h:
        return "horizontal"
    if h > w:
        return "vertical"
    return "square"


def _can_place(occupied: np.ndarray, top: int, left: int, mask: np.ndarray):
    h, w = mask.shape
    r0 = max(0, top - 1)
    c0 = max(0, left - 1)
    r1 = min(occupied.shape[0], top + h + 1)
    c1 = min(occupied.shape[1], left + w + 1)
    if occupied[r0:r1, c0:c1].any():
        return False
    return True


def _place_mask(grid: np.ndarray, occupied: np.ndarray, top: int, left: int, mask: np.ndarray, color: int):
    h, w = mask.shape
    region = grid[top : top + h, left : left + w]
    region[mask] = color
    occupied[top : top + h, left : left + w] |= mask



def _gen_attempt(rng: np.random.Generator):
    grid_h = int(rng.integers(12, 20))
    grid_w = int(rng.integers(12, 22))
    obj_count = int(rng.integers(3, 7))
    if grid_w < obj_count + 4:
        return None

    base_kinds = ["horizontal", "vertical", "square"]
    kinds = base_kinds.copy()
    while len(kinds) < obj_count:
        kinds.append(base_kinds[int(rng.integers(0, 3))])
    rng.shuffle(kinds)

    available_colors = list(range(1, 10))
    rng.shuffle(available_colors)
    obj_colors = available_colors[:obj_count]
    remaining = [c for c in range(1, 10) if c not in obj_colors]
    rng.shuffle(remaining)
    if len(remaining) < obj_count:
        return None
    target_colors = remaining[:obj_count]
    mapping = dict(zip(obj_colors, target_colors))

    grid = np.zeros((grid_h, grid_w), dtype=int)
    occupied = np.zeros((grid_h - 2, grid_w), dtype=bool)
    objects = []

    for color, kind in zip(obj_colors, kinds):
        placed = False
        for _ in range(200):
            mh, mw = _sample_dims(kind, rng)
            if mh > grid_h - 3 or mw > grid_w:
                continue
            mask = _random_connected_mask(mh, mw, rng)
            if mask is None or _category(mask) != kind:
                continue
            top = int(rng.integers(0, grid_h - 2 - mh + 1))
            left = int(rng.integers(0, grid_w - mw + 1))
            if not _can_place(occupied, top, left, mask):
                continue
            _place_mask(grid, occupied, top, left, mask, color)
            objects.append((color, top, left, mask.copy()))
            placed = True
            break
        if not placed:
            return None

    legend_width = obj_count
    if legend_width > grid_w:
        return None
    legend_start = int(rng.integers(0, grid_w - legend_width + 1))
    for idx, color in enumerate(obj_colors):
        c = legend_start + idx
        grid[grid_h - 2, c] = color
        grid[grid_h - 1, c] = mapping[color]

    if np.any(grid[grid_h - 2 :, :legend_start] != BG) or np.any(grid[grid_h - 2 :, legend_start + legend_width :] != BG):
        return None

    # Validate assumptions explicitly.
    mapping2, c0, c1 = _legend_mapping(grid)
    if c0 != legend_start or c1 != legend_start + legend_width - 1 or mapping2 != mapping:
        return None
    work = grid.copy()
    work[-2:, :] = BG
    comps = _find_components(work != BG)
    if len(comps) != obj_count:
        return None
    comp_colors = sorted(int(work[r, c]) for comp in comps for r, c in [comp[0]])
    if sorted(obj_colors) != comp_colors:
        return None
    if set(_category(mask) for _, _, _, mask in objects) != {"horizontal", "vertical", "square"}:
        return None
    out = main(grid)
    if out.shape != grid.shape:
        return None
    if np.any(out[-2:, :] != BG):
        return None
    return grid



def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
