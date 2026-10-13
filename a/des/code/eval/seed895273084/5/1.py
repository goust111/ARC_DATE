import random
from collections import Counter, deque

import numpy as np


BG = 0
GEN_RETRIES = 400
GRID_H_RANGE = (18, 30)
GRID_W_RANGE = (18, 30)
PALETTE_ENTRY_RANGE = (2, 4)
OBJECT_COUNT_RANGE = (1, 3)
TEXTURE_SIZES = (2, 3)
OBJECT_INNER_H_RANGE = (3, 6)
OBJECT_INNER_W_RANGE = (3, 6)
OBJECT_AREA_RANGE = (5, 12)
PLACE_RETRIES = 120
GROW_RETRIES = 20


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _background_color(grid: np.ndarray) -> int:
    if np.any(grid == BG):
        return BG
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _edge_main_color(grid: np.ndarray, bg: int) -> int:
    h, w = grid.shape
    edge_vals = []
    for c in range(w):
        if int(grid[0, c]) != bg:
            edge_vals.append(int(grid[0, c]))
        if int(grid[h - 1, c]) != bg:
            edge_vals.append(int(grid[h - 1, c]))
    for r in range(1, h - 1):
        if int(grid[r, 0]) != bg:
            edge_vals.append(int(grid[r, 0]))
        if int(grid[r, w - 1]) != bg:
            edge_vals.append(int(grid[r, w - 1]))
    if edge_vals:
        return Counter(edge_vals).most_common(1)[0][0]

    vals, counts = np.unique(grid, return_counts=True)
    non_bg = [(int(v), int(c)) for v, c in zip(vals, counts) if int(v) != bg]
    assert non_bg, "expected at least one non-background color"
    return max(non_bg, key=lambda item: item[1])[0]


def _parse_palette(sub: np.ndarray, main_color: int, bg: int) -> dict[int, np.ndarray] | None:
    for k in range(4):
        rot = np.rot90(sub, k)
        h, w = rot.shape
        tex_size = w - 4
        if tex_size not in TEXTURE_SIZES:
            continue
        step = tex_size + 1
        if h < step + 1 or (h - 1) % step != 0:
            continue
        n_rows = (h - 1) // step
        if n_rows < 2:
            continue
        sep_rows = set(range(0, h, step))
        if sep_rows != set(range(0, h, step)):
            continue
        if not all(np.all(rot[r, :] == main_color) for r in sep_rows):
            continue
        if not np.all(rot[:, 0] == main_color):
            continue
        if not np.all(rot[:, 2] == main_color):
            continue
        if not np.all(rot[:, w - 1] == main_color):
            continue

        mapping: dict[int, np.ndarray] = {}
        ok = True
        for idx in range(n_rows):
            r0 = 1 + idx * step
            band = rot[r0 : r0 + tex_size, :]
            label_col = band[:, 1]
            non_bg_rows = [rr for rr in range(tex_size) if int(label_col[rr]) != bg]
            if len(non_bg_rows) != 1:
                ok = False
                break
            label_row = non_bg_rows[0]
            label_color = int(label_col[label_row])
            if label_color in (bg, main_color) or label_color in mapping:
                ok = False
                break
            if np.any((label_col != bg) & (label_col != label_color)):
                ok = False
                break

            tex = band[:, 3 : 3 + tex_size]
            if tex.shape != (tex_size, tex_size):
                ok = False
                break
            if np.any((tex == bg) | (tex == main_color)):
                ok = False
                break
            if len(np.unique(tex)) < 2:
                ok = False
                break
            if np.any(band[:, 0] != main_color) or np.any(band[:, 2] != main_color) or np.any(
                band[:, w - 1] != main_color
            ):
                ok = False
                break
            mapping[label_color] = tex.copy()

        if ok and mapping:
            return mapping
    return None


def _find_palette(grid: np.ndarray, main_color: int, bg: int) -> tuple[tuple[int, int, int, int], dict[int, np.ndarray]]:
    main_mask = grid == main_color
    h, w = grid.shape
    candidates = []
    for comp in _components(main_mask):
        if not any(r in (0, h - 1) or c in (0, w - 1) for r, c in comp):
            continue
        r0, c0, r1, c1 = _bbox(comp)
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        candidates.append((area, (r0, c0, r1, c1)))
    candidates.sort(reverse=True)

    for _area, bbox in candidates:
        r0, c0, r1, c1 = bbox
        mapping = _parse_palette(grid[r0 : r1 + 1, c0 : c1 + 1], main_color, bg)
        if mapping is not None:
            return bbox, mapping
    raise AssertionError("failed to locate a valid palette on the grid edge")


def _exterior_mask(passable: np.ndarray) -> np.ndarray:
    h, w = passable.shape
    seen = np.zeros_like(passable, dtype=bool)
    q: deque[tuple[int, int]] = deque()
    for r in range(h):
        for c in (0, w - 1):
            if passable[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if passable[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and passable[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
    return seen


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each enclosed region with the texture selected by its seed color.

    The palette sits on one grid edge. Each palette entry contains a label color
    and a square texture. Every non-palette closed region contains one seed
    pixel; that seed color selects the palette entry, and the seed coordinate is
    the texture tiling anchor.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _background_color(g)
    main_color = _edge_main_color(g, bg)
    palette_bbox, mapping = _find_palette(g, main_color, bg)
    r0, c0, r1, c1 = palette_bbox

    palette_mask = np.zeros_like(g, dtype=bool)
    palette_mask[r0 : r1 + 1, c0 : c1 + 1] = True

    blocked = (g == main_color) | palette_mask
    passable = ~blocked
    outside = _exterior_mask(passable)
    enclosed = passable & ~outside

    out = g.copy()
    for comp in _components(enclosed):
        seeds = [(r, c) for r, c in comp if int(g[r, c]) != bg]
        assert len(seeds) == 1, "each enclosed region must contain exactly one seed"
        sr, sc = seeds[0]
        seed_color = int(g[sr, sc])
        assert seed_color in mapping, "seed color must appear in the palette"
        texture = mapping[seed_color]
        th, tw = texture.shape
        for r, c in comp:
            if int(g[r, c]) != bg:
                continue
            out[r, c] = int(texture[(r - sr) % th, (c - sc) % tw])
    return out


def _make_texture(tex_size: int, palette: list[int]) -> np.ndarray:
    n_colors = random.randint(2, min(4, len(palette)))
    colors = random.sample(palette, n_colors)
    tex = np.array([[random.choice(colors) for _ in range(tex_size)] for _ in range(tex_size)], dtype=int)
    if len(np.unique(tex)) < 2:
        tex[0, 0] = colors[0]
        tex[-1, -1] = colors[1]
    return tex


def _build_palette(main_color: int, labels: list[int], textures: dict[int, np.ndarray], tex_size: int) -> np.ndarray:
    n_rows = len(labels)
    height = n_rows * (tex_size + 1) + 1
    width = tex_size + 4
    pal = np.full((height, width), BG, dtype=int)
    pal[:, 0] = main_color
    pal[:, 2] = main_color
    pal[:, width - 1] = main_color
    for r in range(0, height, tex_size + 1):
        pal[r, :] = main_color

    for idx, label in enumerate(labels):
        r0 = 1 + idx * (tex_size + 1)
        label_row = r0 + random.randrange(tex_size)
        pal[label_row, 1] = label
        pal[r0 : r0 + tex_size, 3 : 3 + tex_size] = textures[label]
    return pal


def _grow_blob(h: int, w: int, target: int) -> set[tuple[int, int]] | None:
    for _ in range(GROW_RETRIES):
        seed = (random.randrange(h), random.randrange(w))
        cells = {seed}
        frontier = set()
        for nr, nc in _neighbors4(*seed):
            if 0 <= nr < h and 0 <= nc < w:
                frontier.add((nr, nc))
        while len(cells) < target and frontier:
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            cells.add(cell)
            for nr, nc in _neighbors4(*cell):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        if len(cells) == target:
            return cells
    return None


def _make_object(main_color: int, seed_color: int) -> np.ndarray | None:
    ih = random.randint(*OBJECT_INNER_H_RANGE)
    iw = random.randint(*OBJECT_INNER_W_RANGE)
    area_max = min(OBJECT_AREA_RANGE[1], ih * iw)
    area_min = min(area_max, OBJECT_AREA_RANGE[0])
    if area_max < 2:
        return None
    target = random.randint(max(2, area_min), area_max)
    blob = _grow_blob(ih, iw, target)
    if blob is None:
        return None

    inside = np.zeros((ih + 2, iw + 2), dtype=bool)
    for r, c in blob:
        inside[r + 1, c + 1] = True

    border = np.zeros_like(inside)
    for r, c in np.argwhere(inside):
        rr, cc = int(r), int(c)
        for nr, nc in _neighbors4(rr, cc):
            if not inside[nr, nc]:
                border[nr, nc] = True

    if not np.any(border):
        return None

    local = np.full_like(inside, BG, dtype=int)
    local[border] = main_color
    seed_r, seed_c = random.choice(tuple(blob))
    local[seed_r + 1, seed_c + 1] = seed_color
    return local


def _reserve(mask: np.ndarray, r0: int, c0: int, h: int, w: int, margin: int = 1) -> None:
    rr0 = max(0, r0 - margin)
    cc0 = max(0, c0 - margin)
    rr1 = min(mask.shape[0], r0 + h + margin)
    cc1 = min(mask.shape[1], c0 + w + margin)
    mask[rr0:rr1, cc0:cc1] = True


def _place_patch(grid: np.ndarray, occupied: np.ndarray, patch: np.ndarray) -> bool:
    h, w = grid.shape
    ph, pw = patch.shape
    if ph > h - 2 or pw > w - 2:
        return False
    for _ in range(PLACE_RETRIES):
        r0 = random.randint(1, h - ph - 1)
        c0 = random.randint(1, w - pw - 1)
        if occupied[r0 : r0 + ph, c0 : c0 + pw].any():
            continue
        grid[r0 : r0 + ph, c0 : c0 + pw] = np.where(patch != BG, patch, grid[r0 : r0 + ph, c0 : c0 + pw])
        _reserve(occupied, r0, c0, ph, pw)
        return True
    return False


def _place_palette(grid: np.ndarray, occupied: np.ndarray, palette: np.ndarray, edge: str) -> None:
    h, w = grid.shape
    ph, pw = palette.shape
    if edge == "top":
        r0 = 0
        c0 = random.randint(1, w - pw - 1) if w - pw >= 2 else 0
    elif edge == "bottom":
        r0 = h - ph
        c0 = random.randint(1, w - pw - 1) if w - pw >= 2 else 0
    elif edge == "left":
        r0 = random.randint(1, h - ph - 1) if h - ph >= 2 else 0
        c0 = 0
    else:
        r0 = random.randint(1, h - ph - 1) if h - ph >= 2 else 0
        c0 = w - pw
    grid[r0 : r0 + ph, c0 : c0 + pw] = palette
    _reserve(occupied, r0, c0, ph, pw)


def _gen_attempt() -> np.ndarray | None:
    tex_size = random.choice(TEXTURE_SIZES)
    n_labels = random.randint(*PALETTE_ENTRY_RANGE)
    edge = random.choice(["top", "bottom", "left", "right"])

    canonical_h = n_labels * (tex_size + 1) + 1
    canonical_w = tex_size + 4
    if edge in ("top", "bottom"):
        palette_h, palette_w = canonical_h, canonical_w
    else:
        rot = random.choice([1, 3])
        palette_h, palette_w = np.rot90(np.zeros((canonical_h, canonical_w), dtype=int), rot).shape

    min_h = max(GRID_H_RANGE[0], palette_h + (8 if edge in ("top", "bottom") else 2))
    min_w = max(GRID_W_RANGE[0], palette_w + (8 if edge in ("left", "right") else 2))
    if min_h > GRID_H_RANGE[1] or min_w > GRID_W_RANGE[1]:
        return None
    h = random.randint(min_h, GRID_H_RANGE[1])
    w = random.randint(min_w, GRID_W_RANGE[1])

    main_color = random.randint(1, 9)
    usable = [c for c in range(1, 10) if c != main_color]
    if len(usable) < n_labels + 2:
        return None
    random.shuffle(usable)
    labels = usable[:n_labels]
    texture_pool = usable[n_labels:]
    if len(texture_pool) < 2:
        return None

    textures = {label: _make_texture(tex_size, texture_pool) for label in labels}
    palette = _build_palette(main_color, labels, textures, tex_size)
    if edge in ("left", "right"):
        palette = np.rot90(palette, random.choice([1, 3]))

    grid = np.full((h, w), BG, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    _place_palette(grid, occupied, palette, edge)

    max_objects = min(OBJECT_COUNT_RANGE[1], n_labels)
    n_objects = random.randint(OBJECT_COUNT_RANGE[0], max_objects)
    seeds = [random.choice(labels) for _ in range(n_objects)]
    if n_objects > 1 and len(labels) > 1 and len(set(seeds)) == 1:
        seeds[random.randrange(n_objects)] = random.choice([c for c in labels if c != seeds[0]])

    for seed_color in seeds:
        placed = False
        for _ in range(PLACE_RETRIES):
            obj = _make_object(main_color, seed_color)
            if obj is None:
                continue
            if _place_patch(grid, occupied, obj):
                placed = True
                break
        if not placed:
            return None

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
