import random

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9
PALETTE_COLORS = (1, 2, 3, 4, 6, 7, 8)
MAX_GEN_ATTEMPTS = 400

TEMPLATE_MASKS = [
    np.array([[1, 0, 0], [0, 1, 1], [1, 0, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 0], [1, 0, 1]], dtype=bool),
    np.array([[0, 1, 1], [1, 0, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 0], [0, 1, 1]], dtype=bool),
]


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.flatnonzero(np.any(mask, axis=1))
    cols = np.flatnonzero(np.any(mask, axis=0))
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _boundary(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros((h, w), dtype=bool)
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        for nr, nc in _neighbors4(rr, cc):
            if nr < 0 or nr >= h or nc < 0 or nc >= w or not mask[nr, nc]:
                out[rr, cc] = True
                break
    return out


def _interior(mask: np.ndarray) -> np.ndarray:
    return mask & (~_boundary(mask))


def _bbox_dims(mask: np.ndarray) -> tuple[int, int]:
    rows, cols = np.where(mask)
    return int(rows.max() - rows.min() + 1), int(cols.max() - cols.min() + 1)


def _legend_rows(grid: np.ndarray) -> tuple[int, int]:
    maroon_rows = np.flatnonzero(np.any(grid == MAROON, axis=1))
    assert maroon_rows.size > 0, "expected at least one maroon outline"
    candidates = [r for r in range(int(maroon_rows[0])) if np.any(grid[r] != BLACK)]
    assert candidates, "expected a non-empty legend above the outlines"
    r0 = int(min(candidates))
    r1 = int(max(candidates))
    assert r1 - r0 + 1 == 3, "legend band must be 3 rows tall"
    return r0, r1


def _parse_legend(grid: np.ndarray) -> tuple[dict[int, np.ndarray], int]:
    r0, r1 = _legend_rows(grid)
    band = grid[r0 : r1 + 1]
    legend: dict[int, np.ndarray] = {}
    c = 0
    while c < band.shape[1]:
        if not np.any(band[:, c] != BLACK):
            c += 1
            continue
        start = c
        while c < band.shape[1] and np.any(band[:, c] != BLACK):
            c += 1
        end = c
        assert end - start == 4, "each legend block must be 1 + 3 columns wide"
        key_vals = [int(v) for v in np.unique(band[:, start]) if int(v) != BLACK]
        assert len(key_vals) == 1, "legend key column must contain exactly one index color"
        key = key_vals[0]
        template = band[:, start + 1 : end].copy()
        assert template.shape == (3, 3)
        legend[key] = template
    assert legend, "legend must contain at least one entry"
    return legend, int(r1 + 1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the top legend to tile a 3x3 template inside each closed maroon outline.
    """
    g = np.asarray(grid, dtype=int)
    legend, lower_start = _parse_legend(g)
    sub = g[lower_start:, :]
    h, w = sub.shape
    out = g.copy()

    for comp in _components(sub != MAROON):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
            continue
        seeds = [(r, c, int(sub[r, c])) for r, c in comp if int(sub[r, c]) != BLACK]
        assert len(seeds) == 1, "each enclosed region must contain exactly one seed"
        seed_r, seed_c, color = seeds[0]
        assert color in legend, "seed color must appear in the legend"
        template = legend[color]
        for r, c in comp:
            tr = (r - seed_r + 1) % 3
            tc = (c - seed_c + 1) % 3
            out[lower_start + r, c] = int(template[tr, tc])

    return out


def _template_is_nontrivial(template: np.ndarray) -> bool:
    values = set(map(int, np.unique(template)))
    if values != {GREY, int(template[template != GREY][0])}:
        return False
    for dr in range(3):
        for dc in range(3):
            if dr == 0 and dc == 0:
                continue
            if np.array_equal(template, np.roll(template, shift=(dr, dc), axis=(0, 1))):
                return False
    return True


def _make_template(color: int) -> np.ndarray:
    for _ in range(200):
        mask = random.choice(TEMPLATE_MASKS).copy()
        mask = np.rot90(mask, random.randint(0, 3))
        if random.random() < 0.5:
            mask = np.fliplr(mask)
        if random.random() < 0.5:
            mask = np.flipud(mask)
        if random.random() < 0.5:
            mask = ~mask
        template = np.full((3, 3), GREY, dtype=int)
        template[mask] = color
        if _template_is_nontrivial(template):
            return template
    raise RuntimeError("failed to build a nontrivial template")


def _carve_notch(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    side = random.choice(("top", "bottom", "left", "right"))
    if side in {"top", "bottom"}:
        if h <= 2 or w <= 1:
            return out
        depth = random.randint(1, h - 2)
        span = random.randint(1, w - 1)
        c0 = random.randint(0, w - span)
        rows = slice(0, depth) if side == "top" else slice(h - depth, h)
        out[rows, c0 : c0 + span] = False
    else:
        if w <= 2 or h <= 1:
            return out
        depth = random.randint(1, w - 2)
        span = random.randint(1, h - 1)
        r0 = random.randint(0, h - span)
        cols = slice(0, depth) if side == "left" else slice(w - depth, w)
        out[r0 : r0 + span, cols] = False
    return out


def _shape_matches_role(interior: np.ndarray, role: str) -> bool:
    area = int(np.sum(interior))
    if area == 0:
        return False
    ih, iw = _bbox_dims(interior)
    if role == "small":
        return area <= 4 or min(ih, iw) <= 2
    if role == "large":
        return area >= 12 and min(ih, iw) >= 4
    return area >= 4


def _make_filled_shape(role: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dims = {
        "small": ((3, 5), (3, 6), (0, 1)),
        "medium": ((4, 7), (4, 8), (0, 2)),
        "large": ((6, 9), (6, 10), (1, 3)),
    }
    (h0, h1), (w0, w1), notch_range = dims[role]
    for _ in range(250):
        mask = np.ones((random.randint(h0, h1), random.randint(w0, w1)), dtype=bool)
        for _ in range(random.randint(notch_range[0], notch_range[1])):
            mask = _carve_notch(mask)
        if not np.any(mask):
            continue
        mask = _crop_mask(mask)
        if len(_components(mask)) != 1:
            continue
        boundary = _boundary(mask)
        interior = mask & (~boundary)
        if len(_components(boundary)) != 1:
            continue
        if len(_components(interior)) != 1:
            continue
        if not _shape_matches_role(interior, role):
            continue
        return mask, boundary, interior
    raise RuntimeError(f"failed to generate a valid {role} shape")


def _seeded_patch(role: str, color: int) -> np.ndarray:
    filled, boundary, interior = _make_filled_shape(role)
    patch = np.full(filled.shape, BLACK, dtype=int)
    patch[boundary] = MAROON
    cells = [tuple(map(int, rc)) for rc in np.argwhere(interior)]
    seed_r, seed_c = random.choice(cells)
    patch[seed_r, seed_c] = color
    return patch


def _legend_width(entry_count: int) -> int:
    return 4 * entry_count + (entry_count - 1)


def _draw_legend(grid: np.ndarray, templates: dict[int, np.ndarray]) -> None:
    keys = list(templates)
    width = _legend_width(len(keys))
    start_c = random.randint(1, grid.shape[1] - width - 1)
    c = start_c
    for i, key in enumerate(keys):
        grid[1, c] = key
        grid[:, c + 1 : c + 4] = templates[key]
        c += 4
        if i != len(keys) - 1:
            c += 1


def _choose_shape_colors(colors: list[int], shape_count: int) -> list[int]:
    chosen = [random.choice(colors) for _ in range(shape_count)]
    if shape_count >= 2 and random.random() < 0.7:
        dup_color = random.choice(colors)
        i, j = random.sample(range(shape_count), 2)
        chosen[i] = dup_color
        chosen[j] = dup_color
    return chosen


def _place_patches(grid: np.ndarray, lower_start: int, patches: list[np.ndarray]) -> bool:
    h, w = grid.shape
    blocked = np.zeros((h, w), dtype=bool)
    for patch in patches:
        ph, pw = patch.shape
        placed = False
        for _ in range(200):
            r0 = random.randint(lower_start + 1, h - ph - 2)
            c0 = random.randint(1, w - pw - 2)
            rr0 = r0 - 1
            rr1 = r0 + ph + 1
            cc0 = c0 - 1
            cc1 = c0 + pw + 1
            if np.any(blocked[rr0:rr1, cc0:cc1]):
                continue
            view = grid[r0 : r0 + ph, c0 : c0 + pw]
            if np.any(view != BLACK):
                continue
            view[patch != BLACK] = patch[patch != BLACK]
            blocked[rr0:rr1, cc0:cc1] = True
            placed = True
            break
        if not placed:
            return False
    return True


def _generate_attempt() -> np.ndarray | None:
    legend_count = random.randint(2, 4)
    legend_colors = random.sample(PALETTE_COLORS, legend_count)
    templates = {color: _make_template(color) for color in legend_colors}

    shape_count = random.randint(2, 5)
    roles = ["small", "large"] + [random.choice(("small", "medium", "large")) for _ in range(shape_count - 2)]
    random.shuffle(roles)
    shape_colors = _choose_shape_colors(legend_colors, shape_count)
    patches = [_seeded_patch(role, color) for role, color in zip(roles, shape_colors, strict=False)]
    random.shuffle(patches)

    legend_w = _legend_width(legend_count)
    min_w = max(legend_w + 2, max(p.shape[1] for p in patches) + 4, 24)
    min_h = max(3 + 2 + max(p.shape[0] for p in patches) + 3, 18)
    width = random.randint(min_w, max(min_w, 34))
    height = random.randint(min_h, max(min_h, 30))
    grid = np.full((height, width), BLACK, dtype=int)

    _draw_legend(grid[:3, :], templates)
    lower_start = 3 + random.randint(2, 4)
    if lower_start + max(p.shape[0] for p in patches) + 2 >= height:
        return None
    if not _place_patches(grid, lower_start, patches):
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid sample after {MAX_GEN_ATTEMPTS} attempts")
