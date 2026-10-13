import random
from collections import deque

import numpy as np


BG = 0
RED = 2
GREY = 5
MAROON = 9

TEMPLATE_COLOR_POOL = [1, 3, 4, 6, 7, 8]
DIRECTIONS = ("up", "right", "down", "left")
ROTATION_K = {"up": 0, "right": 3, "down": 2, "left": 1}

GEN_ATTEMPTS = 400
GEN_SIZE_CHOICES = [5, 6]
GEN_FENCE_COUNT_RANGE = (2, 4)
GEN_PADDING = 1


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _extract_templates(grid: np.ndarray) -> list[np.ndarray]:
    maroon_points = [tuple(map(int, p)) for p in np.argwhere(grid == MAROON)]
    assert maroon_points, "template library border not found"
    r0, c0, r1, c1 = _bbox(maroon_points)
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]

    full_rows = [r for r in range(patch.shape[0]) if np.all(patch[r, :] == MAROON)]
    full_cols = [c for c in range(patch.shape[1]) if np.all(patch[:, c] == MAROON)]
    assert full_rows == [0, patch.shape[0] - 1], "template library must have only top/bottom full maroon rows"
    assert len(full_cols) == 5, "template library must contain 4 horizontally separated templates"

    strides = np.diff(full_cols)
    assert len(set(map(int, strides))) == 1, "template separators must be evenly spaced"
    size = int(strides[0] - 1)
    assert size >= 2
    assert patch.shape == (size + 2, 4 * size + 5)

    templates: list[np.ndarray] = []
    for i in range(4):
        left = full_cols[i] + 1
        right = full_cols[i + 1]
        tmpl = patch[1:-1, left:right].copy()
        assert tmpl.shape == (size, size)
        templates.append(tmpl)
    return templates


def _classify_fence(grid: np.ndarray, comp: list[tuple[int, int]], size: int) -> tuple[int, int, str, int] | None:
    r0, c0, r1, c1 = _bbox(comp)
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    h, w = patch.shape
    if (h, w) != (size + 2, size + 2):
        return None

    border = np.zeros_like(patch, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    grey_mask = patch == GREY
    if np.any(grey_mask & ~border):
        return None

    missing = np.argwhere(border & ~grey_mask)
    if len(missing) != 1:
        return None
    mr, mc = map(int, missing[0])
    if mr == 0 and 0 < mc < w - 1:
        gap_dir = "up"
    elif mr == h - 1 and 0 < mc < w - 1:
        gap_dir = "down"
    elif mc == 0 and 0 < mr < h - 1:
        gap_dir = "left"
    elif mc == w - 1 and 0 < mr < h - 1:
        gap_dir = "right"
    else:
        return None

    interior = patch[1:-1, 1:-1]
    allowed = {BG, RED}
    if not set(map(int, np.unique(interior))).issubset(allowed):
        return None
    marker_count = int(np.sum(interior == RED))
    if not (1 <= marker_count <= 4):
        return None

    return r0, c0, gap_dir, marker_count


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count red markers inside each grey fenced rectangle, choose the matching template
    from the maroon-bordered 4-template library, rotate it toward the fence gap, and
    fill the fence interior.
    """
    g = np.asarray(grid, dtype=int)
    templates = _extract_templates(g)
    size = int(templates[0].shape[0])

    fences: list[tuple[int, int, str, int]] = []
    for comp in _components_of_color(g, GREY):
        item = _classify_fence(g, comp, size)
        if item is not None:
            fences.append(item)

    assert fences, "no valid fenced rectangles found"

    out = g.copy()
    for r0, c0, gap_dir, marker_count in fences:
        tmpl = templates[marker_count - 1]
        rotated = np.rot90(tmpl, ROTATION_K[gap_dir])
        out[r0 + 1 : r0 + 1 + size, c0 + 1 : c0 + 1 + size] = rotated
    return out


def _arrow_masks(size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows, cols = np.indices((size, size))
    mid = (size - 1) / 2.0
    head_h = max(2, size // 2)
    tip_extra = 0.5 if size % 2 == 0 else 0.0
    stem_half = 0.5 if size % 2 == 0 else 0.0
    foot_half = 1.5 if size % 2 == 0 else 1.0

    head = (rows < head_h) & (np.abs(cols - mid) <= rows + tip_extra)
    stem = (rows >= head_h) & (np.abs(cols - mid) <= stem_half)
    foot = (rows == size - 1) & (np.abs(cols - mid) <= foot_half)
    arrow = head | stem | foot
    spine = arrow & (np.abs(cols - mid) <= stem_half)
    return arrow, head, foot, spine


def _build_template(size: int, variant: int, colors: list[int]) -> np.ndarray:
    rows, cols = np.indices((size, size))
    mid = (size - 1) / 2.0
    arrow, head, foot, spine = _arrow_masks(size)
    tpl = np.full((size, size), BG, dtype=int)
    a, b, c, d = colors

    if variant == 0:
        tpl[head & (cols <= mid)] = a
        tpl[head & (cols > mid)] = b
        tpl[arrow & ~head] = c
        tpl[foot] = d
    elif variant == 1:
        tpl[arrow] = a
        tpl[spine] = b
        tpl[head & (rows == rows[head].max())] = c
        tpl[foot & (cols < mid)] = d
    elif variant == 2:
        tpl[head] = a
        tpl[arrow & ~head] = b
        tpl[head & ((rows + cols) % 2 == 0)] = c
        tpl[foot & (cols > mid)] = d
    elif variant == 3:
        tpl[arrow] = a
        tpl[head & (cols < mid)] = b
        tpl[spine & (rows >= 1)] = c
        badge_row = max(1, size // 2 - 1)
        tpl[(rows == badge_row) & head & (cols > mid)] = d
        tpl[foot & (np.abs(cols - mid) <= (0.5 if size % 2 == 0 else 0.0))] = c
    else:
        raise AssertionError("unexpected template variant")
    return tpl


def _make_templates(size: int, rng: random.Random) -> list[np.ndarray]:
    templates: list[np.ndarray] = []
    for variant in range(4):
        colors = rng.sample(TEMPLATE_COLOR_POOL, 4)
        tpl = _build_template(size, variant, colors)
        if np.array_equal(tpl, np.rot90(tpl, 1)):
            return []
        templates.append(tpl)

    for i in range(4):
        for j in range(i + 1, 4):
            if np.array_equal(templates[i], templates[j]):
                return []
    return templates


def _library_patch(templates: list[np.ndarray]) -> np.ndarray:
    size = int(templates[0].shape[0])
    patch = np.full((size + 2, 4 * size + 5), BG, dtype=int)
    patch[0, :] = MAROON
    patch[-1, :] = MAROON
    for k in range(5):
        c = k * (size + 1)
        patch[:, c] = MAROON
    for i, tmpl in enumerate(templates):
        c0 = 1 + i * (size + 1)
        patch[1:-1, c0 : c0 + size] = tmpl
    return patch


def _expand_rect(rect: tuple[int, int, int, int], pad: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = rect
    return r0 - pad, c0 - pad, r1 + pad, c1 + pad


def _intersects(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _place_corner_patch(
    grid: np.ndarray, patch: np.ndarray, corner: str, rng: random.Random
) -> tuple[int, int, tuple[int, int, int, int]]:
    h, w = grid.shape
    ph, pw = patch.shape
    margin = rng.choice([0, 1])
    if corner == "tl":
        r0, c0 = margin, margin
    elif corner == "tr":
        r0, c0 = margin, w - pw - margin
    elif corner == "bl":
        r0, c0 = h - ph - margin, margin
    elif corner == "br":
        r0, c0 = h - ph - margin, w - pw - margin
    else:
        raise AssertionError("unknown corner")
    grid[r0 : r0 + ph, c0 : c0 + pw] = patch
    return r0, c0, (r0, c0, r0 + ph - 1, c0 + pw - 1)


def _draw_fence(size: int, gap_dir: str, marker_count: int, rng: random.Random) -> np.ndarray:
    patch = np.full((size + 2, size + 2), BG, dtype=int)
    patch[0, :] = GREY
    patch[-1, :] = GREY
    patch[:, 0] = GREY
    patch[:, -1] = GREY

    gap_idx = rng.randint(1, size)
    if gap_dir == "up":
        patch[0, gap_idx] = BG
    elif gap_dir == "down":
        patch[-1, gap_idx] = BG
    elif gap_dir == "left":
        patch[gap_idx, 0] = BG
    elif gap_dir == "right":
        patch[gap_idx, -1] = BG
    else:
        raise AssertionError("unknown gap direction")

    cells = [(r, c) for r in range(1, size + 1) for c in range(1, size + 1)]
    for r, c in rng.sample(cells, marker_count):
        patch[r, c] = RED
    return patch


def _try_place_fences(
    grid: np.ndarray,
    size: int,
    fence_count: int,
    blocked: list[tuple[int, int, int, int]],
    rng: random.Random,
) -> bool:
    h, w = grid.shape
    counts = rng.sample([1, 2, 3, 4], fence_count)
    dirs = rng.sample(list(DIRECTIONS), fence_count)
    outer = size + 2

    for marker_count, gap_dir in zip(counts, dirs):
        placed = False
        for _ in range(200):
            r0 = rng.randint(1, h - outer - 1)
            c0 = rng.randint(1, w - outer - 1)
            rect = (r0, c0, r0 + outer - 1, c0 + outer - 1)
            pad_rect = _expand_rect(rect, GEN_PADDING)
            if any(_intersects(pad_rect, other) for other in blocked):
                continue
            grid[r0 : r0 + outer, c0 : c0 + outer] = _draw_fence(size, gap_dir, marker_count, rng)
            blocked.append(pad_rect)
            placed = True
            break
        if not placed:
            return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    size = rng.choice(GEN_SIZE_CHOICES)
    fence_count = rng.randint(*GEN_FENCE_COUNT_RANGE)
    templates = _make_templates(size, rng)
    if not templates:
        return None

    lib = _library_patch(templates)
    outer = size + 2
    h_min = max(20, lib.shape[0] + outer + 4, 2 * outer + 5)
    w_min = max(lib.shape[1] + outer + 4, 2 * outer + 5)
    h = rng.randint(h_min, h_min + 8)
    w = rng.randint(w_min, w_min + 10)
    grid = np.full((h, w), BG, dtype=int)

    corner = rng.choice(["tl", "tr", "bl", "br"])
    _, _, lib_rect = _place_corner_patch(grid, lib, corner, rng)
    blocked = [_expand_rect(lib_rect, GEN_PADDING)]

    if not _try_place_fences(grid, size, fence_count, blocked, rng):
        return None

    grey_comps = _components_of_color(grid, GREY)
    if len(grey_comps) != fence_count:
        return None
    if len(_extract_templates(grid)) != 4:
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
