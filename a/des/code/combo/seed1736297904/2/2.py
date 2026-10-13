import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_RETRIES = 300
REGION_RETRIES = 120


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    return [
        (r + dr, c + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if dr or dc
    ]


def _components(mask: np.ndarray, conn8: bool = False) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    neigh = _neighbors8 if conn8 else _neighbors4
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in neigh(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in points]
    cs = [c for _, c in points]
    return min(rs), min(cs), max(rs), max(cs)


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _boundary_from_shape(shape: np.ndarray) -> np.ndarray:
    h, w = shape.shape
    border = np.zeros_like(shape, dtype=bool)
    for r in range(h):
        for c in range(w):
            if not shape[r, c]:
                continue
            for nr, nc in _neighbors4(r, c):
                if not (0 <= nr < h and 0 <= nc < w) or not shape[nr, nc]:
                    border[r, c] = True
                    break
    return border


def _enclosed_nonwall(sub_wall: np.ndarray) -> np.ndarray:
    h, w = sub_wall.shape
    outside = np.zeros_like(sub_wall, dtype=bool)
    q: deque[tuple[int, int]] = deque()
    for r in range(h):
        for c in range(w):
            if r not in (0, h - 1) and c not in (0, w - 1):
                continue
            if not sub_wall[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and not sub_wall[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))
    return (~sub_wall) & (~outside)


def _detect_palette(grid: np.ndarray) -> list[int]:
    h, w = grid.shape
    candidates: list[tuple[int, int, list[int]]] = []
    for c in range(w):
        col = grid[:, c]
        if np.any(col == GREY):
            continue
        rows = np.flatnonzero((col != BG) & (col != GREY))
        if rows.size < 2:
            continue
        if rows[-1] - rows[0] + 1 != rows.size:
            continue
        if np.any(col[: rows[0]] != BG) or np.any(col[rows[-1] + 1 :] != BG):
            continue
        colors = [int(v) for v in col[rows[0] : rows[-1] + 1]]
        if len(colors) != len(set(colors)):
            continue
        candidates.append((c, rows[0], colors))
    assert candidates, "palette column not found"
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][2]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the black holes inside each grey-bordered region with the palette color
    indexed by that region's hole count.
    """
    g = np.asarray(grid, dtype=int)
    palette = _detect_palette(g)
    out = g.copy()

    wall_mask = g == GREY
    for comp in _components(wall_mask, conn8=True):
        r0, c0, r1, c1 = _bbox(comp)
        r0 = max(0, r0 - 1)
        c0 = max(0, c0 - 1)
        r1 = min(g.shape[0] - 1, r1 + 1)
        c1 = min(g.shape[1] - 1, c1 + 1)

        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        sub_wall = np.zeros_like(sub, dtype=bool)
        for gr, gc in comp:
            if r0 <= gr <= r1 and c0 <= gc <= c1:
                sub_wall[gr - r0, gc - c0] = True

        inside = _enclosed_nonwall(sub_wall)
        hole_mask = inside & (sub == BG)
        hole_count = len(_components(hole_mask, conn8=False))
        if hole_count == 0:
            continue
        assert hole_count <= len(palette), "hole count exceeds palette length"
        out[r0 : r1 + 1, c0 : c1 + 1][hole_mask] = palette[hole_count - 1]

    return out


def _connected(mask: np.ndarray) -> bool:
    return bool(mask.any()) and len(_components(mask, conn8=False)) == 1


def _carve_notch(shape: np.ndarray, rng: random.Random) -> None:
    h, w = shape.shape
    side = rng.choice(["top", "bottom", "left", "right"])
    if side in ("top", "bottom"):
        depth = rng.randint(1, max(1, h // 3))
        width = rng.randint(1, max(1, w - 2))
        start = rng.randint(0, w - width)
        if side == "top":
            shape[:depth, start : start + width] = False
        else:
            shape[h - depth :, start : start + width] = False
    else:
        depth = rng.randint(1, max(1, w // 3))
        height = rng.randint(1, max(1, h - 2))
        start = rng.randint(0, h - height)
        if side == "left":
            shape[start : start + height, :depth] = False
        else:
            shape[start : start + height, w - depth :] = False


def _sample_holes(interior: np.ndarray, hole_count: int, rng: random.Random) -> np.ndarray | None:
    h, w = interior.shape
    blocked = np.zeros_like(interior, dtype=bool)
    holes = np.zeros_like(interior, dtype=bool)
    cells = [(r, c) for r in range(h) for c in range(w) if interior[r, c]]

    for _ in range(hole_count):
        scored: list[tuple[int, int, int]] = []
        for r, c in cells:
            if blocked[r, c]:
                continue
            score = 0
            for nr, nc in _neighbors8(r, c):
                if 0 <= nr < h and 0 <= nc < w and interior[nr, nc] and not blocked[nr, nc]:
                    score += 1
            scored.append((score, r, c))
        if not scored:
            return None
        scored.sort(reverse=True)
        top = scored[: min(len(scored), 10)]
        _, ar, ac = rng.choice(top)
        comp = {(ar, ac)}

        roll = rng.random()
        target_size = 1
        if roll < 0.25:
            target_size = 2
        elif roll < 0.35:
            target_size = 3

        while len(comp) < target_size:
            frontier = []
            for cr, cc in comp:
                for nr, nc in _neighbors4(cr, cc):
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if interior[nr, nc] and not blocked[nr, nc] and (nr, nc) not in comp:
                        frontier.append((nr, nc))
            if not frontier:
                break
            comp.add(rng.choice(frontier))

        for r, c in comp:
            holes[r, c] = True
        for r, c in comp:
            for nr in range(r - 1, r + 2):
                for nc in range(c - 1, c + 2):
                    if 0 <= nr < h and 0 <= nc < w:
                        blocked[nr, nc] = True

    if len(_components(holes, conn8=False)) != hole_count:
        return None
    return holes


def _make_region_patch(hole_count: int, base_color: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(REGION_RETRIES):
        sh = rng.randint(max(7, hole_count + 4), hole_count + 8)
        sw = rng.randint(max(7, hole_count + 4), hole_count + 9)
        shape = np.ones((sh, sw), dtype=bool)

        for _ in range(rng.randint(1, 4)):
            _carve_notch(shape, rng)
        if rng.random() < 0.5:
            _carve_notch(shape, rng)

        if not shape.any() or not _connected(shape):
            continue
        shape = _trim_mask(shape)
        if shape.shape[0] < 5 or shape.shape[1] < 5:
            continue

        border = _boundary_from_shape(shape)
        interior = shape & (~border)
        if int(interior.sum()) < hole_count * 3 + 4:
            continue

        holes = _sample_holes(interior, hole_count, rng)
        if holes is None:
            continue

        patch = np.zeros(shape.shape, dtype=int)
        patch[shape] = base_color
        patch[border] = GREY
        patch[holes] = BG
        if np.count_nonzero(patch == base_color) < hole_count + 3:
            continue
        return patch, holes
    return None


def _find_positions(
    occ: np.ndarray, patch_shape: tuple[int, int], min_col: int
) -> list[tuple[int, int]]:
    h, w = occ.shape
    ph, pw = patch_shape
    positions: list[tuple[int, int]] = []
    for r0 in range(1, h - ph):
        for c0 in range(max(1, min_col), w - pw):
            if occ[r0 - 1 : r0 + ph + 1, c0 - 1 : c0 + pw + 1].any():
                continue
            positions.append((r0, c0))
    return positions


def _paste_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    ph, pw = patch.shape
    sub = grid[top : top + ph, left : left + pw]
    mask = patch != BG
    sub[mask] = patch[mask]


def _choose_base_colors(
    palette: list[int], hole_counts: list[int], rng: random.Random
) -> list[int]:
    chosen: list[int] = []
    used: set[int] = set()
    for h in hole_counts:
        choices = [c for c in COLORS if c != palette[h - 1] and c not in used]
        if not choices:
            choices = [c for c in COLORS if c != palette[h - 1]]
        base = rng.choice(choices)
        chosen.append(base)
        used.add(base)
    return chosen


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    palette_len = rng.randint(2, 5)
    region_count = rng.randint(2, min(4, palette_len))
    hole_counts = rng.sample(range(1, palette_len + 1), region_count)
    palette = rng.sample(COLORS, palette_len)
    base_colors = _choose_base_colors(palette, hole_counts, rng)

    regions = []
    for h, base in zip(hole_counts, base_colors):
        made = _make_region_patch(h, base, rng)
        if made is None:
            return None
        patch, holes = made
        fill_color = palette[h - 1]
        regions.append(
            {
                "hole_count": h,
                "patch": patch,
                "holes": holes,
                "fill_color": fill_color,
                "area": int(np.count_nonzero(patch == GREY) + np.count_nonzero(patch == base)),
            }
        )

    regions.sort(key=lambda item: item["area"], reverse=True)
    heights = sorted((item["patch"].shape[0] for item in regions), reverse=True)
    widths = sorted((item["patch"].shape[1] for item in regions), reverse=True)

    palette_col = rng.choice([0, 1])
    min_h = max(18, sum(heights[:2]) + 4, heights[0] + 4, palette_len + 2)
    min_w = max(22, palette_col + 5 + sum(widths[:2]) + 2, palette_col + 8 + widths[0])
    max_h = min(30, min_h + 5)
    max_w = min(30, min_w + 5)
    if min_h > max_h or min_w > max_w:
        return None

    grid = np.zeros((rng.randint(min_h, max_h), rng.randint(min_w, max_w)), dtype=int)
    occ = np.zeros_like(grid, dtype=bool)

    palette_row = rng.randint(0, grid.shape[0] - palette_len)
    grid[palette_row : palette_row + palette_len, palette_col] = np.array(palette, dtype=int)
    occ[
        max(0, palette_row - 1) : min(grid.shape[0], palette_row + palette_len + 1),
        max(0, palette_col - 1) : min(grid.shape[1], palette_col + 2),
    ] = True

    expected = grid.copy()
    min_region_col = palette_col + 3
    for item in regions:
        patch = item["patch"]
        holes = item["holes"]
        positions = _find_positions(occ, patch.shape, min_region_col)
        if not positions:
            return None
        top, left = rng.choice(positions)
        _paste_patch(grid, patch, top, left)
        _paste_patch(expected, patch, top, left)
        ph, pw = patch.shape
        occ[top - 1 : top + ph + 1, left - 1 : left + pw + 1] = True

        for hr, hc in np.argwhere(holes):
            expected[top + int(hr), left + int(hc)] = int(item["fill_color"])

    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted retries")
