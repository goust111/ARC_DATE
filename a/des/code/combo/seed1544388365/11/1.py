from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BLACK = 0
GRAY = 5
DIR4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]
DIR8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]
PALETTE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _touches_border(coords: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in coords)


def _dilate4(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    outside = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()
    for r in range(h):
        for c in (0, w - 1):
            if not mask[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not mask[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((int(nr), int(nc)))
    return mask | (~mask & ~outside)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _detect_wall_color(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid[grid != BLACK], return_counts=True)
    return int(vals[np.argmax(cnts)])


def _find_legend(grid: np.ndarray, wall_color: int) -> tuple[dict[int, np.ndarray], np.ndarray]:
    h, w = grid.shape
    mask = (grid != BLACK) & (grid != wall_color)
    legend_mask = np.zeros((h, w), dtype=bool)
    mapping: dict[int, np.ndarray] = {}

    for comp in _components(mask):
        if len(comp) <= 1:
            continue
        r0, c0, r1, c1 = _bbox(comp)
        bh = r1 - r0 + 1
        bw = c1 - c0 + 1
        if bh not in (2, 3) or bw != bh + 1:
            continue
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        key_mask = (patch[:, 0] != BLACK) & (patch[:, 0] != wall_color)
        if int(key_mask.sum()) != 1:
            continue
        pattern = patch[:, 1:]
        if np.any((pattern == BLACK) | (pattern == wall_color)):
            continue
        key_row = int(np.flatnonzero(key_mask)[0])
        key_color = int(patch[key_row, 0])
        colors = {int(v) for v in np.unique(pattern)}
        if len(colors) != 2 or key_color not in colors:
            continue
        mapping[key_color] = pattern.copy()
        for r, c in comp:
            legend_mask[r, c] = True

    assert mapping, "legend not found"
    return mapping, legend_mask


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    wall_color = _detect_wall_color(g)
    mapping, legend_mask = _find_legend(g, wall_color)

    open_mask = (g != wall_color) & (~legend_mask)
    out = g.copy()

    for comp in _components(open_mask):
        if _touches_border(comp, h, w):
            continue
        seed_colors = {int(g[r, c]) for r, c in comp if int(g[r, c]) != BLACK}
        assert len(seed_colors) == 1, "each enclosed region must contain one seed color"
        key_color = next(iter(seed_colors))
        pattern = mapping[key_color]
        ph, pw = pattern.shape
        r0, c0, _, _ = _bbox(comp)
        for r, c in comp:
            out[r, c] = int(pattern[(r - r0) % ph, (c - c0) % pw])
    return out


def _make_pattern(size: int, key_color: int, other_color: int, rng: random.Random) -> np.ndarray:
    for _ in range(100):
        pattern = np.full((size, size), other_color, dtype=int)
        mode = rng.choice(["checker", "stripes_h", "stripes_v", "diag", "random"])
        if mode == "checker":
            phase = rng.randrange(2)
            for r in range(size):
                for c in range(size):
                    if (r + c + phase) % 2 == 0:
                        pattern[r, c] = key_color
        elif mode == "stripes_h":
            phase = rng.randrange(2)
            for r in range(size):
                if (r + phase) % 2 == 0:
                    pattern[r, :] = key_color
        elif mode == "stripes_v":
            phase = rng.randrange(2)
            for c in range(size):
                if (c + phase) % 2 == 0:
                    pattern[:, c] = key_color
        elif mode == "diag":
            if rng.random() < 0.5:
                for r in range(size):
                    pattern[r, r] = key_color
            else:
                for r in range(size):
                    pattern[r, size - 1 - r] = key_color
            if size == 3 and rng.random() < 0.5:
                pattern[1, :] = key_color
        else:
            for r in range(size):
                for c in range(size):
                    pattern[r, c] = key_color if rng.random() < 0.5 else other_color
        vals = {int(v) for v in np.unique(pattern)}
        if vals == {key_color, other_color}:
            return pattern
    raise RuntimeError("failed to build pattern")


def _random_shape(rng: random.Random) -> np.ndarray | None:
    for _ in range(200):
        h = rng.randint(3, 6)
        w = rng.randint(3, 6)
        min_area = max(6, (h * w * 3 + 4) // 5)
        max_area = h * w - 1
        if min_area > max_area:
            continue
        target = rng.randint(min_area, max_area)
        mask = np.zeros((h, w), dtype=bool)
        sr = rng.randrange(h)
        sc = rng.randrange(w)
        mask[sr, sc] = True
        while int(mask.sum()) < target:
            frontier: list[tuple[int, int]] = []
            seen = set()
            for r, c in zip(*np.where(mask), strict=False):
                for dr, dc in DIR4:
                    nr, nc = int(r + dr), int(c + dc)
                    if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc] and (nr, nc) not in seen:
                        seen.add((nr, nc))
                        frontier.append((nr, nc))
            if not frontier:
                break
            frontier.sort(key=lambda p: abs(p[0] - (h - 1) / 2.0) + abs(p[1] - (w - 1) / 2.0))
            pick_pool = frontier[: max(1, len(frontier) // 2)]
            nr, nc = rng.choice(pick_pool)
            mask[nr, nc] = True
        mask = _fill_holes(mask)
        mask = _crop_mask(mask)
        area = int(mask.sum())
        hh, ww = mask.shape
        if area < 6 or hh < 2 or ww < 2:
            continue
        if area == hh * ww and rng.random() < 0.85:
            continue
        if area * 10 < 6 * hh * ww:
            continue
        if len(_components(mask)) != 1:
            continue
        return mask
    return None


def _region_from_shape(shape: np.ndarray, key_color: int, rng: random.Random) -> dict:
    ih, iw = shape.shape
    interior = np.zeros((ih + 2, iw + 2), dtype=bool)
    interior[1 : ih + 1, 1 : iw + 1] = shape
    boundary = _dilate4(interior) & (~interior)
    footprint = interior | boundary
    patch = np.zeros((ih + 2, iw + 2), dtype=int)
    patch[boundary] = GRAY
    cells = list(zip(*np.where(interior), strict=False))
    seed_r, seed_c = rng.choice(cells)
    patch[int(seed_r), int(seed_c)] = key_color
    return {
        "patch": patch,
        "footprint": footprint,
        "interior": interior,
        "key_color": key_color,
    }


def _random_partition(total: int, parts: int, rng: random.Random) -> list[int]:
    cuts = sorted(rng.randrange(total + 1) for _ in range(parts - 1))
    values = []
    prev = 0
    for cut in cuts:
        values.append(cut - prev)
        prev = cut
    values.append(total - prev)
    return values


def _reserve_with_margin(forbidden: np.ndarray, footprint: np.ndarray, top: int, left: int) -> None:
    fh, fw = footprint.shape
    for rr in range(fh):
        for cc in range(fw):
            if not footprint[rr, cc]:
                continue
            gr = top + rr
            gc = left + cc
            for dr, dc in DIR8 + [(0, 0)]:
                nr, nc = gr + dr, gc + dc
                if 0 <= nr < forbidden.shape[0] and 0 <= nc < forbidden.shape[1]:
                    forbidden[nr, nc] = True


def _place_region(
    grid: np.ndarray,
    forbidden: np.ndarray,
    region: dict,
    max_col_exclusive: int,
    rng: random.Random,
) -> dict | None:
    patch = region["patch"]
    footprint = region["footprint"]
    interior = region["interior"]
    ph, pw = patch.shape
    h, w = grid.shape
    row_lo = 1
    row_hi = h - ph - 1
    col_lo = 1
    col_hi = max_col_exclusive - pw - 1
    if row_lo > row_hi or col_lo > col_hi:
        return None
    positions = [(r, c) for r in range(row_lo, row_hi + 1) for c in range(col_lo, col_hi + 1)]
    rng.shuffle(positions)
    for top, left in positions[:200]:
        blocked = forbidden[top : top + ph, left : left + pw]
        if bool((blocked & footprint).any()):
            continue
        grid[top : top + ph, left : left + pw][patch != BLACK] = patch[patch != BLACK]
        _reserve_with_margin(forbidden, footprint, top, left)
        coords = [(top + r, left + c) for r, c in zip(*np.where(interior), strict=False)]
        return {"coords": coords, "key_color": int(region["key_color"])}
    return None


def _build_legend(
    grid: np.ndarray,
    entries: list[dict],
    rng: random.Random,
) -> tuple[dict[int, np.ndarray], int]:
    h, w = grid.shape
    legend_strip_w = 4
    legend_x = w - legend_strip_w - 1
    min_height = 2 + sum(entry["size"] for entry in entries) + (len(entries) - 1)
    slack = h - min_height
    if slack < 0:
        raise RuntimeError("grid too short for legend")
    extras = _random_partition(slack, len(entries) + 1, rng)
    row = 1 + extras[0]
    mapping: dict[int, np.ndarray] = {}
    for idx, entry in enumerate(entries):
        size = int(entry["size"])
        key_color = int(entry["key_color"])
        pattern = entry["pattern"]
        key_row = row + size // 2
        grid[key_row, legend_x] = key_color
        grid[row : row + size, legend_x + 1 : legend_x + 1 + size] = pattern
        mapping[key_color] = pattern.copy()
        row += size
        if idx < len(entries) - 1:
            row += 1 + extras[idx + 1]
    return mapping, legend_x


def _paint_expected(grid: np.ndarray, regions: list[dict], mapping: dict[int, np.ndarray]) -> np.ndarray:
    out = grid.copy()
    for region in regions:
        coords = region["coords"]
        r0, c0, _, _ = _bbox(coords)
        pattern = mapping[region["key_color"]]
        ph, pw = pattern.shape
        for r, c in coords:
            out[r, c] = int(pattern[(r - r0) % ph, (c - c0) % pw])
    return out


def _generate_case(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    n_entries = rng.randint(2, 4)
    sizes = [rng.choice([2, 3]) for _ in range(n_entries)]
    min_height = 2 + sum(sizes) + (n_entries - 1)
    h = rng.randint(max(18, min_height), 28)
    w = rng.randint(22, 32)

    available = PALETTE_COLORS[:]
    rng.shuffle(available)
    key_colors = available[:n_entries]
    other_colors = available[n_entries : 2 * n_entries]
    if len(other_colors) < n_entries:
        return None

    entries = []
    for key_color, other_color, size in zip(key_colors, other_colors, sizes, strict=False):
        entries.append(
            {
                "key_color": key_color,
                "size": size,
                "pattern": _make_pattern(size, key_color, other_color, rng),
            }
        )

    grid = np.zeros((h, w), dtype=int)
    mapping, legend_x = _build_legend(grid, entries, rng)

    max_col_exclusive = legend_x - 2
    if max_col_exclusive < 10:
        return None

    n_regions = rng.randint(n_entries, min(5, n_entries + 2))
    region_keys = key_colors[:]
    while len(region_keys) < n_regions:
        region_keys.append(rng.choice(key_colors))
    rng.shuffle(region_keys)

    forbidden = np.zeros((h, w), dtype=bool)
    forbidden[:, max_col_exclusive:] = True
    regions: list[dict] = []
    for key_color in region_keys:
        shape = _random_shape(rng)
        if shape is None:
            return None
        region = _region_from_shape(shape, key_color, rng)
        placed = _place_region(grid, forbidden, region, max_col_exclusive, rng)
        if placed is None:
            return None
        regions.append(placed)

    non_black = grid[grid != BLACK]
    vals, cnts = np.unique(non_black, return_counts=True)
    color_counts = {int(v): int(c) for v, c in zip(vals, cnts, strict=False)}
    if color_counts.get(GRAY, 0) <= max(v for k, v in color_counts.items() if k != GRAY):
        return None

    expected = _paint_expected(grid, regions, mapping)
    if not np.array_equal(main(grid), expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(400):
        case = _generate_case(rng)
        if case is not None:
            return case[0]
    raise RuntimeError("failed to generate a valid sample")
