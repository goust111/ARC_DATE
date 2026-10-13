from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
MARKER = 4

GEN_ATTEMPTS = 400
SHAPE_ATTEMPTS = 300
PLACE_ATTEMPTS = 300

GRID_H_RANGE = (30, 36)
GRID_W_RANGE = (30, 36)
COUNTS = (1, 2, 3, 4)
CORNERS = ("tl", "tr", "bl", "br")
NON_MARKER_COLORS = [1, 2, 3, 5, 6, 7, 8, 9]


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _count_components(mask: np.ndarray) -> int:
    if not np.any(mask):
        return 0
    return len(_components(mask))


def _tight_crop(mask: np.ndarray) -> np.ndarray:
    rows, cols = np.where(mask)
    if rows.size == 0:
        return mask.copy()
    return mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]


def _dilate4(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, constant_values=False)
    return (
        padded[1:-1, 1:-1]
        | padded[:-2, 1:-1]
        | padded[2:, 1:-1]
        | padded[1:-1, :-2]
        | padded[1:-1, 2:]
    )


def _erode4(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, constant_values=False)
    return (
        padded[1:-1, 1:-1]
        & padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
    )


def _boundary4(mask: np.ndarray) -> np.ndarray:
    return mask & ~_erode4(mask)


def _dilate_n(mask: np.ndarray, steps: int) -> np.ndarray:
    out = mask.copy()
    for _ in range(steps):
        out = _dilate4(out)
    return out


def _erode_n(mask: np.ndarray, steps: int) -> np.ndarray:
    out = mask.copy()
    for _ in range(steps):
        out = _erode4(out)
        if not np.any(out):
            break
    return out


def _flood_from_border(passable: np.ndarray) -> np.ndarray:
    h, w = passable.shape
    out = np.zeros_like(passable, dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if passable[r, c] and not out[r, c]:
                out[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if passable[r, c] and not out[r, c]:
                out[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and passable[nr, nc] and not out[nr, nc]:
                out[nr, nc] = True
                q.append((nr, nc))
    return out


def _has_hole(mask: np.ndarray) -> bool:
    if not np.any(mask):
        return False
    inv = ~mask
    outside = _flood_from_border(inv)
    return bool(np.any(inv & ~outside))


def _reconstruct_region(loop_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    passable = ~loop_mask
    outside = _flood_from_border(passable)
    interior = passable & ~outside
    return loop_mask | interior, interior


def _detect_palette(grid: np.ndarray, bg: int) -> tuple[dict[int, int], np.ndarray]:
    h, w = grid.shape

    def scan(top: bool, left: bool) -> tuple[dict[int, int], np.ndarray, int] | None:
        if w < 2:
            return None
        rows: list[int] = []
        row_iter = range(h) if top else range(h - 1, -1, -1)
        for r in row_iter:
            if left:
                ok = grid[r, 0] == MARKER and grid[r, 1] not in (bg, MARKER)
                if w > 2:
                    ok = ok and grid[r, 2] == bg
            else:
                ok = grid[r, w - 2] == MARKER and grid[r, w - 1] not in (bg, MARKER)
                if w > 2:
                    ok = ok and grid[r, w - 3] == bg
            if ok:
                rows.append(r)
            else:
                break
        if not rows:
            return None

        rows = sorted(rows)
        mask = np.zeros_like(grid, dtype=bool)
        mapping: dict[int, int] = {}
        ycol = 0 if left else w - 2
        ccol = 1 if left else w - 1
        for idx, r in enumerate(rows, start=1):
            mapping[idx] = int(grid[r, ccol])
            mask[r, ycol] = True
            mask[r, ccol] = True
        return mapping, mask, len(rows)

    candidates = []
    for top in (True, False):
        for left in (True, False):
            candidate = scan(top, left)
            if candidate is not None:
                candidates.append(candidate)

    if not candidates:
        raise AssertionError("palette not found")

    mapping, mask, _size = max(candidates, key=lambda item: item[2])
    return mapping, mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Shrink each colored loop inward by its interior marker count and repaint it
    using the palette row indexed by that same count.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    palette_map, palette_mask = _detect_palette(g, bg)

    out = np.full_like(g, bg)
    out[palette_mask] = g[palette_mask]

    object_mask = (g != bg) & (g != MARKER) & ~palette_mask
    components = _components(object_mask)
    for coords in components:
        color = int(g[coords[0, 0], coords[0, 1]])
        if not np.all(g[coords[:, 0], coords[:, 1]] == color):
            continue

        r0 = max(int(coords[:, 0].min()) - 1, 0)
        r1 = min(int(coords[:, 0].max()) + 1, g.shape[0] - 1)
        c0 = max(int(coords[:, 1].min()) - 1, 0)
        c1 = min(int(coords[:, 1].max()) + 1, g.shape[1] - 1)

        loop_mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        loop_mask[coords[:, 0] - r0, coords[:, 1] - c0] = True

        region, interior = _reconstruct_region(loop_mask)
        marker_count = int(np.count_nonzero(g[r0 : r1 + 1, c0 : c1 + 1][interior] == MARKER))
        if marker_count <= 0 or marker_count not in palette_map:
            continue

        shrunk = _erode_n(region, marker_count)
        if not np.any(shrunk):
            continue
        new_loop = _boundary4(shrunk)
        if not np.any(new_loop):
            continue

        color_out = palette_map[marker_count]
        patch = out[r0 : r1 + 1, c0 : c1 + 1]
        patch[new_loop] = color_out

    return out


def _random_polyomino(
    rng: random.Random,
    min_h: int,
    max_h: int,
    min_w: int,
    max_w: int,
    min_area: int,
    max_area: int,
) -> np.ndarray:
    h = rng.randint(min_h, max_h)
    w = rng.randint(min_w, max_w)
    target = rng.randint(min_area, min(max_area, h * w))
    start = (rng.randrange(h), rng.randrange(w))
    cells = {start}
    frontier = set()

    def add_frontier(r: int, c: int) -> None:
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                frontier.add((nr, nc))

    add_frontier(*start)
    while len(cells) < target and frontier:
        nr, nc = rng.choice(tuple(frontier))
        frontier.remove((nr, nc))
        cells.add((nr, nc))
        add_frontier(nr, nc)

    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return _tight_crop(mask)


def _random_core(rng: random.Random, thick: bool) -> np.ndarray | None:
    for _ in range(SHAPE_ATTEMPTS):
        mask = _random_polyomino(
            rng,
            min_h=3,
            max_h=6,
            min_w=3,
            max_w=6,
            min_area=4,
            max_area=10,
        )
        if _has_hole(mask):
            continue
        if min(mask.shape) < 2:
            continue
        boundary = _boundary4(mask)
        if _count_components(boundary) != 1:
            continue
        has_inner = bool(np.any(_erode4(mask)))
        if thick and not has_inner:
            continue
        if not thick and has_inner:
            continue
        return mask
    return None


def _choose_markers(rng: random.Random, interior: np.ndarray, n: int) -> list[tuple[int, int]]:
    pts = [(int(r), int(c)) for r, c in np.argwhere(interior)]
    rng.shuffle(pts)
    chosen: list[tuple[int, int]] = []
    used = set()
    for r, c in pts:
        if all(abs(r - rr) + abs(c - cc) >= 2 for rr, cc in chosen):
            chosen.append((r, c))
            used.add((r, c))
            if len(chosen) == n:
                return chosen
    for r, c in pts:
        if (r, c) in used:
            continue
        chosen.append((r, c))
        if len(chosen) == n:
            return chosen
    return chosen


def _carved_region(rng: random.Random, marker_count: int, vanish: bool) -> np.ndarray | None:
    for _ in range(SHAPE_ATTEMPTS):
        if vanish:
            if marker_count < 2:
                return None
            h = rng.randint(max(4, marker_count + 2), 2 * marker_count + 2)
            w = rng.randint(max(5, marker_count + 4), 2 * marker_count + 5)
            if rng.random() < 0.5:
                h = rng.randint(max(4, marker_count + 2), 2 * marker_count)
            else:
                w = rng.randint(max(4, marker_count + 2), 2 * marker_count)
        else:
            h = rng.randint(2 * marker_count + 4, 2 * marker_count + 8)
            w = rng.randint(2 * marker_count + 4, 2 * marker_count + 8)

        mask = np.ones((h, w), dtype=bool)
        for _ in range(rng.randint(1, 4)):
            side = rng.choice(("top", "bottom", "left", "right"))
            if side in ("top", "bottom"):
                if h <= 4 or w <= 4:
                    continue
                span = rng.randint(2, w - 2)
                start = rng.randint(1, w - span - 1)
                depth_cap = h - 2
                if depth_cap <= 0:
                    continue
                if vanish:
                    depth_max = min(depth_cap, max(1, marker_count))
                else:
                    depth_max = min(depth_cap, max(1, h // 2 - 1))
                depth = rng.randint(1, depth_max)
                rows = slice(0, depth) if side == "top" else slice(h - depth, h)
                mask[rows, start : start + span] = False
            else:
                if h <= 4 or w <= 4:
                    continue
                span = rng.randint(2, h - 2)
                start = rng.randint(1, h - span - 1)
                depth_cap = w - 2
                if depth_cap <= 0:
                    continue
                if vanish:
                    depth_max = min(depth_cap, max(1, marker_count))
                else:
                    depth_max = min(depth_cap, max(1, w // 2 - 1))
                depth = rng.randint(1, depth_max)
                cols = slice(0, depth) if side == "left" else slice(w - depth, w)
                mask[start : start + span, cols] = False

        mask = _tight_crop(mask)
        if not np.any(mask):
            continue
        if _count_components(mask) != 1:
            continue
        if _has_hole(mask):
            continue
        if _count_components(_boundary4(mask)) != 1:
            continue
        return mask
    return None


def _generate_region(rng: random.Random, marker_count: int, vanish: bool) -> tuple[np.ndarray, list[tuple[int, int]]] | None:
    for _ in range(SHAPE_ATTEMPTS):
        region = _carved_region(rng, marker_count, vanish)
        if region is None:
            continue

        boundary = _boundary4(region)
        interior = region & ~boundary
        if int(np.count_nonzero(interior)) < marker_count:
            continue

        shrunk = _erode_n(region, marker_count)
        if vanish:
            if marker_count < 2 or np.any(shrunk):
                continue
        else:
            if not np.any(shrunk):
                continue
            if not np.any(_erode4(shrunk)):
                continue
            if _count_components(_boundary4(shrunk)) != 1:
                continue

        markers = _choose_markers(rng, interior, marker_count)
        if len(markers) != marker_count:
            continue
        return region, markers
    return None


def _palette_bbox(corner: str, h: int, w: int, rows: int) -> tuple[int, int, int, int]:
    r0 = 0 if corner in ("tl", "tr") else h - rows
    c0 = 0 if corner in ("tl", "bl") else w - 2
    return r0, c0, r0 + rows - 1, c0 + 1


def _mark_blocked(blocked: np.ndarray, top: int, left: int, height: int, width: int, pad: int = 2) -> None:
    h, w = blocked.shape
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(h, top + height + pad)
    c1 = min(w, left + width + pad)
    blocked[r0:r1, c0:c1] = True


def _place_loop(
    rng: random.Random,
    grid: np.ndarray,
    blocked: np.ndarray,
    region: np.ndarray,
    markers: list[tuple[int, int]],
    loop_color: int,
) -> bool:
    h, w = grid.shape
    rh, rw = region.shape
    boundary = _boundary4(region)
    for _ in range(PLACE_ATTEMPTS):
        top = rng.randint(1, h - rh - 1)
        left = rng.randint(1, w - rw - 1)
        if np.any(blocked[top : top + rh, left : left + rw]):
            continue
        if np.any(grid[top : top + rh, left : left + rw] != BG):
            continue

        patch = grid[top : top + rh, left : left + rw]
        patch[boundary] = loop_color
        for mr, mc in markers:
            patch[mr, mc] = MARKER
        _mark_blocked(blocked, top, left, rh, rw, pad=2)
        return True
    return False


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    grid = np.full((h, w), BG, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    corner = rng.choice(CORNERS)
    palette_rows = len(COUNTS)
    pr0, pc0, pr1, pc1 = _palette_bbox(corner, h, w, palette_rows)
    _mark_blocked(blocked, pr0, pc0, pr1 - pr0 + 1, pc1 - pc0 + 1, pad=1)

    colors = NON_MARKER_COLORS.copy()
    rng.shuffle(colors)
    loop_colors = colors[: len(COUNTS)]
    palette_colors = colors[len(COUNTS) : 2 * len(COUNTS)]

    vanish_count = rng.choice([None, 2, 3, 4])
    specs = []
    for idx, marker_count in enumerate(COUNTS):
        vanish = marker_count == vanish_count
        shape = _generate_region(rng, marker_count, vanish=vanish)
        if shape is None:
            return None
        region, markers = shape
        specs.append(
            {
                "count": marker_count,
                "region": region,
                "markers": markers,
                "loop_color": loop_colors[idx],
                "area": int(np.count_nonzero(region)),
            }
        )

    specs.sort(key=lambda item: item["area"], reverse=True)
    for spec in specs:
        ok = _place_loop(
            rng,
            grid,
            blocked,
            spec["region"],
            spec["markers"],
            int(spec["loop_color"]),
        )
        if not ok:
            return None

    for row_idx, palette_color in enumerate(palette_colors):
        rr = pr0 + row_idx
        grid[rr, pc0] = MARKER
        grid[rr, pc0 + 1] = palette_color

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    if np.count_nonzero(out != BG) <= 2 * palette_rows:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid shrinking-loop puzzle")
