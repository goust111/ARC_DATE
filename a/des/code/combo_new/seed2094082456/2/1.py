from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if seen[rr, cc]:
            continue
        seen[rr, cc] = True
        q = deque([(rr, cc)])
        pts: list[tuple[int, int]] = []
        while q:
            cr, cc2 = q.popleft()
            pts.append((int(cr), int(cc2)))
            for dr, dc in DIR4:
                nr = int(cr + dr)
                nc = int(cc2 + dc)
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _filled_box(points: list[tuple[int, int]], r0: int, c0: int, r1: int, c1: int) -> bool:
    return len(points) == (r1 - r0 + 1) * (c1 - c0 + 1)


def _enclosed_cells_from_component(
    shape: tuple[int, int], component: list[tuple[int, int]]
) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(component)
    sub_h = r1 - r0 + 1
    sub_w = c1 - c0 + 1
    wall = np.zeros((sub_h, sub_w), dtype=bool)
    for r, c in component:
        wall[int(r - r0), int(c - c0)] = True

    open_mask = ~wall
    seen = np.zeros_like(open_mask, dtype=bool)
    q: deque[tuple[int, int]] = deque()
    for rr in range(sub_h):
        for cc in (0, sub_w - 1):
            if open_mask[rr, cc] and not seen[rr, cc]:
                seen[rr, cc] = True
                q.append((rr, cc))
    for cc in range(sub_w):
        for rr in (0, sub_h - 1):
            if open_mask[rr, cc] and not seen[rr, cc]:
                seen[rr, cc] = True
                q.append((rr, cc))

    while q:
        rr, cc = q.popleft()
        for dr, dc in DIR4:
            nr = int(rr + dr)
            nc = int(cc + dc)
            if 0 <= nr < sub_h and 0 <= nc < sub_w and open_mask[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))

    enclosed = open_mask & ~seen
    coords = np.argwhere(enclosed)
    if coords.size == 0:
        return np.zeros((0, 2), dtype=int)
    coords = coords.astype(int, copy=False)
    coords[:, 0] += int(r0)
    coords[:, 1] += int(c0)
    return coords


def _legend_palette(grid: np.ndarray, bg: int) -> tuple[list[int], np.ndarray]:
    candidates: list[tuple[int, int, int]] = []
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        for comp in _components(grid == color):
            r0, c0, r1, c1 = _bbox(comp)
            if (r1 - r0 + 1, c1 - c0 + 1) != (2, 2):
                continue
            if not _filled_box(comp, r0, c0, r1, c1):
                continue
            candidates.append((int(r0), int(c0), int(color)))

    by_col: dict[int, dict[int, int]] = {}
    for r0, c0, color in candidates:
        by_col.setdefault(int(c0), {})[int(r0)] = int(color)

    best: list[tuple[int, int, int]] = []
    best_col = None
    for c0, rows in by_col.items():
        for start_r in rows:
            run: list[tuple[int, int, int]] = []
            cur = int(start_r)
            while cur in rows:
                run.append((int(cur), int(c0), int(rows[cur])))
                cur += 3
            if len(run) > len(best) or (
                len(run) == len(best) and best_col is not None and int(c0) < int(best_col)
            ):
                best = run
                best_col = int(c0)
            if len(run) > len(best) and best_col is None:
                best = run
                best_col = int(c0)

    assert len(best) >= 2, "could not identify left legend"
    best.sort()
    palette = [color for _r0, _c0, color in best]
    legend_mask = np.zeros_like(grid, dtype=bool)
    for r0, c0, _color in best:
        legend_mask[int(r0) : int(r0) + 2, int(c0) : int(c0) + 2] = True
    return palette, legend_mask


def _outline_regions(
    grid: np.ndarray, bg: int, palette: list[int], legend_mask: np.ndarray
) -> tuple[int, list[tuple[list[tuple[int, int]], np.ndarray]]]:
    palette_set = set(map(int, palette))
    best_color = None
    best_regions: list[tuple[list[tuple[int, int]], np.ndarray]] = []
    best_score = (-1, -1, -1)
    for color in map(int, np.unique(grid)):
        if color == bg or color in palette_set:
            continue
        mask = (grid == color) & ~legend_mask
        if not mask.any():
            continue
        regions: list[tuple[list[tuple[int, int]], np.ndarray]] = []
        enclosed_components = 0
        enclosed_area = 0
        for comp in _components(mask):
            enclosed = _enclosed_cells_from_component(grid.shape, comp)
            if enclosed.size == 0:
                continue
            regions.append((comp, enclosed))
            enclosed_components += 1
            enclosed_area += int(len(enclosed))
        score = (enclosed_components, enclosed_area, int(mask.sum()))
        if score > best_score:
            best_score = score
            best_color = int(color)
            best_regions = regions

    assert best_color is not None and best_score[0] > 0, "could not identify closed outlines"
    return int(best_color), best_regions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each closed outline by looking up the number of interior hole markers in the left legend.

    Input structure:
    - The background is the dominant color.
    - Along the left side sits a vertical legend made of 2x2 monochrome swatches, with one background row between consecutive swatches.
    - Elsewhere are several irregular closed outline components in one neutral color, each surrounding background interior cells that may contain scattered single-pixel hole markers in another color.

    Rule:
    - Read the legend from top to bottom so the first swatch corresponds to one hole marker, the second to two, and so on.
    - For every closed outline component, find the pixels enclosed by that outline and count how many interior marker pixels it contains.
    - Repaint the entire outline with the legend color indexed by that count, and erase the interior marker pixels back to background.
    - Keep the legend and all other background pixels unchanged.

    Key insight:
    - The interior marker count is a lookup-table index: each region must first be recognized as a closed contour, then its hidden count selects the replacement color from the ordered legend.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(Counter(map(int, g.flat)).most_common(1)[0][0])

    palette, legend_mask = _legend_palette(g, bg)
    _outline_color, regions = _outline_regions(g, bg, palette, legend_mask)

    hole_counts = Counter()
    for _comp, enclosed in regions:
        for r, c in enclosed:
            value = int(g[int(r), int(c)])
            if value != bg:
                hole_counts[value] += 1
    assert hole_counts, "could not identify hole marker color"
    hole_color = int(hole_counts.most_common(1)[0][0])

    out = g.copy()
    for comp, enclosed in regions:
        count = sum(int(g[int(r), int(c)] == hole_color) for r, c in enclosed)
        assert 1 <= count <= len(palette), "hole count falls outside legend"
        new_color = int(palette[count - 1])
        for r, c in comp:
            out[int(r), int(c)] = new_color
        for r, c in enclosed:
            if int(g[int(r), int(c)]) == hole_color:
                out[int(r), int(c)] = bg

    return out


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0 = int(coords[:, 0].min())
    r1 = int(coords[:, 0].max())
    c0 = int(coords[:, 1].min())
    c1 = int(coords[:, 1].max())
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _dilate8(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    padded = np.pad(mask, 1, constant_values=False)
    out = np.zeros((h, w), dtype=bool)
    for dr in range(3):
        for dc in range(3):
            out |= padded[dr : dr + h, dc : dc + w]
    return out


def _make_interior(required_area: int, rng: random.Random) -> np.ndarray | None:
    for _ in range(120):
        canvas = np.zeros((12, 12), dtype=bool)
        base_h = rng.randint(2, 5)
        base_w = rng.randint(2, 5)
        r0 = rng.randint(3, 5)
        c0 = rng.randint(3, 5)
        canvas[r0 : r0 + base_h, c0 : c0 + base_w] = True

        sides = ["top", "bottom", "left", "right"]
        rng.shuffle(sides)
        for side in sides[: rng.randint(0, 3)]:
            if side in ("top", "bottom"):
                bulge_h = rng.randint(1, 2)
                bulge_w = rng.randint(2, base_w)
                bulge_c0 = c0 + rng.randint(0, base_w - bulge_w)
                if side == "top":
                    bulge_r0 = r0 - bulge_h
                else:
                    bulge_r0 = r0 + base_h
                if 0 <= bulge_r0 and bulge_r0 + bulge_h <= canvas.shape[0]:
                    canvas[bulge_r0 : bulge_r0 + bulge_h, bulge_c0 : bulge_c0 + bulge_w] = True
            else:
                bulge_w = rng.randint(1, 2)
                bulge_h = rng.randint(2, base_h)
                bulge_r0 = r0 + rng.randint(0, base_h - bulge_h)
                if side == "left":
                    bulge_c0 = c0 - bulge_w
                else:
                    bulge_c0 = c0 + base_w
                if 0 <= bulge_c0 and bulge_c0 + bulge_w <= canvas.shape[1]:
                    canvas[bulge_r0 : bulge_r0 + bulge_h, bulge_c0 : bulge_c0 + bulge_w] = True

        interior = np.pad(_trim_mask(canvas), 1, constant_values=False)
        if int(interior.sum()) < required_area or int(interior.sum()) > 18:
            continue
        if min(interior.shape) < 4:
            continue

        outline = _dilate8(interior) & ~interior
        if len(_components(outline)) != 1:
            continue

        comp = _components(outline)[0]
        enclosed = _enclosed_cells_from_component(outline.shape, comp)
        enclosed_mask = np.zeros_like(interior, dtype=bool)
        if enclosed.size > 0:
            enclosed_mask[enclosed[:, 0], enclosed[:, 1]] = True
        if not np.array_equal(enclosed_mask, interior):
            continue
        return interior
    return None


def _make_region_patch(
    bg: int, outline_color: int, hole_color: int, hole_count: int, rng: random.Random
) -> tuple[np.ndarray, np.ndarray]:
    required_area = int(hole_count + rng.randint(1, 4))
    interior = _make_interior(required_area, rng)
    assert interior is not None
    outline = _dilate8(interior) & ~interior
    patch = np.full(interior.shape, bg, dtype=int)
    patch[outline] = int(outline_color)

    interior_cells = [tuple(map(int, cell)) for cell in np.argwhere(interior)]
    for r, c in rng.sample(interior_cells, int(hole_count)):
        patch[int(r), int(c)] = int(hole_color)
    return patch, outline


def _place_patches(
    patches: list[np.ndarray], h: int, w: int, scene_col0: int, rng: random.Random
) -> list[tuple[int, int]] | None:
    occupied = np.zeros((h, w), dtype=bool)
    placements: list[tuple[int, int]] = []
    for patch in patches:
        ph, pw = map(int, patch.shape)
        options: list[tuple[int, int]] = []
        if ph + 2 > h or pw + 1 > w:
            return None
        row_lo = 1
        row_hi = h - ph - 1
        col_lo = int(scene_col0)
        col_hi = w - pw - 1
        if row_lo > row_hi or col_lo > col_hi:
            return None
        for _ in range(250):
            r0 = rng.randint(row_lo, row_hi)
            c0 = rng.randint(col_lo, col_hi)
            rr0 = max(0, r0 - 1)
            rr1 = min(h, r0 + ph + 1)
            cc0 = max(0, c0 - 1)
            cc1 = min(w, c0 + pw + 1)
            if not occupied[rr0:rr1, cc0:cc1].any():
                options.append((int(r0), int(c0)))
        if not options:
            return None
        r0, c0 = rng.choice(options)
        placements.append((int(r0), int(c0)))
        rr0 = max(0, r0 - 1)
        rr1 = min(h, r0 + ph + 1)
        cc0 = max(0, c0 - 1)
        cc1 = min(w, c0 + pw + 1)
        occupied[rr0:rr1, cc0:cc1] = True
    return placements


def _expected_output(
    grid: np.ndarray,
    patches: list[np.ndarray],
    outlines: list[np.ndarray],
    placements: list[tuple[int, int]],
    counts: list[int],
    palette: list[int],
    bg: int,
    hole_color: int,
) -> np.ndarray:
    out = grid.copy()
    for patch, outline, (r0, c0), count in zip(patches, outlines, placements, counts):
        ph, pw = patch.shape
        sub = out[r0 : r0 + ph, c0 : c0 + pw]
        sub[outline] = int(palette[count - 1])
        sub[patch == hole_color] = int(bg)
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    palette_len = rng.randint(3, 5)
    region_count = rng.randint(2, min(4, palette_len))
    colors = list(range(10))
    rng.shuffle(colors)
    bg = int(colors[0])
    outline_color = int(colors[1])
    hole_color = int(colors[2])
    palette = [int(c) for c in colors[3 : 3 + palette_len]]
    counts = rng.sample(list(range(1, palette_len + 1)), region_count)

    patches: list[np.ndarray] = []
    outlines: list[np.ndarray] = []
    for count in counts:
        region = _make_region_patch(bg, outline_color, hole_color, int(count), rng)
        patches.append(region[0])
        outlines.append(region[1])

    legend_h = 3 * palette_len - 1
    max_patch_h = max(int(p.shape[0]) for p in patches)
    max_patch_w = max(int(p.shape[1]) for p in patches)

    min_h = max(legend_h + 2, max_patch_h + 2, 15)
    min_w = max(2 + 3 + max_patch_w + 2, 18)
    if min_h > 30 or min_w > 30:
        return None

    h = rng.randint(min_h, 30)
    w = rng.randint(min_w, 30)
    legend_left = rng.randint(0, 1)
    legend_top = rng.randint(0, h - legend_h)
    scene_col0 = legend_left + 2 + rng.randint(3, 5)

    placements = _place_patches(patches, h, w, scene_col0, rng)
    if placements is None:
        return None

    grid = np.full((h, w), bg, dtype=int)
    for idx, color in enumerate(palette):
        r0 = legend_top + 3 * idx
        grid[r0 : r0 + 2, legend_left : legend_left + 2] = int(color)

    for patch, (r0, c0) in zip(patches, placements):
        ph, pw = patch.shape
        sub = grid[r0 : r0 + ph, c0 : c0 + pw]
        mask = patch != bg
        if np.any(sub[mask] != bg):
            return None
        sub[mask] = patch[mask]

    expected = _expected_output(grid, patches, outlines, placements, counts, palette, bg, hole_color)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(300):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
