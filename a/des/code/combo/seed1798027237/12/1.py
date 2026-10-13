from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GREY = 5
GEN_ATTEMPTS = 300
GRID_SIDE_MIN = 18
GRID_SIDE_MAX = 36
LEGEND_LEN_MIN = 3
LEGEND_LEN_MAX = 5
REGION_COUNT_MIN = 2
REGION_COUNT_MAX = 4


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in DIR4:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(pts: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(pts[:, 0].min()),
        int(pts[:, 1].min()),
        int(pts[:, 0].max()),
        int(pts[:, 1].max()),
    )


def _most_common_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _legend_spec_for_grey(
    grid: np.ndarray, bg: int, grey: int
) -> tuple[list[int], np.ndarray] | None:
    h, w = grid.shape
    colors = [int(c) for c in np.unique(grid) if int(c) not in (bg, grey)]
    if not colors:
        return None

    entries = []
    for color in colors:
        comps = _components(grid == color)
        if len(comps) != 1:
            return None
        pts = comps[0]
        entries.append({"color": color, "bbox": _bbox(pts), "pts": pts})

    legend_mask = np.isin(grid, colors)
    all_pts = np.argwhere(legend_mask)
    r0, c0, r1, c1 = _bbox(all_pts)
    sides = []
    if r0 == 0:
        sides.append("top")
    if r1 == h - 1:
        sides.append("bottom")
    if c0 == 0:
        sides.append("left")
    if c1 == w - 1:
        sides.append("right")
    if len(sides) != 1:
        return None

    side = sides[0]
    if side in ("top", "bottom"):
        if (r1 - r0 + 1) > 4:
            return None
        ordered = sorted(entries, key=lambda e: (e["bbox"][1], e["bbox"][0]))
        prev_end = -1
        row_overlap_lo = max(int(e["bbox"][0]) for e in ordered)
        row_overlap_hi = min(int(e["bbox"][2]) for e in ordered)
        if row_overlap_lo > row_overlap_hi:
            return None
        for entry in ordered:
            if int(entry["bbox"][1]) <= prev_end:
                return None
            prev_end = int(entry["bbox"][3])
    else:
        if (c1 - c0 + 1) > 4:
            return None
        ordered = sorted(entries, key=lambda e: (e["bbox"][0], e["bbox"][1]))
        prev_end = -1
        col_overlap_lo = max(int(e["bbox"][1]) for e in ordered)
        col_overlap_hi = min(int(e["bbox"][3]) for e in ordered)
        if col_overlap_lo > col_overlap_hi:
            return None
        for entry in ordered:
            if int(entry["bbox"][0]) <= prev_end:
                return None
            prev_end = int(entry["bbox"][2])

    palette = [int(entry["color"]) for entry in ordered]
    return palette, legend_mask


def _detect_grey_and_legend(grid: np.ndarray, bg: int) -> tuple[int, list[int], np.ndarray]:
    colors = [int(c) for c in np.unique(grid) if int(c) != bg]
    preferred = []
    if GREY in colors:
        preferred.append(GREY)
    preferred.extend([c for c in colors if c != GREY])

    for grey in preferred:
        spec = _legend_spec_for_grey(grid, bg, grey)
        if spec is not None:
            palette, legend_mask = spec
            return grey, palette, legend_mask

    raise AssertionError("could not identify grey boundary color and edge legend")


def _inside_of_barrier(barrier: np.ndarray) -> np.ndarray:
    h, w = barrier.shape
    seen = np.zeros((h, w), dtype=bool)
    q = deque()

    for r in range(h):
        for c in (0, w - 1):
            if not barrier[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not barrier[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c, h, w):
            if not barrier[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))

    return (~seen) & (~barrier)


def _grey_containers(grey_mask: np.ndarray) -> list[dict]:
    h, w = grey_mask.shape
    containers: list[dict] = []
    for pts in _components(grey_mask):
        r0, c0, r1, c1 = _bbox(pts)
        er0 = max(0, r0 - 1)
        ec0 = max(0, c0 - 1)
        er1 = min(h - 1, r1 + 1)
        ec1 = min(w - 1, c1 + 1)
        barrier = np.zeros((er1 - er0 + 1, ec1 - ec0 + 1), dtype=bool)
        boundary_set: set[tuple[int, int]] = set()
        for r, c in pts:
            rr = int(r - er0)
            cc = int(c - ec0)
            barrier[rr, cc] = True
            boundary_set.add((int(r), int(c)))
        inside_crop = _inside_of_barrier(barrier)
        if not inside_crop.any():
            continue
        inside_pts = {
            (int(er0 + r), int(ec0 + c)) for r, c in np.argwhere(inside_crop)
        }
        containers.append(
            {
                "boundary": boundary_set,
                "inside": inside_pts,
                "inside_area": int(inside_crop.sum()),
            }
        )
    return containers


def _top_level_regions(work: np.ndarray, grey: int) -> list[dict]:
    containers = _grey_containers(work == grey)
    parents: list[int | None] = [None] * len(containers)

    for i, container in enumerate(containers):
        rep = next(iter(container["boundary"]))
        parent = None
        parent_area = None
        for j, other in enumerate(containers):
            if i == j:
                continue
            if rep not in other["inside"]:
                continue
            area = int(other["inside_area"])
            if parent is None or area < parent_area:
                parent = j
                parent_area = area
        parents[i] = parent

    bg_components = []
    h, w = work.shape
    for pts in _components(work == _most_common_color(work)):
        rep = tuple(map(int, pts[0]))
        touches_border = bool(
            np.any(pts[:, 0] == 0)
            or np.any(pts[:, 0] == h - 1)
            or np.any(pts[:, 1] == 0)
            or np.any(pts[:, 1] == w - 1)
        )
        bg_components.append({"rep": rep, "touches_border": touches_border})

    regions = []
    for i, container in enumerate(containers):
        if parents[i] is not None:
            continue
        inside = container["inside"]
        enclosed_bg = sum(
            1
            for comp in bg_components
            if not comp["touches_border"] and comp["rep"] in inside
        )
        regions.append(
            {
                "inside": inside,
                "hole_count": enclosed_bg - 1,
            }
        )
    return regions


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    grey, palette, legend_mask = _detect_grey_and_legend(g, bg)

    work = g.copy()
    work[legend_mask] = bg

    out = g.copy()
    for region in _top_level_regions(work, grey):
        holes = int(region["hole_count"])
        assert 0 <= holes < len(palette), "hole count exceeds legend range"
        fill_color = int(palette[holes])
        for r, c in region["inside"]:
            if not legend_mask[r, c] and int(g[r, c]) != grey:
                out[r, c] = fill_color
    return out


def _draw_rect_frame(arr: np.ndarray, top: int, left: int, height: int, width: int, color: int) -> None:
    arr[top, left : left + width] = color
    arr[top + height - 1, left : left + width] = color
    arr[top : top + height, left] = color
    arr[top : top + height, left + width - 1] = color


def _make_region(k: int, bg: int, rng: random.Random) -> np.ndarray:
    if k == 0:
        inner_h = rng.randint(3, 8)
        inner_w = rng.randint(4, 10)
        region = np.full((inner_h + 2, inner_w + 2), bg, dtype=int)
        _draw_rect_frame(region, 0, 0, inner_h + 2, inner_w + 2, GREY)
        return region

    slots = list(range(k))
    ncols = rng.randint(1, min(3, k))
    nrows = (k + ncols - 1) // ncols
    hole_sizes = [rng.choice([(3, 3), (3, 4), (4, 3), (4, 4)]) for _ in slots]
    row_heights = [0] * nrows
    col_widths = [0] * ncols
    for idx, (hh, hw) in enumerate(hole_sizes):
        rr = idx // ncols
        cc = idx % ncols
        row_heights[rr] = max(row_heights[rr], hh)
        col_widths[cc] = max(col_widths[cc], hw)

    pad_top = rng.randint(1, 2)
    pad_bottom = rng.randint(1, 2)
    pad_left = rng.randint(1, 3)
    pad_right = rng.randint(1, 3)
    gap_y = rng.randint(1, 2)
    gap_x = rng.randint(1, 2)

    inner_h = pad_top + pad_bottom + sum(row_heights) + gap_y * (nrows - 1)
    inner_w = pad_left + pad_right + sum(col_widths) + gap_x * (ncols - 1)
    region = np.full((inner_h + 2, inner_w + 2), bg, dtype=int)
    _draw_rect_frame(region, 0, 0, inner_h + 2, inner_w + 2, GREY)

    base_y = 1 + pad_top
    for rr in range(nrows):
        base_x = 1 + pad_left
        for cc in range(ncols):
            idx = rr * ncols + cc
            if idx >= k:
                break
            hole_h, hole_w = hole_sizes[idx]
            slot_h = row_heights[rr]
            slot_w = col_widths[cc]
            top = base_y + rng.randint(0, slot_h - hole_h)
            left = base_x + rng.randint(0, slot_w - hole_w)
            _draw_rect_frame(region, top, left, hole_h, hole_w, GREY)
            base_x += slot_w + gap_x
        base_y += row_heights[rr] + gap_y

    return region


def _place_legend(
    grid: np.ndarray, occ: np.ndarray, palette: list[int], side: str, rng: random.Random
) -> None:
    h, w = grid.shape
    if side in ("top", "bottom"):
        block_h = rng.randint(1, 2)
        block_w = rng.randint(2, 3)
        legend_h = block_h
        legend_w = block_w * len(palette)
        if w - legend_w < 3:
            raise ValueError("legend too wide")
        col0 = rng.randint(1, w - legend_w - 2)
        row0 = 0 if side == "top" else h - legend_h
        for idx, color in enumerate(palette):
            c0 = col0 + idx * block_w
            grid[row0 : row0 + block_h, c0 : c0 + block_w] = int(color)
    else:
        block_h = rng.randint(2, 3)
        block_w = rng.randint(1, 2)
        legend_h = block_h * len(palette)
        legend_w = block_w
        if h - legend_h < 3:
            raise ValueError("legend too tall")
        row0 = rng.randint(1, h - legend_h - 2)
        col0 = 0 if side == "left" else w - legend_w
        for idx, color in enumerate(palette):
            r0 = row0 + idx * block_h
            grid[r0 : r0 + block_h, col0 : col0 + block_w] = int(color)

    occ[
        max(0, row0 - 1) : min(h, row0 + legend_h + 1),
        max(0, col0 - 1) : min(w, col0 + legend_w + 1),
    ] = True


def _can_place(occ: np.ndarray, top: int, left: int, height: int, width: int) -> bool:
    h, w = occ.shape
    if top < 1 or left < 1 or top + height > h - 1 or left + width > w - 1:
        return False
    pad = occ[
        max(0, top - 1) : min(h, top + height + 1),
        max(0, left - 1) : min(w, left + width + 1),
    ]
    return not pad.any()


def _place_region(grid: np.ndarray, occ: np.ndarray, region: np.ndarray, rng: random.Random) -> bool:
    h, w = grid.shape
    rh, rw = region.shape
    candidates = [
        (r, c)
        for r in range(1, h - rh)
        for c in range(1, w - rw)
        if _can_place(occ, r, c, rh, rw)
    ]
    if not candidates:
        return False
    top, left = rng.choice(candidates)
    grey_mask = region == GREY
    sub = grid[top : top + rh, left : left + rw]
    sub[grey_mask] = GREY
    occ[
        max(0, top - 1) : min(h, top + rh + 1),
        max(0, left - 1) : min(w, left + rw + 1),
    ] = True
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    bg = rng.choice([c for c in range(10) if c != GREY])
    legend_len = rng.randint(LEGEND_LEN_MIN, LEGEND_LEN_MAX)
    palette = rng.sample([c for c in range(10) if c not in (bg, GREY)], legend_len)
    side = rng.choice(["top", "bottom", "left", "right"])

    region_count = rng.randint(REGION_COUNT_MIN, REGION_COUNT_MAX)
    hole_counts = [0, rng.randint(1, legend_len - 1)]
    while len(hole_counts) < region_count:
        hole_counts.append(rng.randint(0, legend_len - 1))
    rng.shuffle(hole_counts)
    if len(set(hole_counts)) < 2:
        return None

    regions = [_make_region(k, bg, rng) for k in hole_counts]
    max_region_h = max(region.shape[0] for region in regions)
    max_region_w = max(region.shape[1] for region in regions)
    total_area = sum(int(region.shape[0] * region.shape[1]) for region in regions)

    min_h = max(GRID_SIDE_MIN, max_region_h + 4)
    min_w = max(GRID_SIDE_MIN, max_region_w + 4)
    if side in ("left", "right"):
        min_h = max(min_h, legend_len * 3 + 4)
    else:
        min_w = max(min_w, legend_len * 3 + 4)

    max_h = min(GRID_SIDE_MAX, min_h + max(4, int(np.sqrt(total_area))))
    max_w = min(GRID_SIDE_MAX, min_w + max(4, int(np.sqrt(total_area))))
    if min_h > max_h or min_w > max_w:
        return None
    h = rng.randint(min_h, max_h)
    w = rng.randint(min_w, max_w)
    grid = np.full((h, w), bg, dtype=int)
    occ = np.zeros((h, w), dtype=bool)

    try:
        _place_legend(grid, occ, palette, side, rng)
    except ValueError:
        return None

    order = list(range(region_count))
    rng.shuffle(order)
    for idx in order:
        if not _place_region(grid, occ, regions[idx], rng):
            return None

    if _most_common_color(grid) != bg:
        return None

    try:
        grey, found_palette, legend_mask = _detect_grey_and_legend(grid, bg)
    except AssertionError:
        return None
    if grey != GREY or found_palette != palette:
        return None

    work = grid.copy()
    work[legend_mask] = bg
    regions_found = _top_level_regions(work, grey)
    found_counts = sorted(int(region["hole_count"]) for region in regions_found)
    if found_counts != sorted(hole_counts):
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
    raise RuntimeError("failed to generate valid input")
