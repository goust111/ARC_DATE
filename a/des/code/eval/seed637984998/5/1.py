from __future__ import annotations

from collections import deque

import numpy as np


BG = 0
GREY = 5
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
GENERATION_RETRIES = 400


def _components(mask: np.ndarray) -> list[dict]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            touches_boundary = r in (0, h - 1) or c in (0, w - 1)
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
                        if nr in (0, h - 1) or nc in (0, w - 1):
                            touches_boundary = True
            arr = np.array(coords, dtype=int)
            r0 = int(arr[:, 0].min())
            c0 = int(arr[:, 1].min())
            r1 = int(arr[:, 0].max())
            c1 = int(arr[:, 1].max())
            comps.append(
                {
                    "coords": arr,
                    "area": int(len(coords)),
                    "bbox": (r0, c0, r1, c1),
                    "touches_boundary": touches_boundary,
                }
            )
    return comps


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    for comp in _components(~mask):
        if comp["touches_boundary"]:
            continue
        pts = comp["coords"]
        out[pts[:, 0], pts[:, 1]] = True
    return out


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    pts = np.argwhere(mask)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _ring_around(room: np.ndarray) -> np.ndarray:
    h, w = room.shape
    ring = np.zeros((h, w), dtype=bool)
    for r, c in np.argwhere(room):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = int(r + dr), int(c + dc)
                if 0 <= nr < h and 0 <= nc < w and not room[nr, nc]:
                    ring[nr, nc] = True
    return ring


def _neighbor_count(mask: np.ndarray, r: int, c: int) -> int:
    h, w = mask.shape
    total = 0
    for dr, dc in DIR4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
            total += 1
    return total


def _extract_legend(grid: np.ndarray) -> tuple[list[int], np.ndarray]:
    candidates = []
    for color in map(int, np.unique(grid)):
        if color in (BG, GREY):
            continue
        for comp in _components(grid == color):
            r0, c0, r1, c1 = comp["bbox"]
            height = int(r1 - r0 + 1)
            width = int(c1 - c0 + 1)
            if height >= 2 and height == width and comp["area"] == height * width:
                candidates.append(
                    {
                        "color": color,
                        "coords": comp["coords"],
                        "bbox": comp["bbox"],
                        "side": height,
                    }
                )

    assert len(candidates) >= 2, "expected a vertical legend of colored squares"
    assert len({item["side"] for item in candidates}) == 1, "legend blocks must share one size"
    assert len({item["bbox"][1] for item in candidates}) == 1, "legend blocks must align vertically"
    assert len({item["bbox"][3] for item in candidates}) == 1, "legend blocks must align vertically"

    candidates.sort(key=lambda item: item["bbox"][0])
    colors = [int(item["color"]) for item in candidates]
    assert len(colors) == len(set(colors)), "legend colors must be distinct"

    prev_bottom = -10**9
    for item in candidates:
        r0, _c0, r1, _c1 = item["bbox"]
        assert r0 > prev_bottom, "legend blocks must not overlap"
        prev_bottom = int(r1)

    mask = np.zeros_like(grid, dtype=bool)
    for item in candidates:
        pts = item["coords"]
        mask[pts[:, 0], pts[:, 1]] = True
    return colors, mask


def _grey_components(grid: np.ndarray) -> tuple[list[dict], np.ndarray]:
    comps = _components(grid == GREY)
    ids = np.full(grid.shape, -1, dtype=int)
    for idx, comp in enumerate(comps):
        pts = comp["coords"]
        ids[pts[:, 0], pts[:, 1]] = int(idx)
    return comps, ids


def _enclosed_regions(grid: np.ndarray, legend_mask: np.ndarray, grey_ids: np.ndarray) -> list[dict]:
    passable = (grid != GREY) & ~legend_mask
    regions = []
    seen_boundaries = set()
    h, w = grid.shape
    for comp in _components(passable):
        if comp["touches_boundary"]:
            continue
        pts = comp["coords"]
        dot_count = int(np.count_nonzero(grid[pts[:, 0], pts[:, 1]] != BG))
        assert dot_count >= 1, "each enclosed region must contain at least one colored dot"
        boundary_ids = set()
        for r, c in pts:
            for dr, dc in DIR4:
                nr, nc = int(r + dr), int(c + dc)
                if 0 <= nr < h and 0 <= nc < w and grey_ids[nr, nc] >= 0:
                    boundary_ids.add(int(grey_ids[nr, nc]))
        assert len(boundary_ids) == 1, "each enclosed region must belong to one grey boundary"
        boundary_id = next(iter(boundary_ids))
        assert boundary_id not in seen_boundaries, "grey boundaries must not enclose multiple regions"
        seen_boundaries.add(boundary_id)
        regions.append({"coords": pts, "dot_count": dot_count, "boundary_id": boundary_id})
    assert regions, "expected at least one enclosed region"
    return regions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each grey closed boundary by indexing a vertical legend with the number
    of colored dots inside its enclosed region, then erase the dots.
    """
    g = np.asarray(grid, dtype=int)
    legend_colors, legend_mask = _extract_legend(g)
    grey_comps, grey_ids = _grey_components(g)
    regions = _enclosed_regions(g, legend_mask, grey_ids)

    out = g.copy()
    for region in regions:
        dot_count = int(region["dot_count"])
        assert dot_count <= len(legend_colors), "dot count exceeds legend length"
        new_color = int(legend_colors[dot_count - 1])

        boundary = grey_comps[int(region["boundary_id"])]["coords"]
        out[boundary[:, 0], boundary[:, 1]] = new_color

        pts = region["coords"]
        out[pts[:, 0], pts[:, 1]] = BG

    return out


def _make_growth_room(inner_h: int, inner_w: int, target_area: int, rng: np.random.Generator) -> np.ndarray | None:
    mask = np.zeros((inner_h, inner_w), dtype=bool)
    start = (int(rng.integers(inner_h)), int(rng.integers(inner_w)))
    mask[start] = True
    frontier = set()
    frontier.update(
        (start[0] + dr, start[1] + dc)
        for dr, dc in DIR4
        if 0 <= start[0] + dr < inner_h and 0 <= start[1] + dc < inner_w
    )
    while frontier and int(mask.sum()) < target_area:
        cells = list(frontier)
        weights = np.array([1 + _neighbor_count(mask, r, c) ** 2 for r, c in cells], dtype=float)
        idx = int(rng.choice(len(cells), p=weights / weights.sum()))
        r, c = cells[idx]
        frontier.remove((r, c))
        mask[r, c] = True
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < inner_h and 0 <= nc < inner_w and not mask[nr, nc]:
                frontier.add((nr, nc))

    mask = _fill_holes(mask)
    if int(mask.sum()) < target_area:
        return None
    r0, c0, r1, c1 = _bbox(mask)
    bbox_area = (r1 - r0 + 1) * (c1 - c0 + 1)
    if mask.sum() * 2 < bbox_area:
        return None
    return mask


def _make_rect_union_room(inner_h: int, inner_w: int, target_area: int, rng: np.random.Generator) -> np.ndarray | None:
    mask = np.zeros((inner_h, inner_w), dtype=bool)
    max_rect_h = min(inner_h, 5)
    max_rect_w = min(inner_w, 5)

    for step in range(8):
        rh = int(rng.integers(2, max_rect_h + 1))
        rw = int(rng.integers(2, max_rect_w + 1))
        r0 = int(rng.integers(0, inner_h - rh + 1))
        c0 = int(rng.integers(0, inner_w - rw + 1))
        rect = np.zeros_like(mask)
        rect[r0 : r0 + rh, c0 : c0 + rw] = True
        if step == 0 or np.any(rect & mask):
            mask |= rect
        if int(mask.sum()) >= target_area:
            break

    mask = _fill_holes(mask)
    if int(mask.sum()) < target_area:
        return None
    r0, c0, r1, c1 = _bbox(mask)
    bbox_area = (r1 - r0 + 1) * (c1 - c0 + 1)
    if mask.sum() * 3 < bbox_area * 2:
        return None
    return mask


def _sample_dot_positions(room: np.ndarray, dot_count: int, rng: np.random.Generator) -> list[tuple[int, int]] | None:
    coords = [tuple(map(int, cell)) for cell in np.argwhere(room)]
    order = rng.permutation(len(coords))
    chosen = []
    for idx in order:
        r, c = coords[int(idx)]
        if all(max(abs(r - rr), abs(c - cc)) > 1 for rr, cc in chosen):
            chosen.append((r, c))
        if len(chosen) == dot_count:
            return chosen

    if len(coords) < dot_count:
        return None
    order = rng.permutation(len(coords))[:dot_count]
    return [coords[int(idx)] for idx in order]


def _make_region_patch(dot_count: int, mode: str, rng: np.random.Generator) -> np.ndarray | None:
    for _ in range(100):
        min_dim = 4 if dot_count <= 2 else 5
        max_dim = 7 if dot_count <= 4 else 8
        inner_h = int(rng.integers(min_dim, max_dim + 1))
        inner_w = int(rng.integers(min_dim, max_dim + 1))
        min_area = dot_count + int(rng.integers(4, 8))
        max_area = min(inner_h * inner_w, min_area + int(rng.integers(4, 9)))
        if min_area > max_area:
            continue
        target_area = int(rng.integers(min_area, max_area + 1))

        if mode == "growth":
            inner_room = _make_growth_room(inner_h, inner_w, target_area, rng)
        else:
            inner_room = _make_rect_union_room(inner_h, inner_w, target_area, rng)
        if inner_room is None:
            continue

        room = np.zeros((inner_h + 2, inner_w + 2), dtype=bool)
        room[1:-1, 1:-1] = inner_room
        boundary = _ring_around(room)

        if len(_components(room)) != 1 or len(_components(boundary)) != 1:
            continue

        dots = _sample_dot_positions(room, dot_count, rng)
        if dots is None:
            continue

        patch = np.full(room.shape, BG, dtype=int)
        patch[boundary] = GREY
        dot_colors = rng.choice(COLOR_POOL, size=dot_count, replace=False)
        for (r, c), color in zip(dots, dot_colors):
            patch[int(r), int(c)] = int(color)
        return patch
    return None


def _sample_dot_counts(region_count: int, legend_len: int, rng: np.random.Generator) -> list[int]:
    values = np.arange(1, legend_len + 1)
    if region_count <= legend_len and rng.random() < 0.5:
        counts = rng.choice(values, size=region_count, replace=False)
    else:
        counts = rng.choice(values, size=region_count, replace=True)
        if region_count > 1 and len(set(map(int, counts.tolist()))) == 1:
            idx = int(rng.integers(region_count))
            counts[idx] = int(values[(int(counts[idx]) % legend_len)])
    rng.shuffle(counts)
    return [int(x) for x in counts]


def _place_patch(grid: np.ndarray, blocked: np.ndarray, patch: np.ndarray, rng: np.random.Generator) -> bool:
    h, w = grid.shape
    ph, pw = patch.shape
    if ph + 2 >= h or pw + 2 >= w:
        return False
    for _ in range(250):
        r0 = int(rng.integers(1, h - ph))
        c0 = int(rng.integers(1, w - pw))
        r1, c1 = r0 + ph, c0 + pw
        rr0 = max(0, r0 - 1)
        cc0 = max(0, c0 - 1)
        rr1 = min(h, r1 + 1)
        cc1 = min(w, c1 + 1)
        if blocked[rr0:rr1, cc0:cc1].any():
            continue
        non_bg = patch != BG
        region = grid[r0:r1, c0:c1]
        if np.any(region[non_bg] != BG):
            continue
        region[non_bg] = patch[non_bg]
        blocked[rr0:rr1, cc0:cc1] = True
        return True
    return False


def _validate_grid(grid: np.ndarray, expected_regions: int) -> bool:
    try:
        legend_colors, legend_mask = _extract_legend(grid)
        grey_comps, grey_ids = _grey_components(grid)
        regions = _enclosed_regions(grid, legend_mask, grey_ids)
        if len(regions) != expected_regions:
            return False
        if len({region["boundary_id"] for region in regions}) != expected_regions:
            return False
        out = main(grid)
        if np.any(out == GREY):
            return False
        if not np.array_equal(out[legend_mask], grid[legend_mask]):
            return False
        for region in regions:
            pts = region["coords"]
            if np.any(out[pts[:, 0], pts[:, 1]] != BG):
                return False
            boundary = grey_comps[int(region["boundary_id"])]["coords"]
            expected_color = legend_colors[int(region["dot_count"]) - 1]
            if not np.all(out[boundary[:, 0], boundary[:, 1]] == expected_color):
                return False
        return True
    except Exception:
        return False


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    legend_len = int(rng.integers(3, 6))
    block_size = int(rng.integers(2, 4))
    gap = int(rng.integers(1, 3))
    legend_height = legend_len * block_size + (legend_len - 1) * gap

    h = int(rng.integers(max(24, legend_height + 4), 31))
    w = int(rng.integers(24, 31))
    grid = np.full((h, w), BG, dtype=int)

    side = "left" if rng.random() < 0.5 else "right"
    vertical_anchor = "top" if rng.random() < 0.5 else "bottom"
    legend_col = 0 if side == "left" else w - block_size
    legend_row0 = 0 if vertical_anchor == "top" else h - legend_height
    legend_colors = rng.choice(COLOR_POOL, size=legend_len, replace=False)
    for idx, color in enumerate(legend_colors):
        r0 = legend_row0 + idx * (block_size + gap)
        grid[r0 : r0 + block_size, legend_col : legend_col + block_size] = int(color)

    blocked = np.zeros((h, w), dtype=bool)
    blocked[0, :] = True
    blocked[-1, :] = True
    blocked[:, 0] = True
    blocked[:, -1] = True
    if side == "left":
        blocked[:, : block_size + 2] = True
    else:
        blocked[:, w - block_size - 2 :] = True

    region_count = int(rng.integers(2, 5))
    dot_counts = _sample_dot_counts(region_count, legend_len, rng)
    modes = ["growth", "rects"] * region_count
    rng.shuffle(modes)

    patches = []
    for dot_count, mode in zip(dot_counts, modes[:region_count]):
        patch = _make_region_patch(dot_count, mode, rng)
        if patch is None:
            return None
        patches.append((int((patch != BG).sum()), patch))

    patches.sort(key=lambda item: item[0], reverse=True)
    for _area, patch in patches:
        if not _place_patch(grid, blocked, patch, rng):
            return None

    if not _validate_grid(grid, region_count):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
