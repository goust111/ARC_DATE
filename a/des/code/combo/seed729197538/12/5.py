from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
TEMPLATE_COLOR = 5
PIECE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

GRID_MIN = 24
GRID_MAX = 30
GEN_RETRIES = 400
PARTITION_RETRIES = 160
PLACEMENT_RETRIES = 320


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _shape_key(coords: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    r0, c0, _, _ = _bbox(coords)
    return tuple(sorted((r - r0, c - c0) for r, c in coords))


def _cropped_mask(coords: list[tuple[int, int]]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask


def _mode_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _extract_template_regions(
    grid: np.ndarray,
) -> tuple[tuple[int, int, int, int], list[dict[str, object]]]:
    gray_pts = np.argwhere(grid == TEMPLATE_COLOR)
    assert len(gray_pts) > 0, "template lines must exist"
    r0, c0 = gray_pts.min(axis=0)
    r1, c1 = gray_pts.max(axis=0)
    inside = grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] != TEMPLATE_COLOR
    regions = []
    seen_shapes: set[tuple[tuple[int, int], ...]] = set()
    for comp in _components(inside):
        abs_coords = [(r + int(r0), c + int(c0)) for r, c in comp]
        key = _shape_key(abs_coords)
        assert key not in seen_shapes, "template region shapes must be unique up to translation"
        seen_shapes.add(key)
        regions.append({"coords": abs_coords, "shape": key})
    assert regions, "template must contain at least one fillable region"
    return (int(r0), int(c0), int(r1), int(c1)), regions


def _extract_pieces(
    grid: np.ndarray, template_bbox: tuple[int, int, int, int], bg: int
) -> tuple[list[dict[str, object]], np.ndarray]:
    r0, c0, r1, c1 = template_bbox
    outside = np.ones(grid.shape, dtype=bool)
    outside[r0 : r1 + 1, c0 : c1 + 1] = False
    pieces: list[dict[str, object]] = []
    for color in np.unique(grid[outside]):
        color = int(color)
        if color in {bg, TEMPLATE_COLOR}:
            continue
        for comp in _components(outside & (grid == color)):
            pieces.append({"color": color, "shape": _shape_key(comp)})
    return pieces, outside


def _color_counts(grid: np.ndarray, outside: np.ndarray, bg: int) -> Counter:
    counts: Counter = Counter()
    for value in grid[outside].flat:
        color = int(value)
        if color not in {bg, TEMPLATE_COLOR}:
            counts[color] += 1
    return counts


def _color_adjacencies(grid: np.ndarray, outside: np.ndarray, bg: int) -> set[tuple[int, int]]:
    h, w = grid.shape
    adj: set[tuple[int, int]] = set()
    for r in range(h):
        for c in range(w):
            if not outside[r, c]:
                continue
            a = int(grid[r, c])
            if a in {bg, TEMPLATE_COLOR}:
                continue
            for dr, dc in ((1, 0), (0, 1)):
                nr, nc = r + dr, c + dc
                if not (0 <= nr < h and 0 <= nc < w and outside[nr, nc]):
                    continue
                b = int(grid[nr, nc])
                if b in {bg, TEMPLATE_COLOR} or a == b:
                    continue
                adj.add(tuple(sorted((a, b))))
    return adj


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each gray-bounded template region with the matching fragment color.

    If two same-shape fragments compete for one region, use their global color
    counts and whether those colors touch outside the template to decide which
    one wins.
    """
    g = np.asarray(grid, dtype=int)
    bg = _mode_color(g)

    template_bbox, regions = _extract_template_regions(g)
    pieces, outside = _extract_pieces(g, template_bbox, bg)

    shape_to_region: dict[tuple[tuple[int, int], ...], int] = {}
    for idx, region in enumerate(regions):
        shape_to_region[region["shape"]] = idx

    region_colors: dict[int, list[int]] = {idx: [] for idx in range(len(regions))}
    for piece in pieces:
        shape = piece["shape"]
        assert shape in shape_to_region, "every outside fragment must match one template region"
        region_colors[shape_to_region[shape]].append(int(piece["color"]))

    counts = _color_counts(g, outside, bg)
    adj = _color_adjacencies(g, outside, bg)

    out = g.copy()
    out[outside & (g != bg) & (g != TEMPLATE_COLOR)] = bg

    for idx, region in enumerate(regions):
        colors = region_colors[idx]
        assert 1 <= len(colors) <= 2, "each region must have one or two matching fragments"
        if len(colors) == 1:
            fill = int(colors[0])
        else:
            a, b = map(int, colors)
            assert a != b, "competing fragments must have different colors"
            assert counts[a] != counts[b], "competing colors must have different global counts"
            touching = tuple(sorted((a, b))) in adj
            if touching:
                fill = a if counts[a] < counts[b] else b
            else:
                fill = a if counts[a] > counts[b] else b
        for r, c in region["coords"]:
            out[r, c] = fill
    return out


def _is_rectangle(coords: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    return len(coords) == (r1 - r0 + 1) * (c1 - c0 + 1)


def _weighted_choice_index(weights: list[float], rng) -> int:
    total = sum(weights)
    pick = rng.random() * total
    acc = 0.0
    for idx, weight in enumerate(weights):
        acc += weight
        if pick <= acc:
            return idx
    return len(weights) - 1


def _random_partition(rng, h: int, w: int, n_regions: int) -> np.ndarray | None:
    cells = [(r, c) for r in range(h) for c in range(w)]
    max_area = int(np.ceil(h * w * 0.45))
    for _ in range(PARTITION_RETRIES):
        seeds = rng.sample(cells, n_regions)
        part = -np.ones((h, w), dtype=int)
        regions = [set() for _ in range(n_regions)]
        for idx, cell in enumerate(seeds):
            part[cell] = idx
            regions[idx].add(cell)

        ok = True
        while np.any(part < 0):
            frontiers: list[list[tuple[int, int]]] = []
            weights: list[float] = []
            frontier_regions: list[int] = []
            for rid in range(n_regions):
                frontier = set()
                for r, c in regions[rid]:
                    for nr, nc in _neighbors4(r, c):
                        if 0 <= nr < h and 0 <= nc < w and part[nr, nc] < 0:
                            frontier.add((nr, nc))
                if frontier:
                    frontiers.append(list(frontier))
                    frontier_regions.append(rid)
                    weights.append((len(frontier) ** 1.2) / (len(regions[rid]) ** 0.7))
            if not weights:
                ok = False
                break
            choice = _weighted_choice_index(weights, rng)
            rid = frontier_regions[choice]
            cell = rng.choice(frontiers[choice])
            part[cell] = rid
            regions[rid].add(cell)

        if not ok:
            continue

        region_lists = [sorted((int(r), int(c)) for r, c in reg) for reg in regions]
        areas = [len(reg) for reg in region_lists]
        if min(areas) < 2 or max(areas) > max_area:
            continue
        if not any(not _is_rectangle(reg) for reg in region_lists):
            continue

        shapes = {_shape_key(reg) for reg in region_lists}
        if len(shapes) != n_regions:
            continue
        return part
    return None


def _render_template(part: np.ndarray, cell_size: int) -> np.ndarray:
    h, w = part.shape
    out = np.full((h * cell_size + h + 1, w * cell_size + w + 1), TEMPLATE_COLOR, dtype=int)
    step = cell_size + 1
    for r in range(h):
        for c in range(w):
            rr = 1 + r * step
            cc = 1 + c * step
            out[rr : rr + cell_size, cc : cc + cell_size] = BG
            if c + 1 < w and part[r, c] == part[r, c + 1]:
                out[rr : rr + cell_size, cc + cell_size] = BG
            if r + 1 < h and part[r, c] == part[r + 1, c]:
                out[rr + cell_size, cc : cc + cell_size] = BG
    return out


def _template_regions_from_patch(template: np.ndarray) -> list[dict[str, object]]:
    regions = []
    seen = set()
    for comp in _components(template == BG):
        key = _shape_key(comp)
        if key in seen:
            return []
        seen.add(key)
        regions.append(
            {
                "coords": comp,
                "shape": key,
                "mask": _cropped_mask(comp),
                "area": len(comp),
            }
        )
    return regions


def _single_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.full(mask.shape, BG, dtype=int)
    patch[mask] = color
    return patch


def _adjacent_pair_patch(mask: np.ndarray, color_a: int, color_b: int, rng) -> np.ndarray | None:
    base = {(int(r), int(c)) for r, c in np.argwhere(mask)}
    h, w = mask.shape
    candidates = []
    for dr in range(-h - 1, h + 2):
        for dc in range(-w - 1, w + 2):
            if dr == 0 and dc == 0:
                continue
            shifted = {(r + dr, c + dc) for r, c in base}
            if base & shifted:
                continue
            touching = False
            for r, c in base:
                if (
                    (r + 1, c) in shifted
                    or (r - 1, c) in shifted
                    or (r, c + 1) in shifted
                    or (r, c - 1) in shifted
                ):
                    touching = True
                    break
            if not touching:
                continue
            all_pts = list(base) + list(shifted)
            r0, c0, r1, c1 = _bbox(all_pts)
            area = (r1 - r0 + 1) * (c1 - c0 + 1)
            span = max(r1 - r0 + 1, c1 - c0 + 1)
            candidates.append((area, span, abs(dr) + abs(dc), dr, dc))
    if not candidates:
        return None

    candidates.sort()
    best_area = candidates[0][0]
    best = [cand for cand in candidates if cand[0] == best_area]
    _, _, _, dr, dc = rng.choice(best)
    shifted = {(r + dr, c + dc) for r, c in base}
    all_pts = list(base) + list(shifted)
    r0, c0, r1, c1 = _bbox(all_pts)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), BG, dtype=int)
    for r, c in base:
        patch[r - r0, c - c0] = color_a
    for r, c in shifted:
        patch[r - r0, c - c0] = color_b
    return patch


def _intervals_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return not (a1 < b0 or b1 < a0)


def _empty_gap(a0: int, a1: int, b0: int, b1: int) -> int:
    if a1 < b0:
        return b0 - a1 - 1
    if b1 < a0:
        return a0 - b1 - 1
    return -1


def _bboxes_conflict(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    row_overlap = _intervals_overlap(ar0, ar1, br0, br1)
    col_overlap = _intervals_overlap(ac0, ac1, bc0, bc1)
    if row_overlap and col_overlap:
        return True
    row_gap = _empty_gap(ar0, ar1, br0, br1)
    col_gap = _empty_gap(ac0, ac1, bc0, bc1)
    if row_gap == 0 and col_overlap:
        return True
    if col_gap == 0 and row_overlap:
        return True
    return False


def _place_items(
    rng,
    grid_shape: tuple[int, int],
    template_bbox: tuple[int, int, int, int],
    items: list[np.ndarray],
) -> list[tuple[tuple[int, int, int, int], np.ndarray]] | None:
    h, w = grid_shape
    ordered = sorted(items, key=lambda patch: int(np.count_nonzero(patch)), reverse=True)
    placed: list[tuple[tuple[int, int, int, int], np.ndarray]] = []
    for patch in ordered:
        ph, pw = patch.shape
        options = []
        for _ in range(PLACEMENT_RETRIES):
            r0 = rng.randint(0, h - ph)
            c0 = rng.randint(0, w - pw)
            bbox = (r0, c0, r0 + ph - 1, c0 + pw - 1)
            if _bboxes_conflict(bbox, template_bbox):
                continue
            if any(_bboxes_conflict(bbox, other_bbox) for other_bbox, _ in placed):
                continue
            options.append((bbox, patch))
            if len(options) >= 16:
                break
        if not options:
            return None
        placed.append(rng.choice(options))
    return placed


def _gen_attempt(rng) -> np.ndarray | None:
    cell_size = rng.choice([2, 3])
    if cell_size == 2:
        coarse_h = rng.randint(4, 5)
        coarse_w = rng.randint(4, 5)
    else:
        coarse_h = rng.randint(3, 4)
        coarse_w = rng.randint(3, 4)

    n_regions = rng.randint(3, min(5, coarse_h * coarse_w // 2))
    part = _random_partition(rng, coarse_h, coarse_w, n_regions)
    if part is None:
        return None

    template = _render_template(part, cell_size)
    regions = _template_regions_from_patch(template)
    if len(regions) != n_regions:
        return None

    contestable = [i for i, reg in enumerate(regions) if max(reg["mask"].shape) <= 6]
    if not contestable:
        return None

    max_dups = min(2, n_regions - 2, len(contestable))
    if max_dups <= 0:
        return None
    dup_count = 1 if max_dups == 1 else rng.choice([1, 2])

    contested = rng.sample(contestable, dup_count)
    remaining = [i for i in range(n_regions) if i not in contested]
    if len(remaining) < dup_count:
        return None
    anchors = rng.sample(remaining, dup_count)

    colors = PIECE_COLORS[:]
    rng.shuffle(colors)
    single_region_colors: dict[int, int] = {}
    result_colors: dict[int, int] = {}
    pair_infos = []

    if dup_count == 1:
        branch_flags = [rng.random() < 0.5]
    else:
        branch_flags = [True, False]
        rng.shuffle(branch_flags)

    for idx, region_id in enumerate(contested):
        color_a = colors.pop()
        color_b = colors.pop()
        major, minor = (color_a, color_b) if rng.random() < 0.5 else (color_b, color_a)
        anchor_id = anchors[idx]
        adjacent = branch_flags[idx]
        pair_infos.append(
            {
                "region": region_id,
                "anchor": anchor_id,
                "major": major,
                "minor": minor,
                "adjacent": adjacent,
            }
        )
        single_region_colors[anchor_id] = major
        result_colors[region_id] = minor if adjacent else major
        result_colors[anchor_id] = major

    for region_id in range(n_regions):
        if region_id in single_region_colors or region_id in contested:
            continue
        color = colors.pop()
        single_region_colors[region_id] = color
        result_colors[region_id] = color

    items: list[np.ndarray] = []
    for info in pair_infos:
        mask = regions[info["region"]]["mask"]
        if info["adjacent"]:
            patch = _adjacent_pair_patch(mask, info["major"], info["minor"], rng)
            if patch is None:
                return None
            items.append(patch)
        else:
            items.append(_single_patch(mask, info["major"]))
            items.append(_single_patch(mask, info["minor"]))

    for region_id, color in single_region_colors.items():
        items.append(_single_patch(regions[region_id]["mask"], color))

    template_h, template_w = template.shape
    max_item_h = max(patch.shape[0] for patch in items)
    max_item_w = max(patch.shape[1] for patch in items)
    grid_h_min = max(GRID_MIN, template_h + max_item_h + 3)
    grid_w_min = max(GRID_MIN, template_w + max_item_w + 3)
    if grid_h_min > GRID_MAX or grid_w_min > GRID_MAX:
        return None

    grid_h = rng.randint(grid_h_min, GRID_MAX)
    grid_w = rng.randint(grid_w_min, GRID_MAX)
    grid = np.full((grid_h, grid_w), BG, dtype=int)

    tr = (grid_h - template_h) // 2
    tc = (grid_w - template_w) // 2
    grid[tr : tr + template_h, tc : tc + template_w] = template
    template_bbox = (tr, tc, tr + template_h - 1, tc + template_w - 1)

    placements = _place_items(rng, grid.shape, template_bbox, items)
    if placements is None:
        return None

    for (r0, c0, _r1, _c1), patch in placements:
        sub = grid[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]]
        mask = patch != BG
        sub[mask] = patch[mask]

    expected = grid.copy()
    outside = np.ones(grid.shape, dtype=bool)
    outside[tr : tr + template_h, tc : tc + template_w] = False
    expected[outside & (expected != BG) & (expected != TEMPLATE_COLOR)] = BG
    for region_id, color in result_colors.items():
        for rr, cc in regions[region_id]["coords"]:
            expected[tr + rr, tc + cc] = color

    try:
        solved = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None
    if np.array_equal(grid, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(random)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
