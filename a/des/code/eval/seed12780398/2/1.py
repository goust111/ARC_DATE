import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
MAROON = 9
PATTERN_COLORS = [1, 2, 3, 4, 6, 7, 8]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

SHAPES = [
    (1, 1),
    (1, 2),
    (2, 1),
    (2, 2),
    (1, 3),
    (3, 1),
    (2, 3),
    (3, 2),
]

SAMPLE_H = 3
SAMPLE_W = 3
PATTERN_BOX_H = 3
PATTERN_BOX_W = 3
GEN_RETRIES = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in points]
    cs = [c for _, c in points]
    return min(rs), min(cs), max(rs), max(cs)


def _submask_from_points(points: list[tuple[int, int]]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(points)
    sub = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in points:
        sub[r - r0, c - c0] = True
    return sub, (r0, c0, r1, c1)


def _shape_key(mask: np.ndarray) -> tuple[int, int, tuple[int, ...]]:
    pts = np.argwhere(mask)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    sub = mask[r0 : r1 + 1, c0 : c1 + 1]
    return int(sub.shape[0]), int(sub.shape[1]), tuple(int(x) for x in sub.flatten())


def _full_height_maroon_cols(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> list[int]:
    cols: list[int] = []
    for c in range(c0, c1 + 1):
        if np.all(grid[r0 : r1 + 1, c] == MAROON):
            cols.append(c)
    return cols


def _full_width_maroon_rows(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> list[int]:
    rows: list[int] = []
    for r in range(r0, r1 + 1):
        if np.all(grid[r, c0 : c1 + 1] == MAROON):
            rows.append(r)
    return rows


def _find_legend_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    maroon_comps = _components(grid == MAROON)
    assert maroon_comps, "legend border not found"
    best = max(maroon_comps, key=len)
    return _bbox(best)


def _parse_pattern_cell(cell: np.ndarray) -> np.ndarray:
    pts = np.argwhere(cell != BG)
    assert len(pts) > 0, "pattern cell must contain non-background colors"
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return cell[r0 : r1 + 1, c0 : c1 + 1].copy()


def _parse_legend(grid: np.ndarray) -> tuple[dict[tuple[int, int, tuple[int, ...]], np.ndarray], tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _find_legend_bbox(grid)
    v_lines = _full_height_maroon_cols(grid, r0, c0, r1, c1)
    h_lines = _full_width_maroon_rows(grid, r0, c0, r1, c1)
    mapping: dict[tuple[int, int, tuple[int, ...]], np.ndarray] = {}

    if len(h_lines) > 2:
        assert len(v_lines) == 3, "vertical legend must have one internal divider"
        left, divider, right = v_lines
        for ra, rb in zip(h_lines[:-1], h_lines[1:]):
            sr0, sr1 = ra + 1, rb - 1
            if sr0 > sr1:
                continue
            sample = grid[sr0 : sr1 + 1, left + 1 : divider]
            pattern_cell = grid[sr0 : sr1 + 1, divider + 1 : right]
            key = _shape_key(sample == BG)
            mapping[key] = _parse_pattern_cell(pattern_cell)
    else:
        assert len(h_lines) == 2, "horizontal legend must have only top/bottom frame rows"
        top, bottom = h_lines
        spans = [(a + 1, b - 1) for a, b in zip(v_lines[:-1], v_lines[1:]) if a + 1 <= b - 1]
        assert len(spans) % 2 == 0 and spans, "horizontal legend must contain sample/pattern span pairs"
        for i in range(0, len(spans), 2):
            sc0, sc1 = spans[i]
            pc0, pc1 = spans[i + 1]
            sample = grid[top + 1 : bottom, sc0 : sc1 + 1]
            pattern_cell = grid[top + 1 : bottom, pc0 : pc1 + 1]
            key = _shape_key(sample == BG)
            mapping[key] = _parse_pattern_cell(pattern_cell)

    assert mapping, "legend mapping must be non-empty"
    return mapping, (r0, c0, r1, c1)


def _outside_mask(comp: np.ndarray) -> np.ndarray:
    h, w = comp.shape
    outside = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()
    for r in range(h):
        for c in (0, w - 1):
            if not comp[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not comp[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not comp[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))
    return outside


def _region_holes_and_fill(comp: np.ndarray) -> tuple[list[dict[str, object]], np.ndarray]:
    h, w = comp.shape
    outside = _outside_mask(comp)
    hole_mask = (~comp) & (~outside)
    holes: list[dict[str, object]] = []
    for pts in _components(hole_mask):
        sub, (r0, c0, _r1, _c1) = _submask_from_points(pts)
        holes.append(
            {
                "top": int(r0),
                "left": int(c0),
                "area": int(len(pts)),
                "key": _shape_key(sub),
            }
        )

    boundary = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if not comp[r, c]:
                continue
            if r == 0 or r == h - 1 or c == 0 or c == w - 1:
                boundary[r, c] = True
                continue
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if outside[nr, nc]:
                    boundary[r, c] = True
                    break

    fill_mask = (~outside) & (~boundary)
    return holes, fill_mask


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    mapping, legend_bbox = _parse_legend(g)
    out = g.copy()

    region_mask = g == GREY
    r0, c0, r1, c1 = legend_bbox
    region_mask[r0 : r1 + 1, c0 : c1 + 1] = False

    for pts in _components(region_mask):
        comp, (br0, bc0, br1, bc1) = _submask_from_points(pts)
        holes, fill_mask = _region_holes_and_fill(comp)
        assert holes, "every region must contain at least one hole"
        holes.sort(key=lambda item: (int(item["top"]), int(item["left"])))
        max_area = max(int(item["area"]) for item in holes)
        chosen = next(item for item in holes if int(item["area"]) == max_area)
        key = chosen["key"]
        assert key in mapping, "region hole shape must exist in legend"
        pattern = mapping[key]
        ph, pw = pattern.shape

        for rr, cc in np.argwhere(fill_mask):
            out[br0 + rr, bc0 + cc] = int(pattern[rr % ph, cc % pw])

    return out


def _random_pattern(rng: random.Random, used: set[bytes]) -> np.ndarray:
    while True:
        ph, pw = rng.choice([(2, 2), (3, 3)])
        motif = rng.choice(["checker", "rows", "cols", "random"])
        colors = rng.sample(PATTERN_COLORS, rng.randint(2, min(4, ph * pw)))
        arr = np.zeros((ph, pw), dtype=int)

        if motif == "checker":
            for r in range(ph):
                for c in range(pw):
                    arr[r, c] = colors[(r + c) % len(colors)]
        elif motif == "rows":
            row_colors = [rng.choice(colors) for _ in range(ph)]
            if len(set(row_colors)) == 1:
                row_colors[-1] = colors[(colors.index(row_colors[-1]) + 1) % len(colors)]
            for r in range(ph):
                arr[r, :] = row_colors[r]
        elif motif == "cols":
            col_colors = [rng.choice(colors) for _ in range(pw)]
            if len(set(col_colors)) == 1:
                col_colors[-1] = colors[(colors.index(col_colors[-1]) + 1) % len(colors)]
            for c in range(pw):
                arr[:, c] = col_colors[c]
        else:
            for r in range(ph):
                for c in range(pw):
                    arr[r, c] = rng.choice(colors)

        if len(set(int(x) for x in arr.flatten())) < 2:
            continue
        key = arr.tobytes()
        if key in used:
            continue
        used.add(key)
        return arr


def _make_legend_patch(
    entries: list[tuple[tuple[int, int], np.ndarray]],
    orientation: str,
) -> np.ndarray:
    if orientation == "right":
        n = len(entries)
        h = 1 + n * SAMPLE_H + (n - 1) + 1
        w = 1 + SAMPLE_W + 1 + PATTERN_BOX_W + 1
        legend = np.full((h, w), BG, dtype=int)
        legend[0, :] = MAROON
        legend[-1, :] = MAROON
        legend[:, 0] = MAROON
        legend[:, -1] = MAROON
        divider = 1 + SAMPLE_W
        legend[:, divider] = MAROON

        for i, (shape, pattern) in enumerate(entries):
            rs = 1 + i * (SAMPLE_H + 1)
            re = rs + SAMPLE_H
            if i < n - 1:
                legend[re, :] = MAROON
            sample = legend[rs:re, 1 : 1 + SAMPLE_W]
            sample[:, :] = GREY
            hh, ww = shape
            r_off = (SAMPLE_H - hh) // 2
            c_off = (SAMPLE_W - ww) // 2
            sample[r_off : r_off + hh, c_off : c_off + ww] = BG

            box = legend[rs:re, divider + 1 : divider + 1 + PATTERN_BOX_W]
            ph, pw = pattern.shape
            pr = (PATTERN_BOX_H - ph) // 2
            pc = (PATTERN_BOX_W - pw) // 2
            box[pr : pr + ph, pc : pc + pw] = pattern
        return legend

    n = len(entries)
    h = 1 + SAMPLE_H + 1
    w = 1 + n * (SAMPLE_W + 1 + PATTERN_BOX_W) + (n - 1) + 1
    legend = np.full((h, w), BG, dtype=int)
    legend[0, :] = MAROON
    legend[-1, :] = MAROON
    legend[:, 0] = MAROON
    legend[:, -1] = MAROON

    col = 1
    for i, (shape, pattern) in enumerate(entries):
        divider = col + SAMPLE_W
        legend[:, divider] = MAROON
        sample = legend[1 : 1 + SAMPLE_H, col : col + SAMPLE_W]
        sample[:, :] = GREY
        hh, ww = shape
        r_off = (SAMPLE_H - hh) // 2
        c_off = (SAMPLE_W - ww) // 2
        sample[r_off : r_off + hh, c_off : c_off + ww] = BG

        box = legend[1 : 1 + SAMPLE_H, divider + 1 : divider + 1 + PATTERN_BOX_W]
        ph, pw = pattern.shape
        pr = (PATTERN_BOX_H - ph) // 2
        pc = (PATTERN_BOX_W - pw) // 2
        box[pr : pr + ph, pc : pc + pw] = pattern

        col = divider + 1 + PATTERN_BOX_W
        if i < n - 1:
            legend[:, col] = MAROON
            col += 1
    return legend


def _is_connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return False
    seen = np.zeros_like(mask, dtype=bool)
    start = tuple(int(x) for x in pts[0])
    q = deque([start])
    seen[start] = True
    count = 0
    while q:
        r, c = q.popleft()
        count += 1
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
    return count == int(mask.sum())


def _boundary_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    boundary = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            if r == 0 or r == h - 1 or c == 0 or c == w - 1:
                boundary[r, c] = True
                continue
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if not mask[nr, nc]:
                    boundary[r, c] = True
                    break
    return boundary


def _place_holes(
    rng: random.Random,
    h: int,
    w: int,
    hole_shapes: list[tuple[int, int]],
) -> list[tuple[int, int, int, int]] | None:
    used = np.zeros((h, w), dtype=bool)
    placements: list[tuple[int, int, int, int]] = []
    for hh, ww in sorted(hole_shapes, key=lambda item: item[0] * item[1], reverse=True):
        candidates: list[tuple[int, int]] = []
        for r in range(1, h - hh):
            for c in range(1, w - ww):
                if used[max(0, r - 1) : min(h, r + hh + 1), max(0, c - 1) : min(w, c + ww + 1)].any():
                    continue
                candidates.append((r, c))
        if not candidates:
            return None
        r, c = rng.choice(candidates)
        placements.append((r, c, hh, ww))
        used[max(0, r - 1) : min(h, r + hh + 1), max(0, c - 1) : min(w, c + ww + 1)] = True
    return placements


def _carve_notches(
    rng: random.Random,
    mask: np.ndarray,
    reserved: np.ndarray,
) -> np.ndarray:
    h, w = mask.shape
    notch_count = rng.randint(0, 3)
    for _ in range(notch_count):
        for _try in range(30):
            side = rng.choice(["top", "bottom", "left", "right"])
            trial = mask.copy()
            if side in ("top", "bottom"):
                depth = rng.randint(1, max(1, h // 3))
                span = rng.randint(1, max(1, w - 2))
                start = rng.randint(0, w - span)
                if span >= w:
                    continue
                rr = slice(0, depth) if side == "top" else slice(h - depth, h)
                cc = slice(start, start + span)
            else:
                depth = rng.randint(1, max(1, w // 3))
                span = rng.randint(1, max(1, h - 2))
                start = rng.randint(0, h - span)
                if span >= h:
                    continue
                rr = slice(start, start + span)
                cc = slice(0, depth) if side == "left" else slice(w - depth, w)
            if reserved[rr, cc].any():
                continue
            trial[rr, cc] = False
            if not _is_connected(trial):
                continue
            core = trial & (~_boundary_mask(trial))
            if int(core.sum()) < 4:
                continue
            mask = trial
            break
    return mask


def _region_from_holes(
    rng: random.Random,
    hole_shapes: list[tuple[int, int]],
) -> np.ndarray | None:
    total_area = sum(h * w for h, w in hole_shapes)
    min_side = max(max(h for h, _ in hole_shapes), max(w for _, w in hole_shapes)) + 4
    for _ in range(120):
        h = rng.randint(max(7, min_side), max(10, min_side + 4))
        w = rng.randint(max(7, min_side), max(10, min_side + 4))
        if h * w < total_area * 4 + 20:
            continue
        holes = _place_holes(rng, h, w, hole_shapes)
        if holes is None:
            continue

        mask = np.ones((h, w), dtype=bool)
        reserved = np.zeros((h, w), dtype=bool)
        for r, c, hh, ww in holes:
            reserved[max(0, r - 1) : min(h, r + hh + 1), max(0, c - 1) : min(w, c + ww + 1)] = True
        mask = _carve_notches(rng, mask, reserved)
        for r, c, hh, ww in holes:
            mask[r : r + hh, c : c + ww] = False

        if not _is_connected(mask):
            continue
        found_holes, fill_mask = _region_holes_and_fill(mask)
        if len(found_holes) != len(hole_shapes):
            continue

        want = sorted((hh, ww) for hh, ww in hole_shapes)
        got = sorted((int(item["key"][0]), int(item["key"][1])) for item in found_holes)
        if want != got:
            continue
        if int(fill_mask.sum()) < 8:
            continue
        return mask
    return None


def _choose_hole_recipe(
    rng: random.Random,
    branch: str,
    target: tuple[int, int],
    legend_shapes: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    if branch == "single":
        return [target]
    if branch == "same":
        count = rng.choice([2, 3])
        return [target] * count

    target_area = target[0] * target[1]
    smaller = [shape for shape in legend_shapes if shape[0] * shape[1] < target_area]
    if not smaller:
        return [target]
    extras = rng.choice([1, 2])
    recipe = [target]
    for _ in range(extras):
        recipe.append(rng.choice(smaller))
    return recipe


def _branch_targets(
    rng: random.Random,
    legend_shapes: list[tuple[int, int]],
) -> list[tuple[str, tuple[int, int]]]:
    branch_types = ["single", "same", "mixed"]
    if len(legend_shapes) == 4:
        branch_types.append(rng.choice(branch_types))
    rng.shuffle(branch_types)

    sorted_shapes = sorted(legend_shapes, key=lambda shape: shape[0] * shape[1])
    mixed_candidates = [shape for shape in sorted_shapes if shape[0] * shape[1] > sorted_shapes[0][0] * sorted_shapes[0][1]]
    assignments: list[tuple[str, tuple[int, int]]] = []
    unused = legend_shapes[:]
    rng.shuffle(unused)
    for branch in branch_types:
        if branch == "mixed" and mixed_candidates:
            choices = [shape for shape in unused if shape in mixed_candidates] or mixed_candidates
            target = rng.choice(choices)
        else:
            target = unused[0] if unused else rng.choice(legend_shapes)
        if target in unused:
            unused.remove(target)
        assignments.append((branch, target))
    return assignments


def _place_rectangles(
    rng: random.Random,
    area_h: int,
    area_w: int,
    patches: list[np.ndarray],
) -> list[tuple[int, int]] | None:
    occupied = np.zeros((area_h, area_w), dtype=bool)
    placements: list[tuple[int, int]] = []
    ordered = sorted(enumerate(patches), key=lambda item: item[1].shape[0] * item[1].shape[1], reverse=True)
    tmp: dict[int, tuple[int, int]] = {}

    for idx, patch in ordered:
        ph, pw = patch.shape
        candidates: list[tuple[int, int]] = []
        for r in range(area_h - ph + 1):
            for c in range(area_w - pw + 1):
                rr0 = max(0, r - 1)
                rr1 = min(area_h, r + ph + 1)
                cc0 = max(0, c - 1)
                cc1 = min(area_w, c + pw + 1)
                if occupied[rr0:rr1, cc0:cc1].any():
                    continue
                candidates.append((r, c))
        if not candidates:
            return None
        tmp[idx] = rng.choice(candidates)
        r, c = tmp[idx]
        occupied[r : r + ph, c : c + pw] = True

    for i in range(len(patches)):
        placements.append(tmp[i])
    return placements


def _build_grid(
    rng: random.Random,
    legend_patch: np.ndarray,
    orientation: str,
    regions: list[np.ndarray],
) -> np.ndarray | None:
    legend_h, legend_w = legend_patch.shape
    if orientation == "right":
        h = rng.randint(max(22, legend_h + 2), max(28, legend_h + 8))
        w = rng.randint(24, 34)
        if w <= legend_w + 8:
            w = legend_w + rng.randint(10, 16)
        grid = np.full((h, w), BG, dtype=int)
        legend_r = rng.randint(0, h - legend_h)
        legend_c = w - legend_w
        area_h = h
        area_w = legend_c - 2
        if area_w < 10:
            return None
        placements = _place_rectangles(rng, area_h, area_w, regions)
        if placements is None:
            return None
    else:
        w = rng.randint(max(28, legend_w), max(36, legend_w + 4))
        h = rng.randint(18, 28)
        if h <= legend_h + 8:
            h = legend_h + rng.randint(9, 15)
        grid = np.full((h, w), BG, dtype=int)
        legend_r = h - legend_h
        legend_c = rng.randint(0, w - legend_w)
        area_h = legend_r - 2
        area_w = w
        if area_h < 10:
            return None
        placements = _place_rectangles(rng, area_h, area_w, regions)
        if placements is None:
            return None

    grid[legend_r : legend_r + legend_h, legend_c : legend_c + legend_w] = legend_patch
    for region, (r, c) in zip(regions, placements):
        grid[r : r + region.shape[0], c : c + region.shape[1]][region] = GREY
    return grid


def _generate_candidate(rng: random.Random) -> np.ndarray | None:
    entry_count = rng.choice([3, 4])
    legend_shapes = rng.sample(SHAPES, entry_count)
    if len({h * w for h, w in legend_shapes}) == 1:
        return None

    pattern_used: set[bytes] = set()
    entries = [(shape, _random_pattern(rng, pattern_used)) for shape in legend_shapes]
    orientation = rng.choice(["right", "bottom"])
    legend_patch = _make_legend_patch(entries, orientation)

    regions: list[np.ndarray] = []
    for branch, target in _branch_targets(rng, legend_shapes):
        hole_recipe = _choose_hole_recipe(rng, branch, target, legend_shapes)
        region = _region_from_holes(rng, hole_recipe)
        if region is None:
            return None
        regions.append(region)

    grid = _build_grid(rng, legend_patch, orientation, regions)
    if grid is None:
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _generate_candidate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
