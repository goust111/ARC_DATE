from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
COLOR_POOL = tuple(range(1, 10))
GEN_RETRY_LIMIT = 300
TEMPLATE_KINDS = ("plain", "split", "double", "cross", "elbow")


def neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for dr, dc in DIR4]


def get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _anchor(coords: list[tuple[int, int]]) -> tuple[int, int]:
    return min(coords)


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, list[list[tuple[int, int]]]]:
    h, w = mask.shape
    labels = -np.ones((h, w), dtype=int)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] >= 0:
                continue
            cid = len(comps)
            q = deque([(r, c)])
            labels[r, c] = cid
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and labels[nr, nc] < 0:
                        labels[nr, nc] = cid
                        q.append((nr, nc))
            comps.append(coords)
    return labels, comps


def _polygon_region_groups(grid: np.ndarray) -> tuple[int, dict[int, list[dict]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    fg_mask = g != bg
    fg_labels, fg_comps = _label_components(fg_mask)
    fg_anchors = [_anchor(coords) for coords in fg_comps]

    bg_labels, bg_comps = _label_components(g == bg)
    _ = bg_labels  # kept for symmetry with the foreground pass

    groups: dict[int, list[dict]] = defaultdict(list)
    for coords in bg_comps:
        if any(r in (0, h - 1) or c in (0, w - 1) for r, c in coords):
            continue

        adj_counts: Counter[int] = Counter()
        boundary_pixels: set[tuple[int, int]] = set()
        for r, c in coords:
            for nr, nc in neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and fg_mask[nr, nc]:
                    wall_id = int(fg_labels[nr, nc])
                    adj_counts[wall_id] += 1
                    boundary_pixels.add((nr, nc))

        if not adj_counts:
            continue

        owner = max(
            adj_counts.items(),
            key=lambda kv: (kv[1], -fg_anchors[kv[0]][0], -fg_anchors[kv[0]][1]),
        )[0]
        owner_boundary = sorted((r, c) for r, c in boundary_pixels if int(fg_labels[r, c]) == owner)
        color_counts = Counter(int(g[r, c]) for r, c in owner_boundary)
        groups[owner].append(
            {
                "coords": coords,
                "area": len(coords),
                "anchor": _anchor(coords),
                "boundary_scan": owner_boundary,
                "boundary_color_counts": color_counts,
            }
        )
    return bg, groups


def _pick_fill_color(grid: np.ndarray, region: dict) -> int:
    counts: Counter[int] = region["boundary_color_counts"]
    max_count = max(counts.values())
    for r, c in region["boundary_scan"]:
        color = int(grid[r, c])
        if counts[color] == max_count:
            return color
    raise AssertionError("region boundary must have a most frequent color")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect colored orthogonal line segments into polygons and fill each polygon's
    interior using the boundary-majority color of its smallest interior subregion.
    """
    g = np.asarray(grid, dtype=int)
    out = np.array(g, copy=True)

    _bg, groups = _polygon_region_groups(g)
    for regions in groups.values():
        if not regions:
            continue
        target = min(regions, key=lambda region: (region["area"], region["anchor"][0], region["anchor"][1]))
        fill_color = _pick_fill_color(g, target)
        for region in regions:
            rr, cc = zip(*region["coords"])
            out[rr, cc] = fill_color
    return out


def _line_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = a
    r1, c1 = b
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        return [(r0, c) for c in range(c0, c1 + step, step)]
    assert c0 == c1
    step = 1 if r1 >= r0 else -1
    return [(r, c0) for r in range(r0, r1 + step, step)]


def _render_segments(
    shape: tuple[int, int],
    segments: list[tuple[tuple[int, int], tuple[int, int]]],
    colors: list[int],
    order: list[int] | None = None,
) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    use_order = list(range(len(segments))) if order is None else order
    for idx in use_order:
        for r, c in _line_cells(*segments[idx]):
            grid[r, c] = int(colors[idx])
    return grid


def _validate_polygon_grid(grid: np.ndarray) -> bool:
    fg_mask = grid != 0
    _fg_labels, fg_comps = _label_components(fg_mask)
    if len(fg_comps) != 1:
        return False

    _bg, groups = _polygon_region_groups(grid)
    if len(groups) != 1:
        return False
    regions = next(iter(groups.values()))
    if not regions:
        return False

    min_area = min(region["area"] for region in regions)
    if sum(region["area"] == min_area for region in regions) != 1:
        return False

    target = min(regions, key=lambda region: (region["area"], region["anchor"][0], region["anchor"][1]))
    counts: Counter[int] = target["boundary_color_counts"]
    if not counts:
        return False
    max_count = max(counts.values())
    if sum(count == max_count for count in counts.values()) != 1:
        return False
    return True


def _build_plain(rng: random.Random) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    h = rng.randint(6, 11)
    w = rng.randint(6, 13)
    return [
        ((0, 0), (0, w - 1)),
        ((0, w - 1), (h - 1, w - 1)),
        ((h - 1, w - 1), (h - 1, 0)),
        ((h - 1, 0), (0, 0)),
    ]


def _build_split(rng: random.Random) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    for _ in range(60):
        h = rng.randint(6, 11)
        w = rng.randint(7, 13)
        x = rng.randint(2, w - 3)
        if x - 1 == w - x - 2:
            continue
        return [
            ((0, 0), (0, x)),
            ((0, x), (0, w - 1)),
            ((0, w - 1), (h - 1, w - 1)),
            ((h - 1, w - 1), (h - 1, x)),
            ((h - 1, x), (h - 1, 0)),
            ((h - 1, 0), (0, 0)),
            ((0, x), (h - 1, x)),
        ]
    raise RuntimeError("failed to build split template")


def _build_double(rng: random.Random) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    for _ in range(80):
        h = rng.randint(6, 11)
        w = rng.randint(9, 15)
        x1 = rng.randint(2, w - 5)
        x2 = rng.randint(x1 + 2, w - 3)
        widths = [x1 - 1, x2 - x1 - 1, w - x2 - 2]
        if min(widths) < 1 or widths.count(min(widths)) != 1:
            continue
        return [
            ((0, 0), (0, x1)),
            ((0, x1), (0, x2)),
            ((0, x2), (0, w - 1)),
            ((0, w - 1), (h - 1, w - 1)),
            ((h - 1, w - 1), (h - 1, x2)),
            ((h - 1, x2), (h - 1, x1)),
            ((h - 1, x1), (h - 1, 0)),
            ((h - 1, 0), (0, 0)),
            ((0, x1), (h - 1, x1)),
            ((0, x2), (h - 1, x2)),
        ]
    raise RuntimeError("failed to build double template")


def _build_cross(rng: random.Random) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    for _ in range(80):
        h = rng.randint(8, 12)
        w = rng.randint(8, 14)
        x = rng.randint(2, w - 3)
        y = rng.randint(2, h - 3)
        areas = [
            (y - 1) * (x - 1),
            (y - 1) * (w - x - 2),
            (h - y - 2) * (x - 1),
            (h - y - 2) * (w - x - 2),
        ]
        if min(areas) < 1 or areas.count(min(areas)) != 1:
            continue
        return [
            ((0, 0), (0, x)),
            ((0, x), (0, w - 1)),
            ((0, w - 1), (y, w - 1)),
            ((y, w - 1), (h - 1, w - 1)),
            ((h - 1, w - 1), (h - 1, x)),
            ((h - 1, x), (h - 1, 0)),
            ((h - 1, 0), (y, 0)),
            ((y, 0), (0, 0)),
            ((0, x), (y, x)),
            ((y, x), (h - 1, x)),
            ((y, 0), (y, x)),
            ((y, x), (y, w - 1)),
        ]
    raise RuntimeError("failed to build cross template")


def _build_elbow(rng: random.Random) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    for _ in range(80):
        h = rng.randint(8, 12)
        w = rng.randint(8, 14)
        x = rng.randint(2, w - 3)
        y = rng.randint(2, h - 3)
        small = (y - 1) * (w - x - 2)
        total = (h - 2) * (w - 2)
        if small < 1 or small * 2 == total:
            continue
        return [
            ((0, 0), (0, x)),
            ((0, x), (0, w - 1)),
            ((0, w - 1), (y, w - 1)),
            ((y, w - 1), (h - 1, w - 1)),
            ((h - 1, w - 1), (h - 1, 0)),
            ((h - 1, 0), (0, 0)),
            ((0, x), (y, x)),
            ((y, x), (y, w - 1)),
        ]
    raise RuntimeError("failed to build elbow template")


def _build_segments(rng: random.Random, kind: str) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    if kind == "plain":
        return _build_plain(rng)
    if kind == "split":
        return _build_split(rng)
    if kind == "double":
        return _build_double(rng)
    if kind == "cross":
        return _build_cross(rng)
    if kind == "elbow":
        return _build_elbow(rng)
    raise ValueError(kind)


def _choose_target_segments(
    segments: list[tuple[tuple[int, int], tuple[int, int]]],
    boundary_pixels: set[tuple[int, int]],
) -> set[int]:
    seg_boundary = [set(_line_cells(*segment)) & boundary_pixels for segment in segments]
    ranked = sorted(range(len(segments)), key=lambda idx: (-len(seg_boundary[idx]), idx))
    target_set: set[int] = set()
    covered: set[tuple[int, int]] = set()
    for idx in ranked:
        if not seg_boundary[idx]:
            continue
        target_set.add(idx)
        covered |= seg_boundary[idx]
        if len(covered) * 2 > len(boundary_pixels):
            break
    if not target_set:
        target_set.add(ranked[0])
    return target_set


def _random_transform(grid: np.ndarray, rng: random.Random) -> np.ndarray:
    out = np.array(grid, copy=True)
    if rng.random() < 0.5:
        out = np.fliplr(out)
    out = np.rot90(out, rng.randrange(4))
    return np.array(out, copy=True)


def _generate_polygon_patch(rng: random.Random) -> np.ndarray:
    for _ in range(120):
        kind = rng.choice(TEMPLATE_KINDS)
        segments = _build_segments(rng, kind)

        max_r = max(max(a[0], b[0]) for a, b in segments)
        max_c = max(max(a[1], b[1]) for a, b in segments)
        shape = (max_r + 1, max_c + 1)
        mask_grid = _render_segments(shape, segments, [1] * len(segments))
        if not _validate_polygon_grid(mask_grid):
            continue

        _bg, groups = _polygon_region_groups(mask_grid)
        regions = next(iter(groups.values()))
        target = min(regions, key=lambda region: (region["area"], region["anchor"][0], region["anchor"][1]))
        target_set = _choose_target_segments(segments, set(target["boundary_scan"]))

        palette = rng.sample(list(COLOR_POOL), rng.randint(3, 6))
        target_color = rng.choice(palette)
        other_colors = [color for color in palette if color != target_color] or [target_color]

        colors = []
        for idx in range(len(segments)):
            if idx in target_set:
                colors.append(target_color)
            else:
                colors.append(rng.choice(other_colors))

        if len(set(colors)) == 1 and len(segments) > 1:
            for idx in range(len(segments)):
                if idx not in target_set:
                    colors[idx] = other_colors[0]
                    break

        order = [idx for idx in range(len(segments)) if idx not in target_set] + [
            idx for idx in range(len(segments)) if idx in target_set
        ]
        patch = _render_segments(shape, segments, colors, order)
        patch = _random_transform(patch, rng)
        if _validate_polygon_grid(patch):
            return patch
    raise RuntimeError("failed to generate polygon patch")


def _generate_isolated_segment(rng: random.Random) -> np.ndarray:
    length = rng.randint(2, 6)
    color = rng.choice(COLOR_POOL)
    if rng.random() < 0.5:
        patch = np.zeros((1, length), dtype=int)
        patch[0, :] = color
    else:
        patch = np.zeros((length, 1), dtype=int)
        patch[:, 0] = color
    return patch


def _place_patch(grid: np.ndarray, blocked: np.ndarray, patch: np.ndarray, rng: random.Random) -> bool:
    h, w = grid.shape
    ph, pw = patch.shape
    coords = np.argwhere(patch != 0)
    if coords.size == 0 or ph + 2 > h or pw + 2 > w:
        return False

    for _ in range(150):
        top = rng.randint(1, h - ph - 1)
        left = rng.randint(1, w - pw - 1)
        if any(blocked[top + r, left + c] for r, c in coords):
            continue

        for r, c in coords:
            rr = top + int(r)
            cc = left + int(c)
            grid[rr, cc] = int(patch[r, c])
        for r, c in coords:
            rr = top + int(r)
            cc = left + int(c)
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        blocked[nr, nc] = True
        return True
    return False


def _valid_global_instance(grid: np.ndarray) -> bool:
    _bg, groups = _polygon_region_groups(grid)
    if not groups:
        return False

    changed = False
    out = main(grid)
    if np.any((grid != 0) & (out != grid)):
        return False

    for regions in groups.values():
        if not regions:
            continue
        min_area = min(region["area"] for region in regions)
        if sum(region["area"] == min_area for region in regions) != 1:
            return False
        target = min(regions, key=lambda region: (region["area"], region["anchor"][0], region["anchor"][1]))
        counts: Counter[int] = target["boundary_color_counts"]
        max_count = max(counts.values())
        if sum(count == max_count for count in counts.values()) != 1:
            return False
        fill_color = _pick_fill_color(grid, target)
        for region in regions:
            for r, c in region["coords"]:
                if int(out[r, c]) != fill_color:
                    return False
                changed = True
    return changed


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    polygon_count = rng.choice((1, 2))
    polygons = [_generate_polygon_patch(rng) for _ in range(polygon_count)]
    isolated_count = rng.choice((0, 1, 2, 3))
    isolated = [_generate_isolated_segment(rng) for _ in range(isolated_count)]
    objects = polygons + isolated

    max_h = max(patch.shape[0] for patch in objects)
    max_w = max(patch.shape[1] for patch in objects)
    min_h = max(18, max_h + 2)
    min_w = max(18, max_w + 2)
    if min_h > 30 or min_w > 30:
        return None

    h = rng.randint(min_h, 30)
    w = rng.randint(min_w, 30)
    grid = np.zeros((h, w), dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    for patch in sorted(objects, key=lambda arr: arr.shape[0] * arr.shape[1], reverse=True):
        if not _place_patch(grid, blocked, patch, rng):
            return None

    if not _valid_global_instance(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
