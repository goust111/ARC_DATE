from __future__ import annotations

import itertools
import random
from collections import Counter

import numpy as np


BLACK = 0
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
COLOR_POOL = tuple(range(10))

STAMP_SHAPES = (
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=bool),
    np.array(
        [[1, 1, 1, 1], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0]],
        dtype=bool,
    ),
    np.array(
        [[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 1]],
        dtype=bool,
    ),
    np.array(
        [[1, 1, 1, 1], [1, 0, 0, 1], [1, 1, 0, 1], [1, 1, 1, 1]],
        dtype=bool,
    ),
    np.array(
        [[1, 0, 0, 0], [1, 1, 1, 0], [0, 0, 1, 0], [0, 0, 1, 1]],
        dtype=bool,
    ),
)


def _neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _touches_edge(coords: np.ndarray, height: int, width: int) -> bool:
    return bool(
        np.any(coords[:, 0] == 0)
        or np.any(coords[:, 0] == height - 1)
        or np.any(coords[:, 1] == 0)
        or np.any(coords[:, 1] == width - 1)
    )


def _components(mask: np.ndarray) -> list[np.ndarray]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if not (0 <= nr < height and 0 <= nc < width):
                        continue
                    if seen[nr, nc] or not mask[nr, nc]:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            parts.append(np.asarray(coords, dtype=int))
    return parts


def _erode4(mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(mask, dtype=bool)
    if mask.shape[0] < 3 or mask.shape[1] < 3:
        return out
    out[1:-1, 1:-1] = (
        mask[1:-1, 1:-1]
        & mask[:-2, 1:-1]
        & mask[2:, 1:-1]
        & mask[1:-1, :-2]
        & mask[1:-1, 2:]
    )
    return out


def _extract_regions(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    height, width = grid.shape
    regions: list[dict[str, object]] = []
    for coords in _components(grid == bg):
        if _touches_edge(coords, height, width):
            continue
        border_colors = set()
        comp_mask = np.zeros(grid.shape, dtype=bool)
        comp_mask[coords[:, 0], coords[:, 1]] = True
        for r, c in coords:
            for nr, nc in _neighbors4(int(r), int(c)):
                if not (0 <= nr < height and 0 <= nc < width):
                    continue
                if comp_mask[nr, nc]:
                    continue
                border_colors.add(int(grid[nr, nc]))
        if border_colors != {BLACK}:
            continue

        r0, c0, r1, c1 = _bbox(coords)
        local_mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        local_mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
        regions.append(
            {
                "bbox": (r0, c0, r1, c1),
                "mask": local_mask,
            }
        )
    return regions


def _extract_patch(grid: np.ndarray, coords: np.ndarray) -> dict[str, object]:
    values = grid[coords[:, 0], coords[:, 1]]
    counts = Counter(int(v) for v in values)
    stamp_colors = [color for color, count in counts.items() if count > 1]
    assert len(stamp_colors) == 1, "expected exactly one repeated stamp color per auxiliary cluster"
    stamp_color = stamp_colors[0]

    stamp_mask = values == stamp_color
    stamp_coords = coords[stamp_mask]
    other_coords = coords[~stamp_mask]
    other_values = values[~stamp_mask]
    assert other_coords.shape[0] >= 2, "color strip must have length at least two"

    sr0, sc0, sr1, sc1 = _bbox(stamp_coords)
    stamp = np.zeros((sr1 - sr0 + 1, sc1 - sc0 + 1), dtype=bool)
    stamp[stamp_coords[:, 0] - sr0, stamp_coords[:, 1] - sc0] = True

    rows = np.unique(other_coords[:, 0])
    cols = np.unique(other_coords[:, 1])
    assert rows.size == 1 or cols.size == 1, "color strip must be a single row or column"
    if rows.size == 1:
        order = np.argsort(other_coords[:, 1], kind="stable")
    else:
        order = np.argsort(other_coords[:, 0], kind="stable")

    palette = [int(other_values[i]) for i in order]
    assert len(set(palette)) == len(palette), "color strip colors must be distinct"
    return {
        "bbox": _bbox(coords),
        "stamp": stamp,
        "palette": palette,
    }


def _interval_gap(a0: int, a1: int, b0: int, b1: int) -> int:
    if a1 < b0:
        return b0 - a1
    if b1 < a0:
        return a0 - b1
    return 0


def _bbox_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    return _interval_gap(a[0], a[2], b[0], b[2]) + _interval_gap(a[1], a[3], b[1], b[3])


def _center_distance(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    ar = a[0] + a[2]
    ac = a[1] + a[3]
    br = b[0] + b[2]
    bc = b[1] + b[3]
    return abs(ar - br) + abs(ac - bc)


def _pair_regions_and_patches(
    regions: list[dict[str, object]], patches: list[dict[str, object]]
) -> list[tuple[dict[str, object], dict[str, object]]]:
    assert len(regions) == len(patches) and regions, "expected one auxiliary cluster per enclosed region"
    n = len(regions)
    best_key = None
    best_perm = None
    for perm in itertools.permutations(range(n)):
        gap = 0
        center = 0
        for region_idx, patch_idx in enumerate(perm):
            region_bbox = regions[region_idx]["bbox"]
            patch_bbox = patches[patch_idx]["bbox"]
            gap += _bbox_gap(region_bbox, patch_bbox)
            center += _center_distance(region_bbox, patch_bbox)
        key = (gap, center)
        if best_key is None or key < best_key:
            best_key = key
            best_perm = perm
    assert best_perm is not None
    return [(regions[i], patches[best_perm[i]]) for i in range(n)]


def _paint_region(
    out: np.ndarray,
    region: dict[str, object],
    patch: dict[str, object],
    bg: int,
) -> None:
    r0, c0, _r1, _c1 = region["bbox"]
    inside = region["mask"]
    stamp = patch["stamp"]
    palette = patch["palette"]
    stamp_h, stamp_w = stamp.shape
    index = 0
    for top in range(0, inside.shape[0] - stamp_h + 1, stamp_h):
        for left in range(0, inside.shape[1] - stamp_w + 1, stamp_w):
            window = inside[top : top + stamp_h, left : left + stamp_w]
            if not np.all(window):
                continue
            color = palette[index % len(palette)]
            index += 1
            view = out[r0 + top : r0 + top + stamp_h, c0 + left : c0 + left + stamp_w]
            view[stamp] = color
            view[~stamp] = bg


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)

    regions = _extract_regions(g, bg)
    aux_mask = (g != bg) & (g != BLACK)
    patches = [_extract_patch(g, coords) for coords in _components(aux_mask)]
    pairs = _pair_regions_and_patches(regions, patches)

    out = g.copy()
    out[aux_mask] = bg
    for region, patch in pairs:
        _paint_region(out, region, patch, bg)
    return out


def _slot_count(interior: np.ndarray, stamp: np.ndarray) -> int:
    count = 0
    h, w = stamp.shape
    for top in range(0, interior.shape[0] - h + 1, h):
        for left in range(0, interior.shape[1] - w + 1, w):
            if np.all(interior[top : top + h, left : left + w]):
                count += 1
    return count


def _make_area(rng: random.Random, stamp_h: int, stamp_w: int, polygon_kind: str) -> np.ndarray:
    min_h = max(7, stamp_h + 3)
    min_w = max(7, stamp_w + 3)
    outer_h = rng.randint(min_h, min_h + 5)
    outer_w = rng.randint(min_w, min_w + 6)
    area = np.ones((outer_h, outer_w), dtype=bool)
    if polygon_kind == "rect":
        return area

    side = rng.choice(("top", "bottom", "left", "right"))
    if side in ("top", "bottom"):
        if outer_w < 7 or outer_h < 7:
            return area
        notch_w = rng.randint(2, outer_w - 4)
        notch_d = rng.randint(2, outer_h - 4)
        start = rng.randint(2, outer_w - notch_w - 2)
        rows = slice(0, notch_d) if side == "top" else slice(outer_h - notch_d, outer_h)
        area[rows, start : start + notch_w] = False
    else:
        if outer_h < 7 or outer_w < 7:
            return area
        notch_h = rng.randint(2, outer_h - 4)
        notch_d = rng.randint(2, outer_w - 4)
        start = rng.randint(2, outer_h - notch_h - 2)
        cols = slice(0, notch_d) if side == "left" else slice(outer_w - notch_d, outer_w)
        area[start : start + notch_h, cols] = False
    return area


def _build_stamp_patch(
    rng: random.Random,
    bg: int,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    base = rng.choice(STAMP_SHAPES)
    stamp = np.rot90(base, k=rng.randrange(4)).copy()
    h, w = stamp.shape

    side = rng.choice(("top", "bottom", "left", "right"))
    if side in ("top", "bottom"):
        max_len = min(4, w)
        strip_len = rng.randint(2, max_len)
        edge = stamp[0, :] if side == "top" else stamp[-1, :]
        starts = [s for s in range(w - strip_len + 1) if edge[s : s + strip_len].any()]
        start = rng.choice(starts)
    else:
        max_len = min(4, h)
        strip_len = rng.randint(2, max_len)
        edge = stamp[:, 0] if side == "left" else stamp[:, -1]
        starts = [s for s in range(h - strip_len + 1) if edge[s : s + strip_len].any()]
        start = rng.choice(starts)

    available = [c for c in COLOR_POOL if c not in (BLACK, bg)]
    chosen = rng.sample(available, strip_len + 1)
    stamp_color = chosen[0]
    palette = chosen[1:]

    if side in ("top", "bottom"):
        patch = np.full((h + 1, w), bg, dtype=int)
        row = 0 if side == "top" else h
        stamp_row = 1 if side == "top" else 0
        patch[stamp_row : stamp_row + h, :][stamp] = stamp_color
        for i, color in enumerate(palette):
            patch[row, start + i] = color
    else:
        patch = np.full((h, w + 1), bg, dtype=int)
        col = 0 if side == "left" else w
        stamp_col = 1 if side == "left" else 0
        patch[:, stamp_col : stamp_col + w][stamp] = stamp_color
        for i, color in enumerate(palette):
            patch[start + i, col] = color

    return stamp, patch, palette


def _compose_group(
    rng: random.Random,
    boundary: np.ndarray,
    patch: np.ndarray,
    bg: int,
) -> np.ndarray:
    bh, bw = boundary.shape
    ph, pw = patch.shape
    gap = rng.randint(1, 2)
    direction = rng.choice(("top", "bottom", "left", "right"))

    if direction in ("top", "bottom"):
        canvas_h = bh + ph + gap + 6
        canvas_w = max(bw, pw) + 6
        canvas = np.full((canvas_h, canvas_w), bg, dtype=int)
        if direction == "top":
            boundary_r = ph + gap + 2
            patch_r = 2
        else:
            boundary_r = 2
            patch_r = boundary_r + bh + gap
        boundary_c = rng.randint(2, canvas_w - bw - 2)
        low = max(2, boundary_c - pw + 1)
        high = min(canvas_w - pw - 2, boundary_c + bw - 1)
        patch_c = rng.randint(low, high)
        canvas[boundary_r : boundary_r + bh, boundary_c : boundary_c + bw][boundary] = BLACK
        patch_view = canvas[patch_r : patch_r + ph, patch_c : patch_c + pw]
        patch_view[patch != bg] = patch[patch != bg]
    else:
        canvas_h = max(bh, ph) + 6
        canvas_w = bw + pw + gap + 6
        canvas = np.full((canvas_h, canvas_w), bg, dtype=int)
        if direction == "left":
            boundary_c = pw + gap + 2
            patch_c = 2
        else:
            boundary_c = 2
            patch_c = boundary_c + bw + gap
        boundary_r = rng.randint(2, canvas_h - bh - 2)
        low = max(2, boundary_r - ph + 1)
        high = min(canvas_h - ph - 2, boundary_r + bh - 1)
        patch_r = rng.randint(low, high)
        canvas[boundary_r : boundary_r + bh, boundary_c : boundary_c + bw][boundary] = BLACK
        patch_view = canvas[patch_r : patch_r + ph, patch_c : patch_c + pw]
        patch_view[patch != bg] = patch[patch != bg]

    occupied = np.argwhere(canvas != bg)
    r0, c0, r1, c1 = _bbox(occupied)
    return canvas[r0 : r1 + 1, c0 : c1 + 1]


def _build_group(rng: random.Random, bg: int, polygon_kind: str) -> np.ndarray | None:
    for _ in range(100):
        stamp, patch, _palette = _build_stamp_patch(rng, bg)
        for _ in range(100):
            area = _make_area(rng, stamp.shape[0], stamp.shape[1], polygon_kind)
            interior = _erode4(area)
            if interior.sum() == 0:
                continue
            if len(_components(interior)) != 1:
                continue
            if _slot_count(interior, stamp) == 0:
                continue
            boundary = area & ~interior
            group = _compose_group(rng, boundary, patch, bg)
            return group
    return None


def _place_groups(rng: random.Random, groups: list[np.ndarray], bg: int) -> np.ndarray | None:
    for _ in range(200):
        height = rng.randint(22, 30)
        width = rng.randint(22, 30)
        if any(group.shape[0] + 2 > height or group.shape[1] + 2 > width for group in groups):
            continue

        occ = np.zeros((height, width), dtype=bool)
        placed: list[tuple[np.ndarray, int, int]] = []
        success = True
        for group in sorted(groups, key=lambda arr: arr.shape[0] * arr.shape[1], reverse=True):
            gh, gw = group.shape
            placed_here = False
            for _ in range(300):
                top = rng.randint(1, height - gh - 1)
                left = rng.randint(1, width - gw - 1)
                r0 = max(0, top - 2)
                r1 = min(height, top + gh + 2)
                c0 = max(0, left - 2)
                c1 = min(width, left + gw + 2)
                if occ[r0:r1, c0:c1].any():
                    continue
                occ[top : top + gh, left : left + gw] |= group != bg
                placed.append((group, top, left))
                placed_here = True
                break
            if not placed_here:
                success = False
                break
        if not success:
            continue

        grid = np.full((height, width), bg, dtype=int)
        for group, top, left in placed:
            gh, gw = group.shape
            view = grid[top : top + gh, left : left + gw]
            mask = group != bg
            view[mask] = group[mask]
        if _get_bg(grid) != bg:
            continue
        return grid
    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(400):
        bg = rng.choice([c for c in COLOR_POOL if c != BLACK])
        group_count = rng.randint(1, 3)
        kinds = [rng.choice(("rect", "notch")) for _ in range(group_count)]
        if group_count >= 2:
            kinds[0] = "rect"
            kinds[1] = "notch"
        groups = []
        failed = False
        for kind in kinds:
            group = _build_group(rng, bg, kind)
            if group is None:
                failed = True
                break
            groups.append(group)
        if failed:
            continue

        grid = _place_groups(rng, groups, bg)
        if grid is None:
            continue

        out = main(grid)
        if out.shape != grid.shape:
            continue
        if not np.any(out != grid):
            continue
        return grid
    raise RuntimeError("failed to generate a valid puzzle input")
