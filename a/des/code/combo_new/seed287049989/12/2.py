from __future__ import annotations

from collections import defaultdict, deque
import random

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_RETRIES = 400
LOCAL_SHAPE_RETRIES = 300


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _components_of_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or visited[r, c]:
                continue
            q = deque([(r, c)])
            visited[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _analyze_regions(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    visited = np.zeros((h, w), dtype=bool)
    comp_id = np.full((h, w), -1, dtype=int)
    regions: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == bg or visited[r, c]:
                continue
            q = deque([(r, c)])
            visited[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(g[nr, nc]) == color:
                        visited[nr, nc] = True
                        q.append((nr, nc))

            coords = np.array(pts, dtype=int)
            rid = len(regions)
            comp_id[coords[:, 0], coords[:, 1]] = rid
            regions.append(
                {
                    "id": rid,
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "area": int(coords.shape[0]),
                    "hole_count": 0,
                }
            )

    for coords in _components_of_mask(g == bg):
        if np.any(coords[:, 0] == 0) or np.any(coords[:, 0] == h - 1) or np.any(coords[:, 1] == 0) or np.any(coords[:, 1] == w - 1):
            continue
        owners = set()
        for r, c in coords:
            for dr, dc in DIR4:
                nr, nc = int(r + dr), int(c + dc)
                if 0 <= nr < h and 0 <= nc < w and comp_id[nr, nc] != -1:
                    owners.add(int(comp_id[nr, nc]))
        if len(owners) == 1:
            regions[next(iter(owners))]["hole_count"] += 1

    return regions


def _draw_square_ring(out: np.ndarray, r0: int, c0: int, size: int, layer: int, color: int) -> None:
    a = layer
    b = size - 1 - layer
    rr0 = r0 + a
    rr1 = r0 + b
    cc0 = c0 + a
    cc1 = c0 + b
    out[rr0, cc0 : cc1 + 1] = color
    out[rr1, cc0 : cc1 + 1] = color
    out[rr0 : rr1 + 1, cc0] = color
    out[rr0 : rr1 + 1, cc1] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group colored regions by their hole counts, then repack each group as a nested square-ring pattern.

    Input structure:
    - The background is black.
    - The foreground consists of disconnected 4-connected monochrome regions, each with its own color.
    - A region may contain zero, one, or two enclosed black holes.

    Rule:
    - Count how many enclosed black 4-connected background components belong to each colored region.
    - Group regions by that hole count, and sort each group by region area from largest to smallest.
    - For each group, draw a square of side 2*n-1 where n is the group size: the largest region colors the outer ring, the next colors the next ring, and the smallest colors the center.
    - Arrange the group squares from left to right in ascending hole-count order with one black column between them, vertically centering smaller squares.

    Key insight:
    - The decisive latent feature is not color or position, but how many enclosed background holes each region contains.
    """
    g = np.asarray(grid, dtype=int)
    bg = 0 if np.any(g == 0) else _most_common_color(g)
    regions = _analyze_regions(g, bg)
    if not regions:
        return np.full((1, 1), bg, dtype=int)

    groups: dict[int, list[dict]] = defaultdict(list)
    for region in regions:
        groups[int(region["hole_count"])].append(region)

    hole_keys = sorted(groups)
    side_lengths = [2 * len(groups[k]) - 1 for k in hole_keys]
    out_h = max(side_lengths)
    out_w = sum(side_lengths) + max(0, len(side_lengths) - 1)
    out = np.full((out_h, out_w), bg, dtype=int)

    col = 0
    for hole_count, side in zip(hole_keys, side_lengths):
        members = sorted(groups[hole_count], key=lambda item: (-int(item["area"]), int(item["bbox"][0]), int(item["bbox"][1])))
        row = (out_h - side) // 2
        for layer, item in enumerate(members):
            _draw_square_ring(out, row, col, side, layer, int(item["color"]))
        col += side + 1

    return out


def _sample_group_counts(rng: random.Random) -> dict[int, int]:
    patterns = [
        {0: 2, 1: 2, 2: 2},
        {0: 3, 1: 2, 2: 2},
        {0: 2, 1: 3, 2: 2},
        {0: 2, 1: 2, 2: 3},
        {0: 3, 1: 3, 2: 2},
        {0: 3, 1: 2, 2: 3},
        {0: 2, 1: 3, 2: 3},
        {0: 4, 1: 2, 2: 2},
        {0: 2, 1: 4, 2: 2},
        {0: 2, 1: 2, 2: 4},
    ]
    return dict(rng.choice(patterns))


def _separated(rect_a: tuple[int, int, int, int], rect_b: tuple[int, int, int, int], gap: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = rect_a
    br0, bc0, br1, bc1 = rect_b
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _make_region_mask(rng: random.Random, hole_count: int, compact: bool) -> np.ndarray:
    for _ in range(LOCAL_SHAPE_RETRIES):
        margin_max = 1 if compact else 2
        if hole_count == 0:
            core_h = rng.randint(3, 5 if compact else 6)
            core_w = rng.randint(3, 5 if compact else 6)
        elif hole_count == 1:
            core_h = rng.randint(4 if compact else 5, 6 if compact else 7)
            core_w = rng.randint(4 if compact else 5, 6 if compact else 7)
        else:
            core_h = rng.randint(5 if compact else 6, 7 if compact else 8)
            core_w = rng.randint(5 if compact else 6, 7 if compact else 8)

        top = rng.randint(0, margin_max)
        bottom = rng.randint(0, margin_max)
        left = rng.randint(0, margin_max)
        right = rng.randint(0, margin_max)
        h = core_h + top + bottom
        w = core_w + left + right
        if h > 10 or w > 10:
            continue

        mask = np.zeros((h, w), dtype=bool)
        r0, c0 = top, left
        mask[r0 : r0 + core_h, c0 : c0 + core_w] = True

        sides = ["top", "bottom", "left", "right"]
        rng.shuffle(sides)
        for side in sides:
            if rng.random() < 0.35:
                continue
            if side == "top" and top > 0:
                ph = rng.randint(1, top)
                pw = rng.randint(max(1, core_w // 2), core_w)
                start = rng.randint(0, core_w - pw)
                mask[r0 - ph : r0, c0 + start : c0 + start + pw] = True
            elif side == "bottom" and bottom > 0:
                ph = rng.randint(1, bottom)
                pw = rng.randint(max(1, core_w // 2), core_w)
                start = rng.randint(0, core_w - pw)
                mask[r0 + core_h : r0 + core_h + ph, c0 + start : c0 + start + pw] = True
            elif side == "left" and left > 0:
                pw = rng.randint(1, left)
                ph = rng.randint(max(1, core_h // 2), core_h)
                start = rng.randint(0, core_h - ph)
                mask[r0 + start : r0 + start + ph, c0 - pw : c0] = True
            elif side == "right" and right > 0:
                pw = rng.randint(1, right)
                ph = rng.randint(max(1, core_h // 2), core_h)
                start = rng.randint(0, core_h - ph)
                mask[r0 + start : r0 + start + ph, c0 + core_w : c0 + core_w + pw] = True

        holes: list[tuple[int, int, int, int]] = []
        inner_r0 = r0 + 1
        inner_r1 = r0 + core_h - 2
        inner_c0 = c0 + 1
        inner_c1 = c0 + core_w - 2
        if inner_r0 > inner_r1 or inner_c0 > inner_c1:
            continue

        ok = True
        for _hole_index in range(hole_count):
            placed = False
            for _ in range(50):
                max_hole_h = min(3, inner_r1 - inner_r0 + 1)
                max_hole_w = min(3, inner_c1 - inner_c0 + 1)
                hh = rng.randint(1, max_hole_h)
                ww = rng.randint(1, max_hole_w)
                rr = rng.randint(inner_r0, inner_r1 - hh + 1)
                cc = rng.randint(inner_c0, inner_c1 - ww + 1)
                rect = (rr, cc, rr + hh - 1, cc + ww - 1)
                if all(_separated(rect, prev, gap=1) for prev in holes):
                    holes.append(rect)
                    placed = True
                    break
            if not placed:
                ok = False
                break
        if not ok:
            continue

        for hr0, hc0, hr1, hc1 in holes:
            mask[hr0 : hr1 + 1, hc0 : hc1 + 1] = False

        if rng.random() < 0.45:
            bbox_r0, bbox_c0, bbox_r1, bbox_c1 = _bbox(np.argwhere(mask))
            side = rng.choice(["top", "bottom", "left", "right"])
            if side in {"top", "bottom"} and bbox_c1 - bbox_c0 >= 2:
                width = rng.randint(1, max(1, (bbox_c1 - bbox_c0 + 1) // 2))
                start = rng.randint(bbox_c0, bbox_c1 - width + 1)
                if side == "top":
                    depth = rng.randint(1, max(1, min(2, bbox_r1 - bbox_r0)))
                    mask[bbox_r0 : bbox_r0 + depth, start : start + width] = False
                else:
                    depth = rng.randint(1, max(1, min(2, bbox_r1 - bbox_r0)))
                    mask[bbox_r1 - depth + 1 : bbox_r1 + 1, start : start + width] = False
            elif side in {"left", "right"} and bbox_r1 - bbox_r0 >= 2:
                height = rng.randint(1, max(1, (bbox_r1 - bbox_r0 + 1) // 2))
                start = rng.randint(bbox_r0, bbox_r1 - height + 1)
                if side == "left":
                    depth = rng.randint(1, max(1, min(2, bbox_c1 - bbox_c0)))
                    mask[start : start + height, bbox_c0 : bbox_c0 + depth] = False
                else:
                    depth = rng.randint(1, max(1, min(2, bbox_c1 - bbox_c0)))
                    mask[start : start + height, bbox_c1 - depth + 1 : bbox_c1 + 1] = False

        if not np.any(mask):
            continue
        analysis = _analyze_regions(mask.astype(int), 0)
        if len(analysis) != 1:
            continue
        region = analysis[0]
        if int(region["hole_count"]) != hole_count:
            continue
        if int(region["area"]) < (4 + hole_count * 4):
            continue
        return mask

    raise RuntimeError(f"failed to create a region with {hole_count} holes")


def _place_masks(rng: random.Random, masks: list[np.ndarray], height: int, width: int) -> list[tuple[np.ndarray, int, int]] | None:
    occ = np.zeros((height, width), dtype=bool)
    placements: list[tuple[np.ndarray, int, int]] = []

    for mask in sorted(masks, key=lambda arr: (-(arr.shape[0] * arr.shape[1]), -int(arr.sum()))):
        mh, mw = mask.shape
        candidates: list[tuple[int, int]] = []
        for r0 in range(1, height - mh):
            for c0 in range(1, width - mw):
                rr0 = max(0, r0 - 1)
                rr1 = min(height, r0 + mh + 1)
                cc0 = max(0, c0 - 1)
                cc1 = min(width, c0 + mw + 1)
                if np.any(occ[rr0:rr1, cc0:cc1]):
                    continue
                candidates.append((r0, c0))
        if not candidates:
            return None
        r0, c0 = rng.choice(candidates)
        occ[r0 : r0 + mh, c0 : c0 + mw] |= mask
        placements.append((mask, r0, c0))

    return placements


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    group_counts = _sample_group_counts(rng)
    total_objects = sum(group_counts.values())
    compact = total_objects >= 8
    colors = rng.sample(list(range(1, 10)), total_objects)

    specs: list[dict] = []
    color_idx = 0
    for hole_count in [0, 1, 2]:
        used_areas: set[int] = set()
        for _ in range(group_counts[hole_count]):
            for _attempt in range(80):
                mask = _make_region_mask(rng, hole_count, compact=compact)
                area = int(mask.sum())
                if area in used_areas:
                    continue
                used_areas.add(area)
                specs.append(
                    {
                        "hole_count": hole_count,
                        "mask": mask,
                        "area": area,
                        "color": colors[color_idx],
                    }
                )
                color_idx += 1
                break
            else:
                return None

    max_h = max(mask["mask"].shape[0] for mask in specs)
    max_w = max(mask["mask"].shape[1] for mask in specs)
    min_h = max(20, max_h + 2, 14 + total_objects)
    min_w = max(20, max_w + 2, 14 + total_objects)
    height = rng.randint(min_h, 30)
    width = rng.randint(min_w, 30)

    placements = _place_masks(rng, [spec["mask"] for spec in specs], height, width)
    if placements is None:
        return None

    placed_grid = np.zeros((height, width), dtype=int)
    by_signature: dict[tuple[int, int, int], list[dict]] = defaultdict(list)
    for spec in specs:
        key = (spec["mask"].shape[0], spec["mask"].shape[1], spec["area"])
        by_signature[key].append(spec)

    for mask, r0, c0 in placements:
        key = (mask.shape[0], mask.shape[1], int(mask.sum()))
        candidates = by_signature[key]
        if not candidates:
            return None
        spec = candidates.pop()
        placed_grid[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]][mask] = int(spec["color"])

    regions = _analyze_regions(placed_grid, 0)
    if len(regions) != total_objects:
        return None

    hole_groups: dict[int, list[int]] = defaultdict(list)
    for region in regions:
        hole_groups[int(region["hole_count"])].append(int(region["area"]))

    if set(hole_groups) != {0, 1, 2}:
        return None
    if any(len(hole_groups[k]) != group_counts[k] for k in [0, 1, 2]):
        return None
    if any(len(set(hole_groups[k])) != len(hole_groups[k]) for k in [0, 1, 2]):
        return None

    out = main(placed_grid)
    if max(placed_grid.shape) > 30 or max(out.shape) > 30:
        return None

    return placed_grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
