from __future__ import annotations

import random

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]

BG = 0
FILL_BY_BENDS = {1: 1, 2: 2, 3: 4}
GREEN = 3
ORANGE = 7
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _neighbors4(r: int, c: int) -> list[Coord]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _bbox(cells: set[Coord]) -> BBox:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _boxes_overlap(a: BBox, b: BBox, margin: int = 0) -> bool:
    return not (
        a[2] + margin < b[0]
        or b[2] + margin < a[0]
        or a[3] + margin < b[1]
        or b[3] + margin < a[1]
    )


def _components(mask: np.ndarray) -> list[set[Coord]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[set[Coord]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: set[Coord] = set()
            while stack:
                rr, cc = stack.pop()
                comp.add((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _count_bends(cells: set[Coord]) -> int:
    r0, c0, r1, c1 = _bbox(cells)
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in cells:
        patch[r - r0, c - c0] = True

    bends = 0
    for r in range(patch.shape[0] - 1):
        for c in range(patch.shape[1] - 1):
            if int(np.sum(patch[r : r + 2, c : c + 2])) == 3:
                bends += 1
    return bends


def _segment_cells(base_r: int, base_c: int, length: int, shift: int = 0) -> list[Coord]:
    return [(base_r + shift + t, base_c + shift - t) for t in range(length)]


def _region_cells_from_geometry(base_r: int, base_c: int, length: int, gap: int) -> set[Coord]:
    s1 = base_r + base_c
    s2 = s1 + 2 * gap
    d0 = base_r - base_c
    d1 = d0 + 2 * (length - 1)
    cells: set[Coord] = set()
    for r in range(base_r, base_r + gap + length):
        for c in range(base_c - length + 1, base_c + gap + 1):
            s = r + c
            d = r - c
            if s1 < s < s2 and d0 < d < d1:
                cells.add((r, c))
    return cells


def _safe_region_cells(region: set[Coord], orange: set[Coord]) -> set[Coord]:
    safe: set[Coord] = set()
    for r, c in region:
        if all((nr, nc) not in orange for nr, nc in _neighbors4(r, c)):
            safe.add((r, c))
    return safe


def _find_orange_segments(grid: np.ndarray) -> list[dict]:
    orange_cells = [(int(r), int(c)) for r, c in np.argwhere(grid == ORANGE)]
    by_sum: dict[int, list[Coord]] = {}
    for r, c in orange_cells:
        by_sum.setdefault(r + c, []).append((r, c))

    segments: list[dict] = []
    for s, pts in by_sum.items():
        pts.sort()
        run: list[Coord] = []
        for pt in pts:
            if not run or (pt[0] == run[-1][0] + 1 and pt[1] == run[-1][1] - 1):
                run.append(pt)
            else:
                if len(run) >= 2:
                    ds = [r - c for r, c in run]
                    segments.append(
                        {
                            "cells": tuple(run),
                            "s": int(s),
                            "length": len(run),
                            "d_min": int(min(ds)),
                            "d_max": int(max(ds)),
                        }
                    )
                run = [pt]
        if len(run) >= 2:
            ds = [r - c for r, c in run]
            segments.append(
                {
                    "cells": tuple(run),
                    "s": int(s),
                    "length": len(run),
                    "d_min": int(min(ds)),
                    "d_max": int(max(ds)),
                }
            )
    return segments


def _pair_segments(segments: list[dict]) -> list[tuple[dict, dict]]:
    groups: dict[tuple[int, int, int], list[dict]] = {}
    for seg in segments:
        key = (seg["d_min"], seg["d_max"], seg["length"])
        groups.setdefault(key, []).append(seg)

    pairs: list[tuple[dict, dict]] = []
    for group in groups.values():
        group.sort(key=lambda seg: seg["s"])
        if len(group) % 2 != 0:
            raise ValueError("orange segments could not be paired evenly")
        for i in range(0, len(group), 2):
            pairs.append((group[i], group[i + 1]))
    return pairs


def _region_from_pair(seg_a: dict, seg_b: dict) -> set[Coord]:
    if seg_a["s"] > seg_b["s"]:
        seg_a, seg_b = seg_b, seg_a

    cells = list(seg_a["cells"]) + list(seg_b["cells"])
    r0 = min(r for r, _ in cells)
    r1 = max(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    c1 = max(c for _, c in cells)

    region: set[Coord] = set()
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            s = r + c
            d = r - c
            if seg_a["s"] < s < seg_b["s"] and seg_a["d_min"] < d < seg_a["d_max"]:
                region.add((r, c))
    return region


def _search_shape(safe_cells: set[Coord], target_bends: int) -> set[Coord] | None:
    if len(safe_cells) < 4:
        return None

    safe_list = list(safe_cells)
    random.shuffle(safe_list)
    safe_set = set(safe_list)
    nbrs = {
        pt: [q for q in _neighbors4(pt[0], pt[1]) if q in safe_set]
        for pt in safe_list
    }

    lengths = list(range(max(4, target_bends + 2), 9))
    random.shuffle(lengths)

    def dfs(path: list[Coord], used: set[Coord], prev_dir: Coord | None, turns: int, target_len: int) -> list[Coord] | None:
        if len(path) == target_len:
            return path[:] if _count_bends(used) == target_bends else None
        if turns > target_bends:
            return None
        if turns + (target_len - len(path)) < target_bends:
            return None

        last = path[-1]
        candidates = nbrs[last][:]
        random.shuffle(candidates)
        for nxt in candidates:
            if nxt in used:
                continue
            attached = sum((adj in used) for adj in _neighbors4(nxt[0], nxt[1]))
            if attached != 1:
                continue
            step = (nxt[0] - last[0], nxt[1] - last[1])
            new_turns = turns + int(prev_dir is not None and step != prev_dir)
            if new_turns > target_bends:
                continue
            path.append(nxt)
            used.add(nxt)
            ans = dfs(path, used, step, new_turns, target_len)
            if ans is not None:
                return ans
            used.remove(nxt)
            path.pop()
        return None

    for target_len in lengths:
        starts = safe_list[:]
        random.shuffle(starts)
        for start in starts[: min(50, len(starts))]:
            ans = dfs([start], {start}, None, 0, target_len)
            if ans is not None:
                return set(ans)
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each diagonal-bounded parallelogram with a color determined by the bend count of its enclosed green shape.

    Input structure:
    - The grid has a black background, several pairs of orange anti-diagonal segments, and one green 4-connected bent shape inside each paired region.
    - Each orange segment is a short run on a constant r+c anti-diagonal, and paired segments share the same r-c span.
    - Green shapes do not overlap the orange boundaries.

    Rule:
    - Detect all orange anti-diagonal segments and pair segments that share the same r-c range, ordered by their r+c value.
    - For each paired segment set, recover the interior parallelogram region between the two orange boundaries.
    - Find the unique green connected component inside that region and count its L-bends by counting 2x2 windows containing exactly three green cells.
    - Fill the region's background cells with blue for 1 bend, red for 2 bends, and yellow for 3 bends, while preserving the orange boundaries and green shape.

    Key insight:
    - The filled area is determined geometrically by the paired diagonal boundaries, but the fill color is determined topologically by the bend count of the separate green object inside that area.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    green_components = _components(g == GREEN)
    orange_segments = _find_orange_segments(g)
    for seg_a, seg_b in _pair_segments(orange_segments):
        region = _region_from_pair(seg_a, seg_b)
        enclosed = [comp for comp in green_components if comp & region]
        if len(enclosed) != 1:
            raise ValueError("each paired region must contain exactly one green shape")
        bends = _count_bends(enclosed[0])
        fill_color = FILL_BY_BENDS[bends]
        for r, c in region:
            if out[r, c] == BG:
                out[r, c] = fill_color

    return out


def generate_input() -> np.ndarray:
    for _ in range(400):
        n_regions = random.choice([3, 4])
        height = random.randint(max(18, 6 * n_regions), 28)
        width = random.randint(max(18, 6 * n_regions), 28)
        grid = np.zeros((height, width), dtype=int)

        targets = [1, 2, 3]
        if n_regions == 4:
            targets.append(random.randint(1, 3))
        random.shuffle(targets)

        placed: list[dict] = []
        used_boxes: list[BBox] = []
        used_ranges: set[tuple[int, int]] = set()
        ok = True

        for target_bends in targets:
            made = False
            for _ in range(250):
                length = random.choice([3, 3, 4, 4])
                gap = random.randint(5, 8)
                max_r = height - (gap + length)
                min_c = length - 1
                max_c = width - gap - 1
                if max_r < 0 or min_c > max_c:
                    continue

                base_r = random.randint(0, max_r)
                base_c = random.randint(min_c, max_c)
                d0 = base_r - base_c
                d1 = d0 + 2 * (length - 1)
                if (d0, d1) in used_ranges:
                    continue

                bbox = (base_r, base_c - length + 1, base_r + gap + length - 1, base_c + gap)
                if any(_boxes_overlap(bbox, other, margin=1) for other in used_boxes):
                    continue

                seg1 = set(_segment_cells(base_r, base_c, length, shift=0))
                seg2 = set(_segment_cells(base_r, base_c, length, shift=gap))
                orange = seg1 | seg2
                region = _region_cells_from_geometry(base_r, base_c, length, gap)
                safe = _safe_region_cells(region, orange)
                shape = _search_shape(safe, target_bends)
                if shape is None:
                    continue

                placed.append(
                    {
                        "orange": orange,
                        "shape": shape,
                        "region": region,
                        "bends": target_bends,
                    }
                )
                used_boxes.append(bbox)
                used_ranges.add((d0, d1))
                made = True
                break

            if not made:
                ok = False
                break

        if not ok:
            continue

        for item in placed:
            for r, c in item["orange"]:
                grid[r, c] = ORANGE
            for r, c in item["shape"]:
                grid[r, c] = GREEN

        if len(_components(grid == GREEN)) != n_regions:
            continue

        out = main(grid)
        valid = True
        for item in placed:
            fill_color = FILL_BY_BENDS[item["bends"]]
            for cell in item["orange"]:
                if out[cell] != ORANGE:
                    valid = False
                    break
            if not valid:
                break
            for cell in item["shape"]:
                if out[cell] != GREEN:
                    valid = False
                    break
            if not valid:
                break
            for cell in item["region"]:
                if cell not in item["shape"] and out[cell] != fill_color:
                    valid = False
                    break
            if not valid:
                break

        if valid:
            return grid

    raise RuntimeError("failed to generate a valid ARC instance")
