from __future__ import annotations

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5
TEAL = 8

GEN_ATTEMPTS = 500
ROW_SIZE_RANGE = (6, 8)
COL_SIZE_RANGE = (6, 8)
RECT_MIN_SIDE = 4
MASK_ATTEMPTS = 800


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r0 in range(h):
        for c0 in range(w):
            if not mask[r0, c0] or seen[r0, c0]:
                continue
            stack = [(r0, c0)]
            seen[r0, c0] = True
            cells: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _spans_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(int(a0), int(b0)) <= min(int(a1), int(b1))


def _bboxes_share_row_or_col(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return _spans_overlap(ar0, ar1, br0, br1) or _spans_overlap(ac0, ac1, bc0, bc1)


def _merge_fragment_groups(components: list[np.ndarray]) -> list[dict]:
    groups = [{"coords": comp.copy(), "bbox": _bbox_coords(comp)} for comp in components]
    changed = True
    while changed:
        changed = False
        for i in range(len(groups)):
            if changed:
                break
            for j in range(i + 1, len(groups)):
                if _bboxes_share_row_or_col(groups[i]["bbox"], groups[j]["bbox"]):
                    merged = np.vstack([groups[i]["coords"], groups[j]["coords"]])
                    groups[i] = {"coords": merged, "bbox": _bbox_coords(merged)}
                    del groups[j]
                    changed = True
                    break
    return groups


def _intervals_from_line_mask(is_divider: np.ndarray) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    i = 0
    n = int(is_divider.shape[0])
    while i < n:
        if bool(is_divider[i]):
            i += 1
            continue
        j = i
        while j < n and not bool(is_divider[j]):
            j += 1
        intervals.append((i, j))
        i = j
    return intervals


def _extract_region_infos(grid: np.ndarray) -> list[dict]:
    row_intervals = _intervals_from_line_mask(np.all(grid == GREY, axis=1))
    col_intervals = _intervals_from_line_mask(np.all(grid == GREY, axis=0))
    infos: list[dict] = []
    for rr, (rs, re) in enumerate(row_intervals):
        for cc, (cs, ce) in enumerate(col_intervals):
            region = grid[rs:re, cs:ce]
            teal_mask = region == TEAL
            components = _find_components(teal_mask)
            assert components, "each region must contain teal fragments"
            groups = _merge_fragment_groups(components)
            assert len(groups) == 1, "each region must merge to one rectangle"
            lr0, lc0, lr1, lc1 = groups[0]["bbox"]
            infos.append(
                {
                    "region_index": (rr, cc),
                    "region_slice": (rs, re, cs, ce),
                    "count": int(np.count_nonzero(teal_mask)),
                    "components": components,
                    "component_count": len(components),
                    "bbox_local": (lr0, lc0, lr1, lc1),
                    "bbox_global": (rs + lr0, cs + lc0, rs + lr1, cs + lc1),
                    "area": int((lr1 - lr0 + 1) * (lc1 - lc0 + 1)),
                }
            )
    return infos


def _rank_fill_colors(counts: list[int]) -> list[int]:
    assert len(counts) == 9, "expected nine regions"
    assert len(set(counts)) == 9, "region counts must be distinct"
    fills = [BLACK] * len(counts)
    order = sorted(range(len(counts)), key=lambda idx: counts[idx], reverse=True)
    for rank, idx in enumerate(order, start=1):
        if rank <= 3:
            fills[idx] = BLUE
        elif rank <= 6:
            fills[idx] = RED
        else:
            fills[idx] = GREEN
    return fills


def main(grid: np.ndarray) -> np.ndarray:
    """Merge fragmented teal rectangles inside a 3x3 divider grid and fill each one by cross-region count rank.

    Input structure:
    - A black background is split into 3x3 rectangular regions by full gray divider rows and columns.
    - Each region contains one set of teal fragments belonging to an incomplete rectangle outline.
    - Fragments may be 4-disconnected, but fragments from the same rectangle align so their bounding boxes overlap in row span or column span.

    Rule:
    - Locate the full gray divider rows and columns to recover the nine regions.
    - Inside each region, find the 4-connected teal fragments and merge them transitively when their bounding boxes overlap in row span or column span.
    - Count the original teal pixels in every region, rank the nine regions from largest count to smallest, map ranks 1-3 to blue, 4-6 to red, and 7-9 to green, then fill every black cell inside that region's merged rectangle bbox with the mapped color while keeping existing teal pixels unchanged.

    Key insight:
    - The rectangle extent is not given by connectivity alone: fragments must first be merged by aligned bbox overlap, and only then can the cross-region teal-pixel ranking determine the fill color for each completed rectangle.
    """
    g = np.asarray(grid, dtype=int)
    infos = _extract_region_infos(g)
    fills = _rank_fill_colors([info["count"] for info in infos])
    out = g.copy()
    for info, fill in zip(infos, fills):
        r0, c0, r1, c1 = info["bbox_global"]
        patch = out[r0 : r1 + 1, c0 : c1 + 1]
        patch[patch == BLACK] = int(fill)
    return out


def _perimeter_cycle(h: int, w: int) -> list[tuple[int, int]]:
    cycle = [(0, c) for c in range(w)]
    cycle += [(r, w - 1) for r in range(1, h - 1)]
    cycle += [(h - 1, c) for c in range(w - 1, -1, -1)]
    cycle += [(r, 0) for r in range(h - 2, 0, -1)]
    return cycle


def _random_composition(total: int, parts: int) -> list[int] | None:
    if total < parts or parts <= 0:
        return None
    if parts == 1:
        return [int(total)]
    if total == parts:
        return [1] * parts
    cuts = np.sort(np.random.choice(np.arange(1, total), size=parts - 1, replace=False))
    vals = np.diff(np.concatenate(([0], cuts, [total])))
    return [int(v) for v in vals]


def _overlap_graph(components: list[np.ndarray]) -> list[set[int]]:
    bboxes = [_bbox_coords(comp) for comp in components]
    graph = [set() for _ in components]
    for i in range(len(components)):
        for j in range(i + 1, len(components)):
            if _bboxes_share_row_or_col(bboxes[i], bboxes[j]):
                graph[i].add(j)
                graph[j].add(i)
    return graph


def _is_chain_like(components: list[np.ndarray]) -> bool:
    if len(components) < 3:
        return False
    graph = _overlap_graph(components)
    seen = {0}
    stack = [0]
    while stack:
        cur = stack.pop()
        for nxt in graph[cur]:
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    edge_count = sum(len(nei) for nei in graph) // 2
    max_edges = len(components) * (len(components) - 1) // 2
    return len(seen) == len(components) and edge_count < max_edges


def _mask_valid(mask: np.ndarray, target_components: int, need_chain: bool) -> bool:
    h, w = mask.shape
    if not np.any(mask[0, :]) or not np.any(mask[h - 1, :]):
        return False
    if not np.any(mask[:, 0]) or not np.any(mask[:, w - 1]):
        return False
    components = _find_components(mask)
    if len(components) != target_components:
        return False
    groups = _merge_fragment_groups(components)
    if len(groups) != 1 or tuple(groups[0]["bbox"]) != (0, 0, h - 1, w - 1):
        return False
    if need_chain and not _is_chain_like(components):
        return False
    return True


def _corners_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, 0] = True
    mask[0, w - 1] = True
    mask[h - 1, 0] = True
    mask[h - 1, w - 1] = True
    return mask


def _mask_from_runs(h: int, w: int, count: int, target_components: int, need_chain: bool) -> np.ndarray | None:
    if count == 4 and target_components == 4:
        mask = _corners_mask(h, w)
        if _mask_valid(mask, target_components, need_chain):
            return mask
        return None

    cycle = _perimeter_cycle(h, w)
    n = len(cycle)
    if target_components < 1 or target_components > count or count > n:
        return None
    if target_components > 1 and n - count < target_components:
        return None

    for _ in range(MASK_ATTEMPTS):
        run_lengths = _random_composition(count, target_components)
        if run_lengths is None:
            continue
        if target_components == 1:
            gap_lengths = [n - count]
        else:
            gap_lengths = _random_composition(n - count, target_components)
            if gap_lengths is None:
                continue
        start = int(np.random.randint(n))
        mask = np.zeros((h, w), dtype=bool)
        idx = start
        for run_len, gap_len in zip(run_lengths, gap_lengths):
            for _ in range(run_len):
                r, c = cycle[idx % n]
                mask[r, c] = True
                idx += 1
            idx += gap_len
        if _mask_valid(mask, target_components, need_chain):
            return mask
    return None


def _choose_target_components(count: int, perimeter: int, max_count: int) -> tuple[int, bool]:
    if count == 4:
        return 4, True
    if count == max_count:
        return 1, False
    if count == 6:
        return 2, False
    if count == 8:
        return 3, False

    feasible = [m for m in (1, 2, 3, 4) if m <= count and (m == 1 or perimeter - count >= m)]
    if count <= 5:
        weights = {m: 1 + (m >= 3) * 3 + (m == 4) * 2 for m in feasible}
    elif count <= 8:
        weights = {m: 1 + (m == 2) * 3 + (m == 3) * 2 for m in feasible}
    else:
        weights = {m: 1 + (m == 1) * 4 + (m == 2) * 2 for m in feasible}
    choices = [m for m, wt in weights.items() for _ in range(wt)]
    return int(np.random.choice(np.array(choices, dtype=int))), False


def _region_spans(sizes: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for size in sizes:
        spans.append((start, start + size))
        start += size + 1
    return spans


def _build_expected_output(grid: np.ndarray, metas: list[dict]) -> np.ndarray:
    out = grid.copy()
    fills = _rank_fill_colors([meta["count"] for meta in metas])
    for meta, fill in zip(metas, fills):
        r0, c0, r1, c1 = meta["bbox_global"]
        patch = out[r0 : r1 + 1, c0 : c1 + 1]
        patch[patch == BLACK] = int(fill)
    return out


def _has_area_count_inversion(metas: list[dict]) -> bool:
    counts = [meta["count"] for meta in metas]
    areas = [meta["area"] for meta in metas]
    for i in range(len(metas)):
        for j in range(i + 1, len(metas)):
            if (counts[i] - counts[j]) * (areas[i] - areas[j]) < 0:
                return True
    return False


def _semantic_check(grid: np.ndarray, metas: list[dict]) -> bool:
    infos = _extract_region_infos(grid)
    if len(infos) != 9 or len(metas) != 9:
        return False
    for info, meta in zip(infos, metas):
        if info["region_index"] != meta["region_index"]:
            return False
        if info["count"] != meta["count"]:
            return False
        if tuple(info["bbox_global"]) != tuple(meta["bbox_global"]):
            return False

    component_counts = [info["component_count"] for info in infos]
    if not any(cc == 1 for cc in component_counts):
        return False
    if not any(cc == 2 for cc in component_counts):
        return False
    if not any(cc >= 3 for cc in component_counts):
        return False
    if not any(_is_chain_like(info["components"]) for info in infos if info["component_count"] >= 3):
        return False

    if not _has_area_count_inversion(metas):
        return False

    marginful = 0
    for meta in metas:
        rs, re, cs, ce = meta["region_slice"]
        r0, c0, r1, c1 = meta["bbox_global"]
        if (r0 > rs) or (c0 > cs) or (r1 < re - 1) or (c1 < ce - 1):
            marginful += 1
    if marginful < 5:
        return False

    expected = _build_expected_output(grid, metas)
    actual = main(grid)
    return np.array_equal(actual, expected)


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        row_sizes = [int(np.random.randint(ROW_SIZE_RANGE[0], ROW_SIZE_RANGE[1] + 1)) for _ in range(3)]
        col_sizes = [int(np.random.randint(COL_SIZE_RANGE[0], COL_SIZE_RANGE[1] + 1)) for _ in range(3)]
        row_spans = _region_spans(row_sizes)
        col_spans = _region_spans(col_sizes)
        height = row_spans[-1][1]
        width = col_spans[-1][1]

        metas: list[dict] = []
        for rr, (rs, re) in enumerate(row_spans):
            for cc, (cs, ce) in enumerate(col_spans):
                region_h = re - rs
                region_w = ce - cs
                rect_h = int(np.random.randint(RECT_MIN_SIDE, region_h + 1))
                rect_w = int(np.random.randint(RECT_MIN_SIDE, region_w + 1))
                top = int(np.random.randint(0, region_h - rect_h + 1))
                left = int(np.random.randint(0, region_w - rect_w + 1))
                metas.append(
                    {
                        "region_index": (rr, cc),
                        "region_slice": (rs, re, cs, ce),
                        "rect_local": (top, left, top + rect_h - 1, left + rect_w - 1),
                        "bbox_global": (rs + top, cs + left, rs + top + rect_h - 1, cs + left + rect_w - 1),
                        "area": int(rect_h * rect_w),
                        "perimeter": int(2 * (rect_h + rect_w) - 4),
                    }
                )

        min_perimeter = min(meta["perimeter"] for meta in metas)
        available = list(range(4, min_perimeter + 1))
        if len(available) < 9:
            continue

        max_count = max(available)
        special = {4, 6, 8, max_count}
        if not special.issubset(set(available)):
            continue
        pool = [v for v in available if v not in special]
        if len(pool) < 5:
            continue
        counts = list(special) + list(np.random.choice(np.array(pool, dtype=int), size=5, replace=False))
        counts = [int(v) for v in counts]
        np.random.shuffle(counts)
        for meta, count in zip(metas, counts):
            meta["count"] = int(count)

        ok = True
        for meta in metas:
            rs, re, cs, ce = meta["region_slice"]
            lr0, lc0, lr1, lc1 = meta["rect_local"]
            rect_h = lr1 - lr0 + 1
            rect_w = lc1 - lc0 + 1
            target_components, need_chain = _choose_target_components(meta["count"], meta["perimeter"], max_count)
            mask = _mask_from_runs(rect_h, rect_w, meta["count"], target_components, need_chain)
            if mask is None:
                ok = False
                break
            meta["mask"] = mask
            meta["target_components"] = target_components
            meta["need_chain"] = need_chain
            meta["component_count"] = len(_find_components(mask))
            meta["area"] = int(rect_h * rect_w)
            meta["bbox_global"] = (rs + lr0, cs + lc0, rs + lr1, cs + lc1)
        if not ok:
            continue

        grid = np.full((height, width), BLACK, dtype=int)
        divider_rows = [row_spans[0][1], row_spans[1][1]]
        divider_cols = [col_spans[0][1], col_spans[1][1]]
        for r in divider_rows:
            grid[r, :] = GREY
        for c in divider_cols:
            grid[:, c] = GREY

        for meta in metas:
            r0, c0, r1, c1 = meta["bbox_global"]
            patch = grid[r0 : r1 + 1, c0 : c1 + 1]
            patch[meta["mask"]] = TEAL

        if _semantic_check(grid, metas):
            return grid

    raise RuntimeError("failed to generate a valid ARC sample")
