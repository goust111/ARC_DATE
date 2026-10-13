from __future__ import annotations

import itertools
from collections import Counter

import numpy as np


ALL_COLORS = [1, 2, 3, 4, 5, 6, 7, 8, 9]
BG = 0
MAX_GEN_ATTEMPTS = 300


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(sorted(comp))
    return out


def _segments(length: int, dividers: list[int]) -> list[tuple[int, int]]:
    divs = sorted(int(x) for x in dividers)
    segs: list[tuple[int, int]] = []
    start = 0
    for d in divs:
        if start <= d - 1:
            segs.append((start, d - 1))
        start = d + 1
    if start <= length - 1:
        segs.append((start, length - 1))
    return segs


def _band_lookup(length: int, segments: list[tuple[int, int]]) -> list[int]:
    band = [-1] * length
    for idx, (a, b) in enumerate(segments):
        for x in range(a, b + 1):
            band[x] = idx
    return band


def _find_divider(grid: np.ndarray, bg: int) -> tuple[int, list[int], list[int]]:
    h, w = grid.shape
    row_colors = {
        int(grid[r, 0])
        for r in range(h)
        if int(grid[r, 0]) != bg and np.all(grid[r, :] == grid[r, 0])
    }
    col_colors = {
        int(grid[0, c])
        for c in range(w)
        if int(grid[0, c]) != bg and np.all(grid[:, c] == grid[0, c])
    }
    colors = row_colors & col_colors
    if len(colors) != 1:
        raise AssertionError("expected exactly one divider color")
    divider = next(iter(colors))
    div_rows = [r for r in range(h) if np.all(grid[r, :] == divider)]
    div_cols = [c for c in range(w) if np.all(grid[:, c] == divider)]
    if not div_rows or not div_cols:
        raise AssertionError("expected full divider rows and columns")
    return divider, div_rows, div_cols


def _is_singleton_pixel(grid: np.ndarray, r: int, c: int) -> bool:
    val = int(grid[r, c])
    for nr, nc in _neighbors4(r, c):
        if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and int(grid[nr, nc]) == val:
            return False
    return True


def _build_regions(
    grid: np.ndarray, div_rows: list[int], div_cols: list[int]
) -> tuple[dict[tuple[int, int], dict], list[int], list[int]]:
    row_segs = _segments(grid.shape[0], div_rows)
    col_segs = _segments(grid.shape[1], div_cols)
    row_band = _band_lookup(grid.shape[0], row_segs)
    col_band = _band_lookup(grid.shape[1], col_segs)
    regions: dict[tuple[int, int], dict] = {}
    for ri, (r0, r1) in enumerate(row_segs):
        for ci, (c0, c1) in enumerate(col_segs):
            regions[(ri, ci)] = {
                "bbox": (r0, c0, r1, c1),
                "h": r1 - r0 + 1,
                "w": c1 - c0 + 1,
            }
    return regions, row_band, col_band


def _region_local(coord: tuple[int, int], bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r, c = coord
    r0, c0, _r1, _c1 = bbox
    return r - r0, c - c0


def _region_global(coord: tuple[int, int], bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r, c = coord
    r0, c0, _r1, _c1 = bbox
    return r0 + r, c0 + c


def _find_pairs(
    grid: np.ndarray,
    bg: int,
    divider: int,
    div_rows: list[int],
    div_cols: list[int],
    row_band: list[int],
    col_band: list[int],
) -> list[dict]:
    h, w = grid.shape
    div_row_set = set(div_rows)
    div_col_set = set(div_cols)
    pairs: list[dict] = []
    used_regions: set[tuple[int, int]] = set()

    for c in div_cols:
        if c == 0 or c == w - 1:
            continue
        for r in range(h):
            if r in div_row_set:
                continue
            lv = int(grid[r, c - 1])
            rv = int(grid[r, c + 1])
            if lv in (bg, divider) or rv in (bg, divider) or lv == rv:
                continue
            if not _is_singleton_pixel(grid, r, c - 1) or not _is_singleton_pixel(grid, r, c + 1):
                continue
            a = (row_band[r], col_band[c - 1])
            b = (row_band[r], col_band[c + 1])
            if a in used_regions or b in used_regions:
                raise AssertionError("region participates in multiple pairs")
            pairs.append(
                {
                    "a": a,
                    "b": b,
                    "a_msg": (r, c - 1),
                    "b_msg": (r, c + 1),
                    "orientation": "vertical",
                }
            )
            used_regions.add(a)
            used_regions.add(b)

    for r in div_rows:
        if r == 0 or r == h - 1:
            continue
        for c in range(w):
            if c in div_col_set:
                continue
            uv = int(grid[r - 1, c])
            dv = int(grid[r + 1, c])
            if uv in (bg, divider) or dv in (bg, divider) or uv == dv:
                continue
            if not _is_singleton_pixel(grid, r - 1, c) or not _is_singleton_pixel(grid, r + 1, c):
                continue
            a = (row_band[r - 1], col_band[c])
            b = (row_band[r + 1], col_band[c])
            if a in used_regions or b in used_regions:
                raise AssertionError("region participates in multiple pairs")
            pairs.append(
                {
                    "a": a,
                    "b": b,
                    "a_msg": (r - 1, c),
                    "b_msg": (r + 1, c),
                    "orientation": "horizontal",
                }
            )
            used_regions.add(a)
            used_regions.add(b)

    if not pairs:
        raise AssertionError("expected at least one messenger pair")
    return pairs


def _extract_sprite(
    grid: np.ndarray,
    region_bbox: tuple[int, int, int, int],
    bg: int,
    divider: int,
    messenger: tuple[int, int],
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = region_bbox
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    mask = (patch != bg) & (patch != divider)
    mr, mc = _region_local(messenger, region_bbox)
    mask[mr, mc] = False
    comps = _components(mask)
    big = [comp for comp in comps if len(comp) >= 3]
    if len(big) != 1:
        raise AssertionError("expected exactly one non-distractor sprite in paired region")
    return big[0]


def _paint_local(out: np.ndarray, region_bbox: tuple[int, int, int, int], coords: list[tuple[int, int]], color: int) -> None:
    r0, c0, r1, c1 = region_bbox
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    for rr, cc in coords:
        if not (0 <= rr < h and 0 <= cc < w):
            raise AssertionError("local sprite escaped target region")
        out[r0 + rr, c0 + cc] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap each paired region's interior sprite across the divider and recolor it by the target-side messenger cell.

    Input structure:
    - Black background cells are partitioned into rectangular regions by one divider color that fills entire rows and columns.
    - Some adjacent region pairs have a facing cross-divider messenger pair: two singleton pixels of different colors, one on each side of the same divider cell.
    - Every paired region contains exactly one larger sprite plus optional interior distractor dots or 2-pixel pairs.

    Rule:
    - For every valid messenger pair, identify the two adjacent regions that own those messenger pixels.
    - In each paired region, keep only the unique non-messenger 4-connected component of area at least 3 as the real sprite; size-1 and size-2 components are distractors.
    - Remove the two real sprites, swap their region-relative occupied cells, and repaint each moved sprite to the color of the messenger pixel already on the destination side.
    - Keep divider rows/columns, messenger pixels, unpaired regions, and all distractors unchanged.

    Key insight:
    - The only actionable signals are singleton pixels that straddle a divider; they encode both which two regions swap and which color the incoming sprite must take on each side.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    divider, div_rows, div_cols = _find_divider(g, bg)
    regions, row_band, col_band = _build_regions(g, div_rows, div_cols)
    pairs = _find_pairs(g, bg, divider, div_rows, div_cols, row_band, col_band)

    out = g.copy()
    sprite_cache: dict[tuple[int, int], list[tuple[int, int]]] = {}

    for pair in pairs:
        a = pair["a"]
        b = pair["b"]
        bbox_a = regions[a]["bbox"]
        bbox_b = regions[b]["bbox"]
        sprite_a = sprite_cache.setdefault(a, _extract_sprite(g, bbox_a, bg, divider, pair["a_msg"]))
        sprite_b = sprite_cache.setdefault(b, _extract_sprite(g, bbox_b, bg, divider, pair["b_msg"]))

        for rr, cc in sprite_a:
            gr, gc = _region_global((rr, cc), bbox_a)
            out[gr, gc] = bg
        for rr, cc in sprite_b:
            gr, gc = _region_global((rr, cc), bbox_b)
            out[gr, gc] = bg

        color_a = int(g[pair["a_msg"]])
        color_b = int(g[pair["b_msg"]])
        for rr, cc in sprite_b:
            gr, gc = _region_global((rr, cc), bbox_a)
            if out[gr, gc] != bg:
                raise AssertionError("incoming sprite overlaps preserved content")
            out[gr, gc] = color_a
        for rr, cc in sprite_a:
            gr, gc = _region_global((rr, cc), bbox_b)
            if out[gr, gc] != bg:
                raise AssertionError("incoming sprite overlaps preserved content")
            out[gr, gc] = color_b

    return out


def _halo(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    out.add((rr, cc))
    return out


def _random_connected_shape(
    rng: np.random.Generator, max_h: int, max_w: int
) -> list[tuple[int, int]] | None:
    if max_h < 2 or max_w < 2:
        return None
    for _ in range(60):
        box_h = int(rng.integers(2, min(4, max_h) + 1))
        box_w = int(rng.integers(2, min(4, max_w) + 1))
        area = int(rng.integers(3, min(6, box_h * box_w) + 1))
        start = (int(rng.integers(0, box_h)), int(rng.integers(0, box_w)))
        cells = {start}
        while len(cells) < area:
            frontier: set[tuple[int, int]] = set()
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < box_h and 0 <= nc < box_w and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            choice = list(frontier)[int(rng.integers(0, len(frontier)))]
            cells.add(choice)
        if len(cells) != area:
            continue
        coords = sorted(cells)
        r0, c0, r1, c1 = _bbox(coords)
        norm = sorted((r - r0, c - c0) for r, c in coords)
        rows = {r for r, _ in norm}
        cols = {c for _, c in norm}
        if len(rows) >= 2 and len(cols) >= 2:
            return norm
    return None


def _translate_shape(coords: list[tuple[int, int]], dr: int, dc: int) -> list[tuple[int, int]]:
    return sorted((r + dr, c + dc) for r, c in coords)


def _shape_size(coords: list[tuple[int, int]]) -> tuple[int, int]:
    r0, c0, r1, c1 = _bbox(coords)
    return r1 - r0 + 1, c1 - c0 + 1


def _choose_local_shape(
    rng: np.random.Generator,
    common_h: int,
    common_w: int,
) -> list[tuple[int, int]] | None:
    inner_h = common_h - 2
    inner_w = common_w - 2
    for _ in range(60):
        shape = _random_connected_shape(rng, inner_h, inner_w)
        if shape is None:
            return None
        shape_h, shape_w = _shape_size(shape)
        max_r = common_h - shape_h - 1
        max_c = common_w - shape_w - 1
        if max_r < 1 or max_c < 1:
            continue
        off_r = int(rng.integers(1, max_r + 1))
        off_c = int(rng.integers(1, max_c + 1))
        return _translate_shape(shape, off_r, off_c)
    return None


def _pick_color(
    rng: np.random.Generator,
    divider: int,
    forbid: set[int] | None = None,
) -> int:
    bad = {divider, BG}
    if forbid is not None:
        bad |= {int(x) for x in forbid if x is not None}
    choices = [c for c in ALL_COLORS if c not in bad]
    return int(choices[int(rng.integers(0, len(choices)))])


def _sample_layout(rng: np.random.Generator) -> tuple[list[int], list[int]]:
    while True:
        row_count = int(rng.integers(2, 4))
        col_count = int(rng.integers(2, 4))
        heights = [int(rng.integers(5, 9)) for _ in range(row_count)]
        widths = [int(rng.integers(5, 9)) for _ in range(col_count)]
        total_h = sum(heights) + row_count - 1
        total_w = sum(widths) + col_count - 1
        if total_h <= 30 and total_w <= 30:
            return heights, widths


def _layout_to_regions(
    heights: list[int], widths: list[int]
) -> tuple[dict[tuple[int, int], dict], list[int], list[int], tuple[int, int]]:
    row_starts: list[int] = []
    divider_rows: list[int] = []
    pos = 0
    for i, h in enumerate(heights):
        row_starts.append(pos)
        pos += h
        if i != len(heights) - 1:
            divider_rows.append(pos)
            pos += 1
    total_h = pos

    col_starts: list[int] = []
    divider_cols: list[int] = []
    pos = 0
    for i, w in enumerate(widths):
        col_starts.append(pos)
        pos += w
        if i != len(widths) - 1:
            divider_cols.append(pos)
            pos += 1
    total_w = pos

    regions: dict[tuple[int, int], dict] = {}
    for ri, (r0, h) in enumerate(zip(row_starts, heights)):
        for ci, (c0, w) in enumerate(zip(col_starts, widths)):
            regions[(ri, ci)] = {"bbox": (r0, c0, r0 + h - 1, c0 + w - 1), "h": h, "w": w}
    return regions, divider_rows, divider_cols, (total_h, total_w)


def _candidate_edges(regions: dict[tuple[int, int], dict]) -> list[dict]:
    row_ids = sorted({r for r, _ in regions})
    col_ids = sorted({c for _, c in regions})
    edges: list[dict] = []
    for ri in row_ids:
        for ci in col_ids:
            if (ri, ci + 1) in regions:
                edges.append({"a": (ri, ci), "b": (ri, ci + 1), "orientation": "vertical"})
            if (ri + 1, ci) in regions:
                edges.append({"a": (ri, ci), "b": (ri + 1, ci), "orientation": "horizontal"})
    return edges


def _choose_pairs(
    rng: np.random.Generator, regions: dict[tuple[int, int], dict], edges: list[dict]
) -> list[dict]:
    options_by_size: dict[int, list[list[dict]]] = {}
    max_pairs = min(3, len(edges), len(regions) // 2)
    for size in range(1, max_pairs + 1):
        combos: list[list[dict]] = []
        for idxs in itertools.combinations(range(len(edges)), size):
            used: set[tuple[int, int]] = set()
            combo: list[dict] = []
            ok = True
            for idx in idxs:
                edge = edges[idx]
                if edge["a"] in used or edge["b"] in used:
                    ok = False
                    break
                used.add(edge["a"])
                used.add(edge["b"])
                combo.append(edge)
            if ok:
                combos.append(combo)
        if combos:
            options_by_size[size] = combos
    if not options_by_size:
        raise AssertionError("no valid pairings")

    sizes = sorted(options_by_size)
    weights = []
    for size in sizes:
        if size == 1:
            weights.append(4)
        elif size == 2:
            weights.append(3)
        else:
            weights.append(1)
    desired = sizes[int(rng.choice(len(sizes), p=np.array(weights, dtype=float) / np.sum(weights)))]
    combos = options_by_size[desired]
    if desired >= 2:
        mixed = [combo for combo in combos if len({edge["orientation"] for edge in combo}) >= 2]
        if mixed and rng.random() < 0.7:
            combos = mixed
    return [dict(edge) for edge in combos[int(rng.integers(0, len(combos)))]]


def _pair_messenger_cells(edge: dict, regions: dict[tuple[int, int], dict], rng: np.random.Generator) -> None:
    reg_a = regions[edge["a"]]
    reg_b = regions[edge["b"]]
    if edge["orientation"] == "vertical":
        row = int(rng.integers(1, reg_a["h"] - 1))
        edge["a_msg_local"] = (row, reg_a["w"] - 1)
        edge["b_msg_local"] = (row, 0)
    else:
        col = int(rng.integers(1, reg_a["w"] - 1))
        edge["a_msg_local"] = (reg_a["h"] - 1, col)
        edge["b_msg_local"] = (0, col)


def _plan_pair_sprites(
    rng: np.random.Generator, edge: dict, regions: dict[tuple[int, int], dict]
) -> bool:
    reg_a = regions[edge["a"]]
    reg_b = regions[edge["b"]]
    common_h = min(reg_a["h"], reg_b["h"])
    common_w = min(reg_a["w"], reg_b["w"])
    for _ in range(80):
        sprite_a = _choose_local_shape(rng, common_h, common_w)
        sprite_b = _choose_local_shape(rng, common_h, common_w)
        if sprite_a is None or sprite_b is None:
            return False
        if set(sprite_a) == set(sprite_b):
            continue
        edge["sprite_a_local"] = sprite_a
        edge["sprite_b_local"] = sprite_b
        return True
    return False


def _place_distractor(
    rng: np.random.Generator,
    h: int,
    w: int,
    blocked: set[tuple[int, int]],
) -> set[tuple[int, int]] | None:
    halo = _halo(blocked, h, w)
    for _ in range(80):
        if rng.random() < 0.5:
            cell = (int(rng.integers(1, h - 1)), int(rng.integers(1, w - 1)))
            cand = {cell}
        else:
            r = int(rng.integers(1, h - 1))
            c = int(rng.integers(1, w - 1))
            dr, dc = ((1, 0) if rng.random() < 0.5 else (0, 1))
            cell2 = (r + dr, c + dc)
            if not (1 <= cell2[0] < h - 1 and 1 <= cell2[1] < w - 1):
                continue
            cand = {(r, c), cell2}
        if cand & halo:
            continue
        return cand
    return None


def _build_case(rng: np.random.Generator) -> dict | None:
    heights, widths = _sample_layout(rng)
    regions, divider_rows, divider_cols, shape = _layout_to_regions(heights, widths)
    divider = int(ALL_COLORS[int(rng.integers(0, len(ALL_COLORS)))])
    edges = _candidate_edges(regions)
    pairs = _choose_pairs(rng, regions, edges)

    region_plans: dict[tuple[int, int], dict] = {
        rid: {
            "messenger_local": None,
            "messenger_color": None,
            "sprite_local": [],
            "sprite_color": None,
            "incoming_local": [],
            "distractors": [],
        }
        for rid in regions
    }

    for edge in pairs:
        _pair_messenger_cells(edge, regions, rng)
        msg_a = _pick_color(rng, divider)
        msg_b = _pick_color(rng, divider, {msg_a})
        edge["msg_a_color"] = msg_a
        edge["msg_b_color"] = msg_b
        if not _plan_pair_sprites(rng, edge, regions):
            return None
        src_a = _pick_color(rng, divider, {msg_a, msg_b})
        src_b = _pick_color(rng, divider, {msg_a, msg_b, src_a})
        edge["src_a_color"] = src_a
        edge["src_b_color"] = src_b
        region_plans[edge["a"]]["messenger_local"] = edge["a_msg_local"]
        region_plans[edge["a"]]["messenger_color"] = msg_a
        region_plans[edge["a"]]["sprite_local"] = edge["sprite_a_local"]
        region_plans[edge["a"]]["sprite_color"] = src_a
        region_plans[edge["a"]]["incoming_local"] = edge["sprite_b_local"]
        region_plans[edge["b"]]["messenger_local"] = edge["b_msg_local"]
        region_plans[edge["b"]]["messenger_color"] = msg_b
        region_plans[edge["b"]]["sprite_local"] = edge["sprite_b_local"]
        region_plans[edge["b"]]["sprite_color"] = src_b
        region_plans[edge["b"]]["incoming_local"] = edge["sprite_a_local"]

    unpaired = [rid for rid, plan in region_plans.items() if plan["messenger_local"] is None]
    force_distractor_region = None
    if unpaired and rng.random() < 0.75:
        force_distractor_region = unpaired[int(rng.integers(0, len(unpaired)))]
    elif rng.random() < 0.5:
        force_distractor_region = list(region_plans)[int(rng.integers(0, len(region_plans)))]

    placed_any_distractor = False
    for rid, plan in region_plans.items():
        h = regions[rid]["h"]
        w = regions[rid]["w"]
        reserved = set(plan["sprite_local"]) | set(plan["incoming_local"])
        if plan["messenger_local"] is not None:
            reserved.add(plan["messenger_local"])
        blocked = set(reserved)
        want = 0
        if rid == force_distractor_region:
            want = 1
        if plan["messenger_local"] is None:
            if rng.random() < 0.8:
                want = max(want, int(rng.integers(1, 3)))
        else:
            if rng.random() < 0.6:
                want = max(want, 1)
                if rng.random() < 0.25:
                    want = 2
        for _ in range(want):
            distractor = _place_distractor(rng, h, w, blocked)
            if distractor is None:
                continue
            plan["distractors"].append(
                {
                    "cells": sorted(distractor),
                    "color": _pick_color(
                        rng,
                        divider,
                        {plan["sprite_color"], plan["messenger_color"]} if plan["messenger_color"] is not None else {plan["sprite_color"]},
                    ),
                }
            )
            blocked |= distractor
            placed_any_distractor = True

    if not placed_any_distractor:
        return None

    grid = np.zeros(shape, dtype=int)
    for r in divider_rows:
        grid[r, :] = divider
    for c in divider_cols:
        grid[:, c] = divider

    for rid, plan in region_plans.items():
        bbox = regions[rid]["bbox"]
        if plan["messenger_local"] is not None:
            gr, gc = _region_global(plan["messenger_local"], bbox)
            grid[gr, gc] = int(plan["messenger_color"])
        for rr, cc in plan["sprite_local"]:
            gr, gc = _region_global((rr, cc), bbox)
            grid[gr, gc] = int(plan["sprite_color"])
        for distractor in plan["distractors"]:
            for rr, cc in distractor["cells"]:
                gr, gc = _region_global((rr, cc), bbox)
                grid[gr, gc] = int(distractor["color"])

    expected = grid.copy()
    for edge in pairs:
        bbox_a = regions[edge["a"]]["bbox"]
        bbox_b = regions[edge["b"]]["bbox"]
        for rr, cc in edge["sprite_a_local"]:
            gr, gc = _region_global((rr, cc), bbox_a)
            expected[gr, gc] = BG
        for rr, cc in edge["sprite_b_local"]:
            gr, gc = _region_global((rr, cc), bbox_b)
            expected[gr, gc] = BG
        for rr, cc in edge["sprite_b_local"]:
            gr, gc = _region_global((rr, cc), bbox_a)
            expected[gr, gc] = int(edge["msg_a_color"])
        for rr, cc in edge["sprite_a_local"]:
            gr, gc = _region_global((rr, cc), bbox_b)
            expected[gr, gc] = int(edge["msg_b_color"])

    if np.array_equal(grid, expected):
        return None

    case = {
        "input": grid,
        "expected": expected,
        "regions": regions,
        "pairs": pairs,
        "plans": region_plans,
        "divider": divider,
        "divider_rows": divider_rows,
        "divider_cols": divider_cols,
    }
    if not _validate_case(case):
        return None
    if not np.array_equal(main(case["input"]), case["expected"]):
        return None
    return case


def _validate_case(case: dict) -> bool:
    grid = case["input"]
    expected = case["expected"]
    divider = case["divider"]
    if grid.ndim != 2 or expected.ndim != 2:
        return False
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return False

    paired_regions = set()
    for edge in case["pairs"]:
        paired_regions.add(edge["a"])
        paired_regions.add(edge["b"])

    for rid, region in case["regions"].items():
        bbox = region["bbox"]
        r0, c0, r1, c1 = bbox
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        local_fg = patch != BG
        if case["plans"][rid]["messenger_local"] is not None:
            mr, mc = case["plans"][rid]["messenger_local"]
            for rr in range(region["h"]):
                for cc in range(region["w"]):
                    touching_border = rr in (0, region["h"] - 1) or cc in (0, region["w"] - 1)
                    if touching_border and local_fg[rr, cc] and (rr, cc) != (mr, mc):
                        return False
            local_fg[mr, mc] = False
        else:
            for rr in range(region["h"]):
                for cc in range(region["w"]):
                    touching_border = rr in (0, region["h"] - 1) or cc in (0, region["w"] - 1)
                    if touching_border and local_fg[rr, cc]:
                        return False

        comps = _components(local_fg)
        big = [comp for comp in comps if len(comp) >= 3]
        if rid in paired_regions:
            if len(big) != 1:
                return False
        else:
            if big:
                return False
        if any(len(comp) > 2 for comp in comps if comp not in big):
            return False

    return len(case["pairs"]) >= 1 and not np.array_equal(grid, expected) and divider in ALL_COLORS


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GEN_ATTEMPTS):
        case = _build_case(rng)
        if case is not None:
            return case["input"]
    raise RuntimeError("failed to generate a valid messenger-swap input")
