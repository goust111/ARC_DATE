from __future__ import annotations

import random
from collections import Counter, defaultdict

import numpy as np


DIR4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]
GEN_ATTEMPTS = 300


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(cells: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _rect_cycle(rect: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = rect
    pts: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        pts.append((r0, c))
    for r in range(r0 + 1, r1 + 1):
        pts.append((r, c1))
    for c in range(c1 - 1, c0 - 1, -1):
        pts.append((r1, c))
    for r in range(r1 - 1, r0, -1):
        pts.append((r, c0))
    return pts


def _render_block_cells(
    rect_info: dict,
    anchor_idx: int,
    size: int,
) -> tuple[tuple[int, int], ...]:
    cycle = rect_info["cycle"]
    length = len(cycle)
    ar, ac = cycle[anchor_idx]
    br, bc = cycle[(anchor_idx + 1) % length]
    dr = br - ar
    dc = bc - ac
    out_r = -dc
    out_c = dr
    if size == 1:
        cells = [(ar + out_r, ac + out_c)]
    else:
        cells = [
            (ar + out_r, ac + out_c),
            (ar + 2 * out_r, ac + 2 * out_c),
            (ar + out_r + dr, ac + out_c + dc),
            (ar + 2 * out_r + dr, ac + 2 * out_c + dc),
        ]
    return tuple(sorted((int(r), int(c)) for r, c in cells))


def _border_prefix(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ints = mask.astype(int)
    return np.cumsum(ints, axis=1), np.cumsum(ints, axis=0)


def _row_sum(row_pref: np.ndarray, r: int, c0: int, c1: int) -> int:
    total = int(row_pref[r, c1])
    if c0:
        total -= int(row_pref[r, c0 - 1])
    return total


def _col_sum(col_pref: np.ndarray, c: int, r0: int, r1: int) -> int:
    total = int(col_pref[r1, c])
    if r0:
        total -= int(col_pref[r0 - 1, c])
    return total


def _side_kind(rect: tuple[int, int, int, int], point: tuple[int, int]) -> str:
    r0, c0, r1, c1 = rect
    r, c = point
    horizontal = (r == r0 or r == r1) and c0 <= c <= c1
    vertical = (c == c0 or c == c1) and r0 <= r <= r1
    if horizontal and not vertical:
        return "H"
    if vertical and not horizontal:
        return "V"
    return "C"


def _infer_roles(grid: np.ndarray) -> tuple[int, int, int]:
    counts = Counter(map(int, grid.flat))
    bg = int(counts.most_common(1)[0][0])
    non_bg = [c for c in counts if c != bg]
    assert non_bg
    track = int(max(non_bg, key=lambda c: (int(counts[c]), -int(c))))

    marker_cands: list[tuple[int, int]] = []
    for color in non_bg:
        if color == track:
            continue
        comps = _components(grid == int(color))
        if len(comps) >= 2 and all(len(comp) == 1 for comp in comps):
            marker_cands.append((int(counts[color]), int(color)))
    assert marker_cands, "could not infer marker color"
    marker = max(marker_cands)[1]
    return bg, track, marker


def _find_block_components(grid: np.ndarray, bg: int, track: int, marker: int) -> list[dict]:
    blocks: list[dict] = []
    for color in sorted(set(map(int, np.unique(grid))) - {bg, track, marker}):
        comps = _components(grid == int(color))
        assert len(comps) == 1, "block colors must be unique"
        comp = comps[0]
        r0, c0, r1, c1 = _bbox(comp)
        area = len(comp)
        if area == 1:
            size = 1
        else:
            assert area == 4 and r1 - r0 == 1 and c1 - c0 == 1
            expect = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
            assert set(comp) == expect
            size = 2
        blocks.append(
            {
                "color": int(color),
                "size": int(size),
                "cells": tuple(sorted((int(r), int(c)) for r, c in comp)),
            }
        )
    assert blocks, "expected at least one block"
    return blocks


def _candidate_rectangles(grid: np.ndarray, track: int, marker: int) -> list[dict]:
    rail_mask = (grid == int(track)) | (grid == int(marker))
    h, w = grid.shape
    row_pref, col_pref = _border_prefix(rail_mask)
    rects: list[dict] = []
    for r0 in range(h - 3):
        for r1 in range(r0 + 3, h):
            height = r1 - r0 + 1
            if height < 4:
                continue
            for c0 in range(w - 3):
                for c1 in range(c0 + 3, w):
                    width = c1 - c0 + 1
                    if width < 4:
                        continue
                    if _row_sum(row_pref, r0, c0, c1) != width:
                        continue
                    if _row_sum(row_pref, r1, c0, c1) != width:
                        continue
                    if _col_sum(col_pref, c0, r0, r1) != height:
                        continue
                    if _col_sum(col_pref, c1, r0, r1) != height:
                        continue
                    rect = (int(r0), int(c0), int(r1), int(c1))
                    cycle = _rect_cycle(rect)
                    if not any(int(grid[r, c]) == int(marker) for r, c in cycle):
                        continue
                    rects.append(
                        {
                            "bbox": rect,
                            "cycle": cycle,
                            "index_of": {pt: idx for idx, pt in enumerate(cycle)},
                        }
                    )
    return rects


def _analyze_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg, track, marker = _infer_roles(g)
    blocks = _find_block_components(g, bg, track, marker)
    rects = _candidate_rectangles(g, track, marker)
    rail_mask = (g == int(track)) | (g == int(marker))

    matches_by_block: list[dict] = []
    used_rects: dict[int, dict] = {}
    for block in blocks:
        matches: list[tuple[int, int]] = []
        block_set = set(block["cells"])
        for rid, rect_info in enumerate(rects):
            cycle = rect_info["cycle"]
            # Quick reject: the block must at least touch the rectangle's neighborhood.
            touched = False
            for r, c in block_set:
                for dr, dc in DIR4:
                    nr = r + dr
                    nc = c + dc
                    if (nr, nc) in rect_info["index_of"]:
                        touched = True
                        break
                if touched:
                    break
            if not touched:
                continue
            for anchor_idx in range(len(cycle)):
                rendered = _render_block_cells(rect_info, anchor_idx, int(block["size"]))
                if set(rendered) != block_set:
                    continue
                if any(not (0 <= r < g.shape[0] and 0 <= c < g.shape[1]) for r, c in rendered):
                    continue
                matches.append((int(rid), int(anchor_idx)))
        assert len(matches) == 1, f"ambiguous block match for color {block['color']}: {matches}"
        rid, anchor_idx = matches[0]
        used_rects[rid] = rects[rid]
        matches_by_block.append(
            {
                "color": int(block["color"]),
                "size": int(block["size"]),
                "cells": tuple(block["cells"]),
                "rect_id": int(rid),
                "anchor_idx": int(anchor_idx),
            }
        )

    assert len(used_rects) == len(blocks), "each rectangle should carry one block"

    used_rect_list = []
    remap: dict[int, int] = {}
    for new_id, old_id in enumerate(sorted(used_rects)):
        remap[int(old_id)] = int(new_id)
        rect = dict(used_rects[old_id])
        rect["marker_count"] = sum(int(g[r, c]) == int(marker) for r, c in rect["cycle"])
        assert rect["marker_count"] >= 1
        used_rect_list.append(rect)

    pixel_to_rects: dict[tuple[int, int], list[int]] = defaultdict(list)
    for old_id, rect in used_rects.items():
        new_id = remap[int(old_id)]
        for pt in rect["cycle"]:
            pixel_to_rects[pt].append(int(new_id))
    assert max((len(v) for v in pixel_to_rects.values()), default=0) <= 2

    norm_blocks = []
    for block in matches_by_block:
        norm_blocks.append(
            {
                "color": int(block["color"]),
                "size": int(block["size"]),
                "cells": tuple(block["cells"]),
                "rect_id": int(remap[int(block["rect_id"])]),
                "anchor_idx": int(block["anchor_idx"]),
            }
        )

    return {
        "bg": int(bg),
        "track": int(track),
        "marker": int(marker),
        "rail_mask": rail_mask,
        "rects": used_rect_list,
        "blocks": norm_blocks,
        "pixel_to_rects": pixel_to_rects,
    }


def _simulate_anchor(info: dict, rect_id: int, anchor_idx: int, steps: int) -> tuple[int, int]:
    current_rect = int(rect_id)
    current_idx = int(anchor_idx)
    for _ in range(int(steps)):
        rect_info = info["rects"][current_rect]
        cycle = rect_info["cycle"]
        current_idx = (current_idx + 1) % len(cycle)
        point = cycle[current_idx]
        owners = info["pixel_to_rects"][point]
        if len(owners) == 2:
            other = owners[0] if owners[1] == current_rect else owners[1]
            current_rect = int(other)
            current_idx = int(info["rects"][current_rect]["index_of"][point])
    return current_rect, current_idx


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each conveyor block along its rectangle rail by the marker count of its
    starting rail; at a shared rail pixel the block switches to the other rail.
    """
    g = np.asarray(grid, dtype=int)
    info = _analyze_scene(g)
    out = g.copy()

    for block in info["blocks"]:
        for r, c in block["cells"]:
            out[r, c] = int(info["bg"])

    placed: set[tuple[int, int]] = set()
    for block in info["blocks"]:
        rect = info["rects"][int(block["rect_id"])]
        steps = int(rect["marker_count"])
        end_rect, end_anchor = _simulate_anchor(info, int(block["rect_id"]), int(block["anchor_idx"]), steps)
        rendered = _render_block_cells(info["rects"][end_rect], end_anchor, int(block["size"]))
        for r, c in rendered:
            assert 0 <= r < g.shape[0] and 0 <= c < g.shape[1]
            assert not bool(info["rail_mask"][r, c])
            assert (r, c) not in placed
            placed.add((r, c))
            out[r, c] = int(block["color"])

    return out


def _rect_border_set(rect: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    return set(_rect_cycle(rect))


def _valid_intersection(rect_a: tuple[int, int, int, int], rect_b: tuple[int, int, int, int]) -> bool:
    shared = _rect_border_set(rect_a) & _rect_border_set(rect_b)
    return len(shared) <= 1


def _touch_allowed_cells(rect_info: dict, anchor_idx: int, size: int) -> set[tuple[int, int]]:
    cycle = rect_info["cycle"]
    touch = {cycle[anchor_idx]}
    if size == 2:
        touch.add(cycle[(anchor_idx + 1) % len(cycle)])
    return touch


def _legal_anchor_cells(
    rect_info: dict,
    anchor_idx: int,
    size: int,
    h: int,
    w: int,
    rail_set: set[tuple[int, int]],
    occupied: set[tuple[int, int]],
) -> tuple[tuple[int, int], ...] | None:
    cells = _render_block_cells(rect_info, anchor_idx, size)
    if any(not (0 <= r < h and 0 <= c < w) for r, c in cells):
        return None
    if any((r, c) in rail_set or (r, c) in occupied for r, c in cells):
        return None

    allowed = _touch_allowed_cells(rect_info, anchor_idx, size)
    for r, c in cells:
        for dr, dc in DIR4:
            nr = r + dr
            nc = c + dc
            if (nr, nc) in rail_set and (nr, nc) not in allowed:
                return None
    return cells


def _choose_block_config(
    info: dict,
    rect_id: int,
    h: int,
    w: int,
    rail_set: set[tuple[int, int]],
    occupied: set[tuple[int, int]],
    prefer_cross: bool | None,
) -> dict | None:
    rect_info = info["rects"][rect_id]
    cycle = rect_info["cycle"]
    shared_idx = [idx for idx, pt in enumerate(cycle) if len(info["pixel_to_rects"][pt]) == 2]
    marker_slots = [idx for idx, pt in enumerate(cycle) if len(info["pixel_to_rects"][pt]) == 1]
    candidates: list[dict] = []
    for size in random.sample([1, 2], k=2):
        for anchor_idx in range(len(cycle)):
            if len(info["pixel_to_rects"][cycle[anchor_idx]]) != 1:
                continue
            cells = _legal_anchor_cells(rect_info, anchor_idx, size, h, w, rail_set, occupied)
            if cells is None:
                continue
            near_block = {anchor_idx, (anchor_idx + 1) % len(cycle)}
            slots = [idx for idx in marker_slots if idx not in near_block]
            if len(slots) < 1:
                continue
            max_k = min(4, len(slots))
            for k in range(1, max_k + 1):
                cross = any(1 <= (shared - anchor_idx) % len(cycle) <= k for shared in shared_idx)
                candidates.append(
                    {
                        "rect_id": int(rect_id),
                        "size": int(size),
                        "anchor_idx": int(anchor_idx),
                        "k": int(k),
                        "cells": tuple(cells),
                        "crosses": bool(cross),
                    }
                )
    if not candidates:
        return None

    if prefer_cross is not None:
        filtered = [cand for cand in candidates if cand["crosses"] == prefer_cross]
        if filtered:
            candidates = filtered
        elif prefer_cross:
            return None

    candidates.sort(
        key=lambda cand: (
            cand["crosses"],
            cand["size"] == 2,
            -cand["k"],
            -cand["anchor_idx"],
        ),
        reverse=True,
    )
    return random.choice(candidates[: min(8, len(candidates))])


def _build_generator_rects(h: int, w: int) -> list[tuple[int, int, int, int]] | None:
    base_h = random.randint(6, 9)
    base_w = random.randint(6, 9)
    if h < base_h + 14 or w < base_w + 14:
        return None
    r0 = random.randint(7, h - base_h - 7)
    c0 = random.randint(7, w - base_w - 7)
    rects = [(r0, c0, r0 + base_h - 1, c0 + base_w - 1)]

    target_count = random.choice([2, 3])
    while len(rects) < target_count:
        placed = False
        for _ in range(80):
            parent = random.choice(rects)
            pr0, pc0, pr1, pc1 = parent
            height = random.randint(5, 8)
            width = random.randint(5, 8)
            mode = random.choice(["tl", "tr", "bl", "br"])

            if mode == "tl":
                rect = (pr0 - height + 1, pc0 - width + 1, pr0, pc0)
            elif mode == "tr":
                rect = (pr0 - height + 1, pc1, pr0, pc1 + width - 1)
            elif mode == "bl":
                rect = (pr1, pc0 - width + 1, pr1 + height - 1, pc0)
            else:
                rect = (pr1, pc1, pr1 + height - 1, pc1 + width - 1)

            top, left, bottom, right = rect
            if top < 3 or left < 3 or bottom > h - 4 or right > w - 4:
                continue
            if rect in rects:
                continue
            ok = True
            for other in rects:
                shared = _rect_border_set(rect) & _rect_border_set(other)
                if other == parent:
                    if len(shared) != 1 or not _valid_intersection(rect, other):
                        ok = False
                        break
                else:
                    if shared:
                        ok = False
                        break
            if ok:
                rects.append(rect)
                placed = True
                break
        if not placed:
            return None
    return rects


def _pick_marker_indices(
    cycle: list[tuple[int, int]],
    slots: list[int],
    k: int,
) -> list[int] | None:
    random.shuffle(slots)
    chosen: list[int] = []
    chosen_pts: list[tuple[int, int]] = []
    for idx in slots:
        pt = cycle[idx]
        if any(abs(pt[0] - rr) + abs(pt[1] - cc) == 1 for rr, cc in chosen_pts):
            continue
        chosen.append(idx)
        chosen_pts.append(pt)
        if len(chosen) == k:
            return chosen
    return None


def _paint_scene(
    h: int,
    w: int,
    bg: int,
    track: int,
    marker: int,
    block_colors: list[int],
    rects: list[tuple[int, int, int, int]],
) -> np.ndarray | None:
    grid = np.full((h, w), int(bg), dtype=int)
    rect_infos = []
    rail_set: set[tuple[int, int]] = set()
    for rect in rects:
        cycle = _rect_cycle(rect)
        rect_infos.append({"bbox": rect, "cycle": cycle, "index_of": {pt: idx for idx, pt in enumerate(cycle)}})
        for r, c in cycle:
            grid[r, c] = int(track)
            rail_set.add((r, c))

    pixel_to_rects: dict[tuple[int, int], list[int]] = defaultdict(list)
    for rid, info in enumerate(rect_infos):
        for pt in info["cycle"]:
            pixel_to_rects[pt].append(int(rid))
    if max((len(v) for v in pixel_to_rects.values()), default=0) > 2:
        return None

    scene_info = {"rects": rect_infos, "pixel_to_rects": pixel_to_rects}
    occupied: set[tuple[int, int]] = set()
    configs: dict[int, dict] = {}

    shared_rects = [
        rid
        for rid, rect_info in enumerate(rect_infos)
        if any(len(pixel_to_rects[pt]) == 2 for pt in rect_info["cycle"])
    ]
    must_cross = random.choice(shared_rects) if shared_rects and random.random() < 0.65 else None
    order = list(range(len(rect_infos)))
    if must_cross is not None:
        order.remove(must_cross)
        random.shuffle(order)
        order = [must_cross] + order
    else:
        random.shuffle(order)

    for rid in order:
        prefer_cross: bool | None
        if rid == must_cross:
            prefer_cross = True
        elif rid in shared_rects and random.random() < 0.35:
            prefer_cross = True
        elif rid in shared_rects and random.random() < 0.35:
            prefer_cross = False
        else:
            prefer_cross = None
        cfg = _choose_block_config(scene_info, rid, h, w, rail_set, occupied, prefer_cross)
        if cfg is None:
            return None
        configs[rid] = cfg
        for cell in cfg["cells"]:
            occupied.add(cell)
        for r, c in cfg["cells"]:
            for rr in range(r - 1, r + 2):
                for cc in range(c - 1, c + 2):
                    if 0 <= rr < h and 0 <= cc < w:
                        occupied.add((rr, cc))

    final_occupied: set[tuple[int, int]] = set()
    for rid, cfg in configs.items():
        end_rect, end_anchor = _simulate_anchor(scene_info, rid, cfg["anchor_idx"], cfg["k"])
        end_cells = _render_block_cells(rect_infos[end_rect], end_anchor, cfg["size"])
        if any(cell in rail_set for cell in end_cells):
            return None
        if any(cell in final_occupied for cell in end_cells):
            return None
        final_occupied.update(end_cells)
        cfg["end_rect"] = int(end_rect)
        cfg["end_anchor"] = int(end_anchor)
        cfg["end_cells"] = tuple(end_cells)

    for rid, cfg in configs.items():
        rect_info = rect_infos[rid]
        cycle = rect_info["cycle"]
        marker_slots = [idx for idx, pt in enumerate(cycle) if len(pixel_to_rects[pt]) == 1]
        preferred = [idx for idx in marker_slots if idx not in {cfg["anchor_idx"], (cfg["anchor_idx"] + 1) % len(cycle)}]
        slots = preferred if len(preferred) >= cfg["k"] else marker_slots
        if len(slots) < cfg["k"]:
            return None
        chosen = _pick_marker_indices(cycle, slots[:], cfg["k"])
        if chosen is None:
            return None
        for idx in chosen:
            r, c = cycle[idx]
            grid[r, c] = int(marker)

    for rid, cfg in configs.items():
        color = int(block_colors[rid])
        for r, c in cfg["cells"]:
            grid[r, c] = color

    analyzed = _analyze_scene(grid)
    if len(analyzed["blocks"]) != len(rects):
        return None
    if {tuple(rect["bbox"]) for rect in analyzed["rects"]} != set(rects):
        return None
    _ = main(grid)
    return grid


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(18, 28)
    w = random.randint(18, 28)
    palette = random.sample(range(10), random.choice([6, 7, 8]))
    bg, track, marker = palette[:3]
    rects = _build_generator_rects(h, w)
    if rects is None:
        return None
    if len(palette) < 3 + len(rects):
        return None
    block_colors = palette[3 : 3 + len(rects)]
    grid = _paint_scene(h, w, bg, track, marker, block_colors, rects)
    if grid is None:
        return None
    counts = Counter(map(int, grid.flat))
    if int(counts[bg]) <= max(int(counts[track]), int(counts[marker])):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
