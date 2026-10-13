from __future__ import annotations

import random
from collections import defaultdict, deque

import numpy as np


PALETTE = (1, 3, 4, 5, 7)
RED = 2
DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
GRID_BANDS = (3, 4, 5)
MAX_GEN_ATTEMPTS = 400
MAX_COMBO_ATTEMPTS = 200


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= int(r) < int(shape[0]) and 0 <= int(c) < int(shape[1])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    seen = np.zeros((H, W), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    target = int(color)
    for r in range(H):
        for c in range(W):
            if seen[r, c] or int(g[r, c]) != target:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(g.shape, nr, nc) and not seen[nr, nc] and int(g[nr, nc]) == target:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(comp)
    return out


def _label_nonred_components(grid: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    comp_id = np.full((H, W), -1, dtype=int)
    comps: list[dict] = []
    cid = 0
    for r in range(H):
        for c in range(W):
            if comp_id[r, c] >= 0 or int(g[r, c]) == RED:
                continue
            color = int(g[r, c])
            q = deque([(r, c)])
            comp_id[r, c] = cid
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc):
                    if not _in_bounds(g.shape, nr, nc):
                        continue
                    if comp_id[nr, nc] >= 0 or int(g[nr, nc]) != color:
                        continue
                    comp_id[nr, nc] = cid
                    q.append((nr, nc))
            comps.append({"color": color, "coords": coords, "area": len(coords)})
            cid += 1
    return comp_id, comps


def _allowed_base_indices(bar_len: int) -> tuple[int, ...]:
    if bar_len <= 1:
        return (0,)
    if bar_len % 2 == 1:
        return (bar_len // 2,)
    return (bar_len // 2 - 1, bar_len // 2)


def _parse_marker(cells: list[tuple[int, int]]) -> dict:
    pts = sorted((int(r), int(c)) for r, c in cells)
    n = len(pts)
    assert 3 <= n <= 5, "marker size must be 3..5"
    matches: list[dict] = []

    for tip_r, tip_c in pts:
        others = [(r, c) for r, c in pts if (r, c) != (tip_r, tip_c)]
        if len(others) != n - 1:
            continue

        # Horizontal stem, vertical bar.
        cols = {c for _r, c in others}
        rows = sorted(r for r, _c in others)
        if len(cols) == 1 and rows == list(range(rows[0], rows[0] + len(rows))):
            bar_c = next(iter(cols))
            if tip_c == bar_c + 1 and tip_r in rows and rows.index(tip_r) in _allowed_base_indices(len(rows)):
                matches.append(
                    {
                        "cells": pts,
                        "tip": (tip_r, tip_c),
                        "direction": (0, 1),
                        "bar_cells": sorted(others),
                    }
                )
            if tip_c == bar_c - 1 and tip_r in rows and rows.index(tip_r) in _allowed_base_indices(len(rows)):
                matches.append(
                    {
                        "cells": pts,
                        "tip": (tip_r, tip_c),
                        "direction": (0, -1),
                        "bar_cells": sorted(others),
                    }
                )

        # Vertical stem, horizontal bar.
        rows2 = {r for r, _c in others}
        cols2 = sorted(c for _r, c in others)
        if len(rows2) == 1 and cols2 == list(range(cols2[0], cols2[0] + len(cols2))):
            bar_r = next(iter(rows2))
            if tip_r == bar_r + 1 and tip_c in cols2 and cols2.index(tip_c) in _allowed_base_indices(len(cols2)):
                matches.append(
                    {
                        "cells": pts,
                        "tip": (tip_r, tip_c),
                        "direction": (1, 0),
                        "bar_cells": sorted(others),
                    }
                )
            if tip_r == bar_r - 1 and tip_c in cols2 and cols2.index(tip_c) in _allowed_base_indices(len(cols2)):
                matches.append(
                    {
                        "cells": pts,
                        "tip": (tip_r, tip_c),
                        "direction": (-1, 0),
                        "bar_cells": sorted(others),
                    }
                )

    uniq: list[dict] = []
    seen = set()
    for item in matches:
        key = (item["direction"], item["tip"], tuple(item["bar_cells"]))
        if key not in seen:
            seen.add(key)
            uniq.append(item)
    assert len(uniq) == 1, "each red component must be one unambiguous directional marker"
    marker = uniq[0]
    marker["size"] = len(pts)
    marker["top_left"] = min(pts)
    return marker


def _extract_markers(grid: np.ndarray) -> list[dict]:
    comps = _components_of_color(grid, RED)
    assert 2 <= len(comps) <= 3, "expected 2 or 3 red markers"
    markers = [_parse_marker(comp) for comp in comps]
    markers.sort(key=lambda m: m["top_left"])
    return markers


def _walk_to_component(comp_id: np.ndarray, start: tuple[int, int], direction: tuple[int, int]) -> int:
    r, c = map(int, start)
    dr, dc = map(int, direction)
    while _in_bounds(comp_id.shape, r, c):
        cid = int(comp_id[r, c])
        if cid >= 0:
            return cid
        r += dr
        c += dc
    raise AssertionError("ray must hit a non-red component")


def _manhattan_to_coords(r: int, c: int, coords: list[tuple[int, int]]) -> int:
    return min(abs(int(r) - rr) + abs(int(c) - cc) for rr, cc in coords)


def _solve_with_markers(
    grid: np.ndarray,
    markers: list[dict],
    *,
    source_colors_override: dict[tuple[int, int], int] | None = None,
    return_debug: bool = False,
):
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    comp_id, comps = _label_nonred_components(g)
    out = g.copy()
    target_mask = np.zeros((H, W), dtype=bool)
    prepared: list[dict] = []

    for marker in markers:
        dr, dc = marker["direction"]
        tip_r, tip_c = marker["tip"]
        target_comp = _walk_to_component(comp_id, (tip_r + dr, tip_c + dc), (dr, dc))
        source_ids = {
            _walk_to_component(comp_id, (br - dr, bc - dc), (-dr, -dc))
            for br, bc in marker["bar_cells"]
        }
        assert len(source_ids) == 1, "bar back-side must point to one source region"
        source_comp = next(iter(source_ids))
        source_color = (
            int(source_colors_override[marker["top_left"]])
            if source_colors_override is not None
            else int(comps[source_comp]["color"])
        )
        target_coords = [tuple(map(int, xy)) for xy in comps[target_comp]["coords"]]
        prepared.append(
            {
                **marker,
                "target_comp": int(target_comp),
                "source_comp": int(source_comp),
                "source_color": source_color,
                "target_coords": target_coords,
                "depth": int(marker["size"] - 2),
                "priority": (-int(marker["size"]), int(marker["top_left"][0]), int(marker["top_left"][1])),
            }
        )
        for rr, cc in target_coords:
            target_mask[rr, cc] = True
            out[rr, cc] = source_color

    claims: dict[tuple[int, int], dict] = {}
    claim_sets: dict[tuple[int, int], set[tuple[int, int]]] = {m["top_left"]: set() for m in prepared}
    overlap_cells: dict[tuple[int, int], list[dict]] = defaultdict(list)

    for marker in prepared:
        color = int(marker["source_color"])
        depth = int(marker["depth"])
        if depth <= 0:
            continue
        for r in range(H):
            for c in range(W):
                if int(g[r, c]) == RED or target_mask[r, c]:
                    continue
                dist = _manhattan_to_coords(r, c, marker["target_coords"])
                if 1 <= dist <= depth:
                    key = (int(r), int(c))
                    claim_sets[marker["top_left"]].add(key)
                    overlap_cells[key].append(marker)
                    winner = claims.get(key)
                    if winner is None or marker["priority"] < winner["priority"]:
                        claims[key] = marker
                        out[r, c] = color

    for marker in prepared:
        dr, dc = marker["direction"]
        back = (-dr, -dc)
        for r, c in marker["cells"]:
            rr, cc = int(r), int(c)
            while True:
                rr += back[0]
                cc += back[1]
                assert _in_bounds(g.shape, rr, cc), "marker must have a source-side cell"
                if int(g[rr, cc]) != RED:
                    out[r, c] = int(out[rr, cc])
                    break

    if not return_debug:
        return out

    return out, {
        "markers": prepared,
        "claims": claims,
        "claim_sets": claim_sets,
        "overlaps": {k: v for k, v in overlap_cells.items() if len(v) >= 2},
        "target_mask": target_mask,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the region pointed to by each red marker, then diffuse that recolor outward by a marker-size-controlled depth.

    Input structure:
    - The grid is a mosaic of 4-connected monochrome regions using colors 1, 3, 4, 5, and 7.
    - Two or three 4-connected red markers lie on region boundaries.
    - Each red marker is a short bar with one perpendicular stem tip; the stem direction indicates a target side.

    Rule:
    - For each red marker, identify the stem tip and the straight bar behind it.
    - The first non-red region in the stem direction is the target region; the region immediately behind the bar is the source region.
    - Repaint the whole target region with the source region's color.
    - If the marker has N pixels, repaint every non-red, non-target cell whose Manhattan distance to the original target region is between 1 and N-2 inclusive.
    - Process all markers simultaneously; when multiple diffusions cover one cell, the larger marker wins, and equal sizes break ties by the marker whose top-left cell is earlier in row-major order.
    - Finally, replace each red marker cell with the final color found on its source side.

    Key insight:
    - The same marker geometry carries two meanings at once: the stem chooses which region changes, and the marker's pixel count chooses how far that color spreads.
    """
    markers = _extract_markers(np.asarray(grid, dtype=int))
    return _solve_with_markers(np.asarray(grid, dtype=int), markers)


def _sample_partition(count: int) -> list[int]:
    while True:
        bands = [random.choice(GRID_BANDS) for _ in range(count)]
        if 12 <= sum(bands) <= 18:
            return bands


def _build_rectangles() -> tuple[np.ndarray, list[dict], dict[tuple[int, int], int]]:
    n_rows = random.choice((3, 4))
    n_cols = random.choice((3, 4))
    row_heights = _sample_partition(n_rows)
    col_widths = _sample_partition(n_cols)
    H, W = sum(row_heights), sum(col_widths)
    grid = np.zeros((H, W), dtype=int)

    row_starts = [0]
    for h in row_heights[:-1]:
        row_starts.append(row_starts[-1] + h)
    col_starts = [0]
    for w in col_widths[:-1]:
        col_starts.append(col_starts[-1] + w)

    rects: list[dict] = []
    rect_id_of: dict[tuple[int, int], int] = {}
    for ri, r0 in enumerate(row_starts):
        for ci, c0 in enumerate(col_starts):
            h = row_heights[ri]
            w = col_widths[ci]
            r1 = r0 + h - 1
            c1 = c0 + w - 1
            used = set()
            if ri > 0:
                used.add(int(rects[(ri - 1) * n_cols + ci]["color"]))
            if ci > 0:
                used.add(int(rects[ri * n_cols + ci - 1]["color"]))
            color_choices = [c for c in PALETTE if c not in used]
            color = random.choice(color_choices)
            rect_id = len(rects)
            rects.append(
                {
                    "id": rect_id,
                    "ri": ri,
                    "ci": ci,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "height": int(h),
                    "width": int(w),
                    "color": int(color),
                }
            )
            grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)
            rect_id_of[(ri, ci)] = rect_id
    return grid, rects, rect_id_of


def _candidate_cells_for_edge(src: dict, dst: dict, direction: tuple[int, int], size: int) -> list[dict]:
    L = size - 1
    base_indices = _allowed_base_indices(L)
    out: list[dict] = []
    sr0, sc0, sr1, sc1 = map(int, src["bbox"])
    dr0, dc0, dr1, dc1 = map(int, dst["bbox"])

    if direction == (0, 1):
        if int(src["width"]) < 2 or int(dst["width"]) < 2 or int(src["height"]) < L:
            return out
        bar_c = sc1
        for start_r in range(sr0, sr1 - L + 2):
            for base_idx in base_indices:
                base_r = start_r + base_idx
                bar = [(r, bar_c) for r in range(start_r, start_r + L)]
                tip = (base_r, bar_c + 1)
                out.append({"cells": sorted(bar + [tip]), "tip": tip, "direction": direction})
        return out

    if direction == (0, -1):
        if int(src["width"]) < 2 or int(dst["width"]) < 2 or int(src["height"]) < L:
            return out
        bar_c = sc0
        for start_r in range(sr0, sr1 - L + 2):
            for base_idx in base_indices:
                base_r = start_r + base_idx
                bar = [(r, bar_c) for r in range(start_r, start_r + L)]
                tip = (base_r, bar_c - 1)
                out.append({"cells": sorted(bar + [tip]), "tip": tip, "direction": direction})
        return out

    if direction == (1, 0):
        if int(src["height"]) < 2 or int(dst["height"]) < 2 or int(src["width"]) < L:
            return out
        bar_r = sr1
        for start_c in range(sc0, sc1 - L + 2):
            for base_idx in base_indices:
                base_c = start_c + base_idx
                bar = [(bar_r, c) for c in range(start_c, start_c + L)]
                tip = (bar_r + 1, base_c)
                out.append({"cells": sorted(bar + [tip]), "tip": tip, "direction": direction})
        return out

    if int(src["height"]) < 2 or int(dst["height"]) < 2 or int(src["width"]) < L:
        return out
    bar_r = sr0
    for start_c in range(sc0, sc1 - L + 2):
        for base_idx in base_indices:
            base_c = start_c + base_idx
            bar = [(bar_r, c) for c in range(start_c, start_c + L)]
            tip = (bar_r - 1, base_c)
            out.append({"cells": sorted(bar + [tip]), "tip": tip, "direction": direction})
    return out


def _all_marker_candidates(rects: list[dict], rect_id_of: dict[tuple[int, int], int]) -> list[dict]:
    by_pos = {(rect["ri"], rect["ci"]): rect for rect in rects}
    out: list[dict] = []
    for rect in rects:
        ri, ci = int(rect["ri"]), int(rect["ci"])
        if (ri, ci + 1) in by_pos:
            dst = by_pos[(ri, ci + 1)]
            for src, target, direction in ((rect, dst, (0, 1)), (dst, rect, (0, -1))):
                for size in (4, 5):
                    for item in _candidate_cells_for_edge(src, target, direction, size):
                        marker = _parse_marker(item["cells"])
                        out.append(
                            {
                                **marker,
                                "source_rect": int(src["id"]),
                                "target_rect": int(target["id"]),
                                "source_color": int(src["color"]),
                                "target_bbox": tuple(map(int, target["bbox"])),
                            }
                        )
        if (ri + 1, ci) in by_pos:
            dst = by_pos[(ri + 1, ci)]
            for src, target, direction in ((rect, dst, (1, 0)), (dst, rect, (-1, 0))):
                for size in (4, 5):
                    for item in _candidate_cells_for_edge(src, target, direction, size):
                        marker = _parse_marker(item["cells"])
                        out.append(
                            {
                                **marker,
                                "source_rect": int(src["id"]),
                                "target_rect": int(target["id"]),
                                "source_color": int(src["color"]),
                                "target_bbox": tuple(map(int, target["bbox"])),
                            }
                        )
    return out


def _halo8(cells: list[tuple[int, int]]) -> set[tuple[int, int]]:
    halo = set()
    for r, c in cells:
        halo.add((int(r), int(c)))
        for rr, cc in _neighbors8(r, c):
            halo.add((int(rr), int(cc)))
    return halo


def _bbox_distance(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    ar0, ac0, ar1, ac1 = map(int, a)
    br0, bc0, br1, bc1 = map(int, b)
    dr = max(0, br0 - ar1 - 1, ar0 - br1 - 1)
    dc = max(0, bc0 - ac1 - 1, ac0 - bc1 - 1)
    return int(dr + dc)


def _choose_markers(candidates: list[dict], mode: str) -> list[dict] | None:
    num_markers = 3 if random.random() < 0.6 else 2
    if len(candidates) < num_markers:
        return None

    for _ in range(MAX_COMBO_ATTEMPTS):
        chosen: list[dict] = []
        occupied: set[tuple[int, int]] = set()
        used_targets: set[int] = set()
        pool = candidates[:]
        random.shuffle(pool)

        while len(chosen) < num_markers:
            valid = []
            for cand in pool:
                if int(cand["target_rect"]) in used_targets:
                    continue
                if _halo8(cand["cells"]) & occupied:
                    continue
                valid.append(cand)
            if not valid:
                break

            if chosen:
                def score(cand: dict) -> tuple[int, int, int]:
                    d = min(_bbox_distance(cand["target_bbox"], prev["target_bbox"]) for prev in chosen)
                    same_size_bonus = 0 if any(int(prev["size"]) == int(cand["size"]) for prev in chosen) else 1
                    big_size_bonus = -int(cand["size"])
                    if mode == "tie_overlap":
                        return (same_size_bonus, d, big_size_bonus)
                    if mode == "overlap_big":
                        return (d, same_size_bonus, big_size_bonus)
                    return (random.randint(0, 1000), d, big_size_bonus)

                valid.sort(key=score)
                pick = random.choice(valid[: max(1, min(8, len(valid)))])
            else:
                pick = random.choice(valid)

            chosen.append(pick)
            used_targets.add(int(pick["target_rect"]))
            occupied |= set(pick["cells"])

        if len(chosen) == num_markers:
            return sorted(chosen, key=lambda m: m["top_left"])
    return None


def _place_markers(grid: np.ndarray, markers: list[dict]) -> np.ndarray:
    g = np.array(grid, dtype=int, copy=True)
    for marker in markers:
        for r, c in marker["cells"]:
            g[r, c] = RED
    return g


def _semantic_output(grid: np.ndarray, markers: list[dict]):
    overrides = {m["top_left"]: int(m["source_color"]) for m in markers}
    out, debug = _solve_with_markers(grid, markers, source_colors_override=overrides, return_debug=True)
    return out, debug


def _mode_satisfied(mode: str, debug: dict) -> bool:
    overlaps = list(debug["overlaps"].items())
    if mode == "plain":
        return True
    if not overlaps:
        return False
    if mode == "overlap_big":
        for cell, markers in overlaps:
            sizes = {int(m["size"]) for m in markers}
            if len(sizes) >= 2:
                winner = min(markers, key=lambda m: m["priority"])
                if int(winner["size"]) == max(sizes):
                    return True
        return False
    for _cell, markers in overlaps:
        sizes = {int(m["size"]) for m in markers}
        if len(sizes) == 1:
            winner = min(markers, key=lambda m: m["priority"])
            if winner["top_left"] == min((m["top_left"] for m in markers)):
                return True
    return False


def _validate_instance(inp: np.ndarray, markers: list[dict]) -> bool:
    if inp.ndim != 2 or max(inp.shape) > 30:
        return False
    if not (2 <= len(markers) <= 3):
        return False

    try:
        parsed = _extract_markers(inp)
        if [m["top_left"] for m in parsed] != [m["top_left"] for m in sorted(markers, key=lambda m: m["top_left"])]:
            return False
        expected, debug = _semantic_output(inp, markers)
        actual = main(inp)
    except Exception:
        return False

    if not np.array_equal(expected, actual):
        return False
    if np.array_equal(inp, actual):
        return False

    source_colors = {int(m["source_color"]) for m in markers}
    if len(source_colors) < 2:
        return False

    target_changed = 0
    expansion_changed = 0
    for marker in debug["markers"]:
        if not debug["claim_sets"][marker["top_left"]]:
            return False
        for r, c in marker["target_coords"]:
            if int(inp[r, c]) != int(actual[r, c]):
                target_changed += 1
        for r, c in debug["claim_sets"][marker["top_left"]]:
            if int(inp[r, c]) != int(actual[r, c]):
                expansion_changed += 1
    if target_changed == 0 or expansion_changed == 0:
        return False

    mode = "plain"
    sizes = [int(m["size"]) for m in markers]
    if len(set(sizes)) == 1:
        mode = "tie_overlap"
    elif max(sizes) != min(sizes):
        mode = "overlap_big"
    if mode != "plain" and debug["overlaps"] and not _mode_satisfied(mode, debug):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    mode = random.choices(["overlap_big", "tie_overlap", "plain"], weights=[4, 3, 3])[0]
    base, rects, rect_id_of = _build_rectangles()
    candidates = _all_marker_candidates(rects, rect_id_of)
    if len(candidates) < 2:
        return None

    markers = _choose_markers(candidates, mode)
    if markers is None:
        return None

    inp = _place_markers(base, markers)

    try:
        _ = main(inp)
    except Exception:
        return None

    expected, debug = _semantic_output(inp, markers)
    if not _mode_satisfied(mode, debug):
        return None
    if not _validate_instance(inp, markers):
        return None

    if np.array_equal(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("failed to generate a valid puzzle instance")
