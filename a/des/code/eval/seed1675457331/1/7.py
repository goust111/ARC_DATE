import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}
MAX_GEN_RETRIES = 400


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _extract_table(g: np.ndarray, border: int) -> tuple[np.ndarray, dict[int, int]]:
    h, w = g.shape
    assert h >= 4 and w >= 4
    mask = np.zeros_like(g, dtype=bool)
    mapping: dict[int, int] = {}
    c = 1
    while c < w - 1:
        top = int(g[1, c])
        bottom = int(g[2, c])
        if top in (BG, border) or bottom in (BG, border):
            break
        mapping[top] = bottom
        mask[1, c] = True
        mask[2, c] = True
        c += 1
    assert len(mapping) >= 2
    return mask, mapping


def _find_objects(
    g: np.ndarray, table_mask: np.ndarray, border: int
) -> tuple[list[dict], np.ndarray, dict[tuple[int, int], int]]:
    h, w = g.shape
    seen = np.zeros_like(g, dtype=bool)
    tower_mask = np.zeros_like(g, dtype=bool)
    towers: list[dict] = []
    filters: dict[tuple[int, int], int] = {}

    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if seen[r, c] or table_mask[r, c]:
                continue
            color = int(g[r, c])
            if color in (BG, border):
                continue

            q = deque([(r, c)])
            seen[r, c] = True
            coords = [(r, c)]
            while q:
                rr, cc = q.popleft()
                for nr, nc in _neighbors4(rr, cc):
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or table_mask[nr, nc]:
                        continue
                    if int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
                    coords.append((nr, nc))

            r0, c0, r1, c1 = _bbox(coords)
            area = len(coords)
            if area == 9 and (r1 - r0 + 1, c1 - c0 + 1) == (3, 3):
                want = {(rr, cc) for rr in range(r0, r1 + 1) for cc in range(c0, c1 + 1)}
                if set(coords) != want:
                    raise AssertionError("tower must be a solid 3x3 block")
                for rr, cc in coords:
                    tower_mask[rr, cc] = True
                towers.append(
                    {
                        "color": color,
                        "bbox": (r0, c0, r1, c1),
                        "center": (r0 + 1, c0 + 1),
                    }
                )
            elif area == 1:
                filters[(r, c)] = color
            else:
                raise AssertionError("unexpected non-table object shape")

    towers.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))
    tower_colors = [int(t["color"]) for t in towers]
    assert len(tower_colors) == len(set(tower_colors))
    return towers, tower_mask, filters


def _raw_rays(
    g: np.ndarray, towers: list[dict], tower_mask: np.ndarray, table_mask: np.ndarray
) -> dict[tuple[int, str], dict]:
    border = int(g[0, 0])
    rays: dict[tuple[int, str], dict] = {}
    for idx, tower in enumerate(towers):
        r0, c0, r1, c1 = tower["bbox"]
        cr, cc = tower["center"]
        starts = {
            "U": (r0, cc),
            "D": (r1, cc),
            "L": (cr, c0),
            "R": (cr, c1),
        }
        for name, (sr, sc) in starts.items():
            dr, dc = DIRS[name]
            rr, cc2 = sr, sc
            cells: list[tuple[int, int]] = []
            while True:
                rr += dr
                cc2 += dc
                if int(g[rr, cc2]) == border:
                    break
                if tower_mask[rr, cc2] or table_mask[rr, cc2]:
                    break
                cells.append((rr, cc2))
            rays[(idx, name)] = {
                "tower_idx": idx,
                "dir": name,
                "axis": "h" if dr == 0 else "v",
                "cells": cells,
                "cell_set": set(cells),
            }
    return rays


def _solve(grid: np.ndarray, return_meta: bool = False):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    border = int(g[0, 0])
    assert np.all(g[0, :] == border)
    assert np.all(g[h - 1, :] == border)
    assert np.all(g[:, 0] == border)
    assert np.all(g[:, w - 1] == border)

    table_mask, mapping = _extract_table(g, border)
    towers, tower_mask, filters = _find_objects(g, table_mask, border)
    assert towers

    out = g.copy()
    best_dist = np.full(g.shape, -1, dtype=int)
    visit_count = np.zeros(g.shape, dtype=int)
    meta = {
        "passes": 0,
        "blocks": 0,
        "border_stops": 0,
        "tower_stops": 0,
        "table_stops": 0,
    }

    def paint(rr: int, cc: int, color: int, dist: int) -> None:
        visit_count[rr, cc] += 1
        if dist > int(best_dist[rr, cc]):
            best_dist[rr, cc] = dist
            out[rr, cc] = int(color)

    for tower in towers:
        base_color = int(tower["color"])
        r0, c0, r1, c1 = tower["bbox"]
        cr, cc = tower["center"]
        starts = [
            ((r0, cc), "U"),
            ((r1, cc), "D"),
            ((cr, c0), "L"),
            ((cr, c1), "R"),
        ]
        for (sr, sc), name in starts:
            dr, dc = DIRS[name]
            rr, cc2 = sr, sc
            dist = 0
            current = base_color
            while True:
                rr += dr
                cc2 += dc
                dist += 1
                if int(g[rr, cc2]) == border:
                    meta["border_stops"] += 1
                    break
                if tower_mask[rr, cc2]:
                    meta["tower_stops"] += 1
                    break
                if table_mask[rr, cc2]:
                    meta["table_stops"] += 1
                    break
                if (rr, cc2) in filters:
                    if int(filters[(rr, cc2)]) != current:
                        meta["blocks"] += 1
                        break
                    assert current in mapping
                    current = int(mapping[current])
                    meta["passes"] += 1
                    paint(rr, cc2, current, dist)
                    continue
                if int(g[rr, cc2]) != BG:
                    raise AssertionError("unexpected obstacle outside table/towers/filters")
                paint(rr, cc2, current, dist)

    meta["overlap_cells"] = int(np.count_nonzero(visit_count > 1))
    meta["changed_cells"] = int(np.count_nonzero(out != g))
    return (out, meta) if return_meta else out


def main(grid: np.ndarray) -> np.ndarray:
    return _solve(grid, return_meta=False)


def _random_derangement(colors: list[int], rng: random.Random) -> list[int]:
    if len(colors) == 2:
        return [colors[1], colors[0]]
    out = colors[:]
    for _ in range(200):
        rng.shuffle(out)
        if all(a != b for a, b in zip(colors, out)):
            return out
    return colors[1:] + colors[:1]


def _compatible_centers(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return abs(a[0] - b[0]) > 4 or abs(a[1] - b[1]) > 4


def _sample_extra_center(
    existing: list[tuple[int, int]],
    row_range: range,
    col_range: range,
    rng: random.Random,
    require_new_row: bool,
    require_new_col: bool,
) -> tuple[int, int] | None:
    used_rows = {r for r, _ in existing}
    used_cols = {c for _, c in existing}
    for _ in range(300):
        cand = (rng.choice(row_range), rng.choice(col_range))
        if require_new_row and cand[0] in used_rows:
            continue
        if require_new_col and cand[1] in used_cols:
            continue
        if all(_compatible_centers(cand, other) for other in existing):
            return cand
    return None


def _sample_tower_centers(
    h: int, w: int, table_len: int, pattern: str, rng: random.Random
) -> list[tuple[int, int]] | None:
    row_range = range(6, h - 3)
    col_range = range(table_len + 5, w - 3)
    if len(row_range) < 3 or len(col_range) < 6:
        return None

    for _ in range(300):
        want = 4 if rng.random() < 0.45 else 3
        centers: list[tuple[int, int]] = []
        if pattern == "independent":
            while len(centers) < want:
                cand = _sample_extra_center(
                    centers,
                    row_range,
                    col_range,
                    rng,
                    require_new_row=True,
                    require_new_col=True,
                )
                if cand is None:
                    break
                centers.append(cand)
            if len(centers) == want:
                return centers
            continue

        if pattern == "row_pair":
            r_pair = rng.choice(row_range)
            c1, c2 = sorted(rng.sample(list(col_range), 2))
            if c2 - c1 < 8:
                continue
            centers = [(r_pair, c1), (r_pair, c2)]
            third = _sample_extra_center(
                centers,
                row_range,
                col_range,
                rng,
                require_new_row=True,
                require_new_col=False,
            )
            if third is None or third[1] in (c1, c2):
                continue
            centers.append(third)
            if want == 4:
                extra = _sample_extra_center(
                    centers,
                    row_range,
                    col_range,
                    rng,
                    require_new_row=True,
                    require_new_col=True,
                )
                if extra is None:
                    continue
                centers.append(extra)
            return centers

        if pattern == "col_pair":
            c_pair = rng.choice(col_range)
            r1, r2 = sorted(rng.sample(list(row_range), 2))
            if r2 - r1 < 8:
                continue
            centers = [(r1, c_pair), (r2, c_pair)]
            third = _sample_extra_center(
                centers,
                row_range,
                col_range,
                rng,
                require_new_row=False,
                require_new_col=True,
            )
            if third is None or third[0] in (r1, r2):
                continue
            centers.append(third)
            if want == 4:
                extra = _sample_extra_center(
                    centers,
                    row_range,
                    col_range,
                    rng,
                    require_new_row=True,
                    require_new_col=True,
                )
                if extra is None:
                    continue
                centers.append(extra)
            return centers

    return None


def _paint_tower(g: np.ndarray, center: tuple[int, int], color: int) -> None:
    r, c = center
    g[r - 1 : r + 2, c - 1 : c + 2] = color


def _is_isolated_black(g: np.ndarray, r: int, c: int) -> bool:
    if int(g[r, c]) != BG:
        return False
    return all(int(g[nr, nc]) == BG for nr, nc in _neighbors4(r, c))


def _candidate_positions(
    g: np.ndarray,
    ray: dict,
    tower_rows: set[int],
    tower_cols: set[int],
    reserved: set[tuple[int, int]],
    after_idx: int = -1,
) -> list[tuple[int, tuple[int, int]]]:
    out: list[tuple[int, tuple[int, int]]] = []
    for idx, (r, c) in enumerate(ray["cells"]):
        if idx <= after_idx or (r, c) in reserved:
            continue
        if ray["axis"] == "h" and c in tower_cols:
            continue
        if ray["axis"] == "v" and r in tower_rows:
            continue
        if _is_isolated_black(g, r, c):
            out.append((idx, (r, c)))
    return out


def _mismatch_color(current: int, color_keys: list[int], rng: random.Random) -> int:
    choices = [color for color in color_keys if color != current]
    return rng.choice(choices)


def _try_place_pass_profile(
    g: np.ndarray,
    ray: dict,
    start_color: int,
    mapping: dict[int, int],
    tower_rows: set[int],
    tower_cols: set[int],
    reserved: set[tuple[int, int]],
    profile: str,
    rng: random.Random,
) -> list[tuple[tuple[int, int], int]] | None:
    colors = list(mapping)
    first_options = _candidate_positions(g, ray, tower_rows, tower_cols, reserved)
    rng.shuffle(first_options)
    for first_idx, first_pos in first_options:
        temp = g.copy()
        temp[first_pos] = start_color
        if profile == "single":
            return [(first_pos, start_color)]

        next_color = int(mapping[start_color])
        second_options = _candidate_positions(
            temp, ray, tower_rows, tower_cols, reserved, after_idx=first_idx
        )
        rng.shuffle(second_options)
        for _second_idx, second_pos in second_options:
            if profile == "chain":
                return [(first_pos, start_color), (second_pos, next_color)]
            stop_color = _mismatch_color(next_color, colors, rng)
            return [(first_pos, start_color), (second_pos, stop_color)]
    return None


def _try_place_block(
    g: np.ndarray,
    ray: dict,
    start_color: int,
    color_keys: list[int],
    tower_rows: set[int],
    tower_cols: set[int],
    reserved: set[tuple[int, int]],
    rng: random.Random,
) -> list[tuple[tuple[int, int], int]] | None:
    options = _candidate_positions(g, ray, tower_rows, tower_cols, reserved)
    if not options:
        return None
    idx, pos = rng.choice(options)
    _ = idx
    return [(pos, _mismatch_color(start_color, color_keys, rng))]


def _pick_cross_candidate(
    towers: list[dict], rays: dict[tuple[int, str], dict], rng: random.Random
) -> tuple[dict, dict] | None:
    candidates: list[tuple[dict, dict]] = []
    for i, tower_a in enumerate(towers):
        ra, ca = tower_a["center"]
        for j, tower_b in enumerate(towers):
            if i == j:
                continue
            rb, cb = tower_b["center"]
            if ra == rb or ca == cb:
                continue
            hdir = "R" if cb > ca else "L"
            vdir = "D" if ra > rb else "U"
            ray_h = rays[(i, hdir)]
            ray_v = rays[(j, vdir)]
            cross = (ra, cb)
            if cross in ray_h["cell_set"] and cross in ray_v["cell_set"]:
                candidates.append((ray_h, ray_v))
    if not candidates:
        return None
    return rng.choice(candidates)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(18, 27)
    w = rng.randint(21, 30)
    table_len = rng.randint(4, 6)
    colors = rng.sample(list(COLOR_POOL), table_len)
    outputs = _random_derangement(colors, rng)
    mapping = {src: dst for src, dst in zip(colors, outputs)}

    pattern = rng.choice(["independent", "row_pair", "col_pair"])
    centers = _sample_tower_centers(h, w, table_len, pattern, rng)
    if centers is None:
        return None

    tower_colors = rng.sample(colors, len(centers))
    g = np.full((h, w), BG, dtype=int)
    g[0, :] = GREY
    g[h - 1, :] = GREY
    g[:, 0] = GREY
    g[:, w - 1] = GREY
    for idx, src in enumerate(colors, start=1):
        g[1, idx] = src
        g[2, idx] = mapping[src]

    for center, color in zip(centers, tower_colors):
        _paint_tower(g, center, color)

    table_mask = np.zeros_like(g, dtype=bool)
    table_mask[1, 1 : table_len + 1] = True
    table_mask[2, 1 : table_len + 1] = True
    towers = []
    tower_mask = np.zeros_like(g, dtype=bool)
    for center, color in zip(centers, tower_colors):
        r, c = center
        towers.append({"color": color, "bbox": (r - 1, c - 1, r + 1, c + 1), "center": center})
        tower_mask[r - 1 : r + 2, c - 1 : c + 2] = True
    towers.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))
    rays = _raw_rays(g, towers, tower_mask, table_mask)

    pair = _pick_cross_candidate(towers, rays, rng)
    if pair is None:
        return None
    reserved_rays = {
        (pair[0]["tower_idx"], pair[0]["dir"]),
        (pair[1]["tower_idx"], pair[1]["dir"]),
    }
    reserved_cells: set[tuple[int, int]] = set(pair[0]["cells"]) | set(pair[1]["cells"])

    free_rays = [ray for key, ray in rays.items() if key not in reserved_rays and len(ray["cells"]) >= 2]
    if len(free_rays) < 2:
        return None

    tower_rows = {int(t["center"][0]) for t in towers}
    tower_cols = {int(t["center"][1]) for t in towers}

    pass_profile = rng.choice(["single", "chain", "pass_then_block"])
    need_pass = 1 if pass_profile == "single" else 2
    color_keys = list(mapping)

    pass_ray_order = free_rays[:]
    rng.shuffle(pass_ray_order)
    pass_placements = None
    pass_ray_id = None
    for ray in pass_ray_order:
        if len(_candidate_positions(g, ray, tower_rows, tower_cols, reserved_cells)) < need_pass:
            continue
        tower_color = int(towers[ray["tower_idx"]]["color"])
        pass_placements = _try_place_pass_profile(
            g,
            ray,
            tower_color,
            mapping,
            tower_rows,
            tower_cols,
            reserved_cells,
            pass_profile,
            rng,
        )
        if pass_placements is not None:
            pass_ray_id = (ray["tower_idx"], ray["dir"])
            break
    if pass_placements is None or pass_ray_id is None:
        return None
    for pos, color in pass_placements:
        g[pos] = color

    block_candidates = [ray for ray in free_rays if (ray["tower_idx"], ray["dir"]) != pass_ray_id]
    rng.shuffle(block_candidates)
    block_placements = None
    for ray in block_candidates:
        tower_color = int(towers[ray["tower_idx"]]["color"])
        block_placements = _try_place_block(
            g,
            ray,
            tower_color,
            color_keys,
            tower_rows,
            tower_cols,
            reserved_cells,
            rng,
        )
        if block_placements is not None:
            break
    if block_placements is None:
        return None
    for pos, color in block_placements:
        g[pos] = color

    extra_kind = rng.choice(["none", "single", "block"])
    if extra_kind != "none":
        extra_rays = [
            ray
            for ray in block_candidates
            if block_placements is not None and (ray["tower_idx"], ray["dir"]) != pass_ray_id
        ]
        rng.shuffle(extra_rays)
        for ray in extra_rays:
            if extra_kind == "single":
                placements = _try_place_pass_profile(
                    g,
                    ray,
                    int(towers[ray["tower_idx"]]["color"]),
                    mapping,
                    tower_rows,
                    tower_cols,
                    reserved_cells,
                    "single",
                    rng,
                )
            else:
                placements = _try_place_block(
                    g,
                    ray,
                    int(towers[ray["tower_idx"]]["color"]),
                    color_keys,
                    tower_rows,
                    tower_cols,
                    reserved_cells,
                    rng,
                )
            if placements is None:
                continue
            for pos, color in placements:
                g[pos] = color
            break

    out, meta = _solve(g, return_meta=True)
    if meta["passes"] < 1 or meta["blocks"] < 1:
        return None
    if meta["overlap_cells"] < 1:
        return None
    if meta["changed_cells"] < 12:
        return None
    if np.array_equal(out, g):
        return None
    return g


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid signal-tower puzzle")
