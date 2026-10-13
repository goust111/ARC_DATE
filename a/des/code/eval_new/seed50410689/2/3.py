from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_RETRIES = 400
CHAIN_RETRIES = 120
CHAIN_TRANSLATION_RETRIES = 80
KEY_PLACEMENT_RETRIES = 120
FAMILIES = ("line", "l", "t")


def _bg_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _inside(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _translate(rel: tuple[tuple[int, int], ...], anchor: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    ar, ac = anchor
    return tuple((ar + dr, ac + dc) for dr, dc in rel)


def _normalize_rel(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    anchor = min((int(r), int(c)) for r, c in coords)
    ar, ac = anchor
    return tuple(sorted((int(r) - ar, int(c) - ac) for r, c in coords))


def _components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((int(rr), int(cc)))
                for dr, dc in DIRS4:
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if not _inside(h, w, nr, nc) or seen[nr, nc]:
                        continue
                    if int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            comps.append({"color": color, "cells": tuple(sorted(cells))})
    return comps


def _extract_scene(grid: np.ndarray) -> tuple[list[dict], dict[int, dict]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _bg_color(g)
    comps = _components(g, bg)
    cell_to_comp: dict[tuple[int, int], int] = {}
    for idx, comp in enumerate(comps):
        for cell in comp["cells"]:
            cell_to_comp[cell] = idx

    chain_comp_ids: set[int] = set()
    lock_comp_ids: set[int] = set()
    chains: list[dict] = []

    for idx, comp in enumerate(comps):
        cells = comp["cells"]
        if len(cells) < 2:
            continue
        cell_set = set(cells)
        degrees = {}
        for cell in cells:
            r, c = cell
            degrees[cell] = sum((r + dr, c + dc) in cell_set for dr, dc in DIRS4)
        if any(d not in (1, 2) for d in degrees.values()):
            continue
        endpoints = [cell for cell, deg in degrees.items() if deg == 1]
        if len(endpoints) != 2:
            continue

        endpoint_set = set(endpoints)
        lock_records = []
        ok = True
        for cell in cells:
            r, c = cell
            ext = []
            for dr, dc in DIRS4:
                nr = int(r + dr)
                nc = int(c + dc)
                if not _inside(h, w, nr, nc):
                    continue
                if (nr, nc) in cell_set or int(g[nr, nc]) == bg:
                    continue
                ext.append((nr, nc))
            if cell in endpoint_set:
                if len(ext) != 1:
                    ok = False
                    break
                lock = ext[0]
                lock_idx = cell_to_comp.get(lock)
                if lock_idx is None or len(comps[lock_idx]["cells"]) != 1:
                    ok = False
                    break
                lock_records.append(
                    {
                        "endpoint": cell,
                        "lock": lock,
                        "lock_color": int(comps[lock_idx]["color"]),
                        "lock_comp_id": int(lock_idx),
                    }
                )
            elif ext:
                ok = False
                break

        if not ok:
            continue
        if len({rec["lock"] for rec in lock_records}) != 2:
            continue

        chain_comp_ids.add(idx)
        lock_comp_ids.update(rec["lock_comp_id"] for rec in lock_records)
        chains.append({"color": int(comp["color"]), "cells": cells, "locks": tuple(lock_records)})

    keys_by_color: dict[int, dict] = {}
    for idx, comp in enumerate(comps):
        if idx in chain_comp_ids or idx in lock_comp_ids:
            continue
        color = int(comp["color"])
        rel = _normalize_rel(comp["cells"])
        assert color not in keys_by_color, "key colors must be unique"
        keys_by_color[color] = {"cells": comp["cells"], "rel": rel}

    assert chains, "expected at least one chain"
    assert keys_by_color, "expected at least one key"
    return chains, keys_by_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Teleport each colored key shape onto the matching lock at a chain end and recolor it to the chain.

    Input structure:
    - The background is black.
    - Several monochrome 4-connected chain paths appear; each simple path has one different-colored lock pixel adjacent to each endpoint.
    - Additional small monochrome key shapes are scattered away from the chains, and each key color matches a lock color.

    Rule:
    - Identify every chain by finding a single-color path component whose two endpoints each touch exactly one singleton lock pixel.
    - For each lock color, locate the key of the same color and take its first pixel in row-major order as the anchor.
    - Translate that whole key shape so the anchor lands on the matching lock, repaint the translated shape with the chain color, erase the original key, and leave the chain path itself unchanged.

    Key insight:
    Lock colors are lookup keys for remote shapes, while the destination chain color controls the final repainting after the shape is transported.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    bg = _bg_color(g)

    chains, keys_by_color = _extract_scene(g)
    out = g.copy()

    for key in keys_by_color.values():
        for r, c in key["cells"]:
            out[int(r), int(c)] = bg

    for chain in chains:
        chain_color = int(chain["color"])
        for lock_info in chain["locks"]:
            lock_color = int(lock_info["lock_color"])
            assert lock_color in keys_by_color, f"missing key for lock color {lock_color}"
            rel = keys_by_color[lock_color]["rel"]
            target = _translate(rel, lock_info["lock"])
            for rr, cc in target:
                assert _inside(h, w, int(rr), int(cc)), "translated key must stay in bounds"
                if (int(rr), int(cc)) != tuple(lock_info["lock"]):
                    assert int(out[int(rr), int(cc)]) == bg, "translated key may only cover its lock"
            for rr, cc in target:
                out[int(rr), int(cc)] = chain_color

    return out


def _transform_coords(coords: list[tuple[int, int]], rot: int, flip: bool) -> tuple[tuple[int, int], ...]:
    pts = [(int(r), int(c)) for r, c in coords]
    for _ in range(rot % 4):
        pts = [(-c, r) for r, c in pts]
    if flip:
        pts = [(r, -c) for r, c in pts]
    return _normalize_rel(pts)


def _build_shape_catalog() -> dict[str, list[tuple[tuple[int, int], ...]]]:
    base: dict[str, list[list[tuple[int, int]]]] = {
        "line": [[(0, i) for i in range(n)] for n in (3, 4, 5)],
        "l": [],
        "t": [],
    }

    for vr in range(2, 5):
        for hr in range(2, 5):
            if vr + hr - 1 in (3, 4, 5):
                base["l"].append([(i, 0) for i in range(vr)] + [(0, j) for j in range(1, hr)])

    for stem_len in (2, 3):
        base["t"].append([(0, 0), (0, 1), (0, 2)] + [(i, 1) for i in range(1, stem_len)])

    out: dict[str, list[tuple[tuple[int, int], ...]]] = {}
    for family, shapes in base.items():
        variants = set()
        for shape in shapes:
            for rot in range(4):
                for flip in (False, True):
                    variants.add(_transform_coords(shape, rot, flip))
        out[family] = sorted(variants)
    return out


SHAPES_BY_FAMILY = _build_shape_catalog()


def _random_shape_order() -> list[tuple[tuple[int, int], ...]]:
    families = list(FAMILIES)
    random.shuffle(families)
    order: list[tuple[tuple[int, int], ...]] = []
    for family in families:
        variants = list(SHAPES_BY_FAMILY[family])
        random.shuffle(variants)
        order.extend(variants)
    return order


def _random_polyline() -> tuple[list[tuple[int, int]], tuple[int, int], tuple[int, int]]:
    for _ in range(200):
        seg_count = random.choice([1, 2, 3])
        dir_idx = random.randrange(4)
        dirs = [DIRS4[dir_idx]]
        for _ in range(1, seg_count):
            dir_idx = (dir_idx + random.choice([-1, 1])) % 4
            dirs.append(DIRS4[dir_idx])

        lengths = [random.randint(2, 5) for _ in range(seg_count)]
        cells = [(0, 0)]
        used = {(0, 0)}
        r = c = 0
        ok = True
        for (dr, dc), length in zip(dirs, lengths):
            for _ in range(length):
                r += dr
                c += dc
                if (r, c) in used:
                    ok = False
                    break
                used.add((r, c))
                cells.append((r, c))
            if not ok:
                break
        if not ok:
            continue

        start_lock = (cells[0][0] - dirs[0][0], cells[0][1] - dirs[0][1])
        end_lock = (cells[-1][0] + dirs[-1][0], cells[-1][1] + dirs[-1][1])
        if start_lock in used or end_lock in used or start_lock == end_lock:
            continue
        return cells, start_lock, end_lock
    raise RuntimeError("failed to build chain path")


def _buffer_conflict(cells: tuple[tuple[int, int], ...] | list[tuple[int, int]], occupied: set[tuple[int, int]]) -> bool:
    cand = {(int(r), int(c)) for r, c in cells}
    for r, c in cand:
        if (r, c) in occupied:
            return True
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if (rr, cc) in occupied and (rr, cc) not in cand:
                    return True
    return False


def _choose_lock_shapes(
    h: int,
    w: int,
    path_cells: tuple[tuple[int, int], ...],
    lock_positions: tuple[tuple[int, int], tuple[int, int]],
    prior_input: set[tuple[int, int]],
    prior_output: set[tuple[int, int]],
) -> tuple[dict, dict] | None:
    path_set = set(path_cells)
    lock_set = set(lock_positions)
    shared_forbidden = set(prior_input) | set(prior_output) | path_set | lock_set
    valid_per_lock: list[list[dict]] = []

    for lock in lock_positions:
        options: list[dict] = []
        for rel in _random_shape_order():
            cells = _translate(rel, lock)
            if any(not _inside(h, w, rr, cc) for rr, cc in cells):
                continue
            if any((rr, cc) in shared_forbidden and (rr, cc) != lock for rr, cc in cells):
                continue
            options.append({"rel": rel, "cells": cells})
        if not options:
            return None
        valid_per_lock.append(options)

    left = list(valid_per_lock[0])
    right = list(valid_per_lock[1])
    random.shuffle(left)
    random.shuffle(right)
    for a in left:
        set_a = set(a["cells"])
        for b in right:
            if set_a.isdisjoint(b["cells"]):
                return a, b
    return None


def _render_input(h: int, w: int, chains: list[dict], keys: list[dict]) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    for chain in chains:
        for r, c in chain["path"]:
            grid[int(r), int(c)] = int(chain["color"])
        for lock in chain["locks"]:
            rr, cc = lock["pos"]
            grid[int(rr), int(cc)] = int(lock["color"])
    for key in keys:
        for r, c in _translate(key["shape"], key["anchor"]):
            grid[int(r), int(c)] = int(key["color"])
    return grid


def _render_output(h: int, w: int, chains: list[dict]) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    for chain in chains:
        color = int(chain["color"])
        for r, c in chain["path"]:
            grid[int(r), int(c)] = color
        for lock in chain["locks"]:
            for r, c in _translate(lock["shape"], lock["pos"]):
                grid[int(r), int(c)] = color
    return grid


def _place_key(
    h: int,
    w: int,
    rel: tuple[tuple[int, int], ...],
    occupied: set[tuple[int, int]],
) -> tuple[int, int] | None:
    drs = [dr for dr, _ in rel]
    dcs = [dc for _, dc in rel]
    r_lo = -min(drs)
    r_hi = h - 1 - max(drs)
    c_lo = -min(dcs)
    c_hi = w - 1 - max(dcs)
    if r_lo > r_hi or c_lo > c_hi:
        return None

    for _ in range(KEY_PLACEMENT_RETRIES):
        anchor = (random.randint(r_lo, r_hi), random.randint(c_lo, c_hi))
        cells = _translate(rel, anchor)
        if _buffer_conflict(cells, occupied):
            continue
        return anchor
    return None


def _generate_attempt() -> np.ndarray | None:
    chain_count = random.choice([1, 2, 3])
    min_dim = 12 + 5 * chain_count
    h = random.randint(min_dim, 30)
    w = random.randint(min_dim, 30)

    palette = list(range(1, 10))
    random.shuffle(palette)
    chain_colors = palette[:chain_count]
    lock_colors = palette[chain_count : chain_count + 2 * chain_count]

    input_occ: set[tuple[int, int]] = set()
    output_occ: set[tuple[int, int]] = set()
    chains: list[dict] = []
    key_specs: list[dict] = []

    for chain_idx in range(chain_count):
        chain_color = int(chain_colors[chain_idx])
        lock_pair = [int(lock_colors[2 * chain_idx]), int(lock_colors[2 * chain_idx + 1])]
        placed = False

        for _ in range(CHAIN_RETRIES):
            local_path, local_lock_a, local_lock_b = _random_polyline()
            all_local = local_path + [local_lock_a, local_lock_b]
            min_r = min(r for r, _ in all_local)
            max_r = max(r for r, _ in all_local)
            min_c = min(c for _, c in all_local)
            max_c = max(c for _, c in all_local)

            r_lo = -min_r
            r_hi = h - 1 - max_r
            c_lo = -min_c
            c_hi = w - 1 - max_c
            if r_lo > r_hi or c_lo > c_hi:
                continue

            for _ in range(CHAIN_TRANSLATION_RETRIES):
                tr = random.randint(r_lo, r_hi)
                tc = random.randint(c_lo, c_hi)
                path = tuple((r + tr, c + tc) for r, c in local_path)
                lock_a = (local_lock_a[0] + tr, local_lock_a[1] + tc)
                lock_b = (local_lock_b[0] + tr, local_lock_b[1] + tc)
                candidate_input = path + (lock_a, lock_b)
                if _buffer_conflict(candidate_input, input_occ):
                    continue
                if any(cell in output_occ for cell in candidate_input):
                    continue

                chosen = _choose_lock_shapes(h, w, path, (lock_a, lock_b), input_occ, output_occ)
                if chosen is None:
                    continue

                shape_a, shape_b = chosen
                chain = {
                    "color": chain_color,
                    "path": path,
                    "locks": [
                        {"pos": lock_a, "color": lock_pair[0], "shape": shape_a["rel"]},
                        {"pos": lock_b, "color": lock_pair[1], "shape": shape_b["rel"]},
                    ],
                }
                chains.append(chain)
                input_occ.update(path)
                input_occ.update((lock_a, lock_b))
                output_occ.update(path)
                output_occ.update(shape_a["cells"])
                output_occ.update(shape_b["cells"])
                key_specs.append({"color": lock_pair[0], "shape": shape_a["rel"]})
                key_specs.append({"color": lock_pair[1], "shape": shape_b["rel"]})
                placed = True
                break

            if placed:
                break

        if not placed:
            return None

    keys = list(key_specs)
    random.shuffle(keys)
    rendered_keys: list[dict] = []
    for key in sorted(keys, key=lambda item: len(item["shape"]), reverse=True):
        anchor = _place_key(h, w, key["shape"], input_occ)
        if anchor is None:
            return None
        cells = _translate(key["shape"], anchor)
        input_occ.update(cells)
        rendered_keys.append({"color": int(key["color"]), "shape": key["shape"], "anchor": anchor})

    inp = _render_input(h, w, chains, rendered_keys)
    out = _render_output(h, w, chains)
    if np.array_equal(inp, out):
        return None
    if not np.array_equal(main(inp), out):
        return None

    colors = Counter(map(int, inp.ravel()))
    assert colors[BG] > max(v for k, v in colors.items() if k != BG)
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed after repeated retries")
