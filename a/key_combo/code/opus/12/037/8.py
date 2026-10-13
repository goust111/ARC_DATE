from __future__ import annotations

import itertools
import random

import numpy as np


BG = 0
CTRL = 2
RAY = 3
BLOCK = 4

DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SHAPES = [(3, 3), (3, 4), (4, 3)]
GRID_ROWS = (18, 24)
GRID_COLS = (18, 24)
BLOCK_COUNT = (4, 6)
LENGTHS = (1, 2, 3, 4)
MIN_BLOCK_DISTANCE = 4
GEN_ATTEMPTS = 400
ASSIGN_ATTEMPTS = 250
PAIR_ATTEMPTS = 250


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.asarray(coords, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _rect_distance(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    dr = 0
    if ar1 < br0:
        dr = br0 - ar1
    elif br1 < ar0:
        dr = ar0 - br1
    dc = 0
    if ac1 < bc0:
        dc = bc0 - ac1
    elif bc1 < ac0:
        dc = ac0 - bc1
    return int(dr + dc)


def _compatible_dirs(h: int, w: int) -> list[tuple[int, int]]:
    dirs: list[tuple[int, int]] = []
    if h % 2 == 1:
        dirs.extend([(0, -1), (0, 1)])
    if w % 2 == 1:
        dirs.extend([(-1, 0), (1, 0)])
    return dirs


def _direction_from_bias(block_coords: np.ndarray, ctrl_coords: np.ndarray) -> tuple[int, int]:
    block_center = block_coords.mean(axis=0)
    ctrl_center = ctrl_coords.mean(axis=0)
    vr = float(block_center[0] - ctrl_center[0])
    vc = float(block_center[1] - ctrl_center[1])
    assert abs(vr) > 1e-9 or abs(vc) > 1e-9, "control centroid must not equal block centroid"
    if abs(vc) >= abs(vr):
        return (0, 1 if vc > 0 else -1)
    return (1 if vr > 0 else -1, 0)


def _maybe_direction_from_bias(block_coords: np.ndarray, ctrl_coords: np.ndarray) -> tuple[int, int] | None:
    block_center = block_coords.mean(axis=0)
    ctrl_center = ctrl_coords.mean(axis=0)
    if np.allclose(block_center, ctrl_center):
        return None
    return _direction_from_bias(block_coords, ctrl_coords)


def _ray_start(block: dict, direction: tuple[int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = block["bbox"]
    if direction == (0, -1):
        return ((r0 + r1) // 2, c0 - 1)
    if direction == (0, 1):
        return ((r0 + r1) // 2, c1 + 1)
    if direction == (-1, 0):
        return (r0 - 1, (c0 + c1) // 2)
    return (r1 + 1, (c0 + c1) // 2)


def _extract_blocks(grid: np.ndarray) -> tuple[list[dict], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    comps = _components(g != BG)
    blocks: list[dict] = []
    labels = np.full(g.shape, -1, dtype=int)
    for coords in comps:
        r0, c0, r1, c1 = _bbox(coords)
        region = g[r0 : r1 + 1, c0 : c1 + 1]
        assert coords.shape[0] == region.size, "foreground objects must be solid rectangles"
        assert set(map(int, np.unique(region))) <= {CTRL, BLOCK}, "unexpected colors in a block"
        ctrl_coords = np.argwhere(region == CTRL) + np.array([r0, c0])
        assert 1 <= len(ctrl_coords) <= 4, "control count must be in 1..4"
        block = {
            "bbox": (r0, c0, r1, c1),
            "coords": coords,
            "ctrl_coords": ctrl_coords.astype(int),
        }
        blocks.append(block)
    blocks.sort(key=lambda b: (b["bbox"][0], b["bbox"][1], b["bbox"][2], b["bbox"][3]))
    for idx, block in enumerate(blocks):
        rr, cc = block["coords"][:, 0], block["coords"][:, 1]
        labels[rr, cc] = idx
        block["id"] = idx
        block["direction"] = _direction_from_bias(block["coords"], block["ctrl_coords"])
        block["length"] = int(len(block["ctrl_coords"]))
        block["start"] = _ray_start(block, block["direction"])
    return blocks, labels


def _trace_ray(
    grid: np.ndarray,
    labels: np.ndarray,
    block_id: int,
    start: tuple[int, int],
    direction: tuple[int, int],
    length: int,
) -> tuple[list[tuple[int, int]], dict]:
    h, w = grid.shape
    r, c = map(int, start)
    dr, dc = map(int, direction)
    steps_left = int(length)
    bounced = False
    hit_other = False
    cells: list[tuple[int, int]] = []

    while steps_left > 0 and 0 <= r < h and 0 <= c < w:
        if int(grid[r, c]) != BG:
            break
        cells.append((r, c))
        steps_left -= 1
        if steps_left == 0:
            break

        nr, nc = r + dr, c + dc
        if not (0 <= nr < h and 0 <= nc < w):
            break
        if int(grid[nr, nc]) == BG:
            r, c = nr, nc
            continue

        if not bounced and int(labels[nr, nc]) != block_id:
            bounced = True
            hit_other = True
            dr, dc = -dr, -dc
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                break
            if int(grid[nr, nc]) != BG:
                break
            r, c = nr, nc
            continue
        break

    return cells, {"bounced": hit_other, "remaining": steps_left}


def _solve(grid: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    blocks, labels = _extract_blocks(grid)
    out = np.asarray(grid, dtype=int).copy()
    traces: list[dict] = []
    for block in blocks:
        cells, info = _trace_ray(
            grid,
            labels,
            int(block["id"]),
            tuple(block["start"]),
            tuple(block["direction"]),
            int(block["length"]),
        )
        for r, c in cells:
            out[r, c] = RAY
        traces.append(
            {
                "id": int(block["id"]),
                "bbox": tuple(block["bbox"]),
                "direction": tuple(block["direction"]),
                "length": int(block["length"]),
                "cells": tuple(cells),
                "bounced": bool(info["bounced"]),
            }
        )
    return out, traces


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cast one green ray from each yellow block, using its red controls to encode direction and travel distance.

    Input structure:
    - The grid is a black background with several small solid rectangular blocks.
    - Each block is filled with yellow except for 1-4 embedded red control pixels.
    - Blocks are separated, axis-aligned, and remain fixed in place.

    Rule:
    - For each block, count its red control pixels; that count is the ray length.
    - Compare the centroid of the red pixels with the centroid of the whole block, reverse that offset, and quantize it to one of the four cardinal directions; ties go horizontally.
    - Start just outside the block on that side and draw one green pixel per step through background cells only.
    - If the next step would enter a different block before the length is spent, reverse direction once and continue with the remaining steps; if another block is met after the reversal, stop there. Hitting the grid boundary also stops the ray.

    Key insight:
    - The same red control set is decoded twice: its population gives the ray length, while its biased placement gives the opposite-facing ray direction.
    """
    solved, _ = _solve(np.asarray(grid, dtype=int))
    return solved


def _pattern_catalog() -> dict[tuple[int, int, tuple[int, int], int], list[tuple[tuple[int, int], ...]]]:
    catalog: dict[tuple[int, int, tuple[int, int], int], list[tuple[tuple[int, int], ...]]] = {}
    for h, w in SHAPES:
        block_coords = np.asarray([(r, c) for r in range(h) for c in range(w)], dtype=float)
        cells = [(r, c) for r in range(h) for c in range(w)]
        center = np.array([(h - 1) / 2.0, (w - 1) / 2.0], dtype=float)
        for direction in _compatible_dirs(h, w):
            for n in LENGTHS:
                scored: list[tuple[float, tuple[tuple[int, int], ...]]] = []
                for combo in itertools.combinations(cells, n):
                    ctrl = np.asarray(combo, dtype=float)
                    actual = _maybe_direction_from_bias(block_coords, ctrl)
                    if actual != direction:
                        continue
                    bias = ctrl.mean(axis=0) - center
                    if direction == (0, -1):
                        score = float(bias[1] - 0.35 * abs(bias[0]))
                    elif direction == (0, 1):
                        score = float(-bias[1] - 0.35 * abs(bias[0]))
                    elif direction == (-1, 0):
                        score = float(bias[0] - 0.35 * abs(bias[1]))
                    else:
                        score = float(-bias[0] - 0.35 * abs(bias[1]))
                    if score < 0.45:
                        continue
                    scored.append((score, tuple(sorted((int(r), int(c)) for r, c in combo))))
                scored.sort(key=lambda item: (-item[0], item[1]))
                unique: list[tuple[tuple[int, int], ...]] = []
                seen = set()
                for _score, combo in scored:
                    if combo not in seen:
                        seen.add(combo)
                        unique.append(combo)
                assert unique, f"no patterns for {(h, w, direction, n)}"
                catalog[(h, w, direction, n)] = unique[:80]
    return catalog


PATTERNS = _pattern_catalog()


def _shape_for_direction(direction: tuple[int, int]) -> tuple[int, int]:
    if direction[0] == 0:
        return random.choice([(3, 3), (3, 4)])
    return random.choice([(3, 3), (4, 3)])


def _choose_pattern(h: int, w: int, direction: tuple[int, int], length: int) -> tuple[tuple[int, int], ...]:
    return random.choice(PATTERNS[(h, w, direction, length)])


def _can_place(bbox: tuple[int, int, int, int], placed: list[dict]) -> bool:
    return all(_rect_distance(bbox, other["bbox"]) >= MIN_BLOCK_DISTANCE for other in placed)


def _add_block(placed: list[dict], r0: int, c0: int, h: int, w: int) -> dict:
    block = {"bbox": (r0, c0, r0 + h - 1, c0 + w - 1), "h": h, "w": w}
    placed.append(block)
    return block


def _place_bounce_pair(h: int, w: int) -> tuple[list[dict], int, tuple[int, int]] | None:
    for _ in range(PAIR_ATTEMPTS):
        direction = random.choice(DIRS)
        sh, sw = _shape_for_direction(direction)
        oh, ow = random.choice(SHAPES)
        blocks: list[dict] = []

        if direction == (0, 1):
            max_sr = h - sh
            max_sc = w - (sw + 3 + ow)
            if max_sr < 0 or max_sc < 0:
                continue
            sr = random.randint(0, max_sr)
            sc = random.randint(0, max_sc)
            ray_row = sr + sh // 2
            min_or = max(0, ray_row - oh + 1)
            max_or = min(ray_row, h - oh)
            if min_or > max_or:
                continue
            orow = random.randint(min_or, max_or)
            ocol = sc + sw + 3
        elif direction == (0, -1):
            max_sr = h - sh
            min_sc = ow + 3
            max_sc = w - sw
            if max_sr < 0 or min_sc > max_sc:
                continue
            sr = random.randint(0, max_sr)
            sc = random.randint(min_sc, max_sc)
            ray_row = sr + sh // 2
            min_or = max(0, ray_row - oh + 1)
            max_or = min(ray_row, h - oh)
            if min_or > max_or:
                continue
            orow = random.randint(min_or, max_or)
            ocol = sc - ow - 3
        elif direction == (1, 0):
            max_sr = h - (sh + 3 + oh)
            max_sc = w - sw
            if max_sr < 0 or max_sc < 0:
                continue
            sr = random.randint(0, max_sr)
            sc = random.randint(0, max_sc)
            ray_col = sc + sw // 2
            min_oc = max(0, ray_col - ow + 1)
            max_oc = min(ray_col, w - ow)
            if min_oc > max_oc:
                continue
            orow = sr + sh + 3
            ocol = random.randint(min_oc, max_oc)
        else:
            min_sr = oh + 3
            max_sr = h - sh
            max_sc = w - sw
            if min_sr > max_sr or max_sc < 0:
                continue
            sr = random.randint(min_sr, max_sr)
            sc = random.randint(0, max_sc)
            ray_col = sc + sw // 2
            min_oc = max(0, ray_col - ow + 1)
            max_oc = min(ray_col, w - ow)
            if min_oc > max_oc:
                continue
            orow = sr - oh - 3
            ocol = random.randint(min_oc, max_oc)

        src = {"bbox": (sr, sc, sr + sh - 1, sc + sw - 1), "h": sh, "w": sw}
        obs = {"bbox": (orow, ocol, orow + oh - 1, ocol + ow - 1), "h": oh, "w": ow}
        if _rect_distance(src["bbox"], obs["bbox"]) != MIN_BLOCK_DISTANCE:
            continue
        blocks.extend([src, obs])
        return blocks, 0, direction
    return None


def _place_random_blocks(h: int, w: int, count: int, placed: list[dict], need_vertical: bool, need_horizontal: bool) -> bool:
    shape_pool: list[tuple[int, int]] = []
    if need_horizontal:
        shape_pool.append((3, 4))
    if need_vertical:
        shape_pool.append((4, 3))
    while len(shape_pool) < count:
        shape_pool.append(random.choice(SHAPES))
    random.shuffle(shape_pool)

    for shape in shape_pool[:count]:
        bh, bw = shape
        candidates: list[tuple[int, int]] = []
        for r0 in range(h - bh + 1):
            for c0 in range(w - bw + 1):
                bbox = (r0, c0, r0 + bh - 1, c0 + bw - 1)
                if _can_place(bbox, placed):
                    candidates.append((r0, c0))
        if not candidates:
            return False
        r0, c0 = random.choice(candidates)
        _add_block(placed, r0, c0, bh, bw)
    return True


def _placements() -> tuple[int, int, list[dict], int, tuple[int, int]] | None:
    h = random.randint(*GRID_ROWS)
    w = random.randint(*GRID_COLS)
    count = random.randint(*BLOCK_COUNT)
    pair = _place_bounce_pair(h, w)
    if pair is None:
        return None
    placed, bounce_src, bounce_dir = pair
    need_vertical = bounce_dir[0] == 0
    need_horizontal = bounce_dir[1] == 0
    if not _place_random_blocks(h, w, count - 2, placed, need_vertical, need_horizontal):
        return None
    bounce_bbox = placed[bounce_src]["bbox"]
    placed.sort(key=lambda b: (b["bbox"][0], b["bbox"][1], b["bbox"][2], b["bbox"][3]))
    bounce_src = next(i for i, b in enumerate(placed) if b["bbox"] == bounce_bbox)
    for idx, block in enumerate(placed):
        block["id"] = idx
        block["start_by_dir"] = {d: _ray_start(block, d) for d in _compatible_dirs(block["h"], block["w"])}
    return h, w, placed, bounce_src, bounce_dir


def _occupancy(placed: list[dict], shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    grid = np.zeros(shape, dtype=int)
    labels = np.full(shape, -1, dtype=int)
    for block in placed:
        r0, c0, r1, c1 = block["bbox"]
        grid[r0 : r1 + 1, c0 : c1 + 1] = BLOCK
        labels[r0 : r1 + 1, c0 : c1 + 1] = int(block["id"])
    return grid, labels


def _options_for_block(block: dict, grid: np.ndarray, labels: np.ndarray) -> list[dict]:
    opts: list[dict] = []
    for direction in _compatible_dirs(block["h"], block["w"]):
        start = block["start_by_dir"][direction]
        for length in LENGTHS:
            cells, info = _trace_ray(grid, labels, int(block["id"]), start, direction, length)
            if not cells:
                continue
            opts.append(
                {
                    "direction": direction,
                    "length": int(length),
                    "cells": tuple(cells),
                    "bounced": bool(info["bounced"]),
                }
            )
    return opts


def _choose_assignments(
    placed: list[dict],
    grid: np.ndarray,
    labels: np.ndarray,
    bounce_src: int,
    bounce_dir: tuple[int, int],
) -> dict[int, dict] | None:
    options = {int(block["id"]): _options_for_block(block, grid, labels) for block in placed}
    forced = [
        opt
        for opt in options[bounce_src]
        if opt["direction"] == bounce_dir and opt["length"] == 4 and opt["bounced"]
    ]
    if not forced:
        return None

    ids = [int(block["id"]) for block in placed]
    for _ in range(ASSIGN_ATTEMPTS):
        chosen: dict[int, dict] = {bounce_src: random.choice(forced)}
        for block_id in ids:
            if block_id == bounce_src:
                continue
            pool = options[block_id]
            if not pool:
                break
            chosen[block_id] = random.choice(pool)
        if len(chosen) != len(ids):
            continue

        dirs = {tuple(opt["direction"]) for opt in chosen.values()}
        lengths = {int(opt["length"]) for opt in chosen.values()}
        bounce_count = sum(int(opt["bounced"]) for opt in chosen.values())
        if len(dirs) < 2 or len(lengths) < 2 or bounce_count < 1 or bounce_count == len(ids):
            continue
        if all(int(opt["length"]) == 4 for opt in chosen.values()):
            continue

        cell_counts: dict[tuple[int, int], int] = {}
        for opt in chosen.values():
            for cell in opt["cells"]:
                cell_counts[cell] = cell_counts.get(cell, 0) + 1
        overlap = sum(v - 1 for v in cell_counts.values() if v > 1)
        if overlap > 2:
            continue
        if sum(len(opt["cells"]) for opt in chosen.values()) < len(ids) + 4:
            continue
        return chosen
    return None


def _render_input(shape: tuple[int, int], placed: list[dict], chosen: dict[int, dict]) -> tuple[np.ndarray, list[dict]]:
    inp = np.zeros(shape, dtype=int)
    plan: list[dict] = []
    for block in placed:
        r0, c0, r1, c1 = block["bbox"]
        inp[r0 : r1 + 1, c0 : c1 + 1] = BLOCK
        pick = chosen[int(block["id"])]
        pattern = _choose_pattern(block["h"], block["w"], tuple(pick["direction"]), int(pick["length"]))
        for pr, pc in pattern:
            inp[r0 + pr, c0 + pc] = CTRL
        plan.append(
            {
                "bbox": tuple(block["bbox"]),
                "direction": tuple(pick["direction"]),
                "length": int(pick["length"]),
                "cells": tuple(pick["cells"]),
                "bounced": bool(pick["bounced"]),
            }
        )
    plan.sort(key=lambda item: (item["bbox"][0], item["bbox"][1], item["bbox"][2], item["bbox"][3]))
    return inp, plan


def _semantic_accepts(inp: np.ndarray, plan: list[dict]) -> bool:
    try:
        solved, traces = _solve(inp)
    except Exception:
        return False

    if len(traces) != len(plan):
        return False
    for got, want in zip(traces, plan):
        if tuple(got["bbox"]) != tuple(want["bbox"]):
            return False
        if tuple(got["direction"]) != tuple(want["direction"]):
            return False
        if int(got["length"]) != int(want["length"]):
            return False
        if tuple(got["cells"]) != tuple(want["cells"]):
            return False
        if bool(got["bounced"]) != bool(want["bounced"]):
            return False

    ray_mask = solved == RAY
    if np.any(inp[ray_mask] != BG):
        return False

    if len({p["direction"] for p in plan}) < 2:
        return False
    if len({p["length"] for p in plan}) < 2:
        return False
    if not any(p["bounced"] for p in plan):
        return False
    if not any(not p["bounced"] for p in plan):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    layout = _placements()
    if layout is None:
        return None
    h, w, placed, bounce_src, bounce_dir = layout
    occ, labels = _occupancy(placed, (h, w))
    chosen = _choose_assignments(placed, occ, labels, bounce_src, bounce_dir)
    if chosen is None:
        return None
    inp, plan = _render_input((h, w), placed, chosen)
    if not _semantic_accepts(inp, plan):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
