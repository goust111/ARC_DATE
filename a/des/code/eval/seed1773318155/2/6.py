import random
from collections import deque

import numpy as np


DIR_VECS = {
    "N": (-1, 0),
    "S": (1, 0),
    "W": (0, -1),
    "E": (0, 1),
}
OPPOSITE = {
    "N": "S",
    "S": "N",
    "W": "E",
    "E": "W",
}

BG_COLORS = (0, 5)
OBJECT_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
GRID_H_RANGE = (22, 31)
GRID_W_RANGE = (22, 31)
PAIR_COUNT_RANGE = (2, 4)
CORE_SHORT_RANGE = (3, 5)
CORE_LONG_RANGE = (3, 7)
GAP_RANGE = (4, 10)
MARGIN = 1
GEN_ATTEMPTS = 400
PAIR_PLACE_ATTEMPTS = 120


def _checkerboard(shape: tuple[int, int], even_color: int, odd_color: int) -> np.ndarray:
    h, w = shape
    rr, cc = np.indices((h, w))
    return np.where(((rr + cc) & 1) == 0, int(even_color), int(odd_color)).astype(int)


def _recover_background(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    colors = [int(c) for c in np.unique(g)]
    assert len(colors) >= 3, "expected checkerboard plus foreground objects"

    rr, cc = np.indices(g.shape)
    even = ((rr + cc) & 1) == 0

    best_score = -1
    best_pair = None
    for i, color_a in enumerate(colors):
        for color_b in colors[i + 1 :]:
            score_ab = int(np.sum(g[even] == color_a) + np.sum(g[~even] == color_b))
            if score_ab > best_score:
                best_score = score_ab
                best_pair = (color_a, color_b)
            score_ba = int(np.sum(g[even] == color_b) + np.sum(g[~even] == color_a))
            if score_ba > best_score:
                best_score = score_ba
                best_pair = (color_b, color_a)

    assert best_pair is not None
    return _checkerboard(g.shape, best_pair[0], best_pair[1])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((int(rr), int(cc)))
                for dr, dc in DIR_VECS.values():
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _decode_object(mask: np.ndarray) -> tuple[str, tuple[int, int], tuple[int, int]]:
    h, w = mask.shape
    candidates: list[tuple[str, tuple[int, int], tuple[int, int]]] = []

    if w >= 2:
        core = mask[:, :-1]
        extra_rows = np.flatnonzero(mask[:, -1])
        missing = np.argwhere(~core)
        if extra_rows.size == 1 and missing.shape == (1, 2) and int(missing[0, 1]) == 0:
            candidates.append(("E", (int(extra_rows[0]), w - 1), (int(missing[0, 0]), 0)))

        core = mask[:, 1:]
        extra_rows = np.flatnonzero(mask[:, 0])
        missing = np.argwhere(~core)
        if (
            extra_rows.size == 1
            and missing.shape == (1, 2)
            and int(missing[0, 1]) == core.shape[1] - 1
        ):
            candidates.append(("W", (int(extra_rows[0]), 0), (int(missing[0, 0]), w - 1)))

    if h >= 2:
        core = mask[:-1, :]
        extra_cols = np.flatnonzero(mask[-1, :])
        missing = np.argwhere(~core)
        if extra_cols.size == 1 and missing.shape == (1, 2) and int(missing[0, 0]) == 0:
            candidates.append(("S", (h - 1, int(extra_cols[0])), (0, int(missing[0, 1]))))

        core = mask[1:, :]
        extra_cols = np.flatnonzero(mask[0, :])
        missing = np.argwhere(~core)
        if (
            extra_cols.size == 1
            and missing.shape == (1, 2)
            and int(missing[0, 0]) == core.shape[0] - 1
        ):
            candidates.append(("N", (0, int(extra_cols[0])), (h - 1, int(missing[0, 1]))))

    assert len(candidates) == 1, "each object must have one protrusion and one opposite notch"
    return candidates[0]


def _extract_objects(grid: np.ndarray, bg: np.ndarray) -> list[dict]:
    fg_mask = grid != bg
    comps = _components(fg_mask)
    assert comps, "expected at least one foreground object"

    objects: list[dict] = []
    for idx, coords in enumerate(comps):
        rows = [r for r, _ in coords]
        cols = [c for _, c in coords]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        submask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        rel_coords: list[tuple[int, int]] = []
        colors = {int(grid[r, c]) for r, c in coords}
        assert len(colors) == 1, "foreground objects must be monochrome"
        for r, c in coords:
            rr = int(r - r0)
            cc = int(c - c0)
            submask[rr, cc] = True
            rel_coords.append((rr, cc))
        direction, protrusion_rel, indent_rel = _decode_object(submask)
        objects.append(
            {
                "idx": idx,
                "color": int(next(iter(colors))),
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "rel_coords": rel_coords,
                "direction": direction,
                "indent_direction": OPPOSITE[direction],
                "protrusion": (int(r0 + protrusion_rel[0]), int(c0 + protrusion_rel[1])),
                "indent": (int(r0 + indent_rel[0]), int(c0 + indent_rel[1])),
            }
        )
    return objects


def _candidate_distance(source: dict, target: dict) -> int | None:
    if target["indent_direction"] != OPPOSITE[source["direction"]]:
        return None

    sr, sc = source["protrusion"]
    tr, tc = target["indent"]
    direction = source["direction"]

    if direction == "E":
        if sr != tr:
            return None
        dist = int(tc - sc)
    elif direction == "W":
        if sr != tr:
            return None
        dist = int(sc - tc)
    elif direction == "S":
        if sc != tc:
            return None
        dist = int(tr - sr)
    else:
        if sc != tc:
            return None
        dist = int(sr - tr)

    return dist if dist > 0 else None


def _match_pairs(objects: list[dict]) -> list[tuple[int, int, int]]:
    edges: list[tuple[int, int, int]] = []
    for source in objects:
        candidates = []
        for target in objects:
            if source["idx"] == target["idx"]:
                continue
            dist = _candidate_distance(source, target)
            if dist is not None:
                candidates.append((dist, source["idx"], target["idx"]))
        if not candidates:
            continue
        candidates.sort()
        assert len(candidates) == 1 or candidates[0][0] < candidates[1][0], "pairing must be unambiguous"
        edges.append(candidates[0])

    edges.sort()
    used: set[int] = set()
    pairs: list[tuple[int, int, int]] = []
    for dist, src_idx, tgt_idx in edges:
        if src_idx in used or tgt_idx in used:
            continue
        used.add(src_idx)
        used.add(tgt_idx)
        pairs.append((src_idx, tgt_idx, dist))
    return pairs


def _paint_object(out: np.ndarray, obj: dict, top: int, left: int) -> None:
    color = int(obj["color"])
    for rr, cc in obj["rel_coords"]:
        out[top + rr, left + cc] = color


def _solve(grid: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int, int]], list[dict], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    bg = _recover_background(g)
    objects = _extract_objects(g, bg)
    pairs = _match_pairs(objects)

    out = bg.copy()
    used: set[int] = set()
    by_idx = {obj["idx"]: obj for obj in objects}

    for src_idx, tgt_idx, dist in pairs:
        source = by_idx[src_idx]
        target = by_idx[tgt_idx]
        dr, dc = DIR_VECS[source["direction"]]
        source_steps = (dist + 1) // 2
        target_steps = dist // 2
        sr0, sc0, _, _ = source["bbox"]
        tr0, tc0, _, _ = target["bbox"]
        _paint_object(out, source, sr0 + dr * source_steps, sc0 + dc * source_steps)
        _paint_object(out, target, tr0 - dr * target_steps, tc0 - dc * target_steps)
        used.add(src_idx)
        used.add(tgt_idx)

    for obj in objects:
        if obj["idx"] not in used:
            r0, c0, _, _ = obj["bbox"]
            _paint_object(out, obj, r0, c0)
    return out, pairs, objects, bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each jigsaw-like colored object pair toward the midpoint so the protrusion
    of the rear object plugs into the notch of the front object, preserving the
    checkerboard background and each object's color.
    """
    return _solve(np.asarray(grid, dtype=int))[0]


def _make_shape(direction: str, core_h: int, core_w: int, slot: int) -> np.ndarray:
    if direction == "E":
        arr = np.zeros((core_h, core_w + 1), dtype=bool)
        arr[:, :core_w] = True
        arr[slot, 0] = False
        arr[slot, core_w] = True
        return arr
    if direction == "W":
        arr = np.zeros((core_h, core_w + 1), dtype=bool)
        arr[:, 1:] = True
        arr[slot, core_w] = False
        arr[slot, 0] = True
        return arr
    if direction == "S":
        arr = np.zeros((core_h + 1, core_w), dtype=bool)
        arr[:core_h, :] = True
        arr[0, slot] = False
        arr[core_h, slot] = True
        return arr
    arr = np.zeros((core_h + 1, core_w), dtype=bool)
    arr[1:, :] = True
    arr[core_h, slot] = False
    arr[0, slot] = True
    return arr


def _shape_dims(direction: str) -> tuple[int, int]:
    if direction in ("E", "W"):
        return (
            random.randint(CORE_SHORT_RANGE[0], CORE_SHORT_RANGE[1]),
            random.randint(CORE_LONG_RANGE[0], CORE_LONG_RANGE[1]),
        )
    return (
        random.randint(CORE_LONG_RANGE[0], CORE_LONG_RANGE[1]),
        random.randint(CORE_SHORT_RANGE[0], CORE_SHORT_RANGE[1]),
    )


def _region_clear(mask: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int = 1) -> bool:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(mask.shape[0] - 1, r1 + pad)
    cc1 = min(mask.shape[1] - 1, c1 + pad)
    return not bool(mask[rr0 : rr1 + 1, cc0 : cc1 + 1].any())


def _mark_region(mask: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int = 1) -> None:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(mask.shape[0] - 1, r1 + pad)
    cc1 = min(mask.shape[1] - 1, c1 + pad)
    mask[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _stamp_shape(grid: np.ndarray, mask: np.ndarray, color: int, top: int, left: int) -> None:
    rows, cols = np.where(mask)
    grid[top + rows, left + cols] = int(color)


def _place_pair(
    grid: np.ndarray,
    occupied: np.ndarray,
    direction: str,
    pair_colors: tuple[int, int],
    used_h_rows: set[int],
    used_v_cols: set[int],
) -> bool:
    h, w = grid.shape
    color_a, color_b = pair_colors

    for _ in range(PAIR_PLACE_ATTEMPTS):
        src_core_h, src_core_w = _shape_dims(direction)
        tgt_core_h, tgt_core_w = _shape_dims(direction)
        if direction in ("E", "W"):
            max_slot = min(src_core_h, tgt_core_h) - 2
        else:
            max_slot = min(src_core_w, tgt_core_w) - 2
        if max_slot < 1:
            continue
        slot = random.randint(1, max_slot)
        source_mask = _make_shape(direction, src_core_h, src_core_w, slot)
        target_mask = _make_shape(direction, tgt_core_h, tgt_core_w, slot)
        gap = random.randint(GAP_RANGE[0], GAP_RANGE[1] - 1)

        if direction == "E":
            pair_h = max(source_mask.shape[0], target_mask.shape[0])
            pair_w = source_mask.shape[1] + gap + target_mask.shape[1] - 1
            row_low = MARGIN + slot
            row_high = h - MARGIN - (pair_h - 1 - slot)
            if row_low > row_high:
                continue
            join_row = random.randint(row_low, row_high)
            if join_row in used_h_rows:
                continue
            top = join_row - slot
            left_low = MARGIN
            left_high = w - MARGIN - pair_w
            if left_low > left_high:
                continue
            source_left = random.randint(left_low, left_high)
            target_left = source_left + source_mask.shape[1] - 1 + gap
            bbox = (
                top,
                source_left,
                top + pair_h - 1,
                source_left + pair_w - 1,
            )
            source_top = top
            target_top = top
        elif direction == "W":
            pair_h = max(source_mask.shape[0], target_mask.shape[0])
            pair_w = target_mask.shape[1] + gap + source_mask.shape[1] - 1
            row_low = MARGIN + slot
            row_high = h - MARGIN - (pair_h - 1 - slot)
            if row_low > row_high:
                continue
            join_row = random.randint(row_low, row_high)
            if join_row in used_h_rows:
                continue
            top = join_row - slot
            left_low = MARGIN
            left_high = w - MARGIN - pair_w
            if left_low > left_high:
                continue
            target_left = random.randint(left_low, left_high)
            source_left = target_left + target_mask.shape[1] - 1 + gap
            bbox = (
                top,
                target_left,
                top + pair_h - 1,
                target_left + pair_w - 1,
            )
            source_top = top
            target_top = top
        elif direction == "S":
            pair_h = source_mask.shape[0] + gap + target_mask.shape[0] - 1
            pair_w = max(source_mask.shape[1], target_mask.shape[1])
            col_low = MARGIN + slot
            col_high = w - MARGIN - (pair_w - 1 - slot)
            if col_low > col_high:
                continue
            join_col = random.randint(col_low, col_high)
            if join_col in used_v_cols:
                continue
            left = join_col - slot
            top_low = MARGIN
            top_high = h - MARGIN - pair_h
            if top_low > top_high:
                continue
            source_top = random.randint(top_low, top_high)
            target_top = source_top + source_mask.shape[0] - 1 + gap
            bbox = (
                source_top,
                left,
                source_top + pair_h - 1,
                left + pair_w - 1,
            )
            source_left = left
            target_left = left
        else:
            pair_h = target_mask.shape[0] + gap + source_mask.shape[0] - 1
            pair_w = max(source_mask.shape[1], target_mask.shape[1])
            col_low = MARGIN + slot
            col_high = w - MARGIN - (pair_w - 1 - slot)
            if col_low > col_high:
                continue
            join_col = random.randint(col_low, col_high)
            if join_col in used_v_cols:
                continue
            left = join_col - slot
            top_low = MARGIN
            top_high = h - MARGIN - pair_h
            if top_low > top_high:
                continue
            target_top = random.randint(top_low, top_high)
            source_top = target_top + target_mask.shape[0] - 1 + gap
            bbox = (
                target_top,
                left,
                target_top + pair_h - 1,
                left + pair_w - 1,
            )
            source_left = left
            target_left = left

        if not _region_clear(occupied, *bbox, pad=1):
            continue

        _stamp_shape(grid, source_mask, color_a, source_top, source_left)
        _stamp_shape(grid, target_mask, color_b, target_top, target_left)
        _mark_region(occupied, *bbox, pad=1)
        if direction in ("E", "W"):
            used_h_rows.add(join_row)
        else:
            used_v_cols.add(join_col)
        return True
    return False


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GRID_H_RANGE[0], GRID_H_RANGE[1] - 1)
    w = random.randint(GRID_W_RANGE[0], GRID_W_RANGE[1] - 1)
    grid = _checkerboard((h, w), BG_COLORS[0], BG_COLORS[1])
    occupied = np.zeros((h, w), dtype=bool)

    pair_count = random.randint(PAIR_COUNT_RANGE[0], PAIR_COUNT_RANGE[1])
    directions = random.sample(list(DIR_VECS.keys()), k=pair_count)
    colors = random.sample(OBJECT_COLORS, k=2 * pair_count)
    used_h_rows: set[int] = set()
    used_v_cols: set[int] = set()

    for i, direction in enumerate(directions):
        if not _place_pair(
            grid,
            occupied,
            direction,
            (colors[2 * i], colors[2 * i + 1]),
            used_h_rows,
            used_v_cols,
        ):
            return None

    try:
        out, pairs, objects, bg = _solve(grid)
    except AssertionError:
        return None

    if len(objects) != 2 * pair_count or len(pairs) != pair_count:
        return None
    used = {idx for src, tgt, _ in pairs for idx in (src, tgt)}
    if len(used) != len(objects):
        return None
    if np.array_equal(out, bg) or np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
