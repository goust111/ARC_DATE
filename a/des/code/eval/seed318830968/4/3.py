import random
from collections import defaultdict, deque

import numpy as np


BG = 8
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 9)
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIRS8 = DIRS4 + ((1, 1), (1, -1), (-1, 1), (-1, -1))
ARROW_DIRS = ("right", "left", "down", "up")
MAX_GENERATION_ATTEMPTS = 300


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _component_coords(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    neigh = DIRS4 if connectivity == 4 else DIRS8
    components: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                coords.append((cr, cc))
                for dr, dc in neigh:
                    nr, nc = cr + dr, cc + dc
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _enclosed_mask(frame_mask: np.ndarray) -> np.ndarray:
    h, w = frame_mask.shape
    open_mask = ~frame_mask
    seen = np.zeros_like(frame_mask, dtype=bool)
    queue: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if open_mask[r, c] and not seen[r, c]:
                seen[r, c] = True
                queue.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if open_mask[r, c] and not seen[r, c]:
                seen[r, c] = True
                queue.append((r, c))

    while queue:
        r, c = queue.popleft()
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if _in_bounds(frame_mask.shape, nr, nc) and open_mask[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                queue.append((nr, nc))

    return open_mask & ~seen


def _build_arrow(direction: str, tip: tuple[int, int], length: int) -> dict[str, object]:
    r, c = tip
    if direction == "right":
        shaft = [(r, cc) for cc in range(c - length, c)]
        head = [(r - 1, c), (r + 1, c)]
        tail = (r, c - length - 1)
        delta = (0, 1)
    elif direction == "left":
        shaft = [(r, cc) for cc in range(c + 1, c + length + 1)]
        head = [(r - 1, c), (r + 1, c)]
        tail = (r, c + length + 1)
        delta = (0, -1)
    elif direction == "down":
        shaft = [(rr, c) for rr in range(r - length, r)]
        head = [(r, c - 1), (r, c + 1)]
        tail = (r - length - 1, c)
        delta = (1, 0)
    else:
        shaft = [(rr, c) for rr in range(r + 1, r + length + 1)]
        head = [(r, c - 1), (r, c + 1)]
        tail = (r + length + 1, c)
        delta = (-1, 0)

    grey_cells = shaft + head
    cells = grey_cells + [tail]
    return {
        "direction": direction,
        "tip": tip,
        "length": int(length),
        "shaft": shaft,
        "head": head,
        "grey_cells": grey_cells,
        "tail": tail,
        "cells": cells,
        "delta": delta,
    }


def _detect_arrows(grid: np.ndarray) -> list[dict[str, object]]:
    h, w = grid.shape
    found: dict[tuple[tuple[tuple[int, int], ...], tuple[int, int]], dict[str, object]] = {}

    for direction in ARROW_DIRS:
        for length in range(3, 6):
            for r in range(h):
                for c in range(w):
                    arrow = _build_arrow(direction, (r, c), length)
                    cells = arrow["cells"]
                    grey_cells = arrow["grey_cells"]
                    tail = arrow["tail"]

                    if not all(_in_bounds(grid.shape, rr, cc) for rr, cc in cells):
                        continue
                    if int(grid[r, c]) != BG:
                        continue
                    if any(int(grid[rr, cc]) != GREY for rr, cc in grey_cells):
                        continue
                    tail_color = int(grid[tail])
                    if tail_color in {BG, GREY}:
                        continue

                    grey_nbrs = 0
                    for dr, dc in DIRS4:
                        nr, nc = tail[0] + dr, tail[1] + dc
                        if _in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) == GREY:
                            grey_nbrs += 1
                    if grey_nbrs != 1:
                        continue

                    grey_key = tuple(sorted(grey_cells))
                    found[(grey_key, tail)] = {**arrow, "color": tail_color}

    arrows = list(found.values())
    used: set[tuple[int, int]] = set()
    for arrow in arrows:
        overlap = used & set(arrow["cells"])
        assert not overlap, "arrow indicators must be disjoint"
        used.update(arrow["cells"])
    assert arrows, "expected at least one arrow"
    return arrows


def _find_shapes(grid: np.ndarray, arrows: list[dict[str, object]]) -> tuple[list[dict[str, object]], np.ndarray]:
    grey_mask = grid == GREY
    for arrow in arrows:
        for r, c in arrow["grey_cells"]:
            grey_mask[r, c] = False

    comps = _component_coords(grey_mask, connectivity=8)
    assert comps, "expected grey closed shapes"
    shape_id_grid = np.full(grid.shape, -1, dtype=int)
    shapes: list[dict[str, object]] = []

    for shape_id, coords in enumerate(comps):
        r0, c0, r1, c1 = _bbox(coords)
        frame_mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        frame_mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
        enclosed = _enclosed_mask(frame_mask)
        assert np.any(enclosed), "shape frame must enclose at least one cell"

        window = grid[r0 : r1 + 1, c0 : c1 + 1]
        pattern_mask = enclosed & (window != BG) & (window != GREY)
        colors = np.unique(window[pattern_mask])
        assert colors.size == 1, "each shape must contain exactly one pattern color"

        enclosed_rows = sorted({int(r) for r, _c in np.argwhere(enclosed)})
        enclosed_cols = sorted({int(c) for _r, c in np.argwhere(enclosed)})
        for r, c in coords:
            shape_id_grid[r, c] = shape_id

        shapes.append(
            {
                "id": shape_id,
                "bbox": (r0, c0, r1, c1),
                "frame_mask": frame_mask,
                "enclosed": enclosed,
                "pattern_mask": pattern_mask,
                "pattern_color": int(colors[0]),
                "enclosed_rows": enclosed_rows,
                "enclosed_cols": enclosed_cols,
            }
        )

    return shapes, shape_id_grid


def _ray_target(shape_id_grid: np.ndarray, arrow: dict[str, object]) -> int:
    r, c = arrow["tip"]
    dr, dc = arrow["delta"]
    while _in_bounds(shape_id_grid.shape, r, c):
        shape_id = int(shape_id_grid[r, c])
        if shape_id != -1:
            return shape_id
        r += dr
        c += dc
    raise AssertionError("arrow must point to a grey shape")


def _stripe_order_key(arrow: dict[str, object], vertical: bool) -> tuple[int, int]:
    tr, tc = arrow["tail"]
    return (tc, tr) if vertical else (tr, tc)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repaint the interiors of grey closed shapes using arrow-tail colors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert np.any(g == GREY), "expected grey shapes and arrows"
    assert np.any(g == BG), "expected azure background"

    arrows = _detect_arrows(g)
    shapes, shape_id_grid = _find_shapes(g, arrows)
    assert shapes, "expected at least one targetable shape"

    by_target: dict[int, list[dict[str, object]]] = defaultdict(list)
    for arrow in arrows:
        target = _ray_target(shape_id_grid, arrow)
        by_target[target].append(arrow)

    out = g.copy()
    for arrow in arrows:
        for r, c in arrow["cells"]:
            out[r, c] = BG

    for shape in shapes:
        target_arrows = by_target.get(shape["id"], [])
        if not target_arrows:
            continue

        r0, c0, r1, c1 = shape["bbox"]
        window = out[r0 : r1 + 1, c0 : c1 + 1]
        enclosed = shape["enclosed"]

        if len(target_arrows) == 1:
            color = int(target_arrows[0]["color"])
            window[shape["pattern_mask"]] = color
            continue

        vertical = (c1 - c0) >= (r1 - r0)
        ordered = sorted(target_arrows, key=lambda arrow: _stripe_order_key(arrow, vertical))
        colors = [int(arrow["color"]) for arrow in ordered]

        if vertical:
            col_order = {col: idx for idx, col in enumerate(shape["enclosed_cols"])}
            for rr, cc in np.argwhere(enclosed):
                window[rr, cc] = colors[col_order[int(cc)] % len(colors)]
        else:
            row_order = {row: idx for idx, row in enumerate(shape["enclosed_rows"])}
            for rr, cc in np.argwhere(enclosed):
                window[rr, cc] = colors[row_order[int(rr)] % len(colors)]

    return out


def _rect_frame(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _chamfer_frame(h: int, w: int) -> np.ndarray:
    mask = _rect_frame(h, w)
    mask[0, 0] = False
    mask[0, -1] = False
    mask[-1, 0] = False
    mask[-1, -1] = False
    return mask


def _shape_template(rng: random.Random) -> dict[str, object]:
    kind = rng.choice(("rect", "rect", "chamfer"))
    if kind == "rect":
        h = rng.randint(3, 5)
        w = rng.randint(3, 5)
        frame = _rect_frame(h, w)
    else:
        h = rng.randint(4, 5)
        w = rng.randint(4, 5)
        frame = _chamfer_frame(h, w)

    enclosed = _enclosed_mask(frame)
    assert np.any(enclosed)
    pattern = _pattern_mask(enclosed, rng)
    return {"frame": frame, "enclosed": enclosed, "pattern": pattern}


def _pattern_mask(enclosed: np.ndarray, rng: random.Random) -> np.ndarray:
    rr, cc = np.indices(enclosed.shape)
    row_vals = sorted({int(v) for v in rr[enclosed]})
    col_vals = sorted({int(v) for v in cc[enclosed]})
    mid_row = row_vals[len(row_vals) // 2]
    mid_col = col_vals[len(col_vals) // 2]
    max_sum = int(rr.max() + cc.max())

    options = [
        enclosed,
        enclosed & (rr == cc),
        enclosed & ((rr + cc) == max_sum),
        enclosed & (rr == mid_row),
        enclosed & (cc == mid_col),
        enclosed & (((rr + cc) % 2) == 0),
        enclosed & ((rr == mid_row) | (cc == mid_col)),
    ]
    valid = [mask for mask in options if np.any(mask)]
    return valid[rng.randrange(len(valid))].copy()


def _stamp_shape(
    grid: np.ndarray,
    top: int,
    left: int,
    frame: np.ndarray,
    pattern: np.ndarray,
    pattern_color: int,
) -> dict[str, object]:
    h, w = frame.shape
    window = grid[top : top + h, left : left + w]
    window[frame] = GREY
    window[pattern] = pattern_color

    coords = np.argwhere(frame) + np.array([top, left], dtype=int)
    enclosed = _enclosed_mask(frame)
    return {
        "bbox": (top, left, top + h - 1, left + w - 1),
        "frame_coords": {tuple(map(int, coord)) for coord in coords},
        "enclosed": enclosed,
    }


def _can_place(occupied: np.ndarray, top: int, left: int, h: int, w: int, gap: int = 2) -> bool:
    h0, w0 = occupied.shape
    r0 = max(0, top - gap)
    c0 = max(0, left - gap)
    r1 = min(h0, top + h + gap)
    c1 = min(w0, left + w + gap)
    return not np.any(occupied[r0:r1, c0:c1])


def _reserve(occupied: np.ndarray, top: int, left: int, h: int, w: int) -> None:
    occupied[top : top + h, left : left + w] = True


def _shape_line_hits(shape: dict[str, object]) -> dict[str, dict[int, int]]:
    rows_left: dict[int, int] = {}
    rows_right: dict[int, int] = {}
    cols_top: dict[int, int] = {}
    cols_bottom: dict[int, int] = {}
    for r, c in shape["frame_coords"]:
        rows_left[r] = c if r not in rows_left else min(rows_left[r], c)
        rows_right[r] = c if r not in rows_right else max(rows_right[r], c)
        cols_top[c] = r if c not in cols_top else min(cols_top[c], r)
        cols_bottom[c] = r if c not in cols_bottom else max(cols_bottom[c], r)
    return {
        "right": rows_left,
        "left": rows_right,
        "down": cols_top,
        "up": cols_bottom,
    }


def _ray_clear(grid: np.ndarray, direction: str, tip: tuple[int, int], hit: int) -> bool:
    r, c = tip
    if direction == "right":
        return bool(np.all(grid[r, c:hit] == BG))
    if direction == "left":
        return bool(np.all(grid[r, hit + 1 : c + 1] == BG))
    if direction == "down":
        return bool(np.all(grid[r:hit, c] == BG))
    return bool(np.all(grid[hit + 1 : r + 1, c] == BG))


def _candidate_tip_range(hit: int, span: int, direction: str) -> range:
    if direction in {"right", "down"}:
        return range(hit - 1, span - 1, -1)
    return range(hit + 1, span)


def _arrow_candidate(
    grid: np.ndarray,
    shape_id: int,
    direction: str,
    tip: tuple[int, int],
    length: int,
    hit_value: int,
) -> dict[str, object] | None:
    arrow = _build_arrow(direction, tip, length)
    if not all(_in_bounds(grid.shape, r, c) for r, c in arrow["cells"]):
        return None
    if any(int(grid[r, c]) != BG for r, c in arrow["cells"]):
        return None
    if not _ray_clear(grid, direction, tip, hit_value):
        return None

    reserved = set(arrow["cells"])
    if direction == "right":
        reserved.update((tip[0], cc) for cc in range(tip[1], hit_value))
    elif direction == "left":
        reserved.update((tip[0], cc) for cc in range(hit_value + 1, tip[1] + 1))
    elif direction == "down":
        reserved.update((rr, tip[1]) for rr in range(tip[0], hit_value))
    else:
        reserved.update((rr, tip[1]) for rr in range(hit_value + 1, tip[0] + 1))

    return {
        **arrow,
        "target": shape_id,
        "reserved": reserved,
    }


def _enumerate_arrow_candidates(grid: np.ndarray, shapes: list[dict[str, object]]) -> dict[int, list[dict[str, object]]]:
    by_shape: dict[int, list[dict[str, object]]] = defaultdict(list)
    h, w = grid.shape

    for shape_id, shape in enumerate(shapes):
        hits = _shape_line_hits(shape)
        for direction, line_hits in hits.items():
            for axis_value, hit_value in line_hits.items():
                if direction in {"right", "left"}:
                    r = axis_value
                    if not (1 <= r < h - 1):
                        continue
                    span = 6 if direction == "right" else w - 6
                    tips = _candidate_tip_range(hit_value, span, direction)
                    for tip_c in tips:
                        for length in range(3, 6):
                            cand = _arrow_candidate(grid, shape_id, direction, (r, tip_c), length, hit_value)
                            if cand is not None:
                                by_shape[shape_id].append(cand)
                else:
                    c = axis_value
                    if not (1 <= c < w - 1):
                        continue
                    span = 6 if direction == "down" else h - 6
                    tips = _candidate_tip_range(hit_value, span, direction)
                    for tip_r in tips:
                        for length in range(3, 6):
                            cand = _arrow_candidate(grid, shape_id, direction, (tip_r, c), length, hit_value)
                            if cand is not None:
                                by_shape[shape_id].append(cand)

    for shape_id, cands in by_shape.items():
        uniq: dict[tuple, dict[str, object]] = {}
        for cand in cands:
            key = (cand["direction"], cand["tip"], cand["length"], cand["tail"])
            uniq[key] = cand
        by_shape[shape_id] = list(uniq.values())
    return by_shape


def _stripe_vertical(shape: dict[str, object]) -> bool:
    r0, c0, r1, c1 = shape["bbox"]
    return (c1 - c0) >= (r1 - r0)


def _shape_stripeable(shape: dict[str, object]) -> bool:
    if _stripe_vertical(shape):
        return len(shape["enclosed"].any(axis=0).nonzero()[0]) >= 2
    return len(shape["enclosed"].any(axis=1).nonzero()[0]) >= 2


def _compatible(selection: list[dict[str, object]], cand: dict[str, object]) -> bool:
    reserved = cand["reserved"]
    return all(not (reserved & other["reserved"]) for other in selection)


def _pick_multi_arrows(
    rng: random.Random,
    shape: dict[str, object],
    candidates: list[dict[str, object]],
) -> list[dict[str, object]] | None:
    if not _shape_stripeable(shape):
        return None

    vertical = _stripe_vertical(shape)
    feasible: list[list[dict[str, object]]] = []
    target_count = rng.choice((2, 2, 3))
    pool = candidates[:]
    rng.shuffle(pool)

    for count in (target_count, 2):
        if len(pool) < count:
            continue
        for first_idx in range(len(pool)):
            picked = [pool[first_idx]]
            used_keys = {_stripe_order_key(pool[first_idx], vertical)}
            for cand in pool[first_idx + 1 :]:
                if len(picked) == count:
                    break
                key = _stripe_order_key(cand, vertical)
                if key in used_keys:
                    continue
                if _compatible(picked, cand):
                    picked.append(cand)
                    used_keys.add(key)
            if len(picked) == count:
                feasible.append(picked)
        if feasible:
            break

    return rng.choice(feasible) if feasible else None


def _choose_arrows(
    rng: random.Random,
    shapes: list[dict[str, object]],
    candidates_by_shape: dict[int, list[dict[str, object]]],
) -> list[dict[str, object]] | None:
    selection: list[dict[str, object]] = []
    shape_ids = list(range(len(shapes)))
    rng.shuffle(shape_ids)
    multi_mode = rng.random() < 0.5

    if multi_mode:
        multi_options = [
            shape_id
            for shape_id in shape_ids
            if len(candidates_by_shape.get(shape_id, [])) >= 2 and _shape_stripeable(shapes[shape_id])
        ]
        rng.shuffle(multi_options)
        for shape_id in multi_options:
            picked = _pick_multi_arrows(rng, shapes[shape_id], candidates_by_shape[shape_id])
            if picked is not None:
                selection.extend(picked)
                break
        if not selection:
            multi_mode = False

    if not multi_mode:
        available = [shape_id for shape_id in shape_ids if candidates_by_shape.get(shape_id)]
        if not available:
            return None
        rng.shuffle(available)
        wanted = rng.randint(1, min(3, len(available)))
        for shape_id in available:
            pool = candidates_by_shape[shape_id][:]
            rng.shuffle(pool)
            for cand in pool:
                if _compatible(selection, cand):
                    selection.append(cand)
                    break
            if len(selection) >= wanted:
                break

    for shape_id in shape_ids:
        if rng.random() < 0.4:
            pool = candidates_by_shape.get(shape_id, [])[:]
            rng.shuffle(pool)
            for cand in pool:
                if _compatible(selection, cand):
                    selection.append(cand)
                    break

    uniq: dict[tuple[tuple[int, int], str, int], dict[str, object]] = {}
    for cand in selection:
        uniq[(cand["tail"], cand["direction"], cand["length"])] = cand
    selection = list(uniq.values())
    return selection or None


def _assign_tail_colors(rng: random.Random, arrows: list[dict[str, object]]) -> None:
    by_target: dict[int, list[dict[str, object]]] = defaultdict(list)
    for arrow in arrows:
        by_target[arrow["target"]].append(arrow)

    for target_arrows in by_target.values():
        if len(target_arrows) == 1:
            target_arrows[0]["color"] = rng.choice(PALETTE)
            continue
        colors = list(PALETTE)
        rng.shuffle(colors)
        while len(colors) < len(target_arrows):
            colors.extend(PALETTE)
        for arrow, color in zip(target_arrows, colors, strict=False):
            arrow["color"] = color


def _draw_arrows(grid: np.ndarray, arrows: list[dict[str, object]]) -> None:
    for arrow in arrows:
        for r, c in arrow["grey_cells"]:
            grid[r, c] = GREY
        tr, tc = arrow["tail"]
        grid[tr, tc] = int(arrow["color"])


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(18, 26)
    w = rng.randint(18, 26)
    grid = np.full((h, w), BG, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    num_shapes = rng.randint(2, 4)
    shapes: list[dict[str, object]] = []
    for idx in range(num_shapes):
        placed = False
        for _ in range(150):
            template = _shape_template(rng)
            frame = template["frame"]
            pattern = template["pattern"]
            sh, sw = frame.shape
            top = rng.randint(3, h - sh - 4)
            left = rng.randint(3, w - sw - 4)
            if not _can_place(occupied, top, left, sh, sw, gap=3):
                continue

            pattern_color = rng.choice(PALETTE)
            shape = _stamp_shape(grid, top, left, frame, pattern, pattern_color)
            shape["pattern_color"] = pattern_color
            shapes.append(shape)
            _reserve(occupied, top, left, sh, sw)
            placed = True
            break
        if not placed:
            return None

    candidates_by_shape = _enumerate_arrow_candidates(grid, shapes)
    arrows = _choose_arrows(rng, shapes, candidates_by_shape)
    if arrows is None:
        return None

    _assign_tail_colors(rng, arrows)
    _draw_arrows(grid, arrows)

    out = main(grid)
    if out.shape != grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    seed = random.randrange(1 << 30)
    for offset in range(MAX_GENERATION_ATTEMPTS):
        rng = random.Random(seed + offset)
        try:
            grid = _gen_attempt(rng)
            if grid is not None:
                return grid
        except Exception:
            continue
    raise RuntimeError("failed to generate a valid input")
