import numpy as np


def _bg_color(grid: np.ndarray) -> int:
    vals = np.bincount(grid.ravel(), minlength=10)
    return int(np.argmax(vals))


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


def _components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            arr = np.array(pts, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": _bbox(arr),
                    "area": int(len(pts)),
                }
            )
    return comps


def _shape_from_coords(coords: np.ndarray) -> dict:
    r0, c0, r1, c1 = _bbox(coords)
    rel = [(int(r - r0), int(c - c0)) for r, c in coords]
    rel.sort()
    rel_arr = np.array(rel, dtype=int)
    return {
        "rel": rel,
        "h": int(r1 - r0 + 1),
        "w": int(c1 - c0 + 1),
        "max_r": int(rel_arr[:, 0].max()),
        "max_c": int(rel_arr[:, 1].max()),
        "center_r": float(rel_arr[:, 0].mean()),
        "center_c": float(rel_arr[:, 1].mean()),
    }


def _gap_run(inner: np.ndarray) -> tuple[int, int] | None:
    runs: list[tuple[int, int]] = []
    start = None
    for i, flag in enumerate(inner):
        if not bool(flag):
            if start is None:
                start = i
        elif start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(inner) - 1))
    if len(runs) != 1:
        return None
    a, b = runs[0]
    if b - a + 1 < 2:
        return None
    return a, b


def _parse_frame(comp: dict) -> dict | None:
    coords = comp["coords"]
    r0, c0, r1, c1 = comp["bbox"]
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 5 or w < 5:
        return None

    mask = np.zeros((h, w), dtype=bool)
    for r, c in coords:
        rr = int(r - r0)
        cc = int(c - c0)
        if rr not in (0, h - 1) and cc not in (0, w - 1):
            return None
        mask[rr, cc] = True

    if mask[1:-1, 1:-1].any():
        return None
    if not (mask[0, 0] and mask[0, -1] and mask[-1, 0] and mask[-1, -1]):
        return None

    side_arrays = {
        "top": mask[0, :],
        "bottom": mask[-1, :],
        "left": mask[:, 0],
        "right": mask[:, -1],
    }

    gap_side = None
    gap_bounds = None
    for side, arr in side_arrays.items():
        inner = arr[1:-1]
        run = _gap_run(inner)
        if run is None:
            if not bool(inner.all()):
                return None
            continue
        if gap_side is not None:
            return None
        gap_side = side
        gap_bounds = run

    if gap_side is None or gap_bounds is None:
        return None

    gap_len = int(gap_bounds[1] - gap_bounds[0] + 1)
    expected_area = 2 * h + 2 * w - 4 - gap_len
    if int(comp["area"]) != expected_area:
        return None

    if gap_side in ("top", "bottom"):
        gap_start = c0 + 1 + int(gap_bounds[0])
        gap_end = c0 + 1 + int(gap_bounds[1])
        gap_center = 0.5 * (gap_start + gap_end)
    else:
        gap_start = r0 + 1 + int(gap_bounds[0])
        gap_end = r0 + 1 + int(gap_bounds[1])
        gap_center = 0.5 * (gap_start + gap_end)

    return {
        "color": int(comp["color"]),
        "coords": comp["coords"],
        "bbox": comp["bbox"],
        "side": gap_side,
        "gap_start": int(gap_start),
        "gap_end": int(gap_end),
        "gap_center": float(gap_center),
        "interior_top": int(r0 + 1),
        "interior_bottom": int(r1 - 1),
        "interior_left": int(c0 + 1),
        "interior_right": int(c1 - 1),
    }


def _fragment_distance(frame: dict, comp: dict) -> int | None:
    if int(comp["color"]) != int(frame["color"]):
        return None
    r0, c0, r1, c1 = comp["bbox"]
    fr0, fc0, fr1, fc1 = frame["bbox"]
    side = frame["side"]
    if side == "top":
        if r1 >= fr0 or c0 < frame["gap_start"] or c1 > frame["gap_end"]:
            return None
        return int(fr0 - r1)
    if side == "bottom":
        if r0 <= fr1 or c0 < frame["gap_start"] or c1 > frame["gap_end"]:
            return None
        return int(r0 - fr1)
    if side == "left":
        if c1 >= fc0 or r0 < frame["gap_start"] or r1 > frame["gap_end"]:
            return None
        return int(fc0 - c1)
    if c0 <= fc1 or r0 < frame["gap_start"] or r1 > frame["gap_end"]:
        return None
    return int(c0 - fc1)


def _cells_at(shape: dict, top: int, left: int) -> set[tuple[int, int]]:
    return {(top + rr, left + cc) for rr, cc in shape["rel"]}


def _inside_frame(cells: set[tuple[int, int]], frame: dict) -> bool:
    return all(
        frame["interior_top"] <= r <= frame["interior_bottom"] and frame["interior_left"] <= c <= frame["interior_right"]
        for r, c in cells
    )


def _project_fragment(frame: dict, comp: dict, occupied: set[tuple[int, int]]) -> set[tuple[int, int]]:
    shape = _shape_from_coords(comp["coords"])
    r0, c0, _r1, _c1 = comp["bbox"]
    side = frame["side"]

    if side == "top":
        rebound = 1 if float(comp["coords"][:, 1].mean()) < frame["gap_center"] else -1
        top = frame["interior_bottom"] - shape["max_r"]
        ideal_left = frame["interior_right"] - shape["max_c"] if rebound > 0 else frame["interior_left"]
        row_shift = 0
        col_shift = -rebound
        final_top = int(top)
        final_left = int(ideal_left)
    elif side == "bottom":
        rebound = 1 if float(comp["coords"][:, 1].mean()) < frame["gap_center"] else -1
        top = frame["interior_top"]
        ideal_left = frame["interior_right"] - shape["max_c"] if rebound > 0 else frame["interior_left"]
        row_shift = 0
        col_shift = -rebound
        final_top = int(top)
        final_left = int(ideal_left)
    elif side == "left":
        rebound = 1 if float(comp["coords"][:, 0].mean()) < frame["gap_center"] else -1
        left = frame["interior_right"] - shape["max_c"]
        ideal_top = frame["interior_bottom"] - shape["max_r"] if rebound > 0 else frame["interior_top"]
        row_shift = -rebound
        col_shift = 0
        final_top = int(ideal_top)
        final_left = int(left)
    else:
        rebound = 1 if float(comp["coords"][:, 0].mean()) < frame["gap_center"] else -1
        left = frame["interior_left"]
        ideal_top = frame["interior_bottom"] - shape["max_r"] if rebound > 0 else frame["interior_top"]
        row_shift = -rebound
        col_shift = 0
        final_top = int(ideal_top)
        final_left = int(left)

    cells = _cells_at(shape, final_top, final_left)
    while cells & occupied:
        next_top = final_top + row_shift
        next_left = final_left + col_shift
        next_cells = _cells_at(shape, next_top, next_left)
        if not _inside_frame(next_cells, frame):
            break
        final_top = next_top
        final_left = next_left
        cells = next_cells

    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Determine each frame's opening side from its missing border segment.
    Move same-color fragments from outside through the opening, then reflect them
    once inside the window based on their offset from the opening center line.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    comps = _components(g, bg)

    frame_infos: list[dict] = []
    frame_ids: set[int] = set()
    for idx, comp in enumerate(comps):
        parsed = _parse_frame(comp)
        if parsed is not None:
            frame_infos.append(parsed)
            frame_ids.add(idx)

    fragments = [comp for idx, comp in enumerate(comps) if idx not in frame_ids]
    assignments: dict[int, list[tuple[int, dict]]] = {i: [] for i in range(len(frame_infos))}

    for comp in fragments:
        candidates: list[tuple[int, int]] = []
        for i, frame in enumerate(frame_infos):
            dist = _fragment_distance(frame, comp)
            if dist is not None:
                candidates.append((dist, i))
        if not candidates:
            continue
        candidates.sort()
        dist, idx = candidates[0]
        assignments[idx].append((int(dist), comp))

    out = g.copy()
    for items in assignments.values():
        for _dist, comp in items:
            for r, c in comp["coords"]:
                out[int(r), int(c)] = bg

    for idx, frame in enumerate(frame_infos):
        items = sorted(assignments[idx], key=lambda x: x[0])
        occupied: set[tuple[int, int]] = set()
        for _dist, comp in items:
            cells = _project_fragment(frame, comp, occupied)
            occupied |= cells
            for r, c in cells:
                out[int(r), int(c)] = int(comp["color"])

    return out


def _normalize_shape(coords: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    norm = sorted((r - min_r, c - min_c) for r, c in coords)
    return tuple((int(r), int(c)) for r, c in norm)


def _rotate_shape(coords: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    max_r = max(r for r, _ in coords)
    return _normalize_shape(tuple((c, max_r - r) for r, c in coords))


def _all_shapes() -> list[tuple[tuple[int, int], ...]]:
    bases = [
        ((0, 0),),
        ((0, 0), (0, 1)),
        ((0, 0), (1, 0)),
        ((0, 0), (0, 1), (0, 2)),
        ((0, 0), (1, 0), (2, 0)),
        ((0, 0), (1, 0), (1, 1)),
        ((0, 1), (1, 0), (1, 1)),
    ]
    seen: set[tuple[tuple[int, int], ...]] = set()
    shapes: list[tuple[tuple[int, int], ...]] = []
    for base in bases:
        cur = _normalize_shape(base)
        for _ in range(4):
            if cur not in seen:
                seen.add(cur)
                shapes.append(cur)
            cur = _rotate_shape(cur)
    return shapes


_SHAPES = _all_shapes()


def _shape_meta(shape: tuple[tuple[int, int], ...]) -> dict:
    arr = np.array(shape, dtype=int)
    return {
        "coords": shape,
        "h": int(arr[:, 0].max() + 1),
        "w": int(arr[:, 1].max() + 1),
        "center_r": float(arr[:, 0].mean()),
        "center_c": float(arr[:, 1].mean()),
    }


_SHAPE_META = [_shape_meta(shape) for shape in _SHAPES]


def _frame_cells(h: int, w: int, side: str, gap_start: int, gap_len: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for c in range(w):
        if not (side == "top" and gap_start <= c <= gap_start + gap_len - 1):
            cells.add((0, c))
        if not (side == "bottom" and gap_start <= c <= gap_start + gap_len - 1):
            cells.add((h - 1, c))
    for r in range(1, h - 1):
        if not (side == "left" and gap_start <= r <= gap_start + gap_len - 1):
            cells.add((r, 0))
        if not (side == "right" and gap_start <= r <= gap_start + gap_len - 1):
            cells.add((r, w - 1))
    return cells


def _touches(existing: set[tuple[int, int]], cells: set[tuple[int, int]]) -> bool:
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if (r + dr, c + dc) in existing:
                    return True
    return False


def _choose_signs(rng: np.random.Generator, count: int) -> list[int]:
    if count == 2:
        signs = [-1, 1]
    elif count == 3:
        signs = [-1, 1, int(rng.choice([-1, 1]))]
    else:
        signs = [-1, -1, 1, 1]
    rng.shuffle(signs)
    return signs


def _allowed_shapes(side: str, sign_count: int, interior_h: int, interior_w: int, gap_len: int) -> list[dict]:
    allowed: list[dict] = []
    if side in ("top", "bottom"):
        max_span = interior_w - sign_count + 1
        for meta in _SHAPE_META:
            if meta["h"] <= interior_h and meta["w"] <= max_span and meta["w"] <= gap_len:
                allowed.append(meta)
    else:
        max_span = interior_h - sign_count + 1
        for meta in _SHAPE_META:
            if meta["w"] <= interior_w and meta["h"] <= max_span and meta["h"] <= gap_len:
                allowed.append(meta)
    return allowed


def _candidate_offsets(side: str, sign: int, gap_start: int, gap_len: int, meta: dict) -> list[int]:
    offsets: list[int] = []
    gap_center = gap_start + 0.5 * (gap_len - 1)
    if side in ("top", "bottom"):
        for left in range(gap_start, gap_start + gap_len - meta["w"] + 1):
            center = left + meta["center_c"]
            if sign < 0 and center < gap_center:
                offsets.append(int(left))
            if sign > 0 and center > gap_center:
                offsets.append(int(left))
    else:
        for top in range(gap_start, gap_start + gap_len - meta["h"] + 1):
            center = top + meta["center_r"]
            if sign < 0 and center < gap_center:
                offsets.append(int(top))
            if sign > 0 and center > gap_center:
                offsets.append(int(top))
    return offsets


def _place_local_piece(side: str, h: int, w: int, axis_offset: int, distance: int, meta: dict) -> set[tuple[int, int]]:
    if side == "top":
        top = -distance - meta["h"] + 1
        left = axis_offset
    elif side == "bottom":
        top = h - 1 + distance
        left = axis_offset
    elif side == "left":
        top = axis_offset
        left = -distance - meta["w"] + 1
    else:
        top = axis_offset
        left = w - 1 + distance
    return {(top + rr, left + cc) for rr, cc in meta["coords"]}


def _build_cluster(
    rng: np.random.Generator,
    color: int,
    side: str,
    h: int,
    w: int,
    gap_start: int,
    gap_len: int,
    piece_count: int,
) -> dict | None:
    frame = _frame_cells(h, w, side, gap_start, gap_len)
    pieces: list[dict] = []
    used_local = set(frame)
    signs = _choose_signs(rng, piece_count)
    group_counts = {sign: signs.count(sign) for sign in (-1, 1)}
    allowed_by_sign = {
        sign: _allowed_shapes(side, group_counts[sign], h - 2, w - 2, gap_len) for sign in (-1, 1)
    }
    if not allowed_by_sign[-1] or not allowed_by_sign[1]:
        return None

    distances = list(rng.choice(np.arange(1, 6), size=piece_count, replace=False))
    reuse_bias = bool(rng.integers(0, 2))
    metas_by_sign: dict[int, list[dict]] = {-1: [], 1: []}

    for idx in range(piece_count):
        sign = signs[idx]
        dist = int(distances[idx])
        candidates = allowed_by_sign[sign]
        order = list(range(len(candidates)))
        rng.shuffle(order)
        if reuse_bias and metas_by_sign[sign]:
            repeated = metas_by_sign[sign][int(rng.integers(len(metas_by_sign[sign])))]
            order = [candidates.index(repeated)] + [i for i in order if candidates[i] is not repeated]

        placed = None
        for meta_idx in order:
            meta = candidates[meta_idx]
            offsets = _candidate_offsets(side, sign, gap_start, gap_len, meta)
            if not offsets:
                continue
            rng.shuffle(offsets)
            for axis_offset in offsets:
                cells = _place_local_piece(side, h, w, int(axis_offset), dist, meta)
                if _touches(used_local, cells):
                    continue
                placed = {"coords": cells, "meta": meta}
                break
            if placed is not None:
                break
        if placed is None:
            return None
        pieces.append(placed)
        used_local |= placed["coords"]
        metas_by_sign[sign].append(placed["meta"])

    cluster_cells = set(frame)
    for piece in pieces:
        cluster_cells |= piece["coords"]

    rows = [r for r, _ in cluster_cells]
    cols = [c for _, c in cluster_cells]
    min_r = min(rows)
    min_c = min(cols)
    max_r = max(rows)
    max_c = max(cols)
    return {
        "color": int(color),
        "side": side,
        "cells": cluster_cells,
        "frame_cells": frame,
        "pieces": pieces,
        "min_r": int(min_r),
        "min_c": int(min_c),
        "max_r": int(max_r),
        "max_c": int(max_c),
    }


def _mark_padded(mask: np.ndarray, cells: set[tuple[int, int]]) -> None:
    h, w = mask.shape
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    mask[nr, nc] = True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    grid_h = int(rng.integers(18, 29))
    grid_w = int(rng.integers(18, 29))
    grid = np.zeros((grid_h, grid_w), dtype=int)
    occupied_pad = np.zeros((grid_h, grid_w), dtype=bool)

    n_frames = int(rng.integers(1, 4))
    sides = ["top", "bottom", "left", "right"]
    rng.shuffle(sides)
    palette = list(range(1, 10))
    rng.shuffle(palette)

    for i in range(n_frames):
        side = sides[i]
        color = palette[i]
        placed = False
        for _ in range(120):
            h = int(rng.integers(7, 12))
            w = int(rng.integers(7, 12))
            gap_span = w - 2 if side in ("top", "bottom") else h - 2
            gap_len = int(rng.integers(4, gap_span + 1))
            gap_start = int(rng.integers(1, gap_span - gap_len + 2))
            piece_count = int(rng.integers(2, 5))

            cluster = _build_cluster(rng, color, side, h, w, gap_start, gap_len, piece_count)
            if cluster is None:
                continue

            row_lo = -cluster["min_r"]
            row_hi = grid_h - 1 - cluster["max_r"]
            col_lo = -cluster["min_c"]
            col_hi = grid_w - 1 - cluster["max_c"]
            if row_lo > row_hi or col_lo > col_hi:
                continue

            for _ in range(80):
                base_r = int(rng.integers(row_lo, row_hi + 1))
                base_c = int(rng.integers(col_lo, col_hi + 1))
                abs_cells = {(base_r + r, base_c + c) for r, c in cluster["cells"]}
                if any(occupied_pad[r, c] for r, c in abs_cells):
                    continue
                for r, c in abs_cells:
                    grid[r, c] = color
                _mark_padded(occupied_pad, abs_cells)
                placed = True
                break
            if placed:
                break
        if not placed:
            return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(400):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
