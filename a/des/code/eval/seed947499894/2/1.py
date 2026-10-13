from __future__ import annotations

import numpy as np


DIRS = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}
DIR_FROM_VEC = {v: k for k, v in DIRS.items()}
FOUR_NEIGHBORS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _bbox(cells: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize_cells(cells: list[tuple[int, int]] | set[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    if not cells:
        return tuple()
    r0, c0, _r1, _c1 = _bbox(list(cells))
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in cells))


def _shape_size(sig: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    if not sig:
        return 0, 0
    return max(r for r, _ in sig) + 1, max(c for _, c in sig) + 1


def _translate(sig: tuple[tuple[int, int], ...], dr: int, dc: int) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for r, c in sig]


def _transform_signature(
    sig: tuple[tuple[int, int], ...],
    rotation: int = 0,
    flip_lr: bool = False,
) -> tuple[tuple[int, int], ...]:
    pts = [(int(r), int(c)) for r, c in sig]
    if flip_lr:
        pts = [(r, -c) for r, c in pts]
    for _ in range(rotation % 4):
        pts = [(c, -r) for r, c in pts]
    return _normalize_cells(pts)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for dr, dc in FOUR_NEIGHBORS:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(comp)
    return out


def _is_connected(cells: list[tuple[int, int]] | tuple[tuple[int, int], ...] | set[tuple[int, int]]) -> bool:
    if not cells:
        return False
    cell_set = {tuple(p) for p in cells}
    stack = [next(iter(cell_set))]
    seen = {stack[0]}
    while stack:
        r, c = stack.pop()
        for dr, dc in FOUR_NEIGHBORS:
            nb = (r + dr, c + dc)
            if nb in cell_set and nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return len(seen) == len(cell_set)


def _rect_cells(r0: int, c0: int, h: int, w: int) -> set[tuple[int, int]]:
    return {(r, c) for r in range(r0, r0 + h) for c in range(c0, c0 + w)}


def _padded(cells: set[tuple[int, int]], h: int, w: int, pad: int = 1) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in range(-pad, pad + 1):
            for dc in range(-pad, pad + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    out.add((nr, nc))
    return out


def _is_frame_component(g: np.ndarray, comp: list[tuple[int, int]], color: int) -> bool:
    r0, c0, r1, c1 = _bbox(comp)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    if len(comp) != 2 * (h + w) - 4:
        return False
    cell_set = set(comp)
    for c in range(c0, c1 + 1):
        if (r0, c) not in cell_set or (r1, c) not in cell_set:
            return False
    for r in range(r0, r1 + 1):
        if (r, c0) not in cell_set or (r, c1) not in cell_set:
            return False
    sub = g[r0 : r1 + 1, c0 : c1 + 1]
    border = np.zeros(sub.shape, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return bool(np.all(sub[border] == color))


def _find_frames(g: np.ndarray, bg: int) -> list[dict]:
    frames: list[dict] = []
    frame_id = 0
    for color in [int(v) for v in np.unique(g) if int(v) != bg]:
        for comp in _components(g == color):
            if not _is_frame_component(g, comp, color):
                continue
            r0, c0, r1, c1 = _bbox(comp)
            inside: list[tuple[int, int]] = []
            for r in range(r0 + 1, r1):
                for c in range(c0 + 1, c1):
                    v = int(g[r, c])
                    if v != bg and v != color:
                        inside.append((int(r), int(c)))
            if not inside:
                continue
            frames.append(
                {
                    "id": frame_id,
                    "border_color": int(color),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "height": int(r1 - r0 + 1),
                    "width": int(c1 - c0 + 1),
                    "shape_cells": inside,
                    "shape_sig": _normalize_cells(inside),
                }
            )
            frame_id += 1
    return frames


def _key_matches(
    key_sig: tuple[tuple[int, int], ...],
    frame_sig_to_ids: dict[tuple[tuple[int, int], ...], list[int]],
) -> set[tuple[int, str]]:
    matches: set[tuple[int, str]] = set()
    key_cells = list(key_sig)
    for removed in key_cells:
        rem = [p for p in key_cells if p != removed]
        if not _is_connected(rem):
            continue
        rem_sig = _normalize_cells(rem)
        ids = frame_sig_to_ids.get(rem_sig, [])
        if len(ids) != 1:
            continue
        dirs = {
            (removed[0] - rr, removed[1] - cc)
            for rr, cc in rem
            if abs(removed[0] - rr) + abs(removed[1] - cc) == 1
        }
        if len(dirs) != 1:
            continue
        vec = next(iter(dirs))
        if vec not in DIR_FROM_VEC:
            continue
        matches.add((int(ids[0]), DIR_FROM_VEC[vec]))
    return matches


def _find_keys(g: np.ndarray, bg: int, frames: list[dict]) -> list[dict]:
    h, w = g.shape
    outside = np.ones((h, w), dtype=bool)
    frame_sig_to_ids: dict[tuple[tuple[int, int], ...], list[int]] = {}
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        outside[r0 : r1 + 1, c0 : c1 + 1] = False
        frame_sig_to_ids.setdefault(frame["shape_sig"], []).append(int(frame["id"]))

    keys: list[dict] = []
    for color in [int(v) for v in np.unique(g) if int(v) != bg]:
        for comp in _components((g == color) & outside):
            if len(comp) < 4:
                continue
            comp_sig = _normalize_cells(comp)
            matches = _key_matches(comp_sig, frame_sig_to_ids)
            if len(matches) != 1:
                continue
            frame_id, direction = next(iter(matches))
            keys.append(
                {
                    "frame_id": int(frame_id),
                    "color": int(color),
                    "direction": direction,
                    "cells": comp,
                }
            )
    return keys


def _analyze_grid(g: np.ndarray) -> tuple[int, list[dict], list[dict]]:
    bg = _most_common_color(g)
    frames = _find_frames(g, bg)
    keys = _find_keys(g, bg, frames)
    return bg, frames, keys


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each outside key head to the identical inside-frame shape, then tile that
    frame's content along the key handle direction using the key color.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    _bg, frames, keys = _analyze_grid(g)
    frame_by_id = {int(frame["id"]): frame for frame in frames}

    for key in keys:
        frame = frame_by_id.get(int(key["frame_id"]))
        if frame is None:
            continue
        direction = key["direction"]
        dr, dc = DIRS[direction]
        step = int(frame["height"] if direction in ("U", "D") else frame["width"])
        if step <= 0:
            continue
        base_cells = frame["shape_cells"]
        shift = 0
        while True:
            any_in_bounds = False
            for r, c in base_cells:
                rr = int(r + dr * step * shift)
                cc = int(c + dc * step * shift)
                if 0 <= rr < h and 0 <= cc < w:
                    out[rr, cc] = int(key["color"])
                    any_in_bounds = True
            if not any_in_bounds:
                break
            shift += 1

    return out


BASE_SHAPES = [
    ((0, 0), (0, 1), (1, 0), (2, 0)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
    ((0, 0), (0, 1), (1, 1), (1, 2)),
    ((0, 1), (1, 0), (1, 1), (2, 0)),
    ((0, 0), (1, 0), (1, 1), (2, 1)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1), (2, 2)),
    ((0, 0), (0, 1), (1, 1), (2, 0), (2, 1)),
    ((0, 1), (1, 1), (2, 0), (2, 1), (2, 2)),
    ((0, 0), (1, 0), (1, 1), (1, 2), (2, 2)),
    ((0, 0), (0, 1), (1, 1), (2, 1), (2, 2)),
    ((0, 1), (1, 0), (1, 1), (2, 1), (2, 2), (2, 3)),
]


def _variant_pool() -> list[tuple[tuple[int, int], ...]]:
    pool: set[tuple[tuple[int, int], ...]] = set()
    for shape in BASE_SHAPES:
        base = _normalize_cells(shape)
        for flip in (False, True):
            for rot in range(4):
                pool.add(_transform_signature(base, rotation=rot, flip_lr=flip))
    return sorted(pool)


SHAPE_POOL = _variant_pool()


def _handle_candidates(
    head_sig: tuple[tuple[int, int], ...],
    direction: str,
) -> list[tuple[tuple[tuple[int, int], ...], tuple[int, int]]]:
    dr, dc = DIRS[direction]
    head = set(head_sig)
    out: list[tuple[tuple[tuple[int, int], ...], tuple[int, int]]] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    for r, c in head:
        hr, hc = r + dr, c + dc
        if (hr, hc) in head:
            continue
        neighbors = [(rr, cc) for rr, cc in head if abs(hr - rr) + abs(hc - cc) == 1]
        if len(neighbors) != 1:
            continue
        key_sig = _normalize_cells(head | {(hr, hc)})
        if key_sig in seen:
            continue
        seen.add(key_sig)
        out.append((key_sig, (hr, hc)))
    return out


def _expansion_cells(
    shape_cells: list[tuple[int, int]],
    direction: str,
    step: int,
    h: int,
    w: int,
) -> tuple[set[tuple[int, int]], int]:
    dr, dc = DIRS[direction]
    cells: set[tuple[int, int]] = set()
    copies = 0
    shift = 0
    while True:
        any_in_bounds = False
        for r, c in shape_cells:
            rr = int(r + dr * step * shift)
            cc = int(c + dc * step * shift)
            if 0 <= rr < h and 0 <= cc < w:
                cells.add((rr, cc))
                any_in_bounds = True
        if not any_in_bounds:
            break
        copies += 1
        shift += 1
    return cells, copies


def _build_expected(grid: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = grid.copy()
    h, w = out.shape
    for spec in specs:
        direction = spec["direction"]
        step = int(spec["frame_h"] if direction in ("U", "D") else spec["frame_w"])
        dr, dc = DIRS[direction]
        shift = 0
        while True:
            any_in_bounds = False
            for r, c in spec["shape_abs"]:
                rr = int(r + dr * step * shift)
                cc = int(c + dc * step * shift)
                if 0 <= rr < h and 0 <= cc < w:
                    out[rr, cc] = int(spec["key_color"])
                    any_in_bounds = True
            if not any_in_bounds:
                break
            shift += 1
    return out


def _sample_shape_assignment(
    rng: np.random.Generator,
    directions: list[str],
) -> list[dict] | None:
    frame_count = len(directions)
    for _ in range(200):
        idxs = rng.choice(len(SHAPE_POOL), size=frame_count, replace=False)
        frame_sigs = [SHAPE_POOL[int(i)] for i in idxs]
        if len(set(frame_sigs)) != frame_count:
            continue
        frame_sig_to_ids = {sig: [i] for i, sig in enumerate(frame_sigs)}
        specs: list[dict] = []
        ok = True
        for i, direction in enumerate(directions):
            head_sig = frame_sigs[i]
            valid_keys: list[tuple[tuple[tuple[int, int], ...], tuple[int, int]]] = []
            for key_sig, handle_pos in _handle_candidates(head_sig, direction):
                if _key_matches(key_sig, frame_sig_to_ids) == {(i, direction)}:
                    valid_keys.append((key_sig, handle_pos))
            if not valid_keys:
                ok = False
                break
            choice = valid_keys[int(rng.integers(0, len(valid_keys)))]
            specs.append(
                {
                    "shape_sig": head_sig,
                    "key_sig": choice[0],
                    "direction": direction,
                }
            )
        if ok:
            return specs
    return None


def _frame_position_limits(
    direction: str,
    frame_h: int,
    frame_w: int,
    shape_rel: list[tuple[int, int]],
    h: int,
    w: int,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    min_r = min(r for r, _ in shape_rel)
    max_r = max(r for r, _ in shape_rel)
    min_c = min(c for _, c in shape_rel)
    max_c = max(c for _, c in shape_rel)

    row_lo, row_hi = 1, h - frame_h - 1
    col_lo, col_hi = 1, w - frame_w - 1
    if row_lo > row_hi or col_lo > col_hi:
        return None

    if direction == "U":
        row_lo = max(row_lo, frame_h - min_r, h // 2)
    elif direction == "D":
        row_hi = min(row_hi, h - frame_h - 1 - max_r, h // 2 - 1)
    elif direction == "L":
        col_lo = max(col_lo, frame_w - min_c, w // 2)
    else:
        col_hi = min(col_hi, w - frame_w - 1 - max_c, w // 2 - 1)

    if row_lo > row_hi or col_lo > col_hi:
        return None
    return (int(row_lo), int(row_hi)), (int(col_lo), int(col_hi))


def _place_frame(
    rng: np.random.Generator,
    spec: dict,
    grid_h: int,
    grid_w: int,
    occupied_input: set[tuple[int, int]],
    reserved_exact: set[tuple[int, int]],
) -> dict | None:
    shape_h, shape_w = _shape_size(spec["shape_sig"])
    for _ in range(200):
        top_pad = int(rng.integers(0, 3))
        bottom_pad = int(rng.integers(0, 3))
        left_pad = int(rng.integers(0, 3))
        right_pad = int(rng.integers(0, 3))
        frame_h = shape_h + top_pad + bottom_pad + 2
        frame_w = shape_w + left_pad + right_pad + 2
        shape_rel = _translate(spec["shape_sig"], 1 + top_pad, 1 + left_pad)
        limits = _frame_position_limits(
            spec["direction"], frame_h, frame_w, shape_rel, grid_h, grid_w
        )
        if limits is None:
            continue
        (row_lo, row_hi), (col_lo, col_hi) = limits
        if row_lo > row_hi or col_lo > col_hi:
            continue
        r0 = int(rng.integers(row_lo, row_hi + 1))
        c0 = int(rng.integers(col_lo, col_hi + 1))
        bbox_cells = _rect_cells(r0, c0, frame_h, frame_w)
        if bbox_cells & reserved_exact:
            continue
        if _padded(bbox_cells, grid_h, grid_w, pad=1) & occupied_input:
            continue
        shape_abs = [(r0 + r, c0 + c) for r, c in shape_rel]
        expansion, copies = _expansion_cells(
            shape_abs,
            spec["direction"],
            frame_h if spec["direction"] in ("U", "D") else frame_w,
            grid_h,
            grid_w,
        )
        if copies < 2:
            continue
        if expansion & reserved_exact:
            continue
        return {
            **spec,
            "frame_h": frame_h,
            "frame_w": frame_w,
            "bbox": (r0, c0, r0 + frame_h - 1, c0 + frame_w - 1),
            "shape_abs": shape_abs,
            "bbox_cells": bbox_cells,
            "expansion_cells": expansion,
        }
    return None


def _place_key(
    rng: np.random.Generator,
    key_sig: tuple[tuple[int, int], ...],
    grid_h: int,
    grid_w: int,
    occupied_input: set[tuple[int, int]],
    reserved_exact: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    key_h, key_w = _shape_size(key_sig)
    for _ in range(300):
        r0 = int(rng.integers(0, grid_h - key_h + 1))
        c0 = int(rng.integers(0, grid_w - key_w + 1))
        abs_cells = [(r0 + r, c0 + c) for r, c in key_sig]
        cell_set = set(abs_cells)
        if cell_set & reserved_exact:
            continue
        if _padded(cell_set, grid_h, grid_w, pad=1) & occupied_input:
            continue
        return abs_cells
    return None


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    frame_count = int(rng.integers(2, 5))
    directions = [str(d) for d in rng.choice(np.array(list(DIRS)), size=frame_count, replace=False)]
    shape_specs = _sample_shape_assignment(rng, directions)
    if shape_specs is None:
        return None

    grid_h = int(rng.integers(24, 33))
    grid_w = int(rng.integers(24, 33))
    if frame_count == 4:
        grid_h = max(grid_h, 28)
        grid_w = max(grid_w, 28)

    palette = [int(v) for v in rng.permutation(10)]
    bg = palette[0]
    border = palette[1]
    key_colors = palette[2 : 2 + frame_count]
    inner_colors = palette[2 + frame_count : 2 + 2 * frame_count]

    specs: list[dict] = []
    for i, spec in enumerate(shape_specs):
        specs.append(
            {
                **spec,
                "border_color": int(border),
                "key_color": int(key_colors[i]),
                "inner_color": int(inner_colors[i]),
            }
        )

    order = sorted(range(frame_count), key=lambda i: -len(specs[i]["shape_sig"]))
    placed: list[dict | None] = [None] * frame_count
    occupied_input: set[tuple[int, int]] = set()
    reserved_exact: set[tuple[int, int]] = set()

    for i in order:
        placement = _place_frame(rng, specs[i], grid_h, grid_w, occupied_input, reserved_exact)
        if placement is None:
            return None
        placed[i] = placement
        occupied_input |= placement["bbox_cells"]
        reserved_exact |= placement["bbox_cells"] | placement["expansion_cells"]

    final_specs = [p for p in placed if p is not None]
    if len(final_specs) != frame_count:
        return None

    for spec in final_specs:
        key_abs = _place_key(rng, spec["key_sig"], grid_h, grid_w, occupied_input, reserved_exact)
        if key_abs is None:
            return None
        spec["key_abs"] = key_abs
        key_set = set(key_abs)
        occupied_input |= key_set
        reserved_exact |= key_set

    grid = np.full((grid_h, grid_w), bg, dtype=int)
    for spec in final_specs:
        r0, c0, r1, c1 = spec["bbox"]
        grid[r0, c0 : c1 + 1] = spec["border_color"]
        grid[r1, c0 : c1 + 1] = spec["border_color"]
        grid[r0 : r1 + 1, c0] = spec["border_color"]
        grid[r0 : r1 + 1, c1] = spec["border_color"]
        for r, c in spec["shape_abs"]:
            grid[r, c] = spec["inner_color"]
        for r, c in spec["key_abs"]:
            grid[r, c] = spec["key_color"]

    if _most_common_color(grid) != bg:
        return None

    analyzed_bg, frames, keys = _analyze_grid(grid)
    if analyzed_bg != bg or len(frames) != frame_count or len(keys) != frame_count:
        return None

    expected = _build_expected(grid, final_specs)
    got = main(grid)
    if np.array_equal(got, grid):
        return None
    if not np.array_equal(got, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(500):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
