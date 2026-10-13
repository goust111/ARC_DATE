from __future__ import annotations

from collections import Counter
from typing import Iterable

import numpy as np


BG = 0
MARKER = 1
SYMBOL_COLORS = [2, 3, 4, 5, 6, 7, 8, 9]
BLOCK = 5
RNG = np.random.default_rng()


def _mask_from_rows(rows: list[str]) -> np.ndarray:
    return np.array([[ch == "#" for ch in row] for row in rows], dtype=bool)


BASE_MASKS = {
    0: _mask_from_rows(
        [
            "#####",
            "#...#",
            "#...#",
            "#...#",
            "#####",
        ]
    ),
    1: _mask_from_rows(
        [
            "#####",
            "#...#",
            "##..#",
            "#...#",
            "#####",
        ]
    ),
    2: _mask_from_rows(
        [
            "#####",
            "#.#.#",
            "##..#",
            "#...#",
            "#####",
        ]
    ),
    3: _mask_from_rows(
        [
            "#####",
            "#.#.#",
            "#...#",
            "#.#.#",
            "#####",
        ]
    ),
    4: _mask_from_rows(
        [
            "#####",
            "#.#.#",
            "##.##",
            "#...#",
            "#####",
        ]
    ),
    5: _mask_from_rows(
        [
            "#####",
            "#.#.#",
            "##.##",
            "#.#.#",
            "#####",
        ]
    ),
}
ASYMMETRIC_BASE_IDS = (2, 5)


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return np.zeros((0, 0), dtype=bool)
    r0, c0, r1, c1 = _bbox_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    trimmed = _trim_mask(mask)
    return tuple(tuple(int(v) for v in row) for row in trimmed.astype(int))


def _canonical_shape_key(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    variants = []
    for rot in range(4):
        turned = np.rot90(mask, rot)
        variants.append(_mask_key(turned))
        variants.append(_mask_key(np.fliplr(turned)))
    return min(variants)


def _transform_mask(mask: np.ndarray, code: int) -> np.ndarray:
    rot = int(code) % 4
    flipped = int(code) >= 4
    out = np.rot90(mask, rot)
    if flipped:
        out = np.fliplr(out)
    return out


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
    seen = np.zeros_like(mask, dtype=bool)
    coords_list: list[np.ndarray] = []
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            coords_list.append(np.array(comp, dtype=int))
    return coords_list


def _has_hole(mask: np.ndarray) -> bool:
    bg = ~mask
    seen = np.zeros_like(bg, dtype=bool)
    h, w = bg.shape
    for r in range(h):
        for c in range(w):
            if not bg[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            touches_edge = False
            while stack:
                rr, cc = stack.pop()
                if rr in (0, h - 1) or cc in (0, w - 1):
                    touches_edge = True
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bg[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            if not touches_edge:
                return True
    return False


def _find_marker_side(grid: np.ndarray, bg: int) -> tuple[str, int]:
    candidates = []
    if np.all(grid[0, :] == grid[0, 0]) and int(grid[0, 0]) != bg:
        candidates.append(("top", int(grid[0, 0])))
    if np.all(grid[-1, :] == grid[-1, 0]) and int(grid[-1, 0]) != bg:
        candidates.append(("bottom", int(grid[-1, 0])))
    if np.all(grid[:, 0] == grid[0, 0]) and int(grid[0, 0]) != bg:
        candidates.append(("left", int(grid[0, 0])))
    if np.all(grid[:, -1] == grid[0, -1]) and int(grid[0, -1]) != bg:
        candidates.append(("right", int(grid[0, -1])))
    assert len(candidates) == 1, "expected exactly one full-edge marker line"
    return candidates[0]


def _direction_from_side(side: str) -> str:
    return {
        "top": "right",
        "right": "down",
        "bottom": "left",
        "left": "up",
    }[side]


def _remove_unique_shapes(symbolic: np.ndarray, shape_grid: list[list[object | None]], bg: int) -> np.ndarray:
    freq = Counter(key for row in shape_grid for key in row if key is not None)
    out = symbolic.copy()
    for r in range(symbolic.shape[0]):
        for c in range(symbolic.shape[1]):
            key = shape_grid[r][c]
            if key is not None and freq[key] == 1:
                out[r, c] = bg
    return out


def _compact(symbolic: np.ndarray, direction: str, bg: int) -> np.ndarray:
    h, w = symbolic.shape
    out = np.full_like(symbolic, bg)
    if direction == "left":
        for r in range(h):
            vals = [int(v) for v in symbolic[r] if int(v) != bg]
            out[r, : len(vals)] = vals
    elif direction == "right":
        for r in range(h):
            vals = [int(v) for v in symbolic[r] if int(v) != bg]
            out[r, w - len(vals) :] = vals
    elif direction == "up":
        for c in range(w):
            vals = [int(v) for v in symbolic[:, c] if int(v) != bg]
            out[: len(vals), c] = vals
    elif direction == "down":
        for c in range(w):
            vals = [int(v) for v in symbolic[:, c] if int(v) != bg]
            out[h - len(vals) :, c] = vals
    else:
        raise ValueError(f"bad direction: {direction}")
    return out


def _parse_symbolic(grid: np.ndarray) -> tuple[np.ndarray, list[list[object | None]], str, int]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    side, marker_color = _find_marker_side(g, bg)
    symbol_mask = (g != bg) & (g != marker_color)
    components = _component_coords(symbol_mask)
    assert components, "expected symbol components"

    items = []
    for coords in components:
        bbox = _bbox_coords(coords)
        patch = _crop_bbox(g, bbox)
        colors = sorted(set(map(int, patch[patch != bg])))
        assert len(colors) == 1, "each symbol must be monochrome"
        color = colors[0]
        rr = coords[:, 0] - bbox[0]
        cc = coords[:, 1] - bbox[1]
        local_mask = np.zeros(patch.shape, dtype=bool)
        local_mask[rr, cc] = True
        assert patch.shape[0] == patch.shape[1], "symbol bbox must be square"
        assert _has_hole(local_mask), "each symbol must be hollow"
        items.append(
            {
                "r0": bbox[0],
                "c0": bbox[1],
                "color": color,
                "shape_key": _canonical_shape_key(local_mask),
                "block_shape": patch.shape,
            }
        )

    block_shapes = {item["block_shape"] for item in items}
    assert len(block_shapes) == 1, "all symbols must share one block size"
    row_positions = sorted({int(item["r0"]) for item in items})
    col_positions = sorted({int(item["c0"]) for item in items})
    row_index = {r0: i for i, r0 in enumerate(row_positions)}
    col_index = {c0: i for i, c0 in enumerate(col_positions)}

    symbolic = np.full((len(row_positions), len(col_positions)), bg, dtype=int)
    shape_grid: list[list[object | None]] = [[None for _ in col_positions] for _ in row_positions]
    for item in items:
        r = row_index[int(item["r0"])]
        c = col_index[int(item["c0"])]
        assert symbolic[r, c] == bg, "at most one symbol per coarse cell"
        symbolic[r, c] = int(item["color"])
        shape_grid[r][c] = item["shape_key"]

    return symbolic, shape_grid, side, bg


def main(input_grid: np.ndarray) -> np.ndarray:
    """Collapse hollow symbols to pixels, drop shape singletons, then compact clockwise from the marked edge.

    Input structure:
    - The grid has a dominant background, one full-edge marker line, and several single-color hollow symbols.
    - Every symbol sits in a same-sized square block on a coarse lattice; some shapes are rotations or flips of others.
    - Fully empty coarse rows or columns do not occur, so the occupied symbol blocks determine the lattice axes.

    Rule:
    - Replace each hollow symbol by a single pixel of that symbol's color in the corresponding coarse-grid cell.
    - Group symbol shapes up to rotation and mirror symmetry; keep only shapes whose group size is at least two and delete singleton-shape symbols.
    - Use the marker edge to decode the clockwise compaction direction: top -> right, right -> down, bottom -> left, left -> up; then stably slide the remaining pixels to that side.

    Key insight:
    - The puzzle composes three necessary layers: read the big image as a coarse symbolic grid, classify shapes modulo dihedral symmetry to remove the lone forms, and only then apply the marker-selected gravity direction.
    """
    symbolic, shape_grid, side, bg = _parse_symbolic(np.asarray(input_grid, dtype=int))
    filtered = _remove_unique_shapes(symbolic, shape_grid, bg)
    direction = _direction_from_side(side)
    return _compact(filtered, direction, bg).astype(int)


def _sample_distinct_transforms(mask_id: int, count: int) -> list[int]:
    base = BASE_MASKS[mask_id]
    codes = list(range(8))
    RNG.shuffle(codes)
    out = []
    raw_keys = set()
    for code in codes:
        raw_key = _mask_key(_transform_mask(base, code))
        if raw_key in raw_keys:
            continue
        raw_keys.add(raw_key)
        out.append(code)
        if len(out) == count:
            return out
    raise RuntimeError("failed to find enough distinct transforms")


def _independent_expected_from_meta(meta: dict) -> np.ndarray:
    rows, cols = meta["shape_ids"].shape
    symbolic = np.asarray(meta["symbolic"], dtype=int)
    exact_ids = meta["exact_ids"]
    canonical_ids = meta["shape_ids"]
    bg = int(meta["bg"])
    direction = str(meta["direction"])

    canonical_grid: list[list[object | None]] = [[None for _ in range(cols)] for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            value = int(symbolic[r, c])
            if value != bg:
                canonical_grid[r][c] = int(canonical_ids[r, c])
    filtered = _remove_unique_shapes(symbolic, canonical_grid, bg)

    wrong_exact_grid: list[list[object | None]] = [[None for _ in range(cols)] for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            value = int(symbolic[r, c])
            if value != bg:
                wrong_exact_grid[r][c] = exact_ids[r, c]
    wrong_exact = _compact(_remove_unique_shapes(symbolic, wrong_exact_grid, bg), direction, bg)
    wrong_no_delete = _compact(symbolic, direction, bg)
    other_dirs = {
        name: _compact(filtered, name, bg)
        for name in ("left", "right", "up", "down")
        if name != direction
    }
    correct = _compact(filtered, direction, bg)

    assert np.any(filtered != symbolic), "singleton removal must matter"
    assert np.any(correct != filtered), "compaction must matter"
    assert np.any(correct != wrong_no_delete), "skipping singleton removal must fail"
    assert np.any(correct != wrong_exact), "rotation/flip invariance must matter"
    assert all(np.any(correct != candidate) for candidate in other_dirs.values()), "marker direction must matter"

    return correct


def _render_case(meta: dict) -> np.ndarray:
    rows = int(meta["rows"])
    cols = int(meta["cols"])
    side = str(meta["side"])
    row_gap = int(meta["row_gap"])
    col_gap = int(meta["col_gap"])

    if side == "top":
        top_margin, bottom_margin = 2, 1
        left_margin, right_margin = 1, 1
        marker_r = 0
        marker_c = None
        start_r, start_c = 2, 1
    elif side == "bottom":
        top_margin, bottom_margin = 1, 2
        left_margin, right_margin = 1, 1
        marker_r = None
        marker_c = None
        start_r, start_c = 1, 1
    elif side == "left":
        top_margin, bottom_margin = 1, 1
        left_margin, right_margin = 2, 1
        marker_r = None
        marker_c = 0
        start_r, start_c = 1, 2
    elif side == "right":
        top_margin, bottom_margin = 1, 1
        left_margin, right_margin = 1, 2
        marker_r = None
        marker_c = None
        start_r, start_c = 1, 1
    else:
        raise ValueError(side)

    height = top_margin + rows * BLOCK + (rows - 1) * row_gap + bottom_margin
    width = left_margin + cols * BLOCK + (cols - 1) * col_gap + right_margin
    grid = np.full((height, width), BG, dtype=int)

    if side == "top":
        grid[0, :] = MARKER
    elif side == "bottom":
        grid[-1, :] = MARKER
    elif side == "left":
        grid[:, 0] = MARKER
    elif side == "right":
        grid[:, -1] = MARKER

    for item in meta["items"]:
        sr, sc = item["pos"]
        r0 = start_r + sr * (BLOCK + row_gap)
        c0 = start_c + sc * (BLOCK + col_gap)
        color = int(item["color"])
        mask = item["mask"]
        grid[r0 : r0 + BLOCK, c0 : c0 + BLOCK][mask] = color

    assert max(grid.shape) <= 30
    return grid


def _build_symbol_records(capacity: int, min_total: int) -> list[dict]:
    remaining_ids = set(BASE_MASKS)
    mandatory_id = int(RNG.choice(ASYMMETRIC_BASE_IDS))
    remaining_ids.remove(mandatory_id)

    max_total = min(8, capacity - 1)
    if max_total < max(5, min_total):
        raise RuntimeError("capacity too small")

    for _ in range(200):
        dup_group_count = int(RNG.integers(1, 4))
        group_sizes = [int(RNG.integers(2, 4))]
        while len(group_sizes) < dup_group_count:
            group_sizes.append(int(RNG.integers(2, 4)))
        unique_count = int(RNG.integers(1, 3))
        total = sum(group_sizes) + unique_count
        if max(5, min_total) <= total <= max_total:
            break
    else:
        raise RuntimeError("cannot fit symbol counts")

    dup_ids = [mandatory_id]
    others = list(remaining_ids)
    RNG.shuffle(others)
    dup_ids.extend(others[: len(group_sizes) - 1])
    used = set(dup_ids)
    unique_ids = [sid for sid in BASE_MASKS if sid not in used][:unique_count]
    if len(unique_ids) < unique_count:
        raise RuntimeError("not enough unique shapes")

    records = []
    for idx, (mask_id, count) in enumerate(zip(dup_ids, group_sizes)):
        if idx == 0:
            codes = _sample_distinct_transforms(mask_id, count)
        else:
            codes = list(RNG.integers(0, 8, size=count))
        for code in codes:
            transformed = _transform_mask(BASE_MASKS[mask_id], code)
            records.append(
                {
                    "shape_id": mask_id,
                    "exact_id": _mask_key(transformed),
                    "mask": transformed,
                }
            )

    for mask_id in unique_ids:
        code = int(RNG.integers(0, 8))
        transformed = _transform_mask(BASE_MASKS[mask_id], code)
        records.append(
            {
                "shape_id": mask_id,
                "exact_id": _mask_key(transformed),
                "mask": transformed,
            }
        )

    return records


def _positions_cover_axes(positions: Iterable[tuple[int, int]], rows: int, cols: int) -> bool:
    pos = list(positions)
    return {r for r, _ in pos} == set(range(rows)) and {c for _, c in pos} == set(range(cols))


def _make_meta() -> dict | None:
    rows = int(RNG.integers(3, 5))
    cols = int(RNG.integers(3, 5))
    capacity = rows * cols
    try:
        records = _build_symbol_records(capacity, max(rows, cols) + 1)
    except RuntimeError:
        return None
    total = len(records)

    all_positions = [(r, c) for r in range(rows) for c in range(cols)]
    side = str(RNG.choice(["top", "right", "bottom", "left"]))
    direction = _direction_from_side(side)

    for _ in range(200):
        chosen_idx = RNG.choice(len(all_positions), size=total, replace=False)
        positions = [all_positions[int(i)] for i in chosen_idx]
        if not _positions_cover_axes(positions, rows, cols):
            continue

        items = []
        symbolic = np.full((rows, cols), BG, dtype=int)
        shape_ids = np.full((rows, cols), -1, dtype=int)
        exact_ids = np.empty((rows, cols), dtype=object)
        exact_ids[:, :] = None

        order = RNG.permutation(total)
        colors = RNG.permutation(np.array(SYMBOL_COLORS))[:total]
        for color, rec_idx, pos in zip(colors, order, positions):
            rec = records[int(rec_idx)]
            item = {
                "pos": pos,
                "color": int(color),
                "shape_id": int(rec["shape_id"]),
                "exact_id": rec["exact_id"],
                "mask": rec["mask"].copy(),
            }
            items.append(item)
            r, c = pos
            symbolic[r, c] = int(color)
            shape_ids[r, c] = int(rec["shape_id"])
            exact_ids[r, c] = rec["exact_id"]

        meta = {
            "bg": BG,
            "rows": rows,
            "cols": cols,
            "side": side,
            "direction": direction,
            "row_gap": int(RNG.integers(1, 3)),
            "col_gap": int(RNG.integers(1, 3)),
            "items": items,
            "symbolic": symbolic,
            "shape_ids": shape_ids,
            "exact_ids": exact_ids,
        }

        try:
            expected = _independent_expected_from_meta(meta)
        except AssertionError:
            continue
        if np.count_nonzero(expected != BG) < 2:
            continue
        meta["expected"] = expected
        return meta

    return None


def generate_input() -> np.ndarray:
    for _ in range(400):
        meta = _make_meta()
        if meta is None:
            continue
        grid = _render_case(meta)
        out = main(grid)
        if np.array_equal(out, meta["expected"]):
            return grid
    raise RuntimeError("failed to generate a valid input")
