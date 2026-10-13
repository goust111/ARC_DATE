from __future__ import annotations

import random

import numpy as np


_DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
_SIDES = ("top", "bottom", "left", "right")

_GEN_BG = 0
_GEN_COLORS = tuple(range(1, 10))
_GEN_ATTEMPTS = 400


def _dominant_color(grid: np.ndarray) -> int:
    flat = np.asarray(grid, dtype=int).ravel()
    counts = np.bincount(flat)
    return int(np.argmax(counts))


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
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
                cr, cc = stack.pop()
                cells.append((int(cr), int(cc)))
                for dr, dc in _DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))

            rows = [cell_r for cell_r, _cell_c in cells]
            cols = [cell_c for _cell_r, cell_c in cells]
            comps.append(
                {
                    "color": color,
                    "cells": cells,
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                    "area": len(cells),
                }
            )

    return comps


def _border_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _line_orientation(coords: np.ndarray) -> str | None:
    if coords.shape[0] < 2:
        return None

    rows = coords[:, 0]
    cols = coords[:, 1]

    if np.all(rows == rows[0]):
        ordered = np.sort(cols)
        expected = np.arange(int(ordered[0]), int(ordered[0]) + ordered.size)
        if np.array_equal(ordered, expected):
            return "horizontal"

    if np.all(cols == cols[0]):
        ordered = np.sort(rows)
        expected = np.arange(int(ordered[0]), int(ordered[0]) + ordered.size)
        if np.array_equal(ordered, expected):
            return "vertical"

    return None


def _gap_side(local_r: int, local_c: int, h: int, w: int) -> str | None:
    if local_r == 0 and 0 < local_c < w - 1:
        return "top"
    if local_r == h - 1 and 0 < local_c < w - 1:
        return "bottom"
    if local_c == 0 and 0 < local_r < h - 1:
        return "left"
    if local_c == w - 1 and 0 < local_r < h - 1:
        return "right"
    return None


def _parse_frame(grid: np.ndarray, comp: dict, bg: int) -> dict | None:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 3 or w < 3:
        return None

    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    border = _border_mask(h, w)
    border_values = patch[border]
    color = int(comp["color"])

    if not np.all((border_values == bg) | (border_values == color)):
        return None
    if int(np.count_nonzero(border_values == bg)) != 1:
        return None
    if int(np.count_nonzero(border_values == color)) != int(border.sum()) - 1:
        return None

    comp_mask = np.zeros((h, w), dtype=bool)
    for rr, cc in comp["cells"]:
        comp_mask[int(rr - r0), int(cc - c0)] = True
    if np.any(comp_mask & ~border):
        return None
    if int(comp["area"]) != int(border.sum()) - 1:
        return None

    missing = np.argwhere(border & ~comp_mask)
    if missing.shape != (1, 2):
        return None
    gap_r, gap_c = map(int, missing[0])
    gap_side = _gap_side(gap_r, gap_c, h, w)
    if gap_side is None or int(grid[r0 + gap_r, c0 + gap_c]) != bg:
        return None

    if h <= 2 or w <= 2:
        return None
    interior = patch[1:-1, 1:-1]
    marker_coords = np.argwhere(interior != bg)
    if marker_coords.shape[0] < 2:
        return None

    marker_colors = {int(value) for value in np.unique(interior[interior != bg])}
    if len(marker_colors) != 1:
        return None
    marker_color = next(iter(marker_colors))
    if marker_color == color:
        return None

    marker_orientation = _line_orientation(marker_coords)
    if marker_orientation is None:
        return None

    marker_cells = [(r0 + 1 + int(rr), c0 + 1 + int(cc)) for rr, cc in marker_coords]
    return {
        "bbox": (r0, c0, r1, c1),
        "patch": patch.copy(),
        "border_cells": list(comp["cells"]),
        "marker_cells": marker_cells,
        "marker_orientation": marker_orientation,
        "gap_side": gap_side,
    }


def _extract_objects(grid: np.ndarray) -> tuple[int, list[dict]]:
    bg = _dominant_color(grid)
    objects: list[dict] = []
    covered = np.zeros_like(grid, dtype=bool)

    for comp in _find_components(grid, bg):
        parsed = _parse_frame(grid, comp, bg)
        if parsed is None:
            continue
        objects.append(parsed)
        for rr, cc in parsed["border_cells"]:
            covered[int(rr), int(cc)] = True
        for rr, cc in parsed["marker_cells"]:
            covered[int(rr), int(cc)] = True

    assert objects, "expected at least one gapped frame object"
    assert np.array_equal(covered, grid != bg), "all foreground must belong to a frame object"
    objects.sort(key=lambda item: item["bbox"])
    return bg, objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror each gapped frame across the axis implied by its inner marker line and attach the copy outside the gap side.

    Input structure:
    - The background is black.
    - The foreground consists of disjoint one-pixel-thick rectangular frames, each drawn in one color with exactly one missing non-corner border cell.
    - Inside each frame, a different-colored contiguous marker segment lies entirely on one row or one column.

    Rule:
    - For each frame, inspect the marker segment orientation.
    - A horizontal marker segment means mirror the whole framed patch left-right, while a vertical marker segment means mirror it up-down.
    - Translate the mirrored copy outward along the frame's gap side by exactly one frame width or height so the copy touches the original from that side.
    - Keep every original frame unchanged and draw all translated mirrored copies on the same black background.

    Key insight:
    - The inner marker line and the border gap encode different decisions: the line chooses the reflection axis, while the gap chooses where the reflected copy is placed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg, objects = _extract_objects(g)
    h0, w0 = g.shape
    copy_specs: list[tuple[np.ndarray, int, int]] = []
    min_r = 0
    min_c = 0
    max_r = int(h0 - 1)
    max_c = int(w0 - 1)

    for obj in objects:
        r0, c0, r1, c1 = map(int, obj["bbox"])
        patch = obj["patch"]
        h, w = patch.shape
        if obj["marker_orientation"] == "horizontal":
            mirrored = patch[:, ::-1]
        else:
            mirrored = patch[::-1, :]

        if obj["gap_side"] == "top":
            dr, dc = -h, 0
        elif obj["gap_side"] == "bottom":
            dr, dc = h, 0
        elif obj["gap_side"] == "left":
            dr, dc = 0, -w
        else:
            dr, dc = 0, w

        target_r = int(r0 + dr)
        target_c = int(c0 + dc)
        copy_specs.append((mirrored, target_r, target_c))
        min_r = min(min_r, target_r)
        min_c = min(min_c, target_c)
        max_r = max(max_r, target_r + h - 1)
        max_c = max(max_c, target_c + w - 1)

    out = np.full((max_r - min_r + 1, max_c - min_c + 1), bg, dtype=int)
    row_offset = -min_r
    col_offset = -min_c
    out[row_offset : row_offset + h0, col_offset : col_offset + w0] = g

    for patch, target_r, target_c in copy_specs:
        h, w = patch.shape
        view = out[target_r + row_offset : target_r + row_offset + h, target_c + col_offset : target_c + col_offset + w]
        mask = patch != bg
        view[mask] = patch[mask]

    return out


def _choose_asymmetric_segment(axis_len: int, rng: random.Random) -> tuple[int, int]:
    lengths = list(range(2, axis_len))
    rng.shuffle(lengths)
    for length in lengths:
        starts = [start for start in range(axis_len - length + 1) if 2 * start != axis_len - length]
        if starts:
            return int(rng.choice(starts)), int(length)
    raise RuntimeError("failed to choose a non-symmetric segment")


def _make_object(border_color: int, marker_color: int, orientation: str, gap_side: str, rng: random.Random) -> dict:
    if orientation == "horizontal":
        h = rng.randint(4, 6)
        w = rng.randint(5, 7)
    else:
        h = rng.randint(5, 7)
        w = rng.randint(4, 6)

    patch = np.full((h, w), _GEN_BG, dtype=int)
    patch[0, :] = border_color
    patch[-1, :] = border_color
    patch[:, 0] = border_color
    patch[:, -1] = border_color

    if gap_side in {"top", "bottom"}:
        gap_idx = rng.randint(1, w - 2)
        gap_r = 0 if gap_side == "top" else h - 1
        gap_c = gap_idx
    else:
        gap_idx = rng.randint(1, h - 2)
        gap_r = gap_idx
        gap_c = 0 if gap_side == "left" else w - 1
    patch[gap_r, gap_c] = _GEN_BG

    inner_h = h - 2
    inner_w = w - 2
    if orientation == "horizontal":
        row = rng.randint(0, inner_h - 1)
        start, length = _choose_asymmetric_segment(inner_w, rng)
        patch[1 + row, 1 + start : 1 + start + length] = marker_color
    else:
        col = rng.randint(0, inner_w - 1)
        start, length = _choose_asymmetric_segment(inner_h, rng)
        patch[1 + start : 1 + start + length, 1 + col] = marker_color

    if gap_side == "top":
        union_h, union_w = 2 * h, w
        origin_r, origin_c = h, 0
    elif gap_side == "bottom":
        union_h, union_w = 2 * h, w
        origin_r, origin_c = 0, 0
    elif gap_side == "left":
        union_h, union_w = h, 2 * w
        origin_r, origin_c = 0, w
    else:
        union_h, union_w = h, 2 * w
        origin_r, origin_c = 0, 0

    return {
        "patch": patch,
        "union_h": union_h,
        "union_w": union_w,
        "origin_r": origin_r,
        "origin_c": origin_c,
    }


def _free_positions(occupied: np.ndarray, rect_h: int, rect_w: int, pad: int = 1) -> list[tuple[int, int]]:
    h, w = occupied.shape
    positions: list[tuple[int, int]] = []
    for r in range(h - rect_h + 1):
        r0 = max(0, r - pad)
        r1 = min(h, r + rect_h + pad)
        for c in range(w - rect_w + 1):
            c0 = max(0, c - pad)
            c1 = min(w, c + rect_w + pad)
            if not np.any(occupied[r0:r1, c0:c1]):
                positions.append((int(r), int(c)))
    return positions


def _balanced_orientations(count: int, rng: random.Random) -> list[str]:
    base = count // 2
    horizontal = base
    vertical = base
    if count % 2 == 1:
        if rng.random() < 0.5:
            horizontal += 1
        else:
            vertical += 1
    orientations = ["horizontal"] * horizontal + ["vertical"] * vertical
    rng.shuffle(orientations)
    return orientations


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    obj_count = rng.randint(1, 4)
    colors = rng.sample(list(_GEN_COLORS), 2 * obj_count)
    gap_sides = rng.sample(list(_SIDES), obj_count)
    orientations = _balanced_orientations(obj_count, rng)

    specs = [
        _make_object(colors[2 * idx], colors[2 * idx + 1], orientations[idx], gap_sides[idx], rng)
        for idx in range(obj_count)
    ]
    specs.sort(key=lambda item: item["union_h"] * item["union_w"], reverse=True)

    total_union_area = sum(spec["union_h"] * spec["union_w"] for spec in specs)
    target_area = int(np.ceil(total_union_area * 1.5))
    min_h = max(12, max(spec["union_h"] for spec in specs) + 1, int(np.ceil(np.sqrt(target_area))))
    min_w = max(12, max(spec["union_w"] for spec in specs) + 1, int(np.ceil(target_area / min_h)))
    if min_h > 30 or min_w > 30:
        return None

    grid_h = rng.randint(min_h, 30)
    min_w = max(min_w, int(np.ceil(target_area / grid_h)))
    if min_w > 30:
        return None
    grid_w = rng.randint(min_w, 30)
    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    placements: list[tuple[dict, int, int]] = []

    for spec in specs:
        positions = _free_positions(occupied, spec["union_h"], spec["union_w"], pad=1)
        if not positions:
            return None
        union_r, union_c = rng.choice(positions)
        occupied[union_r : union_r + spec["union_h"], union_c : union_c + spec["union_w"]] = True
        placements.append((spec, int(union_r), int(union_c)))

    grid = np.full((grid_h, grid_w), _GEN_BG, dtype=int)
    for spec, union_r, union_c in placements:
        patch = spec["patch"]
        h, w = patch.shape
        r0 = union_r + spec["origin_r"]
        c0 = union_c + spec["origin_c"]
        view = grid[r0 : r0 + h, c0 : c0 + w]
        mask = patch != _GEN_BG
        view[mask] = patch[mask]

    try:
        out = main(grid)
    except Exception:
        return None
    if max(grid.shape) > 30 or max(out.shape) > 30:
        return None
    if out.shape != grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        rng = random.Random()
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
