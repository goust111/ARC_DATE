from __future__ import annotations

import random

import numpy as np


BG = 0
DIRECTIONS = ("up", "down", "left", "right")
DIR_TO_UNIT = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
SCAN_TO_PROJECTION = {
    (-1, 0): "down",
    (1, 0): "up",
    (0, -1): "right",
    (0, 1): "left",
}
PALETTE_COLORS = tuple(range(1, 10))
GRID_H_RANGE = (14, 24)
GRID_W_RANGE = (14, 24)
PALETTE_LEN_RANGE = (4, 6)
OBJECT_COUNT_RANGE = (2, 4)
SHAPE_SIDE_RANGE = (2, 4)
SHAPE_AREA_RANGE = (3, 6)
MAX_ATTEMPTS = 400


def _find_palette(grid: np.ndarray) -> tuple[int, list[int]]:
    row = np.asarray(grid[-1], dtype=int)
    runs: list[tuple[int, list[int]]] = []
    c = 0
    while c < row.size:
        if int(row[c]) == BG:
            c += 1
            continue
        start = c
        colors: list[int] = []
        while c < row.size and int(row[c]) != BG:
            colors.append(int(row[c]))
            c += 1
        if len(colors) >= 2 and len(set(colors)) == len(colors):
            runs.append((start, colors))
    assert runs, "expected a distinct non-black palette strip on the bottom row"
    start, colors = max(runs, key=lambda item: len(item[1]))
    return start, colors


def _components(scene: np.ndarray) -> list[dict]:
    h, w = scene.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(scene[r, c])
            if color == BG or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(scene[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            coords = np.array(pts, dtype=int)
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "size": int(coords.shape[0]),
                }
            )
    return comps


def _scan_stamp_from_arrow(
    arrow: tuple[int, int],
    color: int,
    scene: np.ndarray,
    stamp_lookup: np.ndarray,
    stamps: list[dict],
) -> tuple[int, str]:
    h, w = scene.shape
    ar, ac = arrow
    candidates: list[tuple[int, str]] = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        r = ar
        c = ac
        gap = 0
        while True:
            r += dr
            c += dc
            if not (0 <= r < h and 0 <= c < w):
                break
            value = int(scene[r, c])
            if value == BG:
                gap += 1
                continue
            if gap >= 1:
                stamp_id = int(stamp_lookup[r, c])
                if stamp_id >= 0 and int(stamps[stamp_id]["color"]) == color:
                    candidates.append((stamp_id, SCAN_TO_PROJECTION[(dr, dc)]))
            break
    assert len(candidates) == 1, "each arrow must identify exactly one same-color stamp"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project each colored stamp repeatedly in its arrow direction using the bottom palette order.

    Input structure:
    - Black is the background.
    - The bottom row contains one contiguous palette strip of distinct non-black colors read from left to right.
    - Above the palette are several 4-connected monochrome stamp objects, and each used stamp color also has one detached same-color single-pixel arrow dot.

    Rule:
    - Locate the palette strip and map every palette color to its left-to-right index.
    - Pair each arrow dot with the first same-color stamp reached by scanning through background along a cardinal ray; the opposite of that scan direction is the stamp's projection direction.
    - For each stamp, copy its exact shape every (stamp span in that direction + 1) cells, recoloring successive copies with the next palette colors until the palette ends or the next copy would leave the non-palette area.

    Key insight:
    - The bottom strip simultaneously encodes color succession and the maximum number of recolored copies, while the detached one-pixel marker supplies only the translation direction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h >= 2 and w >= 2, "grid is too small"

    _, palette = _find_palette(g)
    palette_index = {color: idx for idx, color in enumerate(palette)}
    scene = g[:-1, :]
    comps = _components(scene)

    stamps: list[dict] = []
    arrows: list[dict] = []
    for comp in comps:
        color = int(comp["color"])
        assert color in palette_index, "all non-palette objects must use palette colors"
        if int(comp["size"]) == 1:
            arrows.append(comp)
        else:
            stamps.append(comp)

    stamp_lookup = np.full(scene.shape, -1, dtype=int)
    for stamp_id, stamp in enumerate(stamps):
        coords = np.asarray(stamp["coords"], dtype=int)
        stamp_lookup[coords[:, 0], coords[:, 1]] = stamp_id

    pairings: dict[int, str] = {}
    for arrow in arrows:
        coords = np.asarray(arrow["coords"], dtype=int)
        arrow_pt = (int(coords[0, 0]), int(coords[0, 1]))
        stamp_id, direction = _scan_stamp_from_arrow(
            arrow_pt,
            int(arrow["color"]),
            scene,
            stamp_lookup,
            stamps,
        )
        assert stamp_id not in pairings, "each stamp should have one arrow"
        pairings[stamp_id] = direction

    assert len(pairings) == len(stamps), "every stamp must have exactly one arrow"

    out = g.copy()
    scene_h = h - 1
    for stamp_id, stamp in enumerate(stamps):
        color = int(stamp["color"])
        color_pos = int(palette_index[color])
        direction = pairings[stamp_id]
        coords = np.asarray(stamp["coords"], dtype=int)
        r0, c0, r1, c1 = stamp["bbox"]
        span = (r1 - r0 + 1) if direction in ("up", "down") else (c1 - c0 + 1)
        step = span + 1
        unit_r, unit_c = DIR_TO_UNIT[direction]
        dr_step = unit_r * step
        dc_step = unit_c * step

        for offset, next_color in enumerate(palette[color_pos + 1 :], start=1):
            shifted = coords + np.array([dr_step * offset, dc_step * offset], dtype=int)
            if (
                shifted[:, 0].min() < 0
                or shifted[:, 0].max() >= scene_h
                or shifted[:, 1].min() < 0
                or shifted[:, 1].max() >= w
            ):
                break
            out[shifted[:, 0], shifted[:, 1]] = int(next_color)

    return out


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _random_shape(rng: random.Random) -> np.ndarray:
    h = rng.randint(*SHAPE_SIDE_RANGE)
    w = rng.randint(*SHAPE_SIDE_RANGE)
    target = rng.randint(SHAPE_AREA_RANGE[0], min(SHAPE_AREA_RANGE[1], h * w - 1))

    start = (rng.randrange(h), rng.randrange(w))
    cells = {start}
    while len(cells) < target:
        frontier: list[tuple[int, int]] = []
        for r, c in cells:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            break
        cells.add(rng.choice(frontier))

    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True

    if rng.random() < 0.5:
        mask = np.fliplr(mask)
    mask = np.rot90(mask, rng.randrange(4))
    return _crop_mask(mask)


def _arrow_options(mask: np.ndarray) -> dict[str, list[tuple[int, int]]]:
    h, w = mask.shape
    options = {direction: [] for direction in DIRECTIONS}
    for c in range(w):
        if mask[0, c] and not mask[h - 1, c]:
            options["up"].append((-2, c))
        if mask[h - 1, c] and not mask[0, c]:
            options["down"].append((h + 1, c))
    for r in range(h):
        if mask[r, 0] and not mask[r, w - 1]:
            options["left"].append((r, -2))
        if mask[r, w - 1] and not mask[r, 0]:
            options["right"].append((r, w + 1))
    return options


def _max_fit(scene_h: int, width: int, top: int, left: int, mask: np.ndarray, direction: str) -> int:
    h, w = mask.shape
    span = h if direction in ("up", "down") else w
    step = span + 1
    if direction == "up":
        return top // step
    if direction == "down":
        return (scene_h - top - h) // step
    if direction == "left":
        return left // step
    return (width - left - w) // step


def _paint_cells(grid: np.ndarray, coords: np.ndarray, color: int) -> None:
    grid[coords[:, 0], coords[:, 1]] = color


def _place_object(
    grid: np.ndarray,
    input_occ: np.ndarray,
    output_occ: np.ndarray,
    palette: list[int],
    color_idx: int,
    direction: str,
    stop_mode: str,
    rng: random.Random,
) -> bool:
    scene_h, width = grid.shape[0] - 1, grid.shape[1]
    color = int(palette[color_idx])
    remaining = len(palette) - 1 - color_idx
    assert remaining >= 1

    for _ in range(120):
        mask = _random_shape(rng)
        rel_coords = np.argwhere(mask)
        options = _arrow_options(mask)[direction]
        if not options:
            continue
        rng.shuffle(options)
        for arrow_rel in options:
            min_rel_r = min(0, arrow_rel[0])
            max_rel_r = max(mask.shape[0] - 1, arrow_rel[0])
            min_rel_c = min(0, arrow_rel[1])
            max_rel_c = max(mask.shape[1] - 1, arrow_rel[1])

            row_lo = -min_rel_r
            row_hi = scene_h - 1 - max_rel_r
            col_lo = -min_rel_c
            col_hi = width - 1 - max_rel_c
            if row_lo > row_hi or col_lo > col_hi:
                continue

            candidates: list[tuple[int, int, int]] = []
            for top in range(row_lo, row_hi + 1):
                for left in range(col_lo, col_hi + 1):
                    max_copies = _max_fit(scene_h, width, top, left, mask, direction)
                    if max_copies < 1:
                        continue
                    if stop_mode == "boundary" and max_copies >= remaining:
                        continue
                    if stop_mode == "palette" and max_copies < remaining:
                        continue

                    stamp_abs = rel_coords + np.array([top, left], dtype=int)
                    arrow_abs = np.array([[top + arrow_rel[0], left + arrow_rel[1]]], dtype=int)

                    if input_occ[stamp_abs[:, 0], stamp_abs[:, 1]].any():
                        continue
                    if input_occ[arrow_abs[:, 0], arrow_abs[:, 1]].any():
                        continue

                    span = mask.shape[0] if direction in ("up", "down") else mask.shape[1]
                    step = span + 1
                    unit_r, unit_c = DIR_TO_UNIT[direction]
                    dr_step = unit_r * step
                    dc_step = unit_c * step
                    copy_count = min(max_copies, remaining)

                    blocked = False
                    if output_occ[stamp_abs[:, 0], stamp_abs[:, 1]].any():
                        blocked = True
                    if output_occ[arrow_abs[:, 0], arrow_abs[:, 1]].any():
                        blocked = True
                    if blocked:
                        continue
                    for k in range(1, copy_count + 1):
                        shifted = stamp_abs + np.array([dr_step * k, dc_step * k], dtype=int)
                        if output_occ[shifted[:, 0], shifted[:, 1]].any():
                            blocked = True
                            break
                    if blocked:
                        continue

                    candidates.append((top, left, copy_count))

            if not candidates:
                continue

            top, left, copy_count = rng.choice(candidates)
            stamp_abs = rel_coords + np.array([top, left], dtype=int)
            arrow_abs = np.array([[top + arrow_rel[0], left + arrow_rel[1]]], dtype=int)
            _paint_cells(grid, stamp_abs, color)
            _paint_cells(grid, arrow_abs, color)
            input_occ[stamp_abs[:, 0], stamp_abs[:, 1]] = True
            input_occ[arrow_abs[:, 0], arrow_abs[:, 1]] = True
            output_occ[stamp_abs[:, 0], stamp_abs[:, 1]] = True
            output_occ[arrow_abs[:, 0], arrow_abs[:, 1]] = True

            span = mask.shape[0] if direction in ("up", "down") else mask.shape[1]
            step = span + 1
            unit_r, unit_c = DIR_TO_UNIT[direction]
            dr_step = unit_r * step
            dc_step = unit_c * step
            for k in range(1, copy_count + 1):
                shifted = stamp_abs + np.array([dr_step * k, dc_step * k], dtype=int)
                output_occ[shifted[:, 0], shifted[:, 1]] = True
            return True
    return False


def _attempt_grid(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    palette_len = rng.randint(PALETTE_LEN_RANGE[0], min(PALETTE_LEN_RANGE[1], min(len(PALETTE_COLORS), w - 2)))
    if palette_len < 4:
        return None

    palette = rng.sample(PALETTE_COLORS, palette_len)
    grid = np.zeros((h, w), dtype=int)

    start_min = 1
    start_max = w - palette_len - 1
    if start_max < start_min:
        return None
    palette_start = rng.randint(start_min, start_max)
    grid[h - 1, palette_start : palette_start + palette_len] = palette

    input_occ = np.zeros_like(grid, dtype=bool)
    output_occ = np.zeros_like(grid, dtype=bool)
    input_occ[h - 1, palette_start : palette_start + palette_len] = True
    output_occ[h - 1, palette_start : palette_start + palette_len] = True

    max_objects = min(OBJECT_COUNT_RANGE[1], palette_len - 1, len(DIRECTIONS))
    if max_objects < OBJECT_COUNT_RANGE[0]:
        return None
    object_count = rng.randint(OBJECT_COUNT_RANGE[0], max_objects)

    boundary_count = object_count // 2
    if boundary_count > palette_len - 2:
        boundary_count = palette_len - 2
    if boundary_count < 1:
        return None

    boundary_indices = rng.sample(list(range(palette_len - 2)), boundary_count)
    remaining_indices = [idx for idx in range(palette_len - 1) if idx not in set(boundary_indices)]
    palette_indices = rng.sample(remaining_indices, object_count - boundary_count)

    specs: list[dict] = []
    for idx in boundary_indices:
        specs.append({"color_idx": idx, "mode": "boundary"})
    for idx in palette_indices:
        specs.append({"color_idx": idx, "mode": "palette"})
    rng.shuffle(specs)

    directions = rng.sample(list(DIRECTIONS), object_count)
    for spec, direction in zip(specs, directions):
        spec["direction"] = direction

    specs.sort(key=lambda spec: (spec["mode"] != "boundary", -spec["color_idx"]))
    for spec in specs:
        placed = _place_object(
            grid,
            input_occ,
            output_occ,
            palette,
            int(spec["color_idx"]),
            str(spec["direction"]),
            str(spec["mode"]),
            rng,
        )
        if not placed:
            return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_ATTEMPTS):
        grid = _attempt_grid(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted attempts")
