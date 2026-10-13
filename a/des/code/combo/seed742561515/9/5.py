from __future__ import annotations

from collections import deque

import numpy as np
import random


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


SIDE_TO_DIR = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

HORIZONTAL_SIDES = {"left", "right"}
VERTICAL_SIDES = {"top", "bottom"}


def _neighbors4(r: int, c: int) -> tuple[Coord, Coord, Coord, Coord]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _bbox(coords: set[Coord]) -> BBox:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _is_connected(cells: set[Coord]) -> bool:
    if not cells:
        return False
    start = next(iter(cells))
    seen = {start}
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in cells and (nr, nc) not in seen:
                seen.add((nr, nc))
                queue.append((nr, nc))
    return len(seen) == len(cells)


def _find_components(g: np.ndarray) -> tuple[list[dict], dict[Coord, int]]:
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []
    cell_to_comp: dict[Coord, int] = {}

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == 0 or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: set[Coord] = set()
            while queue:
                cr, cc = queue.popleft()
                coords.add((cr, cc))
                cell_to_comp[(cr, cc)] = len(components)
                for nr, nc in _neighbors4(cr, cc):
                    if _in_bounds(g.shape, nr, nc) and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        queue.append((nr, nc))

            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "area": len(coords),
                }
            )

    return components, cell_to_comp


def _is_hollow_frame(g: np.ndarray, comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    color = int(comp["color"])
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    if int(comp["area"]) != 2 * h + 2 * w - 4:
        return False

    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            on_border = r in (r0, r1) or c in (c0, c1)
            if on_border:
                if int(g[r, c]) != color:
                    return False
            elif int(g[r, c]) != 0:
                return False
    return True


def _frame_side_candidates(g: np.ndarray, frame: dict) -> list[tuple[int, int, str]]:
    r0, c0, r1, c1 = frame["bbox"]
    candidates: list[tuple[int, int, str]] = []

    if r0 > 0:
        for c in range(c0, c1 + 1):
            if int(g[r0 - 1, c]) != 0:
                candidates.append((r0 - 1, c, "top"))
    if r1 + 1 < g.shape[0]:
        for c in range(c0, c1 + 1):
            if int(g[r1 + 1, c]) != 0:
                candidates.append((r1 + 1, c, "bottom"))
    if c0 > 0:
        for r in range(r0, r1 + 1):
            if int(g[r, c0 - 1]) != 0:
                candidates.append((r, c0 - 1, "left"))
    if c1 + 1 < g.shape[1]:
        for r in range(r0, r1 + 1):
            if int(g[r, c1 + 1]) != 0:
                candidates.append((r, c1 + 1, "right"))

    return candidates


def _same_color_neighbors(g: np.ndarray, r: int, c: int, color: int) -> list[Coord]:
    out = []
    for nr, nc in _neighbors4(r, c):
        if _in_bounds(g.shape, nr, nc) and int(g[nr, nc]) == color:
            out.append((nr, nc))
    return out


def _trace_indicator_line(
    g: np.ndarray,
    component_cells: set[Coord],
    endpoint: Coord,
    side: str,
    color: int,
) -> list[Coord]:
    dr, dc = SIDE_TO_DIR[side]
    line_cells = [endpoint]
    curr = endpoint

    while True:
        nr, nc = curr[0] + dr, curr[1] + dc
        assert (nr, nc) in component_cells, "indicator line must reach a source object"

        forward = (nr + dr, nc + dc)
        neighbors = [cell for cell in _same_color_neighbors(g, nr, nc, color) if cell in component_cells]
        extras = [cell for cell in neighbors if cell != curr and cell != forward]
        if extras:
            return line_cells

        if forward not in component_cells:
            return line_cells

        line_cells.append((nr, nc))
        curr = (nr, nc)


def _mask_from_cells(cells: set[Coord]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(cells)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in cells:
        mask[r - r0, c - c0] = True
    return mask


def main(grid: np.ndarray) -> np.ndarray:
    """Project mirrored copies of source objects into the target frames pointed to by their indicator lines."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    components, cell_to_comp = _find_components(g)
    frames = [comp for comp in components if _is_hollow_frame(g, comp)]
    assert frames, "expected at least one target frame"

    out = g.copy()
    used_sources: set[int] = set()

    for frame in frames:
        candidates = _frame_side_candidates(g, frame)
        assert len(candidates) == 1, "each frame must have exactly one adjacent indicator-line endpoint"
        endpoint_r, endpoint_c, side = candidates[0]
        source_idx = cell_to_comp[(endpoint_r, endpoint_c)]
        assert source_idx not in used_sources, "each source component must point to exactly one frame"
        used_sources.add(source_idx)

        source_comp = components[source_idx]
        color = int(source_comp["color"])
        line_cells = set(_trace_indicator_line(g, source_comp["coords"], (endpoint_r, endpoint_c), side, color))
        object_cells = source_comp["coords"] - line_cells

        assert object_cells, "source component must contain an object beyond the indicator line"
        assert _is_connected(object_cells), "removing the line must leave one connected source object"

        mask = _mask_from_cells(object_cells)
        if side in HORIZONTAL_SIDES:
            mirrored = np.fliplr(mask)
        else:
            mirrored = np.flipud(mask)

        fr0, fc0, fr1, fc1 = frame["bbox"]
        inner = out[fr0 + 1 : fr1, fc0 + 1 : fc1]
        inner_h, inner_w = inner.shape
        obj_h, obj_w = mirrored.shape
        assert obj_h <= inner_h and obj_w <= inner_w, "mirrored object must fit inside its frame"

        start_r = (inner_h - obj_h) // 2
        start_c = (inner_w - obj_w) // 2
        dest = inner[start_r : start_r + obj_h, start_c : start_c + obj_w]
        assert np.all(dest[mirrored] == 0), "target frame interior must be empty before projection"
        dest[mirrored] = int(frame["color"])

    return out


# --------------- generator ---------------

MIN_GRID_H, MAX_GRID_H = 20, 30
MIN_GRID_W, MAX_GRID_W = 20, 30
PAIR_COUNT_CHOICES = (2, 3)
COLOR_POOL = tuple(range(1, 10))

OBJECT_DIM_RANGE = (3, 5)
REMOVAL_COUNT_RANGE = (1, 4)
OBJECT_ATTEMPTS = 200

FRAME_PAD_RANGE = (0, 2)
LINE_LENGTH_RANGE = (2, 5)

PLACEMENT_BORDER = 1
PLACEMENT_PADDING = 2
PLACEMENT_ATTEMPTS = 120
SCENE_ATTEMPTS = 300


def _connected_mask(mask: np.ndarray) -> bool:
    cells = {(int(r), int(c)) for r, c in np.argwhere(mask)}
    return _is_connected(cells)


def _attachment_candidates(mask: np.ndarray, side: str) -> list[Coord]:
    h, w = mask.shape
    candidates: list[Coord] = []
    if side == "left":
        for r in range(h):
            if mask[r, 0] and ((r > 0 and mask[r - 1, 0]) or (r + 1 < h and mask[r + 1, 0])):
                candidates.append((r, 0))
    elif side == "right":
        c = w - 1
        for r in range(h):
            if mask[r, c] and ((r > 0 and mask[r - 1, c]) or (r + 1 < h and mask[r + 1, c])):
                candidates.append((r, c))
    elif side == "top":
        for c in range(w):
            if mask[0, c] and ((c > 0 and mask[0, c - 1]) or (c + 1 < w and mask[0, c + 1])):
                candidates.append((0, c))
    else:
        r = h - 1
        for c in range(w):
            if mask[r, c] and ((c > 0 and mask[r, c - 1]) or (c + 1 < w and mask[r, c + 1])):
                candidates.append((r, c))
    return candidates


def _random_object_mask(side: str) -> np.ndarray | None:
    for _ in range(OBJECT_ATTEMPTS):
        h = random.randint(*OBJECT_DIM_RANGE)
        w = random.randint(*OBJECT_DIM_RANGE)
        mask = np.ones((h, w), dtype=bool)

        cells = [(r, c) for r in range(h) for c in range(w)]
        random.shuffle(cells)
        removed = 0
        target_removals = random.randint(*REMOVAL_COUNT_RANGE)
        for r, c in cells:
            if removed >= target_removals:
                break
            trial = mask.copy()
            trial[r, c] = False
            if trial.sum() < 5:
                continue
            if not _connected_mask(trial):
                continue
            mask = trial
            removed += 1

        if removed == 0:
            continue
        if np.array_equal(mask, np.fliplr(mask)) or np.array_equal(mask, np.flipud(mask)):
            continue
        if not _attachment_candidates(mask, side):
            continue
        return mask
    return None


def _draw_frame(arr: np.ndarray, bbox: BBox, color: int) -> None:
    r0, c0, r1, c1 = bbox
    arr[r0, c0 : c1 + 1] = color
    arr[r1, c0 : c1 + 1] = color
    arr[r0 : r1 + 1, c0] = color
    arr[r0 : r1 + 1, c1] = color


def _build_pair(side: str, src_color: int, frame_color: int) -> tuple[np.ndarray, np.ndarray] | None:
    mask = _random_object_mask(side)
    if mask is None:
        return None
    obj_h, obj_w = mask.shape

    pad_r = random.randint(*FRAME_PAD_RANGE)
    pad_c = random.randint(*FRAME_PAD_RANGE)
    frame_h = obj_h + 2 * pad_r + 2
    frame_w = obj_w + 2 * pad_c + 2
    line_len = random.randint(*LINE_LENGTH_RANGE)

    attach_r, attach_c = random.choice(_attachment_candidates(mask, side))

    object_cells = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    line_cells: list[Coord] = []

    if side == "right":
        wall_offset = random.randint(1, frame_h - 2)
        for c in range(obj_w, obj_w + line_len):
            line_cells.append((attach_r, c))
        frame_r0 = attach_r - wall_offset
        frame_c0 = obj_w + line_len
    elif side == "left":
        wall_offset = random.randint(1, frame_h - 2)
        for c in range(-line_len, 0):
            line_cells.append((attach_r, c))
        frame_r0 = attach_r - wall_offset
        frame_c0 = -line_len - frame_w
    elif side == "bottom":
        wall_offset = random.randint(1, frame_w - 2)
        for r in range(obj_h, obj_h + line_len):
            line_cells.append((r, attach_c))
        frame_r0 = obj_h + line_len
        frame_c0 = attach_c - wall_offset
    else:
        wall_offset = random.randint(1, frame_w - 2)
        for r in range(-line_len, 0):
            line_cells.append((r, attach_c))
        frame_r0 = -line_len - frame_h
        frame_c0 = attach_c - wall_offset

    frame_bbox = (frame_r0, frame_c0, frame_r0 + frame_h - 1, frame_c0 + frame_w - 1)

    all_rows = [r for r, _c in object_cells + line_cells]
    all_cols = [c for _r, c in object_cells + line_cells]
    all_rows.extend([frame_bbox[0], frame_bbox[2]])
    all_cols.extend([frame_bbox[1], frame_bbox[3]])
    row_shift = -min(all_rows)
    col_shift = -min(all_cols)

    shifted_object = {(r + row_shift, c + col_shift) for r, c in object_cells}
    shifted_line = {(r + row_shift, c + col_shift) for r, c in line_cells}
    fr0, fc0, fr1, fc1 = (
        frame_bbox[0] + row_shift,
        frame_bbox[1] + col_shift,
        frame_bbox[2] + row_shift,
        frame_bbox[3] + col_shift,
    )

    patch_h = max(max(r for r, _c in shifted_object | shifted_line), fr1) + 1
    patch_w = max(max(c for _r, c in shifted_object | shifted_line), fc1) + 1
    inp = np.zeros((patch_h, patch_w), dtype=int)
    out = np.zeros((patch_h, patch_w), dtype=int)

    for r, c in shifted_object | shifted_line:
        inp[r, c] = src_color
        out[r, c] = src_color

    _draw_frame(inp, (fr0, fc0, fr1, fc1), frame_color)
    _draw_frame(out, (fr0, fc0, fr1, fc1), frame_color)

    if side in HORIZONTAL_SIDES:
        mirrored = np.fliplr(mask)
    else:
        mirrored = np.flipud(mask)

    start_r = fr0 + 1 + pad_r
    start_c = fc0 + 1 + pad_c
    for r, c in np.argwhere(mirrored):
        out[start_r + int(r), start_c + int(c)] = frame_color

    return inp, out


def _random_sides(n_pairs: int) -> list[str]:
    sides = [random.choice(tuple(HORIZONTAL_SIDES)), random.choice(tuple(VERTICAL_SIDES))]
    while len(sides) < n_pairs:
        sides.append(random.choice(("left", "right", "top", "bottom")))
    random.shuffle(sides)
    return sides


def _place_patches(patches: list[tuple[np.ndarray, np.ndarray]], h: int, w: int) -> tuple[np.ndarray, np.ndarray] | None:
    forbidden = np.zeros((h, w), dtype=bool)
    placements: list[tuple[int, int]] = [(-1, -1)] * len(patches)
    order = sorted(
        range(len(patches)),
        key=lambda idx: patches[idx][0].shape[0] * patches[idx][0].shape[1],
        reverse=True,
    )

    for idx in order:
        patch_h, patch_w = patches[idx][0].shape
        row_lo = PLACEMENT_BORDER
        row_hi = h - PLACEMENT_BORDER - patch_h
        col_lo = PLACEMENT_BORDER
        col_hi = w - PLACEMENT_BORDER - patch_w
        if row_hi < row_lo or col_hi < col_lo:
            return None

        candidates = [(r, c) for r in range(row_lo, row_hi + 1) for c in range(col_lo, col_hi + 1)]
        random.shuffle(candidates)
        placed = False
        for r, c in candidates[:PLACEMENT_ATTEMPTS]:
            rr0 = max(0, r - PLACEMENT_PADDING)
            rr1 = min(h, r + patch_h + PLACEMENT_PADDING)
            cc0 = max(0, c - PLACEMENT_PADDING)
            cc1 = min(w, c + patch_w + PLACEMENT_PADDING)
            if forbidden[rr0:rr1, cc0:cc1].any():
                continue
            forbidden[rr0:rr1, cc0:cc1] = True
            placements[idx] = (r, c)
            placed = True
            break
        if not placed:
            return None

    inp = np.zeros((h, w), dtype=int)
    out = np.zeros((h, w), dtype=int)
    for (patch_in, patch_out), (r, c) in zip(patches, placements):
        patch_h, patch_w = patch_in.shape
        in_view = inp[r : r + patch_h, c : c + patch_w]
        out_view = out[r : r + patch_h, c : c + patch_w]
        in_mask = patch_in != 0
        out_mask = patch_out != 0
        assert not np.any(in_view[in_mask]), "patch placement must not overlap"
        assert not np.any(out_view[out_mask]), "patch placement must not overlap"
        in_view[in_mask] = patch_in[in_mask]
        out_view[out_mask] = patch_out[out_mask]

    return inp, out


def _gen_attempt() -> np.ndarray | None:
    n_pairs = random.choice(PAIR_COUNT_CHOICES)
    colors = random.sample(COLOR_POOL, 2 * n_pairs)
    sides = _random_sides(n_pairs)

    patches: list[tuple[np.ndarray, np.ndarray]] = []
    for i, side in enumerate(sides):
        pair = _build_pair(side, colors[2 * i], colors[2 * i + 1])
        if pair is None:
            return None
        patches.append(pair)

    max_h = max(patch_in.shape[0] for patch_in, _patch_out in patches)
    max_w = max(patch_in.shape[1] for patch_in, _patch_out in patches)
    min_h = max(MIN_GRID_H, max_h + 2 * PLACEMENT_BORDER)
    min_w = max(MIN_GRID_W, max_w + 2 * PLACEMENT_BORDER)
    if min_h > MAX_GRID_H or min_w > MAX_GRID_W:
        return None

    for _ in range(40):
        h = random.randint(min_h, MAX_GRID_H)
        w = random.randint(min_w, MAX_GRID_W)
        placed = _place_patches(patches, h, w)
        if placed is None:
            continue
        inp, out = placed
        try:
            if np.array_equal(main(inp), out):
                return inp
        except AssertionError:
            return None

    return None


def generate_input() -> np.ndarray:
    for _ in range(SCENE_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
