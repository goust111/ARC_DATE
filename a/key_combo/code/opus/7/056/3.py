from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


def _bg_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _neighbors4(r: int, c: int) -> tuple[Coord, Coord, Coord, Coord]:
    return (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)


def _bbox_from_coords(coords: list[Coord]) -> BBox:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components_same_color(grid: np.ndarray, mask: np.ndarray | None = None) -> list[dict]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if visited[r, c]:
                continue
            if mask is not None and not bool(mask[r, c]):
                continue
            color = int(grid[r, c])
            if color == 0 and mask is None:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            coords: list[Coord] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if visited[nr, nc]:
                        continue
                    if mask is not None and not bool(mask[nr, nc]):
                        continue
                    if int(grid[nr, nc]) != color:
                        continue
                    visited[nr, nc] = True
                    stack.append((nr, nc))
            if coords:
                comps.append(
                    {
                        "color": color,
                        "coords": coords,
                        "bbox": _bbox_from_coords(coords),
                        "area": len(coords),
                    }
                )
    return comps


def _components_from_mask(mask: np.ndarray) -> list[list[Coord]]:
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=bool)
    comps: list[list[Coord]] = []
    for r in range(h):
        for c in range(w):
            if visited[r, c] or not bool(mask[r, c]):
                continue
            stack = [(r, c)]
            visited[r, c] = True
            coords: list[Coord] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and bool(mask[nr, nc]):
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _cropped_mask(coords: list[Coord]) -> tuple[np.ndarray, tuple[int, int]]:
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        out[r - r0, c - c0] = True
    return out, (r0, c0)


def _shape_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for flip in (False, True):
        base = np.fliplr(mask) if flip else mask
        for k in range(4):
            var = np.rot90(base, k)
            key = (tuple(map(int, var.shape)), var.astype(np.uint8).tobytes())
            if key not in seen:
                seen.add(key)
                variants.append(var.copy())
    return variants


def _canonical_shape_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    keys = []
    for variant in _shape_variants(mask):
        keys.append((tuple(map(int, variant.shape)), variant.astype(np.uint8).tobytes()))
    return min(keys)


def _estimate_axis2(frame_mask: np.ndarray) -> int:
    h, w = frame_mask.shape
    center_axis2 = w - 1
    best = (10**9, 10**9, center_axis2)
    for axis2 in range(2 * w - 1):
        score = 0
        for c in range(w):
            mc = axis2 - c
            col = frame_mask[:, c]
            if 0 <= mc < w:
                if c > mc:
                    continue
                if c < mc:
                    score += int(np.count_nonzero(col != frame_mask[:, mc]))
            else:
                score += int(np.count_nonzero(col))
        cand = (score, abs(axis2 - center_axis2), axis2)
        if cand < best:
            best = cand
    return best[2]


def _mirror_vertical(mask: np.ndarray, axis2: int) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros_like(mask, dtype=bool)
    rows, cols = np.where(mask)
    for r, c in zip(rows.tolist(), cols.tolist()):
        mc = axis2 - c
        if 0 <= mc < w:
            out[r, mc] = True
    return out


def _majority_color(values: list[int]) -> int:
    return int(Counter(values).most_common(1)[0][0])


def _sprite_shape_colors(grid: np.ndarray, frame_bbox: BBox, bg: int) -> dict[tuple[tuple[int, int], bytes], Counter]:
    h, w = grid.shape
    r0, c0, r1, c1 = frame_bbox
    outside_mask = np.ones((h, w), dtype=bool)
    outside_mask[r0 : r1 + 1, c0 : c1 + 1] = False
    comps = _components_same_color(grid, mask=outside_mask & (grid != bg))
    by_shape: dict[tuple[tuple[int, int], bytes], Counter] = defaultdict(Counter)
    for comp in comps:
        sprite_mask, _ = _cropped_mask(comp["coords"])
        key = _canonical_shape_key(sprite_mask)
        by_shape[key][int(comp["color"])] += 1
    return by_shape


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete a nearly mirror-symmetric frame's missing holes, then fill every hole using shape-matched outside sprites.

    Input structure:
    - Black background surrounds one large monochrome frame object near the center.
    - Inside the frame's bounding box, several black holes appear; some hole counterparts are missing on one side of a vertical symmetry axis.
    - Outside the frame, small monochrome sprites act as shape keys whose colors determine the hole fills.

    Rule:
    - Find the large frame and estimate its vertical mirror axis from the mostly symmetric frame pattern.
    - Mirror the existing hole mask across that axis and remove frame pixels on the missing side to complete the symmetric hole set.
    - For each completed hole shape, find an outside sprite with the same shape up to rotation or reflection, fill the hole with that sprite's color, and erase all outside sprites to black.

    Key insight:
    - The asymmetry is itself the instruction: first mirror-complete the hidden holes, then use the external sprites as shape-coded color keys for the final fill.
    """
    g = np.asarray(grid, dtype=int)
    bg = 0

    frame_candidates = [comp for comp in _components_same_color(g) if int(comp["color"]) != bg]
    assert frame_candidates, "expected a large central frame component"
    frame = max(frame_candidates, key=lambda comp: int(comp["area"]))
    frame_color = int(frame["color"])
    fr0, fc0, fr1, fc1 = frame["bbox"]

    sub = g[fr0 : fr1 + 1, fc0 : fc1 + 1]
    frame_mask = sub == frame_color
    axis2 = _estimate_axis2(frame_mask)

    hole_mask = sub != frame_color
    completed_holes = hole_mask | _mirror_vertical(hole_mask, axis2)

    sprite_map = _sprite_shape_colors(g, frame["bbox"], bg)

    out = g.copy()
    outside = np.ones_like(g, dtype=bool)
    outside[fr0 : fr1 + 1, fc0 : fc1 + 1] = False
    out[outside & (out != bg)] = bg

    for comp in _components_from_mask(completed_holes):
        hole_shape, (hr0, hc0) = _cropped_mask(comp)
        key = _canonical_shape_key(hole_shape)
        assert key in sprite_map, "every completed hole shape must have a matching outside sprite"
        color = max(sprite_map[key].items(), key=lambda kv: (kv[1], -kv[0]))[0]
        for rr, cc in comp:
            out[fr0 + rr, fc0 + cc] = int(color)

    return out


FRAME_COLOR = 1
SPRITE_COLORS = [2, 3, 4, 7, 8, 9]
GEN_ATTEMPTS = 400


def _shape_library() -> list[np.ndarray]:
    raw = [
        [[1]],
        [[1, 1]],
        [[1, 1, 1]],
        [[1, 0], [1, 1]],
        [[1, 1], [1, 1]],
        [[1, 1, 1], [0, 1, 0]],
        [[1, 1, 0], [0, 1, 1]],
        [[1, 0], [1, 1], [0, 1]],
        [[1, 0, 1], [1, 1, 1]],
    ]
    shapes: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for arr in raw:
        mask = np.array(arr, dtype=bool)
        key = _canonical_shape_key(mask)
        if key not in seen:
            seen.add(key)
            shapes.append(mask)
    return shapes


SHAPE_LIBRARY = _shape_library()


def _stamp_mask(base: np.ndarray, shape: np.ndarray, top: int, left: int, value: bool = True) -> None:
    rows, cols = np.where(shape)
    for r, c in zip(rows.tolist(), cols.tolist()):
        base[top + r, left + c] = value


def _cell_list(shape: np.ndarray, top: int, left: int) -> list[Coord]:
    rows, cols = np.where(shape)
    return [(top + int(r), left + int(c)) for r, c in zip(rows.tolist(), cols.tolist())]


def _can_place_hole(occupied: np.ndarray, shape: np.ndarray, top: int, left: int) -> bool:
    h, w = occupied.shape
    sh, sw = shape.shape
    if top < 1 or left < 1 or top + sh > h - 1 or left + sw > w - 1:
        return False
    rr0 = max(0, top - 1)
    rr1 = min(h, top + sh + 1)
    cc0 = max(0, left - 1)
    cc1 = min(w, left + sw + 1)
    return not bool(np.any(occupied[rr0:rr1, cc0:cc1]))


def _mirror_left(shape: np.ndarray, left: int, axis2: int) -> int:
    return int(axis2 - (left + shape.shape[1] - 1))


def _can_place_sprite_bbox(occupied: np.ndarray, top: int, left: int, height: int, width: int) -> bool:
    h, w = occupied.shape
    if top < 0 or left < 0 or top + height > h or left + width > w:
        return False
    rr0 = max(0, top - 1)
    rr1 = min(h, top + height + 1)
    cc0 = max(0, left - 1)
    cc1 = min(w, left + width + 1)
    return not bool(np.any(occupied[rr0:rr1, cc0:cc1]))


def _expected_output(
    grid_shape: tuple[int, int],
    frame_bbox: BBox,
    frame_color: int,
    pairs: list[dict],
) -> np.ndarray:
    out = np.zeros(grid_shape, dtype=int)
    fr0, fc0, fr1, fc1 = frame_bbox
    out[fr0 : fr1 + 1, fc0 : fc1 + 1] = frame_color
    for pair in pairs:
        color = int(pair["color"])
        for coords in (pair["left_cells"], pair["right_cells"]):
            for rr, cc in coords:
                out[fr0 + rr, fc0 + cc] = color
    return out


def _place_sprites(
    grid: np.ndarray,
    occupied: np.ndarray,
    sprite_specs: list[tuple[np.ndarray, int]],
    rng: random.Random,
) -> bool:
    for shape, color in sprite_specs:
        oriented = rng.choice(_shape_variants(shape))
        sh, sw = oriented.shape
        candidates = [(r, c) for r in range(grid.shape[0] - sh + 1) for c in range(grid.shape[1] - sw + 1)]
        rng.shuffle(candidates)
        placed = False
        for top, left in candidates:
            if not _can_place_sprite_bbox(occupied, top, left, sh, sw):
                continue
            rows, cols = np.where(oriented)
            for r, c in zip(rows.tolist(), cols.tolist()):
                grid[top + r, left + c] = color
            occupied[top : top + sh, left : left + sw] = True
            placed = True
            break
        if not placed:
            return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    grid_h = rng.randint(18, 26)
    grid_w = rng.randint(18, 26)
    frame_h = rng.randint(10, min(16, grid_h - 4))
    frame_w = rng.randint(11, min(17, grid_w - 4))
    frame_r0 = rng.randint(2, grid_h - frame_h - 2)
    frame_c0 = rng.randint(2, grid_w - frame_w - 2)
    frame_bbox = (frame_r0, frame_c0, frame_r0 + frame_h - 1, frame_c0 + frame_w - 1)

    shape_count = rng.randint(3, min(5, len(SHAPE_LIBRARY), len(SPRITE_COLORS)))
    chosen_shapes = rng.sample(SHAPE_LIBRARY, shape_count)
    chosen_colors = rng.sample(SPRITE_COLORS, shape_count)
    axis2 = frame_w - 1

    completed_holes = np.zeros((frame_h, frame_w), dtype=bool)
    pairs: list[dict] = []

    total_pairs_target = rng.randint(max(4, shape_count), shape_count + 3)
    per_shape_counts = [1] * shape_count
    for _ in range(total_pairs_target - shape_count):
        per_shape_counts[rng.randrange(shape_count)] += 1

    for shape, color, pair_count in zip(chosen_shapes, chosen_colors, per_shape_counts):
        sh, sw = shape.shape
        mirrored_shape = np.fliplr(shape)
        placed = 0
        for _ in range(120):
            if placed >= pair_count:
                break
            top = rng.randint(1, frame_h - sh - 1)
            left = rng.randint(1, frame_w - sw - 1)
            mirror_left = _mirror_left(shape, left, axis2)
            if left >= mirror_left:
                continue
            if mirror_left < 1 or mirror_left + sw > frame_w - 1:
                continue
            if not _can_place_hole(completed_holes, shape, top, left):
                continue
            temp_holes = completed_holes.copy()
            _stamp_mask(temp_holes, shape, top, left, True)
            if not _can_place_hole(temp_holes, mirrored_shape, top, mirror_left):
                continue

            _stamp_mask(completed_holes, shape, top, left, True)
            _stamp_mask(completed_holes, mirrored_shape, top, mirror_left, True)
            pairs.append(
                {
                    "shape": shape,
                    "key": _canonical_shape_key(shape),
                    "color": color,
                    "left_cells": _cell_list(shape, top, left),
                    "right_cells": _cell_list(mirrored_shape, top, mirror_left),
                }
            )
            placed += 1
        if placed < pair_count:
            return None

    frame_density = 1.0 - float(np.count_nonzero(completed_holes)) / float(frame_h * frame_w)
    if not (0.68 <= frame_density <= 0.9):
        return None

    states = []
    removed_indices = []
    intact_indices = []
    for idx in range(len(pairs)):
        state = rng.choices(["intact", "remove_left", "remove_right"], weights=[3, 2, 2])[0]
        states.append(state)
        if state == "intact":
            intact_indices.append(idx)
        else:
            removed_indices.append(idx)
    if not removed_indices:
        flip_idx = rng.randrange(len(pairs))
        states[flip_idx] = rng.choice(["remove_left", "remove_right"])
        removed_indices = [flip_idx]
        intact_indices = [i for i, state in enumerate(states) if state == "intact"]
    if not intact_indices:
        keep_idx = rng.randrange(len(pairs))
        states[keep_idx] = "intact"

    input_holes = completed_holes.copy()
    for pair, state in zip(pairs, states):
        if state == "remove_left":
            for rr, cc in pair["left_cells"]:
                input_holes[rr, cc] = False
        elif state == "remove_right":
            for rr, cc in pair["right_cells"]:
                input_holes[rr, cc] = False

    grid = np.zeros((grid_h, grid_w), dtype=int)
    grid[frame_r0 : frame_r0 + frame_h, frame_c0 : frame_c0 + frame_w] = FRAME_COLOR
    hole_rows, hole_cols = np.where(input_holes)
    for rr, cc in zip(hole_rows.tolist(), hole_cols.tolist()):
        grid[frame_r0 + rr, frame_c0 + cc] = 0

    occupied = np.zeros_like(grid, dtype=bool)
    occupied[frame_r0 : frame_r0 + frame_h, frame_c0 : frame_c0 + frame_w] = True
    sprite_specs = list(zip(chosen_shapes, chosen_colors))
    rng.shuffle(sprite_specs)
    if not _place_sprites(grid, occupied, sprite_specs, rng):
        return None

    expected = _expected_output(grid.shape, frame_bbox, FRAME_COLOR, pairs)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None

    if np.array_equal(actual, grid):
        return None
    if np.count_nonzero(actual == FRAME_COLOR) <= np.count_nonzero(grid == FRAME_COLOR) // 2:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid puzzle-3 input")
