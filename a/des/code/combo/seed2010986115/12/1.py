from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR_TO_VEC = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
DIR_NAMES = tuple(DIR_TO_VEC)

NUM_OBJECT_OPTIONS = [2, 3, 4]
NUM_OBJECT_WEIGHTS = [2, 3, 2]
SHAPE_AREA_RANGE = (3, 6)
SHAPE_BOX_LIMIT = 4
FRAME_EXTRA_MARGIN = (0, 2)
MIN_CORRIDOR_LEN = 4
PLACEMENT_RETRIES = 120
KEY_PATCH_RETRIES = 24
GEN_RETRIES = 300


def _components(mask: np.ndarray, connectivity: int = 4) -> list[list[tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 4:
        dirs = DIRS4
    else:
        dirs = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc]

    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in dirs:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _coords_to_mask(coords: list[tuple[int, int]] | set[tuple[int, int]]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask


def _shape_signature(mask: np.ndarray) -> tuple[int, int, tuple[int, ...]]:
    arr = np.asarray(mask, dtype=bool)
    return (
        int(arr.shape[0]),
        int(arr.shape[1]),
        tuple(int(v) for v in arr.astype(np.uint8).ravel()),
    )


def _all_transforms(mask: np.ndarray) -> list[np.ndarray]:
    arr = np.asarray(mask, dtype=bool)
    out = []
    for k in range(4):
        rot = np.rot90(arr, k)
        out.append(rot)
        out.append(np.fliplr(rot))
    return out


def _canonical_shape(mask: np.ndarray) -> tuple[int, int, tuple[int, ...]]:
    return min(_shape_signature(t) for t in _all_transforms(mask))


def _transform_by_index(mask: np.ndarray, index: int) -> np.ndarray:
    arr = np.rot90(np.asarray(mask, dtype=bool), index % 4)
    if index >= 4:
        arr = np.fliplr(arr)
    return arr


def _perimeter_coords(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    coords = [(r0, c) for c in range(c0, c1 + 1)]
    coords += [(r1, c) for c in range(c0, c1 + 1)]
    coords += [(r, c0) for r in range(r0 + 1, r1)]
    coords += [(r, c1) for r in range(r0 + 1, r1)]
    return coords


def _line_to_edge(shape: tuple[int, int], start: tuple[int, int], step: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    r, c = start
    dr, dc = step
    cells = []
    while 0 <= r < h and 0 <= c < w:
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def _ray_cells(grid: np.ndarray, start: tuple[int, int], step: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = grid.shape
    r, c = start
    dr, dc = step
    cells = []
    while 0 <= r < h and 0 <= c < w and int(grid[r, c]) == BG:
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def _classify_frame(grid: np.ndarray, comp: list[tuple[int, int]]) -> dict | None:
    pts = set(comp)
    r0, c0, r1, c1 = _bbox(comp)
    if r1 - r0 + 1 < 5 or c1 - c0 + 1 < 5:
        return None

    perimeter = set(_perimeter_coords(r0, c0, r1, c1))
    if not pts.issubset(perimeter):
        return None
    missing = list(perimeter - pts)
    if len(missing) != 1 or len(pts) != len(perimeter) - 1:
        return None

    gap = missing[0]
    gr, gc = gap
    if (gr in (r0, r1)) and (gc in (c0, c1)):
        return None
    if gr == r0:
        direction = "up"
    elif gr == r1:
        direction = "down"
    elif gc == c0:
        direction = "left"
    else:
        direction = "right"

    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    inner_comps = _components(inner != BG, connectivity=4)
    if len(inner_comps) != 1:
        return None
    island_local = inner_comps[0]
    ih, iw = inner.shape
    if any(r in (0, ih - 1) or c in (0, iw - 1) for r, c in island_local):
        return None

    island_mask = _coords_to_mask(island_local)
    return {
        "bbox": (r0, c0, r1, c1),
        "gap": gap,
        "direction": direction,
        "island_mask": island_mask,
        "canonical": _canonical_shape(island_mask),
    }


def _find_frames(grid: np.ndarray) -> list[dict]:
    frames = []
    for color in np.unique(grid):
        color = int(color)
        if color == BG:
            continue
        for comp in _components(grid == color, connectivity=4):
            frame = _classify_frame(grid, comp)
            if frame is not None:
                frames.append(frame)
    return frames


def _find_keys(grid: np.ndarray, frames: list[dict]) -> list[dict]:
    masked = np.zeros_like(grid, dtype=bool)
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        masked[r0 : r1 + 1, c0 : c1 + 1] = True

    keys = []
    for comp in _components((grid != BG) & ~masked, connectivity=4):
        color_counts = Counter(int(grid[r, c]) for r, c in comp)
        if len(color_counts) != 2:
            raise AssertionError("each exterior object must be a two-color key")
        singleton_colors = [color for color, count in color_counts.items() if count == 1]
        if len(singleton_colors) != 1:
            raise AssertionError("each key must have exactly one singleton marker color")
        marker_color = singleton_colors[0]
        body_color = next(color for color in color_counts if color != marker_color)

        body_coords = [(r, c) for r, c in comp if int(grid[r, c]) == body_color]
        if len(body_coords) < 2:
            raise AssertionError("key bodies must have at least two pixels")
        body_mask = _coords_to_mask(body_coords)
        if len(_components(body_mask, connectivity=4)) != 1:
            raise AssertionError("key body must stay connected")

        keys.append(
            {
                "marker_color": marker_color,
                "canonical": _canonical_shape(body_mask),
            }
        )
    return keys


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frames = _find_frames(g)
    assert frames, "expected at least one framed island"
    keys = _find_keys(g, frames)
    assert len(frames) == len(keys), "frame/key counts must match"

    frame_by_shape: dict[tuple[int, int, tuple[int, ...]], dict] = {}
    for frame in frames:
        sig = frame["canonical"]
        assert sig not in frame_by_shape, "shape matching must be unambiguous"
        frame_by_shape[sig] = frame

    key_by_shape: dict[tuple[int, int, tuple[int, ...]], dict] = {}
    for key in keys:
        sig = key["canonical"]
        assert sig not in key_by_shape, "shape matching must be unambiguous"
        key_by_shape[sig] = key

    assert set(frame_by_shape) == set(key_by_shape), "every key must match one island"

    out = g.copy()
    for sig, frame in frame_by_shape.items():
        marker_color = int(key_by_shape[sig]["marker_color"])
        step = DIR_TO_VEC[frame["direction"]]
        r, c = frame["gap"]
        while 0 <= r < g.shape[0] and 0 <= c < g.shape[1] and int(out[r, c]) == BG:
            out[r, c] = marker_color
            r += step[0]
            c += step[1]
    return out


def _normalize_cells(cells: set[tuple[int, int]]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(cells)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in cells:
        mask[r - r0, c - c0] = True
    return mask


def _random_shape(rng: random.Random) -> np.ndarray | None:
    for _ in range(200):
        target_area = rng.randint(*SHAPE_AREA_RANGE)
        cells = {(0, 0)}
        while len(cells) < target_area:
            frontier = []
            for r, c in cells:
                for dr, dc in DIRS4:
                    cand = (r + dr, c + dc)
                    if cand not in cells:
                        frontier.append(cand)
            cells.add(rng.choice(frontier))

        mask = _normalize_cells(cells)
        h, w = mask.shape
        if h > SHAPE_BOX_LIMIT or w > SHAPE_BOX_LIMIT:
            continue
        if h == 1 and w == 1:
            continue
        return mask
    return None


def _sample_unique_shapes(rng: random.Random, count: int) -> list[np.ndarray] | None:
    shapes = []
    seen = set()
    while len(shapes) < count:
        shape = _random_shape(rng)
        if shape is None:
            return None
        sig = _canonical_shape(shape)
        if sig in seen:
            continue
        seen.add(sig)
        shapes.append(shape)
    return shapes


def _mark_cells(occupied: np.ndarray, buffered: np.ndarray, cells: list[tuple[int, int]]) -> None:
    h, w = occupied.shape
    for r, c in cells:
        occupied[r, c] = True
    for r, c in cells:
        buffered[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)] = True


def _can_place_cells(
    cells: list[tuple[int, int]],
    buffered: np.ndarray,
    forbidden: set[tuple[int, int]] | None = None,
    allowed: set[tuple[int, int]] | None = None,
) -> bool:
    h, w = buffered.shape
    forbidden = forbidden or set()
    allowed = allowed or set()
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if r in (0, h - 1) or c in (0, w - 1):
            return False
        if buffered[r, c]:
            return False
        if (r, c) in forbidden and (r, c) not in allowed:
            return False
    return True


def _build_key_patch(body_mask: np.ndarray, body_color: int, marker_color: int, rng: random.Random) -> np.ndarray:
    h, w = body_mask.shape
    canvas = np.zeros((h + 2, w + 2), dtype=int)
    canvas[1 : h + 1, 1 : w + 1][body_mask] = body_color

    candidates = set()
    body_coords = np.argwhere(canvas == body_color)
    for r, c in body_coords:
        for dr, dc in DIRS4:
            nr, nc = int(r + dr), int(c + dc)
            if 0 <= nr < h + 2 and 0 <= nc < w + 2 and canvas[nr, nc] == BG:
                candidates.add((nr, nc))
    mr, mc = rng.choice(list(candidates))
    canvas[mr, mc] = marker_color

    coords = np.argwhere(canvas != BG)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return canvas[r0 : r1 + 1, c0 : c1 + 1]


def _patch_cells(patch: np.ndarray, top: int, left: int) -> list[tuple[int, int]]:
    return [(top + int(r), left + int(c)) for r, c in np.argwhere(patch != BG)]


def _grid_size(rng: random.Random, n_objects: int) -> tuple[int, int]:
    min_side = 16 + 3 * n_objects
    max_side = min_side + 6
    return rng.randint(min_side, max_side), rng.randint(min_side, max_side)


def _blocked_indices(rng: random.Random, count: int) -> set[int]:
    choices = list(range(count + 1))
    weights = [1] + [3] * (count - 1) + [1]
    blocked_count = rng.choices(choices, weights=weights, k=1)[0]
    return set(rng.sample(range(count), blocked_count))


def _sample_frame_anchor(
    rng: random.Random,
    h: int,
    w: int,
    frame_h: int,
    frame_w: int,
    direction: str,
) -> tuple[int, int] | None:
    r_min = 1
    c_min = 1
    r_max = h - frame_h - 1
    c_max = w - frame_w - 1
    if direction == "up":
        r_min = max(r_min, MIN_CORRIDOR_LEN - 1)
    elif direction == "down":
        r_max = min(r_max, h - frame_h - MIN_CORRIDOR_LEN + 1)
    elif direction == "left":
        c_min = max(c_min, MIN_CORRIDOR_LEN - 1)
    else:
        c_max = min(c_max, w - frame_w - MIN_CORRIDOR_LEN + 1)
    if r_min > r_max or c_min > c_max:
        return None
    return rng.randint(r_min, r_max), rng.randint(c_min, c_max)


def _place_frames(rng: random.Random, specs: list[dict], shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]] | None:
    h, w = shape
    grid = np.zeros((h, w), dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    buffered = np.zeros((h, w), dtype=bool)
    placed: list[dict | None] = [None] * len(specs)

    order = sorted(range(len(specs)), key=lambda i: specs[i]["frame_h"] * specs[i]["frame_w"], reverse=True)
    for index in order:
        spec = specs[index]
        placed_here = None
        for _ in range(PLACEMENT_RETRIES):
            anchor = _sample_frame_anchor(rng, h, w, spec["frame_h"], spec["frame_w"], spec["direction"])
            if anchor is None:
                return None
            r0, c0 = anchor
            r1 = r0 + spec["frame_h"] - 1
            c1 = c0 + spec["frame_w"] - 1

            max_row_shift = spec["inner_h"] - spec["island_mask"].shape[0] - 2
            max_col_shift = spec["inner_w"] - spec["island_mask"].shape[1] - 2
            row_shift = rng.randint(0, max_row_shift)
            col_shift = rng.randint(0, max_col_shift)

            if spec["direction"] in ("up", "down"):
                gap = (r0 if spec["direction"] == "up" else r1, rng.randint(c0 + 1, c1 - 1))
            else:
                gap = (rng.randint(r0 + 1, r1 - 1), c0 if spec["direction"] == "left" else c1)

            border_cells = [cell for cell in _perimeter_coords(r0, c0, r1, c1) if cell != gap]
            island_top = r0 + 2 + row_shift
            island_left = c0 + 2 + col_shift
            island_cells = [
                (island_top + int(rr), island_left + int(cc))
                for rr, cc in np.argwhere(spec["island_mask"])
            ]
            all_cells = border_cells + island_cells
            if not _can_place_cells(all_cells, buffered):
                continue

            for r, c in border_cells:
                grid[r, c] = spec["frame_color"]
            for r, c in island_cells:
                grid[r, c] = spec["island_color"]
            _mark_cells(occupied, buffered, all_cells)
            placed_here = {
                **spec,
                "bbox": (r0, c0, r1, c1),
                "gap": gap,
                "border_cells": border_cells,
                "island_cells": island_cells,
                "corridor": _line_to_edge((h, w), gap, DIR_TO_VEC[spec["direction"]]),
            }
            break
        if placed_here is None:
            return None
        placed[index] = placed_here

    return grid, occupied, buffered, [item for item in placed if item is not None]


def _place_key(
    rng: random.Random,
    grid: np.ndarray,
    occupied: np.ndarray,
    buffered: np.ndarray,
    key_spec: dict,
    all_corridors: set[tuple[int, int]],
) -> dict | None:
    h, w = grid.shape
    own_corridor = key_spec["corridor"]
    own_set = set(own_corridor)
    other_forbidden = all_corridors - (own_set if key_spec["blocked"] else set())
    corridor_index = {cell: idx for idx, cell in enumerate(own_corridor)}

    for _ in range(KEY_PATCH_RETRIES):
        body_mask = _transform_by_index(key_spec["base_shape"], rng.randrange(8))
        patch = _build_key_patch(body_mask, key_spec["body_color"], key_spec["marker_color"], rng)
        patch_coords = [(int(r), int(c)) for r, c in np.argwhere(patch != BG)]

        if key_spec["blocked"]:
            candidates = own_corridor[MIN_CORRIDOR_LEN:]
            rng.shuffle(candidates)
            for target in candidates:
                rel_coords = patch_coords[:]
                rng.shuffle(rel_coords)
                for pr, pc in rel_coords:
                    top = target[0] - pr
                    left = target[1] - pc
                    cells = _patch_cells(patch, top, left)
                    if not _can_place_cells(cells, buffered, other_forbidden, own_set):
                        continue
                    hit_indices = [corridor_index[cell] for cell in cells if cell in corridor_index]
                    if not hit_indices or min(hit_indices) < MIN_CORRIDOR_LEN:
                        continue
                    for r, c in cells:
                        grid[r, c] = patch[r - top, c - left]
                    _mark_cells(occupied, buffered, cells)
                    return {
                        **key_spec,
                        "patch": patch,
                        "cells": cells,
                        "top": top,
                        "left": left,
                    }
        else:
            if h - patch.shape[0] - 2 < 1 or w - patch.shape[1] - 2 < 1:
                continue
            for _ in range(PLACEMENT_RETRIES):
                top = rng.randint(1, h - patch.shape[0] - 2)
                left = rng.randint(1, w - patch.shape[1] - 2)
                cells = _patch_cells(patch, top, left)
                if not _can_place_cells(cells, buffered, all_corridors):
                    continue
                for r, c in cells:
                    grid[r, c] = patch[r - top, c - left]
                _mark_cells(occupied, buffered, cells)
                return {
                    **key_spec,
                    "patch": patch,
                    "cells": cells,
                    "top": top,
                    "left": left,
                }
    return None


def _expected_from_specs(grid: np.ndarray, frames: list[dict], keys: list[dict]) -> np.ndarray:
    out = grid.copy()
    marker_by_index = {key["index"]: key["marker_color"] for key in keys}
    for frame in frames:
        color = int(marker_by_index[frame["index"]])
        for r, c in _ray_cells(grid, frame["gap"], DIR_TO_VEC[frame["direction"]]):
            out[r, c] = color
    return out


def _validate_corridors(frame_grid: np.ndarray, frames: list[dict]) -> bool:
    seen: set[tuple[int, int]] = set()
    for frame in frames:
        corridor = frame["corridor"]
        if len(corridor) < MIN_CORRIDOR_LEN:
            return False
        if len(_ray_cells(frame_grid, frame["gap"], DIR_TO_VEC[frame["direction"]])) != len(corridor):
            return False
        if seen.intersection(corridor):
            return False
        seen.update(corridor)
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_objects = rng.choices(NUM_OBJECT_OPTIONS, weights=NUM_OBJECT_WEIGHTS, k=1)[0]
    base_shapes = _sample_unique_shapes(rng, n_objects)
    if base_shapes is None:
        return None

    h, w = _grid_size(rng, n_objects)
    blocked = _blocked_indices(rng, n_objects)
    frame_specs = []
    for index, base_shape in enumerate(base_shapes):
        island_mask = _transform_by_index(base_shape, rng.randrange(8))
        frame_color, island_color, key_body_color, marker_color = rng.sample(range(1, 10), 4)
        extra_h = rng.randint(*FRAME_EXTRA_MARGIN)
        extra_w = rng.randint(*FRAME_EXTRA_MARGIN)
        inner_h = island_mask.shape[0] + 2 + extra_h
        inner_w = island_mask.shape[1] + 2 + extra_w
        frame_specs.append(
            {
                "index": index,
                "base_shape": base_shape,
                "island_mask": island_mask,
                "shape_sig": _canonical_shape(base_shape),
                "direction": rng.choice(DIR_NAMES),
                "frame_h": inner_h + 2,
                "frame_w": inner_w + 2,
                "inner_h": inner_h,
                "inner_w": inner_w,
                "frame_color": frame_color,
                "island_color": island_color,
                "body_color": key_body_color,
                "marker_color": marker_color,
                "blocked": index in blocked,
            }
        )

    placed_frames_result = _place_frames(rng, frame_specs, (h, w))
    if placed_frames_result is None:
        return None
    grid, occupied, buffered, frames = placed_frames_result
    if not _validate_corridors(grid, frames):
        return None

    key_specs = [
        {
            "index": frame["index"],
            "base_shape": frame["base_shape"],
            "shape_sig": frame["shape_sig"],
            "body_color": frame["body_color"],
            "marker_color": frame["marker_color"],
            "blocked": frame["blocked"],
            "corridor": frame["corridor"],
        }
        for frame in frames
    ]
    key_order = sorted(range(len(key_specs)), key=lambda i: (not key_specs[i]["blocked"], -len(key_specs[i]["corridor"])))
    all_corridors = set(cell for frame in frames for cell in frame["corridor"])

    keys: list[dict | None] = [None] * len(key_specs)
    for order_index in key_order:
        placed_key = _place_key(rng, grid, occupied, buffered, key_specs[order_index], all_corridors)
        if placed_key is None:
            return None
        keys[order_index] = placed_key
    keys = [item for item in keys if item is not None]

    for frame in frames:
        path = _ray_cells(grid, frame["gap"], DIR_TO_VEC[frame["direction"]])
        if len(path) < 2:
            return None
        reaches_edge = len(path) == len(frame["corridor"])
        if frame["blocked"] and reaches_edge:
            return None
        if not frame["blocked"] and not reaches_edge:
            return None

    expected = _expected_from_specs(grid, frames, keys)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
