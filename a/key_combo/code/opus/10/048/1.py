from __future__ import annotations

import random
from typing import Iterable

import numpy as np


NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
FOREGROUND = 1
BACKGROUND = 0


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in NEIGHBORS4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _extract_objects(grid: np.ndarray, bg: int) -> list[dict]:
    objects = []
    for coords in _components_mask(grid != bg):
        colors = {int(grid[r, c]) for r, c in coords}
        assert len(colors) == 1, "each foreground object must be monochromatic"
        color = next(iter(colors))
        objects.append(
            {
                "coords": coords.astype(int),
                "color": int(color),
                "bbox": _bbox_coords(coords),
            }
        )
    return objects


def _row_extents(coords: np.ndarray) -> dict[int, tuple[int, int]]:
    extents: dict[int, tuple[int, int]] = {}
    for r_raw, c_raw in coords:
        r = int(r_raw)
        c = int(c_raw)
        if r not in extents:
            extents[r] = (c, c)
        else:
            lo, hi = extents[r]
            extents[r] = (min(lo, c), max(hi, c))
    return extents


def _pair_info(a: dict, b: dict) -> dict | None:
    a_rows = _row_extents(a["coords"])
    b_rows = _row_extents(b["coords"])
    shared_rows = sorted(set(a_rows) & set(b_rows))
    if not shared_rows:
        return None

    segments = []
    min_gap: int | None = None
    for r in shared_rows:
        a0, a1 = a_rows[r]
        b0, b1 = b_rows[r]
        if a1 < b0:
            left_hi = a1
            right_lo = b0
        elif b1 < a0:
            left_hi = b1
            right_lo = a0
        else:
            raise AssertionError("shared-row objects must not overlap")
        gap = int(right_lo - left_hi - 1)
        segments.append((int(r), int(left_hi + 1), int(right_lo - 1), gap))
        min_gap = gap if min_gap is None else min(min_gap, gap)

    return {
        "pair": (int(a["id"]), int(b["id"])),
        "gap": int(min_gap),
        "segments": segments,
        "shared_rows": shared_rows,
    }


def _shift_coords(coords: np.ndarray, dc: int, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    shifted = coords + np.array([0, int(dc)], dtype=int)
    keep = (
        (0 <= shifted[:, 0])
        & (shifted[:, 0] < h)
        & (0 <= shifted[:, 1])
        & (shifted[:, 1] < w)
    )
    return shifted[keep]


def _shifted_objects(objects: list[dict], shape: tuple[int, int]) -> list[dict]:
    shifted = []
    for rank, obj in enumerate(objects, start=1):
        dc = -1 if rank % 2 == 1 else 1
        moved = _shift_coords(obj["coords"], dc, shape)
        assert len(moved) > 0, "shifted object cannot disappear completely"
        shifted.append({"id": obj["id"], "color": obj["color"], "coords": moved})
    return shifted


def _render_objects(objects: Iterable[dict], shape: tuple[int, int], bg: int) -> np.ndarray:
    grid = np.full(shape, int(bg), dtype=int)
    for obj in objects:
        coords = obj["coords"]
        grid[coords[:, 0], coords[:, 1]] = int(obj["color"])
    return grid


def _analyze_pairs(objects: list[dict]) -> list[dict]:
    infos = []
    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            info = _pair_info(objects[i], objects[j])
            if info is not None:
                infos.append(info)
    return infos


def _unique_min_pair(objects: list[dict]) -> dict:
    infos = _analyze_pairs(objects)
    assert infos, "expected at least one object pair with shared rows"
    min_gap = min(info["gap"] for info in infos)
    winners = [info for info in infos if info["gap"] == min_gap]
    assert len(winners) == 1, "post-shift closest pair must be unique"
    return winners[0]


def _solve_from_objects(objects: list[dict], shape: tuple[int, int], bg: int) -> np.ndarray:
    moved = _shifted_objects(objects, shape)
    out = _render_objects(moved, shape, bg)
    winner = _unique_min_pair(moved)
    color = int(moved[0]["color"])
    for r, c0, c1, gap in winner["segments"]:
        if gap > 0:
            out[r, c0 : c1 + 1] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Shift all blue objects by alternating direction, then connect only the uniquely closest post-shift aligned pair.

    Input structure
    - The grid has a dominant black background and 4-6 blue 4-connected foreground objects.
    - Each object is monochromatic, non-overlapping, and has a distinct bbox top row, so the top-to-bottom order is unique.
    - Objects may overlap in row span, but the decisive horizontal relationship is defined only after the rank-based shift.

    Rule
    - Sort the foreground objects by bbox top row.
    - Shift rank 1, 3, 5, ... one column left and rank 2, 4, 6, ... one column right, dropping any out-of-bounds pixels.
    - Among post-shift object pairs that share at least one occupied row, compute the minimum horizontal gap on those shared rows.
    - Find the unique smallest-gap pair and fill every gap cell between the two objects on each shared row; keep all other shifted pixels unchanged.

    Key insight
    - The connection target is not marked in the input: the alternating shift creates a new closest aligned pair, and only that emergent pair is bridged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)
    objects = _extract_objects(g, bg)
    assert 4 <= len(objects) <= 6, "expected 4-6 foreground objects"

    top_rows = [obj["bbox"][0] for obj in objects]
    assert len(set(top_rows)) == len(top_rows), "foreground objects must have distinct top rows"

    ordered = []
    for obj_id, obj in enumerate(sorted(objects, key=lambda item: item["bbox"][0])):
        ordered.append(
            {
                "id": obj_id,
                "color": obj["color"],
                "coords": obj["coords"],
                "bbox": obj["bbox"],
            }
        )

    return _solve_from_objects(ordered, g.shape, bg)


SHAPE_LIBRARY = [
    np.array([[1, 1], [1, 0]], dtype=int),
    np.array([[1, 1], [0, 1]], dtype=int),
    np.array([[1, 0], [1, 1], [0, 1]], dtype=int),
    np.array([[0, 1], [1, 1], [1, 0]], dtype=int),
    np.array([[1, 1], [1, 0], [1, 0]], dtype=int),
    np.array([[1, 0], [1, 0], [1, 1]], dtype=int),
    np.array([[1, 1], [0, 1], [0, 1]], dtype=int),
    np.array([[0, 1], [0, 1], [1, 1]], dtype=int),
    np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=int),
    np.array([[0, 1, 1], [0, 1, 0], [1, 1, 0]], dtype=int),
    np.array([[1, 0], [1, 1], [1, 0], [0, 1]], dtype=int),
    np.array([[0, 1], [1, 1], [0, 1], [1, 0]], dtype=int),
]

TARGET_FINAL_GAPS = (1, 2)
GEN_ATTEMPTS = 400


def _shape_signature(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return mask.shape, mask.astype(np.uint8).tobytes()


def _random_shape() -> np.ndarray:
    mask = random.choice(SHAPE_LIBRARY).copy()
    if random.random() < 0.5:
        mask = np.fliplr(mask)
    if random.random() < 0.5:
        mask = np.flipud(mask)
    return mask.astype(int)


def _sample_shapes(count: int) -> list[np.ndarray]:
    seen = set()
    shapes = []
    attempts = 0
    while len(shapes) < count and attempts < 200:
        attempts += 1
        shape = _random_shape()
        sig = _shape_signature(shape)
        if sig in seen:
            continue
        seen.add(sig)
        shapes.append(shape)
    if len(shapes) != count:
        raise RuntimeError("failed to sample distinct asymmetric shapes")
    return shapes


def _coords_from_mask(mask: np.ndarray, top: int, left: int) -> np.ndarray:
    coords = np.argwhere(mask != 0).astype(int)
    coords[:, 0] += int(top)
    coords[:, 1] += int(left)
    return coords


def _final_left_bounds(rank: int, width: int, grid_w: int) -> tuple[int, int]:
    min_left = 0
    max_left = grid_w - width
    if rank % 2 == 0:
        min_left = 1
    else:
        max_left -= 1
    return min_left, max_left


def _sample_top_rows(shapes: list[np.ndarray], chain_start: int) -> tuple[list[int], int]:
    count = len(shapes)
    heights = [int(shape.shape[0]) for shape in shapes]
    tops = [0] * count
    cursor = random.randint(1, 2)

    for idx in range(chain_start):
        tops[idx] = cursor
        cursor += heights[idx] + random.randint(1, 2)

    h0, h1, h2 = heights[chain_start : chain_start + 3]
    t0 = cursor
    d01 = random.randint(1, h0 - 1)
    t1 = t0 + d01
    lo = t0 + h0
    hi = t1 + h1 - 1
    if lo > hi:
        raise ValueError("no feasible chain overlap")
    t2 = random.randint(lo, hi)
    tops[chain_start] = t0
    tops[chain_start + 1] = t1
    tops[chain_start + 2] = t2

    cursor = max(t0 + h0, t1 + h1, t2 + h2) + random.randint(1, 2)
    for idx in range(chain_start + 3, count):
        tops[idx] = cursor
        cursor += heights[idx] + random.randint(1, 2)

    grid_h = cursor + random.randint(1, 2)
    return tops, grid_h


def _render_ranked_objects(shapes: list[np.ndarray], tops: list[int], lefts: list[int]) -> list[dict]:
    objects = []
    for idx, (shape, top, left) in enumerate(zip(shapes, tops, lefts)):
        objects.append({"id": idx, "color": FOREGROUND, "coords": _coords_from_mask(shape, top, left)})
    return objects


def _ordered_from_grid(grid: np.ndarray) -> list[dict] | None:
    bg = _get_bg(grid)
    objects = _extract_objects(grid, bg)
    top_rows = [obj["bbox"][0] for obj in objects]
    if len(set(top_rows)) != len(top_rows):
        return None
    ordered = []
    for obj_id, obj in enumerate(sorted(objects, key=lambda item: item["bbox"][0])):
        ordered.append(
            {
                "id": obj_id,
                "color": obj["color"],
                "coords": obj["coords"],
                "bbox": obj["bbox"],
            }
        )
    return ordered


def _pair_gap_lookup(infos: list[dict]) -> dict[tuple[int, int], int]:
    return {tuple(info["pair"]): int(info["gap"]) for info in infos}


def _semantic_checks(inp: np.ndarray, target_pair: tuple[int, int], target_final_gap: int) -> bool:
    ordered = _ordered_from_grid(inp)
    if ordered is None or not (4 <= len(ordered) <= 6):
        return False

    pre_infos = _analyze_pairs(ordered)
    if not pre_infos:
        return False
    pre_lookup = _pair_gap_lookup(pre_infos)
    if target_pair not in pre_lookup:
        return False
    pre_min = min(pre_lookup.values())
    if pre_lookup[target_pair] != pre_min:
        return False
    if sum(gap == pre_min for gap in pre_lookup.values()) < 2:
        return False

    moved = _shifted_objects(ordered, inp.shape)
    moved_grid = _render_objects(moved, inp.shape, BACKGROUND)
    post_infos = _analyze_pairs(moved)
    if not post_infos:
        return False
    post_lookup = _pair_gap_lookup(post_infos)
    if target_pair not in post_lookup:
        return False
    post_min = min(post_lookup.values())
    if post_lookup[target_pair] != post_min:
        return False
    if sum(gap == post_min for gap in post_lookup.values()) != 1:
        return False
    if post_lookup[target_pair] != target_final_gap:
        return False

    out = main(inp)
    if np.count_nonzero(out == FOREGROUND) <= np.count_nonzero(moved_grid == FOREGROUND):
        return False
    if not np.array_equal(out, _solve_from_objects(ordered, inp.shape, BACKGROUND)):
        return False
    return True


def _place_isolated_lefts(shapes: list[np.ndarray], chain_start: int, grid_w: int) -> list[int] | None:
    lefts = [0] * len(shapes)
    for idx, shape in enumerate(shapes):
        if chain_start <= idx <= chain_start + 2:
            continue
        lo, hi = _final_left_bounds(idx + 1, int(shape.shape[1]), grid_w)
        if lo > hi:
            return None
        side_bias = random.choice(("left", "right"))
        if side_bias == "left":
            hi = min(hi, max(lo, grid_w // 2 - 2))
        else:
            lo = max(lo, min(hi, grid_w // 2))
        lefts[idx] = random.randint(lo, hi)
    return lefts


def _search_chain_lefts(
    shapes: list[np.ndarray], tops: list[int], lefts: list[int], chain_start: int, grid_w: int
) -> list[int] | None:
    chain = [chain_start, chain_start + 1, chain_start + 2]
    ranges = []
    for idx in chain:
        lo, hi = _final_left_bounds(idx + 1, int(shapes[idx].shape[1]), grid_w)
        if lo > hi:
            return None
        vals = list(range(lo, hi + 1))
        random.shuffle(vals)
        ranges.append(vals)

    target_pair = (chain_start + 1, chain_start + 2)
    decoy_pair = (chain_start, chain_start + 1)

    for left_a in ranges[0]:
        for left_b in ranges[1]:
            if left_b <= left_a:
                continue
            for left_c in ranges[2]:
                if left_c <= left_b:
                    continue
                candidate_lefts = list(lefts)
                candidate_lefts[chain[0]] = left_a
                candidate_lefts[chain[1]] = left_b
                candidate_lefts[chain[2]] = left_c
                final_objects = _render_ranked_objects(shapes, tops, candidate_lefts)
                final_grid = _render_objects(final_objects, (max(tops) + 6, grid_w), BACKGROUND)
                if len(_extract_objects(final_grid, BACKGROUND)) != len(shapes):
                    continue

                try:
                    post_infos = _pair_gap_lookup(_analyze_pairs(final_objects))
                except AssertionError:
                    continue
                if target_pair not in post_infos or decoy_pair not in post_infos:
                    continue
                target_gap = post_infos[target_pair]
                decoy_gap = post_infos[decoy_pair]
                if target_gap not in TARGET_FINAL_GAPS or decoy_gap != target_gap + 4:
                    continue
                if sum(gap == target_gap for gap in post_infos.values()) != 1:
                    continue
                return candidate_lefts
    return None


def _generate_case() -> np.ndarray | None:
    count = random.randint(4, 6)
    chain_start = 0
    if count >= 5 and random.random() < 0.5:
        chain_start = 2

    shapes = _sample_shapes(count)
    try:
        tops, grid_h = _sample_top_rows(shapes, chain_start)
    except ValueError:
        return None
    if grid_h > 24:
        return None

    min_width = sum(int(shapes[idx].shape[1]) for idx in range(chain_start, chain_start + 3)) + 8
    grid_w = random.randint(max(15, min_width), min(20, max(15, min_width + 4)))
    lefts = _place_isolated_lefts(shapes, chain_start, grid_w)
    if lefts is None:
        return None
    lefts = _search_chain_lefts(shapes, tops, lefts, chain_start, grid_w)
    if lefts is None:
        return None

    final_objects = _render_ranked_objects(shapes, tops, lefts)
    final_grid = _render_objects(final_objects, (grid_h, grid_w), BACKGROUND)
    if len(_extract_objects(final_grid, BACKGROUND)) != count:
        return None

    input_objects = []
    for rank, obj in enumerate(final_objects, start=1):
        dc = 1 if rank % 2 == 1 else -1
        moved = _shift_coords(obj["coords"], dc, final_grid.shape)
        if len(moved) != len(obj["coords"]):
            return None
        input_objects.append({"id": obj["id"], "color": obj["color"], "coords": moved})
    inp = _render_objects(input_objects, final_grid.shape, BACKGROUND)
    if len(_extract_objects(inp, BACKGROUND)) != count:
        return None

    target_pair = (chain_start + 1, chain_start + 2)
    target_gap = _pair_gap_lookup(_analyze_pairs(final_objects))[target_pair]
    if not _semantic_checks(inp, target_pair, target_gap):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_case()
        if grid is not None:
            return grid.astype(int)
    raise RuntimeError("failed to generate a valid input after repeated attempts")
