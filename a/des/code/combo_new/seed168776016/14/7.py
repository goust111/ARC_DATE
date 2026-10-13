from __future__ import annotations

import math
import random
from collections import deque

import numpy as np

BG = 0
BRIDGE = 4
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
NON_SPECIAL_COLORS = (1, 2, 3, 5, 6, 7, 8, 9)

PAIR_RETRIES = 200
SINGLE_RETRIES = 120
PLACEMENT_RETRIES = 300
GEN_RETRIES = 400


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in DIR4:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _mask_components(mask: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=int)
    components: list[np.ndarray] = []
    comp_id = 0
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or labels[r, c] != 0:
                continue
            comp_id += 1
            q = deque([(r, c)])
            labels[r, c] = comp_id
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if bool(mask[nr, nc]) and labels[nr, nc] == 0:
                        labels[nr, nc] = comp_id
                        q.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components, labels


def _color_components(grid: np.ndarray) -> tuple[dict[int, dict[str, object]], np.ndarray]:
    mask = (grid != BG) & (grid != BRIDGE)
    comp_coords, labels = _mask_components(mask)
    components: dict[int, dict[str, object]] = {}
    for comp_id, coords in enumerate(comp_coords, start=1):
        color = int(grid[int(coords[0, 0]), int(coords[0, 1])])
        components[comp_id] = {
            "coords": coords,
            "color": color,
            "area": int(coords.shape[0]),
        }
    return components, labels


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _trim_patch(patch: np.ndarray) -> np.ndarray:
    coords = np.argwhere(patch != BG)
    r0, c0, r1, c1 = _bbox(coords)
    return patch[r0 : r1 + 1, c0 : c1 + 1].copy()


def _corner_occupancy(patch: np.ndarray) -> int:
    h, w = patch.shape
    corners = ((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1))
    return sum(int(patch[r, c] != BG) for r, c in corners)


def _analyze_bridges(grid: np.ndarray) -> tuple[dict[int, dict[str, object]], list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    components, labels = _color_components(g)
    bridge_pairs: list[dict[str, object]] = []
    used_components: set[int] = set()
    h, w = g.shape
    for r, c in np.argwhere(g == BRIDGE):
        rr = int(r)
        cc = int(c)
        adj = sorted({int(labels[nr, nc]) for nr, nc in _neighbors4(rr, cc, h, w) if labels[nr, nc] != 0})
        assert len(adj) == 2, "each yellow bridge must touch exactly two colored components"
        a_id, b_id = adj
        assert a_id not in used_components and b_id not in used_components, "each island may participate in at most one bridge"
        a = components[a_id]
        b = components[b_id]
        assert int(a["color"]) != int(b["color"]), "a bridge must connect different-colored islands"
        used_components.add(a_id)
        used_components.add(b_id)
        bridge_pairs.append({"bridge": (rr, cc), "a": a_id, "b": b_id})
    return components, bridge_pairs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Merge every two-island pair signaled by a yellow bridge, then tag the merged bbox corners with the absorbed island color.

    Input structure:
    - The background is black.
    - Non-black, non-yellow pixels form monochrome 4-connected islands with irregular shapes and varying areas.
    - Some island pairs are linked by exactly one yellow bridge pixel that is 4-neighbor-adjacent to both islands.
    - An island participates in at most one bridge; unbridged islands are independent.

    Rule:
    - Find each yellow bridge pixel and identify the two colored islands it touches.
    - Compare the two island areas; the larger island is the main island and the smaller one is the absorbed island.
    - Recolor every absorbed-island pixel and the bridge pixel to the main-island color so the pair becomes one object.
    - Compute the tight bounding box of that merged object and place one pixel of the absorbed island's original color at each bbox corner that is still background.
    - Leave islands with no bridge unchanged.

    Key insight:
    - The yellow pixel is a structural connector: it tells both which two objects must be merged and where to add the absorbed color's corner markers after recoloring.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    components, bridge_pairs = _analyze_bridges(g)
    base = g.copy()
    marker_jobs: list[tuple[tuple[int, int, int, int], int]] = []

    for pair in bridge_pairs:
        comp_a = components[int(pair["a"])]
        comp_b = components[int(pair["b"])]
        if int(comp_a["area"]) >= int(comp_b["area"]):
            main_comp, sub_comp = comp_a, comp_b
        else:
            main_comp, sub_comp = comp_b, comp_a

        main_color = int(main_comp["color"])
        sub_color = int(sub_comp["color"])
        sub_coords = np.asarray(sub_comp["coords"], dtype=int)
        base[sub_coords[:, 0], sub_coords[:, 1]] = main_color
        br, bc = pair["bridge"]
        base[int(br), int(bc)] = main_color

        merged_coords = np.vstack(
            (
                np.asarray(main_comp["coords"], dtype=int),
                sub_coords,
                np.array([[int(br), int(bc)]], dtype=int),
            )
        )
        marker_jobs.append((_bbox(merged_coords), sub_color))

    out = base.copy()
    for (r0, c0, r1, c1), marker_color in marker_jobs:
        for rr, cc in ((r0, c0), (r0, c1), (r1, c0), (r1, c1)):
            if int(base[rr, cc]) == BG:
                out[rr, cc] = int(marker_color)

    return out


def _path_between(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    rr, cc = a
    tr, tc = b
    cells = [(rr, cc)]
    row_first = bool(random.getrandbits(1))
    if row_first:
        step = 1 if tr >= rr else -1
        while rr != tr:
            rr += step
            cells.append((rr, cc))
        step = 1 if tc >= cc else -1
        while cc != tc:
            cc += step
            cells.append((rr, cc))
    else:
        step = 1 if tc >= cc else -1
        while cc != tc:
            cc += step
            cells.append((rr, cc))
        step = 1 if tr >= rr else -1
        while rr != tr:
            rr += step
            cells.append((rr, cc))
    return cells


def _seed_shape(height: int, width: int, anchor: tuple[int, int], forced: set[tuple[int, int]]) -> set[tuple[int, int]]:
    shape = {anchor}
    for target in forced:
        for cell in _path_between(anchor, target):
            shape.add(cell)
    return shape


def _grow_shape(height: int, width: int, shape: set[tuple[int, int]], target_area: int) -> set[tuple[int, int]]:
    while len(shape) < target_area:
        frontier: set[tuple[int, int]] = set()
        for r, c in shape:
            for dr, dc in DIR4:
                rr = r + dr
                cc = c + dc
                if 0 <= rr < height and 0 <= cc < width and (rr, cc) not in shape:
                    frontier.add((rr, cc))
        if not frontier:
            break
        shape.add(random.choice(tuple(frontier)))
    return shape


def _sample_area(seed_area: int, capacity: int) -> int:
    low = max(seed_area, 3, int(math.ceil(capacity * random.uniform(0.35, 0.55))))
    high = max(low, capacity - random.randint(0, max(1, capacity // 4)))
    return random.randint(low, high)


def _single_blob_patch() -> np.ndarray:
    for _ in range(SINGLE_RETRIES):
        height = random.randint(3, 7)
        width = random.randint(3, 7)
        anchor = (random.randint(0, height - 1), random.randint(0, width - 1))
        forced: set[tuple[int, int]] = set()
        if random.random() < 0.5:
            forced.add(random.choice(((0, 0), (0, width - 1), (height - 1, 0), (height - 1, width - 1))))
        if random.random() < 0.35:
            forced.add((random.randint(0, height - 1), random.randint(0, width - 1)))
        seed = _seed_shape(height, width, anchor, forced)
        area = _sample_area(len(seed), height * width)
        shape = _grow_shape(height, width, seed, area)
        patch = np.zeros((height, width), dtype=int)
        color = random.choice(NON_SPECIAL_COLORS)
        for r, c in shape:
            patch[r, c] = color
        patch = _trim_patch(patch)
        if patch.shape[0] >= 2 and patch.shape[1] >= 2:
            return patch
    raise RuntimeError("failed to build single-island patch")


def _horizontal_pair_patch(require_skipped_corner: bool) -> np.ndarray | None:
    height = random.randint(4, 8)
    left_w = random.randint(3, 6)
    right_w = random.randint(3, 6)
    bridge_row = random.randint(0, height - 1)
    left_anchor = (bridge_row, left_w - 1)
    right_anchor = (bridge_row, 0)

    corner_labels = ("tl", "tr", "bl", "br")
    chosen = set(random.sample(corner_labels, random.randint(1, 3))) if require_skipped_corner else set()
    left_forced = set()
    right_forced = set()
    if "tl" in chosen:
        left_forced.add((0, 0))
    if "bl" in chosen:
        left_forced.add((height - 1, 0))
    if "tr" in chosen:
        right_forced.add((0, right_w - 1))
    if "br" in chosen:
        right_forced.add((height - 1, right_w - 1))
    if random.random() < 0.35:
        left_forced.add((random.randint(0, height - 1), 0))
    if random.random() < 0.35:
        right_forced.add((random.randint(0, height - 1), right_w - 1))

    left_seed = _seed_shape(height, left_w, left_anchor, left_forced)
    right_seed = _seed_shape(height, right_w, right_anchor, right_forced)
    left_area = _sample_area(len(left_seed), height * left_w)
    right_area = _sample_area(len(right_seed), height * right_w)
    if left_area == right_area:
        return None

    left_shape = _grow_shape(height, left_w, left_seed, left_area)
    right_shape = _grow_shape(height, right_w, right_seed, right_area)
    colors = random.sample(NON_SPECIAL_COLORS, 2)

    patch = np.zeros((height, left_w + 1 + right_w), dtype=int)
    for r, c in left_shape:
        patch[r, c] = colors[0]
    patch[bridge_row, left_w] = BRIDGE
    for r, c in right_shape:
        patch[r, left_w + 1 + c] = colors[1]
    patch = _trim_patch(patch)
    if min(patch.shape) < 3:
        return None
    if _corner_occupancy(patch) == 0 and require_skipped_corner:
        return None
    if _corner_occupancy(patch) != 0 and not require_skipped_corner:
        return None
    _, pairs = _analyze_bridges(patch)
    if len(pairs) != 1:
        return None
    return patch


def _vertical_pair_patch(require_skipped_corner: bool) -> np.ndarray | None:
    width = random.randint(4, 8)
    top_h = random.randint(3, 6)
    bottom_h = random.randint(3, 6)
    bridge_col = random.randint(0, width - 1)
    top_anchor = (top_h - 1, bridge_col)
    bottom_anchor = (0, bridge_col)

    corner_labels = ("tl", "tr", "bl", "br")
    chosen = set(random.sample(corner_labels, random.randint(1, 3))) if require_skipped_corner else set()
    top_forced = set()
    bottom_forced = set()
    if "tl" in chosen:
        top_forced.add((0, 0))
    if "tr" in chosen:
        top_forced.add((0, width - 1))
    if "bl" in chosen:
        bottom_forced.add((bottom_h - 1, 0))
    if "br" in chosen:
        bottom_forced.add((bottom_h - 1, width - 1))
    if random.random() < 0.35:
        top_forced.add((0, random.randint(0, width - 1)))
    if random.random() < 0.35:
        bottom_forced.add((bottom_h - 1, random.randint(0, width - 1)))

    top_seed = _seed_shape(top_h, width, top_anchor, top_forced)
    bottom_seed = _seed_shape(bottom_h, width, bottom_anchor, bottom_forced)
    top_area = _sample_area(len(top_seed), top_h * width)
    bottom_area = _sample_area(len(bottom_seed), bottom_h * width)
    if top_area == bottom_area:
        return None

    top_shape = _grow_shape(top_h, width, top_seed, top_area)
    bottom_shape = _grow_shape(bottom_h, width, bottom_seed, bottom_area)
    colors = random.sample(NON_SPECIAL_COLORS, 2)

    patch = np.zeros((top_h + 1 + bottom_h, width), dtype=int)
    for r, c in top_shape:
        patch[r, c] = colors[0]
    patch[top_h, bridge_col] = BRIDGE
    for r, c in bottom_shape:
        patch[top_h + 1 + r, c] = colors[1]
    patch = _trim_patch(patch)
    if min(patch.shape) < 3:
        return None
    if _corner_occupancy(patch) == 0 and require_skipped_corner:
        return None
    if _corner_occupancy(patch) != 0 and not require_skipped_corner:
        return None
    _, pairs = _analyze_bridges(patch)
    if len(pairs) != 1:
        return None
    return patch


def _pair_patch(require_skipped_corner: bool) -> np.ndarray:
    for _ in range(PAIR_RETRIES):
        if random.random() < 0.5:
            patch = _horizontal_pair_patch(require_skipped_corner)
        else:
            patch = _vertical_pair_patch(require_skipped_corner)
        if patch is not None:
            return patch
    raise RuntimeError("failed to build bridge pair patch")


def _place_patch(grid: np.ndarray, reserved: np.ndarray, patch: np.ndarray) -> bool:
    h, w = grid.shape
    ph, pw = patch.shape
    row_min = 1
    col_min = 1
    row_max = h - ph - 1
    col_max = w - pw - 1
    if row_max < row_min or col_max < col_min:
        return False
    for _ in range(PLACEMENT_RETRIES):
        r0 = random.randint(row_min, row_max)
        c0 = random.randint(col_min, col_max)
        if reserved[r0 - 1 : r0 + ph + 1, c0 - 1 : c0 + pw + 1].any():
            continue
        mask = patch != BG
        sub = grid[r0 : r0 + ph, c0 : c0 + pw]
        if np.any(mask & (sub != BG)):
            continue
        sub[mask] = patch[mask]
        reserved[r0 - 1 : r0 + ph + 1, c0 - 1 : c0 + pw + 1] = True
        return True
    return False


def _valid_generated_instance(grid: np.ndarray) -> bool:
    components, pairs = _analyze_bridges(grid)
    if not pairs:
        return False
    if max(grid.shape) > 30:
        return False
    bridged = {int(p["a"]) for p in pairs} | {int(p["b"]) for p in pairs}
    if not bridged:
        return False
    for pair in pairs:
        a = components[int(pair["a"])]
        b = components[int(pair["b"])]
        if int(a["area"]) == int(b["area"]):
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.randint(1, 3)
    single_count = random.randint(0, 3)
    require_skipped_count = random.randint(0, pair_count)
    skip_flags = [True] * require_skipped_count + [False] * (pair_count - require_skipped_count)
    random.shuffle(skip_flags)

    patches = [_pair_patch(flag) for flag in skip_flags]
    for _ in range(single_count):
        patches.append(_single_blob_patch())

    patches.sort(key=lambda p: p.shape[0] * p.shape[1], reverse=True)
    padded_area = sum((p.shape[0] + 2) * (p.shape[1] + 2) for p in patches)
    side_floor = max(max(p.shape[0] for p in patches), max(p.shape[1] for p in patches), int(math.ceil(math.sqrt(padded_area))) + 1)
    if side_floor > 30:
        return None
    h = random.randint(max(12, side_floor), 30)
    w = random.randint(max(12, side_floor), 30)

    grid = np.zeros((h, w), dtype=int)
    reserved = np.zeros((h, w), dtype=bool)
    for patch in patches:
        if not _place_patch(grid, reserved, patch):
            return None

    if not _valid_generated_instance(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid bridge-island puzzle after {GEN_RETRIES} attempts")
