from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
GREY = 5
SHAPE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_RETRIES = 300
PATCH_RETRIES = 80
PLACEMENT_GAP = 1


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    return int(mins[0]), int(mins[1]), int(maxs[0]), int(maxs[1])


def _normalize(coords: np.ndarray) -> frozenset[tuple[int, int]]:
    mins = coords.min(axis=0)
    rel = coords - mins
    return frozenset((int(r), int(c)) for r, c in rel)


def _find_mask_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(cells, dtype=int))
    return comps


def _find_shape_components(grid: np.ndarray) -> tuple[list[dict], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    label_map = np.full((h, w), -1, dtype=int)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color in {BG, GREY} or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((int(rr), int(cc)))
                label_map[rr, cc] = len(comps)
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            coords = np.asarray(cells, dtype=int)
            comps.append(
                {
                    "id": len(comps),
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox_from_coords(coords),
                    "area": int(len(cells)),
                }
            )
    return comps, label_map


def _line_orientation(coords: np.ndarray) -> str:
    rows = np.unique(coords[:, 0])
    cols = np.unique(coords[:, 1])
    if len(rows) == 1 and len(cols) >= 2:
        return "h"
    if len(cols) == 1 and len(rows) >= 2:
        return "v"
    raise AssertionError("grey connector must be a straight nontrivial segment")


def _pair_side(smaller: dict, larger: dict, orient: str) -> str:
    sr0, sc0, sr1, sc1 = smaller["bbox"]
    lr0, lc0, lr1, lc1 = larger["bbox"]
    if orient == "h":
        if sc1 < lc0:
            return "left"
        if sc0 > lc1:
            return "right"
        raise AssertionError("horizontal connector requires left/right-separated shapes")
    if sr1 < lr0:
        return "above"
    if sr0 > lr1:
        return "below"
    raise AssertionError("vertical connector requires above/below-separated shapes")


def _fold_large_component(larger: dict, side: str) -> tuple[np.ndarray, np.ndarray]:
    coords = np.asarray(larger["coords"], dtype=int)
    r0, c0, r1, c1 = larger["bbox"]
    local = coords - np.array([r0, c0], dtype=int)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)

    if side in {"left", "right"}:
        assert w % 2 == 0, "vertical fold needs an even-width large shape"
        half = w // 2
        if side == "left":
            keep = local[:, 1] < half
            fold = local[:, 1] >= half
        else:
            keep = local[:, 1] >= half
            fold = local[:, 1] < half

        near = coords[keep]
        far = local[fold]
        folded = np.column_stack(
            [
                far[:, 0] + r0,
                (w - 1 - far[:, 1]) + c0,
            ]
        )
        return near.astype(int), folded.astype(int)

    assert h % 2 == 0, "horizontal fold needs an even-height large shape"
    half = h // 2
    if side == "above":
        keep = local[:, 0] < half
        fold = local[:, 0] >= half
    else:
        keep = local[:, 0] >= half
        fold = local[:, 0] < half

    near = coords[keep]
    far = local[fold]
    folded = np.column_stack(
        [
            (h - 1 - far[:, 0]) + r0,
            far[:, 1] + c0,
        ]
    )
    return near.astype(int), folded.astype(int)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fold each grey-linked larger shape toward its smaller partner and recolor the folded overlap with the smaller shape's color.

    Input structure:
    - The background is black.
    - Grey straight line segments each connect exactly two monochrome 4-connected shapes.
    - In every linked pair, one shape is larger, and the smaller shape is congruent to the larger shape's far half after folding across the axis perpendicular to the grey segment.

    Rule:
    - Use each grey segment to identify one pair of colored shapes.
    - A horizontal grey segment implies a vertical fold axis, while a vertical grey segment implies a horizontal fold axis.
    - For each pair, keep the half of the larger shape nearest the smaller one fixed, reflect the opposite half across the larger shape's center line, and place the smaller shape onto that reflected footprint.
    - Paint the reflected footprint in the smaller shape's color, keep any uncovered near-side pixels in the larger shape's color, and remove the grey connector.

    Key insight:
    - The grey connector carries two pieces of information at once: which shapes belong together and which perpendicular axis the larger shape must fold across.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    shape_comps, label_map = _find_shape_components(g)
    grey_comps = _find_mask_components(g == GREY)
    assert grey_comps, "expected at least one grey connector"

    by_id = {int(comp["id"]): comp for comp in shape_comps}
    used_ids: set[int] = set()
    out = np.full_like(g, BG)

    for grey in grey_comps:
        orient = _line_orientation(grey)
        adjacent: set[int] = set()
        for r, c in grey:
            for nr, nc in _neighbors4(int(r), int(c), g.shape[0], g.shape[1]):
                comp_id = int(label_map[nr, nc])
                if comp_id >= 0:
                    adjacent.add(comp_id)
        assert len(adjacent) == 2, "each grey connector must touch exactly two shapes"

        first_id, second_id = sorted(adjacent)
        assert first_id not in used_ids and second_id not in used_ids, "shape pairs must be disjoint"
        used_ids.add(first_id)
        used_ids.add(second_id)

        a = by_id[first_id]
        b = by_id[second_id]
        assert int(a["area"]) != int(b["area"]), "pair must contain one larger and one smaller shape"
        smaller, larger = (a, b) if int(a["area"]) < int(b["area"]) else (b, a)

        side = _pair_side(smaller, larger, orient)
        near_coords, folded_far_coords = _fold_large_component(larger, side)
        assert _normalize(np.asarray(smaller["coords"], dtype=int)) == _normalize(folded_far_coords)

        pair_grid = np.full_like(g, BG)
        pair_grid[near_coords[:, 0], near_coords[:, 1]] = int(larger["color"])
        pair_grid[folded_far_coords[:, 0], folded_far_coords[:, 1]] = int(smaller["color"])
        overlap = (out != BG) & (pair_grid != BG)
        assert not bool(overlap.any()), "different pairs must not overlap after folding"
        out[pair_grid != BG] = pair_grid[pair_grid != BG]

    assert len(used_ids) == len(shape_comps), "every shape must belong to exactly one grey-linked pair"
    return out


def _path_segment(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = start
    tr, tc = end
    out = [(int(r), int(c))]
    while (r, c) != (tr, tc):
        moves: list[tuple[int, int]] = []
        if r != tr:
            moves.append((1 if tr > r else -1, 0))
        if c != tc:
            moves.append((0, 1 if tc > c else -1))
        dr, dc = random.choice(moves)
        r += dr
        c += dc
        out.append((int(r), int(c)))
    return out


def _connecting_path(a: tuple[int, int], b: tuple[int, int], h: int, w: int) -> list[tuple[int, int]]:
    waypoints = [a]
    if a[0] == b[0] and h > 1 and random.random() < 0.6:
        pivot_r = random.randrange(h)
        if pivot_r != a[0]:
            waypoints.extend([(pivot_r, a[1]), (pivot_r, b[1])])
    elif a[1] == b[1] and w > 1 and random.random() < 0.6:
        pivot_c = random.randrange(w)
        if pivot_c != a[1]:
            waypoints.extend([(a[0], pivot_c), (b[0], pivot_c)])
    waypoints.append(b)

    path: list[tuple[int, int]] = []
    for start, end in zip(waypoints, waypoints[1:]):
        segment = _path_segment(start, end)
        if path:
            segment = segment[1:]
        path.extend(segment)
    return path


def _grow_connected_mask(
    h: int,
    w: int,
    mandatory: list[tuple[int, int]],
    target_area: int,
) -> np.ndarray:
    cells = set()
    for idx, point in enumerate(mandatory):
        if idx == 0:
            cells.add((int(point[0]), int(point[1])))
            continue
        for step in _connecting_path(mandatory[idx - 1], point, h, w):
            cells.add((int(step[0]), int(step[1])))

    target_area = max(int(target_area), len(cells))

    frontier = set()
    for r, c in cells:
        for nr, nc in _neighbors4(r, c, h, w):
            if (nr, nc) not in cells:
                frontier.add((nr, nc))

    while len(cells) < target_area and frontier:
        nr, nc = random.choice(tuple(frontier))
        frontier.remove((nr, nc))
        cells.add((int(nr), int(nc)))
        for rr, cc in _neighbors4(nr, nc, h, w):
            if (rr, cc) not in cells:
                frontier.add((rr, cc))

    assert len(cells) == target_area
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    comps = _find_mask_components(mask)
    assert len(comps) == 1
    return mask


def _trim_to_input(inp_patch: np.ndarray, out_patch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    occupied = inp_patch != BG
    coords = np.argwhere(occupied)
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    return inp_patch[r0 : r1 + 1, c0 : c1 + 1].copy(), out_patch[r0 : r1 + 1, c0 : c1 + 1].copy()


def _build_pair_patch(large_color: int, small_color: int) -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(PATCH_RETRIES):
        h = random.randint(4, 7)
        half_w = random.randint(3, 5)
        anchor_r = random.randrange(h)
        far_outer_r = random.randrange(h)
        near_mandatory = [(anchor_r, 0), (anchor_r, half_w - 1)]
        far_mandatory = [(anchor_r, 0), (far_outer_r, half_w - 1)]

        near_min = len(set(_connecting_path(near_mandatory[0], near_mandatory[1], h, half_w)))
        far_min = len(set(_connecting_path(far_mandatory[0], far_mandatory[1], h, half_w)))
        near_area = random.randint(near_min, h * half_w)
        far_area = random.randint(far_min, h * half_w)

        near = _grow_connected_mask(h, half_w, near_mandatory, near_area)
        far = _grow_connected_mask(h, half_w, far_mandatory, far_area)
        small = np.fliplr(far)
        if not bool(small[anchor_r, -1]):
            continue

        gap = random.randint(2, 4)
        width = half_w + gap + 2 * half_w
        inp = np.full((h, width), BG, dtype=int)
        out = np.full((h, width), BG, dtype=int)

        inp[:, :half_w][small] = small_color
        inp[anchor_r, half_w : half_w + gap] = GREY
        large_start = half_w + gap
        inp[:, large_start : large_start + half_w][near] = large_color
        inp[:, large_start + half_w : large_start + 2 * half_w][far] = large_color

        overlay = np.full((h, half_w), BG, dtype=int)
        overlay[near] = large_color
        overlay[small] = small_color
        out[:, large_start : large_start + half_w] = overlay

        inp, out = _trim_to_input(inp, out)
        rot_k = random.randrange(4)
        inp = np.rot90(inp, rot_k).copy()
        out = np.rot90(out, rot_k).copy()
        inp, out = _trim_to_input(inp, out)

        if max(inp.shape) > 30 or max(out.shape) > 30:
            continue
        if np.array_equal(inp, out):
            continue
        try:
            solved = main(inp)
        except AssertionError:
            continue
        if not np.array_equal(solved, out):
            continue
        return inp, out
    return None


def _place_patches(
    patches: list[tuple[np.ndarray, np.ndarray]],
    h: int,
    w: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    reserved = np.zeros((h, w), dtype=bool)
    placements: list[tuple[int, int] | None] = [None] * len(patches)
    order = sorted(range(len(patches)), key=lambda i: patches[i][0].shape[0] * patches[i][0].shape[1], reverse=True)
    position_cache: dict[int, list[tuple[int, int]]] = {}

    for idx in order:
        ph, pw = patches[idx][0].shape
        positions = [(r, c) for r in range(h - ph + 1) for c in range(w - pw + 1)]
        random.shuffle(positions)
        position_cache[idx] = positions

    def backtrack(pos: int) -> bool:
        if pos == len(order):
            return True
        idx = order[pos]
        ph, pw = patches[idx][0].shape
        for r, c in position_cache[idx]:
            r0 = max(0, r - PLACEMENT_GAP)
            c0 = max(0, c - PLACEMENT_GAP)
            r1 = min(h, r + ph + PLACEMENT_GAP)
            c1 = min(w, c + pw + PLACEMENT_GAP)
            if bool(reserved[r0:r1, c0:c1].any()):
                continue
            reserved[r0:r1, c0:c1] = True
            placements[idx] = (r, c)
            if backtrack(pos + 1):
                return True
            reserved[r0:r1, c0:c1] = False
            placements[idx] = None
        return False

    if not backtrack(0):
        return None

    inp = np.full((h, w), BG, dtype=int)
    out = np.full((h, w), BG, dtype=int)
    for idx, placement in enumerate(placements):
        assert placement is not None
        r, c = placement
        inp_patch, out_patch = patches[idx]
        ph, pw = inp_patch.shape
        in_mask = inp_patch != BG
        out_mask = out_patch != BG
        assert not bool((inp[r : r + ph, c : c + pw] != BG)[in_mask].any())
        assert not bool((out[r : r + ph, c : c + pw] != BG)[out_mask].any())
        inp[r : r + ph, c : c + pw][in_mask] = inp_patch[in_mask]
        out[r : r + ph, c : c + pw][out_mask] = out_patch[out_mask]
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        n_pairs = random.randint(1, 3)
        colors = random.sample(SHAPE_COLORS, 2 * n_pairs)
        patches: list[tuple[np.ndarray, np.ndarray]] = []
        ok = True
        for pair_idx in range(n_pairs):
            patch = _build_pair_patch(colors[2 * pair_idx], colors[2 * pair_idx + 1])
            if patch is None:
                ok = False
                break
            patches.append(patch)
        if not ok:
            continue

        total_reserved = sum((inp.shape[0] + 2 * PLACEMENT_GAP) * (inp.shape[1] + 2 * PLACEMENT_GAP) for inp, _out in patches)
        lower = max(
            12,
            max(inp.shape[0] for inp, _out in patches) + 2,
            max(inp.shape[1] for inp, _out in patches) + 2,
            int(np.ceil(total_reserved ** 0.5)),
        )
        if lower > 30:
            continue

        h = random.randint(lower, 30)
        w = random.randint(lower, 30)
        placed = _place_patches(patches, h, w)
        if placed is None:
            continue
        inp, expected = placed
        try:
            solved = main(inp)
        except AssertionError:
            continue
        if not np.array_equal(solved, expected):
            continue
        return inp
    raise RuntimeError("failed to generate a valid ARC sample")
