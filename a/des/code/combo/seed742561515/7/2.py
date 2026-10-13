from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
MAROON = 9
INPUT_COLORS = (1, 2, 3, 4, 5, 6, 7, 8)
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


GEN_RETRIES = 400
WORK_H = 30
WORK_W = 30
RECT_DIMS = (
    (2, 4),
    (2, 5),
    (2, 6),
    (3, 4),
    (3, 5),
    (3, 6),
    (4, 4),
    (4, 5),
    (4, 6),
    (5, 4),
    (5, 5),
    (5, 6),
)
MAX_PARTITION_ATTEMPTS = 160
MAX_MOVE_PLACEMENT_ATTEMPTS = 200


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out = []
    for dr, dc in DIRS4:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w:
            out.append((rr, cc))
    return out


def _bbox_from_cells(cells: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _connected_cells(cells: set[tuple[int, int]]) -> bool:
    if not cells:
        return False
    start = next(iter(cells))
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (nr, nc) in cells and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == len(cells)


def _make_patch_from_rel(rel: set[tuple[int, int]], h: int, w: int, color: int) -> np.ndarray:
    patch = np.full((h, w), BG, dtype=int)
    for r, c in rel:
        patch[r, c] = int(color)
    return patch


def _cells_to_piece(
    cells: set[tuple[int, int]],
    color: int,
    *,
    full_rect_shape: tuple[int, int] | None = None,
) -> dict:
    r0, c0, r1, c1 = _bbox_from_cells(cells)
    rel = {(r - r0, c - c0) for r, c in cells}
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    piece = {
        "color": int(color),
        "cells": set(cells),
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "rel": rel,
        "h": int(h),
        "w": int(w),
        "area": int(len(cells)),
        "patch": _make_patch_from_rel(rel, h, w, color),
    }
    if full_rect_shape is not None:
        piece["rect_h"] = int(full_rect_shape[0])
        piece["rect_w"] = int(full_rect_shape[1])
    return piece


def _find_objects(grid: np.ndarray) -> list[dict]:
    h, w = map(int, grid.shape)
    seen = np.zeros((h, w), dtype=bool)
    objects: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = {(int(r), int(c))}
            while q:
                rr, cc = q.popleft()
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    cells.add((int(nr), int(nc)))
                    q.append((nr, nc))
            objects.append(_cells_to_piece(cells, color))
    objects.sort(key=lambda obj: (obj["bbox"][0], obj["bbox"][1], obj["color"]))
    return objects


def _enumerate_pair_candidates(piece_a: dict, piece_b: dict) -> list[dict]:
    area_a = int(piece_a["area"])
    area_b = int(piece_b["area"])
    total = area_a + area_b
    rel_a = set(piece_a["rel"])
    rel_b = set(piece_b["rel"])
    h_a = int(piece_a["h"])
    w_a = int(piece_a["w"])
    h_b = int(piece_b["h"])
    w_b = int(piece_b["w"])
    candidates: list[dict] = []

    for rect_h in range(max(h_a, h_b), total + 1):
        if total % rect_h != 0:
            continue
        rect_w = total // rect_h
        if rect_w < max(w_a, w_b):
            continue
        rect_cells = {(r, c) for r in range(rect_h) for c in range(rect_w)}
        for off_r in range(rect_h - h_a + 1):
            for off_c in range(rect_w - w_a + 1):
                placed_a = {(r + off_r, c + off_c) for r, c in rel_a}
                if len(placed_a) != area_a:
                    continue
                comp = rect_cells - placed_a
                if len(comp) != area_b:
                    continue
                br0, bc0, br1, bc1 = _bbox_from_cells(comp)
                if br1 - br0 + 1 != h_b or bc1 - bc0 + 1 != w_b:
                    continue
                norm_b = {(r - br0, c - bc0) for r, c in comp}
                if norm_b != rel_b:
                    continue
                candidates.append(
                    {
                        "rect_h": int(rect_h),
                        "rect_w": int(rect_w),
                        "off_a": (int(off_r), int(off_c)),
                        "off_b": (int(br0), int(bc0)),
                    }
                )
    return candidates


def _candidate_rect_key(piece_a: dict, piece_b: dict, cand: dict) -> tuple[int, int]:
    a_r0, a_c0, _a_r1, _a_c1 = piece_a["bbox"]
    b_r0, b_c0, _b_r1, _b_c1 = piece_b["bbox"]
    key_a = (int(a_r0 - cand["off_a"][0]), int(a_c0 - cand["off_a"][1]))
    key_b = (int(b_r0 - cand["off_b"][0]), int(b_c0 - cand["off_b"][1]))
    return min(key_a, key_b)


def _canonical_candidate(piece_a: dict, piece_b: dict, candidates: list[dict]) -> dict:
    assert candidates
    return min(
        candidates,
        key=lambda cand: (
            _candidate_rect_key(piece_a, piece_b, cand),
            int(cand["rect_h"]),
            int(cand["rect_w"]),
            int(cand["off_a"][0]),
            int(cand["off_a"][1]),
            int(cand["off_b"][0]),
            int(cand["off_b"][1]),
        ),
    )


def _render_pair_patch(piece_a: dict, piece_b: dict, cand: dict) -> np.ndarray:
    rect_h = int(cand["rect_h"])
    rect_w = int(cand["rect_w"])
    patch = np.full((rect_h, rect_w), BG, dtype=int)
    off_a_r, off_a_c = cand["off_a"]
    off_b_r, off_b_c = cand["off_b"]
    area_a = int(piece_a["area"])
    area_b = int(piece_b["area"])

    if area_a > area_b:
        color_a = int(piece_a["color"])
        color_b = MAROON
    elif area_b > area_a:
        color_a = MAROON
        color_b = int(piece_b["color"])
    else:
        split_row = (rect_h + 1) // 2
        color_a = None
        color_b = None

    for r, c in piece_a["rel"]:
        rr = r + off_a_r
        cc = c + off_a_c
        if area_a == area_b:
            patch[rr, cc] = int(piece_a["color"]) if rr < split_row else MAROON
        else:
            patch[rr, cc] = int(color_a)

    for r, c in piece_b["rel"]:
        rr = r + off_b_r
        cc = c + off_b_c
        if area_a == area_b:
            patch[rr, cc] = int(piece_b["color"]) if rr < split_row else MAROON
        else:
            patch[rr, cc] = int(color_b)

    return patch


def _pack_patches(patches: list[np.ndarray]) -> np.ndarray:
    assert patches
    out_h = max(int(patch.shape[0]) for patch in patches)
    out_w = sum(int(patch.shape[1]) for patch in patches) + max(0, len(patches) - 1)
    out = np.full((out_h, out_w), BG, dtype=int)
    col = 0
    for patch in patches:
        h, w = map(int, patch.shape)
        out[:h, col : col + w] = patch
        col += w + 1
    return out


def _unique_pairing(objects: list[dict]) -> tuple[list[tuple[int, int]], dict[tuple[int, int], list[dict]]]:
    n = len(objects)
    edge_map: dict[tuple[int, int], list[dict]] = {}
    for i in range(n):
        for j in range(i + 1, n):
            cands = _enumerate_pair_candidates(objects[i], objects[j])
            if cands:
                edge_map[(i, j)] = cands

    solutions: list[list[tuple[int, int]]] = []
    used = [False] * n
    current: list[tuple[int, int]] = []

    def dfs() -> None:
        if len(solutions) > 1:
            return
        try:
            i = next(idx for idx in range(n) if not used[idx])
        except StopIteration:
            solutions.append(current.copy())
            return
        used[i] = True
        for j in range(i + 1, n):
            if used[j] or (i, j) not in edge_map:
                continue
            used[j] = True
            current.append((i, j))
            dfs()
            current.pop()
            used[j] = False
        used[i] = False

    dfs()
    assert len(solutions) == 1, "expected a unique complementary pairing"
    return solutions[0], edge_map


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    objects = _find_objects(g)
    assert objects and len(objects) % 2 == 0, "expected an even number of foreground objects"

    pairing, edge_map = _unique_pairing(objects)
    rendered: list[tuple[tuple[int, int], np.ndarray]] = []
    for i, j in pairing:
        cand = _canonical_candidate(objects[i], objects[j], edge_map[(i, j)])
        rendered.append((_candidate_rect_key(objects[i], objects[j], cand), _render_pair_patch(objects[i], objects[j], cand)))

    rendered.sort(key=lambda item: (item[0][0], item[0][1], item[1].shape[0], item[1].shape[1]))
    return _pack_patches([patch for _key, patch in rendered])


def _random_connected_subset(h: int, w: int, target: int) -> set[tuple[int, int]]:
    start = (random.randrange(h), random.randrange(w))
    cells = {start}
    frontier = set(_neighbors4(start[0], start[1], h, w))
    while len(cells) < target:
        pick = random.choice(tuple(frontier))
        frontier.remove(pick)
        cells.add(pick)
        for nbr in _neighbors4(pick[0], pick[1], h, w):
            if nbr not in cells:
                frontier.add(nbr)
    return cells


def _partition_rectangle(h: int, w: int, equal_area: bool) -> tuple[dict, dict, np.ndarray] | None:
    total = h * w
    all_cells = {(r, c) for r in range(h) for c in range(w)}
    if equal_area:
        if total % 2 != 0:
            return None
        area_choices = [total // 2]
    else:
        area_choices = [area for area in range(2, total - 1) if area * 2 != total]
        if not area_choices:
            return None

    for _ in range(MAX_PARTITION_ATTEMPTS):
        target = random.choice(area_choices)
        cells_a = _random_connected_subset(h, w, target)
        cells_b = all_cells - cells_a
        if len(cells_b) < 2 or not _connected_cells(cells_b):
            continue
        piece_a = _cells_to_piece(cells_a, 1, full_rect_shape=(h, w))
        piece_b = _cells_to_piece(cells_b, 2, full_rect_shape=(h, w))
        if len(_enumerate_pair_candidates(piece_a, piece_b)) != 1:
            continue
        mask_a = np.zeros((h, w), dtype=bool)
        for r, c in cells_a:
            mask_a[r, c] = True
        return piece_a, piece_b, mask_a
    return None


def _paint_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = map(int, patch.shape)
    region = grid[top : top + h, left : left + w]
    mask = patch != BG
    region[mask] = patch[mask]


def _mark_blocked(blocked: np.ndarray, top: int, left: int, h: int, w: int, pad: int = 1) -> None:
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(blocked.shape[0], top + h + pad)
    c1 = min(blocked.shape[1], left + w + pad)
    blocked[r0:r1, c0:c1] = True


def _can_place_bbox(blocked: np.ndarray, top: int, left: int, h: int, w: int, pad: int = 1) -> bool:
    if top < 0 or left < 0 or top + h > blocked.shape[0] or left + w > blocked.shape[1]:
        return False
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(blocked.shape[0], top + h + pad)
    c1 = min(blocked.shape[1], left + w + pad)
    return not bool(blocked[r0:r1, c0:c1].any())


def _build_expected_patch(
    rect_h: int,
    rect_w: int,
    piece_a: dict,
    piece_b: dict,
    color_a: int,
    color_b: int,
) -> np.ndarray:
    patch = np.full((rect_h, rect_w), BG, dtype=int)
    area_a = int(piece_a["area"])
    area_b = int(piece_b["area"])
    if area_a == area_b:
        split_row = (rect_h + 1) // 2
        for r, c in piece_a["cells"]:
            patch[r, c] = int(color_a) if r < split_row else MAROON
        for r, c in piece_b["cells"]:
            patch[r, c] = int(color_b) if r < split_row else MAROON
        return patch

    keep_a = area_a > area_b
    for r, c in piece_a["cells"]:
        patch[r, c] = int(color_a) if keep_a else MAROON
    for r, c in piece_b["cells"]:
        patch[r, c] = MAROON if keep_a else int(color_b)
    return patch


def _crop_with_random_margin(grid: np.ndarray) -> np.ndarray:
    coords = np.argwhere(grid != BG)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    top_margin = random.randint(0, min(2, int(r0)))
    left_margin = random.randint(0, min(2, int(c0)))
    bottom_margin = random.randint(0, min(2, int(grid.shape[0] - 1 - r1)))
    right_margin = random.randint(0, min(2, int(grid.shape[1] - 1 - c1)))
    return grid[
        int(r0 - top_margin) : int(r1 + bottom_margin) + 1,
        int(c0 - left_margin) : int(c1 + right_margin) + 1,
    ]


def _choose_rect_dims(equal_flags: list[bool]) -> list[tuple[int, int]]:
    available = list(RECT_DIMS)
    chosen: list[tuple[int, int]] = []
    for equal_area in equal_flags:
        options = [dim for dim in available if (dim[0] * dim[1]) % 2 == 0 or not equal_area]
        if not options:
            raise ValueError("not enough rectangle dimensions")
        dim = random.choice(options)
        available.remove(dim)
        chosen.append(dim)
    return chosen


def _layout_rectangles(rect_dims: list[tuple[int, int]]) -> dict[int, tuple[int, int]] | None:
    indices = list(range(len(rect_dims)))
    random.shuffle(indices)
    row_count = 1 if len(indices) == 2 and random.random() < 0.5 else 2
    if row_count == 1:
        groups = [indices]
    else:
        split = random.randint(1, len(indices) - 1)
        groups = [indices[:split], indices[split:]]

    placements: dict[int, tuple[int, int]] = {}
    row_top = random.randint(1, 3)
    max_right = 0
    for group in groups:
        row_height = max(rect_dims[idx][0] for idx in group)
        col = random.randint(1, 3)
        for idx in group:
            rect_h, rect_w = rect_dims[idx]
            placements[idx] = (row_top, col)
            col += rect_w + random.randint(2, 4)
        max_right = max(max_right, col - 1)
        row_top += row_height + random.randint(2, 4)

    max_bottom = max(placements[idx][0] + rect_dims[idx][0] - 1 for idx in placements)
    if max_right >= WORK_W - 2 or max_bottom >= 12:
        return None
    return placements


def _generate_attempt() -> np.ndarray | None:
    pair_count = random.randint(2, 4)
    equal_flags = [bool(random.getrandbits(1)) for _ in range(pair_count)]
    rect_dims = _choose_rect_dims(equal_flags)
    anchor_rects = _layout_rectangles(rect_dims)
    if anchor_rects is None:
        return None

    colors = random.sample(INPUT_COLORS, 2 * pair_count)
    grid = np.full((WORK_H, WORK_W), BG, dtype=int)
    blocked = np.zeros((WORK_H, WORK_W), dtype=bool)
    specs = []

    max_anchor_bottom = 0
    for idx, ((rect_h, rect_w), equal_area) in enumerate(zip(rect_dims, equal_flags)):
        partition = _partition_rectangle(rect_h, rect_w, equal_area)
        if partition is None:
            return None
        piece_a, piece_b, _mask_a = partition
        color_a = colors[2 * idx]
        color_b = colors[2 * idx + 1]
        piece_a["color"] = int(color_a)
        piece_a["patch"] = _make_patch_from_rel(piece_a["rel"], piece_a["h"], piece_a["w"], color_a)
        piece_b["color"] = int(color_b)
        piece_b["patch"] = _make_patch_from_rel(piece_b["rel"], piece_b["h"], piece_b["w"], color_b)
        top, left = anchor_rects[idx]
        _mark_blocked(blocked, top, left, rect_h, rect_w, pad=1)
        max_anchor_bottom = max(max_anchor_bottom, top + rect_h - 1)
        specs.append(
            {
                "rect_h": int(rect_h),
                "rect_w": int(rect_w),
                "piece_a": piece_a,
                "piece_b": piece_b,
                "color_a": int(color_a),
                "color_b": int(color_b),
                "anchor_top_left": (int(top), int(left)),
                "expected_patch": _build_expected_patch(rect_h, rect_w, piece_a, piece_b, color_a, color_b),
            }
        )

    lower_zone_top = max_anchor_bottom + random.randint(4, 6)
    if lower_zone_top >= WORK_H - 4:
        return None

    for spec in specs:
        piece_a = spec["piece_a"]
        piece_b = spec["piece_b"]
        anchor_top, anchor_left = spec["anchor_top_left"]
        anchor_is_a = bool(random.getrandbits(1))
        anchor_piece = piece_a if anchor_is_a else piece_b
        moved_piece = piece_b if anchor_is_a else piece_a

        anchor_bbox = anchor_piece["bbox"]
        anchor_patch_top = anchor_top + anchor_bbox[0]
        anchor_patch_left = anchor_left + anchor_bbox[1]
        _paint_patch(grid, anchor_piece["patch"], anchor_patch_top, anchor_patch_left)

        moved_bbox = moved_piece["bbox"]
        target_key = (anchor_top, anchor_left)
        placed = False
        for _ in range(MAX_MOVE_PLACEMENT_ATTEMPTS):
            move_top = random.randint(lower_zone_top, WORK_H - moved_piece["h"] - 1)
            move_left = random.randint(1, WORK_W - moved_piece["w"] - 1)
            if not _can_place_bbox(blocked, move_top, move_left, moved_piece["h"], moved_piece["w"], pad=1):
                continue
            alt_key = (move_top - moved_bbox[0], move_left - moved_bbox[1])
            if alt_key <= target_key:
                continue
            _paint_patch(grid, moved_piece["patch"], move_top, move_left)
            _mark_blocked(blocked, move_top, move_left, moved_piece["h"], moved_piece["w"], pad=1)
            placed = True
            break
        if not placed:
            return None

    cropped = _crop_with_random_margin(grid)
    expected = _pack_patches(
        [
            spec["expected_patch"]
            for spec in sorted(specs, key=lambda spec: (spec["anchor_top_left"][0], spec["anchor_top_left"][1]))
        ]
    )
    try:
        actual = main(cropped)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return cropped


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
