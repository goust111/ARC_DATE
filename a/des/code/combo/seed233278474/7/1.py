from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9
PALETTE_COLORS = (1, 2, 3, 4, 6, 7, 8)

GRID_H_RANGE = (18, 30)
GRID_W_RANGE = (18, 30)
FRAME_COUNT_RANGE = (2, 4)
INTERIOR_SIZE_RANGE = (5, 9)
PLACEMENT_RETRIES = 200
GEN_RETRIES = 300

SHAPE_LIBRARY = [
    np.array([[0, 0, 0], [1, 1, 0], [0, 0, 0]], dtype=bool),
    np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=bool),
    np.array([[0, 1, 0], [0, 1, 1], [0, 0, 0]], dtype=bool),
    np.array([[0, 1, 1], [0, 1, 1], [0, 0, 0]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0], [0, 0, 0]], dtype=bool),
    np.array([[0, 1, 1], [1, 1, 0], [0, 0, 0]], dtype=bool),
    np.array([[0, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 0, 1], [1, 1, 1], [0, 0, 0]], dtype=bool),
]


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    return int(mins[0]), int(mins[1]), int(maxs[0]), int(maxs[1])


def _crop_true(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert len(coords) > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _shape_key(mask: np.ndarray) -> tuple[int, int, bytes]:
    cropped = _crop_true(mask)
    best = None
    for k in range(4):
        rot = _crop_true(np.rot90(cropped, k))
        key = (int(rot.shape[0]), int(rot.shape[1]), rot.astype(np.uint8).tobytes())
        if best is None or key < best:
            best = key
    assert best is not None
    return best


def _is_rect_frame(coords: np.ndarray) -> tuple[bool, tuple[int, int, int, int] | None]:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False, None
    patch = np.zeros((h, w), dtype=bool)
    patch[coords[:, 0] - r0, coords[:, 1] - c0] = True
    want = np.zeros((h, w), dtype=bool)
    want[0, :] = True
    want[-1, :] = True
    want[:, 0] = True
    want[:, -1] = True
    return bool(np.array_equal(patch, want)), (r0, c0, r1, c1)


def _parse_candidate_patch(patch: np.ndarray) -> tuple[int, tuple[int, int, bytes]]:
    assert patch.shape == (3, 3)
    colors = set(int(v) for v in np.unique(patch))
    assert BLACK not in colors
    assert MAROON in colors
    fills = [c for c in colors if c != MAROON]
    assert len(fills) == 1
    fill = int(fills[0])
    assert fill not in {BLACK, GREY, MAROON}
    hole = patch == MAROON
    assert hole.any()
    return fill, _shape_key(hole)


def _bottom_candidates(grid: np.ndarray) -> list[dict]:
    h, _ = grid.shape
    if h < 3:
        return []
    strip = grid[h - 3 : h, :] != BLACK
    out = []
    for coords in _components(strip):
        r0, c0, r1, c1 = _bbox(coords)
        if (r0, r1) != (0, 2) or (r1 - r0 + 1, c1 - c0 + 1) != (3, 3):
            continue
        patch = grid[h - 3 : h, c0 : c1 + 1]
        fill, key = _parse_candidate_patch(patch)
        out.append({"bbox": (h - 3, c0, h - 1, c1), "fill": fill, "key": key})
    out.sort(key=lambda item: item["bbox"][1])
    return out


def _right_candidates(grid: np.ndarray) -> list[dict]:
    _, w = grid.shape
    if w < 3:
        return []
    strip = grid[:, w - 3 : w] != BLACK
    out = []
    for coords in _components(strip):
        r0, c0, r1, c1 = _bbox(coords)
        if (c0, c1) != (0, 2) or (r1 - r0 + 1, c1 - c0 + 1) != (3, 3):
            continue
        patch = grid[r0 : r1 + 1, w - 3 : w]
        fill, key = _parse_candidate_patch(patch)
        out.append({"bbox": (r0, w - 3, r1, w - 1), "fill": fill, "key": key})
    out.sort(key=lambda item: item["bbox"][0])
    return out


def _extract_candidates(grid: np.ndarray) -> list[dict]:
    bottom = _bottom_candidates(grid)
    right = _right_candidates(grid)
    if len(bottom) >= 2 and len(bottom) >= len(right):
        return bottom
    if len(right) >= 2 and len(right) > len(bottom):
        return right
    if bottom and not right:
        return bottom
    if right and not bottom:
        return right
    raise AssertionError("failed to identify a unique edge candidate library")


def _apply_halo(interior: np.ndarray, fill: int) -> np.ndarray:
    out = interior.copy()
    maroon = interior == MAROON
    halo = np.zeros_like(maroon, dtype=bool)
    h, w = maroon.shape
    for r, c in np.argwhere(maroon):
        for nr, nc in _neighbors4(int(r), int(c), h, w):
            if interior[nr, nc] == BLACK:
                halo[nr, nc] = True
    out[halo] = fill
    return out


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    candidates = _extract_candidates(g)
    key_to_fill: dict[tuple[int, int, bytes], int] = {}
    for cand in candidates:
        key = cand["key"]
        assert key not in key_to_fill, "candidate shapes must be unique up to rotation"
        key_to_fill[key] = int(cand["fill"])
        r0, c0, r1, c1 = cand["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = BLACK

    grey_components = _components(g == GREY)
    assert grey_components, "expected at least one grey frame"
    frames = []
    for comp in grey_components:
        ok, box = _is_rect_frame(comp)
        assert ok and box is not None, "every grey component must be a rectangular frame"
        frames.append(box)

    seen_keys = set()
    for r0, c0, r1, c1 in frames:
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        colors = set(int(v) for v in np.unique(interior))
        assert colors <= {BLACK, MAROON}, "frame interiors may only contain black and maroon"
        maroon = interior == MAROON
        assert maroon.any(), "each frame must contain a maroon template"
        key = _shape_key(maroon)
        assert key not in seen_keys, "frame templates must be unique up to rotation"
        seen_keys.add(key)
        assert key in key_to_fill, "every frame template must match a candidate block"
        out[r0 + 1 : r1, c0 + 1 : c1] = _apply_halo(interior, key_to_fill[key])

    return out


def _can_place(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> bool:
    h0, w0 = occupied.shape
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    if r0 < 0 or c0 < 0 or r1 >= h0 or c1 >= w0:
        return False
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h0 - 1, r1 + pad)
    cc1 = min(w0 - 1, c1 + pad)
    return not occupied[rr0 : rr1 + 1, cc0 : cc1 + 1].any()


def _mark(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> None:
    h0, w0 = occupied.shape
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h0 - 1, r1 + pad)
    cc1 = min(w0 - 1, c1 + pad)
    occupied[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _build_expected(grid: np.ndarray, placements: list[dict], library_boxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    out = grid.copy()
    for r0, c0, r1, c1 in library_boxes:
        out[r0 : r1 + 1, c0 : c1 + 1] = BLACK
    for item in placements:
        fr0, fc0, fr1, fc1 = item["frame_bbox"]
        interior = grid[fr0 + 1 : fr1, fc0 + 1 : fc1]
        out[fr0 + 1 : fr1, fc0 + 1 : fc1] = _apply_halo(interior, int(item["fill"]))
    return out


def _gen_attempt() -> np.ndarray | None:
    frame_count = random.randint(*FRAME_COUNT_RANGE)
    orientation = random.choice(["bottom", "right"])
    shape_indices = random.sample(range(len(SHAPE_LIBRARY)), frame_count)
    fills = random.sample(PALETTE_COLORS, frame_count)

    placements: list[dict] = []
    for shape_idx, fill in zip(shape_indices, fills):
        interior_h = random.randint(*INTERIOR_SIZE_RANGE)
        interior_w = random.randint(*INTERIOR_SIZE_RANGE)
        shape_mask = np.rot90(SHAPE_LIBRARY[shape_idx], random.randrange(4))
        library_mask = np.rot90(SHAPE_LIBRARY[shape_idx], random.randrange(4))
        top_margin_max = interior_h - 4
        left_margin_max = interior_w - 4
        if top_margin_max < 1 or left_margin_max < 1:
            return None
        shape_r = random.randint(1, top_margin_max)
        shape_c = random.randint(1, left_margin_max)
        placements.append(
            {
                "fill": int(fill),
                "shape_mask": shape_mask,
                "library_mask": library_mask,
                "interior_h": int(interior_h),
                "interior_w": int(interior_w),
                "shape_r": int(shape_r),
                "shape_c": int(shape_c),
            }
        )

    library_span = frame_count * 3 + (frame_count - 1)
    min_h = GRID_H_RANGE[0]
    min_w = GRID_W_RANGE[0]
    if orientation == "bottom":
        min_h = max(min_h, max(item["interior_h"] + 2 for item in placements) + 4)
        min_w = max(min_w, library_span)
    else:
        min_w = max(min_w, max(item["interior_w"] + 2 for item in placements) + 4)
        min_h = max(min_h, library_span)
    if min_h > GRID_H_RANGE[1] or min_w > GRID_W_RANGE[1]:
        return None

    h = random.randint(min_h, GRID_H_RANGE[1])
    w = random.randint(min_w, GRID_W_RANGE[1])
    usable_h = h - 4 if orientation == "bottom" else h
    usable_w = w if orientation == "bottom" else w - 4
    if usable_h < 3 or usable_w < 3:
        return None

    grid = np.full((h, w), BLACK, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    order = list(range(frame_count))
    random.shuffle(order)
    order.sort(key=lambda idx: placements[idx]["interior_h"] * placements[idx]["interior_w"], reverse=True)
    for idx in order:
        item = placements[idx]
        frame_h = item["interior_h"] + 2
        frame_w = item["interior_w"] + 2
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            r0 = random.randint(0, usable_h - frame_h)
            c0 = random.randint(0, usable_w - frame_w)
            if not _can_place(occupied, r0, c0, frame_h, frame_w, pad=1):
                continue
            _mark(occupied, r0, c0, frame_h, frame_w, pad=1)
            item["frame_bbox"] = (r0, c0, r0 + frame_h - 1, c0 + frame_w - 1)
            placed = True
            break
        if not placed:
            return None

    for item in placements:
        r0, c0, r1, c1 = item["frame_bbox"]
        grid[r0, c0 : c1 + 1] = GREY
        grid[r1, c0 : c1 + 1] = GREY
        grid[r0 : r1 + 1, c0] = GREY
        grid[r0 : r1 + 1, c1] = GREY
        sr = r0 + 1 + item["shape_r"]
        sc = c0 + 1 + item["shape_c"]
        patch = grid[sr : sr + 3, sc : sc + 3]
        patch[item["shape_mask"]] = MAROON

    library_boxes = []
    library_order = list(range(frame_count))
    random.shuffle(library_order)
    if orientation == "bottom":
        start_c = random.randint(0, w - library_span)
        row0 = h - 3
        for slot, idx in enumerate(library_order):
            c0 = start_c + 4 * slot
            patch = np.full((3, 3), placements[idx]["fill"], dtype=int)
            patch[placements[idx]["library_mask"]] = MAROON
            grid[row0 : row0 + 3, c0 : c0 + 3] = patch
            library_boxes.append((row0, c0, row0 + 2, c0 + 2))
    else:
        start_r = random.randint(0, h - library_span)
        col0 = w - 3
        for slot, idx in enumerate(library_order):
            r0 = start_r + 4 * slot
            patch = np.full((3, 3), placements[idx]["fill"], dtype=int)
            patch[placements[idx]["library_mask"]] = MAROON
            grid[r0 : r0 + 3, col0 : col0 + 3] = patch
            library_boxes.append((r0, col0, r0 + 2, col0 + 2))

    expected = _build_expected(grid, placements, library_boxes)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
