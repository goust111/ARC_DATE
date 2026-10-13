from __future__ import annotations

import random
from typing import Iterable

import numpy as np


PAYLOAD_COLOR = 9
FRAME_COLORS = [1, 2, 3, 4, 5, 6, 7, 8]

FRAME_COUNT_RANGE = (2, 4)
INNER_SIZE_RANGE = (5, 9)
ANCHOR_COUNT_CHOICES = [1, 2]
GRID_MIN_SIZE = 24
GRID_MAX_SIZE = 38
OBJECT_PADDING = 1
MAX_GENERATION_ATTEMPTS = 400


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    if connectivity == 4:
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        steps = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ]

    comps: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        stack = [(r, c)]
        seen[r, c] = True
        cells: list[tuple[int, int]] = []
        while stack:
            cr, cc = stack.pop()
            cells.append((cr, cc))
            for dr, dc in steps:
                nr = cr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(np.array(cells, dtype=int))
    return comps


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(np.min(coords[:, 0])),
        int(np.min(coords[:, 1])),
        int(np.max(coords[:, 0])),
        int(np.max(coords[:, 1])),
    )


def _is_hollow_frame(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = bbox_coords(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    if len(coords) != 2 * (h + w) - 4:
        return False
    mask = np.zeros((h, w), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    if not mask[0, :].all():
        return False
    if not mask[-1, :].all():
        return False
    if not mask[:, 0].all():
        return False
    if not mask[:, -1].all():
        return False
    return not mask[1:-1, 1:-1].any()


def _find_frames(
    grid: np.ndarray, bg: int
) -> tuple[list[dict[str, object]], np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    frame_mask = np.zeros(g.shape, dtype=bool)
    anchor_mask = np.zeros(g.shape, dtype=bool)
    frames: list[dict[str, object]] = []

    for color in [int(c) for c in np.unique(g) if int(c) != bg]:
        for coords in components_mask(g == color, connectivity=4):
            if not _is_hollow_frame(coords):
                continue
            r0, c0, r1, c1 = bbox_coords(coords)
            anchors: list[tuple[int, int]] = []
            for rr in range(r0 + 1, r1):
                for cc in range(c0 + 1, c1):
                    if g[rr, cc] == color:
                        anchors.append((rr, cc))
            if not (1 <= len(anchors) <= 2):
                continue
            frame_mask[coords[:, 0], coords[:, 1]] = True
            for rr, cc in anchors:
                anchor_mask[rr, cc] = True
            frames.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "anchors": tuple(sorted(anchors)),
                }
            )

    by_color: dict[int, dict[str, object]] = {}
    for frame in frames:
        color = int(frame["color"])
        assert color not in by_color, "frame colors must be unique"
        by_color[color] = frame
    return frames, frame_mask, anchor_mask


def _solve_translation(
    source_pts: Iterable[tuple[int, int]], target_pts: Iterable[tuple[int, int]]
) -> tuple[int, int]:
    src = [tuple(map(int, pt)) for pt in source_pts]
    dst = [tuple(map(int, pt)) for pt in target_pts]
    assert len(src) == len(dst) and src, "source and target anchors must match in count"
    dst_set = set(dst)
    candidates: set[tuple[int, int]] = set()
    for sr, sc in src:
        for tr, tc in dst:
            dr = tr - sr
            dc = tc - sc
            shifted = {(rr + dr, cc + dc) for rr, cc in src}
            if shifted == dst_set:
                candidates.add((dr, dc))
    assert len(candidates) == 1, "fragment alignment must be unique"
    return next(iter(candidates))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Route each external two-color fragment into the frame of the same non-maroon color,
    align its routing pixels with the frame anchors, and stamp only the maroon payload.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = get_bg(g)
    frames, frame_mask, anchor_mask = _find_frames(g, bg)
    assert frames, "expected at least one framed target"
    frames_by_color = {int(frame["color"]): frame for frame in frames}

    movable_mask = (g != bg) & (~frame_mask) & (~anchor_mask)
    out = g.copy()
    out[movable_mask] = bg

    for coords in components_mask(movable_mask, connectivity=4):
        colors = {int(g[r, c]) for r, c in coords}
        owner_candidates = colors & set(frames_by_color)
        assert len(owner_candidates) == 1, "each fragment must point to exactly one frame color"
        owner_color = next(iter(owner_candidates))
        payload_colors = colors - {owner_color}
        assert len(payload_colors) == 1, "each fragment must have exactly one payload color"
        payload_color = next(iter(payload_colors))

        owner_pts = [(int(r), int(c)) for r, c in coords if int(g[r, c]) == owner_color]
        assert 1 <= len(owner_pts) <= 2, "fragments need one or two routing pixels"
        frame = frames_by_color[owner_color]
        anchors = list(frame["anchors"])
        shift_r, shift_c = _solve_translation(owner_pts, anchors)

        fr0, fc0, fr1, fc1 = frame["bbox"]
        for r, c in coords:
            if int(g[r, c]) == owner_color:
                continue
            rr = int(r) + shift_r
            cc = int(c) + shift_c
            assert fr0 < rr < fr1 and fc0 < cc < fc1, "payload must land inside its frame"
            out[rr, cc] = payload_color

    return out


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < h and 0 <= nc < w:
            out.append((nr, nc))
    return out


def _random_path(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = a
    br, bc = b
    steps: list[tuple[int, int]] = []
    if br != r:
        steps.extend([(1 if br > r else -1, 0)] * abs(br - r))
    if bc != c:
        steps.extend([(0, 1 if bc > c else -1)] * abs(bc - c))
    random.shuffle(steps)
    path = [(r, c)]
    for dr, dc in steps:
        r += dr
        c += dc
        path.append((r, c))
    return path


def _sample_anchors(inner_h: int, inner_w: int, count: int) -> list[tuple[int, int]]:
    core_cells = [(r, c) for r in range(1, inner_h - 1) for c in range(1, inner_w - 1)]
    if count == 1:
        return [random.choice(core_cells)]
    for _ in range(100):
        a, b = random.sample(core_cells, 2)
        dist = abs(a[0] - b[0]) + abs(a[1] - b[1])
        if 1 <= dist <= 6:
            return [a, b]
    return random.sample(core_cells, 2)


def _grow_fragment(
    inner_h: int, inner_w: int, anchors: list[tuple[int, int]]
) -> set[tuple[int, int]]:
    cells = set(anchors)
    if len(anchors) == 2:
        cells.update(_random_path(anchors[0], anchors[1]))

    extra_cap = min(8, inner_h * inner_w - len(cells))
    target = len(cells) + random.randint(2, max(2, extra_cap))
    target = min(target, inner_h * inner_w)

    while len(cells) < target:
        frontier = {
            nb
            for r, c in cells
            for nb in _neighbors4(r, c, inner_h, inner_w)
            if nb not in cells
        }
        if not frontier:
            break
        cells.add(random.choice(tuple(frontier)))

    while len(cells) - len(anchors) < 2:
        frontier = {
            nb
            for r, c in cells
            for nb in _neighbors4(r, c, inner_h, inner_w)
            if nb not in cells
        }
        if not frontier:
            break
        cells.add(random.choice(tuple(frontier)))
    return cells


def _patch_from_cells(
    cells: set[tuple[int, int]], anchors: list[tuple[int, int]], owner_color: int
) -> tuple[np.ndarray, tuple[int, int], list[tuple[int, int]]]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0 = min(rows)
    c0 = min(cols)
    h = max(rows) - r0 + 1
    w = max(cols) - c0 + 1
    patch = np.zeros((h, w), dtype=int)
    anchor_set = set(anchors)
    payload_local: list[tuple[int, int]] = []
    for r, c in sorted(cells):
        rr = r - r0
        cc = c - c0
        if (r, c) in anchor_set:
            patch[rr, cc] = owner_color
        else:
            patch[rr, cc] = PAYLOAD_COLOR
            payload_local.append((r, c))
    return patch, (r0, c0), payload_local


def _can_place(used: np.ndarray, top: int, left: int, h: int, w: int, pad: int) -> bool:
    H, W = used.shape
    if top < 1 or left < 1 or top + h > H - 1 or left + w > W - 1:
        return False
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(H, top + h + pad)
    c1 = min(W, left + w + pad)
    return not used[r0:r1, c0:c1].any()


def _occupy(used: np.ndarray, top: int, left: int, h: int, w: int, pad: int) -> None:
    H, W = used.shape
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(H, top + h + pad)
    c1 = min(W, left + w + pad)
    used[r0:r1, c0:c1] = True


def _place_box(used: np.ndarray, h: int, w: int, pad: int) -> tuple[int, int] | None:
    H, W = used.shape
    if H - h - 1 < 1 or W - w - 1 < 1:
        return None
    for _ in range(250):
        top = random.randint(1, H - h - 1)
        left = random.randint(1, W - w - 1)
        if _can_place(used, top, left, h, w, pad):
            _occupy(used, top, left, h, w, pad)
            return top, left
    return None


def _draw_frame(grid: np.ndarray, top: int, left: int, inner_h: int, inner_w: int, color: int) -> None:
    bottom = top + inner_h + 1
    right = left + inner_w + 1
    grid[top, left : right + 1] = color
    grid[bottom, left : right + 1] = color
    grid[top : bottom + 1, left] = color
    grid[top : bottom + 1, right] = color


def _gen_attempt() -> np.ndarray | None:
    frame_count = random.randint(*FRAME_COUNT_RANGE)
    frame_colors = random.sample(FRAME_COLORS, frame_count)

    specs: list[dict[str, object]] = []
    total_padded_area = 0
    max_h = 0
    max_w = 0

    for color in frame_colors:
        inner_h = random.randint(*INNER_SIZE_RANGE)
        inner_w = random.randint(*INNER_SIZE_RANGE)
        anchor_count = random.choice(ANCHOR_COUNT_CHOICES)
        anchors = _sample_anchors(inner_h, inner_w, anchor_count)
        cells = _grow_fragment(inner_h, inner_w, anchors)
        patch, crop_origin, payload_local = _patch_from_cells(cells, anchors, color)
        outer_h = inner_h + 2
        outer_w = inner_w + 2
        patch_h, patch_w = patch.shape
        total_padded_area += (outer_h + 2 * OBJECT_PADDING) * (outer_w + 2 * OBJECT_PADDING)
        total_padded_area += (patch_h + 2 * OBJECT_PADDING) * (patch_w + 2 * OBJECT_PADDING)
        max_h = max(max_h, outer_h, patch_h)
        max_w = max(max_w, outer_w, patch_w)
        specs.append(
            {
                "color": color,
                "inner_h": inner_h,
                "inner_w": inner_w,
                "anchors": anchors,
                "cells": cells,
                "patch": patch,
                "crop_origin": crop_origin,
                "payload_local": payload_local,
            }
        )

    base = int(np.ceil(np.sqrt(total_padded_area * random.uniform(1.35, 1.7))))
    low_h = max(GRID_MIN_SIZE, max_h + 4, base - 3)
    low_w = max(GRID_MIN_SIZE, max_w + 4, base - 3)
    high_h = min(GRID_MAX_SIZE, low_h + 8)
    high_w = min(GRID_MAX_SIZE, low_w + 8)
    if low_h > high_h or low_w > high_w:
        return None
    H = random.randint(low_h, high_h)
    W = random.randint(low_w, high_w)

    grid = np.zeros((H, W), dtype=int)
    expected = np.zeros((H, W), dtype=int)
    used = np.zeros((H, W), dtype=bool)

    frame_order = sorted(
        range(frame_count),
        key=lambda idx: int(specs[idx]["inner_h"]) * int(specs[idx]["inner_w"]),
        reverse=True,
    )
    for idx in frame_order:
        spec = specs[idx]
        outer_h = int(spec["inner_h"]) + 2
        outer_w = int(spec["inner_w"]) + 2
        pos = _place_box(used, outer_h, outer_w, OBJECT_PADDING)
        if pos is None:
            return None
        top, left = pos
        spec["frame_top"] = top
        spec["frame_left"] = left
        _draw_frame(grid, top, left, int(spec["inner_h"]), int(spec["inner_w"]), int(spec["color"]))
        _draw_frame(expected, top, left, int(spec["inner_h"]), int(spec["inner_w"]), int(spec["color"]))

        for ar, ac in spec["anchors"]:
            rr = top + 1 + ar
            cc = left + 1 + ac
            grid[rr, cc] = int(spec["color"])
            expected[rr, cc] = int(spec["color"])

        for pr, pc in spec["payload_local"]:
            rr = top + 1 + pr
            cc = left + 1 + pc
            expected[rr, cc] = PAYLOAD_COLOR

    fragment_order = sorted(
        range(frame_count),
        key=lambda idx: int(np.count_nonzero(specs[idx]["patch"])),
        reverse=True,
    )
    for idx in fragment_order:
        spec = specs[idx]
        patch = np.asarray(spec["patch"], dtype=int)
        ph, pw = patch.shape
        pos = _place_box(used, ph, pw, OBJECT_PADDING)
        if pos is None:
            return None
        top, left = pos
        spec["fragment_top"] = top
        spec["fragment_left"] = left
        nonzero = np.argwhere(patch != 0)
        for rr, cc in nonzero:
            grid[top + int(rr), left + int(cc)] = int(patch[rr, cc])

    if np.array_equal(grid, expected):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
