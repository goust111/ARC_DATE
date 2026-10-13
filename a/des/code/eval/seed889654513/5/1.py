from __future__ import annotations

import random
from collections import deque

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR_TO_ROT_K = {
    (-1, 0): 0,  # pointer above tail -> keep as-is
    (0, 1): 3,   # pointer right of tail -> clockwise 90
    (1, 0): 2,   # pointer below tail -> 180
    (0, -1): 1,  # pointer left of tail -> counter-clockwise 90
}


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(cells: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    return not (
        a[2] + pad < b[0]
        or b[2] + pad < a[0]
        or a[3] + pad < b[1]
        or b[3] + pad < a[1]
    )


def _color_patch_from_cells(cells: list[tuple[int, int]], color: int) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(cells)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), -1, dtype=int)
    for r, c in cells:
        patch[r - r0, c - c0] = int(color)
    return patch


def _find_frames(grid: np.ndarray, bg: int) -> list[dict[str, int]]:
    frames: list[dict[str, int]] = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        for pts in _components(grid == color):
            r0, c0, r1, c1 = _bbox(pts)
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            if h < 3 or w < 3:
                continue
            if len(pts) != 2 * h + 2 * w - 4:
                continue
            cell_set = set(pts)
            ok = True
            for cc in range(c0, c1 + 1):
                if (r0, cc) not in cell_set or (r1, cc) not in cell_set:
                    ok = False
                    break
            if not ok:
                continue
            for rr in range(r0, r1 + 1):
                if (rr, c0) not in cell_set or (rr, c1) not in cell_set:
                    ok = False
                    break
            if not ok:
                continue
            if np.any(grid[r0 + 1 : r1, c0 + 1 : c1] == color):
                continue
            frames.append(
                {
                    "color": color,
                    "r0": int(r0),
                    "c0": int(c0),
                    "r1": int(r1),
                    "c1": int(c1),
                }
            )
    frames.sort(key=lambda item: (item["r0"], item["c0"]))
    return frames


def _frame_key(grid: np.ndarray, bg: int, frame: dict[str, int]) -> int:
    inner = grid[frame["r0"] + 1 : frame["r1"], frame["c0"] + 1 : frame["c1"]]
    colors = {int(v) for v in np.unique(inner) if int(v) != bg}
    return len(colors)


def _find_external_sprites(grid: np.ndarray, bg: int, frame_cover: np.ndarray) -> list[dict[str, object]]:
    mask = (grid != bg) & (~frame_cover)
    sprites: list[dict[str, object]] = []
    for pts in _components(mask):
        counts: dict[int, list[tuple[int, int]]] = {}
        for r, c in pts:
            counts.setdefault(int(grid[r, c]), []).append((r, c))
        assert len(counts) == 2, "each external object should contain one sprite color and one pointer color"

        pointer_colors = [color for color, cells in counts.items() if len(cells) == 1]
        assert len(pointer_colors) == 1, "pointer color must be the unique singleton color"
        pointer_color = pointer_colors[0]
        sprite_color = next(color for color in counts if color != pointer_color)

        pointer = counts[pointer_color][0]
        same_color_cells = counts[sprite_color]
        tail_candidates = [
            (r, c)
            for r, c in same_color_cells
            if abs(r - pointer[0]) + abs(c - pointer[1]) == 1
        ]
        assert len(tail_candidates) == 1, "pointer must touch exactly one same-color tail pixel"
        tail = tail_candidates[0]

        sprite_cells = [cell for cell in same_color_cells if cell != tail]
        assert sprite_cells, "sprite must contain at least one pixel besides the tail"
        dr = int(pointer[0] - tail[0])
        dc = int(pointer[1] - tail[1])
        assert (dr, dc) in DIR_TO_ROT_K, "pointer direction must be cardinal"

        sprite_patch = _color_patch_from_cells(sprite_cells, sprite_color)
        sprites.append(
            {
                "cells": pts,
                "sprite_patch": sprite_patch,
                "rot_k": int(DIR_TO_ROT_K[(dr, dc)]),
                "height_key": int(sprite_patch.shape[0]),
            }
        )
    sprites.sort(key=lambda item: (item["height_key"], len(item["cells"])))
    return sprites


def _paste_patch(dest: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    region = dest[top : top + h, left : left + w]
    mask = patch >= 0
    region[mask] = patch[mask]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each rectangular frame with the outside sprite whose original bbox height equals the
    number of distinct marker colors inside that frame, then rotate by the two-pixel indicator
    and place the sprite centered inside the frame.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)

    frames = _find_frames(g, bg)
    assert frames, "expected at least one rectangular frame"

    frame_cover = np.zeros(g.shape, dtype=bool)
    frame_by_key: dict[int, dict[str, int]] = {}
    for frame in frames:
        frame_cover[frame["r0"] : frame["r1"] + 1, frame["c0"] : frame["c1"] + 1] = True
        key = _frame_key(g, bg, frame)
        assert key > 0, "frame interiors must contain at least one marker color"
        assert key not in frame_by_key, "matching keys must be unique"
        frame_by_key[key] = frame

    sprites = _find_external_sprites(g, bg, frame_cover)
    assert sprites, "expected at least one external sprite+indicator object"
    sprite_by_key: dict[int, dict[str, object]] = {}
    for sprite in sprites:
        key = int(sprite["height_key"])
        assert key not in sprite_by_key, "sprite height keys must be unique"
        sprite_by_key[key] = sprite

    assert frame_by_key.keys() == sprite_by_key.keys(), "frame keys and sprite keys must match exactly"

    out = g.copy()
    out[(g != bg) & (~frame_cover)] = bg
    for frame in frames:
        out[frame["r0"] + 1 : frame["r1"], frame["c0"] + 1 : frame["c1"]] = bg

    for key, frame in frame_by_key.items():
        sprite = sprite_by_key[key]
        patch = np.rot90(np.asarray(sprite["sprite_patch"], dtype=int), int(sprite["rot_k"]))
        inner_h = frame["r1"] - frame["r0"] - 1
        inner_w = frame["c1"] - frame["c0"] - 1
        ph, pw = patch.shape
        assert ph <= inner_h and pw <= inner_w, "rotated sprite must fit inside the matched frame"
        top = frame["r0"] + 1 + (inner_h - ph) // 2
        left = frame["c0"] + 1 + (inner_w - pw) // 2
        _paste_patch(out, patch, top, left)

    return out


# --------------- generator ---------------

GEN_ATTEMPTS = 400
GRID_SIDE_RANGE = (24, 32)
OBJECT_COUNT_RANGE = (2, 4)
MATCH_KEYS = (1, 2, 3, 4, 5)
PLACEMENT_ATTEMPTS = 300


def _random_sprite_mask(height: int) -> np.ndarray:
    for _ in range(200):
        width_limit = random.randint(1, min(5, height + 1))
        backbone_col = random.randint(0, width_limit - 1)
        cells = {(r, backbone_col) for r in range(height)}
        max_area = min(height * width_limit, height + max(1, 2 * width_limit + random.randint(0, height)))
        target_area = random.randint(height, max_area)
        stalled = 0
        while len(cells) < target_area and stalled < 100:
            r, c = random.choice(tuple(cells))
            nbrs = []
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width_limit and (nr, nc) not in cells:
                    nbrs.append((nr, nc))
            if nbrs:
                cells.add(random.choice(nbrs))
                stalled = 0
            else:
                stalled += 1

        r0, c0, r1, c1 = _bbox(cells)
        if r1 - r0 + 1 != height:
            continue
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in cells:
            mask[r - r0, c - c0] = True
        return mask
    raise RuntimeError("failed to generate sprite mask")


def _colorize_mask(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.full(mask.shape, -1, dtype=int)
    patch[mask] = int(color)
    return patch


def _build_external_patch(
    sprite_mask: np.ndarray, sprite_color: int, pointer_color: int, direction: tuple[int, int]
) -> np.ndarray:
    sh, sw = sprite_mask.shape
    if direction == (-1, 0):
        patch = np.full((sh + 2, sw), -1, dtype=int)
        patch[2:, :][sprite_mask] = int(sprite_color)
        cols = [c for c in range(sw) if sprite_mask[0, c]]
        anchor_c = random.choice(cols)
        patch[1, anchor_c] = int(sprite_color)
        patch[0, anchor_c] = int(pointer_color)
        return patch
    if direction == (1, 0):
        patch = np.full((sh + 2, sw), -1, dtype=int)
        patch[:sh, :][sprite_mask] = int(sprite_color)
        cols = [c for c in range(sw) if sprite_mask[sh - 1, c]]
        anchor_c = random.choice(cols)
        patch[sh, anchor_c] = int(sprite_color)
        patch[sh + 1, anchor_c] = int(pointer_color)
        return patch
    if direction == (0, -1):
        patch = np.full((sh, sw + 2), -1, dtype=int)
        patch[:, 2:][sprite_mask] = int(sprite_color)
        rows = [r for r in range(sh) if sprite_mask[r, 0]]
        anchor_r = random.choice(rows)
        patch[anchor_r, 1] = int(sprite_color)
        patch[anchor_r, 0] = int(pointer_color)
        return patch
    if direction == (0, 1):
        patch = np.full((sh, sw + 2), -1, dtype=int)
        patch[:, :sw][sprite_mask] = int(sprite_color)
        rows = [r for r in range(sh) if sprite_mask[r, sw - 1]]
        anchor_r = random.choice(rows)
        patch[anchor_r, sw] = int(sprite_color)
        patch[anchor_r, sw + 1] = int(pointer_color)
        return patch
    raise ValueError(direction)


def _sample_box(h: int, w: int, box_h: int, box_w: int, margin: int = 1) -> tuple[int, int, int, int]:
    r_lo = margin
    c_lo = margin
    r_hi = h - box_h - margin
    c_hi = w - box_w - margin
    if r_hi < r_lo:
        r_lo = 0
        r_hi = h - box_h
    if c_hi < c_lo:
        c_lo = 0
        c_hi = w - box_w
    r0 = random.randint(r_lo, r_hi)
    c0 = random.randint(c_lo, c_hi)
    return r0, c0, r0 + box_h - 1, c0 + box_w - 1


def _place_boxes(shape: tuple[int, int], sizes: list[tuple[int, int]], pad: int) -> list[tuple[int, int, int, int]] | None:
    h, w = shape
    placed: list[tuple[int, int, int, int]] = []
    for box_h, box_w in sizes:
        found = None
        for _ in range(PLACEMENT_ATTEMPTS):
            box = _sample_box(h, w, box_h, box_w, margin=1)
            if all(not _boxes_overlap(box, other, pad=pad) for other in placed):
                found = box
                break
        if found is None:
            return None
        placed.append(found)
    return placed


def _random_frame_markers(inner_h: int, inner_w: int, count: int, marker_pool: list[int]) -> list[tuple[int, int, int]]:
    cells = [(r, c) for r in range(inner_h) for c in range(inner_w)]
    assert count <= len(cells)
    coords = random.sample(cells, count)
    colors = random.sample(marker_pool, count)
    return [(r, c, color) for (r, c), color in zip(coords, colors, strict=False)]


def _expected_from_spec(shape: tuple[int, int], bg: int, frame_color: int, specs: list[dict[str, object]]) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    for spec in specs:
        frame = spec["frame_box"]
        r0, c0, r1, c1 = frame
        out[r0, c0 : c1 + 1] = int(frame_color)
        out[r1, c0 : c1 + 1] = int(frame_color)
        out[r0 : r1 + 1, c0] = int(frame_color)
        out[r0 : r1 + 1, c1] = int(frame_color)

        patch = np.rot90(np.asarray(spec["sprite_patch"], dtype=int), int(spec["rot_k"]))
        inner_h = r1 - r0 - 1
        inner_w = c1 - c0 - 1
        ph, pw = patch.shape
        top = r0 + 1 + (inner_h - ph) // 2
        left = c0 + 1 + (inner_w - pw) // 2
        _paste_patch(out, patch, top, left)
    return out


def _generate_case() -> tuple[np.ndarray, np.ndarray] | None:
    n = random.randint(*OBJECT_COUNT_RANGE)
    keys = random.sample(MATCH_KEYS, n)

    colors = list(range(10))
    random.shuffle(colors)
    bg = colors[0]
    frame_color = colors[1]
    sprite_colors = colors[2 : 2 + n]
    pointer_colors = colors[2 + n : 2 + 2 * n]
    marker_pool = colors[2:]

    directions = [random.choice(DIR4) for _ in range(n)]

    specs: list[dict[str, object]] = []
    frame_sizes: list[tuple[int, int]] = []
    external_sizes: list[tuple[int, int]] = []
    for key, sprite_color, pointer_color, direction in zip(
        keys, sprite_colors, pointer_colors, directions, strict=False
    ):
        mask = _random_sprite_mask(key)
        sprite_patch = _colorize_mask(mask, sprite_color)
        rot_k = DIR_TO_ROT_K[direction]
        rotated = np.rot90(sprite_patch, rot_k)
        rh, rw = rotated.shape
        inner_h = rh + random.randint(1, 4)
        inner_w = rw + random.randint(1, 4)
        inner_h = max(inner_h, 3)
        inner_w = max(inner_w, 3)

        frame_h = inner_h + 2
        frame_w = inner_w + 2
        external_patch = _build_external_patch(mask, sprite_color, pointer_color, direction)
        markers = _random_frame_markers(inner_h, inner_w, key, marker_pool)

        specs.append(
            {
                "key": int(key),
                "sprite_patch": sprite_patch,
                "rot_k": int(rot_k),
                "external_patch": external_patch,
                "markers": markers,
            }
        )
        frame_sizes.append((frame_h, frame_w))
        external_sizes.append(tuple(map(int, external_patch.shape)))

    side_low = max(GRID_SIDE_RANGE[0], 10 + 4 * n)
    side_high = max(side_low, GRID_SIDE_RANGE[1])
    h = random.randint(side_low, side_high)
    w = random.randint(side_low, side_high)

    frame_boxes = _place_boxes((h, w), frame_sizes, pad=2)
    if frame_boxes is None:
        return None

    all_boxes = frame_boxes[:]
    external_boxes: list[tuple[int, int, int, int]] = []
    for patch_h, patch_w in external_sizes:
        found = None
        for _ in range(PLACEMENT_ATTEMPTS):
            box = _sample_box(h, w, patch_h, patch_w, margin=1)
            if all(not _boxes_overlap(box, other, pad=1) for other in all_boxes):
                found = box
                break
        if found is None:
            return None
        external_boxes.append(found)
        all_boxes.append(found)

    inp = np.full((h, w), int(bg), dtype=int)
    for spec, frame_box, external_box in zip(specs, frame_boxes, external_boxes, strict=False):
        spec["frame_box"] = frame_box
        spec["external_box"] = external_box

        r0, c0, r1, c1 = frame_box
        inp[r0, c0 : c1 + 1] = int(frame_color)
        inp[r1, c0 : c1 + 1] = int(frame_color)
        inp[r0 : r1 + 1, c0] = int(frame_color)
        inp[r0 : r1 + 1, c1] = int(frame_color)

        for rr, cc, color in spec["markers"]:
            inp[r0 + 1 + rr, c0 + 1 + cc] = int(color)

        patch = np.asarray(spec["external_patch"], dtype=int)
        er0, ec0, _, _ = external_box
        region = inp[er0 : er0 + patch.shape[0], ec0 : ec0 + patch.shape[1]]
        mask = patch >= 0
        region[mask] = patch[mask]

    expected = _expected_from_spec((h, w), bg, frame_color, specs)
    if not np.array_equal(main(inp), expected):
        return None
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        case = _generate_case()
        if case is not None:
            return case[0]
    raise RuntimeError("failed to generate a valid sample")
