from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import bbox, bbox_coords, colors_present, components_mask, crop_bbox, get_bg, paste


BG = 0
FRAME_COUNT_RANGE = (2, 4)
ROOM_DIM_RANGE = (3, 5)
OUTER_MARGIN_RANGE = (0, 2)
PLACEMENT_GAP = 2
CANVAS_LIMIT = 30
GENERATE_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 250

SPRITE_SHAPES = [
    np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [1, 0, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 0, 1], [1, 1, 0], [0, 1, 0]], dtype=bool),
]

ROOM_LAYOUTS = {
    0: [(2, 2)],
    1: [(1, 1), (3, 3)],
    2: [(1, 2), (2, 1), (2, 3), (3, 2)],
    3: [(1, 3), (3, 1)],
}
COMPLEX_LAYOUTS = [(1, 2), (2, 1), (1, 3), (3, 1), (2, 2), (2, 3), (3, 2), (3, 3)]


def _bbox_hw(box: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = box
    return r1 - r0 + 1, c1 - c0 + 1


def _frame_info_from_mask(mask: np.ndarray) -> dict | None:
    box = bbox(mask)
    if box is None:
        return None
    patch = crop_bbox(mask.astype(int), box).astype(bool)
    h, w = patch.shape
    if h < 5 or w < 5:
        return None

    row_counts = patch.sum(axis=1)
    col_counts = patch.sum(axis=0)
    dense_rows = [r for r, count in enumerate(row_counts) if int(count) >= w - 1]
    dense_cols = [c for c, count in enumerate(col_counts) if int(count) >= h - 1]
    if len(dense_rows) < 2 or len(dense_cols) < 2:
        return None
    if dense_rows[0] != 0 or dense_rows[-1] != h - 1 or dense_cols[0] != 0 or dense_cols[-1] != w - 1:
        return None

    repaired = np.zeros_like(patch, dtype=bool)
    repaired[dense_rows, :] = True
    repaired[:, dense_cols] = True

    if np.any(patch & ~repaired):
        return None
    if not np.any(~repaired):
        return None

    rooms = len(components_mask(~repaired, connectivity=4))
    return {
        "bbox": box,
        "dense_rows": dense_rows,
        "dense_cols": dense_cols,
        "repaired": repaired,
        "room_count": int(rooms),
    }


def _find_frames(grid: np.ndarray) -> list[dict]:
    bg = int(get_bg(grid))
    frames: list[dict] = []
    for color in colors_present(grid, exclude={bg}):
        info = _frame_info_from_mask(grid == int(color))
        if info is None:
            continue
        info["color"] = int(color)
        frames.append(info)
    return frames


def _rotate_sprite_in_place(out: np.ndarray, grid: np.ndarray, color: int, turns: int) -> None:
    coords = np.argwhere(grid == int(color))
    assert len(coords) > 0
    r0, c0, r1, c1 = bbox_coords(coords)
    patch = grid[r0 : r1 + 1, c0 : c1 + 1] == int(color)
    rotated = np.rot90(patch, int(turns) % 4)
    region = out[r0 : r1 + 1, c0 : c1 + 1]
    region[region == int(color)] = BG
    region[rotated] = int(color)


def main(input_grid: np.ndarray) -> np.ndarray:
    """Repair one-cell wall gaps and rotate each inner sprite by the repaired room count modulo four.

    Input structure:
    - A black background contains several monochrome rectangular partitioned frames.
    - Each frame color draws full-width horizontal walls and full-height vertical walls, but some wall lines have a single missing cell.
    - Exactly one small non-90-symmetric sprite of another color sits inside each frame.

    Rule:
    - For every frame color, find its bounding box and infer the dense wall rows and columns inside that box.
    - Fill every inferred row and column completely with the frame color, which repairs every one-cell gap and restores the closed partitioned frame.
    - Count the 4-connected non-wall regions inside the repaired frame, then rotate the sprite inside that frame counterclockwise by 90 degrees times (room_count mod 4).

    Key insight:
    - The visible room count is only correct after wall repair; the repaired partitions, not the damaged input, determine the sprite rotation.
    """
    g = np.asarray(input_grid, dtype=int)
    assert g.ndim == 2

    out = g.copy()
    frames = _find_frames(g)
    assert frames

    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        patch = out[r0 : r1 + 1, c0 : c1 + 1]
        patch[frame["repaired"]] = int(frame["color"])

    frame_by_sprite: dict[int, dict] = {}
    frame_colors = {int(frame["color"]) for frame in frames}
    for color in colors_present(g, exclude={BG}):
        if int(color) in frame_colors:
            continue
        coords = np.argwhere(g == int(color))
        r0, c0, r1, c1 = bbox_coords(coords)
        matches = []
        for frame in frames:
            fr0, fc0, fr1, fc1 = frame["bbox"]
            if fr0 < r0 and c0 > fc0 and r1 < fr1 and c1 < fc1:
                matches.append(frame)
        assert len(matches) == 1
        frame_by_sprite[int(color)] = matches[0]

    assert len(frame_by_sprite) == len(frames)
    for color, frame in frame_by_sprite.items():
        _rotate_sprite_in_place(out, g, int(color), int(frame["room_count"]) % 4)

    return out


def _line_positions(lengths: list[int]) -> list[int]:
    positions = [0]
    cursor = 0
    for size in lengths:
        cursor += int(size) + 1
        positions.append(cursor)
    return positions


def _room_rects(row_sizes: list[int], col_sizes: list[int]) -> list[tuple[int, int, int, int]]:
    rows = _line_positions(row_sizes)
    cols = _line_positions(col_sizes)
    rects = []
    for i in range(len(row_sizes)):
        for j in range(len(col_sizes)):
            rects.append((rows[i] + 1, cols[j] + 1, rows[i + 1] - 1, cols[j + 1] - 1))
    return rects


def _build_repaired_mask(row_sizes: list[int], col_sizes: list[int]) -> tuple[np.ndarray, list[int], list[int]]:
    h = sum(row_sizes) + len(row_sizes) + 1
    w = sum(col_sizes) + len(col_sizes) + 1
    repaired = np.zeros((h, w), dtype=bool)
    rows = _line_positions(row_sizes)
    cols = _line_positions(col_sizes)
    repaired[rows, :] = True
    repaired[:, cols] = True
    return repaired, rows, cols


def _count_rooms_from_mask(wall_mask: np.ndarray) -> int:
    return len(components_mask(~wall_mask, connectivity=4))


def _damage_walls(
    repaired: np.ndarray,
    wall_rows: list[int],
    wall_cols: list[int],
    rng: random.Random,
    *,
    force_internal_gap: bool,
) -> tuple[np.ndarray, list[dict]]:
    damaged = repaired.copy()
    h, w = repaired.shape
    candidates: list[dict] = []
    for r in wall_rows:
        cells = [c for c in range(w) if c not in wall_cols]
        if cells:
            candidates.append({"axis": "h", "pos": r, "cells": cells, "internal": r not in (0, h - 1)})
    for c in wall_cols:
        cells = [r for r in range(h) if r not in wall_rows]
        if cells:
            candidates.append({"axis": "v", "pos": c, "cells": cells, "internal": c not in (0, w - 1)})

    internal = [item for item in candidates if item["internal"]]
    chosen: list[dict] = []
    if force_internal_gap and internal:
        chosen.append(rng.choice(internal))

    remaining = [item for item in candidates if item not in chosen]
    rng.shuffle(remaining)
    extra = rng.randint(0, min(2, len(remaining)))
    chosen.extend(remaining[:extra])
    if not chosen:
        chosen.append(rng.choice(candidates))

    gaps: list[dict] = []
    for item in chosen:
        if item["axis"] == "h":
            r = int(item["pos"])
            c = int(rng.choice(item["cells"]))
            damaged[r, c] = False
            gaps.append({"axis": "h", "r": r, "c": c, "internal": bool(item["internal"])})
        else:
            c = int(item["pos"])
            r = int(rng.choice(item["cells"]))
            damaged[r, c] = False
            gaps.append({"axis": "v", "r": r, "c": c, "internal": bool(item["internal"])})
    return damaged, gaps


def _sample_layout(rng: random.Random, desired_mod: int | None = None, *, prefer_complex: bool) -> tuple[int, int]:
    if desired_mod is None:
        options = COMPLEX_LAYOUTS if prefer_complex else sum(ROOM_LAYOUTS.values(), [])
    else:
        options = ROOM_LAYOUTS[int(desired_mod)]
        if prefer_complex:
            complex_options = [item for item in options if item != (1, 1)]
            if complex_options:
                options = complex_options
    return rng.choice(options)


def _make_frame_case(
    rng: random.Random,
    wall_color: int,
    sprite_color: int,
    *,
    desired_mod: int | None,
    force_internal_gap: bool,
) -> dict:
    room_rows, room_cols = _sample_layout(rng, desired_mod, prefer_complex=force_internal_gap)
    row_sizes = [rng.randint(*ROOM_DIM_RANGE) for _ in range(room_rows)]
    col_sizes = [rng.randint(*ROOM_DIM_RANGE) for _ in range(room_cols)]
    repaired, wall_rows, wall_cols = _build_repaired_mask(row_sizes, col_sizes)
    room_rects = _room_rects(row_sizes, col_sizes)
    sprite_shape = rng.choice(SPRITE_SHAPES)
    sprite_room = rng.choice(room_rects)

    sr0, sc0, sr1, sc1 = sprite_room
    room_h = sr1 - sr0 + 1
    room_w = sc1 - sc0 + 1
    top = rng.randint(sr0, sr1 - 2)
    left = rng.randint(sc0, sc1 - 2)
    assert top + 2 <= sr1 and left + 2 <= sc1

    damaged, gaps = _damage_walls(
        repaired,
        wall_rows,
        wall_cols,
        rng,
        force_internal_gap=force_internal_gap,
    )
    input_patch = np.full(repaired.shape, BG, dtype=int)
    input_patch[damaged] = int(wall_color)
    sprite_cells = np.argwhere(sprite_shape)
    for dr, dc in sprite_cells:
        input_patch[top + int(dr), left + int(dc)] = int(sprite_color)

    output_patch = np.full(repaired.shape, BG, dtype=int)
    output_patch[repaired] = int(wall_color)
    rotated = np.rot90(sprite_shape, _count_rooms_from_mask(repaired) % 4)
    sprite_box = (top, left, top + 2, left + 2)
    for dr, dc in np.argwhere(rotated):
        output_patch[top + int(dr), left + int(dc)] = int(sprite_color)

    damaged_room_count = _count_rooms_from_mask(damaged)
    repaired_room_count = _count_rooms_from_mask(repaired)
    assert repaired_room_count == room_rows * room_cols
    if any(gap["internal"] for gap in gaps):
        assert damaged_room_count < repaired_room_count
    assert room_h >= 3 and room_w >= 3
    return {
        "input_patch": input_patch,
        "output_patch": output_patch,
        "room_count": repaired_room_count,
        "damaged_room_count": damaged_room_count,
        "gaps": gaps,
        "sprite_box": sprite_box,
    }


def _rects_separate(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _place_patches(rng: random.Random, cases: list[dict]) -> tuple[np.ndarray, np.ndarray] | None:
    order = sorted(range(len(cases)), key=lambda idx: -cases[idx]["input_patch"].size)
    placements: dict[int, tuple[int, int]] = {}
    boxes: list[tuple[int, int, int, int]] = []

    for idx in order:
        patch = cases[idx]["input_patch"]
        h, w = patch.shape
        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            r0 = rng.randint(0, CANVAS_LIMIT - h)
            c0 = rng.randint(0, CANVAS_LIMIT - w)
            cand = (r0, c0, r0 + h - 1, c0 + w - 1)
            if all(_rects_separate(cand, box, PLACEMENT_GAP) for box in boxes):
                placements[idx] = (r0, c0)
                boxes.append(cand)
                placed = True
                break
        if not placed:
            return None

    input_grid = np.full((CANVAS_LIMIT, CANVAS_LIMIT), BG, dtype=int)
    output_grid = np.full((CANVAS_LIMIT, CANVAS_LIMIT), BG, dtype=int)
    for idx, case in enumerate(cases):
        r0, c0 = placements[idx]
        paste(input_grid, case["input_patch"], r0, c0)
        paste(output_grid, case["output_patch"], r0, c0)

    coords = np.argwhere(input_grid != BG)
    assert len(coords) > 0
    r0, c0, r1, c1 = bbox_coords(coords)
    top = min(r0, rng.randint(*OUTER_MARGIN_RANGE))
    left = min(c0, rng.randint(*OUTER_MARGIN_RANGE))
    bottom = min(CANVAS_LIMIT - 1 - r1, rng.randint(*OUTER_MARGIN_RANGE))
    right = min(CANVAS_LIMIT - 1 - c1, rng.randint(*OUTER_MARGIN_RANGE))
    crop = (r0 - top, c0 - left, r1 + bottom, c1 + right)
    return crop_bbox(input_grid, crop), crop_bbox(output_grid, crop)


def _generate_case(rng: random.Random | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    rng = rng or random.Random()
    frame_count = rng.randint(*FRAME_COUNT_RANGE)
    colors = rng.sample(range(1, 10), 2 * frame_count)
    wall_colors = colors[:frame_count]
    sprite_colors = colors[frame_count:]

    anchor_mod = rng.randint(0, 3)
    internal_gap_index = rng.randrange(frame_count)
    cases: list[dict] = []
    mods: list[int] = []
    for idx in range(frame_count):
        desired_mod = anchor_mod if idx == 0 else None
        force_internal_gap = idx == internal_gap_index
        case = _make_frame_case(
            rng,
            wall_colors[idx],
            sprite_colors[idx],
            desired_mod=desired_mod,
            force_internal_gap=force_internal_gap,
        )
        cases.append(case)
        mods.append(int(case["room_count"]) % 4)

    if len(set(mods)) == 1:
        return _generate_case(rng)

    placed = _place_patches(rng, cases)
    if placed is None:
        return _generate_case(rng)
    input_grid, output_grid = placed

    meta = {
        "room_mods": mods,
        "has_internal_gap": any(any(gap["internal"] for gap in case["gaps"]) for case in cases),
        "room_counts": [int(case["room_count"]) for case in cases],
        "damaged_room_counts": [int(case["damaged_room_count"]) for case in cases],
    }
    return input_grid, output_grid, meta


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_ATTEMPTS):
        grid, expected, meta = _generate_case()
        actual = main(grid)
        if not np.array_equal(actual, expected):
            continue
        if not meta["has_internal_gap"]:
            continue
        if max(grid.shape) > CANVAS_LIMIT or max(expected.shape) > CANVAS_LIMIT:
            continue
        return grid
    raise RuntimeError("failed to generate valid input")
