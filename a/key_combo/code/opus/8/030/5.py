from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import colors_present, find_components, get_bg, random_free_location


BG = 0
FRAME_COLOR = 1
MOTIF_COLORS = (2, 3, 4, 5, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 400
FRAME_COUNT_CHOICES = (2, 3)
CANVAS_ROWS = (18, 28)
CANVAS_COLS = (18, 28)
WIDE_INNER_HEIGHTS = (4, 5, 6)
WIDE_INNER_WIDTHS = (7, 8, 9, 10)
TALL_INNER_HEIGHTS = (7, 8, 9, 10)
TALL_INNER_WIDTHS = (4, 5, 6)
SQUARE_INNER_SIDES = (6, 7, 8)
FRAME_GAP = 2
FRAME_RECTS_CONNECTIVITY = 8
MOTIF_SIZE_RANGE = (3, 6)


def _infer_background_and_frame(g: np.ndarray) -> tuple[int, int]:
    bg = int(get_bg(g))
    colors = colors_present(g, exclude={bg})
    assert colors, "expected foreground colors"
    counts = {int(color): int(np.count_nonzero(g == int(color))) for color in colors}
    largest = max(counts.values())
    frame_candidates = [color for color, area in counts.items() if area == largest]
    assert len(frame_candidates) == 1, "frame color must be uniquely largest"
    return bg, int(frame_candidates[0])


def _frame_objects(g: np.ndarray) -> tuple[int, int, list[dict]]:
    bg, frame_color = _infer_background_and_frame(g)
    frames = find_components(
        g,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        colors=[frame_color],
        return_format="dict",
    )
    assert frames, "expected at least one frame"
    frames = sorted(frames, key=lambda comp: tuple(map(int, comp["bbox"])))
    return bg, frame_color, frames


def _frame_border_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _local_mirror(lr: int, lc: int, inner_h: int, inner_w: int) -> tuple[int, int]:
    if inner_w >= inner_h:
        return int(lr), int(inner_w - 1 - lc)
    return int(inner_h - 1 - lr), int(lc)


def _solve_frame(
    g: np.ndarray,
    out: np.ndarray,
    bg: int,
    frame_color: int,
    bbox: tuple[int, int, int, int],
) -> dict:
    r0, c0, r1, c1 = map(int, bbox)
    frame_patch = g[r0 : r1 + 1, c0 : c1 + 1]
    h, w = frame_patch.shape
    assert h >= 3 and w >= 3, "frame must enclose a non-empty interior"

    border = _frame_border_mask(h, w)
    assert np.array_equal(frame_patch == frame_color, border), "frame must be a 1-cell rectangular border"

    inner = out[r0 + 1 : r1, c0 + 1 : c1]
    original_inner = g[r0 + 1 : r1, c0 + 1 : c1]
    inner_h, inner_w = inner.shape
    motif_mask = (original_inner != bg) & (original_inner != frame_color)
    motif_coords = np.argwhere(motif_mask)
    assert motif_coords.size > 0, "each frame must contain a motif"

    mirrored_positions: set[tuple[int, int]] = set()
    for lr, lc in motif_coords:
        mr, mc = _local_mirror(int(lr), int(lc), inner_h, inner_w)
        mirrored_positions.add((mr, mc))
        if inner[mr, mc] == bg:
            inner[mr, mc] = original_inner[int(lr), int(lc)]

    cycle: list[int] = []
    seen: set[int] = set()
    for lr in range(inner_h):
        for lc in range(inner_w):
            value = int(inner[lr, lc])
            if value != bg and value != frame_color and value not in seen:
                seen.add(value)
                cycle.append(value)

    assert 2 <= len(cycle) <= 3, "motif must use 2-3 colors"

    blank_coords = np.argwhere(inner == bg)
    for lr, lc in blank_coords:
        inner[int(lr), int(lc)] = cycle[(int(lr) + int(lc)) % len(cycle)]

    return {
        "bbox": (r0, c0, r1, c1),
        "inner_shape": (inner_h, inner_w),
        "motif_coords": motif_coords,
        "mirrored_positions": mirrored_positions,
        "cycle": cycle,
        "filled_count": int(blank_coords.shape[0]),
    }


def _expected_output(g: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    g = np.asarray(g, dtype=int)
    bg, frame_color, frames = _frame_objects(g)
    out = g.copy()
    infos = []
    for frame in frames:
        infos.append(_solve_frame(g, out, bg, frame_color, frame["bbox"]))
    return out, infos


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror each frame's motif according to the frame aspect ratio, then fill the remaining interior with a diagonal cycle from the motif colors.

    Input structure:
    - A black background contains 2-3 separated blue rectangular frames with 1-cell-thick borders.
    - Each frame encloses a small multicolor motif made of 3-6 non-blue pixels, while the rest of the interior is black.
    - Wide frames place the motif on the left side, tall frames place it on the top side, and square frames behave like wide frames.

    Rule:
    - Identify the blue frame color as the unique non-background color with the largest total area, and treat all other non-background colors as motif colors.
    - For each frame interior, mirror every motif pixel across the interior vertical axis when width >= height, otherwise across the interior horizontal axis; mirrored pixels are added only onto black cells.
    - Scan the completed motif inside that frame in reading order and record each distinct motif color at its first appearance to form a color cycle.
    - Fill every remaining black interior cell at local coordinate (r, c) with cycle[(r + c) mod K], where K is the cycle length; keep the frame border and the exterior unchanged.

    Key insight:
    - The frame aspect ratio determines which mirror to apply, and the mirrored motif simultaneously decides both which cells remain gaps and the frame-specific color order used to fill those gaps.
    """
    out, _ = _expected_output(np.asarray(grid, dtype=int))
    return out


def _orientation_from_inner(inner_h: int, inner_w: int) -> str:
    if inner_w > inner_h:
        return "wide"
    if inner_h > inner_w:
        return "tall"
    return "square"


def _frame_patch(inner_h: int, inner_w: int) -> np.ndarray:
    patch = np.zeros((inner_h + 2, inner_w + 2), dtype=int)
    patch[0, :] = FRAME_COLOR
    patch[-1, :] = FRAME_COLOR
    patch[:, 0] = FRAME_COLOR
    patch[:, -1] = FRAME_COLOR
    return patch


def _allowed_source_mask(inner_h: int, inner_w: int) -> np.ndarray:
    orient = _orientation_from_inner(inner_h, inner_w)
    mask = np.zeros((inner_h, inner_w), dtype=bool)
    if orient in {"wide", "square"}:
        max_c = (inner_w - 1) // 2
        mask[:, : max_c + 1] = True
    else:
        max_r = (inner_h - 1) // 2
        mask[: max_r + 1, :] = True
    return mask


def _axis_cells(inner_h: int, inner_w: int) -> list[tuple[int, int]]:
    orient = _orientation_from_inner(inner_h, inner_w)
    if orient in {"wide", "square"} and inner_w % 2 == 1:
        c = inner_w // 2
        return [(r, c) for r in range(inner_h)]
    if orient == "tall" and inner_h % 2 == 1:
        r = inner_h // 2
        return [(r, c) for c in range(inner_w)]
    return []


def _frontier(cells: set[tuple[int, int]], allowed: set[tuple[int, int]]) -> list[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (r + dr, c + dc)
            if nxt in allowed and nxt not in cells:
                out.add(nxt)
    return list(out)


def _generate_motif_cells(inner_h: int, inner_w: int, cell_count: int) -> set[tuple[int, int]]:
    allowed_mask = _allowed_source_mask(inner_h, inner_w)
    allowed = {tuple(map(int, rc)) for rc in np.argwhere(allowed_mask)}
    assert allowed
    axis = set(_axis_cells(inner_h, inner_w))
    off_axis = [cell for cell in allowed if cell not in axis]
    assert off_axis, "allowed source side must contain an off-axis cell"

    cells = {random.choice(off_axis)}
    want_axis = bool(axis) and random.random() < 0.45 and cell_count >= 4
    while len(cells) < cell_count:
        candidates = _frontier(cells, allowed)
        if not candidates:
            cells = {random.choice(off_axis)}
            continue
        if want_axis:
            axis_candidates = [cell for cell in candidates if cell in axis]
            if axis_candidates:
                cells.add(random.choice(axis_candidates))
                want_axis = False
                continue
        cells.add(random.choice(candidates))

    assert any(
        _local_mirror(int(r), int(c), inner_h, inner_w) != (int(r), int(c))
        for r, c in cells
    ), "motif must contain at least one pixel that creates a new mirrored cell"
    return cells


def _choose_cycle(colors: list[int]) -> list[int]:
    cycle = colors[:]
    random.shuffle(cycle)
    if cycle == sorted(cycle) and len(cycle) > 1:
        cycle = cycle[1:] + cycle[:1]
    return cycle


def _paint_motif(inner_h: int, inner_w: int) -> tuple[np.ndarray, dict]:
    count = random.randint(*MOTIF_SIZE_RANGE)
    cells = _generate_motif_cells(inner_h, inner_w, count)
    ordered_cells = sorted(cells)

    color_count = random.randint(2, min(3, count))
    palette = random.sample(MOTIF_COLORS, color_count)
    cycle = _choose_cycle(palette)

    motif = np.zeros((inner_h, inner_w), dtype=int)
    assignments: dict[tuple[int, int], int] = {}
    for idx, cell in enumerate(ordered_cells):
        if idx < color_count:
            assignments[cell] = cycle[idx]
        else:
            assignments[cell] = random.choice(cycle)

    for (r, c), color in assignments.items():
        motif[int(r), int(c)] = int(color)

    return motif, {
        "count": count,
        "colors": tuple(cycle),
        "cells": tuple(ordered_cells),
        "has_axis_cell": any(cell in set(_axis_cells(inner_h, inner_w)) for cell in cells),
    }


def _sample_frame_specs() -> list[tuple[str, int, int]]:
    frame_count = random.choice(FRAME_COUNT_CHOICES)
    specs: list[tuple[str, int, int]] = [
        ("wide", random.choice(WIDE_INNER_HEIGHTS), random.choice(WIDE_INNER_WIDTHS)),
        ("tall", random.choice(TALL_INNER_HEIGHTS), random.choice(TALL_INNER_WIDTHS)),
    ]
    if frame_count == 3:
        if random.random() < 0.7:
            side = random.choice(SQUARE_INNER_SIDES)
            specs.append(("square", side, side))
        else:
            if random.random() < 0.5:
                specs.append(("wide", random.choice(WIDE_INNER_HEIGHTS), random.choice(WIDE_INNER_WIDTHS)))
            else:
                specs.append(("tall", random.choice(TALL_INNER_HEIGHTS), random.choice(TALL_INNER_WIDTHS)))
    random.shuffle(specs)
    return specs


def _canvas_bounds(specs: list[tuple[str, int, int]]) -> tuple[int, int]:
    bbox_heights = [inner_h + 2 for _kind, inner_h, _inner_w in specs]
    bbox_widths = [inner_w + 2 for _kind, _inner_h, inner_w in specs]
    total_bbox_area = sum(h * w for h, w in zip(bbox_heights, bbox_widths))
    min_h = max(CANVAS_ROWS[0], max(bbox_heights) + 2)
    min_w = max(CANVAS_COLS[0], max(bbox_widths) + 2)
    if len(specs) == 3:
        min_h = max(min_h, 22)
        min_w = max(min_w, 22)
    if total_bbox_area > 190:
        min_h = max(min_h, 24)
        min_w = max(min_w, 24)
    return min_h, min_w


def _place_frames(specs: list[tuple[str, int, int]], shape: tuple[int, int]) -> list[tuple[str, int, int, int, int]]:
    H, W = map(int, shape)
    placement_canvas = np.zeros((H, W), dtype=int)
    placed: list[tuple[str, int, int, int, int]] = []
    for kind, inner_h, inner_w in sorted(specs, key=lambda item: (item[1] + 2) * (item[2] + 2), reverse=True):
        rect = np.ones((inner_h + 2, inner_w + 2), dtype=int)
        r0, c0 = random_free_location(
            placement_canvas,
            rect,
            transparent=0,
            padding=FRAME_GAP,
            padding_connectivity=FRAME_RECTS_CONNECTIVITY,
        )
        placement_canvas[r0 : r0 + inner_h + 2, c0 : c0 + inner_w + 2] = 1
        placed.append((kind, inner_h, inner_w, int(r0), int(c0)))
    return placed


def _build_grid(
    shape: tuple[int, int],
    placed: list[tuple[str, int, int, int, int]],
) -> tuple[np.ndarray, list[dict]]:
    H, W = map(int, shape)
    grid = np.zeros((H, W), dtype=int)
    motif_infos: list[dict] = []
    for kind, inner_h, inner_w, r0, c0 in placed:
        patch = _frame_patch(inner_h, inner_w)
        grid[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]] = patch
        motif, info = _paint_motif(inner_h, inner_w)
        grid[r0 + 1 : r0 + 1 + inner_h, c0 + 1 : c0 + 1 + inner_w] = motif
        motif_infos.append(
            {
                "kind": kind,
                "inner_shape": (inner_h, inner_w),
                "bbox": (r0, c0, r0 + inner_h + 1, c0 + inner_w + 1),
                **info,
            }
        )
    return grid, motif_infos


def _validate_generated(inp: np.ndarray, motif_infos: list[dict]) -> bool:
    if not isinstance(inp, np.ndarray) or inp.ndim != 2 or max(inp.shape) > 30:
        return False

    try:
        bg, frame_color, frames = _frame_objects(inp)
    except AssertionError:
        return False
    if bg != BG or frame_color != FRAME_COLOR:
        return False
    if len(frames) not in {2, 3}:
        return False

    frame_bboxes = {tuple(map(int, frame["bbox"])) for frame in frames}
    info_bboxes = {tuple(map(int, info["bbox"])) for info in motif_infos}
    if frame_bboxes != info_bboxes:
        return False

    out, solved_infos = _expected_output(inp)
    if np.array_equal(out, inp):
        return False

    orientations = set()
    non_sorted_cycle_seen = False
    total_new_pixels = 0

    for info, solved in zip(sorted(motif_infos, key=lambda x: x["bbox"]), solved_infos):
        bbox = tuple(map(int, info["bbox"]))
        r0, c0, r1, c1 = bbox
        original_inner = inp[r0 + 1 : r1, c0 + 1 : c1]
        output_inner = out[r0 + 1 : r1, c0 + 1 : c1]
        inner_h, inner_w = map(int, info["inner_shape"])
        orientations.add(_orientation_from_inner(inner_h, inner_w))

        motif_mask = (original_inner != BG) & (original_inner != FRAME_COLOR)
        if not (3 <= int(motif_mask.sum()) <= 6):
            return False
        motif_colors = list(dict.fromkeys(int(value) for value in original_inner[motif_mask]))
        if not (2 <= len(set(motif_colors)) <= 3):
            return False

        allowed_mask = _allowed_source_mask(inner_h, inner_w)
        if np.any(motif_mask & ~allowed_mask):
            return False

        if solved["cycle"] == sorted(solved["cycle"]):
            pass
        else:
            non_sorted_cycle_seen = True
        mirrored_new = 0
        for lr, lc in solved["motif_coords"]:
            mr, mc = _local_mirror(int(lr), int(lc), inner_h, inner_w)
            if original_inner[mr, mc] == BG:
                mirrored_new += 1
                if output_inner[mr, mc] != original_inner[int(lr), int(lc)]:
                    return False
        if mirrored_new == 0:
            return False
        total_new_pixels += mirrored_new

        if np.any(output_inner == BG):
            return False
        if solved["filled_count"] <= 0:
            return False

    if not {"wide", "tall"} <= orientations:
        return False
    if not non_sorted_cycle_seen:
        return False
    if total_new_pixels < len(frames):
        return False

    return True


def _generate_attempt() -> np.ndarray | None:
    specs = _sample_frame_specs()
    min_h, min_w = _canvas_bounds(specs)
    if min_h > CANVAS_ROWS[1] or min_w > CANVAS_COLS[1]:
        return None

    H = random.randint(min_h, CANVAS_ROWS[1])
    W = random.randint(min_w, CANVAS_COLS[1])
    try:
        placed = _place_frames(specs, (H, W))
    except ValueError:
        return None

    inp, motif_infos = _build_grid((H, W), placed)
    if not _validate_generated(inp, motif_infos):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise AssertionError("failed to generate a valid puzzle instance")
