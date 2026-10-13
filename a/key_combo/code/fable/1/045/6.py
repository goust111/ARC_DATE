from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, crop_bbox, find_components, get_bg, in_bounds, paste


PALETTE = np.array([Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON], dtype=int)
SIZE_OPTIONS = ((4, 4), (4, 5), (5, 4), (5, 5))
GENERATION_RETRIES = 400


@dataclass
class FrameSpec:
    color: int
    bbox: tuple[int, int, int, int]
    gap: tuple[int, int]
    direction: tuple[int, int]
    stamp: np.ndarray
    local_cells: tuple[tuple[int, int], ...]
    copies: int


@dataclass
class Plan:
    frame: FrameSpec
    top_lefts: list[tuple[int, int]]
    stopped_by_collision: bool


def _perimeter_coords(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    pts: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        pts.append((r0, c))
        pts.append((r1, c))
    for r in range(r0 + 1, r1):
        pts.append((r, c0))
        pts.append((r, c1))
    return pts


def _gap_direction(gap: tuple[int, int], bbox: tuple[int, int, int, int]) -> tuple[int, int] | None:
    r, c = gap
    r0, c0, r1, c1 = bbox
    options: list[tuple[int, int]] = []
    if r == r0:
        options.append((-1, 0))
    if r == r1:
        options.append((1, 0))
    if c == c0:
        options.append((0, -1))
    if c == c1:
        options.append((0, 1))
    if len(options) != 1:
        return None
    return options[0]


def _frame_from_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int, bg: int) -> FrameSpec | None:
    r0, c0, r1, c1 = bbox
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None

    stamp = crop_bbox(grid, bbox)
    if not np.all((stamp == bg) | (stamp == color)):
        return None

    perimeter = _perimeter_coords(bbox)
    missing = [(r, c) for r, c in perimeter if int(grid[r, c]) == bg]
    if len(missing) != 1:
        return None
    if any(int(grid[r, c]) != color for r, c in perimeter if (r, c) not in missing):
        return None

    direction = _gap_direction(missing[0], bbox)
    if direction is None:
        return None

    interior = stamp[1:-1, 1:-1]
    if interior.size == 0:
        return None
    copies = int(np.sum(interior == bg))
    if copies <= 0:
        return None

    local_cells = tuple((int(r), int(c)) for r, c in np.argwhere(stamp != bg))
    return FrameSpec(
        color=int(color),
        bbox=(int(r0), int(c0), int(r1), int(c1)),
        gap=(int(missing[0][0]), int(missing[0][1])),
        direction=(int(direction[0]), int(direction[1])),
        stamp=stamp,
        local_cells=local_cells,
        copies=int(copies),
    )


def _extract_frames(grid: np.ndarray) -> tuple[int, list[FrameSpec]]:
    bg = int(get_bg(grid))
    frames_by_key: dict[tuple[int, tuple[int, int, int, int]], FrameSpec] = {}
    for comp in find_components(grid, bg=bg, connectivity=4, monochromatic=True, return_format="dict"):
        color = int(comp["color"])
        bbox = tuple(map(int, comp["bbox"]))
        key = (color, bbox)
        if key in frames_by_key:
            continue
        frame = _frame_from_bbox(grid, bbox, color, bg)
        if frame is not None:
            frames_by_key[key] = frame

    frames = sorted(frames_by_key.values(), key=lambda fr: fr.bbox)
    assert frames, "expected at least one gapped hollow rectangle"

    covered = np.zeros_like(grid, dtype=bool)
    for frame in frames:
        r0, c0, r1, c1 = frame.bbox
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        assert np.all((sub == bg) | (sub == frame.color)), "frame bbox must contain only its own color or background"
        covered[r0 : r1 + 1, c0 : c1 + 1] |= (sub == frame.color)
    assert np.array_equal(covered, grid != bg), "all colored pixels must belong to some frame bbox"

    return bg, frames


def _translation_step(frame: FrameSpec) -> tuple[int, int]:
    dr, dc = frame.direction
    h, w = frame.stamp.shape
    return int(dr * (h + 1)), int(dc * (w + 1))


def _visible_cells(frame: FrameSpec, top_left: tuple[int, int], shape: tuple[int, int]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    r0, c0 = top_left
    for rr, cc in frame.local_cells:
        r = r0 + rr
        c = c0 + cc
        if in_bounds(shape, r, c):
            out.append((int(r), int(c)))
    return out


def _would_overlap(frame: FrameSpec, top_left: tuple[int, int], shape: tuple[int, int], occupied: np.ndarray) -> bool:
    for r, c in _visible_cells(frame, top_left, shape):
        if occupied[r, c]:
            return True
    return False


def _mark_occupied(frame: FrameSpec, top_left: tuple[int, int], shape: tuple[int, int], occupied: np.ndarray) -> None:
    for r, c in _visible_cells(frame, top_left, shape):
        occupied[r, c] = True


def _plan_copies(grid: np.ndarray, frames: list[FrameSpec], bg: int) -> list[Plan]:
    occupied_orig = grid != bg
    plans: list[Plan] = []
    for frame in frames:
        step_r, step_c = _translation_step(frame)
        r0, c0, _r1, _c1 = frame.bbox
        occupied = occupied_orig.copy()
        top_lefts: list[tuple[int, int]] = []
        stopped = False
        for k in range(1, frame.copies + 1):
            top_left = (int(r0 + k * step_r), int(c0 + k * step_c))
            if _would_overlap(frame, top_left, grid.shape, occupied):
                stopped = True
                break
            top_lefts.append(top_left)
            _mark_occupied(frame, top_left, grid.shape, occupied)
        plans.append(Plan(frame=frame, top_lefts=top_lefts, stopped_by_collision=stopped))
    return plans


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp each gapped hollow rectangle outward, using its remaining cavity black cells as the copy count.

    Input structure:
    - Black background with several non-touching monochrome hollow rectangles.
    - Each rectangle has exactly one missing border pixel on one side, and its cavity may contain same-color solid cells.
    - Each object's full stamp is the rectangle border, its single gap, and its cavity solids inside one bounding box.

    Rule:
    - For each rectangle, count the black cells still remaining in its interior cavity.
    - Copy that whole bbox pattern outward along the gap side's normal direction, spacing consecutive copies by one blank cell.
    - Stop a rectangle's sequence as soon as the next copy would overlap another original rectangle or one of its own earlier copies.
    - Clip any out-of-bounds part of an accepted copy, and keep the original rectangles unchanged.

    Key insight:
    - The same object provides all three hidden parameters at once: the gap chooses direction, the cavity black-cell count chooses how many copies to attempt, and the whole bbox is the stamp that gets repeated.
    """
    g = np.asarray(grid, dtype=int)
    bg, frames = _extract_frames(g)
    plans = _plan_copies(g, frames, bg)

    out = g.copy()
    occupied = g != bg
    for plan in plans:
        for top_left in plan.top_lefts:
            assert not _would_overlap(plan.frame, top_left, g.shape, occupied), "copy overlaps another planned mark"
            out = paste(out, plan.frame.stamp, top_left[0], top_left[1], transparent=bg, clip=True)
            _mark_occupied(plan.frame, top_left, g.shape, occupied)
    return out


def _build_patch(
    rng: np.random.Generator,
    h: int,
    w: int,
    color: int,
    copies: int,
    gap_side: int,
) -> np.ndarray:
    patch = np.full((h, w), Color.BLACK, dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color

    if gap_side == 0:
        patch[0, int(rng.integers(1, w - 1))] = Color.BLACK
    elif gap_side == 1:
        patch[-1, int(rng.integers(1, w - 1))] = Color.BLACK
    elif gap_side == 2:
        patch[int(rng.integers(1, h - 1)), 0] = Color.BLACK
    else:
        patch[int(rng.integers(1, h - 1)), -1] = Color.BLACK

    interior = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
    fill_count = len(interior) - int(copies)
    if fill_count > 0:
        chosen = rng.choice(len(interior), size=fill_count, replace=False)
        for idx in np.atleast_1d(chosen):
            r, c = interior[int(idx)]
            patch[r, c] = color
    return patch


def _bbox_from_top_left(top_left: tuple[int, int], patch: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = top_left
    h, w = patch.shape
    return int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1)


def _touch_or_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    rows_separated = ar1 + 1 < br0 or br1 + 1 < ar0
    cols_separated = ac1 + 1 < bc0 or bc1 + 1 < ac0
    return not (rows_separated or cols_separated)


def _place_random_bbox(
    rng: np.random.Generator,
    shape: tuple[int, int],
    patch: np.ndarray,
    existing: list[tuple[int, int, int, int]],
    tries: int = 120,
) -> tuple[int, int] | None:
    h, w = patch.shape
    H, W = shape
    for _ in range(tries):
        r0 = int(rng.integers(0, H - h + 1))
        c0 = int(rng.integers(0, W - w + 1))
        bbox = _bbox_from_top_left((r0, c0), patch)
        if any(_touch_or_overlap(bbox, other) for other in existing):
            continue
        return (r0, c0)
    return None


def _collision_anchor(
    rng: np.random.Generator,
    shape: tuple[int, int],
    patch: np.ndarray,
    gap_side: int,
    blocked_step: int,
) -> tuple[int, int] | None:
    h, w = patch.shape
    H, W = shape
    if gap_side == 0:
        r_lo, r_hi = blocked_step * (h + 1), H - h
        c_lo, c_hi = 0, W - w
    elif gap_side == 1:
        r_lo, r_hi = 0, H - h - blocked_step * (h + 1)
        c_lo, c_hi = 0, W - w
    elif gap_side == 2:
        r_lo, r_hi = 0, H - h
        c_lo, c_hi = blocked_step * (w + 1), W - w
    else:
        r_lo, r_hi = 0, H - h
        c_lo, c_hi = 0, W - w - blocked_step * (w + 1)
    if r_lo > r_hi or c_lo > c_hi:
        return None
    return (int(rng.integers(r_lo, r_hi + 1)), int(rng.integers(c_lo, c_hi + 1)))


def _copy_stats(frame: FrameSpec, top_lefts: list[tuple[int, int]], shape: tuple[int, int]) -> tuple[int, int]:
    visible = 0
    total_cells = 0
    for top_left in top_lefts:
        cells = _visible_cells(frame, top_left, shape)
        if cells:
            visible += 1
            total_cells += len(cells)
    return visible, total_cells


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    H = int(rng.integers(18, 31))
    W = int(rng.integers(18, 31))
    n_objects = int(rng.integers(3, 5))
    colors = [int(c) for c in rng.choice(PALETTE, size=n_objects, replace=False)]

    h, w = SIZE_OPTIONS[int(rng.integers(len(SIZE_OPTIONS)))]
    interior_area = (h - 2) * (w - 2)
    min_copies = max(2, (interior_area + 1) // 2)
    max_copies = min(5, interior_area)
    copies_a = int(rng.integers(min_copies, max_copies + 1))
    copies_b_choices = [k for k in range(min_copies, max_copies + 1) if k != copies_a]
    copies_b = int(rng.choice(copies_b_choices if copies_b_choices else [copies_a]))
    gap_a = int(rng.integers(4))
    patch_a = _build_patch(rng, h, w, colors[0], copies_a, gap_a)
    patch_b = _build_patch(rng, h, w, colors[1], copies_b, int(rng.integers(4)))
    blocked_step = int(rng.integers(2, min(copies_a, 3) + 1))
    anchor_a = _collision_anchor(rng, (H, W), patch_a, gap_a, blocked_step)
    if anchor_a is None:
        return None

    step_r = (-1 if gap_a == 0 else 1 if gap_a == 1 else 0) * (h + 1)
    step_c = (-1 if gap_a == 2 else 1 if gap_a == 3 else 0) * (w + 1)
    anchor_b = (anchor_a[0] + blocked_step * step_r, anchor_a[1] + blocked_step * step_c)
    bbox_a = _bbox_from_top_left(anchor_a, patch_a)
    bbox_b = _bbox_from_top_left(anchor_b, patch_b)
    if _touch_or_overlap(bbox_a, bbox_b):
        return None

    patches = [patch_a, patch_b]
    anchors = [anchor_a, anchor_b]
    boxes = [bbox_a, bbox_b]

    for i in range(2, n_objects):
        hh, ww = SIZE_OPTIONS[int(rng.integers(len(SIZE_OPTIONS)))]
        area = (hh - 2) * (ww - 2)
        copies = int(rng.integers(max(1, (area + 1) // 2), min(5, area) + 1))
        patch = _build_patch(rng, hh, ww, colors[i], copies, int(rng.integers(4)))
        anchor = _place_random_bbox(rng, (H, W), patch, boxes)
        if anchor is None:
            return None
        patches.append(patch)
        anchors.append(anchor)
        boxes.append(_bbox_from_top_left(anchor, patch))

    grid = np.full((H, W), Color.BLACK, dtype=int)
    for patch, (r0, c0) in zip(patches, anchors):
        grid = paste(grid, patch, r0, c0, transparent=Color.BLACK, clip=False)

    bg, frames = _extract_frames(grid)
    if len(frames) != n_objects:
        return None

    plans = _plan_copies(grid, frames, bg)
    occupied = grid != bg
    collision_seen = False
    free_seen = False
    for plan in plans:
        visible_copies, visible_cells = _copy_stats(plan.frame, plan.top_lefts, grid.shape)
        if plan.stopped_by_collision and visible_copies >= 1:
            collision_seen = True
        if not plan.stopped_by_collision and visible_copies >= 1:
            free_seen = True
        for top_left in plan.top_lefts:
            if _would_overlap(plan.frame, top_left, grid.shape, occupied):
                return None
            _mark_occupied(plan.frame, top_left, grid.shape, occupied)

    if not collision_seen or not free_seen:
        return None
    if int(np.count_nonzero(main(grid))) <= int(np.count_nonzero(grid)):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid stamping input")
