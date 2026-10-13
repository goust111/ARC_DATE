from __future__ import annotations

import importlib.util
import random
from pathlib import Path

import numpy as np


_ARC_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
_ARC_UTILS = None
if _ARC_UTILS_PATH.exists():
    try:
        _spec = importlib.util.spec_from_file_location("_arc_utils_shared", str(_ARC_UTILS_PATH))
        if _spec is not None and _spec.loader is not None:
            _ARC_UTILS = importlib.util.module_from_spec(_spec)
            _spec.loader.exec_module(_ARC_UTILS)
    except Exception:
        _ARC_UTILS = None


if _ARC_UTILS is not None:
    Color = _ARC_UTILS.Color
    scale_sprite = _ARC_UTILS.scale_sprite
    _shared_bbox = _ARC_UTILS.bbox
    _shared_crop_bbox = _ARC_UTILS.crop_bbox
else:
    class Color:
        BLACK = 0
        BLUE = 1
        RED = 2
        GREEN = 3
        TEAL = 8

    def scale_sprite(sprite: np.ndarray, factor: int) -> np.ndarray:
        return np.kron(np.asarray(sprite, dtype=int), np.ones((factor, factor), dtype=int))

    def _shared_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
        coords = np.argwhere(mask)
        if coords.size == 0:
            return None
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        return int(r0), int(c0), int(r1), int(c1)

    def _shared_crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
        r0, c0, r1, c1 = bbox
        return np.asarray(grid, dtype=int)[r0 : r1 + 1, c0 : c1 + 1].copy()


DIRECTIONS = ("up", "down", "left", "right")
OPPOSITE = {"up": "down", "down": "up", "left": "right", "right": "left"}
DIR_STEP = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

TARGET_K_CHOICES = (2, 3)
GRID_MIN = 18
GRID_MAX = 26
CUP_SIZE_CHOICES = (5, 6, 7)
GAP_LEN_CHOICES = (2, 3)
MOTIF_GAP_CHOICES = (1, 2)
MAX_GENERATE_ATTEMPTS = 400


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    return _shared_bbox(np.asarray(mask, dtype=bool))


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    return np.asarray(_shared_crop_bbox(np.asarray(grid, dtype=int), bbox), dtype=int)


def _side_border_cells(bbox: tuple[int, int, int, int], side: str) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    if side == "top":
        return [(r0, c) for c in range(c0 + 1, c1)]
    if side == "bottom":
        return [(r1, c) for c in range(c0 + 1, c1)]
    if side == "left":
        return [(r, c0) for r in range(r0 + 1, r1)]
    return [(r, c1) for r in range(r0 + 1, r1)]


def _is_consecutive(cells: list[tuple[int, int]], side: str) -> bool:
    if not cells:
        return True
    values = sorted(c for _, c in cells) if side in ("top", "bottom") else sorted(r for r, _ in cells)
    return all(values[i] + 1 == values[i + 1] for i in range(len(values) - 1))


def _detect_opening(grid: np.ndarray, cup_bbox: tuple[int, int, int, int]) -> tuple[str, list[tuple[int, int]]]:
    candidates: list[tuple[str, list[tuple[int, int]]]] = []
    for side in ("top", "bottom", "left", "right"):
        border = _side_border_cells(cup_bbox, side)
        missing = [(r, c) for r, c in border if int(grid[r, c]) != Color.BLUE]
        if missing:
            if len(missing) not in (2, 3) or not _is_consecutive(missing, side):
                raise AssertionError("invalid cup opening")
            candidates.append((side, missing))
    assert len(candidates) == 1, "expected exactly one cup opening"
    side, gap_cells = candidates[0]
    direction = {"top": "up", "bottom": "down", "left": "left", "right": "right"}[side]
    return direction, gap_cells


def _find_motif(grid: np.ndarray) -> tuple[tuple[int, int, int, int], np.ndarray]:
    mask = np.isin(grid, [Color.RED, Color.GREEN])
    box = _bbox(mask)
    assert box is not None, "expected a red-green motif"
    r0, c0, r1, c1 = box
    assert (r1 - r0 + 1, c1 - c0 + 1) == (2, 2), "motif must be 2x2"
    motif = _crop_bbox(grid, box)
    assert np.all(np.isin(motif, [Color.RED, Color.GREEN])), "motif must use red and green only"
    assert len(set(map(int, motif.ravel()))) == 2, "motif must show both colors"
    return box, motif


def _plan_scaled_copies(
    shape: tuple[int, int], motif_bbox: tuple[int, int, int, int], direction: str
) -> list[tuple[int, int, int]]:
    rows, cols = shape
    r0, c0, r1, c1 = motif_bbox
    placements: list[tuple[int, int, int]] = []
    scale = 1

    if direction == "right":
        cursor = c0
        while True:
            size = 2 * scale
            top = r0 - scale + 1
            left = cursor
            if top < 0 or left < 0 or top + size > rows or left + size > cols:
                break
            placements.append((scale, top, left))
            cursor += size
            scale *= 2
        return placements

    if direction == "left":
        cursor = c1
        while True:
            size = 2 * scale
            top = r0 - scale + 1
            left = cursor - size + 1
            if top < 0 or left < 0 or top + size > rows or left + size > cols:
                break
            placements.append((scale, top, left))
            cursor = left - 1
            scale *= 2
        return placements

    if direction == "down":
        cursor = r0
        while True:
            size = 2 * scale
            top = cursor
            left = c0 - scale + 1
            if top < 0 or left < 0 or top + size > rows or left + size > cols:
                break
            placements.append((scale, top, left))
            cursor += size
            scale *= 2
        return placements

    cursor = r1
    while True:
        size = 2 * scale
        top = cursor - size + 1
        left = c0 - scale + 1
        if top < 0 or left < 0 or top + size > rows or left + size > cols:
            break
        placements.append((scale, top, left))
        cursor = top - 1
        scale *= 2
    return placements


def _opening_capacity(shape: tuple[int, int], cup_bbox: tuple[int, int, int, int], direction: str) -> int:
    rows, cols = shape
    r0, c0, r1, c1 = cup_bbox
    if direction == "up":
        return r0 + 1
    if direction == "down":
        return rows - r1
    if direction == "left":
        return c0 + 1
    return cols - c1


def _ray_cells(gap_cells: list[tuple[int, int]], direction: str, length: int) -> list[tuple[int, int]]:
    dr, dc = DIR_STEP[direction]
    cells: list[tuple[int, int]] = []
    for r, c in gap_cells:
        for step in range(length):
            cells.append((r + dr * step, c + dc * step))
    return cells


def _placement_cells(placements: list[tuple[int, int, int]]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for scale, top, left in placements:
        size = 2 * scale
        for r in range(top, top + size):
            for c in range(left, left + size):
                cells.add((r, c))
    return cells


def _bbox_cells(bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _parse_input(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    blue_mask = g == Color.BLUE
    cup_bbox = _bbox(blue_mask)
    assert cup_bbox is not None, "expected a blue cup outline"
    direction, gap_cells = _detect_opening(g, cup_bbox)
    motif_bbox, motif = _find_motif(g)
    placements = _plan_scaled_copies(g.shape, motif_bbox, direction)
    assert placements, "expected at least one motif copy"
    return {
        "cup_bbox": cup_bbox,
        "direction": direction,
        "gap_cells": gap_cells,
        "motif_bbox": motif_bbox,
        "motif": motif,
        "placements": placements,
    }


def _render_output(
    grid: np.ndarray,
    info: dict,
    *,
    direction: str | None = None,
    ray_length: int | None = None,
    max_copy_count: int | None = None,
) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    cup_r0, cup_c0, cup_r1, cup_c1 = info["cup_bbox"]
    out[cup_r0 + 1 : cup_r1, cup_c0 + 1 : cup_c1] = Color.TEAL

    use_direction = info["direction"] if direction is None else direction
    placements = info["placements"] if direction is None else _plan_scaled_copies(g.shape, info["motif_bbox"], use_direction)
    if max_copy_count is not None:
        placements = placements[:max_copy_count]
    for scale, top, left in placements:
        stamp = scale_sprite(info["motif"], scale)
        patch = out[top : top + stamp.shape[0], left : left + stamp.shape[1]]
        mask = patch != Color.BLUE
        patch[mask] = stamp[mask]

    length = len(placements) if ray_length is None else ray_length
    dr, dc = DIR_STEP[use_direction]
    for r0, c0 in info["gap_cells"]:
        for step in range(length):
            r = r0 + dr * step
            c = c0 + dc * step
            if not (0 <= r < out.shape[0] and 0 <= c < out.shape[1]):
                break
            if out[r, c] != Color.BLACK:
                break
            out[r, c] = Color.TEAL
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the cup, expand the nearby 2x2 motif by dyadic scaling along the cup opening, and emit a matching-length ray.

    Input structure:
    - The background is black.
    - One blue rectangular cup outline has exactly one side with a consecutive 2-3 pixel gap that defines the opening direction.
    - One nearby 2x2 motif made of red and green pixels sits on the side opposite the opening.

    Rule:
    - Detect the cup opening direction from the only border side with missing non-corner pixels.
    - Fill every interior cell of the cup's bounding box with teal.
    - Starting from the 2x2 motif, place scaled copies with factors 1, 2, 4, ... along the opening direction, keeping each copy centered on the motif's row-seam or column-seam and stopping before the next copy would leave the grid.
    - Let k be the number of placed copies; from each opening-gap pixel, draw a teal ray of length k outward along the same opening direction, stopping early if a non-background pixel blocks the path.

    Key insight:
    - The cup opening is a shared control signal: it chooses the motif expansion direction and, through the resulting copy count k, also fixes the ray length.
    """
    info = _parse_input(np.asarray(grid, dtype=int))
    return _render_output(np.asarray(grid, dtype=int), info)


def _make_motif() -> np.ndarray:
    mask = random.randint(1, 14)
    vals = []
    for bit in range(4):
        vals.append(Color.RED if (mask >> bit) & 1 else Color.GREEN)
    motif = np.array(vals, dtype=int).reshape(2, 2)
    if len(set(map(int, motif.ravel()))) != 2:
        return _make_motif()
    return motif


def _make_cup_patch(height: int, width: int, direction: str, gap_len: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    patch = np.zeros((height, width), dtype=int)
    patch[0, :] = Color.BLUE
    patch[-1, :] = Color.BLUE
    patch[:, 0] = Color.BLUE
    patch[:, -1] = Color.BLUE

    if direction in ("up", "down"):
        span = width - 2
        gap_start = 1 + (span - gap_len) // 2
        gap_cells = [(0 if direction == "up" else height - 1, c) for c in range(gap_start, gap_start + gap_len)]
    else:
        span = height - 2
        gap_start = 1 + (span - gap_len) // 2
        gap_cells = [(r, 0 if direction == "left" else width - 1) for r in range(gap_start, gap_start + gap_len)]

    for r, c in gap_cells:
        patch[r, c] = Color.BLACK
    return patch, gap_cells


def _absolute_gap_cells(cup_bbox: tuple[int, int, int, int], rel_gap_cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    r0, c0, _, _ = cup_bbox
    return [(r0 + r, c0 + c) for r, c in rel_gap_cells]


def _motif_anchor_choices(
    shape: tuple[int, int], cup_bbox: tuple[int, int, int, int], direction: str, gap_dist: int
) -> list[tuple[int, int]]:
    rows, cols = shape
    r0, c0, r1, c1 = cup_bbox
    anchors: list[tuple[int, int]] = []
    if direction == "up":
        motif_r0 = r1 + gap_dist + 1
        if 0 <= motif_r0 <= rows - 2:
            anchors.extend((motif_r0, c) for c in range(cols - 1))
    elif direction == "down":
        motif_r0 = r0 - gap_dist - 2
        if 0 <= motif_r0 <= rows - 2:
            anchors.extend((motif_r0, c) for c in range(cols - 1))
    elif direction == "left":
        motif_c0 = c1 + gap_dist + 1
        if 0 <= motif_c0 <= cols - 2:
            anchors.extend((r, motif_c0) for r in range(rows - 1))
    else:
        motif_c0 = c0 - gap_dist - 2
        if 0 <= motif_c0 <= cols - 2:
            anchors.extend((r, motif_c0) for r in range(rows - 1))
    return anchors


def _collect_candidate_layouts(
    rows: int,
    cols: int,
    direction: str,
    target_k: int,
    cup_h: int,
    cup_w: int,
    gap_len: int,
    gap_dist: int,
) -> list[dict]:
    cup_patch, rel_gap_cells = _make_cup_patch(cup_h, cup_w, direction, gap_len)
    candidates: list[dict] = []
    for cup_r0 in range(rows - cup_h + 1):
        for cup_c0 in range(cols - cup_w + 1):
            cup_bbox = (cup_r0, cup_c0, cup_r0 + cup_h - 1, cup_c0 + cup_w - 1)
            if _opening_capacity((rows, cols), cup_bbox, direction) < target_k + 1:
                continue

            cup_cells = _bbox_cells(cup_bbox)
            gap_cells = _absolute_gap_cells(cup_bbox, rel_gap_cells)
            ray_cells = set(_ray_cells(gap_cells, direction, target_k))

            for motif_r0, motif_c0 in _motif_anchor_choices((rows, cols), cup_bbox, direction, gap_dist):
                motif_bbox = (motif_r0, motif_c0, motif_r0 + 1, motif_c0 + 1)
                placements = _plan_scaled_copies((rows, cols), motif_bbox, direction)
                if len(placements) != target_k:
                    continue
                copy_cells = _placement_cells(placements)
                if copy_cells & cup_cells:
                    continue
                if copy_cells & ray_cells:
                    continue
                candidates.append(
                    {
                        "cup_bbox": cup_bbox,
                        "cup_patch": cup_patch,
                        "gap_cells": gap_cells,
                        "motif_bbox": motif_bbox,
                        "placements": placements,
                    }
                )
    return candidates


def _semantic_checks(inp: np.ndarray) -> bool:
    info = _parse_input(inp)
    out = _render_output(inp, info)
    k = len(info["placements"])
    if k < 2:
        return False
    if _opening_capacity(inp.shape, info["cup_bbox"], info["direction"]) <= k:
        return False
    for other in DIRECTIONS:
        if other == info["direction"]:
            continue
        if np.array_equal(_render_output(inp, info, direction=other), out):
            return False
    if np.array_equal(_render_output(inp, info, ray_length=k - 1), out):
        return False
    if np.array_equal(_render_output(inp, info, max_copy_count=k - 1, ray_length=k - 1), out):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    rows = random.randint(GRID_MIN, GRID_MAX)
    cols = random.randint(GRID_MIN, GRID_MAX)
    direction = random.choice(DIRECTIONS)
    target_k = random.choice(TARGET_K_CHOICES)
    cup_h = random.choice(CUP_SIZE_CHOICES)
    cup_w = random.choice(CUP_SIZE_CHOICES)
    gap_len = random.choice([g for g in GAP_LEN_CHOICES if g <= (cup_w - 2 if direction in ("up", "down") else cup_h - 2)])
    gap_dist = random.choice(MOTIF_GAP_CHOICES)

    layouts = _collect_candidate_layouts(rows, cols, direction, target_k, cup_h, cup_w, gap_len, gap_dist)
    if not layouts:
        return None

    layout = random.choice(layouts)
    motif = _make_motif()
    inp = np.zeros((rows, cols), dtype=int)

    cup_r0, cup_c0, cup_r1, cup_c1 = layout["cup_bbox"]
    inp[cup_r0 : cup_r1 + 1, cup_c0 : cup_c1 + 1] = layout["cup_patch"]
    motif_r0, motif_c0, _, _ = layout["motif_bbox"]
    inp[motif_r0 : motif_r0 + 2, motif_c0 : motif_c0 + 2] = motif

    info = _parse_input(inp)
    if len(info["placements"]) != target_k:
        return None
    if not _semantic_checks(inp):
        return None
    if not np.array_equal(main(inp), _render_output(inp, info)):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATE_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid cup-scaling puzzle input")
