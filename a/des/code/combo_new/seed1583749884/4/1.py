from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
DIR_STEP = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _classify_component(g: np.ndarray, coords: list[tuple[int, int]]) -> tuple[str, dict] | None:
    r0, c0, r1, c1 = _bbox(coords)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    values = [int(g[r, c]) for r, c in coords]
    colors = set(values)

    if h == 2 and w == 2 and len(coords) == 4 and len(colors) == 1:
        color = values[0]
        if color not in (BLACK, GREY):
            return "anchor", {
                "color": int(color),
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "cells": set(coords),
            }

    if h < 3 or w < 3:
        return None

    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    comp_mask = np.zeros((h, w), dtype=bool)
    for r, c in coords:
        comp_mask[r - r0, c - c0] = True
    if len(coords) != int(border.sum()) or not np.array_equal(comp_mask, border):
        return None

    sub = g[r0 : r1 + 1, c0 : c1 + 1]
    if bool(np.any(sub[~border] != BLACK)):
        return None

    marker_cells = [(r, c) for r, c in coords if int(g[r, c]) != GREY]
    if len(marker_cells) != 1:
        return None
    marker = marker_cells[0]
    marker_color = int(g[marker])
    if marker_color in (BLACK, GREY):
        return None
    if any(int(g[r, c]) != GREY for r, c in coords if (r, c) != marker):
        return None

    side: list[str] = []
    if marker[0] == r0:
        side.append("top")
    if marker[0] == r1:
        side.append("bottom")
    if marker[1] == c0:
        side.append("left")
    if marker[1] == c1:
        side.append("right")
    if len(side) != 1:
        return None

    return "frame", {
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "marker": (int(marker[0]), int(marker[1])),
        "side": side[0],
        "color": int(marker_color),
    }


def _extract_scene(g: np.ndarray) -> tuple[list[dict], dict[tuple[int, int], dict]]:
    frames: list[dict] = []
    cell_to_anchor: dict[tuple[int, int], dict] = {}

    for coords in _components(g != BLACK):
        classified = _classify_component(g, coords)
        assert classified is not None, "unexpected foreground component"
        kind, info = classified
        if kind == "frame":
            frames.append(info)
        else:
            for cell in info["cells"]:
                cell_to_anchor[cell] = info

    assert frames, "expected at least one frame"
    return frames, cell_to_anchor


def _find_anchor(frame: dict, cell_to_anchor: dict[tuple[int, int], dict], shape: tuple[int, int]) -> dict:
    h, w = map(int, shape)
    dr, dc = DIR_STEP[frame["side"]]
    r, c = frame["marker"]
    color = int(frame["color"])

    r += dr
    c += dc
    while 0 <= r < h and 0 <= c < w:
        anchor = cell_to_anchor.get((int(r), int(c)))
        if anchor is not None and int(anchor["color"]) == color:
            return anchor
        r += dr
        c += dc

    raise AssertionError("expected a matching anchor in the marker direction")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each hollow grey frame using the first same-color anchor found in the marker's outward direction.

    Input structure:
    - The background is black.
    - Several one-pixel-thick hollow grey rectangular frames appear in the grid.
    - Each frame has exactly one non-grey, non-black marker pixel embedded on one side of its border.
    - Separate 2x2 solid anchor blocks sit outside the frames.

    Rule:
    - For each frame, identify which side contains the marker; that side defines the outward search direction.
    - Step away from the marker along that row or column until the first 2x2 anchor whose color matches the marker is encountered.
    - Fill every black interior cell of the frame with that anchor color.
    - Leave the grey borders, marker pixels, and anchor blocks unchanged.

    Key insight:
    The marker encodes two things at once: its side gives the search direction, and its color selects the matching anchor.
    """
    g = np.asarray(grid, dtype=int)
    frames, cell_to_anchor = _extract_scene(g)

    out = g.copy()
    for frame in frames:
        anchor = _find_anchor(frame, cell_to_anchor, g.shape)
        r0, c0, r1, c1 = frame["bbox"]
        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        interior[interior == BLACK] = int(anchor["color"])

    return out


# --------------- generator ---------------

GEN_HEIGHT_RANGE = (18, 31)
GEN_WIDTH_RANGE = (18, 31)
GEN_FRAME_COUNT_RANGE = (2, 5)
GEN_FRAME_HEIGHT_RANGE = (4, 10)
GEN_FRAME_WIDTH_RANGE = (4, 10)
GEN_SEPARATION_RANGE = (1, 7)
GEN_DISTRACTOR_RANGE = (0, 4)
GEN_CROP_MARGIN = 2
GEN_PAIR_TRIES = 120
GEN_ANCHOR_TRIES = 120
GEN_ATTEMPTS = 500


def _expanded_bbox(
    r0: int, c0: int, r1: int, c1: int, shape: tuple[int, int], pad: int = 1
) -> tuple[int, int, int, int]:
    h, w = map(int, shape)
    return (
        max(0, int(r0 - pad)),
        max(0, int(c0 - pad)),
        min(h - 1, int(r1 + pad)),
        min(w - 1, int(c1 + pad)),
    )


def _direction_list(count: int) -> list[str]:
    dirs: list[str] = []
    while len(dirs) < count:
        batch = list(DIR_STEP)
        random.shuffle(batch)
        dirs.extend(batch)
    return dirs[:count]


def _pair_geometry(
    r0: int, c0: int, h: int, w: int, side: str, marker_offset: int, sep: int, shift: int
) -> tuple[tuple[int, int], tuple[int, int]]:
    r1 = int(r0 + h - 1)
    c1 = int(c0 + w - 1)
    if side == "top":
        marker = (int(r0), int(c0 + marker_offset))
        anchor = (int(r0 - sep - 2), int(marker[1] + shift))
    elif side == "bottom":
        marker = (int(r1), int(c0 + marker_offset))
        anchor = (int(r1 + sep + 1), int(marker[1] + shift))
    elif side == "left":
        marker = (int(r0 + marker_offset), int(c0))
        anchor = (int(marker[0] + shift), int(c0 - sep - 2))
    else:
        marker = (int(r0 + marker_offset), int(c1))
        anchor = (int(marker[0] + shift), int(c1 + sep + 1))
    return marker, anchor


def _place_pair(
    grid: np.ndarray, reserved: np.ndarray, color: int, side: str
) -> tuple[tuple[int, int, int, int], int] | None:
    h_grid, w_grid = grid.shape

    for _ in range(GEN_PAIR_TRIES):
        h = random.randint(GEN_FRAME_HEIGHT_RANGE[0], GEN_FRAME_HEIGHT_RANGE[1] - 1)
        w = random.randint(GEN_FRAME_WIDTH_RANGE[0], GEN_FRAME_WIDTH_RANGE[1] - 1)
        sep = random.randint(GEN_SEPARATION_RANGE[0], GEN_SEPARATION_RANGE[1] - 1)

        if side in ("top", "bottom"):
            marker_offset = random.randint(1, w - 2)
            valid_shifts: list[int] = []
            for shift in (-1, 0):
                c_lo = max(0, -marker_offset - shift)
                c_hi = min(w_grid - w, w_grid - 2 - marker_offset - shift)
                if c_lo <= c_hi:
                    valid_shifts.append(shift)
            if not valid_shifts:
                continue
            shift = random.choice(valid_shifts)
            c_lo = max(0, -marker_offset - shift)
            c_hi = min(w_grid - w, w_grid - 2 - marker_offset - shift)
            if side == "top":
                r_lo = sep + 2
                r_hi = h_grid - h
            else:
                r_lo = 0
                r_hi = h_grid - h - sep - 2
        else:
            marker_offset = random.randint(1, h - 2)
            valid_shifts = []
            for shift in (-1, 0):
                r_lo = max(0, -marker_offset - shift)
                r_hi = min(h_grid - h, h_grid - 2 - marker_offset - shift)
                if r_lo <= r_hi:
                    valid_shifts.append(shift)
            if not valid_shifts:
                continue
            shift = random.choice(valid_shifts)
            r_lo = max(0, -marker_offset - shift)
            r_hi = min(h_grid - h, h_grid - 2 - marker_offset - shift)
            if side == "left":
                c_lo = sep + 2
                c_hi = w_grid - w
            else:
                c_lo = 0
                c_hi = w_grid - w - sep - 2

        if r_lo > r_hi or c_lo > c_hi:
            continue

        r0 = random.randint(int(r_lo), int(r_hi))
        c0 = random.randint(int(c_lo), int(c_hi))
        marker, anchor = _pair_geometry(r0, c0, h, w, side, marker_offset, sep, shift)
        ar0, ac0 = anchor
        r1 = int(r0 + h - 1)
        c1 = int(c0 + w - 1)

        if not (0 <= ar0 <= h_grid - 2 and 0 <= ac0 <= w_grid - 2):
            continue

        pair_r0 = min(r0, ar0)
        pair_c0 = min(c0, ac0)
        pair_r1 = max(r1, ar0 + 1)
        pair_c1 = max(c1, ac0 + 1)
        ex_r0, ex_c0, ex_r1, ex_c1 = _expanded_bbox(pair_r0, pair_c0, pair_r1, pair_c1, grid.shape, pad=1)
        if bool(reserved[ex_r0 : ex_r1 + 1, ex_c0 : ex_c1 + 1].any()):
            continue

        grid[r0, c0 : c1 + 1] = GREY
        grid[r1, c0 : c1 + 1] = GREY
        grid[r0 : r1 + 1, c0] = GREY
        grid[r0 : r1 + 1, c1] = GREY
        grid[marker] = int(color)
        grid[ar0 : ar0 + 2, ac0 : ac0 + 2] = int(color)
        reserved[ex_r0 : ex_r1 + 1, ex_c0 : ex_c1 + 1] = True
        return (int(r0), int(c0), int(r1), int(c1)), int(color)

    return None


def _place_distractor_anchor(grid: np.ndarray, reserved: np.ndarray, color: int) -> bool:
    h, w = grid.shape
    for _ in range(GEN_ANCHOR_TRIES):
        r0 = random.randint(0, h - 2)
        c0 = random.randint(0, w - 2)
        ex_r0, ex_c0, ex_r1, ex_c1 = _expanded_bbox(r0, c0, r0 + 1, c0 + 1, grid.shape, pad=1)
        if bool(reserved[ex_r0 : ex_r1 + 1, ex_c0 : ex_c1 + 1].any()):
            continue
        grid[r0 : r0 + 2, c0 : c0 + 2] = int(color)
        reserved[ex_r0 : ex_r1 + 1, ex_c0 : ex_c1 + 1] = True
        return True
    return False


def _crop(grid: np.ndarray, expected: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coords = np.argwhere(grid != BLACK)
    r0, c0 = map(int, coords.min(axis=0))
    r1, c1 = map(int, coords.max(axis=0))
    h, w = grid.shape

    top = random.randint(0, min(GEN_CROP_MARGIN, r0))
    left = random.randint(0, min(GEN_CROP_MARGIN, c0))
    bottom = random.randint(0, min(GEN_CROP_MARGIN, h - 1 - r1))
    right = random.randint(0, min(GEN_CROP_MARGIN, w - 1 - c1))

    rr0 = int(r0 - top)
    cc0 = int(c0 - left)
    rr1 = int(r1 + bottom)
    cc1 = int(c1 + right)
    return grid[rr0 : rr1 + 1, cc0 : cc1 + 1].copy(), expected[rr0 : rr1 + 1, cc0 : cc1 + 1].copy()


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GEN_HEIGHT_RANGE[0], GEN_HEIGHT_RANGE[1] - 1)
    w = random.randint(GEN_WIDTH_RANGE[0], GEN_WIDTH_RANGE[1] - 1)
    grid = np.full((h, w), BLACK, dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    n_frames = random.randint(GEN_FRAME_COUNT_RANGE[0], GEN_FRAME_COUNT_RANGE[1] - 1)
    colors = random.sample(PALETTE, n_frames)
    directions = _direction_list(n_frames)
    records: list[tuple[tuple[int, int, int, int], int]] = []

    for color, side in zip(colors, directions, strict=False):
        placed = _place_pair(grid, reserved, int(color), side)
        if placed is None:
            return None
        records.append(placed)

    remaining = [color for color in PALETTE if color not in colors]
    random.shuffle(remaining)
    n_distractors = min(random.randint(GEN_DISTRACTOR_RANGE[0], GEN_DISTRACTOR_RANGE[1] - 1), len(remaining))
    for color in remaining[:n_distractors]:
        _place_distractor_anchor(grid, reserved, int(color))

    expected = grid.copy()
    for (r0, c0, r1, c1), color in records:
        expected[r0 + 1 : r1, c0 + 1 : c1] = int(color)

    cropped_grid, cropped_expected = _crop(grid, expected)
    if max(cropped_grid.shape) > 30 or max(cropped_expected.shape) > 30:
        return None

    try:
        if np.array_equal(main(cropped_grid), cropped_expected):
            return cropped_grid
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid sample after {GEN_ATTEMPTS} attempts")
