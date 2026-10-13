from __future__ import annotations

import random

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

ARM_ORDER = ("top", "bottom", "left", "right")
CORNERS = ("top_left", "top_right", "bottom_left", "bottom_right")

TRI_L = np.array(
    [
        [1, 0],
        [1, 1],
    ],
    dtype=bool,
)
TET_L = np.array(
    [
        [1, 0],
        [1, 0],
        [1, 1],
    ],
    dtype=bool,
)

MAX_GENERATION_TRIES = 300


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _mask_components(mask: np.ndarray, connectivity: int = 4) -> list[list[tuple[int, int]]]:
    dirs = DIR4 if int(connectivity) == 4 else [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in dirs:
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)

    return comps


def _color_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if bool(seen[r, c]) or int(grid[r, c]) == int(bg):
                continue
            color = int(grid[r, c])
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not bool(seen[nr, nc])
                        and int(grid[nr, nc]) == int(color)
                    ):
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            r0, c0, r1, c1 = _bbox(coords)
            comps.append(
                {
                    "color": int(color),
                    "area": int(len(coords)),
                    "coords": coords,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                }
            )

    return comps


def _component_mask(comp: dict) -> np.ndarray:
    r0, c0, r1, c1 = comp["bbox"]
    mask = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=bool)
    for r, c in comp["coords"]:
        mask[int(r - r0), int(c - c0)] = True
    return mask


def _paint_mask(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    h, w = mask.shape
    patch = grid[int(r0) : int(r0) + h, int(c0) : int(c0) + w]
    patch[mask] = int(color)


def _clear_coords(grid: np.ndarray, coords: list[tuple[int, int]], bg: int) -> None:
    for r, c in coords:
        grid[int(r), int(c)] = int(bg)


def _marker_direction(marker: tuple[int, int], center_rows: np.ndarray, center_cols: np.ndarray) -> str:
    mr, mc = map(int, marker)
    if mr < int(center_rows[0]):
        assert int(center_cols[0]) <= mc <= int(center_cols[-1])
        return "top"
    if mr > int(center_rows[-1]):
        assert int(center_cols[0]) <= mc <= int(center_cols[-1])
        return "bottom"
    if mc < int(center_cols[0]):
        assert int(center_rows[0]) <= mr <= int(center_rows[-1])
        return "left"
    assert mc > int(center_cols[-1]) and int(center_rows[0]) <= mr <= int(center_rows[-1])
    return "right"


def _candidate_bbox(
    direction: str,
    marker: tuple[int, int],
    center_rows: np.ndarray,
    center_cols: np.ndarray,
    shape: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    mr, mc = map(int, marker)
    h, w = map(int, shape)
    if direction in {"top", "bottom"}:
        if w != int(len(center_cols)):
            return None
        c0 = int(center_cols[0])
        if direction == "top":
            return int(mr - h), c0, int(mr - 1), int(c0 + w - 1)
        return int(mr + 1), c0, int(mr + h), int(c0 + w - 1)

    if h != int(len(center_rows)):
        return None
    r0 = int(center_rows[0])
    if direction == "left":
        return r0, int(mc - w), int(r0 + h - 1), int(mc - 1)
    return r0, int(mc + 1), int(r0 + h - 1), int(mc + w)


def _slot_fit(
    grid: np.ndarray,
    skeleton_color: int,
    mask: np.ndarray,
    bbox: tuple[int, int, int, int] | None,
) -> bool:
    if bbox is None:
        return False
    r0, c0, r1, c1 = map(int, bbox)
    h, w = grid.shape
    if r0 < 0 or c0 < 0 or r1 >= h or c1 >= w:
        return False
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    if sub.shape != mask.shape:
        return False
    return bool(np.all(sub[mask] != skeleton_color) and np.all(sub[mask] == _get_bg(grid)) and np.all(sub[~mask] == int(skeleton_color)))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each scattered L fragment and dock it into the matching arm-end notch of the central cross.

    Input structure:
    - The background is the dominant color.
    - A large central 4-connected object consists of a single-color cross skeleton plus four colored singleton marker pixels on its arms.
    - Four small monochrome L-shaped pieces of area 3 or 4 sit away from the center, and each piece color matches exactly one arm marker color.

    Rule:
    - Pair every L piece with the unique singleton marker of the same color.
    - Use the marker position to determine which arm the piece belongs to, then test the four rotations of that piece in the arm-end rectangular tip window immediately beyond the marker.
    - The valid rotation is the one whose cells exactly cover the background notch while the rest of that tip window stays skeleton-colored; erase the original corner piece and the marker pixel, and place the rotated piece into the notch.

    Key insight:
    - Color gives the piece-to-arm assignment, but color alone is insufficient: the arm-end background pattern around that marker uniquely determines which rotation of the L fragment completes the tip.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)

    mono = _color_components(g, bg)
    assert mono, "expected foreground components"

    skeleton = max(mono, key=lambda comp: int(comp["area"]))
    skeleton_color = int(skeleton["color"])

    central_comps = _mask_components(g != int(bg), connectivity=4)
    central_coords = max(central_comps, key=len)
    central_mask = np.zeros_like(g, dtype=bool)
    for r, c in central_coords:
        central_mask[int(r), int(c)] = True

    row_counts = central_mask.sum(axis=1)
    col_counts = central_mask.sum(axis=0)
    center_rows = np.where(row_counts == int(row_counts.max()))[0]
    center_cols = np.where(col_counts == int(col_counts.max()))[0]
    assert len(center_rows) == 2 and np.all(np.diff(center_rows) == 1)
    assert len(center_cols) == 2 and np.all(np.diff(center_cols) == 1)

    piece_comps = [comp for comp in mono if comp is not skeleton and int(comp["area"]) in {3, 4}]
    marker_comps = [comp for comp in mono if comp is not skeleton and int(comp["area"]) == 1]
    assert len(piece_comps) == 4 and len(marker_comps) == 4

    pieces_by_color = {int(comp["color"]): comp for comp in piece_comps}
    markers_by_color = {int(comp["color"]): comp for comp in marker_comps}
    assert set(pieces_by_color) == set(markers_by_color) and len(pieces_by_color) == 4

    out = g.copy()
    for comp in piece_comps + marker_comps:
        _clear_coords(out, comp["coords"], bg)

    for color, piece in pieces_by_color.items():
        marker_comp = markers_by_color[int(color)]
        marker = marker_comp["coords"][0]
        direction = _marker_direction(marker, center_rows, center_cols)
        piece_mask = _component_mask(piece)

        matches: list[tuple[np.ndarray, tuple[int, int, int, int]]] = []
        for k in range(4):
            rotated = np.rot90(piece_mask, k)
            bbox = _candidate_bbox(direction, marker, center_rows, center_cols, rotated.shape)
            if _slot_fit(g, skeleton_color, rotated, bbox):
                matches.append((rotated, bbox))

        assert len(matches) == 1, "each piece should have one arm-end fit"
        rotated, (r0, c0, _r1, _c1) = matches[0]
        _paint_mask(out, rotated, int(r0), int(c0), int(color))

    return out


def _arm_base_mask(size: int) -> np.ndarray:
    return TRI_L.copy() if int(size) == 3 else TET_L.copy()


def _allowed_slot_rotations(size: int, arm: str) -> list[int]:
    if int(size) == 3:
        return [0, 1, 2, 3]
    if arm in {"top", "bottom"}:
        return [0, 2]
    return [1, 3]


def _slot_spec(
    arm: str,
    slot_mask: np.ndarray,
    center_r0: int,
    center_c0: int,
    top_len: int,
    bottom_len: int,
    left_len: int,
    right_len: int,
) -> dict:
    h, w = slot_mask.shape
    center_rows = np.array([center_r0, center_r0 + 1], dtype=int)
    center_cols = np.array([center_c0, center_c0 + 1], dtype=int)

    if arm == "top":
        r0 = int(center_r0 - top_len)
        c0 = int(center_c0)
        marker_cols = [int(center_cols[i]) for i in range(2) if bool(slot_mask[h - 1, i])]
        marker = (int(r0 + h), random.choice(marker_cols))
    elif arm == "bottom":
        r0 = int(center_r0 + 2 + bottom_len - h)
        c0 = int(center_c0)
        marker_cols = [int(center_cols[i]) for i in range(2) if bool(slot_mask[0, i])]
        marker = (int(r0 - 1), random.choice(marker_cols))
    elif arm == "left":
        r0 = int(center_r0)
        c0 = int(center_c0 - left_len)
        marker_rows = [int(center_rows[i]) for i in range(2) if bool(slot_mask[i, w - 1])]
        marker = (random.choice(marker_rows), int(c0 + w))
    else:
        r0 = int(center_r0)
        c0 = int(center_c0 + 2 + right_len - w)
        marker_rows = [int(center_rows[i]) for i in range(2) if bool(slot_mask[i, 0])]
        marker = (random.choice(marker_rows), int(c0 - 1))

    slot_cells = [(int(r0 + dr), int(c0 + dc)) for dr, dc in np.argwhere(slot_mask)]
    return {"r0": int(r0), "c0": int(c0), "mask": slot_mask, "marker": marker, "cells": slot_cells}


def _corner_position(
    corner: str,
    piece_shape: tuple[int, int],
    h: int,
    w: int,
    top_margin: int,
    bottom_margin: int,
    left_margin: int,
    right_margin: int,
) -> tuple[int, int]:
    ph, pw = map(int, piece_shape)
    if corner == "top_left":
        r0 = random.randint(0, int(top_margin - ph))
        c0 = random.randint(0, int(left_margin - pw))
    elif corner == "top_right":
        r0 = random.randint(0, int(top_margin - ph))
        c0 = random.randint(int(w - right_margin), int(w - pw))
    elif corner == "bottom_left":
        r0 = random.randint(int(h - bottom_margin), int(h - ph))
        c0 = random.randint(0, int(left_margin - pw))
    else:
        r0 = random.randint(int(h - bottom_margin), int(h - ph))
        c0 = random.randint(int(w - right_margin), int(w - pw))
    return int(r0), int(c0)


def _build_case() -> tuple[np.ndarray, np.ndarray]:
    bg, skeleton_color, *piece_colors = random.sample(list(range(10)), 6)

    top_len = random.randint(4, 8)
    bottom_len = random.randint(4, 8)
    left_len = random.randint(4, 8)
    right_len = random.randint(4, 8)

    top_margin = random.randint(3, 6)
    bottom_margin = random.randint(3, 6)
    left_margin = random.randint(3, 6)
    right_margin = random.randint(3, 6)

    h = int(top_margin + top_len + 2 + bottom_len + bottom_margin)
    w = int(left_margin + left_len + 2 + right_len + right_margin)
    assert h <= 30 and w <= 30

    center_r0 = int(top_margin + top_len)
    center_c0 = int(left_margin + left_len)
    center_rows = np.array([center_r0, center_r0 + 1], dtype=int)
    center_cols = np.array([center_c0, center_c0 + 1], dtype=int)

    grid = np.full((h, w), int(bg), dtype=int)
    grid[int(top_margin) : int(h - bottom_margin), int(center_c0) : int(center_c0 + 2)] = int(skeleton_color)
    grid[int(center_r0) : int(center_r0 + 2), int(left_margin) : int(w - right_margin)] = int(skeleton_color)

    arm_data: dict[str, dict] = {}
    for arm, color in zip(ARM_ORDER, piece_colors):
        size = random.choice([3, 4])
        base = _arm_base_mask(size)
        slot_rot = random.choice(_allowed_slot_rotations(size, arm))
        slot_mask = np.rot90(base, slot_rot)
        spec = _slot_spec(arm, slot_mask, center_r0, center_c0, top_len, bottom_len, left_len, right_len)
        arm_data[arm] = {"color": int(color), "base": base, "slot": spec}
        _paint_mask(grid, slot_mask, spec["r0"], spec["c0"], bg)
        for r, c in spec["cells"]:
            grid[int(r), int(c)] = int(bg)
        mr, mc = spec["marker"]
        grid[int(mr), int(mc)] = int(color)

    corners = list(CORNERS)
    random.shuffle(corners)
    expected = grid.copy()

    for arm, corner in zip(ARM_ORDER, corners):
        color = int(arm_data[arm]["color"])
        piece_mask = np.rot90(arm_data[arm]["base"], random.randint(0, 3))
        pr0, pc0 = _corner_position(
            corner,
            piece_mask.shape,
            h,
            w,
            top_margin,
            bottom_margin,
            left_margin,
            right_margin,
        )
        _paint_mask(grid, piece_mask, pr0, pc0, color)

        for dr, dc in np.argwhere(piece_mask):
            expected[int(pr0 + dr), int(pc0 + dc)] = int(bg)

        mr, mc = arm_data[arm]["slot"]["marker"]
        expected[int(mr), int(mc)] = int(bg)
        for r, c in arm_data[arm]["slot"]["cells"]:
            expected[int(r), int(c)] = int(color)

    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_TRIES):
        grid, expected = _build_case()
        try:
            actual = main(grid)
        except AssertionError:
            continue
        if np.array_equal(actual, expected):
            return grid
    raise RuntimeError("failed to generate a valid cross-notch L-piece puzzle")
