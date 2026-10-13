import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
COLOR_POOL = (3, 4, 5, 6, 7, 8, 9)
GEN_RETRIES = 400
_GEN_CALL_COUNT = 0


ARROW_PATTERNS = {
    frozenset({(0, 2), (1, 0), (1, 1), (1, 2), (2, 2)}): {
        "direction": "right",
        "delta": (0, 1),
        "apex": (1, 2),
    },
    frozenset({(0, 0), (1, 0), (1, 1), (1, 2), (2, 0)}): {
        "direction": "left",
        "delta": (0, -1),
        "apex": (1, 0),
    },
    frozenset({(0, 1), (1, 1), (2, 0), (2, 1), (2, 2)}): {
        "direction": "down",
        "delta": (1, 0),
        "apex": (2, 1),
    },
    frozenset({(0, 0), (0, 1), (0, 2), (1, 1), (2, 1)}): {
        "direction": "up",
        "delta": (-1, 0),
        "apex": (0, 1),
    },
}


ARROW_OFFSETS_FROM_APEX = {
    "right": ((0, 0), (0, -1), (0, -2), (-1, 0), (1, 0)),
    "left": ((0, 0), (0, 1), (0, 2), (-1, 0), (1, 0)),
    "down": ((0, 0), (-1, 0), (-2, 0), (0, -1), (0, 1)),
    "up": ((0, 0), (1, 0), (2, 0), (0, -1), (0, 1)),
}


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _paint_frame(arr: np.ndarray, r0: int, c0: int, r1: int, c1: int, color: int) -> None:
    arr[r0, c0 : c1 + 1] = color
    arr[r1, c0 : c1 + 1] = color
    arr[r0 : r1 + 1, c0] = color
    arr[r0 : r1 + 1, c1] = color


def _find_frames(grid: np.ndarray) -> tuple[list[dict], dict[tuple[int, int], int]]:
    frames: list[dict] = []
    cell_to_frame: dict[tuple[int, int], int] = {}
    for pts in _components(grid == BLUE):
        r0, c0, r1, c1 = _bbox(pts)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        border = set()
        for c in range(c0, c1 + 1):
            border.add((r0, c))
            border.add((r1, c))
        for r in range(r0, r1 + 1):
            border.add((r, c0))
            border.add((r, c1))
        if set(pts) != border:
            continue
        frame_idx = len(frames)
        frame = {
            "bbox": (r0, c0, r1, c1),
            "inner_h": r1 - r0 - 1,
            "inner_w": c1 - c0 - 1,
        }
        frames.append(frame)
        for pt in border:
            cell_to_frame[pt] = frame_idx
    return frames, cell_to_frame


def _find_arrows(grid: np.ndarray) -> list[dict]:
    arrows: list[dict] = []
    for pts in _components(grid == RED):
        r0, c0, _, _ = _bbox(pts)
        norm = frozenset((r - r0, c - c0) for r, c in pts)
        info = ARROW_PATTERNS.get(norm)
        if info is None:
            continue
        ar, ac = info["apex"]
        arrows.append(
            {
                "cells": pts,
                "direction": info["direction"],
                "delta": info["delta"],
                "apex": (r0 + ar, c0 + ac),
            }
        )
    return arrows


def _trace_strip(
    grid: np.ndarray,
    apex: tuple[int, int],
    delta: tuple[int, int],
    cell_to_frame: dict[tuple[int, int], int],
) -> tuple[list[tuple[int, int]], list[int], int]:
    h, w = grid.shape
    dr, dc = delta
    rr, cc = apex[0] + dr, apex[1] + dc
    strip_cells: list[tuple[int, int]] = []
    colors: list[int] = []

    while 0 <= rr < h and 0 <= cc < w:
        value = int(grid[rr, cc])
        if value == BLUE:
            if not colors or (rr, cc) not in cell_to_frame:
                raise AssertionError("strip must terminate at a frame border")
            return strip_cells, colors, cell_to_frame[(rr, cc)]
        if value in (BLACK, RED):
            break
        strip_cells.append((rr, cc))
        colors.append(value)
        rr += dr
        cc += dc

    raise AssertionError("failed to trace a colored strip from arrow to frame")


def _fill_frame(out: np.ndarray, frame: dict, direction: str, colors: list[int]) -> None:
    r0, c0, r1, c1 = frame["bbox"]
    if direction == "right":
        assert len(colors) == frame["inner_w"]
        for i, color in enumerate(colors):
            out[r0 + 1 : r1, c0 + 1 + i] = color
    elif direction == "left":
        assert len(colors) == frame["inner_w"]
        for i, color in enumerate(colors):
            out[r0 + 1 : r1, c1 - 1 - i] = color
    elif direction == "down":
        assert len(colors) == frame["inner_h"]
        for i, color in enumerate(colors):
            out[r0 + 1 + i, c0 + 1 : c1] = color
    elif direction == "up":
        assert len(colors) == frame["inner_h"]
        for i, color in enumerate(colors):
            out[r1 - 1 - i, c0 + 1 : c1] = color
    else:
        raise AssertionError(f"unknown direction: {direction}")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Push each colored strip into the adjacent blue frame according to the red arrow direction.

    Input structure:
    - Background is black.
    - The grid contains several blue hollow rectangular frames with black interiors.
    - Each frame has a red 5-pixel arrow outside one side, and between the arrow tip and the frame there is a 1-cell-thick colored strip whose length matches the frame's interior depth along that direction.

    Rule:
    - Detect each arrow orientation (up, down, left, or right).
    - Read the colored strip in the order from the arrow tip toward the frame.
    - Fill successive interior rows or columns of the pointed frame from the near side inward, using one strip color per whole row or column.
    - Erase the red arrows and the original colored strips to black while keeping the blue frame borders unchanged.

    Key insight: The arrow orientation determines both which frame side is active and whether the strip colors map to rows or columns, while the strip order determines the inward fill order.
    """
    g = np.asarray(grid, dtype=int)
    frames, cell_to_frame = _find_frames(g)
    arrows = _find_arrows(g)

    out = g.copy()
    for arrow in arrows:
        strip_cells, colors, frame_idx = _trace_strip(g, arrow["apex"], arrow["delta"], cell_to_frame)
        _fill_frame(out, frames[frame_idx], arrow["direction"], colors)
        for r, c in strip_cells:
            out[r, c] = BLACK
        for r, c in arrow["cells"]:
            out[r, c] = BLACK

    return out


def _choose_directions(rng: random.Random, count: int) -> list[str]:
    global _GEN_CALL_COUNT
    base = ["right", "down", "left", "up"]
    shift = _GEN_CALL_COUNT % len(base)
    _GEN_CALL_COUNT += 1
    rotated = base[shift:] + base[:shift]
    dirs = rotated[:count]
    rng.shuffle(dirs)
    return dirs


def _make_patch(direction: str, inner_h: int, inner_w: int, colors: list[int]) -> dict:
    outer_h = inner_h + 2
    outer_w = inner_w + 2

    if direction in ("right", "left"):
        patch = np.zeros((outer_h, outer_w + inner_w + 3), dtype=int)
        frame_r0 = 0
        frame_c0 = inner_w + 3 if direction == "right" else 0
        strip_row = outer_h // 2
        if direction == "right":
            for i, color in enumerate(colors):
                patch[strip_row, 3 + i] = color
            apex = (strip_row, 2)
        else:
            start_c = outer_w
            for i, color in enumerate(colors):
                patch[strip_row, start_c + inner_w - 1 - i] = color
            apex = (strip_row, outer_w + inner_w)
    else:
        patch = np.zeros((outer_h + inner_h + 3, outer_w), dtype=int)
        frame_r0 = inner_h + 3 if direction == "down" else 0
        frame_c0 = 0
        strip_col = outer_w // 2
        if direction == "down":
            for i, color in enumerate(colors):
                patch[3 + i, strip_col] = color
            apex = (2, strip_col)
        else:
            start_r = outer_h
            for i, color in enumerate(colors):
                patch[start_r + inner_h - 1 - i, strip_col] = color
            apex = (outer_h + inner_h, strip_col)

    frame_r1 = frame_r0 + outer_h - 1
    frame_c1 = frame_c0 + outer_w - 1
    _paint_frame(patch, frame_r0, frame_c0, frame_r1, frame_c1, BLUE)

    arrow_cells: list[tuple[int, int]] = []
    for dr, dc in ARROW_OFFSETS_FROM_APEX[direction]:
        rr = apex[0] + dr
        cc = apex[1] + dc
        patch[rr, cc] = RED
        arrow_cells.append((rr, cc))

    strip_cells: list[tuple[int, int]] = []
    if direction == "right":
        strip_cells = [(outer_h // 2, 3 + i) for i in range(inner_w)]
    elif direction == "left":
        strip_cells = [(outer_h // 2, outer_w + i) for i in range(inner_w)]
    elif direction == "down":
        strip_cells = [(3 + i, outer_w // 2) for i in range(inner_h)]
    else:
        strip_cells = [(outer_h + i, outer_w // 2) for i in range(inner_h)]

    return {
        "patch": patch,
        "direction": direction,
        "colors": list(colors),
        "frame_bbox_local": (frame_r0, frame_c0, frame_r1, frame_c1),
        "arrow_cells_local": arrow_cells,
        "strip_cells_local": strip_cells,
    }


def _apply_expected_transform(grid: np.ndarray, objects: list[dict]) -> np.ndarray:
    out = grid.copy()
    for obj in objects:
        r0, c0, r1, c1 = obj["frame_bbox"]
        colors = obj["colors"]
        direction = obj["direction"]
        if direction == "right":
            for i, color in enumerate(colors):
                out[r0 + 1 : r1, c0 + 1 + i] = color
        elif direction == "left":
            for i, color in enumerate(colors):
                out[r0 + 1 : r1, c1 - 1 - i] = color
        elif direction == "down":
            for i, color in enumerate(colors):
                out[r0 + 1 + i, c0 + 1 : c1] = color
        else:
            for i, color in enumerate(colors):
                out[r1 - 1 - i, c0 + 1 : c1] = color
        for r, c in obj["erase_cells"]:
            out[r, c] = BLACK
    return out


def _place_patches(rng: random.Random, specs: list[dict]) -> tuple[np.ndarray, list[dict]] | None:
    max_h = max(spec["patch"].shape[0] for spec in specs)
    max_w = max(spec["patch"].shape[1] for spec in specs)
    total_area = sum(spec["patch"].shape[0] * spec["patch"].shape[1] for spec in specs)

    min_h = max(max_h + 2, min(30, int(np.sqrt(total_area) * 1.2) + 6))
    min_w = max(max_w + 2, min(30, int(np.sqrt(total_area) * 1.2) + 6))
    if min_h > 30 or min_w > 30:
        return None

    for _ in range(12):
        grid_h = rng.randint(min_h, 30)
        grid_w = rng.randint(min_w, 30)
        grid = np.zeros((grid_h, grid_w), dtype=int)
        occupied = np.zeros((grid_h, grid_w), dtype=bool)
        placed: list[dict] = []
        order = sorted(
            range(len(specs)),
            key=lambda idx: specs[idx]["patch"].shape[0] * specs[idx]["patch"].shape[1],
            reverse=True,
        )
        ok = True

        for idx in order:
            spec = specs[idx]
            patch = spec["patch"]
            ph, pw = patch.shape
            if ph + 2 > grid_h or pw + 2 > grid_w:
                ok = False
                break

            placed_here = False
            for _ in range(300):
                top = rng.randint(1, grid_h - ph - 1)
                left = rng.randint(1, grid_w - pw - 1)
                if occupied[top - 1 : top + ph + 1, left - 1 : left + pw + 1].any():
                    continue
                mask = patch != BLACK
                sub = grid[top : top + ph, left : left + pw]
                if np.any(sub[mask] != BLACK):
                    continue
                sub[mask] = patch[mask]
                occupied[top : top + ph, left : left + pw] = True
                fr0, fc0, fr1, fc1 = spec["frame_bbox_local"]
                erase_cells = [(top + r, left + c) for r, c in spec["arrow_cells_local"] + spec["strip_cells_local"]]
                placed.append(
                    {
                        "direction": spec["direction"],
                        "colors": list(spec["colors"]),
                        "frame_bbox": (top + fr0, left + fc0, top + fr1, left + fc1),
                        "erase_cells": erase_cells,
                    }
                )
                placed_here = True
                break

            if not placed_here:
                ok = False
                break

        if ok:
            return grid, placed

    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    count = rng.choice((2, 3, 4))
    directions = _choose_directions(rng, count)
    specs: list[dict] = []

    for direction in directions:
        if count == 4:
            inner_h = rng.randint(2, 4)
            inner_w = rng.randint(2, 4)
        else:
            inner_h = rng.randint(2, 6)
            inner_w = rng.randint(2, 6)
        depth = inner_w if direction in ("right", "left") else inner_h
        colors = rng.sample(COLOR_POOL, depth)
        specs.append(_make_patch(direction, inner_h, inner_w, colors))

    placed = _place_patches(rng, specs)
    if placed is None:
        return None

    grid, objects = placed
    expected = _apply_expected_transform(grid, objects)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a valid puzzle")
