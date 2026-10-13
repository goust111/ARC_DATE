import itertools
import random
from collections import Counter, deque

import numpy as np


PALETTE = tuple(range(10))
FRAME_SIZES = (5, 7)
OBJECT_COUNT_CHOICES = (2, 3)
CANVAS_LIMIT = 30
PLACEMENT_GAP = 1
GEN_ATTEMPTS = 400
SIDE_TO_ROT90_K = {
    "top": 0,
    "right": 3,   # clockwise 90
    "bottom": 2,
    "left": 1,    # counterclockwise 90
}


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _monochrome_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if visited[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            queue = deque([(r, c)])
            visited[r, c] = True
            cells: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if visited[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    visited[nr, nc] = True
                    queue.append((nr, nc))
            comps.append({"color": color, "cells": cells, "bbox": _bbox(cells)})
    return comps


def _gap_side(size: int, missing: tuple[int, int]) -> str | None:
    center = size // 2
    if missing == (0, center):
        return "top"
    if missing == (center, size - 1):
        return "right"
    if missing == (size - 1, center):
        return "bottom"
    if missing == (center, 0):
        return "left"
    return None


def _detect_frames(grid: np.ndarray, bg: int) -> list[dict]:
    frames: list[dict] = []
    for comp in _monochrome_components(grid, bg):
        frame_color = int(comp["color"])
        r0, c0, r1, c1 = comp["bbox"]
        size = int(r1 - r0 + 1)
        if size != int(c1 - c0 + 1) or size < 5 or size % 2 == 0:
            continue

        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        cells = {(r - r0, c - c0) for r, c in comp["cells"]}

        border = []
        for rr in range(size):
            for cc in range(size):
                if rr in (0, size - 1) or cc in (0, size - 1):
                    border.append((rr, cc))

        if len(cells) != len(border) - 1:
            continue
        if any(pos not in border for pos in cells):
            continue
        if bool((sub[1:-1, 1:-1] == frame_color).any()):
            continue

        missing = [pos for pos in border if pos not in cells]
        if len(missing) != 1:
            continue
        if int(sub[missing[0]]) != bg:
            continue

        side = _gap_side(size, missing[0])
        if side is None:
            continue

        inner = sub[1:-1, 1:-1]
        pattern_mask = (inner != bg) & (inner != frame_color)
        if not bool(pattern_mask.any()):
            continue
        coords = np.argwhere(pattern_mask)
        pr0, pc0 = coords.min(axis=0)
        pr1, pc1 = coords.max(axis=0)
        if (int(pr1 - pr0 + 1), int(pc1 - pc0 + 1)) != (3, 3):
            continue

        pattern = inner[pr0 : pr1 + 1, pc0 : pc1 + 1]
        if bool((pattern == bg).any()) or bool((pattern == frame_color).any()):
            continue

        counts = Counter(map(int, pattern.ravel()))
        if len(counts) != 2:
            continue
        marker_color, marker_count = min(counts.items(), key=lambda kv: (kv[1], kv[0]))
        body_color, body_count = max(counts.items(), key=lambda kv: (kv[1], -kv[0]))
        if not (1 <= int(marker_count) <= 4 and int(body_count) >= 5):
            continue

        frames.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "frame_color": frame_color,
                "gap_side": side,
                "pattern": pattern.astype(int),
                "marker_color": int(marker_color),
                "body_color": int(body_color),
            }
        )

    frames.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each framed 3x3 motif from its border gap direction, then scale it by its marker-pixel count.

    Input structure:
    - The background color is dominant.
    - Several hollow square frames appear on the grid; each frame is one color, one pixel thick, and has exactly one one-cell gap at the center of a side.
    - Inside each frame there is a full 3x3 motif made from a body color plus a minority marker color.

    Rule:
    - Detect every square frame, read which side contains the central gap, and rotate the enclosed 3x3 motif: top -> 0 degrees, right -> clockwise 90 degrees, bottom -> 180 degrees, left -> counterclockwise 90 degrees.
    - Count the marker-color pixels in the rotated motif; if the count is N, enlarge the rotated 3x3 motif by an integer factor N so each source cell becomes an N x N block.
    - Sort the framed objects in input reading order and place the enlarged motifs left-to-right in one output row with one background column between neighbors; align all motifs to the top and pad below with background.

    Key insight:
    - The frame geometry and the interior colors encode two independent steps: the gap direction chooses the rotation, while the minority marker-color count chooses the scale factor.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    frames = _detect_frames(g, bg)
    assert frames, "no valid framed motifs found"

    tiles: list[np.ndarray] = []
    for frame in frames:
        rotated = np.rot90(frame["pattern"], SIDE_TO_ROT90_K[frame["gap_side"]])
        scale = int((rotated == int(frame["marker_color"])).sum())
        assert 1 <= scale <= 4
        tile = np.repeat(np.repeat(rotated, scale, axis=0), scale, axis=1)
        tiles.append(tile.astype(int))

    out_h = max(tile.shape[0] for tile in tiles)
    out_w = sum(tile.shape[1] for tile in tiles) + len(tiles) - 1
    out = np.full((out_h, out_w), bg, dtype=int)

    col = 0
    for tile in tiles:
        h, w = tile.shape
        out[:h, col : col + w] = tile
        col += w + 1

    return out


def _build_marker_masks() -> dict[int, list[tuple[tuple[int, int], ...]]]:
    cells = [(r, c) for r in range(3) for c in range(3)]
    by_count = {1: [], 2: [], 3: [], 4: []}
    for count in range(1, 5):
        for combo in itertools.combinations(cells, count):
            arr = np.zeros((3, 3), dtype=int)
            for r, c in combo:
                arr[r, c] = 1
            rotations = {tuple(np.rot90(arr, k).ravel()) for k in range(4)}
            if len(rotations) == 4:
                by_count[count].append(tuple(sorted(combo)))
    return by_count


MARKER_MASKS = _build_marker_masks()


def _make_pattern(
    rng: random.Random,
    body_color: int,
    marker_color: int,
    marker_count: int,
) -> np.ndarray:
    pattern = np.full((3, 3), body_color, dtype=int)
    mask = rng.choice(MARKER_MASKS[marker_count])
    for r, c in mask:
        pattern[r, c] = marker_color
    return pattern


def _make_object(
    bg: int,
    frame_color: int,
    body_color: int,
    marker_color: int,
    size: int,
    gap_side: str,
    marker_count: int,
    rng: random.Random,
) -> np.ndarray:
    obj = np.full((size, size), bg, dtype=int)
    obj[0, :] = frame_color
    obj[-1, :] = frame_color
    obj[:, 0] = frame_color
    obj[:, -1] = frame_color

    center = size // 2
    if gap_side == "top":
        obj[0, center] = bg
    elif gap_side == "right":
        obj[center, -1] = bg
    elif gap_side == "bottom":
        obj[-1, center] = bg
    else:
        obj[center, 0] = bg

    pattern = _make_pattern(rng, body_color, marker_color, marker_count)
    interior_span = size - 2
    top = 1 + rng.randrange(interior_span - 3 + 1)
    left = 1 + rng.randrange(interior_span - 3 + 1)
    obj[top : top + 3, left : left + 3] = pattern
    return obj


def _rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + pad < br0
        or br1 + pad < ar0
        or ac1 + pad < bc0
        or bc1 + pad < ac0
    )


def _place_objects(rng: random.Random, objects: list[np.ndarray], bg: int) -> np.ndarray | None:
    canvas = np.full((CANVAS_LIMIT, CANVAS_LIMIT), bg, dtype=int)
    placed_rects: list[tuple[int, int, int, int]] = []
    for obj in objects:
        h, w = obj.shape
        placed = False
        for _ in range(200):
            r0 = rng.randrange(1, CANVAS_LIMIT - h - 1)
            c0 = rng.randrange(1, CANVAS_LIMIT - w - 1)
            rect = (r0, c0, r0 + h - 1, c0 + w - 1)
            if any(_rects_overlap(rect, other, PLACEMENT_GAP) for other in placed_rects):
                continue
            canvas[r0 : r0 + h, c0 : c0 + w] = obj
            placed_rects.append(rect)
            placed = True
            break
        if not placed:
            return None

    used = np.argwhere(canvas != bg)
    r0, c0 = used.min(axis=0)
    r1, c1 = used.max(axis=0)

    extra_top = min(int(r0), rng.randint(1, 4))
    extra_left = min(int(c0), rng.randint(1, 4))
    extra_bottom = min(int(CANVAS_LIMIT - 1 - r1), rng.randint(1, 4))
    extra_right = min(int(CANVAS_LIMIT - 1 - c1), rng.randint(1, 4))

    cropped = canvas[
        int(r0 - extra_top) : int(r1 + extra_bottom) + 1,
        int(c0 - extra_left) : int(c1 + extra_right) + 1,
    ]
    return cropped.astype(int)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    bg = rng.choice(PALETTE)
    object_count = rng.choice(OBJECT_COUNT_CHOICES)

    marker_counts = [rng.randint(1, 4) for _ in range(object_count)]
    if 3 * sum(marker_counts) + object_count - 1 > CANVAS_LIMIT:
        return None

    colors = list(PALETTE)
    colors.remove(bg)
    rng.shuffle(colors)
    chosen = colors[: 3 * object_count]

    objects: list[np.ndarray] = []
    for idx in range(object_count):
        frame_color, body_color, marker_color = chosen[3 * idx : 3 * idx + 3]
        size = rng.choice(FRAME_SIZES)
        gap_side = rng.choice(("top", "right", "bottom", "left"))
        obj = _make_object(
            bg=bg,
            frame_color=frame_color,
            body_color=body_color,
            marker_color=marker_color,
            size=size,
            gap_side=gap_side,
            marker_count=marker_counts[idx],
            rng=rng,
        )
        objects.append(obj)

    grid = _place_objects(rng, objects, bg)
    if grid is None:
        return None

    frames = _detect_frames(grid, bg)
    if len(frames) != object_count:
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    if grid.ndim != 2 or out.ndim != 2:
        return None
    if max(grid.shape) > CANVAS_LIMIT or max(out.shape) > CANVAS_LIMIT:
        return None
    return grid.astype(int)


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_ATTEMPTS} attempts")
