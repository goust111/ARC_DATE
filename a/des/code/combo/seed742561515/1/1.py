from __future__ import annotations

import random

import numpy as np


BG = 0
NON_BG_COLORS = tuple(range(1, 10))
SIDE_TO_DIR = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def _perimeter_coords(h: int, w: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    for c in range(w):
        coords.append((0, c))
    for r in range(1, h - 1):
        coords.append((r, w - 1))
    for c in range(w - 1, -1, -1):
        coords.append((h - 1, c))
    for r in range(h - 2, 0, -1):
        coords.append((r, 0))
    return coords


def _same_color_component(
    grid: np.ndarray, start: tuple[int, int], visited: np.ndarray
) -> dict[str, object]:
    h, w = grid.shape
    sr, sc = start
    color = int(grid[sr, sc])
    stack = [(sr, sc)]
    visited[sr, sc] = True
    pts: list[tuple[int, int]] = []
    while stack:
        r, c = stack.pop()
        pts.append((r, c))
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if not (1 <= nr < h - 1 and 1 <= nc < w - 1):
                continue
            if visited[nr, nc] or int(grid[nr, nc]) != color:
                continue
            visited[nr, nc] = True
            stack.append((nr, nc))
    rows = [r for r, _ in pts]
    cols = [c for _, c in pts]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    return {
        "color": color,
        "bbox": (r0, c0, r1, c1),
        "pts": pts,
    }


def _find_rect_blocks(grid: np.ndarray) -> list[dict[str, object]]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    blocks: list[dict[str, object]] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if visited[r, c] or int(grid[r, c]) == BG:
                continue
            comp = _same_color_component(grid, (r, c), visited)
            r0, c0, r1, c1 = comp["bbox"]
            color = int(comp["color"])
            sub = grid[r0 : r1 + 1, c0 : c1 + 1]
            if np.all(sub == color):
                blocks.append(comp)
    return blocks


def _gap_color(grid: np.ndarray, side: str, row: int, col: int) -> int:
    if side in ("top", "bottom"):
        left = int(grid[row, col - 1]) if col - 1 >= 0 else BG
        right = int(grid[row, col + 1]) if col + 1 < grid.shape[1] else BG
        return left if left != BG else right
    up = int(grid[row - 1, col]) if row - 1 >= 0 else BG
    down = int(grid[row + 1, col]) if row + 1 < grid.shape[0] else BG
    return up if up != BG else down


def _detect_gaps(grid: np.ndarray) -> list[dict[str, object]]:
    h, w = grid.shape
    gaps: list[dict[str, object]] = []
    for c in range(1, w - 1):
        if int(grid[0, c]) == BG:
            gaps.append({"side": "top", "row": 0, "col": c, "color": _gap_color(grid, "top", 0, c)})
    for c in range(1, w - 1):
        if int(grid[h - 1, c]) == BG:
            gaps.append(
                {
                    "side": "bottom",
                    "row": h - 1,
                    "col": c,
                    "color": _gap_color(grid, "bottom", h - 1, c),
                }
            )
    for r in range(1, h - 1):
        if int(grid[r, 0]) == BG:
            gaps.append({"side": "left", "row": r, "col": 0, "color": _gap_color(grid, "left", r, 0)})
    for r in range(1, h - 1):
        if int(grid[r, w - 1]) == BG:
            gaps.append(
                {
                    "side": "right",
                    "row": r,
                    "col": w - 1,
                    "color": _gap_color(grid, "right", r, w - 1),
                }
            )
    return gaps


def _match_gap_to_block(
    gap: dict[str, object], blocks: list[dict[str, object]], used: set[int]
) -> int:
    side = str(gap["side"])
    row = int(gap["row"])
    col = int(gap["col"])
    color = int(gap["color"])
    candidates: list[tuple[int, int]] = []
    for idx, block in enumerate(blocks):
        if idx in used or int(block["color"]) != color:
            continue
        r0, c0, r1, c1 = block["bbox"]
        if side in ("top", "bottom"):
            if not (c0 <= col <= c1):
                continue
            priority = r0 if side == "top" else -r1
        else:
            if not (r0 <= row <= r1):
                continue
            priority = c0 if side == "left" else -c1
        candidates.append((priority, idx))
    assert candidates, f"no aligned block found for gap {gap}"
    candidates.sort()
    if len(candidates) >= 2:
        assert candidates[0][0] != candidates[1][0], f"ambiguous block match for gap {gap}"
    return int(candidates[0][1])


def _transform(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    gaps = _detect_gaps(g)
    blocks = _find_rect_blocks(g)
    used: set[int] = set()
    matches: list[tuple[dict[str, object], dict[str, object]]] = []
    for gap in gaps:
        block_idx = _match_gap_to_block(gap, blocks, used)
        used.add(block_idx)
        matches.append((gap, blocks[block_idx]))

    out = g.copy()
    for _gap, block in matches:
        for r, c in block["pts"]:
            out[r, c] = BG

    h, w = g.shape
    for gap, block in matches:
        r0, c0, r1, c1 = block["bbox"]
        color = int(block["color"])
        side = str(gap["side"])
        if side == "top":
            out[0:r0, c0 : c1 + 1] = color
        elif side == "bottom":
            out[r1 + 1 : h, c0 : c1 + 1] = color
        elif side == "left":
            out[r0 : r1 + 1, 0:c0] = color
        else:
            out[r0 : r1 + 1, c1 + 1 : w] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Push each interior color block into its aligned border gap.

    The outer frame is a one-cell ring of colored segments separated by one-cell
    black gaps. Each gap is paired with an interior monochrome rectangle of the
    same color. The rectangle is translated toward the gap based on the gap's
    side, and the swept path is filled with that color while the rectangle's
    original location becomes black.
    """
    return _transform(np.asarray(grid, dtype=int))


def _sample_spaced_positions(
    rng: random.Random, values: list[int], count: int, min_gap: int
) -> list[int] | None:
    if count == 0:
        return []
    if len(values) < count:
        return None
    for _ in range(200):
        picked = sorted(rng.sample(values, count))
        ok = True
        for a, b in zip(picked, picked[1:], strict=False):
            if b - a < min_gap:
                ok = False
                break
        if ok:
            return picked
    return None


def _sample_side_counts(rng: random.Random) -> dict[str, int]:
    mode = rng.random()
    if mode < 0.2:
        top = rng.randint(1, 2)
        bottom = rng.randint(1, 2)
        left = 0
        right = 0
    elif mode < 0.45:
        top = rng.randint(0, 2)
        bottom = rng.randint(0, 2)
        left = 1
        right = rng.randint(0, 1)
    elif mode < 0.7:
        top = rng.randint(0, 2)
        bottom = rng.randint(0, 2)
        left = rng.randint(0, 1)
        right = 1
    else:
        top = 1 + rng.randint(0, 1)
        bottom = 1 + rng.randint(0, 1)
        left = rng.randint(0, 1)
        right = rng.randint(0, 1)
    counts = {"top": top, "bottom": bottom, "left": left, "right": right}
    if sum(counts.values()) < 2:
        counts["top"] += 1
        counts["bottom"] += 1
    return counts


def _build_border(
    h: int, w: int, gap_coords: list[tuple[int, int]], rng: random.Random
) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    coords = _perimeter_coords(h, w)
    idx_of = {coord: i for i, coord in enumerate(coords)}
    gap_idxs = sorted(idx_of[coord] for coord in gap_coords)
    colors = rng.sample(list(NON_BG_COLORS), len(gap_idxs))
    p = len(coords)
    for color, start, end in zip(colors, gap_idxs, gap_idxs[1:] + gap_idxs[:1], strict=False):
        cur = (start + 1) % p
        while cur != end:
            r, c = coords[cur]
            grid[r, c] = int(color)
            cur = (cur + 1) % p
    return grid


def _same_run_left(grid: np.ndarray, row: int, col: int, color: int) -> int:
    count = 0
    c = col - 1
    while c >= 0 and int(grid[row, c]) == color:
        count += 1
        c -= 1
    return count


def _same_run_up(grid: np.ndarray, row: int, col: int, color: int) -> int:
    count = 0
    r = row - 1
    while r >= 0 and int(grid[r, col]) == color:
        count += 1
        r -= 1
    return count


def _placement_output(
    grid: np.ndarray, placements: list[dict[str, object]]
) -> np.ndarray:
    out = grid.copy()
    h, w = grid.shape
    for placement in placements:
        r0, c0, r1, c1 = placement["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = BG
    for placement in placements:
        r0, c0, r1, c1 = placement["bbox"]
        side = str(placement["side"])
        color = int(placement["color"])
        if side == "top":
            out[0:r0, c0 : c1 + 1] = color
        elif side == "bottom":
            out[r1 + 1 : h, c0 : c1 + 1] = color
        elif side == "left":
            out[r0 : r1 + 1, 0:c0] = color
        else:
            out[r0 : r1 + 1, c1 + 1 : w] = color
    return out


def _generate_case(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(16, 22)
    w = rng.randint(16, 22)
    counts = _sample_side_counts(rng)
    total = sum(counts.values())
    if total > len(NON_BG_COLORS):
        return None

    top_cols = _sample_spaced_positions(rng, list(range(1, w - 1)), counts["top"], 3)
    bottom_cols = _sample_spaced_positions(rng, list(range(1, w - 1)), counts["bottom"], 3)
    if top_cols is None or bottom_cols is None:
        return None

    middle_rows = list(range(6, h - 5))
    if len(middle_rows) < counts["left"] + counts["right"]:
        return None
    lr_rows = _sample_spaced_positions(rng, middle_rows, counts["left"] + counts["right"], 4)
    if lr_rows is None:
        return None

    gap_coords: list[tuple[int, int]] = []
    gap_coords.extend((0, c) for c in top_cols)
    gap_coords.extend((h - 1, c) for c in bottom_cols)
    for i in range(counts["left"]):
        gap_coords.append((lr_rows[i], 0))
    for i in range(counts["right"]):
        gap_coords.append((lr_rows[counts["left"] + i], w - 1))

    border = _build_border(h, w, gap_coords, rng)
    gaps = _detect_gaps(border)
    if len(gaps) != total:
        return None

    inp = border.copy()
    placements: list[dict[str, object]] = []

    for gap in gaps:
        side = str(gap["side"])
        row = int(gap["row"])
        col = int(gap["col"])
        color = int(gap["color"])
        if side in ("top", "bottom"):
            max_w = min(3, _same_run_left(border, row, col, color) + 1)
            if max_w < 1:
                return None
            bw = rng.randint(1, max_w)
            bh = rng.randint(1, 3)
            c1 = col
            c0 = c1 - bw + 1
            if c0 < 1:
                return None
            if side == "top":
                r0 = rng.randint(2, 4)
                r1 = r0 + bh - 1
                if r1 >= 6:
                    return None
            else:
                r1 = rng.randint(h - 5, h - 3)
                r0 = r1 - bh + 1
                if r0 <= h - 7:
                    return None
        else:
            max_h = min(3, _same_run_up(border, row, col, color) + 1)
            if max_h < 1:
                return None
            bh = rng.randint(1, max_h)
            bw = rng.randint(1, 3)
            r1 = row
            r0 = r1 - bh + 1
            if r0 < 4:
                return None
            if side == "left":
                c0 = rng.randint(2, 4)
                c1 = c0 + bw - 1
                if c1 >= 6:
                    return None
            else:
                c1 = rng.randint(w - 5, w - 3)
                c0 = c1 - bw + 1
                if c0 <= w - 7:
                    return None

        if not np.all(inp[r0 : r1 + 1, c0 : c1 + 1] == BG):
            return None
        inp[r0 : r1 + 1, c0 : c1 + 1] = color
        placements.append(
            {
                "side": side,
                "color": color,
                "bbox": (r0, c0, r1, c1),
            }
        )

    expected = _placement_output(inp, placements)
    if int(np.bincount(inp.ravel(), minlength=10).argmax()) != BG:
        return None
    if len(_find_rect_blocks(inp)) != total:
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(400):
        grid = _generate_case(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed after 400 attempts")
