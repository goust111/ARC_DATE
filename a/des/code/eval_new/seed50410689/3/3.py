import random
from collections import deque

import numpy as np


BG = 8
FRAME = 5
ARROW = 9
BLOCK_COLORS = [1, 2, 3, 4, 6, 7, 9]
SIDES = ["top", "bottom", "left", "right"]
GEN_ATTEMPTS = 400


def _components(mask: np.ndarray, connectivity: int = 4) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    if connectivity == 4:
        nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        nbrs = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in nbrs:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), max(rows), min(cols), max(cols)


def _detect_frames(grid: np.ndarray) -> list[dict[str, int]]:
    frames = []
    for comp in _components(grid == FRAME, connectivity=4):
        r0, r1, c0, c1 = _bbox(comp)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        border = {
            (r, c)
            for r in range(r0, r1 + 1)
            for c in range(c0, c1 + 1)
            if r in (r0, r1) or c in (c0, c1)
        }
        if set(comp) != border:
            continue
        frames.append(
            {
                "r0": r0,
                "r1": r1,
                "c0": c0,
                "c1": c1,
                "h": r1 - r0 + 1,
                "w": c1 - c0 + 1,
            }
        )
    frames.sort(key=lambda f: (f["r0"], f["c0"]))
    return frames


def _arrow_template(side: str, line: int, h: int, w: int) -> set[tuple[int, int]]:
    if side == "left":
        return {(line - 1, 0), (line, 1), (line + 1, 0)}
    if side == "right":
        return {(line - 1, w - 1), (line, w - 2), (line + 1, w - 1)}
    if side == "top":
        return {(0, line - 1), (1, line), (0, line + 1)}
    return {(h - 1, line - 1), (h - 2, line), (h - 1, line + 1)}


def _detect_arrows(grid: np.ndarray) -> tuple[list[dict[str, object]], np.ndarray]:
    h, w = grid.shape
    arrow_mask = np.zeros_like(grid, dtype=bool)
    arrows = []
    for comp in _components(grid == ARROW, connectivity=8):
        pts = set(comp)
        if len(pts) != 3:
            continue
        sides = set()
        if any(c == 0 for _, c in pts):
            sides.add("left")
        if any(c == w - 1 for _, c in pts):
            sides.add("right")
        if any(r == 0 for r, _ in pts):
            sides.add("top")
        if any(r == h - 1 for r, _ in pts):
            sides.add("bottom")
        if len(sides) != 1:
            continue
        side = next(iter(sides))
        if side in ("left", "right"):
            rows = sorted(r for r, _ in pts)
            if rows[2] - rows[0] != 2:
                continue
            line = rows[1]
        else:
            cols = sorted(c for _, c in pts)
            if cols[2] - cols[0] != 2:
                continue
            line = cols[1]
        if pts != _arrow_template(side, line, h, w):
            continue
        if side == "left":
            dr, dc = 0, 1
            axis = "row"
        elif side == "right":
            dr, dc = 0, -1
            axis = "row"
        elif side == "top":
            dr, dc = 1, 0
            axis = "col"
        else:
            dr, dc = -1, 0
            axis = "col"
        for r, c in pts:
            arrow_mask[r, c] = True
        arrows.append({"side": side, "line": line, "dr": dr, "dc": dc, "axis": axis, "cells": pts})
    arrows.sort(key=lambda a: (str(a["side"]), int(a["line"])))
    return arrows, arrow_mask


def _pair_frames_arrows(
    frames: list[dict[str, int]], arrows: list[dict[str, object]]
) -> list[tuple[dict[str, int], dict[str, object]]]:
    pairs = []
    used: set[int] = set()
    for frame in frames:
        matches = []
        for idx, arrow in enumerate(arrows):
            line = int(arrow["line"])
            if arrow["axis"] == "row" and frame["r0"] <= line <= frame["r1"]:
                matches.append((idx, arrow))
            if arrow["axis"] == "col" and frame["c0"] <= line <= frame["c1"]:
                matches.append((idx, arrow))
        assert len(matches) == 1, "each frame must match exactly one arrow"
        idx, arrow = matches[0]
        assert idx not in used, "each arrow must match exactly one frame"
        used.add(idx)
        pairs.append((frame, arrow))
    assert len(used) == len(arrows) == len(frames), "frame/arrow counts must agree"
    return pairs


def _scan_palette(grid: np.ndarray, arrow: dict[str, object], arrow_mask: np.ndarray) -> list[int]:
    h, w = grid.shape
    line = int(arrow["line"])
    dr = int(arrow["dr"])
    dc = int(arrow["dc"])
    if dr == 0 and dc == 1:
        coords = [(line, c) for c in range(w)]
    elif dr == 0 and dc == -1:
        coords = [(line, c) for c in range(w - 1, -1, -1)]
    elif dr == 1 and dc == 0:
        coords = [(r, line) for r in range(h)]
    else:
        coords = [(r, line) for r in range(h - 1, -1, -1)]

    palette: list[int] = []
    in_run = False
    last_color = None
    for r, c in coords:
        if arrow_mask[r, c]:
            in_run = False
            last_color = None
            continue
        color = int(grid[r, c])
        if color in (BG, FRAME):
            in_run = False
            last_color = None
            continue
        if (not in_run) or color != last_color:
            palette.append(color)
            in_run = True
            last_color = color
    assert palette, "each arrow scan must encounter at least one color block"
    return palette


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each hollow frame with cyclic stripes whose colors are collected along the matching edge-arrow scan line.

    Input structure:
    - Background is Azure.
    - Several interior 1x2 or 2x1 monochrome color blocks are scattered through the grid.
    - There are 2-3 Grey hollow rectangular frames with Azure interiors.
    - Maroon 3-pixel arrowheads sit on a single outer edge; each arrow aligns by row or column with exactly one Grey frame.

    Rule:
    - Detect each edge arrow and pair it with the unique Grey frame whose row-range or column-range contains the arrow's center line.
    - Starting from that arrow, scan straight inward across the whole row or column and record the colors of the non-background blocks encountered in order, ignoring Azure, Grey frame pixels, and the arrow pixels themselves.
    - Fill the paired frame's interior by cycling through that color list in stripes perpendicular to the scan direction: horizontal scans produce vertical stripes, vertical scans produce horizontal stripes.

    Key insight:
    - The arrow simultaneously chooses which row or column is read as a palette and how the palette must be rotated to stripe the empty rectangle.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    frames = _detect_frames(g)
    arrows, arrow_mask = _detect_arrows(g)
    assert frames and arrows, "expected frames and edge arrows"
    assert len({str(arrow["side"]) for arrow in arrows}) == 1, "all arrows must lie on one edge"

    for frame, arrow in _pair_frames_arrows(frames, arrows):
        palette = _scan_palette(g, arrow, arrow_mask)
        inner_h = frame["h"] - 2
        inner_w = frame["w"] - 2
        if arrow["axis"] == "row":
            for k in range(inner_w):
                out[frame["r0"] + 1 : frame["r1"], frame["c0"] + 1 + k] = palette[k % len(palette)]
        else:
            for k in range(inner_h):
                out[frame["r0"] + 1 + k, frame["c0"] + 1 : frame["c1"]] = palette[k % len(palette)]

    return out


def _interval_starts(
    total_len: int, lengths: list[int], rng: random.Random, margin: int = 3, gap: int = 2
) -> list[int] | None:
    slack = total_len - 2 * margin - sum(lengths) - gap * (len(lengths) - 1)
    if slack < 0:
        return None
    extras = [0] * (len(lengths) + 1)
    for _ in range(slack):
        extras[rng.randrange(len(extras))] += 1
    starts = []
    pos = margin + extras[0]
    for i, length in enumerate(lengths):
        starts.append(pos)
        pos += length
        if i < len(lengths) - 1:
            pos += gap + extras[i + 1]
    return starts


def _draw_frame(grid: np.ndarray, frame: dict[str, int]) -> None:
    r0, r1, c0, c1 = frame["r0"], frame["r1"], frame["c0"], frame["c1"]
    grid[r0, c0 : c1 + 1] = FRAME
    grid[r1, c0 : c1 + 1] = FRAME
    grid[r0 : r1 + 1, c0] = FRAME
    grid[r0 : r1 + 1, c1] = FRAME


def _cells_with_padding(cells: list[tuple[int, int]], h: int, w: int, pad: int = 1) -> set[tuple[int, int]]:
    out = set()
    for r, c in cells:
        for rr in range(r - pad, r + pad + 1):
            for cc in range(c - pad, c + pad + 1):
                if 0 <= rr < h and 0 <= cc < w:
                    out.add((rr, cc))
    return out


def _reserve_box(blocked: np.ndarray, frame: dict[str, int], pad: int = 1) -> None:
    h, w = blocked.shape
    r0 = max(0, frame["r0"] - pad)
    r1 = min(h - 1, frame["r1"] + pad)
    c0 = max(0, frame["c0"] - pad)
    c1 = min(w - 1, frame["c1"] + pad)
    blocked[r0 : r1 + 1, c0 : c1 + 1] = True


def _block_cells(blocked: np.ndarray, cells: list[tuple[int, int]], pad: int = 1) -> None:
    h, w = blocked.shape
    for r, c in _cells_with_padding(cells, h, w, pad=pad):
        blocked[r, c] = True


def _make_block_cells(
    side: str, scan_line: int, anchor: int, orient: str, bias: int
) -> list[tuple[int, int]]:
    if side in ("left", "right"):
        r = scan_line
        c = anchor
        if orient == "h":
            return [(r, c), (r, c + 1)] if bias == 0 else [(r, c - 1), (r, c)]
        return [(r, c), (r + 1, c)] if bias == 0 else [(r - 1, c), (r, c)]
    r = anchor
    c = scan_line
    if orient == "v":
        return [(r, c), (r + 1, c)] if bias == 0 else [(r - 1, c), (r, c)]
    return [(r, c), (r, c + 1)] if bias == 0 else [(r, c - 1), (r, c)]


def _place_sequence_blocks(
    grid: np.ndarray,
    blocked: np.ndarray,
    side: str,
    scan_line: int,
    colors: list[int],
    rng: random.Random,
) -> bool:
    h, w = grid.shape
    if side in ("left", "right"):
        anchors = list(range(3, w - 3))
        if side == "right":
            anchors.reverse()
    else:
        anchors = list(range(3, h - 3))
        if side == "bottom":
            anchors.reverse()

    cursor = 0
    for color in colors:
        placed = False
        while cursor < len(anchors):
            anchor = anchors[cursor]
            cursor += rng.randint(2, 4)
            options = [("h", 0), ("h", 1), ("v", 0), ("v", 1)]
            rng.shuffle(options)
            for orient, bias in options:
                cells = _make_block_cells(side, scan_line, anchor, orient, bias)
                if any(not (1 <= r < h - 1 and 1 <= c < w - 1) for r, c in cells):
                    continue
                if any(blocked[r, c] for r, c in cells):
                    continue
                if len(set(cells)) != 2:
                    continue
                for r, c in cells:
                    grid[r, c] = color
                _block_cells(blocked, cells, pad=1)
                placed = True
                break
            if placed:
                break
        if not placed:
            return False
    return True


def _place_extra_blocks(
    grid: np.ndarray,
    blocked: np.ndarray,
    side: str,
    scan_lines: set[int],
    count: int,
    rng: random.Random,
) -> bool:
    h, w = grid.shape
    for _ in range(count):
        placed = False
        for _attempt in range(80):
            color = rng.choice(BLOCK_COLORS)
            if side in ("left", "right"):
                orient = rng.choice(["h", "v"])
                r = rng.randint(2, h - 3)
                c = rng.randint(2, w - 3)
                if orient == "h":
                    cells = [(r, c), (r, c + 1)]
                else:
                    cells = [(r, c), (r + 1, c)]
                if any(rr in scan_lines for rr, _ in cells):
                    continue
            else:
                orient = rng.choice(["h", "v"])
                r = rng.randint(2, h - 3)
                c = rng.randint(2, w - 3)
                if orient == "h":
                    cells = [(r, c), (r, c + 1)]
                else:
                    cells = [(r, c), (r + 1, c)]
                if any(cc in scan_lines for _, cc in cells):
                    continue
            if any(blocked[rr, cc] for rr, cc in cells):
                continue
            for rr, cc in cells:
                grid[rr, cc] = color
            _block_cells(blocked, cells, pad=1)
            placed = True
            break
        if not placed:
            return False
    return True


def _build_attempt(rng: random.Random) -> np.ndarray | None:
    side = rng.choice(SIDES)
    n_frames = rng.choice([2, 3])

    if side in ("top", "bottom"):
        min_w = 18 if n_frames == 2 else 24
        h = rng.randint(18, 30)
        w = rng.randint(min_w, 30)
    else:
        min_h = 18 if n_frames == 2 else 24
        h = rng.randint(min_h, 30)
        w = rng.randint(18, 30)

    grid = np.full((h, w), BG, dtype=int)

    frame_heights = [rng.randint(5, 8) for _ in range(n_frames)]
    frame_widths = [rng.randint(5, 8) for _ in range(n_frames)]

    if side in ("top", "bottom"):
        starts = _interval_starts(w, frame_widths, rng, margin=3, gap=2)
        if starts is None:
            return None
        frames = []
        for fw, fh, c0 in zip(frame_widths, frame_heights, starts):
            row_max = h - fh - 3
            if row_max < 3:
                return None
            r0 = rng.randint(3, row_max)
            frames.append({"r0": r0, "r1": r0 + fh - 1, "c0": c0, "c1": c0 + fw - 1, "h": fh, "w": fw})
    else:
        starts = _interval_starts(h, frame_heights, rng, margin=3, gap=2)
        if starts is None:
            return None
        frames = []
        for fh, fw, r0 in zip(frame_heights, frame_widths, starts):
            col_max = w - fw - 3
            if col_max < 3:
                return None
            c0 = rng.randint(3, col_max)
            frames.append({"r0": r0, "r1": r0 + fh - 1, "c0": c0, "c1": c0 + fw - 1, "h": fh, "w": fw})

    blocked = np.zeros((h, w), dtype=bool)
    for frame in frames:
        _draw_frame(grid, frame)
        _reserve_box(blocked, frame, pad=1)

    scan_lines = set()
    for frame in frames:
        if side in ("top", "bottom"):
            line = (frame["c0"] + frame["c1"]) // 2
        else:
            line = (frame["r0"] + frame["r1"]) // 2
        scan_lines.add(line)
        cells = list(_arrow_template(side, line, h, w))
        if any(grid[r, c] != BG for r, c in cells):
            return None
        for r, c in cells:
            grid[r, c] = ARROW
        _block_cells(blocked, cells, pad=1)

    frame_order = sorted(frames, key=lambda f: (f["r0"], f["c0"]))
    for frame in frame_order:
        line = (frame["c0"] + frame["c1"]) // 2 if side in ("top", "bottom") else (frame["r0"] + frame["r1"]) // 2
        seq_len = rng.randint(2, 4)
        colors = rng.sample(BLOCK_COLORS, seq_len)
        if not _place_sequence_blocks(grid, blocked, side, line, colors, rng):
            return None

    if not _place_extra_blocks(grid, blocked, side, scan_lines, rng.randint(0, 3), rng):
        return None

    frames_found = _detect_frames(grid)
    arrows_found, arrow_mask = _detect_arrows(grid)
    if len(frames_found) != n_frames or len(arrows_found) != n_frames:
        return None
    try:
        pairs = _pair_frames_arrows(frames_found, arrows_found)
    except AssertionError:
        return None
    if len({str(a["side"]) for _, a in pairs}) != 1:
        return None
    for _frame, arrow in pairs:
        palette = _scan_palette(grid, arrow, arrow_mask)
        if len(palette) < 2:
            return None

    _ = main(grid)
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _build_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
