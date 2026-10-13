from __future__ import annotations

import random

import numpy as np


BG = 0
RED = 2
GREEN = 3
GREY = 5
PALETTE_COLORS = [1, 4, 6, 7, 8, 9]
GEN_ATTEMPTS = 400


def _most_common_color(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[cnts.argmax()])


def _components_of_color(
    grid: np.ndarray, color: int, row_min: int = 0, row_max: int | None = None
) -> list[dict[str, object]]:
    h, w = map(int, grid.shape)
    if row_max is None:
        row_max = h
    visited = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []

    for r in range(row_min, row_max):
        for c in range(w):
            if visited[r, c] or int(grid[r, c]) != int(color):
                continue

            stack = [(r, c)]
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((int(cr), int(cc)))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if not (row_min <= nr < row_max and 0 <= nc < w):
                        continue
                    if visited[nr, nc] or int(grid[nr, nc]) != int(color):
                        continue
                    visited[nr, nc] = True
                    stack.append((nr, nc))

            arr = np.array(coords, dtype=int)
            r0, c0 = map(int, arr.min(axis=0))
            r1, c1 = map(int, arr.max(axis=0))
            comps.append(
                {
                    "color": int(color),
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(len(coords)),
                }
            )
    return comps


def _find_separator_row(grid: np.ndarray, bg: int) -> tuple[int, int]:
    candidates = []
    for r in range(grid.shape[0]):
        row = grid[r, :]
        if np.all(row == row[0]) and int(row[0]) != int(bg):
            candidates.append((int(r), int(row[0])))
    assert len(candidates) == 1, "expected one full-width separator row"
    return candidates[0]


def _extract_palette_blocks(grid: np.ndarray, sep_r: int, bg: int, sep_color: int) -> list[dict[str, object]]:
    lower = grid[sep_r + 1 :, :]
    colors = [int(c) for c in np.unique(lower) if int(c) not in {int(bg), int(sep_color)}]
    blocks: list[dict[str, object]] = []

    for color in colors:
        comps = _components_of_color(grid, color, row_min=sep_r + 1)
        assert len(comps) == 1, "each palette color must form one component"
        comp = comps[0]
        r0, c0, r1, c1 = comp["bbox"]
        bh = int(r1) - int(r0) + 1
        bw = int(c1) - int(c0) + 1
        assert (bh, bw) == (3, 3), "palette blocks must be 3x3"

        patch = grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
        border = np.ones((3, 3), dtype=bool)
        border[1, 1] = False
        assert np.all(patch[border] == int(color)), "palette border must be monochrome"
        assert np.all((patch == int(color)) | (patch == int(bg))), "palette block may only contain color and bg"

        center = int(patch[1, 1])
        assert center in {int(bg), int(color)}, "palette center must be bg or border color"
        blocks.append(
            {
                "color": int(color),
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "hole": bool(center == int(bg)),
            }
        )

    assert blocks, "expected palette blocks below separator"
    assert len({int(block["bbox"][0]) for block in blocks}) == 1, "palette blocks must share one row"
    blocks.sort(key=lambda block: int(block["bbox"][1]))
    return blocks


def _is_frame_component(grid: np.ndarray, comp: dict[str, object]) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    h = int(r1) - int(r0) + 1
    w = int(c1) - int(c0) + 1
    if h < 3 or w < 3:
        return False

    color = int(comp["color"])
    patch = grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return bool(np.array_equal(patch == color, border))


def _find_frames(grid: np.ndarray, sep_r: int, bg: int) -> list[dict[str, object]]:
    upper = grid[:sep_r, :]
    colors = [int(c) for c in np.unique(upper) if int(c) != int(bg)]
    frames: list[dict[str, object]] = []
    for color in colors:
        for comp in _components_of_color(grid, color, row_max=sep_r):
            if _is_frame_component(grid, comp):
                frames.append(comp)

    assert frames, "expected at least one frame"
    assert len({int(comp["color"]) for comp in frames}) == 1, "all frames must share one color"
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repaint each rectangular frame border using a palette block selected by the
    parity and count of the foreground dots inside the frame.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    sep_r, sep_color = _find_separator_row(g, bg)
    blocks = _extract_palette_blocks(g, sep_r, bg, sep_color)
    hole_blocks = [block for block in blocks if bool(block["hole"])]
    solid_blocks = [block for block in blocks if not bool(block["hole"])]
    assert hole_blocks and solid_blocks, "palette must contain both hole and solid blocks"

    out = g.copy()
    for frame in _find_frames(g, sep_r, bg):
        r0, c0, r1, c1 = frame["bbox"]
        interior = g[int(r0) + 1 : int(r1), int(c0) + 1 : int(c1)]
        non_bg = set(map(int, np.unique(interior))) - {int(bg)}
        assert len(non_bg) == 1, "frame interior must contain one foreground color"
        count = int(np.count_nonzero(interior != int(bg)))
        assert count > 0, "frame interior must contain at least one marker"

        subset = hole_blocks if count % 2 == 1 else solid_blocks
        choice = subset[(count - 1) % len(subset)]
        coords = frame["coords"]
        out[coords[:, 0], coords[:, 1]] = int(choice["color"])

    return out


def _draw_palette_block(grid: np.ndarray, top: int, left: int, color: int, hole: bool) -> None:
    grid[top : top + 3, left : left + 3] = int(color)
    if hole:
        grid[top + 1, left + 1] = BG


def _paint_frame(grid: np.ndarray, r0: int, c0: int, h: int, w: int, color: int) -> None:
    grid[r0, c0 : c0 + w] = int(color)
    grid[r0 + h - 1, c0 : c0 + w] = int(color)
    grid[r0 : r0 + h, c0] = int(color)
    grid[r0 : r0 + h, c0 + w - 1] = int(color)


def _expected_from_layout(
    grid: np.ndarray,
    frames: list[dict[str, int]],
    hole_blocks: list[dict[str, int]],
    solid_blocks: list[dict[str, int]],
) -> np.ndarray:
    out = grid.copy()
    for frame in frames:
        count = int(frame["count"])
        subset = hole_blocks if count % 2 == 1 else solid_blocks
        color = int(subset[(count - 1) % len(subset)]["color"])
        _paint_frame(out, int(frame["r0"]), int(frame["c0"]), int(frame["h"]), int(frame["w"]), color)
    return out


def _sample_palette_layout(total_blocks: int, width: int) -> tuple[int, list[int]] | None:
    gaps = [random.randint(1, 2) for _ in range(total_blocks - 1)]
    required = 3 * total_blocks + sum(gaps)
    slack = width - required
    if slack < 2:
        return None
    left = random.randint(1, slack - 1)
    return left, gaps


def _frame_size_ranges(frame_count: int) -> tuple[tuple[int, int], tuple[int, int]]:
    if frame_count >= 5:
        return (4, 6), (4, 7)
    if frame_count == 4:
        return (4, 7), (4, 8)
    return (4, 8), (4, 10)


def _place_frames(
    height: int,
    width: int,
    sep_r: int,
    frame_count: int,
    hole_count: int,
    solid_count: int,
) -> list[dict[str, int]] | None:
    occupied = np.zeros((height, width), dtype=bool)
    frames: list[dict[str, int]] = []
    parities = [1, 0]
    while len(parities) < frame_count:
        parities.append(random.randint(0, 1))
    random.shuffle(parities)

    hrange, wrange = _frame_size_ranges(frame_count)
    for parity in parities:
        placed = False
        for _ in range(120):
            fh = random.randint(*hrange)
            fw = random.randint(*wrange)
            if sep_r - fh - 1 <= 1 or width - fw - 1 <= 1:
                return None

            interior_area = (fh - 2) * (fw - 2)
            subset_len = hole_count if parity == 1 else solid_count
            max_count = min(interior_area, 9)
            choices = [n for n in range(1, max_count + 1) if n % 2 == parity and n > 0]
            if not choices:
                continue

            large = [n for n in choices if n > subset_len]
            small = [n for n in choices if n <= subset_len]
            if large and small:
                count = random.choice(large if random.random() < 0.5 else small)
            else:
                count = random.choice(choices)

            r0 = random.randint(1, sep_r - fh - 1)
            c0 = random.randint(1, width - fw - 1)
            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(height, r0 + fh + 1)
            cc1 = min(width, c0 + fw + 1)
            if occupied[rr0:rr1, cc0:cc1].any():
                continue

            occupied[rr0:rr1, cc0:cc1] = True
            frames.append({"r0": r0, "c0": c0, "h": fh, "w": fw, "count": int(count)})
            placed = True
            break
        if not placed:
            return None

    return frames


def _generate_one() -> np.ndarray | None:
    frame_count = random.randint(2, 5)
    hole_count = random.randint(2, 3)
    solid_count = random.randint(2, 3)
    total_blocks = hole_count + solid_count

    width = random.randint(max(26, 3 * total_blocks + total_blocks), 34)
    height = random.randint(16, 24)
    sep_r = height - 4

    palette_layout = _sample_palette_layout(total_blocks, width)
    if palette_layout is None:
        return None
    palette_left, gaps = palette_layout

    frames = _place_frames(height, width, sep_r, frame_count, hole_count, solid_count)
    if frames is None:
        return None

    colors = random.sample(PALETTE_COLORS, total_blocks)
    kinds = [True] * hole_count + [False] * solid_count
    pairs = list(zip(colors, kinds, strict=False))
    random.shuffle(pairs)

    grid = np.full((height, width), BG, dtype=int)
    grid[sep_r, :] = GREY

    palette_blocks: list[dict[str, int]] = []
    c = palette_left
    top = sep_r + 1
    for idx, (color, hole) in enumerate(pairs):
        _draw_palette_block(grid, top, c, int(color), bool(hole))
        palette_blocks.append({"color": int(color), "hole": int(bool(hole)), "left": int(c)})
        c += 3
        if idx < len(gaps):
            c += gaps[idx]

    for frame in frames:
        r0, c0 = int(frame["r0"]), int(frame["c0"])
        fh, fw = int(frame["h"]), int(frame["w"])
        _paint_frame(grid, r0, c0, fh, fw, GREEN)

        cells = [
            (r, c)
            for r in range(r0 + 1, r0 + fh - 1)
            for c in range(c0 + 1, c0 + fw - 1)
        ]
        for rr, cc in random.sample(cells, int(frame["count"])):
            grid[rr, cc] = RED

    hole_blocks = sorted(
        [{"color": block["color"]} for block in palette_blocks if block["hole"] == 1],
        key=lambda block: int(next(b["left"] for b in palette_blocks if b["color"] == block["color"])),
    )
    solid_blocks = sorted(
        [{"color": block["color"]} for block in palette_blocks if block["hole"] == 0],
        key=lambda block: int(next(b["left"] for b in palette_blocks if b["color"] == block["color"])),
    )
    expected = _expected_from_layout(grid, frames, hole_blocks, solid_blocks)

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_one()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
