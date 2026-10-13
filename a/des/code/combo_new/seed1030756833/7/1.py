from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
DIRS = ("up", "down", "left", "right")
DIR_TO_SIDE = {
    "up": "top",
    "down": "bottom",
    "left": "left",
    "right": "right",
}


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in points]
    cols = [c for _r, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _same_color_components(g: np.ndarray) -> list[dict]:
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r0 in range(h):
        for c0 in range(w):
            if seen[r0, c0] or int(g[r0, c0]) == BG:
                continue
            color = int(g[r0, c0])
            q = deque([(r0, c0)])
            seen[r0, c0] = True
            pts: list[tuple[int, int]] = []
            while q:
                r, c = q.popleft()
                pts.append((int(r), int(c)))
                for rr, cc in _neighbors4(int(r), int(c)):
                    if 0 <= rr < h and 0 <= cc < w and not seen[rr, cc] and int(g[rr, cc]) == color:
                        seen[rr, cc] = True
                        q.append((rr, cc))
            r0b, c0b, r1b, c1b = _bbox(pts)
            comps.append(
                {
                    "color": color,
                    "points": pts,
                    "bbox": (r0b, c0b, r1b, c1b),
                    "area": len(pts),
                }
            )
    return comps


def _frame_from_component(g: np.ndarray, comp: dict) -> dict | None:
    pts = set(comp["points"])
    r0, c0, r1, c1 = comp["bbox"]
    if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 3:
        return None

    border = {
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if r in (r0, r1) or c in (c0, c1)
    }
    missing = list(border - pts)
    if len(missing) != 1 or pts - border:
        return None

    # The frame interior must be blank in the input.
    if np.any(g[r0 + 1 : r1, c0 + 1 : c1] != BG):
        return None

    gr, gc = missing[0]
    if gr == r0 and c0 < gc < c1:
        direction = "up"
    elif gr == r1 and c0 < gc < c1:
        direction = "down"
    elif gc == c0 and r0 < gr < r1:
        direction = "left"
    elif gc == c1 and r0 < gr < r1:
        direction = "right"
    else:
        return None

    return {
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "color": int(comp["color"]),
        "gap": (int(gr), int(gc)),
        "direction": direction,
    }


def _classify_edge_block(bbox: tuple[int, int, int, int], h: int, w: int) -> str | None:
    r0, c0, r1, c1 = bbox
    distances = {
        "top": r0,
        "bottom": h - 1 - r1,
        "left": c0,
        "right": w - 1 - c1,
    }
    side, dist = min(distances.items(), key=lambda item: item[1])
    if dist > 3:
        return None

    center_r2 = r0 + r1
    center_c2 = c0 + c1
    if side in ("top", "bottom"):
        if center_c2 <= 4 or center_c2 >= 2 * w - 6:
            return None
    else:
        if center_r2 <= 4 or center_r2 >= 2 * h - 6:
            return None
    return side


def _find_edge_blocks(g: np.ndarray, frame_components: set[int]) -> dict[str, list[dict]]:
    h, w = g.shape
    blocks = {side: [] for side in ("top", "bottom", "left", "right")}
    for idx, comp in enumerate(_same_color_components(g)):
        if idx in frame_components:
            continue
        r0, c0, r1, c1 = comp["bbox"]
        bh = r1 - r0 + 1
        bw = c1 - c0 + 1
        if bh < 2 or bw < 2 or bh > 4 or bw > 4:
            continue
        if comp["area"] != bh * bw:
            continue
        side = _classify_edge_block(comp["bbox"], h, w)
        if side is None:
            continue
        blocks[side].append(
            {
                "color": int(comp["color"]),
                "area": int(comp["area"]),
                "bbox": comp["bbox"],
            }
        )

    for side in blocks:
        blocks[side].sort(key=lambda b: (-int(b["area"]), int(b["bbox"][0]), int(b["bbox"][1])))
    return blocks


def _find_frames_and_component_ids(g: np.ndarray) -> tuple[list[dict], set[int]]:
    frames: list[dict] = []
    frame_component_ids: set[int] = set()
    comps = _same_color_components(g)
    for idx, comp in enumerate(comps):
        frame = _frame_from_component(g, comp)
        if frame is not None:
            frames.append(frame)
            frame_component_ids.add(idx)
    return frames, frame_component_ids


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each hollow frame with the edge-palette color indicated by its missing side.

    Input structure:
    - The background is black (0).
    - Several one-pixel-thick rectangular hollow frames are scattered in the interior; each frame has one same-color border with exactly one non-corner border pixel missing.
    - Small isolated solid color blocks sit near the top, bottom, left, and right grid edges and do not overlap any frame.
    - Multiple blocks on the same edge direction are visibly ordered by their block area.

    Rule:
    - Detect which side of each frame contains the single missing border pixel: top, bottom, left, or right.
    - Match that side to the corresponding edge color pool: top->up, bottom->down, left->left, right->right.
    - For each direction, sort its frames from left to right and then top to bottom, and sort that edge's color blocks by area from large to small.
    - Fill every blank interior pixel of each frame with the assigned edge-block color, leaving the border and the missing border gap unchanged.

    Key insight: The tiny gap in a frame is a hidden direction key, and that direction selects a visible color pool on the matching grid edge.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    frames, frame_component_ids = _find_frames_and_component_ids(g)
    pools = _find_edge_blocks(g, frame_component_ids)

    for direction in DIRS:
        side = DIR_TO_SIDE[direction]
        group = [f for f in frames if f["direction"] == direction]
        group.sort(key=lambda f: (int(f["bbox"][1]), int(f["bbox"][0])))
        assert len(pools[side]) >= len(group), "each frame direction needs a matching edge color block"
        for frame, block in zip(group, pools[side], strict=False):
            r0, c0, r1, c1 = frame["bbox"]
            interior = out[r0 + 1 : r1, c0 + 1 : c1]
            interior[interior == BG] = int(block["color"])

    return out


# ------------------------------ generator ------------------------------

GRID_SIZE_RANGE = (24, 30)
FRAME_H_RANGE = (4, 7)
FRAME_W_RANGE = (4, 7)
GEN_RETRIES = 500


def _can_place(mask: np.ndarray, r: int, c: int, h: int, w: int, pad: int = 1) -> bool:
    H, W = mask.shape
    if r < 0 or c < 0 or r + h > H or c + w > W:
        return False
    rr0 = max(0, r - pad)
    cc0 = max(0, c - pad)
    rr1 = min(H, r + h + pad)
    cc1 = min(W, c + w + pad)
    return not bool(mask[rr0:rr1, cc0:cc1].any())


def _reserve(mask: np.ndarray, r: int, c: int, h: int, w: int) -> None:
    mask[r : r + h, c : c + w] = True


def _place_block(
    grid: np.ndarray,
    mask: np.ndarray,
    side: str,
    size: int,
    color: int,
) -> dict | None:
    H, W = grid.shape
    for _ in range(100):
        if side == "top":
            r = random.randint(0, 1)
            c = random.randint(4, W - size - 5)
        elif side == "bottom":
            r = H - size - random.randint(0, 1)
            c = random.randint(4, W - size - 5)
        elif side == "left":
            r = random.randint(4, H - size - 5)
            c = random.randint(0, 1)
        else:
            r = random.randint(4, H - size - 5)
            c = W - size - random.randint(0, 1)
        if _can_place(mask, r, c, size, size, pad=1):
            grid[r : r + size, c : c + size] = color
            _reserve(mask, r, c, size, size)
            return {"side": side, "bbox": (r, c, r + size - 1, c + size - 1), "area": size * size, "color": color}
    return None


def _draw_frame(grid: np.ndarray, frame: dict, color: int) -> None:
    r0, c0, r1, c1 = frame["bbox"]
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    gr, gc = frame["gap"]
    grid[gr, gc] = BG
    frame["color"] = int(color)


def _place_frame(mask: np.ndarray, direction: str) -> dict | None:
    H, W = mask.shape
    for _ in range(200):
        fh = random.randint(*FRAME_H_RANGE)
        fw = random.randint(*FRAME_W_RANGE)
        r = random.randint(5, H - fh - 6)
        c = random.randint(5, W - fw - 6)
        if not _can_place(mask, r, c, fh, fw, pad=1):
            continue

        r0, c0, r1, c1 = r, c, r + fh - 1, c + fw - 1
        if direction == "up":
            gap = (r0, random.randint(c0 + 1, c1 - 1))
        elif direction == "down":
            gap = (r1, random.randint(c0 + 1, c1 - 1))
        elif direction == "left":
            gap = (random.randint(r0 + 1, r1 - 1), c0)
        else:
            gap = (random.randint(r0 + 1, r1 - 1), c1)

        _reserve(mask, r, c, fh, fw)
        return {"bbox": (r0, c0, r1, c1), "gap": gap, "direction": direction}
    return None


def _expected_output(grid: np.ndarray, frames: list[dict], blocks: list[dict]) -> np.ndarray:
    out = grid.copy()
    by_side: dict[str, list[dict]] = {side: [] for side in ("top", "bottom", "left", "right")}
    for block in blocks:
        by_side[block["side"]].append(block)
    for side in by_side:
        by_side[side].sort(key=lambda b: (-int(b["area"]), int(b["bbox"][0]), int(b["bbox"][1])))

    for direction in DIRS:
        side = DIR_TO_SIDE[direction]
        group = [f for f in frames if f["direction"] == direction]
        group.sort(key=lambda f: (int(f["bbox"][1]), int(f["bbox"][0])))
        for frame, block in zip(group, by_side[side], strict=False):
            r0, c0, r1, c1 = frame["bbox"]
            out[r0 + 1 : r1, c0 + 1 : c1] = int(block["color"])
    return out


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_SIZE_RANGE)
    W = random.randint(*GRID_SIZE_RANGE)
    grid = np.zeros((H, W), dtype=int)
    occupied = np.zeros((H, W), dtype=bool)

    extra_dirs = set(random.sample(list(DIRS), random.randint(0, len(DIRS))))
    counts = {d: 1 + int(d in extra_dirs) for d in DIRS}

    total_blocks = sum(counts.values())
    block_colors = random.sample(range(1, 10), total_blocks)
    color_i = 0
    blocks: list[dict] = []
    for direction in DIRS:
        side = DIR_TO_SIDE[direction]
        sizes = [3, 2] if counts[direction] == 2 else [random.choice([2, 3])]
        random.shuffle(sizes)
        for size in sizes:
            block = _place_block(grid, occupied, side, size, block_colors[color_i])
            color_i += 1
            if block is None:
                return None
            blocks.append(block)

    frames: list[dict] = []
    for direction in DIRS:
        for _ in range(counts[direction]):
            frame = _place_frame(occupied, direction)
            if frame is None:
                return None
            frames.append(frame)

    by_side: dict[str, list[dict]] = {side: [] for side in ("top", "bottom", "left", "right")}
    for block in blocks:
        by_side[block["side"]].append(block)
    for side in by_side:
        by_side[side].sort(key=lambda b: (-int(b["area"]), int(b["bbox"][0]), int(b["bbox"][1])))

    for direction in DIRS:
        side = DIR_TO_SIDE[direction]
        group = [f for f in frames if f["direction"] == direction]
        group.sort(key=lambda f: (int(f["bbox"][1]), int(f["bbox"][0])))
        for frame, block in zip(group, by_side[side], strict=False):
            choices = [c for c in range(1, 10) if c != int(block["color"])]
            _draw_frame(grid, frame, random.choice(choices))

    expected = _expected_output(grid, frames, blocks)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(actual, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid seed1030756833_7_1 input")
