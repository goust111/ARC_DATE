import random
from typing import Dict, List, Tuple

import numpy as np


BG = 0
START_COLOR = 2
PATH_COLOR = 3
RAIL_COLORS = [1, 4, 5, 6, 7, 8, 9]

WORK_H = 24
WORK_W = 24
NUM_RAILS_RANGE = (3, 6)
GAP_RANGE = (1, 3)
RAIL_LEN_RANGE = (2, 5)
START_MARGIN = 5
CROP_MARGIN = 2
GEN_RETRIES = 400


def _line_cells(start: Tuple[int, int], dr: int, dc: int, length: int) -> List[Tuple[int, int]]:
    return [(start[0] + dr * k, start[1] + dc * k) for k in range(1, length + 1)]


def _rail_cells(start: Tuple[int, int], dr: int, dc: int, gap: int, rail_len: int) -> List[Tuple[int, int]]:
    cells = []
    for k in range(gap + 1, gap + rail_len + 1):
        cells.append((start[0] + dr * k, start[1] + dc * k))
    return cells


def _extract_rails(grid: np.ndarray) -> Dict[int, dict]:
    body = grid[:-1]
    rails: Dict[int, dict] = {}
    for color in sorted(set(map(int, body.ravel())) - {BG, START_COLOR}):
        pts = np.argwhere(body == color)
        rows = sorted(set(int(r) for r, _ in pts))
        cols = sorted(set(int(c) for _, c in pts))
        if len(rows) == 1:
            row = rows[0]
            c0, c1 = min(cols), max(cols)
            assert cols == list(range(c0, c1 + 1))
            rails[color] = {"orientation": "h", "row": row, "c0": c0, "c1": c1}
        else:
            assert len(cols) == 1
            col = cols[0]
            r0, r1 = min(rows), max(rows)
            assert rows == list(range(r0, r1 + 1))
            rails[color] = {"orientation": "v", "col": col, "r0": r0, "r1": r1}
    return rails


def main(grid: np.ndarray) -> np.ndarray:
    """
    Follow a bottom-row color script to visit matching guide rails and draw the routed polyline.

    Input structure:
    - The background is black.
    - The bottom row contains a one-pixel-tall instruction strip: its non-black pixels,
      read left to right, are the rail colors to visit in order.
    - Above the strip are several single-color straight rails, each one cell thick and
      either horizontal or vertical, plus one unique red start pixel.

    Rule:
    - Read the instruction colors from the bottom row from left to right.
    - For each color, find the rail with that color. If the rail is horizontal, the brush
      moves horizontally from its current point until it reaches the near endpoint of that
      rail; if the rail is vertical, it moves vertically in the same way. The traversed gap
      cells are painted green.
    - After reaching the rail, continue along that rail to its opposite endpoint. Rail cells
      keep their rail color, and the brush stops on the far endpoint.
    - After all instructions are processed, erase the entire instruction strip to black.

    Key insight:
    - The bottom row gives the visit order, while each visited rail's orientation determines
      whether the next path segment is horizontal or vertical.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 2

    commands = [int(v) for v in g[-1] if int(v) != BG]
    start_pts = np.argwhere(g[:-1] == START_COLOR)
    assert len(start_pts) == 1
    current = (int(start_pts[0, 0]), int(start_pts[0, 1]))
    rails = _extract_rails(g)

    out = g.copy()
    out[-1, :] = BG

    for color in commands:
        rail = rails[color]
        if rail["orientation"] == "h":
            row = rail["row"]
            assert current[0] == row
            if current[1] < rail["c0"]:
                for c in range(current[1] + 1, rail["c0"]):
                    out[row, c] = PATH_COLOR
                current = (row, rail["c1"])
            else:
                assert current[1] > rail["c1"]
                for c in range(rail["c1"] + 1, current[1]):
                    out[row, c] = PATH_COLOR
                current = (row, rail["c0"])
        else:
            col = rail["col"]
            assert current[1] == col
            if current[0] < rail["r0"]:
                for r in range(current[0] + 1, rail["r0"]):
                    out[r, col] = PATH_COLOR
                current = (rail["r1"], col)
            else:
                assert current[0] > rail["r1"]
                for r in range(rail["r1"] + 1, current[0]):
                    out[r, col] = PATH_COLOR
                current = (rail["r0"], col)

    return out


def _build_chain(rng: random.Random) -> Tuple[Tuple[int, int], list]:
    num_rails = rng.randint(*NUM_RAILS_RANGE)
    start = (
        rng.randint(START_MARGIN, WORK_H - START_MARGIN - 1),
        rng.randint(START_MARGIN, WORK_W - START_MARGIN - 1),
    )
    current = start
    used = {start}
    segments = []

    for _ in range(num_rails):
        candidates = []
        dirs = [(0, 1, "h"), (0, -1, "h"), (1, 0, "v"), (-1, 0, "v")]
        rng.shuffle(dirs)
        gaps = list(range(GAP_RANGE[0], GAP_RANGE[1] + 1))
        lens = list(range(RAIL_LEN_RANGE[0], RAIL_LEN_RANGE[1] + 1))
        rng.shuffle(gaps)
        rng.shuffle(lens)

        for dr, dc, orientation in dirs:
            for gap in gaps:
                for rail_len in lens:
                    total_len = gap + rail_len
                    cells = _line_cells(current, dr, dc, total_len)
                    if any(not (0 <= r < WORK_H and 0 <= c < WORK_W) for r, c in cells):
                        continue
                    if any(cell in used for cell in cells):
                        continue
                    near = cells[gap]
                    far = cells[-1]
                    candidates.append(
                        {
                            "orientation": orientation,
                            "dr": dr,
                            "dc": dc,
                            "gap": gap,
                            "rail_len": rail_len,
                            "near": near,
                            "far": far,
                            "all_cells": cells,
                            "rail_cells": cells[gap:],
                        }
                    )

        if not candidates:
            raise RuntimeError("dead end while building chain")

        segment = rng.choice(candidates)
        segments.append(segment)
        used.update(segment["all_cells"])
        current = segment["far"]

    orientations = {seg["orientation"] for seg in segments}
    if len(orientations) < 2:
        raise RuntimeError("degenerate single-axis chain")
    return start, segments


def _crop_body(body: np.ndarray, rng: random.Random, command_count: int) -> np.ndarray:
    pts = np.argwhere(body != BG)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    top = rng.randint(0, min(CROP_MARGIN, int(r0)))
    bottom = rng.randint(0, min(CROP_MARGIN, int(body.shape[0] - 1 - r1)))
    left = rng.randint(0, min(CROP_MARGIN, int(c0)))
    right = rng.randint(0, min(CROP_MARGIN, int(body.shape[1] - 1 - c1)))

    crop = body[r0 - top : r1 + bottom + 1, c0 - left : c1 + right + 1]
    min_width = command_count + rng.randint(0, 3)
    if crop.shape[1] >= min_width:
        return crop

    final_w = min_width
    padded = np.full((crop.shape[0], final_w), BG, dtype=int)
    offset = rng.randint(0, final_w - crop.shape[1])
    padded[:, offset : offset + crop.shape[1]] = crop
    return padded


def _gen_attempt(rng: random.Random) -> np.ndarray:
    start, segments = _build_chain(rng)
    colors = rng.sample(RAIL_COLORS, len(segments))

    body = np.full((WORK_H, WORK_W), BG, dtype=int)
    body[start] = START_COLOR
    for color, segment in zip(colors, segments):
        for r, c in segment["rail_cells"]:
            body[r, c] = color

    crop = _crop_body(body, rng, len(colors))
    final = np.full((crop.shape[0] + 1, crop.shape[1]), BG, dtype=int)
    final[:-1] = crop

    cmd_start = rng.randint(0, final.shape[1] - len(colors))
    final[-1, cmd_start : cmd_start + len(colors)] = np.array(colors, dtype=int)
    return final


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        try:
            grid = _gen_attempt(rng)
        except RuntimeError:
            continue
        if grid.ndim == 2 and max(grid.shape) <= 30:
            return grid
    raise RuntimeError("failed to generate a valid input")
