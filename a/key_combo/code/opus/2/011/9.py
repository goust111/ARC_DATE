import random
from collections import Counter, deque
from typing import Dict, List, Tuple

import numpy as np


def _components(mask: np.ndarray) -> List[List[Tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: List[List[Tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: List[Tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _sorted_bars(counter: Counter, first_row: Dict[int, int]) -> List[int]:
    return sorted(counter.keys(), key=lambda color: (counter[color], first_row[color]))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Parse:
        Extract each isolated 4x4 non-black object and classify it as a hollow
        frame or a solid block by inspecting its 2x2 interior.
    Analyze:
        Count colors separately inside the frame class and inside the solid
        block class while also tracking each color's first input row.
    Transform:
        Sort both classwise color-frequency tables in ascending count order and
        convert them into left-aligned horizontal color bars.
    Return:
        Build the final histogram grid with frame bars on top, a gray separator
        row in the middle, and solid-block bars beneath it.
    """
    grid = np.asarray(grid, dtype=int)
    frames: Counter = Counter()
    blocks: Counter = Counter()
    first_row_frame: Dict[int, int] = {}
    first_row_block: Dict[int, int] = {}

    for comp in _components(grid != 0):
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        top, bottom = min(rows), max(rows)
        left, right = min(cols), max(cols)
        box = grid[top : bottom + 1, left : right + 1]
        color = int(box[box != 0][0])
        if box.shape != (4, 4):
            continue
        if np.all(box[1:3, 1:3] == 0):
            frames[color] += 1
            first_row_frame.setdefault(color, top)
        else:
            blocks[color] += 1
            first_row_block.setdefault(color, top)

    frame_order = _sorted_bars(frames, first_row_frame)
    block_order = _sorted_bars(blocks, first_row_block)
    width = max(max(frames.values()), max(blocks.values()))
    out = np.zeros((len(frame_order) + 1 + len(block_order), width), dtype=int)

    r = 0
    for color in frame_order:
        out[r, : frames[color]] = color
        r += 1
    out[r, :] = 5
    r += 1
    for color in block_order:
        out[r, : blocks[color]] = color
        r += 1
    return out


def generate_input() -> np.ndarray:
    h = 18
    w = 18
    slots = [(1, 1), (1, 7), (1, 13), (7, 1), (7, 7), (7, 13), (13, 1), (13, 7), (13, 13)]

    for _ in range(5000):
        grid = np.zeros((h, w), dtype=int)
        count = random.randint(6, 8)
        chosen = random.sample(slots, count)
        kinds = ["frame", "frame", "block", "block"] + [random.choice(["frame", "block"]) for _ in range(count - 4)]
        random.shuffle(kinds)
        colors = [random.choice([1, 2, 3, 4, 7]) for _ in range(count)]

        if len({colors[i] for i, kind in enumerate(kinds) if kind == "frame"}) < 2:
            continue
        if len({colors[i] for i, kind in enumerate(kinds) if kind == "block"}) < 2:
            continue

        for (top, left), kind, color in zip(chosen, kinds, colors):
            if kind == "frame":
                grid[top : top + 4, left : left + 4] = color
                grid[top + 1 : top + 3, left + 1 : left + 3] = 0
            else:
                grid[top : top + 4, left : left + 4] = color
        return grid
    raise RuntimeError("failed to generate a valid task-9 input")
