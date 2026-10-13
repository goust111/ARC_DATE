from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np


_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, crop_bbox, find_components, get_bg, paste, random_free_location


GEN_ATTEMPTS = 500
MIN_CHAIN_LEN = 2
MAX_CHAIN_LEN = 7
FRAME_SIDE_RANGE = (3, 7)


def _is_hollow_square(region: np.ndarray, color: int) -> bool:
    h, w = map(int, region.shape)
    if h != w or h < 3:
        return False
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return np.array_equal(region == int(color), border)


def _extract_frames(grid: np.ndarray, bg: int) -> dict[int, tuple[int, int, int, int]]:
    frames: dict[int, tuple[int, int, int, int]] = {}
    for comp in find_components(grid, bg=bg, connectivity=4, monochromatic=True):
        color = int(comp["color"])
        region = crop_bbox(grid, comp["bbox"])
        if _is_hollow_square(region, color):
            assert color not in frames, "each frame color must be unique"
            frames[color] = tuple(map(int, comp["bbox"]))
    assert frames, "expected at least one hollow square frame"
    return frames


def _frame_pointer(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    bg: int,
    frame_colors: set[int],
    self_color: int,
) -> int | None:
    r0, c0, r1, c1 = map(int, bbox)
    inside = grid[r0 + 1 : r1, c0 + 1 : c1]
    marks = np.argwhere(inside != bg)
    assert marks.shape[0] <= 1, "each frame interior must contain at most one control pixel"
    if marks.shape[0] == 0:
        return None
    rr, cc = map(int, marks[0])
    color = int(inside[rr, cc])
    assert color in frame_colors, "control pixel must point to another frame color"
    assert color != int(self_color), "frame cannot point to itself"
    return color


def _chain_colors_from_grid(grid: np.ndarray, bg: int, frames: dict[int, tuple[int, int, int, int]]) -> list[int]:
    frame_colors = set(frames)
    pointers = {
        color: _frame_pointer(grid, bbox, bg, frame_colors, color)
        for color, bbox in frames.items()
    }
    referenced = {pointer for pointer in pointers.values() if pointer is not None}
    heads = list(frame_colors - referenced)
    tails = [color for color, pointer in pointers.items() if pointer is None]
    assert len(heads) == 1, "exactly one frame color must be unreferenced"
    assert len(tails) == 1, "exactly one frame must have no control pixel"

    chain: list[int] = []
    seen: set[int] = set()
    cur: int | None = heads[0]
    while cur is not None:
        assert cur not in seen, "pointer chain must be acyclic"
        seen.add(cur)
        chain.append(cur)
        cur = pointers[cur]

    assert len(chain) == len(frames), "pointer chain must cover all frames"
    return chain


def _draw_chain_output(chain_colors: list[int], bg: int) -> np.ndarray:
    n = len(chain_colors)
    side = 4 * n - 3
    out = np.full((side, side), int(bg), dtype=int)
    for index, color in enumerate(chain_colors):
        offset = 2 * index
        ring_side = side - 4 * index
        if ring_side == 1:
            out[offset, offset] = int(color)
            continue
        r0 = c0 = offset
        r1 = c1 = offset + ring_side - 1
        out[r0, c0 : c1 + 1] = int(color)
        out[r1, c0 : c1 + 1] = int(color)
        out[r0 : r1 + 1, c0] = int(color)
        out[r0 : r1 + 1, c1] = int(color)
    return out


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Reorder scattered hollow square frames by their interior color pointers, then
    redraw the chain as concentric square rings from outermost to innermost.
    """
    grid = np.asarray(input_grid, dtype=int)
    bg = get_bg(grid)
    frames = _extract_frames(grid, bg)
    chain_colors = _chain_colors_from_grid(grid, bg, frames)
    return _draw_chain_output(chain_colors, bg)


def _make_frame_patch(side: int, frame_color: int, bg: int, next_color: int | None) -> np.ndarray:
    patch = np.full((side, side), int(bg), dtype=int)
    patch[0, :] = int(frame_color)
    patch[-1, :] = int(frame_color)
    patch[:, 0] = int(frame_color)
    patch[:, -1] = int(frame_color)
    if next_color is not None:
        patch[random.randint(1, side - 2), random.randint(1, side - 2)] = int(next_color)
    return patch


def _make_instance() -> tuple[np.ndarray, np.ndarray]:
    for _ in range(GEN_ATTEMPTS):
        bg = random.choice(Color.ALL_COLORS)
        palette = [color for color in Color.ALL_COLORS if color != bg]
        chain_len = random.randint(MIN_CHAIN_LEN, min(MAX_CHAIN_LEN, len(palette)))
        chain_colors = random.sample(palette, chain_len)
        sides = [random.randint(*FRAME_SIDE_RANGE) for _ in range(chain_len)]

        reserve_area = sum((side + 2) ** 2 for side in sides)
        min_dim = max(max(sides) + 2, int(np.ceil(np.sqrt(reserve_area))) + 2)
        if min_dim > 30:
            continue
        height = random.randint(min_dim, 30)
        width = random.randint(min_dim, 30)

        grid = np.full((height, width), int(bg), dtype=int)
        occupied = np.zeros((height, width), dtype=int)

        specs = []
        for index, color in enumerate(chain_colors):
            next_color = chain_colors[index + 1] if index + 1 < chain_len else None
            patch = _make_frame_patch(sides[index], color, bg, next_color)
            specs.append({"side": sides[index], "patch": patch})

        try:
            order = list(range(chain_len))
            random.shuffle(order)
            order.sort(key=lambda idx: int(specs[idx]["side"]), reverse=True)
            for idx in order:
                patch = np.asarray(specs[idx]["patch"], dtype=int)
                reserve = np.ones_like(patch, dtype=int)
                r0, c0 = random_free_location(occupied, reserve, transparent=0, border=1, padding=1)
                paste(occupied, reserve, r0, c0)
                paste(grid, patch, r0, c0, transparent=bg)
        except ValueError:
            continue

        expected = _draw_chain_output(chain_colors, bg)
        return grid, expected

    raise RuntimeError("failed to generate a valid pointer-chain instance")


def generate_input() -> np.ndarray:
    return _make_instance()[0]
