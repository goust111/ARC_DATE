from __future__ import annotations

from collections import Counter, deque
import math
import random

import numpy as np


ORIENTATIONS = ("tl", "tr", "bl", "br")
CORNER_COORDS = {
    "tl": (0, 0),
    "tr": (0, -1),
    "bl": (-1, 0),
    "br": (-1, -1),
}
OPPOSITE_CORNER = {
    "tl": "br",
    "tr": "bl",
    "bl": "tr",
    "br": "tl",
}

BG_COLOR = 0
MARKER_COLORS = tuple(range(1, 10))
BODY_COLORS = tuple(range(1, 10))
GEN_RETRIES = 400
PLACE_RETRIES = 300


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    if BG_COLOR in values:
        return BG_COLOR
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(grid: np.ndarray, bg: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) != bg:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _missing_corner(patch: np.ndarray, bg: int) -> str:
    matches = []
    for name, (r, c) in CORNER_COORDS.items():
        if int(patch[r, c]) == bg:
            matches.append(name)
    if len(matches) != 1:
        raise ValueError("each block must have exactly one missing corner")
    return matches[0]


def _extract_blocks(grid: np.ndarray) -> tuple[int, list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    blocks: list[dict[str, object]] = []
    for comp in _components(g, bg):
        pts = np.array(comp, dtype=int)
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        side_h = int(r1 - r0 + 1)
        side_w = int(c1 - c0 + 1)
        if side_h != side_w:
            raise ValueError("each block bbox must be square")
        patch = np.full((side_h, side_h), bg, dtype=int)
        for r, c in comp:
            patch[r - r0, c - c0] = int(g[r, c])
        if int(np.count_nonzero(patch != bg)) != side_h * side_h - 1:
            raise ValueError("each block must occupy a full square except for one corner")
        blocks.append(
            {
                "patch": patch,
                "side": side_h,
                "orientation": _missing_corner(patch, bg),
            }
        )
    if not blocks:
        raise ValueError("input must contain at least one block")
    return bg, blocks


def _rotate_to_target(patch: np.ndarray, bg: int, target: str) -> np.ndarray:
    for k in range(4):
        rotated = np.rot90(patch, k)
        if _missing_corner(rotated, bg) == target:
            return rotated
    raise ValueError("rotation to target orientation failed")


def _center_offset(outer: int, inner: int) -> int:
    return (outer - inner) // 2


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate corner-missing square blocks to a common orientation and nest them at the center.

    Input structure:
    - Black background surrounds several disconnected multicolor blocks.
    - Each block is an SxS square with exactly one corner cell removed, leaving an L-like square-with-hole shape.
    - The main body of each block has its own color, and a shared marker color sits on the corner opposite the missing one.

    Rule:
    - For each block, inspect its square bounding box and find which of the four corners is the only background corner; that is the missing-corner orientation.
    - Count these orientations across all blocks and choose the majority orientation as the common target.
    - Rotate every non-matching block by 90, 180, or 270 degrees so its missing corner faces the target orientation; the marker rotates with the block.
    - Create an output grid whose size equals the largest block side length, place the largest rotated block at the center, then place smaller rotated blocks on top in descending size order with the same center.

    Key insight:
    - The only background corner inside each square bbox reveals the block's orientation, and the global target is hidden as a majority vote over those local orientations before centered size-ordered overlaying.
    """
    bg, blocks = _extract_blocks(np.asarray(grid, dtype=int))
    counts = Counter(str(block["orientation"]) for block in blocks)
    target = max(ORIENTATIONS, key=lambda name: (counts[name], -ORIENTATIONS.index(name)))
    max_side = max(int(block["side"]) for block in blocks)
    out = np.full((max_side, max_side), bg, dtype=int)

    for block in sorted(blocks, key=lambda item: int(item["side"]), reverse=True):
        patch = _rotate_to_target(np.asarray(block["patch"], dtype=int), bg, target)
        side = int(block["side"])
        top = _center_offset(max_side, side)
        left = _center_offset(max_side, side)
        mask = patch != bg
        window = out[top : top + side, left : left + side]
        window[mask] = patch[mask]

    return out


def _make_block(side: int, body_color: int, marker_color: int, orientation: str) -> np.ndarray:
    patch = np.full((side, side), body_color, dtype=int)
    miss_r, miss_c = CORNER_COORDS[orientation]
    opp_r, opp_c = CORNER_COORDS[OPPOSITE_CORNER[orientation]]
    patch[miss_r, miss_c] = BG_COLOR
    patch[opp_r, opp_c] = marker_color
    return patch


def _choose_sizes(rng: random.Random, count: int) -> list[int]:
    pools = (
        [2, 4, 6, 8, 10],
        [3, 5, 7, 9, 11],
    )
    pool = list(rng.choice(pools))
    rng.shuffle(pool)
    return sorted(pool[:count], reverse=True)


def _choose_orientations(rng: random.Random, count: int) -> tuple[str, list[str]]:
    target = rng.choice(ORIENTATIONS)
    others = [name for name in ORIENTATIONS if name != target]
    if count == 5:
        orientations = [target, target] + others
    else:
        orientations = [target, target] + rng.sample(others, 2)
    rng.shuffle(orientations)
    return target, orientations


def _canvas_shape(rng: random.Random, sizes: list[int]) -> tuple[int, int]:
    area = sum(side * side for side in sizes)
    largest = max(sizes)
    minimum = max(largest + 3, int(math.ceil(math.sqrt(area * 2.0))))
    height = rng.randint(minimum, 30)
    width = rng.randint(minimum, 30)
    return height, width


def _try_place(grid: np.ndarray, blocked: np.ndarray, patch: np.ndarray, rng: random.Random) -> bool:
    h, w = grid.shape
    side = patch.shape[0]
    if h - side - 1 < 1 or w - side - 1 < 1:
        return False
    for _ in range(PLACE_RETRIES):
        top = rng.randint(1, h - side - 1)
        left = rng.randint(1, w - side - 1)
        if blocked[top : top + side, left : left + side].any():
            continue
        mask = patch != BG_COLOR
        window = grid[top : top + side, left : left + side]
        window[mask] = patch[mask]
        blocked[max(0, top - 1) : min(h, top + side + 1), max(0, left - 1) : min(w, left + side + 1)] = True
        return True
    return False


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    block_count = rng.choice((4, 5))
    _, orientations = _choose_orientations(rng, block_count)
    sizes = _choose_sizes(rng, block_count)
    marker_color = rng.choice(MARKER_COLORS)
    body_palette = [color for color in BODY_COLORS if color != marker_color]
    body_colors = rng.sample(body_palette, block_count)

    blocks = [
        _make_block(side, body_color, marker_color, orientation)
        for side, body_color, orientation in zip(sizes, body_colors, orientations)
    ]

    height, width = _canvas_shape(rng, sizes)
    grid = np.full((height, width), BG_COLOR, dtype=int)
    blocked = np.zeros((height, width), dtype=bool)

    for patch in blocks:
        if not _try_place(grid, blocked, patch, rng):
            return None

    try:
        _, extracted = _extract_blocks(grid)
    except ValueError:
        return None

    if len(extracted) != block_count:
        return None
    extracted_sizes = sorted(int(block["side"]) for block in extracted)
    if len(set(extracted_sizes)) != block_count:
        return None
    orient_counts = Counter(str(block["orientation"]) for block in extracted)
    common = orient_counts.most_common(2)
    if common[0][1] <= 1:
        return None
    if len(common) > 1 and common[0][1] == common[1][1]:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid missing-corner block puzzle")
