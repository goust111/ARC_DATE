from __future__ import annotations

import math
import random
from typing import Iterable

import numpy as np


BG = 8
PALETTE = [0, 1, 2, 3, 4, 5, 6, 7, 9]
GEN_RETRIES = 400

# Each configuration is chosen so that:
# 1. The intended cavity sizes are unique.
# 2. A size-only solver still has at least two perfect matchings.
# 3. The intended hole-count scales fit inside 30x30 once framed.
CONFIGS = [
    {"bases": (4, 8), "scales": (2, 2)},
    {"bases": (4, 5, 8), "scales": (2, 1, 2)},
    {"bases": (4, 5, 8), "scales": (2, 2, 2)},
    {"bases": (4, 5, 8), "scales": (2, 3, 2)},
    {"bases": (4, 6, 8), "scales": (2, 1, 2)},
    {"bases": (4, 6, 8), "scales": (2, 2, 2)},
    {"bases": (3, 4, 5, 8), "scales": (1, 2, 1, 2)},
    {"bases": (3, 4, 5, 8), "scales": (1, 2, 2, 2)},
    {"bases": (3, 4, 5, 8), "scales": (1, 2, 3, 2)},
]


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1]


def _label_mask(mask: np.ndarray) -> tuple[np.ndarray, int]:
    labels = np.zeros(mask.shape, dtype=int)
    next_id = 0
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != 0:
                continue
            next_id += 1
            stack = [(r, c)]
            labels[r, c] = next_id
            while stack:
                cr, cc = stack.pop()
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and labels[nr, nc] == 0:
                        labels[nr, nc] = next_id
                        stack.append((nr, nc))
    return labels, next_id


def _monochrome_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            coords = np.array(cells, dtype=int)
            comps.append({"color": color, "coords": coords, "bbox": _bbox(coords)})
    return comps


def _frame_mask(side: int) -> np.ndarray:
    mask = np.zeros((side, side), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _is_frame_patch(patch: np.ndarray, color: int, bg: int) -> bool:
    h, w = patch.shape
    if h != w or h < 5:
        return False
    border = _frame_mask(h)
    return bool(np.all(patch[border] == color) and np.all(patch[~border] == bg))


def _count_holes(patch: np.ndarray, bg: int) -> int:
    labels, n = _label_mask(patch == bg)
    holes = 0
    for label_id in range(1, n + 1):
        coords = np.argwhere(labels == label_id)
        touches_edge = np.any(
            (coords[:, 0] == 0)
            | (coords[:, 0] == patch.shape[0] - 1)
            | (coords[:, 1] == 0)
            | (coords[:, 1] == patch.shape[1] - 1)
        )
        if not touches_edge:
            holes += 1
    return int(holes)


def _is_sprite_patch(patch: np.ndarray, color: int, bg: int) -> bool:
    h, w = patch.shape
    if h != w or h < 3:
        return False
    if not np.all(patch[0, :] == color):
        return False
    if not np.all(patch[-1, :] == color):
        return False
    if not np.all(patch[:, 0] == color):
        return False
    if not np.all(patch[:, -1] == color):
        return False
    if _is_frame_patch(patch, color, bg):
        return False
    return _count_holes(patch, bg) >= 1


def _scale_sprite(sprite: np.ndarray, factor: int) -> np.ndarray:
    return np.kron(sprite, np.ones((factor, factor), dtype=int))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count each sprite's holes, upscale by that count, and place the enlarged sprite into the matching hollow frame.

    Input structure:
    - The background is cyan, with several 1-pixel-thick square hollow frames whose interiors are empty background squares.
    - Elsewhere are several monochrome square sprites of different colors, each with one or more enclosed background holes.
    - Objects are separated from each other and each frame interior size matches exactly one sprite after the sprite is scaled by its own hole count.

    Rule:
    - Count the number of 4-connected background hole components inside each sprite.
    - Use that hole count n as the sprite's pixel-upscaling factor.
    - Find the hollow frame whose interior side length equals the scaled sprite side length, move the scaled sprite into that interior, and recolor the scaled holes to the frame color.
    - Clear the sprite from its original location and leave the rest of the grid unchanged.

    Key insight:
    - Frame size alone is insufficient because several global size-only matchings are possible; the hole count written inside each sprite is the unique scale instruction.
    """
    g = np.asarray(grid, dtype=int)
    comps = _monochrome_components(g, BG)

    frames = []
    sprites = []
    for comp in comps:
        color = int(comp["color"])
        box = comp["bbox"]
        patch = _crop(g, box)
        if _is_frame_patch(patch, color, BG):
            frames.append(
                {
                    "color": color,
                    "bbox": box,
                    "outer_side": int(patch.shape[0]),
                    "cavity_side": int(patch.shape[0] - 2),
                }
            )
        elif _is_sprite_patch(patch, color, BG):
            sprites.append(
                {
                    "color": color,
                    "bbox": box,
                    "patch": patch.copy(),
                    "side": int(patch.shape[0]),
                    "holes": _count_holes(patch, BG),
                    "coords": comp["coords"],
                }
            )

    assert frames and sprites, "expected both frames and sprites"
    assert len(frames) == len(sprites), "expected equal counts of frames and sprites"

    frame_by_cavity = {}
    for frame in frames:
        cavity = int(frame["cavity_side"])
        assert cavity not in frame_by_cavity, "frame interior sizes must be unique"
        frame_by_cavity[cavity] = frame

    assignments = []
    used_cavities = set()
    for sprite in sprites:
        target_side = int(sprite["side"]) * int(sprite["holes"])
        assert target_side in frame_by_cavity, "every sprite must fit exactly one frame after scaling"
        assert target_side not in used_cavities, "frame pairing must be one-to-one"
        used_cavities.add(target_side)
        assignments.append((sprite, frame_by_cavity[target_side]))

    out = g.copy()
    for sprite, _frame in assignments:
        for r, c in sprite["coords"]:
            out[int(r), int(c)] = BG

    for sprite, frame in assignments:
        scaled = _scale_sprite(sprite["patch"], int(sprite["holes"]))
        scaled = scaled.copy()
        scaled[scaled == BG] = int(frame["color"])
        r0, c0, _r1, _c1 = frame["bbox"]
        cavity_side = int(frame["cavity_side"])
        assert scaled.shape == (cavity_side, cavity_side)
        out[r0 + 1 : r0 + 1 + cavity_side, c0 + 1 : c0 + 1 + cavity_side] = scaled

    return out


def _pick_isolated_holes(side: int, count: int) -> list[tuple[int, int]] | None:
    cells = [(r, c) for r in range(1, side - 1) for c in range(1, side - 1)]
    for _ in range(120):
        random.shuffle(cells)
        chosen = []
        for r, c in cells:
            if all(abs(r - rr) + abs(c - cc) > 1 for rr, cc in chosen):
                chosen.append((r, c))
                if len(chosen) == count:
                    return chosen
    return None


def _make_sprite(side: int, holes: int, color: int) -> np.ndarray | None:
    patch = np.full((side, side), color, dtype=int)
    hole_cells = _pick_isolated_holes(side, holes)
    if hole_cells is None:
        return None
    for r, c in hole_cells:
        patch[r, c] = BG
    if _count_holes(patch, BG) != holes:
        return None
    return patch


def _make_frame(cavity_side: int, color: int) -> np.ndarray:
    side = cavity_side + 2
    patch = np.full((side, side), color, dtype=int)
    patch[1:-1, 1:-1] = BG
    return patch


def _can_place(occ: np.ndarray, r: int, c: int, h: int, w: int, pad: int = 1) -> bool:
    h0, w0 = occ.shape
    if r < 0 or c < 0 or r + h > h0 or c + w > w0:
        return False
    rr0 = max(0, r - pad)
    cc0 = max(0, c - pad)
    rr1 = min(h0, r + h + pad)
    cc1 = min(w0, c + w + pad)
    return not bool(occ[rr0:rr1, cc0:cc1].any())


def _place_patch(canvas: np.ndarray, occ: np.ndarray, patch: np.ndarray, pad: int = 1) -> tuple[int, int] | None:
    h, w = patch.shape
    positions = [
        (r, c)
        for r in range(canvas.shape[0] - h + 1)
        for c in range(canvas.shape[1] - w + 1)
        if _can_place(occ, r, c, h, w, pad=pad)
    ]
    if not positions:
        return None
    r, c = random.choice(positions)
    canvas[r : r + h, c : c + w] = patch
    occ[r : r + h, c : c + w] = True
    return r, c


def _count_size_only_matchings(sprite_sides: list[int], cavity_sides: list[int]) -> int:
    n = len(sprite_sides)
    used = [False] * n

    def dfs(i: int) -> int:
        if i == n:
            return 1
        total = 0
        for j in range(n):
            if used[j]:
                continue
            if cavity_sides[j] % sprite_sides[i] != 0:
                continue
            used[j] = True
            total += dfs(i + 1)
            used[j] = False
        return total

    return dfs(0)


def _expected_output(inp: np.ndarray, frames: list[dict], sprites: list[dict]) -> np.ndarray:
    out = inp.copy()
    for sprite in sprites:
        r0, c0, r1, c1 = sprite["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = BG
    for sprite, frame in zip(sprites, frames):
        scaled = _scale_sprite(sprite["patch"], int(sprite["holes"])).copy()
        scaled[scaled == BG] = int(frame["color"])
        r0, c0, _r1, _c1 = frame["bbox"]
        side = int(frame["cavity_side"])
        out[r0 + 1 : r0 + 1 + side, c0 + 1 : c0 + 1 + side] = scaled
    return out


def _build_instance() -> np.ndarray | None:
    config = random.choice(CONFIGS)
    bases = list(map(int, config["bases"]))
    scales = list(map(int, config["scales"]))
    cavity_sides = [b * s for b, s in zip(bases, scales)]

    sprite_colors = random.sample(PALETTE, len(bases))
    frame_colors = random.sample([c for c in PALETTE if c not in sprite_colors], len(bases))

    sprites = []
    for side, holes, color in zip(bases, scales, sprite_colors):
        patch = _make_sprite(int(side), int(holes), int(color))
        if patch is None:
            return None
        sprites.append({"side": int(side), "holes": int(holes), "color": int(color), "patch": patch})

    frames = [
        {
            "cavity_side": int(cavity),
            "outer_side": int(cavity + 2),
            "color": int(color),
            "patch": _make_frame(int(cavity), int(color)),
        }
        for cavity, color in zip(cavity_sides, frame_colors)
    ]

    size_only_matchings = _count_size_only_matchings(bases, cavity_sides)
    if size_only_matchings < 2:
        return None
    intended = [b * h for b, h in zip(bases, scales)]
    if len(set(intended)) != len(intended):
        return None

    frame_area = sum(f["outer_side"] ** 2 for f in frames)
    sprite_area = sum(s["side"] ** 2 for s in sprites)
    max_side = max([f["outer_side"] for f in frames] + [s["side"] for s in sprites])

    min_dim = max(max_side, int(math.ceil(math.sqrt(frame_area + sprite_area + 30))))
    candidate_dims = [
        (h, w)
        for h in range(min_dim, 31)
        for w in range(min_dim, 31)
        if h * w >= frame_area + sprite_area + 30
    ]
    if not candidate_dims:
        return None
    H, W = random.choice(candidate_dims)

    canvas = np.full((H, W), BG, dtype=int)
    occ = np.zeros((H, W), dtype=bool)

    frame_order = list(range(len(frames)))
    sprite_order = list(range(len(sprites)))
    random.shuffle(frame_order)
    random.shuffle(sprite_order)

    for idx in sorted(frame_order, key=lambda i: frames[i]["outer_side"], reverse=True):
        pos = _place_patch(canvas, occ, frames[idx]["patch"], pad=1)
        if pos is None:
            return None
        r, c = pos
        side = frames[idx]["outer_side"]
        frames[idx]["bbox"] = (r, c, r + side - 1, c + side - 1)

    for idx in sorted(sprite_order, key=lambda i: sprites[i]["side"], reverse=True):
        pos = _place_patch(canvas, occ, sprites[idx]["patch"], pad=1)
        if pos is None:
            return None
        r, c = pos
        side = sprites[idx]["side"]
        sprites[idx]["bbox"] = (r, c, r + side - 1, c + side - 1)

    # Keep expected pairing aligned with hole-count scaling, not with placement order.
    paired_frames = [None] * len(sprites)
    cavity_to_frame = {int(frame["cavity_side"]): frame for frame in frames}
    for i, sprite in enumerate(sprites):
        paired_frames[i] = cavity_to_frame[int(sprite["side"]) * int(sprite["holes"])]

    expected = _expected_output(canvas, paired_frames, sprites)

    actual = main(canvas)
    if not np.array_equal(actual, expected):
        return None

    # Semantic checks beyond self-consistency.
    if _count_size_only_matchings([s["side"] for s in sprites], [f["cavity_side"] for f in frames]) < 2:
        return None
    if np.array_equal(canvas, expected):
        return None
    if max(canvas.shape) > 30 or max(expected.shape) > 30:
        return None

    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _build_instance()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to construct a valid puzzle")
