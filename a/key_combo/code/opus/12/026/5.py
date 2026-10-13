from __future__ import annotations

import numpy as np


BACKGROUND = 0
OUTLINE = 1
FILL = 8

QUADRANT_ORDER = (
    ("lu", 0, 0),
    ("ru", 0, 1),
    ("ll", 1, 0),
    ("rl", 1, 1),
)

OPENING_WEIGHTS = ("top", "left", "right", "bottom", "top", "left")
MAX_GENERATION_ATTEMPTS = 400


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise ValueError("empty mask")
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _mirror_mosaic(grid: np.ndarray) -> np.ndarray:
    return np.block(
        [
            [np.rot90(grid, 2), np.flipud(grid)],
            [np.fliplr(grid), grid],
        ]
    )


def _analyze_cup(grid: np.ndarray) -> dict[str, object]:
    nonzero = grid != BACKGROUND
    if not np.any(nonzero):
        raise ValueError("no cup found")

    colors = {int(v) for v in np.unique(grid[nonzero])}
    if len(colors) != 1:
        raise ValueError("expected exactly one outline color")
    outline_color = next(iter(colors))

    r0, c0, r1, c1 = _bbox(nonzero)
    local = grid[r0 : r1 + 1, c0 : c1 + 1]
    outline_mask = local == outline_color
    if np.any(local[1:-1, 1:-1] != BACKGROUND):
        raise ValueError("cup outline should have empty interior")

    fill_mask = local == BACKGROUND
    top = np.flatnonzero(fill_mask[0, :])
    bottom = np.flatnonzero(fill_mask[-1, :])
    left = np.flatnonzero(fill_mask[:, 0])
    right = np.flatnonzero(fill_mask[:, -1])
    side_map = {
        "top": top,
        "bottom": bottom,
        "left": left,
        "right": right,
    }
    candidates = {name: idxs for name, idxs in side_map.items() if len(idxs) > 0}
    if len(candidates) != 1:
        raise ValueError("expected exactly one opening side")

    side, idxs = next(iter(candidates.items()))
    if len(idxs) not in {1, 2}:
        raise ValueError("opening width must be 1 or 2")
    if len(idxs) == 2 and int(idxs[1]) != int(idxs[0]) + 1:
        raise ValueError("opening must be contiguous")

    if side == "top":
        hole = [(0, int(c)) for c in idxs]
        direction = (-1, 0)
    elif side == "bottom":
        hole = [(local.shape[0] - 1, int(c)) for c in idxs]
        direction = (1, 0)
    elif side == "left":
        hole = [(int(r), 0) for r in idxs]
        direction = (0, -1)
    else:
        hole = [(int(r), local.shape[1] - 1) for r in idxs]
        direction = (0, 1)

    return {
        "bbox": (r0, c0, r1, c1),
        "outline_color": outline_color,
        "local": local,
        "fill_mask": fill_mask,
        "hole": hole,
        "direction": direction,
        "opening_side": side,
    }


def _paint_ray(out: np.ndarray, start: tuple[int, int], direction: tuple[int, int]) -> None:
    r, c = start
    dr, dc = direction
    r += dr
    c += dc
    while 0 <= r < out.shape[0] and 0 <= c < out.shape[1]:
        if int(out[r, c]) != BACKGROUND:
            break
        out[r, c] = FILL
        r += dr
        c += dc


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror one cup into four quadrants, fill every mirrored cup, and emit teal rays from the mirrored openings in draw order.

    Input structure:
    - A black rectangular grid contains exactly one blue cup outline.
    - The cup is a hollow rectangular border with one contiguous opening of width 1 or 2 on exactly one side.
    - The cup stays at least one cell away from every input border so the mirrored quadrants have empty space around them.

    Rule:
    - Build a 2H x 2W mirror mosaic with the original in the lower-right, its left-right mirror in the lower-left, its up-down mirror in the upper-right, and its 180-degree rotation in the upper-left.
    - For each mirrored cup, determine the opening side from that mirrored outline, fill every black cell inside its bounding box with teal, and treat the opening cells as the ray origins.
    - Draw teal rays outward from each opening cell, one cup at a time in the order upper-left, upper-right, lower-left, lower-right; each ray stops before the first non-black cell, including earlier teal rays.

    Key insight:
    - The mirror placement does more than duplicate geometry: it changes which side is open, and that mirrored opening side is exactly what decides the ray direction and later collisions.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("input must be a 2D grid")

    H, W = g.shape
    base = _mirror_mosaic(g)
    out = base.copy()

    for _name, qr, qc in QUADRANT_ORDER:
        sr = qr * H
        sc = qc * W
        quadrant = base[sr : sr + H, sc : sc + W]
        info = _analyze_cup(quadrant)
        r0, c0, r1, c1 = info["bbox"]
        fill_mask = info["fill_mask"]
        hole = info["hole"]
        direction = info["direction"]

        sub_out = out[sr + r0 : sr + r1 + 1, sc + c0 : sc + c1 + 1]
        sub_out[fill_mask] = FILL

        for hr, hc in hole:
            _paint_ray(out, (sr + r0 + hr, sc + c0 + hc), direction)

    return out


def _make_cup_sprite(
    box_h: int,
    box_w: int,
    side: str,
    hole_len: int,
    hole_start: int,
) -> np.ndarray:
    sprite = np.zeros((box_h, box_w), dtype=int)
    sprite[0, :] = OUTLINE
    sprite[-1, :] = OUTLINE
    sprite[:, 0] = OUTLINE
    sprite[:, -1] = OUTLINE

    if side == "top":
        sprite[0, hole_start : hole_start + hole_len] = BACKGROUND
    elif side == "bottom":
        sprite[-1, hole_start : hole_start + hole_len] = BACKGROUND
    elif side == "left":
        sprite[hole_start : hole_start + hole_len, 0] = BACKGROUND
    else:
        sprite[hole_start : hole_start + hole_len, -1] = BACKGROUND
    return sprite


def _sample_hole(rng: np.random.Generator, side: str, box_h: int, box_w: int) -> tuple[int, int]:
    span = box_w - 2 if side in {"top", "bottom"} else box_h - 2
    if span < 1:
        raise ValueError("cup side too short for an opening")
    hole_len = int(rng.choice([1, 2])) if span >= 2 else 1
    hole_len = min(hole_len, span)
    start = int(rng.integers(1, span - hole_len + 2))
    return hole_len, start


def _validate_input(grid: np.ndarray) -> dict[str, object]:
    if grid.ndim != 2 or max(grid.shape) > 15:
        raise ValueError("input must fit inside 15x15 before doubling")
    info = _analyze_cup(grid)
    r0, c0, r1, c1 = info["bbox"]
    H, W = grid.shape
    if r0 < 1 or c0 < 1 or r1 > H - 2 or c1 > W - 2:
        raise ValueError("cup must stay away from the border")
    return info


def _ray_cells(out: np.ndarray, start: tuple[int, int], direction: tuple[int, int]) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    r, c = start
    dr, dc = direction
    r += dr
    c += dc
    while 0 <= r < out.shape[0] and 0 <= c < out.shape[1] and int(out[r, c]) == FILL:
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def _validate_output_semantics(inp: np.ndarray, out: np.ndarray) -> None:
    H, W = inp.shape
    if out.shape != (2 * H, 2 * W):
        raise ValueError("output shape must be doubled")

    base = _mirror_mosaic(inp)
    if not np.array_equal(out[base != BACKGROUND], base[base != BACKGROUND]):
        raise ValueError("outline pixels must stay unchanged in output")

    inward_collision_seen = False
    ray_pixel_count = 0

    for _name, qr, qc in QUADRANT_ORDER:
        sr = qr * H
        sc = qc * W
        quadrant = base[sr : sr + H, sc : sc + W]
        info = _analyze_cup(quadrant)
        r0, c0, r1, c1 = info["bbox"]
        fill_mask = info["fill_mask"]
        hole = info["hole"]
        direction = info["direction"]

        cup_out = out[sr + r0 : sr + r1 + 1, sc + c0 : sc + c1 + 1]
        if not np.all(cup_out[fill_mask] == FILL):
            raise ValueError("cup interior and opening must be filled")

        for hr, hc in hole:
            start = (sr + r0 + hr, sc + c0 + hc)
            cells = _ray_cells(out, start, direction)
            ray_pixel_count += len(cells)
            dr, dc = direction
            stop_r = start[0] + (len(cells) + 1) * dr
            stop_c = start[1] + (len(cells) + 1) * dc
            if 0 <= stop_r < out.shape[0] and 0 <= stop_c < out.shape[1]:
                if len(cells) > 0 and int(out[stop_r, stop_c]) != BACKGROUND:
                    inward_collision_seen = True

    if ray_pixel_count < 4:
        raise ValueError("rays should remain visually detectable")

    opening_side = _validate_input(inp)["opening_side"]
    if opening_side in {"top", "left"} and not inward_collision_seen:
        raise ValueError("inward-opening variants should exhibit blocking or meeting rays")


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()

    for _ in range(MAX_GENERATION_ATTEMPTS):
        box_h = int(rng.integers(4, 7))
        box_w = int(rng.integers(4, 8))
        opening_side = str(rng.choice(OPENING_WEIGHTS))
        hole_len, hole_start = _sample_hole(rng, opening_side, box_h, box_w)

        extra_h = int(rng.integers(2, 6))
        extra_w = int(rng.integers(2, 6))
        H = box_h + extra_h
        W = box_w + extra_w
        if max(H, W) > 15:
            continue

        max_top = H - box_h - 1
        max_left = W - box_w - 1
        if max_top < 1 or max_left < 1:
            continue

        top = int(rng.integers(1, max_top + 1))
        left = int(rng.integers(1, max_left + 1))

        sprite = _make_cup_sprite(box_h, box_w, opening_side, hole_len, hole_start)
        grid = np.zeros((H, W), dtype=int)
        grid[top : top + box_h, left : left + box_w] = sprite

        info = _validate_input(grid)
        out = main(grid)
        _validate_output_semantics(grid, out)

        bbox_area = (info["bbox"][2] - info["bbox"][0] + 1) * (info["bbox"][3] - info["bbox"][1] + 1)
        if np.count_nonzero(out == FILL) <= bbox_area:
            continue
        return grid

    raise RuntimeError("failed to generate a valid mirrored-cup input")
