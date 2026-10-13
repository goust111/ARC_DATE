from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
FRAME_COLOR = 5
NON_FRAME_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
SPRITE_BAND = 4
BAND_GAP = 1
FRAME_GAP = 1
GEN_RETRIES = 300


SPRITE_MASKS: dict[int, tuple[tuple[str, ...], ...]] = {
    3: (
        ("111", "100", "111"),
        ("111", "110", "101"),
        ("111", "010", "101"),
        ("111", "101", "110"),
    ),
    4: (
        ("1111", "1000", "1110", "1001"),
        ("1111", "1010", "1011", "1001"),
        ("1111", "1100", "1010", "1001"),
        ("1111", "1001", "1110", "1101"),
    ),
}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each hollow square by rotating the matching outside sprite according to the anchor quadrant.

    Input structure:
    - Background is black, and several equal-size hollow square frames are drawn in grey with a one-pixel border.
    - Each frame interior contains exactly one colored anchor pixel; the anchor color is unique and its position lies in one of the four interior quadrants.
    - Outside the frames, small 3x3 or 4x4 multicolor sprites are scattered near the grid edge on black background.
    - Each sprite has one single-pixel identifier color at its top-left cell, and that color matches exactly one frame anchor.

    Rule:
    - Detect every hollow square frame and read the single anchor inside it.
    - Match the anchor color to the outside sprite whose top-left identifier pixel has the same color.
    - Rotate that sprite by 0 degrees for a top-left anchor, 90 degrees clockwise for top-right, 180 degrees for bottom-right, and 90 degrees counterclockwise for bottom-left.
    - Paste the rotated sprite into the frame interior with the anchor position used as the sprite's top-left corner, keeping the grey border unchanged and clearing the outside sprites to black.

    Key insight:
    - The anchor color chooses which template to use, while the anchor quadrant separately encodes the rotation that must be applied before stamping it into the frame.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)

    frames = _find_frames(g, bg)
    frame_border_mask = np.zeros(g.shape, dtype=bool)
    frame_area_mask = np.zeros(g.shape, dtype=bool)
    for frame in frames:
        frame_border_mask[frame["r0"] : frame["r1"] + 1, frame["c0"] : frame["c1"] + 1] |= frame["border_mask"]
        frame_area_mask[int(frame["r0"]) : int(frame["r1"]) + 1, int(frame["c0"]) : int(frame["c1"]) + 1] = True

    sprites = _find_sprites(g, bg, frame_area_mask)

    out = np.full_like(g, bg)
    out[frame_border_mask] = g[frame_border_mask]

    for frame in frames:
        sprite = sprites[int(frame["anchor_color"])]
        rot_k = _rotation_k_for_quadrant(frame)
        rotated = np.rot90(sprite, rot_k)
        _paste_inside_frame(out, rotated, frame, bg)

    return out


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC sample")


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_frames = rng.randint(2, 4)
    frame_size = rng.choice([10, 11, 12])
    arrangement = _choose_arrangement(n_frames, rng)
    band_side = rng.choice(["top", "bottom"])

    sprite_sizes = [rng.choice([3, 4]) for _ in range(n_frames)]
    min_frame_h = arrangement[0] * frame_size + (arrangement[0] - 1) * FRAME_GAP
    min_frame_w = arrangement[1] * frame_size + (arrangement[1] - 1) * FRAME_GAP
    min_h = SPRITE_BAND + BAND_GAP + min_frame_h
    min_w = max(min_frame_w, sum(sprite_sizes) + (n_frames - 1) * FRAME_GAP)
    if min_h > 30 or min_w > 30:
        return None

    height = rng.randint(min_h, 30)
    width = rng.randint(min_w, 30)
    grid = np.zeros((height, width), dtype=int)

    frame_positions = _place_frame_boxes(height, width, frame_size, arrangement, n_frames, band_side, rng)
    if frame_positions is None:
        return None

    sprite_specs = _build_sprite_specs(n_frames, sprite_sizes, rng)
    sprite_order = list(range(n_frames))
    rng.shuffle(sprite_order)
    sprite_boxes = _place_sprites(grid, [sprite_specs[i]["size"] for i in sprite_order], band_side, rng)
    if sprite_boxes is None:
        return None

    expected = np.zeros_like(grid)
    for box in frame_positions:
        r0, c0 = box
        _draw_frame(grid, r0, c0, frame_size)
        _draw_frame(expected, r0, c0, frame_size)

    interior_size = frame_size - 2
    for i, (r0, c0) in enumerate(frame_positions):
        spec = sprite_specs[i]
        quadrant = rng.choice(["tl", "tr", "br", "bl"])
        anchor_r, anchor_c = _choose_anchor_position(r0, c0, interior_size, spec["size"], quadrant, rng)
        grid[anchor_r, anchor_c] = spec["anchor_color"]
        rotated = np.rot90(spec["patch"], _rotation_k_from_name(quadrant))
        _paste_expected(expected, rotated, r0, c0, frame_size, anchor_r, anchor_c)

    for order_idx, spec_idx in enumerate(sprite_order):
        sr, sc = sprite_boxes[order_idx]
        patch = sprite_specs[spec_idx]["patch"]
        if np.any(grid[sr : sr + patch.shape[0], sc : sc + patch.shape[1]] != BG):
            return None
        mask = patch != BG
        grid[sr : sr + patch.shape[0], sc : sc + patch.shape[1]][mask] = patch[mask]

    try:
        actual = main(grid)
    except Exception:
        return None

    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(actual, grid):
        return None
    if max(grid.shape) > 30 or max(actual.shape) > 30:
        return None
    return grid


def _build_sprite_specs(n_frames: int, sprite_sizes: list[int], rng: random.Random) -> list[dict[str, np.ndarray | int]]:
    colors = rng.sample(NON_FRAME_COLORS, 2 * n_frames)
    anchor_colors = colors[:n_frames]
    body_colors = colors[n_frames:]
    specs = []
    for size, anchor_color, body_color in zip(sprite_sizes, anchor_colors, body_colors):
        patch = _make_sprite_patch(size, anchor_color, body_color, rng)
        specs.append(
            {
                "size": size,
                "anchor_color": anchor_color,
                "body_color": body_color,
                "patch": patch,
            }
        )
    return specs


def _make_sprite_patch(size: int, anchor_color: int, body_color: int, rng: random.Random) -> np.ndarray:
    mask = _mask_from_strings(rng.choice(SPRITE_MASKS[size]))
    variant = rng.randrange(8)
    mask = _dihedral(mask, variant)
    patch = np.zeros((size, size), dtype=int)
    patch[mask] = int(body_color)
    patch[0, 0] = int(anchor_color)
    return patch


def _place_frame_boxes(
    height: int,
    width: int,
    frame_size: int,
    arrangement: tuple[int, int],
    n_frames: int,
    band_side: str,
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    rows, cols = arrangement
    frame_region_top = 0 if band_side == "bottom" else SPRITE_BAND + BAND_GAP
    frame_region_bottom = height - (SPRITE_BAND + BAND_GAP) if band_side == "bottom" else height
    usable_h = frame_region_bottom - frame_region_top
    usable_w = width
    min_h = rows * frame_size + (rows - 1) * FRAME_GAP
    min_w = cols * frame_size + (cols - 1) * FRAME_GAP
    if usable_h < min_h or usable_w < min_w:
        return None

    extra_h = usable_h - min_h
    extra_w = usable_w - min_w
    gaps_h = _split_nonnegative(extra_h, rows + 1, rng)
    gaps_w = _split_nonnegative(extra_w, cols + 1, rng)

    row_positions = [frame_region_top + gaps_h[0]]
    for i in range(1, rows):
        row_positions.append(row_positions[-1] + frame_size + FRAME_GAP + gaps_h[i])

    col_positions = [gaps_w[0]]
    for j in range(1, cols):
        col_positions.append(col_positions[-1] + frame_size + FRAME_GAP + gaps_w[j])

    slots = [(r, c) for r in row_positions for c in col_positions]
    chosen = rng.sample(slots, n_frames)
    chosen.sort()
    return chosen


def _place_sprites(
    grid: np.ndarray,
    sprite_sizes: list[int],
    band_side: str,
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    height, width = grid.shape
    total_w = sum(sprite_sizes) + (len(sprite_sizes) - 1) * FRAME_GAP
    if total_w > width:
        return None
    extra_w = width - total_w
    gaps = _split_nonnegative(extra_w, len(sprite_sizes) + 1, rng)
    col = gaps[0]
    boxes = []
    for idx, size in enumerate(sprite_sizes):
        row = 0 if band_side == "top" else height - size
        boxes.append((row, col))
        col += size
        if idx < len(sprite_sizes) - 1:
            col += FRAME_GAP + gaps[idx + 1]
    return boxes


def _choose_anchor_position(
    frame_r0: int,
    frame_c0: int,
    interior_size: int,
    sprite_size: int,
    quadrant: str,
    rng: random.Random,
) -> tuple[int, int]:
    max_rel = interior_size - sprite_size
    top_max = min(max_rel, int(np.ceil(interior_size / 2.0)) - 1)
    bottom_min = int(np.ceil(interior_size / 2.0))
    left_max = min(max_rel, int(np.ceil(interior_size / 2.0)) - 1)
    right_min = int(np.ceil(interior_size / 2.0))

    if quadrant == "tl":
        rel_r = rng.randint(0, top_max)
        rel_c = rng.randint(0, left_max)
    elif quadrant == "tr":
        rel_r = rng.randint(0, top_max)
        rel_c = rng.randint(right_min, max_rel)
    elif quadrant == "br":
        rel_r = rng.randint(bottom_min, max_rel)
        rel_c = rng.randint(right_min, max_rel)
    else:
        rel_r = rng.randint(bottom_min, max_rel)
        rel_c = rng.randint(0, left_max)

    return frame_r0 + 1 + rel_r, frame_c0 + 1 + rel_c


def _draw_frame(grid: np.ndarray, r0: int, c0: int, size: int) -> None:
    r1 = r0 + size - 1
    c1 = c0 + size - 1
    grid[r0, c0 : c1 + 1] = FRAME_COLOR
    grid[r1, c0 : c1 + 1] = FRAME_COLOR
    grid[r0 : r1 + 1, c0] = FRAME_COLOR
    grid[r0 : r1 + 1, c1] = FRAME_COLOR


def _paste_expected(
    out: np.ndarray,
    patch: np.ndarray,
    frame_r0: int,
    frame_c0: int,
    frame_size: int,
    anchor_r: int,
    anchor_c: int,
) -> None:
    frame_r1 = frame_r0 + frame_size - 1
    frame_c1 = frame_c0 + frame_size - 1
    for dr in range(patch.shape[0]):
        for dc in range(patch.shape[1]):
            value = int(patch[dr, dc])
            if value == BG:
                continue
            rr = anchor_r + dr
            cc = anchor_c + dc
            if frame_r0 < rr < frame_r1 and frame_c0 < cc < frame_c1:
                out[rr, cc] = value


def _background_color(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[np.argmax(counts)])


def _find_frames(grid: np.ndarray, bg: int) -> list[dict[str, int | np.ndarray]]:
    frames = []
    for color in (int(c) for c in np.unique(grid) if int(c) != bg):
        for coords in _components(grid == color, connectivity=4):
            r0, c0, r1, c1 = _bbox(coords)
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            if h != w or h < 5:
                continue
            border = np.zeros((h, w), dtype=bool)
            border[0, :] = True
            border[-1, :] = True
            border[:, 0] = True
            border[:, -1] = True
            local = np.zeros((h, w), dtype=bool)
            local[coords[:, 0] - r0, coords[:, 1] - c0] = True
            if not np.array_equal(local, border):
                continue
            inner = grid[r0 + 1 : r1, c0 + 1 : c1]
            anchor_pts = np.argwhere((inner != bg) & (inner != color))
            if anchor_pts.shape[0] != 1:
                continue
            anchor_r = int(r0 + 1 + anchor_pts[0, 0])
            anchor_c = int(c0 + 1 + anchor_pts[0, 1])
            frames.append(
                {
                    "r0": r0,
                    "c0": c0,
                    "r1": r1,
                    "c1": c1,
                    "anchor_r": anchor_r,
                    "anchor_c": anchor_c,
                    "anchor_color": int(grid[anchor_r, anchor_c]),
                    "border_mask": local,
                }
            )
    frames.sort(key=lambda frame: (int(frame["r0"]), int(frame["c0"])))
    return frames


def _find_sprites(grid: np.ndarray, bg: int, frame_mask: np.ndarray) -> dict[int, np.ndarray]:
    sprites: dict[int, np.ndarray] = {}
    outside = (grid != bg) & (~frame_mask)
    for coords in _components(outside, connectivity=8):
        r0, c0, r1, c1 = _bbox(coords)
        patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
        patch[coords[:, 0] - r0, coords[:, 1] - c0] = grid[coords[:, 0], coords[:, 1]]
        marker_color = int(patch[0, 0])
        if marker_color == bg:
            continue
        marker_positions = np.argwhere(patch == marker_color)
        if marker_positions.shape[0] != 1 or tuple(marker_positions[0]) != (0, 0):
            continue
        sprites[marker_color] = patch
    return sprites


def _paste_inside_frame(out: np.ndarray, patch: np.ndarray, frame: dict[str, int | np.ndarray], bg: int) -> None:
    anchor_r = int(frame["anchor_r"])
    anchor_c = int(frame["anchor_c"])
    r0 = int(frame["r0"])
    c0 = int(frame["c0"])
    r1 = int(frame["r1"])
    c1 = int(frame["c1"])
    for dr in range(patch.shape[0]):
        for dc in range(patch.shape[1]):
            value = int(patch[dr, dc])
            if value == bg:
                continue
            rr = anchor_r + dr
            cc = anchor_c + dc
            if r0 < rr < r1 and c0 < cc < c1:
                out[rr, cc] = value


def _rotation_k_for_quadrant(frame: dict[str, int | np.ndarray]) -> int:
    size = int(frame["r1"]) - int(frame["r0"]) - 1
    rel_r = int(frame["anchor_r"]) - int(frame["r0"]) - 1
    rel_c = int(frame["anchor_c"]) - int(frame["c0"]) - 1
    top = rel_r < size / 2.0
    left = rel_c < size / 2.0
    if top and left:
        return 0
    if top and not left:
        return 3
    if not top and not left:
        return 2
    return 1


def _rotation_k_from_name(quadrant: str) -> int:
    return {"tl": 0, "tr": 3, "br": 2, "bl": 1}[quadrant]


def _choose_arrangement(n_frames: int, rng: random.Random) -> tuple[int, int]:
    if n_frames == 2:
        return rng.choice([(1, 2), (2, 1)])
    return (2, 2)


def _split_nonnegative(total: int, parts: int, rng: random.Random) -> list[int]:
    values = [0] * parts
    for _ in range(total):
        values[rng.randrange(parts)] += 1
    return values


def _mask_from_strings(rows: Iterable[str]) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)


def _dihedral(mask: np.ndarray, variant: int) -> np.ndarray:
    k = variant % 4
    out = np.rot90(mask, k)
    if variant >= 4:
        out = np.fliplr(out)
    return out


def _components(mask: np.ndarray, connectivity: int) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    if connectivity == 4:
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        neighbors = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ]

    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in neighbors:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )
