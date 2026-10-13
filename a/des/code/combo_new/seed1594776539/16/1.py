from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np


BG = 0

STAMP_TO_SIDE = {
    "right": "left",
    "left": "right",
    "down": "top",
    "up": "bottom",
}

SIDE_TO_STAMP = {side: stamp for stamp, side in STAMP_TO_SIDE.items()}

OUTWARD_STEP = {
    "left": (0, -1),
    "right": (0, 1),
    "top": (-1, 0),
    "bottom": (1, 0),
}

ROTATION_K = {"right": 0, "down": 1, "left": 2, "up": 3}

RIGHT_TEMPLATES = [
    {"anchor": (1, 0), "body": [(1, 1)]},
    {"anchor": (1, 0), "body": [(0, 1), (1, 1)]},
    {"anchor": (1, 0), "body": [(1, 1), (2, 1)]},
    {"anchor": (1, 0), "body": [(0, 1), (1, 1), (2, 1)]},
    {"anchor": (1, 0), "body": [(1, 1), (1, 2)]},
    {"anchor": (1, 0), "body": [(0, 1), (1, 1), (1, 2)]},
    {"anchor": (1, 0), "body": [(1, 1), (2, 1), (1, 2)]},
    {"anchor": (1, 0), "body": [(0, 1), (1, 1), (2, 1), (1, 2)]},
    {"anchor": (0, 0), "body": [(0, 1), (1, 1)]},
    {"anchor": (2, 0), "body": [(1, 1), (2, 1)]},
]

GEN_ATTEMPTS = 400
PAIR_ATTEMPTS = 150
POSITION_ATTEMPTS = 60
GRID_MIN = 16
GRID_MAX = 30
FRAME_COUNT_MIN = 2
FRAME_COUNT_MAX = 4
FRAME_SIZE_MIN = 5
FRAME_SIZE_MAX = 9


def _components4(mask: np.ndarray) -> Tuple[List[List[Tuple[int, int]]], np.ndarray]:
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=int)
    comps: List[List[Tuple[int, int]]] = []
    cid = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != 0:
                continue
            cid += 1
            stack = [(r, c)]
            labels[r, c] = cid
            coords: List[Tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and labels[nr, nc] == 0:
                        labels[nr, nc] = cid
                        stack.append((nr, nc))
            comps.append(coords)
    return comps, labels


def _bbox(coords: Sequence[Tuple[int, int]]) -> Tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _border_cells(r0: int, c0: int, r1: int, c1: int) -> List[Tuple[int, int]]:
    cells: List[Tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        if r1 != r0:
            cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        if c1 != c0:
            cells.append((r, c1))
    return cells


def _find_frames(
    g: np.ndarray, comps: Sequence[Sequence[Tuple[int, int]]]
) -> List[Dict[str, object]]:
    frames: List[Dict[str, object]] = []
    for cid, coords in enumerate(comps, start=1):
        arr = np.array(coords, dtype=int)
        colors = np.unique(g[arr[:, 0], arr[:, 1]])
        if len(colors) != 1:
            continue
        r0, c0, r1, c1 = _bbox(coords)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        border = set(_border_cells(r0, c0, r1, c1))
        pts = set(coords)
        if not pts.issubset(border):
            continue
        missing = list(border - pts)
        if len(missing) != 1:
            continue
        gap = missing[0]
        if gap in {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}:
            continue
        if np.any(g[r0 + 1 : r1, c0 + 1 : c1] != BG):
            continue
        if gap[0] == r0:
            side = "top"
        elif gap[0] == r1:
            side = "bottom"
        elif gap[1] == c0:
            side = "left"
        else:
            side = "right"
        frames.append(
            {
                "cid": cid,
                "color": int(colors[0]),
                "bbox": (r0, c0, r1, c1),
                "gap": gap,
                "side": side,
            }
        )
    frames.sort(key=lambda item: (item["bbox"], item["gap"]))  # type: ignore[index]
    return frames


def _infer_stamp_direction(
    anchor: Tuple[int, int], body: Sequence[Tuple[int, int]]
) -> str:
    ar, ac = anchor
    rel = [(r - ar, c - ac) for r, c in body]
    if all(dc > 0 for _, dc in rel):
        return "right"
    if all(dc < 0 for _, dc in rel):
        return "left"
    if all(dr > 0 for dr, _ in rel):
        return "down"
    if all(dr < 0 for dr, _ in rel):
        return "up"
    raise AssertionError("sprite anchor does not define a single stamp direction")


def _find_sprite_candidates(
    g: np.ndarray, comps: Sequence[Sequence[Tuple[int, int]]], frame_ids: set[int]
) -> Dict[int, Dict[str, object]]:
    sprites: Dict[int, Dict[str, object]] = {}
    for cid, coords in enumerate(comps, start=1):
        if cid in frame_ids:
            continue
        r0, c0, r1, c1 = _bbox(coords)
        if r1 - r0 + 1 > 3 or c1 - c0 + 1 > 3:
            continue
        arr = np.array(coords, dtype=int)
        vals = [int(v) for v in g[arr[:, 0], arr[:, 1]]]
        colors = sorted(set(vals))
        if len(colors) != 2:
            continue
        sprites[cid] = {
            "cid": cid,
            "coords": list(coords),
            "colors": colors,
        }
    return sprites


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp each scattered sprite onto the inner wall of the nearby gapped frame it faces.

    Input structure:
    - The background is black.
    - The grid contains several monochrome hollow rectangular frames; each frame is one pixel thick and has exactly one one-pixel gap on one side.
    - Outside the frames are small connected sprites no larger than 3x3. Each sprite has one body color plus one anchor pixel whose color matches the nearby frame.

    Rule:
    - Detect every frame, recover its unique gap, and follow the outward ray from that gap to the sprite sitting on the same row or column.
    - Infer the stamping direction from the anchor's side inside the sprite: left anchor means stamp right, right means left, top means down, bottom means up.
    - Remove the anchor pixel, align it with the gap position, and paint the remaining sprite shape inside the frame with the sprite's body color. Leave the gap open.

    Key insight:
    - Frame matching is determined by spatial alignment with the gap, while the odd-colored anchor is not noise: it encodes which way the sprite must be pushed through the opening.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    comps, labels = _components4(g != BG)
    frames = _find_frames(g, comps)
    frame_ids = {int(frame["cid"]) for frame in frames}
    sprites = _find_sprite_candidates(g, comps, frame_ids)

    out = g.copy()
    used_sprites: set[int] = set()

    for frame in frames:
        gap_r, gap_c = frame["gap"]  # type: ignore[misc]
        side = str(frame["side"])
        dr, dc = OUTWARD_STEP[side]
        sprite = None
        for step in range(1, max(h, w) + 1):
            rr = gap_r + dr * step
            cc = gap_c + dc * step
            if not (0 <= rr < h and 0 <= cc < w):
                break
            if g[rr, cc] == BG:
                continue
            cid = int(labels[rr, cc])
            sprite = sprites.get(cid)
            break
        assert sprite is not None, "every frame must have a sprite on its gap ray"
        assert int(sprite["cid"]) not in used_sprites, "each sprite should match exactly one frame"
        used_sprites.add(int(sprite["cid"]))

        coords = sprite["coords"]  # type: ignore[assignment]
        frame_color = int(frame["color"])
        anchor_cells = [(r, c) for r, c in coords if int(g[r, c]) == frame_color]
        assert len(anchor_cells) == 1, "paired sprite must contain exactly one frame-colored anchor"
        anchor = anchor_cells[0]
        body = [(r, c) for r, c in coords if (r, c) != anchor]
        direction = _infer_stamp_direction(anchor, body)
        assert direction == SIDE_TO_STAMP[side], "anchor direction must agree with frame gap"
        body_color = int(g[body[0]])
        body_offsets = [(r - anchor[0], c - anchor[1]) for r, c in body]

        for off_r, off_c in body_offsets:
            rr = gap_r + int(off_r)
            cc = gap_c + int(off_c)
            assert 0 <= rr < h and 0 <= cc < w
            out[rr, cc] = body_color

    return out


def _rot90_cw(point: Tuple[int, int], size: int = 3) -> Tuple[int, int]:
    r, c = point
    return c, size - 1 - r


def _oriented_template(rng: np.random.Generator, stamp_dir: str) -> Dict[str, object]:
    base = RIGHT_TEMPLATES[int(rng.integers(0, len(RIGHT_TEMPLATES)))]
    coords = [tuple(base["anchor"])] + [tuple(cell) for cell in base["body"]]
    k = ROTATION_K[stamp_dir]
    for _ in range(k):
        coords = [_rot90_cw(point) for point in coords]
    min_r = min(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    coords = [(r - min_r, c - min_c) for r, c in coords]
    anchor = coords[0]
    body = coords[1:]
    height = 1 + max(r for r, _ in coords)
    width = 1 + max(c for _, c in coords)
    return {"anchor": anchor, "body": body, "cells": coords, "shape": (height, width)}


def _valid_gap_positions(
    frame_h: int,
    frame_w: int,
    side: str,
    body_offsets: Sequence[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    positions: List[Tuple[int, int]] = []
    if side in {"left", "right"}:
        border_c = 0 if side == "left" else frame_w - 1
        interior_ok = all(1 <= border_c + dc <= frame_w - 2 for _, dc in body_offsets)
        if not interior_ok:
            return positions
        for gap_r in range(1, frame_h - 1):
            if all(1 <= gap_r + dr <= frame_h - 2 for dr, _ in body_offsets):
                positions.append((gap_r, border_c))
    else:
        border_r = 0 if side == "top" else frame_h - 1
        interior_ok = all(1 <= border_r + dr <= frame_h - 2 for dr, _ in body_offsets)
        if not interior_ok:
            return positions
        for gap_c in range(1, frame_w - 1):
            if all(1 <= gap_c + dc <= frame_w - 2 for _, dc in body_offsets):
                positions.append((border_r, gap_c))
    return positions


def _sprite_input_positions(
    template: Dict[str, object], side: str, gap: Tuple[int, int], distance: int
) -> List[Tuple[int, int]]:
    anchor_r, anchor_c = template["anchor"]  # type: ignore[misc]
    cells = template["cells"]  # type: ignore[assignment]
    gap_r, gap_c = gap

    if side == "left":
        same_row_cols = [c for r, c in cells if r == anchor_r]
        shift_r = gap_r - int(anchor_r)
        shift_c = gap_c - distance - max(int(c) for c in same_row_cols)
    elif side == "right":
        same_row_cols = [c for r, c in cells if r == anchor_r]
        shift_r = gap_r - int(anchor_r)
        shift_c = gap_c + distance - min(int(c) for c in same_row_cols)
    elif side == "top":
        same_col_rows = [r for r, c in cells if c == anchor_c]
        shift_r = gap_r - distance - max(int(r) for r in same_col_rows)
        shift_c = gap_c - int(anchor_c)
    else:
        same_col_rows = [r for r, c in cells if c == anchor_c]
        shift_r = gap_r + distance - min(int(r) for r in same_col_rows)
        shift_c = gap_c - int(anchor_c)

    return [(shift_r + int(r), shift_c + int(c)) for r, c in cells]


def _pair_layout(
    rng: np.random.Generator,
    grid: np.ndarray,
    protect: np.ndarray,
    frame_color: int,
    body_color: int,
    stamp_dir: str,
) -> Tuple[Dict[str, object], List[Tuple[int, int]]] | None:
    h, w = grid.shape
    side = STAMP_TO_SIDE[stamp_dir]

    for _ in range(PAIR_ATTEMPTS):
        template = _oriented_template(rng, stamp_dir)
        body_offsets = [
            (int(r) - int(template["anchor"][0]), int(c) - int(template["anchor"][1]))
            for r, c in template["body"]  # type: ignore[assignment]
        ]
        frame_h = int(rng.integers(FRAME_SIZE_MIN, FRAME_SIZE_MAX + 1))
        frame_w = int(rng.integers(FRAME_SIZE_MIN, FRAME_SIZE_MAX + 1))
        gaps = _valid_gap_positions(frame_h, frame_w, side, body_offsets)
        if not gaps:
            continue
        gap_rel = gaps[int(rng.integers(0, len(gaps)))]
        distance = int(rng.integers(1, 4))

        frame_cells = set(_border_cells(0, 0, frame_h - 1, frame_w - 1))
        frame_cells.remove(gap_rel)
        sprite_cells = _sprite_input_positions(template, side, gap_rel, distance)
        union_cells = list(frame_cells) + sprite_cells
        min_r = min(r for r, _ in union_cells)
        max_r = max(r for r, _ in union_cells)
        min_c = min(c for _, c in union_cells)
        max_c = max(c for _, c in union_cells)

        row_lo = 1 - min_r
        row_hi = h - 2 - max_r
        col_lo = 1 - min_c
        col_hi = w - 2 - max_c
        if row_lo > row_hi or col_lo > col_hi:
            continue

        for _ in range(POSITION_ATTEMPTS):
            base_r = int(rng.integers(row_lo, row_hi + 1))
            base_c = int(rng.integers(col_lo, col_hi + 1))
            abs_r0 = base_r + min_r
            abs_r1 = base_r + max_r
            abs_c0 = base_c + min_c
            abs_c1 = base_c + max_c
            if protect[abs_r0 - 1 : abs_r1 + 2, abs_c0 - 1 : abs_c1 + 2].any():
                continue

            frame_abs = [(base_r + r, base_c + c) for r, c in frame_cells]
            gap_abs = (base_r + gap_rel[0], base_c + gap_rel[1])
            sprite_abs = [(base_r + r, base_c + c) for r, c in sprite_cells]
            if any(grid[r, c] != BG for r, c in frame_abs + sprite_abs):
                continue

            anchor_abs = sprite_abs[template["cells"].index(template["anchor"])]  # type: ignore[index]
            body_abs = [pos for pos in sprite_abs if pos != anchor_abs]
            for r, c in frame_abs:
                grid[r, c] = frame_color
            for r, c in body_abs:
                grid[r, c] = body_color
            grid[anchor_abs] = frame_color
            protect[abs_r0 - 1 : abs_r1 + 2, abs_c0 - 1 : abs_c1 + 2] = True

            spec = {
                "frame_color": frame_color,
                "body_color": body_color,
                "stamp_dir": stamp_dir,
                "side": side,
                "gap": gap_abs,
                "body_offsets": body_offsets,
                "anchor": anchor_abs,
                "body": body_abs,
                "bbox": (abs_r0, abs_c0, abs_r1, abs_c1),
            }
            return spec, frame_abs + sprite_abs
    return None


def _expected_output(grid: np.ndarray, specs: Sequence[Dict[str, object]]) -> np.ndarray:
    out = grid.copy()
    for spec in specs:
        gap_r, gap_c = spec["gap"]  # type: ignore[misc]
        body_color = int(spec["body_color"])
        for dr, dc in spec["body_offsets"]:  # type: ignore[assignment]
            out[gap_r + int(dr), gap_c + int(dc)] = body_color
    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(GRID_MIN, GRID_MAX + 1))
    w = int(rng.integers(GRID_MIN, GRID_MAX + 1))
    grid = np.zeros((h, w), dtype=int)
    protect = np.zeros((h, w), dtype=bool)

    frame_count = int(rng.integers(FRAME_COUNT_MIN, FRAME_COUNT_MAX + 1))
    colors = rng.permutation(np.arange(1, 10)).tolist()
    if 2 * frame_count > len(colors):
        return None
    frame_colors = [int(c) for c in colors[:frame_count]]
    body_colors = [int(c) for c in colors[frame_count : 2 * frame_count]]

    stamp_dirs = ["right", "left", "down", "up"]
    rng.shuffle(stamp_dirs)
    chosen_dirs = stamp_dirs[:frame_count]

    specs: List[Dict[str, object]] = []
    for i in range(frame_count):
        placed = _pair_layout(
            rng,
            grid,
            protect,
            frame_colors[i],
            body_colors[i],
            chosen_dirs[i],
        )
        if placed is None:
            return None
        spec, _ = placed
        specs.append(spec)

    expected = _expected_output(grid, specs)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed to build a valid frame-stamping puzzle")
