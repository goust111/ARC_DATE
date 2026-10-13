from __future__ import annotations

from itertools import product
import random

import numpy as np


_FRAME_SIZE = 7
_INNER_SIZE = 5
_KEY_SIZE = 3
_TOP_ZONE = 4
_BOTTOM_ZONE = 4
_MAX_ATTEMPTS = 300
_DIR_ORDER = ("up", "right", "down", "left")
_FRAME_GAPS = {
    "up": (0, 3),
    "right": (3, 6),
    "down": (6, 3),
    "left": (3, 0),
}
_MARKER_OFFSETS = {
    "up": (-1, 1),
    "right": (1, 3),
    "down": (3, 1),
    "left": (1, -1),
}
_KEY_ORIGIN_IN_OBJECT = {
    "up": (1, 0),
    "right": (0, 0),
    "down": (0, 0),
    "left": (0, 1),
}
_ROT_K = {"up": 0, "right": 3, "down": 2, "left": 1}
_KEY_MASKS = (
    np.array([[0, 1, 0], [1, 1, 1], [1, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[0, 1, 1], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 0], [1, 1, 1], [1, 1, 1]], dtype=bool),
    np.array([[0, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=bool),
)


def _dominant_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == bg or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            arr = np.array(coords, dtype=int)
            r0 = int(arr[:, 0].min())
            c0 = int(arr[:, 1].min())
            r1 = int(arr[:, 0].max())
            c1 = int(arr[:, 1].max())
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (r0, c0, r1, c1),
                    "area": int(arr.shape[0]),
                }
            )
    return comps


def _frame_direction(patch: np.ndarray, bg: int, color: int) -> str | None:
    if patch.shape != (_FRAME_SIZE, _FRAME_SIZE):
        return None
    if not np.all(patch[1:-1, 1:-1] == bg):
        return None
    border = np.full((_FRAME_SIZE, _FRAME_SIZE), bg, dtype=int)
    border[0, :] = color
    border[-1, :] = color
    border[:, 0] = color
    border[:, -1] = color
    matches = []
    for direction, (rr, cc) in _FRAME_GAPS.items():
        sample = border.copy()
        sample[rr, cc] = bg
        if np.array_equal(sample, patch):
            matches.append(direction)
    return matches[0] if len(matches) == 1 else None


def _extract_frames(grid: np.ndarray, bg: int, comps: list[dict]) -> list[dict]:
    frames = []
    for comp in comps:
        r0, c0, r1, c1 = comp["bbox"]
        patch = np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int)
        direction = _frame_direction(patch, bg, int(comp["color"]))
        if direction is None:
            continue
        frames.append(
            {
                "color": int(comp["color"]),
                "direction": direction,
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
            }
        )
    return frames


def _extract_keys(grid: np.ndarray, bg: int, comps: list[dict], frames: list[dict]) -> dict[tuple[int, str], np.ndarray]:
    frame_bboxes = {tuple(frame["bbox"]) for frame in frames}
    markers: dict[tuple[int, int], int] = {}
    key_comps = []
    for comp in comps:
        if tuple(comp["bbox"]) in frame_bboxes:
            continue
        r0, c0, r1, c1 = comp["bbox"]
        if int(comp["area"]) == 1:
            markers[(int(r0), int(c0))] = int(comp["color"])
            continue
        if (r1 - r0 + 1, c1 - c0 + 1) == (_KEY_SIZE, _KEY_SIZE):
            key_comps.append(comp)

    keys: dict[tuple[int, str], np.ndarray] = {}
    for comp in key_comps:
        r0, c0, r1, c1 = comp["bbox"]
        patch = np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int)
        matches = []
        for side, (dr, dc) in _MARKER_OFFSETS.items():
            pos = (int(r0 + dr), int(c0 + dc))
            if pos in markers:
                matches.append((side, int(markers[pos])))
        assert len(matches) == 1, "each key must have exactly one centered marker"
        side, marker_color = matches[0]
        key = (marker_color, side)
        assert key not in keys, "marker-color and marker-side combination must be unique"
        keys[key] = patch
    return keys


def _tile_fill(key_patch: np.ndarray, direction: str, bg: int) -> np.ndarray:
    rotated = np.rot90(np.asarray(key_patch, dtype=int), _ROT_K[direction])
    return np.tile(rotated, (2, 2))[:_INNER_SIZE, :_INNER_SIZE]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each gapped frame to the key with the same color-marker direction and tile the rotated key into the frame interior.

    Input structure:
    - The background is black. Several 7x7 hollow rectangular frames appear on the grid, each drawn in one non-background color with a 5x5 black interior.
    - Every frame has exactly one 1-pixel gap at the center of one side.
    - Separate 3x3 monochrome key shapes sit near the outer edges, and each key has one adjacent single-pixel marker whose color matches some frame and whose side relative to the key is one of up/right/down/left.

    Rule:
    - For every frame, use its gap direction and frame color to select the unique key whose marker is on the same side and whose marker color matches the frame color.
    - Rotate the selected 3x3 key pattern by the frame direction: up = none, right = 90 degrees clockwise, down = 180 degrees, left = 90 degrees counterclockwise.
    - Tile that rotated key pattern over the frame's 5x5 interior from the top-left corner, clipping overflow and only painting non-background key pixels.
    - Keep the original frames, keys, and marker pixels in place.

    Key insight:
    - The frame gap direction is reused twice: first as the lookup index for the correct key marker side, then as the rotation that orients the key pattern before tiling.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _dominant_color(g)
    comps = _find_components(g, bg)
    frames = _extract_frames(g, bg, comps)
    assert frames, "expected at least one gapped frame"
    keys = _extract_keys(g, bg, comps, frames)

    out = g.copy()
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        key = (int(frame["color"]), str(frame["direction"]))
        assert key in keys, "every frame must have a matching key"
        fill = _tile_fill(keys[key], frame["direction"], bg)
        target = out[r0 + 1 : r1, c0 + 1 : c1]
        mask = fill != bg
        target[mask] = fill[mask]
    return out


def _key_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.zeros((_KEY_SIZE, _KEY_SIZE), dtype=int)
    patch[np.asarray(mask, dtype=bool)] = int(color)
    return patch


def _draw_frame(grid: np.ndarray, top: int, left: int, color: int, direction: str) -> None:
    grid[top, left : left + _FRAME_SIZE] = color
    grid[top + _FRAME_SIZE - 1, left : left + _FRAME_SIZE] = color
    grid[top : top + _FRAME_SIZE, left] = color
    grid[top : top + _FRAME_SIZE, left + _FRAME_SIZE - 1] = color
    rr, cc = _FRAME_GAPS[direction]
    grid[top + rr, left + cc] = 0


def _place_key(grid: np.ndarray, top: int, left: int, patch: np.ndarray, frame_color: int, direction: str) -> None:
    h = 4 if direction in ("up", "down") else 3
    w = 4 if direction in ("left", "right") else 3
    assert top >= 0 and left >= 0 and top + h <= grid.shape[0] and left + w <= grid.shape[1]
    kr = top + _KEY_ORIGIN_IN_OBJECT[direction][0]
    kc = left + _KEY_ORIGIN_IN_OBJECT[direction][1]
    mask = patch != 0
    grid[kr : kr + _KEY_SIZE, kc : kc + _KEY_SIZE][mask] = patch[mask]
    mr = kr + _MARKER_OFFSETS[direction][0]
    mc = kc + _MARKER_OFFSETS[direction][1]
    grid[mr, mc] = frame_color


def _expected_output(grid: np.ndarray, frame_specs: list[dict], key_map: dict[tuple[int, str], np.ndarray]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for frame in frame_specs:
        r0 = int(frame["top"])
        c0 = int(frame["left"])
        direction = str(frame["direction"])
        color = int(frame["color"])
        fill = _tile_fill(key_map[(color, direction)], direction, 0)
        target = out[r0 + 1 : r0 + 1 + _INNER_SIZE, c0 + 1 : c0 + 1 + _INNER_SIZE]
        mask = fill != 0
        target[mask] = fill[mask]
    return out


def _sample_layout():
    color_count, direction_count = random.choice(((2, 2), (2, 3), (3, 2)))
    n_obj = color_count * direction_count
    cols = n_obj // 2
    top_gap = random.randint(1, 2)
    mid_gap = random.randint(1, 4)
    bottom_gap = random.randint(1, 2)
    if cols == 2:
        left_margin = random.randint(1, 6)
        right_margin = random.randint(1, 6)
        col_gap = random.randint(1, 5)
    else:
        left_margin = random.randint(1, 3)
        right_margin = random.randint(1, 3)
        col_gap = random.randint(1, 2)
    height = _TOP_ZONE + top_gap + _FRAME_SIZE + mid_gap + _FRAME_SIZE + bottom_gap + _BOTTOM_ZONE
    width = left_margin + cols * _FRAME_SIZE + (cols - 1) * col_gap + right_margin
    if height > 30 or width > 30:
        return None
    return {
        "color_count": color_count,
        "direction_count": direction_count,
        "n_obj": n_obj,
        "cols": cols,
        "height": height,
        "width": width,
        "top_gap": top_gap,
        "mid_gap": mid_gap,
        "bottom_gap": bottom_gap,
        "left_margin": left_margin,
        "col_gap": col_gap,
    }


def _gen_attempt() -> np.ndarray | None:
    layout = _sample_layout()
    if layout is None:
        return None

    color_count = int(layout["color_count"])
    direction_count = int(layout["direction_count"])
    n_obj = int(layout["n_obj"])
    cols = int(layout["cols"])
    h = int(layout["height"])
    w = int(layout["width"])
    top_gap = int(layout["top_gap"])
    mid_gap = int(layout["mid_gap"])
    left_margin = int(layout["left_margin"])
    col_gap = int(layout["col_gap"])

    frame_colors = random.sample(list(range(1, 10)), color_count)
    remaining = [c for c in range(1, 10) if c not in frame_colors]
    key_colors = random.sample(remaining, n_obj)
    directions = random.sample(list(_DIR_ORDER), direction_count)

    combos = [(fc, dr) for fc, dr in product(frame_colors, directions)]
    random.shuffle(combos)

    masks = random.sample(list(_KEY_MASKS), n_obj)
    key_map: dict[tuple[int, str], np.ndarray] = {}
    for (frame_color, direction), key_color, mask in zip(combos, key_colors, masks):
        patch = _key_patch(np.rot90(mask, random.randint(0, 3)), key_color)
        key_map[(frame_color, direction)] = patch

    grid = np.zeros((h, w), dtype=int)

    frame_rows = [_TOP_ZONE + top_gap, _TOP_ZONE + top_gap + _FRAME_SIZE + mid_gap]
    frame_cols = [left_margin + i * (_FRAME_SIZE + col_gap) for i in range(cols)]
    frame_slots = [(r, c) for r in frame_rows for c in frame_cols]
    random.shuffle(frame_slots)

    frame_specs = []
    for (frame_color, direction), (top, left) in zip(combos, frame_slots):
        _draw_frame(grid, top, left, frame_color, direction)
        frame_specs.append({"top": top, "left": left, "color": frame_color, "direction": direction})

    key_slots = []
    for col in frame_cols:
        key_slots.append(("top", col))
    for col in frame_cols:
        key_slots.append(("bottom", col))
    random.shuffle(key_slots)

    for combo, (zone, base_col) in zip(combos, key_slots):
        frame_color, direction = combo
        patch = key_map[combo]
        obj_h = 4 if direction in ("up", "down") else 3
        obj_w = 4 if direction in ("left", "right") else 3
        row_lo = 0 if zone == "top" else h - _BOTTOM_ZONE
        row_hi = row_lo + (4 - obj_h)
        top = random.randint(row_lo, row_hi)
        left = random.randint(base_col, base_col + _FRAME_SIZE - obj_w)
        _place_key(grid, top, left, patch, frame_color, direction)

    expected = _expected_output(grid, frame_specs, key_map)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to produce a valid sample")
