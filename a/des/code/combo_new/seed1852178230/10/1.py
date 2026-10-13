from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
CORNER_TO_CCW = {
    "top_left": 0,
    "top_right": 3,
    "bottom_right": 2,
    "bottom_left": 1,
}
COLOR_POOL = tuple(range(1, 10))
GEN_RETRIES = 300


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_components(grid: np.ndarray, bg: int = BG) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append({"color": color, "coords": coords, "bbox": _bbox(coords)})
    return comps


def _component_mask(comp: dict) -> np.ndarray:
    r0, c0, r1, c1 = comp["bbox"]
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in comp["coords"]:
        mask[r - r0, c - c0] = True
    return mask


def _is_rect_frame(comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False

    mask = _component_mask(comp)
    expected = np.zeros_like(mask)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    return np.array_equal(mask, expected)


def _bbox_distance(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    row_gap = max(0, br0 - ar1 - 1, ar0 - br1 - 1)
    col_gap = max(0, bc0 - ac1 - 1, ac0 - bc1 - 1)
    return row_gap + col_gap


def _pair_frames_and_shards(comps: list[dict]) -> list[tuple[dict, dict]]:
    by_color: dict[int, list[dict]] = {}
    for comp in comps:
        by_color.setdefault(int(comp["color"]), []).append(comp)

    pairs: list[tuple[dict, dict]] = []
    for color, color_comps in by_color.items():
        frames = [comp for comp in color_comps if _is_rect_frame(comp)]
        shards = [comp for comp in color_comps if not _is_rect_frame(comp)]
        if not frames:
            continue
        assert len(shards) >= len(frames), f"color {color} is missing matching shards"

        frame_pool = frames[:]
        shard_pool = shards[:]
        while frame_pool:
            best: tuple[int, int, int] | None = None
            for fi, frame in enumerate(frame_pool):
                for si, shard in enumerate(shard_pool):
                    dist = _bbox_distance(frame["bbox"], shard["bbox"])
                    cand = (dist, fi, si)
                    if best is None or cand < best:
                        best = cand
            assert best is not None
            _, fi, si = best
            pairs.append((frame_pool.pop(fi), shard_pool.pop(si)))
    assert pairs, "expected at least one frame-shard pair"
    return pairs


def _marker_info(interior: np.ndarray) -> tuple[str, int, np.ndarray]:
    marker_mask = interior != BG
    assert np.any(marker_mask), "frame interior must contain a marker"
    marker_colors = {int(v) for v in interior[marker_mask]}
    assert len(marker_colors) == 1, "marker inside a frame must use one color"
    marker_color = next(iter(marker_colors))

    corners = {
        "top_left": bool(marker_mask[0, 0]),
        "top_right": bool(marker_mask[0, -1]),
        "bottom_right": bool(marker_mask[-1, -1]),
        "bottom_left": bool(marker_mask[-1, 0]),
    }
    active = [name for name, present in corners.items() if present]
    assert len(active) == 1, "marker must occupy exactly one interior corner"
    return active[0], marker_color, marker_mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each outer shard by the frame's corner marker and insert it to complete the framed rectangle.

    Input structure:
    - The background is black.
    - Several one-pixel-thick rectangular frames appear in distinct colors.
    - Each frame contains a small differently colored L marker in exactly one interior corner.
    - Outside each frame, a separated shard with the same color as the frame represents the missing interior area.

    Rule:
    - Detect every rectangular frame and the same-colored shard paired with it.
    - Read which corner holds the interior L marker: top-left means no rotation, top-right means 90 degrees clockwise, bottom-right means 180 degrees, and bottom-left means 90 degrees counterclockwise.
    - Rotate the shard by that amount and place it into the frame interior so that it exactly complements the marker.
    - Recolor the inserted shard to the marker color and erase the original outer shard.

    Key insight:
    - The marker's corner is both the placement cue and the rotation code; after rotation, the shard and the marker must tile the full interior rectangle.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    comps = _find_components(g, bg=BG)
    pairs = _pair_frames_and_shards(comps)

    out = g.copy()
    for frame, shard in pairs:
        fr0, fc0, fr1, fc1 = frame["bbox"]
        ir0, ic0, ir1, ic1 = fr0 + 1, fc0 + 1, fr1 - 1, fc1 - 1
        assert ir0 <= ir1 and ic0 <= ic1, "frame must have a non-empty interior"

        interior = g[ir0 : ir1 + 1, ic0 : ic1 + 1]
        corner, marker_color, marker_mask = _marker_info(interior)
        rotated_shard = np.rot90(_component_mask(shard), CORNER_TO_CCW[corner])
        assert rotated_shard.shape == interior.shape, "rotated shard must match frame interior size"

        combined = rotated_shard | marker_mask
        assert np.all(combined), "rotated shard and marker must fill the full interior"
        assert not np.any(rotated_shard & marker_mask), "rotated shard must complement the marker"

        for r, c in shard["coords"]:
            out[r, c] = BG
        out[ir0 : ir1 + 1, ic0 : ic1 + 1] = marker_color

    return out


def _corner_marker_mask(shape: tuple[int, int], corner: str) -> np.ndarray:
    h, w = shape
    assert h >= 3 and w >= 3, "generator uses a 3-cell marker and needs at least 3x3 interiors"
    mask = np.zeros((h, w), dtype=bool)
    if corner == "top_left":
        cells = [(0, 0), (0, 1), (1, 0)]
    elif corner == "top_right":
        cells = [(0, w - 1), (0, w - 2), (1, w - 1)]
    elif corner == "bottom_right":
        cells = [(h - 1, w - 1), (h - 1, w - 2), (h - 2, w - 1)]
    elif corner == "bottom_left":
        cells = [(h - 1, 0), (h - 1, 1), (h - 2, 0)]
    else:
        raise AssertionError(f"unknown corner {corner}")
    for r, c in cells:
        mask[r, c] = True
    return mask


def _shape_coords(mask: np.ndarray, offset_r: int, offset_c: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    for r, c in zip(*np.where(mask)):
        coords.append((offset_r + int(r), offset_c + int(c)))
    return coords


def _build_group(
    rng: random.Random,
    frame_color: int,
    marker_color: int,
    corner: str,
    max_dim: int,
) -> dict:
    interior_h = rng.randint(3, max_dim)
    interior_w = rng.randint(3, max_dim)
    frame_h = interior_h + 2
    frame_w = interior_w + 2

    marker_mask = _corner_marker_mask((interior_h, interior_w), corner)
    target_mask = ~marker_mask
    input_shard = np.rot90(target_mask, (4 - CORNER_TO_CCW[corner]) % 4)
    shard_h, shard_w = input_shard.shape

    side = rng.choice(["top", "right", "bottom", "left"])
    gap = rng.randint(1, 2)

    if side in ("top", "bottom"):
        offset = rng.randint(-(shard_w - 2), frame_w - 2)
        frame_c0 = max(0, -offset)
        shard_c0 = frame_c0 + offset
        if side == "top":
            shard_r0 = 0
            frame_r0 = shard_h + gap
        else:
            frame_r0 = 0
            shard_r0 = frame_h + gap
    else:
        offset = rng.randint(-(shard_h - 2), frame_h - 2)
        frame_r0 = max(0, -offset)
        shard_r0 = frame_r0 + offset
        if side == "left":
            shard_c0 = 0
            frame_c0 = shard_w + gap
        else:
            frame_c0 = 0
            shard_c0 = frame_w + gap

    group_h = max(frame_r0 + frame_h, shard_r0 + shard_h)
    group_w = max(frame_c0 + frame_w, shard_c0 + shard_w)

    frame_mask = np.zeros((frame_h, frame_w), dtype=bool)
    frame_mask[0, :] = True
    frame_mask[-1, :] = True
    frame_mask[:, 0] = True
    frame_mask[:, -1] = True

    frame_coords = _shape_coords(frame_mask, frame_r0, frame_c0)
    marker_coords = _shape_coords(marker_mask, frame_r0 + 1, frame_c0 + 1)
    shard_coords = _shape_coords(input_shard, shard_r0, shard_c0)

    return {
        "h": group_h,
        "w": group_w,
        "frame_color": frame_color,
        "marker_color": marker_color,
        "frame_coords": frame_coords,
        "marker_coords": marker_coords,
        "shard_coords": shard_coords,
        "interior_bbox": (frame_r0 + 1, frame_c0 + 1, frame_r0 + interior_h, frame_c0 + interior_w),
    }


def _layout_row(groups: list[dict], order: list[int], rng: random.Random) -> dict[int, tuple[int, int]]:
    positions: dict[int, tuple[int, int]] = {}
    cursor = 0
    for idx in order:
        y = rng.randint(0, 2)
        positions[idx] = (y, cursor)
        cursor += groups[idx]["w"] + rng.randint(2, 4)
    return positions


def _layout_col(groups: list[dict], order: list[int], rng: random.Random) -> dict[int, tuple[int, int]]:
    positions: dict[int, tuple[int, int]] = {}
    cursor = 0
    for idx in order:
        x = rng.randint(0, 2)
        positions[idx] = (cursor, x)
        cursor += groups[idx]["h"] + rng.randint(2, 4)
    return positions


def _footprint(groups: list[dict], positions: dict[int, tuple[int, int]]) -> tuple[int, int]:
    h = 0
    w = 0
    for idx, (r0, c0) in positions.items():
        h = max(h, r0 + groups[idx]["h"])
        w = max(w, c0 + groups[idx]["w"])
    return h, w


def _arrange_groups(groups: list[dict], rng: random.Random) -> tuple[dict[int, tuple[int, int]], int, int]:
    order = list(range(len(groups)))
    rng.shuffle(order)

    modes = ["row", "col"]
    if len(groups) == 3:
        modes.append("grid")
    mode = rng.choice(modes)

    if mode == "row":
        positions = _layout_row(groups, order, rng)
    elif mode == "col":
        positions = _layout_col(groups, order, rng)
    else:
        split = rng.choice([1, 2])
        row1 = order[:split]
        row2 = order[split:]
        top = _layout_row(groups, row1, rng)
        top_h, top_w = _footprint(groups, top)
        bottom = _layout_row(groups, row2, rng)
        bottom_h, bottom_w = _footprint(groups, bottom)
        row_gap = rng.randint(2, 4)
        bottom_shift = rng.randint(0, max(0, top_w - bottom_w + 2))
        positions = dict(top)
        for idx, (r0, c0) in bottom.items():
            positions[idx] = (r0 + top_h + row_gap, c0 + bottom_shift)

    core_h, core_w = _footprint(groups, positions)
    if core_h > 30 or core_w > 30:
        raise AssertionError("group arrangement exceeds ARC size limit")

    extra_h = rng.randint(0, min(30 - core_h, 6))
    extra_w = rng.randint(0, min(30 - core_w, 6))
    top_margin = rng.randint(0, extra_h)
    left_margin = rng.randint(0, extra_w)

    shifted = {idx: (r0 + top_margin, c0 + left_margin) for idx, (r0, c0) in positions.items()}
    return shifted, core_h + extra_h, core_w + extra_w


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_groups = rng.choices([1, 2, 3], weights=[2, 3, 2])[0]
    colors = rng.sample(COLOR_POOL, 2 * n_groups)
    frame_colors = colors[:n_groups]
    marker_colors = colors[n_groups:]

    corners = rng.sample(list(CORNER_TO_CCW), n_groups)
    max_dim = 5 if n_groups == 3 else 6
    groups = [
        _build_group(rng, frame_colors[i], marker_colors[i], corners[i], max_dim)
        for i in range(n_groups)
    ]

    try:
        positions, h, w = _arrange_groups(groups, rng)
    except AssertionError:
        return None
    grid = np.zeros((h, w), dtype=int)
    expected = np.zeros((h, w), dtype=int)

    for idx, group in enumerate(groups):
        base_r, base_c = positions[idx]

        for r, c in group["frame_coords"]:
            rr, cc = base_r + r, base_c + c
            if grid[rr, cc] != BG:
                return None
            grid[rr, cc] = group["frame_color"]
            expected[rr, cc] = group["frame_color"]

        for r, c in group["marker_coords"]:
            rr, cc = base_r + r, base_c + c
            if grid[rr, cc] != BG:
                return None
            grid[rr, cc] = group["marker_color"]

        for r, c in group["shard_coords"]:
            rr, cc = base_r + r, base_c + c
            if grid[rr, cc] != BG:
                return None
            grid[rr, cc] = group["frame_color"]

        ir0, ic0, ir1, ic1 = group["interior_bbox"]
        expected[base_r + ir0 : base_r + ir1 + 1, base_c + ic0 : base_c + ic1 + 1] = group["marker_color"]

    try:
        solved = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        try:
            grid = _gen_attempt(rng)
        except Exception:
            grid = None
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
