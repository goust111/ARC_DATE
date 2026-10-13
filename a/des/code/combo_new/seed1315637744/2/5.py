from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
RED = 2
GEN_RETRIES = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _red_frame_mask(shape: tuple[int, int], depth: int) -> np.ndarray:
    h, w = shape
    r0, c0 = depth, depth
    r1, c1 = h - depth - 1, w - depth - 1
    mask = np.zeros((h, w), dtype=bool)
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    return mask


def _is_rect_frame(red_mask: np.ndarray) -> bool:
    coords = np.argwhere(red_mask)
    if coords.size == 0:
        return False
    r0, c0, r1, c1 = _bbox(coords)
    if r0 == r1 or c0 == c1:
        return False
    return np.array_equal(red_mask, _red_frame_mask(red_mask.shape, r0))


def _outer_boundary(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    padded = np.pad(mask, 1, constant_values=False)
    exterior = np.zeros_like(padded, dtype=bool)
    q = deque([(0, 0)])
    exterior[0, 0] = True
    ph, pw = padded.shape
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < ph and 0 <= nc < pw and not padded[nr, nc] and not exterior[nr, nc]:
                exterior[nr, nc] = True
                q.append((nr, nc))
    boundary = np.zeros_like(mask, dtype=bool)
    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            pr, pc = r + 1, c + 1
            for nr, nc in _neighbors4(pr, pc):
                if exterior[nr, nc]:
                    boundary[r, c] = True
                    break
    return boundary


def _marker_depth(stamp: np.ndarray) -> int:
    red = stamp == RED
    assert _is_rect_frame(red), "each stamp must contain one rectangular red frame"
    current = stamp != BG
    depth = 0
    while True:
        boundary = _outer_boundary(current)
        if np.any(boundary & red):
            return depth
        current = current & ~boundary
        depth += 1
        assert np.any(current & red), "red frame vanished before reaching the outer boundary"


def _extract_stamps(region: np.ndarray, tile_shape: tuple[int, int]) -> list[np.ndarray] | None:
    comps = _components(region != BG)
    if len(comps) < 2:
        return None
    stamps: list[np.ndarray] = []
    for comp in comps:
        r0, c0, r1, c1 = _bbox(comp)
        if (r1 - r0 + 1, c1 - c0 + 1) != tile_shape:
            return None
        crop = region[r0 : r1 + 1, c0 : c1 + 1]
        if not np.any(crop == RED):
            return None
        stamps.append(crop.copy())
    return stamps


def _find_layout(grid: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    h, w = grid.shape
    found: list[tuple[str, int, bool, np.ndarray, list[np.ndarray]]] = []

    def try_split(axis: str, idx: int, first: np.ndarray, second: np.ndarray) -> None:
        for canvas_on_first in (True, False):
            canvas = first if canvas_on_first else second
            stamps = second if canvas_on_first else first
            if canvas.size == 0 or stamps.size == 0 or not np.all(canvas == BG):
                continue
            stamp_crops = _extract_stamps(stamps, canvas.shape)
            if stamp_crops is not None:
                found.append((axis, idx, canvas_on_first, canvas.copy(), stamp_crops))

    for r in range(1, h - 1):
        if grid[r, 0] != BG and np.all(grid[r] == grid[r, 0]):
            try_split("row", r, grid[:r, :], grid[r + 1 :, :])
    for c in range(1, w - 1):
        if grid[0, c] != BG and np.all(grid[:, c] == grid[0, c]):
            try_split("col", c, grid[:, :c], grid[:, c + 1 :])

    unique = {(axis, idx, side): (canvas, stamps) for axis, idx, side, canvas, stamps in found}
    assert len(unique) == 1, "separator must be uniquely identifiable"
    return next(iter(unique.values()))


def _overlay_stamps(stamps: list[np.ndarray]) -> np.ndarray:
    ordered = sorted(((_marker_depth(stamp), idx, stamp) for idx, stamp in enumerate(stamps)))
    out = np.zeros_like(stamps[0])
    for _depth, _idx, stamp in ordered:
        mask = (stamp != BG) & (stamp != RED)
        out[mask] = stamp[mask]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Overlay the stamp tiles onto the blank canvas in the order encoded by each tile's red nested frame.

    Input structure:
    - One full non-black separator row or column splits the grid into two regions.
    - One side is a pure-black canvas whose shape matches a single stamp tile.
    - The other side contains several same-size stamps on black background, separated by black gaps.
    - Each stamp contains exactly one 1-pixel-thick red rectangular frame embedded at some nesting depth.

    Rule:
    - Detect the separator and identify the blank canvas side.
    - Extract each stamp as one non-black connected component of the stamp side.
    - For each stamp, measure how many foreground layers lie between the red frame and the stamp's outer contour.
    - Sort stamps from shallowest to deepest frame and overlay only their non-red pixels onto the blank canvas; later stamps overwrite earlier ones.

    Key insight:
    - Spatial order of the stamps is irrelevant; the red frame is an ordering marker, and its distance from the stamp exterior determines the stamping sequence.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    canvas, stamps = _find_layout(g)
    assert stamps, "expected at least two stamps"
    out = _overlay_stamps(stamps)
    assert out.shape == canvas.shape
    return out


def _stamp_is_valid(stamp: np.ndarray, depth: int) -> bool:
    if not np.any((stamp != BG) & (stamp != RED)):
        return False
    comps = _components(stamp != BG)
    if len(comps) != 1:
        return False
    if _bbox(comps[0]) != (0, 0, stamp.shape[0] - 1, stamp.shape[1] - 1):
        return False
    return _marker_depth(stamp) == depth


def _make_stamp(shape: tuple[int, int], color: int, depth: int, rng: random.Random) -> np.ndarray:
    h, w = shape
    stamp = np.full((h, w), color, dtype=int)
    stamp[_red_frame_mask(shape, depth)] = RED

    inner_r0, inner_r1 = depth + 1, h - depth - 2
    inner_c0, inner_c1 = depth + 1, w - depth - 2
    if inner_r0 <= inner_r1 and inner_c0 <= inner_c1:
        window_count = rng.randint(0, 3)
        for _ in range(window_count):
            for _attempt in range(20):
                r0 = rng.randint(inner_r0, inner_r1)
                r1 = rng.randint(r0, inner_r1)
                c0 = rng.randint(inner_c0, inner_c1)
                c1 = rng.randint(c0, inner_c1)
                candidate = stamp.copy()
                candidate[r0 : r1 + 1, c0 : c1 + 1] = BG
                if _stamp_is_valid(candidate, depth):
                    stamp = candidate
                    break

    assert _stamp_is_valid(stamp, depth)
    return stamp


def _pick_layout(rng: random.Random) -> tuple[str, bool, int, int, int, int]:
    for _ in range(200):
        orientation = rng.choice(["vertical", "horizontal"])
        canvas_first = rng.choice([False, True])
        stamp_count = rng.choice([2, 3])
        gap = rng.randint(1, 3)
        if orientation == "vertical":
            stamp_h = rng.randint(6, 12)
            stamp_w = rng.randint(5, 10)
            total = stamp_w + 1 + stamp_count * stamp_w + (stamp_count - 1) * gap
            if total > 30:
                continue
        else:
            stamp_h = rng.randint(5, 10)
            stamp_w = rng.randint(6, 12)
            total = stamp_h + 1 + stamp_count * stamp_h + (stamp_count - 1) * gap
            if total > 30:
                continue
        max_depth = min((stamp_h - 2) // 2, (stamp_w - 2) // 2)
        if max_depth + 1 >= stamp_count:
            return orientation, canvas_first, stamp_count, stamp_h, stamp_w, gap
    raise RuntimeError("failed to pick a feasible layout")


def _depths_for(shape: tuple[int, int], count: int, rng: random.Random) -> list[int]:
    max_depth = min((shape[0] - 2) // 2, (shape[1] - 2) // 2)
    if count == 3 and max_depth >= 2 and rng.random() < 0.6:
        depths = [0, 1, 2]
    else:
        depths = sorted(rng.sample(range(max_depth + 1), count))
    rng.shuffle(depths)
    return depths


def _build_stamp_strip(stamps: list[np.ndarray], gap: int, vertical: bool) -> np.ndarray:
    count = len(stamps)
    h, w = stamps[0].shape
    if vertical:
        strip = np.zeros((h, count * w + (count - 1) * gap), dtype=int)
        col = 0
        for stamp in stamps:
            strip[:, col : col + w] = stamp
            col += w + gap
        return strip
    strip = np.zeros((count * h + (count - 1) * gap, w), dtype=int)
    row = 0
    for stamp in stamps:
        strip[row : row + h, :] = stamp
        row += h + gap
    return strip


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    orientation, canvas_first, stamp_count, stamp_h, stamp_w, gap = _pick_layout(rng)
    shape = (stamp_h, stamp_w)

    palette = [c for c in range(1, 10) if c != RED]
    rng.shuffle(palette)
    sep_color = palette.pop()
    stamp_colors = [palette.pop() for _ in range(stamp_count)]
    depths = _depths_for(shape, stamp_count, rng)

    try:
        stamp_data = [
            {"depth": depth, "stamp": _make_stamp(shape, color, depth, rng)}
            for color, depth in zip(stamp_colors, depths)
        ]
    except AssertionError:
        return None

    rng.shuffle(stamp_data)
    stamps = [item["stamp"] for item in stamp_data]
    expected = _overlay_stamps(stamps)
    visible = {int(v) for v in np.unique(expected) if v != BG}
    if len(visible) < 2 or np.any(expected == RED):
        return None

    strip = _build_stamp_strip(stamps, gap, vertical=(orientation == "vertical"))
    canvas = np.zeros(shape, dtype=int)

    if orientation == "vertical":
        sep = np.full((stamp_h, 1), sep_color, dtype=int)
        parts = [canvas, sep, strip] if canvas_first else [strip, sep, canvas]
        grid = np.concatenate(parts, axis=1)
    else:
        sep = np.full((1, stamp_w), sep_color, dtype=int)
        parts = [canvas, sep, strip] if canvas_first else [strip, sep, canvas]
        grid = np.concatenate(parts, axis=0)

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
