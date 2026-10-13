from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BLACK = 0
GREY = 5
KEY_PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)

MAX_GENERATION_ATTEMPTS = 400

BBox = tuple[int, int, int, int]


def _background_color(grid: np.ndarray) -> int:
    return int(Counter(int(v) for v in grid.ravel()).most_common(1)[0][0])


def _layer_values(block: np.ndarray, layer: int) -> list[int]:
    side = int(block.shape[0])
    lo = int(layer)
    hi = side - layer - 1
    if lo == hi:
        return [int(block[lo, lo])]
    vals = []
    vals.extend(int(v) for v in block[lo, lo : hi + 1])
    vals.extend(int(v) for v in block[hi, lo : hi + 1])
    vals.extend(int(v) for v in block[lo + 1 : hi, lo])
    vals.extend(int(v) for v in block[lo + 1 : hi, hi])
    return vals


def _concentric_colors(block: np.ndarray, bg: int) -> tuple[int, ...] | None:
    side = int(block.shape[0])
    if side < 3 or block.shape[0] != block.shape[1]:
        return None
    if np.any(block == int(bg)):
        return None

    colors = []
    for layer in range((side + 1) // 2):
        vals = _layer_values(block, layer)
        unique = set(vals)
        if len(unique) != 1:
            return None
        colors.append(int(vals[0]))

    if len(colors) < 2 or len(set(colors)) != len(colors):
        return None
    return tuple(colors)


def _corner_slices(shape: tuple[int, int], corner: str, side: int) -> tuple[slice, slice]:
    H, W = shape
    if corner == "tl":
        return slice(0, side), slice(0, side)
    if corner == "tr":
        return slice(0, side), slice(W - side, W)
    if corner == "bl":
        return slice(H - side, H), slice(0, side)
    if corner == "br":
        return slice(H - side, H), slice(W - side, W)
    raise ValueError(f"unknown corner {corner}")


def _find_key(grid: np.ndarray, bg: int) -> tuple[tuple[int, ...], np.ndarray]:
    H, W = grid.shape
    candidates: list[tuple[tuple[int, ...], np.ndarray]] = []
    max_side = min(H, W)

    for side in range(3, max_side + 1):
        for corner in ("tl", "tr", "bl", "br"):
            rs, cs = _corner_slices((H, W), corner, side)
            block = grid[rs, cs]
            colors = _concentric_colors(block, bg)
            if colors is None:
                continue
            mask = np.zeros((H, W), dtype=bool)
            mask[rs, cs] = True
            candidates.append((colors, mask))

    assert len(candidates) == 1, "expected one corner concentric-ring key"
    return candidates[0]


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    H, W = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []

    for r0, c0 in np.argwhere(mask):
        r = int(r0)
        c = int(c0)
        if seen[r, c]:
            continue

        coords: list[tuple[int, int]] = []
        q: deque[tuple[int, int]] = deque([(r, c)])
        seen[r, c] = True
        while q:
            rr, cc = q.popleft()
            coords.append((rr, cc))
            for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _frame_area(box: BBox) -> int:
    r0, c0, r1, c1 = box
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    return 2 * h + 2 * w - 4


def _is_rectangular_frame(coords: np.ndarray, box: BBox) -> bool:
    r0, c0, r1, c1 = box
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3 or len(coords) != _frame_area(box):
        return False
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        if rr not in (r0, r1) and cc not in (c0, c1):
            return False
    return True


def _find_gray_frames(grid: np.ndarray) -> list[dict]:
    frames = []
    for coords in _components4(grid == GREY):
        box = _bbox(coords)
        assert _is_rectangular_frame(coords, box), "every gray component must be a hollow rectangular frame"
        frames.append({"bbox": box, "coords": coords})

    assert frames, "expected at least one gray frame"
    return frames


def _contains(outer: BBox, inner: BBox) -> bool:
    return outer[0] < inner[0] and outer[1] < inner[1] and outer[2] > inner[2] and outer[3] > inner[3]


def _depths(frames: list[dict]) -> list[int]:
    boxes = [frame["bbox"] for frame in frames]
    return [1 + sum(1 for other in boxes if other != box and _contains(other, box)) for box in boxes]


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor every gray rectangular frame by its nesting depth using the concentric-ring key.

    Input structure:
    - One corner contains a solid square key made of one-cell-thick, tightly
      adjacent concentric square rings, with a distinct color on each layer.
    - Away from the key are gray hollow rectangular frames; separate frame
      groups are disjoint, while nested frames are fully contained with a gap.
    - Frame size and position are not tied to the recolor result.

    Rule:
    - Read the key colors from the outer ring inward.
    - Delete the key object by replacing its cells with the background color.
    - For each gray frame, compute depth as one plus the number of other gray
      frames whose bounding boxes strictly contain it.
    - Recolor the whole frame with the depth-th key color, cycling through the
      key sequence when the depth exceeds the number of key layers.

    Key insight:
    The key is a visible depth-to-color lookup table; the correct color of a
    frame depends on containment depth, not on the frame's size or location.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _background_color(g)
    key_colors, key_mask = _find_key(g, bg)

    working = g.copy()
    working[key_mask] = bg
    frames = _find_gray_frames(working)
    depths = _depths(frames)

    out = working.copy()
    for frame, depth in zip(frames, depths):
        coords = frame["coords"]
        color = key_colors[(int(depth) - 1) % len(key_colors)]
        out[coords[:, 0], coords[:, 1]] = int(color)
    return out


def _draw_key(grid: np.ndarray, colors: tuple[int, ...]) -> np.ndarray:
    side = 2 * len(colors) - 1
    for layer, color in enumerate(colors):
        lo = layer
        hi = side - layer - 1
        grid[lo, lo : hi + 1] = int(color)
        grid[hi, lo : hi + 1] = int(color)
        grid[lo : hi + 1, lo] = int(color)
        grid[lo : hi + 1, hi] = int(color)
    return grid


def _draw_frame(grid: np.ndarray, box: BBox, color: int = GREY) -> None:
    r0, c0, r1, c1 = box
    grid[r0, c0 : c1 + 1] = int(color)
    grid[r1, c0 : c1 + 1] = int(color)
    grid[r0 : r1 + 1, c0] = int(color)
    grid[r0 : r1 + 1, c1] = int(color)


def _nested_boxes(outer: BBox, depth: int) -> list[BBox]:
    r0, c0, r1, c1 = outer
    return [(r0 + 2 * i, c0 + 2 * i, r1 - 2 * i, c1 - 2 * i) for i in range(depth)]


def _required_outer_size(depth: int) -> int:
    return 3 + 4 * (int(depth) - 1)


def _intersects(a: BBox, b: BBox, pad: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    ar0 -= pad
    ac0 -= pad
    ar1 += pad
    ac1 += pad
    br0 -= pad
    bc0 -= pad
    br1 += pad
    bc1 += pad
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _sample_deep_outer(H: int, W: int, key_side: int, depth: int, rng: random.Random) -> BBox | None:
    required = _required_outer_size(depth)
    zones = [
        (key_side + 1, 1, H - 2, W - 2),
        (1, key_side + 1, H - 2, W - 2),
    ]
    rng.shuffle(zones)

    for zr0, zc0, zr1, zc1 in zones:
        zone_h = zr1 - zr0 + 1
        zone_w = zc1 - zc0 + 1
        if zone_h < required or zone_w < required:
            continue
        max_h = min(zone_h, required + 5)
        max_w = min(zone_w, required + 7)
        h = rng.randint(required, max_h)
        w = rng.randint(required, max_w)
        r0 = rng.randint(zr0, zr1 - h + 1)
        c0 = rng.randint(zc0, zc1 - w + 1)
        return r0, c0, r0 + h - 1, c0 + w - 1
    return None


def _sample_extra_outer(
    H: int,
    W: int,
    key_box: BBox,
    existing: list[BBox],
    depth: int,
    rng: random.Random,
) -> BBox | None:
    required = _required_outer_size(depth)
    if H - 2 < required or W - 2 < required:
        return None

    for _ in range(220):
        max_h = min(H - 2, required + 7)
        max_w = min(W - 2, required + 8)
        h = rng.randint(required, max_h)
        w = rng.randint(required, max_w)
        r0 = rng.randint(1, H - h - 1)
        c0 = rng.randint(1, W - w - 1)
        box = (r0, c0, r0 + h - 1, c0 + w - 1)
        if _intersects(box, key_box, pad=1):
            continue
        if any(_intersects(box, other, pad=1) for other in existing):
            continue
        return box
    return None


def _frame_border_coords(box: BBox) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = box
    coords = []
    for c in range(c0, c1 + 1):
        coords.append((r0, c))
        coords.append((r1, c))
    for r in range(r0 + 1, r1):
        coords.append((r, c0))
        coords.append((r, c1))
    return coords


def _expected_output(grid: np.ndarray, key_side: int, boxes: list[BBox], key_colors: tuple[int, ...]) -> np.ndarray:
    out = grid.copy()
    out[:key_side, :key_side] = BLACK

    depths = [1 + sum(1 for other in boxes if other != box and _contains(other, box)) for box in boxes]
    for box, depth in zip(boxes, depths):
        color = key_colors[(depth - 1) % len(key_colors)]
        for r, c in _frame_border_coords(box):
            out[r, c] = int(color)
    return out


def _semantically_rich(boxes: list[BBox], key_len: int) -> bool:
    depths = [1 + sum(1 for other in boxes if other != box and _contains(other, box)) for box in boxes]
    if max(depths) <= key_len:
        return False

    by_depth: dict[int, set[int]] = {}
    for box, depth in zip(boxes, depths):
        r0, c0, r1, c1 = box
        by_depth.setdefault(depth, set()).add((r1 - r0 + 1) * (c1 - c0 + 1))
    return any(len(areas) >= 2 for areas in by_depth.values())


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    H = rng.randint(28, 30)
    W = rng.randint(28, 30)
    key_len = rng.randint(2, 4)
    key_side = 2 * key_len - 1
    key_colors = tuple(rng.sample(KEY_PALETTE, key_len))
    cycle_depth = key_len + 1

    deep_outer = _sample_deep_outer(H, W, key_side, cycle_depth, rng)
    if deep_outer is None:
        return None

    key_box = (0, 0, key_side - 1, key_side - 1)
    boxes = _nested_boxes(deep_outer, cycle_depth)
    top_level_boxes = [deep_outer]

    extra_count = rng.randint(1, 3)
    for i in range(extra_count):
        preferred_depths = [2, 1, 3] if i == 0 else [rng.choice((1, 2, 3)), 1]
        placed = False
        for depth in preferred_depths:
            outer = _sample_extra_outer(H, W, key_box, top_level_boxes, depth, rng)
            if outer is None:
                continue
            boxes.extend(_nested_boxes(outer, depth))
            top_level_boxes.append(outer)
            placed = True
            break
        if i == 0 and not placed:
            return None

    if not _semantically_rich(boxes, key_len):
        return None

    grid = np.full((H, W), BLACK, dtype=int)
    _draw_key(grid, key_colors)
    for box in boxes:
        _draw_frame(grid, box)

    expected = _expected_output(grid, key_side, boxes, key_colors)
    k = rng.randint(0, 3)
    inp = np.rot90(grid, k).copy()
    expected = np.rot90(expected, k).copy()

    if inp.shape[0] > 30 or inp.shape[1] > 30 or expected.shape[0] > 30 or expected.shape[1] > 30:
        return None
    if np.array_equal(inp, expected):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        rng = random.Random()
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid deep-key input")
