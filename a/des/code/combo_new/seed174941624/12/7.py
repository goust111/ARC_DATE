from __future__ import annotations

import random

import numpy as np


BG = 0
MARKER = 2
FRAME = 8
PATTERN_COLORS = (1, 3, 4, 5, 6, 7, 9)

KIND_TO_OFFSET = {
    "TL": (-1, -1),
    "TR": (-1, 1),
    "BL": (1, -1),
    "BR": (1, 1),
}

GENERATION_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 200
PATTERN_ATTEMPTS = 200


def _ray_end(g: np.ndarray, r: int, c: int, dr: int, dc: int, color: int) -> tuple[int, int]:
    h, w = g.shape
    rr, cc = int(r), int(c)
    while 0 <= rr + dr < h and 0 <= cc + dc < w and int(g[rr + dr, cc + dc]) == int(color):
        rr += dr
        cc += dc
    return rr, cc


def _marker_positions(r0: int, c0: int, r1: int, c1: int) -> dict[str, tuple[int, int]]:
    return {
        "TL": (int(r0 - 1), int(c0 - 1)),
        "TR": (int(r0 - 1), int(c1 + 1)),
        "BL": (int(r1 + 1), int(c0 - 1)),
        "BR": (int(r1 + 1), int(c1 + 1)),
    }


def _border_is_frame(g: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> bool:
    return bool(
        np.all(g[r0, c0 : c1 + 1] == FRAME)
        and np.all(g[r1, c0 : c1 + 1] == FRAME)
        and np.all(g[r0 : r1 + 1, c0] == FRAME)
        and np.all(g[r0 : r1 + 1, c1] == FRAME)
    )


def _outside_ring_is_bg(g: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> bool:
    h, w = g.shape
    if not (1 <= r0 and r1 + 1 < h and 1 <= c0 and c1 + 1 < w):
        return False
    return bool(
        np.all(g[r0 - 1, c0 : c1 + 1] == BG)
        and np.all(g[r1 + 1, c0 : c1 + 1] == BG)
        and np.all(g[r0 : r1 + 1, c0 - 1] == BG)
        and np.all(g[r0 : r1 + 1, c1 + 1] == BG)
    )


def _candidate_bbox_from_marker(g: np.ndarray, mr: int, mc: int, kind: str) -> tuple[int, int, int, int] | None:
    dr, dc = KIND_TO_OFFSET[kind]
    corner_r = int(mr - dr)
    corner_c = int(mc - dc)
    h, w = g.shape
    if not (0 <= corner_r < h and 0 <= corner_c < w):
        return None
    if int(g[corner_r, corner_c]) != FRAME:
        return None

    if kind == "TL":
        _r0, c1 = _ray_end(g, corner_r, corner_c, 0, 1, FRAME)
        r1, _c0 = _ray_end(g, corner_r, corner_c, 1, 0, FRAME)
        r0, c0 = corner_r, corner_c
    elif kind == "TR":
        _r0, c0 = _ray_end(g, corner_r, corner_c, 0, -1, FRAME)
        r1, _c1 = _ray_end(g, corner_r, corner_c, 1, 0, FRAME)
        r0, c1 = corner_r, corner_c
    elif kind == "BL":
        r0, _c0 = _ray_end(g, corner_r, corner_c, -1, 0, FRAME)
        _r1, c1 = _ray_end(g, corner_r, corner_c, 0, 1, FRAME)
        r1, c0 = corner_r, corner_c
    elif kind == "BR":
        r0, _c1 = _ray_end(g, corner_r, corner_c, -1, 0, FRAME)
        _r1, c0 = _ray_end(g, corner_r, corner_c, 0, -1, FRAME)
        r1, c1 = corner_r, corner_c
    else:
        raise AssertionError(f"unknown marker kind: {kind}")

    if r1 - r0 < 4 or c1 - c0 < 4:
        return None
    return int(r0), int(c0), int(r1), int(c1)


def _detect_frame(g: np.ndarray, mr: int, mc: int) -> tuple[str, tuple[int, int, int, int]]:
    valid: list[tuple[str, tuple[int, int, int, int]]] = []
    for kind in ("TL", "TR", "BL", "BR"):
        bbox = _candidate_bbox_from_marker(g, mr, mc, kind)
        if bbox is None:
            continue
        r0, c0, r1, c1 = bbox
        if not _border_is_frame(g, r0, c0, r1, c1):
            continue
        if not _outside_ring_is_bg(g, r0, c0, r1, c1):
            continue

        markers = _marker_positions(r0, c0, r1, c1)
        colors = {name: int(g[pos]) for name, pos in markers.items()}
        if colors[kind] != MARKER:
            continue
        if any(colors[name] != BG for name in markers if name != kind):
            continue
        valid.append((kind, bbox))

    assert len(valid) == 1, f"red marker at {(mr, mc)} must identify exactly one frame"
    return valid[0]


def _transform_inside(pattern: np.ndarray, kind: str) -> np.ndarray:
    if kind == "TL":
        return np.rot90(pattern, 1)
    if kind == "TR":
        return np.rot90(pattern, 3)
    if kind == "BL":
        return np.flipud(pattern)
    if kind == "BR":
        return np.fliplr(pattern)
    raise AssertionError(f"unknown marker kind: {kind}")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transform each fenced interior pattern according to the red pixel placed at one outer corner of its fence.

    Input structure:
    - The background is black.
    - Several one-pixel-thick azure rectangular fences appear on the grid, and the fences do not touch each other.
    - Each fence encloses a non-symmetric multicolor interior pattern, with at least a 3x3 interior area.
    - Exactly one of the four diagonal cells just outside a fence's corners contains a single red marker pixel.

    Rule:
    - For each fence, locate which outer diagonal corner cell contains the red marker.
    - A top-left outer marker means rotate the interior 90 degrees counterclockwise; a top-right outer marker means rotate it 90 degrees clockwise.
    - A bottom-left outer marker means flip the interior across the horizontal midline; a bottom-right outer marker means flip it across the vertical midline.
    - Write the transformed interior back into the original fenced area, keep the azure fence unchanged, and erase every red marker to black.

    Key insight:
    - The red marker's corner position is a code for which geometric transform to apply to the pattern inside the matching azure fence.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    marker_coords = [tuple(map(int, rc)) for rc in np.argwhere(g == MARKER)]
    assert marker_coords, "expected at least one red marker"

    seen_boxes: set[tuple[int, int, int, int]] = set()
    for mr, mc in marker_coords:
        kind, bbox = _detect_frame(g, mr, mc)
        assert bbox not in seen_boxes, "each frame should have exactly one red marker"
        seen_boxes.add(bbox)

        r0, c0, r1, c1 = bbox
        inside = g[r0 + 1 : r1, c0 + 1 : c1]
        transformed = _transform_inside(inside, kind)
        assert transformed.shape == inside.shape, "90-degree rotations require a square fenced interior"
        out[r0 + 1 : r1, c0 + 1 : c1] = transformed
        out[mr, mc] = BG

    return out


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _variant_key(arr: np.ndarray) -> bytes:
    return arr.tobytes() + str(tuple(arr.shape)).encode()


def _sample_pattern(size: int) -> np.ndarray | None:
    total = int(size * size)
    min_filled = max(4, total // 3)
    max_filled = min(total - 1, max(min_filled, (2 * total) // 3))

    for _ in range(PATTERN_ATTEMPTS):
        pattern = np.zeros((size, size), dtype=int)
        color_count = random.randint(2, min(4, len(PATTERN_COLORS), total))
        palette = random.sample(PATTERN_COLORS, color_count)
        filled = random.randint(min_filled, max_filled)

        cells = random.sample(range(total), filled)
        colors = palette.copy()
        colors.extend(random.choice(palette) for _ in range(filled - len(palette)))
        random.shuffle(colors)

        for idx, color in zip(cells, colors):
            rr, cc = divmod(int(idx), size)
            pattern[rr, cc] = int(color)

        variants = [
            pattern,
            np.rot90(pattern, 1),
            np.rot90(pattern, 3),
            np.flipud(pattern),
            np.fliplr(pattern),
        ]
        keys = {_variant_key(variant) for variant in variants}
        if len(keys) != len(variants):
            continue
        return pattern

    return None


def _place_frame(
    occupied: list[tuple[int, int, int, int]],
    h: int,
    w: int,
    outer_size: int,
) -> tuple[int, int, int, int] | None:
    if h - outer_size - 2 < 2 or w - outer_size - 2 < 2:
        return None

    for _ in range(PLACEMENT_ATTEMPTS):
        r0 = random.randint(2, h - outer_size - 2)
        c0 = random.randint(2, w - outer_size - 2)
        r1 = r0 + outer_size - 1
        c1 = c0 + outer_size - 1
        clearance = (int(r0 - 2), int(c0 - 2), int(r1 + 2), int(c1 + 2))
        if any(_boxes_overlap(clearance, other) for other in occupied):
            continue
        occupied.append(clearance)
        return int(r0), int(c0), int(r1), int(c1)
    return None


def _draw_frame(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> None:
    grid[r0, c0 : c1 + 1] = FRAME
    grid[r1, c0 : c1 + 1] = FRAME
    grid[r0 : r1 + 1, c0] = FRAME
    grid[r0 : r1 + 1, c1] = FRAME


def _draw_marker(grid: np.ndarray, kind: str, r0: int, c0: int, r1: int, c1: int) -> None:
    markers = _marker_positions(r0, c0, r1, c1)
    mr, mc = markers[kind]
    grid[mr, mc] = MARKER


def _kinds_for_sample(n_frames: int) -> list[str]:
    assert n_frames >= 4
    kinds = list(random.sample(("TL", "TR", "BL", "BR"), 4))
    for _ in range(n_frames - 4):
        kinds.append(random.choice(("TL", "TR", "BL", "BR")))
    random.shuffle(kinds)
    return kinds


def _generate_one() -> np.ndarray | None:
    n_frames = 5 if random.random() < 0.35 else 4
    inner_hi = 5 if n_frames == 5 else 6
    min_dim = 26 if n_frames == 5 else 22
    h = random.randint(min_dim, 30)
    w = random.randint(min_dim, 30)
    grid = np.full((h, w), BG, dtype=int)

    occupied: list[tuple[int, int, int, int]] = []
    for kind in _kinds_for_sample(n_frames):
        inner_size = random.randint(3, inner_hi)
        pattern = _sample_pattern(inner_size)
        if pattern is None:
            return None

        bbox = _place_frame(occupied, h, w, inner_size + 2)
        if bbox is None:
            return None
        r0, c0, r1, c1 = bbox
        _draw_frame(grid, r0, c0, r1, c1)
        grid[r0 + 1 : r1, c0 + 1 : c1] = pattern
        _draw_marker(grid, kind, r0, c0, r1, c1)

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _generate_one()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid fenced-transform input")
