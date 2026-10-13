from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
SIDES = ("top", "right", "bottom", "left")
SIDE_TO_ROT_K = {
    "top": 0,
    "right": 3,
    "bottom": 2,
    "left": 1,
}
NON_BG_COLORS = tuple(range(1, 10))
MAX_GENERATION_ATTEMPTS = 400


def _find_components(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _is_square_ring(coords: np.ndarray) -> bool:
    rows = coords[:, 0]
    cols = coords[:, 1]
    r0 = int(rows.min())
    r1 = int(rows.max())
    c0 = int(cols.min())
    c1 = int(cols.max())
    side = r1 - r0 + 1
    if side < 3 or c1 - c0 + 1 != side:
        return False
    if len(coords) != 4 * side - 4:
        return False
    return bool(np.all((rows == r0) | (rows == r1) | (cols == c0) | (cols == c1)))


def _outside_line(grid: np.ndarray, bbox: tuple[int, int, int, int], side: str) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    if side == "top":
        if r0 == 0:
            return np.empty((0,), dtype=int)
        return grid[r0 - 1, c0 : c1 + 1]
    if side == "bottom":
        if r1 + 1 >= grid.shape[0]:
            return np.empty((0,), dtype=int)
        return grid[r1 + 1, c0 : c1 + 1]
    if side == "left":
        if c0 == 0:
            return np.empty((0,), dtype=int)
        return grid[r0 : r1 + 1, c0 - 1]
    if c1 + 1 >= grid.shape[1]:
        return np.empty((0,), dtype=int)
    return grid[r0 : r1 + 1, c1 + 1]


def _find_containers(grid: np.ndarray) -> list[dict]:
    containers: list[dict] = []
    for color in map(int, np.unique(grid)):
        if color == BG:
            continue
        for coords in _find_components(grid, color):
            if not _is_square_ring(coords):
                continue
            rows = coords[:, 0]
            cols = coords[:, 1]
            bbox = (int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max()))
            counts = {}
            for side in SIDES:
                line = _outside_line(grid, bbox, side)
                counts[side] = int(np.count_nonzero(line != BG))
            marked_sides = [side for side, count in counts.items() if count > 0]
            assert len(marked_sides) == 1, "each container must have markers on exactly one side"
            side = marked_sides[0]
            containers.append(
                {
                    "border_color": color,
                    "bbox": bbox,
                    "side": side,
                    "marker_count": counts[side],
                }
            )
    containers.sort(key=lambda item: item["bbox"])
    return containers


def _apply_inner_transform(inner: np.ndarray, side: str, mirror: bool) -> np.ndarray:
    transformed = np.rot90(inner, SIDE_TO_ROT_K[side])
    if mirror:
        transformed = np.fliplr(transformed)
    return transformed


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each container's interior according to the side of its exterior markers,
    then horizontally flip the rotated interior when the marker count is odd.

    - Top markers:    keep orientation.
    - Right markers:  rotate 90 degrees clockwise.
    - Bottom markers: rotate 180 degrees.
    - Left markers:   rotate 90 degrees counterclockwise.
    - Odd marker count adds a left-right mirror after the rotation.

    Borders and markers stay unchanged; only the pixels strictly inside each
    square border are transformed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    containers = _find_containers(g)
    assert containers, "expected at least one container"

    out = g.copy()
    for container in containers:
        r0, c0, r1, c1 = container["bbox"]
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        assert inner.shape[0] == inner.shape[1] and inner.shape[0] >= 1
        transformed = _apply_inner_transform(
            inner,
            container["side"],
            container["marker_count"] % 2 == 1,
        )
        out[r0 + 1 : r1, c0 + 1 : c1] = transformed
    return out


def _random_pattern(size: int, forbidden: set[int]) -> np.ndarray | None:
    usable = [color for color in NON_BG_COLORS if color not in forbidden]
    if len(usable) < 2:
        return None
    palette_size = random.randint(2, min(4, len(usable)))
    palette = random.sample(usable, palette_size)
    pattern = np.array(
        [[random.choice(palette) for _ in range(size)] for _ in range(size)],
        dtype=int,
    )
    if len(set(map(int, pattern.ravel()))) < 2:
        return None
    original = pattern
    for side in SIDES:
        for mirror in (False, True):
            if side == "top" and not mirror:
                continue
            if np.array_equal(_apply_inner_transform(original, side, mirror), original):
                return None
    return pattern


def _marker_segment(side_len: int, odd: bool) -> tuple[int, int]:
    counts = [count for count in range(1, side_len + 1) if (count % 2 == 1) == odd]
    count = random.choice(counts)
    start = random.randint(0, side_len - count)
    return start, count


def _reserve_box(r0: int, c0: int, side: int) -> tuple[int, int, int, int]:
    return r0 - 1, c0 - 1, r0 + side, c0 + side


def _box_is_free(occupied: np.ndarray, box: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = box
    return not bool(occupied[r0 : r1 + 1, c0 : c1 + 1].any())


def _mark_box(occupied: np.ndarray, box: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = box
    occupied[r0 : r1 + 1, c0 : c1 + 1] = True


def _draw_container(grid: np.ndarray, spec: dict) -> None:
    r0 = spec["r0"]
    c0 = spec["c0"]
    side = spec["side_len"]
    border = spec["border_color"]
    marker = spec["marker_color"]

    grid[r0, c0 : c0 + side] = border
    grid[r0 + side - 1, c0 : c0 + side] = border
    grid[r0 : r0 + side, c0] = border
    grid[r0 : r0 + side, c0 + side - 1] = border
    grid[r0 + 1 : r0 + side - 1, c0 + 1 : c0 + side - 1] = spec["pattern"]

    start = spec["marker_start"]
    count = spec["marker_count"]
    if spec["side"] == "top":
        grid[r0 - 1, c0 + start : c0 + start + count] = marker
    elif spec["side"] == "bottom":
        grid[r0 + side, c0 + start : c0 + start + count] = marker
    elif spec["side"] == "left":
        grid[r0 + start : r0 + start + count, c0 - 1] = marker
    else:
        grid[r0 + start : r0 + start + count, c0 + side] = marker


def _expected_output(inp: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = inp.copy()
    for spec in specs:
        r0 = spec["r0"]
        c0 = spec["c0"]
        side = spec["side_len"]
        inner = inp[r0 + 1 : r0 + side - 1, c0 + 1 : c0 + side - 1]
        transformed = _apply_inner_transform(inner, spec["side"], spec["marker_count"] % 2 == 1)
        out[r0 + 1 : r0 + side - 1, c0 + 1 : c0 + side - 1] = transformed
    return out


def _sample_specs() -> list[dict] | None:
    count = random.randint(2, 4)
    while True:
        directions = [random.choice(SIDES) for _ in range(count)]
        mirrors = [bool(random.getrandbits(1)) for _ in range(count)]
        if len(set(directions)) >= min(2, count) and len(set(mirrors)) >= min(2, count):
            break

    border_colors = random.sample(NON_BG_COLORS, count)
    non_border = [color for color in NON_BG_COLORS if color not in border_colors]
    specs: list[dict] = []
    non_identity = False
    for i in range(count):
        side_len = random.randint(5, 8)
        start, marker_count = _marker_segment(side_len, mirrors[i])
        marker_color = random.choice(non_border)
        pattern = _random_pattern(side_len - 2, {border_colors[i], marker_color})
        if pattern is None:
            return None
        specs.append(
            {
                "side_len": side_len,
                "side": directions[i],
                "marker_start": start,
                "marker_count": marker_count,
                "border_color": border_colors[i],
                "marker_color": marker_color,
                "pattern": pattern,
            }
        )
        if directions[i] != "top" or marker_count % 2 == 1:
            non_identity = True
    if not non_identity:
        return None
    return specs


def _place_specs(grid_shape: tuple[int, int], specs: list[dict]) -> bool:
    h, w = grid_shape
    occupied = np.zeros((h, w), dtype=bool)
    order = sorted(range(len(specs)), key=lambda idx: specs[idx]["side_len"], reverse=True)
    for idx in order:
        spec = specs[idx]
        side = spec["side_len"]
        positions: list[tuple[int, int]] = []
        for r0 in range(1, h - side):
            for c0 in range(1, w - side):
                box = _reserve_box(r0, c0, side)
                if _box_is_free(occupied, box):
                    positions.append((r0, c0))
        if not positions:
            return False
        r0, c0 = random.choice(positions)
        spec["r0"] = r0
        spec["c0"] = c0
        _mark_box(occupied, _reserve_box(r0, c0, side))
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(18, 28)
    w = random.randint(18, 28)
    specs = _sample_specs()
    if specs is None:
        return None
    if not _place_specs((h, w), specs):
        return None

    inp = np.full((h, w), BG, dtype=int)
    for spec in specs:
        _draw_container(inp, spec)

    detected = _find_containers(inp)
    if len(detected) != len(specs):
        return None

    expected = _expected_output(inp, specs)
    if np.array_equal(inp, expected):
        return None
    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
