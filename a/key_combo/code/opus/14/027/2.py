from __future__ import annotations

from collections import deque

import numpy as np


BG = 0
BLUE = 1
GREY = 5
FILL_COLORS = (2, 3, 4, 6, 7, 8, 9)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise AssertionError("expected non-empty mask")
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _find_marker(grid: np.ndarray) -> tuple[tuple[int, int], int]:
    coords = np.argwhere((grid != BG) & (grid != BLUE) & (grid != GREY))
    assert coords.shape[0] == 1, "expected exactly one internal marker"
    r, c = map(int, coords[0])
    return (r, c), int(grid[r, c])


def _find_grey(grid: np.ndarray) -> tuple[int, int]:
    coords = np.argwhere(grid == GREY)
    assert coords.shape[0] == 1, "expected exactly one grey control pixel"
    return tuple(map(int, coords[0]))


def _opening_direction(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> str:
    r0, c0, r1, c1 = bbox
    counts = {
        "up": int(np.count_nonzero(grid[r0, c0 : c1 + 1] == BLUE)),
        "down": int(np.count_nonzero(grid[r1, c0 : c1 + 1] == BLUE)),
        "left": int(np.count_nonzero(grid[r0 : r1 + 1, c0] == BLUE)),
        "right": int(np.count_nonzero(grid[r0 : r1 + 1, c1] == BLUE)),
    }
    smallest = min(counts.values())
    openings = [side for side, count in counts.items() if count == smallest]
    assert len(openings) == 1, "cup must have a unique opening side"
    return openings[0]


def _axis_from_grey(
    bbox: tuple[int, int, int, int], grey: tuple[int, int], opening: str
) -> tuple[str, float]:
    r0, c0, r1, c1 = bbox
    gr, gc = grey
    in_rows = r0 <= gr <= r1
    in_cols = c0 <= gc <= c1
    assert in_rows ^ in_cols, "grey must lie in exactly one bbox band"

    if in_rows:
        axis_kind = "vertical"
        axis_value = (c0 + c1) / 2.0
        assert opening in {"left", "right"}, "row-band grey is only valid for side-opening cups"
        if opening == "left":
            assert gc < c0, "left-opening cup must have grey pixel on its mouth side"
        else:
            assert gc > c1, "right-opening cup must have grey pixel on its mouth side"
    else:
        axis_kind = "horizontal"
        axis_value = (r0 + r1) / 2.0
        assert opening in {"up", "down"}, "column-band grey is only valid for vertical-opening cups"
        if opening == "up":
            assert gr < r0, "up-opening cup must have grey pixel on its mouth side"
        else:
            assert gr > r1, "down-opening cup must have grey pixel on its mouth side"

    return axis_kind, float(axis_value)


def _reflect_blue(
    coords: np.ndarray, bbox: tuple[int, int, int, int], axis_kind: str, axis_value: float
) -> np.ndarray:
    mirrored = coords.copy()
    if axis_kind == "vertical":
        mirrored[:, 1] = np.rint(2.0 * axis_value - mirrored[:, 1]).astype(int)
    else:
        mirrored[:, 0] = np.rint(2.0 * axis_value - mirrored[:, 0]).astype(int)

    r0, c0, r1, c1 = bbox
    assert np.all((r0 <= mirrored[:, 0]) & (mirrored[:, 0] <= r1))
    assert np.all((c0 <= mirrored[:, 1]) & (mirrored[:, 1] <= c1))
    return mirrored


def _collect_component(grid: np.ndarray, seed: tuple[int, int]) -> tuple[set[tuple[int, int]], bool]:
    H, W = grid.shape
    sr, sc = seed
    assert 0 <= sr < H and 0 <= sc < W and int(grid[sr, sc]) != BLUE

    q = deque([seed])
    seen = {seed}
    touches_boundary = False

    while q:
        r, c = q.popleft()
        if r in (0, H - 1) or c in (0, W - 1):
            touches_boundary = True
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in seen and int(grid[nr, nc]) != BLUE:
                seen.add((nr, nc))
                q.append((nr, nc))

    return seen, touches_boundary


def _fill_component(
    grid: np.ndarray, seed: tuple[int, int], fill_color: int
) -> tuple[np.ndarray, set[tuple[int, int]]]:
    seen, touches_boundary = _collect_component(grid, seed)
    assert not touches_boundary, "reflection must create an enclosed fillable region"
    out = grid.copy()
    for r, c in seen:
        out[r, c] = int(fill_color)
    return out, seen


def _parse_input(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    blue_mask = g == BLUE
    assert blue_mask.any(), "expected one blue cup outline"
    bbox = _bbox(blue_mask)
    marker, marker_color = _find_marker(g)
    grey = _find_grey(g)
    r0, c0, r1, c1 = bbox
    mr, mc = marker
    assert r0 <= mr <= r1 and c0 <= mc <= c1, "marker must lie inside the cup bbox"
    opening = _opening_direction(g, bbox)
    axis_kind, axis_value = _axis_from_grey(bbox, grey, opening)
    return {
        "grid": g,
        "bbox": bbox,
        "marker": marker,
        "marker_color": marker_color,
        "grey": grey,
        "opening": opening,
        "axis_kind": axis_kind,
        "axis_value": axis_value,
    }


def _simulate_transform(grid: np.ndarray, axis_override: str | None = None) -> np.ndarray | None:
    info = _parse_input(grid)
    g = info["grid"]
    bbox = info["bbox"]
    marker = info["marker"]
    marker_color = info["marker_color"]
    axis_kind = info["axis_kind"] if axis_override is None else axis_override
    axis_value = (bbox[1] + bbox[3]) / 2.0 if axis_kind == "vertical" else (bbox[0] + bbox[2]) / 2.0

    out = g.copy()
    out[out == GREY] = BG

    blue_coords = np.argwhere(out == BLUE)
    mirrored = _reflect_blue(blue_coords, bbox, axis_kind, axis_value)
    out[mirrored[:, 0], mirrored[:, 1]] = BLUE

    try:
        filled, _ = _fill_component(out, marker, marker_color)
    except AssertionError:
        return None
    return filled


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer the hidden reflection axis from the grey control pixel, mirror the blue cup, and fill the newly closed interior with the marker color.

    Input structure:
    - The grid has a black background, one connected blue cup outline with exactly one open side, one non-blue/non-black marker pixel inside the cup bbox, and one grey control pixel outside the cup.
    - The blue cup is a one-pixel-thick U/C-shaped contour whose bbox defines the implicit reflection center line.
    - The grey pixel lies in exactly one bbox band: either within the cup's row span or within its column span, but not both.

    Rule:
    - Detect which side of the blue contour is open.
    - Use the grey pixel to choose the reflection axis: row-band grey means reflect across the cup bbox's center column, while column-band grey means reflect across the cup bbox's center row.
    - Mirror the full blue contour across that axis, erase the grey pixel, and keep all blue contour pixels blue.
    - Starting from the internal marker, flood-fill the enclosed region created by the original and reflected contours with the marker's color.

    Key insight:
    - The mirror axis is not drawn directly; it is encoded by the grey pixel's placement relative to the cup mouth, and reflection is necessary because only the mirrored contour turns the open cup into a closed fillable container.
    """
    out = _simulate_transform(np.asarray(grid, dtype=int))
    assert out is not None, "input does not produce a closed mirrored cup"
    return out


DIR_SPECS = {
    "up": {"shape": (5, 4), "marker": (3, 1)},
    "down": {"shape": (4, 4), "marker": (1, 2)},
    "left": {"shape": (4, 5), "marker": (2, 3)},
    "right": {"shape": (4, 4), "marker": (1, 1)},
}


def _build_cup(direction: str, color: int) -> np.ndarray:
    h, w = DIR_SPECS[direction]["shape"]
    sprite = np.zeros((h, w), dtype=int)
    sprite[0, :] = BLUE
    sprite[-1, :] = BLUE
    sprite[:, 0] = BLUE
    sprite[:, -1] = BLUE

    if direction == "up":
        sprite[0, 1:-1] = BG
    elif direction == "down":
        sprite[-1, 1:-1] = BG
    elif direction == "left":
        sprite[1:-1, 0] = BG
    else:
        sprite[1:-1, -1] = BG

    mr, mc = DIR_SPECS[direction]["marker"]
    sprite[mr, mc] = int(color)
    return sprite


def _grey_offset(direction: str, h: int, w: int) -> tuple[int, int]:
    if direction == "up":
        return -1, w // 2
    if direction == "down":
        return h, (w - 1) // 2
    if direction == "left":
        return h // 2, -1
    return (h - 1) // 2, w


def _randint_closed(lo: int, hi: int) -> int:
    assert lo <= hi
    return int(np.random.randint(lo, hi + 1))


def _place_cup(H: int, W: int, direction: str) -> tuple[int, int]:
    h, w = DIR_SPECS[direction]["shape"]

    if direction == "up":
        r0 = _randint_closed(H // 2, H - h - 1)
        c0 = _randint_closed(1, W // 2 - w)
    elif direction == "down":
        r0 = _randint_closed(1, H // 2 - h)
        c0 = _randint_closed(W // 2, W - w - 2)
    elif direction == "left":
        r0 = _randint_closed(H // 2, H - h - 2)
        c0 = _randint_closed(W // 2, W - w - 1)
    else:
        r0 = _randint_closed(1, H // 2 - h)
        c0 = _randint_closed(1, W // 2 - w)

    return r0, c0


def _validate_generated(grid: np.ndarray) -> None:
    info = _parse_input(grid)
    out = main(grid)
    assert not np.any(out == GREY), "grey control pixel must be erased"
    assert not np.array_equal(out, grid), "transformation must change the grid"

    fill_color = info["marker_color"]
    fill_count = int(np.count_nonzero(out == fill_color))
    assert fill_count >= 4, "marker color must expand beyond the original marker"

    wrong_axis = "horizontal" if info["axis_kind"] == "vertical" else "vertical"
    wrong_out = _simulate_transform(grid, axis_override=wrong_axis)
    assert wrong_out is None or not np.array_equal(wrong_out, out), "axis cue must matter"

    raw = np.asarray(grid, dtype=int).copy()
    raw[raw == GREY] = BG
    _, touches_boundary = _collect_component(raw, info["marker"])
    assert touches_boundary, "original cup must remain open before reflection"


def generate_input() -> np.ndarray:
    for _ in range(400):
        H = _randint_closed(12, 15)
        W = _randint_closed(12, 15)
        direction = str(np.random.choice(["up", "down", "left", "right"]))
        fill_color = int(np.random.choice(FILL_COLORS))

        grid = np.zeros((H, W), dtype=int)
        cup = _build_cup(direction, fill_color)
        h, w = cup.shape
        r0, c0 = _place_cup(H, W, direction)
        grid[r0 : r0 + h, c0 : c0 + w] = cup

        dr, dc = _grey_offset(direction, h, w)
        grid[r0 + dr, c0 + dc] = GREY

        try:
            _validate_generated(grid)
        except AssertionError:
            continue
        return grid

    raise RuntimeError("failed to generate a valid implicit-axis mirrored-cup example")
