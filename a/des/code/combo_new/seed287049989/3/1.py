from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


def _mode_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _uniform_color(values: np.ndarray) -> int | None:
    if values.size == 0:
        return None
    color = int(values.flat[0])
    if np.all(values == color):
        return color
    return None


def _collect_region(
    grid: np.ndarray,
    bg: int,
    wall: int,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
) -> dict[str, object] | None:
    if r1 - r0 < 3 or c1 - c0 < 3:
        return None
    if not (
        int(grid[r0, c0]) == wall
        and int(grid[r0, c1]) == wall
        and int(grid[r1, c0]) == wall
        and int(grid[r1, c1]) == wall
    ):
        return None

    top = _uniform_color(grid[r0, c0 + 1 : c1])
    bottom = _uniform_color(grid[r1, c0 + 1 : c1])
    left = _uniform_color(grid[r0 + 1 : r1, c0])
    right = _uniform_color(grid[r0 + 1 : r1, c1])
    if None in (top, bottom, left, right):
        return None

    side_colors = {
        "top": int(top),
        "bottom": int(bottom),
        "left": int(left),
        "right": int(right),
    }
    if any(color in {bg, wall} for color in side_colors.values()):
        return None
    if len(set(side_colors.values())) != 4:
        return None

    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    if interior.size == 0 or np.any(interior == wall):
        return None

    side_palette = set(side_colors.values())
    marker_hits = [
        (r0 + 1 + int(rr), c0 + 1 + int(cc))
        for rr, cc in np.argwhere(np.isin(interior, list(side_palette)))
    ]
    if len(marker_hits) != 1:
        return None

    marker_r, marker_c = marker_hits[0]
    marker_color = int(grid[marker_r, marker_c])
    for nr, nc in _neighbors4(marker_r, marker_c):
        if r0 < nr < r1 and c0 < nc < c1 and int(grid[nr, nc]) == marker_color:
            return None

    object_coords = [
        (r0 + 1 + int(rr), c0 + 1 + int(cc))
        for rr, cc in np.argwhere(interior != bg)
        if (r0 + 1 + int(rr), c0 + 1 + int(cc)) != (marker_r, marker_c)
    ]
    if not object_coords:
        return None
    if any(int(grid[r, c]) in side_palette for r, c in object_coords):
        return None

    door_side = next(side for side, color in side_colors.items() if color == marker_color)
    return {
        "bbox": (r0, c0, r1, c1),
        "door_side": door_side,
        "marker": (marker_r, marker_c),
        "object_coords": object_coords,
    }


def _find_regions(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    by_bbox: dict[tuple[int, int, int, int], dict[str, object]] = {}
    colors = [int(color) for color in np.unique(grid) if int(color) != bg]

    for wall in colors:
        coords = np.argwhere(grid == wall)
        if coords.shape[0] < 4:
            continue
        coord_set = {tuple(map(int, rc)) for rc in coords}
        rows = sorted({int(r) for r, _ in coord_set})
        cols = sorted({int(c) for _, c in coord_set})
        for i, r0 in enumerate(rows):
            for r1 in rows[i + 1 :]:
                if r1 - r0 < 3:
                    continue
                shared_cols = [c for c in cols if (r0, c) in coord_set and (r1, c) in coord_set]
                for j, c0 in enumerate(shared_cols):
                    for c1 in shared_cols[j + 1 :]:
                        region = _collect_region(grid, bg, wall, r0, c0, r1, c1)
                        if region is not None:
                            by_bbox[region["bbox"]] = region

    regions = sorted(by_bbox.values(), key=lambda region: tuple(region["bbox"]))  # type: ignore[arg-type]
    accepted: list[dict[str, object]] = []
    covered = np.zeros(grid.shape, dtype=bool)
    for region in regions:
        r0, c0, r1, c1 = region["bbox"]  # type: ignore[misc]
        mask = np.zeros(grid.shape, dtype=bool)
        mask[r0 : r1 + 1, c0 : c1 + 1] = True
        if np.any(mask & covered):
            continue
        covered |= mask
        accepted.append(region)
    return accepted


def main(grid: np.ndarray) -> np.ndarray:
    """
    Open the color-matched door on each closed box and eject the interior objects off the canvas.

    Input structure:
    - The background is dominant. Each enclosed region is a rectangular frame whose
      four corner pixels share one wall color while each side segment between corners
      is a single non-background side color.
    - Inside every frame there is exactly one isolated marker pixel whose color matches
      one of the four side segments, plus one or more other foreground objects whose
      colors are unrelated to the boundary colors.

    Rule:
    - For each rectangular region, use the marker color to identify the matching side segment.
    - Erase that matching side segment to open the door, keeping the wall-color corners and
      the other three sides unchanged.
    - Erase the marker itself, then translate every other interior foreground object outward
      through that door until it has completely left the grid; because the objects move fully
      off-canvas, no moved object pixels remain in the output.

    Key insight:
    - The marker is a key: its color selects the door side, and that selected side determines
      the outward launch direction for every other object in the same enclosure.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    bg = _mode_color(g)
    regions = _find_regions(g, bg)
    assert regions, "expected at least one keyed rectangular region"

    out = g.copy()
    for region in regions:
        r0, c0, r1, c1 = region["bbox"]  # type: ignore[misc]
        marker_r, marker_c = region["marker"]  # type: ignore[misc]
        out[marker_r, marker_c] = bg
        for r, c in region["object_coords"]:  # type: ignore[misc]
            out[r, c] = bg

        side = region["door_side"]
        if side == "top":
            out[r0, c0 + 1 : c1] = bg
        elif side == "bottom":
            out[r1, c0 + 1 : c1] = bg
        elif side == "left":
            out[r0 + 1 : r1, c0] = bg
        else:
            assert side == "right"
            out[r0 + 1 : r1, c1] = bg

    return out


_BASE_SHAPES = [
    np.array([[1, 1]], dtype=bool),
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 0], [1, 1], [0, 1]], dtype=bool),
]
_DOOR_SIDES = ("top", "bottom", "left", "right")
_SIDE_INDEX = {"top": 0, "bottom": 1, "left": 2, "right": 3}
_COLOR_POOL = tuple(range(10))
_MIN_REGIONS = 1
_MAX_REGIONS = 3
_GEN_RETRIES = 300
_PLACE_RETRIES = 200


def _random_shape() -> np.ndarray:
    shape = random.choice(_BASE_SHAPES).copy()
    shape = np.rot90(shape, random.randint(0, 3)).copy()
    if random.random() < 0.5:
        shape = np.fliplr(shape).copy()
    return shape


def _reserve_with_gap(mask: np.ndarray, coords: list[tuple[int, int]]) -> None:
    for r, c in coords:
        mask[r, c] = True
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1]:
                mask[nr, nc] = True


def _place_region_contents(
    grid: np.ndarray,
    expected: np.ndarray,
    bbox: tuple[int, int, int, int],
    bg: int,
    wall: int,
) -> bool:
    r0, c0, r1, c1 = bbox
    side_colors = random.sample([c for c in _COLOR_POOL if c not in {bg, wall}], 4)
    door_side = random.choice(_DOOR_SIDES)
    marker_color = side_colors[_SIDE_INDEX[door_side]]

    grid[r0, c0] = wall
    grid[r0, c1] = wall
    grid[r1, c0] = wall
    grid[r1, c1] = wall
    grid[r0, c0 + 1 : c1] = side_colors[0]
    grid[r1, c0 + 1 : c1] = side_colors[1]
    grid[r0 + 1 : r1, c0] = side_colors[2]
    grid[r0 + 1 : r1, c1] = side_colors[3]

    expected[r0 : r1 + 1, c0 : c1 + 1] = grid[r0 : r1 + 1, c0 : c1 + 1]
    if door_side == "top":
        expected[r0, c0 + 1 : c1] = bg
    elif door_side == "bottom":
        expected[r1, c0 + 1 : c1] = bg
    elif door_side == "left":
        expected[r0 + 1 : r1, c0] = bg
    else:
        expected[r0 + 1 : r1, c1] = bg

    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    blocked = np.zeros((inner_h, inner_w), dtype=bool)
    object_colors = [c for c in _COLOR_POOL if c not in {bg, wall, *side_colors}]
    object_count = random.randint(1, min(3, len(object_colors)))
    random.shuffle(object_colors)

    placed_any = False
    for color in object_colors[:object_count]:
        shape = _random_shape()
        sh, sw = shape.shape
        positions = []
        for rr in range(inner_h - sh + 1):
            for cc in range(inner_w - sw + 1):
                window = blocked[rr : rr + sh, cc : cc + sw]
                if np.any(window & shape):
                    continue
                positions.append((rr, cc))
        if not positions:
            return False
        rr, cc = random.choice(positions)
        coords: list[tuple[int, int]] = []
        for sr, sc in np.argwhere(shape):
            gr = r0 + 1 + rr + int(sr)
            gc = c0 + 1 + cc + int(sc)
            coords.append((gr, gc))
            grid[gr, gc] = color
            expected[gr, gc] = bg
        _reserve_with_gap(blocked, [(rr + int(sr), cc + int(sc)) for sr, sc in np.argwhere(shape)])
        placed_any = True

    if not placed_any:
        return False

    marker_positions = [
        (rr, cc)
        for rr in range(inner_h)
        for cc in range(inner_w)
        if not blocked[rr, cc]
    ]
    if not marker_positions:
        return False
    mr, mc = random.choice(marker_positions)
    grid[r0 + 1 + mr, c0 + 1 + mc] = marker_color
    expected[r0 + 1 + mr, c0 + 1 + mc] = bg
    return True


def _sample_layout() -> tuple[np.ndarray, np.ndarray] | None:
    n_regions = random.randint(_MIN_REGIONS, _MAX_REGIONS)
    min_dim = 14 + 4 * (n_regions - 1)
    height = random.randint(min_dim, 30)
    width = random.randint(min_dim, 30)

    bg, wall = random.sample(_COLOR_POOL, 2)
    grid = np.full((height, width), bg, dtype=int)
    expected = grid.copy()
    reserved = np.zeros((height, width), dtype=bool)

    for _ in range(n_regions):
        placed = False
        for _ in range(_PLACE_RETRIES):
            outer_h = random.randint(7, min(11, height - 2))
            outer_w = random.randint(7, min(11, width - 2))
            if outer_h >= height or outer_w >= width:
                continue
            r0 = random.randint(1, height - outer_h)
            c0 = random.randint(1, width - outer_w)
            r1 = r0 + outer_h - 1
            c1 = c0 + outer_w - 1

            rr0 = max(0, r0 - 2)
            cc0 = max(0, c0 - 2)
            rr1 = min(height - 1, r1 + 2)
            cc1 = min(width - 1, c1 + 2)
            if np.any(reserved[rr0 : rr1 + 1, cc0 : cc1 + 1]):
                continue

            trial_grid = grid.copy()
            trial_expected = expected.copy()
            if not _place_region_contents(trial_grid, trial_expected, (r0, c0, r1, c1), bg, wall):
                continue
            grid = trial_grid
            expected = trial_expected
            reserved[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
            placed = True
            break
        if not placed:
            return None

    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        sample = _sample_layout()
        if sample is None:
            continue
        inp, expected = sample
        try:
            actual = main(inp)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return inp
    raise RuntimeError("failed to generate a valid keyed-door ejection puzzle")
