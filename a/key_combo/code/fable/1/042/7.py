from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, find_components, in_bounds, neighbors8, paste, random_free_location


BACKGROUND = Color.BLACK
ODD_OUTLINE = Color.ORANGE
EVEN_OUTLINE = Color.BLUE
INPUT_COLORS = (
    Color.RED,
    Color.GREEN,
    Color.YELLOW,
    Color.GREY,
    Color.PINK,
    Color.TEAL,
    Color.MAROON,
)
GEN_RETRY_LIMIT = 400

# Canonical group templates. Coordinates are center pixels before random dihedral transforms.
GROUP_TEMPLATES = (
    ((0, 0), (0, 3)),
    ((0, 0), (4, 0)),
    ((0, 0), (3, 3)),
    ((0, 0), (1, 4)),
    ((0, 0), (0, 3), (0, 7)),
    ((0, 0), (4, 4), (7, 7)),
    ((0, 0), (3, 0), (7, 0)),
    ((0, 0), (0, 4), (3, 7)),
)


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1])))


def _nearest_unique_distances(coords: list[tuple[int, int]]) -> list[int]:
    distances: list[int] = []
    for i, src in enumerate(coords):
        current = [_chebyshev(src, dst) for j, dst in enumerate(coords) if j != i]
        assert current, "each color must appear at least twice"
        best = min(current)
        assert current.count(best) == 1, "nearest same-color pixel must be unique"
        distances.append(int(best))
    return distances


def _extract_scene(grid: np.ndarray) -> dict[int, list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    objects = find_components(g, bg=BACKGROUND, connectivity=8, monochromatic=True)
    positions_by_color: dict[int, list[tuple[int, int]]] = {}
    for obj in objects:
        assert int(obj["area"]) == 1, "every foreground object must be an isolated single pixel"
        r, c = map(int, obj["coords"][0])
        assert 0 < r < g.shape[0] - 1 and 0 < c < g.shape[1] - 1, "each pixel needs room for a 3x3 frame"
        for nr, nc in neighbors8(r, c):
            assert in_bounds(g.shape, nr, nc), "isolation check expects an interior pixel"
            assert int(g[nr, nc]) == BACKGROUND, "each pixel must have an empty 8-neighborhood"
        positions_by_color.setdefault(int(obj["color"]), []).append((r, c))

    assert positions_by_color, "expected at least one colored pixel"
    for coords in positions_by_color.values():
        assert len(coords) >= 2, "each color must appear at least twice"
        _nearest_unique_distances(coords)
    return positions_by_color


def _render_outlines(grid: np.ndarray, positions_by_color: dict[int, list[tuple[int, int]]]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    painted = np.zeros_like(out, dtype=bool)

    for coords in positions_by_color.values():
        nearest = _nearest_unique_distances(coords)
        for (r, c), dist in zip(coords, nearest):
            outline_color = ODD_OUTLINE if dist % 2 == 1 else EVEN_OUTLINE
            for rr in range(r - 1, r + 2):
                for cc in range(c - 1, c + 2):
                    if rr == r and cc == c:
                        continue
                    assert in_bounds(out.shape, rr, cc), "outline must stay inside the grid"
                    assert not painted[rr, cc], "3x3 outlines must not overlap"
                    painted[rr, cc] = True
                    out[rr, cc] = outline_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw a 3x3 hollow frame around each isolated pixel, with frame color set by nearest same-color distance parity.

    Input structure:
    - The background is black.
    - Foreground objects are isolated single pixels, possibly using several colors.
    - Every color appears at least twice, and each pixel has an empty one-cell border around it so a 3x3 frame fits.

    Rule:
    - For each pixel, consider only pixels of the same color and find the uniquely nearest one by Chebyshev distance.
    - If that nearest distance is odd, draw an orange 3x3 hollow square around the pixel; if it is even, draw a blue one.
    - Keep the center pixel unchanged and leave all other background cells black.

    Key insight:
    - The input color only decides which pixels belong to the same comparison set; the visible output change is controlled by the parity of the hidden nearest-neighbor Chebyshev distance.
    """
    g = np.asarray(grid, dtype=int)
    positions_by_color = _extract_scene(g)
    return _render_outlines(g, positions_by_color)


def _transform_coords(coords: tuple[tuple[int, int], ...]) -> list[tuple[int, int]]:
    arr = np.array(coords, dtype=int)
    r = arr[:, 0]
    c = arr[:, 1]
    variant = random.randint(0, 7)
    if variant == 0:
        transformed = np.stack([r, c], axis=1)
    elif variant == 1:
        transformed = np.stack([r, -c], axis=1)
    elif variant == 2:
        transformed = np.stack([-r, c], axis=1)
    elif variant == 3:
        transformed = np.stack([-r, -c], axis=1)
    elif variant == 4:
        transformed = np.stack([c, r], axis=1)
    elif variant == 5:
        transformed = np.stack([c, -r], axis=1)
    elif variant == 6:
        transformed = np.stack([-c, r], axis=1)
    else:
        transformed = np.stack([-c, -r], axis=1)
    transformed -= transformed.min(axis=0)
    return [tuple(map(int, item)) for item in transformed]


def _group_patch(coords: list[tuple[int, int]]) -> tuple[np.ndarray, list[tuple[int, int]]]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    patch = np.zeros((max(rows) + 3, max(cols) + 3), dtype=int)
    shifted = [(r + 1, c + 1) for r, c in coords]
    for r, c in shifted:
        patch[r, c] = 1
    return patch, shifted


def _group_spec() -> dict:
    raw = random.choice(GROUP_TEMPLATES)
    coords = _transform_coords(raw)
    nearest = _nearest_unique_distances(coords)
    patch, shifted = _group_patch(coords)
    return {
        "coords": coords,
        "nearest": nearest,
        "patch": patch,
        "shifted": shifted,
        "has_odd": any(dist % 2 == 1 for dist in nearest),
        "has_even": any(dist % 2 == 0 for dist in nearest),
        "is_triplet": len(coords) == 3,
    }


def _choose_groups() -> list[dict]:
    for _ in range(GEN_RETRY_LIMIT):
        count = random.choice((2, 2, 3, 3, 4))
        require_triplet = random.random() < 0.65
        groups = [_group_spec() for _ in range(count)]
        has_odd = any(group["has_odd"] for group in groups)
        has_even = any(group["has_even"] for group in groups)
        has_triplet = any(group["is_triplet"] for group in groups)
        if has_odd and has_even and (has_triplet or not require_triplet):
            return groups
    raise RuntimeError("failed to choose group templates with both parity branches")


def _place_groups(shape: tuple[int, int], groups: list[dict]) -> list[list[tuple[int, int]]] | None:
    occupancy = np.zeros(shape, dtype=int)
    placed: list[list[tuple[int, int]]] = []

    for group in sorted(groups, key=lambda item: int(item["patch"].shape[0] * item["patch"].shape[1]), reverse=True):
        try:
            top, left = random_free_location(
                occupancy,
                group["patch"],
                transparent=0,
                padding=2,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        paste(occupancy, group["patch"], int(top), int(left), transparent=0)
        coords = [(int(top) + r, int(left) + c) for r, c in group["shifted"]]
        placed.append(coords)
    return placed


def _scene_has_both_parities(positions_by_color: dict[int, list[tuple[int, int]]]) -> bool:
    has_odd = False
    has_even = False
    for coords in positions_by_color.values():
        for dist in _nearest_unique_distances(coords):
            has_odd |= dist % 2 == 1
            has_even |= dist % 2 == 0
    return has_odd and has_even


def _validated_scene(grid: np.ndarray) -> bool:
    try:
        positions_by_color = _extract_scene(grid)
    except AssertionError:
        return False

    used_outline: set[tuple[int, int]] = set()
    for coords in positions_by_color.values():
        _nearest_unique_distances(coords)
        for r, c in coords:
            for rr in range(r - 1, r + 2):
                for cc in range(c - 1, c + 2):
                    if (rr, cc) == (r, c):
                        continue
                    if int(grid[rr, cc]) != BACKGROUND:
                        return False
                    if (rr, cc) in used_outline:
                        return False
                    used_outline.add((rr, cc))
    return _scene_has_both_parities(positions_by_color)


def _attempt_input() -> np.ndarray | None:
    groups = _choose_groups()
    largest_h = max(int(group["patch"].shape[0]) for group in groups)
    largest_w = max(int(group["patch"].shape[1]) for group in groups)
    total_h = sum(int(group["patch"].shape[0]) for group in groups)
    total_w = sum(int(group["patch"].shape[1]) for group in groups)

    min_h = max(12, largest_h, min(24, total_h // 2 + 2))
    min_w = max(12, largest_w, min(24, total_w // 2 + 2))
    if min_h > 30 or min_w > 30:
        return None

    height = random.randint(min_h, 30)
    width = random.randint(min_w, 30)
    placed = _place_groups((height, width), groups)
    if placed is None:
        return None

    colors = random.sample(INPUT_COLORS, k=len(placed))
    grid = np.zeros((height, width), dtype=int)
    for color, coords in zip(colors, placed):
        for r, c in coords:
            grid[r, c] = int(color)

    if not _validated_scene(grid):
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    positions_by_color = _extract_scene(grid)
    odd_seen = any(dist % 2 == 1 for coords in positions_by_color.values() for dist in _nearest_unique_distances(coords))
    even_seen = any(dist % 2 == 0 for coords in positions_by_color.values() for dist in _nearest_unique_distances(coords))
    if not odd_seen or not even_seen:
        return None
    if not np.any(out == ODD_OUTLINE) or not np.any(out == EVEN_OUTLINE):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _attempt_input()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
