import random
from typing import Iterable

import numpy as np


BACKGROUND = 0
MARKER = 5
OBJECT_COLORS = (1, 2, 3, 4)
DIRECTIONS = ("up", "right", "down", "left")
ROTATION_BY_DIRECTION = {
    "up": 0,
    "right": 3,  # np.rot90 uses counter-clockwise steps.
    "down": 2,
    "left": 1,
}
MAX_ATTEMPTS = 400


def main(grid: np.ndarray) -> np.ndarray:
    """Rotate the unique hollow rectangle according to its interior marker, then crop away everything else.

    Input structure:
    - The grid has black background plus several separated monochrome rectangular objects.
    - Exactly one object is a 1-pixel-thick hollow rectangular frame, and its hole contains exactly one grey marker pixel.
    - All other non-grey foreground objects are solid rectangles that do not overlap or touch the frame.

    Rule:
    - Find the unique non-grey connected component whose bbox border is fully filled by its color and whose interior contains background plus a single grey marker.
    - Compute the marker's dominant direction relative to the geometric center of the hole: up -> 0 degrees, right -> 90 degrees clockwise, down -> 180 degrees, left -> 270 degrees clockwise.
    - Remove the marker, rotate only the hollow frame by that amount around its own center, and return the rotated frame's tight bbox crop.

    Key insight:
    - The grey pixel is not output content; it is a direction code that can only be decoded after isolating the enclosing hollow rectangle and its hole geometry.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frame_info = _find_unique_frame(g)
    assert frame_info is not None, "expected exactly one hollow rectangular frame"

    direction = _direction_from_marker(
        frame_info["marker_local"],
        frame_info["hole_shape"],
    )
    k = ROTATION_BY_DIRECTION[direction]
    return np.rot90(frame_info["frame_patch"], k).copy()


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        sample = _generate_attempt()
        if sample is None:
            continue
        try:
            out = main(sample["grid"])
        except AssertionError:
            continue
        if not np.array_equal(out, sample["expected"]):
            continue
        return sample["grid"]
    raise RuntimeError("failed to generate a valid task 9 sample")


def _generate_attempt() -> dict | None:
    rng = random.Random()
    height = rng.randint(13, 17)
    width = rng.randint(13, 17)
    grid = np.zeros((height, width), dtype=int)

    object_count = rng.randint(3, 4)
    solid_count = object_count - 1
    colors = list(OBJECT_COLORS)
    rng.shuffle(colors)
    frame_color = colors[0]
    solid_colors = colors[1 : 1 + solid_count]
    direction = rng.choice(DIRECTIONS)

    frame_patch, expected = _make_frame_with_marker(rng, frame_color, direction)
    solid_patches = [_make_solid_rectangle(rng, color) for color in solid_colors]

    placed: list[dict] = []
    items = [("frame", frame_patch, direction, expected)] + [
        ("solid", patch, None, None) for patch in solid_patches
    ]
    items.sort(key=lambda item: item[1].shape[0] * item[1].shape[1], reverse=True)

    for kind, patch, item_direction, item_expected in items:
        bbox = _place_patch(rng, grid.shape, patch.shape, [entry["bbox"] for entry in placed], gap=1)
        if bbox is None:
            return None
        r0, c0, r1, c1 = bbox
        grid[r0 : r1 + 1, c0 : c1 + 1] = np.maximum(grid[r0 : r1 + 1, c0 : c1 + 1], patch)
        entry = {"kind": kind, "bbox": bbox}
        if kind == "frame":
            entry["direction"] = item_direction
            entry["expected"] = item_expected
        placed.append(entry)

    frame_entry = next(entry for entry in placed if entry["kind"] == "frame")
    if len(_frame_candidates(grid)) != 1:
        return None

    output = frame_entry["expected"]
    if max(grid.shape) > 30 or max(output.shape) > 30:
        return None
    return {"grid": grid, "expected": output}


def _make_frame_with_marker(rng: random.Random, color: int, direction: str) -> tuple[np.ndarray, np.ndarray]:
    hole_h = rng.randint(3, 6)
    hole_w = rng.randint(3, 6)
    if hole_h == hole_w and rng.random() < 0.75:
        if hole_w < 6 and rng.random() < 0.5:
            hole_w += 1
        elif hole_h < 6:
            hole_h += 1
        elif hole_w > 3:
            hole_w -= 1

    patch = np.zeros((hole_h + 2, hole_w + 2), dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color

    candidates = [
        (r, c)
        for r in range(hole_h)
        for c in range(hole_w)
        if (r, c) != ((hole_h - 1) / 2.0, (hole_w - 1) / 2.0)
        if _direction_from_marker((r, c), (hole_h, hole_w)) == direction
    ]
    if not candidates:
        raise AssertionError("expected direction-compatible marker positions")
    marker_r, marker_c = rng.choice(candidates)
    patch[marker_r + 1, marker_c + 1] = MARKER

    frame_only = np.where(patch == color, color, BACKGROUND)
    expected = np.rot90(frame_only, ROTATION_BY_DIRECTION[direction]).copy()
    return patch, expected


def _make_solid_rectangle(rng: random.Random, color: int) -> np.ndarray:
    if rng.random() < 0.35:
        if rng.random() < 0.5:
            h = rng.randint(2, 3)
            w = rng.randint(4, 6)
        else:
            h = rng.randint(4, 6)
            w = rng.randint(2, 3)
    else:
        h = rng.randint(2, 5)
        w = rng.randint(2, 5)
    return np.full((h, w), color, dtype=int)


def _place_patch(
    rng: random.Random,
    grid_shape: tuple[int, int],
    patch_shape: tuple[int, int],
    placed_bboxes: Iterable[tuple[int, int, int, int]],
    gap: int,
) -> tuple[int, int, int, int] | None:
    grid_h, grid_w = grid_shape
    patch_h, patch_w = patch_shape
    if patch_h > grid_h or patch_w > grid_w:
        return None

    for _ in range(200):
        r0 = rng.randint(0, grid_h - patch_h)
        c0 = rng.randint(0, grid_w - patch_w)
        bbox = (r0, c0, r0 + patch_h - 1, c0 + patch_w - 1)
        if all(_bbox_has_gap(bbox, other, gap) for other in placed_bboxes):
            return bbox
    return None


def _bbox_has_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return (
        ar1 + gap < br0
        or br1 + gap < ar0
        or ac1 + gap < bc0
        or bc1 + gap < ac0
    )


def _frame_candidates(grid: np.ndarray) -> list[dict]:
    candidates = []
    seen = np.zeros(grid.shape, dtype=bool)
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            value = int(grid[r, c])
            if value in (BACKGROUND, MARKER) or seen[r, c]:
                continue
            coords = _component_coords(grid, (r, c), value, seen)
            info = _frame_info(grid, coords, value)
            if info is not None:
                candidates.append(info)
    return candidates


def _find_unique_frame(grid: np.ndarray) -> dict | None:
    candidates = _frame_candidates(grid)
    return candidates[0] if len(candidates) == 1 else None


def _frame_info(grid: np.ndarray, coords: list[tuple[int, int]], color: int) -> dict | None:
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None

    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    h, w = patch.shape
    border_mask = np.zeros((h, w), dtype=bool)
    border_mask[0, :] = True
    border_mask[-1, :] = True
    border_mask[:, 0] = True
    border_mask[:, -1] = True

    if not np.all(patch[border_mask] == color):
        return None
    if len(coords) != int(border_mask.sum()):
        return None

    inside = patch[1:-1, 1:-1]
    if inside.size == 0:
        return None
    if np.any((inside != BACKGROUND) & (inside != MARKER)):
        return None

    marker_positions = np.argwhere(inside == MARKER)
    bg_count = int(np.sum(inside == BACKGROUND))
    if len(marker_positions) != 1 or bg_count == 0:
        return None

    marker_local = tuple(int(v) for v in marker_positions[0])
    center_r = (inside.shape[0] - 1) / 2.0
    center_c = (inside.shape[1] - 1) / 2.0
    if marker_local == (center_r, center_c):
        return None

    frame_patch = np.where(patch == color, color, BACKGROUND)
    return {
        "color": color,
        "bbox": (r0, c0, r1, c1),
        "frame_patch": frame_patch,
        "marker_local": marker_local,
        "hole_shape": inside.shape,
    }


def _direction_from_marker(marker_local: tuple[int, int], hole_shape: tuple[int, int]) -> str:
    row = float(marker_local[0])
    col = float(marker_local[1])
    center_row = (hole_shape[0] - 1) / 2.0
    center_col = (hole_shape[1] - 1) / 2.0
    row_delta = row - center_row
    col_delta = col - center_col
    assert row_delta != 0 or col_delta != 0, "marker must not be centered"

    if abs(row_delta) >= abs(col_delta):
        return "up" if row_delta < 0 else "down"
    return "left" if col_delta < 0 else "right"


def _component_coords(
    grid: np.ndarray,
    start: tuple[int, int],
    color: int,
    seen: np.ndarray,
) -> list[tuple[int, int]]:
    stack = [start]
    seen[start] = True
    coords: list[tuple[int, int]] = []

    while stack:
        r, c = stack.pop()
        coords.append((r, c))
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                if not seen[nr, nc] and int(grid[nr, nc]) == color:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
    return coords


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)
