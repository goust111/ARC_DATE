from __future__ import annotations

import importlib.util
import random
from pathlib import Path

import numpy as np


def _load_arc_utils():
    utils_path = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
    spec = importlib.util.spec_from_file_location("shared_arc_utils_fable_8_002_3", utils_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_ARC = _load_arc_utils()
Color = _ARC.Color
bbox = _ARC.bbox
colors_present = _ARC.colors_present
crop_bbox = _ARC.crop_bbox
find_components = _ARC.find_components
get_bg = _ARC.get_bg
paste = _ARC.paste
random_free_location = _ARC.random_free_location


BACKGROUND = Color.BLACK
GREY = Color.GREY
BODY_COLORS = [c for c in Color.NOT_BLACK if c != GREY]
MAX_GENERATION_ATTEMPTS = 600


def _ceil_div(n: int, d: int) -> int:
    return (int(n) + int(d) - 1) // int(d)


def _row_shift(row: int, anchor_row: int, k: int, direction: int, mode: str = "actual") -> int:
    if mode == "top":
        distance = row
        return int(direction) * _ceil_div(distance, k)

    distance = abs(int(row) - int(anchor_row))
    if mode == "actual":
        return int(direction) * _ceil_div(distance, k)
    if mode == "unit":
        return int(direction) * distance
    if mode == "floor":
        return int(direction) * (distance // int(k))
    if mode == "mirror":
        if distance == 0:
            return 0
        sign = -1 if row < anchor_row else 1
        return int(direction) * sign * _ceil_div(distance, k)
    raise ValueError(f"unknown mode: {mode}")


def _render_shifted_patch(
    body_mask: np.ndarray,
    color: int,
    anchor_row: int,
    k: int,
    direction: int,
    mode: str = "actual",
) -> tuple[np.ndarray, int]:
    h, w = body_mask.shape
    shifts = [_row_shift(r, anchor_row, k, direction, mode=mode) for r in range(h)]
    min_shift = min(shifts)
    max_shift = max(shifts)
    left_pad = max(0, -min_shift)
    right_pad = max(0, max_shift)

    out = np.full((h, w + left_pad + right_pad), BACKGROUND, dtype=int)
    for r, shift in enumerate(shifts):
        cols = np.flatnonzero(body_mask[r])
        if cols.size == 0:
            continue
        out[r, cols + left_pad + shift] = int(color)
    return out, left_pad


def _build_body_patch(intervals: list[tuple[int, int]], color: int) -> np.ndarray:
    min_left = min(left for left, _ in intervals)
    max_right = max(right for _, right in intervals)
    h = len(intervals)
    w = max_right - min_left + 1
    patch = np.full((h, w), BACKGROUND, dtype=int)
    for r, (left, right) in enumerate(intervals):
        patch[r, left - min_left : right - min_left + 1] = int(color)
    return patch


def _make_object_spec(
    body_patch: np.ndarray,
    color: int,
    anchor_row: int,
    k: int,
    direction: int,
) -> dict:
    body_mask = body_patch != BACKGROUND
    input_core = body_patch.copy()
    anchor_cols = np.flatnonzero(body_mask[anchor_row])
    assert anchor_cols.size >= k + 1
    if direction < 0:
        input_core[anchor_row, anchor_cols[:k]] = GREY
    else:
        input_core[anchor_row, anchor_cols[-k:]] = GREY

    expected_patch, input_left = _render_shifted_patch(body_mask, color, anchor_row, k, direction, mode="actual")
    input_patch = np.full_like(expected_patch, BACKGROUND)
    paste(input_patch, input_core, 0, input_left, transparent=BACKGROUND)

    reserve_patch = ((input_patch != BACKGROUND) | (expected_patch != BACKGROUND)).astype(int)
    return {
        "body_mask": body_mask,
        "color": int(color),
        "anchor_row": int(anchor_row),
        "k": int(k),
        "direction": int(direction),
        "input_patch": input_patch,
        "expected_patch": expected_patch,
        "input_left": int(input_left),
        "reserve_patch": reserve_patch,
    }


def _sample_intervals(height: int) -> list[tuple[int, int]]:
    start_width = random.randint(4, 7)
    intervals = [(0, start_width - 1)]
    for _ in range(1, height):
        prev_left, prev_right = intervals[-1]
        options = []
        for d_left in (-1, 0, 1):
            for d_right in (-1, 0, 1):
                left = prev_left + d_left
                right = prev_right + d_right
                width = right - left + 1
                if not (4 <= width <= 8):
                    continue
                if left > prev_right or right < prev_left:
                    continue
                options.append((left, right))
        assert options
        intervals.append(random.choice(options))

    if random.random() < 0.35:
        left = min(l for l, _ in intervals)
        right = max(r for _, r in intervals)
        intervals = [(left, right)] * height

    min_left = min(left for left, _ in intervals)
    return [(left - min_left, right - min_left) for left, right in intervals]


def _sample_object(color: int, direction: int, k_hint: int | None, require_floor_gap: bool) -> dict | None:
    for _ in range(200):
        height = random.randint(5, 8)
        intervals = _sample_intervals(height)
        body_patch = _build_body_patch(intervals, color)
        widths = np.sum(body_patch != BACKGROUND, axis=1)
        candidate_rows = [r for r in range(1, height - 1) if int(widths[r]) >= 3]
        if not candidate_rows:
            continue

        anchor_row = random.choice(candidate_rows)
        max_k = min(4, int(widths[anchor_row]) - 1)
        if max_k < 1:
            continue

        if k_hint is not None:
            if k_hint > max_k:
                continue
            k = int(k_hint)
        else:
            k = random.randint(1, max_k)

        if require_floor_gap:
            distances = [abs(r - anchor_row) for r in range(height) if r != anchor_row]
            if not any(d % k != 0 for d in distances):
                continue

        spec = _make_object_spec(body_patch, color, anchor_row, k, direction)
        if np.array_equal(spec["input_patch"], spec["expected_patch"]):
            continue
        return spec
    return None


def _compose_grid(shape: tuple[int, int], placements: list[dict], key: str) -> np.ndarray:
    grid = np.full(shape, BACKGROUND, dtype=int)
    for placement in placements:
        paste(grid, placement[key], placement["r0"], placement["c0"], transparent=BACKGROUND)
    return grid


def _compose_wrong_output(shape: tuple[int, int], placements: list[dict], mode: str) -> np.ndarray | None:
    grid = np.full(shape, BACKGROUND, dtype=int)
    occupied = np.zeros(shape, dtype=bool)

    for placement in placements:
        spec = placement["spec"]
        patch, left_pad = _render_shifted_patch(
            spec["body_mask"],
            spec["color"],
            spec["anchor_row"],
            spec["k"],
            spec["direction"],
            mode=mode,
        )
        r0 = placement["r0"]
        c0 = placement["c0"] + spec["input_left"] - left_pad
        h, w = patch.shape
        if r0 < 0 or c0 < 0 or r0 + h > shape[0] or c0 + w > shape[1]:
            return None
        mask = patch != BACKGROUND
        if np.any(occupied[r0 : r0 + h, c0 : c0 + w] & mask):
            return None
        view = grid[r0 : r0 + h, c0 : c0 + w]
        view[mask] = patch[mask]
        occupied[r0 : r0 + h, c0 : c0 + w] |= mask
    return grid


def _sample_scene() -> tuple[np.ndarray, np.ndarray] | None:
    rows = random.randint(20, 30)
    cols = random.randint(20, 30)
    shape = (rows, cols)
    object_count = random.choice((3, 4))

    colors = random.sample(BODY_COLORS, object_count)
    directions = [-1, 1] + [random.choice((-1, 1)) for _ in range(object_count - 2)]
    random.shuffle(directions)

    k_hints = [1, random.choice((2, 3))]
    while len(k_hints) < object_count:
        k_hints.append(random.choice((1, 2, 3, 4)))
    random.shuffle(k_hints)

    specs = []
    floor_gap_used = False
    for index in range(object_count):
        require_floor_gap = not floor_gap_used and index == object_count - 1
        if not floor_gap_used and k_hints[index] > 1:
            require_floor_gap = True
        spec = _sample_object(colors[index], directions[index], k_hints[index], require_floor_gap)
        if spec is None:
            return None
        floor_gap_used = floor_gap_used or (require_floor_gap and spec["k"] > 1)
        specs.append(spec)

    if not any(spec["direction"] < 0 for spec in specs):
        return None
    if not any(spec["direction"] > 0 for spec in specs):
        return None
    if not any(spec["k"] == 1 for spec in specs):
        return None
    if not any(spec["k"] > 1 for spec in specs):
        return None
    if not any(
        spec["k"] > 1
        and any(abs(r - spec["anchor_row"]) % spec["k"] != 0 for r in range(spec["body_mask"].shape[0]) if r != spec["anchor_row"])
        for spec in specs
    ):
        return None

    occupancy = np.zeros(shape, dtype=int)
    placements = []
    for spec in random.sample(specs, len(specs)):
        try:
            r0, c0 = random_free_location(
                occupancy,
                spec["reserve_patch"],
                transparent=BACKGROUND,
                border=1,
                padding=1,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        paste(occupancy, spec["reserve_patch"], r0, c0, transparent=BACKGROUND)
        placements.append({"spec": spec, "r0": int(r0), "c0": int(c0), "input_patch": spec["input_patch"], "expected_patch": spec["expected_patch"]})

    inp = _compose_grid(shape, placements, "input_patch")
    expected = _compose_grid(shape, placements, "expected_patch")

    if np.array_equal(inp, expected):
        return None
    if len(find_components(inp, bg=BACKGROUND, connectivity=4, monochromatic=False, return_format="objects")) != object_count:
        return None
    if not np.array_equal(main(inp), expected):
        return None

    for wrong_mode in ("unit", "floor", "mirror", "top"):
        wrong = _compose_wrong_output(shape, placements, wrong_mode)
        if wrong is not None and np.array_equal(wrong, expected):
            return None
    return inp, expected


def main(grid: np.ndarray) -> np.ndarray:
    """Use each embedded gray ruler to shear its host solid block into a same-direction staircase.

    Input structure:
    - The grid has a dominant background plus several pairwise separated multicolor objects.
    - Each object is a single-color solid block except for one embedded horizontal gray segment in one row.
    - The gray segment lies fully inside the object row and touches exactly the left or right edge of that row.

    Rule:
    - Process each object independently, keeping the gray-segment row fixed as the anchor row.
    - Let k be the gray-segment length; a row at vertical distance d from the anchor shifts by ceil(d / k) cells.
    - If the gray segment is attached to the left edge, shift every non-anchor row left; if attached to the right edge, shift every non-anchor row right.
    - Recolor the gray pixels back to the object's body color and leave the background elsewhere unchanged.

    Key insight:
    - The gray segment is not a separate object: its row, length, and attachment side jointly encode anchor phase, staircase step height, and shear direction.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = np.full_like(g, bg)

    objects = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="objects")
    for obj in objects:
        mask = obj != bg
        box = bbox(mask)
        assert box is not None
        r0, c0, _, _ = box
        patch = crop_bbox(obj, box)
        patch_mask = patch != bg

        palette = colors_present(patch, exclude={bg})
        assert GREY in palette
        body_colors = [color for color in palette if color != GREY]
        assert len(body_colors) == 1
        body_color = int(body_colors[0])

        grey_coords = np.argwhere(patch == GREY)
        assert grey_coords.shape[0] >= 1
        grey_rows = np.unique(grey_coords[:, 0])
        assert grey_rows.size == 1
        anchor_row = int(grey_rows[0])
        grey_cols = np.sort(grey_coords[:, 1])
        assert np.array_equal(grey_cols, np.arange(int(grey_cols[0]), int(grey_cols[-1]) + 1))
        k = int(grey_cols.size)

        anchor_cols = np.flatnonzero(patch_mask[anchor_row])
        assert anchor_cols.size > k
        if int(grey_cols[0]) == int(anchor_cols[0]):
            direction = -1
        else:
            assert int(grey_cols[-1]) == int(anchor_cols[-1])
            direction = 1

        shifted_patch, left_pad = _render_shifted_patch(
            patch_mask,
            body_color,
            anchor_row,
            k,
            direction,
            mode="actual",
        )
        paste(out, shifted_patch, r0, c0 - left_pad, transparent=bg)

    return out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        scene = _sample_scene()
        if scene is not None:
            return scene[0]
    raise RuntimeError("failed to generate a valid gray-ruler shear input")
