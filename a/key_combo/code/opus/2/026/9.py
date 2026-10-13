from __future__ import annotations

import itertools
import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_ARC_UTILS_DIR))

from arc_utils import bbox, crop_bbox, find_components


BG = 0
PALETTE = (1, 2, 3, 4)
ORIENTATION_ORDER = ((0, 0), (0, 1), (1, 0), (1, 1))
MISSING_TO_QUADRANT = {
    (0, 0): (2, 2),  # missing upper-left -> place in lower-right quadrant
    (0, 1): (2, 0),  # missing upper-right -> place in lower-left quadrant
    (1, 0): (0, 2),  # missing lower-left -> place in upper-right quadrant
    (1, 1): (0, 0),  # missing lower-right -> place in upper-left quadrant
}

GRID_H_RANGE = (10, 16)
GRID_W_RANGE = (10, 16)
PLACEMENT_MARGIN = 1
PLACEMENT_GAP = 1
GEN_ATTEMPTS = 400

_ASSIGNMENTS_BY_COLOR_COUNT = {
    color_count: [
        assignment
        for assignment in itertools.product(PALETTE, repeat=4)
        if len(set(assignment)) == color_count
    ]
    for color_count in (2, 3, 4)
}
_GEN_CALL_COUNT = 0


def _sprite_patch(color: int, missing_corner: tuple[int, int]) -> np.ndarray:
    patch = np.full((2, 2), int(color), dtype=int)
    patch[missing_corner] = BG
    return patch


def _extract_sprites(grid: np.ndarray) -> list[dict]:
    components = find_components(
        grid,
        bg=BG,
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    assert len(components) == 4, "expected exactly four monochromatic sprites"

    sprites = []
    seen_orientations = set()
    for comp in components:
        color = int(comp["color"])
        assert color in PALETTE, "sprite colors must come from the visible puzzle palette"
        assert int(comp["area"]) == 3, "each sprite must have exactly three cells"

        box = tuple(map(int, comp["bbox"]))
        assert (box[2] - box[0] + 1, box[3] - box[1] + 1) == (2, 2), "each sprite must fit a 2x2 bbox"

        patch = crop_bbox(grid, box)
        assert bbox(patch != BG) == (0, 0, 1, 1), "the occupied cells must span the full 2x2 box"

        missing_cells = np.argwhere(patch == BG)
        assert missing_cells.shape == (1, 2), "each 2x2 sprite must miss exactly one corner"
        missing_corner = tuple(map(int, missing_cells[0]))
        assert missing_corner in MISSING_TO_QUADRANT, "the missing cell must be a corner"

        seen_orientations.add(missing_corner)
        sprites.append(
            {
                "color": color,
                "bbox": box,
                "patch": patch,
                "missing_corner": missing_corner,
            }
        )

    assert seen_orientations == set(ORIENTATION_ORDER), "the four L orientations must all appear exactly once"
    return sprites


def _assemble_frame(sprites: list[dict]) -> np.ndarray:
    frame = np.zeros((4, 4), dtype=int)
    for sprite in sprites:
        r0, c0 = MISSING_TO_QUADRANT[sprite["missing_corner"]]
        frame[r0 : r0 + 2, c0 : c0 + 2] = sprite["patch"]

    assert np.count_nonzero(frame[1:3, 1:3]) == 0, "packing must leave a 2x2 central hole"
    return frame


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pack four oriented L sprites into a 4x4 frame, then tile that frame by its packed color count.

    Input structure:
    - The grid has black background and exactly four 4-connected foreground sprites.
    - Every sprite is monochrome, occupies a full 2x2 bounding box, and has exactly one missing corner.
    - The four missing-corner orientations are all different, while colors come from the visible set {1,2,3,4}.

    Rule:
    - Crop each sprite to its 2x2 bounding box and identify which corner is missing.
    - Place that 2x2 crop into the opposite 2x2 quadrant of a 4x4 frame so the four missing corners form the central 2x2 hole.
    - Count the number N of distinct non-black colors in the packed 4x4 frame, then tile the full frame N times horizontally and N times vertically.

    Key insight:
    - The output size is not an explicit input parameter: you must first rebuild the packed 4x4 motif from sprite orientation, and only then can its emergent color set determine the NxN tiling factor.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sprites = _extract_sprites(g)
    frame = _assemble_frame(sprites)
    color_count = len({int(value) for value in frame.reshape(-1) if int(value) != BG})
    assert 2 <= color_count <= 4, "the packed frame must contain between two and four colors"
    return np.tile(frame, (color_count, color_count))


def _boxes_are_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    return (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _choose_positions(height: int, width: int) -> list[tuple[int, int]] | None:
    candidates = [
        (r0, c0)
        for r0 in range(PLACEMENT_MARGIN, height - 2 - PLACEMENT_MARGIN + 1)
        for c0 in range(PLACEMENT_MARGIN, width - 2 - PLACEMENT_MARGIN + 1)
    ]
    random.shuffle(candidates)

    chosen_boxes: list[tuple[int, int, int, int]] = []
    for r0, c0 in candidates:
        box = (r0, c0, r0 + 1, c0 + 1)
        if all(_boxes_are_separated(box, other, PLACEMENT_GAP) for other in chosen_boxes):
            chosen_boxes.append(box)
            if len(chosen_boxes) == 4:
                return [(box[0], box[1]) for box in chosen_boxes]
    return None


def _assignment_color_count(assignment: tuple[int, int, int, int]) -> int:
    return len(set(map(int, assignment)))


def _next_preferred_color_count() -> int:
    global _GEN_CALL_COUNT
    preferred = (2, 3, 4)[_GEN_CALL_COUNT % 3]
    _GEN_CALL_COUNT += 1
    return preferred


def _choose_assignment(preferred_count: int) -> tuple[int, int, int, int]:
    for color_count in (preferred_count, 2, 3, 4):
        if color_count not in (2, 3, 4):
            continue
        options = _ASSIGNMENTS_BY_COLOR_COUNT[color_count]
        if options:
            return random.choice(options)
    raise RuntimeError("exhausted all unique color assignments for this puzzle")


def _render_input(assignment: tuple[int, int, int, int]) -> np.ndarray | None:
    for _ in range(60):
        height = random.randint(*GRID_H_RANGE)
        width = random.randint(*GRID_W_RANGE)
        positions = _choose_positions(height, width)
        if positions is None:
            continue

        grid = np.zeros((height, width), dtype=int)
        random.shuffle(positions)
        for (missing_corner, color), (r0, c0) in zip(zip(ORIENTATION_ORDER, assignment), positions):
            grid[r0 : r0 + 2, c0 : c0 + 2] = _sprite_patch(color, missing_corner)
        return grid
    return None


def _output_signature(grid: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(map(int, grid.shape)), grid.tobytes()


def _validate_case(grid: np.ndarray, assignment: tuple[int, int, int, int]) -> bool:
    try:
        sprites = _extract_sprites(grid)
        parsed_assignment = {
            tuple(map(int, sprite["missing_corner"])): int(sprite["color"])
            for sprite in sprites
        }
        expected_assignment = {
            missing_corner: int(color)
            for missing_corner, color in zip(ORIENTATION_ORDER, assignment)
        }
        if parsed_assignment != expected_assignment:
            return False

        frame = _assemble_frame(sprites)
        color_count = _assignment_color_count(assignment)
        if len({int(value) for value in frame.reshape(-1) if int(value) != BG}) != color_count:
            return False

        out = main(grid)
        expected = np.tile(frame, (color_count, color_count))
        if out.shape != (4 * color_count, 4 * color_count):
            return False
        if not np.array_equal(out, expected):
            return False

        for r0 in range(0, out.shape[0], 4):
            for c0 in range(0, out.shape[1], 4):
                if not np.array_equal(out[r0 : r0 + 4, c0 : c0 + 4], frame):
                    return False
    except AssertionError:
        return False
    return True


def generate_input() -> np.ndarray:
    preferred_count = _next_preferred_color_count()
    for _ in range(GEN_ATTEMPTS):
        assignment = _choose_assignment(preferred_count)
        grid = _render_input(assignment)
        if grid is None:
            continue
        if not _validate_case(grid, assignment):
            continue

        return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
