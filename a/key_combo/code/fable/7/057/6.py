from __future__ import annotations

import importlib.util
import random
from pathlib import Path

import numpy as np


_ARC_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
_ARC_UTILS_SPEC = importlib.util.spec_from_file_location("_arc_utils_shared", _ARC_UTILS_PATH)
assert _ARC_UTILS_SPEC is not None and _ARC_UTILS_SPEC.loader is not None
_ARC_UTILS = importlib.util.module_from_spec(_ARC_UTILS_SPEC)
_ARC_UTILS_SPEC.loader.exec_module(_ARC_UTILS)

Color = _ARC_UTILS.Color
find_components = _ARC_UTILS.find_components
get_bg = _ARC_UTILS.get_bg
bbox_w = _ARC_UTILS.bbox_w


BACKGROUND = Color.BLACK
WALL_COLOR = Color.GREY
OBJECT_COLORS = [Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON]

MIN_HEIGHT = 16
MAX_HEIGHT = 30
MIN_WIDTH = 20
MAX_WIDTH = 30
ROW_GAP = 1
MAX_GENERATION_ATTEMPTS = 500

MANDATORY_PLANS = (
    ("left", True, 2),
    ("right", True, 1),
    ("left", False, 1),
    ("right", False, 2),
)


def _wall_columns(grid: np.ndarray) -> tuple[int, int]:
    cols = np.flatnonzero(np.any(grid == WALL_COLOR, axis=0))
    assert len(cols) == 2, "expected exactly two wall columns"
    left, right = int(cols[0]), int(cols[1])
    assert np.all(grid[:, left] == WALL_COLOR) and np.all(grid[:, right] == WALL_COLOR), "walls must be solid vertical lines"
    wall_coords = np.argwhere(grid == WALL_COLOR)
    assert np.all(np.isin(wall_coords[:, 1], [left, right])), "grey pixels must belong to the two walls"
    return left, right


def _crop_mask(coords: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    shifted = coords - np.array([r0, c0], dtype=int)
    mask[shifted[:, 0], shifted[:, 1]] = True
    return mask


def _parse_objects(
    grid: np.ndarray,
    bg: int,
    left_wall: int,
    right_wall: int,
) -> list[dict]:
    objects: list[dict] = []
    components = find_components(grid, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    for component in components:
        color = int(component["color"])
        if color == WALL_COLOR:
            continue

        coords = component["coords"]
        bbox = tuple(map(int, component["bbox"]))
        r0, c0, r1, c1 = bbox
        mask = _crop_mask(coords, bbox)
        missing = np.argwhere(~mask)

        assert missing.shape == (1, 2), "each object must be a solid rectangle with one notch"
        notch_r, notch_c = map(int, missing[0])
        assert 0 < notch_r < mask.shape[0] - 1, "notch must be away from the corners"
        assert notch_c in {0, mask.shape[1] - 1}, "notch must be on the left or right border"
        if notch_c == 0:
            assert np.all(mask[np.arange(mask.shape[0]) != notch_r, 0]), "left border must be solid except at the notch"
        else:
            assert np.all(mask[np.arange(mask.shape[0]) != notch_r, -1]), "right border must be solid except at the notch"
        assert np.count_nonzero(mask) == mask.size - 1, "object interior must otherwise be solid"

        width = int(bbox_w(bbox))
        gap_left = int(c0 - left_wall - 1)
        gap_right = int(right_wall - c1 - 1)
        assert gap_left >= 0 and gap_right >= 0, "objects must stay between the walls"

        objects.append(
            {
                "color": color,
                "coords": coords.astype(int),
                "bbox": bbox,
                "width": width,
                "height": mask.shape[0],
                "notch_side": "left" if notch_c == 0 else "right",
                "gap_left": gap_left,
                "gap_right": gap_right,
            }
        )

    objects.sort(key=lambda obj: (obj["bbox"][0], obj["bbox"][1]))
    assert objects, "expected at least one movable object"
    for prev, cur in zip(objects, objects[1:]):
        assert prev["bbox"][2] < cur["bbox"][0], "objects must occupy distinct row bands"
    return objects


def _analyze_scene(grid: np.ndarray) -> tuple[np.ndarray, int, int, int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = int(get_bg(g))
    left_wall, right_wall = _wall_columns(g)
    objects = _parse_objects(g, bg, left_wall, right_wall)
    return g, bg, left_wall, right_wall, objects


def _final_shift(obj: dict) -> int:
    gap = int(obj["gap_left"] if obj["notch_side"] == "left" else obj["gap_right"])
    jumps = gap // int(obj["width"])
    return jumps * int(obj["width"])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Jump each notched rectangle horizontally toward its indicated wall in steps of its own width.

    Input structure:
    - The background contains exactly two solid grey vertical wall columns, one on the left side and one on the right side.
    - Between the walls sit several separated monochromatic rectangles, each missing exactly one non-corner cell on its left or right bbox border.
    - Every movable object occupies its own row band, so horizontal jumps never interfere with one another.

    Rule:
    - Read the notch side of each object: a left-border notch points to the left wall and a right-border notch points to the right wall.
    - Let the horizontal jump length equal that object's own bounding-box width, and repeat that jump toward the chosen wall while the next jump would still avoid the wall.
    - Keep both walls fixed and preserve every object's color, shape, row span, and notch; only its horizontal position changes.

    Key insight:
    - Objects do not slide continuously to the wall: they advance in width-sized quanta, so the final gap is exactly the original wall gap modulo the object's own width.
    """
    g, bg, _left_wall, _right_wall, objects = _analyze_scene(grid)

    out = g.copy()
    for obj in objects:
        out[obj["coords"][:, 0], obj["coords"][:, 1]] = bg

    for obj in objects:
        shift = _final_shift(obj)
        dc = -shift if obj["notch_side"] == "left" else shift
        dst = obj["coords"] + np.array([0, dc], dtype=int)
        assert np.all(out[dst[:, 0], dst[:, 1]] == bg), "moved objects must not overlap walls or each other"
        out[dst[:, 0], dst[:, 1]] = obj["color"]

    return out


def _build_object(height: int, width: int, color: int, side: str, bg: int) -> tuple[np.ndarray, int]:
    sprite = np.full((height, width), color, dtype=int)
    notch_row = random.randint(1, height - 2)
    notch_col = 0 if side == "left" else width - 1
    sprite[notch_row, notch_col] = bg
    return sprite, notch_row


def _split_slack(total: int, slots: int) -> list[int]:
    cuts = sorted(random.sample(range(total + slots - 1), slots - 1))
    pieces: list[int] = []
    prev = -1
    for cut in cuts + [total + slots - 1]:
        pieces.append(cut - prev - 1)
        prev = cut
    return pieces


def _sample_object_spec(side: str, exact: bool, jumps: int, corridor: int) -> dict | None:
    for _ in range(200):
        width = random.randint(3, min(6, corridor - 3))
        height = random.randint(3, 5)
        remainder = 0 if exact else random.randint(1, width - 1)
        far_gap = random.randint(1, 4)
        near_gap = jumps * width + remainder
        if width + near_gap + far_gap > corridor:
            continue
        return {
            "side": side,
            "exact": exact,
            "jumps": jumps,
            "remainder": remainder,
            "gap_to_wall": near_gap,
            "far_gap": far_gap,
            "width": width,
            "height": height,
        }
    return None


def _place_object(grid: np.ndarray, left_wall: int, right_wall: int, r0: int, spec: dict, color: int, bg: int) -> None:
    sprite, _notch_row = _build_object(spec["height"], spec["width"], color, spec["side"], bg)
    if spec["side"] == "left":
        c0 = left_wall + 1 + spec["gap_to_wall"]
    else:
        c0 = right_wall - spec["gap_to_wall"] - spec["width"]
    r1 = r0 + spec["height"]
    c1 = c0 + spec["width"]
    assert np.all(grid[r0:r1, c0:c1] == bg), "placement must remain disjoint"
    grid[r0:r1, c0:c1] = sprite


def _semantic_checks(grid: np.ndarray) -> None:
    g, _bg, _left_wall, _right_wall, objects = _analyze_scene(grid)
    out = main(g)
    out_analysis = _analyze_scene(out)[4]

    assert len(objects) == 4, "generator should provide four objects"
    assert len(out_analysis) == len(objects), "object count must stay constant"

    widths = {obj["width"] for obj in objects}
    assert len(widths) >= 2, "examples must rule out a constant jump length shortcut"

    side_counts = {"left": 0, "right": 0}
    exact_seen = False
    remainder_seen = False
    multi_jump_seen = False
    one_jump_seen = False

    for inp_obj, out_obj in zip(objects, out_analysis):
        side = inp_obj["notch_side"]
        side_counts[side] += 1
        gap = int(inp_obj["gap_left"] if side == "left" else inp_obj["gap_right"])
        jumps = gap // int(inp_obj["width"])
        remainder = gap % int(inp_obj["width"])
        expected_shift = jumps * int(inp_obj["width"])
        actual_shift = int(abs(out_obj["bbox"][1] - inp_obj["bbox"][1]))

        assert inp_obj["color"] == out_obj["color"]
        assert inp_obj["bbox"][0] == out_obj["bbox"][0] and inp_obj["bbox"][2] == out_obj["bbox"][2]
        assert inp_obj["width"] == out_obj["width"] and inp_obj["height"] == out_obj["height"]
        assert inp_obj["notch_side"] == out_obj["notch_side"]
        assert actual_shift == expected_shift, "object must move by an integer number of self-width jumps"

        final_gap = int(out_obj["gap_left"] if side == "left" else out_obj["gap_right"])
        assert final_gap == remainder, "final wall gap must equal the original remainder"

        exact_seen |= remainder == 0 and jumps >= 1
        remainder_seen |= remainder > 0 and jumps >= 1
        multi_jump_seen |= jumps >= 2
        one_jump_seen |= jumps == 1

    assert side_counts["left"] >= 2 and side_counts["right"] >= 2, "both directions must be represented clearly"
    assert exact_seen, "generator must include at least one flush-to-wall divisible case"
    assert remainder_seen, "generator must include at least one non-divisible residual-gap case"
    assert multi_jump_seen, "generator must include at least one repeated-jump object"
    assert one_jump_seen, "generator must include at least one single-jump object"


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(MIN_HEIGHT, MAX_HEIGHT)
    width = random.randint(MIN_WIDTH, MAX_WIDTH)
    left_wall = random.randint(1, 3)
    right_wall = random.randint(max(left_wall + 12, width - 5), width - 2)
    corridor = right_wall - left_wall - 1
    if corridor < 10:
        return None

    specs: list[dict] = []
    for side, exact, jumps in MANDATORY_PLANS:
        spec = _sample_object_spec(side, exact, jumps, corridor)
        if spec is None:
            return None
        specs.append(spec)
    if len({spec["width"] for spec in specs}) < 2:
        return None

    total_object_height = sum(spec["height"] for spec in specs)
    base_used = total_object_height + ROW_GAP * (len(specs) - 1) + 2
    if base_used > height:
        return None

    grid = np.full((height, width), BACKGROUND, dtype=int)
    grid[:, left_wall] = WALL_COLOR
    grid[:, right_wall] = WALL_COLOR

    colors = random.sample(OBJECT_COLORS, len(specs))
    slack = height - base_used
    gap_parts = _split_slack(slack, len(specs) + 1)

    r = 1 + gap_parts[0]
    for idx, (spec, color) in enumerate(zip(specs, colors)):
        _place_object(grid, left_wall, right_wall, r, spec, color, BACKGROUND)
        r += spec["height"]
        if idx < len(specs) - 1:
            r += ROW_GAP + gap_parts[idx + 1]

    _semantic_checks(grid)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid quantum-jump ARC input")
