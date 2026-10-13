from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, find_components, paste


TEMPLATE_COLOR = Color.BLUE
ARROW_COLOR = Color.GREEN
BG = Color.BLACK

TEMPLATE_SHAPE = (3, 3)
TEMPLATE_CENTER = (1, 1)
ARROW_BBOXES = ((2, 3), (3, 2))
CORNER_TO_CW = {
    (0, 0): 0,
    (0, 1): 1,
    (1, 1): 2,
    (1, 0): 3,
}

GRID_MIN = 17
GRID_MAX = 24
ARROW_COUNT_MIN = 3
ARROW_COUNT_MAX = 5
PLACEMENT_GAP = 4
GENERATION_RETRIES = 400


def _rotated_template(template: np.ndarray, cw_quarters: int) -> np.ndarray:
    return np.rot90(template, (-int(cw_quarters)) % 4)


def _bbox_gap_ok(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    return (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    stack = [start]
    seen = {start}
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _all_rotations_distinct(template: np.ndarray) -> bool:
    rotations = [_rotated_template(template, k) for k in range(4)]
    return len({rot.tobytes() for rot in rotations}) == 4


def _extract_template(grid: np.ndarray) -> np.ndarray:
    template = np.zeros(TEMPLATE_SHAPE, dtype=int)
    patch = grid[:3, :3]
    template[patch == TEMPLATE_COLOR] = TEMPLATE_COLOR
    return template


def _infer_arrow_rotation(crop: np.ndarray) -> int:
    h, w = crop.shape
    assert (h, w) in ARROW_BBOXES, "arrow bbox must be 2x3 or 3x2"
    assert int(np.count_nonzero(crop == ARROW_COLOR)) == 5, "arrow must contain five green cells"
    mask = crop == ARROW_COLOR
    assert int(mask.sum()) == h * w - 1, "arrow must be a full rectangle minus one corner"

    missing = []
    corner_positions = ((0, 0), (0, w - 1), (h - 1, w - 1), (h - 1, 0))
    for rr, cc in corner_positions:
        if not mask[rr, cc]:
            missing.append((rr, cc))
    assert len(missing) == 1, "arrow must miss exactly one bbox corner"

    rr, cc = missing[0]
    key = (int(rr == h - 1), int(cc == w - 1))
    return CORNER_TO_CW[key]


def _find_arrows(grid: np.ndarray) -> list[dict]:
    comps = find_components(
        grid,
        bg=BG,
        connectivity=4,
        monochromatic=True,
        colors=[ARROW_COLOR],
        return_format="dict",
    )
    arrows: list[dict] = []
    for comp in comps:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        crop = grid[r0 : r1 + 1, c0 : c1 + 1]
        rotation = _infer_arrow_rotation(crop)
        arrows.append(
            {
                "bbox": (r0, c0, r1, c1),
                "rotation": rotation,
            }
        )
    arrows.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))
    return arrows


def _render_output_from_specs(
    shape: tuple[int, int],
    template: np.ndarray,
    arrows: list[dict],
) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    out[:3, :3] = template
    for arrow in arrows:
        r0, c0, r1, c1 = arrow["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = BG
        paste(out, _rotated_template(template, arrow["rotation"]), r0, c0, transparent=BG)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate the top-left template by each arrow's missing-corner direction and stamp it over the arrow position.

    Input structure:
    - The grid has a black background.
    - The top-left 3x3 area contains a blue connected template sprite with 3-5 pixels and no 90-degree rotational symmetry.
    - Elsewhere there are 3-5 green arrow sprites; each is a 2x3 or 3x2 rectangle missing exactly one corner cell.

    Rule:
    - Read the blue template from the top-left 3x3 region.
    - For each green arrow, inspect its tight bbox and find the unique missing corner: top-left means 0 degrees, top-right means 90 degrees clockwise, bottom-right means 180 degrees, bottom-left means 270 degrees clockwise.
    - Erase that arrow back to black, then stamp the correspondingly rotated blue template with template center aligned to the arrow bbox center, processing arrows in row-major bbox order.
    - Keep the original top-left template unchanged.

    Key insight:
    - The arrow does not indicate motion; its missing corner encodes which one of the four distinct template rotations must replace it.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert min(g.shape) >= 3, "grid must contain the top-left 3x3 template area"

    template = _extract_template(g)
    template_mask = template == TEMPLATE_COLOR
    assert 3 <= int(template_mask.sum()) <= 5, "template must use 3-5 cells"
    assert _connected(template_mask), "template must be 4-connected"
    assert _all_rotations_distinct(template), "template rotations must all differ"
    assert not np.any(g[3:, :3] == TEMPLATE_COLOR) and not np.any(g[:3, 3:] == TEMPLATE_COLOR), (
        "input should contain blue only inside the top-left template region"
    )

    arrows = _find_arrows(g)
    assert ARROW_COUNT_MIN <= len(arrows) <= ARROW_COUNT_MAX, "expected 3-5 arrows"

    out = g.copy()
    out[:3, :3] = template
    for arrow in arrows:
        r0, c0, r1, c1 = arrow["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = BG
        paste(out, _rotated_template(template, arrow["rotation"]), r0, c0, transparent=BG)
    return out


def _sample_template() -> np.ndarray:
    for _ in range(200):
        area = random.randint(3, 5)
        mask = np.zeros(TEMPLATE_SHAPE, dtype=bool)
        cells = [(random.randrange(3), random.randrange(3))]
        mask[cells[0]] = True
        while int(mask.sum()) < area:
            frontier = []
            for r, c in np.argwhere(mask):
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = int(r + dr), int(c + dc)
                    if 0 <= nr < 3 and 0 <= nc < 3 and not mask[nr, nc]:
                        frontier.append((nr, nc))
            if not frontier:
                break
            nr, nc = random.choice(frontier)
            mask[nr, nc] = True
        template = np.zeros(TEMPLATE_SHAPE, dtype=int)
        template[mask] = TEMPLATE_COLOR
        if int(mask.sum()) != area:
            continue
        if not _all_rotations_distinct(template):
            continue
        if np.count_nonzero(mask.any(axis=0)) < 2 or np.count_nonzero(mask.any(axis=1)) < 2:
            continue
        return template
    raise RuntimeError("failed to sample a non-degenerate template")


def _direction_multiset(count: int) -> list[int]:
    assert ARROW_COUNT_MIN <= count <= ARROW_COUNT_MAX
    if count == 3:
        dirs = random.sample(range(4), 3)
    elif count == 4:
        dirs = list(range(4))
    else:
        dirs = list(range(4)) + [random.randrange(4)]
    random.shuffle(dirs)
    return dirs


def _bbox_choices(count: int) -> list[tuple[int, int]]:
    choices = [random.choice(ARROW_BBOXES) for _ in range(count)]
    if count >= 2 and len(set(choices)) == 1:
        idx = random.randrange(count)
        choices[idx] = ARROW_BBOXES[1] if choices[idx] == ARROW_BBOXES[0] else ARROW_BBOXES[0]
    return choices


def _candidate_origins(shape: tuple[int, int], existing: list[tuple[int, int, int, int]]) -> list[tuple[int, int]]:
    h, w = shape
    candidates = []
    for r0 in range(4, h - 2):
        for c0 in range(4, w - 2):
            box = (r0, c0, r0 + 2, c0 + 2)
            if all(_bbox_gap_ok(box, other, PLACEMENT_GAP) for other in existing):
                candidates.append((r0, c0))
    return candidates


def _make_arrow_patch(bbox_hw: tuple[int, int], rotation: int) -> np.ndarray:
    h, w = bbox_hw
    patch = np.full((h, w), ARROW_COLOR, dtype=int)
    missing_lookup = {
        0: (0, 0),
        1: (0, w - 1),
        2: (h - 1, w - 1),
        3: (h - 1, 0),
    }
    rr, cc = missing_lookup[rotation]
    patch[rr, cc] = BG
    return patch


def _build_input_from_specs(
    shape: tuple[int, int],
    template: np.ndarray,
    specs: list[dict],
) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    grid[:3, :3] = template
    for spec in specs:
        r0, c0 = spec["origin"]
        patch = _make_arrow_patch(spec["bbox_hw"], spec["rotation"])
        paste(grid, patch, r0, c0, transparent=BG)
    return grid


def _specs_from_input(grid: np.ndarray) -> list[dict]:
    specs = []
    for arrow in _find_arrows(grid):
        r0, c0, r1, c1 = arrow["bbox"]
        specs.append(
            {
                "origin": (r0, c0),
                "bbox_hw": (r1 - r0 + 1, c1 - c0 + 1),
                "rotation": arrow["rotation"],
            }
        )
    return specs


def _semantic_checks(grid: np.ndarray, specs: list[dict], template: np.ndarray, expected: np.ndarray) -> bool:
    if grid.shape != expected.shape:
        return False
    if np.any(grid[3:, :3] == TEMPLATE_COLOR) or np.any(grid[:3, 3:] == TEMPLATE_COLOR):
        return False

    recovered_template = _extract_template(grid)
    if not np.array_equal(recovered_template, template):
        return False
    if not _all_rotations_distinct(template):
        return False

    recovered_specs = _specs_from_input(grid)
    wanted = sorted(specs, key=lambda item: item["origin"])
    if recovered_specs != wanted:
        return False

    directions = {spec["rotation"] for spec in specs}
    if len(directions) < min(3, len(specs)):
        return False
    if len(specs) >= 4 and directions != {0, 1, 2, 3} and len(specs) != 3:
        return False

    stamp_boxes = []
    for spec in specs:
        r0, c0 = spec["origin"]
        stamp_box = (r0, c0, r0 + 2, c0 + 2)
        if not (0 <= r0 and 0 <= c0 and r0 + 2 < grid.shape[0] and c0 + 2 < grid.shape[1]):
            return False
        if not _bbox_gap_ok(stamp_box, (0, 0, 2, 2), 1):
            return False
        if any(not _bbox_gap_ok(stamp_box, other, 0) for other in stamp_boxes):
            return False
        stamp_boxes.append(stamp_box)

    rotations = [_rotated_template(template, k) for k in range(4)]
    for i in range(4):
        for j in range(i + 1, 4):
            if np.array_equal(rotations[i], rotations[j]):
                return False

    alt = _render_output_from_specs(
        grid.shape,
        template,
        [{**spec, "bbox": (spec["origin"][0], spec["origin"][1], spec["origin"][0] + spec["bbox_hw"][0] - 1, spec["origin"][1] + spec["bbox_hw"][1] - 1)} for spec in specs],
    )
    if not np.array_equal(alt, expected):
        return False

    if len(specs) > 1:
        changed = False
        for idx, spec in enumerate(specs):
            if spec["rotation"] != 0:
                wrong_specs = [dict(item) for item in specs]
                wrong_specs[idx]["rotation"] = 0
                wrong_alt = _render_output_from_specs(
                    grid.shape,
                    template,
                    [
                        {
                            **item,
                            "bbox": (
                                item["origin"][0],
                                item["origin"][1],
                                item["origin"][0] + item["bbox_hw"][0] - 1,
                                item["origin"][1] + item["bbox_hw"][1] - 1,
                            ),
                        }
                        for item in wrong_specs
                    ],
                )
                if not np.array_equal(wrong_alt, expected):
                    changed = True
                    break
        if not changed:
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    shape = (
        random.randint(GRID_MIN, GRID_MAX),
        random.randint(GRID_MIN, GRID_MAX),
    )
    count = random.randint(ARROW_COUNT_MIN, ARROW_COUNT_MAX)
    template = _sample_template()
    directions = _direction_multiset(count)
    bbox_choices = _bbox_choices(count)

    specs: list[dict] = []
    taken: list[tuple[int, int, int, int]] = [(0, 0, 2, 2)]
    for rotation, bbox_hw in zip(directions, bbox_choices):
        candidates = _candidate_origins(shape, taken)
        if not candidates:
            return None
        origin = random.choice(candidates)
        specs.append(
            {
                "origin": origin,
                "bbox_hw": bbox_hw,
                "rotation": rotation,
            }
        )
        r0, c0 = origin
        taken.append((r0, c0, r0 + 2, c0 + 2))

    specs.sort(key=lambda item: item["origin"])
    grid = _build_input_from_specs(shape, template, specs)
    expected = _render_output_from_specs(
        shape,
        template,
        [
            {
                "bbox": (
                    spec["origin"][0],
                    spec["origin"][1],
                    spec["origin"][0] + spec["bbox_hw"][0] - 1,
                    spec["origin"][1] + spec["bbox_hw"][1] - 1,
                ),
                "rotation": spec["rotation"],
            }
            for spec in specs
        ],
    )

    if not _semantic_checks(grid, specs, template, expected):
        return None
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid arrow-template input")
