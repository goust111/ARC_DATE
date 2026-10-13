from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, bbox_h, bbox_w, find_components, get_bg, paste


ARROW_PATTERNS = {
    "up": {(0, 0), (0, 1), (0, 2), (2, 1)},
    "right": {(0, 2), (1, 0), (1, 2), (2, 2)},
    "down": {(0, 1), (2, 0), (2, 1), (2, 2)},
    "left": {(0, 0), (1, 0), (1, 2), (2, 0)},
}

DIRS = {
    "up": (-1, 0),
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
}

PAIR_STEPS = [(3, 1), (3, 2), (2, 1), (1, 2), (2, 3), (1, 3)]
BASE_SIDE = 27
GEN_ATTEMPTS = 400


def _intersects(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _bbox_center(box: tuple[int, int, int, int]) -> tuple[float, float]:
    r0, c0, r1, c1 = box
    return (r0 + r1) / 2.0, (c0 + c1) / 2.0


def _make_arrow_patch(a: int, b: int, direction: str) -> np.ndarray:
    patch = np.zeros((3, 3), dtype=int)
    patch[1, 1] = int(b)
    for r, c in ARROW_PATTERNS[direction]:
        patch[r, c] = int(a)
    return patch


def _swap_patch(patch: np.ndarray, a: int, b: int) -> np.ndarray:
    out = patch.copy()
    out[patch == int(a)] = int(b)
    out[patch == int(b)] = int(a)
    return out


def _is_solid_rect(component: dict) -> bool:
    r0, c0, r1, c1 = component["bbox"]
    return int(component["area"]) == (r1 - r0 + 1) * (c1 - c0 + 1)


def _parse_arrow(component: dict, grid: np.ndarray, bg: int) -> dict | None:
    box = component["bbox"]
    if bbox_h(box) != 3 or bbox_w(box) != 3 or int(component["area"]) != 5:
        return None

    r0, c0, r1, c1 = box
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    center = int(patch[1, 1])
    if center == int(bg):
        return None

    colors = sorted(set(map(int, patch.flatten())) - {int(bg)})
    if len(colors) != 2:
        return None

    if int(np.sum(patch == center)) != 1:
        return None

    border_color = [color for color in colors if color != center]
    if len(border_color) != 1:
        return None
    border_color = int(border_color[0])

    border_coords = {
        (int(r), int(c))
        for r, c in np.argwhere(patch == border_color)
    }
    for direction, template in ARROW_PATTERNS.items():
        if border_coords == template:
            return {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "a": border_color,
                "b": center,
                "direction": direction,
                "order_key": (int(r0), int(c0)),
            }
    return None


def _find_arrows(grid: np.ndarray, bg: int) -> list[dict]:
    arrows: list[dict] = []
    for component in find_components(grid, bg=bg, connectivity=4, monochromatic=False):
        arrow = _parse_arrow(component, grid, bg)
        if arrow is not None:
            arrows.append(arrow)
    arrows.sort(key=lambda item: item["order_key"])
    assert 2 <= len(arrows) <= 3, "expected 2 or 3 arrows"
    return arrows


def _find_targets(grid: np.ndarray, bg: int) -> dict[int, list[dict]]:
    targets: dict[int, list[dict]] = {}
    for component in find_components(grid, bg=bg, connectivity=4, monochromatic=True):
        if not _is_solid_rect(component):
            continue
        h = bbox_h(component["bbox"])
        w = bbox_w(component["bbox"])
        if (h, w) not in {(2, 3), (3, 2)}:
            continue
        color = int(component["color"])
        targets.setdefault(color, []).append(
            {
                "color": color,
                "bbox": tuple(map(int, component["bbox"])),
            }
        )
    assert targets, "expected at least one target rectangle"
    return targets


def _step_box(box: tuple[int, int, int, int], direction: str, step: int) -> tuple[int, int, int, int]:
    dr, dc = DIRS[direction]
    r0, c0, r1, c1 = box
    shift_r = 3 * int(step) * int(dr)
    shift_c = 3 * int(step) * int(dc)
    return (r0 + shift_r, c0 + shift_c, r1 + shift_r, c1 + shift_c)


def _box_in_bounds(box: tuple[int, int, int, int], shape: tuple[int, int]) -> bool:
    h, w = shape
    r0, c0, r1, c1 = box
    return 0 <= r0 <= r1 < h and 0 <= c0 <= c1 < w


def _forward_target(
    arrow: dict,
    targets_for_color: list[dict],
    shape: tuple[int, int],
) -> tuple[dict, int]:
    for step in range(1, max(shape) + 1):
        box = _step_box(arrow["bbox"], arrow["direction"], step)
        if not _box_in_bounds(box, shape):
            break
        hits = [target for target in targets_for_color if _intersects(box, target["bbox"])]
        if hits:
            assert len(hits) == 1, "forward hit must be unique"
            return hits[0], int(step)
    raise AssertionError("expected a forward target on the arrow ray")


def _infer_direction_from_target(
    arrow_box: tuple[int, int, int, int],
    target_box: tuple[int, int, int, int],
) -> str:
    ar0, ac0, ar1, ac1 = arrow_box
    tr0, tc0, tr1, tc1 = target_box
    if tc1 < ac0:
        return "left"
    if tc0 > ac1:
        return "right"
    if tr1 < ar0:
        return "up"
    if tr0 > ar1:
        return "down"

    ar, ac = _bbox_center(arrow_box)
    tr, tc = _bbox_center(target_box)
    dr = tr - ar
    dc = tc - ac
    if abs(dc) >= abs(dr):
        return "right" if dc >= 0 else "left"
    return "down" if dr >= 0 else "up"


def _steps_until_overlap(
    arrow_box: tuple[int, int, int, int],
    target_box: tuple[int, int, int, int],
    direction: str,
    shape: tuple[int, int],
) -> int:
    for step in range(1, max(shape) + 1):
        box = _step_box(arrow_box, direction, step)
        if not _box_in_bounds(box, shape):
            break
        if _intersects(box, target_box):
            return int(step)
    raise AssertionError("chosen target is unreachable along the inferred direction")


def _nearest_target(arrow: dict, targets_for_color: list[dict]) -> dict:
    ar, ac = _bbox_center(arrow["bbox"])
    ranked = []
    for target in targets_for_color:
        tr, tc = _bbox_center(target["bbox"])
        dist = abs(tr - ar) + abs(tc - ac)
        ranked.append((dist, target["bbox"][0], target["bbox"][1], target))
    ranked.sort(key=lambda item: item[:3])
    return ranked[0][3]


def _solve_core(
    grid: np.ndarray,
    *,
    target_mode: str,
    swap_mode: str,
) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    arrows = _find_arrows(g, bg)
    targets = _find_targets(g, bg)
    out = g.copy()

    for arrow in arrows:
        assert arrow["b"] in targets, "every arrow center color must have a target"
        candidates = targets[arrow["b"]]
        if target_mode == "forward":
            _target, steps = _forward_target(arrow, candidates, g.shape)
            direction = arrow["direction"]
        elif target_mode == "nearest":
            target = _nearest_target(arrow, candidates)
            direction = _infer_direction_from_target(arrow["bbox"], target["bbox"])
            steps = _steps_until_overlap(arrow["bbox"], target["bbox"], direction, g.shape)
        else:
            raise ValueError("unknown target mode")

        base_patch = _make_arrow_patch(arrow["a"], arrow["b"], direction)
        swapped_patch = _swap_patch(base_patch, arrow["a"], arrow["b"])

        for step in range(1, steps + 1):
            if swap_mode == "alternate":
                patch = swapped_patch if step % 2 == 1 else base_patch
            elif swap_mode == "original":
                patch = base_patch
            elif swap_mode == "swapped":
                patch = swapped_patch
            else:
                raise ValueError("unknown swap mode")
            box = _step_box(arrow["bbox"], direction, step)
            paste(out, patch, box[0], box[1], transparent=bg)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp each two-color arrow forward in its pointed direction, swapping its two colors on alternating steps until it first overlaps a matching target.

    Input structure:
    - The grid has a black background, 2 or 3 separated arrow sprites, and one solid 2x3 or 3x2 target rectangle for each stamping path.
    - Each arrow is a 3x3 asymmetric T-like sprite: four arm pixels use color A, the center uses color B, and the off-center arm placement encodes one of the four cardinal directions.
    - Rectangles use the corresponding center color B; several arrows may share the same B, so direction is needed to know which rectangle lies on that arrow's forward ray.

    Rule:
    - Detect each arrow's pointing direction from its local asymmetric T pattern.
    - Starting three cells ahead of the original arrow, place translated copies of the same 3x3 sprite every three cells along that direction.
    - On odd-numbered copies, swap colors A and B within that copied sprite; on even-numbered copies, keep the original coloring.
    - Stop after placing the first copy whose 3x3 bounding box overlaps a same-color target rectangle on that forward ray.
    - Keep the original arrows unchanged.

    Key insight:
    - The center color names the candidate targets, but the arrow's off-center arm decides which same-color rectangle is the true destination; the copied motif also alternates between original and swapped colors step by step.
    """
    return _solve_core(np.asarray(grid, dtype=int), target_mode="forward", swap_mode="alternate")


def _target_inside_final(final_r: int, final_c: int) -> tuple[int, int, int, int]:
    if random.choice([False, True]):
        top = final_r + random.randint(0, 1)
        return (top, final_c, top + 1, final_c + 2)
    left = final_c + random.randint(0, 1)
    return (final_r, left, final_r + 2, left + 1)


def _draw_rect(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)


def _expanded(box: tuple[int, int, int, int], pad: int, side: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = box
    return (
        max(0, r0 - pad),
        max(0, c0 - pad),
        min(side - 1, r1 + pad),
        min(side - 1, c1 + pad),
    )


def _reserve(mask: np.ndarray, box: tuple[int, int, int, int], pad: int = 1) -> bool:
    r0, c0, r1, c1 = _expanded(box, pad, mask.shape[0])
    if np.any(mask[r0 : r1 + 1, c0 : c1 + 1]):
        return False
    mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return True


def _place_arrow_pair(grid: np.ndarray, occupancy: np.ndarray) -> bool:
    palette = random.sample(list(range(1, 10)), 3)
    shared_b, a1, a2 = map(int, palette)
    steps1, steps2 = random.choice(PAIR_STEPS)
    axis = random.choice(["horizontal", "vertical"])

    if axis == "horizontal":
        row = random.randint(4, 20)
        gap = random.randint(2, 4)
        left_strip = random.randint(1, 3)
        left_arrow_c = left_strip + 3 * steps1
        right_arrow_c = left_arrow_c + 3 + gap
        right_target_c = right_arrow_c + 3 * steps2
        if right_target_c + 2 >= BASE_SIDE - 1:
            return False

        left_box = (row, left_strip, row + 2, left_arrow_c + 2)
        right_box = (row, right_arrow_c, row + 2, right_target_c + 2)
        if not _reserve(occupancy, left_box):
            return False
        if not _reserve(occupancy, right_box):
            return False

        paste(grid, _make_arrow_patch(a1, shared_b, "left"), row, left_arrow_c, transparent=0)
        paste(grid, _make_arrow_patch(a2, shared_b, "right"), row, right_arrow_c, transparent=0)
        _draw_rect(grid, _target_inside_final(row, left_strip), shared_b)
        _draw_rect(grid, _target_inside_final(row, right_target_c), shared_b)
        return True

    col = random.randint(4, 20)
    gap = random.randint(2, 4)
    top_strip = random.randint(1, 3)
    top_arrow_r = top_strip + 3 * steps1
    bottom_arrow_r = top_arrow_r + 3 + gap
    bottom_target_r = bottom_arrow_r + 3 * steps2
    if bottom_target_r + 2 >= BASE_SIDE - 1:
        return False

    top_box = (top_strip, col, top_arrow_r + 2, col + 2)
    bottom_box = (bottom_arrow_r, col, bottom_target_r + 2, col + 2)
    if not _reserve(occupancy, top_box):
        return False
    if not _reserve(occupancy, bottom_box):
        return False

    paste(grid, _make_arrow_patch(a1, shared_b, "up"), top_arrow_r, col, transparent=0)
    paste(grid, _make_arrow_patch(a2, shared_b, "down"), bottom_arrow_r, col, transparent=0)
    _draw_rect(grid, _target_inside_final(top_strip, col), shared_b)
    _draw_rect(grid, _target_inside_final(bottom_target_r, col), shared_b)
    return True


def _generic_strip_box(
    strip_r: int,
    strip_c: int,
    steps: int,
    direction: str,
) -> tuple[int, int, int, int]:
    if direction in {"left", "right"}:
        return (strip_r, strip_c, strip_r + 2, strip_c + 3 * (steps + 1) - 1)
    return (strip_r, strip_c, strip_r + 3 * (steps + 1) - 1, strip_c + 2)


def _place_extra_arrow(grid: np.ndarray, occupancy: np.ndarray) -> bool:
    colors = random.sample(list(range(1, 10)), 2)
    a, b = map(int, colors)
    direction = random.choice(list(DIRS))
    steps = random.randint(1, 3)

    for _ in range(100):
        if direction in {"left", "right"}:
            strip_r = random.randint(1, BASE_SIDE - 4)
            strip_c = random.randint(1, BASE_SIDE - (3 * (steps + 1)) - 1)
        else:
            strip_r = random.randint(1, BASE_SIDE - (3 * (steps + 1)) - 1)
            strip_c = random.randint(1, BASE_SIDE - 4)

        box = _generic_strip_box(strip_r, strip_c, steps, direction)
        if not _reserve(occupancy, box):
            continue

        if direction == "right":
            arrow_r, arrow_c = strip_r, strip_c
            final_r, final_c = strip_r, strip_c + 3 * steps
        elif direction == "left":
            arrow_r, arrow_c = strip_r, strip_c + 3 * steps
            final_r, final_c = strip_r, strip_c
        elif direction == "down":
            arrow_r, arrow_c = strip_r, strip_c
            final_r, final_c = strip_r + 3 * steps, strip_c
        else:
            arrow_r, arrow_c = strip_r + 3 * steps, strip_c
            final_r, final_c = strip_r, strip_c

        paste(grid, _make_arrow_patch(a, b, direction), arrow_r, arrow_c, transparent=0)
        _draw_rect(grid, _target_inside_final(final_r, final_c), b)
        return True
    return False


def _crop_random(grid: np.ndarray) -> np.ndarray:
    coords = np.argwhere(grid != 0)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    top = random.randint(1, min(3, r0))
    left = random.randint(1, min(3, c0))
    bottom = random.randint(1, min(3, BASE_SIDE - 1 - r1))
    right = random.randint(1, min(3, BASE_SIDE - 1 - c1))
    return grid[r0 - top : r1 + bottom + 1, c0 - left : c1 + right + 1].copy()


def _semantic_checks(grid: np.ndarray) -> bool:
    out = main(grid)
    if np.array_equal(out, grid):
        return False

    bg = int(get_bg(grid))
    arrows = _find_arrows(grid, bg)
    targets = _find_targets(grid, bg)

    if len(targets) < 1:
        return False

    parities = []
    reused_center = False
    for arrow in arrows:
        same_color_targets = targets.get(arrow["b"], [])
        if len(same_color_targets) >= 2:
            reused_center = True
        _, steps = _forward_target(arrow, same_color_targets, grid.shape)
        parities.append(steps % 2)

    if not reused_center:
        return False
    if 0 not in parities or 1 not in parities:
        return False

    try:
        naive_nearest = _solve_core(grid, target_mode="nearest", swap_mode="alternate")
    except Exception:
        naive_nearest = None
    naive_original = _solve_core(grid, target_mode="forward", swap_mode="original")
    naive_swapped = _solve_core(grid, target_mode="forward", swap_mode="swapped")

    if isinstance(naive_nearest, np.ndarray) and np.array_equal(out, naive_nearest):
        return False
    if np.array_equal(out, naive_original):
        return False
    if np.array_equal(out, naive_swapped):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    grid = np.zeros((BASE_SIDE, BASE_SIDE), dtype=int)
    occupancy = np.zeros_like(grid, dtype=bool)

    if not _place_arrow_pair(grid, occupancy):
        return None

    if random.choice([False, True]):
        if not _place_extra_arrow(grid, occupancy):
            return None

    cropped = _crop_random(grid)
    try:
        main(cropped)
    except Exception:
        return None
    if not _semantic_checks(cropped):
        return None
    return cropped


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
