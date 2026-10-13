from __future__ import annotations

import random
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np


try:
    from arc_utils import Color, crop_bbox, find_components, get_bg, paste
except ModuleNotFoundError:
    _ARC_UTILS = SourceFileLoader(
        "_arc_utils_fable_2_025_10",
        str(Path("/home/lixinyu/snap/new/solvers/arc_utils.py")),
    ).load_module()
    Color = _ARC_UTILS.Color
    crop_bbox = _ARC_UTILS.crop_bbox
    find_components = _ARC_UTILS.find_components
    get_bg = _ARC_UTILS.get_bg
    paste = _ARC_UTILS.paste


DIRECTIONS = ("up", "right", "down", "left")
DIR_VECS = {
    "up": (-1, 0),
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
}
NON_BG_NON_GREY = [c for c in Color.NOT_BLACK if c != Color.GREY]
MAX_GENERATE_ATTEMPTS = 400


def _rotate_direction_ccw(direction: str, turns: int) -> str:
    vec = DIR_VECS[direction]
    for _ in range(int(turns) % 4):
        vec = (-vec[1], vec[0])
    for name, other in DIR_VECS.items():
        if other == vec:
            return name
    raise AssertionError("unreachable direction rotation")


def _rotation_turns(start: str, target: str) -> int:
    for turns in range(4):
        if _rotate_direction_ccw(start, turns) == target:
            return turns
    raise AssertionError("unreachable rotation lookup")


def _dominant_direction(dr: int, dc: int) -> str | None:
    dr = int(dr)
    dc = int(dc)
    if dr == 0 and dc == 0:
        return None
    if abs(dr) == abs(dc):
        return None
    if abs(dr) > abs(dc):
        return "down" if dr > 0 else "up"
    return "right" if dc > 0 else "left"


def _boxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = map(int, a)
    br0, bc0, br1, bc1 = map(int, b)
    gap = int(gap)
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _center_box(center: tuple[int, int], half: int) -> tuple[int, int, int, int]:
    r, c = map(int, center)
    half = int(half)
    return r - half, c - half, r + half, c + half


def _pointing_direction(sprite: np.ndarray, bg: int) -> str:
    mask = np.asarray(sprite) != int(bg)
    assert mask.ndim == 2 and mask.shape[0] == mask.shape[1] and mask.shape[0] % 2 == 1
    half = mask.shape[0] // 2
    side_indices = {
        "up": np.flatnonzero(mask[0, :]),
        "right": np.flatnonzero(mask[:, -1]),
        "down": np.flatnonzero(mask[-1, :]),
        "left": np.flatnonzero(mask[:, 0]),
    }
    side_counts = {name: int(len(idxs)) for name, idxs in side_indices.items()}
    min_count = min(side_counts.values())
    winners = [name for name, count in side_counts.items() if count == min_count]
    assert min_count == 1 and len(winners) == 1, "template must have one unique tip side"
    tip_dir = winners[0]
    tip_index = int(side_indices[tip_dir][0])
    assert tip_index == half, "the tip pixel must be centered on its side"
    for other_dir, count in side_counts.items():
        if other_dir != tip_dir:
            assert count >= 2, "non-tip sides must stay visibly broader than the tip side"
    for turns in range(1, 4):
        assert not np.array_equal(sprite, np.rot90(sprite, turns)), "template must not be rotation-symmetric"
    return tip_dir


def _extract_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = int(get_bg(g))
    assert bg == Color.BLACK

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    template_candidates = [comp for comp in comps if int(comp["area"]) > 1]
    control_candidates = [comp for comp in comps if int(comp["area"]) == 1]
    assert len(template_candidates) == 1, "expected exactly one multi-pixel template"
    assert control_candidates, "expected at least one gray control point"

    template = template_candidates[0]
    r0, c0, r1, c1 = map(int, template["bbox"])
    side = r1 - r0 + 1
    assert side == c1 - c0 + 1 and side % 2 == 1
    sprite = crop_bbox(g, template["bbox"])
    center = ((r0 + r1) // 2, (c0 + c1) // 2)
    tip_dir = _pointing_direction(sprite, bg)

    controls: list[tuple[int, int]] = []
    for comp in control_candidates:
        rr, cc = map(int, comp["coords"][0])
        assert int(g[rr, cc]) == Color.GREY, "all singleton controls must be gray"
        direction = _dominant_direction(rr - center[0], cc - center[1])
        assert direction is not None, "control points must have a unique dominant axis"
        controls.append((rr, cc))

    return {
        "grid": g,
        "bg": bg,
        "sprite": sprite,
        "template_bbox": (r0, c0, r1, c1),
        "center": center,
        "tip_dir": tip_dir,
        "controls": controls,
    }


def _render_copies(
    grid: np.ndarray,
    sprite: np.ndarray,
    controls: list[tuple[int, int]],
    desired_dirs: list[str],
    tip_dir: str,
    bg: int,
) -> np.ndarray:
    out = np.array(grid, copy=True)
    half = sprite.shape[0] // 2
    for (r, c), desired in zip(controls, desired_dirs):
        turns = _rotation_turns(tip_dir, desired)
        stamp = np.rot90(sprite, turns)
        paste(out, stamp, int(r) - half, int(c) - half, transparent=bg)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp the pointed template onto each gray control pixel, rotating each copy so its tip faces away from the original template center.

    Input structure:
    - The grid has a black background, one connected multicolor template with an odd square bbox, and several isolated gray single-pixel controls.
    - The template has one uniquely narrow side: a centered one-pixel tip that defines its current facing direction.
    - Every control point lies far enough from the template and other controls that centered template copies fit without overlap.

    Rule:
    - Keep the original template unchanged.
    - For each gray control pixel, compare its row/column offset from the original template center; the larger absolute offset gives the dominant cardinal direction.
    - Rotate the template around its own center so the one-pixel tip faces that dominant direction, then stamp the rotated copy centered on the control pixel, covering the gray pixel.

    Key insight:
    - Gray pixels do not encode colors; they are radial bearings from the original template center, and each bearing chooses the rotation that makes the tip point outward.
    """
    scene = _extract_scene(np.asarray(grid, dtype=int))
    center_r, center_c = scene["center"]
    desired_dirs = [
        _dominant_direction(r - center_r, c - center_c)
        for r, c in scene["controls"]
    ]
    assert all(direction is not None for direction in desired_dirs)
    return _render_copies(
        scene["grid"],
        scene["sprite"],
        scene["controls"],
        desired_dirs,  # type: ignore[arg-type]
        scene["tip_dir"],
        scene["bg"],
    )


def _build_right_pointing_mask(size: int) -> np.ndarray:
    size = int(size)
    half = size // 2
    while True:
        mask = np.zeros((size, size), dtype=bool)
        tail_choices = [tail for tail in (3, 5) if tail <= size]
        tail_len = random.choice(tail_choices)
        tail_half = tail_len // 2
        mask[half - tail_half : half + tail_half + 1, 0] = True
        mask[:, 1] = True
        mask[half, 1:size] = True

        top_len = random.randint(2, size - 2)
        bottom_choices = [length for length in range(2, size - 1) if length != top_len]
        bottom_len = random.choice(bottom_choices)
        mask[0, 1 : 1 + top_len] = True
        mask[-1, 1 : 1 + bottom_len] = True

        upper_row = random.randint(1, half - 1)
        lower_row = random.randint(half + 1, size - 2)
        upper_len = random.randint(2, size - 2)
        lower_len_choices = [length for length in range(2, size - 1) if length != upper_len or upper_row != size - 1 - lower_row]
        lower_len = random.choice(lower_len_choices)
        mask[upper_row, 1 : 1 + upper_len] = True
        mask[lower_row, 1 : 1 + lower_len] = True

        if size >= 7 and random.random() < 0.65:
            extra_rows = [row for row in range(1, size - 1) if row not in {upper_row, half, lower_row}]
            extra_row = random.choice(extra_rows)
            extra_len = random.randint(2, size - 3)
            mask[extra_row, 1 : 1 + extra_len] = True

        if int(mask[:, -1].sum()) != 1 or not bool(mask[half, -1]):
            continue
        if min(int(mask[0].sum()), int(mask[-1].sum()), int(mask[:, 0].sum())) < 2:
            continue
        if any(np.array_equal(mask, np.rot90(mask, turns)) for turns in (1, 2, 3)):
            continue
        return mask


def _colorize_mask(mask: np.ndarray) -> np.ndarray:
    size = mask.shape[0]
    half = size // 2
    colors = random.sample(NON_BG_NON_GREY, k=random.randint(4, 5))
    sprite = np.zeros(mask.shape, dtype=int)
    sprite[mask] = colors[0]

    spine_mask = np.zeros_like(mask)
    spine_mask[:, 1] = mask[:, 1]
    sprite[spine_mask] = colors[1]

    center_bar_mask = np.zeros_like(mask)
    center_bar_mask[half, :] = mask[half, :]
    sprite[center_bar_mask] = colors[2]

    top_mask = np.zeros_like(mask)
    top_mask[0, :] = mask[0, :]
    sprite[top_mask] = colors[3]

    bottom_mask = np.zeros_like(mask)
    bottom_mask[-1, :] = mask[-1, :]
    sprite[bottom_mask] = colors[4 % len(colors)]

    accent_candidates = [(int(r), int(c)) for r, c in np.argwhere(mask) if c >= 2 and (r, c) != (half, size - 1)]
    random.shuffle(accent_candidates)
    for idx, (r, c) in enumerate(accent_candidates[:3]):
        sprite[r, c] = colors[(idx + 1) % len(colors)]

    assert len(set(map(int, sprite[mask].tolist()))) >= 3
    return sprite


def _build_template(size: int) -> np.ndarray:
    sprite = _colorize_mask(_build_right_pointing_mask(size))
    for turns in range(1, 4):
        if np.array_equal(sprite, np.rot90(sprite, turns)):
            raise AssertionError("colored template must remain orientation-sensitive")
    return sprite


def _candidate_controls(
    shape: tuple[int, int],
    template_bbox: tuple[int, int, int, int],
    center: tuple[int, int],
    half: int,
) -> dict[str, list[tuple[int, int]]]:
    rows, cols = map(int, shape)
    center_r, center_c = map(int, center)
    candidates = {direction: [] for direction in DIRECTIONS}
    for r in range(half, rows - half):
        for c in range(half, cols - half):
            direction = _dominant_direction(r - center_r, c - center_c)
            if direction is None:
                continue
            box = _center_box((r, c), half)
            if not _boxes_separated(box, template_bbox, gap=1):
                continue
            candidates[direction].append((int(r), int(c)))
    return candidates


def _pick_controls(
    shape: tuple[int, int],
    template_bbox: tuple[int, int, int, int],
    center: tuple[int, int],
    half: int,
) -> list[tuple[int, int]] | None:
    candidates = _candidate_controls(shape, template_bbox, center, half)
    available_dirs = [direction for direction in DIRECTIONS if candidates[direction]]
    if len(available_dirs) < 3:
        return None

    for _ in range(200):
        need_four = len(available_dirs) == 4 and random.random() < 0.7
        required_dirs = DIRECTIONS if need_four else tuple(random.sample(available_dirs, 3))
        total_controls = random.randint(len(required_dirs), min(6, len(required_dirs) + 2))
        plan = list(required_dirs)
        while len(plan) < total_controls:
            plan.append(random.choice(required_dirs))
        random.shuffle(plan)

        selected: list[tuple[int, int]] = []
        selected_boxes: list[tuple[int, int, int, int]] = [template_bbox]
        ok = True
        for direction in plan:
            pool = random.sample(candidates[direction], len(candidates[direction]))
            chosen = None
            for point in pool:
                point_box = _center_box(point, half)
                if all(_boxes_separated(point_box, other, gap=1) for other in selected_boxes):
                    chosen = point
                    break
            if chosen is None:
                ok = False
                break
            selected.append(chosen)
            selected_boxes.append(_center_box(chosen, half))

        if not ok:
            continue

        dirs = {
            _dominant_direction(point[0] - center[0], point[1] - center[1])
            for point in selected
        }
        if None in dirs or len(dirs) < 3:
            continue
        if not ({"up", "down"} & dirs and {"left", "right"} & dirs):
            continue
        return selected
    return None


def _expected_output_from_metadata(
    grid: np.ndarray,
    sprite: np.ndarray,
    controls: list[tuple[int, int]],
    center: tuple[int, int],
    tip_dir: str,
    bg: int,
) -> np.ndarray:
    desired_dirs = [
        _dominant_direction(r - center[0], c - center[1])
        for r, c in controls
    ]
    assert all(direction is not None for direction in desired_dirs)
    return _render_copies(grid, sprite, controls, desired_dirs, tip_dir, bg)  # type: ignore[arg-type]


def _validate_instance(
    grid: np.ndarray,
    sprite: np.ndarray,
    controls: list[tuple[int, int]],
    center: tuple[int, int],
    template_bbox: tuple[int, int, int, int],
    tip_dir: str,
) -> bool:
    bg = Color.BLACK
    expected = _expected_output_from_metadata(grid, sprite, controls, center, tip_dir, bg)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False

    desired_dirs = [_dominant_direction(r - center[0], c - center[1]) for r, c in controls]
    if any(direction is None for direction in desired_dirs):
        return False
    if len(set(desired_dirs)) < 3:
        return False
    if not ({"up", "down"} & set(desired_dirs) and {"left", "right"} & set(desired_dirs)):
        return False
    if all(direction == tip_dir for direction in desired_dirs):
        return False

    fixed = _render_copies(grid, sprite, controls, [tip_dir] * len(controls), tip_dir, bg)
    if np.array_equal(fixed, expected):
        return False

    half = sprite.shape[0] // 2
    boxes = [template_bbox] + [_center_box(point, half) for point in controls]
    for idx, box in enumerate(boxes):
        for other in boxes[idx + 1 :]:
            if not _boxes_separated(box, other, gap=1):
                return False

    return True


def _generate_attempt() -> np.ndarray | None:
    size = random.choice((5, 7))
    sprite = _build_template(size)
    initial_turns = random.randrange(4)
    sprite = np.rot90(sprite, initial_turns)
    tip_dir = _rotate_direction_ccw("right", initial_turns)

    min_dim = max(18, 2 * size + 6)
    rows = random.randint(min_dim, 30)
    cols = random.randint(min_dim, 30)
    half = size // 2

    center_r = random.randint(size + 1, rows - size - 2)
    center_c = random.randint(size + 1, cols - size - 2)
    template_bbox = (center_r - half, center_c - half, center_r + half, center_c + half)

    controls = _pick_controls((rows, cols), template_bbox, (center_r, center_c), half)
    if controls is None:
        return None

    grid = np.full((rows, cols), Color.BLACK, dtype=int)
    paste(grid, sprite, template_bbox[0], template_bbox[1], transparent=Color.BLACK)
    for r, c in controls:
        grid[int(r), int(c)] = Color.GREY

    if not _validate_instance(grid, sprite, controls, (center_r, center_c), template_bbox, tip_dir):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATE_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid backward-stamp-field instance")
