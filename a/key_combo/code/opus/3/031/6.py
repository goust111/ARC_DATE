from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np


_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import (  # noqa: E402
    Color,
    bbox,
    bbox_h,
    bbox_w,
    crop_bbox,
    find_components,
    get_bg,
    paste,
    scale_sprite,
)


TEMPLATE_SIZE = 3
CANVAS_SIZE = 30
FRAME_COUNT_CHOICES = (2, 3)
SCALE_CHOICES = (1, 2, 3)
TEMPLATE_COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
NOISE_COLOR = Color.GREY
BACKGROUND = Color.BLACK
PLACEMENT_GAP = 1
MAX_GENERATION_ATTEMPTS = 400


def _sort_key_bbox(box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    return int(box[0]), int(box[1]), int(box[2]), int(box[3])


def _rectangles_overlap(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + pad < br0
        or br1 + pad < ar0
        or ac1 + pad < bc0
        or bc1 + pad < ac0
    )


def _erase_noise(grid: np.ndarray, bg: int) -> np.ndarray:
    clean = np.asarray(grid, dtype=int).copy()
    clean[clean == int(NOISE_COLOR)] = int(bg)
    return clean


def _find_template(
    grid: np.ndarray, bg: int
) -> tuple[tuple[int, int, int, int], np.ndarray, tuple[int, ...]]:
    candidates: list[tuple[tuple[int, int, int, int], np.ndarray, tuple[int, ...]]] = []
    for comp in find_components(grid, bg=bg, connectivity=4, monochromatic=False):
        box = tuple(map(int, comp["bbox"]))
        if bbox_h(box) != TEMPLATE_SIZE or bbox_w(box) != TEMPLATE_SIZE:
            continue
        patch = crop_bbox(grid, box)
        colors = tuple(sorted(set(map(int, np.unique(patch))) - {int(bg)}))
        if len(colors) < 2:
            continue
        candidates.append((box, patch, colors))
    assert candidates, "expected a 3x3 multicolor template"
    return min(candidates, key=lambda item: _sort_key_bbox(item[0]))


def _is_hollow_frame(
    grid: np.ndarray, comp: dict, bg: int, template_box: tuple[int, int, int, int]
) -> bool:
    box = tuple(map(int, comp["bbox"]))
    if box[0] >= template_box[0] and box[1] >= template_box[1] and box[2] <= template_box[2] and box[3] <= template_box[3]:
        return False
    height = bbox_h(box)
    width = bbox_w(box)
    if height < TEMPLATE_SIZE + 2 or width < TEMPLATE_SIZE + 2:
        return False

    color = int(comp["color"])
    patch = crop_bbox(grid, box)
    if comp["area"] != 2 * (height + width) - 4:
        return False
    if not np.all(patch[0, :] == color):
        return False
    if not np.all(patch[-1, :] == color):
        return False
    if not np.all(patch[:, 0] == color):
        return False
    if not np.all(patch[:, -1] == color):
        return False
    if not np.all(patch[1:-1, 1:-1] == int(bg)):
        return False
    return True


def _find_frames(
    grid: np.ndarray,
    bg: int,
    template_box: tuple[int, int, int, int],
    template_colors: tuple[int, ...],
) -> list[dict]:
    frames: list[dict] = []
    for color in template_colors:
        comps = find_components(
            grid,
            bg=bg,
            connectivity=4,
            monochromatic=True,
            colors=[color],
        )
        for comp in comps:
            if not _is_hollow_frame(grid, comp, bg, template_box):
                continue
            box = tuple(map(int, comp["bbox"]))
            inner_h = bbox_h(box) - 2
            inner_w = bbox_w(box) - 2
            short_side = min(inner_h, inner_w)
            assert short_side % TEMPLATE_SIZE == 0, "frame short side must be a multiple of 3"
            scale = short_side // TEMPLATE_SIZE
            assert scale >= 1
            frames.append(
                {
                    "color": int(color),
                    "bbox": box,
                    "inner_r": int(box[0] + 1),
                    "inner_c": int(box[1] + 1),
                    "inner_h": int(inner_h),
                    "inner_w": int(inner_w),
                    "scale": int(scale),
                }
            )
    assert len(frames) in FRAME_COUNT_CHOICES, "expected two or three hollow frames"
    scales = sorted(frame["scale"] for frame in frames)
    assert len(set(scales)) == len(scales), "frame scales must be distinct"
    return sorted(frames, key=lambda frame: _sort_key_bbox(frame["bbox"]))


def _paint_frame_contents(
    out: np.ndarray, template: np.ndarray, frames: list[dict], bg: int
) -> np.ndarray:
    for frame in frames:
        scaled = scale_sprite(template, frame["scale"])
        filtered = np.where(scaled == int(frame["color"]), int(frame["color"]), int(bg))
        patch = filtered[: frame["inner_h"], : frame["inner_w"]]
        paste(out, patch, frame["inner_r"], frame["inner_c"], transparent=bg)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Scale the top-left 3x3 template to each hollow frame and keep only the template pixels whose color matches that frame.

    Input structure:
    - A black background contains one small 3x3 multicolor template near the upper-left corner.
    - Elsewhere there are two or three non-overlapping hollow rectangular frames with 1-pixel borders.
    - Each frame border color appears in the template, and isolated grey pixels near frames are distractor noise.

    Rule:
    - Erase every grey distractor pixel to black.
    - Read the 3x3 template.
    - For each hollow frame, use the shorter interior side length S to compute an integer scale k = S / 3.
    - Scale the whole template by k, keep only the scaled cells whose color equals the frame border color, and place that filtered result at the frame interior's top-left corner, cropping if needed.
    - Leave the template and frame borders unchanged.

    Key insight:
    - Each frame encodes two independent parameters: its interior size gives the scale factor, and its border color selects which template color survives after scaling.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    bg = int(get_bg(g))
    clean = _erase_noise(g, bg)
    template_box, template, template_colors = _find_template(clean, bg)
    frames = _find_frames(clean, bg, template_box, template_colors)

    out = clean.copy()
    return _paint_frame_contents(out, template, frames, bg)


def _sample_template_counts(color_count: int) -> tuple[int, ...]:
    parts: list[tuple[int, ...]] = []
    if color_count == 2:
        for a in range(2, 8):
            b = 9 - a
            if b >= 2:
                parts.append((a, b))
    elif color_count == 3:
        for a in range(2, 6):
            for b in range(2, 6):
                c = 9 - a - b
                if c >= 2:
                    parts.append((a, b, c))
    else:
        raise ValueError("template must use two or three colors")
    return random.choice(parts)


def _make_template(colors: list[int]) -> np.ndarray:
    counts = _sample_template_counts(len(colors))
    cells: list[int] = []
    for color, count in zip(colors, counts):
        cells.extend([int(color)] * int(count))
    random.shuffle(cells)
    return np.array(cells, dtype=int).reshape(TEMPLATE_SIZE, TEMPLATE_SIZE)


def _frame_patch(color: int, inner_h: int, inner_w: int) -> np.ndarray:
    patch = np.full((inner_h + 2, inner_w + 2), int(color), dtype=int)
    patch[1:-1, 1:-1] = int(BACKGROUND)
    return patch


def _sample_frame_specs(template_colors: list[int]) -> list[dict]:
    frame_count = random.choice(FRAME_COUNT_CHOICES)
    scales = random.sample(list(SCALE_CHOICES), frame_count)
    color_pool = random.sample(template_colors, min(len(template_colors), frame_count))
    while len(color_pool) < frame_count:
        color_pool.append(random.choice(template_colors))
    random.shuffle(color_pool)

    ext_values = [random.randint(0, 2) for _ in range(frame_count)]
    if all(ext == 0 for ext in ext_values) and random.random() < 0.75:
        ext_values[random.randrange(frame_count)] = random.randint(1, 2)

    frames: list[dict] = []
    for scale, color, ext in zip(scales, color_pool, ext_values):
        short_side = TEMPLATE_SIZE * int(scale)
        if ext == 0:
            inner_h = short_side
            inner_w = short_side
        elif random.random() < 0.5:
            inner_h = short_side
            inner_w = short_side + ext
        else:
            inner_h = short_side + ext
            inner_w = short_side
        frames.append(
            {
                "color": int(color),
                "scale": int(scale),
                "inner_h": int(inner_h),
                "inner_w": int(inner_w),
                "outer_h": int(inner_h + 2),
                "outer_w": int(inner_w + 2),
            }
        )
    return frames


def _place_rectangles(
    frames: list[dict], template_box: tuple[int, int, int, int]
) -> list[dict] | None:
    placed: list[dict] = []
    occupied = [template_box]

    for frame in sorted(frames, key=lambda item: (-(item["outer_h"] * item["outer_w"]), -item["scale"])):
        candidates: list[tuple[int, int]] = []
        max_r = CANVAS_SIZE - frame["outer_h"]
        max_c = CANVAS_SIZE - frame["outer_w"]
        for r0 in range(max_r + 1):
            for c0 in range(max_c + 1):
                box = (r0, c0, r0 + frame["outer_h"] - 1, c0 + frame["outer_w"] - 1)
                if c0 < template_box[3] + 2 and r0 < template_box[2] + 2:
                    continue
                if any(_rectangles_overlap(box, other, pad=PLACEMENT_GAP) for other in occupied):
                    continue
                candidates.append((r0, c0))
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)
        frame["bbox"] = (r0, c0, r0 + frame["outer_h"] - 1, c0 + frame["outer_w"] - 1)
        placed.append(frame)
        occupied.append(frame["bbox"])

    return placed


def _noise_candidates(box: tuple[int, int, int, int], shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    r0, c0, r1, c1 = box
    cells: list[tuple[int, int]] = []
    for r in range(max(0, r0 - 1), min(h - 1, r1 + 1) + 1):
        for c in range(max(0, c0 - 1), min(w - 1, c1 + 1) + 1):
            if r0 <= r <= r1 and c0 <= c <= c1:
                continue
            if r in (r0 - 1, r1 + 1) or c in (c0 - 1, c1 + 1):
                cells.append((r, c))
    random.shuffle(cells)
    return cells


def _place_noise(grid: np.ndarray, frames: list[dict]) -> list[tuple[int, int]]:
    noise_count = random.choice((1, 2))
    chosen: list[tuple[int, int]] = []
    frame_order = random.sample(frames, len(frames))
    while len(frame_order) < noise_count:
        frame_order.append(random.choice(frames))

    for frame in frame_order[:noise_count]:
        for cell in _noise_candidates(frame["bbox"], grid.shape):
            if grid[cell] != int(BACKGROUND):
                continue
            if any(max(abs(cell[0] - rr), abs(cell[1] - cc)) <= 1 for rr, cc in chosen):
                continue
            grid[cell] = int(NOISE_COLOR)
            chosen.append(cell)
            break
    assert len(chosen) == noise_count
    return chosen


def _expected_output(
    grid: np.ndarray, template_box: tuple[int, int, int, int], frames: list[dict]
) -> np.ndarray:
    clean = grid.copy()
    clean[clean == int(NOISE_COLOR)] = int(BACKGROUND)
    template = crop_bbox(clean, template_box)
    out = clean.copy()
    for frame in frames:
        scaled = scale_sprite(template, frame["scale"])
        filtered = np.where(scaled == int(frame["color"]), int(frame["color"]), int(BACKGROUND))
        patch = filtered[: frame["inner_h"], : frame["inner_w"]]
        paste(out, patch, frame["bbox"][0] + 1, frame["bbox"][1] + 1, transparent=BACKGROUND)
    return out


def _shift_box(
    box: tuple[int, int, int, int], dr: int, dc: int
) -> tuple[int, int, int, int]:
    return box[0] + dr, box[1] + dc, box[2] + dr, box[3] + dc


def _crop_scene(
    grid: np.ndarray,
    template_box: tuple[int, int, int, int],
    frames: list[dict],
) -> tuple[np.ndarray, tuple[int, int, int, int], list[dict]]:
    scene_box = bbox(grid != int(BACKGROUND))
    assert scene_box is not None
    r0, c0, r1, c1 = scene_box
    top_pad = random.randint(0, min(1, r0))
    left_pad = random.randint(0, min(1, c0))
    bottom_pad = random.randint(0, min(2, grid.shape[0] - 1 - r1))
    right_pad = random.randint(0, min(2, grid.shape[1] - 1 - c1))
    crop_box = (r0 - top_pad, c0 - left_pad, r1 + bottom_pad, c1 + right_pad)
    shifted_template = _shift_box(template_box, -crop_box[0], -crop_box[1])
    shifted_frames: list[dict] = []
    for frame in frames:
        shifted = frame.copy()
        shifted["bbox"] = _shift_box(frame["bbox"], -crop_box[0], -crop_box[1])
        shifted_frames.append(shifted)
    return crop_bbox(grid, crop_box), shifted_template, shifted_frames


def _attempt_generate() -> np.ndarray | None:
    template_color_count = random.choice((2, 3))
    template_colors = random.sample(TEMPLATE_COLOR_POOL, template_color_count)
    template = _make_template(template_colors)

    template_r = random.randint(0, 1)
    template_c = random.randint(0, 1)
    template_box = (
        template_r,
        template_c,
        template_r + TEMPLATE_SIZE - 1,
        template_c + TEMPLATE_SIZE - 1,
    )

    frames = _sample_frame_specs(template_colors)
    placed_frames = _place_rectangles(frames, template_box)
    if placed_frames is None:
        return None

    grid = np.full((CANVAS_SIZE, CANVAS_SIZE), int(BACKGROUND), dtype=int)
    paste(grid, template, template_r, template_c)
    for frame in placed_frames:
        paste(grid, _frame_patch(frame["color"], frame["inner_h"], frame["inner_w"]), frame["bbox"][0], frame["bbox"][1])

    _place_noise(grid, placed_frames)
    cropped, shifted_template_box, shifted_frames = _crop_scene(
        grid, template_box, placed_frames
    )

    expected = _expected_output(cropped, shifted_template_box, shifted_frames)
    try:
        actual = main(cropped)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(actual, cropped):
        return None
    if max(cropped.shape) > 30 or max(actual.shape) > 30:
        return None
    return cropped


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid frame-template ARC input")
