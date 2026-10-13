from __future__ import annotations

from collections import Counter
import random

import numpy as np


MARKER_COLOR = 7
TOUCH_SIDES = ("top", "bottom", "left", "right")
OPPOSITE_SIDE = {
    "top": "bottom",
    "bottom": "top",
    "left": "right",
    "right": "left",
}

# --------------- solver ---------------


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _border_cells(box: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = box
    cells: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        cells.add((int(r0), int(c)))
        cells.add((int(r1), int(c)))
    for r in range(r0, r1 + 1):
        cells.add((int(r), int(c0)))
        cells.add((int(r), int(c1)))
    return cells


def _frame_box_from_component(coords: np.ndarray) -> tuple[int, int, int, int] | None:
    box = _bbox(coords)
    r0, c0, r1, c1 = box
    if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 3:
        return None
    actual = {(int(r), int(c)) for r, c in coords}
    if actual != _border_cells(box):
        return None
    return box


def _marker_touch_side(marker_coords: np.ndarray, inner_shape: tuple[int, int]) -> str:
    if marker_coords.shape != (4, 2):
        raise AssertionError("each frame must contain exactly one 2x2 orange marker")
    mr0, mc0, mr1, mc1 = _bbox(marker_coords)
    if mr1 - mr0 + 1 != 2 or mc1 - mc0 + 1 != 2:
        raise AssertionError("marker must have a 2x2 bbox")
    expected = {(int(r), int(c)) for r in range(mr0, mr1 + 1) for c in range(mc0, mc1 + 1)}
    actual = {(int(r), int(c)) for r, c in marker_coords}
    if actual != expected:
        raise AssertionError("marker must be a solid 2x2 block")

    inner_h, inner_w = map(int, inner_shape)
    touches: list[str] = []
    if mr0 == 0:
        touches.append("top")
    if mr1 == inner_h - 1:
        touches.append("bottom")
    if mc0 == 0:
        touches.append("left")
    if mc1 == inner_w - 1:
        touches.append("right")
    if len(touches) != 1:
        raise AssertionError("marker must touch exactly one inner wall")
    return touches[0]


def _detect_frames(grid: np.ndarray) -> tuple[int, list[dict[str, int | tuple[int, int, int, int] | str]]]:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or g.size == 0:
        raise AssertionError("input must be a non-empty 2D grid")

    bg = int(Counter(map(int, g.flat)).most_common(1)[0][0])
    frames: list[dict[str, int | tuple[int, int, int, int] | str]] = []

    for color in sorted(int(c) for c in np.unique(g) if int(c) not in (bg, MARKER_COLOR)):
        for comp in _components(g == color):
            box = _frame_box_from_component(comp)
            if box is None:
                raise AssertionError("every non-background non-marker component must be a hollow rectangle")
            r0, c0, r1, c1 = box
            inner = g[r0 + 1 : r1, c0 + 1 : c1]
            if inner.shape[0] < 4 or inner.shape[1] < 4:
                raise AssertionError("frame interior must be large enough for an unambiguous 2x2 marker")
            if not np.all((inner == bg) | (inner == MARKER_COLOR)):
                raise AssertionError("frame interior must contain only background and the orange marker")
            marker_coords = np.argwhere(inner == MARKER_COLOR)
            touch = _marker_touch_side(marker_coords, inner.shape)
            frames.append({"box": box, "color": int(color), "touch": touch})

    if not frames:
        raise AssertionError("expected at least one frame")
    if int(np.count_nonzero(g == MARKER_COLOR)) != 4 * len(frames):
        raise AssertionError("all orange marker pixels must belong to exactly one frame each")
    frames.sort(key=lambda item: item["box"])  # deterministic order
    return bg, frames


def _mirrored_border_cells(
    box: tuple[int, int, int, int],
    touch: str,
    grid_shape: tuple[int, int],
) -> set[tuple[int, int]]:
    h, w = map(int, grid_shape)
    out: set[tuple[int, int]] = set()
    r0, c0, r1, c1 = box
    for r, c in _border_cells(box):
        if touch == "top":
            rr, cc = 2 * r1 - r, c
        elif touch == "bottom":
            rr, cc = 2 * r0 - r, c
        elif touch == "left":
            rr, cc = r, 2 * c1 - c
        else:
            rr, cc = r, 2 * c0 - c
        if 0 <= rr < h and 0 <= cc < w:
            out.add((int(rr), int(cc)))
    return out


def _render_with_reflections(
    grid: np.ndarray,
    frames: list[dict[str, int | tuple[int, int, int, int] | str]],
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for frame in frames:
        box = frame["box"]
        color = int(frame["color"])
        touch = str(frame["touch"])
        for r, c in _mirrored_border_cells(box, touch, out.shape):
            out[r, c] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each hollow frame by mirroring its border across the side opposite the orange marker.

    Input structure:
    - The background is the most common color.
    - Each colored object is a 1-pixel-thick hollow rectangle.
    - Inside each rectangle there is exactly one solid 2x2 orange marker touching one inner wall.

    Rule:
    - Find which wall the orange marker touches.
    - Mirror only the rectangle border across the opposite outer edge, so the copy unfolds outside the frame.
    - Keep the original orange marker unchanged; the mirrored copy contains only border pixels.
    - Clip mirrored pixels that fall outside the grid.
    """
    g = np.asarray(grid, dtype=int)
    _bg, frames = _detect_frames(g)
    return _render_with_reflections(g, frames)


# --------------- generator ---------------

GRID_MIN = 18
GRID_MAX = 26
FRAME_COUNT_CHOICES = (2, 3, 4)
FRAME_OUTER_MIN = 6
FRAME_OUTER_MAX_SMALL = 7
FRAME_OUTER_MAX_LARGE = 8
INPUT_MARGIN = 1
OUTPUT_MARGIN = 1
PLACEMENT_TRIES = 300
GEN_ATTEMPTS = 500
CLIP_PROB = 0.5
PALETTE = tuple(range(10))
BG_CHOICES = tuple(c for c in PALETTE if c != MARKER_COLOR)


def _boxes_too_close(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    margin: int,
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + margin < br0
        or br1 + margin < ar0
        or ac1 + margin < bc0
        or bc1 + margin < ac0
    )


def _output_box(
    box: tuple[int, int, int, int],
    touch: str,
    grid_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    h, w = map(int, grid_shape)
    r0, c0, r1, c1 = box
    if touch == "top":
        return (int(r0), int(c0), min(int(h - 1), int(2 * r1 - r0)), int(c1))
    if touch == "bottom":
        return (max(0, int(2 * r0 - r1)), int(c0), int(r1), int(c1))
    if touch == "left":
        return (int(r0), int(c0), int(r1), min(int(w - 1), int(2 * c1 - c0)))
    return (int(r0), max(0, int(2 * c0 - c1)), int(r1), int(c1))


def _placement_is_clipped(
    box: tuple[int, int, int, int],
    touch: str,
    grid_shape: tuple[int, int],
) -> bool:
    h, w = map(int, grid_shape)
    r0, c0, r1, c1 = box
    if touch == "top":
        return int(2 * r1 - r0) > h - 1
    if touch == "bottom":
        return int(2 * r0 - r1) < 0
    if touch == "left":
        return int(2 * c1 - c0) > w - 1
    return int(2 * c0 - c1) < 0


def _marker_cells(
    box: tuple[int, int, int, int],
    touch: str,
    offset: int,
) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = box
    inner_h = int(r1 - r0 - 1)
    inner_w = int(c1 - c0 - 1)
    if touch == "top":
        sr = r0 + 1
        sc = c0 + 1 + offset
    elif touch == "bottom":
        sr = r0 + inner_h - 1
        sc = c0 + 1 + offset
    elif touch == "left":
        sr = r0 + 1 + offset
        sc = c0 + 1
    else:
        sr = r0 + 1 + offset
        sc = c0 + inner_w - 1
    return {
        (int(sr), int(sc)),
        (int(sr + 1), int(sc)),
        (int(sr), int(sc + 1)),
        (int(sr + 1), int(sc + 1)),
    }


def _random_marker_offset(rng: random.Random, box: tuple[int, int, int, int], touch: str) -> int:
    r0, c0, r1, c1 = box
    inner_h = int(r1 - r0 - 1)
    inner_w = int(c1 - c0 - 1)
    if touch in ("top", "bottom"):
        choices = list(range(1, inner_w - 2))
    else:
        choices = list(range(1, inner_h - 2))
    if not choices:
        raise AssertionError("frame must be large enough to place a 2x2 marker away from corners")
    return int(rng.choice(choices))


def _choose_frame_count(rng: random.Random, h: int, w: int) -> int:
    choices = [2, 3]
    if h >= 22 and w >= 22:
        choices.append(4)
    return int(rng.choice(choices))


def _sample_specs(
    rng: random.Random,
    grid_shape: tuple[int, int],
    n_frames: int,
    colors: list[int],
) -> list[dict[str, int | str | bool]]:
    h, w = map(int, grid_shape)
    dirs = list(TOUCH_SIDES)
    rng.shuffle(dirs)
    size_hi = FRAME_OUTER_MAX_LARGE if n_frames <= 3 else FRAME_OUTER_MAX_SMALL
    specs: list[dict[str, int | str | bool]] = []
    for idx in range(n_frames):
        outer_h = rng.randint(FRAME_OUTER_MIN, min(size_hi, h - 2))
        outer_w = rng.randint(FRAME_OUTER_MIN, min(size_hi, w - 2))
        specs.append(
            {
                "color": int(colors[idx]),
                "touch": dirs[idx],
                "outer_h": int(outer_h),
                "outer_w": int(outer_w),
                "clip": bool(rng.random() < CLIP_PROB),
            }
        )
    vertical = {"top", "bottom"}
    specs.sort(
        key=lambda spec: (
            -(
                (2 * int(spec["outer_h"]) - 1) * int(spec["outer_w"])
                if str(spec["touch"]) in vertical
                else int(spec["outer_h"]) * (2 * int(spec["outer_w"]) - 1)
            )
        )
    )
    return specs


def _place_frame(
    rng: random.Random,
    spec: dict[str, int | str | bool],
    grid_shape: tuple[int, int],
    placed_input_boxes: list[tuple[int, int, int, int]],
    placed_output_boxes: list[tuple[int, int, int, int]],
) -> dict[str, int | str | tuple[int, int, int, int]] | None:
    h, w = map(int, grid_shape)
    outer_h = int(spec["outer_h"])
    outer_w = int(spec["outer_w"])
    touch = str(spec["touch"])
    want_clip = bool(spec["clip"])

    rows = list(range(0, h - outer_h + 1))
    cols = list(range(0, w - outer_w + 1))
    rng.shuffle(rows)
    rng.shuffle(cols)

    attempts = 0
    for r0 in rows:
        for c0 in cols:
            attempts += 1
            if attempts > PLACEMENT_TRIES:
                return None
            box = (int(r0), int(c0), int(r0 + outer_h - 1), int(c0 + outer_w - 1))
            if _placement_is_clipped(box, touch, grid_shape) != want_clip:
                continue
            out_box = _output_box(box, touch, grid_shape)
            if any(_boxes_too_close(box, prev, INPUT_MARGIN) for prev in placed_input_boxes):
                continue
            if any(_boxes_too_close(out_box, prev, OUTPUT_MARGIN) for prev in placed_output_boxes):
                continue
            offset = _random_marker_offset(rng, box, touch)
            return {
                "box": box,
                "color": int(spec["color"]),
                "touch": touch,
                "marker_offset": int(offset),
            }
    return None


def _paint_input(
    grid_shape: tuple[int, int],
    bg: int,
    frames: list[dict[str, int | str | tuple[int, int, int, int]]],
) -> np.ndarray:
    grid = np.full(grid_shape, int(bg), dtype=int)
    for frame in frames:
        box = frame["box"]
        color = int(frame["color"])
        for r, c in _border_cells(box):
            grid[r, c] = color
        for r, c in _marker_cells(box, str(frame["touch"]), int(frame["marker_offset"])):
            grid[r, c] = MARKER_COLOR
    return grid


def _expected_output(
    grid: np.ndarray,
    frames: list[dict[str, int | str | tuple[int, int, int, int]]],
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for frame in frames:
        box = frame["box"]
        color = int(frame["color"])
        touch = str(frame["touch"])
        for r, c in _mirrored_border_cells(box, touch, out.shape):
            out[r, c] = color
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    grid_h = rng.randint(GRID_MIN, GRID_MAX)
    grid_w = rng.randint(GRID_MIN, GRID_MAX)
    bg = int(rng.choice(BG_CHOICES))

    frame_colors = [c for c in PALETTE if c not in (bg, MARKER_COLOR)]
    n_frames = _choose_frame_count(rng, grid_h, grid_w)
    chosen_colors = rng.sample(frame_colors, n_frames)
    specs = _sample_specs(rng, (grid_h, grid_w), n_frames, chosen_colors)

    placed_frames: list[dict[str, int | str | tuple[int, int, int, int]]] = []
    input_boxes: list[tuple[int, int, int, int]] = []
    output_boxes: list[tuple[int, int, int, int]] = []

    for spec in specs:
        frame = _place_frame(rng, spec, (grid_h, grid_w), input_boxes, output_boxes)
        if frame is None:
            return None
        box = frame["box"]
        touch = str(frame["touch"])
        placed_frames.append(frame)
        input_boxes.append(box)
        output_boxes.append(_output_box(box, touch, (grid_h, grid_w)))

    grid = _paint_input((grid_h, grid_w), bg, placed_frames)
    expected = _expected_output(grid, placed_frames)

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
