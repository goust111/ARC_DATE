from __future__ import annotations

from collections import deque
import hashlib
import random

import numpy as np


BG = 0
PALETTE = (1, 2, 3)
_SEEN_INPUT_SIGS: set[str] = set()
_SEEN_OUTPUT_SIGS: set[str] = set()


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


def _bbox_shape(box: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = box
    return r1 - r0 + 1, c1 - c0 + 1


def _bbox_area(box: tuple[int, int, int, int]) -> int:
    h, w = _bbox_shape(box)
    return h * w


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + pad < br0
        or br1 + pad < ar0
        or ac1 + pad < bc0
        or bc1 + pad < ac0
    )


def _frame_contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    or0, oc0, or1, oc1 = outer
    ir0, ic0, ir1, ic1 = inner
    return or0 < ir0 and oc0 < ic0 and ir1 < or1 and ic1 < oc1


def _coords_to_tight_sprite(grid: np.ndarray, coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    sprite = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    sprite[coords[:, 0] - r0, coords[:, 1] - c0] = grid[coords[:, 0], coords[:, 1]]
    return sprite


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _is_hollow_rect_component(coords: np.ndarray) -> bool:
    box = _bbox_from_coords(coords)
    r0, c0, r1, c1 = box
    h, w = _bbox_shape(box)
    if h < 4 or w < 4:
        return False
    border = set()
    for c in range(c0, c1 + 1):
        border.add((r0, c))
        border.add((r1, c))
    for r in range(r0 + 1, r1):
        border.add((r, c0))
        border.add((r, c1))
    pts = {tuple(map(int, rc)) for rc in coords}
    return pts == border


def _extract_entities(grid: np.ndarray) -> tuple[dict[int, dict], dict[int, dict]]:
    frames: dict[int, dict] = {}
    sprites: dict[int, dict] = {}
    colors = sorted(int(c) for c in np.unique(grid) if int(c) != BG)
    for color in colors:
        comps = _component_coords(grid == color)
        if not comps:
            continue
        frame_comps = [comp for comp in comps if _is_hollow_rect_component(comp)]
        if len(frame_comps) != 1:
            raise ValueError(f"expected exactly one frame for color {color}, got {len(frame_comps)}")
        frame_coords = frame_comps[0]
        sprite_comps = [comp for comp in comps if comp is not frame_coords]
        if len(sprite_comps) != 1:
            raise ValueError(f"expected exactly one sprite for color {color}, got {len(sprite_comps)}")
        sprite_coords = sprite_comps[0]
        frames[color] = {
            "color": color,
            "coords": frame_coords,
            "bbox": _bbox_from_coords(frame_coords),
        }
        sprites[color] = {
            "color": color,
            "coords": sprite_coords,
            "bbox": _bbox_from_coords(sprite_coords),
            "sprite": _coords_to_tight_sprite(grid, sprite_coords),
        }
    return frames, sprites


def _classify_side(frame_box: tuple[int, int, int, int], sprite_box: tuple[int, int, int, int]) -> str:
    fr0, fc0, fr1, fc1 = frame_box
    sr0, sc0, sr1, sc1 = sprite_box
    if sr1 < fr0 and fc0 < sc0 and sc1 < fc1:
        return "above"
    if sr0 > fr1 and fc0 < sc0 and sc1 < fc1:
        return "below"
    if sc1 < fc0 and fr0 < sr0 and sr1 < fr1:
        return "left"
    if sc0 > fc1 and fr0 < sr0 and sr1 < fr1:
        return "right"
    raise ValueError("sprite is not aligned to a unique reflection side")


def _stamp_plan(
    frame_box: tuple[int, int, int, int],
    sprite_box: tuple[int, int, int, int],
    sprite: np.ndarray,
) -> tuple[str, np.ndarray, tuple[int, int, int, int]]:
    side = _classify_side(frame_box, sprite_box)
    fr0, fc0, fr1, fc1 = frame_box
    sr0, sc0, sr1, sc1 = sprite_box
    sh, sw = sprite.shape
    mid_low_r = (fr0 + fr1 + 1) // 2
    mid_high_r = (fr0 + fr1) // 2
    mid_low_c = (fc0 + fc1 + 1) // 2
    mid_high_c = (fc0 + fc1) // 2

    if side == "above":
        gap = fr0 - sr1 - 1
        stamp = np.flipud(sprite)
        top = mid_low_r + gap
        left = sc0
    elif side == "below":
        gap = sr0 - fr1 - 1
        stamp = np.flipud(sprite)
        top = mid_high_r - gap - sh + 1
        left = sc0
    elif side == "left":
        gap = fc0 - sc1 - 1
        stamp = np.fliplr(sprite)
        top = sr0
        left = mid_low_c + gap
    else:
        gap = sc0 - fc1 - 1
        stamp = np.fliplr(sprite)
        top = sr0
        left = mid_high_c - gap - sw + 1

    box = (top, left, top + sh - 1, left + sw - 1)
    if not (
        fr0 < box[0] <= box[2] < fr1
        and fc0 < box[1] <= box[3] < fc1
    ):
        raise ValueError("reflected stamp does not stay inside the frame interior")
    return side, stamp, box


def _draw_frame(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _paint_stamp(out: np.ndarray, stamp: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, _r1, _c1 = box
    mask = stamp != BG
    rr, cc = np.where(mask)
    out[r0 + rr, c0 + cc] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reflect same-colored outside sprites into matching hollow frames, then recolor stamps inside nested frames to the container-frame color.

    Input structure:
    - The grid uses black as background and contains several monochrome hollow rectangular frames.
    - Each frame color also appears in exactly one small 4-connected asymmetric sprite outside the frames.
    - One frame is strictly nested inside another frame's interior, while at least one other frame is independent.

    Rule:
    - Match every sprite to the hollow frame of the same color.
    - Infer the reflection axis from the sprite position: sprites above/below a frame reflect across its horizontal centerline, and sprites left/right of a frame reflect across its vertical centerline.
    - Mirror the sprite into the opposite half of the frame interior, preserving its orthogonal alignment and its gap from the nearest frame side.
    - Keep the original input intact and add the reflected stamp inside the frame.
    - If a frame is nested inside another frame, recolor the entire reflected stamp inside the inner frame to the immediate outer frame's color; independent frames keep the sprite color.

    Key insight:
    - Color identifies which sprite belongs to which frame, but the final stamp color is controlled by spatial containment, so the same reflection geometry yields different colors in nested and non-nested contexts.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    frames, sprites = _extract_entities(g)

    container_of: dict[int, int | None] = {}
    for color, frame in frames.items():
        box = frame["bbox"]
        container = None
        for other_color, other in frames.items():
            if other_color == color:
                continue
            if _frame_contains(other["bbox"], box):
                if container is None or _bbox_area(other["bbox"]) < _bbox_area(frames[container]["bbox"]):
                    container = other_color
        container_of[color] = container

    for color, frame in frames.items():
        sprite = sprites[color]
        _side, stamp, box = _stamp_plan(frame["bbox"], sprite["bbox"], sprite["sprite"])
        final_color = container_of[color] if container_of[color] is not None else color
        _paint_stamp(out, stamp, box, int(final_color))
        _draw_frame(out, frame["bbox"], color)

    return out


def _mask_connected(mask: np.ndarray) -> bool:
    comps = _component_coords(mask.astype(bool))
    return len(comps) == 1


def _random_sprite(rng: random.Random, color: int) -> np.ndarray:
    shapes = [(2, 2), (2, 3), (3, 2), (3, 3)]
    for _ in range(300):
        h, w = rng.choice(shapes)
        target = rng.randint(3, min(5, h * w))
        mask = np.zeros((h, w), dtype=bool)
        start = (rng.randrange(h), rng.randrange(w))
        mask[start] = True
        while int(mask.sum()) < target:
            frontier: list[tuple[int, int]] = []
            for r, c in np.argwhere(mask):
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                        frontier.append((int(nr), int(nc)))
            if not frontier:
                break
            nr, nc = rng.choice(frontier)
            mask[nr, nc] = True

        if not _mask_connected(mask):
            continue
        coords = np.argwhere(mask)
        r0, c0, r1, c1 = _bbox_from_coords(coords)
        mask = mask[r0 : r1 + 1, c0 : c1 + 1]
        if min(mask.shape) < 2 or not (3 <= int(mask.sum()) <= 5):
            continue
        if np.array_equal(mask, np.flipud(mask)):
            continue
        if np.array_equal(mask, np.fliplr(mask)):
            continue
        if np.array_equal(mask, np.rot90(mask, 1)):
            continue
        if np.array_equal(mask, np.rot90(mask, 2)):
            continue
        sprite = np.zeros(mask.shape, dtype=int)
        sprite[mask] = color
        return sprite
    raise RuntimeError("failed to generate an asymmetric sprite")


def _make_outer_inner_boxes(rng: random.Random, h: int, w: int) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    for _ in range(300):
        outer_h = rng.choice((12, 14))
        outer_w = rng.choice((12, 14))
        inner_h = rng.choice((8, 10))
        inner_w = rng.choice((8, 10))
        if inner_h >= outer_h - 2 or inner_w >= outer_w - 2:
            continue
        top_margin = rng.randint(2, outer_h - inner_h - 2)
        left_margin = rng.randint(2, outer_w - inner_w - 2)
        bottom_margin = outer_h - inner_h - top_margin
        right_margin = outer_w - inner_w - left_margin
        if min(top_margin, left_margin, bottom_margin, right_margin) < 2:
            continue
        or0 = rng.randint(2, h - outer_h - 3)
        oc0 = rng.randint(2, w - outer_w - 3)
        outer = (or0, oc0, or0 + outer_h - 1, oc0 + outer_w - 1)
        inner = (
            or0 + top_margin,
            oc0 + left_margin,
            or0 + top_margin + inner_h - 1,
            oc0 + left_margin + inner_w - 1,
        )
        return outer, inner
    raise RuntimeError("failed to place outer/inner frames")


def _make_independent_box(
    rng: random.Random,
    h: int,
    w: int,
    blocked: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int]:
    for _ in range(400):
        fh = rng.choice((8, 10))
        fw = rng.choice((8, 10))
        r0 = rng.randint(2, h - fh - 3)
        c0 = rng.randint(2, w - fw - 3)
        box = (r0, c0, r0 + fh - 1, c0 + fw - 1)
        if any(_boxes_overlap(box, other, pad=2) for other in blocked):
            continue
        return box
    raise RuntimeError("failed to place the independent frame")


def _choose_sprite_plan(
    rng: random.Random,
    grid: np.ndarray,
    frame_box: tuple[int, int, int, int],
    sprite: np.ndarray,
    blocked_input: list[tuple[int, int, int, int]],
    other_frames: list[tuple[int, int, int, int]],
    blocked_output: list[tuple[int, int, int, int]],
    own_containers: list[tuple[int, int, int, int]],
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int], str]:
    h, w = grid.shape
    fr0, fc0, fr1, fc1 = frame_box
    sh, sw = sprite.shape
    sides = ["above", "below", "left", "right"]
    rng.shuffle(sides)
    gaps = [1, 2]
    rng.shuffle(gaps)

    for side in sides:
        if side in ("above", "below"):
            orth_values = list(range(fc0 + 1, fc1 - sw + 1))
        else:
            orth_values = list(range(fr0 + 1, fr1 - sh + 1))
        rng.shuffle(orth_values)
        for gap in gaps:
            for orth in orth_values:
                if side == "above":
                    input_box = (fr0 - gap - sh, orth, fr0 - gap - 1, orth + sw - 1)
                elif side == "below":
                    input_box = (fr1 + gap + 1, orth, fr1 + gap + sh, orth + sw - 1)
                elif side == "left":
                    input_box = (orth, fc0 - gap - sw, orth + sh - 1, fc0 - gap - 1)
                else:
                    input_box = (orth, fc1 + gap + 1, orth + sh - 1, fc1 + gap + sw)

                ir0, ic0, ir1, ic1 = input_box
                if ir0 < 0 or ic0 < 0 or ir1 >= h or ic1 >= w:
                    continue
                if any(_boxes_overlap(input_box, box, pad=1) for box in blocked_input):
                    continue
                rr, cc = np.where(sprite != BG)
                if np.any(grid[ir0 + rr, ic0 + cc] != BG):
                    continue

                colored = sprite.copy()
                sprite_coords = np.argwhere(colored != BG)
                sprite_box = (
                    ir0 + int(sprite_coords[:, 0].min()),
                    ic0 + int(sprite_coords[:, 1].min()),
                    ir0 + int(sprite_coords[:, 0].max()),
                    ic0 + int(sprite_coords[:, 1].max()),
                )
                try:
                    _side, _stamp, output_box = _stamp_plan(frame_box, sprite_box, colored)
                except ValueError:
                    continue
                if _side != side:
                    continue
                bad = False
                for other in other_frames:
                    if other == frame_box or other in own_containers:
                        continue
                    if _boxes_overlap(output_box, other):
                        bad = True
                        break
                if bad or any(_boxes_overlap(output_box, box) for box in blocked_output):
                    continue
                return input_box, output_box, side

    raise RuntimeError("failed to place a valid sprite")


def _place_sprite_on_grid(grid: np.ndarray, sprite: np.ndarray, box: tuple[int, int, int, int]) -> None:
    r0, c0, _r1, _c1 = box
    mask = sprite != BG
    rr, cc = np.where(mask)
    grid[r0 + rr, c0 + cc] = sprite[rr, cc]


def _semantic_check(inp: np.ndarray, out: np.ndarray) -> None:
    frames, sprites = _extract_entities(inp)
    assert len(frames) == 3
    assert len(sprites) == 3

    containers: dict[int, int | None] = {}
    nested = 0
    axis_kinds: set[str] = set()
    gap_values: list[int] = []
    expected = inp.copy()

    for color, frame in frames.items():
        box = frame["bbox"]
        container = None
        for other_color, other in frames.items():
            if other_color == color:
                continue
            if _frame_contains(other["bbox"], box):
                if container is None or _bbox_area(other["bbox"]) < _bbox_area(frames[container]["bbox"]):
                    container = other_color
        containers[color] = container
        nested += int(container is not None)

    assert nested == 1

    for color, frame in frames.items():
        sprite = sprites[color]
        side, stamp, stamp_box = _stamp_plan(frame["bbox"], sprite["bbox"], sprite["sprite"])
        axis_kinds.add("hv"[side in ("left", "right")])
        fr0, fc0, fr1, fc1 = frame["bbox"]
        sr0, sc0, sr1, sc1 = sprite["bbox"]
        if side == "above":
            gap_values.append(fr0 - sr1 - 1)
        elif side == "below":
            gap_values.append(sr0 - fr1 - 1)
        elif side == "left":
            gap_values.append(fc0 - sc1 - 1)
        else:
            gap_values.append(sc0 - fc1 - 1)

        final_color = containers[color] if containers[color] is not None else color
        _paint_stamp(expected, stamp, stamp_box, int(final_color))
        _draw_frame(expected, frame["bbox"], color)

        sub = out[stamp_box[0] : stamp_box[2] + 1, stamp_box[1] : stamp_box[3] + 1]
        stamp_mask = stamp != BG
        assert np.all(sub[stamp_mask] == int(final_color))
        if containers[color] is not None:
            assert int(final_color) != int(color)
        else:
            assert int(final_color) == int(color)

    assert axis_kinds == {"h", "v"}
    assert any(g == 1 for g in gap_values)
    assert any(g > 1 for g in gap_values)
    assert np.array_equal(out, expected)


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(600):
        h = rng.randint(26, 30)
        w = rng.randint(26, 30)
        grid = np.zeros((h, w), dtype=int)
        colors = list(PALETTE)
        rng.shuffle(colors)

        try:
            outer_box, inner_box = _make_outer_inner_boxes(rng, h, w)
            independent_box = _make_independent_box(rng, h, w, [outer_box])
        except RuntimeError:
            continue
        frame_boxes = {
            colors[0]: outer_box,
            colors[1]: inner_box,
            colors[2]: independent_box,
        }

        for color, box in frame_boxes.items():
            _draw_frame(grid, box, color)

        input_sprite_boxes: list[tuple[int, int, int, int]] = []
        output_blocked: list[tuple[int, int, int, int]] = []
        sprite_sides: list[str] = []

        try:
            for color in (colors[0], colors[1], colors[2]):
                sprite = _random_sprite(rng, color)
                own_box = frame_boxes[color]
                if color == colors[1]:
                    own_containers = [outer_box]
                else:
                    own_containers = []
                other_frames = list(frame_boxes.values())
                blocked_input = list(input_sprite_boxes)
                for frame_candidate in other_frames:
                    if frame_candidate == own_box or frame_candidate in own_containers:
                        continue
                    blocked_input.append(frame_candidate)
                input_box, output_box, side = _choose_sprite_plan(
                    rng,
                    grid,
                    own_box,
                    sprite,
                    blocked_input,
                    other_frames,
                    output_blocked,
                    own_containers,
                )
                _place_sprite_on_grid(grid, sprite, input_box)
                input_sprite_boxes.append(input_box)
                output_blocked.append(output_box)
                sprite_sides.append(side)
        except RuntimeError:
            continue

        axis_groups = {side in ("left", "right") for side in sprite_sides}
        if axis_groups != {False, True}:
            continue

        out = main(grid)
        try:
            _semantic_check(grid, out)
        except Exception:
            continue

        inp_sig = hashlib.md5(grid.tobytes() + str(grid.shape).encode()).hexdigest()
        out_sig = hashlib.md5(out.tobytes() + str(out.shape).encode()).hexdigest()
        if inp_sig in _SEEN_INPUT_SIGS or out_sig in _SEEN_OUTPUT_SIGS:
            continue
        _SEEN_INPUT_SIGS.add(inp_sig)
        _SEEN_OUTPUT_SIGS.add(out_sig)
        return grid

    raise RuntimeError("failed to generate a valid ARC instance")
