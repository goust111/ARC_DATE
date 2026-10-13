from __future__ import annotations

from pathlib import Path
import importlib.util
import random

import numpy as np


# ---------------------------------------------------------------------------
# arc_utils compatibility layer.
# ---------------------------------------------------------------------------

try:  # prefer direct import in normal runtime
    from arc_utils import (
        bbox_h,
        bbox_w,
        components_mask,
        find_components,
        get_bg,
    )
except Exception:  # pragma: no cover
    _UTIL_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
    spec = importlib.util.spec_from_file_location("arc_utils_fallback", str(_UTIL_PATH))
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load arc_utils fallback module")
    _utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_utils)

    bbox_h = _utils.bbox_h
    bbox_w = _utils.bbox_w
    components_mask = _utils.components_mask
    find_components = _utils.find_components
    get_bg = _utils.get_bg


SIDE_DELTAS = {
    "top": (-1, 0),
    "right": (0, 1),
    "bottom": (1, 0),
    "left": (0, -1),
}
SIDES = tuple(SIDE_DELTAS)


# ---------------------------------------------------------------------------
# Decoding helpers
# ---------------------------------------------------------------------------


def _as_bool_mask(h: int, w: int, coords: np.ndarray, r0: int, c0: int) -> np.ndarray:
    mask = np.zeros((int(h), int(w)), dtype=bool)
    rr = coords[:, 0] - int(r0)
    cc = coords[:, 1] - int(c0)
    mask[rr, cc] = True
    return mask


def _is_hollow_frame(component: dict) -> bool:
    if int(component["area"]) < 4:
        return False

    r0, c0, r1, c1 = component["bbox"]
    h = bbox_h((r0, c0, r1, c1))
    w = bbox_w((r0, c0, r1, c1))
    if h < 3 or w < 3:
        return False

    if int(component["area"]) != 2 * h + 2 * w - 4:
        return False

    mask = _as_bool_mask(h, w, component["coords"], r0, c0)
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return np.array_equal(mask, border)


def _inner_bbox(bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = map(int, bbox)
    return r0 + 1, c0 + 1, r1 - 1, c1 - 1


def _extract_frames(g: np.ndarray, bg: int) -> list[dict]:
    frames: list[dict] = []
    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True):
        if not _is_hollow_frame(comp):
            continue
        frames.append(comp)

    for comp in frames:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        comp["bbox_area"] = (r1 - r0 + 1) * (c1 - c0 + 1)
    return frames


def _touch_sides(inner_bbox: tuple[int, int, int, int], target_coords: np.ndarray) -> list[str]:
    ir0, ic0, ir1, ic1 = inner_bbox
    r = target_coords[:, 0]
    c = target_coords[:, 1]
    sides: list[str] = []
    if np.any(r == ir0):
        sides.append("top")
    if np.any(r == ir1):
        sides.append("bottom")
    if np.any(c == ic0):
        sides.append("left")
    if np.any(c == ic1):
        sides.append("right")
    return sides


def _decode_step(inner_bbox: tuple[int, int, int, int], target_bbox: tuple[int, int, int, int], side: str) -> int:
    ir0, ic0, ir1, ic1 = inner_bbox
    tr0, tc0, tr1, tc1 = target_bbox

    if side == "top":
        return max(0, ir1 - int(tr1))
    if side == "bottom":
        return max(0, int(tr0) - ir0)
    if side == "left":
        return max(0, ic1 - int(tc1))
    if side == "right":
        return max(0, int(tc0) - ic0)
    raise ValueError(f"invalid side {side}")


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode nested hollow rectangles as movement commands and move the innermost sprite by the program sum.

    Input structure
    - The background is the dominant color.
    - One or more axis-aligned one-pixel-thick hollow rectangular frames are nested.
    - Each layer is inside the previous frame and touches exactly one side of the previous frame's inner wall.
    - The innermost remaining object is the target sprite.

    Rule
    - Detect all frame components and sort them from outermost to innermost by bounding-box area.
    - For each frame, read the side where the next inner layer touches the frame's inner wall as direction.
    - Read the gap on the opposite side as step count.
    - Sum all direction vectors and translate the innermost sprite by the resulting vector.
    - Erase all frame pixels and keep only the translated sprite on the same original colors.

    Key insight
    - The frame stack is an explicit program tape: each nested frame encodes one directional move.
    """

    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = int(get_bg(g))

    frames = _extract_frames(g, bg)
    assert frames, "expected at least one hollow frame"

    # Outer to inner order.
    frames.sort(key=lambda c: int(c["bbox_area"]), reverse=True)

    for i in range(len(frames) - 1):
        ir0, ic0, ir1, ic1 = _inner_bbox(frames[i]["bbox"])
        nr0, nc0, nr1, nc1 = frames[i + 1]["bbox"]
        assert ir0 <= nr0 <= nr1 <= ir1, "each inner frame must stay within outer interior"
        assert ic0 <= nc0 <= nc1 <= ic1, "each inner frame must stay within outer interior"

    frame_mask = np.zeros_like(g, dtype=bool)
    for frame in frames:
        rr = frame["coords"][:, 0]
        cc = frame["coords"][:, 1]
        frame_mask[rr, cc] = True

    sprite_mask = (g != bg) & (~frame_mask)
    sprite_components = components_mask(sprite_mask, connectivity=4)
    assert sprite_components, "expected at least one non-frame object"
    if len(sprite_components) != 1:
        raise AssertionError("expected exactly one sprite object")

    sprite_coords = sprite_components[0]
    sprite_bbox = _bbox_from_coords(sprite_coords)

    dr_total = 0
    dc_total = 0
    for i, frame in enumerate(frames):
        ir0, ic0, ir1, ic1 = _inner_bbox(frame["bbox"])

        if i + 1 < len(frames):
            target_coords = frames[i + 1]["coords"]
        else:
            target_coords = sprite_coords

        sides = _touch_sides((ir0, ic0, ir1, ic1), target_coords)
        if len(sides) != 1:
            raise AssertionError("each layer must touch exactly one inner wall side")
        side = sides[0]

        target_bbox = _bbox_from_coords(target_coords)
        steps = int(_decode_step((ir0, ic0, ir1, ic1), target_bbox, side))
        if steps <= 0:
            raise AssertionError("step count must be positive")

        dr, dc = SIDE_DELTAS[side]
        dr_total += dr * steps
        dc_total += dc * steps

    H, W = g.shape
    out = np.full_like(g, bg)
    for rr, cc in sprite_coords:
        r = int(rr) + dr_total
        c = int(cc) + dc_total
        if not (0 <= r < H and 0 <= c < W):
            raise AssertionError("translated sprite leaves grid")
        out[r, c] = int(g[int(rr), int(cc)])

    return out


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

_PALETTE = list(range(1, 10))
_BG = 0
_MIN_SIDE = 1


def _draw_frame(g: np.ndarray, color: int, bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = map(int, bbox)
    g[r0, c0 : c1 + 1] = color
    g[r1, c0 : c1 + 1] = color
    g[r0 : r1 + 1, c0] = color
    g[r0 : r1 + 1, c1] = color


def _draw_sprite(g: np.ndarray, color: int, bbox: tuple[int, int, int, int], side: str) -> None:
    r0, c0, r1, c1 = map(int, bbox)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    variant = random.random()

    if variant < 0.5:
        g[r0 : r1 + 1, c0 : c1 + 1] = color
        return

    shape = np.zeros((h, w), dtype=bool)
    if h >= 2 and w >= 2:
        if side == "top":
            shape[0, :] = True
            if w >= 2 and h >= 3:
                stem_col = random.randint(0, w - 1)
                shape[1 : h - 1, stem_col] = True
            elif w >= 2:
                shape[0, :] = True
        elif side == "bottom":
            shape[-1, :] = True
            if w >= 2 and h >= 3:
                stem_col = random.randint(0, w - 1)
                shape[1 : h - 1, stem_col] = True
        elif side == "left":
            shape[:, 0] = True
            if h >= 2 and w >= 3:
                stem_row = random.randint(1, h - 1)
                shape[stem_row, 1 : w - 1] = True
        else:  # right
            shape[:, -1] = True
            if h >= 2 and w >= 3:
                stem_row = random.randint(1, h - 1)
                shape[stem_row, 1 : w - 1] = True

    if not shape.any():
        g[r0 : r1 + 1, c0 : c1 + 1] = color
        return

    g[r0 : r1 + 1, c0 : c1 + 1][shape] = color


def _place_child(
    parent_inner: tuple[int, int, int, int],
    side: str,
    *,
    min_h: int,
    min_w: int,
) -> tuple[int, int, int, int] | None:
    ir0, ic0, ir1, ic1 = parent_inner
    ih = ir1 - ir0 + 1
    iw = ic1 - ic0 + 1

    if ih < min_h or iw < min_w:
        return None

    for _ in range(80):
        if side == "top":
            max_bottom = ih - min_h
            if max_bottom < 1:
                return None
            bottom_gap = random.randint(1, max_bottom)
            r0 = ir0
            r1 = ir1 - bottom_gap

            if h := (r1 - r0 + 1) < min_h:
                continue

            if iw - 2 < min_w:
                continue
            left_gap = random.randint(1, iw - min_w - 1)
            max_right_gap = iw - min_w - left_gap
            if max_right_gap < 1:
                continue
            right_gap = random.randint(1, max_right_gap)
            c0 = ic0 + left_gap
            c1 = ic1 - right_gap

        elif side == "bottom":
            max_top = ih - min_h
            if max_top < 1:
                return None
            top_gap = random.randint(1, max_top)
            r1 = ir1
            r0 = ir0 + top_gap

            if h := (r1 - r0 + 1) < min_h:
                continue

            if iw - 2 < min_w:
                continue
            left_gap = random.randint(1, iw - min_w - 1)
            max_right_gap = iw - min_w - left_gap
            if max_right_gap < 1:
                continue
            right_gap = random.randint(1, max_right_gap)
            c0 = ic0 + left_gap
            c1 = ic1 - right_gap

        elif side == "left":
            max_right = iw - min_w
            if max_right < 1:
                return None
            right_gap = random.randint(1, max_right)
            c0 = ic0
            c1 = ic1 - right_gap

            if w := (c1 - c0 + 1) < min_w:
                continue

            max_top = ih - min_h
            if max_top < 1:
                continue
            top_gap = random.randint(1, max_top)
            max_bottom = ih - min_h - top_gap
            if max_bottom < 1:
                continue
            bottom_gap = random.randint(1, max_bottom)
            r0 = ir0 + top_gap
            r1 = ir1 - bottom_gap

        else:  # right
            if iw - min_w < 1:
                return None
            left_gap = random.randint(1, iw - min_w)
            c0 = ic0 + left_gap
            c1 = ic1

            if w := (c1 - c0 + 1) < min_w:
                continue

            max_top = ih - min_h
            if max_top < 1:
                continue
            top_gap = random.randint(1, max_top)
            max_bottom = ih - min_h - top_gap
            if max_bottom < 1:
                continue
            bottom_gap = random.randint(1, max_bottom)
            r0 = ir0 + top_gap
            r1 = ir1 - bottom_gap

        if r0 < r1 and c0 < c1:
            return int(r0), int(c0), int(r1), int(c1)

    return None


def _gen_attempt() -> np.ndarray | None:
    for _ in range(300):
        h = random.randint(18, 30)
        w = random.randint(18, 30)
        grid = np.full((h, w), _BG, dtype=int)

        frame_count = random.choices((2, 3, 4), weights=(55, 35, 10), k=1)[0]
        palette = random.sample(_PALETTE, frame_count + 1)
        frame_colors = palette[:frame_count]
        sprite_color = int(palette[-1])

        max_h = h - 4
        max_w = w - 4
        outer_h = random.randint(10, max(10, max_h))
        outer_w = random.randint(10, max(10, max_w))
        if outer_h > h - 2 or outer_w > w - 2:
            outer_h = h - 2
            outer_w = w - 2
        if outer_h < 10 or outer_w < 10:
            continue

        outer_r0 = random.randint(1, h - outer_h - 1)
        outer_c0 = random.randint(1, w - outer_w - 1)
        outer_r1 = outer_r0 + outer_h - 1
        outer_c1 = outer_c0 + outer_w - 1

        outer_bbox = (outer_r0, outer_c0, outer_r1, outer_c1)
        _draw_frame(grid, int(frame_colors[0]), outer_bbox)
        parent_inner = _inner_bbox(outer_bbox)

        child_bboxes: list[tuple[int, int, int, int]] = [outer_bbox]
        for i in range(1, frame_count):
            side = random.choice(SIDES)
            child = _place_child(parent_inner, side, min_h=3, min_w=3)
            if child is None:
                break
            _draw_frame(grid, int(frame_colors[i]), child)
            child_bboxes.append(child)
            parent_inner = _inner_bbox(child)

        if len(child_bboxes) != frame_count:
            continue

        # Draw the sprite in the innermost frame.
        sprite_side = random.choice(SIDES)
        sprite_bbox = _place_child(parent_inner, sprite_side, min_h=1, min_w=1)
        if sprite_bbox is None:
            continue
        _draw_sprite(grid, sprite_color, sprite_bbox, sprite_side)

        try:
            out = main(grid)
        except Exception:
            continue

        if np.array_equal(grid, out):
            # trivial no-op shifts are too weak for this task
            continue

        if out.shape != grid.shape:
            continue

        return grid

    return None


def generate_input() -> np.ndarray:
    for _ in range(1000):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 6 input")
