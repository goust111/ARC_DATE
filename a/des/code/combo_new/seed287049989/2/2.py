from __future__ import annotations

import random

import numpy as np


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _mask_connected(mask: np.ndarray) -> bool:
    return bool(mask.any()) and len(_components(mask)) == 1


def _detect_frames(
    grid: np.ndarray, bg: int
) -> tuple[list[dict[str, int]], np.ndarray]:
    non_bg = grid != bg
    frames: list[dict[str, int]] = []
    frame_mask = np.zeros_like(non_bg, dtype=bool)

    for comp in _components(non_bg):
        r0, c0, r1, c1 = _bbox(comp)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 3 or w < 3:
            continue

        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        comp_mask = np.zeros((h, w), dtype=bool)
        for r, c in comp:
            comp_mask[r - r0, c - c0] = True

        border = np.zeros((h, w), dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        if not np.array_equal(comp_mask, border):
            continue
        if not np.all(sub[1:-1, 1:-1] == bg):
            continue

        side_masks = {
            "top": np.zeros((h, w), dtype=bool),
            "right": np.zeros((h, w), dtype=bool),
            "bottom": np.zeros((h, w), dtype=bool),
            "left": np.zeros((h, w), dtype=bool),
        }
        side_masks["top"][0, :] = True
        side_masks["right"][:, -1] = True
        side_masks["bottom"][-1, :] = True
        side_masks["left"][:, 0] = True

        matches: list[tuple[str, int, int]] = []
        for side, side_mask in side_masks.items():
            band_vals = np.unique(sub[side_mask])
            rest_vals = np.unique(sub[border & ~side_mask])
            if len(band_vals) != 1 or len(rest_vals) != 1:
                continue
            band_color = int(band_vals[0])
            frame_color = int(rest_vals[0])
            if band_color == frame_color or band_color == bg or frame_color == bg:
                continue
            matches.append((side, band_color, frame_color))

        if len(matches) != 1:
            continue

        side, band_color, frame_color = matches[0]
        frames.append(
            {
                "r0": r0,
                "c0": c0,
                "r1": r1,
                "c1": c1,
                "side": {"top": 0, "right": 1, "bottom": 2, "left": 3}[side],
                "band_color": band_color,
                "frame_color": frame_color,
            }
        )
        frame_mask[r0 : r1 + 1, c0 : c1 + 1] |= border

    return frames, frame_mask


def _find_fragments(
    grid: np.ndarray, bg: int, frame_mask: np.ndarray
) -> list[dict[str, object]]:
    remaining = (grid != bg) & (~frame_mask)
    fragments: list[dict[str, object]] = []
    for comp in _components(remaining):
        colors = {int(grid[r, c]) for r, c in comp}
        assert len(colors) == 1, "each loose fragment must be monochrome"
        color = next(iter(colors))
        r0, c0, r1, c1 = _bbox(comp)
        shape = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in comp:
            shape[r - r0, c - c0] = True
        fragments.append(
            {
                "color": color,
                "cells": tuple(comp),
                "r0": r0,
                "c0": c0,
                "shape": shape,
            }
        )
    return fragments


def _placement_origin(frame: dict[str, int], frag_h: int, frag_w: int) -> tuple[int, int]:
    side_code = int(frame["side"])
    ir0 = int(frame["r0"]) + 1
    ic0 = int(frame["c0"]) + 1
    ih = int(frame["r1"]) - int(frame["r0"]) - 1
    iw = int(frame["c1"]) - int(frame["c0"]) - 1
    assert 1 <= frag_h <= ih and 1 <= frag_w <= iw, "fragment must fit inside its frame"

    if side_code == 0:  # top band -> stick to bottom interior wall
        return ir0 + ih - frag_h, ic0 + (iw - frag_w) // 2
    if side_code == 1:  # right band -> stick to left interior wall
        return ir0 + (ih - frag_h) // 2, ic0
    if side_code == 2:  # bottom band -> stick to top interior wall
        return ir0, ic0 + (iw - frag_w) // 2
    return ir0 + (ih - frag_h) // 2, ic0 + iw - frag_w  # left band -> right interior wall


def _solve_from_objects(
    grid: np.ndarray,
    bg: int,
    frames_by_color: dict[int, dict[str, int]],
    fragments_by_color: dict[int, dict[str, object]],
) -> np.ndarray:
    out = np.array(grid, copy=True)
    for frag in fragments_by_color.values():
        for r, c in frag["cells"]:
            out[r, c] = bg

    for color, frame in frames_by_color.items():
        frag = fragments_by_color[color]
        shape = np.asarray(frag["shape"], dtype=bool)
        frag_h, frag_w = map(int, shape.shape)
        pr0, pc0 = _placement_origin(frame, frag_h, frag_w)
        patch = out[pr0 : pr0 + frag_h, pc0 : pc0 + frag_w]
        assert np.all(patch[shape] == bg), "frame interiors must start empty"
        patch[shape] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each loose colored fragment to the frame with the same band color and dock it against the opposite inner wall.

    Input structure:
    - The background is black.
    - Several hollow rectangular frames appear as one-pixel-thick borders; three sides are grey and one whole side is a different color band.
    - Away from the frames, there are several small 4-connected monochrome fragments whose colors match the frames' colored sides.

    Rule:
    - Identify each rectangular frame and read which side is the colored band.
    - Pair every loose fragment with the unique frame whose band has the same color.
    - Remove the fragment from its original location, then place it inside the paired frame so it touches the inner wall opposite the colored band.
    - Along the other axis, center the fragment inside the frame interior; if exact centering is impossible, bias left or up.

    Key insight:
    - The colored side is both an ownership key and a placement instruction: color chooses the frame, and side direction tells which opposite wall the fragment must hug.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"
    bg = _most_common_color(g)

    frames, frame_mask = _detect_frames(g, bg)
    assert frames, "expected at least one colored rectangular frame"

    frames_by_color: dict[int, dict[str, int]] = {}
    for frame in frames:
        color = int(frame["band_color"])
        assert color not in frames_by_color, "frame band colors must be unique"
        frames_by_color[color] = frame

    fragments = _find_fragments(g, bg, frame_mask)
    assert fragments, "expected at least one loose fragment"

    fragments_by_color: dict[int, dict[str, object]] = {}
    for frag in fragments:
        color = int(frag["color"])
        assert color in frames_by_color, "every fragment color must match a frame band"
        assert color not in fragments_by_color, "fragment colors must be unique"
        fragments_by_color[color] = frag

    assert set(frames_by_color) == set(
        fragments_by_color
    ), "frame colors and fragment colors must match exactly"
    return _solve_from_objects(g, bg, frames_by_color, fragments_by_color)


# --------------- generator ---------------

BG = 0
GREY = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
SIDES = ("top", "right", "bottom", "left")
GRID_MIN = 18
GRID_MAX = 30
FRAME_COUNT_RANGE = (2, 4)
INTERIOR_BOUNDS = {
    2: (4, 9),
    3: (4, 8),
    4: (3, 7),
}
PLACEMENT_MARGIN = 1
PLACEMENT_ATTEMPTS = 250
SHAPE_ATTEMPTS = 200
GEN_ATTEMPTS = 400


def _boxes_separated(
    candidate: tuple[int, int, int, int],
    others: list[tuple[int, int, int, int]],
    margin: int = PLACEMENT_MARGIN,
) -> bool:
    r0, c0, r1, c1 = candidate
    for or0, oc0, or1, oc1 in others:
        if not (r1 + margin < or0 or or1 + margin < r0 or c1 + margin < oc0 or oc1 + margin < c0):
            return False
    return True


def _random_shape(inner_h: int, inner_w: int, side: str) -> np.ndarray | None:
    for _ in range(SHAPE_ATTEMPTS):
        if side in ("top", "bottom"):
            box_h = random.randint(1, inner_h - 1)
            box_w = random.randint(1, inner_w)
        else:
            box_h = random.randint(1, inner_h)
            box_w = random.randint(1, inner_w - 1)
        if box_h * box_w < 2:
            continue

        lower = max(2, box_h, box_w)
        target = random.randint(lower, box_h * box_w)
        mask = np.ones((box_h, box_w), dtype=bool)
        cells_left = box_h * box_w

        while cells_left > target:
            candidates = [(r, c) for r in range(box_h) for c in range(box_w) if mask[r, c]]
            random.shuffle(candidates)
            removed = False
            for r, c in candidates:
                if int(mask[r, :].sum()) == 1 or int(mask[:, c].sum()) == 1:
                    continue
                trial = mask.copy()
                trial[r, c] = False
                if not _mask_connected(trial):
                    continue
                mask = trial
                cells_left -= 1
                removed = True
                break
            if not removed:
                break

        if cells_left < lower or not _mask_connected(mask):
            continue
        if random.random() < 0.5:
            mask = np.fliplr(mask)
        if random.random() < 0.5:
            mask = np.flipud(mask)
        return mask
    return None


def _draw_frame(grid: np.ndarray, frame: dict[str, object]) -> None:
    r0 = int(frame["r0"])
    c0 = int(frame["c0"])
    r1 = int(frame["r1"])
    c1 = int(frame["c1"])
    color = int(frame["color"])
    side = str(frame["side"])

    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY
    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY

    if side == "top":
        grid[r0, c0 : c1 + 1] = color
    elif side == "right":
        grid[r0 : r1 + 1, c1] = color
    elif side == "bottom":
        grid[r1, c0 : c1 + 1] = color
    else:
        grid[r0 : r1 + 1, c0] = color


def _paste_shape(grid: np.ndarray, r0: int, c0: int, shape: np.ndarray, color: int) -> None:
    patch = grid[r0 : r0 + shape.shape[0], c0 : c0 + shape.shape[1]]
    patch[shape] = color


def _render_case(
    shape_h: int,
    shape_w: int,
    frames: list[dict[str, object]],
    fragments: list[dict[str, object]],
) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full((shape_h, shape_w), BG, dtype=int)
    out = np.full((shape_h, shape_w), BG, dtype=int)

    for frame in frames:
        _draw_frame(inp, frame)
        _draw_frame(out, frame)

    for frag in fragments:
        _paste_shape(inp, int(frag["r0"]), int(frag["c0"]), np.asarray(frag["shape"], dtype=bool), int(frag["color"]))

    for frame in frames:
        color = int(frame["color"])
        frag = next(f for f in fragments if int(f["color"]) == color)
        shape = np.asarray(frag["shape"], dtype=bool)
        frame_for_solver = {
            "r0": int(frame["r0"]),
            "c0": int(frame["c0"]),
            "r1": int(frame["r1"]),
            "c1": int(frame["c1"]),
            "side": {"top": 0, "right": 1, "bottom": 2, "left": 3}[str(frame["side"])],
        }
        pr0, pc0 = _placement_origin(frame_for_solver, int(shape.shape[0]), int(shape.shape[1]))
        _paste_shape(out, pr0, pc0, shape, color)

    return inp, out


def _gen_attempt() -> np.ndarray | None:
    frame_count = random.randint(*FRAME_COUNT_RANGE)
    side_order = list(SIDES)
    random.shuffle(side_order)
    chosen_sides = side_order[:frame_count]
    colors = random.sample(COLOR_POOL, frame_count)

    min_size = GRID_MIN + max(0, frame_count - 2)
    height = random.randint(min_size, GRID_MAX)
    width = random.randint(min_size, GRID_MAX)
    lo, hi = INTERIOR_BOUNDS[frame_count]

    frames: list[dict[str, object]] = []
    fragments: list[dict[str, object]] = []
    placed_boxes: list[tuple[int, int, int, int]] = []

    for color, side in zip(colors, chosen_sides, strict=False):
        inner_h = random.randint(lo, hi)
        inner_w = random.randint(lo, hi)
        shape = _random_shape(inner_h, inner_w, side)
        if shape is None:
            return None

        frame_h = inner_h + 2
        frame_w = inner_w + 2
        frame_box = None
        for _ in range(PLACEMENT_ATTEMPTS):
            r0 = random.randint(0, height - frame_h)
            c0 = random.randint(0, width - frame_w)
            candidate = (r0, c0, r0 + frame_h - 1, c0 + frame_w - 1)
            if _boxes_separated(candidate, placed_boxes):
                frame_box = candidate
                break
        if frame_box is None:
            return None

        placed_boxes.append(frame_box)
        frames.append(
            {
                "r0": frame_box[0],
                "c0": frame_box[1],
                "r1": frame_box[2],
                "c1": frame_box[3],
                "side": side,
                "color": color,
            }
        )
        fragments.append({"color": color, "shape": shape})

    for frag in fragments:
        shape = np.asarray(frag["shape"], dtype=bool)
        fh, fw = map(int, shape.shape)
        frag_box = None
        for _ in range(PLACEMENT_ATTEMPTS):
            r0 = random.randint(0, height - fh)
            c0 = random.randint(0, width - fw)
            candidate = (r0, c0, r0 + fh - 1, c0 + fw - 1)
            if _boxes_separated(candidate, placed_boxes):
                frag_box = candidate
                break
        if frag_box is None:
            return None
        placed_boxes.append(frag_box)
        frag["r0"] = frag_box[0]
        frag["c0"] = frag_box[1]

    inp, expected = _render_case(height, width, frames, fragments)
    if _most_common_color(inp) != BG:
        return None

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 2")
