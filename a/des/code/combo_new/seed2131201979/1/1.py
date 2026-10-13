from __future__ import annotations

import random

import numpy as np


BG = 0
GRAY = 5
FRAME_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _rect_border(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    border: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        border.add((r0, c))
        border.add((r1, c))
    for r in range(r0, r1 + 1):
        border.add((r, c0))
        border.add((r, c1))
    return border


def _parse_frame(grid: np.ndarray, comp: list[tuple[int, int]]) -> dict[str, object] | None:
    r0, c0, r1, c1 = _bbox(comp)
    if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 3:
        return None

    actual = set(comp)
    border = _rect_border(r0, c0, r1, c1)
    if not actual.issubset(border):
        return None

    missing = border - actual
    if len(missing) != 1:
        return None
    gap = next(iter(missing))
    if gap in {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}:
        return None

    if not np.all(grid[r0 + 1 : r1, c0 + 1 : c1] == BG):
        return None

    gr, gc = gap
    if gr == r0:
        side = "top"
    elif gr == r1:
        side = "bottom"
    elif gc == c0:
        side = "left"
    else:
        side = "right"

    return {
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "gap": (int(gr), int(gc)),
        "side": side,
        "cells": actual,
        "inner": (int(r0 + 1), int(c0 + 1), int(r1 - 1), int(c1 - 1)),
    }


def _find_frames(grid: np.ndarray) -> dict[int, dict[str, object]]:
    frames: dict[int, dict[str, object]] = {}
    for color in np.unique(grid):
        color = int(color)
        if color in (BG, GRAY):
            continue
        candidates = []
        for comp in _components(grid == color):
            if len(comp) <= 1:
                continue
            frame = _parse_frame(grid, comp)
            if frame is not None:
                candidates.append(frame)
        assert len(candidates) <= 1, "each frame color should identify at most one frame"
        if candidates:
            frames[color] = candidates[0]
    assert frames, "expected at least one colored frame"
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each gray shape into the uniquely color-matched open frame and seal the frame's gap.

    Input structure:
    - The background is black.
    - Several distinct non-gray colors form one-pixel-thick rectangular frames whose interiors are black.
    - Each frame has exactly one missing border pixel on one side, and gray 4-connected shapes lie elsewhere.
    - Every gray shape has exactly one 4-neighbor single-pixel colored anchor whose color matches exactly one frame.

    Rule:
    - Identify the frame matched by each shape's adjacent anchor color.
    - Read the frame's missing-pixel side to get the entry direction: top gap means move downward, bottom upward, left rightward, right leftward.
    - Translate the whole gray shape rigidly along that axis until it is fully inside the frame and touches the side opposite the gap.
    - Remove the original gray shape and anchor, and fill the frame's missing border pixel with the frame color.
    - Frames without a matched anchored shape remain unchanged.

    Key insight:
    The anchor color resolves which container to use, while the gap side alone determines both the motion axis and the final wall the payload must dock against.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert g.ndim == 2

    frames = _find_frames(g)

    color_comp_size: dict[tuple[int, int], int] = {}
    for color in frames:
        for comp in _components(g == color):
            size = len(comp)
            for cell in comp:
                color_comp_size[cell] = size

    plans: list[dict[str, object]] = []
    used_colors: set[int] = set()
    for comp in _components(g == GRAY):
        adj: dict[tuple[int, int], int] = {}
        for r, c in comp:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w:
                    color = int(g[nr, nc])
                    if color not in (BG, GRAY):
                        adj[(int(nr), int(nc))] = color
        assert len(adj) == 1, "each gray shape must touch exactly one anchor"
        anchor, color = next(iter(adj.items()))
        assert color in frames, "anchor color must match a frame color"
        assert color_comp_size.get(anchor) == 1, "anchor must be a single pixel, not part of the frame border"
        assert color not in used_colors, "generator expects at most one shape per frame"
        used_colors.add(color)

        frame = frames[color]
        inner_r0, inner_c0, inner_r1, inner_c1 = frame["inner"]  # type: ignore[index]
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        side = frame["side"]  # type: ignore[index]

        if side == "top":
            dr, dc = inner_r1 - max(rows), 0
        elif side == "bottom":
            dr, dc = inner_r0 - min(rows), 0
        elif side == "left":
            dr, dc = 0, inner_c1 - max(cols)
        else:
            dr, dc = 0, inner_c0 - min(cols)

        moved = [(int(r + dr), int(c + dc)) for r, c in comp]
        for mr, mc in moved:
            assert inner_r0 <= mr <= inner_r1 and inner_c0 <= mc <= inner_c1, "moved shape must fit inside the frame interior"

        plans.append(
            {
                "shape": comp,
                "anchor": anchor,
                "color": color,
                "gap": frame["gap"],
                "moved": moved,
            }
        )

    out = g.copy()
    for plan in plans:
        for r, c in plan["shape"]:  # type: ignore[index]
            out[r, c] = BG
        ar, ac = plan["anchor"]  # type: ignore[index]
        out[ar, ac] = BG

    for plan in plans:
        gr, gc = plan["gap"]  # type: ignore[index]
        out[gr, gc] = int(plan["color"])
        for r, c in plan["moved"]:  # type: ignore[index]
            assert int(out[r, c]) == BG, "target interior must be empty"
            out[r, c] = GRAY

    return out


def _normalize_cells(cells: set[tuple[int, int]]) -> np.ndarray:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    norm = {(r - min_r, c - min_c) for r, c in cells}
    max_r = max(r for r, _ in norm)
    max_c = max(c for _, c in norm)
    mask = np.zeros((max_r + 1, max_c + 1), dtype=bool)
    for r, c in norm:
        mask[r, c] = True
    return mask


def _random_shape() -> np.ndarray:
    area = random.randint(1, 5)
    cells = {(0, 0)}
    while len(cells) < area:
        frontier: set[tuple[int, int]] = set()
        for r, c in cells:
            for dr, dc in DIRS4:
                frontier.add((r + dr, c + dc))
        frontier -= cells
        cells.add(random.choice(sorted(frontier)))
    return _normalize_cells(cells)


def _shape_cells(mask: np.ndarray, top: int, left: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for r, c in np.argwhere(mask):
        cells.add((int(top + r), int(left + c)))
    return cells


def _random_gap(side: str, frame_h: int, frame_w: int, target_cells: set[tuple[int, int]]) -> tuple[int, int]:
    rows = sorted({r for r, _ in target_cells})
    cols = sorted({c for _, c in target_cells})
    if side == "top":
        return 0, random.choice(cols)
    if side == "bottom":
        return frame_h - 1, random.choice(cols)
    if side == "left":
        return random.choice(rows), 0
    return random.choice(rows), frame_w - 1


def _anchor_candidates(source_cells: set[tuple[int, int]], frame_h: int, frame_w: int, blocked_dir: tuple[int, int]) -> list[tuple[int, int]]:
    candidates: set[tuple[int, int]] = set()
    for r, c in source_cells:
        for dr, dc in DIRS4:
            if (dr, dc) == blocked_dir:
                continue
            nr, nc = r + dr, c + dc
            if (nr, nc) in source_cells:
                continue
            if 0 <= nr < frame_h and 0 <= nc < frame_w:
                continue
            candidates.add((int(nr), int(nc)))
    return sorted(candidates)


def _build_matched_patch(color: int) -> tuple[np.ndarray, np.ndarray]:
    side = random.choice(["top", "bottom", "left", "right"])
    shape = _random_shape()
    shape_h, shape_w = map(int, shape.shape)

    inner_h = random.randint(shape_h, min(7, shape_h + 3))
    inner_w = random.randint(shape_w, min(7, shape_w + 3))
    frame_h = inner_h + 2
    frame_w = inner_w + 2

    if side in ("top", "bottom"):
        target_r0 = frame_h - shape_h - 1 if side == "top" else 1
        target_c0 = random.randint(1, frame_w - shape_w - 1)
        source_r0 = -random.randint(1, 4) - shape_h + 1 if side == "top" else frame_h - 1 + random.randint(1, 4)
        source_c0 = target_c0
        blocked_dir = (1, 0) if side == "top" else (-1, 0)
    else:
        target_r0 = random.randint(1, frame_h - shape_h - 1)
        target_c0 = frame_w - shape_w - 1 if side == "left" else 1
        source_r0 = target_r0
        source_c0 = -random.randint(1, 4) - shape_w + 1 if side == "left" else frame_w - 1 + random.randint(1, 4)
        blocked_dir = (0, 1) if side == "left" else (0, -1)

    target_cells = _shape_cells(shape, target_r0, target_c0)
    source_cells = _shape_cells(shape, source_r0, source_c0)
    gap = _random_gap(side, frame_h, frame_w, target_cells)

    frame_closed = _rect_border(0, 0, frame_h - 1, frame_w - 1)
    frame_open = set(frame_closed)
    frame_open.remove(gap)

    anchor_options = _anchor_candidates(source_cells, frame_h, frame_w, blocked_dir)
    if not anchor_options:
        anchor_options = _anchor_candidates(source_cells, frame_h, frame_w, (99, 99))
    anchor = random.choice(anchor_options)

    all_cells = set(frame_closed) | target_cells | source_cells | {anchor}
    min_r = min(r for r, _ in all_cells)
    min_c = min(c for _, c in all_cells)
    max_r = max(r for r, _ in all_cells)
    max_c = max(c for _, c in all_cells)
    shift_r = 1 - min_r
    shift_c = 1 - min_c
    patch_h = max_r - min_r + 3
    patch_w = max_c - min_c + 3

    inp = np.full((patch_h, patch_w), BG, dtype=int)
    out = np.full((patch_h, patch_w), BG, dtype=int)

    for r, c in frame_open:
        inp[r + shift_r, c + shift_c] = color
    for r, c in frame_closed:
        out[r + shift_r, c + shift_c] = color
    for r, c in source_cells:
        inp[r + shift_r, c + shift_c] = GRAY
    ar, ac = anchor
    inp[ar + shift_r, ac + shift_c] = color
    for r, c in target_cells:
        out[r + shift_r, c + shift_c] = GRAY

    return inp, out


def _build_unmatched_patch(color: int) -> tuple[np.ndarray, np.ndarray]:
    frame_h = random.randint(4, 9)
    frame_w = random.randint(4, 9)
    side = random.choice(["top", "bottom", "left", "right"])
    if side == "top":
        gap = (0, random.randint(1, frame_w - 2))
    elif side == "bottom":
        gap = (frame_h - 1, random.randint(1, frame_w - 2))
    elif side == "left":
        gap = (random.randint(1, frame_h - 2), 0)
    else:
        gap = (random.randint(1, frame_h - 2), frame_w - 1)

    frame = _rect_border(0, 0, frame_h - 1, frame_w - 1)
    frame.remove(gap)
    patch = np.full((frame_h + 2, frame_w + 2), BG, dtype=int)
    for r, c in frame:
        patch[r + 1, c + 1] = color
    return patch.copy(), patch


def _paste_patch(canvas: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    target = canvas[top : top + h, left : left + w]
    mask = patch != BG
    assert not np.any((target != BG) & mask)
    target[mask] = patch[mask]


def _compose_case() -> tuple[np.ndarray, np.ndarray] | None:
    matched = random.randint(1, 3)
    unmatched = random.randint(0, 2)
    total = matched + unmatched
    if total > len(FRAME_COLORS):
        return None

    colors = random.sample(FRAME_COLORS, total)
    items = [_build_matched_patch(color) for color in colors[:matched]]
    items.extend(_build_unmatched_patch(color) for color in colors[matched:])
    items.sort(key=lambda pair: pair[0].shape[0] * pair[0].shape[1], reverse=True)

    H = W = 30
    inp = np.full((H, W), BG, dtype=int)
    out = np.full((H, W), BG, dtype=int)
    reserved = np.zeros((H, W), dtype=bool)

    for pin, pout in items:
        ph, pw = pin.shape
        candidates = [(r, c) for r in range(H - ph + 1) for c in range(W - pw + 1)]
        random.shuffle(candidates)
        placed = False
        for r, c in candidates:
            rr0 = max(0, r - 1)
            rr1 = min(H, r + ph + 1)
            cc0 = max(0, c - 1)
            cc1 = min(W, c + pw + 1)
            if np.any(reserved[rr0:rr1, cc0:cc1]):
                continue
            _paste_patch(inp, pin, r, c)
            _paste_patch(out, pout, r, c)
            reserved[rr0:rr1, cc0:cc1] = True
            placed = True
            break
        if not placed:
            return None

    used = np.argwhere((inp != BG) | (out != BG))
    if used.size == 0:
        return None
    r0, c0 = map(int, used.min(axis=0))
    r1, c1 = map(int, used.max(axis=0))
    inp = inp[r0 : r1 + 1, c0 : c1 + 1]
    out = out[r0 : r1 + 1, c0 : c1 + 1]
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(400):
        case = _compose_case()
        if case is None:
            continue
        inp, out = case
        if inp.ndim != 2 or out.ndim != 2:
            continue
        if max(inp.shape) > 30 or max(out.shape) > 30:
            continue
        try:
            if np.array_equal(main(inp), out):
                return inp
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid input")
