from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
DIRS = ["left", "right", "top", "bottom"]
CANVAS = 30


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _component_from_seed(mask: np.ndarray, seed: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = mask.shape
    sr, sc = seed
    assert 0 <= sr < h and 0 <= sc < w and mask[sr, sc]
    seen = {seed}
    q = deque([seed])
    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return seen


def _find_frames(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    frames: list[tuple[int, int, int, int]] = []
    for pts in _components(grid == GREY):
        rows = [r for r, _ in pts]
        cols = [c for _, c in pts]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        border = set()
        for c in range(c0, c1 + 1):
            border.add((r0, c))
            border.add((r1, c))
        for r in range(r0 + 1, r1):
            border.add((r, c0))
            border.add((r, c1))
        if set(pts) != border:
            continue
        frames.append((r0, c0, r1, c1))
    frames.sort()
    return frames


def _perp_same_color(grid: np.ndarray, r: int, c: int, side: str, color: int) -> bool:
    h, w = grid.shape
    if side in ("left", "right"):
        checks = ((r - 1, c), (r + 1, c))
    else:
        checks = ((r, c - 1), (r, c + 1))
    return any(0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) == color for rr, cc in checks)


def _frame_payload(
    grid: np.ndarray, frame: tuple[int, int, int, int]
) -> tuple[str, int, tuple[int, int], int, int, int, int]:
    r0, c0, r1, c1 = frame
    side_hits: list[tuple[str, int, list[tuple[int, int]]]] = []
    h, w = grid.shape

    if c0 - 1 >= 0:
        pts = [(r, c0 - 1) for r in range(r0 + 1, r1) if int(grid[r, c0 - 1]) not in (BG, GREY)]
        if pts:
            colors = {int(grid[r, c]) for r, c in pts}
            assert len(colors) == 1
            side_hits.append(("left", colors.pop(), pts))
    if c1 + 1 < w:
        pts = [(r, c1 + 1) for r in range(r0 + 1, r1) if int(grid[r, c1 + 1]) not in (BG, GREY)]
        if pts:
            colors = {int(grid[r, c]) for r, c in pts}
            assert len(colors) == 1
            side_hits.append(("right", colors.pop(), pts))
    if r0 - 1 >= 0:
        pts = [(r0 - 1, c) for c in range(c0 + 1, c1) if int(grid[r0 - 1, c]) not in (BG, GREY)]
        if pts:
            colors = {int(grid[r, c]) for r, c in pts}
            assert len(colors) == 1
            side_hits.append(("top", colors.pop(), pts))
    if r1 + 1 < h:
        pts = [(r1 + 1, c) for c in range(c0 + 1, c1) if int(grid[r1 + 1, c]) not in (BG, GREY)]
        if pts:
            colors = {int(grid[r, c]) for r, c in pts}
            assert len(colors) == 1
            side_hits.append(("bottom", colors.pop(), pts))

    assert len(side_hits) == 1
    side, color, contacts = side_hits[0]
    assert len(contacts) == 1
    contact = contacts[0]

    same_color = grid == color
    comp = _component_from_seed(same_color, contact)

    if side == "left":
        step_dir = (0, -1)
    elif side == "right":
        step_dir = (0, 1)
    elif side == "top":
        step_dir = (-1, 0)
    else:
        step_dir = (1, 0)

    arrow: list[tuple[int, int]] = []
    rr, cc = contact
    while (
        0 <= rr < h
        and 0 <= cc < w
        and int(grid[rr, cc]) == color
        and not _perp_same_color(grid, rr, cc, side, color)
    ):
        arrow.append((rr, cc))
        rr += step_dir[0]
        cc += step_dir[1]

    assert arrow
    arrow_set = set(arrow)
    pattern = comp - arrow_set
    assert pattern

    pr0 = min(r for r, _ in pattern)
    pr1 = max(r for r, _ in pattern)
    pc0 = min(c for _, c in pattern)
    pc1 = max(c for _, c in pattern)
    ph = pr1 - pr0 + 1
    pw = pc1 - pc0 + 1
    assert len(pattern) == ph * pw

    end_r, end_c = arrow[-1]
    attached = [(r, c) for r, c in pattern if abs(r - end_r) + abs(c - end_c) == 1]
    assert len(attached) == 1
    ar, ac = attached[0]
    if side in ("left", "right"):
        cross_offset = ar - pr0
    else:
        cross_offset = ac - pc0

    return side, color, contact, len(arrow), ph, pw, cross_offset


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project each small colored rectangle into its paired frame, repeating by arrow length.

    Input structure:
    - Background is black.
    - Several hollow grey rectangular frames appear on the grid, each with a black interior.
    - Exactly one side of each frame has a same-colored outside payload: a 1-pixel-thick arrow touching the frame wall and a small solid rectangle attached at the arrow's far end.
    - Arrows are horizontal or vertical and indicate inward projection from left, right, top, or bottom.

    Rule:
    - For every frame, read the side of the attached arrow to get the inward direction, and read the arrow length to get the repeat count.
    - Recover the attached solid rectangle, then stamp that rectangle inside the frame starting flush against the corresponding inner wall.
    - Shift each next copy by the rectangle's own width or height along the projection axis.
    - If a copy would cross the opposite inner wall, keep only the part that remains inside the frame; grey walls are never overwritten.

    Key insight:
    - The outside monochrome payload combines two signals: the side of the frame gives the projection direction, while the straight arrow segment length gives how many rectangle-width or rectangle-height steps to place inside.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for frame in _find_frames(g):
        r0, c0, r1, c1 = frame
        side, color, contact, repeat_count, ph, pw, cross_offset = _frame_payload(g, frame)
        ir0, ic0, ir1, ic1 = r0 + 1, c0 + 1, r1 - 1, c1 - 1

        if side == "left":
            base_top = contact[0] - cross_offset
            for k in range(repeat_count):
                left = ic0 + k * pw
                for rr in range(base_top, base_top + ph):
                    for cc in range(left, left + pw):
                        if ir0 <= rr <= ir1 and ic0 <= cc <= ic1 and int(out[rr, cc]) == BG:
                            out[rr, cc] = color
        elif side == "right":
            base_top = contact[0] - cross_offset
            for k in range(repeat_count):
                left = ic1 - pw + 1 - k * pw
                for rr in range(base_top, base_top + ph):
                    for cc in range(left, left + pw):
                        if ir0 <= rr <= ir1 and ic0 <= cc <= ic1 and int(out[rr, cc]) == BG:
                            out[rr, cc] = color
        elif side == "top":
            base_left = contact[1] - cross_offset
            for k in range(repeat_count):
                top = ir0 + k * ph
                for rr in range(top, top + ph):
                    for cc in range(base_left, base_left + pw):
                        if ir0 <= rr <= ir1 and ic0 <= cc <= ic1 and int(out[rr, cc]) == BG:
                            out[rr, cc] = color
        else:
            base_left = contact[1] - cross_offset
            for k in range(repeat_count):
                top = ir1 - ph + 1 - k * ph
                for rr in range(top, top + ph):
                    for cc in range(base_left, base_left + pw):
                        if ir0 <= rr <= ir1 and ic0 <= cc <= ic1 and int(out[rr, cc]) == BG:
                            out[rr, cc] = color

    return out


def _draw_frame(patch: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    patch[r0, c0 : c0 + w] = GREY
    patch[r0 + h - 1, c0 : c0 + w] = GREY
    patch[r0 : r0 + h, c0] = GREY
    patch[r0 : r0 + h, c0 + w - 1] = GREY


def _build_patch(color: int, side: str, truncate: bool) -> np.ndarray | None:
    inner_h = random.randint(4, 6)
    inner_w = random.randint(4, 6)
    frame_h = inner_h + 2
    frame_w = inner_w + 2

    if side in ("left", "right"):
        pattern_h = random.randint(2, 3)
        pattern_w = random.randint(1, 3)
        max_full = inner_w // pattern_w
        if max_full < 1:
            return None
        if truncate:
            arrow_len = max_full + random.randint(1, 2)
        else:
            arrow_len = random.randint(1, max_full)
        top = random.randint(1, inner_h - pattern_h + 1)
        attach = random.randint(0, pattern_h - 1)
        row = top + attach
        patch = np.zeros((frame_h, frame_w + arrow_len + pattern_w), dtype=int)
        if side == "left":
            frame_c0 = pattern_w + arrow_len
            _draw_frame(patch, 0, frame_c0, frame_h, frame_w)
            patch[top : top + pattern_h, 0:pattern_w] = color
            patch[row, pattern_w : pattern_w + arrow_len] = color
        else:
            _draw_frame(patch, 0, 0, frame_h, frame_w)
            arrow_c0 = frame_w
            pattern_c0 = frame_w + arrow_len
            patch[top : top + pattern_h, pattern_c0 : pattern_c0 + pattern_w] = color
            patch[row, arrow_c0 : arrow_c0 + arrow_len] = color
        return patch

    pattern_h = random.randint(1, 3)
    pattern_w = random.randint(2, 3)
    max_full = inner_h // pattern_h
    if max_full < 1:
        return None
    if truncate:
        arrow_len = max_full + random.randint(1, 2)
    else:
        arrow_len = random.randint(1, max_full)
    left = random.randint(1, inner_w - pattern_w + 1)
    attach = random.randint(0, pattern_w - 1)
    col = left + attach
    patch = np.zeros((frame_h + arrow_len + pattern_h, frame_w), dtype=int)
    if side == "top":
        frame_r0 = pattern_h + arrow_len
        _draw_frame(patch, frame_r0, 0, frame_h, frame_w)
        patch[0:pattern_h, left : left + pattern_w] = color
        patch[pattern_h : pattern_h + arrow_len, col] = color
    else:
        _draw_frame(patch, 0, 0, frame_h, frame_w)
        arrow_r0 = frame_h
        pattern_r0 = frame_h + arrow_len
        patch[pattern_r0 : pattern_r0 + pattern_h, left : left + pattern_w] = color
        patch[arrow_r0 : arrow_r0 + arrow_len, col] = color
    return patch


def _place_patches(patches: list[np.ndarray]) -> np.ndarray | None:
    canvas = np.zeros((CANVAS, CANVAS), dtype=int)
    occupied = np.zeros((CANVAS, CANVAS), dtype=bool)
    positions: list[tuple[int, int] | None] = [None] * len(patches)

    order = sorted(range(len(patches)), key=lambda i: patches[i].shape[0] * patches[i].shape[1], reverse=True)
    for idx in order:
        patch = patches[idx]
        ph, pw = patch.shape
        placed = False
        for _ in range(400):
            r0 = random.randint(0, CANVAS - ph)
            c0 = random.randint(0, CANVAS - pw)
            rr0 = max(0, r0 - 1)
            rr1 = min(CANVAS, r0 + ph + 1)
            cc0 = max(0, c0 - 1)
            cc1 = min(CANVAS, c0 + pw + 1)
            if occupied[rr0:rr1, cc0:cc1].any():
                continue
            positions[idx] = (r0, c0)
            occupied[r0 : r0 + ph, c0 : c0 + pw] = True
            placed = True
            break
        if not placed:
            return None

    for patch, pos in zip(patches, positions):
        assert pos is not None
        r0, c0 = pos
        ph, pw = patch.shape
        mask = patch != BG
        canvas[r0 : r0 + ph, c0 : c0 + pw][mask] = patch[mask]

    pts = np.argwhere(canvas != BG)
    if pts.size == 0:
        return None
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)

    crop_r0 = random.randint(0, int(r0))
    crop_c0 = random.randint(0, int(c0))
    crop_r1 = random.randint(int(r1), CANVAS - 1)
    crop_c1 = random.randint(int(c1), CANVAS - 1)
    return canvas[crop_r0 : crop_r1 + 1, crop_c0 : crop_c1 + 1]


def _gen_attempt() -> np.ndarray | None:
    n_objects = random.randint(1, 3)
    colors = random.sample(PALETTE, n_objects)
    sides = random.sample(DIRS, n_objects)

    patches: list[np.ndarray] = []
    for color, side in zip(colors, sides):
        patch = _build_patch(color, side, truncate=(random.random() < 0.5))
        if patch is None:
            return None
        patches.append(patch)

    grid = _place_patches(patches)
    if grid is None:
        return None
    if max(grid.shape) > 30:
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed after 400 attempts")
