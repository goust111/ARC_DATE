from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
START_COLOR = 1
END_COLOR = 2
SPRITE_COLORS = tuple(range(3, 10))
GEN_MAX_ATTEMPTS = 500


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q: deque[tuple[int, int]] = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _crop_sprite(grid: np.ndarray, sprite_mask: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    box = _bbox(sprite_mask)
    assert box is not None
    r0, c0, r1, c1 = box
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), BG, dtype=int)
    submask = sprite_mask[r0 : r1 + 1, c0 : c1 + 1]
    patch[submask] = grid[r0 : r1 + 1, c0 : c1 + 1][submask]
    return patch, box


def _paste(dst: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    rr0 = max(0, top)
    cc0 = max(0, left)
    rr1 = min(dst.shape[0], top + h)
    cc1 = min(dst.shape[1], left + w)
    if rr0 >= rr1 or cc0 >= cc1:
        return
    pr0 = rr0 - top
    pc0 = cc0 - left
    view = patch[pr0 : pr0 + (rr1 - rr0), pc0 : pc0 + (cc1 - cc0)]
    mask = view != BG
    dst[rr0:rr1, cc0:cc1][mask] = view[mask]


def _tri(n: int) -> int:
    return n * (n + 1) // 2


def _intersects(shape: tuple[int, int], top: int, left: int, h: int, w: int) -> bool:
    return top < shape[0] and top + h > 0 and left < shape[1] and left + w > 0


def _parse_instance(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    start_pts = np.argwhere(g == START_COLOR)
    end_pts = np.argwhere(g == END_COLOR)
    assert len(start_pts) == 1 and len(end_pts) == 1

    start = tuple(map(int, start_pts[0]))
    end = tuple(map(int, end_pts[0]))
    vector = (end[0] - start[0], end[1] - start[1])
    assert (vector[0] == 0) ^ (vector[1] == 0)

    sprite_mask = (g != BG) & (g != START_COLOR) & (g != END_COLOR)
    comps = _components(sprite_mask)
    assert len(comps) == 1
    sprite_only = np.zeros_like(g, dtype=bool)
    for r, c in comps[0]:
        sprite_only[r, c] = True

    patch, box = _crop_sprite(g, sprite_only)
    colors = set(map(int, patch[patch != BG].ravel()))
    assert len(colors) >= 2
    return {"grid": g, "start": start, "end": end, "vector": vector, "sprite_mask": sprite_only, "patch": patch, "bbox": box}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy the sprite into an accelerating footprint trail defined by the two marker pixels.

    Input structure:
    - The background is black.
    - There is one isolated start marker pixel and one isolated end marker pixel of fixed colors.
    - All remaining foreground cells form one connected multicolor sprite, separated from the markers.

    Rule:
    - The vector from the start marker to the end marker is a horizontal or vertical glide vector v.
    - Keep the original sprite as step 0. For step i >= 1, place a copy at displacement (1+2+...+i) * v from the original sprite.
    - Odd-numbered steps are mirrored across the sprite's own mid-axis parallel to v before placement; even-numbered steps stay unchanged.
    - Continue until the next copy no longer intersects the canvas. Clip out-of-bounds parts, and keep both marker pixels unchanged.

    Key insight:
    - The two single pixels are not endpoints to connect; together they encode both the direction/sign and the stride length of an accelerating glide-reflection sequence.
    """
    parsed = _parse_instance(np.asarray(grid, dtype=int))
    g = parsed["grid"]
    patch = parsed["patch"]
    r0, c0, _r1, _c1 = parsed["bbox"]
    dr, dc = parsed["vector"]
    odd_patch = np.flipud(patch) if dr == 0 else np.fliplr(patch)

    out = g.copy()
    step = 1
    while True:
        factor = _tri(step)
        top = r0 + factor * dr
        left = c0 + factor * dc
        if not _intersects(out.shape, top, left, patch.shape[0], patch.shape[1]):
            break
        _paste(out, odd_patch if step % 2 == 1 else patch, top, left)
        step += 1

    sr, sc = parsed["start"]
    er, ec = parsed["end"]
    out[sr, sc] = START_COLOR
    out[er, ec] = END_COLOR
    return out


def _connected_shape(rng: random.Random, h: int, w: int, area: int) -> set[tuple[int, int]]:
    start = (rng.randrange(h), rng.randrange(w))
    cells = {start}
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        cells.add(rng.choice(frontier))
    return cells


def _random_patch(rng: random.Random, movement_axis: str) -> np.ndarray:
    for _ in range(200):
        h = rng.randint(3, 6)
        w = rng.randint(3, 6)
        area = rng.randint(max(5, (h * w) // 2), h * w - 1)
        cells = _connected_shape(rng, h, w, area)

        patch = np.full((h, w), BG, dtype=int)
        palette_size = rng.randint(2, min(4, len(cells)))
        palette = rng.sample(SPRITE_COLORS, palette_size)
        used_colors = set()
        for cell in cells:
            color = rng.choice(palette)
            used_colors.add(color)
            patch[cell] = color
        if len(used_colors) < 2:
            continue

        box = _bbox(patch != BG)
        assert box is not None
        r0, c0, r1, c1 = box
        patch = patch[r0 : r1 + 1, c0 : c1 + 1]
        if movement_axis == "h":
            if patch.shape[0] < 2 or np.array_equal(patch, np.flipud(patch)):
                continue
        else:
            if patch.shape[1] < 2 or np.array_equal(patch, np.fliplr(patch)):
                continue
        return patch
    raise RuntimeError("failed to generate asymmetric sprite")


def _axis_layouts(span: int, stride: int, last_full_step: int, sign: int, want_partial: bool, min_extent: int) -> list[tuple[int, int]]:
    layouts: list[tuple[int, int]] = []
    next_step = last_full_step + 1
    for extent in range(min_extent, 31):
        for base in range(0, extent - span + 1):
            full_ok = True
            for step in range(last_full_step + 1):
                pos = base + sign * _tri(step) * stride
                if not (0 <= pos and pos + span <= extent):
                    full_ok = False
                    break
            if not full_ok:
                continue
            next_pos = base + sign * _tri(next_step) * stride
            partial = next_pos < extent and next_pos + span > 0 and not (0 <= next_pos and next_pos + span <= extent)
            absent = next_pos >= extent or next_pos + span <= 0
            if want_partial and partial:
                layouts.append((extent, base))
            if not want_partial and absent:
                layouts.append((extent, base))
    return layouts


def _cross_layouts(extent: int, span: int) -> list[tuple[int, int]]:
    layouts: list[tuple[int, int]] = []
    for base in range(1, extent - span):
        marker_lines = list(range(0, base - 1)) + list(range(base + span + 1, extent))
        for marker_line in marker_lines:
            layouts.append((base, marker_line))
    return layouts


def _count_visible_steps(shape: tuple[int, int], box: tuple[int, int, int, int], vector: tuple[int, int], patch_shape: tuple[int, int]) -> tuple[int, int]:
    r0, c0, _r1, _c1 = box
    dr, dc = vector
    full = 0
    any_visible = 0
    step = 0
    while True:
        top = r0 + _tri(step) * dr
        left = c0 + _tri(step) * dc
        visible = _intersects(shape, top, left, patch_shape[0], patch_shape[1])
        if not visible:
            break
        any_visible += 1
        if 0 <= top and 0 <= left and top + patch_shape[0] <= shape[0] and left + patch_shape[1] <= shape[1]:
            full += 1
        step += 1
    return full, any_visible


def _semantic_ok(grid: np.ndarray) -> bool:
    try:
        parsed = _parse_instance(grid)
        patch = parsed["patch"]
        dr, dc = parsed["vector"]
        flipped = np.flipud(patch) if dr == 0 else np.fliplr(patch)
        if np.array_equal(patch, flipped):
            return False

        box = parsed["bbox"]
        full, visible = _count_visible_steps(grid.shape, box, parsed["vector"], patch.shape)
        if full < 3 or visible < 3:
            return False

        stride = abs(dr) + abs(dc)
        span = patch.shape[1] if dr == 0 else patch.shape[0]
        if stride <= span:
            return False

        out = main(grid)
        if out.shape != grid.shape:
            return False
        if out[parsed["start"]] != START_COLOR or out[parsed["end"]] != END_COLOR:
            return False
        return True
    except Exception:
        return False


def _attempt_generate(rng: random.Random) -> np.ndarray | None:
    movement_axis = rng.choice(("h", "v"))
    sign = rng.choice((-1, 1))
    last_full_step = rng.choice((2, 3))
    want_partial = rng.random() < 0.5

    patch = _random_patch(rng, movement_axis)
    move_span = patch.shape[1] if movement_axis == "h" else patch.shape[0]
    stride = rng.randint(move_span + 1, min(move_span + 5, 8))

    axis_layouts = _axis_layouts(move_span, stride, last_full_step, sign, want_partial, min_extent=max(10, move_span + 4))
    if not axis_layouts:
        return None
    axis_extent, axis_base = rng.choice(axis_layouts)

    cross_span = patch.shape[0] if movement_axis == "h" else patch.shape[1]
    cross_extent = rng.randint(max(10, cross_span + 5), min(18, 30))
    cross_layouts = _cross_layouts(cross_extent, cross_span)
    if not cross_layouts:
        return None
    cross_base, marker_line = rng.choice(cross_layouts)

    if movement_axis == "h":
        h, w = cross_extent, axis_extent
        sprite_top, sprite_left = cross_base, axis_base
        marker_cols = [c for c in range(w) if 0 <= c + sign * stride < w]
        if not marker_cols:
            return None
        start_col = rng.choice(marker_cols)
        end_col = start_col + sign * stride
        start = (marker_line, start_col)
        end = (marker_line, end_col)
    else:
        h, w = axis_extent, cross_extent
        sprite_top, sprite_left = axis_base, cross_base
        marker_rows = [r for r in range(h) if 0 <= r + sign * stride < h]
        if not marker_rows:
            return None
        start_row = rng.choice(marker_rows)
        end_row = start_row + sign * stride
        start = (start_row, marker_line)
        end = (end_row, marker_line)

    grid = np.full((h, w), BG, dtype=int)
    _paste(grid, patch, sprite_top, sprite_left)
    grid[start] = START_COLOR
    grid[end] = END_COLOR

    sprite_box = (sprite_top, sprite_left, sprite_top + patch.shape[0] - 1, sprite_left + patch.shape[1] - 1)
    sr0, sc0, sr1, sc1 = sprite_box
    if sr0 - 1 <= start[0] <= sr1 + 1 and sc0 - 1 <= start[1] <= sc1 + 1:
        return None
    if sr0 - 1 <= end[0] <= sr1 + 1 and sc0 - 1 <= end[1] <= sc1 + 1:
        return None
    if abs(start[0] - end[0]) + abs(start[1] - end[1]) <= 1:
        return None
    if not _semantic_ok(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _attempt_generate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid accelerated-footprints input")
