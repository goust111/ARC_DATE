from __future__ import annotations

import random
from collections import Counter, deque
from dataclasses import dataclass

import numpy as np


BLACK = 0
PALETTE = tuple(range(1, 10))
DIRECTIONS = ("up", "down", "left", "right")

GRID_H_RANGE = (15, 26)
GRID_W_RANGE = (15, 26)
SPRITE_COUNT_CHOICES = (3, 4, 4)
SPRITE_SIZE_RANGE = (3, 5)
SHIFT_MIN = 2
SIDE_MARGIN = 2
PATH_PAD = 1
GENERATION_RETRIES = 500
PLACEMENT_RETRIES = 200


@dataclass
class SpriteSpec:
    direction: str
    height: int
    width: int
    marker_r: int
    marker_c: int
    body_color: int
    marker_color: int
    input_top: int
    input_left: int
    final_top: int
    final_left: int
    target_color: int = -1


def neighbors4(r: int, c: int):
    """Canonical ARC `(r,c)` 4-neighborhood, matching arc_utils."""
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    """Inclusive bbox `(r0,c0,r1,c1)` for an `(N,2)` coord array."""
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    """4-connected components for a boolean mask as `(N,2)` coord arrays."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + pad < br0
        or br1 + pad < ar0
        or ac1 + pad < bc0
        or bc1 + pad < ac0
    )


def _marker_candidates(height: int, width: int, direction: str) -> list[tuple[int, int]]:
    center_r = (height - 1) / 2.0
    center_c = (width - 1) / 2.0
    out = []
    for r in range(height):
        for c in range(width):
            dr = r - center_r
            dc = c - center_c
            if abs(dr) == abs(dc):
                continue
            if direction == "up" and dr < 0 and abs(dr) > abs(dc):
                out.append((r, c))
            elif direction == "down" and dr > 0 and abs(dr) > abs(dc):
                out.append((r, c))
            elif direction == "left" and dc < 0 and abs(dc) > abs(dr):
                out.append((r, c))
            elif direction == "right" and dc > 0 and abs(dc) > abs(dr):
                out.append((r, c))
    return out


def _input_patch(spec: SpriteSpec) -> np.ndarray:
    patch = np.full((spec.height, spec.width), spec.body_color, dtype=int)
    patch[spec.marker_r, spec.marker_c] = spec.marker_color
    return patch


def _output_patch(spec: SpriteSpec) -> np.ndarray:
    patch = np.full((spec.height, spec.width), spec.target_color, dtype=int)
    patch[spec.marker_r, spec.marker_c] = BLACK
    return patch


def _place_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    grid[top : top + h, left : left + w] = patch


def _side_interval(spec: SpriteSpec) -> tuple[str, int, int]:
    if spec.direction == "up":
        return "top", spec.final_left - 1, spec.final_left + spec.width - 2
    if spec.direction == "down":
        return "bottom", spec.final_left - 1, spec.final_left + spec.width - 2
    if spec.direction == "left":
        return "left", spec.final_top - 1, spec.final_top + spec.height - 2
    return "right", spec.final_top - 1, spec.final_top + spec.height - 2


def _swept_rect(spec: SpriteSpec) -> tuple[int, int, int, int]:
    in_r0 = spec.input_top
    in_c0 = spec.input_left
    in_r1 = in_r0 + spec.height - 1
    in_c1 = in_c0 + spec.width - 1
    out_r0 = spec.final_top
    out_c0 = spec.final_left
    out_r1 = out_r0 + spec.height - 1
    out_c1 = out_c0 + spec.width - 1
    return (
        min(in_r0, out_r0),
        min(in_c0, out_c0),
        max(in_r1, out_r1),
        max(in_c1, out_c1),
    )


def _choose_color(prev_color: int | None, forbidden: set[int]) -> int:
    choices = [c for c in PALETTE if c != prev_color and c not in forbidden]
    if not choices:
        choices = [c for c in PALETTE if c != prev_color]
    return int(random.choice(choices))


def _split_gap(length: int, allow_multi: bool = True) -> list[int]:
    if length <= 0:
        return []
    max_parts = min(3 if allow_multi else 1, length)
    part_count = random.randint(1, max_parts)
    cuts = sorted(random.sample(range(1, length), part_count - 1)) if part_count > 1 else []
    prev = 0
    parts = []
    for cut in cuts + [length]:
        parts.append(cut - prev)
        prev = cut
    return parts


def _paint_side(length: int, requests: list[tuple[int, int, int]], specs: list[SpriteSpec]) -> list[int]:
    requests = sorted(requests)
    segments: list[tuple[int, int, int | None]] = []
    cursor = 0
    for start, end, idx in requests:
        if cursor < start:
            pos = cursor
            for gap in _split_gap(start - cursor, allow_multi=True):
                segments.append((pos, pos + gap - 1, None))
                pos += gap
        segments.append((start, end, idx))
        cursor = end + 1
    if cursor < length:
        pos = cursor
        for gap in _split_gap(length - cursor, allow_multi=True):
            segments.append((pos, pos + gap - 1, None))
            pos += gap

    if len(segments) == 1 and segments[0][2] is None and length >= 2:
        mid = random.randint(1, length - 1)
        segments = [(0, mid - 1, None), (mid, length - 1, None)]

    seq = [BLACK] * length
    prev_color: int | None = None
    for start, end, idx in segments:
        forbidden = set()
        if idx is not None:
            forbidden.add(specs[idx].body_color)
        color = _choose_color(prev_color, forbidden)
        if idx is not None:
            specs[idx].target_color = color
        for pos in range(start, end + 1):
            seq[pos] = color
        prev_color = color
    return seq


def _build_border(shape: tuple[int, int], specs: list[SpriteSpec]) -> np.ndarray:
    h, w = shape
    border = np.zeros((h, w), dtype=int)
    requests = {"top": [], "bottom": [], "left": [], "right": []}
    for idx, spec in enumerate(specs):
        side, start, end = _side_interval(spec)
        requests[side].append((start, end, idx))

    top_seq = _paint_side(w - 2, requests["top"], specs)
    bottom_seq = _paint_side(w - 2, requests["bottom"], specs)
    left_seq = _paint_side(h - 2, requests["left"], specs)
    right_seq = _paint_side(h - 2, requests["right"], specs)

    border[0, 1 : w - 1] = top_seq
    border[h - 1, 1 : w - 1] = bottom_seq
    border[1 : h - 1, 0] = left_seq
    border[1 : h - 1, w - 1] = right_seq
    border[0, 0] = top_seq[0]
    border[0, w - 1] = top_seq[-1]
    border[h - 1, 0] = bottom_seq[0]
    border[h - 1, w - 1] = bottom_seq[-1]
    return border


def _expected_output(grid: np.ndarray, specs: list[SpriteSpec]) -> np.ndarray:
    out = grid.copy()
    for spec in specs:
        patch = _input_patch(spec)
        mask = patch != BLACK
        view = out[spec.input_top : spec.input_top + spec.height, spec.input_left : spec.input_left + spec.width]
        view[mask] = BLACK

    for spec in specs:
        patch = _output_patch(spec)
        mask = patch != BLACK
        view = out[spec.final_top : spec.final_top + spec.height, spec.final_left : spec.final_left + spec.width]
        assert not np.any(view[mask] != BLACK)
        view[mask] = patch[mask]
    return out


def _direction_from_marker(coords: np.ndarray, marker: tuple[int, int]) -> str:
    r0, c0, r1, c1 = bbox_coords(coords)
    center_r = (r0 + r1) / 2.0
    center_c = (c0 + c1) / 2.0
    dr = marker[0] - center_r
    dc = marker[1] - center_c
    if abs(dr) > abs(dc):
        return "up" if dr < 0 else "down"
    if abs(dc) > abs(dr):
        return "left" if dc < 0 else "right"
    raise AssertionError("marker direction is ambiguous")


def _validate_instance(grid: np.ndarray, specs: list[SpriteSpec]) -> bool:
    h, w = grid.shape
    if max(h, w) > 30:
        return False
    if np.any(grid[0, :] == BLACK) or np.any(grid[-1, :] == BLACK) or np.any(grid[:, 0] == BLACK) or np.any(grid[:, -1] == BLACK):
        return False

    swept = [_swept_rect(spec) for spec in specs]
    for i, rect_a in enumerate(swept):
        for j, rect_b in enumerate(swept):
            if i < j and _rects_overlap(rect_a, rect_b, pad=0):
                return False

    mask = grid != BLACK
    comps = components_mask(mask)
    sprite_comps = []
    border_count = 0
    for comp in comps:
        touches_border = np.any(
            (comp[:, 0] == 0) | (comp[:, 0] == h - 1) | (comp[:, 1] == 0) | (comp[:, 1] == w - 1)
        )
        if touches_border:
            border_count += 1
        else:
            sprite_comps.append(comp)
    if border_count != 1 or len(sprite_comps) != len(specs):
        return False

    for comp in sprite_comps:
        vals = [int(grid[r, c]) for r, c in comp]
        counts = Counter(vals)
        if len(counts) != 2 or sorted(counts.values()) != [1, comp.shape[0] - 1]:
            return False
        body_color = counts.most_common(1)[0][0]
        markers = [(int(r), int(c)) for r, c in comp if int(grid[r, c]) != int(body_color)]
        if len(markers) != 1:
            return False
        try:
            _direction_from_marker(comp, markers[0])
        except AssertionError:
            return False

    for spec in specs:
        patch = _output_patch(spec)
        if spec.direction == "up":
            touched = [spec.final_left + c for c in range(spec.width) if patch[0, c] != BLACK]
            colors = {int(grid[0, c]) for c in touched}
        elif spec.direction == "down":
            touched = [spec.final_left + c for c in range(spec.width) if patch[-1, c] != BLACK]
            colors = {int(grid[-1, c]) for c in touched}
        elif spec.direction == "left":
            touched = [spec.final_top + r for r in range(spec.height) if patch[r, 0] != BLACK]
            colors = {int(grid[r, 0]) for r in touched}
        else:
            touched = [spec.final_top + r for r in range(spec.height) if patch[r, -1] != BLACK]
            colors = {int(grid[r, -1]) for r in touched}
        if colors != {spec.target_color}:
            return False

    expected = _expected_output(grid, specs)
    actual = main(grid)
    return np.array_equal(expected, actual)


def _pick_margin_start(limit: int, size: int) -> int:
    lo = SIDE_MARGIN if limit - size - SIDE_MARGIN >= SIDE_MARGIN else 1
    hi = limit - size - lo
    return random.randint(lo, hi)


def _try_place_sprite(direction: str, h: int, w: int, occupied: list[tuple[int, int, int, int]], grid_h: int, grid_w: int) -> tuple[int, int, int, int] | None:
    if direction == "up":
        max_shift = grid_h - h - 3
        if max_shift < SHIFT_MIN:
            return None
        final_top = 1
        final_left = _pick_margin_start(grid_w, w)
        shift = random.randint(SHIFT_MIN, max_shift)
        input_top = final_top + shift
        input_left = final_left
    elif direction == "down":
        final_top = grid_h - h - 1
        max_shift = final_top - 1
        if max_shift < SHIFT_MIN:
            return None
        final_left = _pick_margin_start(grid_w, w)
        shift = random.randint(SHIFT_MIN, max_shift)
        input_top = final_top - shift
        input_left = final_left
    elif direction == "left":
        max_shift = grid_w - w - 3
        if max_shift < SHIFT_MIN:
            return None
        final_left = 1
        final_top = _pick_margin_start(grid_h, h)
        shift = random.randint(SHIFT_MIN, max_shift)
        input_left = final_left + shift
        input_top = final_top
    else:
        final_left = grid_w - w - 1
        max_shift = final_left - 1
        if max_shift < SHIFT_MIN:
            return None
        final_top = _pick_margin_start(grid_h, h)
        shift = random.randint(SHIFT_MIN, max_shift)
        input_left = final_left - shift
        input_top = final_top

    rect = (
        min(input_top, final_top),
        min(input_left, final_left),
        max(input_top + h - 1, final_top + h - 1),
        max(input_left + w - 1, final_left + w - 1),
    )
    if any(_rects_overlap(rect, other, pad=PATH_PAD) for other in occupied):
        return None
    return input_top, input_left, final_top, final_left


def _gen_attempt() -> np.ndarray | None:
    grid_h = random.randint(*GRID_H_RANGE)
    grid_w = random.randint(*GRID_W_RANGE)
    sprite_count = random.choice(SPRITE_COUNT_CHOICES)
    directions = random.sample(DIRECTIONS, k=sprite_count)
    random.shuffle(directions)

    specs: list[SpriteSpec] = []
    occupied: list[tuple[int, int, int, int]] = []

    for direction in directions:
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            h = random.randint(*SPRITE_SIZE_RANGE)
            w = random.randint(*SPRITE_SIZE_RANGE)
            candidates = _marker_candidates(h, w, direction)
            if not candidates:
                continue
            marker_r, marker_c = random.choice(candidates)
            placement = _try_place_sprite(direction, h, w, occupied, grid_h, grid_w)
            if placement is None:
                continue
            body_color = random.choice(PALETTE)
            marker_color = random.choice([c for c in PALETTE if c != body_color])
            input_top, input_left, final_top, final_left = placement
            spec = SpriteSpec(
                direction=direction,
                height=h,
                width=w,
                marker_r=marker_r,
                marker_c=marker_c,
                body_color=body_color,
                marker_color=marker_color,
                input_top=input_top,
                input_left=input_left,
                final_top=final_top,
                final_left=final_left,
            )
            specs.append(spec)
            occupied.append(_swept_rect(spec))
            placed = True
            break
        if not placed:
            return None

    grid = _build_border((grid_h, grid_w), specs)
    for spec in specs:
        if spec.target_color < 0 or spec.target_color == spec.body_color:
            return None
        patch = _input_patch(spec)
        view = grid[spec.input_top : spec.input_top + spec.height, spec.input_left : spec.input_left + spec.width]
        if np.any(view != BLACK):
            return None
        _place_patch(grid, patch, spec.input_top, spec.input_left)

    if not _validate_instance(grid, specs):
        return None
    return grid


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each marked sprite to the border side pointed to by its internal marker, then recolor it from the touched border segment.

    Input structure:
    - The outermost ring is a non-black border made from several continuous color segments.
    - Inside the border, the background is black and each interior sprite is a mostly single-color 4-connected object.
    - Every sprite contains exactly one differently colored marker cell, whose position is uniquely biased toward up, down, left, or right within the sprite bbox.

    Rule:
    - For each interior sprite, read the marker cell's unique bias relative to the sprite bbox center to get the slide direction.
    - Erase the marker cell, then translate the remaining sprite until it sits flush against the inner side of the border in that direction.
    - Recolor every moved sprite cell with the border-segment color it touches; leave the border unchanged and restore the old sprite location to black.

    Key insight:
    - The marker color itself is irrelevant: first decode direction from the marker's position, then use the destination border segment to decode the final color.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    comps = components_mask(g != BLACK)
    sprites: list[np.ndarray] = []
    for comp in comps:
        touches_border = np.any(
            (comp[:, 0] == 0) | (comp[:, 0] == h - 1) | (comp[:, 1] == 0) | (comp[:, 1] == w - 1)
        )
        if not touches_border:
            sprites.append(comp)

    for comp in sprites:
        out[comp[:, 0], comp[:, 1]] = BLACK

    for comp in sprites:
        values = [int(g[r, c]) for r, c in comp]
        counts = Counter(values)
        body_color = int(counts.most_common(1)[0][0])
        marker_cells = [(int(r), int(c)) for r, c in comp if int(g[r, c]) != body_color]
        assert len(marker_cells) == 1, "each sprite must have exactly one marker cell"
        marker = marker_cells[0]
        direction = _direction_from_marker(comp, marker)

        body_coords = np.array([coord for coord in comp if tuple(map(int, coord)) != marker], dtype=int)
        assert body_coords.size > 0, "erasing the marker must leave a non-empty sprite"

        shift_r = 0
        shift_c = 0
        if direction == "up":
            top = int(body_coords[:, 0].min())
            shift_r = 1 - top
            contact_cols = body_coords[body_coords[:, 0] == top, 1]
            border_colors = {int(g[0, int(c)]) for c in contact_cols}
        elif direction == "down":
            bottom = int(body_coords[:, 0].max())
            shift_r = (h - 2) - bottom
            contact_cols = body_coords[body_coords[:, 0] == bottom, 1]
            border_colors = {int(g[h - 1, int(c)]) for c in contact_cols}
        elif direction == "left":
            left = int(body_coords[:, 1].min())
            shift_c = 1 - left
            contact_rows = body_coords[body_coords[:, 1] == left, 0]
            border_colors = {int(g[int(r), 0]) for r in contact_rows}
        else:
            right = int(body_coords[:, 1].max())
            shift_c = (w - 2) - right
            contact_rows = body_coords[body_coords[:, 1] == right, 0]
            border_colors = {int(g[int(r), w - 1]) for r in contact_rows}

        assert len(border_colors) == 1 and BLACK not in border_colors, "touched border cells must belong to one color segment"
        paint_color = int(next(iter(border_colors)))

        target = body_coords + np.array([[shift_r, shift_c]], dtype=int)
        assert np.all((1 <= target[:, 0]) & (target[:, 0] <= h - 2))
        assert np.all((1 <= target[:, 1]) & (target[:, 1] <= w - 2))
        assert np.all(out[target[:, 0], target[:, 1]] == BLACK), "sprite paths are guaranteed not to intersect"
        out[target[:, 0], target[:, 1]] = paint_color

    return out


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 体内方向针 input")
