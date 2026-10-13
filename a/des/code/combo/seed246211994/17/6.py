from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


ORIENTATIONS = ("tl", "tr", "bl", "br")
GRID_SIDE_MIN = 12
GRID_SIDE_MAX = 20
SPRITE_COUNT_MIN = 1
SPRITE_COUNT_MAX = 4
ARM_MIN = 2
ARM_MAX = 5
PADDING = 1
OFFSET_VALUES = (-6, -5, -4, -3, -2, 2, 3, 4, 5, 6)
OFFSETS = [(dr, dc) for dr in OFFSET_VALUES for dc in OFFSET_VALUES]
MAX_EXTRA_CONTROLS = 2
MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
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


def _local_l_coords(height: int, width: int, orientation: str) -> list[tuple[int, int]]:
    if orientation == "tl":
        return [(0, c) for c in range(width)] + [(r, 0) for r in range(1, height)]
    if orientation == "tr":
        return [(0, c) for c in range(width)] + [(r, width - 1) for r in range(1, height)]
    if orientation == "bl":
        return [(height - 1, c) for c in range(width)] + [(r, 0) for r in range(height - 1)]
    if orientation == "br":
        return [(height - 1, c) for c in range(width)] + [(r, width - 1) for r in range(height - 1)]
    raise ValueError(f"unknown orientation {orientation}")


def _corner_local(height: int, width: int, orientation: str) -> tuple[int, int]:
    if orientation == "tl":
        return 0, 0
    if orientation == "tr":
        return 0, width - 1
    if orientation == "bl":
        return height - 1, 0
    if orientation == "br":
        return height - 1, width - 1
    raise ValueError(f"unknown orientation {orientation}")


def _sprite_cells(sprite: dict, top: int, left: int) -> list[tuple[int, int, int]]:
    cells: list[tuple[int, int, int]] = []
    marker_local = sprite["marker_local"]
    for rr, cc in sprite["local_coords"]:
        color = sprite["marker_color"] if (rr, cc) == marker_local else sprite["body_color"]
        cells.append((top + rr, left + cc, color))
    return cells


def _sprite_coords(sprite: dict, top: int, left: int) -> list[tuple[int, int]]:
    return [(r, c) for r, c, _color in _sprite_cells(sprite, top, left)]


def _analyze_sprite(grid: np.ndarray, coords: list[tuple[int, int]]) -> dict | None:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 2 or width < 2:
        return None
    if len(coords) != height + width - 1:
        return None

    local = {(r - r0, c - c0) for r, c in coords}
    matched_orientation = None
    for orientation in ORIENTATIONS:
        if local == set(_local_l_coords(height, width, orientation)):
            if matched_orientation is not None:
                return None
            matched_orientation = orientation
    if matched_orientation is None:
        return None

    colors = Counter(int(grid[r, c]) for r, c in coords)
    if len(colors) != 2:
        return None
    marker_colors = [color for color, count in colors.items() if count == 1]
    if len(marker_colors) != 1:
        return None
    marker_color = marker_colors[0]
    body_color = next(color for color, count in colors.items() if count != 1)

    marker_rc = next((r, c) for r, c in coords if int(grid[r, c]) == marker_color)
    corner_r, corner_c = _corner_local(height, width, matched_orientation)
    return {
        "coords": list(coords),
        "pixels": [(r, c, int(grid[r, c])) for r, c in coords],
        "orientation": matched_orientation,
        "height": height,
        "width": width,
        "corner": (r0 + corner_r, c0 + corner_c),
        "marker_color": int(marker_color),
        "body_color": int(body_color),
        "marker_rc": marker_rc,
        "bbox": (r0, c0, r1, c1),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each marked L-shaped sprite to the isolated control pixel with the same color.

    The input has a dominant background, several 4-connected L sprites, and isolated
    singleton control pixels. Each sprite is a one-cell-thick L whose body has one
    color except for one embedded marker pixel of a different color.

    If a singleton control pixel has the same color as a sprite's embedded marker,
    translate that entire sprite so its geometric L corner lands on the control
    pixel. Remove the sprite from its original location and consume that control.
    Sprites without a matching control stay in place. Unused control pixels remain.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    h, w = g.shape
    bg = _get_bg(g)

    sprites: list[dict] = []
    controls: dict[int, tuple[int, int]] = {}
    for comp in _find_components(g != bg):
        if len(comp) == 1:
            r, c = comp[0]
            color = int(g[r, c])
            assert color not in controls, "control colors must be unique"
            controls[color] = (r, c)
            continue

        sprite = _analyze_sprite(g, comp)
        assert sprite is not None, "non-singleton component must be a valid L sprite"
        sprites.append(sprite)

    assert sprites, "expected at least one sprite"

    out = g.copy()
    moves: list[tuple[dict, tuple[int, int]]] = []
    for sprite in sprites:
        target = controls.get(sprite["marker_color"])
        if target is not None:
            moves.append((sprite, target))

    for sprite, _target in moves:
        for r, c in sprite["coords"]:
            out[r, c] = bg
    for _sprite, (r, c) in moves:
        out[r, c] = bg

    for sprite, target in moves:
        tr, tc = target
        cr, cc = sprite["corner"]
        dr, dc = tr - cr, tc - cc
        for r, c, color in sprite["pixels"]:
            nr, nc = r + dr, c + dc
            assert 0 <= nr < h and 0 <= nc < w, "translated sprite must stay in bounds"
            assert out[nr, nc] == bg, "translated sprites must land on empty cells"
            out[nr, nc] = color

    return out


def _make_sprite(body_color: int, marker_color: int) -> dict:
    height = random.randint(ARM_MIN, ARM_MAX)
    width = random.randint(ARM_MIN, ARM_MAX)
    orientation = random.choice(ORIENTATIONS)
    local_coords = _local_l_coords(height, width, orientation)
    marker_local = random.choice(local_coords)
    return {
        "body_color": int(body_color),
        "marker_color": int(marker_color),
        "height": height,
        "width": width,
        "orientation": orientation,
        "local_coords": list(local_coords),
        "marker_local": marker_local,
        "corner_local": _corner_local(height, width, orientation),
    }


def _mark_reserved(reserved: np.ndarray, coords: list[tuple[int, int]]) -> None:
    h, w = reserved.shape
    for r, c in coords:
        r0 = max(0, r - PADDING)
        r1 = min(h, r + PADDING + 1)
        c0 = max(0, c - PADDING)
        c1 = min(w, c + PADDING + 1)
        reserved[r0:r1, c0:c1] = True


def _coords_fit(coords: list[tuple[int, int]], shape: tuple[int, int]) -> bool:
    h, w = shape
    return all(0 <= r < h and 0 <= c < w for r, c in coords)


def _can_place(coords: list[tuple[int, int]], reserved: np.ndarray) -> bool:
    return _coords_fit(coords, reserved.shape) and all(not reserved[r, c] for r, c in coords)


def _is_isolated_from(coords: list[tuple[int, int]], point: tuple[int, int]) -> bool:
    pr, pc = point
    return all(max(abs(r - pr), abs(c - pc)) > 1 for r, c in coords)


def _build_expected(
    rows: int,
    cols: int,
    bg: int,
    sprites: list[dict],
    output_tops: list[tuple[int, int]],
    extra_controls: list[tuple[int, int, int]],
) -> np.ndarray:
    grid = np.full((rows, cols), bg, dtype=int)
    for sprite, (top, left) in zip(sprites, output_tops, strict=False):
        for r, c, color in _sprite_cells(sprite, top, left):
            grid[r, c] = color
    for r, c, color in extra_controls:
        grid[r, c] = color
    return grid


def _place_output_objects(
    rows: int,
    cols: int,
    sprites: list[dict],
    matched: list[bool],
    extra_control_colors: list[int],
) -> tuple[list[tuple[int, int]], list[tuple[int, int, int]]] | None:
    reserved = np.zeros((rows, cols), dtype=bool)
    output_tops: list[tuple[int, int] | None] = [None] * len(sprites)

    order = list(range(len(sprites)))
    random.shuffle(order)
    order.sort(key=lambda idx: sprites[idx]["height"] * sprites[idx]["width"], reverse=True)

    for idx in order:
        sprite = sprites[idx]
        candidates: list[tuple[int, int]] = []
        for top in range(rows - sprite["height"] + 1):
            for left in range(cols - sprite["width"] + 1):
                coords = _sprite_coords(sprite, top, left)
                if not _can_place(coords, reserved):
                    continue
                if matched[idx]:
                    if not any(
                        0 <= top - dr <= rows - sprite["height"]
                        and 0 <= left - dc <= cols - sprite["width"]
                        for dr, dc in OFFSETS
                    ):
                        continue
                candidates.append((top, left))
        if not candidates:
            return None
        top_left = random.choice(candidates)
        output_tops[idx] = top_left
        _mark_reserved(reserved, _sprite_coords(sprite, *top_left))

    extra_controls: list[tuple[int, int, int]] = []
    for color in extra_control_colors:
        free_points = [(r, c) for r in range(rows) for c in range(cols) if not reserved[r, c]]
        if not free_points:
            return None
        r, c = random.choice(free_points)
        extra_controls.append((r, c, int(color)))
        _mark_reserved(reserved, [(r, c)])

    return [pos for pos in output_tops if pos is not None], extra_controls


def _build_input(
    rows: int,
    cols: int,
    bg: int,
    sprites: list[dict],
    output_tops: list[tuple[int, int]],
    matched: list[bool],
    extra_controls: list[tuple[int, int, int]],
) -> np.ndarray | None:
    grid = np.full((rows, cols), bg, dtype=int)
    reserved = np.zeros((rows, cols), dtype=bool)

    for sprite, (top, left), is_matched in zip(sprites, output_tops, matched, strict=False):
        if is_matched:
            continue
        coords = _sprite_coords(sprite, top, left)
        if not _can_place(coords, reserved):
            return None
        for r, c, color in _sprite_cells(sprite, top, left):
            grid[r, c] = color
        _mark_reserved(reserved, coords)

    for r, c, color in extra_controls:
        if reserved[r, c]:
            return None
        grid[r, c] = color
        _mark_reserved(reserved, [(r, c)])

    matched_indices = [idx for idx, flag in enumerate(matched) if flag]
    random.shuffle(matched_indices)
    matched_indices.sort(
        key=lambda idx: sprites[idx]["height"] * sprites[idx]["width"], reverse=True
    )

    for idx in matched_indices:
        sprite = sprites[idx]
        out_top, out_left = output_tops[idx]
        corner_r, corner_c = sprite["corner_local"]
        control = (out_top + corner_r, out_left + corner_c)
        if reserved[control]:
            return None

        offsets = OFFSETS[:]
        random.shuffle(offsets)
        chosen = None
        for dr, dc in offsets:
            in_top = out_top - dr
            in_left = out_left - dc
            coords = _sprite_coords(sprite, in_top, in_left)
            if not _can_place(coords, reserved):
                continue
            if not _is_isolated_from(coords, control):
                continue
            chosen = (in_top, in_left)
            break
        if chosen is None:
            return None

        for r, c, color in _sprite_cells(sprite, *chosen):
            grid[r, c] = color
        _mark_reserved(reserved, _sprite_coords(sprite, *chosen))

        cr, cc = control
        if reserved[cr, cc]:
            return None
        grid[cr, cc] = sprite["marker_color"]
        _mark_reserved(reserved, [control])

    return grid


def _gen_attempt() -> np.ndarray | None:
    rows = random.randint(GRID_SIDE_MIN, GRID_SIDE_MAX)
    cols = random.randint(GRID_SIDE_MIN, GRID_SIDE_MAX)
    sprite_count = random.randint(SPRITE_COUNT_MIN, SPRITE_COUNT_MAX)
    match_count = random.randint(0, sprite_count)

    bg = random.randrange(10)
    extra_cap = min(MAX_EXTRA_CONTROLS, 9 - 2 * sprite_count)
    assert extra_cap >= 0
    extra_min = 1 if match_count == 0 else 0
    extra_count = random.randint(extra_min, extra_cap)

    palette = [color for color in range(10) if color != bg]
    total_needed = 2 * sprite_count + extra_count
    colors = random.sample(palette, total_needed)
    body_colors = colors[:sprite_count]
    marker_colors = colors[sprite_count : 2 * sprite_count]
    extra_control_colors = colors[2 * sprite_count :]

    sprites = [_make_sprite(body, marker) for body, marker in zip(body_colors, marker_colors, strict=False)]

    matched = [False] * sprite_count
    matched_indices = random.sample(range(sprite_count), match_count)
    for idx in matched_indices:
        matched[idx] = True

    placed = _place_output_objects(rows, cols, sprites, matched, extra_control_colors)
    if placed is None:
        return None
    output_tops, extra_controls = placed

    expected = _build_expected(rows, cols, bg, sprites, output_tops, extra_controls)
    inp = _build_input(rows, cols, bg, sprites, output_tops, matched, extra_controls)
    if inp is None:
        return None

    try:
        actual = main(inp)
    except AssertionError:
        return None
    return inp if np.array_equal(actual, expected) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
