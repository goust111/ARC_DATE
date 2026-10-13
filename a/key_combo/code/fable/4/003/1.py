from __future__ import annotations

import random

import numpy as np


FRAME_COUNT = 4
ARC_COLORS = tuple(range(10))
PANEL_SIZES = (5, 6)
MAX_GENERATION_ATTEMPTS = 500
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _split_panels(g: np.ndarray) -> tuple[list[np.ndarray], int]:
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    bg = int(g[0, 0])
    height, width = map(int, g.shape)

    separator_rows = [
        r
        for r in range(height)
        if np.all(g[r, :] == int(g[r, 0])) and int(g[r, 0]) != bg
    ]
    assert len(separator_rows) == FRAME_COUNT - 1, "expected three solid separator rows"

    panel_size = int(separator_rows[0])
    assert panel_size > 0, "panel size must be positive"
    assert width == panel_size, "panels must be square"
    assert height == FRAME_COUNT * panel_size + (FRAME_COUNT - 1), "unexpected stack height"

    expected_separators = [panel_size + i * (panel_size + 1) for i in range(FRAME_COUNT - 1)]
    assert separator_rows == expected_separators, "separator rows must evenly split four panels"

    panels: list[np.ndarray] = []
    row0 = 0
    for sep in separator_rows:
        panel = g[row0:sep, :]
        assert panel.shape == (panel_size, panel_size), "all panels must be same square size"
        panels.append(panel.copy())
        row0 = sep + 1

    panel = g[row0:, :]
    assert panel.shape == (panel_size, panel_size), "all panels must be same square size"
    panels.append(panel.copy())
    return panels, bg


def _restore_first_orientation(panels: list[np.ndarray], bg: int) -> np.ndarray:
    panel_size = int(panels[0].shape[0])
    restored = np.full((panel_size, panel_size), bg, dtype=int)
    filled = np.zeros((panel_size, panel_size), dtype=bool)
    aligned_masks: list[np.ndarray] = []

    for frame_idx, panel in enumerate(panels):
        aligned = np.rot90(panel, frame_idx).copy()
        mask = aligned != bg
        assert np.any(mask), "each frame must show part of the sprite"
        conflict = filled & mask & (restored != aligned)
        assert not np.any(conflict), "registered sprite pixels must agree in color"
        restored[mask] = aligned[mask]
        filled |= mask
        aligned_masks.append(mask)

    assert np.any(filled), "restored sprite must be non-empty"
    for mask in aligned_masks:
        assert np.any(filled & ~mask), "every observed frame must be missing part of the sprite"
    return restored


def main(grid: np.ndarray) -> np.ndarray:
    """
    Register four occluded rotation frames, merge the visible pixels, and return the complete next frame.

    Input structure:
    - The input is a vertical stack of four equal square panels separated by three solid full-width rows.
    - The panel background is the shared color found at the whole grid's upper-left corner.
    - The four panels show one non-symmetric foreground sprite rotated clockwise by 0, 90, 180, and 270 degrees.
    - Each panel is missing a different non-empty subset of the sprite pixels.

    Rule:
    - Rotate frame k counterclockwise by (k - 1) quarter turns to register every panel to the first-frame orientation.
    - Take the union of all non-background pixels after registration, requiring overlapping visible pixels to agree in color.
    - Return that completed sprite as one panel; this is the fifth frame because the fourth frame's next clockwise turn completes the cycle.

    Key insight:
    - The panels are not separate objects and no single frame is complete; the complete answer appears only after rotational registration and cross-frame union.
    """
    g = np.asarray(grid, dtype=int)
    panels, bg = _split_panels(g)
    return _restore_first_orientation(panels, bg)


# --------------- generator ---------------


def _rotate_cw(a: np.ndarray, turns: int) -> np.ndarray:
    return np.rot90(a, -int(turns)).copy()


def _foreground_mask(a: np.ndarray, bg: int) -> np.ndarray:
    return a != int(bg)


def _is_rotation_asymmetric(sprite: np.ndarray, bg: int) -> bool:
    mask = _foreground_mask(sprite, bg)
    return all(not np.array_equal(mask, np.rot90(mask, k)) for k in (1, 2, 3))


def _has_solid_foreground_row_in_any_frame(sprite: np.ndarray, bg: int) -> bool:
    for turns in range(FRAME_COUNT):
        frame = _rotate_cw(sprite, turns)
        for r in range(frame.shape[0]):
            if int(frame[r, 0]) != bg and np.all(frame[r, :] == int(frame[r, 0])):
                return True
    return False


def _connected_coords(size: int, count: int) -> list[tuple[int, int]] | None:
    start_choices = [(r, c) for r in range(size) for c in range(size) if (r, c) != (0, 0)]
    coords = {random.choice(start_choices)}
    misses = 0
    while len(coords) < count and misses < 200:
        r, c = random.choice(tuple(coords))
        nbrs = list(NEIGHBORS4)
        random.shuffle(nbrs)
        added = False
        for dr, dc in nbrs:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < size and 0 <= nc < size and (nr, nc) != (0, 0) and (nr, nc) not in coords:
                coords.add((nr, nc))
                added = True
                break
        if not added:
            misses += 1
    if len(coords) != count:
        return None
    return sorted(coords)


def _paint_sprite(size: int, coords: list[tuple[int, int]], bg: int, sep_color: int) -> np.ndarray:
    palette = [c for c in ARC_COLORS if c not in {bg, sep_color}]
    color_count = random.randint(1, min(3, len(palette), len(coords)))
    colors = random.sample(palette, color_count)

    shuffled = coords[:]
    random.shuffle(shuffled)
    assignments: dict[tuple[int, int], int] = {}
    for coord, color in zip(shuffled[:color_count], colors):
        assignments[coord] = color
    for coord in shuffled[color_count:]:
        assignments[coord] = random.choice(colors)

    sprite = np.full((size, size), bg, dtype=int)
    for r, c in coords:
        sprite[r, c] = assignments[(r, c)]
    return sprite


def _make_sprite(size: int, bg: int, sep_color: int) -> np.ndarray | None:
    min_cells = 8 if size == 5 else 9
    max_cells = 12 if size == 5 else 15
    count = random.randint(min_cells, max_cells)
    coords = _connected_coords(size, count)
    if coords is None:
        return None
    sprite = _paint_sprite(size, coords, bg, sep_color)

    if int(sprite[0, 0]) != bg:
        return None
    if not _is_rotation_asymmetric(sprite, bg):
        return None
    if _has_solid_foreground_row_in_any_frame(sprite, bg):
        return None
    return sprite


def _hidden_sets(sprite: np.ndarray, bg: int) -> list[set[tuple[int, int]]]:
    coords = [(int(r), int(c)) for r, c in np.argwhere(sprite != bg)]
    random.shuffle(coords)
    hidden = [{coords[i]} for i in range(FRAME_COUNT)]

    remaining = coords[FRAME_COUNT:]
    max_extra = min(len(remaining), max(1, len(coords) // 3))
    extra_count = random.randint(0, max_extra)
    for coord in random.sample(remaining, extra_count):
        hidden[random.randrange(FRAME_COUNT)].add(coord)
    return hidden


def _apply_hidden(sprite: np.ndarray, hidden: set[tuple[int, int]], bg: int) -> np.ndarray:
    out = sprite.copy()
    for r, c in hidden:
        out[r, c] = bg
    return out


def _stack_frames(frames: list[np.ndarray], sep_color: int) -> np.ndarray:
    rows: list[np.ndarray] = []
    sep = np.full((1, frames[0].shape[1]), int(sep_color), dtype=int)
    for idx, frame in enumerate(frames):
        rows.append(frame)
        if idx != len(frames) - 1:
            rows.append(sep.copy())
    return np.vstack(rows)


def _raw_unregistered_union(frames: list[np.ndarray], bg: int) -> np.ndarray:
    out = np.full_like(frames[0], bg)
    for frame in frames:
        mask = frame != bg
        empty = out == bg
        out[mask & empty] = frame[mask & empty]
    return out


def _build_attempt() -> np.ndarray | None:
    size = random.choice(PANEL_SIZES)
    bg = random.randrange(10)
    sep_color = random.choice([c for c in ARC_COLORS if c != bg])

    sprite = _make_sprite(size, bg, sep_color)
    if sprite is None:
        return None

    hidden = _hidden_sets(sprite, bg)
    if len({frozenset(part) for part in hidden}) != FRAME_COUNT:
        return None

    aligned_incomplete = [_apply_hidden(sprite, part, bg) for part in hidden]
    frames = [_rotate_cw(aligned_incomplete[idx], idx) for idx in range(FRAME_COUNT)]

    # Anti-shortcut checks: copying, advancing one incomplete frame, or raw union cannot solve it.
    if any(np.array_equal(frame, sprite) for frame in frames):
        return None
    if np.array_equal(_rotate_cw(frames[-1], 1), sprite):
        return None
    if np.array_equal(_raw_unregistered_union(frames, bg), sprite):
        return None

    inp = _stack_frames(frames, sep_color)
    try:
        solved = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(solved, sprite):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid fable/4/003/1 input")
