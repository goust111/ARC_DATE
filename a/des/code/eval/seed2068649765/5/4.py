from __future__ import annotations

import numpy as np


_BG = 0
_FRAME_COLOR = 5
_ROTATE_COLOR = 1
_FLIP_COLOR = 2
_PATTERN_COLORS = (3, 4, 6, 7, 8, 9)
_DIRECTIONS = ("up", "down", "left", "right")
_MAX_GEN_ATTEMPTS = 400

_NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
_NEIGHBORS8 = tuple(
    (dr, dc)
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
    if not (dr == 0 and dc == 0)
)

_ARROW_SHAPES: dict[str, tuple[tuple[int, int], ...]] = {
    "right": ((1, 0), (0, 1), (2, 1)),
    "left": ((0, 0), (2, 0), (1, 1)),
    "up": ((0, 0), (0, 2), (1, 1)),
    "down": ((0, 1), (1, 0), (1, 2)),
}
_ARROW_LOOKUP = {
    frozenset(shape): direction for direction, shape in _ARROW_SHAPES.items()
}


def _components(mask: np.ndarray, connectivity: int) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    neighbors = _NEIGHBORS4 if connectivity == 4 else _NEIGHBORS8
    out: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for dr, dc in neighbors:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(comp)
    return out


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _most_common_color(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[int(np.argmax(counts))])


def _is_rectangular_frame(coords: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return False
    border = {
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if r in (r0, r1) or c in (c0, c1)
    }
    return set(coords) == border


def _find_frame(grid: np.ndarray, bg: int) -> tuple[int, int, int, int]:
    candidates: list[tuple[int, int, int, int, int, int]] = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        for comp in _components(grid == color, connectivity=4):
            if not _is_rectangular_frame(comp):
                continue
            r0, c0, r1, c1 = _bbox(comp)
            area = (r1 - r0 + 1) * (c1 - c0 + 1)
            candidates.append((c0, -area, r0, c0, r1, c1))

    assert candidates, "expected one framed source pattern"
    _, _, r0, c0, r1, c1 = min(candidates)
    return int(r0), int(c0), int(r1), int(c1)


def _decode_arrow(coords: list[tuple[int, int]]) -> str | None:
    if len(coords) != 3:
        return None
    r0, c0, _, _ = _bbox(coords)
    normalized = frozenset((r - r0, c - c0) for r, c in coords)
    return _ARROW_LOOKUP.get(normalized)


def _find_arrows(grid: np.ndarray, bg: int, frame_bbox: tuple[int, int, int, int]) -> list[tuple[int, str]]:
    _, _, _, frame_c1 = frame_bbox
    right_region = grid[:, frame_c1 + 1 :]
    arrow_specs: list[tuple[int, int, str]] = []

    for color in np.unique(right_region):
        color = int(color)
        if color == bg:
            continue
        mask = np.zeros_like(grid, dtype=bool)
        mask[:, frame_c1 + 1 :] = grid[:, frame_c1 + 1 :] == color
        for comp in _components(mask, connectivity=8):
            direction = _decode_arrow(comp)
            if direction is None:
                continue
            r0, _, _, _ = _bbox(comp)
            arrow_specs.append((int(r0), color, direction))

    assert arrow_specs, "expected at least one instruction arrow"
    arrow_specs.sort()
    arrow_colors = {color for _, color, _ in arrow_specs}
    assert arrow_colors == {_ROTATE_COLOR, _FLIP_COLOR}, "instruction colors must be blue and red"
    return [(color, direction) for _, color, direction in arrow_specs]


def _apply_step(pattern: np.ndarray, color: int, direction: str) -> np.ndarray:
    if color == _FLIP_COLOR:
        if direction in {"left", "right"}:
            return np.fliplr(pattern).copy()
        return np.flipud(pattern).copy()

    assert color == _ROTATE_COLOR, "unknown arrow color"
    if direction == "up":
        return pattern.copy()
    if direction == "down":
        return np.rot90(pattern, 2).copy()
    if direction == "left":
        return np.rot90(pattern, 1).copy()
    return np.rot90(pattern, -1).copy()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Apply a top-to-bottom chain of arrow-driven rotations and flips to the framed source pattern.

    The grey rectangle on the left carries the source pattern. Blue arrows on the
    right mean rotation (up=0, right=90 CW, down=180, left=90 CCW). Red arrows
    mean reflection (left/right = horizontal, up/down = vertical).
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _most_common_color(g)
    frame_r0, frame_c0, frame_r1, frame_c1 = _find_frame(g, bg)
    assert frame_r1 - frame_r0 >= 2 and frame_c1 - frame_c0 >= 2
    pattern = g[frame_r0 + 1 : frame_r1, frame_c0 + 1 : frame_c1].copy()
    arrows = _find_arrows(g, bg, (frame_r0, frame_c0, frame_r1, frame_c1))

    out = pattern
    for color, direction in arrows:
        out = _apply_step(out, color, direction)
    return out


def _pattern_is_asymmetric(pattern: np.ndarray) -> bool:
    variants = (
        np.fliplr(pattern),
        np.flipud(pattern),
        np.rot90(pattern, 1),
        np.rot90(pattern, 2),
        np.rot90(pattern, 3),
    )
    return not any(np.array_equal(pattern, variant) for variant in variants)


def _make_pattern(rng: np.random.Generator) -> np.ndarray:
    for _ in range(120):
        h = int(rng.integers(4, 7))
        w = int(rng.integers(4, 7))
        color_count = int(rng.integers(2, 5))
        colors = np.array(rng.choice(_PATTERN_COLORS, size=color_count, replace=False), dtype=int)
        pattern = rng.choice(colors, size=(h, w)).astype(int)

        flat = pattern.reshape(-1)
        chosen_slots = rng.choice(flat.size, size=color_count, replace=False)
        for slot, color in zip(chosen_slots, colors):
            flat[int(slot)] = int(color)

        if _pattern_is_asymmetric(pattern):
            return pattern
    raise RuntimeError("failed to build an asymmetric source pattern")


def _sample_sequence(rng: np.random.Generator) -> list[tuple[int, str]]:
    for _ in range(120):
        count = int(rng.integers(3, 6))
        colors = [int(x) for x in rng.choice((_ROTATE_COLOR, _FLIP_COLOR), size=count)]
        directions = [str(x) for x in rng.choice(_DIRECTIONS, size=count)]
        if len(set(colors)) < 2:
            continue
        if len(set(directions)) < 2:
            continue
        return list(zip(colors, directions))
    raise RuntimeError("failed to sample a valid arrow sequence")


def _arrow_dims(direction: str) -> tuple[int, int]:
    rows = [r for r, _ in _ARROW_SHAPES[direction]]
    cols = [c for _, c in _ARROW_SHAPES[direction]]
    return max(rows) + 1, max(cols) + 1


def _paint_arrow(grid: np.ndarray, top: int, left: int, color: int, direction: str) -> None:
    for dr, dc in _ARROW_SHAPES[direction]:
        grid[top + dr, left + dc] = color


def _apply_sequence(pattern: np.ndarray, sequence: list[tuple[int, str]]) -> np.ndarray:
    out = pattern.copy()
    for color, direction in sequence:
        out = _apply_step(out, color, direction)
    return out


def _render_input(
    rng: np.random.Generator,
    pattern: np.ndarray,
    sequence: list[tuple[int, str]],
) -> np.ndarray:
    pattern_h, pattern_w = map(int, pattern.shape)
    frame_h, frame_w = pattern_h + 2, pattern_w + 2
    frame_top = int(rng.integers(1, 4))
    frame_left = int(rng.integers(1, 4))

    arrow_zone_left = frame_left + frame_w + int(rng.integers(3, 7))
    arrow_start = int(rng.integers(1, 4))
    arrow_rows: list[tuple[int, int, str]] = []
    cursor = arrow_start
    for color, direction in sequence:
        arrow_rows.append((cursor, color, direction))
        arrow_h, _ = _arrow_dims(direction)
        cursor += arrow_h + int(rng.integers(2, 4))

    last_arrow_bottom = 0
    for top, _, direction in arrow_rows:
        arrow_h, _ = _arrow_dims(direction)
        last_arrow_bottom = max(last_arrow_bottom, top + arrow_h)

    bottom_margin = int(rng.integers(1, 4))
    right_margin = int(rng.integers(1, 4))
    height = max(frame_top + frame_h, last_arrow_bottom) + bottom_margin
    width = arrow_zone_left + 3 + right_margin
    grid = np.full((height, width), _BG, dtype=int)

    frame_bottom = frame_top + frame_h - 1
    frame_right = frame_left + frame_w - 1
    grid[frame_top, frame_left : frame_right + 1] = _FRAME_COLOR
    grid[frame_bottom, frame_left : frame_right + 1] = _FRAME_COLOR
    grid[frame_top : frame_bottom + 1, frame_left] = _FRAME_COLOR
    grid[frame_top : frame_bottom + 1, frame_right] = _FRAME_COLOR
    grid[frame_top + 1 : frame_bottom, frame_left + 1 : frame_right] = pattern

    for top, color, direction in arrow_rows:
        _paint_arrow(grid, top, arrow_zone_left, color, direction)

    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(_MAX_GEN_ATTEMPTS):
        pattern = _make_pattern(rng)
        sequence = _sample_sequence(rng)
        transformed = _apply_sequence(pattern, sequence)
        if np.array_equal(transformed, pattern):
            continue
        grid = _render_input(rng, pattern, sequence)
        try:
            actual = main(grid)
        except Exception:
            continue
        if np.array_equal(actual, transformed):
            return grid
    raise RuntimeError("failed to generate a valid input grid")
