from __future__ import annotations

from collections import Counter, defaultdict, deque
import math
import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

ARROW_COLORS = (BLUE, RED, GREEN, YELLOW)
DIRECTIONS = ("up", "down", "left", "right")
GEN_RETRIES = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _find_components(grid: np.ndarray, allowed: set[int]) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color not in allowed:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            pts = np.array(coords, dtype=int)
            r0, c0 = pts.min(axis=0)
            r1, c1 = pts.max(axis=0)
            components.append(
                {
                    "color": color,
                    "coords": pts,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(len(coords)),
                }
            )
    return components


def _right_arrow_mask(size: int) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    mid = size // 2
    for r in range(size):
        length = size - 2 * abs(r - mid)
        start = size - length
        mask[r, start:] = True
    return mask


def _arrow_mask(size: int, direction: str) -> np.ndarray:
    base = _right_arrow_mask(size)
    if direction == "right":
        return base
    if direction == "up":
        return np.rot90(base, 1)
    if direction == "left":
        return np.rot90(base, 2)
    if direction == "down":
        return np.rot90(base, 3)
    raise ValueError(direction)


def _component_mask(component: dict) -> np.ndarray:
    r0, c0, r1, c1 = component["bbox"]
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    coords = component["coords"]
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _classify_direction(component: dict) -> str:
    r0, c0, r1, c1 = component["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    assert h == w and h >= 3 and h % 2 == 1, "arrow bbox must be an odd square"
    mask = _component_mask(component)
    matches = [direction for direction in DIRECTIONS if np.array_equal(mask, _arrow_mask(h, direction))]
    assert len(matches) == 1, "component must match exactly one arrow direction"
    return matches[0]


def _unique_mode(colors: list[int]) -> int:
    counts = Counter(colors)
    ranked = counts.most_common()
    assert ranked, "mode requires at least one color"
    if len(ranked) > 1:
        assert ranked[0][1] > ranked[1][1], "expected a unique mode color"
    return int(ranked[0][0])


def _detect_anchor(grid: np.ndarray) -> tuple[int, int]:
    grey_components = _find_components(grid, {GREY})
    anchors = []
    for comp in grey_components:
        r0, c0, r1, c1 = comp["bbox"]
        if comp["area"] != 9 or (r1 - r0 + 1, c1 - c0 + 1) != (3, 3):
            continue
        if np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == GREY):
            anchors.append((r0, c0))
    assert len(anchors) == 1, "expected exactly one grey 3x3 anchor"
    return anchors[0]


def _paint_ray(out: np.ndarray, anchor: tuple[int, int], direction: str, length: int, color: int) -> None:
    r0, c0 = anchor
    r1, c1 = r0 + 2, c0 + 2
    if direction == "up":
        cells = [(r0 - 1 - i, c0 + 1) for i in range(length)]
    elif direction == "down":
        cells = [(r1 + 1 + i, c0 + 1) for i in range(length)]
    elif direction == "left":
        cells = [(r0 + 1, c0 - 1 - i) for i in range(length)]
    elif direction == "right":
        cells = [(r0 + 1, c1 + 1 + i) for i in range(length)]
    else:
        raise ValueError(direction)
    for rr, cc in cells:
        assert 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1], "ray must stay inside the grid"
        out[rr, cc] = color


def _render_output(shape: tuple[int, int], anchor: tuple[int, int], groups: dict[str, list[int]]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    r0, c0 = anchor
    out[r0 : r0 + 3, c0 : c0 + 3] = GREY
    for direction in DIRECTIONS:
        colors = groups.get(direction, [])
        if not colors:
            continue
        _paint_ray(out, anchor, direction, len(colors), _unique_mode(colors))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group scattered arrow sprites by pointing direction, then summarize each group
    as a colored ray around the grey anchor.
    """
    g = np.asarray(grid, dtype=int)
    anchor = _detect_anchor(g)
    allowed = set(int(v) for v in np.unique(g)) - {BG, GREY}
    arrow_components = _find_components(g, allowed)
    assert arrow_components, "expected at least one arrow"

    groups: dict[str, list[int]] = defaultdict(list)
    for comp in arrow_components:
        direction = _classify_direction(comp)
        groups[direction].append(int(comp["color"]))

    return _render_output(g.shape, anchor, groups)


def _sample_group_colors(count: int) -> list[int]:
    mode = random.choice(ARROW_COLORS)
    others = [color for color in ARROW_COLORS if color != mode]
    if count == 1:
        colors = [mode]
    elif count == 2:
        colors = [mode, mode]
    elif count == 3:
        if random.random() < 0.35:
            colors = [mode, mode, mode]
        else:
            colors = [mode, mode, random.choice(others)]
    elif count == 4:
        branch = random.choice(("2+1+1", "3+1", "4"))
        if branch == "2+1+1":
            a, b = random.sample(others, 2)
            colors = [mode, mode, a, b]
        elif branch == "3+1":
            colors = [mode, mode, mode, random.choice(others)]
        else:
            colors = [mode, mode, mode, mode]
    else:
        raise ValueError(count)
    random.shuffle(colors)
    assert _unique_mode(colors) == mode
    return colors


def _paint_arrow(grid: np.ndarray, top: int, left: int, size: int, direction: str, color: int) -> None:
    mask = _arrow_mask(size, direction)
    grid[top : top + size, left : left + size][mask] = color


def _mark_blocked(blocked: np.ndarray, top: int, left: int, size: int, pad: int = 1) -> None:
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(blocked.shape[0], top + size + pad)
    c1 = min(blocked.shape[1], left + size + pad)
    blocked[r0:r1, c0:c1] = True


def _can_place(blocked: np.ndarray, top: int, left: int, size: int) -> bool:
    return not bool(blocked[top : top + size, left : left + size].any())


def _sample_dimensions(counts: dict[str, int]) -> tuple[int, int] | None:
    total = sum(counts.values())
    rows_pack = math.ceil(math.sqrt(total))
    cols_pack = math.ceil(total / rows_pack)
    min_h = max(18, counts["up"] + counts["down"] + 8, rows_pack * 6 + 2)
    min_w = max(18, counts["left"] + counts["right"] + 8, cols_pack * 6 + 2)
    if min_h > 30 or min_w > 30:
        return None
    return random.randint(min_h, 30), random.randint(min_w, 30)


def _sample_anchor(shape: tuple[int, int], counts: dict[str, int]) -> tuple[int, int] | None:
    h, w = shape
    top_min = max(counts["up"], h // 2 - 2)
    top_max = h - counts["down"] - 3
    left_min = max(counts["left"], w // 2 - 2)
    left_max = w - counts["right"] - 3
    if top_min > top_max or left_min > left_max:
        return None
    return random.randint(top_min, top_max), random.randint(left_min, left_max)


def _sample_arrow_specs() -> tuple[list[dict], dict[str, list[int]]]:
    groups: dict[str, list[int]] = {}
    arrows: list[dict] = []
    for direction in DIRECTIONS:
        count = random.randint(1, 4)
        colors = _sample_group_colors(count)
        groups[direction] = colors[:]
        for color in colors:
            arrows.append(
                {
                    "direction": direction,
                    "color": color,
                    "size": random.choice((3, 5)),
                }
            )
    random.shuffle(arrows)
    arrows.sort(key=lambda spec: spec["size"], reverse=True)
    return arrows, groups


def _place_arrows(grid: np.ndarray, anchor: tuple[int, int], arrows: list[dict]) -> bool:
    blocked = np.zeros_like(grid, dtype=bool)
    _mark_blocked(blocked, anchor[0], anchor[1], 3, pad=1)
    for spec in arrows:
        size = int(spec["size"])
        candidates = [(r, c) for r in range(grid.shape[0] - size + 1) for c in range(grid.shape[1] - size + 1)]
        random.shuffle(candidates)
        placed = False
        for top, left in candidates:
            if not _can_place(blocked, top, left, size):
                continue
            _paint_arrow(grid, top, left, size, str(spec["direction"]), int(spec["color"]))
            _mark_blocked(blocked, top, left, size, pad=1)
            placed = True
            break
        if not placed:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    arrows, groups = _sample_arrow_specs()
    counts = {direction: len(colors) for direction, colors in groups.items()}
    shape = _sample_dimensions(counts)
    if shape is None:
        return None
    anchor = _sample_anchor(shape, counts)
    if anchor is None:
        return None

    grid = np.full(shape, BG, dtype=int)
    grid[anchor[0] : anchor[0] + 3, anchor[1] : anchor[1] + 3] = GREY
    if not _place_arrows(grid, anchor, arrows):
        return None

    expected = _render_output(shape, anchor, groups)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
