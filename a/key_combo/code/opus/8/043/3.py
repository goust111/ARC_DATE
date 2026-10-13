from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import random

import numpy as np


Coord = tuple[int, int]

RANK_COLORS = [7, 8, 9, 2, 1]
INPUT_COLOR_POOL = [1, 2, 3, 4, 5, 6]
GEN_RETRIES = 400
TEMPLATE_WEIGHTS = [
    ("hub3", 0.30),
    ("hub3_plus_isolated", 0.40),
    ("pairpair", 0.30),
]


@dataclass(frozen=True)
class Line:
    top: Coord
    bottom: Coord
    step_c: int
    cells: tuple[Coord, ...]
    cell_set: frozenset[Coord]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    colors = sorted(set(map(int, grid.ravel())))
    if exclude is not None:
        colors = [color for color in colors if color not in exclude]
    return colors


def _neighbors8(r: int, c: int) -> list[Coord]:
    out: list[Coord] = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            out.append((r + dr, c + dc))
    return out


def _components8(mask: np.ndarray) -> list[list[Coord]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[list[Coord]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[Coord] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors8(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(comp)
    return components


def _line_from_endpoints(p0: Coord, p1: Coord) -> tuple[Coord, ...]:
    (r0, c0), (r1, c1) = sorted((p0, p1))
    dr = r1 - r0
    dc = c1 - c0
    assert dr > 0 and abs(dr) == abs(dc), "endpoint pair must define a strict 45-degree diagonal"
    step_c = 1 if dc > 0 else -1
    return tuple((r0 + k, c0 + k * step_c) for k in range(dr + 1))


def _rank_and_recolor(mask: np.ndarray, bg: int = 0) -> np.ndarray:
    out = np.full(mask.shape, bg, dtype=int)
    components = _components8(mask)
    ranked = sorted(
        components,
        key=lambda comp: (-len(comp), min(comp)[0], min(comp)[1]),
    )
    for idx, comp in enumerate(ranked):
        if idx < len(RANK_COLORS):
            color = RANK_COLORS[idx]
        else:
            color = RANK_COLORS[3 + (idx - 3) % 2]
        for r, c in comp:
            out[r, c] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Connect same-colored diagonal endpoint pairs, merge the crossed lines into 8-connected components, and recolor components by descending area.

    Input structure:
    - The grid is mostly black background.
    - There are 3 or 4 foreground colors, and each foreground color appears exactly twice as single-pixel endpoints.
    - Each same-colored endpoint pair lies on one strict 45-degree diagonal.

    Rule:
    - For each foreground color, draw the full diagonal segment between its two endpoints, including the endpoints themselves.
    - Process the colors in the order of their first visible endpoint from top to bottom; if two segments pass through the same cell, the later segment overwrites that cell's line color.
    - After all segments are drawn, ignore line colors and split all non-background cells into 8-connected components.
    - Sort the components by total pixel count descending and recolor them in rank order: largest to orange (7), second to teal (8), third to maroon (9), then red (2), blue (1), alternating red/blue if needed further.

    Key insight:
    - The relevant objects are not the endpoint pairs alone but the merged 8-connected structures that only appear after every diagonal segment has been completed and crossings have fused lines together.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be 2D"

    bg = _get_bg(g)
    colors = _colors_present(g, exclude={bg})
    assert 3 <= len(colors) <= 4, "expected 3 or 4 foreground colors"

    line_specs: list[tuple[int, int, int, tuple[Coord, ...]]] = []
    for color in colors:
        coords = [tuple(map(int, rc)) for rc in np.argwhere(g == color)]
        assert len(coords) == 2, "each foreground color must appear exactly twice"
        coords.sort()
        cells = _line_from_endpoints(coords[0], coords[1])
        line_specs.append((coords[0][0], coords[0][1], color, cells))

    out = g.copy()
    for _top_r, _top_c, color, cells in sorted(line_specs):
        for r, c in cells:
            out[r, c] = color

    return _rank_and_recolor(out != bg, bg=bg)


def _build_line(top_r: int, top_c: int, step_c: int, span: int) -> Line:
    cells = tuple((top_r + k, top_c + step_c * k) for k in range(span + 1))
    return Line(
        top=(top_r, top_c),
        bottom=cells[-1],
        step_c=step_c,
        cells=cells,
        cell_set=frozenset(cells),
    )


def _candidate_lines(shape: tuple[int, int]) -> list[Line]:
    h, w = shape
    span_min = 3
    span_max = min(8, min(h, w) - 2)
    out: list[Line] = []
    for span in range(span_min, span_max + 1):
        for step_c in (-1, 1):
            for top_r in range(h - span):
                for top_c in range(w):
                    bottom_c = top_c + step_c * span
                    if 0 <= bottom_c < w:
                        out.append(_build_line(top_r, top_c, step_c, span))
    return out


def _proper_intersection(line_a: Line, line_b: Line) -> Coord | None:
    if line_a.step_c == line_b.step_c:
        return None
    shared = line_a.cell_set & line_b.cell_set
    if len(shared) != 1:
        return None
    point = next(iter(shared))
    if point in (line_a.top, line_a.bottom, line_b.top, line_b.bottom):
        return None
    return point


def _touching8(set_a: frozenset[Coord] | set[Coord], set_b: frozenset[Coord] | set[Coord]) -> bool:
    for r, c in set_a:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if (r + dr, c + dc) in set_b:
                    return True
    return False


def _foreground_mask(shape: tuple[int, int], lines: list[Line]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for line in lines:
        for r, c in line.cells:
            mask[r, c] = True
    return mask


def _cross_pair_count(lines: list[Line]) -> int:
    count = 0
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            if _proper_intersection(lines[i], lines[j]) is not None:
                count += 1
    return count


def _valid_line_set(
    shape: tuple[int, int],
    lines: list[Line],
    expected_components: int,
) -> bool:
    if len({line.top[0] for line in lines}) != len(lines):
        return False

    endpoints: list[Coord] = [point for line in lines for point in (line.top, line.bottom)]
    if len(set(endpoints)) != len(endpoints):
        return False

    for i, line in enumerate(lines):
        for endpoint in (line.top, line.bottom):
            for j, other in enumerate(lines):
                if i != j and endpoint in other.cell_set:
                    return False

    if _cross_pair_count(lines) < 2:
        return False

    mask = _foreground_mask(shape, lines)
    components = _components8(mask)
    if len(components) != expected_components:
        return False

    areas = [len(comp) for comp in components]
    if len(set(areas)) != len(areas):
        return False

    if np.count_nonzero(mask) > int(shape[0] * shape[1] * 0.36):
        return False

    return True


def _choose_hub_triplet(shape: tuple[int, int], candidates: list[Line]) -> list[Line] | None:
    by_slope = {
        -1: [line for line in candidates if line.step_c == -1],
        1: [line for line in candidates if line.step_c == 1],
    }
    for _ in range(600):
        hub = random.choice(candidates)
        crossers = [line for line in by_slope[-hub.step_c] if _proper_intersection(hub, line) is not None]
        if len(crossers) < 2:
            continue
        random.shuffle(crossers)
        for i in range(min(len(crossers), 18)):
            for j in range(i + 1, min(len(crossers), 18)):
                line_a = crossers[i]
                line_b = crossers[j]
                if _proper_intersection(hub, line_a) == _proper_intersection(hub, line_b):
                    continue
                if _touching8(line_a.cell_set, line_b.cell_set):
                    continue
                lines = [hub, line_a, line_b]
                if _valid_line_set(shape, lines, expected_components=1):
                    return lines
    return None


def _choose_isolated_line(shape: tuple[int, int], base_lines: list[Line], candidates: list[Line]) -> Line | None:
    occupied = set().union(*(line.cell_set for line in base_lines))
    taken_rows = {line.top[0] for line in base_lines}
    options = candidates[:]
    random.shuffle(options)
    for line in options:
        if line.top[0] in taken_rows:
            continue
        if any(endpoint in occupied for endpoint in (line.top, line.bottom)):
            continue
        if _touching8(line.cell_set, occupied):
            continue
        lines = base_lines + [line]
        if _valid_line_set(shape, lines, expected_components=2):
            return line
    return None


def _choose_pairpair(shape: tuple[int, int], candidates: list[Line]) -> list[Line] | None:
    by_slope = {
        -1: [line for line in candidates if line.step_c == -1],
        1: [line for line in candidates if line.step_c == 1],
    }
    for _ in range(1000):
        line_a = random.choice(candidates)
        matches_a = [line for line in by_slope[-line_a.step_c] if _proper_intersection(line_a, line) is not None]
        if not matches_a:
            continue
        line_b = random.choice(matches_a)
        pair1 = [line_a, line_b]

        remaining = candidates[:]
        random.shuffle(remaining)
        for line_c in remaining[:120]:
            if line_c in pair1:
                continue
            if line_c.top[0] in {line.top[0] for line in pair1}:
                continue
            matches_c = [line for line in by_slope[-line_c.step_c] if _proper_intersection(line_c, line) is not None]
            random.shuffle(matches_c)
            for line_d in matches_c[:24]:
                if line_d in pair1 or line_d == line_c:
                    continue
                lines = [line_a, line_b, line_c, line_d]
                if _valid_line_set(shape, lines, expected_components=2):
                    return lines
    return None


def _choose_template() -> str:
    names = [name for name, _weight in TEMPLATE_WEIGHTS]
    weights = [weight for _name, weight in TEMPLATE_WEIGHTS]
    return random.choices(names, weights=weights, k=1)[0]


def _render_input(shape: tuple[int, int], lines: list[Line], colors: list[int]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for line, color in zip(lines, colors):
        grid[line.top] = color
        grid[line.bottom] = color
    return grid


def _expected_from_hidden_lines(shape: tuple[int, int], lines: list[Line], colors: list[int]) -> np.ndarray:
    full = np.zeros(shape, dtype=int)
    draw_order = sorted(zip(lines, colors), key=lambda item: (item[0].top[0], item[0].top[1]))
    for line, color in draw_order:
        for r, c in line.cells:
            full[r, c] = color
    return _rank_and_recolor(full != 0, bg=0)


def _semantic_checks(shape: tuple[int, int], lines: list[Line], inp: np.ndarray, out: np.ndarray) -> bool:
    colors = _colors_present(inp, exclude={0})
    if len(colors) != len(lines):
        return False
    if len(colors) not in (3, 4):
        return False
    if any(np.count_nonzero(inp == color) != 2 for color in colors):
        return False
    if not _valid_line_set(shape, lines, expected_components=1 if len(lines) == 3 else 2):
        return False
    if len(set(map(int, out.ravel())) - {0}) != (1 if len(lines) == 3 else 2):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(12, 16)
    w = random.randint(12, 16)
    shape = (h, w)
    candidates = _candidate_lines(shape)
    template = _choose_template()

    if template == "hub3":
        lines = _choose_hub_triplet(shape, candidates)
    elif template == "hub3_plus_isolated":
        hub_lines = _choose_hub_triplet(shape, candidates)
        if hub_lines is None:
            return None
        isolated = _choose_isolated_line(shape, hub_lines, candidates)
        lines = None if isolated is None else hub_lines + [isolated]
    else:
        lines = _choose_pairpair(shape, candidates)

    if lines is None:
        return None

    colors = random.sample(INPUT_COLOR_POOL, len(lines))
    inp = _render_input(shape, lines, colors)
    expected = _expected_from_hidden_lines(shape, lines, colors)

    try:
        actual = main(inp)
    except Exception:
        return None

    if not np.array_equal(actual, expected):
        return None
    if not _semantic_checks(shape, lines, inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
