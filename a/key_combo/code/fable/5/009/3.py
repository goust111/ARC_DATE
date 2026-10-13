from __future__ import annotations

from collections import Counter

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


def _background(grid: np.ndarray) -> int:
    """Return the unique most frequent color."""
    counts = Counter(map(int, grid.ravel()))
    largest = max(counts.values())
    choices = [color for color, count in counts.items() if count == largest]
    assert len(choices) == 1, "background color must be uniquely dominant"
    return int(choices[0])


def _components4(mask: np.ndarray) -> list[list[Coord]]:
    """Small self-contained counterpart of arc_utils.components_mask."""
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    result: list[list[Coord]] = []
    for sr0, sc0 in np.argwhere(mask):
        sr, sc = int(sr0), int(sc0)
        if seen[sr, sc]:
            continue
        seen[sr, sc] = True
        stack = [(sr, sc)]
        component: list[Coord] = []
        while stack:
            r, c = stack.pop()
            component.append((r, c))
            for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= rr < height and 0 <= cc < width and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    stack.append((rr, cc))
        result.append(component)
    return result


def _bar_orientation(cells: list[Coord]) -> str | None:
    """Return H/V exactly for a solid straight bar of length at least two."""
    if len(cells) < 2:
        return None
    rows = sorted({r for r, _c in cells})
    cols = sorted({c for _r, c in cells})
    if len(rows) == 1 and cols == list(range(cols[0], cols[-1] + 1)):
        return "H"
    if len(cols) == 1 and rows == list(range(rows[0], rows[-1] + 1)):
        return "V"
    return None


def _identify_colors_and_key(grid: np.ndarray, bg: int) -> tuple[int, int, list[Coord], str]:
    colors = set(map(int, grid.ravel())) - {bg}
    assert len(colors) == 2, "expected one wall color and one key color"
    candidates: list[tuple[int, list[Coord], str]] = []
    for color in colors:
        components = _components4(grid == color)
        if len(components) == 1:
            orientation = _bar_orientation(components[0])
            if orientation is not None:
                candidates.append((int(color), components[0], orientation))
    assert len(candidates) == 1, "the unique separate straight bar must identify the key"
    key_color, key_cells, orientation = candidates[0]
    wall_color = int(next(iter(colors - {key_color})))
    return wall_color, key_color, key_cells, orientation


def _border_cells(box: BBox) -> set[Coord]:
    r0, c0, r1, c1 = box
    return (
        {(r0, c) for c in range(c0, c1 + 1)}
        | {(r1, c) for c in range(c0, c1 + 1)}
        | {(r, c0) for r in range(r0, r1 + 1)}
        | {(r, c1) for r in range(r0, r1 + 1)}
    )


def _box_info(component: list[Coord]) -> dict[str, object]:
    rows = [r for r, _c in component]
    cols = [c for _r, c in component]
    box = (min(rows), min(cols), max(rows), max(cols))
    r0, c0, r1, c1 = box
    assert r1 - r0 >= 4 and c1 - c0 >= 4, "boxes must have a visible cavity"
    occupied = set(component)
    border = _border_cells(box)
    assert occupied <= border, "box walls must be one cell thick"
    missing = border - occupied
    assert missing, "each box must have one gap"
    assert not ({(r0, c0), (r0, c1), (r1, c0), (r1, c1)} & missing), (
        "gap endpoints must remain bounded by wall"
    )

    side_sets = {
        "top": {(r0, c) for c in range(c0 + 1, c1)},
        "bottom": {(r1, c) for c in range(c0 + 1, c1)},
        "left": {(r, c0) for r in range(r0 + 1, r1)},
        "right": {(r, c1) for r in range(r0 + 1, r1)},
    }
    sides = [side for side, cells in side_sets.items() if missing <= cells]
    assert len(sides) == 1, "a box gap must lie on exactly one side"
    side = sides[0]
    ordered = sorted(c for _r, c in missing) if side in {"top", "bottom"} else sorted(r for r, _c in missing)
    assert ordered == list(range(ordered[0], ordered[-1] + 1)), "box gap must be continuous"
    assert occupied == border - missing, "all non-gap border cells must be present"
    return {
        "bbox": box,
        "gap_side": side,
        "gap": frozenset(missing),
        "span": len(missing),
        "mid2": ordered[0] + ordered[-1],
    }


def _find_boxes(grid: np.ndarray, wall_color: int) -> list[dict[str, object]]:
    components = _components4(grid == wall_color)
    assert len(components) >= 2, "the flight line must contain several boxes"
    return [_box_info(component) for component in components]


def _direction_for_key(
    key_cells: list[Coord], orientation: str, boxes: list[dict[str, object]]
) -> Coord:
    key_rows = [r for r, _c in key_cells]
    key_cols = [c for _r, c in key_cells]
    center2 = min(key_cols) + max(key_cols) if orientation == "H" else min(key_rows) + max(key_rows)
    scores: Counter[Coord] = Counter()

    for info in boxes:
        r0, c0, r1, c1 = info["bbox"]  # type: ignore[misc]
        side = str(info["gap_side"])
        if int(info["mid2"]) != center2:
            continue
        if orientation == "H" and side == "top" and max(key_rows) > r0:
            scores[(-1, 0)] += 1
        elif orientation == "H" and side == "bottom" and min(key_rows) < r1:
            scores[(1, 0)] += 1
        elif orientation == "V" and side == "left" and max(key_cols) > c0:
            scores[(0, -1)] += 1
        elif orientation == "V" and side == "right" and min(key_cols) < c1:
            scores[(0, 1)] += 1

    assert scores, "aligned box gaps must reveal the key's direction"
    best = max(scores.values())
    winners = [direction for direction, score in scores.items() if score == best]
    assert len(winners) == 1, "the visible gaps must determine a unique direction"
    return winners[0]


def _front_side(direction: Coord) -> str:
    return {(-1, 0): "top", (1, 0): "bottom", (0, -1): "left", (0, 1): "right"}[direction]


def _front_line(info: dict[str, object], direction: Coord) -> int:
    r0, c0, r1, c1 = info["bbox"]  # type: ignore[misc]
    return r0 if direction == (-1, 0) else r1 if direction == (1, 0) else c0 if direction == (0, -1) else c1


def _bar_line(cells: list[Coord], orientation: str) -> int:
    return cells[0][0] if orientation == "H" else cells[0][1]


def _fill_cavity(out: np.ndarray, info: dict[str, object], bg: int, key_color: int) -> None:
    r0, c0, r1, c1 = info["bbox"]  # type: ignore[misc]
    interior = out[r0 + 1 : r1, c0 + 1 : c1]
    interior[interior == bg] = key_color


def _solve(grid: np.ndarray, return_trace: bool = False) -> tuple[np.ndarray, list[str]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty two-dimensional grid"
    bg = _background(g)
    wall_color, key_color, key_cells, orientation = _identify_colors_and_key(g, bg)
    boxes = _find_boxes(g, wall_color)
    direction = _direction_for_key(key_cells, orientation, boxes)
    dr, dc = direction
    height, width = g.shape

    # Index the independently recognizable box wall lines encountered by the bar.
    line_to_box: dict[int, dict[str, object]] = {}
    for info in boxes:
        line = _front_line(info, direction)
        assert line not in line_to_box, "nested forward wall lines must be distinct"
        line_to_box[line] = info

    out = g.copy()
    for r, c in key_cells:
        out[r, c] = bg
    current = list(key_cells)
    wall_mask = g == wall_color
    trace: list[str] = []

    while True:
        candidate = [(r + dr, c + dc) for r, c in current]
        if any(not (0 <= r < height and 0 <= c < width) for r, c in candidate):
            trace.append("edge")
            break

        line = _bar_line(candidate, orientation)
        info = line_to_box.get(line)
        if info is not None:
            gap = set(info["gap"]) if str(info["gap_side"]) == _front_side(direction) else set()
            candidate_set = set(candidate)
            if not gap:
                trace.append("solid")
                break
            if candidate_set == gap:
                trace.append("seal")
                current = candidate
                for r, c in current:
                    out[r, c] = key_color
                _fill_cavity(out, info, bg, key_color)
                return out, trace
            if candidate_set < gap:
                trace.append("pass")
                current = candidate
                continue
            trace.append("blocked")
            break

        if any(wall_mask[r, c] for r, c in candidate):
            trace.append("collision")
            break
        current = candidate

    for r, c in current:
        out[r, c] = key_color
    return out, trace


def main(grid: np.ndarray) -> np.ndarray:
    """Slide a bar key through nested gapped boxes until it seals, jams, or reaches an edge.

    Input structure:
    - The unique dominant color is background. One other color draws several
      one-cell-thick nested rectangular boxes, each with one bounded continuous
      gap on exactly one side; the remaining color is one straight bar key.
    - The bar is parallel to the aligned forward gaps, and their common midpoint
      visibly fixes one of the four perpendicular travel directions.

    Rule:
    - Move the key one cell at a time in that direction and compare its span at
      every encountered forward wall: a wider gap is passed, an equal gap is
      filled by the key and causes that box's background cavity to take the key
      color, while a narrower or absent gap stops the key beside the wall.
    - If all forward gaps are wider, the key stops flush with the grid edge.
      Box-wall pixels and every unsealed box remain otherwise unchanged.

    Key insight:
    - Span comparison is recursive: each passed inner gap exposes the next outer
      wall, so the first non-passing event—not the nearest or outermost box—sets
      the final key position and whether a cavity is filled.
    """
    return _solve(grid, return_trace=False)[0]


def _gap_cells(box: BBox, side: str, span: int) -> set[Coord]:
    r0, c0, r1, c1 = box
    if side in {"top", "bottom"}:
        center = (c0 + c1) // 2
        row = r0 if side == "top" else r1
        return {(row, c) for c in range(center - span // 2, center + span // 2 + 1)}
    center = (r0 + r1) // 2
    col = c0 if side == "left" else c1
    return {(r, col) for r in range(center - span // 2, center + span // 2 + 1)}


def _paint_box(grid: np.ndarray, box: BBox, side: str, span: int, bg: int, wall: int) -> None:
    border = _border_cells(box)
    gap = _gap_cells(box, side, span)
    assert gap < border and all(grid[r, c] == bg for r, c in border)
    for r, c in border - gap:
        grid[r, c] = wall


def _expected_from_spec(grid: np.ndarray, spec: dict[str, object]) -> np.ndarray:
    """Independent oracle using sampled terminal geometry, not object recognition."""
    out = grid.copy()
    bg = int(spec["bg"])
    key_color = int(spec["key_color"])
    key_cells = spec["key_cells"]  # type: ignore[assignment]
    for r, c in key_cells:  # type: ignore[union-attr]
        out[r, c] = bg

    direction = spec["direction"]  # type: ignore[assignment]
    dr, dc = direction  # type: ignore[misc]
    branch = str(spec["branch"])
    boxes = spec["boxes"]  # type: ignore[assignment]
    terminal_index = int(spec["terminal_index"])

    if branch == "edge":
        cells = list(key_cells)  # type: ignore[arg-type]
        while True:
            candidate = [(r + dr, c + dc) for r, c in cells]
            if any(not (0 <= r < out.shape[0] and 0 <= c < out.shape[1]) for r, c in candidate):
                break
            cells = candidate
    else:
        terminal_box = boxes[terminal_index]  # type: ignore[index]
        r0, c0, r1, c1 = terminal_box
        if branch == "seal":
            line = r0 if direction == (-1, 0) else r1 if direction == (1, 0) else c0 if direction == (0, -1) else c1
        else:
            line = r0 + 1 if direction == (-1, 0) else r1 - 1 if direction == (1, 0) else c0 + 1 if direction == (0, -1) else c1 - 1
        if direction[0] != 0:  # type: ignore[index]
            center = (c0 + c1) // 2
            length = len(key_cells)  # type: ignore[arg-type]
            cells = [(line, c) for c in range(center - length // 2, center + length // 2 + 1)]
        else:
            center = (r0 + r1) // 2
            length = len(key_cells)  # type: ignore[arg-type]
            cells = [(r, line) for r in range(center - length // 2, center + length // 2 + 1)]

    for r, c in cells:
        out[r, c] = key_color
    if branch == "seal":
        r0, c0, r1, c1 = boxes[terminal_index]  # type: ignore[index]
        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        interior[interior == bg] = key_color
    return out


def _validate_generated(grid: np.ndarray, spec: dict[str, object]) -> bool:
    """Check visible semantics, branch dependencies, uniqueness, and the independent oracle."""
    try:
        if grid.ndim != 2 or max(grid.shape) > 30:
            return False
        bg = _background(grid)
        if bg != int(spec["bg"]):
            return False
        wall, key, key_cells, orientation = _identify_colors_and_key(grid, bg)
        if wall != int(spec["wall_color"]) or key != int(spec["key_color"]):
            return False
        boxes = _find_boxes(grid, wall)
        sampled_boxes = set(spec["boxes"])  # type: ignore[arg-type]
        if {info["bbox"] for info in boxes} != sampled_boxes:
            return False
        direction = _direction_for_key(key_cells, orientation, boxes)
        if direction != spec["direction"]:
            return False

        solved, trace = _solve(grid, return_trace=True)
        branch = str(spec["branch"])
        terminal = int(spec["terminal_index"])
        wanted = ["pass"] * (len(spec["boxes"]) if branch == "edge" else terminal)  # type: ignore[arg-type]
        wanted.append({"seal": "seal", "narrow": "blocked", "solid": "solid", "edge": "edge"}[branch])
        if trace != wanted or trace.count("pass") < 1:
            return False
        expected = _expected_from_spec(grid, spec)
        if not np.array_equal(solved, expected) or np.array_equal(solved, grid):
            return False
        if np.any(solved[grid == wall] != wall):
            return False

        # A one-step shortcut must fail: the terminal is never the first wall.
        if branch != "edge" and terminal < 1:
            return False
        return True
    except (AssertionError, IndexError, TypeError, ValueError):
        return False


def _gen_attempt(
    rng: np.random.Generator, branch: str | None = None, direction: Coord | None = None
) -> np.ndarray | None:
    if branch is None:
        branch = str(rng.choice(("seal", "narrow", "solid", "edge")))
    if direction is None:
        direction = ((-1, 0), (1, 0), (0, -1), (0, 1))[int(rng.integers(4))]
    orientation = "H" if direction[0] else "V"
    key_length = int(rng.choice((3, 5)))
    box_count = int(rng.integers(3, 5))
    terminal_index = box_count if branch == "edge" else int(rng.integers(1, box_count))

    inner_parallel = int(rng.choice((9, 11)))
    inner_perpendicular = int(rng.choice((7, 9)))
    outer_parallel = inner_parallel + 4 * (box_count - 1)
    outer_perpendicular = inner_perpendicular + 4 * (box_count - 1)
    outer_h = outer_perpendicular if orientation == "H" else outer_parallel
    outer_w = outer_parallel if orientation == "H" else outer_perpendicular

    def margins(size: int) -> tuple[int, int, int]:
        extra = int(rng.integers(2, min(7, 30 - size) + 1))
        before = int(rng.integers(1, extra))
        return before, extra - before, size + extra

    top, _bottom, height = margins(outer_h)
    left, _right, width = margins(outer_w)
    palette = list(map(int, rng.choice(np.arange(10), size=3, replace=False)))
    bg, wall_color, key_color = palette
    grid = np.full((height, width), bg, dtype=int)

    outer = (top, left, top + outer_h - 1, left + outer_w - 1)
    boxes: list[BBox] = []
    forward = _front_side(direction)
    lateral_choices = ("left", "right") if orientation == "H" else ("top", "bottom")

    for index in range(box_count):
        inset = 2 * (box_count - 1 - index)
        r0, c0, r1, c1 = outer
        box = (r0 + inset, c0 + inset, r1 - inset, c1 - inset)
        boxes.append(box)
        if branch == "solid" and index == terminal_index:
            side = str(rng.choice(lateral_choices))
            span = 3
        else:
            side = forward
            if branch == "edge" or index < terminal_index:
                span = key_length + 2
            elif index == terminal_index and branch == "seal":
                span = key_length
            elif index == terminal_index and branch == "narrow":
                span = key_length - 2
            else:
                span = int(rng.choice((1, key_length, key_length + 2)))
        _paint_box(grid, box, side, span, bg, wall_color)

    ir0, ic0, ir1, ic1 = boxes[0]
    if direction == (-1, 0):
        key_line, center = ir1 - 2, (ic0 + ic1) // 2
        key_cells = [(key_line, c) for c in range(center - key_length // 2, center + key_length // 2 + 1)]
    elif direction == (1, 0):
        key_line, center = ir0 + 2, (ic0 + ic1) // 2
        key_cells = [(key_line, c) for c in range(center - key_length // 2, center + key_length // 2 + 1)]
    elif direction == (0, -1):
        key_line, center = ic1 - 2, (ir0 + ir1) // 2
        key_cells = [(r, key_line) for r in range(center - key_length // 2, center + key_length // 2 + 1)]
    else:
        key_line, center = ic0 + 2, (ir0 + ir1) // 2
        key_cells = [(r, key_line) for r in range(center - key_length // 2, center + key_length // 2 + 1)]
    if any(grid[r, c] != bg for r, c in key_cells):
        return None
    for r, c in key_cells:
        grid[r, c] = key_color

    spec: dict[str, object] = {
        "bg": bg,
        "wall_color": wall_color,
        "key_color": key_color,
        "key_cells": key_cells,
        "direction": direction,
        "branch": branch,
        "terminal_index": terminal_index,
        "boxes": boxes,
    }
    return grid if _validate_generated(grid, spec) else None


_BRANCHES = ("seal", "narrow", "solid", "edge")
_DIRECTIONS: tuple[Coord, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))
_generation_index = 0


def generate_input() -> np.ndarray:
    """Generate diverse examples while cycling evenly through branch/direction pairs."""
    global _generation_index
    rng = np.random.default_rng()
    # Four consecutive calls cover every terminal behavior.  Across sixteen calls,
    # every behavior also occurs in every direction, while all geometry and colors
    # remain independently random.
    branch = _BRANCHES[_generation_index % len(_BRANCHES)]
    direction = _DIRECTIONS[(_generation_index + _generation_index // 4) % len(_DIRECTIONS)]
    for _ in range(300):
        grid = _gen_attempt(rng, branch=branch, direction=direction)
        if grid is not None:
            _generation_index += 1
            return grid
    raise RuntimeError("failed to generate a valid gapped-box chain after 300 attempts")
