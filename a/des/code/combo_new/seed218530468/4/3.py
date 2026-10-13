from __future__ import annotations

import numpy as np


BLACK = 0
PALETTE = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int)
INPUT_GAP = 2
OUTPUT_GAP = 1
MAX_GEN_RETRIES = 400
PLACEMENT_RETRIES = 120


def _horizontal_isolated(grid: np.ndarray, row: int, c0: int, c1: int) -> bool:
    h, w = grid.shape
    if c0 > 0 and int(grid[row, c0 - 1]) != BLACK:
        return False
    if c1 + 1 < w and int(grid[row, c1 + 1]) != BLACK:
        return False
    for col in range(c0, c1 + 1):
        if row > 0 and int(grid[row - 1, col]) != BLACK:
            return False
        if row + 1 < h and int(grid[row + 1, col]) != BLACK:
            return False
    return True


def _vertical_isolated(grid: np.ndarray, col: int, r0: int, r1: int) -> bool:
    h, w = grid.shape
    if r0 > 0 and int(grid[r0 - 1, col]) != BLACK:
        return False
    if r1 + 1 < h and int(grid[r1 + 1, col]) != BLACK:
        return False
    for row in range(r0, r1 + 1):
        if col > 0 and int(grid[row, col - 1]) != BLACK:
            return False
        if col + 1 < w and int(grid[row, col + 1]) != BLACK:
            return False
    return True


def _find_stripes(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    stripes: list[dict] = []

    for row in range(h):
        col = 0
        while col < w:
            color = int(g[row, col])
            if color == BLACK:
                col += 1
                continue
            start1 = col
            while col < w and int(g[row, col]) == color:
                col += 1
            end1 = col - 1
            if col < w - 1 and int(g[row, col]) == BLACK and int(g[row, col + 1]) == color:
                col2 = col + 1
                while col2 < w and int(g[row, col2]) == color:
                    col2 += 1
                end2 = col2 - 1
                left_len = end1 - start1 + 1
                right_len = end2 - col
                span = end2 - start1 + 1
                if (
                    4 <= span <= 8
                    and left_len > 0
                    and right_len > 0
                    and left_len != right_len
                    and _horizontal_isolated(g, row, start1, end2)
                ):
                    shift = left_len if left_len > right_len else -right_len
                    stripes.append(
                        {
                            "orient": "h",
                            "color": color,
                            "r": row,
                            "c0": start1,
                            "span": span,
                            "hole": left_len,
                            "shift": shift,
                        }
                    )
                    col = end2 + 1
                    continue

    for col in range(w):
        row = 0
        while row < h:
            color = int(g[row, col])
            if color == BLACK:
                row += 1
                continue
            start1 = row
            while row < h and int(g[row, col]) == color:
                row += 1
            end1 = row - 1
            if row < h - 1 and int(g[row, col]) == BLACK and int(g[row + 1, col]) == color:
                row2 = row + 1
                while row2 < h and int(g[row2, col]) == color:
                    row2 += 1
                end2 = row2 - 1
                top_len = end1 - start1 + 1
                bottom_len = end2 - row
                span = end2 - start1 + 1
                if (
                    4 <= span <= 8
                    and top_len > 0
                    and bottom_len > 0
                    and top_len != bottom_len
                    and _vertical_isolated(g, col, start1, end2)
                ):
                    shift = top_len if top_len > bottom_len else -bottom_len
                    stripes.append(
                        {
                            "orient": "v",
                            "color": color,
                            "r0": start1,
                            "c": col,
                            "span": span,
                            "hole": top_len,
                            "shift": shift,
                        }
                    )
                    row = end2 + 1
                    continue

    return stripes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Translate each holed strip according to the hole's off-center position.

    Input structure:
    - Black background with several non-overlapping colored strips.
    - Each strip is a 1-cell-thick horizontal or vertical line of total span 4-8 cells, but one interior cell is black.
    - The black hole splits the strip into two unequal same-color segments, and different strips are separated by background gaps.

    Rule:
    - Detect each full strip span together with its unique black hole.
    - For a horizontal strip, compare the colored lengths on the left and right of the hole; move the completed strip toward the shorter side by the longer side's length.
    - For a vertical strip, compare the colored lengths above and below the hole; move the completed strip toward the shorter side by the longer side's length.
    - Erase the original holed strips and draw the translated strips as complete solid lines in their original colors.

    Key insight: The hole is not damage but an instruction marker: its asymmetric position encodes both translation direction and translation distance.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    out = np.zeros_like(g)

    for stripe in _find_stripes(g):
        color = int(stripe["color"])
        span = int(stripe["span"])
        shift = int(stripe["shift"])
        if stripe["orient"] == "h":
            row = int(stripe["r"])
            c0 = int(stripe["c0"]) + shift
            out[row, c0 : c0 + span] = color
        else:
            r0 = int(stripe["r0"]) + shift
            col = int(stripe["c"])
            out[r0 : r0 + span, col] = color

    return out


def _case_sequence(rng: np.random.Generator, count: int) -> list[int]:
    if count >= 4:
        cases = [0, 1, 2, 3]
        while len(cases) < count:
            cases.append(int(rng.integers(0, 4)))
        rng.shuffle(cases)
        return cases
    return [int(v) for v in rng.choice(4, size=count, replace=False)]


def _short_long_pair(rng: np.random.Generator, span: int) -> tuple[int, int]:
    choices = [(short, span - 1 - short) for short in range(1, span - 1) if short < span - 1 - short]
    return choices[int(rng.integers(0, len(choices)))]


def _base_spec(color: int, case_id: int, span: int, short_len: int, long_len: int) -> dict:
    if case_id == 0:
        return {
            "color": color,
            "orient": "h",
            "span": span,
            "hole": short_len,
            "shift": -long_len,
        }
    if case_id == 1:
        return {
            "color": color,
            "orient": "h",
            "span": span,
            "hole": long_len,
            "shift": long_len,
        }
    if case_id == 2:
        return {
            "color": color,
            "orient": "v",
            "span": span,
            "hole": short_len,
            "shift": -long_len,
        }
    return {
        "color": color,
        "orient": "v",
        "span": span,
        "hole": long_len,
        "shift": long_len,
    }


def _spec_bbox(spec: dict, moved: bool = False) -> tuple[int, int, int, int]:
    span = int(spec["span"])
    delta = int(spec["shift"]) if moved else 0
    if spec["orient"] == "h":
        row = int(spec["r"])
        c0 = int(spec["c0"]) + delta
        return row, c0, row, c0 + span - 1
    r0 = int(spec["r0"]) + delta
    col = int(spec["c"])
    return r0, col, r0 + span - 1, col


def _expanded_slices(
    bbox: tuple[int, int, int, int], height: int, width: int, margin: int
) -> tuple[slice, slice]:
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - margin)
    cc0 = max(0, c0 - margin)
    rr1 = min(height - 1, r1 + margin)
    cc1 = min(width - 1, c1 + margin)
    return slice(rr0, rr1 + 1), slice(cc0, cc1 + 1)


def _blocked(blocked: np.ndarray, bbox: tuple[int, int, int, int], margin: int) -> bool:
    rs, cs = _expanded_slices(bbox, blocked.shape[0], blocked.shape[1], margin)
    return bool(blocked[rs, cs].any())


def _mark(blocked: np.ndarray, bbox: tuple[int, int, int, int], margin: int) -> None:
    rs, cs = _expanded_slices(bbox, blocked.shape[0], blocked.shape[1], margin)
    blocked[rs, cs] = True


def _place_spec(
    base: dict,
    height: int,
    width: int,
    input_blocked: np.ndarray,
    output_blocked: np.ndarray,
    rng: np.random.Generator,
) -> dict | None:
    span = int(base["span"])
    shift = int(base["shift"])
    orient = base["orient"]

    for _ in range(PLACEMENT_RETRIES):
        spec = dict(base)
        if orient == "h":
            low = max(0, -shift)
            high = min(width - span, width - span - shift)
            if low > high:
                return None
            spec["r"] = int(rng.integers(0, height))
            spec["c0"] = int(rng.integers(low, high + 1))
        else:
            low = max(0, -shift)
            high = min(height - span, height - span - shift)
            if low > high:
                return None
            spec["r0"] = int(rng.integers(low, high + 1))
            spec["c"] = int(rng.integers(0, width))

        in_bbox = _spec_bbox(spec, moved=False)
        out_bbox = _spec_bbox(spec, moved=True)
        if _blocked(input_blocked, in_bbox, INPUT_GAP):
            continue
        if _blocked(output_blocked, out_bbox, OUTPUT_GAP):
            continue
        _mark(input_blocked, in_bbox, INPUT_GAP)
        _mark(output_blocked, out_bbox, OUTPUT_GAP)
        return spec

    return None


def _render_input(specs: list[dict], height: int, width: int) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    for spec in specs:
        color = int(spec["color"])
        hole = int(spec["hole"])
        span = int(spec["span"])
        if spec["orient"] == "h":
            row = int(spec["r"])
            c0 = int(spec["c0"])
            grid[row, c0 : c0 + hole] = color
            grid[row, c0 + hole + 1 : c0 + span] = color
        else:
            r0 = int(spec["r0"])
            col = int(spec["c"])
            grid[r0 : r0 + hole, col] = color
            grid[r0 + hole + 1 : r0 + span, col] = color
    return grid


def _render_output(specs: list[dict], height: int, width: int) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    for spec in specs:
        color = int(spec["color"])
        span = int(spec["span"])
        shift = int(spec["shift"])
        if spec["orient"] == "h":
            row = int(spec["r"])
            c0 = int(spec["c0"]) + shift
            grid[row, c0 : c0 + span] = color
        else:
            r0 = int(spec["r0"]) + shift
            col = int(spec["c"])
            grid[r0 : r0 + span, col] = color
    return grid


def _attempt_generate(rng: np.random.Generator) -> np.ndarray | None:
    count = int(rng.integers(2, 6))
    cases = _case_sequence(rng, count)
    n_horizontal = sum(case in (0, 1) for case in cases)
    n_vertical = count - n_horizontal

    low_h = max(12, 5 * n_horizontal - 4, 14 if n_vertical else 12)
    low_w = max(12, 5 * n_vertical - 4, 14 if n_horizontal else 12)
    if low_h > 30 or low_w > 30:
        return None

    height = int(rng.integers(low_h, 31))
    width = int(rng.integers(low_w, 31))
    colors = [int(v) for v in rng.choice(PALETTE, size=count, replace=False)]

    bases = []
    for color, case_id in zip(colors, cases):
        span = int(rng.integers(4, 9))
        short_len, long_len = _short_long_pair(rng, span)
        bases.append(_base_spec(color, int(case_id), span, short_len, long_len))
    bases.sort(key=lambda spec: (int(spec["span"]), abs(int(spec["shift"]))), reverse=True)

    input_blocked = np.zeros((height, width), dtype=bool)
    output_blocked = np.zeros((height, width), dtype=bool)
    specs: list[dict] = []

    for base in bases:
        placed = _place_spec(base, height, width, input_blocked, output_blocked, rng)
        if placed is None:
            return None
        specs.append(placed)

    inp = _render_input(specs, height, width)
    expected = _render_output(specs, height, width)
    if len(_find_stripes(inp)) != len(specs):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GEN_RETRIES):
        grid = _attempt_generate(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {MAX_GEN_RETRIES} attempts")
