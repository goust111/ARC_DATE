import random
from collections import Counter

import numpy as np


BG = 0
GREY = 5
MAX_TRIES = 400
RNG = random.Random()


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(pts: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [p[0] for p in pts]
    cs = [p[1] for p in pts]
    return min(rs), min(cs), max(rs), max(cs)


def _rect_border_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    cells = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _detect_legend(grid: np.ndarray) -> tuple[str, list[int]]:
    h, w = grid.shape
    candidates: list[tuple[str, list[int]]] = []
    edges = {
        "top": [(0, c) for c in range(w)],
        "bottom": [(h - 1, c) for c in range(w)],
        "left": [(r, 0) for r in range(h)],
        "right": [(r, w - 1) for r in range(h)],
    }
    for side, coords in edges.items():
        colors = [int(grid[r, c]) for r, c in coords if int(grid[r, c]) != BG]
        if len(colors) >= 2 and len(set(colors)) == len(colors) and GREY not in colors:
            candidates.append((side, colors))
    assert len(candidates) == 1, "expected exactly one edge legend"
    return candidates[0]


def _frame_components(grid: np.ndarray) -> list[dict]:
    comps = []
    for pts in _components_mask(grid == GREY):
        r0, c0, r1, c1 = _bbox(pts)
        border = _rect_border_cells(r0, c0, r1, c1)
        pset = set(pts)
        missing = sorted(border - pset)
        if len(missing) > 1:
            continue
        if pset != border - set(missing):
            continue
        gap_side = None
        if missing:
            gr, gc = missing[0]
            if gr == r0:
                gap_side = "top"
            elif gr == r1:
                gap_side = "bottom"
            elif gc == c0:
                gap_side = "left"
            else:
                gap_side = "right"
        comps.append(
            {
                "bbox": (r0, c0, r1, c1),
                "pts": pset,
                "gap_side": gap_side,
            }
        )
    return comps


def _contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    or0, oc0, or1, oc1 = outer
    ir0, ic0, ir1, ic1 = inner
    return or0 < ir0 and oc0 < ic0 and ir1 < or1 and ic1 < oc1


def _rotate_points(points: list[tuple[int, int]], center: tuple[int, int], turns_cw: int) -> list[tuple[int, int]]:
    cr, cc = center
    out = []
    for r, c in points:
        dr = r - cr
        dc = c - cc
        for _ in range(turns_cw % 4):
            dr, dc = dc, -dr
        out.append((cr + dr, cc + dc))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    legend_side, legend_colors = _detect_legend(g)
    frames = _frame_components(g)
    assert frames, "expected at least one grey frame"

    for frame in frames:
        bbox = frame["bbox"]
        parents = [other for other in frames if other is not frame and _contains(other["bbox"], bbox)]
        children = [other for other in frames if other is not frame and _contains(bbox, other["bbox"])]
        frame["depth"] = len(parents) + 1
        frame["is_leaf"] = len(children) == 0
        outer_candidates = [p for p in parents if p["gap_side"] is not None]
        frame["outer"] = min(outer_candidates, key=lambda x: x["depth"]) if outer_candidates else (frame if frame["gap_side"] is not None else None)

    out = g.copy()
    side_to_turns = {"top": 0, "right": 1, "bottom": 2, "left": 3}

    leaves = [f for f in frames if f["is_leaf"]]
    assert leaves, "expected innermost frames"
    for frame in leaves:
        depth = frame["depth"]
        assert 1 <= depth <= len(legend_colors), "leaf depth must map into legend"
        outer = frame["outer"]
        assert outer is not None and outer["gap_side"] in side_to_turns, "each frame chain needs one gapped outer frame"
        turns = side_to_turns[outer["gap_side"]]
        r0, c0, r1, c1 = frame["bbox"]
        ir0, ic0, ir1, ic1 = r0 + 1, c0 + 1, r1 - 1, c1 - 1
        pattern = [(r, c) for r in range(ir0, ir1 + 1) for c in range(ic0, ic1 + 1) if int(g[r, c]) not in {BG, GREY}]
        assert pattern, "innermost frame must contain a pattern"
        color = legend_colors[depth - 1]
        old_colors = {int(g[r, c]) for r, c in pattern}
        assert len(old_colors) == 1, "pattern must be monochrome"
        center = ((ir0 + ir1) // 2, (ic0 + ic1) // 2)
        rotated = _rotate_points(pattern, center, turns)
        for r, c in pattern:
            out[r, c] = BG
        for r, c in rotated:
            out[r, c] = color

    return out


def _draw_frame(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int, gap_side: str | None):
    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY
    if gap_side is None:
        return
    mr = (r0 + r1) // 2
    mc = (c0 + c1) // 2
    if gap_side == "top":
        grid[r0, mc] = BG
    elif gap_side == "bottom":
        grid[r1, mc] = BG
    elif gap_side == "left":
        grid[mr, c0] = BG
    else:
        grid[mr, c1] = BG


def _pattern_templates(cavity_side: int) -> list[list[tuple[int, int]]]:
    if cavity_side == 3:
        return [
            [(0, 0), (1, 0), (2, 0), (2, 1)],
            [(0, 1), (1, 1), (2, 1), (2, 0)],
            [(0, 0), (0, 1), (1, 1), (2, 1)],
            [(0, 2), (1, 0), (1, 1), (1, 2), (2, 0)],
        ]
    return [
        [(1, 1), (2, 1), (3, 1), (3, 2)],
        [(1, 3), (2, 3), (3, 3), (3, 2), (2, 2)],
        [(1, 1), (1, 2), (1, 3), (2, 1), (3, 1)],
        [(1, 2), (2, 2), (3, 2), (3, 1), (3, 3)],
    ]


def _can_place(grid: np.ndarray, r0: int, c0: int, size: int) -> bool:
    h, w = grid.shape
    r1 = r0 + size - 1
    c1 = c0 + size - 1
    if r0 < 1 or c0 < 1 or r1 >= h - 1 or c1 >= w - 1:
        return False
    patch = grid[r0 - 1 : r1 + 2, c0 - 1 : c1 + 2]
    return bool(np.all(patch == BG))


def _place_chain(
    grid: np.ndarray,
    r0: int,
    c0: int,
    depth: int,
    cavity_side: int,
    gap_side: str,
    pattern_color: int,
):
    outer_size = cavity_side + 2 + 4 * (depth - 1)
    for layer in range(depth):
        inset = 2 * layer
        size = outer_size - 4 * layer
        fr0 = r0 + inset
        fc0 = c0 + inset
        fr1 = fr0 + size - 1
        fc1 = fc0 + size - 1
        _draw_frame(grid, fr0, fc0, fr1, fc1, gap_side if layer == 0 else None)

    inner_r0 = r0 + 2 * (depth - 1) + 1
    inner_c0 = c0 + 2 * (depth - 1) + 1
    template = RNG.choice(_pattern_templates(cavity_side))
    for dr, dc in template:
        grid[inner_r0 + dr, inner_c0 + dc] = pattern_color


def _legend_cells(side: str, length: int, h: int, w: int) -> list[tuple[int, int]]:
    if side in {"top", "bottom"}:
        max_start = w - length
        start = RNG.randint(0, max_start)
        row = 0 if side == "top" else h - 1
        return [(row, start + i) for i in range(length)]
    max_start = h - length
    start = RNG.randint(0, max_start)
    col = 0 if side == "left" else w - 1
    return [(start + i, col) for i in range(length)]


def _gen_attempt() -> np.ndarray | None:
    legend_len = RNG.randint(2, 4)
    num_groups = RNG.randint(1, 3)
    depths = [RNG.randint(1, legend_len) for _ in range(num_groups)]
    if len(set(depths)) == 1 and legend_len > 1:
        depths[RNG.randrange(num_groups)] = RNG.choice([d for d in range(1, legend_len + 1) if d != depths[0]])

    cavity_choices = [RNG.choice([3, 5]) for _ in range(num_groups)]
    sizes = [cavity_choices[i] + 2 + 4 * (depths[i] - 1) for i in range(num_groups)]
    max_size = max(sizes)

    h = RNG.randint(max(18, max_size + 4), max(28, max_size + 8))
    w = RNG.randint(max(18, max_size + 4), max(30, max_size + 10))
    grid = np.full((h, w), BG, dtype=int)

    colors = [c for c in range(1, 10) if c != GREY]
    RNG.shuffle(colors)
    legend_colors = colors[:legend_len]
    remaining = [c for c in colors if c not in legend_colors]

    legend_side = RNG.choice(["top", "bottom", "left", "right"])
    legend_pos = _legend_cells(legend_side, legend_len, h, w)
    for (r, c), color in zip(legend_pos, legend_colors):
        grid[r, c] = color

    margin_top = 2 if legend_side == "top" else 1
    margin_bottom = 2 if legend_side == "bottom" else 1
    margin_left = 2 if legend_side == "left" else 1
    margin_right = 2 if legend_side == "right" else 1

    directions = ["top", "right", "bottom", "left"]
    RNG.shuffle(directions)
    group_gaps = [directions[i % 4] for i in range(num_groups)]
    if num_groups == 1:
        group_gaps[0] = RNG.choice(directions)

    for i in range(num_groups):
        size = sizes[i]
        placed = False
        for _ in range(120):
            r0 = RNG.randint(margin_top, h - margin_bottom - size)
            c0 = RNG.randint(margin_left, w - margin_right - size)
            if not _can_place(grid, r0, c0, size):
                continue
            _place_chain(grid, r0, c0, depths[i], cavity_choices[i], group_gaps[i], remaining[i])
            placed = True
            break
        if not placed:
            return None

    if len(_detect_legend(grid)[1]) != legend_len:
        return None
    grey_frames = _frame_components(grid)
    if len(grey_frames) != sum(depths):
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_TRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
