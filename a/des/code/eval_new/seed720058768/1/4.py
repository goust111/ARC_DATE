from __future__ import annotations

from collections import defaultdict, deque
import random

import numpy as np


_N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_BG_COLOR = 8
_MARK_COLOR = 5
_CONTENT_COLORS = [0, 1, 2, 3, 4, 6, 7, 9]
_FRAME_COUNT_CHOICES = [2, 3, 4, 5]
_MAX_RETRIES = 300


def _components_by_color(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue

            color = int(grid[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in _N4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            rows = [pt[0] for pt in coords]
            cols = [pt[1] for pt in coords]
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "coord_set": set(coords),
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                }
            )
    return comps


def _corner_kind(comp: dict[str, object], h: int, w: int) -> str | None:
    r0, c0, r1, c1 = comp["bbox"]
    touches = set()
    if r0 == 0:
        touches.add("top")
    if r1 == h - 1:
        touches.add("bottom")
    if c0 == 0:
        touches.add("left")
    if c1 == w - 1:
        touches.add("right")

    if touches == {"top", "left"}:
        return "TL"
    if touches == {"top", "right"}:
        return "TR"
    if touches == {"bottom", "left"}:
        return "BL"
    if touches == {"bottom", "right"}:
        return "BR"
    return None


def _indicator_color(comps: list[dict[str, object]], h: int, w: int) -> int:
    by_color: dict[int, list[dict[str, object]]] = defaultdict(list)
    for comp in comps:
        by_color[int(comp["color"])].append(comp)

    needed = {"TL", "TR", "BL", "BR"}
    matches = []
    for color, group in by_color.items():
        kinds = [_corner_kind(comp, h, w) for comp in group]
        if len(group) == 4 and set(kinds) == needed:
            matches.append(color)

    assert len(matches) == 1, "expected a unique corner-indicator color"
    return int(matches[0])


def _nested_sequence(comps: list[dict[str, object]], indicator_color: int) -> list[int]:
    central = [comp for comp in comps if int(comp["color"]) != indicator_color]
    assert len(central) >= 2, "expected nested frames plus a filled center"

    central.sort(
        key=lambda comp: (
            -((comp["bbox"][2] - comp["bbox"][0] + 1) * (comp["bbox"][3] - comp["bbox"][1] + 1)),
            comp["bbox"][0],
            comp["bbox"][1],
        )
    )

    for outer, inner in zip(central, central[1:]):
        or0, oc0, or1, oc1 = outer["bbox"]
        ir0, ic0, ir1, ic1 = inner["bbox"]
        assert or0 < ir0 and oc0 < ic0 and ir1 < or1 and ic1 < oc1, "central colors must form a strict nesting chain"

    return [int(comp["color"]) for comp in central]


def _tail_rays(comp: dict[str, object], kind: str) -> list[list[tuple[int, int]]]:
    r0, c0, r1, c1 = comp["bbox"]
    coord_set: set[tuple[int, int]] = comp["coord_set"]

    if kind == "TL":
        elbow = (r1, c1)
        dirs = [(-1, 0), (0, -1)]
    elif kind == "TR":
        elbow = (r1, c0)
        dirs = [(-1, 0), (0, 1)]
    elif kind == "BL":
        elbow = (r0, c1)
        dirs = [(1, 0), (0, -1)]
    else:
        elbow = (r0, c0)
        dirs = [(1, 0), (0, 1)]

    assert elbow in coord_set, "corner component must have an inward elbow"
    rays: list[list[tuple[int, int]]] = []
    for dr, dc in dirs:
        arm: list[tuple[int, int]] = []
        step = 1
        while True:
            rr = elbow[0] + step * dr
            cc = elbow[1] + step * dc
            if (rr, cc) not in coord_set:
                break
            if step >= 2:
                arm.append((rr, cc))
            step += 1
        rays.append(arm)
    return rays


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the color order of nested central rectangles, then repaint each corner tail with that sequence.

    Input structure:
    - The background is a uniform field.
    - Near the middle sits a stack of axis-aligned nested rectangular borders, each in a different color, with a solid innermost rectangle of another color.
    - The four corners contain identical monochrome corner indicators: each is an L-shaped 2x2 corner aimed toward the center, plus two straight tail rays extending outward along its arms.

    Rule:
    - Identify the corner-indicator color and keep each 2x2 L shape unchanged.
    - Read the nested central colors from the outermost rectangle inward, then append the innermost filled-rectangle color to the same sequence.
    - For every corner indicator, recolor the tail pixels on both rays from near to far using that sequence.
    - Leave the nested central structure and all non-tail pixels unchanged.

    Key insight:
    - The central nesting encodes an ordered color list, and each corner provides two outward tails whose pixel order matches that list from the L shape outward.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(np.bincount(g.ravel()).argmax())
    h, w = g.shape

    comps = _components_by_color(g, bg)
    marker_color = _indicator_color(comps, h, w)
    sequence = _nested_sequence(comps, marker_color)

    out = g.copy()
    for comp in comps:
        if int(comp["color"]) != marker_color:
            continue
        kind = _corner_kind(comp, h, w)
        assert kind is not None
        for ray in _tail_rays(comp, kind):
            assert len(ray) >= len(sequence), "tail is shorter than the decoded color sequence"
            for color, (r, c) in zip(sequence, ray):
                out[r, c] = int(color)

    return out


def _draw_nested_rectangles(
    grid: np.ndarray,
    top: int,
    left: int,
    fill_h: int,
    fill_w: int,
    frame_colors: list[int],
    fill_color: int,
) -> None:
    frame_count = len(frame_colors)
    outer_h = fill_h + 2 * frame_count
    outer_w = fill_w + 2 * frame_count

    grid[top + frame_count : top + frame_count + fill_h, left + frame_count : left + frame_count + fill_w] = int(fill_color)

    for layer, color in enumerate(frame_colors):
        r0 = top + layer
        c0 = left + layer
        r1 = top + outer_h - 1 - layer
        c1 = left + outer_w - 1 - layer
        grid[r0, c0 : c1 + 1] = int(color)
        grid[r1, c0 : c1 + 1] = int(color)
        grid[r0 : r1 + 1, c0] = int(color)
        grid[r0 : r1 + 1, c1] = int(color)


def _draw_corner_marker(grid: np.ndarray, kind: str, tail_len: int, color: int) -> None:
    h, w = grid.shape
    arm_len = tail_len + 2

    if kind == "TL":
        elbow = (arm_len - 1, arm_len - 1)
        dirs = [(-1, 0), (0, -1)]
    elif kind == "TR":
        elbow = (arm_len - 1, w - arm_len)
        dirs = [(-1, 0), (0, 1)]
    elif kind == "BL":
        elbow = (h - arm_len, arm_len - 1)
        dirs = [(1, 0), (0, -1)]
    else:
        elbow = (h - arm_len, w - arm_len)
        dirs = [(1, 0), (0, 1)]

    er, ec = elbow
    grid[er, ec] = int(color)
    for dr, dc in dirs:
        for step in range(1, arm_len):
            rr = er + step * dr
            cc = ec + step * dc
            grid[rr, cc] = int(color)


def _balanced_split(total: int) -> tuple[int, int]:
    options = [(a, total - a) for a in range(total + 1)]
    balanced = [pair for pair in options if abs(pair[0] - pair[1]) <= 3]
    if balanced:
        options = balanced
    return random.choice(options)


def _paint_expected_tails(grid: np.ndarray, sequence: list[int]) -> np.ndarray:
    out = grid.copy()
    h, w = out.shape
    comps = _components_by_color(out, _BG_COLOR)
    marker_comps = [comp for comp in comps if int(comp["color"]) == _MARK_COLOR]
    assert len(marker_comps) == 4

    for comp in marker_comps:
        kind = _corner_kind(comp, h, w)
        assert kind is not None
        for ray in _tail_rays(comp, kind):
            assert len(ray) == len(sequence)
            for color, (r, c) in zip(sequence, ray):
                out[r, c] = int(color)
    return out


def _gen_attempt() -> np.ndarray | None:
    frame_count = random.choice(_FRAME_COUNT_CHOICES)
    tail_len = frame_count + 1
    min_dim = 4 * frame_count + 9

    if min_dim > 30:
        return None

    h = random.randint(min_dim, 30)
    w = random.randint(min_dim, 30)
    max_fill_h = h - (4 * frame_count + 8)
    max_fill_w = w - (4 * frame_count + 8)
    if max_fill_h < 1 or max_fill_w < 1:
        return None

    fill_h = random.randint(1, min(8, max_fill_h))
    fill_w = random.randint(1, min(8, max_fill_w))
    outer_h = fill_h + 2 * frame_count
    outer_w = fill_w + 2 * frame_count

    pad = tail_len + 3
    extra_h = h - 2 * pad - outer_h
    extra_w = w - 2 * pad - outer_w
    if extra_h < 0 or extra_w < 0:
        return None

    extra_top, extra_bottom = _balanced_split(extra_h)
    extra_left, extra_right = _balanced_split(extra_w)
    top = pad + extra_top
    left = pad + extra_left

    colors = random.sample(_CONTENT_COLORS, tail_len)
    frame_colors = colors[:-1]
    fill_color = colors[-1]

    inp = np.full((h, w), _BG_COLOR, dtype=int)
    _draw_nested_rectangles(inp, top, left, fill_h, fill_w, frame_colors, fill_color)
    for kind in ("TL", "TR", "BL", "BR"):
        _draw_corner_marker(inp, kind, tail_len, _MARK_COLOR)

    out = _paint_expected_tails(inp, colors)
    if np.array_equal(inp, out):
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {_MAX_RETRIES} attempts")
