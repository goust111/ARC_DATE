from __future__ import annotations

from collections import deque
from itertools import combinations
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

CYCLE = (RED, GREEN, YELLOW)
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GEN_HEIGHTS = (10, 11, 12, 13)
GEN_LAYOUTS = ((4, 6), (4, 7), (5, 6))  # (panel_count, panel_width)
GEN_MAX_ATTEMPTS = 500

_GEN_CALL_COUNT = 0


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[np.ndarray] = []
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _full_support(mask: np.ndarray) -> np.ndarray:
    out = np.zeros(mask.shape, dtype=bool)
    if mask.shape[0] < 3 or mask.shape[1] < 3:
        return out
    out[1:-1, 1:-1] = (
        mask[:-2, :-2]
        & mask[:-2, 1:-1]
        & mask[:-2, 2:]
        & mask[1:-1, :-2]
        & mask[1:-1, 1:-1]
        & mask[1:-1, 2:]
        & mask[2:, :-2]
        & mask[2:, 1:-1]
        & mask[2:, 2:]
    )
    return out


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(v) for v in row) for row in mask.astype(int))


def _is_tight_bbox(mask: np.ndarray) -> bool:
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _shape_library() -> list[np.ndarray]:
    shapes: list[np.ndarray] = []
    seen: set[tuple[tuple[int, ...], ...]] = set()
    for h in range(3, 6):
        for w in range(3, 6):
            for r0 in range(h - 2):
                for c0 in range(w - 2):
                    base = np.zeros((h, w), dtype=bool)
                    base[r0 : r0 + 3, c0 : c0 + 3] = True
                    outside = [(r, c) for r in range(h) for c in range(w) if not base[r, c]]
                    for extra_count in range(min(4, len(outside)) + 1):
                        for extra_cells in combinations(outside, extra_count):
                            mask = base.copy()
                            for rr, cc in extra_cells:
                                mask[rr, cc] = True
                            if not _is_tight_bbox(mask):
                                continue
                            if len(_components4(mask)) != 1:
                                continue
                            if int(mask.sum()) > 13:
                                continue
                            if int(_full_support(mask).sum()) != 1:
                                continue
                            key = _mask_key(mask)
                            if key in seen:
                                continue
                            seen.add(key)
                            shapes.append(mask.copy())
    shapes.sort(key=lambda m: (m.shape[0] * m.shape[1], int(m.sum()), m.shape[0], m.shape[1], _mask_key(m)))
    return shapes


SPRITE_LIBRARY = _shape_library()


def _panel_separators(grid: np.ndarray, frame: int) -> list[int]:
    return [c for c in range(grid.shape[1]) if np.all(grid[:, c] == frame)]


def _fit_forward_cycle(colors: list[int]) -> tuple[int, int]:
    matches = [offset for offset in range(3) if all(colors[i] == CYCLE[(offset + i) % 3] for i in range(len(colors)))]
    assert len(matches) == 1, "marker colors must follow one visible forward 3-cycle"
    offset = matches[0]
    next_color = CYCLE[(offset + len(colors)) % 3]
    return offset, next_color


def _extract_sprite_info(panel: np.ndarray) -> dict[str, object]:
    inner = panel[1:-1, 1:-1]
    obj_mask = inner != BLACK
    comps = _components4(obj_mask)
    assert len(comps) == 1, "each panel must contain exactly one sprite component"
    coords = comps[0]
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
    mask = obj_mask[r0 : r1 + 1, c0 : c1 + 1]
    patch = inner[r0 : r1 + 1, c0 : c1 + 1]

    full = _full_support(mask)
    full_coords = np.argwhere(full)
    assert full_coords.shape[0] == 1, "sprite must have exactly one 3x3-supported pixel"
    marker_coords = np.argwhere(mask & (patch != BLUE))
    assert marker_coords.shape[0] == 1, "sprite must contain exactly one singleton non-blue marker"
    mr, mc = map(int, marker_coords[0])
    ir, ic = map(int, full_coords[0])
    assert (mr, mc) == (ir, ic), "marker must sit on the unique full-neighborhood pixel"
    marker_color = int(patch[mr, mc])
    assert marker_color in CYCLE, "marker color must be one of the cycle colors"
    assert set(int(v) for v in patch[mask]) <= {BLUE, RED, GREEN, YELLOW}

    return {
        "mask": mask.copy(),
        "top": r0 + 1,
        "left": c0 + 1,
        "interior_rc": (ir, ic),
        "marker_color": marker_color,
    }


def _analyze_sequence(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert max(g.shape) <= 30

    frame = int(g[0, 0])
    assert np.all(g[0, :] == frame) and np.all(g[-1, :] == frame)
    separators = _panel_separators(g, frame)
    assert len(separators) >= 5, "expected 4-5 shared-border panels"
    assert separators[0] == 0 and separators[-1] == w - 1
    strides = [b - a for a, b in zip(separators, separators[1:])]
    assert len(set(strides)) == 1
    stride = strides[0]
    panel_w = stride + 1
    panel_count = len(separators) - 1
    assert panel_count in (4, 5)

    panels = [g[:, c0 : c1 + 1] for c0, c1 in zip(separators, separators[1:])]
    infos = [_extract_sprite_info(panel) for panel in panels]

    template_mask = infos[0]["mask"]
    template_left = int(infos[0]["left"])
    interior_rc = tuple(int(v) for v in infos[0]["interior_rc"])
    for info in infos[1:]:
        assert int(info["left"]) == template_left, "sprite local column must stay fixed across frames"
        assert np.array_equal(info["mask"], template_mask), "sprite shape must be frame-invariant"
        assert tuple(int(v) for v in info["interior_rc"]) == interior_rc

    tops = [int(info["top"]) for info in infos]
    deltas = [b - a for a, b in zip(tops, tops[1:])]
    assert deltas and len(set(deltas)) == 1
    dy = deltas[0]
    next_top = tops[-1] + dy
    sprite_h, sprite_w = template_mask.shape
    assert dy != 0, "motion extrapolation must be necessary"
    assert 1 <= next_top <= h - 1 - sprite_h

    colors = [int(info["marker_color"]) for info in infos]
    cycle_offset, next_color = _fit_forward_cycle(colors)

    return {
        "frame": frame,
        "panel_h": h,
        "panel_w": panel_w,
        "panel_count": panel_count,
        "stride": stride,
        "sprite_mask": template_mask.copy(),
        "sprite_h": sprite_h,
        "sprite_w": sprite_w,
        "left": template_left,
        "tops": tops,
        "dy": dy,
        "next_top": next_top,
        "interior_rc": interior_rc,
        "marker_colors": colors,
        "cycle_offset": cycle_offset,
        "next_color": next_color,
    }


def _render_panel(
    panel_h: int,
    panel_w: int,
    frame: int,
    sprite_mask: np.ndarray,
    top: int,
    left: int,
    interior_rc: tuple[int, int],
    marker_color: int,
) -> np.ndarray:
    out = np.zeros((panel_h, panel_w), dtype=int)
    out[0, :] = frame
    out[-1, :] = frame
    out[:, 0] = frame
    out[:, -1] = frame
    rr, cc = np.where(sprite_mask)
    out[top + rr, left + cc] = BLUE
    ir, ic = interior_rc
    out[top + ir, left + ic] = marker_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Extrapolate the next panel by continuing the sprite's vertical motion and the marker's red-green-yellow cycle, then repaint the sprite's unique 3x3-supported interior pixel.

    Input structure:
    - The grid is a horizontal row of 4-5 equal-width panels sharing grey vertical border columns on a black background.
    - Each panel contains one 4-connected sprite with the same blue shape at the same local columns, while its top row shifts by a constant amount from panel to panel.
    - Inside the sprite, exactly one pixel has its full 3x3 neighborhood contained in the sprite, and that singleton marker pixel cycles through red, green, and yellow across the panels.

    Rule:
    - Read the panels from left to right as consecutive frames.
    - Extrapolate the sprite one more frame using the constant vertical offset of its bounding-box top row, keeping the local columns and shape unchanged.
    - Build one standalone output panel with the same border and background, draw the extrapolated sprite in blue, and recolor its unique full-neighborhood interior pixel to the next color in the red-green-yellow cycle.

    Key insight:
    - Two independent frame sequences must be continued together: the sprite's position determines where the output object goes, and the marker-color cycle determines how its uniquely supported interior pixel is repainted.
    """
    info = _analyze_sequence(np.asarray(grid, dtype=int))
    return _render_panel(
        int(info["panel_h"]),
        int(info["panel_w"]),
        int(info["frame"]),
        np.asarray(info["sprite_mask"], dtype=bool),
        int(info["next_top"]),
        int(info["left"]),
        tuple(int(v) for v in info["interior_rc"]),
        int(info["next_color"]),
    )


def _shape_options(max_h: int, max_w: int) -> list[np.ndarray]:
    options = [
        mask.copy()
        for mask in SPRITE_LIBRARY
        if mask.shape[0] <= max_h and mask.shape[1] <= max_w and (mask.shape != (3, 3) or int(mask.sum()) > 9)
    ]
    assert options
    return options


def _motion_options(panel_h: int, sprite_h: int, panel_count: int, sign: int) -> list[tuple[int, int]]:
    max_top = panel_h - 1 - sprite_h
    options: list[tuple[int, int]] = []
    for dy in (-2, -1, 1, 2):
        if (1 if dy > 0 else -1) != sign:
            continue
        for start in range(1, max_top + 1):
            positions = [start + i * dy for i in range(panel_count + 1)]
            if all(1 <= pos <= max_top for pos in positions):
                options.append((dy, start))
    return options


def _build_input(
    panel_h: int,
    panel_w: int,
    panel_count: int,
    sprite_mask: np.ndarray,
    left: int,
    start_top: int,
    dy: int,
    cycle_offset: int,
) -> np.ndarray:
    stride = panel_w - 1
    width = panel_count * stride + 1
    grid = np.zeros((panel_h, width), dtype=int)
    grid[0, :] = GREY
    grid[-1, :] = GREY
    for i in range(panel_count + 1):
        grid[:, i * stride] = GREY

    ir, ic = map(int, np.argwhere(_full_support(sprite_mask))[0])
    rr, cc = np.where(sprite_mask)
    for i in range(panel_count):
        top = start_top + i * dy
        x0 = i * stride + left
        grid[top + rr, x0 + cc] = BLUE
        grid[top + ir, x0 + ic] = CYCLE[(cycle_offset + i) % 3]
    return grid


def _self_check_case(grid: np.ndarray) -> None:
    info = _analyze_sequence(grid)
    out = main(grid)

    expected = _render_panel(
        int(info["panel_h"]),
        int(info["panel_w"]),
        int(info["frame"]),
        np.asarray(info["sprite_mask"], dtype=bool),
        int(info["next_top"]),
        int(info["left"]),
        tuple(int(v) for v in info["interior_rc"]),
        int(info["next_color"]),
    )
    assert np.array_equal(out, expected)
    assert int(info["dy"]) != 0
    assert len(set(int(c) for c in info["marker_colors"])) == 3
    assert int(info["panel_count"]) in (4, 5)

    wrong_position = _render_panel(
        int(info["panel_h"]),
        int(info["panel_w"]),
        int(info["frame"]),
        np.asarray(info["sprite_mask"], dtype=bool),
        int(info["tops"][-1]),
        int(info["left"]),
        tuple(int(v) for v in info["interior_rc"]),
        int(info["next_color"]),
    )
    assert not np.array_equal(out, wrong_position), "position extrapolation must affect the result"

    for wrong_color in CYCLE:
        if wrong_color == int(info["next_color"]):
            continue
        wrong = _render_panel(
            int(info["panel_h"]),
            int(info["panel_w"]),
            int(info["frame"]),
            np.asarray(info["sprite_mask"], dtype=bool),
            int(info["next_top"]),
            int(info["left"]),
            tuple(int(v) for v in info["interior_rc"]),
            wrong_color,
        )
        assert not np.array_equal(out, wrong), "marker-color extrapolation must affect the result"

    standalone = _extract_sprite_info(out)
    assert int(standalone["top"]) == int(info["next_top"])
    assert int(standalone["left"]) == int(info["left"])
    assert int(standalone["marker_color"]) == int(info["next_color"])
    assert np.count_nonzero(out == int(info["next_color"])) == 1
    assert np.count_nonzero(out == BLUE) == int(np.asarray(info["sprite_mask"], dtype=bool).sum()) - 1


def _gen_attempt(rng: random.Random, branch: int) -> np.ndarray | None:
    panel_count, panel_w = GEN_LAYOUTS[branch % len(GEN_LAYOUTS)]
    preferred_abs_dy = 2 if panel_count == 4 and (branch // (2 * len(GEN_LAYOUTS))) % 2 == 0 else 1
    panel_h_choices = [13] if preferred_abs_dy == 2 else list(GEN_HEIGHTS)
    panel_h = rng.choice(panel_h_choices)
    interior_h = panel_h - 2
    interior_w = panel_w - 2
    sign = 1 if branch % 2 == 0 else -1
    cycle_offset = (branch // len(GEN_LAYOUTS)) % 3

    shape_pool = _shape_options(interior_h, interior_w)
    if preferred_abs_dy == 2:
        tighter_pool = [mask for mask in shape_pool if mask.shape[0] == 3]
        shape_pool = tighter_pool or shape_pool
    preferred_pool = [mask for mask in shape_pool if mask.shape[0] + mask.shape[1] >= 7]
    sprite_mask = rng.choice(preferred_pool or shape_pool).copy()
    sprite_h, sprite_w = sprite_mask.shape

    left_choices = list(range(1, panel_w - sprite_w))
    if not left_choices:
        return None
    left = rng.choice(left_choices)

    motions = _motion_options(panel_h, sprite_h, panel_count, sign)
    if not motions:
        return None
    preferred = [motion for motion in motions if abs(motion[0]) == preferred_abs_dy]
    dy, start_top = rng.choice(preferred or motions)

    grid = _build_input(panel_h, panel_w, panel_count, sprite_mask, left, start_top, dy, cycle_offset)

    try:
        _self_check_case(grid)
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT

    rng = random.Random()
    for attempt in range(GEN_MAX_ATTEMPTS):
        branch = _GEN_CALL_COUNT + attempt
        grid = _gen_attempt(rng, branch)
        if grid is not None:
            _GEN_CALL_COUNT += 1
            return grid
    raise RuntimeError("failed to generate a valid frame-sequence extrapolation input")
