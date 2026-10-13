from __future__ import annotations

from collections import deque

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
DIRECTIONS = ("up", "down", "left", "right")
ARROW_SPANS = (5, 7)
TEMPLATE_SIZES = (2, 3)
GRID_MIN = 22
GRID_MAX = 30
MIN_GAP = 1
GENERATION_ATTEMPTS = 500
OBJECT_ATTEMPTS = 160
TEMPLATE_ATTEMPTS = 100
_RNG = np.random.default_rng()


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return out


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _monochrome_components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == bg:
                continue
            color = int(g[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            r0, c0, r1, c1 = _bbox(cells)
            patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
            for rr, cc in cells:
                patch[rr - r0, cc - c0] = color
            comps.append(
                {
                    "color": color,
                    "cells": cells,
                    "bbox": (r0, c0, r1, c1),
                    "patch": patch,
                }
            )
    return comps


def _detect_arrow(patch: np.ndarray, bg: int) -> str | None:
    mask = np.asarray(patch) != bg
    h, w = mask.shape

    if h == 3 and w >= 5 and int(mask.sum()) == w + 2:
        if np.all(mask[1, :]) and int(mask[0].sum()) == 1 and np.array_equal(mask[0], mask[2]):
            idx = int(np.flatnonzero(mask[0])[0])
            if idx == 1:
                return "left"
            if idx == w - 2:
                return "right"

    if w == 3 and h >= 5 and int(mask.sum()) == h + 2:
        if np.all(mask[:, 1]) and int(mask[:, 0].sum()) == 1 and np.array_equal(mask[:, 0], mask[:, 2]):
            idx = int(np.flatnonzero(mask[:, 0])[0])
            if idx == 1:
                return "up"
            if idx == h - 2:
                return "down"

    return None


def _sequence_start(center: tuple[int, int], direction: str, shape: tuple[int, int]) -> tuple[int, int]:
    cr, cc = center
    th, tw = shape
    if direction == "right":
        return cr - th // 2, cc
    if direction == "left":
        return cr - th // 2, cc - tw + 1
    if direction == "down":
        return cr, cc - tw // 2
    return cr - th + 1, cc - tw // 2


def _sequence_step(direction: str, shape: tuple[int, int]) -> tuple[int, int]:
    th, tw = shape
    if direction == "right":
        return 0, tw
    if direction == "left":
        return 0, -tw
    if direction == "down":
        return th, 0
    return -th, 0


def _paste_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int, transparent: int = BG) -> None:
    h, w = grid.shape
    ph, pw = patch.shape
    r0 = max(0, top)
    c0 = max(0, left)
    r1 = min(h, top + ph)
    c1 = min(w, left + pw)
    if r0 >= r1 or c0 >= c1:
        return
    src = patch[r0 - top : r1 - top, c0 - left : c1 - left]
    mask = src != transparent
    if np.any(mask):
        region = grid[r0:r1, c0:c1]
        region[mask] = src[mask]


def _arrow_patch(color: int, direction: str, span: int) -> np.ndarray:
    if direction in ("left", "right"):
        patch = np.full((3, span), BG, dtype=int)
        patch[1, :] = color
        idx = 1 if direction == "left" else span - 2
        patch[0, idx] = color
        patch[2, idx] = color
        return patch

    patch = np.full((span, 3), BG, dtype=int)
    patch[:, 1] = color
    idx = 1 if direction == "up" else span - 2
    patch[idx, 0] = color
    patch[idx, 2] = color
    return patch


def _arrow_bbox(center: tuple[int, int], direction: str, span: int) -> tuple[int, int, int, int]:
    cr, cc = center
    half = span // 2
    if direction in ("left", "right"):
        return cr - 1, cc - half, cr + 1, cc + half
    return cr - half, cc - 1, cr + half, cc + 1


def _sequence_bbox(
    center: tuple[int, int], direction: str, shape: tuple[int, int], copies: int
) -> tuple[int, int, int, int]:
    cr, cc = center
    th, tw = shape
    if direction == "right":
        top = cr - th // 2
        return top, cc, top + th - 1, cc + copies * tw - 1
    if direction == "left":
        top = cr - th // 2
        return top, cc - copies * tw + 1, top + th - 1, cc
    if direction == "down":
        left = cc - tw // 2
        return cr, left, cr + copies * th - 1, left + tw - 1
    left = cc - tw // 2
    return cr - copies * th + 1, left, cr, left + tw - 1


def _rect_in_bounds(rect: tuple[int, int, int, int], h: int, w: int) -> bool:
    r0, c0, r1, c1 = rect
    return 0 <= r0 <= r1 < h and 0 <= c0 <= c1 < w


def _rects_too_close(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], min_gap: int = MIN_GAP
) -> bool:
    return not (
        a[2] + min_gap < b[0]
        or b[2] + min_gap < a[0]
        or a[3] + min_gap < b[1]
        or b[3] + min_gap < a[1]
    )


def _collides_any(
    rect: tuple[int, int, int, int], reserved: list[tuple[int, int, int, int]], min_gap: int = MIN_GAP
) -> bool:
    return any(_rects_too_close(rect, other, min_gap=min_gap) for other in reserved)


def _center_ranges(
    h: int, w: int, direction: str, span: int, shape: tuple[int, int]
) -> tuple[int, int, int, int] | None:
    th, tw = shape
    copies = span + 2
    if direction in ("left", "right"):
        r_lo = max(1, th // 2)
        r_hi = min(h - 2, h - th + th // 2)
        if direction == "right":
            c_lo = span // 2
            c_hi = min(w - 1 - span // 2, w - copies * tw)
        else:
            c_lo = max(span // 2, copies * tw - 1)
            c_hi = w - 1 - span // 2
    else:
        c_lo = max(1, tw // 2)
        c_hi = min(w - 2, w - tw + tw // 2)
        if direction == "down":
            r_lo = span // 2
            r_hi = min(h - 1 - span // 2, h - copies * th)
        else:
            r_lo = max(span // 2, copies * th - 1)
            r_hi = h - 1 - span // 2

    if r_lo > r_hi or c_lo > c_hi:
        return None
    return int(r_lo), int(r_hi), int(c_lo), int(c_hi)


def _mask_connected(mask: np.ndarray) -> bool:
    cells = np.argwhere(mask)
    if cells.size == 0:
        return False
    h, w = mask.shape
    start = tuple(map(int, cells[0]))
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c, h, w):
            if mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == int(mask.sum())


def _sample_template_patch(rng: np.random.Generator, color: int) -> np.ndarray:
    size = int(rng.choice(TEMPLATE_SIZES))
    min_area = 3
    max_area = 3 if size == 2 else 7

    for _ in range(TEMPLATE_ATTEMPTS):
        target = int(rng.integers(min_area, max_area + 1))
        mask = np.zeros((size, size), dtype=bool)
        start = (int(rng.integers(size)), int(rng.integers(size)))
        mask[start] = True

        while int(mask.sum()) < target:
            frontier: set[tuple[int, int]] = set()
            for rr, cc in np.argwhere(mask):
                for nr, nc in _neighbors4(int(rr), int(cc), size, size):
                    if not mask[nr, nc]:
                        frontier.add((nr, nc))
            if not frontier:
                break
            choice = list(frontier)[int(rng.integers(len(frontier)))]
            mask[choice] = True

        if int(mask.sum()) != target:
            continue
        if not _mask_connected(mask):
            continue
        if not (mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue
        if bool(mask.all()):
            continue

        patch = np.full((size, size), BG, dtype=int)
        patch[mask] = color
        return patch

    if size == 2:
        patch = np.array([[color, color], [color, BG]], dtype=int)
    else:
        patch = np.array(
            [
                [color, color, BG],
                [BG, color, color],
                [color, BG, color],
            ],
            dtype=int,
        )
    if rng.random() < 0.5:
        patch = np.flipud(patch)
    if rng.random() < 0.5:
        patch = np.fliplr(patch)
    return patch


def _sample_template_bbox(
    rng: np.random.Generator,
    h: int,
    w: int,
    shape: tuple[int, int],
    blocked: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int] | None:
    th, tw = shape
    for _ in range(80):
        side = str(rng.choice(("top", "bottom", "left", "right")))
        if side == "top":
            r0 = 0
            c0 = int(rng.integers(0, w - tw + 1))
        elif side == "bottom":
            r0 = h - th
            c0 = int(rng.integers(0, w - tw + 1))
        elif side == "left":
            r0 = int(rng.integers(0, h - th + 1))
            c0 = 0
        else:
            r0 = int(rng.integers(0, h - th + 1))
            c0 = w - tw
        rect = (r0, c0, r0 + th - 1, c0 + tw - 1)
        if not _collides_any(rect, blocked):
            return rect
    return None


def _draw_component(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    region = grid[top : top + patch.shape[0], left : left + patch.shape[1]]
    mask = patch != BG
    region[mask] = patch[mask]


def _build_expected(inp: np.ndarray, tasks: list[dict[str, object]]) -> np.ndarray:
    out = np.array(inp, copy=True)
    for task in tasks:
        arrow_bbox = task["arrow_bbox"]
        arrow_patch = task["arrow_patch"]
        assert isinstance(arrow_bbox, tuple)
        assert isinstance(arrow_patch, np.ndarray)
        r0, c0, _r1, _c1 = arrow_bbox
        region = out[r0 : r0 + arrow_patch.shape[0], c0 : c0 + arrow_patch.shape[1]]
        region[arrow_patch != BG] = BG

    for task in tasks:
        direction = str(task["direction"])
        center = task["center"]
        template_patch = task["template_patch"]
        arrow_patch = task["arrow_patch"]
        assert isinstance(center, tuple)
        assert isinstance(template_patch, np.ndarray)
        assert isinstance(arrow_patch, np.ndarray)
        copies = int(np.count_nonzero(arrow_patch))
        start_r, start_c = _sequence_start(center, direction, template_patch.shape)
        step_r, step_c = _sequence_step(direction, template_patch.shape)
        for idx in range(copies):
            _paste_patch(out, template_patch, start_r + idx * step_r, start_c + idx * step_c)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Replace each colored arrow with a direction-aligned run of its same-color template.

    Input structure:
    - Black background surrounds several monochrome arrow components and several same-color template components.
    - Each arrow is a 4-connected cardinal arrow: a one-pixel shaft plus a three-pixel-wide triangular head, pointing up, down, left, or right.
    - For every arrow color, there is a separate 2x2 or 3x3 template patch of the same color near the grid edge.

    Rule:
    - For each color, identify the small square template patch and every arrow of that color.
    - Read the arrow's pointing direction and count its total pixels N.
    - Remove the arrow from the grid.
    - Starting from the arrow's center point, place N tightly adjacent copies of the unchanged template along the arrow direction.
    - Keep the original template patch in place as a reference.

    Key insight:
    - The arrow encodes two things at once: its orientation gives the placement direction, and its pixel count gives the number of template copies to draw.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = int(np.bincount(g.ravel()).argmax())
    comps = _monochrome_components(g, bg)

    by_color: dict[int, list[dict[str, object]]] = {}
    for comp in comps:
        color = int(comp["color"])
        by_color.setdefault(color, []).append(comp)

    templates: dict[int, np.ndarray] = {}
    arrows: list[tuple[int, dict[str, object], str]] = []

    for color, group in by_color.items():
        template_candidates: list[dict[str, object]] = []
        color_arrows: list[tuple[dict[str, object], str]] = []
        for comp in group:
            patch = np.asarray(comp["patch"], dtype=int)
            direction = _detect_arrow(patch, bg)
            if direction is not None:
                color_arrows.append((comp, direction))
            elif patch.shape[0] == patch.shape[1] and patch.shape[0] in TEMPLATE_SIZES:
                template_candidates.append(comp)

        if color_arrows:
            assert template_candidates, f"missing template for arrow color {color}"
            template_comp = min(
                template_candidates,
                key=lambda comp: (
                    np.asarray(comp["patch"]).shape[0],
                    int(comp["bbox"][0]),
                    int(comp["bbox"][1]),
                ),
            )
            templates[color] = np.asarray(template_comp["patch"], dtype=int)
            for comp, direction in color_arrows:
                arrows.append((color, comp, direction))

    assert arrows, "expected at least one arrow"

    out = np.array(g, copy=True)
    for _color, comp, _direction in arrows:
        for r, c in comp["cells"]:
            out[r, c] = bg

    for color, comp, direction in arrows:
        template_patch = templates[color]
        r0, c0, r1, c1 = comp["bbox"]
        center = ((int(r0) + int(r1)) // 2, (int(c0) + int(c1)) // 2)
        start_r, start_c = _sequence_start(center, direction, template_patch.shape)
        step_r, step_c = _sequence_step(direction, template_patch.shape)
        copies = len(comp["cells"])
        for idx in range(copies):
            _paste_patch(out, template_patch, start_r + idx * step_r, start_c + idx * step_c, transparent=bg)

    return out


def _sample_object(
    rng: np.random.Generator,
    h: int,
    w: int,
    color: int,
    reserved: list[tuple[int, int, int, int]],
) -> dict[str, object] | None:
    for _ in range(OBJECT_ATTEMPTS):
        template_patch = _sample_template_patch(rng, color)
        direction = str(rng.choice(DIRECTIONS))
        feasible_spans = [
            span for span in ARROW_SPANS if _center_ranges(h, w, direction, span, template_patch.shape) is not None
        ]
        if not feasible_spans:
            continue
        span = int(rng.choice(feasible_spans))
        ranges = _center_ranges(h, w, direction, span, template_patch.shape)
        assert ranges is not None
        r_lo, r_hi, c_lo, c_hi = ranges
        center = (
            int(rng.integers(r_lo, r_hi + 1)),
            int(rng.integers(c_lo, c_hi + 1)),
        )
        arrow_bbox = _arrow_bbox(center, direction, span)
        seq_bbox = _sequence_bbox(center, direction, template_patch.shape, span + 2)
        if not (_rect_in_bounds(arrow_bbox, h, w) and _rect_in_bounds(seq_bbox, h, w)):
            continue
        if _collides_any(arrow_bbox, reserved) or _collides_any(seq_bbox, reserved):
            continue

        blocked = reserved + [arrow_bbox, seq_bbox]
        template_bbox = _sample_template_bbox(rng, h, w, template_patch.shape, blocked)
        if template_bbox is None:
            continue

        arrow_patch = _arrow_patch(color, direction, span)
        return {
            "color": color,
            "direction": direction,
            "center": center,
            "arrow_patch": arrow_patch,
            "arrow_bbox": arrow_bbox,
            "template_patch": template_patch,
            "template_bbox": template_bbox,
            "sequence_bbox": seq_bbox,
        }
    return None


def _attempt_generate(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(GRID_MIN, GRID_MAX + 1))
    w = int(rng.integers(GRID_MIN, GRID_MAX + 1))
    max_objects = 3 if min(h, w) >= 26 else 2
    n_objects = int(rng.integers(1, max_objects + 1))
    colors = [int(c) for c in rng.choice(PALETTE, size=n_objects, replace=False)]

    reserved: list[tuple[int, int, int, int]] = []
    tasks: list[dict[str, object]] = []
    for color in colors:
        task = _sample_object(rng, h, w, color, reserved)
        if task is None:
            return None
        reserved.append(task["arrow_bbox"])
        reserved.append(task["template_bbox"])
        reserved.append(task["sequence_bbox"])
        tasks.append(task)

    inp = np.full((h, w), BG, dtype=int)
    for task in tasks:
        tr0, tc0, _tr1, _tc1 = task["template_bbox"]
        ar0, ac0, _ar1, _ac1 = task["arrow_bbox"]
        _draw_component(inp, task["template_patch"], tr0, tc0)
        _draw_component(inp, task["arrow_patch"], ar0, ac0)

    if int(np.bincount(inp.ravel()).argmax()) != BG:
        return None

    expected = _build_expected(inp, tasks)
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _attempt_generate(_RNG)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for puzzle 7")
