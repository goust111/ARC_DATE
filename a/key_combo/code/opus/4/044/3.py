from __future__ import annotations

from importlib.machinery import SourceFileLoader
import random
from typing import Iterable

import numpy as np


def _load_arc_utils():
    path = "/home/lixinyu/snap/new/solvers/arc_utils.py"
    try:
        return SourceFileLoader("_arc_utils_shared_044_3", path).load_module()
    except Exception:
        return None


_ARC_UTILS = _load_arc_utils()


if _ARC_UTILS is not None:
    Color = _ARC_UTILS.Color
    bbox_coords = _ARC_UTILS.bbox_coords
    find_components = _ARC_UTILS.find_components
else:
    class Color:
        BLACK = 0
        BLUE = 1
        RED = 2
        GREEN = 3
        YELLOW = 4
        ORANGE = 7
        TEAL = 8

    def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        return int(r0), int(c0), int(r1), int(c1)

    def find_components(
        grid: np.ndarray,
        *,
        bg: int | None = None,
        connectivity: int = 4,
        monochromatic: bool = False,
        return_format: str = "dict",
        **_: object,
    ) -> list[dict]:
        if bg is None:
            values, counts = np.unique(grid, return_counts=True)
            bg = int(values[counts.argmax()])
        assert connectivity == 4
        assert not monochromatic
        assert return_format == "dict"

        h, w = grid.shape
        seen = np.zeros((h, w), dtype=bool)
        out: list[dict] = []
        for r in range(h):
            for c in range(w):
                if seen[r, c] or int(grid[r, c]) == bg:
                    continue
                stack = [(r, c)]
                seen[r, c] = True
                coords = []
                while stack:
                    rr, cc = stack.pop()
                    coords.append((rr, cc))
                    for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                        if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) != bg:
                            seen[nr, nc] = True
                            stack.append((nr, nc))
                arr = np.array(coords, dtype=int)
                out.append({"coords": arr, "bbox": bbox_coords(arr), "area": int(arr.shape[0]), "color": None})
        return out


BG = Color.BLACK
INDICATOR = Color.RED
BODY_COLORS = (Color.BLUE, Color.GREEN, Color.YELLOW, Color.ORANGE)
ENVELOPE = Color.TEAL
SIDES = ("top", "bottom", "left", "right")
GRID_H_RANGE = (22, 28)
GRID_W_RANGE = (24, 28)
RECT_COUNT_WEIGHTS = ((2, 2), (3, 3), (4, 2))
MAX_GEN_ATTEMPTS = 400
MIN_MARGIN = 6


def _segments(base_len: int) -> list[tuple[int, int]]:
    assert base_len >= 1
    lo = 0
    hi = base_len - 1
    out = []
    if base_len % 2 == 1:
        while lo <= hi:
            out.append((lo, hi))
            lo += 1
            hi -= 1
    else:
        step = 0
        while lo <= hi:
            out.append((lo, hi))
            if step % 2 == 0:
                hi -= 1
            else:
                lo += 1
            step += 1
    return out


def _indicator_offset(size: int) -> int:
    return (size - 1) // 2


def _indicator_coord(rect: dict) -> tuple[int, int]:
    r0, c0, r1, c1 = rect["bbox"]
    if rect["side"] == "top":
        return r0, c0 + _indicator_offset(rect["w"])
    if rect["side"] == "bottom":
        return r1, c0 + _indicator_offset(rect["w"])
    if rect["side"] == "left":
        return r0 + _indicator_offset(rect["h"]), c0
    return r0 + _indicator_offset(rect["h"]), c1


def _envelope_cells(rect: dict) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = rect["bbox"]
    side = rect["side"]
    cells: list[tuple[int, int]] = []
    if side in {"top", "bottom"}:
        for depth, (lo, hi) in enumerate(_segments(rect["w"]), start=1):
            rr = r0 - depth if side == "top" else r1 + depth
            for cc in range(c0 + lo, c0 + hi + 1):
                cells.append((rr, cc))
    else:
        for depth, (lo, hi) in enumerate(_segments(rect["h"]), start=1):
            cc = c0 - depth if side == "left" else c1 + depth
            for rr in range(r0 + lo, r0 + hi + 1):
                cells.append((rr, cc))
    return cells


def _reserved_cells(rect: dict) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = rect["bbox"]
    rect_cells = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
    return rect_cells | set(_envelope_cells(rect))


def _bounds_ok(shape: tuple[int, int], cells: Iterable[tuple[int, int]]) -> bool:
    h, w = shape
    return all(0 <= r < h and 0 <= c < w for r, c in cells)


def _extract_rectangles(grid: np.ndarray) -> list[dict]:
    comps = find_components(grid, bg=BG, connectivity=4, monochromatic=False, return_format="dict")
    rects: list[dict] = []
    for comp in comps:
        coords = np.asarray(comp["coords"], dtype=int)
        r0, c0, r1, c1 = bbox_coords(coords)
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        colors = set(int(v) for v in np.unique(patch))
        assert BG not in colors, "rectangle bbox must be solid"
        assert INDICATOR in colors, "rectangle must contain one red indicator"
        body_colors = colors - {INDICATOR}
        assert len(body_colors) == 1, "rectangle body must use a single non-red color"
        body = body_colors.pop()

        red_local = np.argwhere(patch == INDICATOR)
        assert red_local.shape[0] == 1, "rectangle must contain exactly one red indicator pixel"
        rr = int(red_local[0, 0]) + r0
        cc = int(red_local[0, 1]) + c0
        h = r1 - r0 + 1
        w = c1 - c0 + 1

        candidates = []
        if rr == r0:
            assert cc == c0 + _indicator_offset(w)
            candidates.append("top")
        if rr == r1:
            assert cc == c0 + _indicator_offset(w)
            candidates.append("bottom")
        if cc == c0:
            assert rr == r0 + _indicator_offset(h)
            candidates.append("left")
        if cc == c1:
            assert rr == r0 + _indicator_offset(h)
            candidates.append("right")
        assert len(candidates) == 1, "indicator must lie on exactly one rectangle side"

        rects.append(
            {
                "bbox": (r0, c0, r1, c1),
                "h": h,
                "w": w,
                "area": h * w,
                "body": int(body),
                "side": candidates[0],
                "indicator": (rr, cc),
            }
        )
    return rects


def _sort_key(rect: dict) -> tuple[int, int, int]:
    r0, c0, _r1, _c1 = rect["bbox"]
    area = int(rect.get("area", int(rect["h"]) * int(rect["w"])))
    return (-area, int(c0), int(r0))


def _oracle_from_rects(grid: np.ndarray, rects: list[dict]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for rect in sorted(rects, key=_sort_key):
        rr, cc = rect["indicator"]
        out[rr, cc] = int(rect["body"])
        for r, c in _envelope_cells(rect):
            if out[r, c] == BG:
                out[r, c] = ENVELOPE
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each red border indicator with the rectangle color and grow a teal half-diamond outward from that side.

    Input structure:
    - The grid has a black background with 2 to 4 separated solid rectangles in distinct body colors.
    - Each rectangle uses one color from blue, green, yellow, or orange, and has exactly one red pixel on the center of one edge.
    - The red pixel marks one of the four rectangle sides and therefore a unique outward growth direction.

    Rule:
    - Parse each multicolor rectangle, identify which side contains its red indicator, and treat that full side as the half-diamond base.
    - Replace the red indicator with the rectangle's own color.
    - Process rectangles by decreasing area, breaking ties by leftmost top-left corner and then uppermost top-left corner; for each one, draw teal rows or columns outward from the indicated side, shrinking the span toward a one-cell tip, with even-length bases alternating the extra shrink starting on the right or bottom so the tip biases left or up.
    - Only paint teal onto black background cells; any non-background cell blocks that envelope pixel.

    Key insight:
    - The same red pixel simultaneously chooses the base side and the outward orientation of the shrinking half-diamond envelope.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    rects = _extract_rectangles(g)
    assert rects, "expected at least one rectangle component"
    return _oracle_from_rects(g, rects)


def _sample_count() -> int:
    values = [count for count, weight in RECT_COUNT_WEIGHTS for _ in range(weight)]
    return random.choice(values)


def _sample_sides(count: int) -> list[str]:
    vertical = ["top", "bottom"]
    horizontal = ["left", "right"]
    if count == 2:
        return [random.choice(vertical), random.choice(horizontal)]
    if count == 3:
        picked = [random.choice(vertical), random.choice(horizontal)]
        remaining = [side for side in SIDES if side not in picked]
        picked.append(random.choice(remaining))
        random.shuffle(picked)
        return picked
    picked = list(SIDES)
    random.shuffle(picked)
    return picked


def _sample_dim(side: str, parity: str | None) -> tuple[int, int]:
    heights = [3, 4, 5]
    widths = [3, 4, 5, 6, 7]
    if side in {"top", "bottom"}:
        if parity is not None:
            widths = [w for w in widths if (w % 2 == 0) == (parity == "even")]
        return random.choice(heights), random.choice(widths)
    if parity is not None:
        heights = [h for h in heights if (h % 2 == 0) == (parity == "even")]
    return random.choice(heights), random.choice(widths)


def _depth(rect: dict) -> int:
    return len(_segments(rect["w"] if rect["side"] in {"top", "bottom"} else rect["h"]))


def _candidate_positions(shape: tuple[int, int], rect: dict) -> list[tuple[int, int]]:
    h, w = shape
    rect_h = rect["h"]
    rect_w = rect["w"]
    positions = []
    for r0 in range(MIN_MARGIN, h - MIN_MARGIN - rect_h + 1):
        for c0 in range(MIN_MARGIN, w - MIN_MARGIN - rect_w + 1):
            r1 = r0 + rect_h - 1
            c1 = c0 + rect_w - 1
            probe = dict(rect)
            probe["bbox"] = (r0, c0, r1, c1)
            if not _bounds_ok(shape, _reserved_cells(probe)):
                continue
            positions.append((r0, c0))
    random.shuffle(positions)
    return positions


def _dilate(cells: Iterable[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    out.add((rr, cc))
    return out


def _place_rectangles(shape: tuple[int, int], specs: list[dict]) -> list[dict] | None:
    blocked: set[tuple[int, int]] = set()
    placed: list[dict] = []
    for spec in sorted(specs, key=lambda item: (-max(item["h"], item["w"]), -_depth(item))):
        candidates = []
        for r0, c0 in _candidate_positions(shape, spec):
            probe = dict(spec)
            probe["bbox"] = (r0, c0, r0 + spec["h"] - 1, c0 + spec["w"] - 1)
            reserved = _reserved_cells(probe)
            if reserved & blocked:
                continue
            candidates.append(probe)
        if not candidates:
            return None
        chosen = random.choice(candidates)
        blocked |= _dilate(_reserved_cells(chosen), shape)
        placed.append(chosen)
    return placed


def _build_input(shape: tuple[int, int], rects: list[dict]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for rect in rects:
        r0, c0, r1, c1 = rect["bbox"]
        rect["area"] = int(rect["h"]) * int(rect["w"])
        grid[r0 : r1 + 1, c0 : c1 + 1] = int(rect["body"])
        rr, cc = _indicator_coord(rect)
        rect["indicator"] = (rr, cc)
        grid[rr, cc] = INDICATOR
    return grid


def _validate_generated(shape: tuple[int, int], rects: list[dict], grid: np.ndarray) -> bool:
    if grid.shape != shape:
        return False
    if int(np.count_nonzero(grid == INDICATOR)) != len(rects):
        return False

    by_bbox = {tuple(rect["bbox"]): rect for rect in rects}
    parsed = _extract_rectangles(grid)
    if len(parsed) != len(rects):
        return False
    for parsed_rect in parsed:
        ref = by_bbox.get(tuple(parsed_rect["bbox"]))
        if ref is None:
            return False
        if parsed_rect["body"] != ref["body"] or parsed_rect["side"] != ref["side"]:
            return False
        if parsed_rect["indicator"] != ref["indicator"]:
            return False

    reserved_sets = [set(_reserved_cells(rect)) for rect in rects]
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            if reserved_sets[i] & reserved_sets[j]:
                return False

    expected = _oracle_from_rects(grid, rects)
    actual = main(grid)
    if np.array_equal(actual, grid):
        return False
    if not np.array_equal(actual, expected):
        return False

    side_kinds = {rect["side"] in {"top", "bottom"} for rect in rects}
    if len(rects) >= 2 and len(side_kinds) < 2:
        return False
    parities = {
        (rect["w"] if rect["side"] in {"top", "bottom"} else rect["h"]) % 2
        for rect in rects
    }
    if len(rects) >= 2 and len(parities) < 2:
        return False
    return True


def _attempt_generate() -> np.ndarray | None:
    shape = (
        random.randint(GRID_H_RANGE[0], GRID_H_RANGE[1]),
        random.randint(GRID_W_RANGE[0], GRID_W_RANGE[1]),
    )
    count = _sample_count()
    sides = _sample_sides(count)
    colors = random.sample(list(BODY_COLORS), count)

    specs: list[dict] = []
    parity_targets = ["even", "odd"] + [None] * max(0, count - 2)
    random.shuffle(parity_targets)
    for color, side, parity in zip(colors, sides, parity_targets, strict=False):
        rect_h, rect_w = _sample_dim(side, parity)
        specs.append({"body": int(color), "side": side, "h": rect_h, "w": rect_w})

    placed = _place_rectangles(shape, specs)
    if placed is None:
        return None
    grid = _build_input(shape, placed)
    if _validate_generated(shape, placed, grid):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid half-diamond envelope input")
