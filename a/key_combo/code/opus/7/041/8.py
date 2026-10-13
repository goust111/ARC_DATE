from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
MARKER_PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
RANK_PALETTE = (2, 1, 3, 4, 6, 7, 8, 9)

GEN_ATTEMPTS = 500
RECT_COUNT_OPTIONS = (2, 3)


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _find_gray_rectangles(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    rects: list[tuple[int, int, int, int]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != GREY:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == GREY:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            area = (r1 - r0 + 1) * (c1 - c0 + 1)
            assert len(coords) == area, "grey component must be a solid rectangle"
            assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == GREY)
            rects.append((r0, c0, r1, c1))

    rects.sort()
    assert rects, "expected at least one grey rectangle"
    return rects


def _quadrant_specs(r0: int, c0: int, r1: int, c1: int) -> list[tuple[tuple[int, int], slice, slice, int]]:
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    assert height >= 2 and width >= 2

    # Odd middle row goes to the upper half; odd middle column goes to the left half.
    r_cut = r0 + (height + 1) // 2
    c_cut = c0 + (width + 1) // 2
    specs = [
        ((r0 - 1, c0 - 1), slice(r0, r_cut), slice(c0, c_cut), (r_cut - r0) * (c_cut - c0)),
        ((r0 - 1, c1 + 1), slice(r0, r_cut), slice(c_cut, c1 + 1), (r_cut - r0) * (c1 + 1 - c_cut)),
        ((r1 + 1, c0 - 1), slice(r_cut, r1 + 1), slice(c0, c_cut), (r1 + 1 - r_cut) * (c_cut - c0)),
        ((r1 + 1, c1 + 1), slice(r_cut, r1 + 1), slice(c_cut, c1 + 1), (r1 + 1 - r_cut) * (c1 + 1 - c_cut)),
    ]
    assert all(area > 0 for _, _, _, area in specs)
    return specs


def _quadrant_areas(height: int, width: int) -> tuple[int, int, int, int]:
    r_cut = (height + 1) // 2
    c_cut = (width + 1) // 2
    return (
        r_cut * c_cut,
        r_cut * (width - c_cut),
        (height - r_cut) * c_cut,
        (height - r_cut) * (width - c_cut),
    )


def main(grid: np.ndarray) -> np.ndarray:
    """Split each grey rectangle into four marker-gated quadrants and globally recolor the active ones by descending area.

    Input structure:
    - The background is black.
    - The grid contains 2-3 solid grey rectangles.
    - Each rectangle has four diagonal outside-corner marker positions; each such
      position is either black (inactive) or a non-black, non-grey colored pixel (active).

    Rule:
    - Detect every solid grey rectangle and inspect its four diagonal marker positions.
    - Erase all visible markers to black.
    - Split each rectangle into top-left, top-right, bottom-left, and bottom-right quadrants;
      when a side length is odd, the extra row stays in the top half and the extra column stays in the left half.
    - Keep inactive quadrants grey, but collect all active quadrants across all rectangles,
      rank them by area from largest to smallest, and repaint them with
      [red, blue, green, yellow, magenta, orange, teal, maroon] in that order.

    Key insight:
    - The diagonal markers no longer encode output colors directly; they only provide binary on/off switches,
      and the visible colors are assigned later by a single global area ranking over every active quadrant.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = g.copy()
    rects = _find_gray_rectangles(g)
    active_quadrants: list[tuple[int, slice, slice]] = []
    all_marker_positions: set[tuple[int, int]] = set()
    visible_marker_positions: set[tuple[int, int]] = set()
    h, w = g.shape

    for r0, c0, r1, c1 in rects:
        active_here = 0
        for marker_pos, rs, cs, area in _quadrant_specs(r0, c0, r1, c1):
            mr, mc = marker_pos
            assert 0 <= mr < h and 0 <= mc < w
            assert marker_pos not in all_marker_positions
            all_marker_positions.add(marker_pos)

            value = int(g[mr, mc])
            assert value == BLACK or value != GREY
            out[mr, mc] = BLACK
            if value != BLACK:
                visible_marker_positions.add(marker_pos)
                active_quadrants.append((area, rs, cs))
                active_here += 1
        assert 0 < active_here < 4, "each rectangle needs at least one active and one inactive quadrant"

    colored_pixels = {
        (int(r), int(c))
        for r, c in np.argwhere((g != BLACK) & (g != GREY))
    }
    assert colored_pixels == visible_marker_positions

    areas = [area for area, _, _ in active_quadrants]
    assert len(active_quadrants) <= len(RANK_PALETTE)
    assert len(set(areas)) == len(areas), "active quadrant areas must be globally unique"

    for rank, (_area, rs, cs) in enumerate(sorted(active_quadrants, key=lambda item: item[0], reverse=True)):
        out[rs, cs] = RANK_PALETTE[rank]

    return out


def _build_option_pool(max_side: int) -> list[dict]:
    pool: list[dict] = []
    for height in range(3, max_side + 1):
        for width in range(3, max_side + 1):
            areas = _quadrant_areas(height, width)
            for mask in range(1, 16):
                active = tuple(idx for idx in range(4) if mask & (1 << idx))
                if len(active) == 4:
                    continue
                active_areas = tuple(areas[idx] for idx in active)
                if len(set(active_areas)) != len(active_areas):
                    continue
                pool.append(
                    {
                        "h": height,
                        "w": width,
                        "active": active,
                        "areas": active_areas,
                    }
                )
    return pool


OPTION_POOL = _build_option_pool(7)


def _choose_specs(rect_count: int) -> list[dict] | None:
    min_active_total = 3 if rect_count == 2 else 4
    max_side = 7 if rect_count == 2 else 6
    pool = [opt for opt in OPTION_POOL if opt["h"] <= max_side and opt["w"] <= max_side]

    for _ in range(300):
        chosen: list[dict] = []
        used_shapes: set[tuple[int, int]] = set()
        used_areas: set[int] = set()
        active_total = 0

        for rect_idx in range(rect_count):
            remaining = rect_count - rect_idx - 1
            candidates = []
            for opt in pool:
                shape = (opt["h"], opt["w"])
                if shape in used_shapes:
                    continue
                if any(area in used_areas for area in opt["areas"]):
                    continue
                new_total = active_total + len(opt["active"])
                if new_total + remaining > 8:
                    continue
                if new_total + 3 * remaining < min_active_total:
                    continue
                candidates.append(opt)
            if not candidates:
                break

            need_more = min_active_total - active_total
            if need_more > remaining + 1:
                richer = [opt for opt in candidates if len(opt["active"]) >= 2]
                if richer:
                    candidates = richer

            picked = random.choice(candidates)
            chosen.append(
                {
                    "h": picked["h"],
                    "w": picked["w"],
                    "active": picked["active"],
                    "areas": picked["areas"],
                }
            )
            used_shapes.add((picked["h"], picked["w"]))
            used_areas.update(picked["areas"])
            active_total += len(picked["active"])

        if len(chosen) != rect_count:
            continue
        if not (min_active_total <= active_total <= 8):
            continue
        if not any(spec["h"] % 2 == 1 or spec["w"] % 2 == 1 for spec in chosen):
            continue
        return chosen

    return None


def _reserve_zone(height: int, width: int, r0: int, c0: int, rect_h: int, rect_w: int) -> tuple[int, int, int, int]:
    return (
        max(0, r0 - 2),
        max(0, c0 - 2),
        min(height, r0 + rect_h + 2),
        min(width, c0 + rect_w + 2),
    )


def _place_rectangles(height: int, width: int, specs: list[dict]) -> list[tuple[int, int]] | None:
    occupied = np.zeros((height, width), dtype=bool)
    placements: list[tuple[int, int] | None] = [None] * len(specs)
    order = sorted(range(len(specs)), key=lambda idx: (specs[idx]["h"] + 4) * (specs[idx]["w"] + 4), reverse=True)

    def search(depth: int) -> bool:
        if depth == len(order):
            return True

        idx = order[depth]
        rect_h = specs[idx]["h"]
        rect_w = specs[idx]["w"]
        candidates: list[tuple[int, int]] = []
        for r0 in range(1, height - rect_h):
            for c0 in range(1, width - rect_w):
                rr0, cc0, rr1, cc1 = _reserve_zone(height, width, r0, c0, rect_h, rect_w)
                if not occupied[rr0:rr1, cc0:cc1].any():
                    candidates.append((r0, c0))
        random.shuffle(candidates)

        for r0, c0 in candidates[:160]:
            rr0, cc0, rr1, cc1 = _reserve_zone(height, width, r0, c0, rect_h, rect_w)
            occupied[rr0:rr1, cc0:cc1] = True
            placements[idx] = (r0, c0)
            if search(depth + 1):
                return True
            placements[idx] = None
            occupied[rr0:rr1, cc0:cc1] = False
        return False

    if not search(0):
        return None
    return [(int(r0), int(c0)) for r0, c0 in placements if r0 is not None and c0 is not None]


def _render_input(height: int, width: int, specs: list[dict], placements: list[tuple[int, int]]) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    for spec, (r0, c0) in zip(specs, placements):
        rect_h = spec["h"]
        rect_w = spec["w"]
        r1 = r0 + rect_h - 1
        c1 = c0 + rect_w - 1
        grid[r0 : r1 + 1, c0 : c1 + 1] = GREY

        colors = random.sample(MARKER_PALETTE, len(spec["active"]))
        active_color_map = {idx: color for idx, color in zip(spec["active"], colors)}
        for idx, (marker_pos, _rs, _cs, _area) in enumerate(_quadrant_specs(r0, c0, r1, c1)):
            mr, mc = marker_pos
            grid[mr, mc] = active_color_map.get(idx, BLACK)

    return grid


def _gen_attempt() -> np.ndarray | None:
    rect_count = random.choice(RECT_COUNT_OPTIONS)
    specs = _choose_specs(rect_count)
    if specs is None:
        return None

    min_side = 15 if rect_count == 2 else 16
    max_side = 20 if rect_count == 2 else 22
    low_h = max(min_side, max(spec["h"] for spec in specs) + 3)
    low_w = max(min_side, max(spec["w"] for spec in specs) + 3)
    if low_h > max_side or low_w > max_side:
        return None

    for _ in range(40):
        height = random.randint(low_h, max_side)
        width = random.randint(low_w, max_side)
        placements = _place_rectangles(height, width, specs)
        if placements is None:
            continue

        grid = _render_input(height, width, specs, placements)
        try:
            out = main(grid)
        except Exception:
            continue
        if np.array_equal(out, grid):
            continue
        return grid

    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
