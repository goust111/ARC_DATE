from __future__ import annotations

import random
from collections import defaultdict, deque

import numpy as np


BG = 0
GREY = 5
ARROW_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
MIN_MARGIN = 4
GEN_RETRIES = 400
PLACEMENT_RETRIES = 300

POINTING_SPECS = {
    "right": {"delta": (0, 1), "source": "left", "wings": ((-1, -1), (1, -1))},
    "left": {"delta": (0, -1), "source": "right", "wings": ((-1, 1), (1, 1))},
    "up": {"delta": (-1, 0), "source": "down", "wings": ((1, -1), (1, 1))},
    "down": {"delta": (1, 0), "source": "up", "wings": ((-1, -1), (-1, 1))},
}

SIDE_TO_POINTING = {
    "left": "right",
    "right": "left",
    "up": "down",
    "down": "up",
}


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= int(r) < int(shape[0]) and 0 <= int(c) < int(shape[1])


def _dominant_color(grid: np.ndarray) -> int:
    counts = np.bincount(np.asarray(grid, dtype=int).ravel(), minlength=10)
    return int(np.argmax(counts))


def _find_rectangles(grid: np.ndarray, bg: int) -> tuple[list[dict], np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    rect_ids = np.full((h, w), -1, dtype=int)
    rectangles: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            queue = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                cells.append((int(rr), int(cc)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if not _in_bounds(grid.shape, nr, nc):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    queue.append((nr, nc))

            rows = [rr for rr, _ in cells]
            cols = [cc for _, cc in cells]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            hh = r1 - r0 + 1
            ww = c1 - c0 + 1
            if 2 <= hh <= 4 and 2 <= ww <= 4 and len(cells) == hh * ww:
                rect_id = len(rectangles)
                for rr, cc in cells:
                    rect_ids[rr, cc] = rect_id
                rectangles.append(
                    {
                        "id": rect_id,
                        "color": color,
                        "r0": r0,
                        "c0": c0,
                        "h": hh,
                        "w": ww,
                    }
                )

    return rectangles, rect_ids


def _ray_target(grid: np.ndarray, rect_ids: np.ndarray, tip: tuple[int, int], delta: tuple[int, int], bg: int) -> int | None:
    r, c = tip
    dr, dc = delta
    for dist in (2, 3):
        ok = True
        for step in range(1, dist):
            rr = r + dr * step
            cc = c + dc * step
            if not _in_bounds(grid.shape, rr, cc) or int(grid[rr, cc]) != bg:
                ok = False
                break
        if not ok:
            continue
        rr = r + dr * dist
        cc = c + dc * dist
        if _in_bounds(grid.shape, rr, cc) and int(rect_ids[rr, cc]) >= 0:
            return int(rect_ids[rr, cc])
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repaint each grey rectangle from the colors of the adjacent arrows pointing at it, then erase the arrows.

    Input structure:
    - The background is black.
    - Several small solid monochrome rectangles sit on the canvas and serve as the recolorable target objects.
    - Around them are sparse 3-pixel colored arrows, each forming a left, right, up, or down V-shape with a black gap before its target rectangle.

    Rule:
    - Detect each arrow's pointing direction from its 3-pixel shape, then follow that ray through the 1-2 black-cell gap to the nearest rectangle.
    - Collect the incoming arrow colors for every target rectangle.
    - A rectangle with one incoming side is recolored uniformly to that arrow color.
    - A rectangle with opposite vertical arrows is split into top and bottom halves; a rectangle with opposite horizontal arrows is split into left and right halves; the matching side color fills each half.
    - All arrow pixels become black in the output.

    Key insight:
    - The arrow shape determines both where to look for the target rectangle and which side of that rectangle owns the corresponding recolor region.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _dominant_color(g)
    rectangles, rect_ids = _find_rectangles(g, bg)
    assert rectangles, "expected at least one solid rectangle"

    non_rect_mask = (g != bg) & (rect_ids < 0)
    used_arrow = np.zeros(g.shape, dtype=bool)
    incoming: dict[int, list[tuple[str, int]]] = defaultdict(list)

    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            if not bool(non_rect_mask[r, c]) or bool(used_arrow[r, c]):
                continue

            color = int(g[r, c])
            matches: list[tuple[str, int, list[tuple[int, int]]]] = []
            for pointing, spec in POINTING_SPECS.items():
                coords = [(int(r), int(c))]
                valid = True
                for wr, wc in spec["wings"]:
                    rr = r + wr
                    cc = c + wc
                    if not _in_bounds(g.shape, rr, cc):
                        valid = False
                        break
                    if bool(used_arrow[rr, cc]) or not bool(non_rect_mask[rr, cc]) or int(g[rr, cc]) != color:
                        valid = False
                        break
                    coords.append((int(rr), int(cc)))
                if not valid:
                    continue

                target = _ray_target(g, rect_ids, (int(r), int(c)), spec["delta"], bg)
                if target is not None:
                    matches.append((pointing, target, coords))

            assert len(matches) <= 1, "arrow detection must be unambiguous"
            if not matches:
                continue

            pointing, target, coords = matches[0]
            for rr, cc in coords:
                used_arrow[rr, cc] = True
            incoming[target].append((POINTING_SPECS[pointing]["source"], color))

    assert not bool(np.any(non_rect_mask & (~used_arrow))), "all non-rectangle foreground pixels must belong to arrows"

    out = g.copy()
    out[used_arrow] = bg

    for rect in rectangles:
        rect_id = int(rect["id"])
        arrows = incoming.get(rect_id, [])
        if not arrows:
            continue

        side_colors: dict[str, int] = {}
        for side, color in arrows:
            if side in side_colors:
                assert side_colors[side] == color, "conflicting colors from the same side are unsupported"
            else:
                side_colors[side] = color

        r0 = int(rect["r0"])
        c0 = int(rect["c0"])
        h = int(rect["h"])
        w = int(rect["w"])

        if len(side_colors) == 1:
            out[r0 : r0 + h, c0 : c0 + w] = next(iter(side_colors.values()))
        elif set(side_colors) == {"up", "down"}:
            split = h // 2
            out[r0 : r0 + split, c0 : c0 + w] = int(side_colors["up"])
            out[r0 + split : r0 + h, c0 : c0 + w] = int(side_colors["down"])
        elif set(side_colors) == {"left", "right"}:
            split = w // 2
            out[r0 : r0 + h, c0 : c0 + split] = int(side_colors["left"])
            out[r0 : r0 + h, c0 + split : c0 + w] = int(side_colors["right"])
        else:
            raise AssertionError("unsupported arrow arrangement")

    return out


def _expanded(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    expanded: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if _in_bounds(shape, rr, cc):
                    expanded.add((int(rr), int(cc)))
    return expanded


def _sample_size(branch: str) -> tuple[int, int]:
    if branch == "vertical":
        return random.choice((2, 4)), random.randint(2, 4)
    if branch == "horizontal":
        return random.randint(2, 4), random.choice((2, 4))
    return random.randint(2, 4), random.randint(2, 4)


def _branch_sides(branch: str) -> list[str]:
    if branch == "vertical":
        return ["up", "down"]
    if branch == "horizontal":
        return ["left", "right"]
    return [random.choice(["up", "down", "left", "right"])]


def _arrow_cells_for_side(
    shape: tuple[int, int],
    bbox: tuple[int, int, int, int],
    side: str,
    gap: int,
    anchor: int,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]] | None:
    r0, c0, h, w = bbox
    r1 = r0 + h - 1
    c1 = c0 + w - 1

    if side == "left":
        tip = (anchor, c0 - gap - 1)
        arrow = {tip, (anchor - 1, tip[1] - 1), (anchor + 1, tip[1] - 1)}
        gap_cells = {(anchor, tip[1] + step) for step in range(1, gap + 1)}
    elif side == "right":
        tip = (anchor, c1 + gap + 1)
        arrow = {tip, (anchor - 1, tip[1] + 1), (anchor + 1, tip[1] + 1)}
        gap_cells = {(anchor, c1 + step) for step in range(1, gap + 1)}
    elif side == "up":
        tip = (r0 - gap - 1, anchor)
        arrow = {tip, (tip[0] - 1, anchor - 1), (tip[0] - 1, anchor + 1)}
        gap_cells = {(tip[0] + step, anchor) for step in range(1, gap + 1)}
    else:
        assert side == "down"
        tip = (r1 + gap + 1, anchor)
        arrow = {tip, (tip[0] + 1, anchor - 1), (tip[0] + 1, anchor + 1)}
        gap_cells = {(r1 + step, anchor) for step in range(1, gap + 1)}

    for rr, cc in arrow | gap_cells:
        if not _in_bounds(shape, rr, cc):
            return None
    return arrow, gap_cells


def _paint_expected(grid: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = np.array(grid, copy=True)
    for spec in specs:
        for r, c in spec["arrow_cells"]:
            out[r, c] = BG

    for spec in specs:
        r0, c0, h, w = spec["bbox"]
        side_colors = spec["side_colors"]
        if spec["branch"] == "single":
            out[r0 : r0 + h, c0 : c0 + w] = next(iter(side_colors.values()))
        elif spec["branch"] == "vertical":
            split = h // 2
            out[r0 : r0 + split, c0 : c0 + w] = int(side_colors["up"])
            out[r0 + split : r0 + h, c0 : c0 + w] = int(side_colors["down"])
        else:
            split = w // 2
            out[r0 : r0 + h, c0 : c0 + split] = int(side_colors["left"])
            out[r0 : r0 + h, c0 + split : c0 + w] = int(side_colors["right"])
    return out


def _build_scene() -> tuple[np.ndarray, list[dict]] | None:
    h = random.randint(18, 30)
    w = random.randint(18, 30)
    grid = np.full((h, w), BG, dtype=int)
    blocked: set[tuple[int, int]] = set()
    specs: list[dict] = []

    branches = ["single", "vertical", "horizontal"]
    if random.random() < 0.5:
        branches.append(random.choice(branches))
    random.shuffle(branches)

    for branch in branches:
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            rect_h, rect_w = _sample_size(branch)
            if h - rect_h - 2 * MIN_MARGIN <= 0 or w - rect_w - 2 * MIN_MARGIN <= 0:
                return None
            r0 = random.randint(MIN_MARGIN, h - rect_h - MIN_MARGIN)
            c0 = random.randint(MIN_MARGIN, w - rect_w - MIN_MARGIN)
            bbox = (r0, c0, rect_h, rect_w)
            rect_cells = {(r, c) for r in range(r0, r0 + rect_h) for c in range(c0, c0 + rect_w)}

            side_colors: dict[str, int] = {}
            side_cells: dict[str, set[tuple[int, int]]] = {}
            side_gaps: dict[str, set[tuple[int, int]]] = {}
            arrow_cells: set[tuple[int, int]] = set()
            gap_cells: set[tuple[int, int]] = set()
            valid = True

            sides = _branch_sides(branch)
            colors = random.sample(ARROW_COLORS, len(sides))
            for side, color in zip(sides, colors):
                if side in {"left", "right"}:
                    anchor = random.randint(r0, r0 + rect_h - 1)
                else:
                    anchor = random.randint(c0, c0 + rect_w - 1)
                gap = random.randint(1, 2)
                built = _arrow_cells_for_side(grid.shape, bbox, side, gap, anchor)
                if built is None:
                    valid = False
                    break
                one_arrow, one_gap = built
                if (one_arrow | one_gap) & rect_cells:
                    valid = False
                    break
                if one_arrow & arrow_cells or one_arrow & gap_cells or one_gap & gap_cells or one_gap & arrow_cells:
                    valid = False
                    break
                arrow_cells |= one_arrow
                gap_cells |= one_gap
                side_colors[side] = int(color)
                side_cells[side] = set(one_arrow)
                side_gaps[side] = set(one_gap)

            if not valid:
                continue

            reserved = rect_cells | arrow_cells | gap_cells
            if reserved & blocked:
                continue

            grid[r0 : r0 + rect_h, c0 : c0 + rect_w] = GREY
            for side, color in side_colors.items():
                for rr, cc in side_cells[side]:
                    grid[rr, cc] = int(color)

            blocked |= _expanded(reserved, grid.shape)
            specs.append(
                {
                    "branch": branch,
                    "bbox": bbox,
                    "side_colors": dict(side_colors),
                    "arrow_cells": set(arrow_cells),
                    "gap_cells": set(gap_cells),
                    "side_cells": side_cells,
                    "side_gaps": side_gaps,
                }
            )

            placed = True
            break

        if not placed:
            return None

    return grid, specs


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        built = _build_scene()
        if built is None:
            continue
        grid, specs = built
        if not specs:
            continue
        solved = main(grid)
        expected = _paint_expected(grid, specs)
        if np.array_equal(solved, expected) and not np.array_equal(grid, expected):
            return grid
    raise RuntimeError("failed to generate a valid arrow-to-rectangle scene")
