from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np


ARROW_TEMPLATES: dict[str, np.ndarray] = {
    "up": np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),
    "down": np.array([[0, 1, 0], [0, 1, 0], [1, 1, 1]], dtype=bool),
    "left": np.array([[1, 0, 0], [1, 1, 1], [1, 0, 0]], dtype=bool),
    "right": np.array([[0, 0, 1], [1, 1, 1], [0, 0, 1]], dtype=bool),
}

ARROW_INFO = {
    "up": {"tip": (0, 1), "step": (-1, 0)},
    "down": {"tip": (2, 1), "step": (1, 0)},
    "left": {"tip": (1, 0), "step": (0, -1)},
    "right": {"tip": (1, 2), "step": (0, 1)},
}

BG = 0
ARROW_COLOR = 5
RECT_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

GEN_RETRIES = 500
RECT_RETRIES = 250
ARROW_GAP_CHOICES = (1, 2, 3, 4)


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _find_components(g: np.ndarray, bg: int) -> list[dict]:
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == int(bg):
                continue
            color = int(g[r, c])
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(int(rr), int(cc)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(arr.shape[0]),
                }
            )

    return comps


def _match_arrow(mask: np.ndarray) -> str | None:
    for name, template in ARROW_TEMPLATES.items():
        if np.array_equal(mask, template):
            return name
    return None


def _segment_sizes(total: int, parts: int) -> list[int]:
    if parts <= 1:
        return [int(total)]
    available = int(total - (parts - 1))
    assert available >= parts, "rectangle too thin to split into visible pieces"
    base, extra = divmod(available, parts)
    return [int(base + (i < extra)) for i in range(parts)]


def _draw_vertical_cuts(out: np.ndarray, bbox: tuple[int, int, int, int], bg: int, parts: int) -> None:
    if parts <= 1:
        return
    r0, c0, r1, c1 = bbox
    sizes = _segment_sizes(int(c1 - c0 + 1), int(parts))
    col = int(c0)
    for size in sizes[:-1]:
        col += int(size)
        out[int(r0) : int(r1) + 1, int(col)] = int(bg)
        col += 1


def _draw_horizontal_cuts(out: np.ndarray, bbox: tuple[int, int, int, int], bg: int, parts: int) -> None:
    if parts <= 1:
        return
    r0, c0, r1, c1 = bbox
    sizes = _segment_sizes(int(r1 - r0 + 1), int(parts))
    row = int(r0)
    for size in sizes[:-1]:
        row += int(size)
        out[int(row), int(c0) : int(c1) + 1] = int(bg)
        row += 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split each colored rectangle into arrow-counted pieces and erase the arrows.

    Input structure:
    - The background is black.
    - Several solid monochrome rectangles of distinct colors are scattered on the grid.
    - Separate grey 3x3 arrow sprites are also present; each arrow is a 5-pixel shape pointing up, down, left, or right.

    Rule:
    - Detect every arrow and trace forward from its tip along the shaft direction until the first solid rectangle it meets; that rectangle is the arrow's target.
    - For each rectangle, count its incoming arrows. If horizontal arrows (left/right) are at least as many as vertical arrows (up/down), draw vertical black separator lines; otherwise draw horizontal black separator lines.
    - The rectangle is divided into k pieces, where k is its number of incoming arrows. The k-1 separator lines are one pixel thick, and the remaining colored span is split as evenly as possible, with any remainder assigned from left to right or top to bottom.
    - All arrows themselves are removed to background in the output.

    Key insight:
    The arrows contribute two different signals at once: line-of-sight decides which rectangle they control, while the majority arrow orientation votes for the perpendicular cut direction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    bg = int(Counter(int(v) for v in [g[0, 0], g[0, -1], g[-1, 0], g[-1, -1]]).most_common(1)[0][0])

    arrows: list[dict] = []
    rectangles: list[dict] = []
    for comp in _find_components(g, bg):
        r0, c0, r1, c1 = comp["bbox"]
        color = int(comp["color"])
        patch = g[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]

        if int(comp["area"]) == 5 and int(r1 - r0 + 1) == 3 and int(c1 - c0 + 1) == 3:
            direction = _match_arrow(patch == color)
            if direction is not None:
                tip_r, tip_c = ARROW_INFO[direction]["tip"]
                arrows.append(
                    {
                        "dir": direction,
                        "bbox": (int(r0), int(c0), int(r1), int(c1)),
                        "coords": comp["coords"],
                        "tip": (int(r0 + tip_r), int(c0 + tip_c)),
                        "step": ARROW_INFO[direction]["step"],
                    }
                )
                continue

        if int(comp["area"]) == int((r1 - r0 + 1) * (c1 - c0 + 1)) and bool(np.all(patch == color)):
            rectangles.append({"bbox": (int(r0), int(c0), int(r1), int(c1)), "color": color})

    assert arrows, "expected at least one arrow"
    assert rectangles, "expected at least one rectangle"

    rect_map = -np.ones((h, w), dtype=int)
    for idx, rect in enumerate(rectangles):
        r0, c0, r1, c1 = rect["bbox"]
        rect_map[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = int(idx)

    incoming: dict[int, list[str]] = defaultdict(list)
    for arrow in arrows:
        tip_r, tip_c = arrow["tip"]
        dr, dc = arrow["step"]
        rr, cc = int(tip_r + dr), int(tip_c + dc)
        target = None
        while 0 <= rr < h and 0 <= cc < w:
            rid = int(rect_map[rr, cc])
            if rid != -1:
                target = int(rid)
                break
            rr += int(dr)
            cc += int(dc)
        assert target is not None, "every arrow must point to a rectangle"
        incoming[int(target)].append(str(arrow["dir"]))

    out = g.copy()
    for arrow in arrows:
        coords = arrow["coords"]
        out[coords[:, 0], coords[:, 1]] = int(bg)

    for rid, rect in enumerate(rectangles):
        dirs = incoming.get(int(rid), [])
        if not dirs:
            continue
        horizontal = sum(d in ("left", "right") for d in dirs)
        vertical = len(dirs) - horizontal
        if horizontal >= vertical:
            _draw_vertical_cuts(out, rect["bbox"], bg, len(dirs))
        else:
            _draw_horizontal_cuts(out, rect["bbox"], bg, len(dirs))

    return out


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    return not (
        int(a[2]) + int(pad) < int(b[0])
        or int(b[2]) + int(pad) < int(a[0])
        or int(a[3]) + int(pad) < int(b[1])
        or int(b[3]) + int(pad) < int(a[1])
    )


def _arrow_cells(top: int, left: int, direction: str) -> list[tuple[int, int]]:
    return [
        (int(top + r), int(left + c))
        for r, c in np.argwhere(ARROW_TEMPLATES[direction])
    ]


def _anchor_span(n: int) -> int:
    if n <= 1:
        return 1
    return 4 * (int(n) - 1) + 1


def _sample_branch() -> tuple[list[str], int, int, str]:
    branch = random.choice(["hmajor", "vmajor", "tie"])
    if branch == "hmajor":
        horizontal, vertical = random.choice([(2, 0), (2, 1), (3, 1)])
    elif branch == "vmajor":
        horizontal, vertical = random.choice([(0, 2), (1, 2), (1, 3)])
    else:
        horizontal, vertical = random.choice([(1, 1), (2, 2)])

    dirs = [random.choice(["left", "right"]) for _ in range(horizontal)]
    dirs += [random.choice(["up", "down"]) for _ in range(vertical)]
    random.shuffle(dirs)
    cut_axis = "vertical" if horizontal >= vertical else "horizontal"
    return dirs, int(horizontal), int(vertical), cut_axis


def _sample_specs(n_rects: int) -> list[dict]:
    colors = list(RECT_COLORS)
    random.shuffle(colors)
    specs: list[dict] = []
    for idx in range(n_rects):
        dirs, h_count, v_count, cut_axis = _sample_branch()
        k = len(dirs)
        per_side = Counter(dirs)
        min_h = max(4, _anchor_span(max(int(per_side["left"]), int(per_side["right"]))))
        min_w = max(4, _anchor_span(max(int(per_side["up"]), int(per_side["down"]))))
        if cut_axis == "vertical":
            min_w = max(min_w, 2 * int(k) - 1)
        else:
            min_h = max(min_h, 2 * int(k) - 1)

        max_h = min(10, max(min_h, min_h + 3))
        max_w = min(10, max(min_w, min_w + 3))
        specs.append(
            {
                "color": int(colors[idx]),
                "dirs": list(dirs),
                "height": random.randint(int(min_h), int(max_h)),
                "width": random.randint(int(min_w), int(max_w)),
            }
        )
    return specs


def _grid_shape(n_rects: int) -> tuple[int, int]:
    if n_rects == 2:
        return random.randint(18, 26), random.randint(18, 26)
    return random.randint(22, 30), random.randint(22, 30)


def _place_rectangles(h: int, w: int, specs: list[dict]) -> list[dict] | None:
    margin = 4
    pad = 3
    placed: list[dict] = []

    for spec in sorted(specs, key=lambda s: (-(s["height"] * s["width"]), -max(s["height"], s["width"]))):
        rh = int(spec["height"])
        rw = int(spec["width"])
        if rh + 2 * margin > h or rw + 2 * margin > w:
            return None
        for _ in range(RECT_RETRIES):
            r0 = random.randint(margin, h - margin - rh)
            c0 = random.randint(margin, w - margin - rw)
            box = (int(r0), int(c0), int(r0 + rh - 1), int(c0 + rw - 1))
            if any(_boxes_overlap(box, other["bbox"], pad=pad) for other in placed):
                continue
            placed.append(
                {
                    "bbox": box,
                    "color": int(spec["color"]),
                    "dirs": list(spec["dirs"]),
                }
            )
            break
        else:
            return None

    return placed


def _candidate_arrow(rect: dict, direction: str, anchor: int, gap: int) -> tuple[int, int, tuple[int, int], tuple[int, int, int, int]]:
    r0, c0, r1, c1 = rect["bbox"]
    if direction == "right":
        tip = (int(anchor), int(c0 - gap - 1))
        top, left = int(anchor - 1), int(tip[1] - 2)
    elif direction == "left":
        tip = (int(anchor), int(c1 + gap + 1))
        top, left = int(anchor - 1), int(tip[1])
    elif direction == "down":
        tip = (int(r0 - gap - 1), int(anchor))
        top, left = int(tip[0] - 2), int(anchor - 1)
    else:
        tip = (int(r1 + gap + 1), int(anchor))
        top, left = int(tip[0]), int(anchor - 1)
    return int(top), int(left), tip, (int(top), int(left), int(top + 2), int(left + 2))


def _bbox_in_bounds(box: tuple[int, int, int, int], h: int, w: int) -> bool:
    r0, c0, r1, c1 = box
    return 0 <= r0 and 0 <= c0 and r1 < h and c1 < w


def _target_hit(grid: np.ndarray, rect: dict, tip: tuple[int, int], step: tuple[int, int]) -> bool:
    h, w = grid.shape
    r0, c0, r1, c1 = rect["bbox"]
    rr, cc = int(tip[0] + step[0]), int(tip[1] + step[1])
    while 0 <= rr < h and 0 <= cc < w:
        value = int(grid[rr, cc])
        if value != BG:
            return r0 <= rr <= r1 and c0 <= cc <= c1
        rr += int(step[0])
        cc += int(step[1])
    return False


def _can_place_arrow(grid: np.ndarray, rect: dict, direction: str, cells: list[tuple[int, int]], tip: tuple[int, int]) -> bool:
    cell_set = set(cells)
    for r, c in cells:
        if int(grid[r, c]) != BG:
            return False
        for nr, nc in _neighbors4(int(r), int(c)):
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and (int(nr), int(nc)) not in cell_set:
                if int(grid[nr, nc]) != BG:
                    return False
    return _target_hit(grid, rect, tip, ARROW_INFO[direction]["step"])


def _place_arrows(grid: np.ndarray, rects: list[dict]) -> bool:
    order = sorted(rects, key=lambda rect: -len(rect["dirs"]))
    for rect in order:
        dirs = list(rect["dirs"])
        dirs.sort(key=lambda d: (-dirs.count(d), d))
        for direction in dirs:
            r0, c0, r1, c1 = rect["bbox"]
            anchors = list(range(int(r0), int(r1) + 1)) if direction in ("left", "right") else list(range(int(c0), int(c1) + 1))
            gaps = list(ARROW_GAP_CHOICES)
            candidates: list[tuple[int, int, tuple[int, int], tuple[int, int, int, int]]] = []
            for anchor in anchors:
                for gap in gaps:
                    candidates.append(_candidate_arrow(rect, direction, int(anchor), int(gap)))
            random.shuffle(candidates)

            placed = False
            for top, left, tip, box in candidates:
                if not _bbox_in_bounds(box, *grid.shape):
                    continue
                cells = _arrow_cells(int(top), int(left), direction)
                if not _can_place_arrow(grid, rect, direction, cells, tip):
                    continue
                for rr, cc in cells:
                    grid[int(rr), int(cc)] = ARROW_COLOR
                placed = True
                break
            if not placed:
                return False
    return True


def _render_rectangles(h: int, w: int, rects: list[dict]) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    for rect in rects:
        r0, c0, r1, c1 = rect["bbox"]
        grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = int(rect["color"])
    return grid


def _gen_attempt() -> np.ndarray | None:
    n_rects = random.choice([2, 3])
    specs = _sample_specs(int(n_rects))
    h, w = _grid_shape(int(n_rects))
    rects = _place_rectangles(int(h), int(w), specs)
    if rects is None:
        return None

    inp = _render_rectangles(int(h), int(w), rects)
    if not _place_arrows(inp, rects):
        return None

    try:
        out = main(inp)
    except Exception:
        return None

    if np.array_equal(inp, out):
        return None
    if int(np.count_nonzero(inp == ARROW_COLOR)) < 10:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid arrow-rectangle puzzle")
