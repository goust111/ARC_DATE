from __future__ import annotations

import random

import numpy as np


BLACK = 0
FRAME_COLOR = 5
SEQUENCE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
MAX_GENERATION_ATTEMPTS = 400

BBox = tuple[int, int, int, int]


def _components(g: np.ndarray) -> list[dict]:
    h, w = map(int, g.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == BLACK or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            pts = np.array(coords, dtype=int)
            r0, c0 = map(int, pts.min(axis=0))
            r1, c1 = map(int, pts.max(axis=0))
            comps.append(
                {
                    "color": color,
                    "coords": pts,
                    "bbox": (r0, c0, r1, c1),
                    "area": int(len(coords)),
                }
            )

    return comps


def _draw_frame(grid: np.ndarray, bbox: BBox, color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = int(color)
    grid[r1, c0 : c1 + 1] = int(color)
    grid[r0 : r1 + 1, c0] = int(color)
    grid[r0 : r1 + 1, c1] = int(color)


def _is_frame(g: np.ndarray, comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    color = int(comp["color"])
    height = int(r1 - r0 + 1)
    width = int(c1 - c0 + 1)
    if height < 3 or width < 3:
        return False

    perimeter = 2 * (height + width) - 4
    if int(comp["area"]) != perimeter:
        return False

    if not np.all(g[r0, c0 : c1 + 1] == color):
        return False
    if not np.all(g[r1, c0 : c1 + 1] == color):
        return False
    if not np.all(g[r0 : r1 + 1, c0] == color):
        return False
    if not np.all(g[r0 : r1 + 1, c1] == color):
        return False

    return True


def _frame_chain(candidates: list[dict]) -> list[dict]:
    by_bbox = {tuple(comp["bbox"]): comp for comp in candidates}
    best: list[dict] = []

    for comp in candidates:
        chain = [comp]
        r0, c0, r1, c1 = comp["bbox"]
        while True:
            inner_bbox = (r0 + 2, c0 + 2, r1 - 2, c1 - 2)
            nxt = by_bbox.get(inner_bbox)
            if nxt is None:
                break
            chain.append(nxt)
            r0, c0, r1, c1 = inner_bbox

        if len(chain) > len(best):
            best = chain
        elif len(chain) == len(best) and chain:
            area = (comp["bbox"][2] - comp["bbox"][0] + 1) * (comp["bbox"][3] - comp["bbox"][1] + 1)
            best_area = (best[0]["bbox"][2] - best[0]["bbox"][0] + 1) * (best[0]["bbox"][3] - best[0]["bbox"][1] + 1)
            if area > best_area:
                best = chain

    assert len(best) >= 2, "expected a nested frame chain"
    return best


def _sequence_candidates(g: np.ndarray, length: int) -> list[tuple[list[tuple[int, int]], list[int]]]:
    h, w = map(int, g.shape)
    found: list[tuple[list[tuple[int, int]], list[int]]] = []

    for r in range(h):
        c = 0
        while c < w:
            if int(g[r, c]) == BLACK:
                c += 1
                continue
            c0 = c
            while c < w and int(g[r, c]) != BLACK:
                c += 1
            c1 = c - 1
            colors = [int(v) for v in g[r, c0 : c1 + 1]]
            if len(colors) == length and len(set(colors)) == length:
                if all((r == 0 or int(g[r - 1, cc]) == BLACK) and (r == h - 1 or int(g[r + 1, cc]) == BLACK) for cc in range(c0, c1 + 1)):
                    found.append(([(r, cc) for cc in range(c0, c1 + 1)], colors))

    for c in range(w):
        r = 0
        while r < h:
            if int(g[r, c]) == BLACK:
                r += 1
                continue
            r0 = r
            while r < h and int(g[r, c]) != BLACK:
                r += 1
            r1 = r - 1
            colors = [int(v) for v in g[r0 : r1 + 1, c]]
            if len(colors) == length and len(set(colors)) == length:
                if all((c == 0 or int(g[rr, c - 1]) == BLACK) and (c == w - 1 or int(g[rr, c + 1]) == BLACK) for rr in range(r0, r1 + 1)):
                    found.append(([(rr, c) for rr in range(r0, r1 + 1)], colors))

    return found


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor nested concentric frames using the ordered colors from a separate sequence bar.

    Input structure:
    - The background is black.
    - One region contains 2 to 5 nested one-pixel-thick rectangular frames with
      one black cell of spacing between consecutive frames.
    - Another disjoint region contains a horizontal or vertical sequence bar made
      of single adjacent pixels of distinct colors, one color per frame layer.

    Rule:
    - Detect the nested frame chain and order its layers from the outermost frame
      to the innermost frame.
    - Read the separate sequence bar in its natural direction: left-to-right for
      a horizontal bar or top-to-bottom for a vertical bar.
    - Recolor frame layer i with sequence color i, keep all black background and
      black gaps unchanged, and clear the sequence bar itself back to black.

    Key insight:
    - The task combines two independent structures: a geometric nesting hierarchy
      from the concentric frames and an ordered color lookup encoded by the bar.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frame_candidates = [comp for comp in _components(g) if _is_frame(g, comp)]
    frames = _frame_chain(frame_candidates)

    bars = _sequence_candidates(g, len(frames))
    assert len(bars) == 1, "expected a unique color sequence bar"
    bar_coords, bar_colors = bars[0]

    out = g.copy()
    for r, c in bar_coords:
        out[r, c] = BLACK
    for comp, color in zip(frames, bar_colors):
        _draw_frame(out, comp["bbox"], color)

    return out


def _frame_bboxes(top: int, left: int, height: int, width: int, layers: int) -> list[BBox]:
    return [
        (top + 2 * i, left + 2 * i, top + height - 1 - 2 * i, left + width - 1 - 2 * i)
        for i in range(layers)
    ]


def _render_expected(inp: np.ndarray, frames: list[BBox], bar_coords: list[tuple[int, int]], bar_colors: list[int]) -> np.ndarray:
    out = inp.copy()
    for r, c in bar_coords:
        out[r, c] = BLACK
    for bbox, color in zip(frames, bar_colors):
        _draw_frame(out, bbox, color)
    return out


def _gen_attempt() -> np.ndarray | None:
    layers = random.randint(2, 5)
    orient = random.choice(("horizontal", "vertical"))
    side = random.choice(("top", "bottom") if orient == "horizontal" else ("left", "right"))

    min_outer = 4 * layers - 1
    max_outer_h = min(26, 30 if orient == "vertical" else 28)
    max_outer_w = min(26, 30 if orient == "horizontal" else 28)
    if min_outer > max_outer_h or min_outer > max_outer_w:
        return None

    outer_h = random.randint(min_outer, max_outer_h)
    outer_w = random.randint(min_outer, max_outer_w)

    if orient == "horizontal":
        height = random.randint(outer_h + 2, 30)
        width = random.randint(max(outer_w, layers), 30)
        frame_left = random.randint(0, width - outer_w)
        bar_c0 = random.randint(0, width - layers)
        if side == "top":
            bar_r = random.randint(0, height - outer_h - 2)
            frame_top = bar_r + 2
        else:
            frame_top = random.randint(0, height - outer_h - 2)
            bar_r = frame_top + outer_h + 1
        bar_coords = [(bar_r, bar_c0 + i) for i in range(layers)]
    else:
        height = random.randint(max(outer_h, layers), 30)
        width = random.randint(outer_w + 2, 30)
        frame_top = random.randint(0, height - outer_h)
        bar_r0 = random.randint(0, height - layers)
        if side == "left":
            bar_c = random.randint(0, width - outer_w - 2)
            frame_left = bar_c + 2
        else:
            frame_left = random.randint(0, width - outer_w - 2)
            bar_c = frame_left + outer_w + 1
        bar_coords = [(bar_r0 + i, bar_c) for i in range(layers)]

    frame_boxes = _frame_bboxes(frame_top, frame_left, outer_h, outer_w, layers)
    bar_colors = random.sample(SEQUENCE_COLORS, layers)

    inp = np.zeros((height, width), dtype=int)
    for bbox in frame_boxes:
        _draw_frame(inp, bbox, FRAME_COLOR)
    for (r, c), color in zip(bar_coords, bar_colors):
        inp[r, c] = int(color)

    expected = _render_expected(inp, frame_boxes, bar_coords, bar_colors)
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid concentric-frame recoloring input")
