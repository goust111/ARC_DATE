from __future__ import annotations

from collections import Counter
import random

import numpy as np


CORNERS = ("TL", "TR", "BR", "BL")
CORNER_TO_RANK = {corner: idx for idx, corner in enumerate(CORNERS)}
CORNER_STEPS = {
    "TL": ((0, 1), (1, 0)),
    "TR": ((0, -1), (1, 0)),
    "BR": ((0, -1), (-1, 0)),
    "BL": ((0, 1), (-1, 0)),
}


GEN_RETRIES = 500
PLACEMENT_RETRIES = 300
RECT_COUNT_CHOICES = (2, 3, 4)
GRID_H_RANGE = (12, 24)
GRID_W_RANGE = (14, 28)
RECT_HEIGHT_RANGE = {2: (5, 9), 3: (5, 8), 4: (4, 7)}
RECT_WIDTH_RANGE = {2: (6, 10), 3: (5, 8), 4: (4, 6)}


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _border_cells(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        cells.append((r, c1))
    return cells


def _draw_rect(grid: np.ndarray, r0: int, c0: int, h: int, w: int, color: int) -> None:
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _marker_pos(rect: dict) -> tuple[int, int]:
    if rect["corner"] == "TL":
        return (rect["r0"], rect["c0"])
    if rect["corner"] == "TR":
        return (rect["r0"], rect["c1"])
    if rect["corner"] == "BR":
        return (rect["r1"], rect["c1"])
    return (rect["r1"], rect["c0"])


def _support_cells(rect: dict) -> list[tuple[int, int]]:
    mr, mc = _marker_pos(rect)
    return [(mr + dr, mc + dc) for dr, dc in CORNER_STEPS[rect["corner"]]]


def _corner_from_marker(
    marker_pos: tuple[int, int], bbox: tuple[int, int, int, int]
) -> str | None:
    r, c = marker_pos
    r0, c0, r1, c1 = bbox
    if (r, c) == (r0, c0):
        return "TL"
    if (r, c) == (r0, c1):
        return "TR"
    if (r, c) == (r1, c1):
        return "BR"
    if (r, c) == (r1, c0):
        return "BL"
    return None


def _on_border(r: int, c: int, bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    return (r == r0 or r == r1 or c == c0 or c == c1) and r0 <= r <= r1 and c0 <= c <= c1


def _marker_supported(
    grid: np.ndarray, marker_pos: tuple[int, int], corner: str, border_color: int
) -> bool:
    h, w = grid.shape
    mr, mc = marker_pos
    for dr, dc in CORNER_STEPS[corner]:
        nr, nc = mr + dr, mc + dc
        if not (0 <= nr < h and 0 <= nc < w):
            return False
        if int(grid[nr, nc]) != border_color:
            return False
    return True


def _reconstruct_rectangles(grid: np.ndarray, bg: int) -> list[dict]:
    counts = Counter(map(int, grid.ravel()))
    rects: list[dict] = []
    used_borders: set[int] = set()

    for marker_color, count in counts.items():
        if marker_color == bg or count != 1:
            continue

        marker_r, marker_c = map(int, np.argwhere(grid == marker_color)[0])
        marker_pos = (marker_r, marker_c)
        neighbor_colors = {
            int(grid[nr, nc])
            for nr, nc in _neighbors4(marker_r, marker_c)
            if 0 <= nr < grid.shape[0]
            and 0 <= nc < grid.shape[1]
            and int(grid[nr, nc]) not in {bg, marker_color}
        }

        candidates = []
        for border_color in sorted(neighbor_colors):
            if border_color in used_borders:
                continue
            coords = np.argwhere(grid == border_color)
            if coords.size == 0:
                continue
            r0 = min(marker_r, int(coords[:, 0].min()))
            c0 = min(marker_c, int(coords[:, 1].min()))
            r1 = max(marker_r, int(coords[:, 0].max()))
            c1 = max(marker_c, int(coords[:, 1].max()))
            bbox = (r0, c0, r1, c1)
            corner = _corner_from_marker(marker_pos, bbox)
            if corner is None:
                continue
            if any(not _on_border(int(r), int(c), bbox) for r, c in coords):
                continue
            if not _marker_supported(grid, marker_pos, corner, border_color):
                continue
            candidates.append((border_color, bbox, corner))

        assert len(candidates) == 1, "each marker must map to exactly one rectangle"
        border_color, (r0, c0, r1, c1), corner = candidates[0]
        used_borders.add(border_color)
        rects.append(
            {
                "r0": r0,
                "c0": c0,
                "r1": r1,
                "c1": c1,
                "h": r1 - r0 + 1,
                "w": c1 - c0 + 1,
                "border": border_color,
                "marker": marker_color,
                "corner": corner,
            }
        )

    assert rects, "expected at least one rectangle"
    rects.sort(key=lambda rect: (CORNER_TO_RANK[rect["corner"]], rect["r0"], rect["c0"]))
    return rects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Separate overlapped rectangular border layers and unfold them side by side in depth order.

    Input structure:
    - The most common color is the background.
    - Several axis-aligned rectangles are drawn as 1-pixel monochrome borders with background interiors.
    - Rectangles overlap by occlusion: front borders overwrite back-border pixels at shared cells.
    - Each rectangle has exactly one corner replaced by a unique marker color that appears nowhere else.

    Rule:
    - Recover every rectangle by pairing each singleton corner marker with its surrounding border color and reconstructing the rectangle's full bounding box.
    - The marker corner encodes depth order: top-left, top-right, bottom-right, bottom-left from front to back.
    - Restore each rectangle without occlusion, change the marker back to the border color, and place the rectangles left to right by depth.
    - Keep each rectangle's original top row; the output height matches the input, and neighboring unfolded rectangles are separated by one background column.

    Key insight:
    - Even though some border pixels are hidden, each visible singleton marker still identifies one rectangle corner, and the remaining visible pixels of that rectangle's border reveal its full bbox and therefore its depth slot.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _bg_color(g)
    rects = _reconstruct_rectangles(g, bg)

    out_w = sum(rect["w"] for rect in rects) + max(0, len(rects) - 1)
    out = np.full((g.shape[0], out_w), bg, dtype=int)
    offset = 0
    for rect in rects:
        _draw_rect(out, rect["r0"], offset, rect["h"], rect["w"], rect["border"])
        offset += rect["w"] + 1
    return out


def _sample_sizes(rng: random.Random, n_rects: int) -> tuple[list[int], list[int]] | None:
    h_lo, h_hi = RECT_HEIGHT_RANGE[n_rects]
    w_lo, w_hi = RECT_WIDTH_RANGE[n_rects]
    for _ in range(100):
        heights = [rng.randint(h_lo, h_hi) for _ in range(n_rects)]
        widths = [rng.randint(w_lo, w_hi) for _ in range(n_rects)]
        if sum(widths) + (n_rects - 1) <= 30:
            return heights, widths
    return None


def _sample_rectangles(
    rng: random.Random, h: int, w: int, heights: list[int], widths: list[int]
) -> list[dict] | None:
    anchor_r = rng.randint(3, h - 4)
    anchor_c = rng.randint(3, w - 4)
    rects: list[dict] = []

    for depth, (rh, rw) in enumerate(zip(heights, widths)):
        row_lo = max(1, anchor_r - rh + 2)
        row_hi = min(anchor_r - 1, h - rh - 1)
        col_lo = max(1, anchor_c - rw + 2)
        col_hi = min(anchor_c - 1, w - rw - 1)
        if row_lo > row_hi or col_lo > col_hi:
            return None

        rects.append(
            {
                "r0": rng.randint(row_lo, row_hi),
                "c0": rng.randint(col_lo, col_hi),
                "h": rh,
                "w": rw,
                "corner": CORNERS[depth],
            }
        )

    for rect in rects:
        rect["r1"] = rect["r0"] + rect["h"] - 1
        rect["c1"] = rect["c0"] + rect["w"] - 1

    return rects


def _strict_contains(a: dict, b: dict) -> bool:
    return (
        a["r0"] < b["r0"]
        and a["c0"] < b["c0"]
        and a["r1"] > b["r1"]
        and a["c1"] > b["c1"]
    )


def _pairwise_geometry_ok(rects: list[dict]) -> bool:
    border_sets = [set(_border_cells(rect["r0"], rect["c0"], rect["r1"], rect["c1"])) for rect in rects]
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            a = rects[i]
            b = rects[j]
            if _strict_contains(a, b) or _strict_contains(b, a):
                return False
            if (
                a["r0"] in {b["r0"], b["r1"]}
                or a["r1"] in {b["r0"], b["r1"]}
                or a["c0"] in {b["c0"], b["c1"]}
                or a["c1"] in {b["c0"], b["c1"]}
            ):
                return False
            overlap = border_sets[i] & border_sets[j]
            if not overlap or len(overlap) > 4:
                return False
    return True


def _render_input(rects: list[dict], bg: int, shape: tuple[int, int]) -> np.ndarray:
    canvas = np.full(shape, bg, dtype=int)
    for rect in reversed(rects):
        _draw_rect(canvas, rect["r0"], rect["c0"], rect["h"], rect["w"], rect["border"])
        mr, mc = _marker_pos(rect)
        canvas[mr, mc] = rect["marker"]
    return canvas


def _render_output(rects: list[dict], bg: int, h: int) -> np.ndarray:
    out_w = sum(rect["w"] for rect in rects) + max(0, len(rects) - 1)
    canvas = np.full((h, out_w), bg, dtype=int)
    offset = 0
    for rect in rects:
        _draw_rect(canvas, rect["r0"], offset, rect["h"], rect["w"], rect["border"])
        offset += rect["w"] + 1
    return canvas


def _visible_signature(grid: np.ndarray, rect: dict) -> list[tuple[int, int]]:
    marker_cell = _marker_pos(rect)
    cells = []
    for r, c in _border_cells(rect["r0"], rect["c0"], rect["r1"], rect["c1"]):
        val = int(grid[r, c])
        if val == rect["border"] or ((r, c) == marker_cell and val == rect["marker"]):
            cells.append((r, c))
    return cells


def _marker_and_bbox_visible(grid: np.ndarray, rect: dict) -> bool:
    mr, mc = _marker_pos(rect)
    if int(grid[mr, mc]) != rect["marker"]:
        return False
    for sr, sc in _support_cells(rect):
        if int(grid[sr, sc]) != rect["border"]:
            return False
    signature = _visible_signature(grid, rect)
    rows = {r for r, _ in signature}
    cols = {c for _, c in signature}
    return (
        rect["r0"] in rows
        and rect["r1"] in rows
        and rect["c0"] in cols
        and rect["c1"] in cols
    )


def _has_real_occlusion(grid: np.ndarray, rects: list[dict]) -> bool:
    for rect in rects:
        for r, c in _border_cells(rect["r0"], rect["c0"], rect["r1"], rect["c1"]):
            expected = rect["marker"] if (r, c) == _marker_pos(rect) else rect["border"]
            if int(grid[r, c]) != expected:
                return True
    return False


def _bg_is_unique_mode(grid: np.ndarray, bg: int) -> bool:
    counts = Counter(map(int, grid.ravel()))
    return counts[bg] > max((count for color, count in counts.items() if color != bg), default=0)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_rects = rng.choice(RECT_COUNT_CHOICES)
    sizes = _sample_sizes(rng, n_rects)
    if sizes is None:
        return None
    heights, widths = sizes

    h_min = max(max(heights) + 2, GRID_H_RANGE[0])
    w_min = max(max(widths) + 2, GRID_W_RANGE[0])
    if h_min > GRID_H_RANGE[1] or w_min > GRID_W_RANGE[1]:
        return None

    bg = rng.randrange(10)
    palette = [color for color in range(10) if color != bg]
    rng.shuffle(palette)
    border_colors = palette[:n_rects]
    marker_colors = palette[n_rects : 2 * n_rects]

    for _ in range(PLACEMENT_RETRIES):
        h = rng.randint(h_min, GRID_H_RANGE[1])
        w = rng.randint(w_min, GRID_W_RANGE[1])
        rects = _sample_rectangles(rng, h, w, heights, widths)
        if rects is None or not _pairwise_geometry_ok(rects):
            continue

        for idx, rect in enumerate(rects):
            rect["border"] = border_colors[idx]
            rect["marker"] = marker_colors[idx]

        inp = _render_input(rects, bg, (h, w))
        if not _bg_is_unique_mode(inp, bg):
            continue
        if not _has_real_occlusion(inp, rects):
            continue
        if not all(_marker_and_bbox_visible(inp, rect) for rect in rects):
            continue

        out = _render_output(rects, bg, h)
        if max(inp.shape) > 30 or max(out.shape) > 30:
            continue
        try:
            actual = main(inp)
        except Exception:
            continue
        if not np.array_equal(actual, out):
            continue
        if np.array_equal(inp, out):
            continue
        return inp

    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to produce a valid sample")
