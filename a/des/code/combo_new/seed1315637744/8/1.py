import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
MARKER_COLORS = tuple(c for c in range(10) if c not in (BG, GREY))
GEN_RETRIES = 300


def _border_path(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    path: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        path.append((r0, c))
    for r in range(r0 + 1, r1 + 1):
        path.append((r, c1))
    for c in range(c1 - 1, c0 - 1, -1):
        path.append((r1, c))
    for r in range(r1 - 1, r0, -1):
        path.append((r, c0))
    return path


def _draw_rectangle(grid: np.ndarray, bbox: tuple[int, int, int, int], dashed: bool) -> None:
    for idx, (r, c) in enumerate(_border_path(bbox)):
        if (not dashed) or idx % 2 == 0:
            grid[r, c] = GREY


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _find_solid_rectangles(grid: np.ndarray) -> list[dict[str, object]]:
    rectangles: list[dict[str, object]] = []
    for comp in _components(grid == GREY):
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue

        border = set(_border_path((r0, c0, r1, c1)))
        comp_set = set(comp)
        if comp_set != border:
            continue

        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        if np.any(interior == GREY):
            continue
        marker_pts = np.argwhere((interior != BG) & (interior != GREY))
        if len(marker_pts) != 1:
            continue
        mr, mc = marker_pts[0]
        mr += r0 + 1
        mc += c0 + 1
        rectangles.append(
            {
                "bbox": (r0, c0, r1, c1),
                "marker_pos": (int(mr), int(mc)),
                "marker_color": int(grid[mr, mc]),
            }
        )
    return rectangles


def _dashed_rect_info(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> dict[str, object] | None:
    r0, c0, r1, c1 = bbox
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None

    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    if np.any(interior == GREY):
        return None
    marker_pts = np.argwhere((interior != BG) & (interior != GREY))
    if len(marker_pts) != 1:
        return None

    for idx, (r, c) in enumerate(_border_path(bbox)):
        want = GREY if idx % 2 == 0 else BG
        if int(grid[r, c]) != want:
            return None

    mr, mc = marker_pts[0]
    mr += r0 + 1
    mc += c0 + 1
    return {
        "bbox": bbox,
        "marker_pos": (int(mr), int(mc)),
        "marker_color": int(grid[mr, mc]),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use a style legend to redraw grey rectangular borders as solid or dashed.

    Input structure:
    - The background is black.
    - Several one-pixel-thick grey rectangular frames appear on the grid; each frame interior contains exactly one colored marker pixel.
    - One corner contains a two-box style legend: one small grey frame is solid and one is dashed, and each legend box also contains one colored marker pixel.

    Rule:
    - Read the legend by matching the marker color inside the solid legend box to the solid-border style, and the marker color inside the dashed legend box to the dashed-border style.
    - For every non-legend grey rectangle, inspect its interior marker color and redraw that rectangle with the matching style.
    - A dashed rectangle keeps every other border pixel along the border path starting from the top-left corner; skipped border pixels become black.
    - Remove every interior marker pixel, and remove the legend itself entirely.

    Key insight:
    - The corner legend does not contribute output pixels; it defines a color-to-style mapping, and the border appearance of the legend boxes is the only cue for how each marked rectangle must be repainted.
    """
    g = np.asarray(grid, dtype=int)
    solid_rects = _find_solid_rectangles(g)
    assert solid_rects, "expected at least one solid grey rectangle"

    all_markers = {
        (int(r), int(c)): int(g[r, c])
        for r, c in np.argwhere((g != BG) & (g != GREY))
    }
    covered_markers = {rect["marker_pos"] for rect in solid_rects}
    unmatched_markers = {pos: color for pos, color in all_markers.items() if pos not in covered_markers}

    matches: list[tuple[int, dict[str, object]]] = []
    h, w = g.shape
    for idx, rect in enumerate(solid_rects):
        r0, c0, r1, c1 = rect["bbox"]
        rect_w = c1 - c0 + 1
        for gap in (1, 2, 3):
            for shift in (-(rect_w + gap), rect_w + gap):
                dc0 = c0 + shift
                dc1 = c1 + shift
                if dc0 < 0 or dc1 >= w:
                    continue
                dashed = _dashed_rect_info(g, (r0, dc0, r1, dc1))
                if dashed is None:
                    continue
                if dashed["marker_pos"] not in unmatched_markers:
                    continue
                matches.append((idx, dashed))

    assert len(matches) == 1, "expected a unique solid/dashed legend pair"
    legend_solid_idx, legend_dashed = matches[0]
    solid_key = int(solid_rects[legend_solid_idx]["marker_color"])
    dashed_key = int(legend_dashed["marker_color"])
    assert solid_key != dashed_key, "legend colors must be distinct"

    out = np.full_like(g, BG)
    for idx, rect in enumerate(solid_rects):
        if idx == legend_solid_idx:
            continue
        marker_color = int(rect["marker_color"])
        assert marker_color in (solid_key, dashed_key), "data marker color must appear in the legend"
        _draw_rectangle(out, rect["bbox"], dashed=marker_color == dashed_key)
    return out


def _expand_bbox(bbox: tuple[int, int, int, int], pad: int, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = shape
    r0, c0, r1, c1 = bbox
    return (
        max(0, r0 - pad),
        max(0, c0 - pad),
        min(h - 1, r1 + pad),
        min(w - 1, c1 + pad),
    )


def _can_place(occupied: np.ndarray, bbox: tuple[int, int, int, int], pad: int = 1) -> bool:
    er0, ec0, er1, ec1 = _expand_bbox(bbox, pad, occupied.shape)
    return not np.any(occupied[er0 : er1 + 1, ec0 : ec1 + 1])


def _mark_occupied(occupied: np.ndarray, bbox: tuple[int, int, int, int], pad: int = 1) -> None:
    er0, ec0, er1, ec1 = _expand_bbox(bbox, pad, occupied.shape)
    occupied[er0 : er1 + 1, ec0 : ec1 + 1] = True


def _rand_interior(rng: random.Random, bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    return rng.randint(r0 + 1, r1 - 1), rng.randint(c0 + 1, c1 - 1)


def _style_assignment(rng: random.Random, count: int) -> list[str]:
    styles = ["solid", "dashed"] * (count // 2)
    if count % 2:
        styles.append(rng.choice(["solid", "dashed"]))
    rng.shuffle(styles)
    return styles


def _generate_case(rng: random.Random) -> tuple[np.ndarray, dict[str, object]] | None:
    h = rng.randint(16, 30)
    w = rng.randint(18, 30)
    solid_color, dashed_color = rng.sample(MARKER_COLORS, 2)

    legend_h = rng.randint(4, min(6, h))
    legend_w = rng.randint(4, min(6, max(4, (w - 1) // 2)))
    gap = rng.randint(1, 2)
    legend_span = 2 * legend_w + gap
    if legend_span > w:
        return None

    corner = rng.choice(["tl", "tr", "bl", "br"])
    top = 0 if corner[0] == "t" else h - legend_h
    left = 0 if corner[1] == "l" else w - legend_span
    solid_on_left = rng.choice([True, False])

    left_bbox = (top, left, top + legend_h - 1, left + legend_w - 1)
    right_bbox = (
        top,
        left + legend_w + gap,
        top + legend_h - 1,
        left + 2 * legend_w + gap - 1,
    )
    legend_solid_bbox = left_bbox if solid_on_left else right_bbox
    legend_dashed_bbox = right_bbox if solid_on_left else left_bbox

    occupied = np.zeros((h, w), dtype=bool)
    _mark_occupied(occupied, left_bbox, pad=1)
    _mark_occupied(occupied, right_bbox, pad=1)

    num_rects = rng.randint(2, 5)
    styles = _style_assignment(rng, num_rects)
    data_rects: list[dict[str, object]] = []
    for style in styles:
        placed = False
        for _ in range(200):
            rect_h = rng.randint(3, min(8, h))
            rect_w = rng.randint(3, min(9, w))
            r0 = rng.randint(0, h - rect_h)
            c0 = rng.randint(0, w - rect_w)
            bbox = (r0, c0, r0 + rect_h - 1, c0 + rect_w - 1)
            if not _can_place(occupied, bbox, pad=1):
                continue
            _mark_occupied(occupied, bbox, pad=1)
            data_rects.append({"bbox": bbox, "style": style})
            placed = True
            break
        if not placed:
            return None

    grid = np.full((h, w), BG, dtype=int)
    _draw_rectangle(grid, legend_solid_bbox, dashed=False)
    _draw_rectangle(grid, legend_dashed_bbox, dashed=True)

    solid_marker = _rand_interior(rng, legend_solid_bbox)
    dashed_marker = _rand_interior(rng, legend_dashed_bbox)
    grid[solid_marker] = solid_color
    grid[dashed_marker] = dashed_color

    for rect in data_rects:
        bbox = rect["bbox"]
        style = rect["style"]
        marker_color = solid_color if style == "solid" else dashed_color
        _draw_rectangle(grid, bbox, dashed=False)
        grid[_rand_interior(rng, bbox)] = marker_color

    meta = {
        "legend_solid_color": solid_color,
        "legend_dashed_color": dashed_color,
        "data_rects": data_rects,
        "shape": (h, w),
    }
    return grid, meta


def _expected_output(meta: dict[str, object]) -> np.ndarray:
    out = np.full(meta["shape"], BG, dtype=int)
    for rect in meta["data_rects"]:
        _draw_rectangle(out, rect["bbox"], dashed=rect["style"] == "dashed")
    return out


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        case = _generate_case(rng)
        if case is not None:
            return case[0]
    raise RuntimeError(f"generate_input failed after {GEN_RETRIES} attempts")
