from __future__ import annotations

import random
from collections import Counter

import numpy as np


class Color:
    BLACK = 0
    BLUE = 1
    RED = 2
    GREEN = 3
    YELLOW = 4
    GREY = 5
    PINK = 6
    ORANGE = 7
    TEAL = 8
    MAROON = 9


MARKER_COLORS = [
    Color.BLUE,
    Color.RED,
    Color.YELLOW,
    Color.GREY,
    Color.PINK,
    Color.ORANGE,
    Color.TEAL,
    Color.MAROON,
]
ALL_SIDES = ("left", "right", "top", "bottom")
LR_SIDES = ("left", "right")
TB_SIDES = ("top", "bottom")

GEN_SIZE_RANGE = (22, 30)
GEN_OBJECT_COUNT_CHOICES = (2, 3, 4)
GEN_DISTANCE_CHOICES = tuple(range(2, 10))
GEN_DISTRACTOR_RANGE = (1, 3)
GEN_RETRIES = 400


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    coords = np.argwhere(grid == int(color))
    if coords.size == 0:
        return []

    h, w = map(int, grid.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r_raw, c_raw in coords:
        r = int(r_raw)
        c = int(c_raw)
        if seen[r, c]:
            continue
        stack = [(r, c)]
        seen[r, c] = True
        comp = []
        while stack:
            cr, cc = stack.pop()
            comp.append((cr, cc))
            for nr, nc in _neighbors4(cr, cc):
                if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and grid[nr, nc] == int(color):
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(np.array(comp, dtype=int))
    return comps


def _green_components(grid: np.ndarray) -> list[np.ndarray]:
    return _components_of_color(grid, Color.GREEN)


def _tab_info(coords: np.ndarray) -> tuple[str, tuple[int, int]]:
    r0, c0, r1, c1 = _bbox_coords(coords)
    top = coords[coords[:, 0] == r0]
    bottom = coords[coords[:, 0] == r1]
    left = coords[coords[:, 1] == c0]
    right = coords[coords[:, 1] == c1]
    candidates = []
    if len(top) == 1:
        candidates.append(("top", top[0]))
    if len(bottom) == 1:
        candidates.append(("bottom", bottom[0]))
    if len(left) == 1:
        candidates.append(("left", left[0]))
    if len(right) == 1:
        candidates.append(("right", right[0]))
    assert len(candidates) == 1, "expected exactly one one-pixel tab side"
    side, tab = candidates[0]
    return side, (int(tab[0]), int(tab[1]))


def _reflect_coords(
    coords: np.ndarray,
    side: str,
    tab: tuple[int, int],
    marker: tuple[int, int],
) -> np.ndarray:
    tr, tc = tab
    mr, mc = marker
    if side in LR_SIDES:
        reflected = coords.copy()
        reflected[:, 1] = int(tc + mc) - coords[:, 1]
        return reflected
    reflected = coords.copy()
    reflected[:, 0] = int(tr + mr) - coords[:, 0]
    return reflected


def _marker_pixels(grid: np.ndarray) -> list[tuple[int, int, int]]:
    markers = []
    colors = sorted(set(map(int, np.unique(grid))) - {Color.BLACK, Color.GREEN})
    for color in colors:
        for comp in _components_of_color(grid, color):
            assert comp.shape[0] == 1, "markers must be isolated single pixels"
            r, c = map(int, comp[0])
            markers.append((int(color), r, c))
    return markers


def _match_marker(
    markers: list[tuple[int, int, int]],
    side: str,
    tab: tuple[int, int],
    used: set[tuple[int, int]],
) -> tuple[int, tuple[int, int]]:
    tr, tc = tab
    candidates = []
    for color, mr, mc in markers:
        if (mr, mc) in used:
            continue
        if side == "left" and mr == tr and mc < tc:
            candidates.append((color, mr, mc))
        elif side == "right" and mr == tr and mc > tc:
            candidates.append((color, mr, mc))
        elif side == "top" and mc == tc and mr < tr:
            candidates.append((color, mr, mc))
        elif side == "bottom" and mc == tc and mr > tr:
            candidates.append((color, mr, mc))
    assert len(candidates) == 1, "expected exactly one marker aligned with each tab"
    color, mr, mc = candidates[0]
    return int(color), (int(mr), int(mc))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror-copy each green tabbed outline so that its tab lands on the aligned marker pixel and inherits that marker's color.

    Input structure:
    - A black background contains one or more 4-connected green outline objects, each with exactly one single-pixel tab on one bbox side.
    - The grid also contains isolated non-green marker pixels; for every object, exactly one marker lies on the tab's row or column in the tab's outward direction.
    - Additional isolated marker pixels may appear as distractors and are not aligned with any tab.

    Rule:
    - Pair each object with the unique aligned marker on the tab's outward row or column.
    - Reflect the whole green object across the axis perpendicular to the tab-marker segment and passing through its midpoint.
    - Paint the reflected copy in the paired marker's color, leaving the original green object unchanged; distractor markers remain as they are.

    Key insight:
    - The marker is not a connection target but the reflected image of the tab itself, so its row or column fixes the reflection direction, its position fixes the mirror axis, and its color paints the whole copy.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    markers = _marker_pixels(g)
    used_markers: set[tuple[int, int]] = set()

    green_components = _green_components(g)
    assert green_components, "expected at least one green outline object"
    for comp in green_components:
        side, tab = _tab_info(comp)
        color, marker = _match_marker(markers, side, tab, used_markers)
        used_markers.add(marker)
        reflected = _reflect_coords(comp, side, tab, marker)
        for rr_raw, cc_raw in reflected:
            rr = int(rr_raw)
            cc = int(cc_raw)
            if 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1]:
                out[rr, cc] = int(color)

    return out


def _draw_frame(patch: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    patch[r0, c0 : c1 + 1] = Color.GREEN
    patch[r1, c0 : c1 + 1] = Color.GREEN
    patch[r0 : r1 + 1, c0] = Color.GREEN
    patch[r0 : r1 + 1, c1] = Color.GREEN


def _make_outline_patch(side: str) -> tuple[np.ndarray, tuple[int, int]]:
    rect_h = random.randint(3, 6)
    rect_w = random.randint(3, 6)
    if side in LR_SIDES:
        patch = np.zeros((rect_h, rect_w + 1), dtype=int)
        frame_r0, frame_c0 = 0, 1 if side == "left" else 0
        tab_r = random.randint(1, rect_h - 2)
        tab_c = 0 if side == "left" else rect_w
    else:
        patch = np.zeros((rect_h + 1, rect_w), dtype=int)
        frame_r0, frame_c0 = (1 if side == "top" else 0), 0
        tab_r = 0 if side == "top" else rect_h
        tab_c = random.randint(1, rect_w - 2)

    _draw_frame(patch, frame_r0, frame_c0, rect_h, rect_w)
    patch[tab_r, tab_c] = Color.GREEN

    frame_r1 = frame_r0 + rect_h - 1
    frame_c1 = frame_c0 + rect_w - 1
    feature_choices = ["none"]
    if rect_h >= 4:
        feature_choices.append("h")
    if rect_w >= 4:
        feature_choices.append("v")
    if rect_h >= 4 and rect_w >= 4:
        feature_choices.append("hv")
    feature = random.choice(feature_choices)

    if "h" in feature:
        rr = random.randint(frame_r0 + 1, frame_r1 - 1)
        patch[rr, frame_c0 + 1 : frame_c1] = Color.GREEN
    if "v" in feature:
        cc = random.randint(frame_c0 + 1, frame_c1 - 1)
        patch[frame_r0 + 1 : frame_r1, cc] = Color.GREEN

    return patch, (int(tab_r), int(tab_c))


def _separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1) -> bool:
    return (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _choose_sides(count: int) -> list[str]:
    if count == 2:
        return [random.choice(LR_SIDES), random.choice(TB_SIDES)]
    if count == 3:
        sides = [random.choice(LR_SIDES), random.choice(TB_SIDES), random.choice(ALL_SIDES)]
        random.shuffle(sides)
        return sides
    sides = list(ALL_SIDES)
    random.shuffle(sides)
    return sides


def _place_object(
    grid: np.ndarray,
    side: str,
    distance: int,
    color: int,
    reserved: list[tuple[int, int, int, int]],
    used_lr_rows: set[int],
    used_tb_cols: set[int],
) -> dict | None:
    h, w = map(int, grid.shape)

    for _ in range(120):
        patch, tab_local = _make_outline_patch(side)
        ph, pw = map(int, patch.shape)

        if side == "left":
            left_min = distance + pw - 1
            left_max = w - pw
            top_min = 0
            top_max = h - ph
        elif side == "right":
            left_min = 0
            left_max = w - (2 * pw - 1 + distance)
            top_min = 0
            top_max = h - ph
        elif side == "top":
            left_min = 0
            left_max = w - pw
            top_min = distance + ph - 1
            top_max = h - ph
        else:
            left_min = 0
            left_max = w - pw
            top_min = 0
            top_max = h - (2 * ph - 1 + distance)

        if left_min > left_max or top_min > top_max:
            continue

        top = random.randint(top_min, top_max)
        left = random.randint(left_min, left_max)

        coords = np.argwhere(patch == Color.GREEN)
        coords = coords + np.array([top, left], dtype=int)
        bbox = _bbox_coords(coords)
        tab = (top + tab_local[0], left + tab_local[1])

        if side in LR_SIDES and tab[0] in used_lr_rows:
            continue
        if side in TB_SIDES and tab[1] in used_tb_cols:
            continue

        if side == "left":
            marker = (tab[0], tab[1] - distance)
        elif side == "right":
            marker = (tab[0], tab[1] + distance)
        elif side == "top":
            marker = (tab[0] - distance, tab[1])
        else:
            marker = (tab[0] + distance, tab[1])

        if side in LR_SIDES and marker[1] in used_tb_cols:
            continue
        if side in TB_SIDES and marker[0] in used_lr_rows:
            continue

        reflected = _reflect_coords(coords, side, tab, marker)
        if reflected[:, 0].min() < 0 or reflected[:, 0].max() >= h or reflected[:, 1].min() < 0 or reflected[:, 1].max() >= w:
            continue
        reflected_bbox = _bbox_coords(reflected)
        marker_bbox = (marker[0], marker[1], marker[0], marker[1])

        if not _separated(bbox, reflected_bbox):
            continue
        if marker_bbox[0] < 0 or marker_bbox[0] >= h or marker_bbox[1] < 0 or marker_bbox[1] >= w:
            continue

        bad = False
        for other in reserved:
            if not _separated(bbox, other):
                bad = True
                break
            if not _separated(reflected_bbox, other):
                bad = True
                break
            if not _separated(marker_bbox, other):
                bad = True
                break
        if bad:
            continue

        if grid[marker] != Color.BLACK:
            continue

        return {
            "coords": coords,
            "bbox": bbox,
            "tab": tab,
            "marker": marker,
            "marker_color": int(color),
            "reflected_bbox": reflected_bbox,
        }

    return None


def _place_distractors(
    grid: np.ndarray,
    reserved: list[tuple[int, int, int, int]],
    colors: list[int],
    forbidden_rows: set[int],
    forbidden_cols: set[int],
) -> bool:
    h, w = map(int, grid.shape)
    for color in colors:
        placed = False
        for _ in range(200):
            r = random.randrange(h)
            c = random.randrange(w)
            if r in forbidden_rows or c in forbidden_cols:
                continue
            if grid[r, c] != Color.BLACK:
                continue
            bbox = (r, c, r, c)
            if any(not _separated(bbox, other) for other in reserved):
                continue
            grid[r, c] = int(color)
            reserved.append(bbox)
            placed = True
            break
        if not placed:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GEN_SIZE_RANGE)
    w = random.randint(*GEN_SIZE_RANGE)
    grid = np.zeros((h, w), dtype=int)

    count = random.choice(GEN_OBJECT_COUNT_CHOICES)
    sides = _choose_sides(count)
    distances = random.sample(GEN_DISTANCE_CHOICES, count)
    distractor_count = random.randint(*GEN_DISTRACTOR_RANGE)
    colors = random.sample(MARKER_COLORS, count + distractor_count)
    object_colors = colors[:count]
    distractor_colors = colors[count:]

    reserved: list[tuple[int, int, int, int]] = []
    used_lr_rows: set[int] = set()
    used_tb_cols: set[int] = set()

    placements = []
    for side, distance, color in zip(sides, distances, object_colors):
        placed = _place_object(grid, side, distance, color, reserved, used_lr_rows, used_tb_cols)
        if placed is None:
            return None
        placements.append(placed)
        reserved.append(placed["bbox"])
        reserved.append(placed["reflected_bbox"])
        reserved.append((placed["marker"][0], placed["marker"][1], placed["marker"][0], placed["marker"][1]))
        if side in LR_SIDES:
            used_lr_rows.add(int(placed["tab"][0]))
        else:
            used_tb_cols.add(int(placed["tab"][1]))

    for placed in placements:
        for r, c in placed["coords"]:
            grid[int(r), int(c)] = Color.GREEN
        mr, mc = placed["marker"]
        grid[mr, mc] = int(placed["marker_color"])

    if not _place_distractors(grid, reserved, distractor_colors, used_lr_rows, used_tb_cols):
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    diff = int(np.count_nonzero(out != grid))
    if diff < 8:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
