from __future__ import annotations

from collections import Counter
from functools import lru_cache
import random

import numpy as np


BLACK = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(comp)
    return out


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise ValueError("expected non-empty mask")
    rs = coords[:, 0]
    cs = coords[:, 1]
    return int(rs.min()), int(cs.min()), int(rs.max()), int(cs.max())


def _largest_component_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    comps = _components(mask)
    if not comps:
        raise ValueError("expected at least one component")
    return _bbox(max(comps, key=len))


def _legend_rows(
    grid: np.ndarray, panel_bbox: tuple[int, int, int, int]
) -> list[dict[str, int]]:
    r0, c0, r1, c1 = panel_bbox
    rows: list[dict[str, int]] = []
    for r in range(r0, r1 + 1):
        rel = grid[r, c0 : c1 + 1]
        cols = np.flatnonzero(rel != BLACK)
        if cols.size == 0:
            continue
        key_c = int(cols[0])
        rows.append({"rel_r": r - r0, "key_c": key_c, "color": int(rel[key_c])})
    if not rows:
        raise ValueError("legend panel must contain keyed rows")
    return rows


def _touch_info(
    coords: list[tuple[int, int]], occ_bbox: tuple[int, int, int, int]
) -> dict[str, set[int]]:
    r0, c0, r1, c1 = occ_bbox
    info = {
        "left_rows": set(),
        "right_rows": set(),
        "top_cols": set(),
        "bottom_cols": set(),
    }
    for r, c in coords:
        if c == c0 - 1 and r0 <= r <= r1:
            info["left_rows"].add(int(r))
        if c == c1 + 1 and r0 <= r <= r1:
            info["right_rows"].add(int(r))
        if r == r0 - 1 and c0 <= c <= c1:
            info["top_cols"].add(int(c))
        if r == r1 + 1 and c0 <= c <= c1:
            info["bottom_cols"].add(int(c))
    return info


def _matching_size(adj_masks: list[int]) -> int:
    n = len(adj_masks)

    @lru_cache(None)
    def dp(mask: int) -> int:
        if mask == 0:
            return 0
        lsb = mask & -mask
        v = lsb.bit_length() - 1
        best = dp(mask ^ lsb)
        options = adj_masks[v] & (mask ^ lsb)
        while options:
            lsb2 = options & -options
            best = max(best, 1 + dp(mask ^ lsb ^ lsb2))
            options ^= lsb2
        return best

    return dp((1 << n) - 1)


def _count_objects(
    grid: np.ndarray,
    panel_bbox: tuple[int, int, int, int],
    occ_bbox: tuple[int, int, int, int],
    colors: list[int],
) -> tuple[dict[int, int], dict[int, int]]:
    r0, c0, r1, c1 = panel_bbox
    scene_mask = np.ones_like(grid, dtype=bool)
    scene_mask[r0 : r1 + 1, c0 : c1 + 1] = False

    real_counts: dict[int, int] = {}
    visible_counts: dict[int, int] = {}
    for color in sorted(set(colors)):
        comps = _components((grid == int(color)) & scene_mask)
        visible_counts[int(color)] = len(comps)
        infos = [_touch_info(comp, occ_bbox) for comp in comps]
        adj_masks = [0] * len(comps)
        for i in range(len(comps)):
            for j in range(i + 1, len(comps)):
                a = infos[i]
                b = infos[j]
                linked = False
                if a["left_rows"] & b["right_rows"]:
                    linked = True
                if a["right_rows"] & b["left_rows"]:
                    linked = True
                if a["top_cols"] & b["bottom_cols"]:
                    linked = True
                if a["bottom_cols"] & b["top_cols"]:
                    linked = True
                if linked:
                    adj_masks[i] |= 1 << j
                    adj_masks[j] |= 1 << i
        real_counts[int(color)] = len(comps) - _matching_size(adj_masks)
    return real_counts, visible_counts


def _render_legend(
    panel_shape: tuple[int, int], rows: list[dict[str, int]], counts: dict[int, int]
) -> np.ndarray:
    out = np.full(panel_shape, BLACK, dtype=int)
    for row in rows:
        rr = int(row["rel_r"])
        key_c = int(row["key_c"])
        color = int(row["color"])
        out[rr, key_c] = color
        for i in range(int(counts[color])):
            cc = key_c + 2 + 2 * i
            if cc >= out.shape[1]:
                raise ValueError("legend row too short for required dot count")
            out[rr, cc] = color
    return out


def _analyze(grid: np.ndarray):
    panel_bbox = _mask_bbox(grid == BLACK)
    occ_bbox = _largest_component_bbox(grid == GREY)
    rows = _legend_rows(grid, panel_bbox)
    colors = [int(row["color"]) for row in rows]
    real_counts, visible_counts = _count_objects(grid, panel_bbox, occ_bbox, colors)
    return panel_bbox, occ_bbox, rows, real_counts, visible_counts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rewrite the legend so each keyed row shows the true count of objects hidden by the grey curtain.

    Input structure:
    - A non-black background contains one solid grey rectangular occluder and several monochrome sprites.
    - Some sprites are fully visible on one side of the occluder; some are split into two visible fragments on opposite sides.
    - A black rectangular legend panel sits in one corner; each used legend row starts with one keyed color pixel and then some decoy dots.

    Rule:
    - Ignore the old legend dots and count objects separately for each keyed color.
    - Count ordinary same-color 4-connected components outside the legend, but merge a left/right pair when both fragments touch opposite occluder sides on at least one common row; likewise merge a top/bottom pair when they touch opposite sides on at least one common column.
    - Output only the legend panel crop, redrawn in black with each keyed row showing the corrected number of dots at every other cell to the right of the key pixel.

    Key insight:
    - Visible components are not always real objects: a grey rectangle can hide the middle of one sprite, so aligned edge-touching fragments across the occluder must be merged before writing the legend.
    """
    g = np.asarray(grid, dtype=int)
    panel_bbox, _occ_bbox, rows, real_counts, _visible_counts = _analyze(g)
    pr0, pc0, pr1, pc1 = panel_bbox
    return _render_legend((pr1 - pr0 + 1, pc1 - pc0 + 1), rows, real_counts)


SHAPES = [
    np.array([[1, 1], [1, 1]], dtype=int),
    np.array([[1, 0], [1, 1]], dtype=int),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=int),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=int),
    np.array([[1, 0], [1, 1], [1, 0]], dtype=int),
    np.array([[1, 1, 1], [1, 0, 0]], dtype=int),
]


def _panel_bbox_for_corner(
    h: int, w: int, panel_h: int, panel_w: int, corner: str
) -> tuple[int, int, int, int]:
    if corner == "tl":
        return 0, 0, panel_h - 1, panel_w - 1
    if corner == "tr":
        return 0, w - panel_w, panel_h - 1, w - 1
    if corner == "bl":
        return h - panel_h, 0, h - 1, panel_w - 1
    return h - panel_h, w - panel_w, h - 1, w - 1


def _pick_counts(rng: random.Random):
    while True:
        bg = rng.choice(PALETTE)
        colors = rng.sample([c for c in PALETTE if c != bg], rng.randint(3, 4))
        real_counts = {color: rng.randint(1, 3) for color in colors}
        if max(real_counts.values()) < 2:
            continue
        split_counts = {}
        budget = 4
        for color in colors:
            max_split = min(2, real_counts[color], budget)
            split_counts[color] = rng.randint(0, max_split)
            budget -= split_counts[color]
        if sum(split_counts.values()) == 0 or sum(split_counts.values()) > 4:
            continue
        if sum(real_counts[color] for color in colors) > 8:
            continue
        if not any(split_counts[color] == 0 for color in colors):
            continue
        if not any(split_counts[color] > 0 for color in colors):
            continue
        unsplit_total = sum(real_counts[color] - split_counts[color] for color in colors)
        if unsplit_total == 0:
            continue
        return bg, colors, real_counts, split_counts


def _make_panel_layout(
    rng: random.Random, colors: list[int], visible_max: int
) -> tuple[list[dict[str, int]], int, int]:
    row_step = rng.choice([1, 2])
    top_pad = rng.randint(0, 1)
    bottom_pad = rng.randint(0, 1)
    key_c = rng.randint(1, 2)
    rows = []
    for i, color in enumerate(colors):
        rows.append({"rel_r": top_pad + i * row_step, "key_c": key_c, "color": int(color)})
    panel_h = rows[-1]["rel_r"] + 1 + bottom_pad
    panel_w = key_c + 2 * visible_max + rng.randint(2, 3)
    return rows, panel_h, panel_w


def _mark_rect(mask: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = bbox
    mask[r0 : r1 + 1, c0 : c1 + 1] = True


def _cells_from_mask(mask: np.ndarray, r0: int, c0: int) -> list[tuple[int, int]]:
    return [(int(r0 + r), int(c0 + c)) for r, c in np.argwhere(mask)]


def _far_from_bbox(cells: list[tuple[int, int]], bbox: tuple[int, int, int, int], margin: int) -> bool:
    r0, c0, r1, c1 = bbox
    for r, c in cells:
        if r0 - margin <= r <= r1 + margin and c0 - margin <= c <= c1 + margin:
            return False
    return True


def _can_place(
    cells: list[tuple[int, int]], reserve: np.ndarray, occupied: np.ndarray
) -> bool:
    h, w = reserve.shape
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if reserve[r, c]:
            return False
    for r, c in cells:
        rr0, rr1 = max(0, r - 1), min(h - 1, r + 1)
        cc0, cc1 = max(0, c - 1), min(w - 1, c + 1)
        if occupied[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
            return False
    return True


def _paint_cells(
    grid: np.ndarray,
    reserve: np.ndarray,
    occupied: np.ndarray,
    cells: list[tuple[int, int]],
    color: int,
) -> None:
    for r, c in cells:
        grid[r, c] = int(color)
        reserve[r, c] = True
        occupied[r, c] = True


def _allocate_bands(span: int, thicknesses: list[int], rng: random.Random) -> list[int] | None:
    needed = sum(thicknesses) + max(0, len(thicknesses) - 1)
    slack = span - needed
    if slack < 0:
        return None
    extras = [0] * (len(thicknesses) + 1)
    for _ in range(slack):
        extras[rng.randrange(len(extras))] += 1
    starts: list[int] = []
    pos = extras[0]
    for i, thick in enumerate(thicknesses):
        starts.append(pos)
        pos += thick + 1 + extras[i + 1]
    return starts


def _place_split_objects(
    grid: np.ndarray,
    reserve: np.ndarray,
    occupied: np.ndarray,
    occ_bbox: tuple[int, int, int, int],
    orientation: str,
    split_sequence: list[int],
    rng: random.Random,
) -> bool:
    if not split_sequence:
        return True
    r0, c0, r1, c1 = occ_bbox
    thicknesses = [rng.randint(1, 2) for _ in split_sequence]
    span = (r1 - r0 + 1) if orientation == "vertical" else (c1 - c0 + 1)
    starts = _allocate_bands(span, thicknesses, rng)
    if starts is None:
        return False

    for color, start, thick in zip(split_sequence, starts, thicknesses):
        if orientation == "vertical":
            left_max = c0 - 2
            right_max = grid.shape[1] - (c1 + 1) - 2
            if min(left_max, right_max) < 2:
                return False
            left_len = rng.randint(2, min(5, left_max))
            right_len = rng.randint(2, min(5, right_max))
            rows = range(r0 + start, r0 + start + thick)
            cells = []
            for rr in rows:
                cells.extend((rr, cc) for cc in range(c0 - left_len, c0))
                cells.extend((rr, cc) for cc in range(c1 + 1, c1 + 1 + right_len))
        else:
            top_max = r0 - 2
            bottom_max = grid.shape[0] - (r1 + 1) - 2
            if min(top_max, bottom_max) < 2:
                return False
            top_len = rng.randint(2, min(5, top_max))
            bottom_len = rng.randint(2, min(5, bottom_max))
            cols = range(c0 + start, c0 + start + thick)
            cells = []
            for cc in cols:
                cells.extend((rr, cc) for rr in range(r0 - top_len, r0))
                cells.extend((rr, cc) for rr in range(r1 + 1, r1 + 1 + bottom_len))
        if not _can_place(cells, reserve, occupied):
            return False
        _paint_cells(grid, reserve, occupied, cells, color)
    return True


def _place_unsplit_objects(
    grid: np.ndarray,
    reserve: np.ndarray,
    occupied: np.ndarray,
    occ_bbox: tuple[int, int, int, int],
    orientation: str,
    colors: list[int],
    rng: random.Random,
) -> bool:
    h, w = grid.shape
    or0, oc0, or1, oc1 = occ_bbox
    for color in colors:
        placed = False
        for _ in range(300):
            shape = np.rot90(rng.choice(SHAPES), k=rng.randrange(4))
            sh, sw = shape.shape
            if orientation == "vertical":
                ranges = []
                if oc0 - sw - 2 >= 1:
                    ranges.append((1, h - sh - 1, 1, oc0 - sw - 2))
                if oc1 + 2 <= w - sw - 1:
                    ranges.append((1, h - sh - 1, oc1 + 2, w - sw - 1))
            else:
                ranges = []
                if or0 - sh - 2 >= 1:
                    ranges.append((1, or0 - sh - 2, 1, w - sw - 1))
                if or1 + 2 <= h - sh - 1:
                    ranges.append((or1 + 2, h - sh - 1, 1, w - sw - 1))
            if not ranges:
                return False
            r_lo, r_hi, c_lo, c_hi = rng.choice(ranges)
            if r_hi < r_lo or c_hi < c_lo:
                continue
            r_start = rng.randint(r_lo, r_hi)
            c_start = rng.randint(c_lo, c_hi)
            cells = _cells_from_mask(shape, r_start, c_start)
            if not _far_from_bbox(cells, occ_bbox, margin=1):
                continue
            if not _can_place(cells, reserve, occupied):
                continue
            _paint_cells(grid, reserve, occupied, cells, color)
            placed = True
            break
        if not placed:
            return False
    return True


def _place_occluder(
    h: int,
    w: int,
    panel_h: int,
    panel_w: int,
    occ_h: int,
    occ_w: int,
    corner: str,
    rng: random.Random,
) -> tuple[int, int, int, int] | None:
    if corner[0] == "t":
        r_lo, r_hi = panel_h + 2, h - occ_h - 2
    else:
        r_lo, r_hi = 2, h - occ_h - panel_h - 2
    if corner[1] == "l":
        c_lo, c_hi = panel_w + 2, w - occ_w - 2
    else:
        c_lo, c_hi = 2, w - occ_w - panel_w - 2
    if r_lo > r_hi or c_lo > c_hi:
        return None

    target_r = max(r_lo, min((h - occ_h) // 2, r_hi))
    target_c = max(c_lo, min((w - occ_w) // 2, c_hi))
    r0 = min(r_hi, max(r_lo, target_r + rng.randint(-1, 1)))
    c0 = min(c_hi, max(c_lo, target_c + rng.randint(-1, 1)))
    return r0, c0, r0 + occ_h - 1, c0 + occ_w - 1


def _build_case(rng: random.Random) -> np.ndarray | None:
    bg, colors, real_counts, split_counts = _pick_counts(rng)
    visible_counts = {color: real_counts[color] + split_counts[color] for color in colors}
    rows, panel_h, panel_w = _make_panel_layout(rng, colors, max(visible_counts.values()))

    orientation = rng.choice(["vertical", "horizontal"])
    split_sequence = []
    for color in colors:
        split_sequence.extend([color] * split_counts[color])
    rng.shuffle(split_sequence)

    if orientation == "vertical":
        occ_w = rng.randint(3, 5)
        occ_h = rng.randint(max(9, len(split_sequence) * 2 + 1), 16)
    else:
        occ_h = rng.randint(3, 5)
        occ_w = rng.randint(max(9, len(split_sequence) * 2 + 1), 16)

    corner = rng.choice(["tl", "tr", "bl", "br"])
    min_h = max(18, panel_h + occ_h + 4)
    min_w = max(20, panel_w + occ_w + 6)
    if min_h > 30 or min_w > 30:
        return None
    h = rng.randint(min_h, 30)
    w = rng.randint(min_w, 30)

    panel_bbox = _panel_bbox_for_corner(h, w, panel_h, panel_w, corner)
    occ_bbox = _place_occluder(h, w, panel_h, panel_w, occ_h, occ_w, corner, rng)
    if occ_bbox is None:
        return None

    grid = np.full((h, w), bg, dtype=int)
    pr0, pc0, pr1, pc1 = panel_bbox
    grid[pr0 : pr1 + 1, pc0 : pc1 + 1] = BLACK
    for row in rows:
        rr = row["rel_r"]
        key_c = row["key_c"]
        color = row["color"]
        grid[pr0 + rr, pc0 + key_c] = color
        for i in range(visible_counts[color]):
            grid[pr0 + rr, pc0 + key_c + 2 + 2 * i] = color

    or0, oc0, or1, oc1 = occ_bbox
    grid[or0 : or1 + 1, oc0 : oc1 + 1] = GREY

    reserve = np.zeros((h, w), dtype=bool)
    occupied = np.zeros((h, w), dtype=bool)
    _mark_rect(reserve, panel_bbox)
    _mark_rect(occupied, panel_bbox)
    _mark_rect(reserve, occ_bbox)

    if not _place_split_objects(grid, reserve, occupied, occ_bbox, orientation, split_sequence, rng):
        return None

    unsplit_colors: list[int] = []
    for color in colors:
        unsplit_colors.extend([color] * (real_counts[color] - split_counts[color]))
    rng.shuffle(unsplit_colors)
    if not _place_unsplit_objects(grid, reserve, occupied, occ_bbox, orientation, unsplit_colors, rng):
        return None

    solver_panel_bbox, _solver_occ_bbox, solver_rows, solver_real, solver_visible = _analyze(grid)
    if solver_panel_bbox != panel_bbox:
        return None
    if solver_real != real_counts:
        return None
    if solver_visible != visible_counts:
        return None
    expected = _render_legend((panel_h, panel_w), solver_rows, real_counts)
    if not np.array_equal(main(grid), expected):
        return None
    if not any(solver_visible[color] != solver_real[color] for color in colors):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _build_case(random)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid curtain-counting task")
