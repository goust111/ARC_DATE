from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 7]
MAX_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _extract_objects(grid: np.ndarray) -> list[dict]:
    comps = _components((grid != BLACK) & (grid != GREY))
    objects = []
    for coords in comps:
        colors = {int(grid[r, c]) for r, c in coords}
        assert len(colors) == 1, "each foreground object must be monochrome"
        color = next(iter(colors))
        b = _bbox(coords)
        width = b[3] - b[1] + 1
        objects.append(
            {
                "color": color,
                "coords": coords,
                "bbox": b,
                "width": int(width),
            }
        )
    return objects


def _shift_scene(grid: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    g = np.asarray(grid, dtype=int)
    out = np.full_like(g, BLACK)
    out[g == GREY] = GREY
    h, w = g.shape
    objects = _extract_objects(g)

    used_colors = set()
    for obj in objects:
        assert obj["color"] not in used_colors, "object colors must be distinct"
        used_colors.add(obj["color"])

        shift = obj["width"]
        dst = obj["coords"] + np.array([0, shift], dtype=int)
        assert bool((0 <= dst[:, 0]).all() and (dst[:, 0] < h).all())
        assert bool((0 <= dst[:, 1]).all() and (dst[:, 1] < w).all())
        assert not bool((out[dst[:, 0], dst[:, 1]] != BLACK).any())
        out[dst[:, 0], dst[:, 1]] = obj["color"]
        obj["shifted_coords"] = dst
    return out, objects


def _enclosed_black_regions(grid: np.ndarray) -> list[np.ndarray]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    regions: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if g[r, c] != BLACK or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            touches_edge = False
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                if rr == 0 or rr == h - 1 or cc == 0 or cc == w - 1:
                    touches_edge = True
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(g.shape, nr, nc) and g[nr, nc] == BLACK and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            if not touches_edge:
                regions.append(np.array(pts, dtype=int))
    return regions


def _region_boundary_color_sets(region: np.ndarray, grid: np.ndarray) -> dict[int, set[tuple[int, int]]]:
    touching: dict[int, set[tuple[int, int]]] = {}
    for r, c in region:
        for nr, nc in _neighbors4(int(r), int(c)):
            if not _in_bounds(grid.shape, nr, nc):
                continue
            color = int(grid[nr, nc])
            if color in (BLACK, GREY):
                continue
            touching.setdefault(color, set()).add((nr, nc))
    return touching


def main(grid: np.ndarray) -> np.ndarray:
    """Shift each colored object right by its own width, then fill the new enclosed cavities with the farthest adjacent object's color.

    Input structure:
    - Black is the background and grey cells are fixed vertical obstacle walls.
    - The non-grey non-black foreground consists of 3-5 separate monochrome 4-connected colored objects.
    - Each colored object's bounding-box width is the distance it will move to the right.

    Rule:
    - Identify every non-grey foreground object and clear its original cells.
    - Repaint each object shifted horizontally right by its own bounding-box width while keeping all grey walls fixed.
    - On the shifted grid, find every 4-connected black region that does not touch the outer border.
    - For each such enclosed region, inspect the non-black non-grey colors on its 4-neighbor boundary and fill the region with the color whose source object has the largest bounding-box width; if widths tie, use the color with more boundary pixels on that region.

    Key insight:
    - The cavity does not exist in the input; it appears only after the translation, and the same visible width attribute controls both motion distance and fill priority.
    """
    shifted, objects = _shift_scene(grid)
    width_by_color = {obj["color"]: obj["width"] for obj in objects}
    out = shifted.copy()

    for region in _enclosed_black_regions(shifted):
        touching = _region_boundary_color_sets(region, shifted)
        assert touching, "each enclosed region must border at least one moved object"

        winner = None
        best_score = None
        ties = 0
        for color, pixels in touching.items():
            score = (width_by_color[color], len(pixels))
            if best_score is None or score > best_score:
                best_score = score
                winner = color
                ties = 1
            elif score == best_score:
                ties += 1
        assert ties == 1, "fill winner must be uniquely determined"

        out[region[:, 0], region[:, 1]] = int(winner)

    return out


def _bridge_shape(width: int, span: int, top_bridge: bool) -> np.ndarray:
    assert 2 <= span <= width <= 7
    mask = np.zeros((2, width), dtype=int)
    bridge_start = width - span
    support_end = bridge_start
    if top_bridge:
        mask[1, bridge_start:] = 1
        mask[0, : support_end + 1] = 1
    else:
        mask[0, bridge_start:] = 1
        mask[1, : support_end + 1] = 1
    return mask


def _extra_shape(width: int) -> np.ndarray:
    if width == 2:
        return np.array([[1, 1], [1, 0]], dtype=int)
    if width == 3:
        return np.array([[1, 0, 0], [1, 1, 1]], dtype=int)
    raise AssertionError("unsupported extra width")


def _stamp_mask(grid: np.ndarray, mask: np.ndarray, color: int, r0: int, c0: int) -> None:
    h, w = mask.shape
    assert 0 <= r0 and 0 <= c0 and r0 + h <= grid.shape[0] and c0 + w <= grid.shape[1]
    rows, cols = np.where(mask == 1)
    rr = rows + r0
    cc = cols + c0
    assert not bool((grid[rr, cc] != BLACK).any())
    grid[rr, cc] = color


def _bridge_top_left(rt: int, cl: int, width: int, span: int, top_bridge: bool) -> tuple[int, int]:
    c0 = cl + 1 - (width - span)
    r0 = rt - 1 if top_bridge else rt
    return int(r0), int(c0)


def _wall_left_min(widths: list[int], span: int) -> int:
    return max(2 * w - span - 1 for w in widths)


def _build_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    templates = [
        {"pairs": [(2, 3, 5), (3, 6, 4)], "extra_width": 2},
        {"pairs": [(2, 3, 6), (3, 5, 4)], "extra_width": 2},
        {"pairs": [(2, 4, 7), (3, 6, 5)], "extra_width": 2},
        {"pairs": [(2, 5, 7), (3, 6, 4)], "extra_width": 3},
    ]
    template = random.choice(templates)
    use_extra = random.random() < 0.5

    num_objects = 4 + int(use_extra)
    colors = random.sample(PALETTE, num_objects)
    random.shuffle(colors)

    pair_specs = template["pairs"][:]
    if random.random() < 0.5:
        pair_specs.reverse()

    chamber_rows = []
    rt = random.randint(4, 5)
    for _ in range(2):
        inner_h = random.randint(2, 3)
        rb = rt + inner_h + 1
        chamber_rows.append((rt, rb))
        rt = rb + random.randint(4, 5)
    h = chamber_rows[-1][1] + 4 + int(use_extra)
    if h > 25:
        return None

    chamber_cols = []
    base = 0
    for (span, top_w, bot_w) in pair_specs:
        lo = max(_wall_left_min([top_w, bot_w], span), 7 + base)
        chamber_cols.append(random.randint(lo, lo + 2))
        base += 1

    max_right_wall = max(cl + span + 1 for cl, (span, _, _) in zip(chamber_cols, pair_specs))
    w = max_right_wall + 4
    if w > 26:
        return None

    inp = np.zeros((h, w), dtype=int)
    shifted = np.zeros((h, w), dtype=int)
    chambers = []
    object_specs = []

    color_idx = 0
    for (rt, rb), cl, (span, top_w, bot_w) in zip(chamber_rows, chamber_cols, pair_specs):
        cr = cl + span + 1
        inp[rt : rb + 1, cl] = GREY
        inp[rt : rb + 1, cr] = GREY
        shifted[rt : rb + 1, cl] = GREY
        shifted[rt : rb + 1, cr] = GREY

        top_color = colors[color_idx]
        bot_color = colors[color_idx + 1]
        color_idx += 2

        top_mask = _bridge_shape(top_w, span, top_bridge=True)
        bot_mask = _bridge_shape(bot_w, span, top_bridge=False)

        top_out_r, top_out_c = _bridge_top_left(rt, cl, top_w, span, True)
        bot_out_r, bot_out_c = _bridge_top_left(rb, cl, bot_w, span, False)
        top_in_r, top_in_c = top_out_r, top_out_c - top_w
        bot_in_r, bot_in_c = bot_out_r, bot_out_c - bot_w

        if min(top_in_c, bot_in_c) < 0:
            return None

        _stamp_mask(inp, top_mask, top_color, top_in_r, top_in_c)
        _stamp_mask(inp, bot_mask, bot_color, bot_in_r, bot_in_c)
        _stamp_mask(shifted, top_mask, top_color, top_out_r, top_out_c)
        _stamp_mask(shifted, bot_mask, bot_color, bot_out_r, bot_out_c)

        winner = top_color if top_w > bot_w else bot_color
        chambers.append(
            {
                "region": (slice(rt + 1, rb), slice(cl + 1, cr)),
                "winner": winner,
                "boundary_colors": {top_color, bot_color},
            }
        )
        object_specs.extend(
            [
                {"color": top_color, "width": top_w},
                {"color": bot_color, "width": bot_w},
            ]
        )

    if use_extra:
        extra_width = template["extra_width"]
        extra_color = colors[color_idx]
        extra_mask = _extra_shape(extra_width)
        extra_r = h - 2
        extra_c_out_max = min(chamber_cols) - 3
        extra_c_out = random.randint(extra_width, max(extra_width, extra_c_out_max))
        extra_c_in = extra_c_out - extra_width
        if extra_c_out + extra_mask.shape[1] > w or extra_c_in < 0:
            return None
        _stamp_mask(inp, extra_mask, extra_color, extra_r, extra_c_in)
        _stamp_mask(shifted, extra_mask, extra_color, extra_r, extra_c_out)
        object_specs.append({"color": extra_color, "width": extra_width})

    widths = [spec["width"] for spec in object_specs]
    if len(widths) != len(set(widths)):
        return None

    if len(_extract_objects(inp)) != len(object_specs):
        return None

    if _enclosed_black_regions(inp):
        return None

    shifted_regions = _enclosed_black_regions(shifted)
    if len(shifted_regions) != len(chambers):
        return None

    expected = shifted.copy()
    for chamber in chambers:
        region = chamber["region"]
        expected[region] = chamber["winner"]

    solved = main(inp)
    if not np.array_equal(solved, expected):
        return None

    shifted_from_solver, objects = _shift_scene(inp)
    if not np.array_equal(shifted_from_solver, shifted):
        return None
    width_by_color = {obj["color"]: obj["width"] for obj in objects}

    regions = _enclosed_black_regions(shifted)
    filled_colors = []
    for region in regions:
        touching = _region_boundary_color_sets(region, shifted)
        if len(touching) != 2:
            return None
        scores = {color: (width_by_color[color], len(pixels)) for color, pixels in touching.items()}
        winner = max(scores, key=scores.get)
        if list(scores.values()).count(scores[winner]) != 1:
            return None
        filled_colors.append(winner)
    if len(set(filled_colors)) != len(filled_colors):
        return None

    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        built = _build_attempt()
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate a valid ARC input")
