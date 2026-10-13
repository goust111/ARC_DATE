from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np


_ARC_UTILS = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS) not in sys.path:
    sys.path.append(str(_ARC_UTILS))

from arc_utils import crop_bbox, find_components, get_bg, neighbors4, paste


BG = 0
BAR = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
GRID_SIDE_RANGE = (17, 24)
SPRITE_AREA_RANGE = (4, 6)
GENERATION_RETRIES = 400
LEFT_SIDE_WEIGHT = 0.8
TOP_SIDE_WEIGHT = 0.86


def _edge_name(shape: tuple[int, int], r: int, c: int) -> str:
    H, W = map(int, shape)
    if r == 0:
        return "top"
    if r == H - 1:
        return "bottom"
    if c == 0:
        return "left"
    if c == W - 1:
        return "right"
    raise AssertionError("cell is not on the border")


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _bbox_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _expand_bbox(bbox: tuple[int, int, int, int], pad: int) -> tuple[int, int, int, int]:
    return bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad


def _cell_set_from_patch(patch: np.ndarray, r0: int, c0: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for rr, cc in np.argwhere(patch != BG):
        cells.add((int(r0 + rr), int(c0 + cc)))
    return cells


def _component_colors(grid: np.ndarray, coords: np.ndarray) -> Counter:
    return Counter(int(grid[int(r), int(c)]) for r, c in coords)


def _marker_info(grid: np.ndarray, coords: np.ndarray) -> tuple[int, tuple[int, int]]:
    counts = _component_colors(grid, coords)
    singles = [color for color, count in counts.items() if count == 1]
    assert len(singles) == 1
    marker_color = int(singles[0])
    marker_pixels = [(int(r), int(c)) for r, c in coords if int(grid[int(r), int(c)]) == marker_color]
    assert len(marker_pixels) == 1
    return marker_color, marker_pixels[0]


def _is_isolated_border_pixel(grid: np.ndarray, r: int, c: int, bg: int) -> bool:
    if int(grid[r, c]) == int(bg):
        return False
    H, W = grid.shape
    if not (r == 0 or r == H - 1 or c == 0 or c == W - 1):
        return False
    for rr, cc in neighbors4(int(r), int(c)):
        if 0 <= rr < H and 0 <= cc < W and int(grid[rr, cc]) != int(bg):
            return False
    return True


def _parse_scene(grid: np.ndarray) -> tuple[int, dict[int, dict], list[dict], list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape

    anchors: dict[int, dict] = {}
    for r in range(H):
        for c in range(W):
            if _is_isolated_border_pixel(g, int(r), int(c), bg):
                color = int(g[r, c])
                anchors[color] = {
                    "color": color,
                    "pos": (int(r), int(c)),
                    "edge": _edge_name(g.shape, int(r), int(c)),
                }

    bars: list[dict] = []
    for comp in find_components(
        g,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        colors=[BAR],
        return_format="dict",
    ):
        r0, c0, r1, c1 = comp["bbox"]
        if not ((r0 == 0 or r1 == H - 1 or c0 == 0 or c1 == W - 1) and ((r0 == r1) ^ (c0 == c1))):
            continue
        edge = _edge_name(g.shape, int(r0 if c0 == c1 else r0), int(c0 if r0 == r1 else c0))
        bars.append(
            {
                "coords": comp["coords"],
                "bbox": comp["bbox"],
                "edge": edge,
            }
        )

    sprites: list[dict] = []
    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict"):
        coords = comp["coords"]
        if any(int(r) in (0, H - 1) or int(c) in (0, W - 1) for r, c in coords):
            continue
        colors = sorted(set(int(g[int(r), int(c)]) for r, c in coords))
        if len(colors) != 2:
            continue
        marker_color, marker_pos = _marker_info(g, coords)
        sprites.append(
            {
                "coords": coords,
                "bbox": comp["bbox"],
                "patch": crop_bbox(g, comp["bbox"]),
                "marker_color": marker_color,
                "marker_pos": marker_pos,
            }
        )

    assert len(anchors) == 2
    assert len(bars) == 2
    assert len(sprites) == 2
    return bg, anchors, bars, sprites


def _signed_step(edge: str, patch_shape: tuple[int, int]) -> tuple[int, int]:
    h, w = map(int, patch_shape)
    if edge == "left":
        return -h, 0
    if edge == "right":
        return h, 0
    if edge == "top":
        return 0, -w
    if edge == "bottom":
        return 0, w
    raise AssertionError(edge)


def _tile_origins(
    start_r: int,
    start_c: int,
    patch_shape: tuple[int, int],
    edge: str,
    grid_shape: tuple[int, int],
) -> list[tuple[int, int]]:
    H, W = map(int, grid_shape)
    h, w = map(int, patch_shape)
    dr, dc = _signed_step(edge, patch_shape)
    origins = [(int(start_r), int(start_c))]
    r = int(start_r + dr)
    c = int(start_c + dc)
    while not (r + h <= 0 or r >= H or c + w <= 0 or c >= W):
        origins.append((int(r), int(c)))
        r += dr
        c += dc
    return origins


def _match_bar(
    current_bbox: tuple[int, int, int, int],
    slide_axis: str,
    bars: list[dict],
) -> dict:
    r0, c0, r1, c1 = current_bbox
    if slide_axis == "horizontal":
        matches = [
            bar
            for bar in bars
            if bar["edge"] in {"left", "right"} and bar["bbox"][0] == r0 and bar["bbox"][2] == r1
        ]
    else:
        matches = [
            bar
            for bar in bars
            if bar["edge"] in {"top", "bottom"} and bar["bbox"][1] == c0 and bar["bbox"][3] == c1
        ]
    assert len(matches) == 1
    return matches[0]


def _plans(grid: np.ndarray, *, slide: bool) -> tuple[int, list[dict], list[dict], list[dict]]:
    bg, anchors, bars, sprites = _parse_scene(grid)
    plans: list[dict] = []
    for sprite in sprites:
        anchor = anchors[sprite["marker_color"]]
        marker_r, marker_c = sprite["marker_pos"]
        dr = 0
        dc = 0
        if anchor["edge"] in {"top", "bottom"}:
            dc = int(anchor["pos"][1] - marker_c) if slide else 0
            slide_axis = "horizontal"
        else:
            dr = int(anchor["pos"][0] - marker_r) if slide else 0
            slide_axis = "vertical"

        r0, c0, r1, c1 = sprite["bbox"]
        shifted_bbox = (int(r0 + dr), int(c0 + dc), int(r1 + dr), int(c1 + dc))
        bar = _match_bar(shifted_bbox, slide_axis, bars)
        origins = _tile_origins(shifted_bbox[0], shifted_bbox[1], sprite["patch"].shape, bar["edge"], grid.shape)
        plans.append(
            {
                "sprite": sprite,
                "anchor": anchor,
                "bar": bar,
                "slide_axis": slide_axis,
                "dr": int(dr),
                "dc": int(dc),
                "bbox": shifted_bbox,
                "origins": origins,
            }
        )
    return bg, bars, sprites, plans


def _render(grid: np.ndarray, *, slide: bool, tile: bool) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg, bars, sprites, plans = _plans(g, slide=slide)
    out = g.copy()

    for bar in bars:
        coords = bar["coords"]
        out[coords[:, 0], coords[:, 1]] = bg
    for sprite in sprites:
        coords = sprite["coords"]
        out[coords[:, 0], coords[:, 1]] = bg

    for plan in plans:
        patch = plan["sprite"]["patch"]
        origins = plan["origins"] if tile else plan["origins"][:1]
        for r0, c0 in origins:
            paste(out, patch, int(r0), int(c0), transparent=bg, overwrite=False, clip=True)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each marked multi-color sprite to its matching border anchor, then tile it orthogonally from that aligned position.

    Input structure:
    - Black is the background.
    - The interior contains exactly two 4-connected multi-color sprites; each sprite has one main color plus one unique single-pixel marker color.
    - The border contains two isolated anchor pixels whose colors match the two marker colors, plus two grey one-cell-thick control bars on other border sides.

    Rule:
    - Match every sprite to the anchor with the same marker color.
    - A top or bottom anchor makes its sprite slide horizontally until the marker column matches the anchor column; a left or right anchor makes its sprite slide vertically until the marker row matches the anchor row.
    - After sliding, match the sprite to the unique perpendicular grey control bar whose border span exactly matches the sprite span on that bar's axis.
    - Starting from the slid position, repeat the sprite by whole-bbox periods along the control-bar axis. Left and top bars use the negative direction of that axis; right and bottom bars use the positive direction. Sprite background cells stay transparent and existing foreground is never overwritten.
    - Erase the grey control bars in the output and keep the anchor pixels.

    Key insight:
    - The anchor decides the slide axis and landing alignment, while the control bar decides the orthogonal signed tiling axis from that landing point.
    """
    return _render(np.asarray(grid, dtype=int), slide=True, tile=True)


def _random_shape() -> np.ndarray:
    for _ in range(100):
        h = random.choice([2, 3])
        w = random.choice([2, 3])
        area = random.randint(SPRITE_AREA_RANGE[0], min(SPRITE_AREA_RANGE[1], h * w))
        start = (random.randrange(h), random.randrange(w))
        cells = {start}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    rr = int(r + dr)
                    cc = int(c + dc)
                    if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in cells:
                        frontier.append((rr, cc))
            if not frontier:
                break
            cells.add(random.choice(frontier))
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        r0, c0 = min(rows), min(cols)
        norm = {(r - r0, c - c0) for r, c in cells}
        hh = max(r for r, _ in norm) + 1
        ww = max(c for _, c in norm) + 1
        if hh == 1 or ww == 1 or len(norm) < SPRITE_AREA_RANGE[0]:
            continue
        patch = np.zeros((hh, ww), dtype=int)
        marker = random.choice(sorted(norm))
        for r, c in norm:
            patch[r, c] = 1
        patch[marker] = 2
        return patch
    raise RuntimeError("failed to build a sprite shape")


def _colorize_shape(mask_patch: np.ndarray, main_color: int, marker_color: int) -> np.ndarray:
    patch = np.zeros_like(mask_patch, dtype=int)
    patch[mask_patch == 1] = int(main_color)
    patch[mask_patch == 2] = int(marker_color)
    return patch


def _marker_rel(patch: np.ndarray) -> tuple[int, int]:
    counts = Counter(int(v) for v in patch.flatten() if int(v) != BG)
    marker_colors = [color for color, count in counts.items() if count == 1]
    assert len(marker_colors) == 1
    marker = np.argwhere(patch == int(marker_colors[0]))
    assert marker.shape[0] == 1
    return int(marker[0, 0]), int(marker[0, 1])


def _sprite_mask_from_plan(plan: dict, shape: tuple[int, int]) -> set[tuple[int, int]]:
    patch = plan["sprite"]["patch"]
    mask: set[tuple[int, int]] = set()
    for r0, c0 in plan["origins"]:
        for rr, cc in np.argwhere(patch != 0):
            ar = int(r0 + rr)
            ac = int(c0 + cc)
            if 0 <= ar < shape[0] and 0 <= ac < shape[1]:
                mask.add((ar, ac))
    return mask


def _mask_from_origins(
    patch: np.ndarray,
    origins: list[tuple[int, int]],
    shape: tuple[int, int],
) -> set[tuple[int, int]]:
    mask: set[tuple[int, int]] = set()
    for r0, c0 in origins:
        for rr, cc in np.argwhere(patch != BG):
            ar = int(r0 + rr)
            ac = int(c0 + cc)
            if 0 <= ar < shape[0] and 0 <= ac < shape[1]:
                mask.add((ar, ac))
    return mask


def _validate_example(grid: np.ndarray) -> None:
    g = np.asarray(grid, dtype=int)
    bg, anchors, bars, sprites = _parse_scene(g)
    assert bg == BG
    assert sorted(anchor["edge"] for anchor in anchors.values()) == ["left", "top"]
    assert {bar["edge"] for bar in bars} & {"left", "right"}
    assert {bar["edge"] for bar in bars} & {"top", "bottom"}
    assert {sprite["marker_color"] for sprite in sprites} == set(anchors)

    out = main(g)
    slide_only = _render(g, slide=True, tile=False)
    tile_without_slide = _render(g, slide=False, tile=True)
    assert not np.array_equal(out, slide_only)
    assert not np.array_equal(out, tile_without_slide)

    _bg2, _bars2, _sprites2, plans = _plans(g, slide=True)
    masks = []
    for plan in plans:
        assert (plan["dr"], plan["dc"]) != (0, 0)
        assert len(plan["origins"]) >= 2
        mask = _sprite_mask_from_plan(plan, g.shape)
        assert mask
        masks.append(mask)

    assert masks[0].isdisjoint(masks[1])
    for anchor in anchors.values():
        r, c = anchor["pos"]
        assert int(out[r, c]) == int(anchor["color"])
        assert all((r, c) not in mask for mask in masks)
    for bar in bars:
        for r, c in bar["coords"]:
            assert int(out[int(r), int(c)]) == BG


def _generate_candidate() -> np.ndarray | None:
    H = random.randint(*GRID_SIDE_RANGE)
    W = random.randint(*GRID_SIDE_RANGE)
    colors = random.sample(PALETTE, 6)
    main_a, marker_a, main_b, marker_b = colors[:4]
    side_a = "left" if random.random() < LEFT_SIDE_WEIGHT else "right"
    side_b = "top" if random.random() < TOP_SIDE_WEIGHT else "bottom"

    patch_a = _colorize_shape(_random_shape(), main_a, marker_a)
    patch_b = _colorize_shape(_random_shape(), main_b, marker_b)
    h_a, w_a = patch_a.shape
    h_b, w_b = patch_b.shape
    mr_a, mc_a = _marker_rel(patch_a)
    mr_b, mc_b = _marker_rel(patch_b)

    if side_a == "left":
        row_a_range = range(h_a + 1, H - h_a - 1)
    else:
        row_a_range = range(1, H - (2 * h_a) + 1)
    if side_b == "top":
        col_b_range = range(w_b + 1, W - w_b - 1)
    else:
        col_b_range = range(1, W - (2 * w_b) + 1)
    if not row_a_range or not col_b_range:
        return None

    col_a_range = range(2, W - w_a - 1)
    row_b_range = range(2, H - h_b - 1)
    if not col_a_range or not row_b_range:
        return None

    for _ in range(120):
        final_a = (random.choice(tuple(row_a_range)), random.choice(tuple(col_a_range)))
        final_b = (random.choice(tuple(row_b_range)), random.choice(tuple(col_b_range)))
        bbox_a = (final_a[0], final_a[1], final_a[0] + h_a - 1, final_a[1] + w_a - 1)
        bbox_b = (final_b[0], final_b[1], final_b[0] + h_b - 1, final_b[1] + w_b - 1)
        if _bbox_overlap(_expand_bbox(bbox_a, 1), _expand_bbox(bbox_b, 1)):
            continue

        top_anchor = (0, final_a[1] + mc_a)
        left_anchor = (final_b[0] + mr_b, 0)
        if top_anchor[1] in (0, W - 1) or left_anchor[0] in (0, H - 1):
            continue

        bar_a_bbox = (bbox_a[0], 0 if side_a == "left" else W - 1, bbox_a[2], 0 if side_a == "left" else W - 1)
        bar_b_bbox = (0 if side_b == "top" else H - 1, bbox_b[1], 0 if side_b == "top" else H - 1, bbox_b[3])

        if side_b == "top" and bbox_b[1] - 1 <= top_anchor[1] <= bbox_b[3] + 1:
            continue
        if side_a == "left" and bbox_a[0] - 1 <= left_anchor[0] <= bbox_a[2] + 1:
            continue
        if top_anchor == left_anchor:
            continue

        origins_a = _tile_origins(final_a[0], final_a[1], patch_a.shape, side_a, (H, W))
        origins_b = _tile_origins(final_b[0], final_b[1], patch_b.shape, side_b, (H, W))
        if len(origins_a) < 2 or len(origins_b) < 2:
            continue

        mask_a = _mask_from_origins(patch_a, origins_a, (H, W))
        mask_b = _mask_from_origins(patch_b, origins_b, (H, W))
        if not mask_a.isdisjoint(mask_b):
            continue
        if top_anchor in mask_a or top_anchor in mask_b or left_anchor in mask_a or left_anchor in mask_b:
            continue

        input_a_cols = [c for c in range(1, W - w_a) if c != final_a[1]]
        input_b_rows = [r for r in range(1, H - h_b) if r != final_b[0]]
        random.shuffle(input_a_cols)
        random.shuffle(input_b_rows)

        input_a_candidates = []
        for init_c_a in input_a_cols:
            init_a = (final_a[0], int(init_c_a))
            init_bbox_a = (init_a[0], init_a[1], init_a[0] + h_a - 1, init_a[1] + w_a - 1)
            if _bbox_overlap(_expand_bbox(init_bbox_a, 1), _expand_bbox(bbox_b, 1)):
                continue
            input_a_candidates.append((init_a, init_bbox_a))
        input_b_candidates = []
        for init_r_b in input_b_rows:
            init_b = (int(init_r_b), final_b[1])
            init_bbox_b = (init_b[0], init_b[1], init_b[0] + h_b - 1, init_b[1] + w_b - 1)
            if _bbox_overlap(_expand_bbox(init_bbox_b, 1), _expand_bbox(bbox_a, 1)):
                continue
            input_b_candidates.append((init_b, init_bbox_b))

        for init_a, init_bbox_a in input_a_candidates[:20]:
            for init_b, init_bbox_b in input_b_candidates[:20]:
                if _bbox_overlap(_expand_bbox(init_bbox_a, 1), _expand_bbox(init_bbox_b, 1)):
                    continue

                grid = np.zeros((H, W), dtype=int)
                paste(grid, patch_a, init_a[0], init_a[1], transparent=BG)
                paste(grid, patch_b, init_b[0], init_b[1], transparent=BG)
                grid[top_anchor] = marker_a
                grid[left_anchor] = marker_b
                grid[bar_a_bbox[0] : bar_a_bbox[2] + 1, bar_a_bbox[1]] = BAR
                grid[bar_b_bbox[0], bar_b_bbox[1] : bar_b_bbox[3] + 1] = BAR

                try:
                    _validate_example(grid)
                except AssertionError:
                    continue
                return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _generate_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
