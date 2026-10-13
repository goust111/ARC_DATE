from __future__ import annotations

import itertools
import random
from collections import Counter, deque

import numpy as np


BG = 0
ALL_COLORS = tuple(range(1, 10))
CARD_COUNT = 4
CARD_SIZES = (3, 4)
BAND_WIDTHS = (2, 3)
MAX_GENERATION_ATTEMPTS = 400
CARD_CORNERS = ("tl", "tr", "bl", "br")


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _colors_present(grid: np.ndarray, bg: int) -> list[int]:
    return sorted(int(v) for v in np.unique(grid) if int(v) != bg)


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _bbox_h(box: tuple[int, int, int, int]) -> int:
    return int(box[2] - box[0] + 1)


def _bbox_w(box: tuple[int, int, int, int]) -> int:
    return int(box[3] - box[1] + 1)


def _intersection_box(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> tuple[int, int, int, int] | None:
    r0 = max(int(a[0]), int(b[0]))
    c0 = max(int(a[1]), int(b[1]))
    r1 = min(int(a[2]), int(b[2]))
    c1 = min(int(a[3]), int(b[3]))
    if r0 > r1 or c0 > c1:
        return None
    return r0, c0, r1, c1


def _components_of_color(grid: np.ndarray, color: int) -> list[dict]:
    mask = grid == int(color)
    visited = np.zeros_like(mask, dtype=bool)
    h, w = grid.shape
    comps: list[dict] = []
    for r_raw, c_raw in np.argwhere(mask):
        r = int(r_raw)
        c = int(c_raw)
        if visited[r, c]:
            continue
        visited[r, c] = True
        q = deque([(r, c)])
        coords: list[tuple[int, int]] = []
        while q:
            rr, cc = q.popleft()
            coords.append((rr, cc))
            for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not visited[nr, nc]:
                    visited[nr, nc] = True
                    q.append((nr, nc))
        comps.append(
            {
                "color": int(color),
                "coords": coords,
                "bbox": _bbox_from_coords(coords),
                "area": len(coords),
            }
        )
    return comps


def _is_band_component(comp: dict) -> bool:
    box = comp["bbox"]
    h = _bbox_h(box)
    w = _bbox_w(box)
    return h >= 9 and 2 <= w <= 3 and h >= w + 6


def _is_card_component(comp: dict) -> bool:
    box = comp["bbox"]
    h = _bbox_h(box)
    w = _bbox_w(box)
    return h == w and h in CARD_SIZES


def _stack_order_from_cards(grid: np.ndarray, card_by_color: dict[int, dict], bg: int) -> list[int]:
    colors = list(card_by_color)
    edges: dict[int, set[int]] = {color: set() for color in colors}
    indeg = {color: 0 for color in colors}
    undirected: dict[int, set[int]] = {color: set() for color in colors}
    overlap_count = 0

    for i, color_a in enumerate(colors):
        for color_b in colors[i + 1 :]:
            inter = _intersection_box(card_by_color[color_a]["bbox"], card_by_color[color_b]["bbox"])
            if inter is None:
                continue
            overlap_count += 1
            r0, c0, r1, c1 = inter
            visible = {
                int(v)
                for v in np.unique(grid[r0 : r1 + 1, c0 : c1 + 1])
                if int(v) in {color_a, color_b} and int(v) != bg
            }
            assert len(visible) == 1, "each overlapping card pair must expose exactly one upper color"
            top = next(iter(visible))
            bottom = color_b if top == color_a else color_a
            if bottom not in edges[top]:
                edges[top].add(bottom)
                indeg[bottom] += 1
            undirected[color_a].add(color_b)
            undirected[color_b].add(color_a)

    assert overlap_count == len(colors) - 1, "stack cards must form a single overlap chain"
    start = colors[0]
    seen = {start}
    q = deque([start])
    while q:
        color = q.popleft()
        for nxt in undirected[color]:
            if nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    assert len(seen) == len(colors), "card overlap chain must be connected"
    assert sum(len(v) for v in undirected.values()) == 2 * (len(colors) - 1)
    assert sum(1 for color in colors if len(undirected[color]) == 1) == 2
    assert all(1 <= len(undirected[color]) <= 2 for color in colors)

    ready = [color for color in colors if indeg[color] == 0]
    order: list[int] = []
    while ready:
        assert len(ready) == 1, "stack order must be uniquely determined"
        color = ready.pop()
        order.append(color)
        for nxt in sorted(edges[color]):
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                ready.append(nxt)
    assert len(order) == len(colors)
    return order


def _band_order_left_to_right(band_by_color: dict[int, dict]) -> list[int]:
    return [color for color, _ in sorted(band_by_color.items(), key=lambda item: item[1]["bbox"][1])]


def _analyze_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)
    colors = _colors_present(g, bg)
    assert len(colors) == CARD_COUNT, "expected one band and one card for each of four colors"

    band_by_color: dict[int, dict] = {}
    card_by_color: dict[int, dict] = {}
    for color in colors:
        comps = _components_of_color(g, color)
        assert len(comps) == 2, "each color must have exactly one band and one stack card"
        bands = [comp for comp in comps if _is_band_component(comp)]
        cards = [comp for comp in comps if _is_card_component(comp)]
        assert len(bands) == 1 and len(cards) == 1
        band_by_color[color] = bands[0]
        card_by_color[color] = cards[0]

        r0, c0, r1, c1 = bands[0]["bbox"]
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        assert np.all((patch == bg) | (patch == color))
        assert np.any(patch == bg), "every band needs at least one gap"

    band_colors = list(band_by_color)
    for i, color_a in enumerate(band_colors):
        for color_b in band_colors[i + 1 :]:
            assert _intersection_box(band_by_color[color_a]["bbox"], band_by_color[color_b]["bbox"]) is None

    card_sizes = {_bbox_h(card["bbox"]) for card in card_by_color.values()}
    assert len(card_sizes) == 1, "all stack cards must use the same size"

    stack_order = _stack_order_from_cards(g, card_by_color, bg)
    return {
        "bg": bg,
        "band_by_color": band_by_color,
        "card_by_color": card_by_color,
        "stack_order": stack_order,
        "band_order": _band_order_left_to_right(band_by_color),
    }


def _fill_from_cyclic_order(
    grid: np.ndarray, band_by_color: dict[int, dict], order: list[int], bg: int, *, use_prev: bool
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    n = len(order)
    for idx, color in enumerate(order):
        fill_color = order[idx - 1] if use_prev else order[(idx + 1) % n]
        r0, c0, r1, c1 = band_by_color[color]["bbox"]
        patch = out[r0 : r1 + 1, c0 : c1 + 1]
        patch[patch == bg] = fill_color
    return out


def _rotations(order: list[int]) -> set[tuple[int, ...]]:
    return {tuple(order[i:] + order[:i]) for i in range(len(order))}


def _choose_band_order(stack_order: list[int]) -> list[int]:
    banned = _rotations(stack_order) | _rotations(list(reversed(stack_order)))
    choices = [list(p) for p in itertools.permutations(stack_order) if tuple(p) not in banned]
    return random.choice(choices)


def _stack_anchor(h: int, w: int, size: int, corner: str) -> tuple[int, int, int, int]:
    if corner == "tl":
        return 1, 1, 1, 1
    if corner == "tr":
        return 1, w - size - 1, 1, -1
    if corner == "bl":
        return h - size - 1, 1, -1, 1
    if corner == "br":
        return h - size - 1, w - size - 1, -1, -1
    raise ValueError(f"unknown corner {corner}")


def _draw_card(grid: np.ndarray, r0: int, c0: int, size: int, color: int) -> None:
    grid[r0 : r0 + size, c0 : c0 + size] = int(color)


def _carve_band_gaps(grid: np.ndarray, r0: int, r1: int, c0: int, c1: int) -> None:
    width = int(c1 - c0 + 1)
    height = int(r1 - r0 + 1)
    if width == 2:
        spine = random.choice([0, 1])
        gap_cols = [1 - spine]
    else:
        spine = 1
        gap_cols = [0, 2]

    for local_col in gap_cols:
        max_count = max(2, height - 2)
        count = random.randint(2, max_count)
        rows = random.sample(range(height), count)
        for rr in rows:
            grid[r0 + rr, c0 + local_col] = BG

    if width == 3:
        mirror_count = random.randint(1, max(1, height // 4))
        rows = random.sample(range(height), mirror_count)
        for rr in rows:
            grid[r0 + rr, c0] = BG
            grid[r0 + rr, c1] = BG


def _generate_attempt() -> tuple[np.ndarray, dict]:
    colors = random.sample(ALL_COLORS, CARD_COUNT)
    stack_order = random.sample(colors, CARD_COUNT)
    band_order = _choose_band_order(stack_order)

    card_size = random.choice(CARD_SIZES)
    step = card_size - 1
    stack_extent = card_size + (CARD_COUNT - 1) * step

    band_widths = [random.choice(BAND_WIDTHS) for _ in range(CARD_COUNT)]
    band_total = sum(band_widths) + (CARD_COUNT - 1)

    width_base = stack_extent + band_total + 2
    width = width_base + random.randint(0, 30 - width_base)

    height_min = max(16, stack_extent + 3)
    height = random.randint(height_min, min(28, height_min + 8))
    grid = np.full((height, width), BG, dtype=int)

    top_margin = random.randint(1, 3)
    bottom_margin = random.randint(1, 3)
    while height - top_margin - bottom_margin < 9:
        top_margin = random.randint(1, 3)
        bottom_margin = random.randint(1, 3)
    band_r0 = top_margin
    band_r1 = height - bottom_margin - 1

    corner = random.choice(CARD_CORNERS)
    if corner in {"tl", "bl"}:
        band_start_min = stack_extent + 2
        band_start_max = width - band_total
    else:
        band_start_min = 0
        band_start_max = width - stack_extent - band_total - 2
    band_cursor = random.randint(band_start_min, band_start_max)

    band_boxes: dict[int, tuple[int, int, int, int]] = {}
    for color, band_width in zip(band_order, band_widths):
        c0 = band_cursor
        c1 = c0 + band_width - 1
        grid[band_r0 : band_r1 + 1, c0 : c1 + 1] = int(color)
        _carve_band_gaps(grid, band_r0, band_r1, c0, c1)
        band_boxes[int(color)] = (band_r0, c0, band_r1, c1)
        band_cursor = c1 + 2

    start_r, start_c, dr_sign, dc_sign = _stack_anchor(height, width, card_size, corner)
    for depth in range(CARD_COUNT - 1, -1, -1):
        color = int(stack_order[depth])
        r0 = start_r + dr_sign * depth * step
        c0 = start_c + dc_sign * depth * step
        _draw_card(grid, r0, c0, card_size, color)

    meta = {
        "stack_order": list(stack_order),
        "band_order": list(band_order),
        "band_boxes": band_boxes,
    }
    return grid, meta


def _validate_example(grid: np.ndarray, meta: dict) -> None:
    info = _analyze_grid(grid)
    assert info["stack_order"] == meta["stack_order"]
    assert info["band_order"] == meta["band_order"]
    for color, box in meta["band_boxes"].items():
        assert info["band_by_color"][color]["bbox"] == box

    true_out = _fill_from_cyclic_order(grid, info["band_by_color"], info["stack_order"], info["bg"], use_prev=True)
    assert np.array_equal(true_out, main(grid))
    assert not np.array_equal(true_out, grid)

    spatial_prev = _fill_from_cyclic_order(grid, info["band_by_color"], info["band_order"], info["bg"], use_prev=True)
    spatial_next = _fill_from_cyclic_order(grid, info["band_by_color"], info["band_order"], info["bg"], use_prev=False)
    assert not np.array_equal(spatial_prev, true_out), "left-neighbor shortcut must fail"
    assert not np.array_equal(spatial_next, true_out), "right-neighbor shortcut must fail"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each vertical band's gaps with the color of the band immediately above it in the corner stack order.

    Input structure
    - The dominant color is the background.
    - Four colors each appear twice: once as a tall vertical band whose bbox contains only that color or background, and once as a same-size solid card inside a partially overlapping corner stack.
    - The stack cards overlap only with their layer-adjacent neighbors, so visible overlap cells encode a unique top-to-bottom occlusion order.

    Rule
    - Recover the full stack order from top layer to bottom layer by reading which card color is visible in each card-card overlap.
    - For each vertical band, replace every background cell inside that band's bbox with the color of the band one step above it in that stack order, wrapping so the top band borrows from the bottom band.
    - Preserve every existing foreground pixel, every background cell outside the band bboxes, and the corner stack itself.

    Key insight
    - The relevant adjacency is not the bands' left-to-right layout: the corner stack uses occlusion to re-index the band colors into a cyclic layer order, and the gaps borrow from that layer neighbor.
    """
    g = np.asarray(grid, dtype=int)
    info = _analyze_grid(g)
    return _fill_from_cyclic_order(g, info["band_by_color"], info["stack_order"], info["bg"], use_prev=True)


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        try:
            grid, meta = _generate_attempt()
            _validate_example(grid, meta)
            return grid
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid input")
