from __future__ import annotations

import random
from collections import defaultdict, deque
from itertools import permutations, product

import numpy as np


BG = 0
PALETTE = list(range(1, 10))
GEN_RETRIES = 400


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _box_component_info(coords: np.ndarray) -> dict[str, object] | None:
    r0, c0, r1, c1 = _bbox_coords(coords)
    hh, ww = r1 - r0 + 1, c1 - c0 + 1
    if hh < 4 or ww < 4:
        return None
    expected = {
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if r in (r0, r1) or c in (c0, c1)
    }
    actual = {tuple(map(int, rc)) for rc in coords}
    if actual != expected:
        return None
    return {"bbox": (r0, c0, r1, c1), "size": (hh, ww)}


def _mask_from_bboxes(shape: tuple[int, int], bboxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r0, c0, r1, c1 in bboxes:
        mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


def _infer_box_group(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    colors = sorted(int(v) for v in np.unique(g) if int(v) != BG)
    groups: dict[tuple[int, int], list[dict[str, object]]] = defaultdict(list)
    for color in colors:
        for comp in _components_mask(g == color):
            info = _box_component_info(comp)
            if info is None:
                continue
            groups[info["size"]].append({"color": color, "bbox": info["bbox"]})

    candidates: list[tuple[int, dict[str, object]]] = []
    for size, boxes in groups.items():
        if len(boxes) < 2:
            continue
        if len({int(box["color"]) for box in boxes}) != len(boxes):
            continue

        bboxes = [box["bbox"] for box in boxes]
        box_mask = _mask_from_bboxes(g.shape, bboxes)
        line_mask = (g != BG) & (~box_mask)
        if not line_mask.any():
            continue

        panel_bbox = _bbox_coords(np.argwhere(line_mask))
        panel_r0, panel_c0, panel_r1, panel_c1 = panel_bbox
        line_colors = {int(v) for v in np.unique(g[line_mask]) if int(v) != BG}
        box_colors = {int(box["color"]) for box in boxes}
        if line_colors != box_colors:
            continue

        lines: dict[int, tuple[str, int]] = {}
        ok = True
        for box in boxes:
            color = int(box["color"])
            coords = np.argwhere((g == color) & (~box_mask))
            if len(coords) < 2:
                ok = False
                break
            r0, c0, r1, c1 = _bbox_coords(coords)
            is_h = c0 == panel_c0 and c1 == panel_c1 and r0 == r1
            is_v = r0 == panel_r0 and r1 == panel_r1 and c0 == c1
            if is_h == is_v:
                ok = False
                break
            key = ("h", r0) if is_h else ("v", c0)
            if color in lines or key in lines.values():
                ok = False
                break
            lines[color] = key

        if not ok:
            continue
        if {kind for kind, _ in lines.values()} != {"h", "v"}:
            continue
        candidates.append(
            (
                len(boxes),
                {
                    "boxes": boxes,
                    "panel_bbox": panel_bbox,
                    "box_mask": box_mask,
                    "lines": lines,
                    "size": size,
                },
            )
        )

    assert candidates, "expected a valid box group"
    best_count = max(count for count, _ in candidates)
    best = [entry for count, entry in candidates if count == best_count]
    assert len(best) == 1, "box group must be uniquely identifiable"
    return best[0]


def _layer_edges(grid: np.ndarray, lines: dict[int, tuple[str, int]]) -> set[tuple[int, int]]:
    h_lines = {color: pos for color, (kind, pos) in lines.items() if kind == "h"}
    v_lines = {color: pos for color, (kind, pos) in lines.items() if kind == "v"}
    edges: set[tuple[int, int]] = set()
    for h_color, row in h_lines.items():
        for v_color, col in v_lines.items():
            visible = int(grid[row, col])
            assert visible in {h_color, v_color}, "line crossings must show one of the two line colors"
            if visible == h_color:
                edges.add((v_color, h_color))
            else:
                edges.add((h_color, v_color))
    return edges


def _unique_topological_order(colors: list[int], edges: set[tuple[int, int]]) -> list[int]:
    higher: dict[int, set[int]] = {color: set() for color in colors}
    indeg: dict[int, int] = {color: 0 for color in colors}
    for lower, upper in edges:
        if upper in higher[lower]:
            continue
        higher[lower].add(upper)
        indeg[upper] += 1

    remaining = set(colors)
    order: list[int] = []
    while remaining:
        zeros = sorted(color for color in remaining if indeg[color] == 0)
        assert len(zeros) == 1, "crossings must determine a unique total order"
        cur = zeros[0]
        remaining.remove(cur)
        order.append(cur)
        for nxt in higher[cur]:
            indeg[nxt] -= 1
    return order


def _extract_sprites(grid: np.ndarray, boxes: list[dict[str, object]]) -> dict[int, np.ndarray]:
    sprites: dict[int, np.ndarray] = {}
    for box in boxes:
        color = int(box["color"])
        r0, c0, r1, c1 = box["bbox"]
        sprites[color] = np.asarray(grid[r0 + 1 : r1, c0 + 1 : c1], dtype=int).copy()
    return sprites


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the unique line stacking order, then overlay the matching boxed sprites from bottom to top.

    Input structure:
    - One side of the grid contains several one-cell-thick colored lines; each line
      spans the full width or full height of that local line region.
    - Another region contains same-sized one-cell-thick colored boxes; each box
      border color matches exactly one line color, and its interior contains a sprite.
    - At line crossings, the visible cell shows the upper line's color.

    Rule:
    - Detect the equal-sized box borders and use their colors to identify the line colors.
    - Reconstruct each line from its full-span bbox and convert every horizontal/vertical
      crossing into a lower/upper layer constraint from the visible crossing color.
    - Solve the uniquely determined bottom-to-top order, then create a blank canvas the
      size of a box interior and overlay each matching interior sprite in that order,
      treating background cells as transparent.

    Key insight:
    - The boxes do not encode order by position; color pairing links them to the line
      region, and only the line occlusion chain reveals the correct sprite overlay order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    structure = _infer_box_group(g)
    boxes = structure["boxes"]
    lines = structure["lines"]
    colors = sorted(lines)
    edges = _layer_edges(g, lines)
    order = _unique_topological_order(colors, edges)

    sprites = _extract_sprites(g, boxes)
    out_h, out_w = next(iter(sprites.values())).shape
    out = np.zeros((out_h, out_w), dtype=int)
    for color in order:
        sprite = sprites[color]
        mask = sprite != BG
        out[mask] = sprite[mask]
    return out


def _valid_orders(colors: list[int], edges: set[tuple[int, int]]) -> list[tuple[int, ...]]:
    orders: list[tuple[int, ...]] = []
    for order in permutations(colors):
        pos = {color: idx for idx, color in enumerate(order)}
        if all(pos[lower] < pos[upper] for lower, upper in edges):
            orders.append(order)
    return orders


def _overlay(sprites: list[np.ndarray]) -> np.ndarray:
    out = np.zeros_like(sprites[0])
    for sprite in sprites:
        mask = sprite != BG
        out[mask] = sprite[mask]
    return out


def _different_color(palette: list[int], avoid: int | None = None) -> int:
    choices = [color for color in palette if color != avoid]
    return int(random.choice(choices))


def _paint_shape(sprite: np.ndarray, color: int, start: tuple[int, int], mode: str, forbidden: set[tuple[int, int]]) -> None:
    h, w = sprite.shape
    r, c = start
    cells: list[tuple[int, int]] = []
    if mode == "point":
        cells = [(r, c)]
    elif mode == "hseg":
        span = random.randint(2, min(4, w))
        c0 = max(0, min(c, w - span))
        cells = [(r, cc) for cc in range(c0, c0 + span)]
    elif mode == "vseg":
        span = random.randint(2, min(4, h))
        r0 = max(0, min(r, h - span))
        cells = [(rr, c) for rr in range(r0, r0 + span)]
    else:
        hh = random.randint(1, min(3, h))
        ww = random.randint(1, min(3, w))
        r0 = max(0, min(r, h - hh))
        c0 = max(0, min(c, w - ww))
        cells = [(rr, cc) for rr in range(r0, r0 + hh) for cc in range(c0, c0 + ww)]

    for rr, cc in cells:
        if (rr, cc) not in forbidden:
            sprite[rr, cc] = color


def _generate_sprites(count: int, h: int, w: int) -> list[np.ndarray]:
    cell_list = list(product(range(h), range(w)))
    area = h * w
    palette = random.sample(PALETTE, random.randint(3, min(6, len(PALETTE))))

    for _ in range(300):
        random.shuffle(cell_list)
        witness_cells = cell_list[:count]
        conflict_cells = cell_list[count : count + count - 1]
        sprites = [np.zeros((h, w), dtype=int) for _ in range(count)]

        for i, (r, c) in enumerate(witness_cells):
            sprites[i][r, c] = _different_color(palette)
        for i, (r, c) in enumerate(conflict_cells):
            sprites[i][r, c] = _different_color(palette)
            sprites[i + 1][r, c] = _different_color(palette, int(sprites[i][r, c]))

        for i, sprite in enumerate(sprites):
            forbidden = set(witness_cells[:i])
            target = random.randint(max(4, area // 4), max(5, (2 * area) // 3))
            tries = 0
            while int(np.count_nonzero(sprite)) < target and tries < area * 6:
                tries += 1
                r, c = random.choice(cell_list)
                if (r, c) in forbidden:
                    continue
                mode = random.choice(("point", "hseg", "vseg", "rect"))
                _paint_shape(sprite, _different_color(palette), (r, c), mode, forbidden)

        comp = _overlay(sprites)
        if np.count_nonzero(comp) < max(count + 1, area // 3):
            continue
        if any(np.array_equal(comp, sprite) for sprite in sprites):
            continue
        if len({sprite.tobytes() for sprite in sprites}) != count:
            continue

        ok = True
        for i, sprite in enumerate(sprites):
            later = sprites[i + 1 :]
            later_mask = np.zeros_like(sprite, dtype=bool)
            for nxt in later:
                later_mask |= nxt != BG
            witness = (sprite != BG) & (~later_mask) & (comp == sprite)
            if not witness.any():
                ok = False
                break
            if i < count - 1 and not ((sprite != BG) & later_mask).any():
                ok = False
                break
        if not ok:
            continue

        for i in range(count - 1):
            swapped = sprites[:]
            swapped[i], swapped[i + 1] = swapped[i + 1], swapped[i]
            if np.array_equal(_overlay(swapped), comp):
                ok = False
                break
        if ok:
            return sprites
    raise RuntimeError("failed to generate non-degenerate sprites")


def _spaced_positions(length: int, count: int) -> list[int]:
    assert length >= 2 * count - 1
    for _ in range(200):
        vals = sorted(random.sample(range(length), count))
        if all(b - a >= 2 for a, b in zip(vals, vals[1:])):
            return vals
    raise RuntimeError("failed to place spaced positions")


def _arrange_boxes(
    count: int,
    outer_h: int,
    outer_w: int,
    panel_h: int,
    panel_w: int,
) -> dict[str, object]:
    gap = random.randint(1, 2)
    sep = random.randint(2, 4)
    side = random.choice(("right", "bottom"))
    options: list[dict[str, object]] = []

    if side == "right":
        for cols in range(1, min(2, count) + 1):
            rows = (count + cols - 1) // cols
            box_h = rows * outer_h + (rows - 1) * gap
            box_w = cols * outer_w + (cols - 1) * gap
            content_h = max(panel_h, box_h)
            content_w = panel_w + sep + box_w
            if content_h <= 26 and content_w <= 26:
                options.append(
                    {
                        "side": side,
                        "rows": rows,
                        "cols": cols,
                        "gap": gap,
                        "sep": sep,
                        "content_h": content_h,
                        "content_w": content_w,
                        "box_h": box_h,
                        "box_w": box_w,
                    }
                )
    else:
        for rows in range(1, min(2, count) + 1):
            cols = (count + rows - 1) // rows
            box_h = rows * outer_h + (rows - 1) * gap
            box_w = cols * outer_w + (cols - 1) * gap
            content_h = panel_h + sep + box_h
            content_w = max(panel_w, box_w)
            if content_h <= 26 and content_w <= 26:
                options.append(
                    {
                        "side": side,
                        "rows": rows,
                        "cols": cols,
                        "gap": gap,
                        "sep": sep,
                        "content_h": content_h,
                        "content_w": content_w,
                        "box_h": box_h,
                        "box_w": box_w,
                    }
                )

    if not options:
        raise RuntimeError("no valid box arrangement")

    layout = random.choice(options)
    content_h = int(layout["content_h"])
    content_w = int(layout["content_w"])
    top = random.randint(1, 30 - content_h - 1)
    left = random.randint(1, 30 - content_w - 1)

    if layout["side"] == "right":
        line_r = top + random.randint(0, content_h - panel_h)
        line_c = left
        box_r = top + random.randint(0, content_h - int(layout["box_h"]))
        box_c = left + panel_w + int(layout["sep"])
    else:
        line_r = top
        line_c = left + random.randint(0, content_w - panel_w)
        box_r = top + panel_h + int(layout["sep"])
        box_c = left + random.randint(0, content_w - int(layout["box_w"]))

    layout["line_origin"] = (line_r, line_c)
    layout["box_origin"] = (box_r, box_c)
    layout["shape"] = (content_h + top + random.randint(0, 30 - top - content_h), content_w + left + random.randint(0, 30 - left - content_w))
    return layout


def _render_attempt() -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    count = random.randint(3, 5)
    start_kind = random.choice(("h", "v"))
    kinds = [start_kind if i % 2 == 0 else ("v" if start_kind == "h" else "h") for i in range(count)]
    h_count = sum(kind == "h" for kind in kinds)
    v_count = count - h_count

    panel_h = random.randint(max(6, 2 * h_count - 1, h_count + 2), min(11, max(6, 2 * h_count - 1, h_count + 2) + 3))
    panel_w = random.randint(max(6, 2 * v_count - 1, v_count + 2), min(11, max(6, 2 * v_count - 1, v_count + 2) + 3))
    rows = _spaced_positions(panel_h, h_count)
    cols = _spaced_positions(panel_w, v_count)

    inner_h = random.randint(3, 6)
    inner_w = random.randint(3, 6)
    outer_h = inner_h + 2
    outer_w = inner_w + 2
    layout = _arrange_boxes(count, outer_h, outer_w, panel_h, panel_w)

    order_colors = random.sample(PALETTE, count)
    h_rows = iter(rows)
    v_cols = iter(cols)
    lines: dict[int, tuple[str, int]] = {}
    for color, kind in zip(order_colors, kinds):
        lines[color] = (kind, next(h_rows) if kind == "h" else next(v_cols))

    sprites = _generate_sprites(count, inner_h, inner_w)
    expected = _overlay(sprites)

    h, w = layout["shape"]
    grid = np.zeros((h, w), dtype=int)
    line_r0, line_c0 = layout["line_origin"]
    for color in order_colors:
        kind, pos = lines[color]
        if kind == "h":
            grid[line_r0 + pos, line_c0 : line_c0 + panel_w] = color
        else:
            grid[line_r0 : line_r0 + panel_h, line_c0 + pos] = color

    box_positions: list[tuple[int, int]] = []
    box_r0, box_c0 = layout["box_origin"]
    rows_n = int(layout["rows"])
    cols_n = int(layout["cols"])
    gap = int(layout["gap"])
    for idx in range(count):
        rr = idx // cols_n
        cc = idx % cols_n
        if rr >= rows_n:
            break
        box_positions.append((box_r0 + rr * (outer_h + gap), box_c0 + cc * (outer_w + gap)))
    assert len(box_positions) >= count
    box_positions = box_positions[:count]
    random.shuffle(box_positions)

    for color, sprite, (r0, c0) in zip(order_colors, sprites, box_positions):
        r1 = r0 + outer_h - 1
        c1 = c0 + outer_w - 1
        grid[r0 : r1 + 1, c0] = color
        grid[r0 : r1 + 1, c1] = color
        grid[r0, c0 : c1 + 1] = color
        grid[r1, c0 : c1 + 1] = color
        mask = sprite != BG
        patch = grid[r0 + 1 : r1, c0 + 1 : c1]
        patch[mask] = sprite[mask]

    meta = {
        "order": order_colors,
        "sprites_by_color": {color: sprite for color, sprite in zip(order_colors, sprites)},
        "expected": expected,
        "lines": {color: (kind, line_r0 + pos if kind == "h" else line_c0 + pos) for color, (kind, pos) in lines.items()},
        "panel_shape": (panel_h, panel_w),
    }
    return grid, expected, meta


def _semantic_ok(grid: np.ndarray, expected: np.ndarray, meta: dict[str, object]) -> bool:
    try:
        structure = _infer_box_group(grid)
    except AssertionError:
        return False

    lines = structure["lines"]
    colors = sorted(lines)
    edges = _layer_edges(grid, lines)
    orders = _valid_orders(colors, edges)
    if len(orders) != 1:
        return False

    solved = main(grid)
    if not np.array_equal(solved, expected):
        return False

    expected_order = tuple(int(c) for c in meta["order"])
    if orders[0] != expected_order:
        return False

    box_mask = structure["box_mask"]
    for color in expected_order:
        visible_count = int(np.count_nonzero((grid == color) & (~box_mask)))
        if visible_count < 2:
            return False

    sprites = [meta["sprites_by_color"][color] for color in expected_order]
    comp = _overlay(sprites)
    if not np.array_equal(comp, expected):
        return False

    for i, sprite in enumerate(sprites):
        later_mask = np.zeros_like(sprite, dtype=bool)
        for nxt in sprites[i + 1 :]:
            later_mask |= nxt != BG
        if not ((sprite != BG) & (~later_mask)).any():
            return False
        if i < len(sprites) - 1 and not ((sprite != BG) & later_mask).any():
            return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        try:
            grid, expected, meta = _render_attempt()
        except RuntimeError:
            continue
        if max(grid.shape) > 30 or max(expected.shape) > 30:
            continue
        if _semantic_ok(grid, expected, meta):
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
