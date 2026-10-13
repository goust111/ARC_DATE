import random

import numpy as np


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(coords)

    return components


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _bbox_contains(bbox: tuple[int, int, int, int], cell: tuple[int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    r, c = cell
    return r0 <= r <= r1 and c0 <= c <= c1


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _color_mapping(grid: np.ndarray, bg: int) -> dict[int, int]:
    h, w = grid.shape
    mapping: dict[int, int] = {}
    top = grid[h - 2]
    bottom = grid[h - 1]
    for c in range(w):
        a = int(top[c])
        b = int(bottom[c])
        if a == bg or b == bg:
            continue
        mapping[a] = b
        mapping[b] = a
    return mapping


def _portal_pairs(work: np.ndarray, bg: int) -> dict[int, list[tuple[int, int]]]:
    pairs: dict[int, list[tuple[int, int]]] = {}
    for color in sorted(set(map(int, np.unique(work))) - {int(bg)}):
        mask = work == int(color)
        coords = np.argwhere(mask)
        if coords.shape[0] != 2:
            continue
        comps = _connected_components(mask)
        if len(comps) == 2 and all(len(comp) == 1 for comp in comps):
            pairs[int(color)] = [tuple(comp[0]) for comp in comps]
    return pairs


def main(grid: np.ndarray) -> np.ndarray:
    """Teleport each portal-anchored object to its paired portal and recolor the moved copy via the bottom mapping table.

    Input structure:
    - The background is the most common color.
    - Above the last two rows there are several same-color portal pairs, each made of exactly two singleton pixels of one color.
    - Small 4-connected multicolor objects appear in the upper area; each object's bounding box contains exactly one portal pixel, while some paired portal endpoints may be empty.
    - The bottom two rows form a width-wide color mapping table, where each non-background column gives a reversible color pair.

    Rule:
    - Identify each portal pair from the two singleton occurrences of the same color.
    - For every foreground object, find the unique portal pixel contained in its bounding box and treat that pixel as the teleport anchor.
    - Erase the object from its original location, paste it at the paired portal while preserving every pixel's offset from the anchor, then replace each moved pixel color using the mapping table when that color appears in a valid mapping column.
    - Keep all portal pixels and the bottom mapping table unchanged in the output.

    Key insight:
    - The puzzle combines three independent cues: matching-color singleton pixels define portal pairs, object placement is anchored by the portal inside each bounding box, and the bottom two-row legend gives the post-teleport recoloring rule.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert h >= 3
    bg = _most_common_color(g)

    work = g[: h - 2]
    mapping = _color_mapping(g, bg)
    portals = _portal_pairs(work, bg)
    portal_at = {
        (int(r), int(c)): int(color)
        for color, cells in portals.items()
        for r, c in cells
    }

    out = g.copy()
    components = _connected_components(work != bg)
    objects: list[dict[str, object]] = []
    for coords in components:
        if len(coords) <= 1:
            continue
        bbox = _bbox(coords)
        anchors = [cell for cell in portal_at if _bbox_contains(bbox, cell)]
        assert len(anchors) == 1
        anchor = anchors[0]
        objects.append(
            {
                "coords": coords,
                "bbox": bbox,
                "anchor": anchor,
                "portal_color": portal_at[anchor],
            }
        )

    for obj in objects:
        for r, c in obj["coords"]:
            out[r, c] = bg

    painted: set[tuple[int, int]] = set()
    objects.sort(key=lambda obj: (obj["anchor"][0], obj["anchor"][1]))
    for obj in objects:
        src_r, src_c = obj["anchor"]
        portal_color = int(obj["portal_color"])
        pair = portals[portal_color]
        if pair[0] == obj["anchor"]:
            dst_r, dst_c = pair[1]
        else:
            dst_r, dst_c = pair[0]

        for r, c in obj["coords"]:
            nr = int(dst_r + (r - src_r))
            nc = int(dst_c + (c - src_c))
            assert 0 <= nr < h - 2 and 0 <= nc < w
            if (nr, nc) not in portal_at:
                assert out[nr, nc] == bg
                assert (nr, nc) not in painted
                painted.add((nr, nc))
            out[nr, nc] = int(mapping.get(int(g[r, c]), int(g[r, c])))

    for color, cells in portals.items():
        for r, c in cells:
            out[r, c] = int(color)

    return out


MAX_GENERATION_ATTEMPTS = 400
OUTER_MARGIN = 1
SECTOR_GAP = 2
SECTOR_SIZE_RANGE = (6, 8)


def _random_connected_cells(height: int, width: int, area: int) -> list[tuple[int, int]]:
    start = (random.randrange(height), random.randrange(width))
    cells = {start}
    while len(cells) < area:
        frontier = set()
        for r, c in cells:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        if not frontier:
            return []
        cells.add(random.choice(tuple(frontier)))
    return list(cells)


def _random_shape_offsets() -> list[tuple[int, int]]:
    area = random.randint(4, 8)
    dims = [
        (hh, ww)
        for hh in range(1, 5)
        for ww in range(1, 5)
        if hh * ww >= area and max(hh, ww) >= 2
    ]
    height, width = random.choice(dims)
    cells = _random_connected_cells(height, width, area)
    if not cells:
        return []
    min_r = min(r for r, _c in cells)
    min_c = min(c for _r, c in cells)
    cropped = [(r - min_r, c - min_c) for r, c in cells]
    anchor_r, anchor_c = random.choice(cropped)
    return [(r - anchor_r, c - anchor_c) for r, c in cropped]


def _footprint(
    anchor: tuple[int, int],
    offsets: list[tuple[int, int]],
) -> tuple[set[tuple[int, int]], tuple[int, int, int, int]]:
    ar, ac = anchor
    cells = {(ar + dr, ac + dc) for dr, dc in offsets}
    bbox = _bbox(list(cells))
    return cells, bbox


def _layout_options(total_pairs: int) -> list[tuple[int, int]]:
    if total_pairs == 2:
        return [(1, 2), (2, 1)]
    if total_pairs == 3:
        return [(1, 3), (2, 2)]
    return [(2, 2)]


def _sector_boxes(rows: int, cols: int) -> list[tuple[int, int, int, int]]:
    row_heights = [random.randint(*SECTOR_SIZE_RANGE) for _ in range(rows)]
    col_widths = [random.randint(*SECTOR_SIZE_RANGE) for _ in range(cols)]

    boxes: list[tuple[int, int, int, int]] = []
    r = OUTER_MARGIN
    for rh in row_heights:
        c = OUTER_MARGIN
        for cw in col_widths:
            boxes.append((r, c, r + rh - 1, c + cw - 1))
            c += cw + SECTOR_GAP
        r += rh + SECTOR_GAP
    return boxes


def _work_shape_from_boxes(boxes: list[tuple[int, int, int, int]]) -> tuple[int, int]:
    max_r = max(r1 for _r0, _c0, r1, _c1 in boxes)
    max_c = max(c1 for _r0, _c0, _r1, c1 in boxes)
    return max_r + OUTER_MARGIN + 1, max_c + OUTER_MARGIN + 1


def _sample_source_and_target(
    box: tuple[int, int, int, int],
    offsets: list[tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int], set[tuple[int, int]], set[tuple[int, int]]] | None:
    r0, c0, r1, c1 = box
    min_dr = min(dr for dr, _dc in offsets)
    max_dr = max(dr for dr, _dc in offsets)
    min_dc = min(dc for _dr, dc in offsets)
    max_dc = max(dc for _dr, dc in offsets)

    anchors = [
        (r, c)
        for r in range(r0 - min_dr, r1 - max_dr + 1)
        for c in range(c0 - min_dc, c1 - max_dc + 1)
    ]
    if len(anchors) < 2:
        return None

    for _ in range(200):
        src = random.choice(anchors)
        dst = random.choice(anchors)
        if src == dst:
            continue
        if abs(src[0] - dst[0]) + abs(src[1] - dst[1]) == 1:
            continue

        src_cells, src_bbox = _footprint(src, offsets)
        dst_cells, dst_bbox = _footprint(dst, offsets)
        if src_cells & dst_cells:
            continue
        if _boxes_overlap(src_bbox, dst_bbox):
            continue
        if _bbox_contains(src_bbox, dst) or _bbox_contains(dst_bbox, src):
            continue
        return src, dst, src_cells, dst_cells

    return None


def _sample_empty_pair(box: tuple[int, int, int, int]) -> tuple[tuple[int, int], tuple[int, int]] | None:
    r0, c0, r1, c1 = box
    cells = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
    for _ in range(200):
        a = random.choice(cells)
        b = random.choice(cells)
        if a == b:
            continue
        if abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1:
            continue
        return a, b
    return None


def _build_expected(
    inp: np.ndarray,
    sources: list[dict[str, object]],
    portals: dict[int, list[tuple[int, int]]],
    bg: int,
    mapping: dict[int, int],
) -> np.ndarray:
    out = inp.copy()
    work_h = inp.shape[0] - 2
    for spec in sources:
        for r, c in spec["source_cells"]:
            out[r, c] = bg

    for spec in sources:
        src_r, src_c = spec["source_anchor"]
        dst_r, dst_c = spec["target_anchor"]
        fill_color = int(spec["fill_color"])
        for dr, dc in spec["offsets"]:
            nr = dst_r + dr
            nc = dst_c + dc
            if (nr, nc) == (dst_r, dst_c):
                out[nr, nc] = int(spec["portal_color"])
            else:
                out[nr, nc] = int(mapping.get(fill_color, fill_color))
            assert 0 <= nr < work_h and 0 <= nc < inp.shape[1]

    for color, cells in portals.items():
        for r, c in cells:
            out[r, c] = int(color)
    return out


def _gen_attempt() -> np.ndarray | None:
    n_sources = random.choice([2, 3])
    extra_empty = random.choice([0, 1])
    total_pairs = n_sources + extra_empty

    rows, cols = random.choice(_layout_options(total_pairs))
    boxes = _sector_boxes(rows, cols)
    used_boxes = random.sample(boxes, total_pairs)
    work_h, work_w = _work_shape_from_boxes(boxes)
    if work_h + 2 > 30 or work_w > 30:
        return None

    mapped_count = random.randint(1, n_sources - 1)
    palette = list(range(10))
    bg = random.choice(palette)
    palette.remove(bg)
    random.shuffle(palette)

    need = total_pairs + n_sources + mapped_count
    if len(palette) < need:
        return None

    portal_colors = palette[:total_pairs]
    fill_colors = palette[total_pairs : total_pairs + n_sources]
    mapped_colors = palette[total_pairs + n_sources : need]
    leftovers = palette[need:]

    mapped_indices = sorted(random.sample(range(n_sources), mapped_count))
    mapped_lookup = {idx: mapped_colors[k] for k, idx in enumerate(mapped_indices)}

    decoy_pairs: list[tuple[int, int]] = []
    if len(leftovers) >= 2 and random.random() < 0.5:
        decoy_pairs.append((leftovers[0], leftovers[1]))

    grid = np.full((work_h + 2, work_w), bg, dtype=int)
    portals: dict[int, list[tuple[int, int]]] = {}
    sources: list[dict[str, object]] = []

    for source_idx in range(n_sources):
        box = used_boxes[source_idx]
        offsets = _random_shape_offsets()
        if not offsets:
            return None
        placement = _sample_source_and_target(box, offsets)
        if placement is None:
            return None
        src_anchor, dst_anchor, src_cells, dst_cells = placement
        portal_color = portal_colors[source_idx]
        fill_color = fill_colors[source_idx]

        portals[portal_color] = [src_anchor, dst_anchor]
        for dr, dc in offsets:
            rr = src_anchor[0] + dr
            cc = src_anchor[1] + dc
            grid[rr, cc] = portal_color if (rr, cc) == src_anchor else fill_color
        grid[dst_anchor[0], dst_anchor[1]] = portal_color

        sources.append(
            {
                "offsets": offsets,
                "source_anchor": src_anchor,
                "target_anchor": dst_anchor,
                "source_cells": src_cells,
                "target_cells": dst_cells,
                "portal_color": portal_color,
                "fill_color": fill_color,
            }
        )

    for empty_idx in range(extra_empty):
        portal_color = portal_colors[n_sources + empty_idx]
        pair = _sample_empty_pair(used_boxes[n_sources + empty_idx])
        if pair is None:
            return None
        portals[portal_color] = [pair[0], pair[1]]
        for r, c in pair:
            grid[r, c] = portal_color

    mapping_pairs = [
        (fill_colors[idx], mapped_lookup[idx])
        for idx in mapped_indices
    ] + decoy_pairs
    valid_columns = random.sample(range(work_w), len(mapping_pairs))
    for col, (a, b) in zip(valid_columns, mapping_pairs):
        if random.random() < 0.5:
            grid[work_h, col] = a
            grid[work_h + 1, col] = b
        else:
            grid[work_h, col] = b
            grid[work_h + 1, col] = a

    if _most_common_color(grid) != bg:
        return None

    work = grid[:work_h]
    if any(int(np.sum(work == color)) != 2 for color in portals):
        return None

    mapping = _color_mapping(grid, bg)
    expected = _build_expected(grid, sources, portals, bg, mapping)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid portal-teleport ARC input")
