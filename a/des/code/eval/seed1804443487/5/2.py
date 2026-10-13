from __future__ import annotations

import random
from collections import defaultdict, deque

import numpy as np


BG = 0
SEPARATOR = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDE_TO_ROT = {"up": 0, "right": 3, "down": 2, "left": 1}
SIDE_OFFSETS = {"up": (-1, 0), "right": (0, 1), "down": (1, 0), "left": (0, -1)}


def _bbox(cells: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _region_spans(total: int, separators: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for line in separators:
        spans.append((start, line))
        start = line + 1
    spans.append((start, total))
    return spans


def _equal_partition_size(total: int, separators: list[int]) -> int | None:
    sizes: list[int] = []
    prev = -1
    for line in separators:
        size = line - prev - 1
        if size <= 0:
            return None
        sizes.append(size)
        prev = line
    tail = total - prev - 1
    if tail <= 0:
        return None
    sizes.append(tail)
    if len(set(sizes)) != 1:
        return None
    return sizes[0]


def _detect_partition(grid: np.ndarray) -> tuple[int, list[int], list[int], list[tuple[int, int]], list[tuple[int, int]]] | None:
    h, w = grid.shape
    best = None
    for color in map(int, np.unique(grid)):
        if color == BG:
            continue
        rows = [r for r in range(h) if np.all(grid[r, :] == color)]
        cols = [c for c in range(w) if np.all(grid[:, c] == color)]
        if len(rows) != len(cols) or len(rows) not in (1, 2):
            continue
        sub_h = _equal_partition_size(h, rows)
        sub_w = _equal_partition_size(w, cols)
        if sub_h is None or sub_w is None:
            continue
        score = (len(rows) + 1, sub_h * sub_w)
        if best is None or score > best[0]:
            best = (score, color, rows, cols)
    if best is None:
        return None
    _, color, rows, cols = best
    return color, rows, cols, _region_spans(grid.shape[0], rows), _region_spans(grid.shape[1], cols)


def _components_in_region(
    grid: np.ndarray, row_span: tuple[int, int], col_span: tuple[int, int], separator_color: int
) -> list[dict[str, object]]:
    rs, re = row_span
    cs, ce = col_span
    region = grid[rs:re, cs:ce]
    h, w = region.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []
    for rr in range(h):
        for cc in range(w):
            color = int(region[rr, cc])
            if color in (BG, separator_color) or seen[rr, cc]:
                continue
            seen[rr, cc] = True
            q = deque([(rr, cc)])
            cells: list[tuple[int, int]] = []
            while q:
                r, c = q.popleft()
                cells.append((rs + r, cs + c))
                for dr, dc in DIRS4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(region[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append({"color": color, "cells": cells})
    return comps


def _template_from_cells(color: int, cells: list[tuple[int, int]]) -> dict[str, object]:
    r0, c0, r1, c1 = _bbox(cells)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in cells:
        mask[r - r0, c - c0] = True
    return {"color": color, "cells": set(cells), "bbox": (r0, c0, r1, c1), "mask": mask}


def _singleton_pairs(region_infos: list[dict[str, object]]) -> dict[int, list[tuple[int, int]]]:
    by_color: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for info in region_infos:
        for r, c, color in info["markers"]:
            by_color[color].append((r, c))
    return by_color


def _adjacent_to_template(pos: tuple[int, int], template_cells: set[tuple[int, int]]) -> bool:
    r, c = pos
    return any((r + dr, c + dc) in template_cells for dr, dc in DIRS4)


def _rotation_from_marker(marker: tuple[int, int], bbox: tuple[int, int, int, int]) -> int | None:
    r, c = marker
    r0, c0, r1, c1 = bbox
    if r < r0:
        return 0
    if c > c1:
        return 3
    if r > r1:
        return 2
    if c < c0:
        return 1
    return None


def _paint_mask(out: np.ndarray, mask: np.ndarray, color: int, top: int, left: int) -> None:
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if mask[r, c]:
                out[top + r, left + c] = color


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    partition = _detect_partition(g)
    if partition is None:
        return g.copy()

    separator_color, _, _, row_spans, col_spans = partition
    region_infos: list[dict[str, object]] = []
    region_of = np.full(g.shape, -1, dtype=int)
    region_id = 0

    for row_span in row_spans:
        for col_span in col_spans:
            rs, re = row_span
            cs, ce = col_span
            region_of[rs:re, cs:ce] = region_id
            comps = _components_in_region(g, row_span, col_span, separator_color)
            template = None
            markers: list[tuple[int, int, int]] = []
            for comp in comps:
                cells = comp["cells"]
                color = int(comp["color"])
                if len(cells) > 1:
                    if template is None or len(cells) > len(template["cells"]):
                        template = _template_from_cells(color, cells)
                elif len(cells) == 1:
                    r, c = cells[0]
                    markers.append((r, c, color))
            region_infos.append(
                {
                    "id": region_id,
                    "row_span": row_span,
                    "col_span": col_span,
                    "template": template,
                    "markers": markers,
                }
            )
            region_id += 1

    marker_pairs = _singleton_pairs(region_infos)
    out = g.copy()
    for positions in marker_pairs.values():
        for r, c in positions:
            out[r, c] = BG

    for color, positions in marker_pairs.items():
        if len(positions) != 2:
            continue
        source_pos = None
        target_pos = None
        source_template = None
        target_region_info = None

        for pos in positions:
            rid = int(region_of[pos[0], pos[1]])
            if rid < 0:
                continue
            info = region_infos[rid]
            template = info["template"]
            if template is not None and _adjacent_to_template(pos, template["cells"]):
                source_pos = pos
                source_template = template
            else:
                target_pos = pos
                target_region_info = info

        if source_pos is None or target_pos is None or source_template is None or target_region_info is None:
            continue

        rot_k = _rotation_from_marker(source_pos, source_template["bbox"])
        if rot_k is None:
            continue
        rotated = np.rot90(source_template["mask"], rot_k)
        ph, pw = rotated.shape
        top = target_pos[0] - ph // 2
        left = target_pos[1] - pw // 2
        rs, re = target_region_info["row_span"]
        cs, ce = target_region_info["col_span"]
        if top < rs or left < cs or top + ph > re or left + pw > ce:
            continue
        _paint_mask(out, rotated, int(source_template["color"]), top, left)

    return out


def _path_to(center: tuple[int, int], target: tuple[int, int]) -> list[tuple[int, int]]:
    rng = random
    r, c = center
    tr, tc = target
    path = [(r, c)]
    while (r, c) != (tr, tc):
        options: list[tuple[int, int]] = []
        if r < tr:
            options.append((r + 1, c))
        elif r > tr:
            options.append((r - 1, c))
        if c < tc:
            options.append((r, c + 1))
        elif c > tc:
            options.append((r, c - 1))
        r, c = rng.choice(options)
        path.append((r, c))
    return path


def _rotation_signature(mask: np.ndarray) -> bool:
    return not any(np.array_equal(mask, np.rot90(mask, k)) for k in (1, 2, 3))


def _grow_template_mask() -> np.ndarray:
    rng = random
    shapes = [(3, 3), (3, 5), (5, 3), (5, 5)]
    for _ in range(400):
        h, w = rng.choice(shapes)
        center = (h // 2, w // 2)
        anchors = {
            "up": (0, rng.randrange(w)),
            "down": (h - 1, rng.randrange(w)),
            "left": (rng.randrange(h), 0),
            "right": (rng.randrange(h), w - 1),
        }
        cells = {center}
        for anchor in anchors.values():
            cells.update(_path_to(center, anchor))

        max_area = h * w - 1
        if len(cells) > max_area:
            continue
        target_area = rng.randint(len(cells), max_area)
        while len(cells) < target_area:
            frontier: list[tuple[int, int]] = []
            for r, c in cells:
                for dr, dc in DIRS4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(rng.choice(frontier))

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        if not mask[center]:
            continue
        if not mask[0, :].any() or not mask[-1, :].any() or not mask[:, 0].any() or not mask[:, -1].any():
            continue
        if mask.sum() < 5:
            continue
        if not _rotation_signature(mask):
            continue
        return mask
    raise RuntimeError("failed to generate a non-degenerate template")


def _side_contacts(mask: np.ndarray, side: str) -> list[tuple[int, int]]:
    h, w = mask.shape
    if side == "up":
        return [(0, c) for c in range(w) if mask[0, c]]
    if side == "down":
        return [(h - 1, c) for c in range(w) if mask[h - 1, c]]
    if side == "left":
        return [(r, 0) for r in range(h) if mask[r, 0]]
    return [(r, w - 1) for r in range(h) if mask[r, w - 1]]


def _choose_source_setup(sub_h: int, sub_w: int, marker_count: int) -> tuple[np.ndarray, int, int, list[str]]:
    rng = random
    for _ in range(300):
        mask = _grow_template_mask()
        ph, pw = mask.shape
        if sub_h - ph < 3 or sub_w - pw < 3:
            continue
        top = rng.randint(1, sub_h - ph - 1)
        left = rng.randint(1, sub_w - pw - 1)
        dirs = rng.sample(["up", "right", "down", "left"], marker_count)
        ok = True
        for side in dirs:
            dr, dc = SIDE_OFFSETS[side]
            choices = _side_contacts(mask, side)
            if not choices:
                ok = False
                break
            rr, cc = rng.choice(choices)
            mr = top + rr + dr
            mc = left + cc + dc
            if not (0 <= mr < sub_h and 0 <= mc < sub_w):
                ok = False
                break
        if ok:
            return mask, top, left, dirs
    raise RuntimeError("failed to place source template")


def _choose_target_center(sub_h: int, sub_w: int, rotated_mask: np.ndarray) -> tuple[int, int]:
    rng = random
    ph, pw = rotated_mask.shape
    top = rng.randint(1, sub_h - ph - 1)
    left = rng.randint(1, sub_w - pw - 1)
    return top + ph // 2, left + pw // 2


def _partition_lines(n: int, size: int) -> list[int]:
    return [size * (idx + 1) + idx for idx in range(n - 1)]


def _pair_distribution(region_count: int, source_count: int) -> list[int]:
    rng = random
    max_pairs = min(5, region_count - source_count, source_count * 3)
    total_pairs = rng.randint(source_count, max_pairs)
    counts = [1] * source_count
    while sum(counts) < total_pairs:
        choices = [idx for idx, count in enumerate(counts) if count < 3]
        counts[rng.choice(choices)] += 1
    return counts


def _render_expected(inp: np.ndarray, placements: list[dict[str, object]]) -> np.ndarray:
    out = inp.copy()
    for item in placements:
        sr, sc = item["source_marker"]
        tr, tc = item["target_marker"]
        out[sr, sc] = BG
        out[tr, tc] = BG
    for item in placements:
        rotated = np.rot90(item["template_mask"], item["rot_k"])
        ph, pw = rotated.shape
        top = item["target_marker"][0] - ph // 2
        left = item["target_marker"][1] - pw // 2
        _paint_mask(out, rotated, int(item["template_color"]), top, left)
    return out


def _gen_attempt() -> np.ndarray | None:
    rng = random
    n = rng.choice([2, 3])
    sub_h = rng.randint(9, 13)
    sub_w = rng.randint(9, 13)
    h = n * sub_h + (n - 1)
    w = n * sub_w + (n - 1)
    grid = np.full((h, w), BG, dtype=int)

    row_lines = _partition_lines(n, sub_h)
    col_lines = _partition_lines(n, sub_w)
    grid[row_lines, :] = SEPARATOR
    grid[:, col_lines] = SEPARATOR

    region_spans_r = _region_spans(h, row_lines)
    region_spans_c = _region_spans(w, col_lines)
    region_boxes: list[tuple[int, int, int, int]] = []
    for rs, re in region_spans_r:
        for cs, ce in region_spans_c:
            region_boxes.append((rs, re, cs, ce))

    source_count = 1 if n == 2 else rng.randint(1, 3)
    pair_counts = _pair_distribution(len(region_boxes), source_count)
    total_pairs = sum(pair_counts)

    region_ids = list(range(len(region_boxes)))
    rng.shuffle(region_ids)
    source_regions = region_ids[:source_count]
    target_regions = region_ids[source_count : source_count + total_pairs]

    if len(target_regions) != total_pairs:
        return None

    needed_colors = source_count + total_pairs
    if needed_colors > len(PALETTE):
        return None
    color_pool = list(PALETTE)
    rng.shuffle(color_pool)
    template_colors = color_pool[:source_count]
    marker_colors = color_pool[source_count : source_count + total_pairs]

    placements: list[dict[str, object]] = []
    target_iter = iter(target_regions)
    marker_iter = iter(marker_colors)

    for source_idx, region_id in enumerate(source_regions):
        rs, re, cs, ce = region_boxes[region_id]
        mask, top, left, dirs = _choose_source_setup(sub_h, sub_w, pair_counts[source_idx])
        template_color = template_colors[source_idx]
        abs_top = rs + top
        abs_left = cs + left
        ph, pw = mask.shape
        for rr in range(ph):
            for cc in range(pw):
                if mask[rr, cc]:
                    grid[abs_top + rr, abs_left + cc] = template_color

        for side in dirs:
            marker_color = next(marker_iter)
            side_r, side_c = rng.choice(_side_contacts(mask, side))
            dr, dc = SIDE_OFFSETS[side]
            source_marker = (abs_top + side_r + dr, abs_left + side_c + dc)
            grid[source_marker] = marker_color

            target_region = next(target_iter)
            trs, tre, tcs, tce = region_boxes[target_region]
            rot_k = SIDE_TO_ROT[side]
            rotated = np.rot90(mask, rot_k)
            center_r, center_c = _choose_target_center(sub_h, sub_w, rotated)
            target_marker = (trs + center_r, tcs + center_c)
            grid[target_marker] = marker_color

            placements.append(
                {
                    "template_color": template_color,
                    "template_mask": mask,
                    "rot_k": rot_k,
                    "source_marker": source_marker,
                    "target_marker": target_marker,
                }
            )

    expected = _render_expected(grid, placements)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("failed to generate a valid puzzle input")
