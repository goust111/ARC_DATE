from __future__ import annotations

import random

import numpy as np


GREY = 5
AZURE = 8
PALETTE_CHOICES = (1, 2, 3, 4, 6, 7, 9)
DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))

GRID_H_RANGE = (22, 30)
GRID_W_RANGE = (22, 30)
LEGEND_LEN_RANGE = (2, 4)
INNER_H_RANGE = (1, 4)
INNER_W_RANGE = (1, 5)
EXTRA_OBJECT_PROB = 0.45
CLIPPED_SAMPLE_PROB = 0.35
PLACEMENT_ATTEMPTS = 150
GENERATION_ATTEMPTS = 400
RESERVED_PAD = 1


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
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
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _rect_boundary(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _merge_boxes(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> tuple[int, int, int, int]:
    return min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])


def _expand_box(
    box: tuple[int, int, int, int], pad: int, h: int, w: int
) -> tuple[int, int, int, int]:
    return (
        max(0, box[0] - pad),
        max(0, box[1] - pad),
        min(h - 1, box[2] + pad),
        min(w - 1, box[3] + pad),
    )


def _clip_box(box: tuple[int, int, int, int], h: int, w: int) -> tuple[int, int, int, int] | None:
    r0 = max(0, box[0])
    c0 = max(0, box[1])
    r1 = min(h - 1, box[2])
    c1 = min(w - 1, box[3])
    if r0 > r1 or c0 > c1:
        return None
    return r0, c0, r1, c1


def _projection_box(
    bbox: tuple[int, int, int, int], dr: int, dc: int
) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    if dr == 1:
        return r1 + 1, c0 + 1, r1 + inner_h, c1 - 1
    if dr == -1:
        return r0 - inner_h, c0 + 1, r0 - 1, c1 - 1
    if dc == 1:
        return r0 + 1, c1 + 1, r1 - 1, c1 + inner_w
    return r0 + 1, c0 - inner_w, r1 - 1, c0 - 1


def _find_legend(grid: np.ndarray, bg: int) -> tuple[list[int], np.ndarray]:
    h, w = grid.shape
    candidates: list[tuple[tuple[int, int, int], list[int], np.ndarray]] = []
    for comp in _components(grid == GREY):
        r0, c0, r1, c1 = _bbox(comp)
        if r1 - r0 + 1 < 4 or c1 - c0 + 1 < 3:
            continue
        if set(comp) != _rect_boundary(r0, c0, r1, c1):
            continue
        inner = grid[r0 + 1 : r1, c0 + 1 : c1]
        if inner.size == 0 or np.any(inner == GREY):
            continue

        dots: list[tuple[int, int, int]] = []
        ok = True
        for rr in range(inner.shape[0]):
            for cc in range(inner.shape[1]):
                value = int(inner[rr, cc])
                if value == bg:
                    continue
                dots.append((r0 + 1 + rr, c0 + 1 + cc, value))
        if not 2 <= len(dots) <= 4:
            continue
        if len({value for _, _, value in dots}) != len(dots):
            continue
        if len({row for row, _, _ in dots}) != len(dots):
            continue
        for rr in range(inner.shape[0]):
            for cc in range(inner.shape[1]):
                value = int(inner[rr, cc])
                if value != bg and (r0 + 1 + rr, c0 + 1 + cc, value) not in dots:
                    ok = False
        if not ok:
            continue

        colors = [value for _, _, value in sorted(dots, key=lambda item: (item[0], item[1]))]
        mask = np.zeros((h, w), dtype=bool)
        mask[r0 : r1 + 1, c0 : c1 + 1] = True
        score = (r0, c0, (r1 - r0 + 1) * (c1 - c0 + 1))
        candidates.append((score, colors, mask))

    assert candidates, "expected one grey legend frame"
    candidates.sort(key=lambda item: item[0])
    _, colors, mask = candidates[0]
    return colors, mask


def _gap_side(missing: tuple[int, int], bbox: tuple[int, int, int, int]) -> str:
    r0, c0, r1, c1 = bbox
    mr, mc = missing
    sides = []
    if mr == r0:
        sides.append("top")
    if mr == r1:
        sides.append("bottom")
    if mc == c0:
        sides.append("left")
    if mc == c1:
        sides.append("right")
    assert len(sides) == 1, "gap must be on exactly one side"
    return sides[0]


def _projection_direction(side: str) -> tuple[int, int]:
    if side == "top":
        return 1, 0
    if side == "bottom":
        return -1, 0
    if side == "left":
        return 0, 1
    return 0, -1


def _find_rings(
    grid: np.ndarray, bg: int, legend_mask: np.ndarray, legend_colors: list[int]
) -> list[dict[str, object]]:
    rings: list[dict[str, object]] = []
    for color in legend_colors:
        mask = (grid == color) & (~legend_mask)
        for comp in _components(mask):
            r0, c0, r1, c1 = _bbox(comp)
            if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 3:
                continue
            boundary = _rect_boundary(r0, c0, r1, c1)
            comp_set = set(comp)
            if not comp_set.issubset(boundary):
                continue
            missing = list(boundary - comp_set)
            if len(missing) != 1:
                continue
            if len(comp_set) != len(boundary) - 1:
                continue
            if not np.all(grid[r0 + 1 : r1, c0 + 1 : c1] == bg):
                continue
            side = _gap_side(missing[0], (r0, c0, r1, c1))
            rings.append(
                {
                    "color": int(color),
                    "bbox": (r0, c0, r1, c1),
                    "gap_side": side,
                    "proj": _projection_direction(side),
                }
            )
    assert rings, "expected at least one gapped rectangular ring"
    rings.sort(key=lambda item: (item["bbox"][0], item["bbox"][1], item["color"]))
    return rings


def main(grid: np.ndarray) -> np.ndarray:
    """
    For each rectangular ring, project its hollow interior away from the gap and recolor the
    projection with the next legend color in cyclic top-to-bottom order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    bg = _most_common_color(g)

    legend_colors, legend_mask = _find_legend(g, bg)
    mapping = {
        int(legend_colors[idx]): int(legend_colors[(idx + 1) % len(legend_colors)])
        for idx in range(len(legend_colors))
    }

    out = g.copy()
    for ring in _find_rings(g, bg, legend_mask, legend_colors):
        r0, c0, r1, c1 = ring["bbox"]
        dr, dc = ring["proj"]
        clipped = _clip_box(_projection_box((r0, c0, r1, c1), dr, dc), h, w)
        if clipped is None:
            continue
        rr0, cc0, rr1, cc1 = clipped
        out[rr0 : rr1 + 1, cc0 : cc1 + 1] = mapping[int(ring["color"])]
    return out


def _draw_legend(grid: np.ndarray, colors: list[int]) -> tuple[int, int, int, int]:
    legend_h = len(colors) + 2
    legend_w = 3
    grid[:legend_h, :legend_w] = AZURE
    grid[0, :legend_w] = GREY
    grid[legend_h - 1, :legend_w] = GREY
    grid[:legend_h, 0] = GREY
    grid[:legend_h, legend_w - 1] = GREY
    for idx, color in enumerate(colors, start=1):
        grid[idx, 1] = color
    return 0, 0, legend_h - 1, legend_w - 1


def _sample_origin(
    h: int, w: int, inner_h: int, inner_w: int, dr: int, dc: int, clipped: bool
) -> tuple[int, int] | None:
    box_h = inner_h + 2
    box_w = inner_w + 2
    if dr != 0:
        if clipped:
            if inner_h < 2:
                return None
            spill = random.randint(1, inner_h - 1)
            if dr == 1:
                r0 = h - (box_h + inner_h) + spill
            else:
                r0 = inner_h - spill
        else:
            low = inner_h if dr == -1 else 0
            high = h - box_h if dr == -1 else h - (box_h + inner_h)
            if low > high:
                return None
            r0 = random.randint(low, high)
        if 0 > r0 or r0 + box_h > h:
            return None
        if box_w > w:
            return None
        c0 = random.randint(0, w - box_w)
        return r0, c0

    if clipped:
        if inner_w < 2:
            return None
        spill = random.randint(1, inner_w - 1)
        if dc == 1:
            c0 = w - (box_w + inner_w) + spill
        else:
            c0 = inner_w - spill
    else:
        low = inner_w if dc == -1 else 0
        high = w - box_w if dc == -1 else w - (box_w + inner_w)
        if low > high:
            return None
        c0 = random.randint(low, high)
    if 0 > c0 or c0 + box_w > w:
        return None
    if box_h > h:
        return None
    r0 = random.randint(0, h - box_h)
    return r0, c0


def _draw_ring(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    color: int,
    gap_side: str,
    gap_offset: int,
) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    if gap_side == "top":
        grid[r0, c0 + gap_offset] = AZURE
    elif gap_side == "bottom":
        grid[r1, c0 + gap_offset] = AZURE
    elif gap_side == "left":
        grid[r0 + gap_offset, c0] = AZURE
    else:
        grid[r0 + gap_offset, c1] = AZURE


def _render_expected(
    grid: np.ndarray, legend_colors: list[int], specs: list[dict[str, object]]
) -> np.ndarray:
    out = grid.copy()
    h, w = grid.shape
    mapping = {
        int(legend_colors[idx]): int(legend_colors[(idx + 1) % len(legend_colors)])
        for idx in range(len(legend_colors))
    }
    for spec in specs:
        bbox = spec["bbox"]
        dr, dc = spec["proj"]
        clipped = _clip_box(_projection_box(bbox, dr, dc), h, w)
        if clipped is None:
            continue
        r0, c0, r1, c1 = clipped
        out[r0 : r1 + 1, c0 : c1 + 1] = mapping[int(spec["color"])]
    return out


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    legend_len = random.randint(*LEGEND_LEN_RANGE)
    legend_colors = random.sample(PALETTE_CHOICES, legend_len)

    object_count = legend_len
    if legend_len < 4 and random.random() < EXTRA_OBJECT_PROB:
        object_count += 1

    colors = list(legend_colors)
    while len(colors) < object_count:
        colors.append(random.choice(legend_colors))
    random.shuffle(colors)

    dirs = list(DIRECTIONS)
    random.shuffle(dirs)
    while len(dirs) < object_count:
        extra = list(DIRECTIONS)
        random.shuffle(extra)
        dirs.extend(extra)
    dirs = dirs[:object_count]
    random.shuffle(dirs)

    clipped_index = random.randrange(object_count) if random.random() < CLIPPED_SAMPLE_PROB else None

    grid = np.full((h, w), AZURE, dtype=int)
    legend_box = _draw_legend(grid, legend_colors)
    reserved = [_expand_box(legend_box, RESERVED_PAD, h, w)]
    specs: list[dict[str, object]] = []

    for idx in range(object_count):
        color = int(colors[idx])
        dr, dc = dirs[idx]
        force_clip = idx == clipped_index
        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            if dr != 0:
                inner_h = random.randint(max(2, INNER_H_RANGE[0]), INNER_H_RANGE[1]) if force_clip else random.randint(*INNER_H_RANGE)
                inner_w = random.randint(*INNER_W_RANGE)
            else:
                inner_h = random.randint(*INNER_H_RANGE)
                inner_w = random.randint(max(2, INNER_W_RANGE[0]), INNER_W_RANGE[1]) if force_clip else random.randint(*INNER_W_RANGE)
            origin = _sample_origin(h, w, inner_h, inner_w, dr, dc, force_clip)
            if origin is None:
                continue
            r0, c0 = origin
            bbox = (r0, c0, r0 + inner_h + 1, c0 + inner_w + 1)
            proj_box = _projection_box(bbox, dr, dc)
            visible_proj = _clip_box(proj_box, h, w)
            if visible_proj is None:
                continue
            reserve = _expand_box(_merge_boxes(bbox, visible_proj), RESERVED_PAD, h, w)
            if any(_boxes_overlap(reserve, other) for other in reserved):
                continue

            if dr == 1:
                gap_side = "top"
                gap_offset = random.randint(1, inner_w)
            elif dr == -1:
                gap_side = "bottom"
                gap_offset = random.randint(1, inner_w)
            elif dc == 1:
                gap_side = "left"
                gap_offset = random.randint(1, inner_h)
            else:
                gap_side = "right"
                gap_offset = random.randint(1, inner_h)

            specs.append(
                {
                    "color": color,
                    "bbox": bbox,
                    "proj": (dr, dc),
                    "gap_side": gap_side,
                    "gap_offset": gap_offset,
                }
            )
            reserved.append(reserve)
            placed = True
            break
        if not placed:
            return None

    for spec in specs:
        _draw_ring(
            grid,
            spec["bbox"],
            int(spec["color"]),
            str(spec["gap_side"]),
            int(spec["gap_offset"]),
        )

    expected = _render_expected(grid, legend_colors, specs)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
