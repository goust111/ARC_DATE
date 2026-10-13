from __future__ import annotations

from collections import Counter
import random

import numpy as np


GRAY = 5
MAROON = 9
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[counts.argmax()])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(comp)
    return out


def _indexed_components(mask: np.ndarray) -> tuple[list[list[tuple[int, int]]], dict[tuple[int, int], int]]:
    comps = _components(mask)
    index: dict[tuple[int, int], int] = {}
    for i, comp in enumerate(comps):
        for cell in comp:
            index[cell] = i
    return comps, index


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _solid_rectangle(grid: np.ndarray, color: int, box: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = box
    return bool(np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == int(color)))


def _copy_masks(grid: np.ndarray) -> tuple[list[tuple[int, int, int, int]], list[np.ndarray]]:
    copy_comps = _components(grid == GRAY)
    assert len(copy_comps) >= 3, "expected at least three gray source copies"

    bboxes = [_bbox(comp) for comp in copy_comps]
    dims = {(r1 - r0 + 1, c1 - c0 + 1) for r0, c0, r1, c1 in bboxes}
    assert len(dims) == 1, "all gray copies must share one bounding-box size"
    box_h, box_w = next(iter(dims))

    masks: list[np.ndarray] = []
    for comp, (r0, c0, _r1, _c1) in zip(copy_comps, bboxes, strict=False):
        mask = np.zeros((box_h, box_w), dtype=bool)
        for r, c in comp:
            mask[r - r0, c - c0] = True
        masks.append(mask)
    return bboxes, masks


def _legend_mapping(grid: np.ndarray, bg: int) -> dict[int, int]:
    h, w = grid.shape
    maroon_comps, maroon_index = _indexed_components(grid == MAROON)
    assert maroon_comps, "legend needs maroon dot groups"

    mapping: dict[int, int] = {}
    for color in map(int, np.unique(grid)):
        if color in {int(bg), GRAY, MAROON}:
            continue
        for comp in _components(grid == color):
            box = _bbox(comp)
            r0, c0, r1, c1 = box
            touches_edge = r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1
            if not touches_edge:
                continue
            assert _solid_rectangle(grid, color, box), "legend color blocks must be solid rectangles"

            dot_ids: set[int] = set()
            for r, c in comp:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) == MAROON:
                        dot_ids.add(maroon_index[(nr, nc)])

            assert len(dot_ids) == 1, "each legend block must touch exactly one maroon dot group"
            dot_count = len(maroon_comps[next(iter(dot_ids))])
            assert dot_count not in mapping, "legend dot counts must be unique"
            mapping[dot_count] = color

    assert mapping, "legend mapping must not be empty"
    return mapping


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild the shared gray prototype by majority vote, then recolor each copy by its missing-cell count.

    Input structure
    - Background is the most common color.
    - Several connected gray objects are incomplete copies of one prototype, all with the same bbox size.
    - Edge legend entries are solid color blocks, each touching a maroon dot group whose size is the lookup key.

    Rule
    - Align the gray copies by bbox and take a strict majority vote per cell to recover the prototype mask.
    - For each copy, count how many prototype cells are missing.
    - Lookup that count in the edge legend.
    - Repaint the full reconstructed prototype at that copy's location with the legend color.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    bboxes, masks = _copy_masks(g)

    stack = np.stack(masks, axis=0)
    prototype = stack.sum(axis=0) > (len(masks) // 2)
    assert bool(prototype.any()), "majority vote must recover a non-empty prototype"

    legend = _legend_mapping(g, bg)
    out = g.copy()

    for box, mask in zip(bboxes, masks, strict=False):
        assert not bool(np.any(mask & (~prototype))), "copies may only miss prototype cells"
        missing = int(np.sum(prototype & (~mask)))
        assert missing in legend, "missing-count lookup must exist in the legend"
        color = legend[missing]

        r0, c0, r1, c1 = box
        patch = out[r0 : r1 + 1, c0 : c1 + 1].copy()
        patch[prototype] = color
        patch[(patch == GRAY) & (~prototype)] = bg
        out[r0 : r1 + 1, c0 : c1 + 1] = patch

    return out


# --------------- generator ---------------

GRID_MIN = 18
GRID_MAX = 30
K_CHOICES = (3, 4, 5)
PROTOTYPE_DIM_RANGE = range(5, 9)
LEGEND_SIDES = ("top", "bottom", "left", "right")
LEGEND_DEPTH_CHOICES = (1, 2, 3)
PROTOTYPE_MODES = ("solid", "light_bites", "deep_bites")
GENERATION_ATTEMPTS = 400


def _is_connected(mask: np.ndarray) -> bool:
    comps = _components(mask)
    return len(comps) == 1


def _safe_missing_cells(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    cells: list[tuple[int, int]] = []
    for r, c in zip(*np.where(mask), strict=False):
        rr = int(r)
        cc = int(c)
        if all(
            0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc])
            for nr, nc in _neighbors4(rr, cc)
        ):
            cells.append((rr, cc))
    return cells


def _carve_edge_bites(mask: np.ndarray, bite_count: int) -> np.ndarray:
    out = mask.copy()
    h, w = out.shape
    for _ in range(bite_count):
        for _attempt in range(60):
            trial = out.copy()
            edge = random.choice(LEGEND_SIDES)
            if edge in {"top", "bottom"}:
                max_depth = min(2, h - 2)
                if max_depth < 1 or w < 3:
                    continue
                depth = random.randint(1, max_depth)
                span = random.randint(1, w - 2)
                c0 = random.randint(1, w - span - 1)
                if edge == "top":
                    trial[:depth, c0 : c0 + span] = False
                else:
                    trial[h - depth :, c0 : c0 + span] = False
            else:
                max_depth = min(2, w - 2)
                if max_depth < 1 or h < 3:
                    continue
                depth = random.randint(1, max_depth)
                span = random.randint(1, h - 2)
                r0 = random.randint(1, h - span - 1)
                if edge == "left":
                    trial[r0 : r0 + span, :depth] = False
                else:
                    trial[r0 : r0 + span, w - depth :] = False

            if _is_connected(trial):
                out = trial
                break
    return out


def _sample_prototype(k: int) -> np.ndarray | None:
    required_safe = sum(range(1, k + 1))
    candidates = [
        (h, w)
        for h in PROTOTYPE_DIM_RANGE
        for w in PROTOTYPE_DIM_RANGE
        if (h - 2) * (w - 2) >= required_safe
    ]
    random.shuffle(candidates)

    for h, w in candidates:
        mask = np.ones((h, w), dtype=bool)
        mode = random.choice(PROTOTYPE_MODES)
        if mode == "light_bites":
            mask = _carve_edge_bites(mask, random.randint(1, 2))
        elif mode == "deep_bites":
            mask = _carve_edge_bites(mask, random.randint(2, 4))

        safe = _safe_missing_cells(mask)
        if len(safe) >= required_safe:
            return mask
    return None


def _missing_sets(mask: np.ndarray, k: int) -> list[tuple[int, set[tuple[int, int]]]] | None:
    safe = _safe_missing_cells(mask)
    needed = sum(range(1, k + 1))
    if len(safe) < needed:
        return None

    counts = list(range(1, k + 1))
    for _ in range(400):
        chosen = random.sample(safe, needed)
        idx = 0
        pairs: list[tuple[int, set[tuple[int, int]]]] = []
        ok = True
        for count in counts:
            removed = set(chosen[idx : idx + count])
            idx += count
            trial = mask.copy()
            for r, c in removed:
                trial[r, c] = False
            if not _is_connected(trial):
                ok = False
                break
            pairs.append((count, removed))
        if ok:
            random.shuffle(pairs)
            return pairs
    return None


def _legend_bounds(side: str, depth: int, k: int) -> tuple[int, int]:
    span = k
    total_len = k * span + (k - 1)
    total_depth = depth + 1
    if side in {"top", "bottom"}:
        return total_depth, total_len
    return total_len, total_depth


def _sample_canvas(side: str, depth: int, k: int, box_h: int, box_w: int) -> tuple[int, int] | None:
    legend_h, legend_w = _legend_bounds(side, depth, k)
    min_h = max(GRID_MIN, legend_h + box_h + 4)
    min_w = max(GRID_MIN, legend_w + box_w + 4)
    if min_h > GRID_MAX or min_w > GRID_MAX:
        return None
    return random.randint(min_h, GRID_MAX), random.randint(min_w, GRID_MAX)


def _place_positions(
    grid_h: int,
    grid_w: int,
    side: str,
    depth: int,
    box_h: int,
    box_w: int,
    k: int,
) -> list[tuple[int, int]] | None:
    margin_top = 1 + (depth + 1 if side == "top" else 0)
    margin_bottom = 1 + (depth + 1 if side == "bottom" else 0)
    margin_left = 1 + (depth + 1 if side == "left" else 0)
    margin_right = 1 + (depth + 1 if side == "right" else 0)

    row_lo = margin_top
    row_hi = grid_h - margin_bottom - box_h
    col_lo = margin_left
    col_hi = grid_w - margin_right - box_w
    if row_hi < row_lo or col_hi < col_lo:
        return None

    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    positions: list[tuple[int, int]] = []
    for _ in range(k):
        for _attempt in range(200):
            r0 = random.randint(row_lo, row_hi)
            c0 = random.randint(col_lo, col_hi)
            r1 = r0 + box_h - 1
            c1 = c0 + box_w - 1
            rr0 = max(0, r0 - 1)
            rr1 = min(grid_h, r1 + 2)
            cc0 = max(0, c0 - 1)
            cc1 = min(grid_w, c1 + 2)
            if bool(occupied[rr0:rr1, cc0:cc1].any()):
                continue
            occupied[rr0:rr1, cc0:cc1] = True
            positions.append((r0, c0))
            break
        else:
            return None
    return positions


def _paint_legend(
    grid: np.ndarray,
    side: str,
    depth: int,
    mapping: dict[int, int],
) -> None:
    h, w = grid.shape
    k = len(mapping)
    span = k
    total_len = k * span + (k - 1)
    order = list(mapping)
    random.shuffle(order)

    if side in {"top", "bottom"}:
        start = random.randint(0, w - total_len)
    else:
        start = random.randint(0, h - total_len)

    for i, count in enumerate(order):
        color = mapping[count]
        offset = i * (span + 1)
        pad = (span - count) // 2

        if side == "top":
            c0 = start + offset
            grid[:depth, c0 : c0 + span] = color
            grid[depth, c0 + pad : c0 + pad + count] = MAROON
        elif side == "bottom":
            c0 = start + offset
            grid[h - depth :, c0 : c0 + span] = color
            grid[h - depth - 1, c0 + pad : c0 + pad + count] = MAROON
        elif side == "left":
            r0 = start + offset
            grid[r0 : r0 + span, :depth] = color
            grid[r0 + pad : r0 + pad + count, depth] = MAROON
        else:
            r0 = start + offset
            grid[r0 : r0 + span, w - depth :] = color
            grid[r0 + pad : r0 + pad + count, w - depth - 1] = MAROON


def _expected_output(
    grid: np.ndarray,
    prototype: np.ndarray,
    placements: list[tuple[int, int, int]],
) -> np.ndarray:
    out = grid.copy()
    box_h, box_w = prototype.shape
    for r0, c0, color in placements:
        patch = out[r0 : r0 + box_h, c0 : c0 + box_w].copy()
        patch[prototype] = color
        out[r0 : r0 + box_h, c0 : c0 + box_w] = patch
    return out


def _background_is_dominant(grid: np.ndarray, bg: int) -> bool:
    counts = Counter(map(int, grid.flat))
    bg_count = counts[int(bg)]
    return all(bg_count > count for color, count in counts.items() if color != int(bg))


def _generate_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    k = random.choice(K_CHOICES)
    prototype = _sample_prototype(k)
    if prototype is None:
        return None

    missing = _missing_sets(prototype, k)
    if missing is None:
        return None

    side = random.choice(LEGEND_SIDES)
    depth = random.choice(LEGEND_DEPTH_CHOICES)
    box_h, box_w = prototype.shape
    canvas = _sample_canvas(side, depth, k, box_h, box_w)
    if canvas is None:
        return None
    grid_h, grid_w = canvas

    positions = _place_positions(grid_h, grid_w, side, depth, box_h, box_w, k)
    if positions is None:
        return None

    bg_choices = [c for c in range(10) if c not in {GRAY, MAROON}]
    bg = random.choice(bg_choices)
    legend_colors = random.sample([c for c in range(10) if c not in {bg, GRAY, MAROON}], k)
    count_to_color = {count: color for (count, _cells), color in zip(sorted(missing), legend_colors, strict=False)}

    grid = np.full((grid_h, grid_w), bg, dtype=int)
    placements: list[tuple[int, int, int]] = []
    for (r0, c0), (count, removed) in zip(positions, missing, strict=False):
        copy_mask = prototype.copy()
        for rr, cc in removed:
            copy_mask[rr, cc] = False
        grid[r0 : r0 + box_h, c0 : c0 + box_w][copy_mask] = GRAY
        placements.append((r0, c0, count_to_color[count]))

    _paint_legend(grid, side, depth, count_to_color)
    expected = _expected_output(grid, prototype, placements)

    if not _background_is_dominant(grid, bg):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        result = _generate_attempt()
        if result is None:
            continue
        grid, _expected = result
        return grid
    raise RuntimeError("failed to generate a valid ARC input")
