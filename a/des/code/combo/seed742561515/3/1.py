from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np


BG = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
ORTHOGONAL = {
    "top": {"left", "right"},
    "bottom": {"left", "right"},
    "left": {"top", "bottom"},
    "right": {"top", "bottom"},
}


@dataclass(frozen=True)
class Region:
    bbox: tuple[int, int, int, int]
    gap: tuple[int, int]
    gap_side: str
    label_color: int
    pattern_color: int
    pattern_coords: tuple[tuple[int, int], ...]
    marker_pos: tuple[int, int]
    marker_color: int


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


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
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(comp)
    return out


def _connected(mask: np.ndarray) -> bool:
    count = int(mask.sum())
    if count == 0:
        return False
    return len(_components(mask.astype(bool))) == 1


def _pattern_projection(
    pattern_coords: tuple[tuple[int, int], ...] | list[tuple[int, int]],
    bbox: tuple[int, int, int, int],
    gap: tuple[int, int],
    current_side: str,
    target_side: str,
) -> list[tuple[int, int]]:
    rows = [r for r, _c in pattern_coords]
    cols = [c for _r, c in pattern_coords]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)
    r0, c0, r1, c1 = bbox
    gap_r, gap_c = gap

    if current_side == "top":
        shift_r = (r0 - 1) - max_r
        if target_side == "left":
            shift_c = (gap_c - 1) - max_c
        elif target_side == "right":
            shift_c = (gap_c + 1) - min_c
        elif target_side == "top":
            shift_c = 0
        else:
            raise AssertionError("unsupported top-side projection")
    elif current_side == "bottom":
        shift_r = (r1 + 1) - min_r
        if target_side == "left":
            shift_c = (gap_c - 1) - max_c
        elif target_side == "right":
            shift_c = (gap_c + 1) - min_c
        elif target_side == "bottom":
            shift_c = 0
        else:
            raise AssertionError("unsupported bottom-side projection")
    elif current_side == "left":
        shift_c = (c0 - 1) - max_c
        if target_side == "top":
            shift_r = (gap_r - 1) - max_r
        elif target_side == "bottom":
            shift_r = (gap_r + 1) - min_r
        elif target_side == "left":
            shift_r = 0
        else:
            raise AssertionError("unsupported left-side projection")
    else:
        assert current_side == "right"
        shift_c = (c1 + 1) - min_c
        if target_side == "top":
            shift_r = (gap_r - 1) - max_r
        elif target_side == "bottom":
            shift_r = (gap_r + 1) - min_r
        elif target_side == "right":
            shift_r = 0
        else:
            raise AssertionError("unsupported right-side projection")

    return [(r + shift_r, c + shift_c) for r, c in pattern_coords]


def _parse_region(grid: np.ndarray, grey_cells: list[tuple[int, int]]) -> Region:
    h, w = grid.shape
    rows = [r for r, _c in grey_cells]
    cols = [c for _r, c in grey_cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    assert r1 - r0 >= 2 and c1 - c0 >= 2

    border = set()
    for c in range(c0, c1 + 1):
        border.add((r0, c))
        border.add((r1, c))
    for r in range(r0, r1 + 1):
        border.add((r, c0))
        border.add((r, c1))

    grey_set = set(grey_cells)
    gap_cells = list(border - grey_set)
    assert not (grey_set - border), "grey component must be a rectangle border with one gap"
    assert len(gap_cells) == 1, "expected exactly one missing border cell"
    gap_r, gap_c = gap_cells[0]
    assert (gap_r, gap_c) not in {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}, "gap cannot be a corner"

    if gap_r == r0:
        gap_side = "top"
        label_pos = (gap_r - 1, gap_c)
    elif gap_r == r1:
        gap_side = "bottom"
        label_pos = (gap_r + 1, gap_c)
    elif gap_c == c0:
        gap_side = "left"
        label_pos = (gap_r, gap_c - 1)
    else:
        assert gap_c == c1
        gap_side = "right"
        label_pos = (gap_r, gap_c + 1)

    lr, lc = label_pos
    assert 0 <= lr < h and 0 <= lc < w
    label_color = int(grid[lr, lc])
    assert label_color not in {BG, GREY}

    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    assert interior.size > 0
    colors = [int(v) for v in np.unique(interior) if int(v) not in {BG, GREY}]
    assert len(colors) == 2, "interior must contain exactly one pattern color and one marker color"

    counts = {color: int(np.count_nonzero(interior == color)) for color in colors}
    marker_colors = [color for color, count in counts.items() if count == 1]
    assert len(marker_colors) == 1
    marker_color = marker_colors[0]
    pattern_color = next(color for color in colors if color != marker_color)
    assert counts[pattern_color] >= 2

    pattern_mask = interior == pattern_color
    assert _connected(pattern_mask), "pattern must be 4-connected"
    pattern_coords = tuple(
        (r0 + 1 + int(r), c0 + 1 + int(c))
        for r, c in np.argwhere(pattern_mask)
    )
    marker_local = np.argwhere(interior == marker_color)
    assert marker_local.shape == (1, 2)
    marker_pos = (r0 + 1 + int(marker_local[0, 0]), c0 + 1 + int(marker_local[0, 1]))

    return Region(
        bbox=(r0, c0, r1, c1),
        gap=(gap_r, gap_c),
        gap_side=gap_side,
        label_color=label_color,
        pattern_color=pattern_color,
        pattern_coords=pattern_coords,
        marker_pos=marker_pos,
        marker_color=marker_color,
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each region's marker color to another region's outside label color.
    The matched region's gap side determines how the current region's pattern is
    projected from its own gap to the outside background. Borders, gaps, and
    labels stay fixed; the interior pattern and marker are erased.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    grey_components = _components(g == GREY)
    assert grey_components, "expected at least one grey frame"
    regions = [_parse_region(g, comp) for comp in grey_components]

    label_to_region: dict[int, Region] = {}
    for region in regions:
        assert region.label_color not in label_to_region, "outside labels must be unique"
        label_to_region[region.label_color] = region

    out = g.copy()
    for region in regions:
        for r, c in region.pattern_coords:
            out[r, c] = BG
        mr, mc = region.marker_pos
        out[mr, mc] = BG

    for region in regions:
        assert region.marker_color in label_to_region, "marker must point to another region label"
        target = label_to_region[region.marker_color]
        new_coords = _pattern_projection(
            region.pattern_coords,
            region.bbox,
            region.gap,
            region.gap_side,
            target.gap_side,
        )
        for nr, nc in new_coords:
            assert 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1], "projection must stay in bounds"
            r0, c0, r1, c1 = region.bbox
            assert not (r0 <= nr <= r1 and c0 <= nc <= c1), "projection must leave the source box"
            assert out[nr, nc] == BG, "projection must land on empty black background"
        for nr, nc in new_coords:
            out[nr, nc] = region.pattern_color

    return out


def _random_shape() -> np.ndarray:
    for _ in range(200):
        h = random.randint(2, 4)
        w = random.randint(2, 4)
        area = random.randint(3, min(6, h * w - 1))
        seed = (random.randrange(h), random.randrange(w))
        cells = {seed}
        frontier = list(_neighbors4(*seed))
        while frontier and len(cells) < area:
            idx = random.randrange(len(frontier))
            r, c = frontier.pop(idx)
            if not (0 <= r < h and 0 <= c < w) or (r, c) in cells:
                continue
            if not any((nr, nc) in cells for nr, nc in _neighbors4(r, c)):
                continue
            cells.add((r, c))
            for nbr in _neighbors4(r, c):
                if nbr not in cells:
                    frontier.append(nbr)
        if len(cells) != area:
            continue
        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        if not _connected(mask):
            continue
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        mask = mask[rows][:, cols]
        if mask.shape[0] >= 2 and mask.shape[1] >= 2:
            return mask
    raise RuntimeError("failed to build a random shape")


def _marker_candidates(
    interior_h: int,
    interior_w: int,
    pattern_coords: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for r in range(interior_h):
        for c in range(interior_w):
            if (r, c) in pattern_coords:
                continue
            if any((nr, nc) in pattern_coords for nr, nc in _neighbors4(r, c)):
                continue
            out.append((r, c))
    return out


def _build_region_tile(
    tile_h: int,
    tile_w: int,
    current_side: str,
    target_side: str,
    label_color: int,
    pattern_color: int,
    marker_color: int,
) -> np.ndarray | None:
    for _ in range(300):
        shape = _random_shape()
        shape_h, shape_w = shape.shape
        frame_h_min = max(6, shape_h + 4)
        frame_w_min = max(6, shape_w + 4)
        frame_h_max = min(8, tile_h - 2 * shape_h)
        frame_w_max = min(8, tile_w - 2 * shape_w)
        if frame_h_min > frame_h_max or frame_w_min > frame_w_max:
            continue

        frame_h = random.randint(frame_h_min, frame_h_max)
        frame_w = random.randint(frame_w_min, frame_w_max)
        top = random.randint(shape_h, tile_h - shape_h - frame_h)
        left = random.randint(shape_w, tile_w - shape_w - frame_w)
        bottom = top + frame_h - 1
        right = left + frame_w - 1

        interior_h = frame_h - 2
        interior_w = frame_w - 2
        patt_top = random.randint(0, interior_h - shape_h)
        patt_left = random.randint(0, interior_w - shape_w)
        rel_pattern = {
            (patt_top + int(r), patt_left + int(c))
            for r, c in np.argwhere(shape)
        }
        marker_slots = _marker_candidates(interior_h, interior_w, rel_pattern)
        if not marker_slots:
            continue
        marker_r, marker_c = random.choice(marker_slots)

        frame_mask = np.zeros((tile_h, tile_w), dtype=bool)
        frame_mask[top : bottom + 1, left] = True
        frame_mask[top : bottom + 1, right] = True
        frame_mask[top, left : right + 1] = True
        frame_mask[bottom, left : right + 1] = True

        pattern_abs = tuple(
            (top + 1 + r, left + 1 + c)
            for r, c in rel_pattern
        )
        marker_abs = (top + 1 + marker_r, left + 1 + marker_c)

        if current_side == "top":
            gap_choices = [(top, c) for c in range(left + 1, right)]
        elif current_side == "bottom":
            gap_choices = [(bottom, c) for c in range(left + 1, right)]
        elif current_side == "left":
            gap_choices = [(r, left) for r in range(top + 1, bottom)]
        else:
            assert current_side == "right"
            gap_choices = [(r, right) for r in range(top + 1, bottom)]
        random.shuffle(gap_choices)

        for gap_r, gap_c in gap_choices:
            if current_side == "top":
                label_pos = (gap_r - 1, gap_c)
            elif current_side == "bottom":
                label_pos = (gap_r + 1, gap_c)
            elif current_side == "left":
                label_pos = (gap_r, gap_c - 1)
            else:
                label_pos = (gap_r, gap_c + 1)
            lr, lc = label_pos
            if not (0 <= lr < tile_h and 0 <= lc < tile_w):
                continue

            projected = _pattern_projection(
                pattern_abs,
                (top, left, bottom, right),
                (gap_r, gap_c),
                current_side,
                target_side,
            )
            ok = True
            for pr, pc in projected:
                if not (0 <= pr < tile_h and 0 <= pc < tile_w):
                    ok = False
                    break
                if top <= pr <= bottom and left <= pc <= right:
                    ok = False
                    break
                if frame_mask[pr, pc] or (pr, pc) == label_pos:
                    ok = False
                    break
            if not ok:
                continue

            tile = np.full((tile_h, tile_w), BG, dtype=int)
            tile[frame_mask] = GREY
            tile[gap_r, gap_c] = BG
            tile[lr, lc] = label_color
            for pr, pc in pattern_abs:
                tile[pr, pc] = pattern_color
            tile[marker_abs] = marker_color
            return tile
    return None


def _sample_region_sides(region_count: int) -> tuple[list[str], list[int]]:
    if region_count == 2:
        sides = [random.choice(["top", "bottom"]), random.choice(["left", "right"])]
        random.shuffle(sides)
        return sides, [1, 0]

    assert region_count == 4
    sides = ["top", "bottom", "left", "right"]
    random.shuffle(sides)
    vertical = [i for i, side in enumerate(sides) if side in {"top", "bottom"}]
    horizontal = [i for i, side in enumerate(sides) if side in {"left", "right"}]
    perm = [-1] * 4
    targets_for_vertical = random.sample(horizontal, len(vertical))
    targets_for_horizontal = random.sample(vertical, len(horizontal))
    for src, tgt in zip(vertical, targets_for_vertical):
        perm[src] = tgt
    for src, tgt in zip(horizontal, targets_for_horizontal):
        perm[src] = tgt
    assert sorted(perm) == list(range(4))
    return sides, perm


def _layout_slots(region_count: int) -> tuple[list[tuple[int, int]], int, int, int, int, int, int]:
    tile_h = random.randint(14, 16)
    tile_w = random.randint(14, 16)
    gap_h = random.randint(2, 4)
    gap_w = random.randint(2, 4)
    margin_h = random.randint(1, 2)
    margin_w = random.randint(1, 2)

    if region_count == 2:
        mode = random.choice(["row", "col", "diag"])
        if mode == "row":
            slots = [(0, 0), (0, 1)]
            rows, cols = 1, 2
        elif mode == "col":
            slots = [(0, 0), (1, 0)]
            rows, cols = 2, 1
        else:
            slots = random.choice([[(0, 0), (1, 1)], [(0, 1), (1, 0)]])
            rows, cols = 2, 2
    else:
        assert region_count == 4
        slots = [(0, 0), (0, 1), (1, 0), (1, 1)]
        rows, cols = 2, 2

    grid_h = rows * tile_h + (rows - 1) * gap_h + 2 * margin_h
    grid_w = cols * tile_w + (cols - 1) * gap_w + 2 * margin_w
    return slots, tile_h, tile_w, gap_h, gap_w, margin_h, margin_w


def generate_input() -> np.ndarray:
    for _ in range(400):
        region_count = random.choice([2, 4])
        sides, perm = _sample_region_sides(region_count)
        colors = random.sample(PALETTE, 2 * region_count)
        label_colors = colors[:region_count]
        pattern_colors = colors[region_count:]

        slots, tile_h, tile_w, gap_h, gap_w, margin_h, margin_w = _layout_slots(region_count)
        grid_h = max(r for r, _c in slots) * (tile_h + gap_h) + tile_h + 2 * margin_h
        grid_w = max(c for _r, c in slots) * (tile_w + gap_w) + tile_w + 2 * margin_w
        grid = np.full((grid_h, grid_w), BG, dtype=int)

        ok = True
        for idx, (slot_r, slot_c) in enumerate(slots):
            target_side = sides[perm[idx]]
            if target_side not in ORTHOGONAL[sides[idx]]:
                ok = False
                break
            tile = _build_region_tile(
                tile_h,
                tile_w,
                sides[idx],
                target_side,
                label_colors[idx],
                pattern_colors[idx],
                label_colors[perm[idx]],
            )
            if tile is None:
                ok = False
                break
            top = margin_h + slot_r * (tile_h + gap_h)
            left = margin_w + slot_c * (tile_w + gap_w)
            grid[top : top + tile_h, left : left + tile_w] = tile
        if not ok:
            continue

        try:
            _ = main(grid)
        except AssertionError:
            continue
        return grid

    raise RuntimeError("failed to generate a valid sample")
