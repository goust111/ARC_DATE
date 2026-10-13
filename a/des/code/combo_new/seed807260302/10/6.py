from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
GREY = 5
COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
MAX_GENERATION_ATTEMPTS = 500


def _components(mask: np.ndarray) -> list[np.ndarray]:
    height, width = map(int, mask.shape)
    seen = np.zeros((height, width), dtype=bool)
    comps: list[np.ndarray] = []

    for sr in range(height):
        for sc in range(width):
            if bool(seen[sr, sc]) or not bool(mask[sr, sc]):
                continue
            seen[sr, sc] = True
            queue: deque[tuple[int, int]] = deque([(sr, sc)])
            cells: list[tuple[int, int]] = []
            while queue:
                r, c = queue.popleft()
                cells.append((int(r), int(c)))
                for dr, dc in DIR4:
                    rr, cc = int(r + dr), int(c + dc)
                    if 0 <= rr < height and 0 <= cc < width and not bool(seen[rr, cc]) and bool(mask[rr, cc]):
                        seen[rr, cc] = True
                        queue.append((rr, cc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = map(int, coords.min(axis=0))
    r1, c1 = map(int, coords.max(axis=0))
    return r0, c0, r1, c1


def _touches_border(coords: np.ndarray, height: int, width: int) -> bool:
    return bool(
        np.any(coords[:, 0] == 0)
        or np.any(coords[:, 1] == 0)
        or np.any(coords[:, 0] == height - 1)
        or np.any(coords[:, 1] == width - 1)
    )


def _hole_count(mask: np.ndarray) -> int:
    height, width = map(int, mask.shape)
    holes = 0
    for comp in _components(~mask):
        if not _touches_border(comp, height, width):
            holes += 1
    return holes


def _find_horizontal_separator(g: np.ndarray) -> int:
    rows = [r for r in range(g.shape[0]) if bool(np.all(g[r, :] == GREY))]
    assert len(rows) == 1, "expected one full grey horizontal separator"
    return int(rows[0])


def _slot_spans(g: np.ndarray, sep_row: int) -> list[tuple[int, int]]:
    lower = g[sep_row + 1 :, :]
    assert lower.size, "expected a lower display area"

    sep_cols = [c for c in range(g.shape[1]) if bool(np.all(lower[:, c] == GREY))]
    runs: list[tuple[int, int]] = []
    for col in sep_cols:
        if not runs or col != runs[-1][1] + 1:
            runs.append((int(col), int(col)))
        else:
            runs[-1] = (runs[-1][0], int(col))
    assert all(start == end for start, end in runs), "vertical separators must be one-pixel lines"

    cols = [start for start, _end in runs]
    if len(cols) == 3:
        spans = [
            (0, cols[0] - 1),
            (cols[0] + 1, cols[1] - 1),
            (cols[1] + 1, cols[2] - 1),
            (cols[2] + 1, g.shape[1] - 1),
        ]
    elif len(cols) == 5 and cols[0] == 0 and cols[-1] == g.shape[1] - 1:
        spans = [(cols[i] + 1, cols[i + 1] - 1) for i in range(4)]
    else:
        raise AssertionError("expected three internal vertical separators, or separators plus outer borders")

    widths = [end - start + 1 for start, end in spans]
    assert len(spans) == 4 and min(widths) > 0 and len(set(widths)) == 1, "expected four equal display slots"
    for start, end in spans:
        assert bool(np.all(lower[:, start : end + 1] == BLACK)), "display slots must start empty"
    return [(int(start), int(end)) for start, end in spans]


def _object_hole_count(patch: np.ndarray, color: int) -> int:
    obj = patch == int(color)
    height, width = map(int, obj.shape)
    holes = 0
    for comp in _components(~obj):
        if _touches_border(comp, height, width):
            continue
        vals = patch[comp[:, 0], comp[:, 1]]
        assert bool(np.all(vals == BLACK)), "object holes must be black background"
        holes += 1
    return int(holes)


def _extract_upper_objects(g: np.ndarray, sep_row: int) -> list[dict[str, int | np.ndarray]]:
    upper = g[:sep_row, :]
    objects: list[dict[str, int | np.ndarray]] = []

    for color in [int(c) for c in np.unique(upper) if int(c) not in (BLACK, GREY)]:
        comps = _components(upper == color)
        assert len(comps) == 1, "each object color must identify one connected block"
        coords = comps[0]
        r0, c0, r1, c1 = _bbox(coords)
        patch = np.full((r1 - r0 + 1, c1 - c0 + 1), BLACK, dtype=int)
        local = coords.copy()
        local[:, 0] -= r0
        local[:, 1] -= c0
        patch[local[:, 0], local[:, 1]] = int(color)
        holes = _object_hole_count(patch, int(color))
        objects.append(
            {
                "color": int(color),
                "r0": int(r0),
                "c0": int(c0),
                "r1": int(r1),
                "c1": int(c1),
                "holes": int(holes),
                "patch": patch,
            }
        )

    objects.sort(key=lambda item: (int(item["c0"]), int(item["r0"])))
    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort each upper sprite into the lower display slot indexed by its number of holes.

    Input structure:
    - The background is black. A full grey horizontal separator splits the grid into an upper source area and a lower display area.
    - The upper area contains non-overlapping 4-connected monochrome sprites, each with a distinct color and 0 to 3 enclosed black holes.
    - The lower area is divided by full-height grey vertical separators into four equal empty display slots, numbered 0, 1, 2, and 3 from left to right.

    Rule:
    - Count the enclosed black holes of each upper sprite.
    - Keep the upper source area and all grey separators unchanged.
    - Copy each sprite's tight patch into display slot N, where N is the sprite's hole count, centered horizontally at the current top of that slot.
    - If several sprites map to one slot, read them by their upper-area left-to-right order and stack their patches from top to bottom.

    Key insight:
    The visible number of enclosed background regions inside a sprite is the hidden category key that selects the destination slot.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, _width = map(int, g.shape)

    sep_row = _find_horizontal_separator(g)
    spans = _slot_spans(g, sep_row)
    objects = _extract_upper_objects(g, sep_row)
    assert objects, "expected at least one upper object"

    out = g.copy()
    lower_top = int(sep_row + 1)
    for start, end in spans:
        out[lower_top:, start : end + 1] = BLACK

    groups: list[list[dict[str, int | np.ndarray]]] = [[] for _ in range(4)]
    for obj in objects:
        holes = int(obj["holes"])
        assert 0 <= holes <= 3, "hole count must select one of four slots"
        groups[holes].append(obj)

    for slot_index, group in enumerate(groups):
        start, end = spans[slot_index]
        slot_width = int(end - start + 1)
        row = lower_top
        for obj in group:
            patch = np.asarray(obj["patch"], dtype=int)
            color = int(obj["color"])
            ph, pw = map(int, patch.shape)
            assert pw <= slot_width, "sprite patch must fit in its display slot"
            assert row + ph <= height, "stacked sprites must fit in the lower area"
            col = int(start + (slot_width - pw) // 2)
            mask = patch == color
            target = out[row : row + ph, col : col + pw]
            assert bool(np.all(target[mask] == BLACK)), "display placement must not overwrite another sprite"
            target[mask] = color
            row += ph

    return out


def _try_place_holes(mask: np.ndarray, count: int) -> bool:
    height, width = map(int, mask.shape)
    carved: set[tuple[int, int]] = set()

    for _idx in range(count):
        placed = False
        for _attempt in range(80):
            if count == 1:
                hole_h, hole_w = random.choice(((1, 1), (1, 2), (2, 1), (2, 2)))
            else:
                hole_h, hole_w = (1, 1)
            if hole_h > height - 2 or hole_w > width - 2:
                continue

            r0 = random.randint(1, height - hole_h - 1)
            c0 = random.randint(1, width - hole_w - 1)
            cells = {(r, c) for r in range(r0, r0 + hole_h) for c in range(c0, c0 + hole_w)}
            if cells & carved:
                continue
            if any((r + dr, c + dc) in carved for r, c in cells for dr, dc in DIR4):
                continue

            for r, c in cells:
                mask[r, c] = False
            carved.update(cells)
            placed = True
            break
        if not placed:
            return False
    return True


def _carve_edge_notches(mask: np.ndarray) -> None:
    height, width = map(int, mask.shape)
    candidates = [(0, 0), (0, width - 1), (height - 1, 0), (height - 1, width - 1)]
    if height >= 4:
        candidates.extend([(random.randint(1, height - 2), 0), (random.randint(1, height - 2), width - 1)])
    if width >= 4:
        candidates.extend([(0, random.randint(1, width - 2)), (height - 1, random.randint(1, width - 2))])
    random.shuffle(candidates)

    for r, c in candidates[: random.randint(0, min(3, len(candidates)))]:
        test = mask.copy()
        test[r, c] = False
        if _hole_count(test) == 0 and len(_components(test)) == 1:
            mask[:, :] = test


def _make_shape(hole_count: int, max_side: int) -> np.ndarray | None:
    min_side = 3 if hole_count <= 1 else 4 if hole_count == 2 else 5
    for _attempt in range(100):
        height = random.randint(min_side, max_side)
        width = random.randint(min_side, max_side)
        mask = np.ones((height, width), dtype=bool)

        if hole_count == 0:
            _carve_edge_notches(mask)
        elif not _try_place_holes(mask, hole_count):
            continue

        if len(_components(mask)) == 1 and _hole_count(mask) == hole_count:
            return mask
    return None


def _mark_blocked(blocked: np.ndarray, row: int, col: int, height: int, width: int) -> None:
    r0 = max(0, int(row - 1))
    c0 = max(0, int(col - 1))
    r1 = min(blocked.shape[0], int(row + height + 1))
    c1 = min(blocked.shape[1], int(col + width + 1))
    blocked[r0:r1, c0:c1] = True


def _can_place(blocked: np.ndarray, row: int, col: int, height: int, width: int) -> bool:
    r0 = max(0, int(row - 1))
    c0 = max(0, int(col - 1))
    r1 = min(blocked.shape[0], int(row + height + 1))
    c1 = min(blocked.shape[1], int(col + width + 1))
    return not bool(blocked[r0:r1, c0:c1].any())


def _paint_shape(grid: np.ndarray, row: int, col: int, mask: np.ndarray, color: int) -> None:
    target = grid[row : row + mask.shape[0], col : col + mask.shape[1]]
    target[mask] = int(color)


def _separator_columns(slot_width: int) -> list[int]:
    return [slot_width, 2 * slot_width + 1, 3 * slot_width + 2]


def _gen_attempt() -> np.ndarray | None:
    slot_width = random.randint(5, 6)
    width = 4 * slot_width + 3

    duplicate_counts = random.sample([0, 1, 2, 3], random.randint(0, 4))
    hole_counts = [0, 1, 2, 3] + duplicate_counts
    random.shuffle(hole_counts)
    colors = random.sample(COLORS, len(hole_counts))

    specs: list[dict[str, int | np.ndarray]] = []
    for holes, color in zip(hole_counts, colors):
        mask = _make_shape(int(holes), slot_width)
        if mask is None:
            return None
        specs.append({"holes": int(holes), "color": int(color), "mask": mask})

    stack_heights = [
        sum(int(np.asarray(spec["mask"]).shape[0]) for spec in specs if int(spec["holes"]) == holes)
        for holes in range(4)
    ]
    min_lower = max(8, max(stack_heights))
    min_top = max(10, max(int(np.asarray(spec["mask"]).shape[0]) for spec in specs) + 2)
    lower_high = min(14, 30 - 1 - min_top)
    if min_lower > lower_high:
        return None

    lower_height = random.randint(min_lower, lower_high)
    top_high = 30 - 1 - lower_height
    if min_top > top_high:
        return None
    top_height = random.randint(min_top, top_high)
    height = top_height + 1 + lower_height

    grid = np.full((height, width), BLACK, dtype=int)
    grid[top_height, :] = GREY
    for col in _separator_columns(slot_width):
        grid[top_height + 1 :, col] = GREY

    blocked = np.zeros((top_height, width), dtype=bool)
    used_cols: set[int] = set()
    order = specs[:]
    random.shuffle(order)
    for spec in order:
        mask = np.asarray(spec["mask"], dtype=bool)
        obj_h, obj_w = map(int, mask.shape)
        candidates = [
            (r, c)
            for r in range(1, top_height - obj_h)
            for c in range(1, width - obj_w)
            if c not in used_cols and _can_place(blocked, r, c, obj_h, obj_w)
        ]
        if not candidates:
            return None
        row, col = random.choice(candidates)
        _paint_shape(grid, row, col, mask, int(spec["color"]))
        _mark_blocked(blocked, row, col, obj_h, obj_w)
        used_cols.add(int(col))

    try:
        out = main(grid)
    except AssertionError:
        return None
    if out.ndim != 2 or max(grid.shape) > 30 or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _attempt in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
