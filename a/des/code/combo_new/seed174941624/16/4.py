from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
DIRECTIONS = ("up", "down", "left", "right")
DIR_VECS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

MIN_RECT_SIZE = 2
MAX_RECT_SIZE = 5
MIN_ARROW_TAIL = 2
MAX_ARROW_TAIL = 4
MIN_GAP = 1
MAX_GAP = 3
GEN_RETRIES = 400
PACK_RETRIES = 200
GROUP_RETRIES = 120


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            components.append(comp)
    return components


def _bbox(coords: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _arrow_offsets(direction: str, tail_len: int) -> set[tuple[int, int]]:
    assert MIN_ARROW_TAIL <= tail_len <= MAX_ARROW_TAIL
    if direction == "up":
        return {(0, 0), (1, -1), (1, 0), (1, 1)} | {(r, 0) for r in range(2, tail_len + 3)}
    if direction == "down":
        return {(0, 0), (-1, -1), (-1, 0), (-1, 1)} | {(-r, 0) for r in range(2, tail_len + 3)}
    if direction == "left":
        return {(0, 0), (-1, 1), (0, 1), (1, 1)} | {(0, c) for c in range(2, tail_len + 3)}
    if direction == "right":
        return {(0, 0), (-1, -1), (0, -1), (1, -1)} | {(0, -c) for c in range(2, tail_len + 3)}
    raise AssertionError(f"unknown direction: {direction}")


def _match_arrow(coords: list[tuple[int, int]]) -> dict[str, object] | None:
    r0, c0, _r1, _c1 = _bbox(coords)
    normalized = {(r - r0, c - c0) for r, c in coords}
    for direction in DIRECTIONS:
        for tail_len in range(MIN_ARROW_TAIL, MAX_ARROW_TAIL + 1):
            offsets = _arrow_offsets(direction, tail_len)
            or0, oc0, _or1, _oc1 = _bbox(offsets)
            pattern = {(r - or0, c - oc0) for r, c in offsets}
            if pattern != normalized:
                continue
            tip = (int(r0 - or0), int(c0 - oc0))
            return {
                "direction": direction,
                "tail_len": int(tail_len),
                "tip": tip,
            }
    return None


def _match_rectangle(coords: list[tuple[int, int]]) -> dict[str, object] | None:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < MIN_RECT_SIZE or w < MIN_RECT_SIZE:
        return None
    if len(coords) != h * w:
        return None
    return {"bbox": (int(r0), int(c0), int(r1), int(c1))}


def _objects_are_separated(objects: list[set[tuple[int, int]]]) -> bool:
    occupied: dict[tuple[int, int], int] = {}
    for idx, cells in enumerate(objects):
        for r, c in cells:
            if (r, c) in occupied:
                return False
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    owner = occupied.get((r + dr, c + dc))
                    if owner is not None and owner != idx:
                        return False
        for cell in cells:
            occupied[cell] = idx
    return True


def _parse_scene(grid: np.ndarray) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    arrows: list[dict[str, object]] = []
    rectangles: list[dict[str, object]] = []
    object_sets: list[set[tuple[int, int]]] = []

    for comp in _components4(g != BG):
        colors = {int(g[r, c]) for r, c in comp}
        assert len(colors) == 1, "each object must be monochrome"
        color = next(iter(colors))

        arrow_info = _match_arrow(comp)
        if arrow_info is not None:
            arrows.append(
                {
                    "type": "arrow",
                    "color": int(color),
                    "coords": [(int(r), int(c)) for r, c in comp],
                    "tip": arrow_info["tip"],
                    "direction": arrow_info["direction"],
                    "tail_len": arrow_info["tail_len"],
                }
            )
            object_sets.append(set(comp))
            continue

        rect_info = _match_rectangle(comp)
        assert rect_info is not None, f"unrecognized object with color {color}"
        rectangles.append(
            {
                "type": "rectangle",
                "color": int(color),
                "coords": [(int(r), int(c)) for r, c in comp],
                "bbox": rect_info["bbox"],
            }
        )
        object_sets.append(set(comp))

    assert arrows, "expected at least one arrow"
    assert rectangles, "expected at least one rectangle"
    assert _objects_are_separated(object_sets), "objects must not touch each other"

    rect_colors = [int(rect["color"]) for rect in rectangles]
    arrow_colors = [int(arrow["color"]) for arrow in arrows]
    assert len(set(rect_colors)) == len(rectangles), "rectangle colors must be unique"
    assert not (set(rect_colors) & set(arrow_colors)), "arrow colors must differ from rectangle colors"
    return arrows, rectangles


def _assign_targets(
    grid: np.ndarray,
    arrows: list[dict[str, object]],
    rectangles: list[dict[str, object]],
) -> list[int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    rect_lookup: dict[tuple[int, int], int] = {}
    for idx, rect in enumerate(rectangles):
        for cell in rect["coords"]:
            rect_lookup[cell] = idx

    targets: list[int] = []
    for arrow in arrows:
        tip_r, tip_c = arrow["tip"]
        dr, dc = DIR_VECS[str(arrow["direction"])]
        rr, cc = int(tip_r + dr), int(tip_c + dc)
        target = None
        while 0 <= rr < h and 0 <= cc < w:
            if g[rr, cc] != BG:
                target = rect_lookup.get((int(rr), int(cc)))
                assert target is not None, "first object along the arrow ray must be a rectangle"
                break
            rr += dr
            cc += dc
        assert target is not None, "every arrow must point to a rectangle"
        targets.append(int(target))
    return targets


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    arrows, rectangles = _parse_scene(g)
    targets = _assign_targets(g, arrows, rectangles)

    out = g.copy()
    for rect_idx in sorted(set(targets)):
        for r, c in rectangles[rect_idx]["coords"]:
            out[r, c] = BG
    for arrow, rect_idx in zip(arrows, targets):
        new_color = int(rectangles[rect_idx]["color"])
        for r, c in arrow["coords"]:
            out[r, c] = new_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each arrow with the color of the rectangle it points to, then erase the targeted rectangle.

    Input structure:
    - The background is black.
    - The foreground contains several monochrome arrow objects and several monochrome filled rectangles.
    - Each arrow is made from a rotated 3x3 arrowhead with a 1-pixel-wide tail of length 2-4 extending from the back of the head.
    - Objects are pairwise separated, rectangle colors are all distinct, and arrow colors do not match rectangle colors.

    Rule:
    - Identify each arrow's direction from its tip and tail orientation.
    - Starting at the tip, scan cell by cell along that direction until the first non-black object is reached.
    - That first object must be the target rectangle for the arrow.
    - Recolor the full arrow to the target rectangle's color, and replace every targeted rectangle with black.
    - If several arrows point to the same rectangle, all of them inherit that rectangle's color while the rectangle is erased only once.

    Key insight:
    - The arrow's original color is irrelevant; only the arrow geometry determines a direction, and that direction selects the nearest rectangle whose color gets transferred back onto the arrow.
    """
    return _solve(np.asarray(grid, dtype=int))


def _direction_assignments(shared_mode: bool) -> list[list[str]]:
    if shared_mode:
        rect_count = random.randint(1, 3)
        arrow_count = random.randint(max(2, rect_count + 1), min(6, 4 * rect_count))
    else:
        rect_count = random.randint(2, 3)
        arrow_count = rect_count

    assignments = [set() for _ in range(rect_count)]
    direction_cycle = list(DIRECTIONS)
    random.shuffle(direction_cycle)
    cycle_idx = 0

    rect_order = list(range(rect_count))
    random.shuffle(rect_order)
    for rect_idx in rect_order:
        direction = direction_cycle[cycle_idx % len(DIRECTIONS)]
        cycle_idx += 1
        assignments[rect_idx].add(direction)

    while sum(len(item) for item in assignments) < arrow_count:
        candidates = [idx for idx, item in enumerate(assignments) if len(item) < len(DIRECTIONS)]
        rect_idx = random.choice(candidates)
        preferred = direction_cycle[cycle_idx % len(DIRECTIONS)]
        cycle_idx += 1
        if preferred in assignments[rect_idx]:
            options = [d for d in DIRECTIONS if d not in assignments[rect_idx]]
            random.shuffle(options)
            preferred = options[0]
        assignments[rect_idx].add(preferred)

    result = [list(item) for item in assignments]
    for item in result:
        random.shuffle(item)
    return result


def _build_group(rect_color: int, directions: list[str], arrow_palette: list[int]) -> dict[str, object] | None:
    for _ in range(GROUP_RETRIES):
        rect_h = random.randint(MIN_RECT_SIZE, min(MAX_RECT_SIZE, 4 + (len(directions) == 1)))
        rect_w = random.randint(MIN_RECT_SIZE, min(MAX_RECT_SIZE, 4 + (len(directions) == 1)))
        rect_cells = {(r, c) for r in range(rect_h) for c in range(rect_w)}

        arrow_cells_list: list[set[tuple[int, int]]] = []
        arrow_specs: list[dict[str, object]] = []
        for direction in directions:
            tail_max = MAX_ARROW_TAIL if len(directions) <= 2 else MAX_ARROW_TAIL - 1
            tail_len = random.randint(MIN_ARROW_TAIL, tail_max)
            gap_max = MAX_GAP if len(directions) == 1 else MAX_GAP - 1
            gap = random.randint(MIN_GAP, gap_max)
            color = random.choice(arrow_palette)

            if direction in {"up", "down"}:
                tip_col = random.randint(0, rect_w - 1)
                tip = (rect_h + gap, tip_col) if direction == "up" else (-gap - 1, tip_col)
            else:
                tip_row = random.randint(0, rect_h - 1)
                tip = (tip_row, rect_w + gap) if direction == "left" else (tip_row, -gap - 1)

            coords = {(tip[0] + dr, tip[1] + dc) for dr, dc in _arrow_offsets(direction, tail_len)}
            arrow_cells_list.append(coords)
            arrow_specs.append(
                {
                    "direction": direction,
                    "tail_len": int(tail_len),
                    "tip": (int(tip[0]), int(tip[1])),
                    "color": int(color),
                    "coords": coords,
                }
            )

        if not _objects_are_separated([rect_cells] + arrow_cells_list):
            continue

        all_cells = set(rect_cells)
        for coords in arrow_cells_list:
            all_cells |= coords
        r0, c0, r1, c1 = _bbox(all_cells)
        patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
        for r, c in rect_cells:
            patch[r - r0, c - c0] = rect_color
        for spec in arrow_specs:
            for r, c in spec["coords"]:
                patch[r - r0, c - c0] = int(spec["color"])
        return {
            "patch": patch,
            "rect_color": int(rect_color),
            "directions": list(directions),
            "arrow_count": len(directions),
        }
    return None


def _random_row_partition(items: list[int], row_count: int) -> list[list[int]]:
    if row_count == 1:
        return [items]
    cuts = sorted(random.sample(range(1, len(items)), row_count - 1))
    rows: list[list[int]] = []
    start = 0
    for cut in cuts + [len(items)]:
        rows.append(items[start:cut])
        start = cut
    return rows


def _pack_groups(groups: list[dict[str, object]]) -> np.ndarray | None:
    patches = [np.asarray(group["patch"], dtype=int) for group in groups]
    count = len(patches)
    for _ in range(PACK_RETRIES):
        order = list(range(count))
        random.shuffle(order)
        row_count = random.randint(1, count)
        rows = _random_row_partition(order, row_count)

        row_gaps = [[random.randint(1, 3) for _ in range(len(row) - 1)] for row in rows]
        col_heights = [max(int(patches[idx].shape[0]) for idx in row) for row in rows]
        row_widths = [
            sum(int(patches[idx].shape[1]) for idx in row) + sum(gaps)
            for row, gaps in zip(rows, row_gaps)
        ]
        between_rows = [random.randint(1, 3) for _ in range(len(rows) - 1)]

        base_h = sum(col_heights) + sum(between_rows)
        base_w = max(row_widths)
        if base_h > 30 or base_w > 30:
            continue

        height = random.randint(base_h, min(30, base_h + 6))
        width = random.randint(base_w, min(30, base_w + 6))
        top_margin = random.randint(0, height - base_h)
        y = top_margin

        grid = np.zeros((height, width), dtype=int)
        for row_idx, row in enumerate(rows):
            row_height = col_heights[row_idx]
            row_width = row_widths[row_idx]
            x = random.randint(0, width - row_width)
            for pos, patch_idx in enumerate(row):
                patch = patches[patch_idx]
                ph, pw = patch.shape
                y_off = random.randint(0, row_height - ph)
                sub = grid[y + y_off : y + y_off + ph, x : x + pw]
                if np.any((sub != BG) & (patch != BG)):
                    break
                write_mask = patch != BG
                sub[write_mask] = patch[write_mask]
                x += pw
                if pos < len(row) - 1:
                    x += row_gaps[row_idx][pos]
            else:
                y += row_height
                if row_idx < len(rows) - 1:
                    y += between_rows[row_idx]
                continue
            break
        else:
            return grid
    return None


def _validate_generated_grid(
    grid: np.ndarray,
    expected_rectangles: int,
    expected_arrows: int,
    shared_mode: bool,
) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False

    try:
        arrows, rectangles = _parse_scene(grid)
        targets = _assign_targets(grid, arrows, rectangles)
        out = _solve(grid)
    except AssertionError:
        return False

    if len(arrows) != expected_arrows or len(rectangles) != expected_rectangles:
        return False
    if len(set(targets)) != expected_rectangles:
        return False

    target_counts = Counter(targets)
    if shared_mode:
        if max(target_counts.values()) < 2:
            return False
    else:
        if any(count != 1 for count in target_counts.values()):
            return False

    if np.array_equal(out, grid):
        return False
    if any(direction not in DIRECTIONS for direction in (arrow["direction"] for arrow in arrows)):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        shared_mode = random.random() < 0.5
        assignments = _direction_assignments(shared_mode)
        rect_count = len(assignments)
        arrow_count = sum(len(item) for item in assignments)

        rect_colors = random.sample(PALETTE, rect_count)
        arrow_palette = [color for color in PALETTE if color not in rect_colors]
        if not arrow_palette:
            continue

        groups: list[dict[str, object]] = []
        for rect_color, directions in zip(rect_colors, assignments):
            group = _build_group(rect_color, directions, arrow_palette)
            if group is None:
                groups = []
                break
            groups.append(group)
        if not groups:
            continue

        grid = _pack_groups(groups)
        if grid is None:
            continue
        if _validate_generated_grid(grid, rect_count, arrow_count, shared_mode):
            return grid

    raise RuntimeError("failed to generate a valid arrow-to-rectangle puzzle")
