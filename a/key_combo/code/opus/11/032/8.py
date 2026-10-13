from __future__ import annotations

import random
from collections import Counter, defaultdict
from itertools import combinations

import numpy as np


BLACK = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7)
MAX_ATTEMPTS = 400

SHAPES: dict[str, tuple[tuple[int, int], ...]] = {
    "square": ((0, 0), (0, 1), (1, 0), (1, 1)),
    "l3": ((0, 0), (1, 0), (1, 1)),
    "line3": ((0, 0), (0, 1), (0, 2)),
    "line4": ((0, 0), (0, 1), (0, 2), (0, 3)),
    "line5": ((0, 0), (0, 1), (0, 2), (0, 3), (0, 4)),
    "t_down": ((0, 0), (0, 1), (0, 2), (1, 1)),
    "t_up": ((0, 1), (1, 0), (1, 1), (1, 2)),
    "cross5": ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
    "p5": ((0, 0), (0, 1), (1, 0), (1, 1), (1, 2)),
    "l4": ((0, 0), (1, 0), (2, 0), (2, 1)),
    "zig4": ((0, 1), (0, 2), (1, 0), (1, 1)),
}
EXTRA_SHAPES = ("line3", "line4", "line5", "t_down", "t_up", "cross5", "p5", "l4", "zig4")


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox(coords: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _component_count(coords: set[tuple[int, int]]) -> int:
    if not coords:
        return 0
    unseen = set(coords)
    count = 0
    while unseen:
        count += 1
        start = unseen.pop()
        stack = [start]
        while stack:
            r, c = stack.pop()
            for nr, nc in _neighbors4(r, c):
                if (nr, nc) in unseen:
                    unseen.remove((nr, nc))
                    stack.append((nr, nc))
    return count


def _normalize(coords: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    return tuple(sorted((r - min_r, c - min_c) for r, c in coords))


def _flip_h(coords: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    max_c = max(c for _, c in coords)
    return tuple((r, max_c - c) for r, c in coords)


def _rot90(coords: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    max_r = max(r for r, _ in coords)
    return tuple((c, max_r - r) for r, c in coords)


def _variants(coords: tuple[tuple[int, int], ...]) -> list[tuple[tuple[int, int], ...]]:
    out: list[tuple[tuple[int, int], ...]] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    cur = coords
    for _ in range(4):
        for candidate in (cur, _flip_h(cur)):
            norm = _normalize(candidate)
            if norm not in seen:
                seen.add(norm)
                out.append(norm)
        cur = _rot90(cur)
    return out


def _translate(coords: tuple[tuple[int, int], ...], dr: int, dc: int) -> set[tuple[int, int]]:
    return {(r + dr, c + dc) for r, c in coords}


def _detect_objects(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    objects: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BLACK or visited[r, c]:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            coords: set[tuple[int, int]] = set()
            while stack:
                cr, cc = stack.pop()
                coords.add((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            objects.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "area": len(coords),
                }
            )
    return objects


def _simulate(
    grid: np.ndarray,
    *,
    swapped_parity: bool = False,
) -> tuple[list[dict], set[tuple[int, int]], dict[tuple[int, int], list[int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    objects = _detect_objects(g)
    assert 4 <= len(objects) <= 6, "expected 4-6 foreground objects"
    top_rows = [obj["bbox"][0] for obj in objects]
    assert len(set(top_rows)) == len(top_rows), "foreground objects must have distinct top rows"
    ordered = sorted(objects, key=lambda obj: obj["bbox"][0])

    occupancy: dict[tuple[int, int], list[int]] = defaultdict(list)
    for rank, obj in enumerate(ordered):
        move_left = (rank % 2 == 0) ^ bool(swapped_parity)
        dc = -1 if move_left else 1
        shifted = {
            (r, c + dc)
            for r, c in obj["coords"]
            if 0 <= c + dc < w and 0 <= r < h
        }
        obj["rank"] = rank
        obj["shifted"] = shifted
        obj["had_clip"] = len(shifted) != len(obj["coords"])
        for cell in shifted:
            occupancy[cell].append(rank)

    collisions = {cell for cell, ranks in occupancy.items() if len(ranks) > 1}
    return ordered, collisions, occupancy


def _render(
    grid: np.ndarray,
    *,
    swapped_parity: bool = False,
    erase_disconnected: bool = True,
    erase_all_colliders: bool = False,
    gray_collisions: bool = True,
) -> np.ndarray:
    ordered, collisions, _ = _simulate(grid, swapped_parity=swapped_parity)
    out = np.full_like(np.asarray(grid, dtype=int), BLACK)

    if gray_collisions:
        for r, c in collisions:
            out[r, c] = GREY

    for obj in ordered:
        remaining = obj["shifted"] - collisions
        had_collision = len(remaining) != len(obj["shifted"])
        if erase_all_colliders and had_collision:
            keep = False
        elif erase_disconnected:
            keep = _component_count(remaining) == 1
        else:
            keep = bool(remaining)
        obj["remaining"] = remaining
        obj["had_collision"] = had_collision
        obj["survives"] = keep
        if keep:
            for r, c in remaining:
                out[r, c] = obj["color"]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Shift objects alternately, gray out collisions, then erase any object broken by the collision loss.

    Input structure:
    - The grid has a black background and 4-6 small single-color 4-connected foreground objects.
    - Every object uses a distinct non-gray color and has a unique top bbox row, so the top-to-bottom order is unambiguous.
    - Objects are spaced apart in the input, but after the prescribed horizontal shifts some moved pixels overlap.

    Rule:
    - Sort the objects by increasing bbox top row.
    - Shift ranks 1, 3, 5, ... one column left and ranks 2, 4, 6, ... one column right; pixels shifted out of bounds disappear.
    - Any cell occupied by two or more moved objects becomes gray, and every colliding object loses that cell.
    - After all collisions, keep an object's remaining original-color pixels only if they still form exactly one 4-connected component; otherwise erase that object's remainder to black.
    - Preserve the gray collision cells in the final output.

    Key insight:
    - Vertical rank fixes the movement direction, but survival is decided later by topology: a compact shape can survive losing one pixel, while a thin bridge can be cut into separate pieces and vanish entirely.
    """
    return _render(grid)


def _pair_specs_l3_vs_square(top_row: int, left_col: int, colors: tuple[int, int]) -> list[dict]:
    top_moved = _translate(SHAPES["l3"], top_row, left_col)
    bottom_moved = _translate(SHAPES["square"], top_row + 1, left_col - 1)
    return [
        {"color": colors[0], "moved": top_moved, "top_row": top_row},
        {"color": colors[1], "moved": bottom_moved, "top_row": top_row + 1},
    ]


def _pair_specs_square_vs_square(top_row: int, left_col: int, colors: tuple[int, int]) -> list[dict]:
    top_moved = _translate(SHAPES["square"], top_row, left_col)
    bottom_moved = _translate(SHAPES["square"], top_row + 1, left_col - 1)
    return [
        {"color": colors[0], "moved": top_moved, "top_row": top_row},
        {"color": colors[1], "moved": bottom_moved, "top_row": top_row + 1},
    ]


def _input_coords_from_moved(moved: set[tuple[int, int]], rank: int) -> set[tuple[int, int]]:
    dc = 1 if rank % 2 == 0 else -1
    return {(r, c + dc) for r, c in moved}


def _in_bounds(coords: set[tuple[int, int]], h: int, w: int) -> bool:
    return all(0 <= r < h and 0 <= c < w for r, c in coords)


def _dilate8(coords: set[tuple[int, int]]) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                out.add((r + dr, c + dc))
    return out


def _compatible(candidate: set[tuple[int, int]], existing: list[set[tuple[int, int]]]) -> bool:
    for other in existing:
        if candidate & other:
            return False
        if _dilate8(candidate) & other:
            return False
    return True


def _add_ranked_group(
    specs: list[dict],
    objects: list[dict],
    input_coords_seen: list[set[tuple[int, int]]],
    moved_coords_seen: list[set[tuple[int, int]]],
    h: int,
    w: int,
) -> bool:
    prepared: list[dict] = []
    for offset, spec in enumerate(specs):
        rank = len(objects) + offset
        moved = set(spec["moved"])
        inp = _input_coords_from_moved(moved, rank)
        if not _in_bounds(moved, h, w) or not _in_bounds(inp, h, w):
            return False
        if _component_count(moved) != 1 or _component_count(inp) != 1:
            return False
        prepared.append({"color": spec["color"], "moved": moved, "input": inp, "top_row": spec["top_row"]})

    for i in range(len(prepared)):
        for j in range(i + 1, len(prepared)):
            if not _compatible(prepared[i]["input"], [prepared[j]["input"]]):
                return False

    for item in prepared:
        if not _compatible(item["input"], input_coords_seen):
            return False
        if not _compatible(item["moved"], moved_coords_seen):
            return False

    for item in prepared:
        objects.append(item)
        input_coords_seen.append(item["input"])
        moved_coords_seen.append(item["moved"])
    return True


def _random_extra_shape() -> tuple[tuple[int, int], ...]:
    name = random.choice(EXTRA_SHAPES)
    return random.choice(_variants(SHAPES[name]))


def _place_extra(
    rank: int,
    color: int,
    min_top_row: int,
    used_top_rows: set[int],
    input_coords_seen: list[set[tuple[int, int]]],
    moved_coords_seen: list[set[tuple[int, int]]],
    h: int,
    w: int,
) -> dict | None:
    row_options = list(range(min_top_row, h))
    random.shuffle(row_options)
    for top_row in row_options:
        if top_row in used_top_rows:
            continue
        for _ in range(40):
            shape = _random_extra_shape()
            max_r = max(r for r, _ in shape)
            max_c = max(c for _, c in shape)
            if top_row + max_r >= h:
                continue
            move_left = rank % 2 == 0
            min_col = 0
            max_col = w - 1 - max_c
            if move_left:
                max_col = min(max_col, w - 2 - max_c)
            else:
                min_col = max(min_col, 1)
            if min_col > max_col:
                continue
            left_col = random.randint(min_col, max_col)
            moved = _translate(shape, top_row, left_col)
            inp = _input_coords_from_moved(moved, rank)
            if not _compatible(inp, input_coords_seen):
                continue
            if not _compatible(moved, moved_coords_seen):
                continue
            return {"color": color, "moved": moved, "input": inp, "top_row": top_row}
    return None


def _render_input(objects: list[dict], shape: tuple[int, int]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for obj in objects:
        for r, c in obj["input"]:
            grid[r, c] = obj["color"]
    return grid


def _validate_case(inp: np.ndarray) -> bool:
    try:
        ordered, collisions, occupancy = _simulate(inp)
    except AssertionError:
        return False
    if len(collisions) < 2:
        return False
    collision_pairs = set()
    for cell, ranks in occupancy.items():
        if len(ranks) > 1:
            for pair in combinations(sorted(ranks), 2):
                collision_pairs.add(pair)
    if len(collision_pairs) < 2:
        return False

    out = main(inp)
    if not np.any(out == GREY):
        return False

    had_erased = False
    had_surviving_collision = False
    for obj in ordered:
        remaining = obj["shifted"] - collisions
        had_collision = len(remaining) != len(obj["shifted"])
        survives = _component_count(remaining) == 1
        if had_collision and not survives:
            had_erased = True
        if had_collision and survives:
            had_surviving_collision = True
    if not had_erased or not had_surviving_collision:
        return False

    if np.array_equal(out, _render(inp, erase_disconnected=False)):
        return False
    if np.array_equal(out, _render(inp, erase_all_colliders=True)):
        return False
    if np.array_equal(out, _render(inp, gray_collisions=False)):
        return False
    if np.array_equal(out, _render(inp, swapped_parity=True)):
        return False

    colors = [int(x) for x in np.unique(inp) if int(x) != BLACK]
    if GREY in colors or len(colors) != len(set(colors)):
        return False
    return True


def _attempt() -> np.ndarray | None:
    extra_count = random.randint(0, 2)
    total_objects = 4 + extra_count
    h = random.randint(14 + extra_count, 18)
    w = random.randint(14, 18)
    colors = random.sample(PALETTE, total_objects)

    objects: list[dict] = []
    input_coords_seen: list[set[tuple[int, int]]] = []
    moved_coords_seen: list[set[tuple[int, int]]] = []

    r1 = random.randint(1, 3)
    r2_min = r1 + 4
    r2_max = h - 3 - extra_count * 2
    if r2_min > r2_max:
        return None
    r2 = random.randint(r2_min, r2_max)

    c1 = random.randint(2, w - 3)
    pair1 = _pair_specs_l3_vs_square(r1, c1, (colors[0], colors[1]))
    if not _add_ranked_group(pair1, objects, input_coords_seen, moved_coords_seen, h, w):
        return None

    for _ in range(40):
        c2 = random.randint(2, w - 3)
        trial_objects = [obj.copy() for obj in objects]
        trial_input = input_coords_seen.copy()
        trial_moved = moved_coords_seen.copy()
        pair2 = _pair_specs_square_vs_square(r2, c2, (colors[2], colors[3]))
        if _add_ranked_group(pair2, trial_objects, trial_input, trial_moved, h, w):
            objects = trial_objects
            input_coords_seen = trial_input
            moved_coords_seen = trial_moved
            break
    else:
        return None

    used_top_rows = {r1, r1 + 1, r2, r2 + 1}
    min_extra_row = r2 + 3
    for color in colors[4:]:
        rank = len(objects)
        extra = _place_extra(
            rank,
            color,
            min_extra_row,
            used_top_rows,
            input_coords_seen,
            moved_coords_seen,
            h,
            w,
        )
        if extra is None:
            return None
        objects.append(extra)
        input_coords_seen.append(extra["input"])
        moved_coords_seen.append(extra["moved"])
        used_top_rows.add(extra["top_row"])

    inp = _render_input(objects, (h, w))
    if not _validate_case(inp):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")


if __name__ == "__main__":
    random.seed(0)
    np.set_printoptions(linewidth=200)
    sample = generate_input()
    print(sample)
    print()
    print(main(sample))
