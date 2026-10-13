from __future__ import annotations

import random
from collections import Counter

import numpy as np


class Color:
    BLACK = 0
    BLUE = 1
    RED = 2
    GREEN = 3
    YELLOW = 4
    GREY = 5
    PINK = 6
    ORANGE = 7
    TEAL = 8
    MAROON = 9


BBox = tuple[int, int, int, int]
Coord = tuple[int, int]
Block = dict[str, object]


# Minimal arc_utils-style helpers, keeping the same (r, c) and bbox conventions.
def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def bbox_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def bbox_h(bbox: BBox) -> int:
    r0, _c0, r1, _c1 = bbox
    return int(r1 - r0 + 1)


def bbox_w(bbox: BBox) -> int:
    _r0, c0, _r1, c1 = bbox
    return int(c1 - c0 + 1)


def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    assert mask.ndim == 2
    if connectivity == 4:
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        steps = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[Coord] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in steps:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def find_components(
    grid: np.ndarray,
    *,
    bg: int | None = None,
    connectivity: int = 4,
    monochromatic: bool = True,
    min_area: int = 1,
) -> list[Block]:
    g = np.asarray(grid, dtype=int)
    if bg is None:
        bg = get_bg(g)
    colors = [int(color) for color in np.unique(g) if int(color) != int(bg)]
    if monochromatic:
        out: list[Block] = []
        for color in colors:
            for coords in components_mask(g == color, connectivity=connectivity):
                if int(coords.shape[0]) < int(min_area):
                    continue
                out.append(
                    {
                        "color": int(color),
                        "coords": coords,
                        "bbox": bbox_coords(coords),
                        "area": int(coords.shape[0]),
                    }
                )
        return out

    mask = g != int(bg)
    out = []
    for coords in components_mask(mask, connectivity=connectivity):
        if int(coords.shape[0]) < int(min_area):
            continue
        colors_in_comp = {int(g[int(r), int(c)]) for r, c in coords}
        out.append(
            {
                "color": -1 if len(colors_in_comp) != 1 else next(iter(colors_in_comp)),
                "coords": coords,
                "bbox": bbox_coords(coords),
                "area": int(coords.shape[0]),
            }
        )
    return out


def _extract_blocks(grid: np.ndarray, bg: int) -> tuple[Block, list[Block]]:
    blocks = find_components(grid, bg=bg, connectivity=4, monochromatic=True)
    assert blocks, "expected at least one colored 2x2 block"

    start_blocks: list[Block] = []
    target_blocks: list[Block] = []
    used_colors: set[int] = set()

    for block in blocks:
        color = int(block["color"])
        assert color not in used_colors, "all blocks must have distinct colors"
        used_colors.add(color)

        coords = np.asarray(block["coords"], dtype=int)
        r0, c0, r1, c1 = block["bbox"]
        assert int(block["area"]) == 4, "every object must be a 2x2 block"
        assert bbox_h(block["bbox"]) == 2 and bbox_w(block["bbox"]) == 2, "every object must be a 2x2 block"
        expected_coords = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
        actual_coords = {tuple(map(int, rc)) for rc in coords}
        assert actual_coords == expected_coords, "block must occupy its full 2x2 bbox"
        assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color), "block must be solid monochromatic"

        if color == Color.RED:
            start_blocks.append(block)
        else:
            target_blocks.append(block)

    assert len(start_blocks) == 1, "expected exactly one red start block"
    assert target_blocks, "expected at least one target block"
    return start_blocks[0], target_blocks


def _visible_step(grid: np.ndarray, bg: int, a: Block, b: Block) -> tuple[str, int] | None:
    ar0, ac0, ar1, ac1 = a["bbox"]
    br0, bc0, br1, bc1 = b["bbox"]

    if ar0 == br0 and ar1 == br1:
        c0 = min(ac1, bc1) + 1
        c1 = max(ac0, bc0) - 1
        if c0 <= c1 and not np.all(grid[ar0 : ar1 + 1, c0 : c1 + 1] == bg):
            return None
        return "h", abs(int(bc0) - int(ac0))

    if ac0 == bc0 and ac1 == bc1:
        r0 = min(ar1, br1) + 1
        r1 = max(ar0, br0) - 1
        if r0 <= r1 and not np.all(grid[r0 : r1 + 1, ac0 : ac1 + 1] == bg):
            return None
        return "v", abs(int(br0) - int(ar0))

    return None


def _walk_sequence(grid: np.ndarray) -> tuple[list[Block], list[Block]]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    current, unvisited = _extract_blocks(g, bg)
    visited = [current]

    while True:
        candidates: list[tuple[int, str, Block]] = []
        for block in unvisited:
            step = _visible_step(g, bg, current, block)
            if step is None:
                continue
            axis, distance = step
            candidates.append((int(distance), axis, block))

        if not candidates:
            return visited, unvisited

        min_distance = min(distance for distance, _axis, _block in candidates)
        nearest = [candidate for candidate in candidates if candidate[0] == min_distance]
        assert len(nearest) == 1, "nearest visible block must be unique"

        _distance, _axis, chosen = nearest[0]
        visited.append(chosen)
        unvisited = [block for block in unvisited if block is not chosen]
        current = chosen


def _draw_nested_rings(colors: list[int]) -> np.ndarray:
    side = 2 * len(colors) - 1
    out = np.zeros((side, side), dtype=int)
    for layer, color in enumerate(colors):
        lo = layer
        hi = side - 1 - layer
        if lo == hi:
            out[lo, hi] = int(color)
            continue
        out[lo, lo : hi + 1] = int(color)
        out[hi, lo : hi + 1] = int(color)
        out[lo : hi + 1, lo] = int(color)
        out[lo : hi + 1, hi] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Follow the nearest visible block walk and redraw the visited colors as nested square rings.

    Input structure
    - The input is a black background containing one red 2x2 start block and several other solid 2x2 target blocks.
    - Every non-red block has a distinct color.
    - Two blocks can interact only when they share the same two-row band or the same two-column band.

    Rule
    - Start from the red block and repeatedly move to the unique nearest unvisited block visible in the same horizontal or vertical 2-cell band.
    - Visibility is blocked by any non-background pixel lying between the two aligned blocks inside that shared band.
    - Let k be the number of visited blocks including the red start; output a (2k-1)x(2k-1) square whose outer ring is red and whose inner rings follow the visit order, with the final visited color at the center.
    - Ignore every block that the greedy walk never reaches.

    Key insight
    - The answer depends on a blocked line-of-sight traversal order, not on object positions alone: extra blocks can be present yet disappear because the walk never visits them.
    """
    visited, _unvisited = _walk_sequence(np.asarray(grid, dtype=int))
    colors = [int(block["color"]) for block in visited]
    return _draw_nested_rings(colors)


BLOCK_SIZE = 2
EDGE_PADDING = 1
TARGET_COLORS = [
    Color.BLUE,
    Color.GREEN,
    Color.YELLOW,
    Color.GREY,
    Color.PINK,
    Color.ORANGE,
    Color.TEAL,
    Color.MAROON,
]
GRID_ROWS_RANGE = (15, 27)
GRID_COLS_RANGE = (15, 27)
PATH_MOVES_RANGE = (2, 5)
MOVE_GAPS = tuple(range(1, 10))
MAX_ATTEMPTS = 400
SCENARIOS = ("lure", "blocked", "both", "mixed")

_GEN_CALL_COUNT = 0
_SEEN_OUTPUT_SIGS: set[tuple[tuple[int, int], bytes]] = set()


def _gen_bbox(pos: Coord) -> BBox:
    r, c = pos
    return int(r), int(c), int(r + BLOCK_SIZE - 1), int(c + BLOCK_SIZE - 1)


def _gen_position_range(length: int) -> range:
    return range(EDGE_PADDING, length - EDGE_PADDING - BLOCK_SIZE + 1)


def _gen_axis(a: Coord, b: Coord) -> str | None:
    if a[0] == b[0]:
        return "h"
    if a[1] == b[1]:
        return "v"
    return None


def _gen_distance(a: Coord, b: Coord) -> int | None:
    axis = _gen_axis(a, b)
    if axis == "h":
        return abs(int(a[1]) - int(b[1]))
    if axis == "v":
        return abs(int(a[0]) - int(b[0]))
    return None


def _gen_rects_overlap(a: BBox, b: BBox) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _gen_blocks_have_gap(a: Coord, b: Coord) -> bool:
    ar0, ac0, ar1, ac1 = _gen_bbox(a)
    br0, bc0, br1, bc1 = _gen_bbox(b)
    row_gap = max(br0 - ar1 - 1, ar0 - br1 - 1)
    col_gap = max(bc0 - ac1 - 1, ac0 - bc1 - 1)
    return row_gap >= 1 or col_gap >= 1


def _gen_far_from_all(pos: Coord, positions: list[Coord]) -> bool:
    return all(_gen_blocks_have_gap(pos, other) for other in positions)


def _gen_corridor(a: Coord, b: Coord) -> tuple[BBox | None, str | None]:
    axis = _gen_axis(a, b)
    if axis == "h":
        r = a[0]
        c0 = min(a[1], b[1]) + BLOCK_SIZE
        c1 = max(a[1], b[1]) - 1
        if c0 <= c1:
            return (r, c0, r + BLOCK_SIZE - 1, c1), axis
        return None, axis
    if axis == "v":
        c = a[1]
        r0 = min(a[0], b[0]) + BLOCK_SIZE
        r1 = max(a[0], b[0]) - 1
        if r0 <= r1:
            return (r0, c, r1, c + BLOCK_SIZE - 1), axis
        return None, axis
    return None, None


def _gen_corridor_clear(a: Coord, b: Coord, positions: list[Coord]) -> bool:
    rect, axis = _gen_corridor(a, b)
    if axis is None:
        return False
    if rect is None:
        return True
    for pos in positions:
        if pos == a or pos == b:
            continue
        if _gen_rects_overlap(rect, _gen_bbox(pos)):
            return False
    return True


def _gen_visible_axis(a: Coord, b: Coord, positions: list[Coord]) -> str | None:
    axis = _gen_axis(a, b)
    if axis is None:
        return None
    return axis if _gen_corridor_clear(a, b, positions) else None


def _gen_turn_count(path: list[Coord]) -> int:
    turns = 0
    prev_axis = None
    for a, b in zip(path, path[1:]):
        axis = _gen_axis(a, b)
        if prev_axis is not None and axis != prev_axis:
            turns += 1
        prev_axis = axis
    return turns


def _gen_walk_is_forced(path: list[Coord], distractors: list[Coord]) -> bool:
    positions = list(path) + list(distractors)
    for i, current in enumerate(path[:-1]):
        candidates: list[tuple[int, Coord]] = []
        for other in list(path[i + 1 :]) + list(distractors):
            axis = _gen_visible_axis(current, other, positions)
            if axis is None:
                continue
            distance = _gen_distance(current, other)
            assert distance is not None
            candidates.append((int(distance), other))
        if not candidates:
            return False
        nearest_distance = min(distance for distance, _other in candidates)
        nearest = [other for distance, other in candidates if distance == nearest_distance]
        if len(nearest) != 1 or nearest[0] != path[i + 1]:
            return False

    endpoint = path[-1]
    return all(_gen_visible_axis(endpoint, other, positions) is None for other in distractors)


def _gen_in_bounds(pos: Coord, rows: int, cols: int) -> bool:
    r, c = pos
    return (
        EDGE_PADDING <= int(r) <= rows - EDGE_PADDING - BLOCK_SIZE
        and EDGE_PADDING <= int(c) <= cols - EDGE_PADDING - BLOCK_SIZE
    )


def _gen_shuffled(items: list[object], rng: random.Random) -> list[object]:
    items = list(items)
    rng.shuffle(items)
    return items


def _gen_next_options(
    rows: int,
    cols: int,
    path: list[Coord],
    total_steps: int,
    target_turns: int,
    rng: random.Random,
) -> list[Coord]:
    current = path[-1]
    prev_axis = _gen_axis(path[-2], path[-1]) if len(path) >= 2 else None
    turns_so_far = _gen_turn_count(path)
    moves_done = len(path) - 1
    moves_left_after = total_steps - (moves_done + 1)

    options: list[tuple[bool, Coord]] = []
    for axis in ("h", "v"):
        new_turns = turns_so_far + int(prev_axis is not None and axis != prev_axis)
        if not (new_turns <= target_turns <= new_turns + moves_left_after):
            continue
        for sign in (-1, 1):
            for gap in MOVE_GAPS:
                offset = sign * (gap + BLOCK_SIZE)
                candidate = (current[0], current[1] + offset) if axis == "h" else (current[0] + offset, current[1])
                if not _gen_in_bounds(candidate, rows, cols):
                    continue
                if not _gen_far_from_all(candidate, path):
                    continue
                candidate_path = path + [candidate]
                if not _gen_walk_is_forced(candidate_path, []):
                    continue
                prefer_turn = prev_axis is not None and axis != prev_axis
                options.append((prefer_turn, candidate))

    turn_first = turns_so_far < target_turns
    if turn_first:
        preferred = [candidate for prefer, candidate in options if prefer]
        fallback = [candidate for prefer, candidate in options if not prefer]
        return _gen_shuffled(preferred, rng) + _gen_shuffled(fallback, rng)
    return _gen_shuffled([candidate for _prefer, candidate in options], rng)


def _gen_extend_path(
    rows: int,
    cols: int,
    total_steps: int,
    target_turns: int,
    path: list[Coord],
    rng: random.Random,
) -> list[Coord] | None:
    if len(path) == total_steps + 1:
        return path if _gen_turn_count(path) == target_turns and _gen_walk_is_forced(path, []) else None

    for candidate in _gen_next_options(rows, cols, path, total_steps, target_turns, rng):
        found = _gen_extend_path(rows, cols, total_steps, target_turns, path + [candidate], rng)
        if found is not None:
            return found
    return None


def _gen_build_path(rows: int, cols: int, total_steps: int, target_turns: int, rng: random.Random) -> list[Coord] | None:
    starts = [(r, c) for r in _gen_position_range(rows) for c in _gen_position_range(cols)]
    for start in _gen_shuffled(starts, rng)[:120]:
        found = _gen_extend_path(rows, cols, total_steps, target_turns, [start], rng)
        if found is not None:
            return found
    return None


def _gen_has_intervening_block(a: Coord, b: Coord, positions: list[Coord]) -> bool:
    axis = _gen_axis(a, b)
    if axis == "h":
        lo, hi = sorted((a[1], b[1]))
        return any(pos not in {a, b} and pos[0] == a[0] and lo < pos[1] < hi for pos in positions)
    if axis == "v":
        lo, hi = sorted((a[0], b[0]))
        return any(pos not in {a, b} and pos[1] == a[1] and lo < pos[0] < hi for pos in positions)
    return False


def _gen_lure_steps(path: list[Coord], distractors: list[Coord], pos: Coord) -> list[int]:
    positions = list(path) + list(distractors)
    steps: list[int] = []
    for i in range(len(path) - 1):
        if _gen_visible_axis(path[i], pos, positions) is None:
            continue
        chosen_distance = _gen_distance(path[i], path[i + 1])
        lure_distance = _gen_distance(path[i], pos)
        assert chosen_distance is not None and lure_distance is not None
        if lure_distance > chosen_distance:
            steps.append(i)
    return steps


def _gen_blocked_steps(path: list[Coord], distractors: list[Coord], pos: Coord) -> list[int]:
    positions = list(path) + list(distractors)
    steps: list[int] = []
    for i in range(len(path) - 1):
        if _gen_axis(path[i], pos) is None:
            continue
        if _gen_visible_axis(path[i], pos, positions) is not None:
            continue
        if _gen_has_intervening_block(path[i], pos, positions):
            steps.append(i)
    return steps


def _gen_candidate_positions(rows: int, cols: int, occupied: list[Coord]) -> list[Coord]:
    candidates = []
    occupied_set = set(occupied)
    for r in _gen_position_range(rows):
        for c in _gen_position_range(cols):
            pos = (r, c)
            if pos in occupied_set:
                continue
            if not _gen_far_from_all(pos, occupied):
                continue
            candidates.append(pos)
    return candidates


def _gen_pick_distractor(
    rows: int,
    cols: int,
    path: list[Coord],
    distractors: list[Coord],
    kind: str,
    rng: random.Random,
) -> tuple[Coord, dict[str, object]] | None:
    catalog = _gen_shuffled(_gen_candidate_positions(rows, cols, list(path) + list(distractors)), rng)
    for pos in catalog:
        trial = distractors + [pos]
        if not _gen_walk_is_forced(path, trial):
            continue
        lure_steps = _gen_lure_steps(path, trial, pos)
        blocked_steps = _gen_blocked_steps(path, trial, pos)
        if kind == "lure" and not lure_steps:
            continue
        if kind == "blocked" and not blocked_steps:
            continue
        if kind == "mixed" and not (lure_steps or blocked_steps):
            continue
        meta = {
            "pos": pos,
            "lure_steps": lure_steps,
            "blocked_steps": blocked_steps,
        }
        return pos, meta
    return None


def _gen_assign_colors(path_len: int, distractor_len: int, rng: random.Random) -> tuple[list[int], list[int]]:
    needed = path_len - 1 + distractor_len
    assert needed <= len(TARGET_COLORS), "not enough distinct non-red colors"
    colors = rng.sample(TARGET_COLORS, needed)
    return colors[: path_len - 1], colors[path_len - 1 :]


def _gen_render(rows: int, cols: int, path: list[Coord], distractors: list[Coord], path_colors: list[int], distractor_colors: list[int]) -> np.ndarray:
    grid = np.zeros((rows, cols), dtype=int)
    ordered_positions = [(path[0], Color.RED)]
    ordered_positions.extend(zip(path[1:], path_colors))
    ordered_positions.extend(zip(distractors, distractor_colors))
    for (r, c), color in ordered_positions:
        grid[r : r + BLOCK_SIZE, c : c + BLOCK_SIZE] = int(color)
    return grid


def _gen_validate(
    grid: np.ndarray,
    path: list[Coord],
    distractors: list[Coord],
    required: list[dict[str, object]],
    path_colors: list[int],
) -> bool:
    if not _gen_walk_is_forced(path, distractors):
        return False

    visited, unvisited = _walk_sequence(grid)
    visited_positions = [(int(block["bbox"][0]), int(block["bbox"][1])) for block in visited]
    unvisited_positions = {(int(block["bbox"][0]), int(block["bbox"][1])) for block in unvisited}
    if visited_positions != path:
        return False
    if unvisited_positions != set(distractors):
        return False

    expected = _draw_nested_rings([Color.RED] + list(path_colors))
    if not np.array_equal(main(grid), expected):
        return False

    positions = list(path) + list(distractors)
    if any(_gen_visible_axis(path[-1], pos, positions) is not None for pos in distractors):
        return False

    for meta in required:
        pos = meta["pos"]
        needs_lure = bool(meta["needs_lure"])
        needs_blocked = bool(meta["needs_blocked"])
        lure_steps = _gen_lure_steps(path, distractors, pos)
        blocked_steps = _gen_blocked_steps(path, distractors, pos)
        if needs_lure and not lure_steps:
            return False
        if needs_blocked and not blocked_steps:
            return False
        if not needs_lure and not needs_blocked and not (lure_steps or blocked_steps):
            return False

    total_blocks = len(path) + len(distractors)
    if total_blocks < 4 or total_blocks > 9:
        return False

    out = expected
    sig = (tuple(out.shape), out.tobytes())
    if sig in _SEEN_OUTPUT_SIGS:
        return False
    return True


def _gen_sample_shape(scenario: str, rng: random.Random) -> tuple[int, int, int, int]:
    rows = rng.randint(*GRID_ROWS_RANGE)
    cols = rng.randint(*GRID_COLS_RANGE)
    steps = rng.randint(*PATH_MOVES_RANGE)
    max_turns = steps - 1
    if scenario in {"blocked", "both"}:
        target_turns = rng.randint(1, max_turns)
    elif scenario == "lure":
        target_turns = rng.randint(0, max_turns)
    else:
        target_turns = rng.randint(0, max_turns)
    return rows, cols, steps, target_turns


def _gen_required_kinds(scenario: str, rng: random.Random) -> list[str]:
    if scenario == "lure":
        return ["lure"]
    if scenario == "blocked":
        return ["blocked"]
    if scenario == "both":
        order = ["lure", "blocked"]
        if rng.random() < 0.5:
            order.reverse()
        return order
    return [rng.choice(["lure", "blocked"])]


def _gen_attempt(scenario: str, rng: random.Random) -> np.ndarray | None:
    rows, cols, steps, target_turns = _gen_sample_shape(scenario, rng)
    path = _gen_build_path(rows, cols, steps, target_turns, rng)
    if path is None:
        return None

    required: list[dict[str, object]] = []
    distractors: list[Coord] = []
    for kind in _gen_required_kinds(scenario, rng):
        if len(path) + len(distractors) >= 9:
            return None
        picked = _gen_pick_distractor(rows, cols, path, distractors, kind, rng)
        if picked is None:
            return None
        pos, meta = picked
        distractors.append(pos)
        required.append(
            {
                "pos": pos,
                "needs_lure": kind == "lure",
                "needs_blocked": kind == "blocked",
                "meta": meta,
            }
        )

    if scenario in {"both", "mixed"} and len(path) + len(distractors) < 9 and rng.random() < 0.6:
        extra = _gen_pick_distractor(rows, cols, path, distractors, "mixed", rng)
        if extra is not None:
            pos, meta = extra
            distractors.append(pos)
            required.append(
                {
                    "pos": pos,
                    "needs_lure": False,
                    "needs_blocked": False,
                    "meta": meta,
                }
            )

    path_colors, distractor_colors = _gen_assign_colors(len(path), len(distractors), rng)
    grid = _gen_render(rows, cols, path, distractors, path_colors, distractor_colors)

    if not _gen_validate(grid, path, distractors, required, path_colors):
        return None
    return grid


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT
    scenario = SCENARIOS[_GEN_CALL_COUNT % len(SCENARIOS)]
    _GEN_CALL_COUNT += 1
    rng = random.Random()

    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(scenario, rng)
        if grid is None:
            continue
        out = main(grid)
        _SEEN_OUTPUT_SIGS.add((tuple(out.shape), out.tobytes()))
        return grid
    raise RuntimeError(f"failed to generate a valid input for scenario={scenario}")
