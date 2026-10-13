from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np

ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import Color, find_components, get_bg


START_COLOR = Color.RED
TARGET_COLOR = Color.TEAL
BLOCKER_COLOR = Color.GREY
BACKGROUND = Color.BLACK
PALETTE = (Color.GREEN, Color.YELLOW, Color.BLUE, Color.ORANGE)
BLOCK_SIZE = 2

GRID_ROWS = (15, 21)
GRID_COLS = (15, 21)
TARGET_COUNT_CHOICES = (6, 7, 8)
STEP_COUNT_CHOICES = (4, 5, 6)
STEP_COUNT_WEIGHTS = np.array((2, 4, 3), dtype=float)
GAP_CHOICES = np.array((1, 2, 3, 4, 5), dtype=int)
GAP_WEIGHTS = np.array((2, 4, 4, 3, 2), dtype=float)
GENERATION_ATTEMPTS = 500


Block = dict


def _bbox_h(box: tuple[int, int, int, int]) -> int:
    return int(box[2] - box[0] + 1)


def _bbox_w(box: tuple[int, int, int, int]) -> int:
    return int(box[3] - box[1] + 1)


def _block_key(block: Block) -> tuple[int, int, int, int]:
    return tuple(map(int, block["bbox"]))


def _block_pos(block: Block) -> tuple[int, int]:
    r0, c0, _r1, _c1 = _block_key(block)
    return r0, c0


def _bbox_from_pos(pos: tuple[int, int]) -> tuple[int, int, int, int]:
    r, c = map(int, pos)
    return r, c, r + BLOCK_SIZE - 1, c + BLOCK_SIZE - 1


def _expanded_box(box: tuple[int, int, int, int], pad: int = 1) -> tuple[int, int, int, int]:
    return box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad


def _rect_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _corridor_between_boxes(
    a_box: tuple[int, int, int, int], b_box: tuple[int, int, int, int]
) -> tuple[str, tuple[int, int, int, int], int] | None:
    ar0, ac0, ar1, ac1 = a_box
    br0, bc0, br1, bc1 = b_box

    if ar0 == br0 and ar1 == br1:
        c0 = min(ac1, bc1) + 1
        c1 = max(ac0, bc0) - 1
        gap = max(0, c1 - c0 + 1)
        return "h", (ar0, c0, ar1, c1), gap

    if ac0 == bc0 and ac1 == bc1:
        r0 = min(ar1, br1) + 1
        r1 = max(ar0, br0) - 1
        gap = max(0, r1 - r0 + 1)
        return "v", (r0, ac0, r1, ac1), gap

    return None


def _corridor_cells_from_boxes(
    a_box: tuple[int, int, int, int], b_box: tuple[int, int, int, int]
) -> list[tuple[int, int]]:
    info = _corridor_between_boxes(a_box, b_box)
    if info is None:
        return []
    _axis, rect, gap = info
    if gap == 0:
        return []
    r0, c0, r1, c1 = rect
    return [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]


def _extract_objects(grid: np.ndarray, bg: int) -> tuple[Block, list[Block], list[Block]]:
    objects = find_components(
        grid,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    starts: list[Block] = []
    targets: list[Block] = []
    blockers: list[Block] = []

    for obj in objects:
        color = int(obj["color"])
        box = tuple(map(int, obj["bbox"]))
        area = int(obj["area"])
        if color == START_COLOR:
            assert area == 4 and _bbox_h(box) == 2 and _bbox_w(box) == 2
            starts.append(obj)
        elif color == TARGET_COLOR:
            assert area == 4 and _bbox_h(box) == 2 and _bbox_w(box) == 2
            targets.append(obj)
        elif color == BLOCKER_COLOR:
            assert area == 1, "grey blockers must be isolated pixels"
            blockers.append(obj)
        else:
            raise AssertionError("unexpected non-background color in input")

    assert len(starts) == 1, "expected exactly one red 2x2 start block"
    assert 6 <= len(targets) <= 8, "expected 6-8 teal 2x2 targets"
    assert blockers, "expected at least one grey blocker"
    return starts[0], targets, blockers


def _visible_step(
    grid: np.ndarray, bg: int, a: Block, b: Block
) -> tuple[str, int] | None:
    a_box = _block_key(a)
    b_box = _block_key(b)
    info = _corridor_between_boxes(a_box, b_box)
    if info is None:
        return None

    axis, rect, gap = info
    if gap == 0:
        return axis, 0

    r0, c0, r1, c1 = rect
    if not np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == bg):
        return None
    return axis, gap


def _paint_corridor(
    out: np.ndarray,
    a: Block,
    b: Block,
    axis: str,
    color: int,
) -> None:
    a_box = _block_key(a)
    b_box = _block_key(b)
    info = _corridor_between_boxes(a_box, b_box)
    assert info is not None
    _axis, rect, gap = info
    assert _axis == axis
    if gap == 0:
        return
    r0, c0, r1, c1 = rect
    out[r0 : r1 + 1, c0 : c1 + 1] = int(color)


def _greedy_walk(
    grid: np.ndarray, bg: int, start: Block, targets: list[Block]
) -> tuple[list[tuple[Block, Block, str]], list[Block]]:
    current = start
    unvisited = list(targets)
    steps: list[tuple[Block, Block, str]] = []

    while True:
        candidates: list[tuple[int, str, Block]] = []
        for block in unvisited:
            step = _visible_step(grid, bg, current, block)
            if step is None:
                continue
            axis, gap = step
            candidates.append((gap, axis, block))

        if not candidates:
            break

        nearest_gap = min(gap for gap, _axis, _block in candidates)
        nearest = [item for item in candidates if item[0] == nearest_gap]
        assert len(nearest) == 1, "expected a unique nearest visible target"

        _gap, axis, chosen = nearest[0]
        steps.append((current, chosen, axis))
        unvisited = [block for block in unvisited if block is not chosen]
        current = chosen

    return steps, unvisited


def main(grid: np.ndarray) -> np.ndarray:
    """
    Walk greedily through visible teal 2x2 targets, color each corridor by cyclic step order, and erase the unreached targets.

    Input structure:
    - The grid is a black background with exactly one red 2x2 start block, 6-8 teal 2x2 target blocks, and several isolated grey blocker pixels.
    - Blocks are separated from each other, and visibility is only possible when two 2x2 blocks share the same two-row band or the same two-column band.
    - Any non-black pixel between aligned blocks, including grey blockers or other blocks, blocks that line of sight.

    Rule:
    - Start from the red block and repeatedly choose the unique nearest visible unvisited teal block, where distance is the number of background cells between the two block boundaries.
    - On step k, fill the two-cell-wide corridor between the current block and the chosen target with the cyclic palette [green, yellow, blue, orange], repeating every four steps.
    - Continue from the reached teal block until no visible unvisited target remains, then erase every unreached teal 2x2 target to black while leaving the start block, visited teal blocks, and grey blockers unchanged.

    Key insight:
    - Corridor colors encode traversal step order rather than endpoint identity, so the output can only be recovered by reconstructing the full greedy visible walk and then deleting the unvisited targets.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = int(get_bg(g))
    assert bg == BACKGROUND, "expected a black background"
    start, targets, _blockers = _extract_objects(g, bg)
    steps, unvisited = _greedy_walk(g, bg, start, targets)

    out = g.copy()
    for step_index, (current, chosen, axis) in enumerate(steps):
        color = PALETTE[step_index % len(PALETTE)]
        _paint_corridor(out, current, chosen, axis, color)

    for block in unvisited:
        r0, c0, r1, c1 = _block_key(block)
        out[r0 : r1 + 1, c0 : c1 + 1] = bg

    return out


def _box_fits(pos: tuple[int, int], rows: int, cols: int) -> bool:
    r, c = pos
    return 1 <= r <= rows - BLOCK_SIZE - 1 and 1 <= c <= cols - BLOCK_SIZE - 1


def _block_far_from_blocks(
    pos: tuple[int, int], blocks: list[tuple[int, int]]
) -> bool:
    box = _expanded_box(_bbox_from_pos(pos), pad=1)
    return all(not _rect_overlap(box, _expanded_box(_bbox_from_pos(other), pad=1)) for other in blocks)


def _corridor_clear_of_blocks(
    a: tuple[int, int],
    b: tuple[int, int],
    blocks: list[tuple[int, int]],
) -> bool:
    info = _corridor_between_boxes(_bbox_from_pos(a), _bbox_from_pos(b))
    if info is None:
        return False
    _axis, rect, _gap = info
    for block in blocks:
        if block == a or block == b:
            continue
        if _rect_overlap(rect, _bbox_from_pos(block)):
            return False
    return True


def _corridor_crosses_existing(
    rect: tuple[int, int, int, int], existing: list[tuple[int, int, int, int]]
) -> bool:
    if rect[0] > rect[2] or rect[1] > rect[3]:
        return False
    return any(_rect_overlap(rect, other) for other in existing)


def _build_path_positions(
    rng: np.random.Generator, rows: int, cols: int, step_count: int
) -> list[tuple[int, int]] | None:
    gap_probs = GAP_WEIGHTS / GAP_WEIGHTS.sum()
    for _ in range(120):
        start = (
            int(rng.integers(1, rows - BLOCK_SIZE - 1)),
            int(rng.integers(1, cols - BLOCK_SIZE - 1)),
        )
        positions = [start]
        corridor_rects: list[tuple[int, int, int, int]] = []
        prev_axis: str | None = None
        desired_turns = int(rng.integers(1, min(step_count, 4) + 1))
        turns = 0
        success = True

        for step_index in range(step_count):
            axis_options = ["h", "v"]
            if prev_axis is not None:
                turn_bias = 0.78 if turns < desired_turns else 0.35
                if rng.random() < turn_bias:
                    axis_options = ["v" if prev_axis == "h" else "h", prev_axis]
                else:
                    axis_options = [prev_axis, "v" if prev_axis == "h" else "h"]

            candidates: list[tuple[tuple[int, int], str, tuple[int, int, int, int]]] = []
            for axis in axis_options:
                gaps = rng.choice(GAP_CHOICES, size=len(GAP_CHOICES), replace=False, p=gap_probs)
                signs = rng.permutation(np.array((-1, 1), dtype=int))
                for gap in gaps:
                    for sign in signs:
                        cr, cc = positions[-1]
                        if axis == "h":
                            nxt = (cr, int(cc + sign * (gap + BLOCK_SIZE)))
                        else:
                            nxt = (int(cr + sign * (gap + BLOCK_SIZE)), cc)

                        if not _box_fits(nxt, rows, cols):
                            continue
                        if not _block_far_from_blocks(nxt, positions):
                            continue
                        if not _corridor_clear_of_blocks(positions[-1], nxt, positions + [nxt]):
                            continue

                        info = _corridor_between_boxes(_bbox_from_pos(positions[-1]), _bbox_from_pos(nxt))
                        assert info is not None
                        _cand_axis, rect, _cand_gap = info
                        if _corridor_crosses_existing(rect, corridor_rects):
                            continue
                        candidates.append((nxt, axis, rect))

            if not candidates:
                success = False
                break

            choice_index = int(rng.integers(len(candidates)))
            nxt, axis, rect = candidates[choice_index]
            if prev_axis is not None and axis != prev_axis:
                turns += 1
            positions.append(nxt)
            corridor_rects.append(rect)
            prev_axis = axis

        if success and len(positions) == step_count + 1:
            return positions

    return None


def _render_input_grid(
    rows: int,
    cols: int,
    path_positions: list[tuple[int, int]],
    blocked_target: tuple[int, int],
    blockers: list[tuple[int, int]],
    extra_targets: list[tuple[int, int]],
) -> np.ndarray:
    grid = np.zeros((rows, cols), dtype=int)

    for index, pos in enumerate(path_positions):
        r, c = pos
        color = START_COLOR if index == 0 else TARGET_COLOR
        grid[r : r + BLOCK_SIZE, c : c + BLOCK_SIZE] = color

    for r, c in [blocked_target, *extra_targets]:
        grid[r : r + BLOCK_SIZE, c : c + BLOCK_SIZE] = TARGET_COLOR

    for r, c in blockers:
        grid[r, c] = BLOCKER_COLOR

    return grid


def _place_blocked_target(
    rng: np.random.Generator,
    rows: int,
    cols: int,
    path_positions: list[tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    row_counts = Counter(r for r, _c in path_positions)
    col_counts = Counter(c for _r, c in path_positions)

    options: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]] = []
    for anchor in path_positions:
        ar, ac = anchor

        if row_counts[ar] == 1:
            for sign in (-1, 1):
                for gap in (3, 4, 5):
                    target = (ar, ac + sign * (gap + BLOCK_SIZE))
                    if not _box_fits(target, rows, cols):
                        continue
                    if not _block_far_from_blocks(target, path_positions):
                        continue
                    info = _corridor_between_boxes(_bbox_from_pos(anchor), _bbox_from_pos(target))
                    assert info is not None
                    _axis, _rect, _gap = info
                    cells = _corridor_cells_from_boxes(_bbox_from_pos(anchor), _bbox_from_pos(target))
                    if len(cells) < 3:
                        continue
                    blocker = cells[len(cells) // 2]
                    options.append((anchor, target, blocker))

        if col_counts[ac] == 1:
            for sign in (-1, 1):
                for gap in (3, 4, 5):
                    target = (ar + sign * (gap + BLOCK_SIZE), ac)
                    if not _box_fits(target, rows, cols):
                        continue
                    if not _block_far_from_blocks(target, path_positions):
                        continue
                    info = _corridor_between_boxes(_bbox_from_pos(anchor), _bbox_from_pos(target))
                    assert info is not None
                    _axis, _rect, _gap = info
                    cells = _corridor_cells_from_boxes(_bbox_from_pos(anchor), _bbox_from_pos(target))
                    if len(cells) < 3:
                        continue
                    blocker = cells[len(cells) // 2]
                    options.append((anchor, target, blocker))

    if not options:
        return None

    order = rng.permutation(len(options))
    for idx in order:
        _anchor, target, blocker = options[int(idx)]
        br, bc = blocker
        blocker_box = (br, bc, br, bc)
        blocked_box = _expanded_box(blocker_box, pad=1)
        occupied = [_expanded_box(_bbox_from_pos(pos), pad=1) for pos in path_positions + [target]]
        if any(_rect_overlap(blocked_box, box) for box in occupied):
            continue
        return target, blocker

    return None


def _pick_extra_target_positions(
    rng: np.random.Generator,
    rows: int,
    cols: int,
    count: int,
    occupied_blocks: list[tuple[int, int]],
    blocker_pixels: list[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    blocked_rows = {r for r, _c in occupied_blocks}
    blocked_cols = {c for _r, c in occupied_blocks}
    blocker_boxes = [_expanded_box((r, c, r, c), pad=1) for r, c in blocker_pixels]
    extras: list[tuple[int, int]] = []

    candidates: list[tuple[int, int]] = []
    for r in range(1, rows - BLOCK_SIZE):
        for c in range(1, cols - BLOCK_SIZE):
            pos = (r, c)
            if r in blocked_rows or c in blocked_cols:
                continue
            if not _block_far_from_blocks(pos, occupied_blocks + extras):
                continue
            if any(_rect_overlap(_expanded_box(_bbox_from_pos(pos), pad=1), box) for box in blocker_boxes):
                continue
            candidates.append(pos)

    if len(candidates) < count:
        return None

    order = rng.permutation(len(candidates))
    for idx in order:
        pos = candidates[int(idx)]
        if not _block_far_from_blocks(pos, occupied_blocks + extras):
            continue
        if any(_rect_overlap(_expanded_box(_bbox_from_pos(pos), pad=1), box) for box in blocker_boxes):
            continue
        extras.append(pos)
        if len(extras) == count:
            return extras

    return None


def _place_noise_blockers(
    rng: np.random.Generator,
    rows: int,
    cols: int,
    occupied_blocks: list[tuple[int, int]],
    blockers: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    occupied_rects = [_expanded_box(_bbox_from_pos(pos), pad=1) for pos in occupied_blocks]
    out = list(blockers)
    target_noise = int(rng.integers(1, 4))
    for _ in range(target_noise):
        for _ in range(80):
            r = int(rng.integers(1, rows - 1))
            c = int(rng.integers(1, cols - 1))
            rect = _expanded_box((r, c, r, c), pad=1)
            if any(_rect_overlap(rect, occ) for occ in occupied_rects):
                continue
            if any(max(abs(r - rr), abs(c - cc)) <= 1 for rr, cc in out):
                continue
            out.append((r, c))
            break
    return out


def _oracle_output(
    grid: np.ndarray,
) -> tuple[np.ndarray, list[tuple[tuple[int, int], tuple[int, int]]], list[tuple[int, int]], list[tuple[int, int]]]:
    bg = int(get_bg(grid))
    start, targets, blockers = _extract_objects(grid, bg)
    steps, unvisited = _greedy_walk(grid, bg, start, targets)
    out = grid.copy()

    visited_pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for step_index, (current, chosen, axis) in enumerate(steps):
        _paint_corridor(out, current, chosen, axis, PALETTE[step_index % len(PALETTE)])
        visited_pairs.append((_block_pos(current), _block_pos(chosen)))

    unvisited_positions: list[tuple[int, int]] = []
    for block in unvisited:
        r0, c0, r1, c1 = _block_key(block)
        unvisited_positions.append((r0, c0))
        out[r0 : r1 + 1, c0 : c1 + 1] = bg

    blocker_positions = [tuple(map(int, block["coords"][0])) for block in blockers]
    return out, visited_pairs, unvisited_positions, blocker_positions


def _has_blocked_aligned_target(
    grid: np.ndarray,
    visited_positions: set[tuple[int, int]],
    unvisited_positions: list[tuple[int, int]],
    blocker_positions: set[tuple[int, int]],
) -> bool:
    for unvisited in unvisited_positions:
        u_box = _bbox_from_pos(unvisited)
        for visited in visited_positions:
            v_box = _bbox_from_pos(visited)
            info = _corridor_between_boxes(v_box, u_box)
            if info is None:
                continue
            _axis, _rect, gap = info
            if gap <= 0:
                continue
            cells = set(_corridor_cells_from_boxes(v_box, u_box))
            if cells & blocker_positions:
                return True
    return False


def _valid_input(grid: np.ndarray, expected_target_count: int) -> bool:
    try:
        oracle_out, visited_pairs, unvisited_positions, blocker_positions = _oracle_output(grid)
    except Exception:
        return False

    if len(unvisited_positions) < 1:
        return False

    start, targets, blockers = _extract_objects(grid, BACKGROUND)
    if len(targets) != expected_target_count:
        return False

    if any(int(block["area"]) != 1 for block in blockers):
        return False

    visited_positions = {tuple(_block_pos(start))}
    for _src, dst in visited_pairs:
        visited_positions.add(dst)

    if len(visited_pairs) < 4:
        return False

    if not _has_blocked_aligned_target(
        grid,
        visited_positions,
        unvisited_positions,
        set(blocker_positions),
    ):
        return False

    solved = main(grid)
    if not np.array_equal(solved, oracle_out):
        return False

    for pos in unvisited_positions:
        r, c = pos
        if np.any(solved[r : r + BLOCK_SIZE, c : c + BLOCK_SIZE] != BACKGROUND):
            return False

    for step_index, (src, dst) in enumerate(visited_pairs, start=1):
        cells = _corridor_cells_from_boxes(_bbox_from_pos(src), _bbox_from_pos(dst))
        expected_color = PALETTE[(step_index - 1) % len(PALETTE)]
        if cells and any(int(solved[r, c]) != int(expected_color) for r, c in cells):
            return False

    return True


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    step_probs = STEP_COUNT_WEIGHTS / STEP_COUNT_WEIGHTS.sum()

    for _attempt in range(GENERATION_ATTEMPTS):
        rows = int(rng.integers(*GRID_ROWS))
        cols = int(rng.integers(*GRID_COLS))
        target_count = int(rng.choice(TARGET_COUNT_CHOICES))
        step_count = int(rng.choice(STEP_COUNT_CHOICES, p=step_probs))
        step_count = min(step_count, target_count - 1)

        path_positions = _build_path_positions(rng, rows, cols, step_count)
        if path_positions is None:
            continue

        blocked = _place_blocked_target(rng, rows, cols, path_positions)
        if blocked is None:
            continue
        blocked_target, forced_blocker = blocked

        remaining_unvisited = target_count - step_count - 1
        if remaining_unvisited < 0:
            continue

        occupied = list(path_positions) + [blocked_target]
        extras = _pick_extra_target_positions(
            rng,
            rows,
            cols,
            remaining_unvisited,
            occupied,
            [forced_blocker],
        )
        if extras is None:
            continue

        blockers = _place_noise_blockers(
            rng,
            rows,
            cols,
            occupied + extras,
            [forced_blocker],
        )
        grid = _render_input_grid(rows, cols, path_positions, blocked_target, blockers, extras)

        if _valid_input(grid, target_count):
            return grid

    raise RuntimeError("failed to generate a valid greedy corridor input")
