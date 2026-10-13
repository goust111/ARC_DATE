from __future__ import annotations

import random
from collections import Counter

import numpy as np


Coord = tuple[int, int]
DIRS: list[Coord] = [(-1, 0), (0, 1), (1, 0), (0, -1)]
BG = 0
COLOR_POOL = list(range(1, 10))
GRID_H_RANGE = (14, 22)
GRID_W_RANGE = (14, 22)
SOURCE_SIZE_RANGE = (5, 9)
DEST_COUNT_RANGE = (2, 4)
GEN_RETRIES = 500


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int) -> list[Coord]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _rotate_offset(dr: int, dc: int, turns: int) -> Coord:
    rr, cc = int(dr), int(dc)
    for _ in range(turns % 4):
        rr, cc = -cc, rr
    return rr, cc


def _rotation_steps(src_dir: Coord, dst_dir: Coord) -> int:
    return (DIRS.index(dst_dir) - DIRS.index(src_dir)) % 4


def _is_connected(cells: set[Coord]) -> bool:
    if not cells:
        return False
    seen = set()
    stack = [next(iter(cells))]
    while stack:
        cell = stack.pop()
        if cell in seen:
            continue
        seen.add(cell)
        for nbr in _neighbors4(*cell):
            if nbr in cells and nbr not in seen:
                stack.append(nbr)
    return seen == cells


def _expanded(cells: set[Coord], shape: tuple[int, int], pad: int = 1) -> set[Coord]:
    out = set(cells)
    frontier = set(cells)
    for _ in range(pad):
        nxt = set()
        for r, c in frontier:
            for rr in range(r - 1, r + 2):
                for cc in range(c - 1, c + 2):
                    if _in_bounds(shape, rr, cc) and (rr, cc) not in out:
                        nxt.add((rr, cc))
        out |= nxt
        frontier = nxt
    return out


def _translate(cells: set[Coord], anchor: Coord) -> set[Coord]:
    ar, ac = anchor
    return {(ar + dr, ac + dc) for dr, dc in cells}


def _pointer_cells(head: Coord, direction: Coord) -> set[Coord]:
    hr, hc = head
    dr, dc = direction
    return {(hr, hc), (hr - dr, hc - dc)}


def _rotated_source_cells(offsets: set[Coord], head: Coord, turns: int) -> set[Coord]:
    hr, hc = head
    return {(hr + rr, hc + cc) for rr, cc in (_rotate_offset(dr, dc, turns) for dr, dc in offsets)}


def _pair_pointers(heads: set[Coord], tails: set[Coord]) -> dict[Coord, Coord] | None:
    head_to_tail: dict[Coord, Coord] = {}
    tail_to_head: dict[Coord, Coord] = {}
    for head in heads:
        nbrs = [nbr for nbr in _neighbors4(*head) if nbr in tails]
        if len(nbrs) != 1:
            return None
        head_to_tail[head] = nbrs[0]
    for tail in tails:
        nbrs = [nbr for nbr in _neighbors4(*tail) if nbr in heads]
        if len(nbrs) != 1:
            return None
        tail_to_head[tail] = nbrs[0]
    for head, tail in head_to_tail.items():
        if tail_to_head.get(tail) != head:
            return None
    return head_to_tail


def _infer_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    colors = [int(c) for c in np.unique(g) if int(c) != bg]
    assert len(colors) == 3, "expected exactly three non-background colors"

    candidates = []
    for head_color in colors:
        for tail_color in colors:
            if tail_color == head_color:
                continue
            source_color = next(c for c in colors if c not in (head_color, tail_color))
            heads = {tuple(map(int, rc)) for rc in np.argwhere(g == head_color)}
            tails = {tuple(map(int, rc)) for rc in np.argwhere(g == tail_color)}
            source = {tuple(map(int, rc)) for rc in np.argwhere(g == source_color)}
            if len(heads) < 3 or len(heads) != len(tails) or len(source) < 4:
                continue
            if not _is_connected(source):
                continue

            pairing = _pair_pointers(heads, tails)
            if pairing is None:
                continue

            anchor_heads = []
            for head in heads:
                if any(nbr in source for nbr in _neighbors4(*head)):
                    anchor_heads.append(head)
            if len(anchor_heads) != 1:
                continue
            if any(any(nbr in source for nbr in _neighbors4(*tail)) for tail in tails):
                continue

            anchor_head = anchor_heads[0]
            anchor_tail = pairing[anchor_head]
            anchor_dir = (anchor_head[0] - anchor_tail[0], anchor_head[1] - anchor_tail[1])
            if anchor_dir not in DIRS:
                continue

            dest_heads = sorted(head for head in heads if head != anchor_head)
            if not dest_heads:
                continue

            offsets = {(r - anchor_head[0], c - anchor_head[1]) for r, c in source}
            fits = True
            dest_dirs: dict[Coord, Coord] = {}
            for head in dest_heads:
                tail = pairing[head]
                direction = (head[0] - tail[0], head[1] - tail[1])
                if direction not in DIRS:
                    fits = False
                    break
                dest_dirs[head] = direction
                turns = _rotation_steps(anchor_dir, direction)
                for rr, cc in (_rotate_offset(dr, dc, turns) for dr, dc in offsets):
                    if not _in_bounds(g.shape, head[0] + rr, head[1] + cc):
                        fits = False
                        break
                if not fits:
                    break
            if not fits:
                continue

            candidates.append(
                {
                    "bg": bg,
                    "source_color": source_color,
                    "head_color": head_color,
                    "tail_color": tail_color,
                    "source": source,
                    "offsets": offsets,
                    "anchor_head": anchor_head,
                    "anchor_tail": anchor_tail,
                    "anchor_dir": anchor_dir,
                    "pairing": pairing,
                    "dest_heads": dest_heads,
                    "dest_dirs": dest_dirs,
                }
            )

    assert len(candidates) == 1, "scene roles must be uniquely inferable"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp the source shape at every destination pointer, rotating by the pointer-direction change from the anchor pointer.

    Input structure:
    - A background grid contains one 4-connected source shape in one color and several two-pixel pointers in two other colors.
    - Each pointer is a head pixel plus a 4-neighbor tail pixel; the tail-to-head direction is the pointer orientation.
    - Exactly one pointer head is 4-adjacent to the source shape; it is the anchor pointer, and the other pointers are destinations.

    Rule:
    - Record every source pixel as an offset from the anchor pointer head.
    - For each destination pointer, rotate that offset set by the 90-degree multiple that turns the anchor orientation into the destination orientation.
    - Clear the original source shape and every pointer pixel, then redraw the rotated source shape in the source color around each destination head.

    Key insight:
    - Each pointer defines a full local coordinate frame, so the destination copy depends on both the head position and the orientation difference, not on translation alone.
    """
    g = np.asarray(grid, dtype=int)
    scene = _infer_scene(g)

    out = np.full_like(g, scene["bg"])
    for head in scene["dest_heads"]:
        turns = _rotation_steps(scene["anchor_dir"], scene["dest_dirs"][head])
        for rr, cc in (_rotate_offset(dr, dc, turns) for dr, dc in scene["offsets"]):
            out[head[0] + rr, head[1] + cc] = scene["source_color"]
    return out


def _rotated_offset_set(offsets: set[Coord], turns: int) -> set[Coord]:
    return {_rotate_offset(dr, dc, turns) for dr, dc in offsets}


def _generate_source_offsets(anchor_dir: Coord) -> set[Coord] | None:
    tail = (-anchor_dir[0], -anchor_dir[1])
    target_size = random.randint(*SOURCE_SIZE_RANGE)
    allowed = {
        (r, c)
        for r in range(-2, 3)
        for c in range(-2, 3)
        if (r, c) != (0, 0)
        and (r, c) != tail
        and abs(r - tail[0]) + abs(c - tail[1]) != 1
    }
    contact_choices = [cell for cell in DIRS if cell in allowed]
    if len(contact_choices) < 2:
        return None

    for _ in range(80):
        size = target_size
        offsets = set(random.sample(contact_choices, k=random.choice([2, 2, 3])))
        while len(offsets) < size:
            frontier = [
                cell
                for cell in allowed
                if cell not in offsets
                and any(abs(cell[0] - rr) + abs(cell[1] - cc) == 1 for rr, cc in offsets)
            ]
            if not frontier:
                break
            offsets.add(random.choice(frontier))
        rows = [r for r, _ in offsets]
        cols = [c for _, c in offsets]
        if len(offsets) != size:
            continue
        if not _is_connected(offsets):
            continue
        if max(rows) - min(rows) < 2 or max(cols) - min(cols) < 2:
            continue
        if any(_rotated_offset_set(offsets, k) == offsets for k in (1, 2, 3)):
            continue
        if not any(abs(r) == 2 or abs(c) == 2 for r, c in offsets):
            continue
        return offsets
    return None


def _choose_dest_dirs(anchor_dir: Coord, count: int) -> list[Coord]:
    while True:
        dirs = [random.choice(DIRS) for _ in range(count)]
        if all(d == anchor_dir for d in dirs):
            continue
        if len(set(dirs)) < 2:
            continue
        return dirs


def _anchor_candidates(shape: tuple[int, int], offsets: set[Coord], anchor_dir: Coord) -> list[Coord]:
    H, W = shape
    tail = (-anchor_dir[0], -anchor_dir[1])
    candidates = []
    for r in range(H):
        for c in range(W):
            cells = {(r + dr, c + dc) for dr, dc in offsets} | {(r, c), (r + tail[0], c + tail[1])}
            if all(_in_bounds(shape, rr, cc) for rr, cc in cells):
                candidates.append((r, c))
    random.shuffle(candidates)
    return candidates


def _dest_candidates(
    shape: tuple[int, int],
    offsets: set[Coord],
    anchor_dir: Coord,
    dest_dir: Coord,
    blocked_input: set[Coord],
    blocked_output: set[Coord],
) -> list[Coord]:
    H, W = shape
    tail = (-dest_dir[0], -dest_dir[1])
    turns = _rotation_steps(anchor_dir, dest_dir)
    candidates = []
    for r in range(H):
        for c in range(W):
            pointer = {(r, c), (r + tail[0], c + tail[1])}
            if not all(_in_bounds(shape, rr, cc) for rr, cc in pointer):
                continue
            if pointer & blocked_input:
                continue
            stamp = _rotated_source_cells(offsets, (r, c), turns)
            if stamp & blocked_output:
                continue
            if not all(_in_bounds(shape, rr, cc) for rr, cc in stamp):
                continue
            candidates.append((r, c))
    random.shuffle(candidates)
    return candidates


def _build_expected(
    shape: tuple[int, int],
    bg: int,
    source_color: int,
    offsets: set[Coord],
    anchor_dir: Coord,
    dest_heads: list[Coord],
    dest_dirs: list[Coord],
) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    for head, direction in zip(dest_heads, dest_dirs):
        turns = _rotation_steps(anchor_dir, direction)
        for rr, cc in _rotated_source_cells(offsets, head, turns):
            out[rr, cc] = source_color
    return out


def _build_input(
    shape: tuple[int, int],
    source_color: int,
    head_color: int,
    tail_color: int,
    offsets: set[Coord],
    anchor_head: Coord,
    anchor_dir: Coord,
    dest_heads: list[Coord],
    dest_dirs: list[Coord],
) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for rr, cc in _translate(offsets, anchor_head):
        grid[rr, cc] = source_color

    all_heads = [anchor_head, *dest_heads]
    all_dirs = [anchor_dir, *dest_dirs]
    for head, direction in zip(all_heads, all_dirs):
        hr, hc = head
        tr, tc = hr - direction[0], hc - direction[1]
        grid[hr, hc] = head_color
        grid[tr, tc] = tail_color
    return grid


def _naive_translation_output(grid: np.ndarray) -> np.ndarray | None:
    scene = _infer_scene(grid)
    out = np.full_like(grid, scene["bg"])
    for head in scene["dest_heads"]:
        for dr, dc in scene["offsets"]:
            rr = head[0] + dr
            cc = head[1] + dc
            if not _in_bounds(grid.shape, rr, cc):
                return None
            out[rr, cc] = scene["source_color"]
    return out


def _validate_generated(grid: np.ndarray, expected: np.ndarray) -> bool:
    try:
        scene = _infer_scene(grid)
        out = main(grid)
    except AssertionError:
        return False

    if not np.array_equal(out, expected):
        return False
    if len(scene["dest_heads"]) < 2:
        return False
    if any(_rotated_offset_set(scene["offsets"], k) == scene["offsets"] for k in (1, 2, 3)):
        return False

    turns = [_rotation_steps(scene["anchor_dir"], scene["dest_dirs"][head]) for head in scene["dest_heads"]]
    if not any(t != 0 for t in turns):
        return False
    if len(set(turns)) < 2:
        return False

    actual_stamps = [
        _rotated_source_cells(scene["offsets"], head, _rotation_steps(scene["anchor_dir"], scene["dest_dirs"][head]))
        for head in scene["dest_heads"]
    ]
    if any(a & b for i, a in enumerate(actual_stamps) for b in actual_stamps[i + 1 :]):
        return False

    naive = _naive_translation_output(grid)
    if naive is not None and np.array_equal(out, naive):
        return False

    counts = Counter(int(v) for v in grid.flat if int(v) != BG)
    if len(counts) != 3:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)
    source_color, head_color, tail_color = random.sample(COLOR_POOL, 3)
    anchor_dir = random.choice(DIRS)
    offsets = _generate_source_offsets(anchor_dir)
    if offsets is None:
        return None

    anchor_heads = _anchor_candidates((H, W), offsets, anchor_dir)
    if not anchor_heads:
        return None
    anchor_head = random.choice(anchor_heads)

    anchor_cells = _translate(offsets, anchor_head) | _pointer_cells(anchor_head, anchor_dir)
    blocked_input = _expanded(anchor_cells, (H, W), pad=1)
    blocked_output: set[Coord] = set()

    dest_count = random.randint(*DEST_COUNT_RANGE)
    dest_dirs = _choose_dest_dirs(anchor_dir, dest_count)
    order = list(range(dest_count))
    random.shuffle(order)
    ordered_dirs = [dest_dirs[i] for i in order]

    ordered_heads: list[Coord] = []
    for direction in ordered_dirs:
        candidates = _dest_candidates((H, W), offsets, anchor_dir, direction, blocked_input, blocked_output)
        if not candidates:
            return None
        head = random.choice(candidates)
        ordered_heads.append(head)
        blocked_input |= _expanded(_pointer_cells(head, direction), (H, W), pad=1)
        blocked_output |= _rotated_source_cells(offsets, head, _rotation_steps(anchor_dir, direction))

    dest_heads = [None] * dest_count
    for idx, head in zip(order, ordered_heads):
        dest_heads[idx] = head

    grid = _build_input(
        (H, W),
        source_color,
        head_color,
        tail_color,
        offsets,
        anchor_head,
        anchor_dir,
        dest_heads,
        dest_dirs,
    )
    expected = _build_expected((H, W), BG, source_color, offsets, anchor_dir, dest_heads, dest_dirs)
    if _validate_generated(grid, expected):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid pointer-stamp input")
