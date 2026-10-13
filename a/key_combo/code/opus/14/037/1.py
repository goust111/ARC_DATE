from __future__ import annotations

import random
from collections import Counter

import numpy as np


DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

TEMPLATES = {
    DIRS["right"]: [
        ([(0, 0), (0, 1), (1, 0), (1, 1), (0, 2)], (0, 2)),
        ([(0, 0), (0, 1), (1, 0), (1, 1), (1, 2)], (1, 2)),
    ],
    DIRS["left"]: [
        ([(0, 1), (0, 2), (1, 1), (1, 2), (0, 0)], (0, 0)),
        ([(0, 1), (0, 2), (1, 1), (1, 2), (1, 0)], (1, 0)),
    ],
    DIRS["down"]: [
        ([(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)], (2, 0)),
        ([(0, 0), (0, 1), (1, 0), (1, 1), (2, 1)], (2, 1)),
    ],
    DIRS["up"]: [
        ([(1, 0), (1, 1), (2, 0), (2, 1), (0, 0)], (0, 0)),
        ([(1, 0), (1, 1), (2, 0), (2, 1), (0, 1)], (0, 1)),
    ],
}

SIZE_CHOICES = {
    2: [(1, 1)],
    3: [(2, 1)],
    4: [(3, 1), (2, 2)],
    5: [(3, 2), (2, 2, 1)],
}


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _is_connected(coords_set: set[tuple[int, int]]) -> bool:
    if not coords_set:
        return False
    start = next(iter(coords_set))
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for nbr in _neighbors4(r, c):
            if nbr in coords_set and nbr not in seen:
                seen.add(nbr)
                stack.append(nbr)
    return len(seen) == len(coords_set)


def _extract_objects(grid: np.ndarray) -> tuple[int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    objects: list[dict] = []
    for comp in _components(g != bg):
        colors = {int(g[r, c]) for r, c in comp}
        assert len(colors) == 1, "each sprite must be monochromatic"
        color = colors.pop()
        coords_set = {tuple(map(int, rc)) for rc in comp}
        degrees = {
            cell: sum((nbr in coords_set) for nbr in _neighbors4(cell[0], cell[1]))
            for cell in coords_set
        }

        candidates: list[tuple[tuple[int, int], tuple[int, int]]] = []
        for cell in coords_set:
            if degrees[cell] != 1:
                continue
            nbr = next(n for n in _neighbors4(cell[0], cell[1]) if n in coords_set)
            if degrees[nbr] < 3:
                continue
            remainder = coords_set - {cell}
            if _is_connected(remainder):
                candidates.append((cell, nbr))

        assert len(candidates) == 1, "each sprite must have one unambiguous protrusion"
        leaf, nbr = candidates[0]
        direction = (leaf[0] - nbr[0], leaf[1] - nbr[1])
        assert direction in TEMPLATES, "protrusion must encode a cardinal direction"
        coords = np.array(sorted(coords_set), dtype=int)
        objects.append(
            {
                "color": int(color),
                "coords": coords,
                "leaf": (int(leaf[0]), int(leaf[1])),
                "direction": direction,
            }
        )

    objects.sort(key=lambda obj: obj["color"])
    return bg, objects


def _copy_objects(objects: list[dict]) -> list[dict]:
    return [
        {
            "color": int(obj["color"]),
            "coords": obj["coords"].copy(),
            "leaf": tuple(obj["leaf"]),
            "direction": tuple(obj["direction"]),
        }
        for obj in objects
    ]


def _in_bounds(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = shape
    return bool(
        np.all((0 <= coords[:, 0]) & (coords[:, 0] < h) & (0 <= coords[:, 1]) & (coords[:, 1] < w))
    )


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _simulate(objects: list[dict], shape: tuple[int, int]) -> tuple[list[dict], dict]:
    state = _copy_objects(objects)
    n = len(state)
    move_counts = [0] * n
    limit = shape[0] * shape[1]

    for _ in range(limit):
        proposals: dict[int, np.ndarray] = {}
        movable = set()
        for idx, obj in enumerate(state):
            dr, dc = obj["direction"]
            proposed = obj["coords"] + np.array([[dr, dc]])
            if _in_bounds(proposed, shape):
                proposals[idx] = proposed
                movable.add(idx)

        changed = True
        while changed:
            changed = False
            non_movers = set(range(n)) - movable
            blocked_cells = {
                tuple(map(int, cell))
                for idx in non_movers
                for cell in state[idx]["coords"]
            }
            for idx in list(movable):
                if any(tuple(map(int, cell)) in blocked_cells for cell in proposals[idx]):
                    movable.remove(idx)
                    changed = True

            if changed:
                continue

            cell_to_idxs: dict[tuple[int, int], list[int]] = {}
            for idx in movable:
                for cell in proposals[idx]:
                    cell_to_idxs.setdefault(tuple(map(int, cell)), []).append(idx)
            conflicts = {idx for ids in cell_to_idxs.values() if len(ids) > 1 for idx in ids}
            if conflicts:
                movable -= conflicts
                changed = True
                continue

            current_sets = {idx: {tuple(map(int, cell)) for cell in state[idx]["coords"]} for idx in movable}
            proposal_sets = {idx: {tuple(map(int, cell)) for cell in proposals[idx]} for idx in movable}
            swaps = set()
            movable_list = sorted(movable)
            for i, idx in enumerate(movable_list):
                di = state[idx]["direction"]
                for jdx in movable_list[i + 1 :]:
                    dj = state[jdx]["direction"]
                    if di != (-dj[0], -dj[1]):
                        continue
                    if proposal_sets[idx] & current_sets[jdx] and proposal_sets[jdx] & current_sets[idx]:
                        swaps.add(idx)
                        swaps.add(jdx)
            if swaps:
                movable -= swaps
                changed = True

        if not movable:
            break

        for idx in movable:
            dr, dc = state[idx]["direction"]
            state[idx]["coords"] = proposals[idx]
            state[idx]["leaf"] = (state[idx]["leaf"][0] + dr, state[idx]["leaf"][1] + dc)
            move_counts[idx] += 1
    else:
        raise AssertionError("movement did not settle")

    return state, {"move_counts": move_counts}


def _render(objects: list[dict], shape: tuple[int, int], bg: int, erase_leaf: bool) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    for obj in objects:
        leaf = tuple(obj["leaf"])
        for r, c in obj["coords"]:
            cell = (int(r), int(c))
            if erase_leaf and cell == leaf:
                continue
            assert out[cell] == bg, "sprites must not overlap"
            out[cell] = int(obj["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each sprite in the direction encoded by its leaf protrusion, then erase the protrusion.

    Input structure:
    - A black background contains 3-5 separated monochromatic foreground sprites.
    - Each sprite is 4-connected and contains exactly one single-pixel leaf protrusion.
    - The protrusion has exactly one same-color orthogonal neighbor, and that neighbor lies inside the sprite body.

    Rule:
    - For each sprite, locate its unique leaf protrusion and read the direction from the protrusion relative to its only same-color neighbor.
    - Move all sprites simultaneously one cell at a time in their own encoded directions; a sprite stops when its next rigid translation would leave the grid or collide with a sprite that is already stopped or cannot move in that step.
    - Repeat until every sprite is stopped, then erase the protrusion pixel from each settled sprite and keep the remaining body.

    Key insight:
    - The leaf pixel is both a structural endpoint and the movement instruction, so the task is object-wise directional gravity rather than a single global fall direction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    bg, objects = _extract_objects(g)
    final_objects, _stats = _simulate(objects, g.shape)
    return _render(final_objects, g.shape, bg, erase_leaf=True)


def _make_template(direction: tuple[int, int]) -> tuple[np.ndarray, tuple[int, int]]:
    coords, leaf = random.choice(TEMPLATES[direction])
    return np.array(coords, dtype=int), tuple(leaf)


def _build_chain_local(direction: tuple[int, int], size: int) -> list[dict]:
    objects = []
    side_offsets = [0]
    for _ in range(1, size):
        side_offsets.append(side_offsets[-1] + random.choice((-1, 0, 1)))

    for idx in range(size):
        coords, leaf = _make_template(direction)
        if direction == DIRS["right"]:
            coords += np.array([[side_offsets[idx], 3 * (size - 1 - idx)]])
            leaf = (leaf[0] + side_offsets[idx], leaf[1] + 3 * (size - 1 - idx))
        elif direction == DIRS["left"]:
            coords += np.array([[side_offsets[idx], 3 * idx]])
            leaf = (leaf[0] + side_offsets[idx], leaf[1] + 3 * idx)
        elif direction == DIRS["down"]:
            coords += np.array([[3 * (size - 1 - idx), side_offsets[idx]]])
            leaf = (leaf[0] + 3 * (size - 1 - idx), leaf[1] + side_offsets[idx])
        else:
            coords += np.array([[3 * idx, side_offsets[idx]]])
            leaf = (leaf[0] + 3 * idx, leaf[1] + side_offsets[idx])
        objects.append({"coords": coords, "leaf": leaf, "direction": direction})

    min_r = min(int(obj["coords"][:, 0].min()) for obj in objects)
    min_c = min(int(obj["coords"][:, 1].min()) for obj in objects)
    if min_r or min_c:
        for obj in objects:
            obj["coords"] = obj["coords"] - np.array([[min_r, min_c]])
            obj["leaf"] = (obj["leaf"][0] - min_r, obj["leaf"][1] - min_c)
    return objects


def _sample_reverse_shifts(size: int, capacity: int) -> list[int] | None:
    if capacity < size:
        return None
    shifts = []
    low = 1
    for idx in range(size):
        high = capacity - (size - 1 - idx)
        if low > high:
            return None
        shift = random.randint(low, high)
        shifts.append(shift)
        low = shift + 1
    return shifts


def _translate_chain_for_input(chain: list[dict], shifts: list[int]) -> list[dict]:
    moved = []
    for obj, shift in zip(chain, shifts):
        dr, dc = obj["direction"]
        delta = np.array([[-dr * shift, -dc * shift]])
        moved.append(
            {
                "coords": obj["coords"] + delta,
                "leaf": (obj["leaf"][0] - dr * shift, obj["leaf"][1] - dc * shift),
                "direction": obj["direction"],
            }
        )
    return moved


def _chain_swept_bbox(input_chain: list[dict], final_chain: list[dict]) -> tuple[int, int, int, int]:
    coords = np.concatenate(
        [obj["coords"] for obj in input_chain] + [obj["coords"] for obj in final_chain], axis=0
    )
    return _bbox(coords)


def _expand_bbox(box: tuple[int, int, int, int], margin: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = box
    return r0 - margin, c0 - margin, r1 + margin, c1 + margin


def _boxes_disjoint(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0


def _shift_chain(chain: list[dict], dr: int, dc: int, color: int) -> list[dict]:
    out = []
    delta = np.array([[dr, dc]])
    for obj in chain:
        out.append(
            {
                "color": color,
                "coords": obj["coords"] + delta,
                "leaf": (obj["leaf"][0] + dr, obj["leaf"][1] + dc),
                "direction": obj["direction"],
            }
        )
    return out


def _settled_offset(
    final_chain: list[dict], direction: tuple[int, int], shape: tuple[int, int]
) -> tuple[int, int]:
    coords = np.concatenate([obj["coords"] for obj in final_chain], axis=0)
    r0, c0, r1, c1 = _bbox(coords)
    h, w = shape
    if direction == DIRS["right"]:
        return 0, w - 1 - c1
    if direction == DIRS["left"]:
        return 0, -c0
    if direction == DIRS["down"]:
        return h - 1 - r1, 0
    return -r0, 0


def _placement_options(
    swept_box: tuple[int, int, int, int], direction: tuple[int, int], shape: tuple[int, int]
) -> list[tuple[int, int]]:
    h, w = shape
    r0, c0, r1, c1 = swept_box
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    opts: list[tuple[int, int]] = []
    if direction in (DIRS["left"], DIRS["right"]):
        for row_shift in range(-r0, h - r1):
            opts.append((row_shift, 0))
    else:
        for col_shift in range(-c0, w - c1):
            opts.append((0, col_shift))
    random.shuffle(opts)
    return opts


def _stopping_modes(objects: list[dict], shape: tuple[int, int]) -> tuple[int, int]:
    border_hits = 0
    object_hits = 0
    occupied = {}
    for idx, obj in enumerate(objects):
        for cell in obj["coords"]:
            occupied[tuple(map(int, cell))] = idx
    for idx, obj in enumerate(objects):
        dr, dc = obj["direction"]
        proposed = obj["coords"] + np.array([[dr, dc]])
        if not _in_bounds(proposed, shape):
            border_hits += 1
            continue
        blocker = False
        for cell in proposed:
            owner = occupied.get(tuple(map(int, cell)))
            if owner is not None and owner != idx:
                blocker = True
                break
        if blocker:
            object_hits += 1
    return border_hits, object_hits


def _attempt() -> np.ndarray | None:
    h = random.randint(14, 18)
    w = random.randint(14, 18)
    shape = (h, w)

    chain_count = random.choice((2, 3))
    total_objects = random.randint(3, 5)
    size_choices = [sizes for sizes in SIZE_CHOICES[total_objects] if len(sizes) == chain_count]
    if not size_choices:
        return None
    chain_sizes = list(random.choice(size_choices))
    random.shuffle(chain_sizes)
    if max(chain_sizes) < 2:
        return None

    horizontal = random.choice([DIRS["left"], DIRS["right"]])
    vertical = random.choice([DIRS["up"], DIRS["down"]])
    directions = [horizontal, vertical]
    if chain_count == 3:
        remaining = [d for d in DIRS.values() if d not in directions]
        directions.append(random.choice(remaining))
    random.shuffle(directions)

    chains = []
    for direction, size in zip(directions, chain_sizes):
        final_chain = _build_chain_local(direction, size)
        final_coords = np.concatenate([obj["coords"] for obj in final_chain], axis=0)
        r0, c0, r1, c1 = _bbox(final_coords)
        capacity = (w - (c1 - c0 + 1)) if direction in (DIRS["left"], DIRS["right"]) else (h - (r1 - r0 + 1))
        shifts = _sample_reverse_shifts(size, capacity)
        if shifts is None:
            return None
        input_chain = _translate_chain_for_input(final_chain, shifts)
        swept = _chain_swept_bbox(input_chain, final_chain)
        chains.append(
            {
                "direction": direction,
                "final_local": final_chain,
                "input_local": input_chain,
                "swept_local": swept,
            }
        )

    placed_input: list[dict] = []
    placed_final: list[dict] = []
    used_boxes: list[tuple[int, int, int, int]] = []
    colors = random.sample(range(1, 10), total_objects)
    color_idx = 0

    for chain in chains:
        anchor_dr, anchor_dc = _settled_offset(chain["final_local"], chain["direction"], shape)
        final_at_border = _shift_chain(chain["final_local"], anchor_dr, anchor_dc, color=0)
        input_at_border = _shift_chain(chain["input_local"], anchor_dr, anchor_dc, color=0)
        swept_at_border = _chain_swept_bbox(input_at_border, final_at_border)

        placed = False
        for orth_dr, orth_dc in _placement_options(swept_at_border, chain["direction"], shape):
            moved_final = _shift_chain(final_at_border, orth_dr, orth_dc, color=0)
            moved_input = _shift_chain(input_at_border, orth_dr, orth_dc, color=0)
            swept = _expand_bbox(_chain_swept_bbox(moved_input, moved_final), 1)
            if any(not _boxes_disjoint(swept, prev) for prev in used_boxes):
                continue

            chain_colors = colors[color_idx : color_idx + len(moved_input)]
            color_idx += len(moved_input)
            for obj, color in zip(moved_input, chain_colors):
                obj["color"] = color
            for obj, color in zip(moved_final, chain_colors):
                obj["color"] = color
            placed_input.extend(moved_input)
            placed_final.extend(moved_final)
            used_boxes.append(swept)
            placed = True
            break

        if not placed:
            return None

    inp = _render(placed_input, shape, bg=0, erase_leaf=False)

    try:
        bg, parsed = _extract_objects(inp)
        if bg != 0 or len(parsed) != total_objects:
            return None
        final_from_parse, stats = _simulate(parsed, shape)
    except AssertionError:
        return None

    expected_full = _render(placed_final, shape, bg=0, erase_leaf=False)
    actual_full = _render(final_from_parse, shape, bg=0, erase_leaf=False)
    if not np.array_equal(expected_full, actual_full):
        return None

    out = main(inp)
    expected_out = _render(placed_final, shape, bg=0, erase_leaf=True)
    if not np.array_equal(out, expected_out):
        return None

    border_hits, object_hits = _stopping_modes(final_from_parse, shape)
    distinct_dirs = {obj["direction"] for obj in parsed}
    if len(distinct_dirs) < 2:
        return None
    if not any(direction in distinct_dirs for direction in (DIRS["left"], DIRS["right"])):
        return None
    if not any(direction in distinct_dirs for direction in (DIRS["up"], DIRS["down"])):
        return None
    if max(stats["move_counts"]) < 2:
        return None
    if object_hits < 1 or border_hits < 1:
        return None
    if np.array_equal(inp, out):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid protrusion-gravity input")
