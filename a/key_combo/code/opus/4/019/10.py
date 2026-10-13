from __future__ import annotations

from collections import Counter, defaultdict
import itertools
import random

import numpy as np


BG = 0
MARKER = 1
SYMBOL_COLORS = [2, 3, 6, 7, 8]
BORDER_COORDS = (
    (0, 0),
    (0, 1),
    (0, 2),
    (1, 0),
    (1, 2),
    (2, 0),
    (2, 1),
    (2, 2),
)
PARTITIONS = [(3, 3), (2, 2, 2), (2, 2, 3), (2, 3, 3)]
SIZE_CHOICES = [19, 23, 27]
SIDES = ["top", "right", "bottom", "left"]


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    mask = grid == int(color)
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in _neighbors8(rr, cc):
                    if _in_bounds((h, w), nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(cells, dtype=int))
    return out


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    seen = {tuple(map(int, coords[0]))}
    stack = [tuple(map(int, coords[0]))]
    points = {tuple(map(int, rc)) for rc in coords}
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors8(r, c):
            if (nr, nc) in points and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == len(points)


def _mask_key(mask: np.ndarray) -> tuple[str, ...]:
    return tuple("".join("1" if cell else "0" for cell in row) for row in mask.astype(bool))


def _shape_signature(mask: np.ndarray) -> tuple[str, ...]:
    variants = []
    for k in range(4):
        rotated = np.rot90(mask, k)
        variants.append(_mask_key(rotated))
        variants.append(_mask_key(np.fliplr(rotated)))
    return min(variants)


def _unique_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen = set()
    for k in range(4):
        for variant in (np.rot90(mask, k), np.fliplr(np.rot90(mask, k))):
            key = _mask_key(variant)
            if key in seen:
                continue
            seen.add(key)
            variants.append(variant.astype(bool).copy())
    return variants


def _valid_symbol_mask(mask: np.ndarray) -> bool:
    if mask.shape != (3, 3):
        return False
    if bool(mask[1, 1]):
        return False
    if int(mask.sum()) < 4:
        return False
    if not mask[0, :].any() or not mask[2, :].any() or not mask[:, 0].any() or not mask[:, 2].any():
        return False
    return _is_connected(mask)


def _all_shape_classes() -> list[dict[str, object]]:
    by_signature: dict[tuple[str, ...], np.ndarray] = {}
    for bits in itertools.product([False, True], repeat=len(BORDER_COORDS)):
        mask = np.zeros((3, 3), dtype=bool)
        for bit, (r, c) in zip(bits, BORDER_COORDS):
            mask[r, c] = bit
        if not _valid_symbol_mask(mask):
            continue
        sig = _shape_signature(mask)
        if sig not in by_signature:
            by_signature[sig] = mask.copy()

    classes = []
    for sig, base in sorted(by_signature.items(), key=lambda item: item[0]):
        variants = _unique_variants(base)
        classes.append({"signature": sig, "base": base, "variants": variants})
    return classes


SHAPE_CLASSES = _all_shape_classes()
GEN_SHAPE_CLASSES = [cls for cls in SHAPE_CLASSES if len(cls["variants"]) >= 2] or SHAPE_CLASSES


def _marker_side(grid: np.ndarray) -> str:
    sides = []
    if np.all(grid[0, :] == MARKER):
        sides.append("top")
    if np.all(grid[:, -1] == MARKER):
        sides.append("right")
    if np.all(grid[-1, :] == MARKER):
        sides.append("bottom")
    if np.all(grid[:, 0] == MARKER):
        sides.append("left")
    assert len(sides) == 1, "expected exactly one full-edge blue marker line"
    return sides[0]


def _direction_from_side(side: str) -> tuple[int, int]:
    return {
        "top": (1, 0),
        "right": (0, -1),
        "bottom": (-1, 0),
        "left": (0, 1),
    }[side]


def _extract_symbols(grid: np.ndarray) -> tuple[str, list[dict[str, object]]]:
    side = _marker_side(grid)
    colors = sorted(set(map(int, np.unique(grid))) - {BG, MARKER})
    symbols: list[dict[str, object]] = []
    for color in colors:
        for coords in _components_of_color(grid, color):
            r0, c0, r1, c1 = _bbox_coords(coords)
            assert (r1 - r0 + 1, c1 - c0 + 1) == (3, 3), "each symbol must occupy a 3x3 bbox"
            patch = grid[r0 : r1 + 1, c0 : c1 + 1]
            assert int(patch[1, 1]) == BG, "symbol centers must be black"
            mask = patch == int(color)
            assert _valid_symbol_mask(mask), "invalid hollow-symbol border mask"
            symbols.append(
                {
                    "bbox": (r0, c0, r1, c1),
                    "center": (r0 + 1, c0 + 1),
                    "color": int(color),
                    "mask": mask.astype(bool).copy(),
                    "signature": _shape_signature(mask),
                }
            )

    symbols.sort(key=lambda sym: (sym["center"][0], sym["center"][1], sym["color"]))
    assert symbols, "expected at least one hollow symbol"
    return side, symbols


def _distance_to_target(pos: tuple[int, int], shape: tuple[int, int], direction: tuple[int, int]) -> int:
    h, w = shape
    r, c = pos
    dr, dc = direction
    if dr == 1:
        return h - 1 - r
    if dr == -1:
        return r
    if dc == 1:
        return w - 1 - c
    return c


def _group_sort_key(
    group: list[int],
    positions: dict[int, tuple[int, int]],
    shape: tuple[int, int],
    direction: tuple[int, int],
) -> tuple[int, int, int]:
    coords = [positions[idx] for idx in group]
    dist = min(_distance_to_target(pos, shape, direction) for pos in coords)
    dr, dc = direction
    if dr != 0:
        orth = min(c for _r, c in coords)
    else:
        orth = min(r for r, _c in coords)
    return dist, orth, min(group)


def _leader_index(group: list[int], positions: dict[int, tuple[int, int]], direction: tuple[int, int]) -> int:
    dr, dc = direction
    if dr == 1:
        return max(group, key=lambda idx: (positions[idx][0], -positions[idx][1], -idx))
    if dr == -1:
        return min(group, key=lambda idx: (positions[idx][0], positions[idx][1], idx))
    if dc == 1:
        return max(group, key=lambda idx: (positions[idx][1], -positions[idx][0], -idx))
    return min(group, key=lambda idx: (positions[idx][1], positions[idx][0], idx))


def _solve_state(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    side, symbols = _extract_symbols(g)
    direction = _direction_from_side(side)

    groups_by_sig: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for idx, symbol in enumerate(symbols):
        groups_by_sig[symbol["signature"]].append(idx)
    groups = [groups_by_sig[sig] for sig in sorted(groups_by_sig)]

    positions = {idx: tuple(map(int, symbol["center"])) for idx, symbol in enumerate(symbols)}
    owner = np.full(g.shape, -1, dtype=int)
    owner[g == MARKER] = -2
    symbol_to_group = {}
    for gid, group in enumerate(groups):
        for idx in group:
            r, c = positions[idx]
            assert owner[r, c] == -1, "compressed centers must be unique"
            owner[r, c] = gid
            symbol_to_group[idx] = gid

    order = sorted(
        range(len(groups)),
        key=lambda gid: _group_sort_key(groups[gid], positions, g.shape, direction),
    )

    steps_by_group: dict[int, int] = {}
    stop_reason_by_group: dict[int, str] = {}
    dr, dc = direction

    for gid in order:
        group = groups[gid]
        steps = 0
        while True:
            blockers = set()
            next_positions = []
            for idx in group:
                r, c = positions[idx]
                nr, nc = r + dr, c + dc
                if not _in_bounds(g.shape, nr, nc):
                    blockers.add("edge")
                    continue
                cell_owner = int(owner[nr, nc])
                if cell_owner not in (-1, gid):
                    blockers.add("group" if cell_owner >= 0 else "marker")
                    continue
                next_positions.append((idx, nr, nc))
            if blockers:
                if "group" in blockers:
                    stop_reason_by_group[gid] = "group"
                elif "edge" in blockers:
                    stop_reason_by_group[gid] = "edge"
                else:
                    stop_reason_by_group[gid] = "marker"
                break

            for idx in group:
                r, c = positions[idx]
                owner[r, c] = -1
            for idx, nr, nc in next_positions:
                positions[idx] = (nr, nc)
            for idx in group:
                r, c = positions[idx]
                owner[r, c] = gid
            steps += 1

        steps_by_group[gid] = steps

    group_color: dict[int, int] = {}
    for gid, group in enumerate(groups):
        lead = _leader_index(group, positions, direction)
        group_color[gid] = int(symbols[lead]["color"])

    out = np.full_like(g, BG)
    out[g == MARKER] = MARKER
    for gid, group in enumerate(groups):
        fill = group_color[gid]
        for idx in group:
            r, c = positions[idx]
            out[r, c] = fill

    return {
        "side": side,
        "direction": direction,
        "symbols": symbols,
        "groups": groups,
        "positions": positions,
        "group_color": group_color,
        "steps_by_group": steps_by_group,
        "stop_reason_by_group": stop_reason_by_group,
        "output": out,
        "order": order,
        "symbol_to_group": symbol_to_group,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress dihedrally equivalent hollow symbols into pixels, slide each shape-class group toward the opposite edge, and recolor every group by its leading member.

    Input structure:
    - The grid has black background, one full-edge blue marker line, and several separated monochrome hollow 3x3 symbols whose centers are black.
    - Each symbol occupies a 3x3 bounding box and is described by which border cells are present; symbols may share a shape up to rotation or reflection.
    - Symbols use visible foreground colors from a small palette, and the marker line indicates which opposite edge the compressed pixels should move toward.

    Rule:
    - Extract every symbol's 3x3 border mask, group symbols by dihedral shape equivalence, and compress each symbol to one pixel at the center of its 3x3 box using its original color.
    - Convert the marker side into motion toward the opposite edge: top->down, right->left, bottom->up, left->right.
    - Slide each shape-equivalence group as one rigid set, processing the group already closest to the target edge first; the whole group advances one cell per step until any member would hit the boundary or a pixel from another group.
    - After sliding, recolor every pixel in a group to the original color of that group's member that is furthest forward in the motion direction, and keep the blue marker line unchanged.

    Key insight:
    - The crucial information is the shape class before compression: the 3x3 masks disappear when each symbol becomes one pixel, but the remembered equivalence groups still control both the collective slide and the final recoloring.
    """
    return _solve_state(np.asarray(grid, dtype=int))["output"]


def _candidate_axes(h: int, w: int, side: str) -> tuple[list[int], list[int], bool]:
    rows = list(range(3, h - 3, 4))
    cols = list(range(3, w - 3, 4))
    if side in ("top", "bottom"):
        return rows, cols, True
    return cols, rows, False


def _ordered_axis_values(axis_values: list[int], side: str) -> list[int]:
    return axis_values if side in ("top", "left") else list(reversed(axis_values))


def _assign_group_positions(
    h: int,
    w: int,
    side: str,
    group_sizes: tuple[int, ...],
) -> dict[int, list[tuple[int, int]]] | None:
    axis_values, orth_values, axis_is_row = _candidate_axes(h, w, side)
    if len(axis_values) < max(group_sizes) + len(group_sizes) - 2:
        return None
    ordered_axis = _ordered_axis_values(axis_values, side)

    back_to_front = list(range(len(group_sizes)))
    random.shuffle(back_to_front)
    lead_idx: dict[int, int] = {}
    cursor = 0
    for gid in back_to_front:
        cursor = max(cursor, group_sizes[gid] - 1)
        if cursor >= len(ordered_axis):
            return None
        lead_idx[gid] = cursor
        cursor += 1

    axis_members: dict[int, list[int]] = {}
    for gid, size in enumerate(group_sizes):
        earlier = ordered_axis[: lead_idx[gid]]
        if len(earlier) < size - 1:
            return None
        chosen = random.sample(earlier, size - 1) + [ordered_axis[lead_idx[gid]]]
        random.shuffle(chosen)
        axis_members[gid] = chosen

    pool_size = min(len(orth_values), random.choice([3, 4]) if len(orth_values) >= 4 else len(orth_values))
    if pool_size < 2:
        return None
    orth_pool = random.sample(orth_values, pool_size)
    used_pairs: set[tuple[int, int]] = set()
    lane_usage: Counter[int] = Counter()
    positions: dict[int, list[tuple[int, int]]] = {}

    place_order = list(range(len(group_sizes)))
    random.shuffle(place_order)
    for gid in place_order:
        coords: list[tuple[int, int]] = []
        for axis_v in axis_members[gid]:
            candidates = [lane for lane in orth_pool if (axis_v, lane) not in used_pairs]
            if not candidates:
                candidates = [lane for lane in orth_values if (axis_v, lane) not in used_pairs]
            if not candidates:
                return None

            reused = [lane for lane in candidates if lane_usage[lane] > 0]
            lane = random.choice(reused if reused and random.random() < 0.75 else candidates)
            used_pairs.add((axis_v, lane))
            lane_usage[lane] += 1
            coords.append((axis_v, lane) if axis_is_row else (lane, axis_v))
        positions[gid] = coords

    shared_lane = False
    if axis_is_row:
        lanes_by_gid = [set(c for _r, c in positions[gid]) for gid in positions]
    else:
        lanes_by_gid = [set(r for r, _c in positions[gid]) for gid in positions]
    for i in range(len(lanes_by_gid)):
        for j in range(i + 1, len(lanes_by_gid)):
            if lanes_by_gid[i] & lanes_by_gid[j]:
                shared_lane = True
                break
        if shared_lane:
            break
    return positions if shared_lane else None


def _leading_member_index(positions: list[tuple[int, int]], direction: tuple[int, int]) -> int:
    if direction == (1, 0):
        return max(range(len(positions)), key=lambda i: (positions[i][0], -positions[i][1], -i))
    if direction == (-1, 0):
        return min(range(len(positions)), key=lambda i: (positions[i][0], positions[i][1], i))
    if direction == (0, 1):
        return max(range(len(positions)), key=lambda i: (positions[i][1], -positions[i][0], -i))
    return min(range(len(positions)), key=lambda i: (positions[i][1], positions[i][0], i))


def _choose_group_colors(size: int, leader_idx: int) -> list[int]:
    colors = [random.choice(SYMBOL_COLORS) for _ in range(size)]
    leader_color = random.choice(SYMBOL_COLORS)
    colors[leader_idx] = leader_color
    if len(set(colors)) == 1:
        alternatives = [c for c in SYMBOL_COLORS if c != leader_color]
        other_idx = random.choice([i for i in range(size) if i != leader_idx])
        colors[other_idx] = random.choice(alternatives)
    return colors


def _different_variant(variants: list[np.ndarray], current: np.ndarray) -> np.ndarray:
    keys = {_mask_key(current)}
    options = [variant for variant in variants if _mask_key(variant) not in keys]
    return random.choice(options) if options else current


def _render_symbol(grid: np.ndarray, center: tuple[int, int], color: int, mask: np.ndarray) -> None:
    r, c = center
    patch = np.full((3, 3), BG, dtype=int)
    patch[mask] = int(color)
    grid[r - 1 : r + 2, c - 1 : c + 2] = patch


def _build_input() -> np.ndarray | None:
    side = random.choice(SIDES)
    h = random.choice(SIZE_CHOICES)
    w = random.choice(SIZE_CHOICES)
    group_sizes = tuple(random.choice(PARTITIONS))
    direction = _direction_from_side(side)

    positions_by_group = _assign_group_positions(h, w, side, group_sizes)
    if positions_by_group is None:
        return None

    classes = random.sample(GEN_SHAPE_CLASSES, len(group_sizes))
    grid = np.full((h, w), BG, dtype=int)
    if side == "top":
        grid[0, :] = MARKER
    elif side == "right":
        grid[:, -1] = MARKER
    elif side == "bottom":
        grid[-1, :] = MARKER
    else:
        grid[:, 0] = MARKER

    for gid, size in enumerate(group_sizes):
        positions = positions_by_group[gid]
        leader_idx = _leading_member_index(positions, direction)
        colors = _choose_group_colors(size, leader_idx)
        variants = classes[gid]["variants"]
        masks = [random.choice(variants).copy() for _ in range(size)]
        if len(variants) > 1 and len({_mask_key(mask) for mask in masks}) == 1:
            change_idx = random.randrange(size)
            masks[change_idx] = _different_variant(variants, masks[change_idx]).copy()

        for pos, color, mask in zip(positions, colors, masks):
            _render_symbol(grid, pos, color, mask)

    return grid


def _semantic_checks(inp: np.ndarray, state: dict[str, object]) -> bool:
    out = state["output"]
    symbols: list[dict[str, object]] = state["symbols"]
    groups: list[list[int]] = state["groups"]
    positions: dict[int, tuple[int, int]] = state["positions"]
    group_color: dict[int, int] = state["group_color"]
    steps_by_group: dict[int, int] = state["steps_by_group"]
    stop_reason_by_group: dict[int, str] = state["stop_reason_by_group"]

    if not np.array_equal(main(inp), out):
        return False
    if max(inp.shape) > 30 or max(out.shape) > 30:
        return False
    if len(symbols) not in (6, 7, 8):
        return False
    if len(groups) not in (2, 3):
        return False
    if any(len(group) not in (2, 3) for group in groups):
        return False
    if np.count_nonzero(out == MARKER) != (inp.shape[1] if state["side"] in ("top", "bottom") else inp.shape[0]):
        return False
    if np.count_nonzero((out != BG) & (out != MARKER)) != len(symbols):
        return False
    if not any(reason == "group" for reason in stop_reason_by_group.values()):
        return False
    if not all(steps > 0 for steps in steps_by_group.values()):
        return False

    any_recolored = False
    for gid, group in enumerate(groups):
        original_colors = [int(symbols[idx]["color"]) for idx in group]
        if len(set(original_colors)) == 1:
            return False
        fill = group_color[gid]
        for idx in group:
            r, c = positions[idx]
            if int(out[r, c]) != fill:
                return False
        any_recolored |= any(color != fill for color in original_colors)
    return any_recolored


def generate_input() -> np.ndarray:
    for _ in range(400):
        inp = _build_input()
        if inp is None:
            continue
        try:
            state = _solve_state(inp)
        except AssertionError:
            continue
        if _semantic_checks(inp, state):
            return inp
    raise RuntimeError("failed to generate a valid grouped-slide ARC input")
