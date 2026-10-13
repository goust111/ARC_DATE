from __future__ import annotations

from collections import Counter, deque
import math
import random

import numpy as np


DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
HORIZONTAL_SIDES = ("top", "bottom")
VERTICAL_SIDES = ("left", "right")
ALL_SIDES = HORIZONTAL_SIDES + VERTICAL_SIDES
SIDE_AXIS = {"top": "h", "bottom": "h", "left": "v", "right": "v"}


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _find_components(mask: np.ndarray) -> list[np.ndarray]:
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    height, width = map(int, mask.shape)

    for r in range(height):
        for c in range(width):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < height and 0 <= nc < width and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _component_subgrid(grid: np.ndarray, coords: np.ndarray, bg: int) -> tuple[tuple[int, int, int, int], np.ndarray]:
    box = _bbox_coords(coords)
    r0, c0, r1, c1 = box
    sub = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
    for r, c in coords:
        sub[int(r) - r0, int(c) - c0] = int(grid[int(r), int(c)])
    return box, sub


def _body_offset(side: str) -> tuple[int, int]:
    if side == "top":
        return 1, 0
    if side == "bottom":
        return 0, 0
    if side == "left":
        return 0, 1
    if side == "right":
        return 0, 0
    raise ValueError(f"unknown side: {side}")


def _board_shape(body_h: int, body_w: int, side: str) -> tuple[int, int]:
    if side in HORIZONTAL_SIDES:
        return body_h + 1, body_w
    return body_h, body_w + 1


def _sequence_matches(seq_a: tuple[int, ...], seq_b: tuple[int, ...]) -> bool:
    if len(seq_a) != len(seq_b):
        return False
    direct = all(a + b == 1 for a, b in zip(seq_a, seq_b))
    mirrored = all(a + b == 1 for a, b in zip(seq_a, reversed(seq_b)))
    return direct or mirrored


def _expected_board_mask(body_h: int, body_w: int, side: str, seq: tuple[int, ...]) -> np.ndarray:
    height, width = _board_shape(body_h, body_w, side)
    mask = np.zeros((height, width), dtype=bool)
    body_r0, body_c0 = _body_offset(side)
    mask[body_r0 : body_r0 + body_h, body_c0 : body_c0 + body_w] = True

    for i, bit in enumerate(seq, start=1):
        if side == "top":
            if bit:
                mask[0, body_c0 + i] = True
            else:
                mask[body_r0, body_c0 + i] = False
        elif side == "bottom":
            if bit:
                mask[height - 1, body_c0 + i] = True
            else:
                mask[body_r0 + body_h - 1, body_c0 + i] = False
        elif side == "left":
            if bit:
                mask[body_r0 + i, 0] = True
            else:
                mask[body_r0 + i, body_c0] = False
        elif side == "right":
            if bit:
                mask[body_r0 + i, width - 1] = True
            else:
                mask[body_r0 + i, body_c0 + body_w - 1] = False
    return mask


def _strict_interior_mask(shape: tuple[int, int], body_h: int, body_w: int, side: str) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    body_r0, body_c0 = _body_offset(side)
    mask[body_r0 + 1 : body_r0 + body_h - 1, body_c0 + 1 : body_c0 + body_w - 1] = True
    return mask


def _try_parse_side(sub: np.ndarray, side: str, base_color: int, bg: int) -> dict | None:
    height, width = map(int, sub.shape)
    if side in HORIZONTAL_SIDES:
        body_h, body_w = height - 1, width
    else:
        body_h, body_w = height, width - 1
    if body_h < 5 or body_w < 5:
        return None

    if side in HORIZONTAL_SIDES:
        seq_len = body_w - 2
    else:
        seq_len = body_h - 2
    if seq_len < 3:
        return None

    body_r0, body_c0 = _body_offset(side)
    occupancy = sub != bg
    seq: list[int] = []
    saw_tab = False
    saw_notch = False

    if side == "top":
        if bool(occupancy[0, 0]) or bool(occupancy[0, width - 1]):
            return None
        if not bool(occupancy[body_r0, body_c0]) or not bool(occupancy[body_r0, body_c0 + body_w - 1]):
            return None
        for i in range(1, body_w - 1):
            tab = bool(occupancy[0, body_c0 + i])
            base = bool(occupancy[body_r0, body_c0 + i])
            if tab and base:
                seq.append(1)
                saw_tab = True
            elif (not tab) and (not base):
                seq.append(0)
                saw_notch = True
            else:
                return None
    elif side == "bottom":
        if bool(occupancy[height - 1, 0]) or bool(occupancy[height - 1, width - 1]):
            return None
        if not bool(occupancy[body_r0 + body_h - 1, body_c0]) or not bool(
            occupancy[body_r0 + body_h - 1, body_c0 + body_w - 1]
        ):
            return None
        for i in range(1, body_w - 1):
            tab = bool(occupancy[height - 1, body_c0 + i])
            base = bool(occupancy[body_r0 + body_h - 1, body_c0 + i])
            if tab and base:
                seq.append(1)
                saw_tab = True
            elif (not tab) and (not base):
                seq.append(0)
                saw_notch = True
            else:
                return None
    elif side == "left":
        if bool(occupancy[0, 0]) or bool(occupancy[height - 1, 0]):
            return None
        if not bool(occupancy[body_r0, body_c0]) or not bool(occupancy[body_r0 + body_h - 1, body_c0]):
            return None
        for i in range(1, body_h - 1):
            tab = bool(occupancy[body_r0 + i, 0])
            base = bool(occupancy[body_r0 + i, body_c0])
            if tab and base:
                seq.append(1)
                saw_tab = True
            elif (not tab) and (not base):
                seq.append(0)
                saw_notch = True
            else:
                return None
    elif side == "right":
        if bool(occupancy[0, width - 1]) or bool(occupancy[height - 1, width - 1]):
            return None
        if not bool(occupancy[body_r0, body_c0 + body_w - 1]) or not bool(
            occupancy[body_r0 + body_h - 1, body_c0 + body_w - 1]
        ):
            return None
        for i in range(1, body_h - 1):
            tab = bool(occupancy[body_r0 + i, width - 1])
            base = bool(occupancy[body_r0 + i, body_c0 + body_w - 1])
            if tab and base:
                seq.append(1)
                saw_tab = True
            elif (not tab) and (not base):
                seq.append(0)
                saw_notch = True
            else:
                return None

    seq_t = tuple(seq)
    if not saw_tab or not saw_notch:
        return None

    expected_mask = _expected_board_mask(body_h, body_w, side, seq_t)
    if not np.array_equal(expected_mask, occupancy):
        return None

    interior_mask = _strict_interior_mask(sub.shape, body_h, body_w, side)
    if np.any(sub[interior_mask] == bg):
        return None
    if np.any(sub[expected_mask & ~interior_mask] != base_color):
        return None

    interior = sub[
        body_r0 + 1 : body_r0 + body_h - 1,
        body_c0 + 1 : body_c0 + body_w - 1,
    ]
    motif_mask = interior != base_color
    return {
        "side": side,
        "axis": SIDE_AXIS[side],
        "body_h": body_h,
        "body_w": body_w,
        "body_r0": body_r0,
        "body_c0": body_c0,
        "seq": seq_t,
        "motif_mask": motif_mask,
        "interior_h": body_h - 2,
        "interior_w": body_w - 2,
    }


def _parse_plate(grid: np.ndarray, coords: np.ndarray, bg: int) -> dict:
    box, sub = _component_subgrid(grid, coords, bg)
    colors = [int(v) for v in sub[sub != bg].flat]
    base_color = int(Counter(colors).most_common(1)[0][0])
    candidates = []
    for side in ALL_SIDES:
        parsed = _try_parse_side(sub, side, base_color, bg)
        if parsed is not None:
            candidates.append(parsed)
    assert len(candidates) == 1, "expected exactly one toothed side"

    parsed = candidates[0]
    r0, c0, _r1, _c1 = box
    parsed.update(
        {
            "bbox": box,
            "base": base_color,
            "interior_r0": r0 + parsed["body_r0"] + 1,
            "interior_c0": c0 + parsed["body_c0"] + 1,
        }
    )
    return parsed


def _plates_match(plate_a: dict, plate_b: dict) -> bool:
    return (
        plate_a["axis"] == plate_b["axis"]
        and plate_a["body_h"] == plate_b["body_h"]
        and plate_a["body_w"] == plate_b["body_w"]
        and _sequence_matches(plate_a["seq"], plate_b["seq"])
    )


def _find_pairs(plates: list[dict]) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    used: set[int] = set()
    for i in range(len(plates)):
        if i in used:
            continue
        candidates = [j for j in range(len(plates)) if j != i and j not in used and _plates_match(plates[i], plates[j])]
        if len(candidates) != 1:
            continue
        j = candidates[0]
        back = [k for k in range(len(plates)) if k != j and k not in used and _plates_match(plates[j], plates[k])]
        if len(back) == 1 and back[0] == i:
            used.add(i)
            used.add(j)
            pairs.append((i, j))
    return pairs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair toothed plates by complementary edge profiles and swap the partner motif masks recolored by source base colors.

    Input structure:
    - A dominant background contains several separated multicolor plates.
    - Each plate is a filled rectangle with exactly one toothed side made from
      one-pixel tabs and notches; the other three sides stay flat.
    - Inside each plate, a small multicolor motif sits strictly in the interior.

    Rule:
    - Recover each plate's base color, body rectangle, toothed side, and tab or
      notch sequence.
    - Pair plates whose toothed sequences are uniquely complementary when the
      two edges are compared face-to-face on the same axis.
    - For every pair, repaint each interior to its own base color, then stamp
      the partner's motif mask at the same relative interior positions using
      the partner's base color. Plates without a unique complement stay unchanged.

    Key insight:
    - Colors, positions, and sizes are distractions; the real pairing signal is
      the unique complementary tab or notch pattern along the single irregular edge.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    components = _find_components(g != bg)
    plates = [_parse_plate(g, coords, bg) for coords in components]

    out = g.copy()
    for i, j in _find_pairs(plates):
        for dst, src in ((plates[i], plates[j]), (plates[j], plates[i])):
            assert dst["motif_mask"].shape == src["motif_mask"].shape
            r0 = dst["interior_r0"]
            c0 = dst["interior_c0"]
            h, w = dst["motif_mask"].shape
            patch = out[r0 : r0 + h, c0 : c0 + w]
            patch[:] = int(dst["base"])
            patch[src["motif_mask"]] = int(src["base"])
    return out


# --------------- generator ---------------

COLORS = list(range(10))
BODY_H_RANGE = (5, 8)
BODY_W_RANGE = (5, 8)
PAIR_COUNT_RANGE = (2, 3)
UNPAIRED_COUNT_RANGE = (0, 2)
PADDING = 2
PLACEMENT_TRIES = 250
GEN_RETRIES = 300


def _random_blob(height: int, width: int, target: int) -> set[tuple[int, int]]:
    start = (random.randrange(height), random.randrange(width))
    cells = {start}
    frontier = [start]
    while len(cells) < target and frontier:
        idx = random.randrange(len(frontier))
        r, c = frontier[idx]
        random.shuffle(DIRS4)
        grown = False
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                cells.add((nr, nc))
                frontier.append((nr, nc))
                grown = True
                break
        if not grown:
            frontier.pop(idx)
    return cells


def _make_motif_interior(body_h: int, body_w: int, base_color: int, bg: int) -> np.ndarray:
    interior_h = body_h - 2
    interior_w = body_w - 2
    area = interior_h * interior_w
    max_cells = min(area - 1, max(3, area // 2 + 1))
    target = random.randint(2, max_cells)
    cells = _random_blob(interior_h, interior_w, target)

    palette = [c for c in COLORS if c not in (bg, base_color)]
    n_colors = random.randint(1, min(3, len(palette)))
    motif_colors = random.sample(palette, n_colors)

    interior = np.full((interior_h, interior_w), int(base_color), dtype=int)
    coords = list(cells)
    random.shuffle(coords)
    for i, (r, c) in enumerate(coords):
        if i < len(motif_colors):
            interior[r, c] = motif_colors[i]
        else:
            interior[r, c] = random.choice(motif_colors)
    return interior


def _render_board(board: dict, bg: int, swapped_mask: np.ndarray | None = None, swapped_color: int | None = None) -> np.ndarray:
    body_h = int(board["body_h"])
    body_w = int(board["body_w"])
    side = str(board["side"])
    height, width = _board_shape(body_h, body_w, side)
    sprite = np.full((height, width), int(bg), dtype=int)
    body_r0, body_c0 = _body_offset(side)
    base = int(board["base"])
    sprite[body_r0 : body_r0 + body_h, body_c0 : body_c0 + body_w] = base

    for i, bit in enumerate(board["seq"], start=1):
        if side == "top":
            if bit:
                sprite[0, body_c0 + i] = base
            else:
                sprite[body_r0, body_c0 + i] = bg
        elif side == "bottom":
            if bit:
                sprite[height - 1, body_c0 + i] = base
            else:
                sprite[body_r0 + body_h - 1, body_c0 + i] = bg
        elif side == "left":
            if bit:
                sprite[body_r0 + i, 0] = base
            else:
                sprite[body_r0 + i, body_c0] = bg
        else:
            if bit:
                sprite[body_r0 + i, width - 1] = base
            else:
                sprite[body_r0 + i, body_c0 + body_w - 1] = bg

    ir0 = body_r0 + 1
    ic0 = body_c0 + 1
    patch = sprite[ir0 : ir0 + body_h - 2, ic0 : ic0 + body_w - 2]
    if swapped_mask is None:
        patch[:] = board["motif_interior"]
    else:
        patch[:] = base
        patch[swapped_mask] = int(swapped_color)
    return sprite


def _random_side(axis: str) -> str:
    return random.choice(HORIZONTAL_SIDES if axis == "h" else VERTICAL_SIDES)


def _random_seq(length: int) -> tuple[int, ...]:
    while True:
        seq = tuple(random.randint(0, 1) for _ in range(length))
        if 0 in seq and 1 in seq:
            return seq


def _partner_seq(seq: tuple[int, ...], mirrored: bool) -> tuple[int, ...]:
    if mirrored:
        return tuple(1 - bit for bit in reversed(seq))
    return tuple(1 - bit for bit in seq)


def _choose_pair_sequences(length: int, existing: list[tuple[int, ...]]) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    for _ in range(300):
        seq_a = _random_seq(length)
        seq_b = _partner_seq(seq_a, mirrored=random.random() < 0.5)
        if seq_a == seq_b:
            continue
        if seq_a in existing or seq_b in existing:
            continue
        if any(_sequence_matches(seq_a, other) or _sequence_matches(seq_b, other) for other in existing):
            continue
        return seq_a, seq_b
    return None


def _choose_unpaired_sequence(length: int, existing: list[tuple[int, ...]]) -> tuple[int, ...] | None:
    for _ in range(300):
        seq = _random_seq(length)
        if seq in existing:
            continue
        if any(_sequence_matches(seq, other) for other in existing):
            continue
        return seq
    return None


def _choose_body_shape(axis: str, seen_shapes: dict[str, list[tuple[int, int]]]) -> tuple[int, int]:
    if seen_shapes[axis] and random.random() < 0.55:
        return random.choice(seen_shapes[axis])
    body_h = random.randint(*BODY_H_RANGE)
    body_w = random.randint(*BODY_W_RANGE)
    seen_shapes[axis].append((body_h, body_w))
    return body_h, body_w


def _board_from_parts(body_h: int, body_w: int, side: str, seq: tuple[int, ...], base: int, bg: int) -> dict:
    motif_interior = _make_motif_interior(body_h, body_w, base, bg)
    return {
        "body_h": body_h,
        "body_w": body_w,
        "side": side,
        "axis": SIDE_AXIS[side],
        "seq": seq,
        "base": base,
        "motif_interior": motif_interior,
        "motif_mask": motif_interior != base,
    }


def _place_sprites(sprites: list[np.ndarray], bg: int) -> tuple[np.ndarray, list[tuple[int, int]]] | None:
    occupied_cells = sum(int(np.count_nonzero(sprite != bg)) for sprite in sprites)
    max_h = max(int(sprite.shape[0]) for sprite in sprites)
    max_w = max(int(sprite.shape[1]) for sprite in sprites)
    side_floor = max(max_h + 2 * PADDING + 1, max_w + 2 * PADDING + 1, int(math.ceil(math.sqrt(occupied_cells * 2.4))))
    if side_floor > 30:
        return None

    for _ in range(80):
        height = random.randint(side_floor, 30)
        width = random.randint(side_floor, 30)
        grid = np.full((height, width), int(bg), dtype=int)
        blocked = np.zeros((height, width), dtype=bool)
        positions: list[tuple[int, int] | None] = [None] * len(sprites)
        order = list(range(len(sprites)))
        random.shuffle(order)

        ok = True
        for idx in order:
            sprite = sprites[idx]
            sh, sw = map(int, sprite.shape)
            placed = False
            for _ in range(PLACEMENT_TRIES):
                r0 = random.randint(0, height - sh)
                c0 = random.randint(0, width - sw)
                rr0 = max(0, r0 - PADDING)
                rr1 = min(height, r0 + sh + PADDING)
                cc0 = max(0, c0 - PADDING)
                cc1 = min(width, c0 + sw + PADDING)
                if np.any(blocked[rr0:rr1, cc0:cc1]):
                    continue
                mask = sprite != bg
                if np.any(mask & (grid[r0 : r0 + sh, c0 : c0 + sw] != bg)):
                    continue
                grid[r0 : r0 + sh, c0 : c0 + sw][mask] = sprite[mask]
                blocked[rr0:rr1, cc0:cc1] = True
                positions[idx] = (r0, c0)
                placed = True
                break
            if not placed:
                ok = False
                break

        if ok and positions.count(None) == 0 and _get_bg(grid) == bg:
            return grid, [(int(r), int(c)) for r, c in positions]
    return None


def _expected_output(
    boards: list[dict], positions: list[tuple[int, int]], bg: int, shape: tuple[int, int]
) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)

    pair_groups: dict[int, list[int]] = {}
    for idx, board in enumerate(boards):
        pair_id = board.get("pair_id")
        if pair_id is not None:
            pair_groups.setdefault(int(pair_id), []).append(idx)

    swapped_sprites: dict[int, np.ndarray] = {}
    for pair_members in pair_groups.values():
        assert len(pair_members) == 2
        a, b = pair_members
        swapped_sprites[a] = _render_board(boards[a], bg, boards[b]["motif_mask"], boards[b]["base"])
        swapped_sprites[b] = _render_board(boards[b], bg, boards[a]["motif_mask"], boards[a]["base"])

    for idx, board in enumerate(boards):
        sprite = swapped_sprites.get(idx, _render_board(board, bg))
        r0, c0 = positions[idx]
        mask = sprite != bg
        out[r0 : r0 + sprite.shape[0], c0 : c0 + sprite.shape[1]][mask] = sprite[mask]
    return out


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.randint(*PAIR_COUNT_RANGE)
    max_unpaired = min(UNPAIRED_COUNT_RANGE[1], 9 - 2 * pair_count)
    unpaired_count = random.randint(UNPAIRED_COUNT_RANGE[0], max_unpaired)
    total_boards = 2 * pair_count + unpaired_count

    bg = random.choice(COLORS)
    base_colors = random.sample([c for c in COLORS if c != bg], total_boards)
    base_iter = iter(base_colors)

    boards: list[dict] = []
    seen_shapes = {"h": [], "v": []}
    seq_registry: dict[tuple[str, int, int], list[tuple[int, ...]]] = {}

    for pair_id in range(pair_count):
        axis = random.choice(["h", "v"])
        body_h, body_w = _choose_body_shape(axis, seen_shapes)
        key = (axis, body_h, body_w)
        length = body_w - 2 if axis == "h" else body_h - 2
        existing = seq_registry.setdefault(key, [])
        seq_pair = _choose_pair_sequences(length, existing)
        if seq_pair is None:
            return None
        seq_a, seq_b = seq_pair
        existing.extend([seq_a, seq_b])

        board_a = _board_from_parts(body_h, body_w, _random_side(axis), seq_a, next(base_iter), bg)
        board_b = _board_from_parts(body_h, body_w, _random_side(axis), seq_b, next(base_iter), bg)
        board_a["pair_id"] = pair_id
        board_b["pair_id"] = pair_id
        boards.extend([board_a, board_b])

    for _ in range(unpaired_count):
        axis = random.choice(["h", "v"])
        body_h, body_w = _choose_body_shape(axis, seen_shapes)
        key = (axis, body_h, body_w)
        length = body_w - 2 if axis == "h" else body_h - 2
        existing = seq_registry.setdefault(key, [])
        seq = _choose_unpaired_sequence(length, existing)
        if seq is None:
            return None
        existing.append(seq)
        board = _board_from_parts(body_h, body_w, _random_side(axis), seq, next(base_iter), bg)
        board["pair_id"] = None
        boards.append(board)

    sprites = [_render_board(board, bg) for board in boards]
    placed = _place_sprites(sprites, bg)
    if placed is None:
        return None
    inp, positions = placed
    expected = _expected_output(boards, positions, bg, inp.shape)

    if inp.shape != expected.shape:
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid toothed-key input")
