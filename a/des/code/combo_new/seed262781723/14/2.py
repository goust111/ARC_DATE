import itertools
import random

import numpy as np


BG = 0
ARROW = 2
PALETTE = [1, 3, 4, 5, 6, 7, 8, 9]
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDES = ["top", "bottom", "left", "right"]
SIDE_TO_MOVE = {
    "top": (1, 0),
    "bottom": (-1, 0),
    "left": (0, 1),
    "right": (0, -1),
}

GROUP_COUNT_RANGE = (1, 3)
RECT_H_RANGE = (4, 7)
RECT_W_RANGE = (4, 7)
PIECE_COUNT_CHOICES = (2, 3)
STEP_RANGE = (2, 5)
MIN_PIECE_AREA = 3
PATCH_ATTEMPTS = 200
PARTITION_ATTEMPTS = 200
PACK_ATTEMPTS = 120
GEN_ATTEMPTS = 300


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in DIRS4:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


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
            cells = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _set_of(coords: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(map(int, cell)) for cell in coords.tolist()}


def _touches_side(coords: np.ndarray, side: str, h: int, w: int) -> bool:
    if side == "top":
        return bool(np.any(coords[:, 0] == 0))
    if side == "bottom":
        return bool(np.any(coords[:, 0] == h - 1))
    if side == "left":
        return bool(np.any(coords[:, 1] == 0))
    return bool(np.any(coords[:, 1] == w - 1))


def _pairwise_well_separated(parts: list[np.ndarray]) -> bool:
    sets = [_set_of(coords) for coords in parts]
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            if sets[i] & sets[j]:
                return False
            for r, c in sets[i]:
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if (r + dr, c + dc) in sets[j]:
                            return False
    return True


def _random_partition(h: int, w: int, sides: list[str]) -> np.ndarray | None:
    for _ in range(PARTITION_ATTEMPTS):
        used: set[tuple[int, int]] = set()
        seeds: list[tuple[int, int]] = []
        for side in sides:
            if side == "top":
                candidates = [(0, c) for c in range(w)]
            elif side == "bottom":
                candidates = [(h - 1, c) for c in range(w)]
            elif side == "left":
                candidates = [(r, 0) for r in range(h)]
            else:
                candidates = [(r, w - 1) for r in range(h)]
            random.shuffle(candidates)
            pick = next((cell for cell in candidates if cell not in used), None)
            if pick is None:
                break
            used.add(pick)
            seeds.append(pick)
        if len(seeds) != len(sides):
            continue

        labels = np.full((h, w), -1, dtype=int)
        regions = [set([seed]) for seed in seeds]
        for idx, (r, c) in enumerate(seeds):
            labels[r, c] = idx

        remaining = h * w - len(seeds)
        while remaining > 0:
            order = list(range(len(seeds)))
            random.shuffle(order)
            order.sort(key=lambda idx: len(regions[idx]))
            grown = False
            for idx in order:
                frontier = []
                for r, c in regions[idx]:
                    for nr, nc in _neighbors4(r, c, h, w):
                        if labels[nr, nc] == -1:
                            frontier.append((nr, nc))
                if not frontier:
                    continue
                nr, nc = random.choice(frontier)
                labels[nr, nc] = idx
                regions[idx].add((nr, nc))
                remaining -= 1
                grown = True
                break
            if not grown:
                break

        if np.any(labels < 0):
            continue

        ok = True
        for idx, side in enumerate(sides):
            coords = np.argwhere(labels == idx)
            if len(coords) < MIN_PIECE_AREA:
                ok = False
                break
            if not _touches_side(coords, side, h, w):
                ok = False
                break
            br0, bc0, br1, bc1 = _bbox(coords)
            if (br1 - br0 + 1) * (bc1 - bc0 + 1) == len(coords):
                ok = False
                break
        if ok:
            return labels
    return None


def _candidate_arrow(
    piece_coords: np.ndarray,
    piece_idx: int,
    move: tuple[int, int],
    owner_of_cell: dict[tuple[int, int], int],
    used_arrows: set[tuple[int, int]],
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    outward = (-move[0], -move[1])
    piece = _set_of(piece_coords)
    candidates: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for r, c in piece:
        side1 = (r + outward[0], c + outward[1])
        side2 = (r + 2 * outward[0], c + 2 * outward[1])
        if side1 in piece or side2 in piece or side1 in used_arrows or side2 in used_arrows:
            continue

        head_neighbors = {
            owner_of_cell[(nr, nc)]
            for nr, nc in ((side1[0] - 1, side1[1]), (side1[0] + 1, side1[1]), (side1[0], side1[1] - 1), (side1[0], side1[1] + 1))
            if (nr, nc) in owner_of_cell
        }
        tail_neighbors = {
            owner_of_cell[(nr, nc)]
            for nr, nc in ((side2[0] - 1, side2[1]), (side2[0] + 1, side2[1]), (side2[0], side2[1] - 1), (side2[0], side2[1] + 1))
            if (nr, nc) in owner_of_cell
        }
        if head_neighbors != {piece_idx} or tail_neighbors:
            continue

        bad = False
        for arrow_cell in used_arrows:
            if abs(arrow_cell[0] - side1[0]) + abs(arrow_cell[1] - side1[1]) <= 1:
                bad = True
                break
            if abs(arrow_cell[0] - side2[0]) + abs(arrow_cell[1] - side2[1]) <= 1:
                bad = True
                break
        if bad:
            continue
        candidates.append((side1, side2))
    return candidates


def _make_group_patch(color: int) -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(PATCH_ATTEMPTS):
        piece_count = random.choice(PIECE_COUNT_CHOICES)
        h = random.randint(*RECT_H_RANGE)
        w = random.randint(*RECT_W_RANGE)
        if h * w < piece_count * (MIN_PIECE_AREA + 1):
            continue

        sides = random.sample(SIDES, piece_count)
        labels = _random_partition(h, w, sides)
        if labels is None:
            continue

        moves = [SIDE_TO_MOVE[side] for side in sides]
        steps = [random.randint(*STEP_RANGE) for _ in range(piece_count)]
        final_parts = [np.argwhere(labels == idx) for idx in range(piece_count)]
        input_parts = [
            coords - step * np.array(move, dtype=int)
            for coords, step, move in zip(final_parts, steps, moves)
        ]
        if not _pairwise_well_separated(input_parts):
            continue

        owner_of_cell: dict[tuple[int, int], int] = {}
        for idx, coords in enumerate(input_parts):
            for cell in _set_of(coords):
                owner_of_cell[cell] = idx

        arrows: list[np.ndarray] = []
        used_arrow_cells: set[tuple[int, int]] = set()
        ok = True
        order = list(range(piece_count))
        random.shuffle(order)
        for idx in order:
            choices = _candidate_arrow(input_parts[idx], idx, moves[idx], owner_of_cell, used_arrow_cells)
            if not choices:
                ok = False
                break
            head, tail = random.choice(choices)
            arrows.append(np.array([head, tail], dtype=int))
            used_arrow_cells.add(head)
            used_arrow_cells.add(tail)
        if not ok:
            continue

        rect_cells = np.array([(r, c) for r in range(h) for c in range(w)], dtype=int)
        all_coords = [rect_cells, *input_parts, *arrows]
        mins = np.vstack([coords.min(axis=0) for coords in all_coords]).min(axis=0)
        maxs = np.vstack([coords.max(axis=0) for coords in all_coords]).max(axis=0)
        shift = -mins
        patch_h = int(maxs[0] - mins[0] + 1)
        patch_w = int(maxs[1] - mins[1] + 1)
        if max(patch_h, patch_w) > 16:
            continue

        inp = np.full((patch_h, patch_w), BG, dtype=int)
        out = np.full((patch_h, patch_w), BG, dtype=int)
        shifted_rect = rect_cells + shift
        out[shifted_rect[:, 0], shifted_rect[:, 1]] = color

        for coords in input_parts:
            shifted = coords + shift
            inp[shifted[:, 0], shifted[:, 1]] = color
        for coords in arrows:
            shifted = coords + shift
            inp[shifted[:, 0], shifted[:, 1]] = ARROW

        try:
            produced = main(inp)
        except AssertionError:
            continue
        if not np.array_equal(produced, out):
            continue
        return inp, out
    return None


def _can_place(occupied: np.ndarray, r0: int, c0: int, ph: int, pw: int, pad: int) -> bool:
    h, w = occupied.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h, r0 + ph + pad)
    cc1 = min(w, c0 + pw + pad)
    return not bool(occupied[rr0:rr1, cc0:cc1].any())


def _mark_place(occupied: np.ndarray, r0: int, c0: int, ph: int, pw: int, pad: int) -> None:
    h, w = occupied.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h, r0 + ph + pad)
    cc1 = min(w, c0 + pw + pad)
    occupied[rr0:rr1, cc0:cc1] = True


def _pack_patches(patches: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray] | None:
    patches = sorted(patches, key=lambda pair: pair[0].shape[0] * pair[0].shape[1], reverse=True)
    max_h = max(pair[0].shape[0] for pair in patches)
    max_w = max(pair[0].shape[1] for pair in patches)
    total_area = sum((pair[0].shape[0] + 2) * (pair[0].shape[1] + 2) for pair in patches)

    for _ in range(PACK_ATTEMPTS):
        min_side = max(max(max_h, max_w), int(np.sqrt(total_area)))
        h = random.randint(min_side, 30)
        w = random.randint(min_side, 30)
        if h < max_h or w < max_w:
            continue

        inp = np.full((h, w), BG, dtype=int)
        out = np.full((h, w), BG, dtype=int)
        occupied = np.zeros((h, w), dtype=bool)

        ok = True
        for inp_patch, out_patch in patches:
            ph, pw = inp_patch.shape
            candidates = [
                (r0, c0)
                for r0 in range(h - ph + 1)
                for c0 in range(w - pw + 1)
                if _can_place(occupied, r0, c0, ph, pw, pad=1)
            ]
            if not candidates:
                ok = False
                break
            r0, c0 = random.choice(candidates)
            inp[r0 : r0 + ph, c0 : c0 + pw] = np.maximum(inp[r0 : r0 + ph, c0 : c0 + pw], inp_patch)
            out[r0 : r0 + ph, c0 : c0 + pw] = np.maximum(out[r0 : r0 + ph, c0 : c0 + pw], out_patch)
            _mark_place(occupied, r0, c0, ph, pw, pad=1)

        if not ok:
            continue
        return inp, out
    return None


def _extract_fragments(grid: np.ndarray):
    h, w = grid.shape
    label = np.full((h, w), -1, dtype=int)
    fragments = []
    idx = 0
    for color in sorted(int(v) for v in np.unique(grid) if v not in (BG, ARROW)):
        for coords in _components(grid == color):
            label[coords[:, 0], coords[:, 1]] = idx
            fragments.append({"id": idx, "color": color, "coords": coords})
            idx += 1
    return fragments, label


def _decode_arrows(grid: np.ndarray, frag_label: np.ndarray) -> dict[int, tuple[int, int]]:
    h, w = grid.shape
    mapping: dict[int, tuple[int, int]] = {}
    for comp in _components(grid == ARROW):
        assert len(comp) == 2, "each arrow must have exactly two red pixels"
        a, b = comp
        assert abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1])) == 1, "arrow pixels must be orthogonally adjacent"

        head_info = []
        for p in (a, b):
            r, c = map(int, p)
            neighbors = {int(frag_label[nr, nc]) for nr, nc in _neighbors4(r, c, h, w) if frag_label[nr, nc] >= 0}
            if neighbors:
                head_info.append(((r, c), neighbors))
        assert len(head_info) == 1, "exactly one red arrow pixel must touch a fragment"
        head, neighbors = head_info[0]
        assert len(neighbors) == 1, "each arrow must identify a single fragment"
        frag_id = next(iter(neighbors))
        tail = tuple(map(int, b if tuple(map(int, a)) == head else a))
        direction = (head[0] - tail[0], head[1] - tail[1])
        assert frag_id not in mapping, "each fragment must have exactly one arrow"
        mapping[frag_id] = direction
    return mapping


def _max_steps(coords: np.ndarray, move: tuple[int, int], h: int, w: int) -> int:
    r0, c0, r1, c1 = _bbox(coords)
    dr, dc = move
    if dr == 1:
        return h - 1 - r1
    if dr == -1:
        return r0
    if dc == 1:
        return w - 1 - c1
    return c0


def _has_contact(piece: set[tuple[int, int]], others: set[tuple[int, int]]) -> bool:
    for r, c in piece:
        for dr, dc in DIRS4:
            if (r + dr, c + dc) in others:
                return True
    return False


def _solve_color_group(parts: list[dict], h: int, w: int) -> list[np.ndarray]:
    ranges = []
    for part in parts:
        max_step = _max_steps(part["coords"], part["move"], h, w)
        assert max_step >= 1, "every fragment must have room to move"
        ranges.append(range(1, max_step + 1))

    total_area = sum(len(part["coords"]) for part in parts)
    solutions: list[list[np.ndarray]] = []
    for steps in itertools.product(*ranges):
        shifted_parts = []
        occupied: set[tuple[int, int]] = set()
        overlap = False
        for part, step in zip(parts, steps):
            dr, dc = part["move"]
            shifted = part["coords"] + np.array([step * dr, step * dc], dtype=int)
            cells = _set_of(shifted)
            if occupied & cells:
                overlap = True
                break
            occupied |= cells
            shifted_parts.append(shifted)
        if overlap or len(occupied) != total_area:
            continue

        union = np.vstack(shifted_parts)
        r0, c0, r1, c1 = _bbox(union)
        if (r1 - r0 + 1) * (c1 - c0 + 1) != total_area:
            continue

        piece_sets = [_set_of(coords) for coords in shifted_parts]
        if not all(_has_contact(piece_sets[i], occupied - piece_sets[i]) for i in range(len(piece_sets))):
            continue

        solutions.append(shifted_parts)
        if len(solutions) > 1:
            break

    assert len(solutions) == 1, "each color group must have a unique rectangle-completing translation"
    return solutions[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each same-colored fragment along its red arrow until the fragments assemble into one solid rectangle.

    Input structure:
    - The background is black.
    - Every non-red foreground color appears as one group of two or three separated 4-connected fragments.
    - Each fragment has one adjacent two-pixel red arrow, with the arrowhead touching the fragment and pointing in the movement direction.

    Rule:
    - Read each red two-pixel marker as a cardinal arrow by comparing the touching head pixel with its red tail pixel.
    - Move each fragment step by step along its own arrow direction.
    - For each color, choose the unique positive translations that make all of that color's fragments merge into one hole-free filled rectangle.
    - Erase all red arrow markers in the output.

    Key insight:
    - The red pair does not encode a distance; it only gives a direction, and the stopping condition is the unique rectangle completed by the translated same-colored fragments.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    fragments, frag_label = _extract_fragments(g)
    assert fragments, "expected at least one fragment"
    directions = _decode_arrows(g, frag_label)
    assert len(directions) == len(fragments), "every fragment must have one arrow"

    by_color: dict[int, list[dict]] = {}
    for frag in fragments:
        move = directions[frag["id"]]
        by_color.setdefault(int(frag["color"]), []).append(
            {"coords": frag["coords"], "move": move, "color": int(frag["color"])}
        )

    out = g.copy()
    out[out == ARROW] = BG
    for frag in fragments:
        coords = frag["coords"]
        out[coords[:, 0], coords[:, 1]] = BG

    for color in sorted(by_color):
        shifted_parts = _solve_color_group(by_color[color], h, w)
        for coords in shifted_parts:
            assert np.all(out[coords[:, 0], coords[:, 1]] == BG), "completed rectangles must not overlap"
            out[coords[:, 0], coords[:, 1]] = color

    return out


def _gen_attempt() -> np.ndarray | None:
    group_count = random.randint(*GROUP_COUNT_RANGE)
    colors = random.sample(PALETTE, group_count)

    patches: list[tuple[np.ndarray, np.ndarray]] = []
    for color in colors:
        pair = _make_group_patch(color)
        if pair is None:
            return None
        patches.append(pair)

    packed = _pack_patches(patches)
    if packed is None:
        return None
    inp, expected = packed

    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 14/2")
