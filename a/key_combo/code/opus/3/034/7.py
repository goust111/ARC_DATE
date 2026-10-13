from __future__ import annotations

import itertools
import random

import numpy as np


BG = 0
ORANGE = 7
PROTO = 8
SLOT = 4
STEP = SLOT + 1
NON_PROTO_COLORS = (1, 2, 3, 4, 5, 6, 9)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_mask(mask: np.ndarray) -> np.ndarray | None:
    box = _bbox(mask)
    if box is None:
        return None
    r0, c0, r1, c1 = box
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _expand_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros((2 * h - 1, w), dtype=bool)
    out[:h, :] |= mask
    out[h - 1 :, :] |= mask[::-1, :]
    return out


def _signature(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    cropped = _crop_mask(mask)
    assert cropped is not None
    expanded = _crop_mask(_expand_mask(cropped))
    assert expanded is not None
    return tuple(map(int, expanded.shape)), expanded.tobytes()


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    stack = [start]
    seen = {start}
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _build_shape_library() -> list[np.ndarray]:
    shapes: list[np.ndarray] = []
    for h, w in itertools.product((2, 3), (2, 3, 4)):
        for bits in range(1, 1 << (h * w)):
            flat = np.array([(bits >> i) & 1 for i in range(h * w)], dtype=bool)
            mask = flat.reshape(h, w)
            area = int(mask.sum())
            if not 3 <= area <= 7:
                continue
            if not mask.any(axis=0).all() or not mask.any(axis=1).all():
                continue
            if not _is_connected(mask):
                continue
            shapes.append(mask.copy())
    shapes.sort(key=lambda x: (int(x.sum()), x.shape[0], x.shape[1], x.tobytes()))
    return shapes


SHAPE_LIBRARY = _build_shape_library()


def _table_dims(grid: np.ndarray) -> tuple[int, int]:
    h, w = map(int, grid.shape)
    assert (h - 1) % STEP == 0 and (w - 1) % STEP == 0
    n_rows = (h - 1) // STEP
    n_cols = (w - 1) // STEP
    assert 4 <= n_rows <= 5 and 4 <= n_cols <= 5
    for r in range(0, h, STEP):
        assert np.all(grid[r, :] == BG)
    for c in range(0, w, STEP):
        assert np.all(grid[:, c] == BG)
    return n_rows, n_cols


def _slot_bbox(slot_r: int, slot_c: int) -> tuple[int, int, int, int]:
    r0 = 1 + slot_r * STEP
    c0 = 1 + slot_c * STEP
    return r0, c0, r0 + SLOT - 1, c0 + SLOT - 1


def _extract_slots(grid: np.ndarray) -> tuple[dict[tuple[int, int], dict], int, int]:
    n_rows, n_cols = _table_dims(grid)
    slots: dict[tuple[int, int], dict] = {}
    for sr in range(n_rows):
        for sc in range(n_cols):
            r0, c0, r1, c1 = _slot_bbox(sr, sc)
            patch = grid[r0 : r1 + 1, c0 : c1 + 1]
            mask = patch != BG
            colors = [int(v) for v in np.unique(patch[mask])]
            assert len(colors) <= 1
            slots[(sr, sc)] = {
                "bbox": (r0, c0, r1, c1),
                "mask": mask.copy(),
                "empty": not bool(mask.any()),
                "color": None if not colors else colors[0],
                "signature": None if not colors else _signature(mask),
            }
    return slots, n_rows, n_cols


def _recolor_slot(out: np.ndarray, slot: dict, color: int) -> None:
    r0, c0, r1, c1 = slot["bbox"]
    view = out[r0 : r1 + 1, c0 : c1 + 1]
    view[slot["mask"]] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor slot patterns by whether their vertically mirrored overlap signature matches the cyan prototype.

    Input structure:
    - A black single-pixel lattice splits the grid into a 4x4 or 5x5 table of 4x4 slots.
    - Each occupied slot contains one small monochrome pattern; empty slots stay black.
    - Exactly one occupied slot uses cyan and serves as the prototype, while other occupied slots use other colors.

    Rule:
    - Crop each occupied slot to the pattern's minimal bbox, then build its hidden signature by stacking the pattern with its vertical flip while overlapping one row.
    - Any slot whose expanded binary signature matches the cyan prototype's signature is recolored cyan.
    - In every row and every column of slots, recolor every non-empty slot strictly between two adjacent matching slots to orange.

    Key insight:
    - The visible pattern is not matched directly; the deciding feature is the unseen mask produced after the overlap-with-vertical-flip expansion.
    """
    g = np.asarray(grid, dtype=int)
    slots, n_rows, n_cols = _extract_slots(g)

    proto_slots = [pos for pos, info in slots.items() if info["color"] == PROTO]
    assert len(proto_slots) == 1
    proto_sig = slots[proto_slots[0]]["signature"]
    assert proto_sig is not None

    match_slots = {
        pos for pos, info in slots.items() if info["signature"] is not None and info["signature"] == proto_sig
    }
    out = g.copy()
    for pos in match_slots:
        _recolor_slot(out, slots[pos], PROTO)

    orange_slots: set[tuple[int, int]] = set()
    for sr in range(n_rows):
        cols = sorted(sc for rr, sc in match_slots if rr == sr)
        for left, right in zip(cols, cols[1:]):
            for sc in range(left + 1, right):
                pos = (sr, sc)
                if not slots[pos]["empty"] and pos not in match_slots:
                    orange_slots.add(pos)

    for sc in range(n_cols):
        rows = sorted(sr for sr, cc in match_slots if cc == sc)
        for top, bottom in zip(rows, rows[1:]):
            for sr in range(top + 1, bottom):
                pos = (sr, sc)
                if not slots[pos]["empty"] and pos not in match_slots:
                    orange_slots.add(pos)

    for pos in orange_slots:
        _recolor_slot(out, slots[pos], ORANGE)
    return out


def _slot_patch(shape: np.ndarray, color: int) -> np.ndarray:
    h, w = shape.shape
    r0 = random.randint(0, SLOT - h)
    c0 = random.randint(0, SLOT - w)
    patch = np.zeros((SLOT, SLOT), dtype=int)
    view = patch[r0 : r0 + h, c0 : c0 + w]
    view[shape] = int(color)
    return patch


def _valid_line_triples(n: int) -> list[tuple[int, int, int]]:
    triples = []
    for combo in itertools.combinations(range(n), 3):
        if any(b - a >= 2 for a, b in zip(combo, combo[1:])):
            triples.append(tuple(map(int, combo)))
    return triples


def _line_layout(n: int, orientation: str) -> tuple[list[tuple[int, int]], list[tuple[int, int]], tuple[int, int]]:
    triples = _valid_line_triples(n)
    assert triples
    picks = random.choice(triples)
    if orientation == "row":
        row = random.randrange(n)
        matches = [(row, c) for c in picks]
        gaps = [(row, c) for a, b in zip(picks, picks[1:]) for c in range(a + 1, b)]
    else:
        col = random.randrange(n)
        matches = [(r, col) for r in picks]
        gaps = [(r, col) for a, b in zip(picks, picks[1:]) for r in range(a + 1, b)]
    proto = random.choice(matches)
    return matches, gaps, proto


def _cross_layout(n: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]], tuple[int, int]]:
    proto = (random.randrange(n), random.randrange(n))
    pr, pc = proto
    row_targets = [c for c in range(n) if abs(c - pc) >= 2]
    col_targets = [r for r in range(n) if abs(r - pr) >= 2]
    assert row_targets and col_targets
    row_match = (pr, random.choice(row_targets))
    col_match = (random.choice(col_targets), pc)

    row_left, row_right = sorted((pc, row_match[1]))
    col_top, col_bottom = sorted((pr, col_match[0]))
    gaps = [(pr, c) for c in range(row_left + 1, row_right)]
    gaps += [(r, pc) for r in range(col_top + 1, col_bottom)]
    matches = [proto, row_match, col_match]
    return matches, gaps, proto


def _render_slots(n: int, patches: dict[tuple[int, int], np.ndarray]) -> np.ndarray:
    size = 1 + n * STEP
    grid = np.zeros((size, size), dtype=int)
    for (sr, sc), patch in patches.items():
        r0, c0, r1, c1 = _slot_bbox(sr, sc)
        assert patch.shape == (SLOT, SLOT)
        grid[r0 : r1 + 1, c0 : c1 + 1] = patch
    return grid


def _expected_output(inp: np.ndarray, match_slots: list[tuple[int, int]], orange_slots: list[tuple[int, int]]) -> np.ndarray:
    out = inp.copy()
    slots, _, _ = _extract_slots(inp)
    for pos in match_slots:
        _recolor_slot(out, slots[pos], PROTO)
    for pos in orange_slots:
        _recolor_slot(out, slots[pos], ORANGE)
    return out


def generate_input() -> np.ndarray:
    for _ in range(400):
        n = random.choice((4, 5))
        mode = random.choice(("line_row", "line_col", "cross"))
        if mode == "line_row":
            match_slots, orange_slots, proto_slot = _line_layout(n, "row")
        elif mode == "line_col":
            match_slots, orange_slots, proto_slot = _line_layout(n, "col")
        else:
            match_slots, orange_slots, proto_slot = _cross_layout(n)

        occupied = set(match_slots) | set(orange_slots)
        remaining = [(r, c) for r in range(n) for c in range(n) if (r, c) not in occupied]
        non_proto_used = len(occupied) - 1
        max_extra = min(2, len(remaining), len(NON_PROTO_COLORS) - non_proto_used)
        extra_count = random.randint(0, max_extra) if max_extra > 0 else 0
        extras = random.sample(remaining, extra_count)
        occupied |= set(extras)

        if len(occupied) - 1 > len(NON_PROTO_COLORS):
            continue

        proto_shape = random.choice(SHAPE_LIBRARY)
        proto_sig = _signature(proto_shape)
        distractor_pool = [shape for shape in SHAPE_LIBRARY if _signature(shape) != proto_sig]
        distractor_count = len(occupied) - len(match_slots)
        if len(distractor_pool) < distractor_count:
            continue

        colors = list(random.sample(NON_PROTO_COLORS, len(occupied) - 1))
        patches: dict[tuple[int, int], np.ndarray] = {proto_slot: _slot_patch(proto_shape, PROTO)}

        color_idx = 0
        for pos in match_slots:
            if pos == proto_slot:
                continue
            patches[pos] = _slot_patch(proto_shape, colors[color_idx])
            color_idx += 1

        distractor_slots = [pos for pos in occupied if pos not in match_slots]
        distractor_shapes = random.sample(distractor_pool, distractor_count)
        for pos, shape in zip(distractor_slots, distractor_shapes):
            patches[pos] = _slot_patch(shape, colors[color_idx])
            color_idx += 1

        inp = _render_slots(n, patches)
        expected = _expected_output(inp, match_slots, orange_slots)
        actual = main(inp)
        if not np.array_equal(actual, expected):
            continue
        return inp

    raise RuntimeError("failed to generate a valid puzzle instance")
