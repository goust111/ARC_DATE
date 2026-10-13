from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAROON = 9

MAX_ATTEMPTS = 500


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.asarray(coords, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _touches_edge(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = shape
    return bool(
        np.any(coords[:, 0] == 0)
        or np.any(coords[:, 0] == h - 1)
        or np.any(coords[:, 1] == 0)
        or np.any(coords[:, 1] == w - 1)
    )


def _stamp_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    h, w = mask.shape
    patch = grid[top : top + h, left : left + w]
    assert patch.shape == mask.shape
    patch[mask] = int(color)


def _extract_summary(left: np.ndarray) -> np.ndarray:
    slots = []
    for coords in _components(left == BG):
        if _touches_edge(coords, left.shape):
            continue
        slots.append(_bbox(coords))

    assert len(slots) == 9, "expected a 3x3 table of enclosed slot regions"
    row_starts = sorted({box[0] for box in slots})
    col_starts = sorted({box[1] for box in slots})
    assert len(row_starts) == 3 and len(col_starts) == 3

    slot_by_anchor = {(r0, c0): (r0, c0, r1, c1) for r0, c0, r1, c1 in slots}
    assert len(slot_by_anchor) == 9

    out = np.zeros((3, 3), dtype=int)
    for i, r0 in enumerate(row_starts):
        for j, c0 in enumerate(col_starts):
            assert (r0, c0) in slot_by_anchor
            sr0, sc0, sr1, sc1 = slot_by_anchor[(r0, c0)]
            window = left[sr0 : sr1 + 1, sc0 : sc1 + 1]
            colors = sorted(set(map(int, window.ravel())) - {BG})
            assert len(colors) <= 1, "each slot may contain at most one object color"
            if colors:
                out[i, j] = colors[0]
    return out


def _sprite_layout(right: np.ndarray) -> tuple[np.ndarray, list[int], list[int]]:
    sprites = []
    for coords in _components(right == BLUE):
        r0, c0, r1, c1 = _bbox(coords)
        patch = right[r0 : r1 + 1, c0 : c1 + 1]
        sprites.append((r0, c0, patch == BLUE))

    assert len(sprites) >= 5, "expected several blue sprite instances"
    mask = sprites[0][2]
    for _r0, _c0, other in sprites[1:]:
        assert other.shape == mask.shape and np.array_equal(other, mask), "sprites must be identical"

    rows = sorted({r0 for r0, _c0, _mask in sprites})
    cols = sorted({c0 for _r0, c0, _mask in sprites})
    assert len(rows) == 3 and len(cols) == 3, "expected a 3x3 sprite lattice"

    occupied = {(r0, c0) for r0, c0, _mask in sprites}
    assert all(any((r, c) in occupied for c in cols) for r in rows)
    assert all(any((r, c) in occupied for r in rows) for c in cols)
    return mask, rows, cols


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete a 3x3 sprite lattice and recolor each sprite from the summary colors encoded in the table on the left.

    Input structure:
    - A black background is split by one full-height maroon separator column into a left table and a right sprite field.
    - The left side contains a grey 3x3 table whose enclosed black slots each hold either no colored pixel or one colored pixel.
    - The right side contains several identical blue 4-connected sprites placed on a 3x3 row-column lattice with some lattice positions missing.

    Rule:
    - Read the left table by locating its nine enclosed black slot regions in row-major order and summarizing each slot by its unique non-black color, or black if empty.
    - On the right side, find the common sprite mask and the three lattice row starts and three lattice column starts from the existing sprite positions.
    - Fill every lattice position with that sprite; recolor sprite (i, j) to the left summary color at (i, j) when it is non-black, otherwise keep it blue, and output only the completed right half.

    Key insight:
    - The left half contributes the color for each lattice slot while the right half contributes the shared sprite shape and the spatial lattice; only combining both halves yields the final answer.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    divider_cols = np.flatnonzero(np.all(g == MAROON, axis=0))
    assert divider_cols.size == 1, "expected a unique full-height separator column"
    divider = int(divider_cols[0])

    left = g[:, :divider]
    right = g[:, divider + 1 :]
    summary = _extract_summary(left)
    mask, rows, cols = _sprite_layout(right)

    out = np.zeros_like(right)
    for i, r0 in enumerate(rows):
        for j, c0 in enumerate(cols):
            color = int(summary[i, j]) if int(summary[i, j]) != BG else BLUE
            _stamp_mask(out, mask, r0, c0, color)
    return out


def _layout_table_axis(sizes: list[int]) -> tuple[list[int], int]:
    starts = []
    pos = 1
    for size in sizes:
        starts.append(pos)
        pos += size + 1
    return starts, pos


def _sample_summary() -> np.ndarray:
    out = np.zeros((3, 3), dtype=int)
    filled = random.sample(range(9), random.randint(5, 8))
    palette = [RED, GREEN, YELLOW]
    distinct = random.randint(2, min(3, len(filled)))
    chosen = random.sample(palette, distinct)
    colors = chosen + [random.choice(chosen) for _ in range(len(filled) - distinct)]
    random.shuffle(colors)
    for idx, color in zip(filled, colors):
        out[idx // 3, idx % 3] = int(color)
    return out


def _sample_axis_positions(size: int = 3) -> tuple[list[int], int]:
    gap1 = random.randint(4, 5)
    gap2 = random.randint(4, 5)
    margin0 = random.randint(1, 2)
    margin1 = random.randint(1, 2)
    starts = [margin0, margin0 + gap1, margin0 + gap1 + gap2]
    total = starts[-1] + size + margin1
    return starts, total


def _sample_occupied(summary: np.ndarray) -> set[tuple[int, int]] | None:
    all_pos = [(r, c) for r in range(3) for c in range(3)]
    for _ in range(200):
        missing_n = random.randint(2, 4)
        missing = set(random.sample(all_pos, missing_n))
        occupied = set(all_pos) - missing
        if not all(any((r, c) in occupied for c in range(3)) for r in range(3)):
            continue
        if not all(any((r, c) in occupied for r in range(3)) for c in range(3)):
            continue
        if not any(summary[r, c] != BG for r, c in occupied):
            continue
        if not any(summary[r, c] != BG for r, c in missing):
            continue
        return occupied
    return None


def _build_left_local(summary: np.ndarray) -> np.ndarray:
    row_heights = [random.randint(3, 4) for _ in range(3)]
    col_widths = [random.randint(3, 4) for _ in range(3)]
    row_starts, table_h = _layout_table_axis(row_heights)
    col_starts, table_w = _layout_table_axis(col_widths)
    margin_top = random.randint(1, 2)
    margin_bottom = random.randint(1, 2)
    margin_left = random.randint(1, 2)
    margin_right = random.randint(1, 2)

    grid = np.zeros((margin_top + table_h + margin_bottom, margin_left + table_w + margin_right), dtype=int)
    top = margin_top
    left = margin_left
    grid[top : top + table_h, left : left + table_w] = GREY

    for i, slot_top in enumerate(row_starts):
        for j, slot_left in enumerate(col_starts):
            h = row_heights[i]
            w = col_widths[j]
            r0 = top + slot_top
            c0 = left + slot_left
            grid[r0 : r0 + h, c0 : c0 + w] = BG
            color = int(summary[i, j])
            if color != BG:
                rr = random.randint(r0 + 1, r0 + h - 2)
                cc = random.randint(c0 + 1, c0 + w - 2)
                grid[rr, cc] = color
    return grid


def _mask_bank() -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    for bits in range(1, 1 << 9):
        mask = np.array([(bits >> i) & 1 for i in range(9)], dtype=bool).reshape(3, 3)
        area = int(mask.sum())
        if area < 4 or area > 6:
            continue
        coords = np.argwhere(mask)
        if coords.size == 0:
            continue
        if len({int(r) for r, _c in coords}) < 2 or len({int(c) for _r, c in coords}) < 2:
            continue
        r0, c0, r1, c1 = _bbox(coords)
        if area == (r1 - r0 + 1) * (c1 - c0 + 1):
            continue
        if len(_components(mask)) != 1:
            continue
        masks.append(mask.copy())
    return masks


SPRITE_BANK = _mask_bank()


def _build_right_locals(summary: np.ndarray, occupied: set[tuple[int, int]], mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    row_starts, h = _sample_axis_positions(mask.shape[0])
    col_starts, w = _sample_axis_positions(mask.shape[1])
    inp = np.zeros((h, w), dtype=int)
    out = np.zeros((h, w), dtype=int)
    for i, r0 in enumerate(row_starts):
        for j, c0 in enumerate(col_starts):
            color = int(summary[i, j]) if int(summary[i, j]) != BG else BLUE
            _stamp_mask(out, mask, r0, c0, color)
            if (i, j) in occupied:
                _stamp_mask(inp, mask, r0, c0, BLUE)
    return inp, out


def _embed_vertical(local: np.ndarray, height: int) -> np.ndarray:
    assert local.shape[0] <= height
    out = np.zeros((height, local.shape[1]), dtype=int)
    offset = random.randint(0, height - local.shape[0])
    out[offset : offset + local.shape[0], :] = local
    return out


def _embed_vertical_pair(local_a: np.ndarray, local_b: np.ndarray, height: int) -> tuple[np.ndarray, np.ndarray]:
    assert local_a.shape == local_b.shape and local_a.shape[0] <= height
    out_a = np.zeros((height, local_a.shape[1]), dtype=int)
    out_b = np.zeros((height, local_b.shape[1]), dtype=int)
    offset = random.randint(0, height - local_a.shape[0])
    out_a[offset : offset + local_a.shape[0], :] = local_a
    out_b[offset : offset + local_b.shape[0], :] = local_b
    return out_a, out_b


def _attempt_generate() -> np.ndarray | None:
    summary = _sample_summary()
    occupied = _sample_occupied(summary)
    if occupied is None:
        return None

    mask = random.choice(SPRITE_BANK)
    left_local = _build_left_local(summary)
    right_in_local, right_out_local = _build_right_locals(summary, occupied, mask)

    h_base = max(left_local.shape[0], right_in_local.shape[0])
    if h_base > 30:
        return None
    height = h_base + random.randint(0, min(2, 30 - h_base))

    left = _embed_vertical(left_local, height)
    right_in, right_out = _embed_vertical_pair(right_in_local, right_out_local, height)

    width = left.shape[1] + 1 + right_in.shape[1]
    if width > 30:
        return None

    divider = np.full((height, 1), MAROON, dtype=int)
    inp = np.concatenate([left, divider, right_in], axis=1)
    try:
        if np.array_equal(main(inp), right_out):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 9")
