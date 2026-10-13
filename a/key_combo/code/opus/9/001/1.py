from __future__ import annotations

import random

import numpy as np


BG = 0
RED = 2
GREY = 5
TEAL = 8
KS = (3, 4, 5)
MAX_GEN_ATTEMPTS = 400

_CANDIDATE_MASKS: dict[int, list[np.ndarray]] | None = None
_USED_MASKS: set[tuple[int, tuple[int, ...]]] = set()


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _separator_positions(size: int, k: int) -> list[int]:
    seps = [k + i * (k + 1) for i in range(k - 1)]
    assert size == k * k + (k - 1)
    return seps


def _slot_spans(seps: list[int], size: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    prev = -1
    for pos in [*seps, size]:
        start = prev + 1
        end = pos - 1
        assert end >= start
        spans.append((start, end))
        prev = pos
    return spans


def _parse_lattice(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]], int]:
    h, w = grid.shape
    sep_rows = [r for r in range(h) if np.all(grid[r, :] == GREY)]
    sep_cols = [c for c in range(w) if np.all(grid[:, c] == GREY)]
    row_spans = _slot_spans(sep_rows, h)
    col_spans = _slot_spans(sep_cols, w)
    k = len(row_spans)
    assert k == len(col_spans) >= 2
    heights = {r1 - r0 + 1 for r0, r1 in row_spans}
    widths = {c1 - c0 + 1 for c0, c1 in col_spans}
    assert len(heights) == 1 and len(widths) == 1
    slot_h = heights.pop()
    slot_w = widths.pop()
    assert slot_h == slot_w == k
    assert sep_rows == _separator_positions(h, k)
    assert sep_cols == _separator_positions(w, k)
    return row_spans, col_spans, k


def _partial_mask(grid: np.ndarray, row_spans: list[tuple[int, int]], col_spans: list[tuple[int, int]]) -> np.ndarray:
    k = len(row_spans)
    mask = np.zeros((k, k), dtype=bool)
    for r, (r0, r1) in enumerate(row_spans):
        for c, (c0, c1) in enumerate(col_spans):
            patch = grid[r0 : r1 + 1, c0 : c1 + 1]
            mask[r, c] = bool(np.any(patch == RED))
    return mask


def _rotation_closure(mask: np.ndarray) -> np.ndarray:
    full = np.zeros_like(mask, dtype=bool)
    for turns in range(4):
        full |= np.rot90(mask, turns)
    return full


def _render_from_mask(mask: np.ndarray) -> np.ndarray:
    k = int(mask.shape[0])
    size = k * k + (k - 1)
    out = np.zeros((size, size), dtype=int)
    for pos in _separator_positions(size, k):
        out[pos, :] = GREY
        out[:, pos] = GREY

    red_patch = np.zeros((k, k), dtype=int)
    red_patch[mask] = RED
    spans = _slot_spans(_separator_positions(size, k), size)
    for r in range(k):
        for c in range(k):
            if not mask[r, c]:
                continue
            patch = red_patch.copy()
            patch[r, c] = TEAL
            r0, r1 = spans[r]
            c0, c1 = spans[c]
            out[r0 : r1 + 1, c0 : c1 + 1] = patch
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Complete a rotationally implied slot mask, then stamp that same mask into every occupied slot.

    Input structure:
    - The grid is a square black canvas partitioned by 1-cell grey separator rows and columns into a k x k array of k x k slots.
    - Some slots contain red connected fragments while most slots are empty; fragment shape only indicates that the slot is occupied.
    - The occupied slots show one full quadrant of an unknown binary mask plus a few extra validation slots elsewhere.

    Rule:
    - Collapse each slot to occupied/unoccupied by checking whether it contains any red pixel.
    - Take the union of that partial slot mask with its 90/180/270 degree rotations to recover the full k x k mask.
    - For every foreground cell (r, c) of the recovered mask, fill the corresponding slot with one red copy of the whole mask and recolor the copy's local pixel (r, c) to teal.
    - Leave empty slots black and keep every grey separator line unchanged.

    Key insight:
    - The recovered mask plays two roles at once: rotation completion decides which slots are active, and that same completed mask is the stamp drawn inside each active slot.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] == g.shape[1]

    row_spans, col_spans, _k = _parse_lattice(g)
    partial = _partial_mask(g, row_spans, col_spans)
    full = _rotation_closure(partial)
    assert np.any(full)
    return _render_from_mask(full)


def _rot90_coord(k: int, rc: tuple[int, int]) -> tuple[int, int]:
    r, c = rc
    return c, k - 1 - r


def _all_orbits(k: int) -> list[tuple[tuple[int, int], ...]]:
    seen: set[tuple[tuple[int, int], ...]] = set()
    orbits: list[tuple[tuple[int, int], ...]] = []
    for r in range(k):
        for c in range(k):
            orbit = []
            cur = (r, c)
            for _ in range(4):
                orbit.append(cur)
                cur = _rot90_coord(k, cur)
            key = tuple(sorted(set(orbit)))
            if key not in seen:
                seen.add(key)
                orbits.append(key)
    orbits.sort(key=lambda orb: (len(orb), orb))
    return orbits


def _main_quadrant_cells(k: int) -> set[tuple[int, int]]:
    limit = (k + 1) // 2
    return {(r, c) for r in range(limit) for c in range(limit)}


def _extra_count(mask: np.ndarray) -> int:
    total = int(mask.sum())
    return 2 if total % 2 == 0 else 1


def _visible_cells(mask: np.ndarray) -> list[tuple[int, int]]:
    k = int(mask.shape[0])
    quad = _main_quadrant_cells(k)
    primary = [(r, c) for r in range(k) for c in range(k) if mask[r, c] and (r, c) in quad]
    outside = [(r, c) for r in range(k) for c in range(k) if mask[r, c] and (r, c) not in quad]
    extra_n = min(_extra_count(mask), max(0, len(outside) - 1))
    return primary + outside[:extra_n]


def _visible_mask(mask: np.ndarray) -> np.ndarray:
    vis = np.zeros_like(mask, dtype=bool)
    for r, c in _visible_cells(mask):
        vis[r, c] = True
    return vis


def _mask_signature(mask: np.ndarray) -> tuple[int, tuple[int, ...]]:
    return int(mask.shape[0]), tuple(int(v) for v in mask.astype(int).ravel())


def _mask_seed(mask: np.ndarray) -> int:
    seed = int(mask.shape[0]) * 1009
    for bit in mask.astype(int).ravel():
        seed = (seed * 131 + int(bit) + 17) & 0xFFFFFFFF
    return seed


def _connected_fragment(k: int, seed: int) -> np.ndarray:
    rng = random.Random(seed)
    max_cells = min(8, k * k - 1)
    target = 3 + (seed % max(1, max_cells - 2))
    for _ in range(60):
        start = (rng.randrange(k), rng.randrange(k))
        cells = {start}
        while len(cells) < target:
            frontier = sorted(
                {
                    nbr
                    for r, c in cells
                    for nbr in _neighbors4(r, c)
                    if 0 <= nbr[0] < k and 0 <= nbr[1] < k and nbr not in cells
                }
            )
            if not frontier:
                break
            cells.add(frontier[rng.randrange(len(frontier))])
        if len(cells) != target:
            continue
        frag = np.zeros((k, k), dtype=bool)
        for r, c in cells:
            frag[r, c] = True
        if not np.array_equal(frag, np.rot90(frag)):
            return frag
    frag = np.zeros((k, k), dtype=bool)
    frag[0, 0] = True
    frag[0, 1] = True
    frag[1, 0] = True
    return frag


def _build_input(mask: np.ndarray) -> np.ndarray:
    k = int(mask.shape[0])
    size = k * k + (k - 1)
    inp = np.zeros((size, size), dtype=int)
    for pos in _separator_positions(size, k):
        inp[pos, :] = GREY
        inp[:, pos] = GREY

    fragment = _connected_fragment(k, _mask_seed(mask))
    spans = _slot_spans(_separator_positions(size, k), size)
    for r, c in _visible_cells(mask):
        r0, r1 = spans[r]
        c0, c1 = spans[c]
        patch = np.zeros((k, k), dtype=int)
        patch[fragment] = RED
        inp[r0 : r1 + 1, c0 : c1 + 1] = patch
    return inp


def _mask_is_semantic(mask: np.ndarray) -> bool:
    k = int(mask.shape[0])
    if np.array_equal(mask, np.zeros_like(mask)) or np.all(mask):
        return False

    visible = _visible_mask(mask)
    total = int(mask.sum())
    shown = int(visible.sum())
    if total < 4 or shown < 2 or shown >= total:
        return False

    quad = _main_quadrant_cells(k)
    primary = sum(1 for r, c in np.argwhere(mask) if (int(r), int(c)) in quad)
    outside = total - primary
    if primary < 2 or outside < 2:
        return False
    if outside <= _extra_count(mask):
        return False

    fill_ratio = total / float(k * k)
    if not (0.18 <= fill_ratio <= 0.72):
        return False

    if not np.array_equal(mask, _rotation_closure(mask)):
        return False
    if not np.array_equal(mask, _rotation_closure(visible)):
        return False
    if np.array_equal(mask, visible):
        return False

    shortcut = _render_from_mask(visible)
    target = _render_from_mask(mask)
    if np.array_equal(shortcut, target):
        return False

    return True


def _enumerate_masks() -> dict[int, list[np.ndarray]]:
    masks_by_k: dict[int, list[np.ndarray]] = {}
    for k in KS:
        valid: list[np.ndarray] = []
        orbits = _all_orbits(k)
        size4_ids = [idx for idx, orbit in enumerate(orbits) if len(orbit) == 4]
        for bits in range(1, 1 << len(orbits)):
            if not any(bits & (1 << idx) for idx in size4_ids):
                continue
            mask = np.zeros((k, k), dtype=bool)
            for idx, orbit in enumerate(orbits):
                if bits & (1 << idx):
                    for r, c in orbit:
                        mask[r, c] = True
            if _mask_is_semantic(mask):
                valid.append(mask)
        masks_by_k[k] = valid
    return masks_by_k


def _ensure_candidates() -> dict[int, list[np.ndarray]]:
    global _CANDIDATE_MASKS
    if _CANDIDATE_MASKS is None:
        _CANDIDATE_MASKS = _enumerate_masks()
    return _CANDIDATE_MASKS


def _generated_case_is_valid(inp: np.ndarray, mask: np.ndarray) -> bool:
    expected = _render_from_mask(mask)
    try:
        actual = main(inp)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False

    partial = _partial_mask(inp, *_parse_lattice(inp)[:2])
    if not np.array_equal(_rotation_closure(partial), mask):
        return False

    visible = _visible_mask(mask)
    if np.array_equal(visible, mask):
        return False
    if np.array_equal(_render_from_mask(visible), expected):
        return False
    return True


def generate_input() -> np.ndarray:
    candidates = _ensure_candidates()
    for _ in range(MAX_GEN_ATTEMPTS):
        ks = [k for k, masks in candidates.items() if any(_mask_signature(mask) not in _USED_MASKS for mask in masks)]
        if not ks:
            _USED_MASKS.clear()
            ks = [k for k, masks in candidates.items() if masks]
        k = random.choice(ks)
        available = [mask for mask in candidates[k] if _mask_signature(mask) not in _USED_MASKS]
        if not available:
            continue
        mask = random.choice(available)
        inp = _build_input(mask)
        if _generated_case_is_valid(inp, mask):
            _USED_MASKS.add(_mask_signature(mask))
            return inp
    raise RuntimeError("failed to generate a valid rotational lattice-stamping input")
