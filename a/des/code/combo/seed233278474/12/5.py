from __future__ import annotations

from collections import defaultdict, deque
import random

import numpy as np


BG = 8  # azure
PALETTE = (1, 2, 3, 4, 5, 6, 7, 9)

PAIR_COUNT_RANGE = (2, 3)
INPUT_HEIGHT_RANGE = (18, 26)
INPUT_WIDTH_RANGE = (20, 30)
PIECE_GAP = 1
GEN_RETRIES = 300


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _extract_components(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == BG:
                continue
            color = int(g[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))

            r0, c0, r1, c1 = _bbox(coords)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for rr, cc in coords:
                mask[rr - r0, cc - c0] = True
            comps.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "mask": mask,
                }
            )

    return comps


def _side_candidate(mask: np.ndarray, side: str) -> tuple[str, tuple[int, ...]] | None:
    h, w = mask.shape
    full = np.ones_like(mask, dtype=bool)

    if side == "left":
        line = mask[:, 0]
        if line.all() or not line.any():
            return None
        full[:, 0] = line
        bits = tuple(int(not cell) for cell in line)
    elif side == "right":
        line = mask[:, w - 1]
        if line.all() or not line.any():
            return None
        full[:, w - 1] = line
        bits = tuple(int(cell) for cell in line)
    elif side == "top":
        line = mask[0, :]
        if line.all() or not line.any():
            return None
        full[0, :] = line
        bits = tuple(int(not cell) for cell in line)
    elif side == "bottom":
        line = mask[h - 1, :]
        if line.all() or not line.any():
            return None
        full[h - 1, :] = line
        bits = tuple(int(cell) for cell in line)
    else:
        raise ValueError(side)

    if np.array_equal(full, mask):
        return side, bits
    return None


def _analyze_piece(comp: dict[str, object]) -> dict[str, object]:
    mask = np.asarray(comp["mask"], dtype=bool)
    candidates = []
    for side in ("left", "right", "top", "bottom"):
        cand = _side_candidate(mask, side)
        if cand is not None:
            candidates.append(cand)

    assert len(candidates) == 1, "expected exactly one jagged edge"
    side, bits = candidates[0]

    patch = np.full(mask.shape, BG, dtype=int)
    patch[mask] = int(comp["color"])
    out = dict(comp)
    out["patch"] = patch
    out["side"] = side
    out["bits"] = bits
    return out


def _paste(canvas: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    mask = patch != BG
    view = canvas[r0 : r0 + h, c0 : c0 + w]
    assert not np.any((view != BG) & mask), "merged pieces must not overlap"
    view[mask] = patch[mask]


def _merge_pair(a: dict[str, object], b: dict[str, object]) -> np.ndarray:
    side_a = str(a["side"])
    side_b = str(b["side"])
    pair_sides = {side_a, side_b}

    if pair_sides == {"right", "left"}:
        left_piece = a if side_a == "right" else b
        right_piece = b if side_a == "right" else a
        left_patch = np.asarray(left_piece["patch"], dtype=int)
        right_patch = np.asarray(right_piece["patch"], dtype=int)
        h = left_patch.shape[0]
        assert right_patch.shape[0] == h
        out = np.full((h, left_patch.shape[1] + right_patch.shape[1] - 1), BG, dtype=int)
        _paste(out, left_patch, 0, 0)
        _paste(out, right_patch, 0, left_patch.shape[1] - 1)
    elif pair_sides == {"bottom", "top"}:
        top_piece = a if side_a == "bottom" else b
        bottom_piece = b if side_a == "bottom" else a
        top_patch = np.asarray(top_piece["patch"], dtype=int)
        bottom_patch = np.asarray(bottom_piece["patch"], dtype=int)
        w = top_patch.shape[1]
        assert bottom_patch.shape[1] == w
        out = np.full((top_patch.shape[0] + bottom_patch.shape[0] - 1, w), BG, dtype=int)
        _paste(out, top_patch, 0, 0)
        _paste(out, bottom_patch, top_patch.shape[0] - 1, 0)
    else:
        raise AssertionError("pair must be left-right or top-bottom")

    assert np.all(out != BG), "paired pieces must reconstruct a full rectangle"
    return out


def _component_key(piece: dict[str, object]) -> tuple[int, int]:
    r0, c0, _r1, _c1 = piece["bbox"]
    return int(c0), int(r0)


def _pair_order_key(a: dict[str, object], b: dict[str, object]) -> tuple[int, int]:
    r0a, c0a, _r1a, _c1a = a["bbox"]
    r0b, c0b, _r1b, _c1b = b["bbox"]
    return min(int(c0a), int(c0b)), min(int(r0a), int(r0b))


def _pack_merged(items: list[tuple[tuple[int, int], np.ndarray]]) -> np.ndarray:
    assert items
    items = sorted(items, key=lambda item: item[0])
    patches = [patch for _key, patch in items]
    out_h = max(patch.shape[0] for patch in patches)
    out_w = sum(patch.shape[1] for patch in patches) + len(patches) - 1
    out = np.full((out_h, out_w), BG, dtype=int)

    col = 0
    for patch in patches:
        _paste(out, patch, 0, col)
        col += patch.shape[1] + 1

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair monochrome fragments by complementary jagged edges and reassemble them into rectangles.

    The output packs the reconstructed rectangles from left to right using the pairs'
    leftmost input positions as order, with one azure separator column and top alignment.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    pieces = [_analyze_piece(comp) for comp in _extract_components(g)]
    assert len(pieces) >= 2 and len(pieces) % 2 == 0

    left_by_bits: defaultdict[tuple[int, ...], list[dict[str, object]]] = defaultdict(list)
    right_by_bits: defaultdict[tuple[int, ...], list[dict[str, object]]] = defaultdict(list)
    top_by_bits: defaultdict[tuple[int, ...], list[dict[str, object]]] = defaultdict(list)
    bottom_by_bits: defaultdict[tuple[int, ...], list[dict[str, object]]] = defaultdict(list)

    for piece in pieces:
        side = str(piece["side"])
        bits = tuple(int(x) for x in piece["bits"])
        if side == "left":
            left_by_bits[bits].append(piece)
        elif side == "right":
            right_by_bits[bits].append(piece)
        elif side == "top":
            top_by_bits[bits].append(piece)
        elif side == "bottom":
            bottom_by_bits[bits].append(piece)
        else:
            raise AssertionError(side)

    merged_items: list[tuple[tuple[int, int], np.ndarray]] = []

    v_keys = set(left_by_bits) | set(right_by_bits)
    for bits in v_keys:
        lefts = sorted(left_by_bits.get(bits, []), key=_component_key)
        rights = sorted(right_by_bits.get(bits, []), key=_component_key)
        assert len(lefts) == len(rights), "vertical jagged patterns must pair uniquely"
        for right_piece, left_piece in zip(rights, lefts):
            merged = _merge_pair(right_piece, left_piece)
            merged_items.append((_pair_order_key(right_piece, left_piece), merged))

    h_keys = set(top_by_bits) | set(bottom_by_bits)
    for bits in h_keys:
        tops = sorted(top_by_bits.get(bits, []), key=_component_key)
        bottoms = sorted(bottom_by_bits.get(bits, []), key=_component_key)
        assert len(tops) == len(bottoms), "horizontal jagged patterns must pair uniquely"
        for bottom_piece, top_piece in zip(bottoms, tops):
            merged = _merge_pair(bottom_piece, top_piece)
            merged_items.append((_pair_order_key(bottom_piece, top_piece), merged))

    assert len(merged_items) * 2 == len(pieces), "every piece must be paired exactly once"
    return _pack_merged(merged_items)


def _sample_bits(length: int, orientation: str, used: set[tuple[str, tuple[int, ...]]]) -> tuple[int, ...]:
    for _ in range(200):
        ones = random.randint(2, length - 2)
        bits = [1] * ones + [0] * (length - ones)
        random.shuffle(bits)
        key = (orientation, tuple(bits))
        if key not in used:
            used.add(key)
            return tuple(bits)
    raise RuntimeError("failed to sample a unique jagged pattern")


def _mask_to_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.full(mask.shape, BG, dtype=int)
    patch[mask] = color
    return patch


def _build_vertical_pair(bits: tuple[int, ...], color_a: int, color_b: int) -> dict[str, object]:
    h = len(bits)
    left_w = random.randint(4, 7)
    right_w = random.randint(4, 7)

    left_mask = np.ones((h, left_w), dtype=bool)
    right_mask = np.ones((h, right_w), dtype=bool)
    for r, bit in enumerate(bits):
        if bit == 0:
            left_mask[r, left_w - 1] = False
        else:
            right_mask[r, 0] = False

    left_patch = _mask_to_patch(left_mask, color_a)
    right_patch = _mask_to_patch(right_mask, color_b)
    merged = np.full((h, left_w + right_w - 1), BG, dtype=int)
    _paste(merged, left_patch, 0, 0)
    _paste(merged, right_patch, 0, left_w - 1)

    return {
        "pieces": [
            {"patch": left_patch, "side": "right", "bits": bits},
            {"patch": right_patch, "side": "left", "bits": bits},
        ],
        "merged": merged,
    }


def _build_horizontal_pair(bits: tuple[int, ...], color_a: int, color_b: int) -> dict[str, object]:
    w = len(bits)
    top_h = random.randint(3, 5)
    bottom_h = random.randint(3, 5)

    top_mask = np.ones((top_h, w), dtype=bool)
    bottom_mask = np.ones((bottom_h, w), dtype=bool)
    for c, bit in enumerate(bits):
        if bit == 0:
            top_mask[top_h - 1, c] = False
        else:
            bottom_mask[0, c] = False

    top_patch = _mask_to_patch(top_mask, color_a)
    bottom_patch = _mask_to_patch(bottom_mask, color_b)
    merged = np.full((top_h + bottom_h - 1, w), BG, dtype=int)
    _paste(merged, top_patch, 0, 0)
    _paste(merged, bottom_patch, top_h - 1, 0)

    return {
        "pieces": [
            {"patch": top_patch, "side": "bottom", "bits": bits},
            {"patch": bottom_patch, "side": "top", "bits": bits},
        ],
        "merged": merged,
    }


def _can_place(blocked: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> bool:
    h, w = patch.shape
    return not blocked[r0 : r0 + h, c0 : c0 + w].any()


def _mark_blocked(blocked: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    rs = max(0, r0 - PIECE_GAP)
    re = min(blocked.shape[0], r0 + h + PIECE_GAP)
    cs = max(0, c0 - PIECE_GAP)
    ce = min(blocked.shape[1], c0 + w + PIECE_GAP)
    blocked[rs:re, cs:ce] = True


def _place_piece(grid: np.ndarray, blocked: np.ndarray, patch: np.ndarray) -> tuple[int, int] | None:
    h, w = patch.shape
    for _ in range(200):
        r0 = random.randint(0, grid.shape[0] - h)
        c0 = random.randint(0, grid.shape[1] - w)
        if not _can_place(blocked, patch, r0, c0):
            continue
        mask = patch != BG
        grid[r0 : r0 + h, c0 : c0 + w][mask] = patch[mask]
        _mark_blocked(blocked, patch, r0, c0)
        return r0, c0
    return None


def _expected_output(pair_layouts: list[dict[str, object]]) -> np.ndarray:
    merged_items = []
    for pair in pair_layouts:
        merged_items.append((pair["order_key"], np.asarray(pair["merged"], dtype=int)))
    return _pack_merged(merged_items)


def _generate_once() -> np.ndarray | None:
    n_pairs = random.randint(*PAIR_COUNT_RANGE)
    colors = random.sample(PALETTE, 2 * n_pairs)
    used_patterns: set[tuple[str, tuple[int, ...]]] = set()

    pair_defs: list[dict[str, object]] = []
    for i in range(n_pairs):
        color_a = colors[2 * i]
        color_b = colors[2 * i + 1]
        if random.random() < 0.5:
            bits = _sample_bits(random.randint(4, 5), "V", used_patterns)
            pair_defs.append(_build_vertical_pair(bits, color_a, color_b))
        else:
            bits = _sample_bits(random.randint(4, 7), "H", used_patterns)
            pair_defs.append(_build_horizontal_pair(bits, color_a, color_b))

    grid_h = random.randint(*INPUT_HEIGHT_RANGE)
    grid_w = random.randint(*INPUT_WIDTH_RANGE)
    grid = np.full((grid_h, grid_w), BG, dtype=int)
    blocked = np.zeros((grid_h, grid_w), dtype=bool)

    pair_layouts: list[dict[str, object]] = []
    pieces_to_place: list[dict[str, object]] = []
    for pair_id, pair_def in enumerate(pair_defs):
        entry = {"pair_id": pair_id, "merged": pair_def["merged"], "positions": []}
        pair_layouts.append(entry)
        for piece in pair_def["pieces"]:
            pieces_to_place.append(
                {
                    "pair_id": pair_id,
                    "patch": np.asarray(piece["patch"], dtype=int),
                }
            )

    random.shuffle(pieces_to_place)
    for piece in pieces_to_place:
        placed = _place_piece(grid, blocked, np.asarray(piece["patch"], dtype=int))
        if placed is None:
            return None
        r0, c0 = placed
        h, w = np.asarray(piece["patch"]).shape
        pair_layouts[piece["pair_id"]]["positions"].append((r0, c0, r0 + h - 1, c0 + w - 1))

    for pair in pair_layouts:
        positions = pair["positions"]
        c0 = min(pos[1] for pos in positions)
        r0 = min(pos[0] for pos in positions)
        pair["order_key"] = (c0, r0)

    expected = _expected_output(pair_layouts)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_once()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
