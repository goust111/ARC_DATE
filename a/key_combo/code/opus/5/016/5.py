from __future__ import annotations

import random
from collections import deque

import numpy as np


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _find_color_components(grid: np.ndarray, color: int) -> list[dict]:
    H, W = grid.shape
    seen = np.zeros((H, W), dtype=bool)
    comps: list[dict] = []
    for r in range(H):
        for c in range(W):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < H and 0 <= nc < W and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            coords = np.array(cells, dtype=int)
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            comps.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(len(cells)),
                }
            )
    return comps


def _occupied_runs(mask_1d: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    i = 0
    n = int(mask_1d.shape[0])
    while i < n:
        if not bool(mask_1d[i]):
            i += 1
            continue
        j = i
        while j + 1 < n and bool(mask_1d[j + 1]):
            j += 1
        runs.append((i, j))
        i = j + 1
    return runs


def _rotate_mask(mask: np.ndarray, turns_cw: int) -> np.ndarray:
    turns_cw = int(turns_cw) % 4
    if turns_cw == 0:
        return mask.copy()
    return np.rot90(mask, -turns_cw).astype(bool)


def _apply_instructions(square: np.ndarray, instructions: list[dict]) -> np.ndarray:
    out = np.array(square, dtype=int)
    matched_colors = [int(inst["color"]) for inst in instructions]
    if matched_colors:
        out[np.isin(out, matched_colors)] = 0

    source_layers = {int(inst["color"]): square == int(inst["color"]) for inst in instructions}
    for inst in instructions:
        color = int(inst["color"])
        turns = int(inst["area"]) % 4
        rotated = _rotate_mask(source_layers[color], turns)
        out[rotated] = color
    return out


def _parse_input(grid: np.ndarray) -> tuple[list[dict], np.ndarray]:
    g = np.array(grid, dtype=int)
    col_runs = _occupied_runs(np.any(g != 0, axis=0))
    if len(col_runs) < 2:
        raise ValueError("expected separated glyph region and square region")
    square_cols = max(col_runs, key=lambda run: (run[1] - run[0] + 1, run[0]))
    c0, c1 = square_cols
    square_rows = np.where(np.any(g[:, c0 : c1 + 1] != 0, axis=1))[0]
    if square_rows.size == 0:
        raise ValueError("missing square rows")
    r0, r1 = int(square_rows[0]), int(square_rows[-1])
    square = g[r0 : r1 + 1, c0 : c1 + 1].copy()
    if square.shape[0] != square.shape[1] or square.shape[0] % 2 != 0:
        raise ValueError("square pattern must have even square bbox")

    left = g[:, :c0]
    glyphs: list[dict] = []
    for color in sorted(int(v) for v in np.unique(left) if int(v) != 0):
        for comp in _find_color_components(left, color):
            rr0, cc0, rr1, cc1 = comp["bbox"]
            if comp["area"] not in (4, 5, 6, 7):
                continue
            if rr1 - rr0 + 1 > 3 or cc1 - cc0 + 1 > 3:
                continue
            glyphs.append(comp)

    glyphs.sort(key=lambda comp: (comp["bbox"][0], comp["bbox"][1]))
    if not (2 <= len(glyphs) <= 4):
        raise ValueError("expected 2-4 glyphs")
    colors = [int(comp["color"]) for comp in glyphs]
    if len(set(colors)) != len(colors):
        raise ValueError("glyph colors must be distinct")
    return glyphs, square


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode each glyph's color and area, then rotate the matching color layer inside the square pattern.

    Input structure:
    - Black (0) is background.
    - The left side contains 2-4 vertically stacked 3x3 monochrome glyphs, each a 4-connected shape of area 4-7.
    - The right side contains one compact even-sized multicolor square pattern whose color set includes every glyph color.

    Rule:
    - Read the glyphs from top to bottom.
    - Each glyph encodes two things at once: its color selects the target color layer in the square, and its area A encodes a clockwise rotation of (A mod 4) * 90 degrees.
    - Rotate each matching color layer around the square's geometric center, composite the rotated layers back in reading order, let later layers overwrite earlier ones, and leave non-matching colors unchanged.
    - Output only the transformed square pattern, tightly cropped.

    Key insight: A single glyph is a dual-channel instruction object: visible color chooses which layer moves, while visible pixel count chooses how far that layer rotates.
    """
    glyphs, square = _parse_input(np.asarray(grid, dtype=int))
    transformed = _apply_instructions(square, glyphs)
    box = _bbox(transformed != 0)
    if box is None:
        return transformed[0:0, 0:0]
    return _crop_bbox(transformed, box)


def _connected_4(mask: np.ndarray) -> bool:
    cells = np.argwhere(mask)
    if cells.size == 0:
        return False
    start = tuple(int(v) for v in cells[0])
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and bool(mask[nr, nc]) and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == int(mask.sum())


def _enumerate_glyph_masks() -> dict[int, list[np.ndarray]]:
    out: dict[int, list[np.ndarray]] = {4: [], 5: [], 6: [], 7: []}
    for bits in range(1 << 9):
        mask = np.array([(bits >> i) & 1 for i in range(9)], dtype=bool).reshape(3, 3)
        area = int(mask.sum())
        if area not in out:
            continue
        if not _connected_4(mask):
            continue
        b = _bbox(mask)
        if b is None:
            continue
        r0, c0, r1, c1 = b
        if r1 - r0 + 1 > 3 or c1 - c0 + 1 > 3:
            continue
        if r1 == r0 or c1 == c0:
            continue
        out[area].append(mask)
    return out


_GLYPH_MASKS = _enumerate_glyph_masks()


def _sample_blob(available: np.ndarray, size: int, rng: random.Random) -> np.ndarray | None:
    coords = [tuple(int(v) for v in rc) for rc in np.argwhere(available)]
    if len(coords) < size:
        return None
    chosen = {coords[rng.randrange(len(coords))]}
    while len(chosen) < size:
        frontier = set()
        for r, c in chosen:
            for nr, nc in _neighbors8(r, c):
                if 0 <= nr < available.shape[0] and 0 <= nc < available.shape[1] and bool(available[nr, nc]) and (nr, nc) not in chosen:
                    frontier.add((nr, nc))
        if frontier and rng.random() < 0.8:
            pool = list(frontier)
        else:
            pool = [coord for coord in coords if coord not in chosen]
        if not pool:
            return None
        chosen.add(pool[rng.randrange(len(pool))])
    mask = np.zeros_like(available, dtype=bool)
    for r, c in chosen:
        mask[r, c] = True
    return mask


def _build_instruction_masks(n: int, turns: list[int], rng: random.Random) -> list[np.ndarray] | None:
    k = len(turns)
    pair_i, pair_j = rng.sample(range(k), 2)
    for _ in range(300):
        masks: list[np.ndarray | None] = [None] * k
        occupied = np.zeros((n, n), dtype=bool)

        pair_sizes = {pair_i: rng.randint(3, min(5, n)), pair_j: rng.randint(3, min(5, n))}
        ok_pair = False
        for _ in range(160):
            m1 = _sample_blob(~occupied, pair_sizes[pair_i], rng)
            if m1 is None:
                continue
            m2 = _sample_blob(~(occupied | m1), pair_sizes[pair_j], rng)
            if m2 is None:
                continue
            if turns[pair_i] and np.array_equal(_rotate_mask(m1, turns[pair_i]), m1):
                continue
            if turns[pair_j] and np.array_equal(_rotate_mask(m2, turns[pair_j]), m2):
                continue
            r1 = _rotate_mask(m1, turns[pair_i])
            r2 = _rotate_mask(m2, turns[pair_j])
            inter = int((r1 & r2).sum())
            if inter < 1:
                continue
            if not (r1 & ~r2).any():
                continue
            if not (r2 & ~r1).any():
                continue
            masks[pair_i] = m1
            masks[pair_j] = m2
            occupied |= m1 | m2
            ok_pair = True
            break
        if not ok_pair:
            continue

        ok = True
        for idx in range(k):
            if masks[idx] is not None:
                continue
            size = rng.randint(2, min(6, n))
            picked = None
            for _ in range(120):
                cand = _sample_blob(~occupied, size, rng)
                if cand is None:
                    continue
                if turns[idx] and np.array_equal(_rotate_mask(cand, turns[idx]), cand):
                    continue
                picked = cand
                break
            if picked is None:
                ok = False
                break
            masks[idx] = picked
            occupied |= picked
        if ok:
            return [mask for mask in masks if mask is not None]
    return None


def _fill_extra_colors(square: np.ndarray, colors: list[int], rng: random.Random) -> None:
    square[:, :] = int(colors[0])
    if len(colors) == 1:
        return
    for extra in colors[1:]:
        size = rng.randint(square.shape[0], max(square.shape[0] + 1, (square.size * 2) // 5))
        for _ in range(40):
            mask = _sample_blob(np.ones_like(square, dtype=bool), size, rng)
            if mask is None:
                continue
            square[mask] = int(extra)
            break


def _build_square(
    n: int,
    instruction_colors: list[int],
    areas: list[int],
    extra_colors: list[int],
    rng: random.Random,
) -> np.ndarray | None:
    turns = [int(area) % 4 for area in areas]
    source_masks = _build_instruction_masks(n, turns, rng)
    if source_masks is None:
        return None

    square = np.zeros((n, n), dtype=int)
    _fill_extra_colors(square, extra_colors, rng)
    for color, mask in zip(instruction_colors, source_masks):
        square[mask] = int(color)
    if any(int((square == int(color)).sum()) < 2 for color in instruction_colors):
        return None
    if any(int((square == int(color)).sum()) == 0 for color in extra_colors):
        return None
    return square


def _build_grid(
    square: np.ndarray,
    instruction_colors: list[int],
    areas: list[int],
    rng: random.Random,
) -> np.ndarray:
    glyph_masks = [rng.choice(_GLYPH_MASKS[int(area)]) for area in areas]
    gaps = [rng.randint(1, 2) for _ in range(len(glyph_masks) - 1)]
    stack_h = 3 * len(glyph_masks) + sum(gaps)
    left_margin = rng.randint(0, 2)
    right_margin = rng.randint(0, 2)
    gap_cols = rng.randint(2, 4)
    square_margin_top = rng.randint(0, 2)
    square_margin_bottom = rng.randint(0, 2)
    glyph_margin_top = rng.randint(0, 2)
    glyph_margin_bottom = rng.randint(0, 2)

    H = max(stack_h + glyph_margin_top + glyph_margin_bottom, square.shape[0] + square_margin_top + square_margin_bottom)
    W = left_margin + 3 + gap_cols + square.shape[1] + right_margin
    grid = np.zeros((H, W), dtype=int)

    square_r0 = rng.randint(0, H - square.shape[0])
    square_c0 = left_margin + 3 + gap_cols
    grid[square_r0 : square_r0 + square.shape[0], square_c0 : square_c0 + square.shape[1]] = square

    top = glyph_margin_top
    for idx, mask in enumerate(glyph_masks):
        color = int(instruction_colors[idx])
        r0 = top
        c0 = left_margin
        patch = np.zeros((3, 3), dtype=int)
        patch[mask] = color
        grid[r0 : r0 + 3, c0 : c0 + 3] = patch
        if idx + 1 < len(glyph_masks):
            top += 3 + gaps[idx]
    return grid


def _validate_generated(grid: np.ndarray) -> None:
    glyphs, square = _parse_input(grid)
    assert square.shape[0] <= 10 and square.shape[0] % 2 == 0
    colors = [int(g["color"]) for g in glyphs]
    areas = [int(g["area"]) for g in glyphs]
    turns = [area % 4 for area in areas]
    assert any(turn != 0 for turn in turns)

    extra_colors = [int(c) for c in np.unique(square) if int(c) not in set(colors) and int(c) != 0]
    assert extra_colors
    for color in colors:
        assert int((square == color).sum()) >= 2

    rotated_layers = [_rotate_mask(square == color, area % 4) for color, area in zip(colors, areas)]
    for layer, turn, color in zip(rotated_layers, turns, colors):
        if turn != 0:
            assert not np.array_equal(layer, square == color)

    order_sensitive = False
    for i in range(len(rotated_layers)):
        for j in range(i + 1, len(rotated_layers)):
            inter = rotated_layers[i] & rotated_layers[j]
            if inter.any() and (rotated_layers[i] & ~rotated_layers[j]).any() and (rotated_layers[j] & ~rotated_layers[i]).any():
                order_sensitive = True
    assert order_sensitive

    truth = main(grid)
    original_square = _crop_bbox(square, _bbox(square != 0))
    assert not np.array_equal(truth, original_square)

    reverse_truth = _crop_bbox(_apply_instructions(square, list(reversed(glyphs))), _bbox(_apply_instructions(square, list(reversed(glyphs))) != 0))
    assert not np.array_equal(truth, reverse_truth)

    no_rotation = [{"color": color, "area": 4} for color in colors]
    zero_truth_full = _apply_instructions(square, no_rotation)
    zero_truth = _crop_bbox(zero_truth_full, _bbox(zero_truth_full != 0))
    assert not np.array_equal(truth, zero_truth)

    shifted = [{"color": colors[(i + 1) % len(colors)], "area": areas[i]} for i in range(len(colors))]
    shifted_full = _apply_instructions(square, shifted)
    shifted_truth = _crop_bbox(shifted_full, _bbox(shifted_full != 0))
    assert not np.array_equal(truth, shifted_truth)


def generate_input() -> np.ndarray:
    for _ in range(500):
        rng = random.Random()
        n = rng.choice([6, 8, 10])
        glyph_count = rng.randint(2, 4)
        palette = rng.sample(list(range(1, 10)), glyph_count + rng.randint(1, 2))
        instruction_colors = palette[:glyph_count]
        extra_colors = palette[glyph_count:]
        areas = rng.sample([4, 5, 6, 7], glyph_count)
        square = _build_square(n, instruction_colors, areas, extra_colors, rng)
        if square is None:
            continue
        grid = _build_grid(square, instruction_colors, areas, rng)
        if max(grid.shape) > 30:
            continue
        try:
            _validate_generated(grid)
        except Exception:
            continue
        return grid
    raise RuntimeError("failed to generate valid input for dual-channel color-layer rotation task")
