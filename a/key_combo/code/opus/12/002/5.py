from __future__ import annotations

from collections import Counter, deque
import itertools

import numpy as np


BG = 0
BLUE = 1
RED = 2
FRAME = 4
NOISE_COLORS = (1, 2, 3, 5, 6, 7, 8, 9)
CARDINALS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_RETRY_LIMIT = 400
_RNG = np.random.default_rng()


def _mode_color(values: list[int]) -> int:
    counts = Counter(map(int, values))
    ranked = counts.most_common()
    assert ranked
    if len(ranked) > 1:
        assert ranked[0][1] > ranked[1][1], f"majority vote tie: {ranked}"
    return int(ranked[0][0])


def _scale_xy(sprite: np.ndarray, sx: int, sy: int) -> np.ndarray:
    return np.repeat(np.repeat(np.asarray(sprite, dtype=int), sy, axis=0), sx, axis=1)


def _select_dense_bands(counts: np.ndarray) -> list[tuple[int, int]]:
    counts = np.asarray(counts, dtype=int)
    n = int(counts.size)
    best_score = None
    best_runs: list[tuple[int, int]] | None = None
    for n_runs, band_len, gap in itertools.product((2, 3), (3, 4), (1, 2)):
        span = n_runs * band_len + (n_runs - 1) * gap
        if span > n:
            continue
        for start in range(0, n - span + 1):
            runs = [(start + i * (band_len + gap), start + i * (band_len + gap) + band_len) for i in range(n_runs)]
            band_idx = np.array([r for r0, r1 in runs for r in range(r0, r1)], dtype=int)
            band_counts = counts[band_idx]
            if band_counts.size == 0 or int(band_counts.min()) < 2:
                continue
            span_counts = counts[start : start + span]
            gap_penalty = int(span_counts.sum()) - int(band_counts.sum())
            score = int(band_counts.sum()) * 10 - gap_penalty * 7 - start
            if best_score is None or score > best_score:
                best_score = score
                best_runs = runs
    assert best_runs is not None, f"failed to locate dense bands in counts={counts.tolist()}"
    return best_runs


def _recover_sprite(grid: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    mask = (g != BG) & (g != FRAME)
    row_bands = _select_dense_bands(mask.sum(axis=1))
    col_bands = _select_dense_bands(mask.sum(axis=0))
    slot_h = row_bands[0][1] - row_bands[0][0]
    slot_w = col_bands[0][1] - col_bands[0][0]
    blocks = [g[r0:r1, c0:c1] for r0, r1 in row_bands for c0, c1 in col_bands]
    assert len(blocks) >= 4
    assert all(block.shape == (slot_h, slot_w) for block in blocks)

    sprite = np.zeros((slot_h, slot_w), dtype=int)
    for r in range(slot_h):
        for c in range(slot_w):
            sprite[r, c] = _mode_color([int(block[r, c]) for block in blocks])
    return sprite, row_bands, col_bands


def _components_of_color(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    H, W = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(H):
        for c in range(W):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in CARDINALS:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _frame_components(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    g = np.asarray(grid, dtype=int)
    frames: list[tuple[int, int, int, int]] = []
    for comp in _components_of_color(g == FRAME):
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 3 or w < 3:
            continue
        border = set()
        for rr in range(r0, r1 + 1):
            border.add((rr, c0))
            border.add((rr, c1))
        for cc in range(c0, c1 + 1):
            border.add((r0, cc))
            border.add((r1, cc))
        if set(comp) != border:
            continue
        frames.append((r0, c0, r1, c1))
    frames.sort()
    assert len(frames) >= 2
    return frames


def _analyze(grid: np.ndarray) -> dict[str, object]:
    sprite, row_bands, col_bands = _recover_sprite(grid)
    frames = _frame_components(grid)
    return {"sprite": sprite, "row_bands": row_bands, "col_bands": col_bands, "frames": frames}


def main(grid: np.ndarray) -> np.ndarray:
    """Recover the repeated noisy sprite, scale it to each yellow frame interior, and paste the scaled sprite inside.

    Input structure:
    - Black background contains a regular 2x3 or 3x2 array of equal sprite slots in the upper region.
    - Each slot holds a noisy copy of the same red/blue sprite on black, and sparse non-yellow noise may appear elsewhere.
    - The lower region contains 2-3 one-cell-thick yellow hollow rectangles whose interior sizes are integer multiples of the sprite size.

    Rule:
    - Detect the dense row bands and dense column bands of the non-yellow slot array.
    - Recover the clean sprite by coordinate-wise majority vote across all slot copies.
    - For every yellow frame, measure its interior width and height, scale the recovered sprite by those integer factors with nearest-neighbor replication, and write the scaled sprite into the frame interior.
    - Keep the slot array, the frame borders, and all other noise unchanged.

    Key insight:
    - The repeated noisy copies reveal the clean sprite by voting, and that recovered sprite then becomes the template used for an independent scaling-and-fill operation elsewhere in the grid.
    """
    g = np.asarray(grid, dtype=int)
    info = _analyze(g)
    sprite = np.asarray(info["sprite"], dtype=int)
    sprite_h, sprite_w = sprite.shape

    out = g.copy()
    for r0, c0, r1, c1 in info["frames"]:
        inner_h = r1 - r0 - 1
        inner_w = c1 - c0 - 1
        assert inner_h % sprite_h == 0 and inner_w % sprite_w == 0
        sy = inner_h // sprite_h
        sx = inner_w // sprite_w
        out[r0 + 1 : r1, c0 + 1 : c1] = _scale_xy(sprite, sx, sy)
    return out


def _random_connected_mask(size: int, rng: np.random.Generator) -> np.ndarray:
    area_lo = 6 if size == 3 else 8
    area_hi = size * size - 1
    for _ in range(200):
        target = int(rng.integers(area_lo, area_hi + 1))
        mask = np.zeros((size, size), dtype=bool)
        start = (int(rng.integers(size)), int(rng.integers(size)))
        mask[start] = True
        while int(mask.sum()) < target:
            frontier: list[tuple[int, int]] = []
            for r, c in np.argwhere(mask):
                for dr, dc in CARDINALS:
                    nr, nc = int(r + dr), int(c + dc)
                    if 0 <= nr < size and 0 <= nc < size and not mask[nr, nc]:
                        frontier.append((nr, nc))
            frontier = list(dict.fromkeys(frontier))
            if not frontier:
                break
            nr, nc = frontier[int(rng.integers(len(frontier)))]
            mask[nr, nc] = True
        if int(mask.sum()) != target:
            continue
        if np.any(mask.sum(axis=0) < 2) or np.any(mask.sum(axis=1) < 2):
            continue
        return mask
    raise RuntimeError("failed to build sprite mask")


def _color_sprite(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    for _ in range(200):
        a_idx, b_idx = rng.choice(len(coords), size=2, replace=False)
        a = coords[int(a_idx)]
        b = coords[int(b_idx)]
        sprite = np.zeros(mask.shape, dtype=int)
        blue_count = 0
        red_count = 0
        for r, c in coords:
            da = abs(r - a[0]) + abs(c - a[1])
            db = abs(r - b[0]) + abs(c - b[1])
            color = BLUE if da <= db else RED
            sprite[r, c] = color
            blue_count += color == BLUE
            red_count += color == RED
        if blue_count >= 2 and red_count >= 2:
            return sprite
    raise RuntimeError("failed to color sprite")


def _build_sprite(size: int, rng: np.random.Generator) -> np.ndarray:
    mask = _random_connected_mask(size, rng)
    sprite = _color_sprite(mask, rng)
    assert np.any(sprite == BLUE) and np.any(sprite == RED) and np.any(sprite == BG)
    return sprite


def _slot_bands(slot_count: int, size: int, gap: int, margin: int) -> list[tuple[int, int]]:
    return [(margin + i * (size + gap), margin + i * (size + gap) + size) for i in range(slot_count)]


def _sample_corruptions(
    sprite: np.ndarray, slot_rows: int, slot_cols: int, rng: np.random.Generator
) -> list[list[tuple[int, int, int]]]:
    fg_coords = [tuple(map(int, rc)) for rc in np.argwhere(sprite != BG)]
    n_slots = slot_rows * slot_cols
    cap = n_slots // 2 - 1
    remaining = {coord: cap for coord in fg_coords}
    result: list[list[tuple[int, int, int]]] = []
    for _ in range(n_slots):
        k = 1 + int(rng.integers(2))
        chosen: list[tuple[int, int]] = []
        candidates = [coord for coord in fg_coords if remaining[coord] > 0]
        while len(chosen) < k and candidates:
            shuffled = candidates.copy()
            rng.shuffle(shuffled)
            pick = None
            for coord in shuffled:
                if coord in chosen:
                    continue
                if any(coord[0] == rr for rr, _ in chosen) and len({r for r, _ in fg_coords}) > 1:
                    continue
                pick = coord
                break
            if pick is None:
                pick = shuffled[0]
            chosen.append(pick)
            candidates = [coord for coord in fg_coords if remaining[coord] > 0 and coord not in chosen]
        slot_changes: list[tuple[int, int, int]] = []
        for r, c in chosen:
            remaining[(r, c)] -= 1
            palette = [col for col in NOISE_COLORS if col != int(sprite[r, c])]
            new_color = int(palette[int(rng.integers(len(palette)))])
            if rng.random() < 0.45:
                new_color = BG
            slot_changes.append((r, c, new_color))
        result.append(slot_changes)
    assert all(v >= 0 for v in remaining.values())
    return result


def _place_noise(
    grid: np.ndarray,
    forbidden: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> None:
    H, W = grid.shape
    choices = [
        (r, c)
        for r in range(H)
        for c in range(W)
        if not forbidden[r, c] and grid[r, c] == BG
    ]
    rng.shuffle(choices)
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    placed = 0
    for r, c in choices:
        if r in used_rows or c in used_cols:
            continue
        grid[r, c] = int(NOISE_COLORS[int(rng.integers(len(NOISE_COLORS)))])
        used_rows.add(r)
        used_cols.add(c)
        placed += 1
        if placed == count:
            return
    raise RuntimeError("failed to place sparse noise")


def _gen_expected_output(inp: np.ndarray, sprite: np.ndarray, frames: list[tuple[int, int, int, int]]) -> np.ndarray:
    out = inp.copy()
    sh, sw = sprite.shape
    for r0, c0, r1, c1 in frames:
        inner_h = r1 - r0 - 1
        inner_w = c1 - c0 - 1
        out[r0 + 1 : r1, c0 + 1 : c1] = _scale_xy(sprite, inner_w // sw, inner_h // sh)
    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    slot_rows, slot_cols = [(2, 3), (3, 2)][int(rng.integers(2))]
    size = int(rng.integers(3, 5))
    row_gap = int(rng.integers(1, 3))
    col_gap = int(rng.integers(1, 3))
    array_h = slot_rows * size + (slot_rows - 1) * row_gap
    array_w = slot_cols * size + (slot_cols - 1) * col_gap

    n_frames = 2 + int(rng.integers(2))
    pair_pool = [(sx, sy) for sx in (1, 2, 3) for sy in (1, 2, 3)]
    rng.shuffle(pair_pool)
    scale_pairs = pair_pool[:n_frames]
    if all(sx == sy for sx, sy in scale_pairs) and rng.random() < 0.7:
        return None
    if all(sx == 1 and sy == 1 for sx, sy in scale_pairs):
        return None

    frame_sizes = [(size * sy + 2, size * sx + 2) for sx, sy in scale_pairs]
    gap_choices = [1 + int(rng.integers(2)) for _ in range(max(0, n_frames - 1))]
    total_frame_w = sum(w for _, w in frame_sizes) + sum(gap_choices)
    max_frame_h = max(h for h, _ in frame_sizes)
    bottom_margin = 1 + int(rng.integers(2))

    H_min_top = 2 * (array_h + 1)
    H_min_frame = 2 * (max_frame_h + bottom_margin)
    W_min = max(20, array_w + 2, total_frame_w + 2)
    H_min = max(20, H_min_top, H_min_frame)
    if H_min > 30 or W_min > 30:
        return None

    H = int(rng.integers(H_min, 31))
    W = int(rng.integers(W_min, 31))
    upper_limit = H // 2 - array_h
    if upper_limit < 1:
        return None
    top_margin = int(rng.integers(1, upper_limit + 1))
    left_margin_max = W - array_w - 1
    if left_margin_max < 1:
        return None
    left_margin = int(rng.integers(1, left_margin_max + 1))

    sprite = _build_sprite(size, rng)
    grid = np.zeros((H, W), dtype=int)
    forbidden = np.zeros((H, W), dtype=bool)

    row_bands = _slot_bands(slot_rows, size, row_gap, top_margin)
    col_bands = _slot_bands(slot_cols, size, col_gap, left_margin)
    slot_corruptions = _sample_corruptions(sprite, slot_rows, slot_cols, rng)
    slot_idx = 0
    for r0, r1 in row_bands:
        for c0, c1 in col_bands:
            block = sprite.copy()
            for rr, cc, color in slot_corruptions[slot_idx]:
                block[rr, cc] = color
            grid[r0:r1, c0:c1] = block
            forbidden[r0:r1, c0:c1] = True
            slot_idx += 1

    frame_left = int(rng.integers(1, W - total_frame_w))
    x = frame_left
    frames: list[tuple[int, int, int, int]] = []
    for idx, ((fh, fw), (sx, sy)) in enumerate(zip(frame_sizes, scale_pairs)):
        top_lo = H // 2
        top_hi = H - bottom_margin - fh
        if top_lo > top_hi:
            return None
        fr0 = int(rng.integers(top_lo, top_hi + 1))
        fc0 = x
        fr1 = fr0 + fh - 1
        fc1 = fc0 + fw - 1
        if np.any(forbidden[fr0 : fr1 + 1, fc0 : fc1 + 1]):
            return None
        grid[fr0 : fr1 + 1, fc0] = FRAME
        grid[fr0 : fr1 + 1, fc1] = FRAME
        grid[fr0, fc0 : fc1 + 1] = FRAME
        grid[fr1, fc0 : fc1 + 1] = FRAME
        forbidden[fr0 : fr1 + 1, fc0 : fc1 + 1] = True
        frames.append((fr0, fc0, fr1, fc1))
        x += fw
        if idx < len(gap_choices):
            x += gap_choices[idx]
        assert (fc1 - fc0 - 1, fr1 - fr0 - 1) == (size * sx, size * sy)

    noise_count = int(rng.integers(4, 8))
    try:
        _place_noise(grid, forbidden, noise_count, rng)
    except RuntimeError:
        return None

    expected = _gen_expected_output(grid, sprite, frames)
    try:
        analysis = _analyze(grid)
        if analysis["row_bands"] != row_bands or analysis["col_bands"] != col_bands:
            return None
        if sorted(analysis["frames"]) != sorted(frames):
            return None
        if not np.array_equal(np.asarray(analysis["sprite"], dtype=int), sprite):
            return None
        if not np.array_equal(main(grid), expected):
            return None
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        sample = _gen_attempt(_RNG)
        if sample is not None:
            return sample
    raise RuntimeError("failed to generate a valid input for puzzle 5")
