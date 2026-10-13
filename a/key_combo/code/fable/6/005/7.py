from __future__ import annotations

import random
from collections import Counter

import numpy as np


BLACK = 0
_GEN_COUNTER = 0


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _dihedral_transforms(arr: np.ndarray) -> list[np.ndarray]:
    outs: list[np.ndarray] = []
    cur = arr.copy()
    for _ in range(4):
        outs.append(cur.copy())
        outs.append(np.fliplr(cur).copy())
        cur = np.rot90(cur)
    return outs


def _regular_intervals(separators: list[int], length: int) -> list[tuple[int, int]]:
    bounds = [-1, *sorted(separators), int(length)]
    intervals = [(bounds[i] + 1, bounds[i + 1]) for i in range(len(bounds) - 1)]
    sizes = [end - start for start, end in intervals]
    assert intervals and min(sizes) > 0
    assert len(set(sizes)) == 1
    return intervals


def _legend_from_corner(grid: np.ndarray) -> tuple[tuple[int, int, int, int], int]:
    H, W = grid.shape

    def top_left() -> tuple[int, int, int, int] | None:
        color = int(grid[0, 0])
        if color == BLACK:
            return None
        width = 0
        while width < W and int(grid[0, width]) == color:
            width += 1
        height = 0
        while height < H and int(grid[height, 0]) == color:
            height += 1
        if min(width, height) < 3:
            return None
        return 0, 0, height - 1, width - 1

    def top_right() -> tuple[int, int, int, int] | None:
        color = int(grid[0, W - 1])
        if color == BLACK:
            return None
        width = 0
        while width < W and int(grid[0, W - 1 - width]) == color:
            width += 1
        height = 0
        while height < H and int(grid[height, W - 1]) == color:
            height += 1
        if min(width, height) < 3:
            return None
        return 0, W - width, height - 1, W - 1

    def bottom_left() -> tuple[int, int, int, int] | None:
        color = int(grid[H - 1, 0])
        if color == BLACK:
            return None
        width = 0
        while width < W and int(grid[H - 1, width]) == color:
            width += 1
        height = 0
        while height < H and int(grid[H - 1 - height, 0]) == color:
            height += 1
        if min(width, height) < 3:
            return None
        return H - height, 0, H - 1, width - 1

    def bottom_right() -> tuple[int, int, int, int] | None:
        color = int(grid[H - 1, W - 1])
        if color == BLACK:
            return None
        width = 0
        while width < W and int(grid[H - 1, W - 1 - width]) == color:
            width += 1
        height = 0
        while height < H and int(grid[H - 1 - height, W - 1]) == color:
            height += 1
        if min(width, height) < 3:
            return None
        return H - height, W - width, H - 1, W - 1

    for builder in (top_left, top_right, bottom_left, bottom_right):
        box = builder()
        if box is None:
            continue
        r0, c0, r1, c1 = box
        color = int(grid[r0, c0])
        patch = _crop(grid, box)
        if not (
            np.all(patch[0, :] == color)
            and np.all(patch[-1, :] == color)
            and np.all(patch[:, 0] == color)
            and np.all(patch[:, -1] == color)
        ):
            continue
        inner = patch[1:-1, 1:-1]
        if inner.size == 0:
            continue
        if not np.any((inner != BLACK) & (inner != color)):
            continue
        return box, color
    raise AssertionError("legend not found")


def _analyze_grid(
    grid: np.ndarray,
) -> dict:
    g = np.asarray(grid, dtype=int)
    legend_box, border_color = _legend_from_corner(g)

    payload = g.copy()
    r0, c0, r1, c1 = legend_box
    payload[r0 : r1 + 1, c0 : c1 + 1] = BLACK

    payload_box = _bbox(payload != BLACK)
    assert payload_box is not None
    pr0, pc0, pr1, pc1 = payload_box
    block_region = payload[pr0 : pr1 + 1, pc0 : pc1 + 1]

    row_seps = [r for r in range(block_region.shape[0]) if np.all(block_region[r, :] == BLACK)]
    col_seps = [c for c in range(block_region.shape[1]) if np.all(block_region[:, c] == BLACK)]
    row_intervals = _regular_intervals(row_seps, int(block_region.shape[0]))
    col_intervals = _regular_intervals(col_seps, int(block_region.shape[1]))
    block_sizes = {r1_ - r0_ for r0_, r1_ in row_intervals} | {c1_ - c0_ for c0_, c1_ in col_intervals}
    assert len(block_sizes) == 1
    block_size = int(next(iter(block_sizes)))
    assert block_size >= 2

    mixed_groups: dict[tuple[int, int], list[np.ndarray]] = {}
    uniform_colors: Counter[int] = Counter()
    block_states: list[tuple[tuple[int, int], np.ndarray]] = []
    for rr in row_intervals:
        for cc in col_intervals:
            br0, br1 = rr
            bc0, bc1 = cc
            block = block_region[br0:br1, bc0:bc1].copy()
            colors = sorted(set(map(int, block.ravel())))
            assert BLACK not in colors
            assert len(colors) in (1, 2)
            block_states.append(((br0, bc0), block))
            if len(colors) == 1:
                uniform_colors[int(colors[0])] += 1
            else:
                mixed_groups[(int(colors[0]), int(colors[1]))] = mixed_groups.get((int(colors[0]), int(colors[1])), []) + [block]

    assert len(mixed_groups) == 2
    assert len(uniform_colors) == 2

    sprites: list[dict] = []
    for color_pair, blocks in mixed_groups.items():
        sprite = blocks[0]
        for other in blocks[1:]:
            assert np.array_equal(other, sprite)
        counts = {color: int(np.count_nonzero(sprite == color)) for color in color_pair}
        assert counts[color_pair[0]] != counts[color_pair[1]]
        majority = max(color_pair, key=lambda color: counts[color])
        minority = min(color_pair, key=lambda color: counts[color])
        sprites.append(
            {
                "sprite": sprite,
                "majority": int(majority),
                "minority": int(minority),
                "pair": tuple(color_pair),
            }
        )

    assert {info["minority"] for info in sprites} == set(uniform_colors)

    legend_patch = _crop(g, legend_box)
    legend_mask = (legend_patch != BLACK) & (legend_patch != border_color)
    legend_sprite_box = _bbox(legend_mask)
    assert legend_sprite_box is not None
    legend_sprite = _crop(legend_patch, legend_sprite_box)

    matches = []
    for idx, info in enumerate(sprites):
        if any(np.array_equal(legend_sprite, t) for t in _dihedral_transforms(info["sprite"])):
            matches.append(idx)
    assert len(matches) == 1
    true_idx = int(matches[0])

    return {
        "legend_box": legend_box,
        "legend_border": int(border_color),
        "payload_box": payload_box,
        "block_region": block_region,
        "row_intervals": row_intervals,
        "col_intervals": col_intervals,
        "sprites": sprites,
        "true_idx": true_idx,
        "false_idx": 1 - true_idx,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the selected sprite code in a block grid while erasing the false code and the corner legend.

    Input structure:
    - Black separator rows and columns split one rectangular payload area into equal square blocks.
    - Each payload block is either a copy of sprite A, a copy of sprite B, or a uniform solid block.
    - The two sprites are two-color patterns with different minority colors; a corner legend frame on black background shows one sprite, possibly rotated or mirrored.

    Rule:
    - Identify the legend frame, crop the sprite drawn inside it, and match it to A or B up to rotation or reflection.
    - The matched sprite is the true code: its sprite blocks become solid blocks of its majority color, and uniform blocks of its minority color become that sprite.
    - The other sprite is a decoy: both its sprite blocks and its minority-color uniform blocks are painted black; separator lines and the whole legend region are also black.

    Key insight:
    - Two visually similar encodings coexist, but only the legend decides which one participates in the sprite↔solid inversion; the other code family must be wiped instead of decoded.
    """
    info = _analyze_grid(grid)
    g = np.asarray(grid, dtype=int)
    out = np.zeros_like(g)

    pr0, pc0, _pr1, _pc1 = info["payload_box"]
    true_sprite = info["sprites"][info["true_idx"]]
    false_sprite = info["sprites"][info["false_idx"]]

    for rr in info["row_intervals"]:
        for cc in info["col_intervals"]:
            br0, br1 = rr
            bc0, bc1 = cc
            block = info["block_region"][br0:br1, bc0:bc1]
            dst = out[pr0 + br0 : pr0 + br1, pc0 + bc0 : pc0 + bc1]

            if np.array_equal(block, true_sprite["sprite"]):
                dst[:, :] = int(true_sprite["majority"])
            elif np.array_equal(block, false_sprite["sprite"]):
                dst[:, :] = BLACK
            else:
                color = int(block[0, 0])
                assert np.all(block == color)
                if color == int(true_sprite["minority"]):
                    dst[:, :] = true_sprite["sprite"]
                else:
                    assert color == int(false_sprite["minority"])
                    dst[:, :] = BLACK

    return out


def _grow_mask(size: int, count: int) -> np.ndarray:
    cells = [(r, c) for r in range(size) for c in range(size)]
    start = random.choice(cells)
    chosen = {start}
    frontier = set()

    def add_frontier(r: int, c: int) -> None:
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < size and 0 <= nc < size and (nr, nc) not in chosen:
                frontier.add((nr, nc))

    add_frontier(*start)
    while len(chosen) < count:
        if not frontier:
            return np.zeros((0, 0), dtype=bool)
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        chosen.add(cell)
        add_frontier(*cell)

    mask = np.zeros((size, size), dtype=bool)
    for r, c in chosen:
        mask[r, c] = True
    return mask


def _valid_shape(mask: np.ndarray) -> bool:
    if mask.size == 0:
        return False
    box = _bbox(mask)
    if box is None:
        return False
    r0, c0, r1, c1 = box
    if r1 == r0 or c1 == c0:
        return False
    if int(mask.sum()) * 2 >= mask.size:
        return False
    if len({t.tobytes() for t in _dihedral_transforms(mask.astype(int))}) < 8:
        return False
    return True


def _make_distinct_masks(size: int) -> tuple[np.ndarray, np.ndarray]:
    max_count = max(3, min(6, (size * size - 1) // 2))
    min_count = 3 if size >= 4 else 2
    target = random.randint(min_count, max_count)
    first = None
    for _ in range(300):
        candidate = _grow_mask(size, target)
        if _valid_shape(candidate):
            first = candidate
            break
    assert first is not None

    for _ in range(300):
        second = _grow_mask(size, target)
        if not _valid_shape(second):
            continue
        if any(np.array_equal(second, t) for t in _dihedral_transforms(first)):
            continue
        return first, second
    raise RuntimeError("failed to build two distinct sprite masks")


def _build_sprite(mask: np.ndarray, majority: int, minority: int) -> np.ndarray:
    return np.where(mask, int(minority), int(majority)).astype(int)


def _state_counts(total: int, dominant_code: int) -> list[int]:
    counts = [1, 1, 1, 1]
    remaining = total - 4
    while remaining > 0:
        if dominant_code == 0:
            pick = random.choice([0, 1, 0, 1, 2, 3])
        else:
            pick = random.choice([2, 3, 2, 3, 0, 1])
        counts[pick] += 1
        remaining -= 1
    return counts


def _render_block_grid(
    rows: int,
    cols: int,
    sprite_a: np.ndarray,
    sprite_b: np.ndarray,
    blank_a: int,
    blank_b: int,
    dominant_code: int,
) -> tuple[np.ndarray, list[int]]:
    total = rows * cols
    counts = _state_counts(total, dominant_code)
    states = [0] * counts[0] + [1] * counts[1] + [2] * counts[2] + [3] * counts[3]
    random.shuffle(states)

    size = int(sprite_a.shape[0])
    grid = np.zeros((rows * size + (rows - 1), cols * size + (cols - 1)), dtype=int)
    idx = 0
    for br in range(rows):
        for bc in range(cols):
            r0 = br * (size + 1)
            c0 = bc * (size + 1)
            state = states[idx]
            idx += 1
            if state == 0:
                patch = sprite_a
            elif state == 1:
                patch = np.full((size, size), int(blank_a), dtype=int)
            elif state == 2:
                patch = sprite_b
            else:
                patch = np.full((size, size), int(blank_b), dtype=int)
            grid[r0 : r0 + size, c0 : c0 + size] = patch
    return grid, counts


def _legend_patch(sprite: np.ndarray, border_color: int, orientation_id: int) -> np.ndarray:
    transformed = _dihedral_transforms(sprite)[orientation_id % 8]
    size = int(sprite.shape[0])
    legend = np.zeros((size + 4, size + 4), dtype=int)
    legend[0, :] = int(border_color)
    legend[-1, :] = int(border_color)
    legend[:, 0] = int(border_color)
    legend[:, -1] = int(border_color)
    legend[2 : 2 + size, 2 : 2 + size] = transformed
    return legend


def _compose_canvas(corner: int, legend: np.ndarray, payload: np.ndarray) -> np.ndarray:
    gap = 2
    lh, lw = legend.shape
    ph, pw = payload.shape
    H = max(lh, ph)
    W = lw + gap + pw
    assert H <= 30 and W <= 30
    canvas = np.zeros((H, W), dtype=int)

    if corner == 0:
        canvas[:lh, :lw] = legend
        canvas[:ph, W - pw :] = payload
    elif corner == 1:
        canvas[:lh, W - lw :] = legend
        canvas[:ph, :pw] = payload
    elif corner == 2:
        canvas[H - lh :, :lw] = legend
        canvas[H - ph :, W - pw :] = payload
    else:
        canvas[H - lh :, W - lw :] = legend
        canvas[H - ph :, :pw] = payload

    return canvas


def _expected_output_from_layout(
    payload: np.ndarray,
    sprite_a: np.ndarray,
    sprite_b: np.ndarray,
    blank_a: int,
    blank_b: int,
    true_code: int,
) -> np.ndarray:
    size = int(sprite_a.shape[0])
    H, W = payload.shape
    out = np.zeros_like(payload)
    row_seps = [r for r in range(H) if np.all(payload[r, :] == BLACK)]
    col_seps = [c for c in range(W) if np.all(payload[:, c] == BLACK)]
    rows = _regular_intervals(row_seps, H)
    cols = _regular_intervals(col_seps, W)
    true_sprite = sprite_a if true_code == 0 else sprite_b
    true_blank = blank_a if true_code == 0 else blank_b
    false_sprite = sprite_b if true_code == 0 else sprite_a
    false_blank = blank_b if true_code == 0 else blank_a
    true_majority = Counter(map(int, true_sprite.ravel())).most_common(1)[0][0]

    for rr in rows:
        for cc in cols:
            r0, r1 = rr
            c0, c1 = cc
            block = payload[r0:r1, c0:c1]
            if np.array_equal(block, true_sprite):
                out[r0:r1, c0:c1] = int(true_majority)
            elif np.array_equal(block, false_sprite):
                out[r0:r1, c0:c1] = BLACK
            elif np.all(block == int(true_blank)):
                out[r0:r1, c0:c1] = true_sprite
            else:
                assert np.all(block == int(false_blank))
                out[r0:r1, c0:c1] = BLACK
    return out


def _semantic_checks(
    grid: np.ndarray,
    expected: np.ndarray,
    counts: list[int],
) -> None:
    assert all(count > 0 for count in counts)
    solved = main(grid)
    assert np.array_equal(solved, expected)
    info = _analyze_grid(grid)
    alt_idx = 1 - int(info["true_idx"])
    alt_sprites = info["sprites"]

    alt = np.zeros_like(grid)
    pr0, pc0, _pr1, _pc1 = info["payload_box"]
    true_sprite = alt_sprites[alt_idx]
    false_sprite = alt_sprites[1 - alt_idx]
    for rr in info["row_intervals"]:
        for cc in info["col_intervals"]:
            br0, br1 = rr
            bc0, bc1 = cc
            block = info["block_region"][br0:br1, bc0:bc1]
            dst = alt[pr0 + br0 : pr0 + br1, pc0 + bc0 : pc0 + bc1]
            if np.array_equal(block, true_sprite["sprite"]):
                dst[:, :] = int(true_sprite["majority"])
            elif np.array_equal(block, false_sprite["sprite"]):
                dst[:, :] = BLACK
            else:
                color = int(block[0, 0])
                if color == int(true_sprite["minority"]):
                    dst[:, :] = true_sprite["sprite"]
                else:
                    dst[:, :] = BLACK
    assert not np.array_equal(alt, expected)


def generate_input() -> np.ndarray:
    global _GEN_COUNTER
    for _ in range(400):
        counter = _GEN_COUNTER
        _GEN_COUNTER += 1

        block_size = 4 if counter % 3 else 5
        rows = 3 + ((counter // 2) % 2)
        cols = 3 if block_size == 5 else 3 + ((counter // 3) % 2)
        true_code = counter % 2
        dominant_code = (counter // 2) % 2
        corner = (counter // 3) % 4
        orientation_id = (counter // 5) % 8

        mask_a, mask_b = _make_distinct_masks(block_size)
        colors = random.sample(list(range(1, 10)), 5)
        maj_a, min_a, maj_b, min_b, border = colors
        sprite_a = _build_sprite(mask_a, maj_a, min_a)
        sprite_b = _build_sprite(mask_b, maj_b, min_b)

        payload, counts = _render_block_grid(rows, cols, sprite_a, sprite_b, min_a, min_b, dominant_code)
        legend = _legend_patch(sprite_a if true_code == 0 else sprite_b, border, orientation_id)
        grid = _compose_canvas(corner, legend, payload)

        expected = np.zeros_like(grid)
        if corner == 0:
            expected[: payload.shape[0], -payload.shape[1] :] = _expected_output_from_layout(
                payload, sprite_a, sprite_b, min_a, min_b, true_code
            )
        elif corner == 1:
            expected[: payload.shape[0], : payload.shape[1]] = _expected_output_from_layout(
                payload, sprite_a, sprite_b, min_a, min_b, true_code
            )
        elif corner == 2:
            expected[-payload.shape[0] :, -payload.shape[1] :] = _expected_output_from_layout(
                payload, sprite_a, sprite_b, min_a, min_b, true_code
            )
        else:
            expected[-payload.shape[0] :, : payload.shape[1]] = _expected_output_from_layout(
                payload, sprite_a, sprite_b, min_a, min_b, true_code
            )

        if max(grid.shape) > 30:
            continue

        try:
            _semantic_checks(grid, expected, counts)
        except AssertionError:
            continue
        return grid

    raise RuntimeError("generate_input: failed after 400 attempts")
