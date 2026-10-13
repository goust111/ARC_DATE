from __future__ import annotations

import itertools

import numpy as np


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[int(np.argmax(counts))])


def _colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, np.unique(np.asarray(grid, dtype=int))))
    if exclude is not None:
        colors -= set(map(int, exclude))
    return sorted(colors)


def _bbox_of_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected a nonempty mask"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _decode_scaled_copy(
    grid: np.ndarray,
    color: int,
    bbox: tuple[int, int, int, int],
    scale: int,
    logical_shape: tuple[int, int],
) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    logical_h, logical_w = logical_shape
    crop = (grid[r0 : r1 + 1, c0 : c1 + 1] == color)
    expected_shape = (logical_h * scale, logical_w * scale)
    assert crop.shape == expected_shape, "bbox dimensions must match the inferred scale"

    if scale == 1:
        return crop.astype(bool)

    blocks = crop.reshape(logical_h, scale, logical_w, scale)
    counts = blocks.sum(axis=(1, 3))
    return counts > (scale * scale) // 2


def _infer_scaled_objects(
    grid: np.ndarray,
) -> tuple[int, dict[int, dict[str, object]], tuple[int, int]]:
    bg = _most_common_color(grid)
    colors = _colors_present(grid, exclude={bg})
    assert len(colors) == 3, "expected exactly three foreground colors"

    objects: list[dict[str, object]] = []
    for color in colors:
        mask = grid == color
        bbox = _bbox_of_mask(mask)
        r0, c0, r1, c1 = bbox
        height = int(r1 - r0 + 1)
        width = int(c1 - c0 + 1)
        objects.append({"color": int(color), "bbox": bbox, "height": height, "width": width})

    for perm in itertools.permutations(range(3)):
        base_h = int(objects[perm[0]]["height"])
        base_w = int(objects[perm[0]]["width"])
        if base_h <= 0 or base_w <= 0:
            continue

        ok = True
        mapping: dict[int, dict[str, object]] = {}
        for scale, idx in zip((1, 2, 3), perm):
            obj = objects[idx]
            if int(obj["height"]) != base_h * scale or int(obj["width"]) != base_w * scale:
                ok = False
                break
            mapping[scale] = obj
        if ok:
            return bg, mapping, (base_h, base_w)

    raise AssertionError("could not match the three bboxes to scales 1, 2, and 3")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore a noisy multi-scale binary sprite by voting across the 1x, 2x, and 3x copies.

    Input structure:
    - A dominant background surrounds exactly three differently colored copies of the same binary sprite.
    - The three copies use scales 1, 2, and 3 via nearest-neighbor pixel replication, and their bboxes keep the same 1:2:3 side-length ratio.
    - Each copy is noisy: a logical cell may be partially corrupted inside its scale block, and a whole block may even vote for the opposite binary value.

    Rule:
    - Identify which colored bbox corresponds to scales 1, 2, and 3 from the common logical height and width implied by the 1:2:3 bbox ratios.
    - For each logical cell, let scale s vote by taking the majority value inside its s by s block; for scale 1 the single pixel is the vote.
    - Take the majority of the three scale votes per logical cell and output the recovered 1x sprite using the 1x copy's color on the original background.

    Key insight:
    - No single copy is reliable; the truth lives at the logical-cell level and only appears after combining block-majority votes across all three scales.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg, scaled_objects, logical_shape = _infer_scaled_objects(g)
    decoded = []
    for scale in (1, 2, 3):
        obj = scaled_objects[scale]
        decoded.append(
            _decode_scaled_copy(
                g,
                int(obj["color"]),
                obj["bbox"],
                scale,
                logical_shape,
            )
        )

    truth = (decoded[0].astype(int) + decoded[1].astype(int) + decoded[2].astype(int)) >= 2
    out = np.full(logical_shape, bg, dtype=int)
    out[truth] = int(scaled_objects[1]["color"])
    return out


def _build_logical_sprite(rng: np.random.Generator) -> np.ndarray:
    while True:
        height = int(rng.integers(4, 6))
        width = int(rng.integers(4, 6))
        sprite = np.zeros((height, width), dtype=bool)

        sprite[0, :] = True
        sprite[-1, :] = True
        sprite[:, 0] = True
        sprite[:, -1] = True

        feature_count = int(rng.integers(2, 5))
        for _ in range(feature_count):
            kind = int(rng.integers(3))
            if kind == 0:
                r0 = int(rng.integers(1, height - 1))
                r1 = int(rng.integers(r0, height - 1))
                c0 = int(rng.integers(1, width - 1))
                c1 = int(rng.integers(c0, width - 1))
                sprite[r0 : r1 + 1, c0 : c1 + 1] = True
            elif kind == 1:
                row = int(rng.integers(1, height - 1))
                c0 = int(rng.integers(1, width - 1))
                c1 = int(rng.integers(c0, width - 1))
                sprite[row, c0 : c1 + 1] = True
            else:
                col = int(rng.integers(1, width - 1))
                r0 = int(rng.integers(1, height - 1))
                r1 = int(rng.integers(r0, height - 1))
                sprite[r0 : r1 + 1, col] = True

        if int(rng.integers(2)) == 1:
            sprite = sprite.T
        if int(rng.integers(2)) == 1:
            sprite = sprite[::-1, :]
        if int(rng.integers(2)) == 1:
            sprite = sprite[:, ::-1]

        interior = sprite[1:-1, 1:-1]
        if np.any(interior) and np.any(~interior):
            return sprite


def _assign_error_owner(shape: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    height, width = shape
    owner = np.zeros((height, width), dtype=int)
    interior = [(r, c) for r in range(1, height - 1) for c in range(1, width - 1)]
    assert len(interior) >= 3, "logical sprite must have at least three interior cells"

    chosen = rng.choice(len(interior), size=3, replace=False)
    for scale, idx in zip((1, 2, 3), chosen.tolist()):
        r, c = interior[int(idx)]
        owner[r, c] = scale

    for r, c in interior:
        if owner[r, c] != 0:
            continue
        roll = int(rng.integers(100))
        if roll < 18:
            owner[r, c] = 1
        elif roll < 36:
            owner[r, c] = 2
        elif roll < 54:
            owner[r, c] = 3

    return owner


def _sample_block(
    truth: bool,
    vote: bool,
    scale: int,
    fg_color: int,
    bg_color: int,
    rng: np.random.Generator,
) -> np.ndarray:
    block = np.full((scale, scale), bg_color, dtype=int)
    if scale == 1:
        if vote:
            block[0, 0] = fg_color
        return block

    cell_count = scale * scale
    majority_min = cell_count // 2 + 1
    minority_max = cell_count - majority_min

    if truth == vote:
        if truth:
            fg_count = cell_count - int(rng.integers(0, minority_max + 1))
        else:
            fg_count = int(rng.integers(0, minority_max + 1))
    else:
        if truth:
            fg_count = int(rng.integers(0, minority_max + 1))
        else:
            fg_count = cell_count - int(rng.integers(0, minority_max + 1))

    if fg_count > 0:
        picks = rng.choice(cell_count, size=fg_count, replace=False)
        flat = block.reshape(-1)
        flat[picks] = fg_color
    return block


def _render_copy(
    sprite: np.ndarray,
    scale: int,
    fg_color: int,
    bg_color: int,
    error_owner: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    logical_h, logical_w = sprite.shape
    out = np.full((logical_h * scale, logical_w * scale), bg_color, dtype=int)
    for r in range(logical_h):
        for c in range(logical_w):
            truth = bool(sprite[r, c])
            vote = truth ^ (int(error_owner[r, c]) == scale)
            block = _sample_block(truth, vote, scale, fg_color, bg_color, rng)
            rr = r * scale
            cc = c * scale
            out[rr : rr + scale, cc : cc + scale] = block
    return out


def _layout_copies(
    logical_shape: tuple[int, int],
    rng: np.random.Generator,
) -> tuple[tuple[int, int], dict[int, tuple[int, int]]]:
    logical_h, logical_w = logical_shape
    gap_a = int(rng.integers(1, 4))
    gap_b = int(rng.integers(1, 4))

    if int(rng.integers(2)) == 0:
        medium_above = int(rng.integers(2)) == 1
        total_h = 3 * logical_h + gap_a
        total_w = 5 * logical_w + gap_b
        relative = {3: (0, 0)}
        right_c = 3 * logical_w + gap_b
        if medium_above:
            relative[2] = (0, right_c)
            relative[1] = (2 * logical_h + gap_a, right_c)
        else:
            relative[1] = (0, right_c)
            relative[2] = (logical_h + gap_a, right_c)
    else:
        medium_left = int(rng.integers(2)) == 1
        total_h = 5 * logical_h + gap_b
        total_w = 3 * logical_w + gap_a
        relative = {3: (0, 0)}
        lower_r = 3 * logical_h + gap_b
        if medium_left:
            relative[2] = (lower_r, 0)
            relative[1] = (lower_r, 2 * logical_w + gap_a)
        else:
            relative[1] = (lower_r, 0)
            relative[2] = (lower_r, logical_w + gap_a)

    assert total_h <= 30 and total_w <= 30, "layout exceeds ARC bounds"
    canvas_h = total_h + int(rng.integers(0, 31 - total_h))
    canvas_w = total_w + int(rng.integers(0, 31 - total_w))
    offset_r = int(rng.integers(0, canvas_h - total_h + 1))
    offset_c = int(rng.integers(0, canvas_w - total_w + 1))

    positions = {
        scale: (offset_r + rel_r, offset_c + rel_c)
        for scale, (rel_r, rel_c) in relative.items()
    }

    if int(rng.integers(2)) == 1:
        positions = {
            scale: (r0, canvas_w - (scale * logical_w) - c0)
            for scale, (r0, c0) in positions.items()
        }
    if int(rng.integers(2)) == 1:
        positions = {
            scale: (canvas_h - (scale * logical_h) - r0, c0)
            for scale, (r0, c0) in positions.items()
        }

    return (canvas_h, canvas_w), positions


def _expected_output(sprite: np.ndarray, bg_color: int, fg_color: int) -> np.ndarray:
    out = np.full(sprite.shape, bg_color, dtype=int)
    out[sprite] = fg_color
    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    sprite = _build_logical_sprite(rng)
    error_owner = _assign_error_owner(sprite.shape, rng)

    palette = list(range(10))
    bg_color = int(rng.choice(palette))
    remaining = [color for color in palette if color != bg_color]
    sampled = rng.choice(remaining, size=3, replace=False).astype(int).tolist()
    rng.shuffle(sampled)
    scale_colors = {scale: int(color) for scale, color in zip((1, 2, 3), sampled)}

    copies = {
        scale: _render_copy(sprite, scale, scale_colors[scale], bg_color, error_owner, rng)
        for scale in (1, 2, 3)
    }

    canvas_shape, positions = _layout_copies(sprite.shape, rng)
    grid = np.full(canvas_shape, bg_color, dtype=int)
    for scale in (1, 2, 3):
        r0, c0 = positions[scale]
        patch = copies[scale]
        h, w = patch.shape
        grid[r0 : r0 + h, c0 : c0 + w] = patch

    expected = _expected_output(sprite, bg_color, scale_colors[1])
    try:
        actual = main(grid)
    except AssertionError:
        return None

    if not np.array_equal(actual, expected):
        return None
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(300):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid multi-scale majority-vote input")
