from __future__ import annotations

import numpy as np


BG = 0
PALETTE = np.arange(1, 10, dtype=int)

RECT_COUNT_RANGE = (2, 6)
LENGTH_RANGE = (3, 10)
THICKNESS_RANGE = (2, 6)
GRID_MARGIN = 2
PLACEMENT_ATTEMPTS = 400
GENERATION_ATTEMPTS = 500


def _rectangle_components(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    boxes: list[tuple[int, int, int, int]] = []

    for start_r in range(height):
        for start_c in range(width):
            if not mask[start_r, start_c] or seen[start_r, start_c]:
                continue

            stack = [(int(start_r), int(start_c))]
            seen[start_r, start_c] = True
            area = 0
            r0 = r1 = int(start_r)
            c0 = c1 = int(start_c)

            while stack:
                r, c = stack.pop()
                area += 1
                r0 = min(r0, r)
                r1 = max(r1, r)
                c0 = min(c0, c)
                c1 = max(c1, c)

                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    rr = r + dr
                    cc = c + dc
                    if 0 <= rr < height and 0 <= cc < width and mask[rr, cc] and not seen[rr, cc]:
                        seen[rr, cc] = True
                        stack.append((int(rr), int(cc)))

            submask = mask[r0 : r1 + 1, c0 : c1 + 1]
            assert bool(submask.all()), "each foreground component must fill its bounding box"
            assert area == int(submask.size), "each foreground component must be a solid rectangle"
            boxes.append((int(r0), int(c0), int(r1), int(c1)))

    assert boxes, "expected at least one striped rectangle"
    return boxes


def _extract_band(patch: np.ndarray) -> np.ndarray:
    repeated_rows = bool(np.all(patch == patch[[0], :]))
    repeated_cols = bool(np.all(patch == patch[:, [0]]))

    if repeated_rows and not repeated_cols:
        band = patch[0, :].copy()
    elif repeated_cols and not repeated_rows:
        band = patch[:, 0].copy()
    else:
        raise AssertionError("rectangle must vary along exactly one axis")

    assert len(set(map(int, band.tolist()))) >= 2, "each rectangle must be multicolored"
    return band.astype(int, copy=False)


def _stack_bands(items: list[tuple[tuple[int, int, int, int], np.ndarray]]) -> np.ndarray:
    ordered = sorted(items, key=lambda item: (int(item[1].size), item[0][0], item[0][1]))
    out_h = len(ordered)
    out_w = max(int(band.size) for _bbox, band in ordered)
    out = np.full((out_h, out_w), BG, dtype=int)

    for offset, (_bbox, band) in enumerate(ordered):
        row = out_h - 1 - offset
        out[row, : band.size] = band
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress each striped rectangle into a one-dimensional band and stack the bands bottom-up by increasing length.

    Input structure:
    - The background is black.
    - The non-black cells form several disjoint solid rectangles.
    - Inside each rectangle, colors repeat unchanged along one axis and vary stripe-by-stripe along the other axis.

    Rule:
    - For each rectangle, detect the axis along which colors change.
    - Read that visible color sequence left-to-right or top-to-bottom to obtain a one-dimensional band.
    - Sort the bands from shortest to longest, then place them as horizontal rows stacked from the bottom upward, left-aligned on a black canvas.

    Key insight:
    - Each 2D striped rectangle redundantly encodes a 1D sequence; solving the task means collapsing that repeated dimension, then sorting and restacking the recovered sequences.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    boxes = _rectangle_components(g != BG)
    bands = []
    for bbox in boxes:
        r0, c0, r1, c1 = bbox
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        bands.append((bbox, _extract_band(patch)))

    return _stack_bands(bands)


def _sample_sequence(
    rng: np.random.Generator,
    length: int,
    used_sequences: set[tuple[int, ...]],
) -> np.ndarray | None:
    for _ in range(200):
        palette_size = int(rng.integers(2, min(5, len(PALETTE)) + 1))
        local_palette = rng.choice(PALETTE, size=palette_size, replace=False)

        seq = np.empty(length, dtype=int)
        seq[0] = int(rng.choice(local_palette))
        for idx in range(1, length):
            choices = local_palette[local_palette != seq[idx - 1]]
            seq[idx] = int(rng.choice(choices))

        if len(set(map(int, seq.tolist()))) < 2:
            continue
        key = tuple(map(int, seq.tolist()))
        if key in used_sequences:
            continue
        used_sequences.add(key)
        return seq
    return None


def _make_patch(band: np.ndarray, orientation: str, thickness: int) -> np.ndarray:
    if orientation == "horizontal":
        return np.repeat(band[np.newaxis, :], thickness, axis=0)
    assert orientation == "vertical"
    return np.repeat(band[:, np.newaxis], thickness, axis=1)


def _choose_orientations(rng: np.random.Generator, count: int) -> list[str]:
    horizontal_count = count // 2
    if count % 2 == 1 and bool(rng.integers(0, 2)):
        horizontal_count += 1
    orientations = ["horizontal"] * horizontal_count + ["vertical"] * (count - horizontal_count)
    rng.shuffle(orientations)
    return orientations


def _target_grid_shape(
    rng: np.random.Generator,
    patches: list[np.ndarray],
) -> tuple[int, int] | None:
    max_h = max(int(patch.shape[0]) for patch in patches)
    max_w = max(int(patch.shape[1]) for patch in patches)
    total_area = sum(int(patch.size) for patch in patches)

    lower = max(
        max_h + GRID_MARGIN,
        max_w + GRID_MARGIN,
        int(np.ceil(np.sqrt(total_area * 2.7))),
    )
    if lower > 30:
        return None

    upper = min(30, lower + 8)
    height = int(rng.integers(lower, upper + 1))
    width = int(rng.integers(lower, upper + 1))
    return height, width


def _place_patches(
    rng: np.random.Generator,
    patches: list[np.ndarray],
    shape: tuple[int, int],
) -> np.ndarray | None:
    height, width = shape
    grid = np.full((height, width), BG, dtype=int)
    blocked = np.zeros((height, width), dtype=bool)

    order = sorted(range(len(patches)), key=lambda idx: int(patches[idx].size), reverse=True)
    for idx in order:
        patch = patches[idx]
        ph, pw = map(int, patch.shape)
        if ph > height or pw > width:
            return None

        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            r0 = int(rng.integers(0, height - ph + 1))
            c0 = int(rng.integers(0, width - pw + 1))
            if bool(blocked[r0 : r0 + ph, c0 : c0 + pw].any()):
                continue

            grid[r0 : r0 + ph, c0 : c0 + pw] = patch
            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(height, r0 + ph + 1)
            cc1 = min(width, c0 + pw + 1)
            blocked[rr0:rr1, cc0:cc1] = True
            placed = True
            break

        if not placed:
            return None

    return grid


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    rect_count = int(rng.integers(RECT_COUNT_RANGE[0], RECT_COUNT_RANGE[1] + 1))
    lengths = rng.choice(
        np.arange(LENGTH_RANGE[0], LENGTH_RANGE[1] + 1, dtype=int),
        size=rect_count,
        replace=False,
    )
    orientations = _choose_orientations(rng, rect_count)

    used_sequences: set[tuple[int, ...]] = set()
    bands: list[np.ndarray] = []
    patches: list[np.ndarray] = []
    for length, orientation in zip(lengths.tolist(), orientations):
        band = _sample_sequence(rng, int(length), used_sequences)
        if band is None:
            return None
        thickness = int(rng.integers(THICKNESS_RANGE[0], THICKNESS_RANGE[1] + 1))
        bands.append(band)
        patches.append(_make_patch(band, orientation, thickness))

    shape = _target_grid_shape(rng, patches)
    if shape is None:
        return None

    grid = _place_patches(rng, patches, shape)
    if grid is None:
        return None

    expected = _stack_bands([((0, 0, 0, 0), band) for band in bands])
    try:
        predicted = main(grid)
    except Exception:
        return None
    if not np.array_equal(predicted, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GENERATION_ATTEMPTS} attempts")
