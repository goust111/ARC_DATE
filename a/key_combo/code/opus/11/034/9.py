from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np


FOUR_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
PALETTE = [1, 2, 3, 4]
MAX_GENERATION_ATTEMPTS = 400


@dataclass(frozen=True)
class SpriteSpec:
    color: int
    horizontal: bool
    logical_h: int
    logical_w: int
    thick_r0: int
    thick_c0: int

    @property
    def dense_h(self) -> int:
        return 2 * self.logical_h - 1

    @property
    def dense_w(self) -> int:
        return 2 * self.logical_w - 1

    @property
    def thick_h(self) -> int:
        return self.dense_h + 2 if self.horizontal else 4 * self.logical_h - 1

    @property
    def thick_w(self) -> int:
        return 4 * self.logical_w - 1 if self.horizontal else self.dense_w + 2

    @property
    def input_r0(self) -> int:
        return self.thick_r0 + 1

    @property
    def input_c0(self) -> int:
        return self.thick_c0 + 1


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _colors_present(grid: np.ndarray) -> list[int]:
    return sorted(int(v) for v in np.unique(grid) if int(v) != 0)


def _thicken_mask(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask.astype(bool), 1)
    out = padded.copy()
    center = padded[1:-1, 1:-1]
    out[:-2, 1:-1] |= center
    out[2:, 1:-1] |= center
    out[1:-1, :-2] |= center
    out[1:-1, 2:] |= center
    return out


def _dense_half_patch(h: int, w: int) -> np.ndarray:
    return np.ones((h, w), dtype=bool)


def _completed_patch(h: int, w: int) -> np.ndarray:
    if w > h:
        return np.ones((h, 2 * w - 1), dtype=bool)
    return np.ones((2 * h - 1, w), dtype=bool)


def _solve_sparse_sprite(grid: np.ndarray, color: int) -> tuple[np.ndarray, int, int]:
    coords = np.argwhere(grid == int(color))
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    dense = _dense_half_patch(h, w)
    completed = _completed_patch(h, w)
    thick = _thicken_mask(completed)
    return thick, r0 - 1, c0 - 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Densify sparse half-sprites, infer the symmetry axis from their bbox aspect ratio, then mirror-complete and thicken them.

    Input structure:
    - The grid has a black background and 2-4 sparse monochrome sprites of distinct colors.
    - Each sprite lies on a step-2 lattice (all of its pixels are on either even-even or odd-odd coordinates).
    - Each sprite is the left half or upper half of a symmetric solid shape, including the symmetry-axis pixels.

    Rule:
    - For each color independently, take the sprite's minimal bounding box and densify it by filling the whole box with that color.
    - Measure the densified box: if width is greater than height, treat it as a left half and mirror it to the right across the box's right edge; otherwise treat it as an upper half and mirror it downward across the box's bottom edge.
    - After completion, thicken the whole shape outward by one pixel onto every black 4-neighbor cell.
    - Draw all transformed sprites back on the black background without changing their colors.

    Key insight:
    - The sparse dots only reveal a half-shape on a two-step lattice; the true filled half must be reconstructed first, and its aspect ratio is what determines whether the hidden symmetry axis is vertical or horizontal.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = np.zeros_like(g)
    for color in _colors_present(g):
        patch, r0, c0 = _solve_sparse_sprite(g, color)
        rr0 = max(0, r0)
        cc0 = max(0, c0)
        rr1 = min(g.shape[0], r0 + patch.shape[0])
        cc1 = min(g.shape[1], c0 + patch.shape[1])
        local = patch[rr0 - r0 : rr1 - r0, cc0 - c0 : cc1 - c0]
        out[rr0:rr1, cc0:cc1][local] = int(color)
    return out


def _alternative_sparse_mirror(grid: np.ndarray, color: int) -> tuple[np.ndarray, int, int]:
    coords = np.argwhere(grid == int(color))
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    local = np.zeros((h, w), dtype=bool)
    local[coords[:, 0] - r0, coords[:, 1] - c0] = True
    if w > h:
        mirrored = np.zeros((h, 2 * w - 1), dtype=bool)
        mirrored[:, :w] |= local
        mirrored[:, w - 1 :] |= np.fliplr(local)
    else:
        mirrored = np.zeros((2 * h - 1, w), dtype=bool)
        mirrored[:h, :] |= local
        mirrored[h - 1 :, :] |= np.flipud(local)
    return _thicken_mask(mirrored), r0 - 1, c0 - 1


def _dense_no_mirror(grid: np.ndarray, color: int) -> tuple[np.ndarray, int, int]:
    coords = np.argwhere(grid == int(color))
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    return _thicken_mask(np.ones((h, w), dtype=bool)), r0 - 1, c0 - 1


def _paint_variant(shape: tuple[int, int], parts: list[tuple[int, np.ndarray, int, int]]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for color, patch, r0, c0 in parts:
        rr0 = max(0, r0)
        cc0 = max(0, c0)
        rr1 = min(shape[0], r0 + patch.shape[0])
        cc1 = min(shape[1], c0 + patch.shape[1])
        local = patch[rr0 - r0 : rr1 - r0, cc0 - c0 : cc1 - c0]
        out[rr0:rr1, cc0:cc1][local] = int(color)
    return out


def _bboxes_too_close(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    margin: int = 1,
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + margin < br0
        or br1 + margin < ar0
        or ac1 + margin < bc0
        or bc1 + margin < ac0
    )


def _choose_orientations(count: int) -> list[bool]:
    horizontal_count = count // 2
    if count % 2 == 1:
        horizontal_count += random.randint(0, 1)
    orientations = [True] * horizontal_count + [False] * (count - horizontal_count)
    random.shuffle(orientations)
    return orientations


def _sample_logical_dims(horizontal: bool) -> tuple[int, int]:
    short = random.randint(2, 4)
    delta = random.choices([1, 2, 3], weights=[4, 3, 1])[0]
    long = min(5, short + delta)
    if long == short:
        long += 1
    if horizontal:
        return short, long
    return long, short


def _thick_bbox_for_spec(spec: SpriteSpec) -> tuple[int, int, int, int]:
    return (
        spec.thick_r0,
        spec.thick_c0,
        spec.thick_r0 + spec.thick_h - 1,
        spec.thick_c0 + spec.thick_w - 1,
    )


def _sparse_bbox_for_spec(spec: SpriteSpec) -> tuple[int, int, int, int]:
    r0 = spec.input_r0
    c0 = spec.input_c0
    return (r0, c0, r0 + spec.dense_h - 1, c0 + spec.dense_w - 1)


def _draw_input_sprite(grid: np.ndarray, spec: SpriteSpec) -> None:
    for dr in range(spec.logical_h):
        for dc in range(spec.logical_w):
            grid[spec.input_r0 + 2 * dr, spec.input_c0 + 2 * dc] = int(spec.color)


def _place_specs(height: int, width: int, colors: list[int], orientations: list[bool]) -> list[SpriteSpec] | None:
    placed: list[SpriteSpec] = []
    thick_boxes: list[tuple[int, int, int, int]] = []

    indexed = list(enumerate(zip(colors, orientations)))
    indexed.sort(key=lambda item: 0 if item[1][1] else 1)

    for _idx, (color, horizontal) in indexed:
        placed_one = None
        for _ in range(120):
            logical_h, logical_w = _sample_logical_dims(horizontal)
            dense_h = 2 * logical_h - 1
            dense_w = 2 * logical_w - 1
            thick_h = dense_h + 2 if horizontal else 4 * logical_h - 1
            thick_w = 4 * logical_w - 1 if horizontal else dense_w + 2

            max_r0 = height - thick_h
            max_c0 = width - thick_w
            if max_r0 < 0 or max_c0 < 0:
                continue

            candidates = [
                (r0, c0)
                for r0 in range(max_r0 + 1)
                for c0 in range(max_c0 + 1)
                if (r0 % 2) == (c0 % 2)
            ]
            random.shuffle(candidates)
            for thick_r0, thick_c0 in candidates:
                spec = SpriteSpec(
                    color=int(color),
                    horizontal=bool(horizontal),
                    logical_h=int(logical_h),
                    logical_w=int(logical_w),
                    thick_r0=int(thick_r0),
                    thick_c0=int(thick_c0),
                )
                thick_box = _thick_bbox_for_spec(spec)
                sparse_box = _sparse_bbox_for_spec(spec)
                if any(_bboxes_too_close(thick_box, old, margin=1) for old in thick_boxes):
                    continue
                if any(_bboxes_too_close(sparse_box, _sparse_bbox_for_spec(old), margin=0) for old in placed):
                    continue
                placed_one = spec
                break
            if placed_one is not None:
                break

        if placed_one is None:
            return None
        placed.append(placed_one)
        thick_boxes.append(_thick_bbox_for_spec(placed_one))

    return placed


def _validate_grid_against_specs(grid: np.ndarray, specs: list[SpriteSpec]) -> None:
    assert 2 <= len(specs) <= 4
    assert all(int(v) in PALETTE for v in _colors_present(grid))
    assert len(_colors_present(grid)) == len(specs)
    assert any(spec.horizontal for spec in specs)
    assert any(not spec.horizontal for spec in specs)

    actual = {int(spec.color): spec for spec in specs}
    for color in _colors_present(grid):
        coords = np.argwhere(grid == int(color))
        box = _bbox_from_coords(coords)
        spec = actual[int(color)]
        assert box == _sparse_bbox_for_spec(spec)
        assert box[2] - box[0] + 1 == spec.dense_h
        assert box[3] - box[1] + 1 == spec.dense_w
        assert (box[3] - box[1]) != (box[2] - box[0])
        assert np.all((coords[:, 0] % 2) == (spec.input_r0 % 2))
        assert np.all((coords[:, 1] % 2) == (spec.input_c0 % 2))

    out = main(grid)
    parts_main = []
    parts_sparse = []
    parts_nomirror = []
    for color in _colors_present(grid):
        parts_main.append((color, *_solve_sparse_sprite(grid, color)))
        parts_sparse.append((color, *_alternative_sparse_mirror(grid, color)))
        parts_nomirror.append((color, *_dense_no_mirror(grid, color)))

    alt_sparse = _paint_variant(grid.shape, parts_sparse)
    alt_nomirror = _paint_variant(grid.shape, parts_nomirror)
    assert not np.array_equal(out, grid)
    assert not np.array_equal(out, alt_sparse)
    assert not np.array_equal(out, alt_nomirror)

    occupied = np.argwhere(out > 0)
    assert occupied.size > 0

    for i, spec_a in enumerate(specs):
        box_a = _thick_bbox_for_spec(spec_a)
        for spec_b in specs[i + 1 :]:
            box_b = _thick_bbox_for_spec(spec_b)
            assert not _bboxes_too_close(box_a, box_b, margin=0)


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        height = random.randint(18, 26)
        width = random.randint(18, 26)
        count = random.randint(2, 4)
        colors = random.sample(PALETTE, count)
        orientations = _choose_orientations(count)
        specs = _place_specs(height, width, colors, orientations)
        if specs is None:
            continue

        grid = np.zeros((height, width), dtype=int)
        for spec in specs:
            _draw_input_sprite(grid, spec)

        try:
            _validate_grid_against_specs(grid, specs)
        except AssertionError:
            continue
        return grid

    raise RuntimeError("failed to generate a valid sparse-sprite symmetry task")
