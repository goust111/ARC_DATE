# Variant of dfadab01
# Dimension: D3 — 标记来源对象 (src)
# Change: 散布在网格中的单像素标记，使用字形字典中的颜色（2、3、5、8），标记位置即字形放置的左上角... → 十字形标记编码字形类型

from __future__ import annotations

import itertools
import random

import numpy as np

"""
Design analysis for the D3 variant.

a) Concrete behavior and trace:
   - Real controls are no longer one colored pixel. Each control is now a small cross:
     a fixed red center (2) plus 1-4 fixed green arm pixels (3) in the four cardinal
     directions. The arm count is the code. Mapping is:
       1 arm -> yellow frame, 2 arms -> blue ring, 3 arms -> pink diagonal blocks,
       4 arms -> orange X.
   - The glyph anchor stays at the control's logical position: the red center cell is
     the top-left corner where the 4x4 glyph is painted in the output.
   - Example trace: in the original task, a single grey pixel 5 at (6, 9) directly
     meant "paint the magenta diagonal-block glyph at (6, 9)". In this variant, the
     input instead shows a red center at (6, 9) with exactly three green arms around
     it. The solver must count the three arms, map that count to the third glyph, and
     paint the same 4x4 magenta diagonal-block pattern at (6, 9).
   - Legend behavior stays structurally the same: an optional sample glyph may appear
     in input, and its label is now a matching cross centered at (+4, +4) from the
     sample glyph's top-left corner. The sample glyph and its label disappear in the
     output.

b) Complexity calibration:
   - Using a fixed center color and a fixed arm color avoids the 1-arm ambiguity that
     would arise if all cross pixels had the same color. Type is still encoded only by
     arm count, not by which colors appear.
   - Arm directions are randomized per marker, so orientation is a nuisance variable
     and the only stable signal is the count 1-4. This makes the D3 change visible.
   - The count vocabulary remains only four cases, mirroring the original four glyph
     types and keeping the task learnable from a few examples.

c) Implicit-assumption audit for main():
   - Original code assumed every control pixel could be found by scanning colors 2/3/5/8.
     That fails here because color no longer identifies the glyph class.
   - Original legend-skip logic relied on the marker coordinate itself being the label
     anchor. That still works if the new marker's logical anchor is the red center.
   - Original painting code safely overwrote output because glyphs never overlapped.
     That assumption remains valid only if generator spacing keeps marker anchors at
     least 4 cells apart; the generator preserves this.
   - The new solver assumes every real marker center is color 2, lies away from the
     border so its possible arms fit, and has 1-4 adjacent arm cells of color 3.
     The generator constructs exactly that.

d) Placement legality:
   - Center cells are restricted to rows/cols 1..H-2 so the full cross fits.
   - Marker anchors sit on a sparse regular grid with 4-6 spacing, so neither input
     crosses nor output 4x4 glyphs overlap.
   - Legend placement is accepted only if the 4x4 sample glyph and the label cross are
     both fully in bounds and disjoint from all real marker pixels.

e) Visual observability:
   - The new input visibly differs from the source task because controls are plus-like
     objects instead of isolated pixels.
   - Because direction sets vary but output depends only on arm count, multiple markers
     per sample support the intended abstraction.

f) Visual cleanliness:
   - Real controls occupy isolated grid positions with black padding around them.
   - Legend glyphs are placed in empty regions so sample colors do not tangle with real
     controls.

g) Color inferability:
   - No hidden sorting or color-id arithmetic is used. Colors only separate roles:
     red=center, green=arms, and the four output glyph colors stay the original fixed
     alphabet.
"""


CENTER_COLOR = 2
ARM_COLOR = 3
CARDINAL_DIRS: tuple[tuple[int, int], ...] = ((-1, 0), (0, 1), (1, 0), (0, -1))

GLYPHS: dict[int, tuple[int, np.ndarray]] = {
    1: (
        4,
        np.array(
            [
                [1, 1, 1, 1],
                [1, 0, 0, 1],
                [1, 0, 0, 1],
                [1, 1, 1, 1],
            ],
            dtype=bool,
        ),
    ),
    2: (
        1,
        np.array(
            [
                [0, 1, 1, 0],
                [1, 0, 0, 1],
                [1, 0, 0, 1],
                [0, 1, 1, 0],
            ],
            dtype=bool,
        ),
    ),
    3: (
        6,
        np.array(
            [
                [1, 1, 0, 0],
                [1, 1, 0, 0],
                [0, 0, 1, 1],
                [0, 0, 1, 1],
            ],
            dtype=bool,
        ),
    ),
    4: (
        7,
        np.array(
            [
                [1, 0, 0, 1],
                [0, 1, 1, 0],
                [0, 1, 1, 0],
                [1, 0, 0, 1],
            ],
            dtype=bool,
        ),
    ),
}

ARM_SUBSETS: dict[int, tuple[tuple[tuple[int, int], ...], ...]] = {
    count: tuple(itertools.combinations(CARDINAL_DIRS, count)) for count in range(1, 5)
}


def _has_source_glyph(g: np.ndarray, r0: int, c0: int, color: int, mask: np.ndarray) -> bool:
    h, w = g.shape
    if r0 < 0 or c0 < 0 or r0 + 4 > h or c0 + 4 > w:
        return False
    patch = g[r0 : r0 + 4, c0 : c0 + 4]
    return bool(np.all(patch[mask] == color) and np.all(patch[~mask] != color))


def _paint_clipped(out: np.ndarray, r0: int, c0: int, color: int, mask: np.ndarray) -> None:
    h, w = out.shape
    for dr in range(4):
        rr = r0 + dr
        if rr < 0 or rr >= h:
            continue
        for dc in range(4):
            cc = c0 + dc
            if 0 <= cc < w and bool(mask[dr, dc]):
                out[rr, cc] = color


def _count_arms(g: np.ndarray, r: int, c: int) -> int:
    h, w = g.shape
    count = 0
    for dr, dc in CARDINAL_DIRS:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w and int(g[rr, cc]) == ARM_COLOR:
            count += 1
    return count


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode each cross marker by its arm count, then expand it into the matching 4x4 glyph.

    Input structure:
    - Black background with control markers made from a red center pixel plus 1-4 green arm pixels in the four cardinal directions.
    - The number of attached arms selects one of four fixed glyphs; the specific arm directions are incidental.
    - Some inputs also contain one sample 4x4 glyph in its output color, with a matching label cross centered four cells below-right of that sample.

    Rule:
    - For each red center pixel, count how many green arm pixels touch it orthogonally.
    - Map arm counts 1, 2, 3, 4 to the four learned glyph/color pairs, and paint the 4x4 glyph with its top-left corner at the center cell, clipping at the grid boundary.
    - Remove any legend glyph and skip the below-right label cross instead of expanding it.

    Key insight:
    - Marker color no longer names the glyph class; the only semantic code is the number of orthogonal arms attached to a fixed-color center.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert set(np.unique(g)).issubset({0, CENTER_COLOR, ARM_COLOR, 1, 4, 6, 7})

    out = np.zeros_like(g)
    centers = np.argwhere(g == CENTER_COLOR)
    assert centers.size

    seen_marker = False
    for r, c in centers:
        arm_count = _count_arms(g, int(r), int(c))
        if arm_count not in GLYPHS:
            continue
        seen_marker = True
        glyph_color, mask = GLYPHS[arm_count]
        if _has_source_glyph(g, int(r) - 4, int(c) - 4, glyph_color, mask):
            continue
        _paint_clipped(out, int(r), int(c), glyph_color, mask)

    assert seen_marker
    return out


def _sample_axis_positions(length: int, count: int, rng: random.Random) -> list[int]:
    max_center = length - 2
    for _ in range(200):
        gaps = [rng.randint(4, 6) for _ in range(count - 1)]
        span = sum(gaps)
        max_start = max_center - span
        if max_start < 1:
            continue
        start = rng.randint(1, max_start)
        pos = [start]
        for gap in gaps:
            pos.append(pos[-1] + gap)
        return pos
    raise RuntimeError("failed to sample axis positions")


def _marker_cells(r: int, c: int, arm_dirs: tuple[tuple[int, int], ...]) -> set[tuple[int, int]]:
    cells = {(r, c)}
    for dr, dc in arm_dirs:
        cells.add((r + dr, c + dc))
    return cells


def _draw_marker(grid: np.ndarray, r: int, c: int, arm_dirs: tuple[tuple[int, int], ...]) -> None:
    grid[r, c] = CENTER_COLOR
    for dr, dc in arm_dirs:
        grid[r + dr, c + dc] = ARM_COLOR


def _choose_arm_dirs(arm_count: int, rng: random.Random) -> tuple[tuple[int, int], ...]:
    return tuple(rng.choice(ARM_SUBSETS[arm_count]))


def _build_expected_output(
    shape: tuple[int, int], markers: list[tuple[int, int, int]]
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for r, c, arm_count in markers:
        glyph_color, mask = GLYPHS[arm_count]
        _paint_clipped(out, r, c, glyph_color, mask)
    return out


def _find_legend(
    shape: tuple[int, int],
    used_counts: list[int],
    occupied: set[tuple[int, int]],
    rng: random.Random,
) -> tuple[int, int, int, tuple[tuple[int, int], ...]] | None:
    h, w = shape
    order = [(r, c) for r in range(h - 5) for c in range(w - 5)]
    rng.shuffle(order)
    count_order = used_counts[:]
    rng.shuffle(count_order)

    for arm_count in count_order:
        glyph_color, mask = GLYPHS[arm_count]
        arm_variants = list(ARM_SUBSETS[arm_count])
        rng.shuffle(arm_variants)
        for r0, c0 in order:
            glyph_cells = {
                (r0 + dr, c0 + dc) for dr in range(4) for dc in range(4) if bool(mask[dr, dc])
            }
            for arm_dirs in arm_variants:
                center_r, center_c = r0 + 4, c0 + 4
                label_cells = _marker_cells(center_r, center_c, arm_dirs)
                all_cells = glyph_cells | label_cells
                if not all(0 <= rr < h and 0 <= cc < w for rr, cc in all_cells):
                    continue
                if all_cells & occupied:
                    continue
                return r0, c0, arm_count, tuple(arm_dirs)
    return None


def generate_input() -> np.ndarray:
    rng = random.Random()

    for _ in range(200):
        size = rng.choice([10, 20])
        h = w = size
        if size == 10:
            row_count = col_count = 2
        else:
            row_count = rng.randint(2, 5)
            col_count = rng.randint(2, 5)

        row_positions = _sample_axis_positions(h, row_count, rng)
        col_positions = _sample_axis_positions(w, col_count, rng)
        slots = [(r, c) for r in row_positions for c in col_positions]
        min_fill = max(4, int(np.ceil(0.6 * len(slots))))
        occupied_count = rng.randint(min_fill, len(slots))
        chosen_slots = rng.sample(slots, occupied_count)

        type_count = rng.randint(2, min(4, occupied_count))
        used_counts = rng.sample([1, 2, 3, 4], type_count)
        arm_counts = used_counts[:] + [rng.choice(used_counts) for _ in range(occupied_count - type_count)]
        rng.shuffle(arm_counts)

        grid = np.zeros((h, w), dtype=int)
        occupied_cells: set[tuple[int, int]] = set()
        markers: list[tuple[int, int, int]] = []
        valid = True

        for (r, c), arm_count in zip(chosen_slots, arm_counts, strict=True):
            arm_dirs = _choose_arm_dirs(arm_count, rng)
            cells = _marker_cells(r, c, arm_dirs)
            if any(cell in occupied_cells for cell in cells):
                valid = False
                break
            if not all(0 <= rr < h and 0 <= cc < w for rr, cc in cells):
                valid = False
                break
            _draw_marker(grid, r, c, arm_dirs)
            occupied_cells |= cells
            markers.append((r, c, arm_count))

        if not valid:
            continue

        if rng.random() < 0.75:
            legend = _find_legend((h, w), used_counts, occupied_cells, rng)
            if legend is not None:
                r0, c0, arm_count, arm_dirs = legend
                glyph_color, mask = GLYPHS[arm_count]
                for dr in range(4):
                    for dc in range(4):
                        if bool(mask[dr, dc]):
                            grid[r0 + dr, c0 + dc] = glyph_color
                _draw_marker(grid, r0 + 4, c0 + 4, arm_dirs)

        expected = _build_expected_output((h, w), markers)
        actual = main(grid)
        if not np.array_equal(actual, expected):
            continue
        return grid

    raise RuntimeError("failed to generate a valid input")
