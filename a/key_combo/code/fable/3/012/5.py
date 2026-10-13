"""Anchor-directed stamp tiling with mirror-orientations from colored control cells.

Input structure
- A black-background grid contains exactly two disjoint objects:
  - a small single-color seed sprite (the stamp),
  - a rectangular-ish dense multicolor control grid with black empty cells.
- Exactly one control-cell color equals the stamp color; that unique cell is the anchor.

Rule
- Let stamp dimensions be H×W and control grid dimensions be Ch×Cw.
- The output canvas is Ch×H rows by Cw×W columns; each control cell maps to one H×W block.
- In the anchor block, paste the stamp unchanged and recolored with its original color.
- For every other non-black control cell at local position (r,c):
  - mirror the stamp horizontally if c is strictly right of the anchor column,
  - mirror the stamp vertically if r is strictly below the anchor row,
  - recolor all stamped pixels to the control-cell color and place in its block.
- Black control cells map to black output blocks.

Key insight
- The mirror rule is driven by the anchor-based direction field: right/below are decided by each control cell's position relative to the anchor, not by a global geometric rule.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ARC_UTILS = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS) not in sys.path:
    sys.path.append(str(_ARC_UTILS))

from arc_utils import Color, bbox_coords, components_mask, crop_bbox, get_bg, paste, random_free_location, random_sprite


def _stamp_block(stamp: np.ndarray, color: int) -> np.ndarray:
    """Return a recolored copy of the stamp with background preserved."""
    out = stamp.copy()
    out[out != Color.BLACK] = int(color)
    return out


def _apply_directional_stamps(stamp: np.ndarray, stamp_color: int, control: np.ndarray) -> np.ndarray:
    """
    Tile recolored stamp blocks according to the anchor-relative direction field.
    """
    g = np.asarray(control, dtype=int)
    h_s, w_s = map(int, stamp.shape)
    h_c, w_c = g.shape
    bg = Color.BLACK

    anchor = np.argwhere(g == stamp_color)
    assert anchor.shape[0] == 1, "expected exactly one anchor cell with stamp color in control grid"
    anchor_r, anchor_c = map(int, anchor[0])

    out = np.full((h_c * h_s, w_c * w_s), bg, dtype=int)
    for r in range(h_c):
        for c in range(w_c):
            cell_color = int(g[r, c])
            if cell_color == bg:
                continue

            tile = stamp
            if c > anchor_c:
                tile = np.fliplr(tile)
            if r > anchor_r:
                tile = np.flipud(tile)

            block = out[r * h_s : (r + 1) * h_s, c * w_s : (c + 1) * w_s]
            recolored = _stamp_block(tile, cell_color)
            block[recolored != bg] = recolored[recolored != bg]

    return out


def _parse_objects(grid: np.ndarray) -> tuple[np.ndarray, int, np.ndarray]:
    """
    Parse one single-color stamp object and one multicolor control grid.
    Returns (stamp_sprite, stamp_color, control_grid_bbox_crop).
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    assert bg == Color.BLACK, "expected black background"

    components = components_mask(g != bg, connectivity=4)
    assert len(components) >= 2, "expected at least stamp and control objects"

    seed_sprite = None
    seed_color = None
    control_grid = None
    for comp in components:
        b = bbox_coords(comp)
        assert b is not None
        r0, c0, r1, c1 = b
        patch = crop_bbox(g, b)
        non_bg = patch != bg
        colors = np.unique(patch[non_bg])

        if colors.size == 1:
            if seed_sprite is not None:
                raise AssertionError("expected exactly one single-color seed sprite")
            seed_sprite = patch
            seed_color = int(colors[0])
        else:
            if control_grid is not None:
                raise AssertionError("expected exactly one multicolor control grid")
            control_grid = patch

    if seed_sprite is None or seed_color is None or control_grid is None:
        raise AssertionError("failed to identify seed and control objects")

    anchor_count = int(np.count_nonzero(control_grid == seed_color))
    assert anchor_count == 1, "expected exactly one anchor cell in control grid"
    return seed_sprite, seed_color, control_grid


def main(grid: np.ndarray) -> np.ndarray:
    """Tile a recolored asymmetric stamp over control-cell blocks using anchor-based mirrors.

    Input structure
    - Black background with two disjoint objects: one small single-color seed sprite and one dense multicolor control grid.
    - Exactly one control-cell has the seed's color and acts as the anchor.
    - Control cells define an H×W tiling lattice for the output, one block per control position.

    Rule
    - Copy the control-grid layout to an output of size (control_h×stamp_h, control_w×stamp_w).
    - For each anchor-relative control-cell:
      - mirror the seed horizontally when the cell is strictly right of the anchor column,
      - mirror it vertically when the cell is strictly below the anchor row,
      - recolor stamped pixels to the control-cell color and place it in the matching block.
    - Keep anchor cell in original orientation and color.
    - Keep blocks black where the control cell is black.

    Key insight
    - The mirror axes are selected by the control-cell's signed offset from the unique anchor cell, so the direction field is anchored by that pixel, not global symmetry.
    """
    stamp, stamp_color, control = _parse_objects(np.asarray(grid, dtype=int))
    return _apply_directional_stamps(stamp, stamp_color, control)


def _is_useful_stamp(stamp: np.ndarray, color: int) -> bool:
    if not np.any(stamp == color):
        return False
    if np.array_equal(stamp, np.fliplr(stamp)):
        return False
    if np.array_equal(stamp, np.flipud(stamp)):
        return False
    return True


def _build_seed() -> tuple[np.ndarray, int]:
    colors = [int(c) for c in Color.NOT_BLACK]
    for _ in range(300):
        seed_color = int(np.random.choice(colors))
        h = int(np.random.randint(2, 6))
        w = int(np.random.randint(2, 6))
        density = float(np.random.uniform(0.35, 0.85))
        stamp = random_sprite(h, w, density=density, symmetry="not_symmetric", color_palette=[seed_color], connectivity=4)
        if _is_useful_stamp(stamp, seed_color):
            return stamp, seed_color
    raise RuntimeError("failed to sample a non-symmetric single-color stamp")


def _build_control(seed_color: int, rows: int, cols: int) -> np.ndarray:
    """
    Build a dense multicolor control grid with one guaranteed anchor and direction anchors in all 4 relative directions.
    """
    other_colors = [int(c) for c in Color.NOT_BLACK if int(c) != seed_color]
    assert rows >= 3 and cols >= 3
    for _ in range(250):
        anchor_r = int(np.random.randint(1, rows - 1))
        anchor_c = int(np.random.randint(1, cols - 1))

        pool = list(other_colors)
        if not pool:
            continue
        np.random.shuffle(pool)
        n_other = int(np.random.randint(2, min(4, len(pool) + 1)))
        palette = pool[:n_other]
        # Start as dense color field, then force the anchor and four neighboring slots.
        control = np.random.choice(palette, size=(rows, cols)).astype(int)
        control[anchor_r, anchor_c] = seed_color

        control[anchor_r - 1, anchor_c] = int(np.random.choice(palette))
        control[anchor_r + 1, anchor_c] = int(np.random.choice(palette))
        control[anchor_r, anchor_c - 1] = int(np.random.choice(palette))
        control[anchor_r, anchor_c + 1] = int(np.random.choice(palette))

        forced = {
            (anchor_r, anchor_c),
            (anchor_r - 1, anchor_c),
            (anchor_r + 1, anchor_c),
            (anchor_r, anchor_c - 1),
            (anchor_r, anchor_c + 1),
        }

        # Add a few black empty cells without breaking connectivity/row-column coverage.
        candidate_holes = [
            (r, c)
            for r in range(rows)
            for c in range(cols)
            if (r, c) not in forced and np.random.random() < 0.24
        ]
        np.random.shuffle(candidate_holes)
        for r, c in candidate_holes[: min(2, len(candidate_holes))]:
            trial = control.copy()
            trial[r, c] = Color.BLACK

            if np.any(np.all(trial[r0, :] == Color.BLACK) for r0 in range(rows)):
                continue
            if np.any(np.all(trial[:, c0] == Color.BLACK) for c0 in range(cols)):
                continue
            if len(components_mask(trial != Color.BLACK, connectivity=4)) != 1:
                continue
            control = trial

        if np.count_nonzero(control == seed_color) != 1:
            continue
        if np.count_nonzero(control != Color.BLACK) == 0:
            continue

        if len(components_mask(control != Color.BLACK, connectivity=4)) != 1:
            continue

        # Keep one block in the dense lattice empty from time to time.
        if np.random.random() < 0.85 and np.count_nonzero(control == Color.BLACK) == 0:
            r, c = int(np.random.randint(rows)), int(np.random.randint(cols))
            if (r, c) not in forced:
                trial = control.copy()
                trial[r, c] = Color.BLACK
                if len(components_mask(trial != Color.BLACK, connectivity=4)) == 1 and not (
                    np.any(np.all(trial[r0, :] == Color.BLACK) for r0 in range(rows))
                    or np.any(np.all(trial[:, c0] == Color.BLACK) for c0 in range(cols))
                ):
                    control = trial

        non_anchor_colors = np.unique(control[control != Color.BLACK])
        if int(seed_color) not in set(map(int, non_anchor_colors.tolist())):
            continue
        if len(non_anchor_colors) < 2:
            continue
        return control

    raise RuntimeError("failed to build a valid control grid")


def generate_input() -> np.ndarray:
    for _ in range(900):
        seed, seed_color = _build_seed()
        seed_h, seed_w = map(int, seed.shape)

        control_h = int(np.random.randint(3, 7))
        control_w = int(np.random.randint(3, 7))
        if seed_h * control_h > 30 or seed_w * control_w > 30:
            continue

        control = _build_control(seed_color, control_h, control_w)

        # Place objects in an independent ARC-like input canvas (≤ 30×30).
        canvas_h = int(np.random.randint(max(seed_h, control_h) + 10, 31))
        canvas_w = int(np.random.randint(max(seed_w, control_w) + 10, 31))
        canvas = np.zeros((canvas_h, canvas_w), dtype=int)

        seed_r = int(np.random.randint(0, canvas_h - seed_h + 1))
        seed_c = int(np.random.randint(0, canvas_w - seed_w + 1))
        paste(canvas, seed, seed_r, seed_c, transparent=Color.BLACK)

        try:
            control_r, control_c = random_free_location(
                canvas,
                control,
                transparent=Color.BLACK,
                border=1,
                padding=1,
                padding_connectivity=8,
            )
        except ValueError:
            continue

        inp = paste(canvas, control, control_r, control_c, transparent=Color.BLACK)

        try:
            expected = _apply_directional_stamps(seed, seed_color, control)
            actual = main(inp)
        except Exception:
            continue

        if np.array_equal(actual, expected):
            if inp.ndim == 2 and max(inp.shape) <= 30:
                return inp

    raise RuntimeError("failed to generate a valid anchor-stamp input")
