# Combo variant of dbff022c (index 40)
# combo_spec: D4/2+D13/0
# Name: 面积决定填充色 + 多色边框

from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

_ROOT = Path(__file__).resolve().parents[5]
_SOLVERS = _ROOT / "solvers"
if str(_SOLVERS) not in sys.path:
    sys.path.insert(0, str(_SOLVERS))

from arc_utils import components_mask, neighbors4


# Design analysis for the combo variant
# -------------------------------------
# spec_variant:
#   preserved:
#     - The scene still consists of separated hollow sprites on a black background.
#     - Only enclosed background holes change between input and output.
#     - Generation remains output-first: first build the solved grid, then erase hole fills.
#   changed_core:
#     - The edge palette is removed entirely, so color lookup can no longer depend on any legend.
#     - Sprite borders are multicolor, but those colors are intentionally non-semantic noise.
#     - The only semantic signal is the area of each enclosed hole.
#   concrete_pixel_behavior:
#     - We use a fixed visible mapping:
#         area 1 -> color 2
#         area 2 -> color 3
#         area 3 -> color 4
#         area 4 -> color 6
#     - Each sprite border uses 2-3 colors drawn from a border-only pool that excludes
#       those fill colors, so border identity and fill identity stay visually separable.
#   behavior_trace:
#     - Original task: detect palette, read border->fill pairs, then fill each hole from
#       the surrounding border color.
#     - Fused task: ignore any border colors, measure the size of the enclosed bg component,
#       and fill from the fixed area mapping.
#     - Example: two sprites can both have majority border color 8, yet a 1-cell hole fills
#       with 2 while a 4-cell hole fills with 6. Conversely, two different multicolor borders
#       around two 2-cell holes both fill with 3.
#   complexity_calibration:
#     - The hole-size alphabet is capped at four areas, which is enough to show a count-based
#       rule without creating a large arbitrary lookup table.
#     - Every generated grid contains four sprites, with one duplicated area and one repeated
#       majority border color across different areas. That gives local evidence that border
#       colors are distractors rather than the control variable.
#   implicit_assumption_audit:
#     - The original `_best_palette` / `_palette_mapping` logic relied on a palette existing;
#       that assumption is false now, so the fused solver removes palette handling entirely.
#     - The original fill step assumed adjacency colors carry meaning. With multicolor shells
#       and no legend, that assumption must be dropped; only the enclosed bg component area
#       remains stable.
#     - The new solver assumes every enclosed bg region to be solved is a true sprite hole and
#       that its area lies in the supported mapping. The generator guarantees both.
#   placement_legality:
#     - Sprites are placed in a 2x2 layout with at least one background cell of separation.
#     - A one-cell outer margin keeps every hole away from the grid boundary, so boundary-touching
#       bg is always the outside world and interior bg components are always holes.
#     - Templates are prevalidated to have exactly one enclosed hole of the intended area.
#   observability_and_cleanliness:
#     - Border colors come from a pool disjoint from fill colors, which keeps the output easy to
#       read and prevents accidental "same color means same answer" shortcuts.
#     - Repeated area with different border colorings, plus repeated majority border with different
#       areas, makes the fusion behavior visible from a small number of examples.
#   main_assumption_check:
#     - The loop over bg components assumes boundary-touching bg must remain unchanged; the generator
#       preserves a full bg frame, so this is safe.
#     - The area->color lookup assumes no unsupported hole areas appear; templates only create areas
#       1..4.
#     - The solver does not group border components at all, so multicolor shells cannot break it.


_AREA_TO_FILL = {1: 2, 2: 3, 3: 4, 4: 6}
_BORDER_POOL = [1, 5, 7, 8, 9]

_GRID_H_MIN, _GRID_H_MAX = 4, 13
_GRID_W_MIN, _GRID_W_MAX = 10, 15
_MAX_RETRIES = 300


def _mask_from_rows(rows: list[str]) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)


_TEMPLATES = {
    1: [
        _mask_from_rows(["111", "101", "111"]),
        _mask_from_rows(["1110", "1011", "1110"]),
        _mask_from_rows(["111", "101", "111", "010"]),
    ],
    2: [
        _mask_from_rows(["1111", "1001", "1111"]),
        _mask_from_rows(["111", "101", "101", "111"]),
    ],
    3: [
        _mask_from_rows(["11111", "10001", "11111"]),
        _mask_from_rows(["111", "101", "101", "101", "111"]),
    ],
    4: [
        _mask_from_rows(["1111", "1001", "1001", "1111"]),
        _mask_from_rows(["1111", "1001", "1001", "1111", "0010"]),
    ],
}


def _bg_boundary(g: np.ndarray) -> int:
    """Most common color along the 1-cell outer border."""
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _get_enclosed_holes(mask: np.ndarray) -> list[np.ndarray]:
    """Return enclosed bg components inside a sprite mask."""
    h, w = mask.shape
    return [
        pts
        for pts in components_mask(~mask, connectivity=4)
        if not any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts)
    ]


def _validate_templates() -> None:
    for area, masks in _TEMPLATES.items():
        for mask in masks:
            holes = _get_enclosed_holes(mask)
            assert len(holes) == 1, f"template area {area} must have exactly one hole"
            assert len(holes[0]) == area, f"template area {area} hole size mismatch"


_validate_templates()


def _fill_enclosed_bg_regions(out: np.ndarray, bg: int) -> None:
    """Fill each non-boundary bg component according to its area."""
    for pts in components_mask(out == bg, connectivity=4):
        if any(r == 0 or c == 0 or r == out.shape[0] - 1 or c == out.shape[1] - 1 for r, c in pts):
            continue
        fill = _AREA_TO_FILL.get(len(pts))
        if fill is None:
            continue
        for r, c in pts:
            out[int(r), int(c)] = int(fill)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Refill hollow multicolor sprites by measuring each enclosed hole's area.

    Input structure:
    - Background dominates the outer border, and there is no palette legend anywhere on the edge.
    - Several separated hollow sprites sit inside the grid, and each border may mix multiple colors.
    - The sprite interiors are blank background regions of small areas such as 1, 2, 3, or 4 cells.

    Rule:
    - Ignore the border colors completely.
    - Find every enclosed background region that does not touch the grid boundary.
    - Fill each such hole from the fixed area mapping: 1->2, 2->3, 3->4, 4->6.

    Key insight: The controlling signal is the size of the empty interior, while the multicolor shell is only a distraction layer left unchanged in the output.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    out = g.copy()
    _fill_enclosed_bg_regions(out, bg)
    return out


def _split_extra(total: int, parts: int, rng: random.Random) -> list[int]:
    vals = [0] * parts
    for _ in range(total):
        vals[rng.randrange(parts)] += 1
    return vals


def _adjacent_border_cells(mask: np.ndarray, hole: np.ndarray) -> set[tuple[int, int]]:
    border_cells = {(int(r), int(c)) for r, c in np.argwhere(mask)}
    out: set[tuple[int, int]] = set()
    for r, c in hole:
        for rr, cc in neighbors4(int(r), int(c)):
            if (rr, cc) in border_cells:
                out.add((rr, cc))
    return out


def _colorize_border(
    mask: np.ndarray,
    major: int,
    accent1: int,
    accent2: int | None,
    rng: random.Random,
) -> tuple[np.ndarray, np.ndarray]:
    """Paint a validated shell with a strict-majority main color plus accent noise."""
    holes = _get_enclosed_holes(mask)
    hole = holes[0]
    border_cells = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    adjacent = list(_adjacent_border_cells(mask, hole))
    rng.shuffle(adjacent)

    sprite = np.zeros(mask.shape, dtype=int)
    for r, c in border_cells:
        sprite[r, c] = int(major)

    max_accents = max(2, (len(border_cells) - 1) // 2)
    accent_total = min(max_accents, max(2, min(len(adjacent), len(border_cells) // 3 + 1)))
    accent2_count = 1 if accent2 is not None and accent_total >= 3 and rng.random() < 0.6 else 0
    accent1_count = accent_total - accent2_count

    chosen: list[tuple[int, int]] = adjacent[:accent1_count]
    if len(chosen) < accent1_count:
        remaining = [cell for cell in border_cells if cell not in chosen]
        rng.shuffle(remaining)
        chosen.extend(remaining[: accent1_count - len(chosen)])
    for r, c in chosen:
        sprite[r, c] = int(accent1)

    if accent2_count:
        remaining = [cell for cell in border_cells if sprite[cell] == major]
        rng.shuffle(remaining)
        for r, c in remaining[:accent2_count]:
            sprite[r, c] = int(accent2)

    return sprite, hole


def _choose_sprite_specs(rng: random.Random) -> list[dict]:
    """Build four sprite specs with explicit anti-border-cue structure."""
    dup_area = rng.choice([1, 2, 3])
    other_areas = rng.sample([a for a in _AREA_TO_FILL if a != dup_area], 2)
    areas = [dup_area, dup_area, other_areas[0], other_areas[1]]

    major0, major1, major2 = rng.sample(_BORDER_POOL, 3)
    majors = [major0, major1, major0, major2]

    specs: list[dict] = []
    for area, major in zip(areas, majors, strict=False):
        accent_choices = [c for c in _BORDER_POOL if c != major]
        accent1 = rng.choice(accent_choices)
        accent2_pool = [c for c in accent_choices if c != accent1]
        accent2 = rng.choice(accent2_pool) if accent2_pool and rng.random() < 0.6 else None
        mask = rng.choice(_TEMPLATES[area]).copy()
        sprite, hole = _colorize_border(mask, major, accent1, accent2, rng)
        specs.append(
            {
                "area": int(area),
                "fill": int(_AREA_TO_FILL[area]),
                "major": int(major),
                "mask": mask,
                "sprite": sprite,
                "hole": hole,
            }
        )

    rng.shuffle(specs)
    return specs


def _layout_positions(specs: list[dict], rng: random.Random) -> tuple[int, int, list[tuple[int, int]]] | None:
    """Place four sprites in a padded 2x2 arrangement."""
    dims = [(int(spec["mask"].shape[0]), int(spec["mask"].shape[1])) for spec in specs]
    (h0, w0), (h1, w1), (h2, w2), (h3, w3) = dims

    left_w = max(w0, w2)
    right_w = max(w1, w3)
    top_h = max(h0, h1)
    bottom_h = max(h2, h3)

    min_h = 1 + top_h + 1 + bottom_h + 1
    min_w = 1 + left_w + 1 + right_w + 1
    if min_h > _GRID_H_MAX or min_w > _GRID_W_MAX:
        return None

    H = rng.randint(max(_GRID_H_MIN, min_h), _GRID_H_MAX)
    W = rng.randint(max(_GRID_W_MIN, min_w), _GRID_W_MAX)
    extra_h = _split_extra(H - min_h, 3, rng)
    extra_w = _split_extra(W - min_w, 3, rng)

    top_margin = 1 + extra_h[0]
    row_gap = 1 + extra_h[1]
    left_margin = 1 + extra_w[0]
    col_gap = 1 + extra_w[1]

    top_y = top_margin
    bottom_y = top_margin + top_h + row_gap
    left_x = left_margin
    right_x = left_margin + left_w + col_gap

    slots = [
        (top_y, left_x, top_h, left_w),
        (top_y, right_x, top_h, right_w),
        (bottom_y, left_x, bottom_h, left_w),
        (bottom_y, right_x, bottom_h, right_w),
    ]

    positions: list[tuple[int, int]] = []
    for spec, (sr, sc, sh, sw) in zip(specs, slots, strict=False):
        h, w = spec["mask"].shape
        dr = rng.randint(0, sh - h)
        dc = rng.randint(0, sw - w)
        positions.append((sr + dr, sc + dc))
    return H, W, positions


def _paint_sprite(grid: np.ndarray, r0: int, c0: int, sprite: np.ndarray) -> None:
    patch = grid[r0 : r0 + sprite.shape[0], c0 : c0 + sprite.shape[1]]
    mask = sprite != 0
    patch[mask] = sprite[mask]


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    bg = 0
    specs = _choose_sprite_specs(rng)
    layout = _layout_positions(specs, rng)
    if layout is None:
        return None
    H, W, positions = layout

    inp = np.full((H, W), bg, dtype=int)
    output = np.full((H, W), bg, dtype=int)
    for spec, (r0, c0) in zip(specs, positions, strict=False):
        _paint_sprite(inp, r0, c0, spec["sprite"])
        _paint_sprite(output, r0, c0, spec["sprite"])
        for hr, hc in spec["hole"]:
            output[r0 + int(hr), c0 + int(hc)] = int(spec["fill"])

    if not np.array_equal(main(inp), output):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate input after {_MAX_RETRIES} attempts")
