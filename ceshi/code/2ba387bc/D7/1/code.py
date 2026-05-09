# Variant of 2ba387bc
# Dimension: D7 — 变换——堆叠时的颜色处理 (transform)
# Change: 方块在分类和堆叠过程中保持原始颜色不变 → 框架变实心，实心变框架

from __future__ import annotations

import numpy as np
from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, paste, random_free_location


# Design analysis for the D7 variant
# a) Concrete behavior:
#    Input stays the same as the original task: several separated 4x4 monochrome objects on a uniform background.
#    A frame is a 4x4 ring with a bg-colored 2x2 hole; a solid is a 4x4 fully filled square.
#    Classification still uses the input shape only. The change happens after sorting/stacking:
#    every frame is output as a solid of the same non-bg color, and every solid is output as a frame
#    of the same color. Example with bg=0:
#      input frame 3 -> output solid 3
#      [[3,3,3,3],      [[3,3,3,3],
#       [3,0,0,3],  ->   [3,3,3,3],
#       [3,0,0,3],       [3,3,3,3],
#       [3,3,3,3]]       [3,3,3,3]]
#      input solid 8 -> output frame 8
#      [[8,8,8,8],      [[8,8,8,8],
#       [8,8,8,8],  ->   [8,0,0,8],
#       [8,8,8,8],       [8,0,0,8],
#       [8,8,8,8]]       [8,8,8,8]]
#    The pipeline difference is therefore localized to the final paste payload, not the grouping/sorting.
# b) Complexity calibration:
#    The variant is visually obvious because each object changes four interior cells in output.
#    We keep both object classes present in every generated sample so the "swap form" rule is always evidenced
#    on both output columns. Colors remain few (1..9, unique per object) and are not used for any logic.
# c) Implicit-assumption audit for main():
#    - `find_components(... monochromatic=False)` assumes each 4x4 object is a single connected component.
#      This still holds because frame borders are 4-connected rings and solids are filled blocks.
#    - `bbox_h/bbox_w == 4` assumes only relevant objects are 4x4. Generator preserves that.
#    - `len(bcols) == 1 and interior == bg` is the frame test. This remains valid because input shapes
#      are unchanged by the variant; only output rendering changes.
#    - Original `paste(out, sub, ...)` relied on "output shape == input shape". That assumption is false now,
#      so `sub` must be replaced by a transformed 4x4 patch before pasting.
# d) Placement legality:
#    Generator uses `random_free_location(..., border=1, padding=1, padding_connectivity=4)` so 4x4 objects
#    never overlap, never touch orthogonally, and stay in-bounds. The solver will thus see each object as an
#    isolated component with a clean 4x4 bbox.
# e) Visual observability:
#    A human sees scattered frames and solids in input, but in output the left column contains filled versions
#    of the input frames and the right column contains hollow versions of the input solids. This is inferable
#    from 2-4 examples because the same colors persist while the 2x2 interior toggles between bg/non-bg.
# f) Visual cleanliness:
#    Objects are sparse, axis-aligned, and non-overlapping; output is the same tidy 8-column stack as original.
# g) Color-change inferability:
#    No color remapping occurs. Only the occupancy of the inner 2x2 changes, so the rule is structural rather
#    than palette-dependent.
# h) Variant-specific self-check:
#    For each generated grid we analytically construct the expected transformed output and reject the sample if
#    it accidentally matches the original task's output. With at least one frame and one solid, changed cells
#    must equal 4 * (n_frames + n_solids), so degeneration is prevented.
# i) New assumptions introduced by the modified main():
#    The transformation helper assumes each valid object has exactly one non-bg color. Generator guarantees this
#    by constructing only monochrome frames/solids.


def _make_frame(col: int, bg: int = 0) -> np.ndarray:
    a = np.full((4, 4), int(col), dtype=int)
    a[1:3, 1:3] = int(bg)
    return a


def _make_solid(col: int) -> np.ndarray:
    return np.full((4, 4), int(col), dtype=int)


def _is_frame(sub: np.ndarray, bg: int) -> bool:
    border = np.concatenate([sub[0, :], sub[-1, :], sub[1:-1, 0], sub[1:-1, -1]])
    bcols = set(map(int, border.flatten()))
    return len(bcols) == 1 and bool((sub[1:3, 1:3] == int(bg)).all())


def _object_color(sub: np.ndarray, bg: int) -> int:
    vals = sub[sub != int(bg)]
    if vals.size == 0:
        raise ValueError("4x4 object has no non-background cells")
    return int(vals.flat[0])


def _swap_form(sub: np.ndarray, bg: int) -> np.ndarray:
    col = _object_color(sub, bg)
    if _is_frame(sub, bg):
        return _make_solid(col)
    return _make_frame(col, bg)


def _expected_from_specs(specs: list[tuple[int, int, bool, int]], bg: int, *, swap_form: bool) -> np.ndarray:
    frames = [(r0, c0, col) for r0, c0, is_frame, col in specs if is_frame]
    solids = [(r0, c0, col) for r0, c0, is_frame, col in specs if not is_frame]
    frames.sort(key=lambda t: (int(t[0]), int(t[1])))
    solids.sort(key=lambda t: (int(t[0]), int(t[1])))

    out = np.full((4 * max(len(frames), len(solids)), 8), int(bg), dtype=int)

    for i, (_r0, _c0, col) in enumerate(frames):
        patch = _make_solid(col) if swap_form else _make_frame(col, bg)
        paste(out, patch, i * 4, 0)
    for i, (_r0, _c0, col) in enumerate(solids):
        patch = _make_frame(col, bg) if swap_form else _make_solid(col)
        paste(out, patch, i * 4, 4)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort scattered 4x4 squares into two columns by type: input frames left, input solids right.

    Variant rule:
    - Classification and ordering are unchanged from 2ba387bc.
    - When stacked into the output, each frame becomes a solid of the same color.
    - When stacked into the output, each solid becomes a frame of the same color.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    frames: list[tuple[int, int, np.ndarray]] = []
    solids: list[tuple[int, int, np.ndarray]] = []

    for comp in comps:
        b = comp["bbox"]
        if bbox_h(b) != 4 or bbox_w(b) != 4:
            continue
        r0, c0, _r1, _c1 = b
        sub = crop_bbox(g, b)

        if _is_frame(sub, bg):
            frames.append((int(r0), int(c0), sub))
        else:
            solids.append((int(r0), int(c0), sub))

    out_h = 4 * max(int(len(frames)), int(len(solids)))
    out = np.full((out_h, 8), int(bg), dtype=int)

    frames.sort(key=lambda t: (int(t[0]), int(t[1])))
    solids.sort(key=lambda t: (int(t[0]), int(t[1])))

    for i, (_r0, _c0, sub) in enumerate(frames):
        paste(out, _swap_form(sub, bg), int(i * 4), 0)
    for i, (_r0, _c0, sub) in enumerate(solids):
        paste(out, _swap_form(sub, bg), int(i * 4), 4)

    return out


def _gen_attempt() -> np.ndarray | None:
    bg = 0
    h = int(np.random.randint(18, 30))
    w = int(np.random.randint(18, 30))
    g = np.full((h, w), bg, dtype=int)
    n_frames = int(np.random.randint(1, 5))
    n_solids = int(np.random.randint(1, 5))

    try:
        colors = list(map(int, np.random.choice(np.arange(1, 10), size=n_frames + n_solids, replace=False)))
        specs: list[tuple[int, int, bool, int]] = []

        for col in colors[:n_frames]:
            patch = _make_frame(col, bg)
            r0, c0 = random_free_location(
                g,
                patch,
                transparent=bg,
                border=1,
                padding=1,
                padding_connectivity=4,
            )
            paste(g, patch, r0, c0, transparent=bg)
            specs.append((int(r0), int(c0), True, int(col)))

        for col in colors[n_frames:]:
            patch = _make_solid(col)
            r0, c0 = random_free_location(
                g,
                patch,
                transparent=bg,
                border=1,
                padding=1,
                padding_connectivity=4,
            )
            paste(g, patch, r0, c0, transparent=bg)
            specs.append((int(r0), int(c0), False, int(col)))

        expected_variant = _expected_from_specs(specs, bg, swap_form=True)
        expected_original = _expected_from_specs(specs, bg, swap_form=False)

        if np.array_equal(expected_variant, expected_original):
            return None
        if not np.array_equal(main(g), expected_variant):
            return None

        return g
    except Exception:
        return None


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 300 attempts")
