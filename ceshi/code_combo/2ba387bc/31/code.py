# Combo variant of 2ba387bc (index 31)
# combo_spec: D1/0+D11/1
# Name: 图案干扰方块+边框一致性分类

from __future__ import annotations

import numpy as np

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, paste, random_free_location


"""
Design analysis for combo D1/0 + D11/1
======================================
source_task:
- Original 2ba387bc sorts scattered 4x4 objects into two output columns.
- Original meaningful families:
  1. frame: monochrome 1px border + bg 2x2 hole
  2. solid: one non-bg color on all 16 cells
- Original class rule was "hole means left, otherwise right".

a) Concretization and behavior trace
- Fused input now contains three visible 4x4 families:
  1. valid uniform-border square:
     - outer border has one non-bg color
     - 2x2 center is monochrome (bg or one fill color)
     - output: keep, stack in the left column
  2. valid mixed-border square:
     - outer border has multiple non-bg colors
     - 2x2 center is monochrome (bg or one fill color)
     - output: keep, stack in the right column
  3. distractor square:
     - still 4x4 and connected, but its 2x2 center is patterned instead of monochrome
     - some distractors intentionally have a monochrome border, so "single border color"
       alone is not enough to decide that a square is meaningful
     - output: ignore completely
- Concrete trace:
  input objects by top row:
  - r=2: [blue border, bg hole] -> valid, border monochrome -> left
  - r=7: [yellow border, pink/green checker center] -> border monochrome but center patterned
    -> distractor -> ignored
  - r=12: [red/teal split border, black hole] -> valid, border mixed -> right
  - r=17: [orange solid 4x4] -> valid, border monochrome -> left
  Original solver path:
  - blue frame -> left
  - yellow distractor -> not a frame -> incorrectly sent right
  - red/teal mixed-border hole -> not a frame -> right
  - orange solid -> right
  Fused solver path:
  - blue frame-like square -> left
  - yellow distractor -> ignored
  - red/teal mixed-border hole -> right
  - orange solid -> left because the border is monochrome
  Main differences:
  - classification changed from "look at center hole vs fill" to "look at border uniformity"
  - distractors force an extra validity test, otherwise monochrome-border noise would leak into
    the left column

b) Complexity calibration
- The valid family is restricted to one clean syntax: full 4x4 border, all border cells non-bg,
  and a monochrome 2x2 center. This keeps the task readable.
- We still show both original-looking subtypes and new ones:
  - left column contains both hole-centered and solid/filled-centered examples
  - right column contains both hole-centered and filled-centered examples
  This gives repeated evidence that center fill is not the class criterion anymore.
- Distractors come from a small library of patterned-center 4x4 squares. At least one distractor
  per sample has a monochrome border, so the D11 change actively interacts with D1 instead of
  being visually separate.

c) Implicit-assumption audit of the original main()
- `find_components(... monochromatic=False)` assumed multi-color 4x4 shapes can still be treated
  as one component. This remains valid and is now essential for mixed-border squares.
- `bbox_h(b) != 4 or bbox_w(b) != 4` assumed only 4x4 candidates matter. Still valid.
- Original border/interior test:
  `if len(border_colors) == 1 and center == bg: frame else: solid`
  depended on two broken assumptions:
  1. every non-frame 4x4 should be kept
  2. center hole vs non-hole is the true class signal
  In the combo, neither is true. We must first test whether the square is valid at all
  (full non-bg border + monochrome 2x2 center), then split valid squares by border uniformity.
- Sorting by `(row, col)` is unchanged and still matches the source task.

d) Placement legality
- Every placed patch uses `random_free_location(..., border=1, padding=1, padding_connectivity=4)`.
  This guarantees no overlaps, no touching components, and no out-of-bounds placement.
- All generated distractors stay connected as foreground under `monochromatic=False`, so
  component detection sees each placed 4x4 as one candidate object.

e) Visual observability
- A human can infer the new rule from a few examples because:
  - some left-column outputs are solid/filled blocks, so "hole => left" is disproved
  - some right-column outputs have bg holes, so "hole => left" is disproved again from the
    opposite direction
  - some input squares with monochrome borders disappear, showing that border uniformity alone
    does not imply membership; the center must also be structurally regular
- No hidden ordering or color-id arithmetic is used; only visible border/center structure matters.

f) Visual cleanliness
- Objects are isolated with at least one-cell padding, so 4x4 candidates are easy to inspect.
- Distractors use only a few crisp templates such as checker centers or row/column stripes,
  avoiding noisy mosaics.
- The right-column mixed-border valid squares use a simple split border syntax so the class signal
  is visually salient.

g) Change inferability
- Colors are preserved verbatim from input to output; there is no remapping.
- The transformation depends on visible geometry:
  - valid if the border is complete and the center is one color
  - among valid ones, left if border is one color, right if border uses multiple colors
- Because the generator always includes repeated support for both center modes on both sides,
  the class rule is inferable from structure rather than memorizing object templates.

h) Generator correctness strategy
- The generator records every placed valid object, constructs the expected output directly, and
  checks `main(input) == expected`.
- It also re-runs the solver-side collector and checks that the realized counts of left/right/
  distractor objects match the intended counts.
- The sample recipe guarantees non-degeneracy versus the original task by always including:
  - at least one valid monochrome-border non-hole square (new solver sends it left, original sent
    it right)
  - at least one monochrome-border distractor (new solver ignores it, naive border-only logic
    would misclassify it)

i) New solver assumptions introduced by this combo
- A meaningful object must have all 12 border cells non-bg and a monochrome 2x2 center.
  The generator guarantees this for every kept object.
- Distractors are exactly the remainder class among connected 4x4 candidates; the generator
  guarantees at least one distractor has a monochrome border so the validity gate is exercised.
- Mixed-border valid squares are generated with two or more border colors but still one center
  color, so the right-column branch is always supported by visible examples.
"""


def _border(sub: np.ndarray) -> np.ndarray:
    return np.concatenate([sub[0, :], sub[-1, :], sub[1:-1, 0], sub[1:-1, -1]])


def _center(sub: np.ndarray) -> np.ndarray:
    return sub[1:3, 1:3]


def _is_valid_square(sub: np.ndarray, bg: int) -> bool:
    border = _border(sub)
    center = _center(sub)
    return bool((border != int(bg)).all()) and len(set(map(int, center.flatten()))) == 1


def _classify_4x4(sub: np.ndarray, bg: int) -> str | None:
    if not _is_valid_square(sub, bg):
        return None
    border_cols = set(map(int, _border(sub).flatten()))
    if len(border_cols) == 1:
        return "uniform"
    return "mixed"


def _collect_objects(
    grid: np.ndarray, bg: int
) -> tuple[list[tuple[int, int, np.ndarray]], list[tuple[int, int, np.ndarray]], list[tuple[int, int, np.ndarray]]]:
    comps = find_components(grid, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    uniforms: list[tuple[int, int, np.ndarray]] = []
    mixeds: list[tuple[int, int, np.ndarray]] = []
    distractors: list[tuple[int, int, np.ndarray]] = []

    for comp in comps:
        b = comp["bbox"]
        if bbox_h(b) != 4 or bbox_w(b) != 4:
            continue
        r0, c0, _r1, _c1 = b
        sub = crop_bbox(grid, b)
        kind = _classify_4x4(sub, bg)
        if kind == "uniform":
            uniforms.append((int(r0), int(c0), sub))
        elif kind == "mixed":
            mixeds.append((int(r0), int(c0), sub))
        else:
            distractors.append((int(r0), int(c0), sub))

    uniforms.sort(key=lambda t: (int(t[0]), int(t[1])))
    mixeds.sort(key=lambda t: (int(t[0]), int(t[1])))
    distractors.sort(key=lambda t: (int(t[0]), int(t[1])))
    return uniforms, mixeds, distractors


def _build_output(
    uniforms: list[tuple[int, int, np.ndarray]],
    mixeds: list[tuple[int, int, np.ndarray]],
    bg: int,
) -> np.ndarray:
    out_h = 4 * max(int(len(uniforms)), int(len(mixeds)))
    out = np.full((out_h, 8), int(bg), dtype=int)
    for i, (_r0, _c0, sub) in enumerate(uniforms):
        paste(out, sub, int(i * 4), 0)
    for i, (_r0, _c0, sub) in enumerate(mixeds):
        paste(out, sub, int(i * 4), 4)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort scattered valid 4x4 squares into two columns.

    Valid square:
    - 4x4 bounding box
    - all outer-border cells are non-background
    - 2x2 center is one color

    Classification:
    - uniform border color -> left column
    - mixed border colors -> right column
    - any other connected 4x4 pattern -> ignored
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    uniforms, mixeds, _distractors = _collect_objects(g, bg)
    return _build_output(uniforms, mixeds, bg)


def _rand_color(*, exclude: set[int] | None = None) -> int:
    exclude = set() if exclude is None else set(map(int, exclude))
    choices = [c for c in range(1, 10) if c not in exclude]
    return int(np.random.choice(choices))


def _make_uniform_square(mode: str, border_col: int, bg: int, fill_col: int | None = None) -> np.ndarray:
    patch = np.full((4, 4), int(border_col), dtype=int)
    if mode == "hole":
        patch[1:3, 1:3] = int(bg)
        return patch
    if mode == "solid":
        return patch
    if mode == "fill":
        if fill_col is None:
            raise ValueError("fill mode requires fill_col")
        patch[1:3, 1:3] = int(fill_col)
        return patch
    raise ValueError(f"unknown uniform mode: {mode}")


def _make_mixed_square(mode: str, col_a: int, col_b: int, bg: int, fill_col: int | None = None) -> np.ndarray:
    patch = np.empty((4, 4), dtype=int)
    patch[0, :] = [int(col_a), int(col_a), int(col_b), int(col_b)]
    patch[3, :] = [int(col_a), int(col_a), int(col_b), int(col_b)]
    patch[1, 0] = int(col_a)
    patch[2, 0] = int(col_a)
    patch[1, 3] = int(col_b)
    patch[2, 3] = int(col_b)
    if mode == "hole":
        patch[1:3, 1:3] = int(bg)
        return patch
    if mode == "fill":
        if fill_col is None:
            raise ValueError("fill mode requires fill_col")
        patch[1:3, 1:3] = int(fill_col)
        return patch
    raise ValueError(f"unknown mixed mode: {mode}")


def _make_distractor(kind: str, colors: tuple[int, ...], bg: int) -> np.ndarray:
    if kind == "mono_checker_center":
        border_col, center_a, center_b = map(int, colors)
        patch = np.full((4, 4), border_col, dtype=int)
        patch[1, 1] = center_a
        patch[1, 2] = center_b
        patch[2, 1] = center_b
        patch[2, 2] = center_a
        return patch
    if kind == "checker":
        col_a, col_b = map(int, colors[:2])
        patch = np.empty((4, 4), dtype=int)
        for r in range(4):
            for c in range(4):
                patch[r, c] = col_a if (r + c) % 2 == 0 else col_b
        return patch
    if kind == "hstripe":
        col_a, col_b = map(int, colors[:2])
        patch = np.full((4, 4), col_a, dtype=int)
        patch[2:, :] = col_b
        return patch
    if kind == "ring_stripe_center":
        border_col, center_a, center_b = map(int, colors)
        patch = np.full((4, 4), border_col, dtype=int)
        patch[1:3, 1] = center_a
        patch[1:3, 2] = center_b
        return patch
    raise ValueError(f"unknown distractor kind: {kind}")


def _place_patch(
    grid: np.ndarray,
    patch: np.ndarray,
    bg: int,
) -> tuple[int, int]:
    r0, c0 = random_free_location(grid, patch, transparent=bg, border=1, padding=1, padding_connectivity=4)
    paste(grid, patch, r0, c0, transparent=bg)
    return int(r0), int(c0)


def _gen_attempt() -> np.ndarray | None:
    bg = 0
    h = int(np.random.randint(26, 38))
    w = int(np.random.randint(26, 38))
    grid = np.full((h, w), bg, dtype=int)

    n_uniform = int(np.random.randint(2, 5))
    n_mixed = int(np.random.randint(2, 5))
    n_distractors = int(np.random.randint(2, 5))

    uniform_modes = ["hole", "solid"]
    while len(uniform_modes) < n_uniform:
        uniform_modes.append(str(np.random.choice(["hole", "solid", "fill"])))

    mixed_modes = ["hole", "fill"]
    while len(mixed_modes) < n_mixed:
        mixed_modes.append(str(np.random.choice(["hole", "fill"])))

    distractor_specs: list[str] = ["mono_checker_center"]
    distractor_library = ["checker", "hstripe", "ring_stripe_center"]
    while len(distractor_specs) < n_distractors:
        distractor_specs.append(str(np.random.choice(distractor_library)))

    records: list[tuple[str, int, int, np.ndarray]] = []

    try:
        for mode in uniform_modes:
            border_col = _rand_color()
            fill_col = None
            if mode == "fill":
                fill_col = _rand_color(exclude={border_col})
            patch = _make_uniform_square(mode, border_col, bg, fill_col)
            if _classify_4x4(patch, bg) != "uniform":
                return None
            r0, c0 = _place_patch(grid, patch, bg)
            records.append(("uniform", int(r0), int(c0), patch))

        for mode in mixed_modes:
            col_a = _rand_color()
            col_b = _rand_color(exclude={col_a})
            fill_col = None
            if mode == "fill":
                fill_col = _rand_color(exclude={col_a, col_b})
            patch = _make_mixed_square(mode, col_a, col_b, bg, fill_col)
            if _classify_4x4(patch, bg) != "mixed":
                return None
            r0, c0 = _place_patch(grid, patch, bg)
            records.append(("mixed", int(r0), int(c0), patch))

        for kind in distractor_specs:
            if kind in {"mono_checker_center", "ring_stripe_center"}:
                border_col = _rand_color()
                center_a = _rand_color(exclude={border_col})
                center_b = _rand_color(exclude={border_col, center_a})
                patch = _make_distractor(kind, (border_col, center_a, center_b), bg)
            else:
                col_a = _rand_color()
                col_b = _rand_color(exclude={col_a})
                patch = _make_distractor(kind, (col_a, col_b), bg)
            if _classify_4x4(patch, bg) is not None:
                return None
            r0, c0 = _place_patch(grid, patch, bg)
            records.append(("distractor", int(r0), int(c0), patch))

        expected_uniforms = [(r0, c0, patch) for kind, r0, c0, patch in records if kind == "uniform"]
        expected_mixeds = [(r0, c0, patch) for kind, r0, c0, patch in records if kind == "mixed"]
        expected_uniforms.sort(key=lambda t: (int(t[0]), int(t[1])))
        expected_mixeds.sort(key=lambda t: (int(t[0]), int(t[1])))
        expected = _build_output(expected_uniforms, expected_mixeds, bg)

        got = main(grid)
        if not np.array_equal(got, expected):
            return None

        uniforms_seen, mixeds_seen, distractors_seen = _collect_objects(grid, bg)
        if len(uniforms_seen) != n_uniform or len(mixeds_seen) != n_mixed or len(distractors_seen) != n_distractors:
            return None

        mono_border_distractors = 0
        for _r0, _c0, sub in distractors_seen:
            if len(set(map(int, _border(sub).flatten()))) == 1:
                mono_border_distractors += 1
        if mono_border_distractors < 1:
            return None

        return grid
    except Exception:
        return None


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 500 attempts")
