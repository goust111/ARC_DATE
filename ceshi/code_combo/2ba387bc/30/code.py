# Combo variant of 2ba387bc (index 30)
# combo_spec: D11/0+D12/0
# Name: 非标尺寸干扰+非黑背景

from __future__ import annotations

import numpy as np
from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, paste, random_free_location

# Design analysis for the combo variant
# a) Concrete behavior and trace:
#    - Original task: bg is usually 0, and every non-bg blob is exactly one isolated
#      4x4 object. `main()` can therefore detect multi-color foreground components as
#      whole objects and classify each 4x4 bbox by its interior.
#    - Combo input: bg is deliberately non-black (here sampled from {5, 8}). Valid
#      objects are still the same 4x4 frames/solids, but extra non-4x4 filled
#      distractors are also scattered around. At least one distractor is orthogonally
#      attached to a valid square using a different color.
#    - Example trace: suppose bg=5, a red frame sits at rows 6:9 and a blue 2x3 bar is
#      attached to its right edge. The legacy solver with `monochromatic=False` merges
#      the red frame and blue bar into one multi-color component whose bbox is not 4x4,
#      so the frame disappears. The combo solver first infers bg=5 via `get_bg`, then
#      finds monochromatic components, recovers the red 4x4 frame as its own component,
#      ignores the blue 2x3 distractor by size, and stacks the frame into output.
#    - Difference summary: D12 removes the hidden shortcut "background is 0"; D11
#      removes the hidden shortcut "one foreground blob equals one semantic object".
#      Together they require identifying bg by frequency and identifying valid objects
#      by monochromatic 4x4 shape checks.
# b) Complexity calibration:
#    - Distractors are restricted to a small rectangle vocabulary such as 1x5, 2x3,
#      5x2, so they are visually obvious and do not explode the search space.
#    - Valid objects and distractors all use distinct colors within one sample to keep
#      monochromatic components separable even when distractors touch valid objects.
#    - Bg is chosen from two non-black values, enough to invalidate bg==0 assumptions
#      without making the task look like arbitrary color remapping.
# c) Implicit assumption audit for the original `main()`:
#    - `bg = int(get_bg(g))` still holds, but only if every later step uses this `bg`
#      instead of hard-coding 0. The combo generator guarantees the true background is
#      still the majority color.
#    - `find_components(... monochromatic=False)` no longer holds because an attached
#      distractor can fuse with a valid square into a larger blob. The solver must use
#      `monochromatic=True`.
#    - `if bbox_h(b) != 4 or bbox_w(b) != 4: continue` still holds as a first filter,
#      but now it is applied to monochromatic components, not to whole multi-color
#      foreground blobs.
#    - `else: solids.append(...)` was too permissive in the original code because every
#      4x4 non-frame blob was assumed to be a solid. The combo solver makes the solid
#      condition explicit to avoid false positives.
# d) Placement legality:
#    - Valid 4x4 objects are placed first with padding so they do not overlap or touch
#      each other directly.
#    - Distractors are filled rectangles placed only on bg cells. Attached distractors
#      may touch exactly one chosen valid object but never overlap it and never touch a
#      second object.
#    - Frame interiors remain true bg cells because valid squares reserve the full 4x4
#      footprint before any transparent pasting.
# e) Visual observability:
#    - Across a few examples, the output always keeps only 4x4 frames/solids while
#      omitting extra bars/rectangles, so a human can infer the size filter.
#    - The non-black background is visible in both the input canvas and the interior of
#      every frame, so a human can infer that "frame interior matches the actual
#      background", not "frame interior is black".
# f) Visual cleanliness:
#    - The number of distractors stays small, they remain rectangular, and at most one
#      forced attachment is required. This keeps the input legible rather than noisy.
#    - Distinct colors per component prevent ambiguous mergers in the monochromatic
#      solver and make each object visually separable.
# g) Change inferability:
#    - No hidden color mapping is introduced. Colors keep their identity; only shape,
#      size, and bg-relative interior matter. This is inferable from 2-4 examples.
# Post-change solver assumption audit:
#    - The revised solver assumes valid objects themselves are monochromatic by border
#      color and still axis-aligned 4x4. The generator guarantees that.
#    - The revised solver assumes non-valid monochromatic 4x4 patterns are absent. The
#      generator only emits true frames, true solids, and non-4x4 distractors.
#    - The revised solver assumes the true bg is the most common color. The generator
#      uses roomy canvases so this remains true despite added distractors.


_FRAME_MASK = np.ones((4, 4), dtype=bool)
_FRAME_MASK[1:3, 1:3] = False

_DISTRACTOR_SHAPES: tuple[tuple[int, int], ...] = (
    (1, 5),
    (5, 1),
    (2, 3),
    (3, 2),
    (2, 5),
    (5, 2),
    (3, 5),
    (5, 3),
)

_BG_CHOICES: tuple[int, ...] = (5, 8)


def _make_frame(col: int, bg: int) -> np.ndarray:
    a = np.full((4, 4), int(col), dtype=int)
    a[1:3, 1:3] = int(bg)
    return a


def _make_solid(col: int) -> np.ndarray:
    return np.full((4, 4), int(col), dtype=int)


def _classify_valid_square(sub: np.ndarray, color: int, area: int, bg: int) -> str | None:
    if sub.shape != (4, 4):
        return None

    color = int(color)
    area = int(area)
    bg = int(bg)

    if area == 16 and bool((sub == color).all()):
        return "solid"

    if area == 12 and bool((sub[_FRAME_MASK] == color).all()) and bool((sub[1:3, 1:3] == bg).all()):
        return "frame"

    return None


def _stack_output(frames: list[tuple[int, int, np.ndarray]], solids: list[tuple[int, int, np.ndarray]], bg: int) -> np.ndarray:
    out_h = 4 * max(int(len(frames)), int(len(solids)))
    out = np.full((out_h, 8), int(bg), dtype=int)
    for i, (_r0, _c0, sub) in enumerate(frames):
        paste(out, sub, int(i * 4), 0)
    for i, (_r0, _c0, sub) in enumerate(solids):
        paste(out, sub, int(i * 4), 4)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep only true 4x4 frame/solid squares on the inferred non-black background.

    Combo logic:
    - infer bg from frequency, without assuming 0
    - find connected components monochromatically so attached distractors do not erase
      valid squares
    - keep only exact 4x4 frame/solid patterns relative to the inferred bg
    - sort each group by input row/col and stack into left/right output columns
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    frames: list[tuple[int, int, np.ndarray]] = []
    solids: list[tuple[int, int, np.ndarray]] = []

    for comp in comps:
        b = comp["bbox"]
        if bbox_h(b) != 4 or bbox_w(b) != 4:
            continue

        r0, c0, _r1, _c1 = b
        sub = crop_bbox(g, b)
        kind = _classify_valid_square(sub, int(comp["color"]), int(comp["area"]), bg)
        if kind == "frame":
            frames.append((int(r0), int(c0), sub))
        elif kind == "solid":
            solids.append((int(r0), int(c0), sub))

    frames.sort(key=lambda t: (int(t[0]), int(t[1])))
    solids.sort(key=lambda t: (int(t[0]), int(t[1])))
    return _stack_output(frames, solids, bg)


def _legacy_main(grid: np.ndarray) -> np.ndarray:
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
        border = np.concatenate([sub[0, :], sub[-1, :], sub[1:-1, 0], sub[1:-1, -1]])
        bcols = set(map(int, border.flatten()))
        if len(bcols) == 1 and bool((sub[1:3, 1:3] == bg).all()):
            frames.append((int(r0), int(c0), sub))
        else:
            solids.append((int(r0), int(c0), sub))

    frames.sort(key=lambda t: (int(t[0]), int(t[1])))
    solids.sort(key=lambda t: (int(t[0]), int(t[1])))
    return _stack_output(frames, solids, bg)


def _bbox_cells(bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    return {(r, c) for r in range(int(r0), int(r1) + 1) for c in range(int(c0), int(c1) + 1)}


def _can_place_attached(
    g: np.ndarray,
    patch: np.ndarray,
    r0: int,
    c0: int,
    target_bbox: tuple[int, int, int, int],
    bg: int,
) -> bool:
    h, w = map(int, patch.shape)
    H, W = g.shape
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False
    if bool((g[r0 : r0 + h, c0 : c0 + w] != bg).any()):
        return False

    target_cells = _bbox_cells(target_bbox)
    touched_target = False
    touched_other = False
    for rr in range(int(r0), int(r0 + h)):
        for cc in range(int(c0), int(c0 + w)):
            for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                if not (0 <= nr < H and 0 <= nc < W):
                    continue
                if int(g[nr, nc]) == bg:
                    continue
                if (nr, nc) in target_cells:
                    touched_target = True
                else:
                    touched_other = True
    return touched_target and not touched_other


def _attached_location(
    g: np.ndarray,
    patch: np.ndarray,
    target_bbox: tuple[int, int, int, int],
    bg: int,
) -> tuple[int, int] | None:
    r0, c0, r1, c1 = target_bbox
    h, w = map(int, patch.shape)
    candidates: list[tuple[int, int]] = []

    if c0 - w >= 0:
        for rr in range(int(r0 - h + 1), int(r1) + 1):
            candidates.append((rr, int(c0 - w)))
    if c1 + 1 + w <= g.shape[1]:
        for rr in range(int(r0 - h + 1), int(r1) + 1):
            candidates.append((rr, int(c1 + 1)))
    if r0 - h >= 0:
        for cc in range(int(c0 - w + 1), int(c1) + 1):
            candidates.append((int(r0 - h), cc))
    if r1 + 1 + h <= g.shape[0]:
        for cc in range(int(c0 - w + 1), int(c1) + 1):
            candidates.append((int(r1 + 1), cc))

    np.random.shuffle(candidates)
    for rr, cc in candidates:
        if _can_place_attached(g, patch, int(rr), int(cc), target_bbox, bg):
            return int(rr), int(cc)
    return None


def _place_distractor(
    g: np.ndarray,
    patch: np.ndarray,
    bg: int,
    anchors: list[tuple[int, int, int, int]],
    force_attach: bool,
) -> bool:
    order = list(range(len(anchors)))
    np.random.shuffle(order)

    if force_attach:
        for idx in order:
            loc = _attached_location(g, patch, anchors[idx], bg)
            if loc is not None:
                paste(g, patch, loc[0], loc[1], transparent=bg)
                return True
        return False

    if bool(np.random.rand() < 0.6):
        for idx in order:
            loc = _attached_location(g, patch, anchors[idx], bg)
            if loc is not None:
                paste(g, patch, loc[0], loc[1], transparent=bg)
                return True

    try:
        r0, c0 = random_free_location(g, patch, transparent=bg, border=1, padding=1, padding_connectivity=4)
        paste(g, patch, int(r0), int(c0), transparent=bg)
        return True
    except Exception:
        return False


def _analyze_combo_features(g: np.ndarray) -> dict[str, int]:
    bg = int(get_bg(g))
    mono = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")

    valid_bboxes: list[tuple[int, int, int, int]] = []
    distractors: list[dict] = []
    for comp in mono:
        b = comp["bbox"]
        if bbox_h(b) == 4 and bbox_w(b) == 4:
            sub = crop_bbox(g, b)
            if _classify_valid_square(sub, int(comp["color"]), int(comp["area"]), bg) is not None:
                valid_bboxes.append(b)
                continue
        distractors.append(comp)

    valid_cells = set()
    for b in valid_bboxes:
        valid_cells |= _bbox_cells(b)

    attached = 0
    for comp in distractors:
        coords = comp["coords"]
        touch_valid = False
        for rr, cc in coords:
            rr = int(rr)
            cc = int(cc)
            for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                if (int(nr), int(nc)) in valid_cells:
                    touch_valid = True
                    break
            if touch_valid:
                break
        if touch_valid:
            attached += 1

    new_out = main(g)
    old_out = _legacy_main(g)
    legacy_diff = int(new_out.shape != old_out.shape or not bool((new_out == old_out).all()))

    return {
        "bg_non_black": int(bg != 0),
        "valid_count": int(len(valid_bboxes)),
        "distractor_count": int(len(distractors)),
        "attached_distractor_count": int(attached),
        "legacy_diff": legacy_diff,
    }


def _gen_attempt() -> np.ndarray | None:
    bg = int(_BG_CHOICES[int(np.random.randint(0, len(_BG_CHOICES)))])
    H = int(np.random.randint(24, 34))
    W = int(np.random.randint(24, 34))
    g = np.full((H, W), bg, dtype=int)

    n_frames = int(np.random.randint(1, 4))
    n_solids = int(np.random.randint(1, 4))
    n_valid = n_frames + n_solids
    n_distractors = int(np.random.randint(1, min(4, 10 - n_valid)))

    palette = [c for c in range(10) if c != bg]
    colors = list(np.random.permutation(np.array(palette, dtype=int)))
    valid_colors = colors[:n_valid]
    distractor_colors = colors[n_valid : n_valid + n_distractors]

    placed_frames: list[tuple[int, int, np.ndarray]] = []
    placed_solids: list[tuple[int, int, np.ndarray]] = []
    anchor_bboxes: list[tuple[int, int, int, int]] = []

    try:
        valid_specs: list[tuple[str, int, np.ndarray]] = []
        for col in valid_colors[:n_frames]:
            valid_specs.append(("frame", int(col), _make_frame(int(col), bg)))
        for col in valid_colors[n_frames:]:
            valid_specs.append(("solid", int(col), _make_solid(int(col))))
        np.random.shuffle(valid_specs)

        for kind, col, patch in valid_specs:
            reserve = np.full((4, 4), int(col), dtype=int)
            r0, c0 = random_free_location(g, reserve, transparent=bg, border=1, padding=2, padding_connectivity=4)
            paste(g, patch, int(r0), int(c0), transparent=bg)
            anchor_bboxes.append((int(r0), int(c0), int(r0 + 3), int(c0 + 3)))
            if kind == "frame":
                placed_frames.append((int(r0), int(c0), patch))
            else:
                placed_solids.append((int(r0), int(c0), patch))

        for i, dcol in enumerate(distractor_colors):
            shape = _DISTRACTOR_SHAPES[int(np.random.randint(0, len(_DISTRACTOR_SHAPES)))]
            patch = np.full(shape, int(dcol), dtype=int)
            ok = _place_distractor(g, patch, bg, anchor_bboxes, force_attach=(i == 0))
            if not ok:
                return None

        placed_frames.sort(key=lambda t: (int(t[0]), int(t[1])))
        placed_solids.sort(key=lambda t: (int(t[0]), int(t[1])))
        expected = _stack_output(placed_frames, placed_solids, bg)
        actual = main(g)
        if actual.shape != expected.shape or not bool((actual == expected).all()):
            return None

        features = _analyze_combo_features(g)
        if features["bg_non_black"] != 1:
            return None
        if features["valid_count"] != n_valid:
            return None
        if features["distractor_count"] < 1:
            return None
        if features["attached_distractor_count"] < 1:
            return None
        if features["legacy_diff"] != 1:
            return None

        return g
    except Exception:
        return None


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 500 attempts")
