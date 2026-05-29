# Combo variant of 2ba387bc (index 43)
# combo_spec: D1/2+D11/0+D12/0
# Name: 非标干扰+非黑背景+面积分类

from __future__ import annotations

import numpy as np
from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, paste, random_free_location

# Design analysis for the combo variant
# a) Concrete behavior and trace:
#    - Original input: a black background with isolated monochrome 4x4 objects. A
#      solver can treat each whole blob as one component and classify by asking
#      whether the 2x2 center is background.
#    - Combo input: the background is non-black (for example 5 or 8), there are
#      still valid monochrome 4x4 frames/solids, and there are extra non-4x4
#      monochrome rectangles. At least one distractor is attached edge-to-edge to a
#      valid 4x4 object with a different color.
#    - Example trace on a gray background 5:
#        * a black solid 4x4 at rows 3:6 touches a red 1x5 bar on its right edge
#        * a blue frame 4x4 is isolated elsewhere
#      Old `monochromatic=False` logic merges the black square and the red bar into a
#      single non-4x4 blob, so that valid square disappears. The original frame test
#      also relies on reading the 2x2 center directly.
#      New logic must:
#        1. infer bg=5 rather than assume 0,
#        2. find monochromatic components so the black 4x4 and red bar separate,
#        3. keep only valid 4x4 components and classify them by non-bg area:
#           area 12 -> frame, area 16 -> solid.
# b) Complexity calibration:
#    - Distractors use a small rectangle vocabulary only, so humans can visually
#      recognize them as "wrong-sized bars/blocks" rather than new semantic objects.
#    - Valid objects remain 4x4 with repeated support for both area classes (12 and
#      16), so the area split is easy to infer from a few examples.
#    - A black foreground object is forced while bg is non-black, making the
#      "background is not 0" change visible rather than cosmetic.
# c) Implicit assumption audit of the original solver:
#    - `bg = get_bg(g)` still holds, but `bg == 0` no longer holds.
#    - `find_components(... monochromatic=False)` no longer holds because attached
#      different-color distractors would swallow a valid square into a larger blob.
#    - `bbox == 4x4` is still required for valid objects, but now it must be applied
#      after monochromatic splitting, not before.
#    - `if frame else solid` is no longer driven by interior inspection; the branch is
#      driven by component area 12 vs 16. We still validate exact 4x4 patterns to
#      reject accidental distractors, but the grouping key itself is area.
# d) Placement legality:
#    - Valid 4x4 objects are placed first with an opaque 4x4 reservation footprint so
#      later objects cannot occupy a frame interior.
#    - Distractors never overlap colored cells. Attached distractors may touch exactly
#      one reserved valid bbox and are rejected if they touch any other object.
# e) Visual observability:
#    - Across 2-4 examples, the user can see three stable cues: non-black canvas, only
#      4x4 objects survive into output, and left/right grouping matches 12-pixel
#      hollow vs 16-pixel filled squares.
# f) Visual cleanliness:
#    - Unique colors prevent same-color mergers. Padding keeps valid objects separated.
#      Distractors are plain rectangles instead of noisy jagged shapes.
# g) Color/change inferability:
#    - No hidden color remapping exists. Colors are preserved from input to output.
#      Color only marks instances; grouping comes from visible geometry and area.
# h) Generator anti-degeneracy checks:
#    - Every sample enforces non-black bg, at least one attached non-4x4 distractor,
#      and `main(input) != _legacy_main(input)`, so the combo does not collapse back
#      to the source task on generated data.
# i) New solver assumptions introduced here:
#    - Valid objects are monochrome 4x4 squares with area exactly 12 or 16.
#    - Distractors use colors distinct from valid objects so monochromatic splitting
#      separates them cleanly.
#    - `generate_input()` explicitly guarantees those assumptions.


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

_BG_CHOICES: tuple[int, ...] = (1, 5, 8)


def _make_frame(col: int, bg: int) -> np.ndarray:
    patch = np.full((4, 4), int(col), dtype=int)
    patch[1:3, 1:3] = int(bg)
    return patch


def _make_solid(col: int) -> np.ndarray:
    return np.full((4, 4), int(col), dtype=int)


def _classify_valid_square(sub: np.ndarray, color: int, area: int, bg: int) -> str | None:
    """Return the area-based class for a legal 4x4 square, else None."""
    if sub.shape != (4, 4):
        return None

    color = int(color)
    area = int(area)
    bg = int(bg)

    if area == 16 and bool((sub == color).all()):
        return "solid"

    if area == 12:
        if not bool((sub[1:3, 1:3] == bg).all()):
            return None
        if not bool((sub[_FRAME_MASK] == color).all()):
            return None
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


def _extract_valid_objects(grid: np.ndarray) -> tuple[int, list[tuple[int, int, np.ndarray]], list[tuple[int, int, np.ndarray]]]:
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
    return bg, frames, solids


def main(grid: np.ndarray) -> np.ndarray:
    """
    Solve the combo by a three-step preprocessing chain:
    1. infer the non-black background,
    2. isolate monochromatic components and ignore non-4x4 distractors,
    3. classify valid 4x4 squares by non-background area (12 frame, 16 solid).
    """
    bg, frames, solids = _extract_valid_objects(grid)
    return _stack_output(frames, solids, bg)


def _legacy_main(grid: np.ndarray) -> np.ndarray:
    """Source-task logic kept only for anti-degeneracy checks in the generator."""
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
    for rr in range(r0, r0 + h):
        for cc in range(c0, c0 + w):
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
            candidates.append((int(rr), int(c0 - w)))
    if c1 + 1 + w <= g.shape[1]:
        for rr in range(int(r0 - h + 1), int(r1) + 1):
            candidates.append((int(rr), int(c1 + 1)))
    if r0 - h >= 0:
        for cc in range(int(c0 - w + 1), int(c1) + 1):
            candidates.append((int(r0 - h), int(cc)))
    if r1 + 1 + h <= g.shape[0]:
        for cc in range(int(c0 - w + 1), int(c1) + 1):
            candidates.append((int(r1 + 1), int(cc)))

    np.random.shuffle(candidates)
    for rr, cc in candidates:
        if _can_place_attached(g, patch, rr, cc, target_bbox, bg):
            return rr, cc
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

    if force_attach or bool(np.random.rand() < 0.7):
        for idx in order:
            loc = _attached_location(g, patch, anchors[idx], bg)
            if loc is not None:
                paste(g, patch, loc[0], loc[1], transparent=bg)
                return True
        if force_attach:
            return False

    try:
        r0, c0 = random_free_location(g, patch, transparent=bg, border=1, padding=1, padding_connectivity=4)
        paste(g, patch, int(r0), int(c0), transparent=bg)
        return True
    except Exception:
        return False


def fusion_metrics(grid: np.ndarray) -> dict[str, int]:
    """
    Quantify whether the generated sample really exercises the combo:
    - bg_nonblack: background is not 0
    - black_foreground: some 0-valued object/distractor exists on that non-black bg
    - distractor_count: count of non-valid monochromatic components
    - attached_distractor_count: how many distractors touch a valid 4x4 object
    - legacy_differs: source-task solver gives a different output
    """
    g = np.array(grid, dtype=int)
    bg, frames, solids = _extract_valid_objects(g)
    valid_bboxes = [(r0, c0, r0 + 3, c0 + 3) for r0, c0, _sub in frames + solids]
    valid_cells: set[tuple[int, int]] = set()
    for bbox in valid_bboxes:
        valid_cells |= _bbox_cells(bbox)

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    distractor_count = 0
    attached_distractor_count = 0
    for comp in comps:
        b = comp["bbox"]
        sub = crop_bbox(g, b)
        if bbox_h(b) == 4 and bbox_w(b) == 4 and _classify_valid_square(sub, int(comp["color"]), int(comp["area"]), bg) is not None:
            continue
        distractor_count += 1

        touches_valid = False
        for rr, cc in comp["coords"]:
            r = int(rr)
            c = int(cc)
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if (int(nr), int(nc)) in valid_cells:
                    touches_valid = True
                    break
            if touches_valid:
                break
        if touches_valid:
            attached_distractor_count += 1

    legacy_out = _legacy_main(g)
    combo_out = main(g)
    return {
        "bg": int(bg),
        "bg_nonblack": int(bg != 0),
        "black_foreground": int(bool((g == 0).any()) and bg != 0),
        "valid_frames": int(len(frames)),
        "valid_solids": int(len(solids)),
        "distractor_count": int(distractor_count),
        "attached_distractor_count": int(attached_distractor_count),
        "legacy_differs": int(not np.array_equal(legacy_out, combo_out)),
    }


def _gen_attempt() -> np.ndarray | None:
    bg = int(_BG_CHOICES[int(np.random.randint(0, len(_BG_CHOICES)))])
    H = int(np.random.randint(24, 36))
    W = int(np.random.randint(24, 36))
    g = np.full((H, W), bg, dtype=int)

    n_frames = int(np.random.randint(1, 4))
    n_solids = int(np.random.randint(1, 4))
    n_valid = n_frames + n_solids
    max_distractors = min(4, 9 - n_valid)
    if max_distractors < 2:
        return None
    n_distractors = int(np.random.randint(2, max_distractors + 1))

    colors = list(np.random.permutation(np.array([c for c in range(10) if c != bg], dtype=int)))
    if 0 not in colors[:n_valid]:
        zero_idx = colors.index(0)
        colors[0], colors[zero_idx] = colors[zero_idx], colors[0]

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
            if not _place_distractor(g, patch, bg, anchor_bboxes, force_attach=(i == 0)):
                return None

        placed_frames.sort(key=lambda t: (int(t[0]), int(t[1])))
        placed_solids.sort(key=lambda t: (int(t[0]), int(t[1])))
        expected = _stack_output(placed_frames, placed_solids, bg)
        actual = main(g)
        if actual.shape != expected.shape or not bool((actual == expected).all()):
            return None

        metrics = fusion_metrics(g)
        if metrics["bg_nonblack"] != 1:
            return None
        if metrics["black_foreground"] != 1:
            return None
        if metrics["distractor_count"] < n_distractors:
            return None
        if metrics["attached_distractor_count"] < 1:
            return None
        if metrics["legacy_differs"] != 1:
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
