# Variant of 2ba387bc
# Dimension: D11 — 干扰项——输入中的额外对象 (src)
# Change: 输入中除4x4方块和背景外无其他对象... → 增加非4x4的干扰色块

from __future__ import annotations

import numpy as np
from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, paste, random_free_location

# Design analysis for the D11 variant
# a) Concrete behavior and trace:
#    - The original task shows only isolated 4x4 objects: either solid 4x4 blocks or
#      4x4 frames with a bg 2x2 interior. Output keeps only those objects and stacks
#      them by type.
#    - This variant adds a few extra non-4x4 rectangles such as 2x3, 1x5, 5x2.
#      At least one distractor is attached orthogonally to a valid 4x4 object using a
#      different color. Humanly, the 4x4 square is still visible, but the full
#      multi-color blob is no longer a 4x4 component.
#    - Old logic: `find_components(... monochromatic=False)` merges touching colors, so
#      an attached distractor changes the valid object's bbox away from 4x4 and the
#      object is lost.
#    - New logic: find monochromatic components, then keep only components whose 4x4
#      crop is exactly a valid frame or valid solid. The attached non-4x4 distractor
#      becomes a separate component and is ignored.
# b) Complexity calibration:
#    - Distractors stay simple filled rectangles with a small shape vocabulary, so the
#      new rule is visually obvious.
#    - At least one attached distractor is forced so the variant is not a no-op with
#      respect to the original solver.
# c) Implicit assumption audit for the original `main()`:
#    - `monochromatic=False` assumed every meaningful object was an isolated whole blob.
#      D11 breaks that because a different-color distractor may touch a valid square.
#    - `else: solids.append(...)` assumed every 4x4 non-frame component was a valid
#      solid. D11 keeps distractors non-4x4, but the revised solver still makes the
#      solid test explicit to avoid accidental false positives.
# d) Placement legality:
#    - Valid 4x4 objects are placed first with a reserved full 4x4 footprint so nothing
#      can land inside a frame interior.
#    - Distractors never overlap existing colored cells. Attached distractors are
#      allowed to touch only one chosen valid object and are rejected if they touch any
#      other object.
# e) Visual observability:
#    - A solver can infer "ignore extra non-4x4 colored bars/blocks; keep only 4x4
#      square types" from a few examples because output never includes those extras.
# f) Visual cleanliness:
#    - Color uniqueness avoids same-color mergers under the new monochromatic analysis
#      and makes each distractor easy to see.
# g) Color inference:
#    - No color remapping exists here; colors are preserved. Color only identifies
#      object instances, not hidden categories.
# Additional note on the source task:
#    - The referenced task `b99e7126` is present in this workspace only as evaluation
#      JSON; its solver source is absent, so solver-reading was skipped as required.


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


def _make_frame(col: int, bg: int = 0) -> np.ndarray:
    a = np.full((4, 4), col, dtype=int)
    a[1:3, 1:3] = bg
    return a


def _make_solid(col: int) -> np.ndarray:
    return np.full((4, 4), col, dtype=int)


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
    Sort only the valid 4x4 squares into two columns by type.

    D11 change: the input may also contain non-4x4 distractor blobs, including blobs
    that touch a valid 4x4 object with another color. We therefore detect components
    monochromatically and keep only true 4x4 frame/solid patterns.
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


def _overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(int(a0), int(b0)) <= min(int(a1), int(b1))


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
    anchor_order = list(range(len(anchors)))
    np.random.shuffle(anchor_order)

    if force_attach:
        for idx in anchor_order:
            loc = _attached_location(g, patch, anchors[idx], bg)
            if loc is not None:
                paste(g, patch, loc[0], loc[1], transparent=bg)
                return True
        return False

    if bool(np.random.rand() < 0.6):
        for idx in anchor_order:
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


def _gen_attempt() -> np.ndarray | None:
    bg = 0
    H = int(np.random.randint(24, 34))
    W = int(np.random.randint(24, 34))
    g = np.full((H, W), bg, dtype=int)

    n_frames = int(np.random.randint(1, 4))
    n_solids = int(np.random.randint(1, 4))
    n_valid = n_frames + n_solids
    n_distractors = int(np.random.randint(1, min(4, 10 - n_valid)))

    colors = list(np.random.permutation(np.arange(1, 10)))
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

        attached_count = 0
        for i, dcol in enumerate(distractor_colors):
            shape = _DISTRACTOR_SHAPES[int(np.random.randint(0, len(_DISTRACTOR_SHAPES)))]
            patch = np.full(shape, int(dcol), dtype=int)
            force_attach = i == 0
            ok = _place_distractor(g, patch, bg, anchor_bboxes, force_attach=force_attach)
            if not ok:
                return None
            if force_attach:
                attached_count += 1

        if attached_count == 0:
            return None

        placed_frames.sort(key=lambda t: (int(t[0]), int(t[1])))
        placed_solids.sort(key=lambda t: (int(t[0]), int(t[1])))
        expected = _stack_output(placed_frames, placed_solids, bg)
        actual = main(g)
        if actual.shape != expected.shape or not bool((actual == expected).all()):
            return None
        return g
    except Exception:
        return None


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 400 attempts")
