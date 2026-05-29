# Combo variant of 2ba387bc (index 46)
# combo_spec: D2/0+D8/2+D11/0
# Name: 逐步移动+干扰物+列排序

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

try:
    from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, paste, random_free_location
except ModuleNotFoundError:
    SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
    if str(SOLVER_DIR) not in sys.path:
        sys.path.insert(0, str(SOLVER_DIR))
    from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, paste, random_free_location


"""
Design analysis for the fused combo variant
==========================================

1. Concrete behavior and trace
   - Base 2ba387bc:
     input contains isolated 4x4 objects on a uniform background:
       * frame = monochrome 1-cell border with bg 2x2 interior
       * solid = monochrome 4x4 fill
     output is a compact 8-column stack: frames on the left, solids on the right,
     each group sorted by input row.
   - Fused D2/0 + D8/2 + D11/0:
     a) only true 4x4 frame/solid objects are movable; non-4x4 colored blobs are distractors
        and stay behind.
     b) movable objects are ordered by input column, tie-broken by row.
     c) the output is no longer "just the packed pile". It is the final state after a
        stepwise relocation process:
          1. take the leftmost remaining valid 4x4 object of its type-order,
          2. erase it from the source scene to background,
          3. place it into the destination stack on the right.
        After all moves finish, the output shows:
          * left panel: original scene with all valid 4x4 objects removed
          * one bg separator column
          * right panel: the packed two-column destination stack
   - Small trace:
     suppose input has frame A at (r=2,c=14), frame B at (r=10,c=4), solid C at (r=6,c=11),
     and a 2x5 distractor attached to A.
     Original solver:
       * uses `monochromatic=False`, so A merges with its attached distractor and is lost
       * sorts surviving objects by row
       * returns only the compact 8-wide stack
     Combo solver:
       * uses `monochromatic=True`, so A is still recognized as a 4x4 frame and the 2x5 bar
         is a separate ignored distractor
       * sorts frames by column, so B is stacked above A
       * erases A/B/C from the source panel, but leaves the distractor in place
       * shows the cleared source panel plus the destination stack at right
     The visible differences are therefore:
       * attached non-4x4 blobs do not move
       * valid objects disappear from their original locations
       * right-stack order depends on left-right position, not top-bottom position

2. Complexity calibration
   - D11 would be too weak if distractors were always isolated far away, because a sloppy
     multicolor-component solver could still work. The generator therefore forces at least one
     distractor to touch a valid 4x4 object with a different color, which breaks the original
     `monochromatic=False` assumption.
   - D2 would be too weak if row-order and column-order matched. The generator rejects such
     samples and requires at least one group whose compact row-sorted stack differs from the
     new column-sorted stack.
   - D8 is made visible by preserving the whole source panel in the output. Humans can see that
     valid squares were moved out while distractors were not, instead of only seeing a compact
     summary pile.
   - Degrees of freedom stay controlled:
       * only two movable classes exist (frame/solid)
       * distractors come from a small rectangle vocabulary
       * no recoloring is introduced

3. Hidden-assumption audit of the original main()
   - `find_components(... monochromatic=False)` assumed every meaningful object was one isolated
     multicolor blob. D11 breaks that when a different-color distractor touches a valid square.
     Fix: detect monochromatic components and classify only exact 4x4 crops.
   - `bbox_h == bbox_w == 4` still holds for valid objects, but no longer for every foreground
     object. The solver must ignore non-4x4 components instead of treating them as errors.
   - Original frame test "uniform border and bg 2x2 center" is still valid, but the old
     fallback `else -> solid` is not. We must make the solid test explicit.
   - Original `sort(key=(row,col))` hard-coded D2's old order. Combo changes this to `(col,row)`.
   - Original output `out = np.full((4*max(...),8), bg)` assumed the answer only needed the
     compact destination pile. D8 breaks that: output must retain the source scene after all
     removals, plus the destination region.

4. Placement legality
   - Valid 4x4 objects are placed first with padding so they never overlap or 4-neighbor touch.
   - Distractors never overlap colored cells.
   - Attached distractors are allowed to touch exactly one chosen valid object and no others.
   - Unique colors are used across all valid objects and distractors so monochromatic component
     detection stays unambiguous.

5. Visual observability
   - In 2-4 examples, a human can see three stable clues:
       * some non-4x4 bars/blocks remain on the left and never appear in the stack
       * true 4x4 squares vanish from the left and reappear in the right-hand staging area
       * within a type, the upper destination item is the leftmost source item
   - No invisible feature such as numeric color value participates in ordering; only geometry and
     visible source position matter.

6. Visual cleanliness
   - The output uses a single separator column, not many overlays or path traces.
   - Distractors are simple solid rectangles, so they are easy to distinguish from the 4x4 movers.
   - Objects remain separated enough in the input that each source patch is legible.

7. Change inferability / anti-degeneracy metrics
   - `order_switch_active`: compact row-sorted stack != compact col-sorted stack
   - `legacy_loss`: number of valid objects detected by the combo solver minus the number detected
     by the original multicolor-component detector. Forced to be at least 1.
   - `attached_distractors`: count of distractor components touching a valid object. Forced >= 1.
   - `moved_pixels`: number of colored valid-object pixels erased from the source panel. Forced > 0.

8. New solver assumptions introduced here, and generator support
   - Every movable object is an exact monochromatic 4x4 frame border or 4x4 solid block.
     Generator guarantees this.
   - Every distractor is not a valid 4x4 frame/solid under the same classifier.
     Generator guarantees non-4x4 rectangles only.
   - Destination stack uses the same source background color and is placed in a fresh right-side
     panel separated by exactly one background column. Generator and solver share this layout.
   - Attached distractors use a different color from the touched valid object so the monochromatic
     detector can still separate them. Generator guarantees global color uniqueness.
"""


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


Obj = tuple[int, int, np.ndarray, tuple[tuple[int, int], ...]]


def _key_by_row(item: Obj) -> tuple[int, int]:
    return int(item[0]), int(item[1])


def _key_by_col(item: Obj) -> tuple[int, int]:
    return int(item[1]), int(item[0])


def _make_frame(col: int, bg: int = 0) -> np.ndarray:
    patch = np.full((4, 4), int(col), dtype=int)
    patch[1:3, 1:3] = int(bg)
    return patch


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


def _collect_valid_objects(grid: np.ndarray, bg: int) -> tuple[list[Obj], list[Obj]]:
    comps = find_components(grid, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    frames: list[Obj] = []
    solids: list[Obj] = []

    for comp in comps:
        bbox = comp["bbox"]
        if bbox_h(bbox) != 4 or bbox_w(bbox) != 4:
            continue

        r0, c0, _r1, _c1 = bbox
        sub = crop_bbox(grid, bbox)
        kind = _classify_valid_square(sub, int(comp["color"]), int(comp["area"]), bg)
        coords = tuple((int(r), int(c)) for r, c in comp["coords"])
        item: Obj = (int(r0), int(c0), sub, coords)
        if kind == "frame":
            frames.append(item)
        elif kind == "solid":
            solids.append(item)

    return frames, solids


def _collect_legacy_objects(grid: np.ndarray, bg: int) -> tuple[list[Obj], list[Obj]]:
    comps = find_components(grid, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    frames: list[Obj] = []
    solids: list[Obj] = []

    for comp in comps:
        bbox = comp["bbox"]
        if bbox_h(bbox) != 4 or bbox_w(bbox) != 4:
            continue

        r0, c0, _r1, _c1 = bbox
        sub = crop_bbox(grid, bbox)
        border = np.concatenate([sub[0, :], sub[-1, :], sub[1:-1, 0], sub[1:-1, -1]])
        border_cols = set(map(int, border.flatten()))
        coords = tuple((int(r), int(c)) for r, c in comp["coords"])
        item: Obj = (int(r0), int(c0), sub, coords)

        if len(border_cols) == 1 and bool((sub[1:3, 1:3] == int(bg)).all()):
            frames.append(item)
        elif len(set(map(int, sub.flatten()))) == 1 and int(sub[0, 0]) != int(bg):
            solids.append(item)

    return frames, solids


def _stack_output(frames: list[Obj], solids: list[Obj], bg: int, *, sort_key) -> np.ndarray:
    frames_sorted = sorted(frames, key=sort_key)
    solids_sorted = sorted(solids, key=sort_key)
    out_h = 4 * max(int(len(frames_sorted)), int(len(solids_sorted)))
    out = np.full((out_h, 8), int(bg), dtype=int)

    for i, (_r0, _c0, sub, _coords) in enumerate(frames_sorted):
        paste(out, sub, int(i * 4), 0)
    for i, (_r0, _c0, sub, _coords) in enumerate(solids_sorted):
        paste(out, sub, int(i * 4), 4)

    return out


def _erase_valid_objects(scene: np.ndarray, frames: list[Obj], solids: list[Obj], bg: int) -> np.ndarray:
    out = np.array(scene, copy=True)
    for _r0, _c0, _sub, coords in frames + solids:
        for r, c in coords:
            out[int(r), int(c)] = int(bg)
    return out


def _compose_output(scene_after_move: np.ndarray, frames: list[Obj], solids: list[Obj], bg: int) -> np.ndarray:
    stack = _stack_output(frames, solids, bg, sort_key=_key_by_col)
    h_scene, w_scene = map(int, scene_after_move.shape)
    h_stack = int(stack.shape[0])
    out_h = max(int(h_scene), int(h_stack))
    out_w = int(w_scene + 1 + 8)
    out = np.full((out_h, out_w), int(bg), dtype=int)

    out[:h_scene, :w_scene] = scene_after_move
    if h_stack > 0:
        out[:h_stack, w_scene + 1 : w_scene + 9] = stack
    return out


def _bbox_cells(bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    return {(int(r), int(c)) for r in range(int(r0), int(r1) + 1) for c in range(int(c0), int(c1) + 1)}


def _can_place_attached(
    grid: np.ndarray,
    patch: np.ndarray,
    r0: int,
    c0: int,
    target_bbox: tuple[int, int, int, int],
    bg: int,
) -> bool:
    h, w = map(int, patch.shape)
    H, W = grid.shape
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False
    if bool((grid[r0 : r0 + h, c0 : c0 + w] != int(bg)).any()):
        return False

    target_cells = _bbox_cells(target_bbox)
    touched_target = False
    touched_other = False
    for rr in range(int(r0), int(r0 + h)):
        for cc in range(int(c0), int(c0 + w)):
            for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                if not (0 <= nr < H and 0 <= nc < W):
                    continue
                if int(grid[nr, nc]) == int(bg):
                    continue
                if (int(nr), int(nc)) in target_cells:
                    touched_target = True
                else:
                    touched_other = True
    return touched_target and not touched_other


def _attached_location(
    grid: np.ndarray,
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
    if c1 + 1 + w <= grid.shape[1]:
        for rr in range(int(r0 - h + 1), int(r1) + 1):
            candidates.append((int(rr), int(c1 + 1)))
    if r0 - h >= 0:
        for cc in range(int(c0 - w + 1), int(c1) + 1):
            candidates.append((int(r0 - h), int(cc)))
    if r1 + 1 + h <= grid.shape[0]:
        for cc in range(int(c0 - w + 1), int(c1) + 1):
            candidates.append((int(r1 + 1), int(cc)))

    np.random.shuffle(candidates)
    for rr, cc in candidates:
        if _can_place_attached(grid, patch, int(rr), int(cc), target_bbox, bg):
            return int(rr), int(cc)
    return None


def _place_distractor(
    grid: np.ndarray,
    patch: np.ndarray,
    bg: int,
    anchors: list[tuple[int, int, int, int]],
    *,
    force_attach: bool,
) -> bool:
    anchor_order = list(range(len(anchors)))
    np.random.shuffle(anchor_order)

    if force_attach or bool(np.random.rand() < 0.6):
        for idx in anchor_order:
            loc = _attached_location(grid, patch, anchors[idx], bg)
            if loc is not None:
                paste(grid, patch, int(loc[0]), int(loc[1]), transparent=bg)
                return True
        if force_attach:
            return False

    try:
        r0, c0 = random_free_location(
            grid,
            patch,
            transparent=bg,
            border=1,
            padding=1,
            padding_connectivity=4,
        )
    except Exception:
        return False

    paste(grid, patch, int(r0), int(c0), transparent=bg)
    return True


def _build_expected_from_records(
    grid: np.ndarray,
    frames: list[Obj],
    solids: list[Obj],
    bg: int,
) -> np.ndarray:
    scene_after_move = _erase_valid_objects(grid, frames, solids, bg)
    return _compose_output(scene_after_move, frames, solids, bg)


def _behavior_metrics(grid: np.ndarray) -> dict[str, int]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    frames, solids = _collect_valid_objects(g, bg)
    legacy_frames, legacy_solids = _collect_legacy_objects(g, bg)

    col_stack = _stack_output(frames, solids, bg, sort_key=_key_by_col)
    row_stack = _stack_output(frames, solids, bg, sort_key=_key_by_row)

    valid_cells: set[tuple[int, int]] = set()
    for _r0, _c0, _sub, coords in frames + solids:
        valid_cells.update(coords)

    attached_distractors = 0
    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict"):
        bbox = comp["bbox"]
        sub = crop_bbox(g, bbox)
        kind = None
        if bbox_h(bbox) == 4 and bbox_w(bbox) == 4:
            kind = _classify_valid_square(sub, int(comp["color"]), int(comp["area"]), bg)
        if kind is not None:
            continue

        coords = [(int(r), int(c)) for r, c in comp["coords"]]
        touches_valid = False
        for r, c in coords:
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if (int(nr), int(nc)) in valid_cells:
                    touches_valid = True
                    break
            if touches_valid:
                break
        if touches_valid:
            attached_distractors += 1

    moved_pixels = int(sum(len(coords) for _r0, _c0, _sub, coords in frames + solids))
    return {
        "n_frames": int(len(frames)),
        "n_solids": int(len(solids)),
        "valid_total": int(len(frames) + len(solids)),
        "legacy_total": int(len(legacy_frames) + len(legacy_solids)),
        "legacy_loss": int(len(frames) + len(solids) - len(legacy_frames) - len(legacy_solids)),
        "attached_distractors": int(attached_distractors),
        "order_switch_active": int(not np.array_equal(col_stack, row_stack)),
        "moved_pixels": int(moved_pixels),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move only the valid 4x4 frame/solid objects into a right-side staging area.

    Combo rule:
    - detect exact 4x4 frame/solid objects monochromatically, ignoring non-4x4 distractors
    - sort each type by input column, tie by row
    - erase every valid object from the source panel
    - stack frames on the right-left column and solids on the right-right column
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    frames, solids = _collect_valid_objects(g, bg)
    scene_after_move = _erase_valid_objects(g, frames, solids, bg)
    return _compose_output(scene_after_move, frames, solids, bg)


def _gen_attempt() -> np.ndarray | None:
    bg = 0
    H = int(np.random.randint(24, 35))
    W = int(np.random.randint(24, 35))
    g = np.full((H, W), int(bg), dtype=int)

    n_frames = int(np.random.randint(1, 4))
    n_solids = int(np.random.randint(1, 4))
    if max(int(n_frames), int(n_solids)) < 2:
        if bool(np.random.rand() < 0.5):
            n_frames = 2
        else:
            n_solids = 2

    n_valid = int(n_frames + n_solids)
    n_distractors = int(np.random.randint(1, min(4, 10 - n_valid)))

    colors = [int(c) for c in np.random.permutation(np.arange(1, 10))]
    valid_colors = colors[:n_valid]
    distractor_colors = colors[n_valid : n_valid + n_distractors]

    frames: list[Obj] = []
    solids: list[Obj] = []
    anchors: list[tuple[int, int, int, int]] = []

    try:
        valid_specs: list[tuple[str, int, np.ndarray]] = []
        for col in valid_colors[:n_frames]:
            valid_specs.append(("frame", int(col), _make_frame(int(col), bg)))
        for col in valid_colors[n_frames:]:
            valid_specs.append(("solid", int(col), _make_solid(int(col))))
        np.random.shuffle(valid_specs)

        for kind, col, patch in valid_specs:
            reserve = np.full((4, 4), int(col), dtype=int)
            r0, c0 = random_free_location(
                g,
                reserve,
                transparent=bg,
                border=1,
                padding=2,
                padding_connectivity=4,
            )
            paste(g, patch, int(r0), int(c0), transparent=bg)
            coords = tuple((int(r), int(c)) for r in range(int(r0), int(r0 + 4)) for c in range(int(c0), int(c0 + 4)) if int(patch[r - int(r0), c - int(c0)]) != int(bg))
            item: Obj = (int(r0), int(c0), patch.copy(), coords)
            anchors.append((int(r0), int(c0), int(r0 + 3), int(c0 + 3)))
            if kind == "frame":
                frames.append(item)
            else:
                solids.append(item)

        attached_count = 0
        for i, dcol in enumerate(distractor_colors):
            shape = _DISTRACTOR_SHAPES[int(np.random.randint(0, len(_DISTRACTOR_SHAPES)))]
            patch = np.full(shape, int(dcol), dtype=int)
            force_attach = i == 0
            ok = _place_distractor(g, patch, bg, anchors, force_attach=force_attach)
            if not ok:
                return None
            if force_attach:
                attached_count += 1

        if attached_count == 0:
            return None

        metrics = _behavior_metrics(g)
        if metrics["n_frames"] != n_frames or metrics["n_solids"] != n_solids:
            return None
        if metrics["attached_distractors"] < 1:
            return None
        if metrics["legacy_loss"] < 1:
            return None
        if metrics["order_switch_active"] < 1:
            return None
        if metrics["moved_pixels"] <= 0:
            return None

        expected = _build_expected_from_records(g, frames, solids, bg)
        actual = main(g)
        if actual.shape != expected.shape or not np.array_equal(actual, expected):
            return None

        legacy_compact = _stack_output(frames, solids, bg, sort_key=_key_by_row)
        if actual.shape == legacy_compact.shape and np.array_equal(actual, legacy_compact):
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

