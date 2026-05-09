# Combo variant of 2ba387bc (index 22)
# combo_spec: D7/2+D14/0
# Name: 跨列颜色互换+反向堆叠

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, paste, random_free_location


"""
Design analysis for combo D7/2 + D14/0
======================================

1. Concrete behavior and trace
   - Input objects stay exactly as in the source task:
       * frame = 4x4 border of one non-bg color with a bg-colored 2x2 hole
       * solid = 4x4 fully filled monochrome square
   - Classification also stays the same: detect all disconnected 4x4 components, then split them
     into frames vs solids from the 2x2 interior.
   - Sorting also stays the same: each group is ordered by input (row, col), top to bottom.
   - The fusion starts only at output construction:
       * left column (frames) is stacked top-down as before;
       * right column (solids) is stacked bottom-up;
       * colors are swapped between objects that occupy the same final output slot.
   - Small example with three frames F1,F2,F3 and three solids S1,S2,S3 after sorting:
       original source task:
         left top->bottom  = F1, F2, F3
         right top->bottom = S1, S2, S3
         colors unchanged
       this combo:
         left top->bottom  = F1, F2, F3
         right top->bottom = S3, S2, S1   (because solids stack from bottom upward)
         slotwise color swap:
           slot 0: F1 takes color(S3), S3 takes color(F1)
           slot 1: F2 takes color(S2), S2 takes color(F2)
           slot 2: F3 takes color(S1), S1 takes color(F3)
     The critical difference from a naive "same-index swap" implementation is that the partner of
     F1 is S3, not S1, because pairing is defined by final physical row slot after stacking.

2. Complexity calibration
   - If the generator allowed only one pair, reversing the right stack would be invisible.
   - If repeated colors were common, the reverse-pair swap could collapse into an output that looks
     identical to a simpler non-fused variant.
   - To keep the rule visible but still simple, the generator uses:
       * equal counts for frames and solids;
       * at least 2 pairs;
       * unique colors across all objects.
     This gives repeated structural evidence for both parts of the combo without introducing extra
     markers, hidden numeric ordering, or noisy object classes.

3. Implicit-assumption audit of the original `main()`
   - `bg = get_bg(g)` still holds: the background remains the dominant color.
   - `find_components(... monochromatic=False)` still assumes each object is an isolated connected
     component. Generator enforces 1-cell padding, so this remains valid.
   - `bbox_h == 4 and bbox_w == 4` still holds because the object geometry is unchanged.
   - Frame detection by "uniform border and bg-filled 2x2 center" still holds because the combo
     changes only output placement and output colors, not input recognition.
   - Original `frames.sort(...)` and `solids.sort(...)` still hold because the group-order rule is
     unchanged.
   - The broken assumptions are only in output assembly:
       * `paste(out, sub, i * 4, 4)` assumed the right column also stacks top-down.
       * `paste(out, sub, ...)` assumed an object's output color equals its input color.
     In the combo solver, output must therefore be built from slot assignments and recolored patches,
     not by directly pasting the extracted input subgrids.

4. Placement legality
   - Generator constructs only 4x4 frame/solid objects.
   - `random_free_location(... border=1, padding=1, padding_connectivity=4)` keeps objects inside
     bounds, non-overlapping, and 4-disconnected, so component extraction is clean.
   - The generator re-runs the solver and rejects any candidate whose reconstructed object lists do
     not match the intended pair count.

5. Visual observability
   - A human can see the left column remains top-down while the right column is reversed.
   - Because colors are unique, the top frame in output taking the bottom solid's color is visible.
   - Across 2-4 examples, the solver can infer a structural rule:
       "group by shape, but solids stack upward and swap colors with the frame in the same row slot."
   - No hidden dependency on the numeric value of the colors is used.

6. Visual cleanliness
   - Input remains a sparse field of separated 4x4 objects on plain background.
   - Output remains a tidy 8-column stack with no overlap, no extra decorations, and no ambiguous
     overlays. The only novelty is the asymmetry between the two columns.

7. Color-change inferability
   - The color mapping is not arbitrary global recoloring. It is a local structural swap between the
     two columns at corresponding output slots.
   - Equal pair counts and unique colors make the correspondence legible from few examples.

8. Quantitative anti-degeneracy check
   - Besides checking `main(generate_input())` runs, the generator compares the combo output against:
       * the original source-task output;
       * a naive "swap colors by sorted index, then reverse the right stack" interpretation.
   - It also measures `reverse_pair_distance`: the Hamming distance between the combo's left-column
     color sequence and the naive same-index-swap left-column sequence. This must be > 0.

9. New assumptions introduced by the combo solver
   - When counts differ, a slot may be occupied on only one side. The solver assumes that only slots
     with two real objects participate in the swap; unmatched objects keep their own color. This is
     the least arbitrary extension of "swap between corresponding positions".
   - The generator avoids these asymmetric cases by using equal counts, so the intended fusion signal
     is always fully visible in produced samples.
"""


ObjectInfo = tuple[int, int, int]  # (row, col, color)


def _make_frame(col: int, bg: int = 0) -> np.ndarray:
    patch = np.full((4, 4), int(col), dtype=int)
    patch[1:3, 1:3] = int(bg)
    return patch


def _make_solid(col: int) -> np.ndarray:
    return np.full((4, 4), int(col), dtype=int)


def _is_frame(sub: np.ndarray, bg: int) -> bool:
    border = np.concatenate([sub[0, :], sub[-1, :], sub[1:-1, 0], sub[1:-1, -1]])
    border_colors = set(map(int, border.flatten()))
    return len(border_colors) == 1 and bool((sub[1:3, 1:3] == int(bg)).all())


def _object_color(sub: np.ndarray, bg: int) -> int:
    values = sub[sub != int(bg)]
    if values.size == 0:
        raise ValueError("object has no non-background pixels")
    return int(values.flat[0])


def _collect_objects(g: np.ndarray, bg: int) -> tuple[list[ObjectInfo], list[ObjectInfo]]:
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    frames: list[ObjectInfo] = []
    solids: list[ObjectInfo] = []

    for comp in comps:
        b = comp["bbox"]
        if bbox_h(b) != 4 or bbox_w(b) != 4:
            continue
        r0, c0, _r1, _c1 = b
        sub = crop_bbox(g, b)
        info = (int(r0), int(c0), _object_color(sub, bg))
        if _is_frame(sub, bg):
            frames.append(info)
        else:
            solids.append(info)

    frames.sort(key=lambda t: (int(t[0]), int(t[1])))
    solids.sort(key=lambda t: (int(t[0]), int(t[1])))
    return frames, solids


def _slotify_frames(frames: list[ObjectInfo], n_slots: int) -> list[ObjectInfo | None]:
    slots: list[ObjectInfo | None] = [None] * int(n_slots)
    for i, info in enumerate(frames):
        slots[int(i)] = info
    return slots


def _slotify_solids_bottom_up(solids: list[ObjectInfo], n_slots: int) -> list[ObjectInfo | None]:
    slots: list[ObjectInfo | None] = [None] * int(n_slots)
    for i, info in enumerate(solids):
        slots[int(n_slots - 1 - i)] = info
    return slots


def _build_original_output(frames: list[ObjectInfo], solids: list[ObjectInfo], bg: int) -> np.ndarray:
    n_slots = max(int(len(frames)), int(len(solids)))
    out = np.full((4 * n_slots, 8), int(bg), dtype=int)

    for i, (_r0, _c0, col) in enumerate(frames):
        paste(out, _make_frame(col, bg), int(i * 4), 0)
    for i, (_r0, _c0, col) in enumerate(solids):
        paste(out, _make_solid(col), int(i * 4), 4)

    return out


def _build_naive_same_index_swap_output(frames: list[ObjectInfo], solids: list[ObjectInfo], bg: int) -> np.ndarray:
    """
    Reference for a wrong interpretation of the combo:
    swap frame[i] with solid[i] by sorted index first, then stack the right column bottom-up.
    """
    n_slots = max(int(len(frames)), int(len(solids)))
    out = np.full((4 * n_slots, 8), int(bg), dtype=int)

    for i, (_r0, _c0, frame_col) in enumerate(frames):
        color = frame_col
        if i < len(solids):
            color = int(solids[i][2])
        paste(out, _make_frame(color, bg), int(i * 4), 0)

    for i, (_r0, _c0, solid_col) in enumerate(solids):
        color = solid_col
        if i < len(frames):
            color = int(frames[i][2])
        paste(out, _make_solid(color), int((n_slots - 1 - i) * 4), 4)

    return out


def _build_combo_output(frames: list[ObjectInfo], solids: list[ObjectInfo], bg: int) -> np.ndarray:
    n_slots = max(int(len(frames)), int(len(solids)))
    out = np.full((4 * n_slots, 8), int(bg), dtype=int)

    left_slots = _slotify_frames(frames, n_slots)
    right_slots = _slotify_solids_bottom_up(solids, n_slots)

    for slot in range(n_slots):
        left = left_slots[slot]
        right = right_slots[slot]
        row0 = int(slot * 4)

        if left is not None:
            left_color = int(right[2]) if right is not None else int(left[2])
            paste(out, _make_frame(left_color, bg), row0, 0)

        if right is not None:
            right_color = int(left[2]) if left is not None else int(right[2])
            paste(out, _make_solid(right_color), row0, 4)

    return out


def _reverse_pair_distance(frames: list[ObjectInfo], solids: list[ObjectInfo]) -> int:
    """
    Measure how different the combo pairing is from naive same-index pairing.
    We compare the left-column output color sequence under the two interpretations.
    """
    n_slots = max(int(len(frames)), int(len(solids)))
    combo_left: list[int | None] = []
    naive_left: list[int | None] = []

    left_slots = _slotify_frames(frames, n_slots)
    right_slots = _slotify_solids_bottom_up(solids, n_slots)

    for slot in range(n_slots):
        left = left_slots[slot]
        right = right_slots[slot]
        if left is None:
            combo_left.append(None)
        else:
            combo_left.append(int(right[2]) if right is not None else int(left[2]))

    for i in range(n_slots):
        if i >= len(frames):
            naive_left.append(None)
        elif i >= len(solids):
            naive_left.append(int(frames[i][2]))
        else:
            naive_left.append(int(solids[i][2]))

    return sum(int(a != b) for a, b in zip(combo_left, naive_left))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combo rule:
    - detect 4x4 frames and solids, sorted by input (row, col);
    - frames stack in the left column from top to bottom;
    - solids stack in the right column from bottom to top;
    - objects occupying the same final row slot exchange colors.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    frames, solids = _collect_objects(g, bg)
    return _build_combo_output(frames, solids, bg)


def _gen_attempt() -> np.ndarray | None:
    bg = 0
    h = int(np.random.randint(24, 33))
    w = int(np.random.randint(24, 33))
    g = np.full((h, w), bg, dtype=int)

    n_pairs = int(np.random.randint(2, 5))
    total = 2 * n_pairs
    if total > 9:
        return None

    try:
        colors = list(map(int, np.random.choice(np.arange(1, 10), size=total, replace=False)))
        frame_colors = colors[:n_pairs]
        solid_colors = colors[n_pairs:]

        objects: list[tuple[str, int]] = [("frame", col) for col in frame_colors]
        objects += [("solid", col) for col in solid_colors]
        np.random.shuffle(objects)

        for kind, col in objects:
            patch = _make_frame(col, bg) if kind == "frame" else _make_solid(col)
            r0, c0 = random_free_location(
                g,
                patch,
                transparent=bg,
                border=1,
                padding=1,
                padding_connectivity=4,
            )
            paste(g, patch, r0, c0, transparent=bg)

        frames, solids = _collect_objects(g, bg)
        if len(frames) != n_pairs or len(solids) != n_pairs:
            return None

        combo_out = _build_combo_output(frames, solids, bg)
        source_out = _build_original_output(frames, solids, bg)
        naive_out = _build_naive_same_index_swap_output(frames, solids, bg)

        if np.array_equal(combo_out, source_out):
            return None
        if np.array_equal(combo_out, naive_out):
            return None
        if _reverse_pair_distance(frames, solids) <= 0:
            return None
        if not np.array_equal(main(g), combo_out):
            return None

        return g
    except Exception:
        return None


def generate_input() -> np.ndarray:
    for _ in range(300):
        candidate = _gen_attempt()
        if candidate is not None:
            return candidate
    raise RuntimeError("generate_input failed after 300 attempts")
