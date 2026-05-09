# Combo variant of 2ba387bc (index 27)
# combo_spec: D9/0+D11/1
# Name: 同色方块+4x4图案干扰

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

try:
    from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, paste, random_free_location
except ModuleNotFoundError:
    sys.path.append(str(Path("/home/lixinyu/snap/new/solvers")))
    from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, paste, random_free_location


"""
Design analysis for combo D9/0 + D11/1
======================================

Task summary carried over from the source task
- The meaningful source objects are still isolated 4x4 blocks on a dominant background.
- True frames are 4x4 hollow squares: one non-background border color and a 2x2 background
  center.
- True solids are 4x4 uniform filled squares in one non-background color.
- Output still stacks frames in the left 4 columns and solids in the right 4 columns, sorted
  by input row then column.

a) Concrete behavior and trace
- The fusion introduces two simultaneous changes:
  1. colors are no longer unique across the meaningful blocks;
  2. some extra 4x4 connected patterns appear in the input, but they are neither true frames
     nor true solids and must be ignored.
- Small trace:
  input top-to-bottom could contain
  - frame(color 3) at row 2,
  - checker distractor(colors 3 and 7) at row 8,
  - solid(color 3) at row 13,
  - filled-frame distractor(colors 5 and 2) at row 18.
  Original solver:
  - frame passes its frame test and goes left;
  - checker distractor fails frame test, falls through `else`, and is incorrectly treated as a
    solid on the right;
  - solid goes right;
  - second distractor also goes right.
  Fused solver:
  - frame is kept as frame;
  - checker distractor is not a frame and not a strict solid, so it is ignored;
  - solid is kept as solid even though it shares color 3 with the frame;
  - second distractor is ignored.
- The visible difference from the original task is therefore twofold in the same sample:
  same-colored frame/solid pairs split across the two output columns, and patterned 4x4 blocks
  disappear entirely.

b) Complexity calibration
- Merely "allowing" shared colors is too weak because many random samples would still use all
  different colors; likewise merely "allowing" distractors is too weak because many samples
  would still degenerate to the source task.
- To make the fusion observable in every sample, the generator enforces:
  - at least one frame color also appears on a solid;
  - at least one distractor exists;
  - relevant frame/solid colors are drawn from a small shared palette so repetition is visible.
- Distractors come from a small library of simple connected 4x4 patterns (checker, horizontal
  stripe, vertical stripe, filled-frame). This is enough variety to show "not every 4x4 square
  is meaningful" without making the task visually noisy.

c) Hidden assumptions audit of the original main()
- `get_bg(g)`: still valid because the background remains the most common color.
- `find_components(... monochromatic=False)`: still valid if every object remains a separated
  foreground component. The generator keeps at least one cell of 4-connectivity padding between
  objects, so even same-colored objects never merge.
- `bbox_h == bbox_w == 4`: still valid for all relevant and distracting objects. This keeps the
  reasoning space narrow and readable.
- Original branch:
    if frame_test: frames.append(...)
    else: solids.append(...)
  This relied on the old invariant that every non-frame 4x4 object was a valid solid. The
  distractor variant breaks that assumption, so the fused solver must explicitly test for
  `solid` and otherwise ignore the component.
- Any solver logic that implicitly keys objects by unique color is now invalid. The fused solver
  must classify by local 4x4 structure only.

d) Placement legality
- Every object is placed with `random_free_location(..., border=1, padding=1,
  padding_connectivity=4)`.
- This guarantees no overlap, no out-of-bounds placement, and no edge-touching between objects.
- Frames are pasted with `transparent=bg` so their hollow center stays background; distractors and
  solids contain no background cells, so they remain full 4x4 connected components.

e) Visual observability
- Across 2-4 examples, a human can infer the fused rule because:
  - some same-colored blocks land in different output columns, so color alone cannot decide;
  - some 4x4 patterned blocks vanish from the output, so "being a 4x4 block" is not enough;
  - the kept-vs-ignored difference is explained by exact internal structure: empty-centered border
    means frame, all-one-color means solid, everything else disappears.
- No hidden numeric-color ordering is used for sorting; sorting remains by visible row position.

f) Visual cleanliness
- All objects remain isolated 4x4 units with background gaps around them.
- Distractors are chosen from clean, regular motifs rather than noisy random mosaics.
- The relevant structure is readable at a glance: hollow border, full solid, or patterned fill.

g) Change inferability
- There is no color remapping. Output colors are copied from the kept input objects.
- The only color-related change is multiplicity: repeated colors across shapes are deliberate and
  visible, but they do not encode any hidden rule.

h) Generator correctness strategy
- The generator first constructs a mixed object list containing frames, solids, and distractors.
- After randomized placement, it builds the expected output only from the placed frames and solids,
  sorted by realized `(row, col)`, and verifies `main(input) == expected`.
- It also re-runs the strict classifier on the realized input to confirm that the intended counts
  of frames, solids, and distractors survived placement.

i) New implicit assumptions introduced by the fused solver
- A true frame is assumed to have one non-background border color and a background 2x2 center.
  The generator guarantees this for every frame.
- A true solid is assumed to be one non-background color over all 16 cells. The generator
  guarantees this for every solid.
- Any other connected 4x4 foreground component is assumed irrelevant and is ignored. The generator
  guarantees that every distractor falls into exactly this remainder class.
- Same-colored meaningful objects are assumed to remain disconnected under 4-connectivity. The
  generator guarantees this with padded placement.

j) Post-generation fusion metrics
- `shared_cross_type_colors = |frame_colors ∩ solid_colors|` must be at least 1.
- `n_distractors` must be at least 1.
- Samples failing either metric are rejected as degenerate because they would not visibly exercise
  the combined variant.
"""


def _is_frame(sub: np.ndarray, bg: int) -> bool:
    if sub.shape != (4, 4):
        return False
    border = np.concatenate([sub[0, :], sub[-1, :], sub[1:-1, 0], sub[1:-1, -1]])
    bcols = set(map(int, border.flatten()))
    return len(bcols) == 1 and next(iter(bcols)) != int(bg) and bool((sub[1:3, 1:3] == int(bg)).all())


def _is_solid(sub: np.ndarray, bg: int) -> bool:
    if sub.shape != (4, 4):
        return False
    vals = set(map(int, sub.flatten()))
    return len(vals) == 1 and int(next(iter(vals))) != int(bg)


def _classify_4x4(sub: np.ndarray, bg: int) -> str | None:
    if _is_frame(sub, bg):
        return "frame"
    if _is_solid(sub, bg):
        return "solid"
    return None


def _collect_objects(
    grid: np.ndarray,
    bg: int,
) -> tuple[list[tuple[int, int, np.ndarray]], list[tuple[int, int, np.ndarray]], list[tuple[int, int, np.ndarray]]]:
    comps = find_components(grid, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    frames: list[tuple[int, int, np.ndarray]] = []
    solids: list[tuple[int, int, np.ndarray]] = []
    distractors: list[tuple[int, int, np.ndarray]] = []

    for comp in comps:
        b = comp["bbox"]
        if bbox_h(b) != 4 or bbox_w(b) != 4:
            continue
        r0, c0, _r1, _c1 = b
        sub = crop_bbox(grid, b)
        kind = _classify_4x4(sub, bg)
        if kind == "frame":
            frames.append((int(r0), int(c0), sub))
        elif kind == "solid":
            solids.append((int(r0), int(c0), sub))
        else:
            distractors.append((int(r0), int(c0), sub))

    frames.sort(key=lambda t: (int(t[0]), int(t[1])))
    solids.sort(key=lambda t: (int(t[0]), int(t[1])))
    distractors.sort(key=lambda t: (int(t[0]), int(t[1])))
    return frames, solids, distractors


def _build_output(
    frames: list[tuple[int, int, np.ndarray]],
    solids: list[tuple[int, int, np.ndarray]],
    bg: int,
) -> np.ndarray:
    out_h = 4 * max(int(len(frames)), int(len(solids)))
    out = np.full((out_h, 8), int(bg), dtype=int)
    for i, (_r0, _c0, sub) in enumerate(frames):
        paste(out, sub, int(i * 4), 0)
    for i, (_r0, _c0, sub) in enumerate(solids):
        paste(out, sub, int(i * 4), 4)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep only true 4x4 frames and true 4x4 solids.

    Fused-variant behavior:
    - Frames and solids may share colors, so classification must use shape.
    - Extra connected 4x4 patterned blocks may exist; they are ignored entirely.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    frames, solids, _distractors = _collect_objects(g, bg)
    return _build_output(frames, solids, bg)


def _make_frame(col: int, bg: int = 0) -> np.ndarray:
    a = np.full((4, 4), int(col), dtype=int)
    a[1:3, 1:3] = int(bg)
    return a


def _make_solid(col: int) -> np.ndarray:
    return np.full((4, 4), int(col), dtype=int)


def _make_distractor(kind: str, col_a: int, col_b: int) -> np.ndarray:
    a = np.full((4, 4), int(col_a), dtype=int)
    b = int(col_b)
    if kind == "checker":
        for r in range(4):
            for c in range(4):
                if (r + c) % 2 == 1:
                    a[r, c] = b
        return a
    if kind == "hstripe":
        a[2:, :] = b
        return a
    if kind == "vstripe":
        a[:, 2:] = b
        return a
    if kind == "filled_frame":
        a[1:3, 1:3] = b
        return a
    raise ValueError(f"unknown distractor kind: {kind}")


def _object_color(sub: np.ndarray) -> int:
    return int(sub[0, 0])


def _analyze_input(g: np.ndarray) -> dict[str, int]:
    bg = int(get_bg(g))
    frames, solids, distractors = _collect_objects(g, bg)
    frame_cols = [_object_color(sub) for _r0, _c0, sub in frames]
    solid_cols = [_object_color(sub) for _r0, _c0, sub in solids]
    shared = set(frame_cols) & set(solid_cols)
    return {
        "n_frames": int(len(frames)),
        "n_solids": int(len(solids)),
        "n_distractors": int(len(distractors)),
        "total_kept": int(len(frames) + len(solids)),
        "distinct_kept_colors": int(len(set(frame_cols + solid_cols))),
        "shared_cross_type_colors": int(len(shared)),
    }


def _choose_primary_colors(n_frames: int, n_solids: int) -> tuple[list[int], list[int], list[int]]:
    total = int(n_frames + n_solids)
    if total <= 2:
        palette_size = 1
    else:
        palette_size = int(np.random.randint(2, min(4, total - 1, 9) + 1))
    palette = list(map(int, np.random.choice(np.arange(1, 10), size=palette_size, replace=False)))
    shared = int(np.random.choice(palette))

    frame_cols = [shared]
    solid_cols = [shared]
    for _ in range(n_frames - 1):
        frame_cols.append(int(np.random.choice(palette)))
    for _ in range(n_solids - 1):
        solid_cols.append(int(np.random.choice(palette)))
    return frame_cols, solid_cols, palette


def _choose_distractor_patch(primary_palette: list[int], distractor_kinds: list[str]) -> np.ndarray:
    kind = str(np.random.choice(distractor_kinds))
    all_cols = list(range(1, 10))

    if len(primary_palette) >= 2 and bool(np.random.randint(0, 2)):
        chosen = list(map(int, np.random.choice(np.array(primary_palette), size=2, replace=False)))
        col_a, col_b = chosen[0], chosen[1]
    else:
        col_a = int(np.random.choice(all_cols))
        col_b = int(np.random.choice(all_cols))
        while col_b == col_a:
            col_b = int(np.random.choice(all_cols))

    patch = _make_distractor(kind, col_a, col_b)
    if _classify_4x4(patch, 0) is not None:
        raise ValueError("distractor patch accidentally matches a kept class")
    return patch


def _gen_attempt() -> np.ndarray | None:
    bg = 0
    h = int(np.random.randint(24, 37))
    w = int(np.random.randint(24, 37))
    g = np.full((h, w), bg, dtype=int)

    n_frames = int(np.random.randint(1, 5))
    n_solids = int(np.random.randint(1, 5))
    n_distractors = int(np.random.randint(1, 4))
    distractor_kinds = ["checker", "hstripe", "vstripe", "filled_frame"]

    frame_cols, solid_cols, primary_palette = _choose_primary_colors(n_frames, n_solids)

    pending: list[tuple[str, np.ndarray]] = []
    for col in frame_cols:
        pending.append(("frame", _make_frame(col, bg)))
    for col in solid_cols:
        pending.append(("solid", _make_solid(col)))
    for _ in range(n_distractors):
        pending.append(("distractor", _choose_distractor_patch(primary_palette, distractor_kinds)))

    order = np.random.permutation(len(pending))
    placed: list[tuple[str, int, int, np.ndarray]] = []

    try:
        for idx in order:
            kind, patch = pending[int(idx)]
            r0, c0 = random_free_location(
                g,
                patch,
                transparent=bg,
                border=1,
                padding=1,
                padding_connectivity=4,
            )
            paste(g, patch, int(r0), int(c0), transparent=bg)
            placed.append((kind, int(r0), int(c0), patch))

        expected_frames = [(r0, c0, patch) for kind, r0, c0, patch in placed if kind == "frame"]
        expected_solids = [(r0, c0, patch) for kind, r0, c0, patch in placed if kind == "solid"]
        expected_frames.sort(key=lambda t: (int(t[0]), int(t[1])))
        expected_solids.sort(key=lambda t: (int(t[0]), int(t[1])))
        expected = _build_output(expected_frames, expected_solids, bg)

        got = main(g)
        if not np.array_equal(got, expected):
            return None

        info = _analyze_input(g)
        if info["n_frames"] != n_frames or info["n_solids"] != n_solids or info["n_distractors"] != n_distractors:
            return None
        if info["shared_cross_type_colors"] < 1:
            return None
        if info["n_distractors"] < 1:
            return None
        if info["distinct_kept_colors"] >= info["total_kept"]:
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
