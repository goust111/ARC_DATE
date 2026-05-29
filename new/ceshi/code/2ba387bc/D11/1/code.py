# Variant of 2ba387bc
# Dimension: D11 — 干扰项——输入中的额外对象 (src)
# Change: 输入中除4x4方块和背景外无其他对象... → 增加4x4但非框架也非纯实心的干扰方块

from __future__ import annotations

import numpy as np

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, paste, random_free_location


"""
Design analysis for D11
======================
source_task:
- None. No external reference solver is required for this variant.

a) Concretization and behavior trace
- Original input only contains two meaningful 4x4 object types on the background:
  1. frame: 1-cell-thick monochrome border with a background-colored 2x2 hole
  2. solid: all 16 cells share one non-background color
- D11 adds a third visible 4x4 category in the input: distractor squares. They are still
  exactly 4x4 and visually square-like, but they are neither a true frame nor a true solid.
  Example patterns: checkerboard, 2-row stripes, 2-column stripes, or a border with a filled
  center of a second color. These must be ignored in the output.
- Small trace:
  input objects by top row = [frame(blue) at r=2, distractor(red/yellow stripes) at r=7,
  solid(green) at r=12].
  Original code path:
  - frame passes the frame test -> stacked left
  - distractor fails frame test -> falls into `else` -> incorrectly stacked right as if solid
  - solid also stacked right
  Variant code path:
  - frame still stacked left
  - distractor passes neither strict frame nor strict solid test -> ignored
  - solid still stacked right
  Difference: the right column now contains only actual solid blocks, not "everything else".

b) Complexity calibration
- The distractor must be visible enough to matter, so every generated input contains at least
  one 4x4 distractor.
- Distractors are restricted to a small library of connected 4x4 patterns. This keeps the task
  readable while still showing that "4x4 square-like" is not sufficient; the solver must detect
  the exact interior rule for frames and the exact uniformity rule for solids.
- We avoid too many degrees of freedom: the transform itself is unchanged, and the only new
  branch is "ignore third category".

c) Implicit-assumption audit of the original main()
- `find_components(... monochromatic=False)`: assumes each meaningful object appears as one
  connected foreground component even if it contains several colors. This still holds because
  generator places every distractor as one connected 4x4 component.
- `if bbox_h(b) != 4 or bbox_w(b) != 4: continue`: assumes relevant objects have 4x4 bounding
  boxes. Still valid; distractors are also 4x4 and therefore reach classification.
- Original `if len(bcols) == 1 and interior == bg: frame else: solid` depended on the old
  invariant that every non-frame 4x4 was automatically a solid. D11 breaks that invariant.
  The fix is to split this into three outcomes: frame / solid / ignore.
- Sorting by `(row, col)` still matches the original spec and does not depend on distractors.

d) Placement legality
- All objects are placed with `random_free_location(..., border=1, padding=1,
  padding_connectivity=4)`, so they do not overlap, do not touch edge-to-edge, and stay
  within the canvas.
- Distractors use only connected 4x4 patterns, so the component finder sees each as exactly
  one candidate object and the bbox-based classifier can inspect the full 4x4 crop safely.

e) Visual observability
- Humans can see the new behavior from a few examples because the input now visibly contains
  extra 4x4 patterned squares that disappear from the output, while normal frames and solids
  still stack as before.
- The rule is inferable from shape content rather than hidden metadata: empty-centered border
  blocks go left, fully uniform blocks go right, patterned 4x4 squares vanish.

f) Visual cleanliness
- No overlays are used. Objects are isolated with padding, so each 4x4 patch is easy to parse.
- Distractors are limited to simple high-contrast internal patterns instead of noisy mosaics.

g) Color-change inferability
- Colors are not remapped. Output preserves the original colors of kept objects.
- Distractor colors do not encode hidden ordering; they only make the ignored third category
  visibly distinct.

h) Generator correctness strategy
- Generator directly constructs the input, then builds the expected output from the placed
  frame/solid records, and verifies `main(input) == expected`.
- It also verifies that the realized input contains the intended numbers of frames, solids, and
  distractors under the same strict classifier used by the solver.

i) New solver assumptions introduced by this variant
- A "solid" is now assumed to be exactly one non-background color across the whole 4x4 crop.
  Generator guarantees this for all genuine solids.
- A "frame" is assumed to have a monochrome non-background border and a background-colored
  2x2 interior. Generator guarantees this for all genuine frames.
- Any other connected 4x4 component is ignored. Generator guarantees every distractor lands in
  that remainder class, so the ignore branch is exercised on every sample.
"""


def _is_frame(sub: np.ndarray, bg: int) -> bool:
    border = np.concatenate([sub[0, :], sub[-1, :], sub[1:-1, 0], sub[1:-1, -1]])
    bcols = set(map(int, border.flatten()))
    return len(bcols) == 1 and next(iter(bcols)) != int(bg) and bool((sub[1:3, 1:3] == int(bg)).all())


def _is_solid(sub: np.ndarray, bg: int) -> bool:
    col = int(sub[0, 0])
    return col != int(bg) and bool((sub == col).all())


def _classify_4x4(sub: np.ndarray, bg: int) -> str | None:
    if _is_frame(sub, bg):
        return "frame"
    if _is_solid(sub, bg):
        return "solid"
    return None


def _collect_objects(grid: np.ndarray, bg: int) -> tuple[list[tuple[int, int, np.ndarray]], list[tuple[int, int, np.ndarray]], list[tuple[int, int, np.ndarray]]]:
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
    Sort scattered 4x4 squares into two columns by type.

    Variant behavior:
    - Frames: 4x4 with one non-bg border color and a bg 2x2 interior -> keep, stack left.
    - Solids: 4x4 filled with one non-bg color -> keep, stack right.
    - Other connected 4x4 patterns -> ignore completely.
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


def _gen_attempt() -> np.ndarray | None:
    bg = 0
    h = int(np.random.randint(22, 34))
    w = int(np.random.randint(22, 34))
    g = np.full((h, w), bg, dtype=int)

    n_frames = int(np.random.randint(1, 5))
    n_solids = int(np.random.randint(1, 5))
    n_distractors = int(np.random.randint(1, 4))
    distractor_kinds = ["checker", "hstripe", "vstripe", "filled_frame"]

    records: list[tuple[str, int, int, np.ndarray]] = []

    try:
        used_primary_cols: set[int] = set()

        for _ in range(n_frames):
            col = int(np.random.randint(1, 10))
            while col in used_primary_cols:
                col = int(np.random.randint(1, 10))
            used_primary_cols.add(col)
            patch = _make_frame(col, bg)
            r0, c0 = random_free_location(g, patch, transparent=bg, border=1, padding=1, padding_connectivity=4)
            paste(g, patch, r0, c0, transparent=bg)
            records.append(("frame", int(r0), int(c0), patch))

        for _ in range(n_solids):
            col = int(np.random.randint(1, 10))
            while col in used_primary_cols:
                col = int(np.random.randint(1, 10))
            used_primary_cols.add(col)
            patch = _make_solid(col)
            r0, c0 = random_free_location(g, patch, transparent=bg, border=1, padding=1, padding_connectivity=4)
            paste(g, patch, r0, c0, transparent=bg)
            records.append(("solid", int(r0), int(c0), patch))

        for _ in range(n_distractors):
            col_a = int(np.random.randint(1, 10))
            col_b = int(np.random.randint(1, 10))
            while col_b == col_a:
                col_b = int(np.random.randint(1, 10))
            kind = str(np.random.choice(distractor_kinds))
            patch = _make_distractor(kind, col_a, col_b)
            if _classify_4x4(patch, bg) is not None:
                return None
            r0, c0 = random_free_location(g, patch, transparent=bg, border=1, padding=1, padding_connectivity=4)
            paste(g, patch, r0, c0, transparent=bg)
            records.append(("distractor", int(r0), int(c0), patch))

        expected_frames = [(r0, c0, patch) for kind, r0, c0, patch in records if kind == "frame"]
        expected_solids = [(r0, c0, patch) for kind, r0, c0, patch in records if kind == "solid"]
        expected_frames.sort(key=lambda t: (int(t[0]), int(t[1])))
        expected_solids.sort(key=lambda t: (int(t[0]), int(t[1])))
        expected = _build_output(expected_frames, expected_solids, bg)

        got = main(g)
        if not np.array_equal(got, expected):
            return None

        frames_seen, solids_seen, distractors_seen = _collect_objects(g, bg)
        if len(frames_seen) != n_frames or len(solids_seen) != n_solids or len(distractors_seen) != n_distractors:
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
