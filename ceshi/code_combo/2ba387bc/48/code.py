# Combo variant of 2ba387bc (index 48)
# combo_spec: D2/1+D5/4+D9/3
# Name: 交错排列+等数量+降序排序

from __future__ import annotations

import numpy as np

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, paste, random_free_location

# Design analysis for this combo variant
# a) Concrete behavior and trace:
#    - Input still consists of separated 4x4 objects on a uniform background.
#      There are two object types:
#      1. frame: one-color 1px border with a bg-colored 2x2 center;
#      2. solid: a full 4x4 monochrome block.
#    - Original seed logic:
#      * classify objects into frames and solids;
#      * sort each group by (row ascending, col ascending);
#      * stack frames in the left 4 columns and solids in the right 4 columns.
#    - Combo logic:
#      * classify objects exactly the same way;
#      * require equal group sizes in generated inputs;
#      * sort frames by (row descending, col ascending);
#      * sort solids by (row descending, col ascending);
#      * output width becomes 4, and objects are placed in one single column as
#        frame_1, solid_1, frame_2, solid_2, ...
#    - Example trace:
#      Suppose frames are at rows [3, 16, 22] and solids at rows [5, 12, 19].
#      Original output would be:
#        left column  = frame rows [3, 16, 22]
#        right column = solid rows [5, 12, 19]
#      Combo output becomes one stack:
#        [frame 22, solid 19, frame 16, solid 12, frame 3, solid 5]
#      Differences are coupled: the column split disappears, ordering is reversed
#      within both groups, and equal counts make the full output strictly alternate.
# b) Complexity calibration:
#    - To keep the pattern visible, the generator uses 2-4 frames and exactly the
#      same number of solids. This gives 4-8 total objects, enough for repeated
#      alternation without overcrowding the board.
#    - Each object gets a unique color, so humans can track movement and ordering.
#    - The generator rejects samples where descending-vs-ascending would produce
#      the same interleaved stack, or where a naive "sort all objects globally by
#      row descending" already matches the intended alternation.
# c) Implicit-assumption audit of the original main():
#    - `find_components(..., monochromatic=False)` assumes every object remains a
#      single isolated 4-connected component even for hollow frames. Still valid.
#    - `bbox_h == 4 and bbox_w == 4` assumes all relevant objects occupy 4x4
#      bboxes. Still valid because the generator emits only 4x4 objects.
#    - The frame test assumes monochrome border + bg 2x2 center. Unchanged.
#    - The original ascending sort no longer matches the combo, so the sort keys
#      must flip to row descending.
#    - The original output height `4 * max(...)` relied on two side-by-side
#      columns sharing row slots. That assumption breaks here because every object
#      occupies its own slot in a single column, so height must be `4 * total`.
#    - The original width 8 and paste columns 0 / 4 are invalid in the combo and
#      must be replaced by width 4 with one shared destination column.
# d) Placement legality:
#    - Input objects are placed with `random_free_location(..., border=1,
#      padding=1, padding_connectivity=4)`, so they stay in bounds, disjoint, and
#      visually separated.
#    - Post-placement validation re-collects components and confirms the intended
#      counts survive segmentation.
# e) Visual observability:
#    - All control variables are human-visible: object type and vertical position.
#      No rule depends on numeric color ids or hidden state.
#    - Equal counts make the entire output alternate frame/solid repeatedly, so
#      the single-column layout is visually diagnostic rather than accidental.
#    - Rejecting samples that already match global row-desc order prevents the
#      dataset from collapsing into "just stack everything bottom-to-top".
# f) Visual cleanliness:
#    - Objects remain sparse and isolated in the input.
#    - The output is a clean single stack of non-overlapping 4x4 blocks, with no
#      extra markers or decorative pixels.
# g) Color inferability:
#    - Colors only identify individual objects; no color remapping is introduced.
# 9) New assumptions introduced by the modified main():
#    - The intended task manifold has equal numbers of frames and solids so the
#      output alternates perfectly from top to bottom; `generate_input()` enforces
#      this exactly.
#    - Descending order is easiest to infer when intra-group row ties do not
#      occur, so the generator enforces distinct top rows within each type, and in
#      practice across all objects.
#    - `main()` still tolerates off-manifold unequal counts by appending leftovers
#      after alternation, but the generator never emits such inputs.


Object = tuple[int, int, np.ndarray]


def _collect_objects(grid: np.ndarray) -> tuple[int, list[Object], list[Object]]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")

    frames: list[Object] = []
    solids: list[Object] = []

    for comp in comps:
        b = comp["bbox"]
        if bbox_h(b) != 4 or bbox_w(b) != 4:
            continue

        r0, c0, _r1, _c1 = b
        sub = crop_bbox(g, b)
        border = np.concatenate([sub[0, :], sub[-1, :], sub[1:-1, 0], sub[1:-1, -1]])
        border_colors = set(map(int, border.flatten()))
        is_frame = len(border_colors) == 1 and bool((sub[1:3, 1:3] == bg).all())

        if is_frame:
            frames.append((int(r0), int(c0), sub))
        else:
            solids.append((int(r0), int(c0), sub))

    return bg, frames, solids


def _sort_group_desc(group: list[Object]) -> list[Object]:
    return sorted(group, key=lambda t: (-int(t[0]), int(t[1])))


def _sort_group_asc(group: list[Object]) -> list[Object]:
    return sorted(group, key=lambda t: (int(t[0]), int(t[1])))


def _interleave(frames: list[Object], solids: list[Object]) -> list[Object]:
    ordered: list[Object] = []
    for i in range(max(len(frames), len(solids))):
        if i < len(frames):
            ordered.append(frames[i])
        if i < len(solids):
            ordered.append(solids[i])
    return ordered


def _render_output(bg: int, frames: list[Object], solids: list[Object], *, descending: bool) -> np.ndarray:
    ordered_frames = _sort_group_desc(frames) if descending else _sort_group_asc(frames)
    ordered_solids = _sort_group_desc(solids) if descending else _sort_group_asc(solids)
    ordered = _interleave(ordered_frames, ordered_solids)

    out = np.full((4 * len(ordered), 4), int(bg), dtype=int)
    for i, (_r0, _c0, sub) in enumerate(ordered):
        paste(out, sub, int(i * 4), 0)
    return out


def _global_desc_type_order(frames: list[Object], solids: list[Object]) -> list[str]:
    tagged = [("frame", *item) for item in frames] + [("solid", *item) for item in solids]
    tagged.sort(key=lambda t: (-int(t[1]), int(t[2])))
    return [str(t[0]) for t in tagged]


def _alternating_type_order(n_pairs: int) -> list[str]:
    order: list[str] = []
    for _ in range(int(n_pairs)):
        order.extend(["frame", "solid"])
    return order


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combo rule:
    - classify 4x4 objects into frames and solids;
    - sort each type bottom-to-top by input row (row descending, col ascending on ties);
    - place them into one single 4-column stack, alternating frame then solid.
    """
    bg, frames, solids = _collect_objects(np.array(grid, dtype=int))
    return _render_output(bg, frames, solids, descending=True)


def _make_frame(col: int, bg: int = 0) -> np.ndarray:
    patch = np.full((4, 4), int(col), dtype=int)
    patch[1:3, 1:3] = int(bg)
    return patch


def _make_solid(col: int) -> np.ndarray:
    return np.full((4, 4), int(col), dtype=int)


def _pick_unique_color(used_cols: set[int]) -> int:
    col = int(np.random.randint(1, 10))
    while col in used_cols:
        col = int(np.random.randint(1, 10))
    used_cols.add(col)
    return col


def _sample_pair_count() -> int:
    return int(np.random.randint(2, 5))


def _place_with_fresh_row(
    grid: np.ndarray,
    patch: np.ndarray,
    *,
    bg: int,
    used_rows: set[int],
) -> tuple[int, int]:
    for _ in range(100):
        r0, c0 = random_free_location(
            grid,
            patch,
            transparent=bg,
            border=1,
            padding=1,
            padding_connectivity=4,
        )
        if int(r0) not in used_rows:
            return int(r0), int(c0)
    raise ValueError("could not find a placement with a fresh row")


def _combo_is_observable(grid: np.ndarray) -> bool:
    bg, frames, solids = _collect_objects(grid)
    if len(frames) != len(solids):
        return False
    if len(frames) < 2:
        return False

    frame_rows = [int(r0) for r0, _c0, _sub in frames]
    solid_rows = [int(r0) for r0, _c0, _sub in solids]
    if len(set(frame_rows)) != len(frame_rows):
        return False
    if len(set(solid_rows)) != len(solid_rows):
        return False

    desc = _render_output(bg, frames, solids, descending=True)
    asc = _render_output(bg, frames, solids, descending=False)
    if np.array_equal(desc, asc):
        return False

    if _global_desc_type_order(frames, solids) == _alternating_type_order(len(frames)):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    bg = 0
    n_pairs = _sample_pair_count()
    total = int(2 * n_pairs)

    h = int(np.random.randint(24, 36))
    w = int(np.random.randint(24, 36))
    grid = np.full((h, w), int(bg), dtype=int)

    try:
        used_cols: set[int] = set()
        used_rows: set[int] = set()

        for _ in range(n_pairs):
            col = _pick_unique_color(used_cols)
            patch = _make_frame(col, bg)
            r0, c0 = _place_with_fresh_row(grid, patch, bg=bg, used_rows=used_rows)
            used_rows.add(int(r0))
            paste(grid, patch, r0, c0, transparent=bg)

        for _ in range(n_pairs):
            col = _pick_unique_color(used_cols)
            patch = _make_solid(col)
            r0, c0 = _place_with_fresh_row(grid, patch, bg=bg, used_rows=used_rows)
            used_rows.add(int(r0))
            paste(grid, patch, r0, c0, transparent=bg)

        bg2, frames, solids = _collect_objects(grid)
        if len(frames) != n_pairs or len(solids) != n_pairs:
            return None
        if total != len(frames) + len(solids):
            return None
        if not _combo_is_observable(grid):
            return None

        expected = _render_output(bg2, frames, solids, descending=True)
        actual = main(grid)
        if actual.shape != (4 * total, 4):
            return None
        if not np.array_equal(actual, expected):
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
