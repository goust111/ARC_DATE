# Variant of b5ca7ac4
# Dimension: D3 — 源对象——框架的结构与颜色组成 (src)
# Change: 每个源对象为5x5方框，外层1像素宽的边框为组标识色（青色8或红色2），内部3x3区域填充统一的其他颜色；所有框架尺寸相同 → 可变尺寸框架（5x5和7x7混合）

from __future__ import annotations

import random

import numpy as np

from arc_utils import crop_bbox, get_bg, paste


# Design analysis for the D3 variant
# a) Concrete behavior:
#    - Every object is still a hollow square frame with a 1-cell border in teal (8) or red (2),
#      but side lengths are now odd and variable. The generator uses a visible mix of 5x5 and 7x7.
#    - The output still moves teal frames to the left side and red frames to the right side, keeping
#      each frame's original top row. Ordering is still by input column: teal from left to right,
#      red from right to left.
#    - The changed step is the inward placement. In the original task, every overlapping frame moved
#      by a fixed width of 5. Here the inward offset depends on the width of the edge-anchored frame
#      that it vertically overlaps. Example on the teal side:
#        original: edge 5x5 at row 2 -> later overlapping 5x5 goes to col 5
#        variant : edge 7x7 at row 2 -> later overlapping 5x5 goes to col 7
#      On the red side, a 5x5 overlapping a 7x7 edge frame lands at col W-7-5 instead of W-10.
# b) Complexity calibration:
#    - Only two sizes are generated, 5 and 7. This keeps the rule learnable from a few examples.
#    - Each sample contains both sizes, and at least one inward frame depends on a 7-wide edge frame,
#      so the size-sensitive offset is visible instead of silently degenerating to the old task.
# c) Implicit assumption audit:
#    - The original fixed-size scan over 5x5 windows no longer works; detection must scan odd sizes.
#    - The original hard-coded inward columns 5 and W-10 no longer work; inward columns must use the
#      width of the overlapped edge frame, and on the red side also the width of the moving frame.
#    - Overlap checks must use each frame's own height instead of a shared SQ constant.
# d) Placement legality:
#    - Input frames are generated non-overlapping and in-bounds.
#    - Edge output frames on the same side have disjoint row spans.
#    - Each inward frame overlaps exactly one edge frame on its own side, so the width-driven inward
#      column is unambiguous and the solver's dependency assumptions are satisfied.
# e) Visual observability:
#    - The rule never sorts by color id or hidden numeric metadata; only visible columns, visible
#      frame widths, and visible row overlaps matter.
#    - Because the output includes both 5x5 and 7x7 frames and at least one 7-driven inward shift,
#      a human can notice that "the second layer starts after the width of the edge frame."
# f) Visual cleanliness:
#    - The generator avoids same-side rectangle overlap in both input and output.
#    - Cross-side output overlap is rejected, which keeps the central area readable even though
#      wide left and right stacks can reach toward the middle.
# g) Color inference:
#    - No color remapping is introduced. Inner fills remain uniform and act only as object identity,
#      preserving the original learnability while changing only the source-object size dimension.


AZURE = 8
RED = 2
FRAME_COLORS = {AZURE, RED}
GRID_H = 22
GRID_W = 22
SIZES = (5, 7)
N_AZURE_RANGE = (3, 4)
N_RED_RANGE = (3, 5)
MAX_INWARD = 2
SIDE_RETRIES = 200
INPUT_COL_RETRIES = 140
GEN_RETRIES = 1000


def _rows_overlap(r1: int, h1: int, r2: int, h2: int) -> bool:
    return not (int(r1 + h1 - 1) < int(r2) or int(r2 + h2 - 1) < int(r1))


def _rects_overlap(a: tuple[int, int, int], b: tuple[int, int, int]) -> bool:
    ar, ac, asz = a
    br, bc, bsz = b
    return _rows_overlap(ar, asz, br, bsz) and not (
        int(ac + asz - 1) < int(bc) or int(bc + bsz - 1) < int(ac)
    )


def _any_pairwise_overlap(entries: list[dict]) -> bool:
    rects = [(int(e["r0"]), int(e["c0"]), int(e["size"])) for e in entries]
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            if _rects_overlap(rects[i], rects[j]):
                return True
    return False


def _make_patch(size: int, frame_color: int, inner_color: int) -> np.ndarray:
    patch = np.full((size, size), frame_color, dtype=int)
    patch[1:-1, 1:-1] = inner_color
    return patch


def _frame_border(patch: np.ndarray) -> np.ndarray:
    return np.concatenate([patch[0, :], patch[-1, :], patch[1:-1, 0], patch[1:-1, -1]])


def _detect_frames(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    found: list[dict] = []
    seen: set[tuple[int, int, int]] = set()

    max_side = min(H, W)
    for size in range(5, max_side + 1, 2):
        for r0 in range(H - size + 1):
            for c0 in range(W - size + 1):
                patch = crop_bbox(g, (r0, c0, r0 + size - 1, c0 + size - 1))
                border = _frame_border(patch)
                if int(border.min()) != int(border.max()):
                    continue
                frame_col = int(border[0])
                if frame_col not in FRAME_COLORS:
                    continue

                inner = patch[1:-1, 1:-1]
                inner_vals = np.unique(inner)
                if len(inner_vals) != 1:
                    continue
                inner_col = int(inner_vals[0])
                if inner_col in {frame_col, bg}:
                    continue

                key = (int(r0), int(c0), int(size))
                if key in seen:
                    continue
                seen.add(key)
                found.append(
                    {
                        "frame": frame_col,
                        "r0": int(r0),
                        "c0": int(c0),
                        "size": int(size),
                        "patch": patch,
                    }
                )
    return found


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move mixed-size bordered frames to the side chosen by their border color.

    Input structure:
    - The background is the most common color.
    - Teal (8) and red (2) square frames are scattered around the grid.
    - Frames keep a 1-cell border and a uniform filled interior, but their side length can vary
      between 5x5 and 7x7 in the generated variant.

    Rule:
    - Teal frames move to the left side; red frames move to the right side.
    - Each frame keeps its original top row, and the processing order still uses the original
      column: teal ascending, red descending.
    - A frame with no row-overlap with an already placed edge frame sits flush on the edge.
    - If it row-overlaps edge frames on its side, it moves inward just past the widest overlapped
      edge frame, so the inward offset depends on frame width rather than a fixed constant.

    Key insight: The color decides the destination side, the input column decides which frame claims
    the edge first, and the width of that edge frame determines how far later overlapping frames must shift.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    out = np.full_like(g, bg)

    frames = _detect_frames(g, bg)

    teal_frames = sorted((f for f in frames if int(f["frame"]) == AZURE), key=lambda f: int(f["c0"]))
    edge_teal: list[dict] = []
    for frame in teal_frames:
        r0 = int(frame["r0"])
        size = int(frame["size"])
        overlaps = [
            placed
            for placed in edge_teal
            if _rows_overlap(r0, size, int(placed["r0"]), int(placed["size"]))
        ]
        dest_c = max((int(p["size"]) for p in overlaps), default=0)
        paste(out, frame["patch"], r0, dest_c, transparent=bg)
        if not overlaps:
            edge_teal.append({"r0": r0, "size": size})

    red_frames = sorted((f for f in frames if int(f["frame"]) == RED), key=lambda f: -int(f["c0"]))
    edge_red: list[dict] = []
    for frame in red_frames:
        r0 = int(frame["r0"])
        size = int(frame["size"])
        overlaps = [
            placed
            for placed in edge_red
            if _rows_overlap(r0, size, int(placed["r0"]), int(placed["size"]))
        ]
        if overlaps:
            widest = max(int(p["size"]) for p in overlaps)
            dest_c = int(W - widest - size)
        else:
            dest_c = int(W - size)
            edge_red.append({"r0": r0, "size": size})
        paste(out, frame["patch"], r0, dest_c, transparent=bg)

    return out


def _sample_side_sizes(n: int, n_edge: int, n_inward: int) -> tuple[list[int], list[int], list[int]]:
    edge_sizes = [random.choice(SIZES) for _ in range(n_edge)]
    inward_sizes = [random.choice(SIZES) for _ in range(n_inward)]

    idx_7 = random.randrange(n_edge)
    edge_sizes[idx_7] = 7
    if 5 not in edge_sizes + inward_sizes:
        if inward_sizes:
            inward_sizes[random.randrange(n_inward)] = 5
        else:
            edge_sizes[1 - idx_7] = 5

    anchor_choices = [i for i, sz in enumerate(edge_sizes) if sz == 7]
    anchors = [random.choice(anchor_choices)]
    for _ in range(1, n_inward):
        anchors.append(random.randrange(n_edge))
    return edge_sizes, inward_sizes, anchors


def _assign_edge_rows(sizes: list[int]) -> list[int] | None:
    order = list(range(len(sizes)))
    random.shuffle(order)
    rows = [None] * len(sizes)
    placed: list[tuple[int, int]] = []
    for idx in order:
        size = int(sizes[idx])
        candidates = list(range(GRID_H - size + 1))
        random.shuffle(candidates)
        picked = None
        for r0 in candidates:
            if all(not _rows_overlap(r0, size, pr, psz) for pr, psz in placed):
                picked = int(r0)
                break
        if picked is None:
            return None
        rows[idx] = picked
        placed.append((picked, size))
    return [int(r) for r in rows]


def _assign_inward_rows(
    edge_rows: list[int],
    edge_sizes: list[int],
    inward_sizes: list[int],
    anchors: list[int],
) -> list[int] | None:
    inward_rows: list[int] = []
    for idx, size in enumerate(inward_sizes):
        anchor = int(anchors[idx])
        candidates = list(range(GRID_H - size + 1))
        random.shuffle(candidates)
        picked = None
        for r0 in candidates:
            if not _rows_overlap(r0, size, edge_rows[anchor], edge_sizes[anchor]):
                continue
            bad = False
            for e_idx, (er, esz) in enumerate(zip(edge_rows, edge_sizes)):
                if e_idx == anchor:
                    continue
                if _rows_overlap(r0, size, er, esz):
                    bad = True
                    break
            if bad:
                continue
            if any(_rows_overlap(r0, size, ir, inward_sizes[j]) for j, ir in enumerate(inward_rows)):
                continue
            picked = int(r0)
            break
        if picked is None:
            return None
        inward_rows.append(picked)
    return inward_rows


def _build_side_layout(
    frame_color: int,
    count_range: tuple[int, int],
    side: str,
    forbidden: list[dict] | None = None,
) -> list[dict] | None:
    n = random.randint(*count_range)
    n_inward = random.randint(1, min(MAX_INWARD, n - 1))
    n_edge = n - n_inward

    for _ in range(SIDE_RETRIES):
        edge_sizes, inward_sizes, anchors = _sample_side_sizes(n, n_edge, n_inward)
        edge_rows = _assign_edge_rows(edge_sizes)
        if edge_rows is None:
            continue
        inward_rows = _assign_inward_rows(edge_rows, edge_sizes, inward_sizes, anchors)
        if inward_rows is None:
            continue

        entries: list[dict] = []
        for idx, (r0, size) in enumerate(zip(edge_rows, edge_sizes)):
            if side == "left":
                c0 = 0
            else:
                c0 = GRID_W - size
            entries.append(
                {
                    "frame": frame_color,
                    "size": int(size),
                    "r0": int(r0),
                    "c0": int(c0),
                    "is_edge": True,
                    "anchor": idx,
                }
            )

        for idx, (r0, size, anchor) in enumerate(zip(inward_rows, inward_sizes, anchors)):
            anchor_size = int(edge_sizes[anchor])
            if side == "left":
                c0 = anchor_size
            else:
                c0 = GRID_W - anchor_size - size
            entries.append(
                {
                    "frame": frame_color,
                    "size": int(size),
                    "r0": int(r0),
                    "c0": int(c0),
                    "is_edge": False,
                    "anchor": int(anchor),
                }
            )

        if _any_pairwise_overlap(entries):
            continue
        if forbidden is not None:
            overlap = False
            for entry in entries:
                rect_a = (int(entry["r0"]), int(entry["c0"]), int(entry["size"]))
                for other in forbidden:
                    rect_b = (int(other["r0"]), int(other["c0"]), int(other["size"]))
                    if _rects_overlap(rect_a, rect_b):
                        overlap = True
                        break
                if overlap:
                    break
            if overlap:
                continue
        if not any((not e["is_edge"]) and int(edge_sizes[int(e["anchor"])]) == 7 for e in entries):
            continue
        if 5 not in [int(e["size"]) for e in entries] or 7 not in [int(e["size"]) for e in entries]:
            continue
        return entries
    return None


def _place_output_and_colorize(entries: list[dict], bg: int) -> list[dict] | None:
    by_frame: dict[int, list[dict]] = {AZURE: [], RED: []}
    for entry in entries:
        by_frame[int(entry["frame"])].append(entry)

    for frame_color, group in by_frame.items():
        avail = [c for c in range(10) if c not in {bg, frame_color}]
        if len(avail) < len(group):
            return None
        colors = random.sample(avail, len(group))
        for entry, inner in zip(group, colors):
            patch = _make_patch(int(entry["size"]), frame_color, int(inner))
            entry["inner"] = int(inner)
            entry["patch"] = patch
    return entries


def _input_order(entries: list[dict], ascending: bool) -> list[dict]:
    edges = [e for e in entries if bool(e["is_edge"])]
    inwards = [e for e in entries if not bool(e["is_edge"])]

    edge_indices = list(range(len(edges)))
    random.shuffle(edge_indices)
    order: list[tuple[str, int]] = [("edge", idx) for idx in edge_indices]

    inward_indices = list(range(len(inwards)))
    random.shuffle(inward_indices)
    for iw_idx in inward_indices:
        anchor_idx = int(inwards[iw_idx]["anchor"])
        latest_pos = max(
            pos
            for pos, (kind, edge_idx) in enumerate(order)
            if kind == "edge" and edge_idx == anchor_idx
        )
        insert_pos = random.randint(latest_pos + 1, len(order))
        order.insert(insert_pos, ("inward", iw_idx))

    resolved = [edges[idx] if kind == "edge" else inwards[idx] for kind, idx in order]
    return resolved if ascending else resolved


def _derive_input_positions(
    entries: list[dict],
    ascending: bool,
    forbidden: list[dict] | None = None,
) -> list[dict] | None:
    ordered = _input_order(entries, ascending=ascending)
    for _ in range(INPUT_COL_RETRIES):
        placed: list[dict] = []
        ok = True
        for idx, base in enumerate(ordered):
            size = int(base["size"])
            out_c = int(base["c0"])
            row = int(base["r0"])

            if ascending:
                lower = 0 if idx == 0 else int(placed[-1]["c0"]) + 1
                for prev in placed:
                    if _rows_overlap(row, size, int(prev["r0"]), int(prev["size"])):
                        lower = max(lower, int(prev["c0"]) + int(prev["size"]))
                upper = GRID_W - size
                if lower > upper:
                    ok = False
                    break
                candidates = [c for c in range(lower, upper + 1) if c != out_c]
            else:
                upper = GRID_W - size if idx == 0 else int(placed[-1]["c0"]) - 1
                for prev in placed:
                    if _rows_overlap(row, size, int(prev["r0"]), int(prev["size"])):
                        upper = min(upper, int(prev["c0"]) - size)
                lower = 0
                if lower > upper:
                    ok = False
                    break
                candidates = [c for c in range(lower, upper + 1) if c != out_c]

            if forbidden is not None:
                filtered = []
                for c0 in candidates:
                    rect_a = (row, int(c0), size)
                    if all(
                        not _rects_overlap(rect_a, (int(other["r0"]), int(other["c0"]), int(other["size"])))
                        for other in forbidden
                    ):
                        filtered.append(int(c0))
                candidates = filtered

            if not candidates:
                ok = False
                break

            c0 = random.choice(candidates)
            placed.append(
                {
                    **base,
                    "c0": int(c0),
                }
            )

        if ok and not _any_pairwise_overlap(placed):
            return placed
    return None


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice([c for c in range(10) if c not in FRAME_COLORS])

    teal_entries = _build_side_layout(AZURE, N_AZURE_RANGE, side="left")
    if teal_entries is None:
        return None
    red_entries = _build_side_layout(RED, N_RED_RANGE, side="right", forbidden=teal_entries)
    if red_entries is None:
        return None

    output_entries = teal_entries + red_entries
    if _any_pairwise_overlap(output_entries):
        return None
    if 5 not in [int(e["size"]) for e in output_entries] or 7 not in [int(e["size"]) for e in output_entries]:
        return None
    if not any((not e["is_edge"]) and ((e["frame"] == AZURE and int(e["c0"]) == 7) or (e["frame"] == RED and int(e["size"]) + int(e["c0"]) == GRID_W - 7)) for e in output_entries):
        return None

    colored_entries = _place_output_and_colorize(output_entries, bg)
    if colored_entries is None:
        return None

    output = np.full((GRID_H, GRID_W), bg, dtype=int)
    for entry in colored_entries:
        paste(output, entry["patch"], int(entry["r0"]), int(entry["c0"]))

    teal_input = _derive_input_positions([e for e in colored_entries if int(e["frame"]) == AZURE], ascending=True)
    if teal_input is None:
        return None
    red_input = _derive_input_positions(
        [e for e in colored_entries if int(e["frame"]) == RED],
        ascending=False,
        forbidden=teal_input,
    )
    if red_input is None:
        return None

    input_entries = teal_input + red_input
    if _any_pairwise_overlap(input_entries):
        return None

    inp = np.full((GRID_H, GRID_W), bg, dtype=int)
    for entry in input_entries:
        paste(inp, entry["patch"], int(entry["r0"]), int(entry["c0"]))

    try:
        if np.array_equal(main(inp), output):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
