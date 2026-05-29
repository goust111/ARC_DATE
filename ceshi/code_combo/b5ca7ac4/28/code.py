# Combo variant of b5ca7ac4 (index 28)
# combo_spec: D3/0+D6/0+D11/1
# Name: 可变尺寸+近邻检测+间隙堆叠

from __future__ import annotations

import random

import numpy as np

from arc_utils import crop_bbox, get_bg, paste


# Design analysis for combo D3/0 + D6/0 + D11/1
# a) Concrete behavior:
#    - The source objects remain hollow square frames with a 1-cell border in teal (8) or red (2),
#      but the generator now mixes 5x5 and 7x7 frames. Rows are still preserved during motion.
#    - The old task moved a later same-side frame inward only when row spans truly overlapped, and the
#      inward offset was exactly one 5-cell frame width. The fused variant changes both terms:
#      a frame also moves inward when the vertical gap to an already claimed edge frame is exactly 1 row,
#      and the inward start is placed after the edge frame plus one blank column.
#    - Example on the teal side: an edge 7x7 at rows 2..8 occupies cols 0..6. A later 5x5 at rows
#      10..14 has row-gap 1 from that edge frame, so it still moves inward to cols 8..12. Under the
#      original task it would have stayed at the edge; under D3 alone it would still stay at the edge;
#      under D6 alone the shift would not show because the trigger would never happen. The combo only
#      becomes visible when all three pieces interact.
# b) Complexity calibration:
#    - Sizes are restricted to 5 and 7. That is enough to make width-sensitive offsets visible while
#      keeping the hypothesis space small enough for ARC-style inference.
#    - Every generated board contains both sizes, and at least one inward frame is a non-overlapping
#      near-neighbor of a 7x7 edge frame. This prevents degeneration back to the original overlap-only,
#      fixed-width behavior.
# c) Implicit assumption audit:
#    - The original fixed 5x5 detector is invalid once frame sizes vary, so detection must scan odd
#      square windows and validate a uniform border plus uniform non-bg interior.
#    - The original overlap predicate `abs(r_a-r_b) < 5` is also invalid. The new trigger condition is
#      based on inclusive row-span gap, with threshold <= 1, and each frame uses its own height.
#    - Hard-coded inward columns (5 and W-10) are no longer correct. The new inward columns depend on
#      the triggering edge frame's placed width and add one blank column between layers.
#    - The solver still assumes depth 2: only frames that successfully claimed the outer edge act as
#      anchors for later frames. The generator explicitly enforces that assumption.
# d) Placement legality:
#    - Input frames are non-overlapping and in-bounds.
#    - Same-side edge frames are generated with row-gap >= 2, so they never trigger each other and can
#      all remain on the outer edge.
#    - Each inward frame is generated to trigger exactly one edge anchor, never multiple anchors, and
#      the final output is rejected if any rectangles overlap across or within sides.
# e) Visual observability:
#    - All cues are visible: border color chooses side, original column chooses processing order, frame
#      width changes horizontal offset, and a 1-row vertical gap still causes an inward move.
#    - Because each board includes a 7-wide near-neighbor case, a human can directly notice both that
#      “touching is not required” and that “the second lane starts one column past the anchor frame.”
# f) Visual cleanliness:
#    - The generator forbids rectangle overlap in both input and output and limits stacking to two
#      lanes per side. That keeps the side stacks legible even with mixed 5x5 and 7x7 frames.
#    - Cross-side output collisions are also rejected so the middle of the board does not become noisy.
# g) Change inferability:
#    - No recoloring rule is added. Inner fills remain uniform identifiers, so the only changed evidence
#      is geometric: size, row-gap, and the visible 1-column buffer between stack lanes.


AZURE = 8
RED = 2
FRAME_COLORS = {AZURE, RED}
GRID_H = 22
GRID_W = 22
SIZES = (5, 7)
STACK_GAP = 1
NEAR_ROW_GAP = 1
N_AZURE_RANGE = (3, 4)
N_RED_RANGE = (3, 5)
MAX_INWARD = 2
SIDE_RETRIES = 300
INPUT_COL_RETRIES = 220
GEN_RETRIES = 1400


def _frame_border(patch: np.ndarray) -> np.ndarray:
    return np.concatenate([patch[0, :], patch[-1, :], patch[1:-1, 0], patch[1:-1, -1]])


def _make_patch(size: int, frame_color: int, inner_color: int) -> np.ndarray:
    patch = np.full((size, size), frame_color, dtype=int)
    patch[1:-1, 1:-1] = inner_color
    return patch


def _row_gap(r1: int, h1: int, r2: int, h2: int) -> int:
    a0 = int(r1)
    a1 = int(r1 + h1 - 1)
    b0 = int(r2)
    b1 = int(r2 + h2 - 1)
    if a1 < b0:
        return int(b0 - a1 - 1)
    if b1 < a0:
        return int(a0 - b1 - 1)
    return 0


def _rows_overlap(r1: int, h1: int, r2: int, h2: int) -> bool:
    return _row_gap(r1, h1, r2, h2) == 0


def _rows_trigger(r1: int, h1: int, r2: int, h2: int) -> bool:
    return _row_gap(r1, h1, r2, h2) <= NEAR_ROW_GAP


def _rects_overlap(a: tuple[int, int, int], b: tuple[int, int, int]) -> bool:
    ar, ac, asz = a
    br, bc, bsz = b
    if not _rows_overlap(ar, asz, br, bsz):
        return False
    return not (int(ac + asz - 1) < int(bc) or int(bc + bsz - 1) < int(ac))


def _any_pairwise_overlap(entries: list[dict]) -> bool:
    rects = [(int(e["r0"]), int(e["c0"]), int(e["size"])) for e in entries]
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            if _rects_overlap(rects[i], rects[j]):
                return True
    return False


def _detect_frames(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    found: list[dict] = []
    seen: set[tuple[int, int, int]] = set()

    for size in range(5, min(H, W) + 1, 2):
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
                if inner_col in {bg, frame_col}:
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


def _plan_group(frames: list[dict], side: str, width: int) -> list[dict]:
    if side == "left":
        ordered = sorted(frames, key=lambda f: int(f["c0"]))
    else:
        ordered = sorted(frames, key=lambda f: -int(f["c0"]))

    edge_anchors: list[dict] = []
    planned: list[dict] = []
    for frame in ordered:
        r0 = int(frame["r0"])
        size = int(frame["size"])
        triggers = [
            anchor
            for anchor in edge_anchors
            if _rows_trigger(r0, size, int(anchor["r0"]), int(anchor["size"]))
        ]
        if side == "left":
            dest_c = (
                max(int(anchor["c0"]) + int(anchor["size"]) for anchor in triggers) + STACK_GAP
                if triggers
                else 0
            )
        else:
            dest_c = (
                min(int(anchor["c0"]) for anchor in triggers) - STACK_GAP - size
                if triggers
                else int(width - size)
            )

        placed = {
            **frame,
            "dest_c": int(dest_c),
            "is_edge": not triggers,
            "triggers": [dict(anchor) for anchor in triggers],
        }
        planned.append(placed)
        if not triggers:
            edge_anchors.append({"r0": r0, "size": size, "c0": int(dest_c)})
    return planned


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move mixed-size bordered frames toward the side indicated by their border color.

    Input structure:
    - The background is the most frequent color.
    - Teal (8) and red (2) hollow square frames are scattered across the board.
    - Each frame still has a 1-cell border and a uniform filled interior, but the side length is
      no longer fixed: the variant mixes 5x5 and 7x7 frames.

    Rule:
    - Teal frames go to the left edge; red frames go to the right edge, while preserving their
      original top row.
    - Processing order still comes from the input columns: teal from smaller columns to larger
      ones, red from larger columns to smaller ones.
    - A frame takes the outer edge lane unless its row span either overlaps an already placed edge
      frame on the same side or is separated from it by only one blank row.
    - When that trigger happens, the frame shifts inward far enough to clear the triggering edge
      frame and leaves a 1-column background gap, so the offset depends on the anchor width.

    Key insight: Border color chooses the destination side, input-column order decides which frames
    claim the edge first, and the inward lane is controlled jointly by frame size, 1-row proximity,
    and the explicit 1-column gap between stacked lanes.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    out = np.full((H, W), bg, dtype=int)

    frames = _detect_frames(g, bg)
    teal_plan = _plan_group([f for f in frames if int(f["frame"]) == AZURE], "left", W)
    red_plan = _plan_group([f for f in frames if int(f["frame"]) == RED], "right", W)

    for placed in teal_plan + red_plan:
        paste(out, placed["patch"], int(placed["r0"]), int(placed["dest_c"]), transparent=bg)

    return out


def _sample_side_specs(n: int, force_special: bool) -> tuple[list[int], list[int], list[int], list[str]]:
    n_inward = random.randint(1, min(MAX_INWARD, n - 1))
    n_edge = n - n_inward

    edge_sizes = [random.choice(SIZES) for _ in range(n_edge)]
    inward_sizes = [random.choice(SIZES) for _ in range(n_inward)]
    anchors = [random.randrange(n_edge) for _ in range(n_inward)]
    relations = [random.choice(["overlap", "near"]) for _ in range(n_inward)]

    if force_special:
        special_idx = random.randrange(n_inward)
        special_anchor = random.randrange(n_edge)
        edge_sizes[special_anchor] = 7
        anchors[special_idx] = special_anchor
        relations[special_idx] = "near"
        if 5 not in edge_sizes + inward_sizes:
            inward_sizes[special_idx] = 5

    return edge_sizes, inward_sizes, anchors, relations


def _assign_edge_rows(sizes: list[int]) -> list[int] | None:
    rows = [None] * len(sizes)
    order = list(range(len(sizes)))
    random.shuffle(order)
    placed: list[tuple[int, int]] = []
    for idx in order:
        size = int(sizes[idx])
        candidates = list(range(GRID_H - size + 1))
        random.shuffle(candidates)
        picked = None
        for r0 in candidates:
            if all(_row_gap(r0, size, pr, psz) > NEAR_ROW_GAP for pr, psz in placed):
                picked = int(r0)
                break
        if picked is None:
            return None
        rows[idx] = picked
        placed.append((picked, size))
    return [int(r0) for r0 in rows]


def _assign_inward_rows(
    edge_rows: list[int],
    edge_sizes: list[int],
    inward_sizes: list[int],
    anchors: list[int],
    relations: list[str],
) -> list[int] | None:
    rows: list[int] = []
    for idx, (size, anchor, relation) in enumerate(zip(inward_sizes, anchors, relations)):
        candidates = list(range(GRID_H - size + 1))
        random.shuffle(candidates)
        picked = None
        for r0 in candidates:
            gap = _row_gap(r0, size, edge_rows[anchor], edge_sizes[anchor])
            if relation == "near":
                if gap != 1:
                    continue
            else:
                if gap != 0:
                    continue

            if any(
                _rows_trigger(r0, size, er, esz)
                for e_idx, (er, esz) in enumerate(zip(edge_rows, edge_sizes))
                if e_idx != anchor
            ):
                continue
            if any(_rows_overlap(r0, size, ir, inward_sizes[j]) for j, ir in enumerate(rows)):
                continue
            picked = int(r0)
            break
        if picked is None:
            return None
        rows.append(picked)
    return rows


def _entry_dest_col(side: str, edge_size: int, moving_size: int, is_edge: bool) -> int:
    if side == "left":
        return 0 if is_edge else int(edge_size + STACK_GAP)
    return int(GRID_W - moving_size) if is_edge else int(GRID_W - edge_size - STACK_GAP - moving_size)


def _build_side_layout(
    frame_color: int,
    count_range: tuple[int, int],
    side: str,
    *,
    force_special: bool,
    forbidden: list[dict] | None = None,
) -> list[dict] | None:
    n = random.randint(*count_range)
    for _ in range(SIDE_RETRIES):
        edge_sizes, inward_sizes, anchors, relations = _sample_side_specs(n, force_special=force_special)
        edge_rows = _assign_edge_rows(edge_sizes)
        if edge_rows is None:
            continue
        inward_rows = _assign_inward_rows(edge_rows, edge_sizes, inward_sizes, anchors, relations)
        if inward_rows is None:
            continue

        entries: list[dict] = []
        for idx, (r0, size) in enumerate(zip(edge_rows, edge_sizes)):
            entries.append(
                {
                    "frame": frame_color,
                    "size": int(size),
                    "r0": int(r0),
                    "c0": _entry_dest_col(side, int(size), int(size), True),
                    "is_edge": True,
                    "anchor": idx,
                    "relation": "edge",
                }
            )

        for idx, (r0, size, anchor, relation) in enumerate(zip(inward_rows, inward_sizes, anchors, relations)):
            entries.append(
                {
                    "frame": frame_color,
                    "size": int(size),
                    "r0": int(r0),
                    "c0": _entry_dest_col(side, int(edge_sizes[anchor]), int(size), False),
                    "is_edge": False,
                    "anchor": int(anchor),
                    "relation": relation,
                }
            )

        if _any_pairwise_overlap(entries):
            continue
        if forbidden is not None:
            bad = False
            for entry in entries:
                rect_a = (int(entry["r0"]), int(entry["c0"]), int(entry["size"]))
                for other in forbidden:
                    rect_b = (int(other["r0"]), int(other["c0"]), int(other["size"]))
                    if _rects_overlap(rect_a, rect_b):
                        bad = True
                        break
                if bad:
                    break
            if bad:
                continue

        if force_special:
            if not any(
                (not bool(e["is_edge"]))
                and e["relation"] == "near"
                and int(edge_sizes[int(e["anchor"])]) == 7
                for e in entries
            ):
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
            entry["inner"] = int(inner)
            entry["patch"] = _make_patch(int(entry["size"]), frame_color, int(inner))
    return entries


def _build_input_order(entries: list[dict]) -> list[dict]:
    edges = [e for e in entries if bool(e["is_edge"])]
    inwards = [e for e in entries if not bool(e["is_edge"])]

    edge_indices = list(range(len(edges)))
    random.shuffle(edge_indices)
    order: list[tuple[str, int]] = [("edge", idx) for idx in edge_indices]

    inward_indices = list(range(len(inwards)))
    random.shuffle(inward_indices)
    for iw_idx in inward_indices:
        anchor_idx = int(inwards[iw_idx]["anchor"])
        latest_anchor_pos = max(
            pos
            for pos, (kind, idx) in enumerate(order)
            if kind == "edge" and idx == anchor_idx
        )
        insert_pos = random.randint(latest_anchor_pos + 1, len(order))
        order.insert(insert_pos, ("inward", iw_idx))

    return [edges[idx] if kind == "edge" else inwards[idx] for kind, idx in order]


def _derive_input_positions(
    entries: list[dict],
    *,
    ascending: bool,
    forbidden: list[dict] | None = None,
) -> list[dict] | None:
    ordered = _build_input_order(entries)
    for _ in range(INPUT_COL_RETRIES):
        placed: list[dict] = []
        ok = True
        for idx, base in enumerate(ordered):
            row = int(base["r0"])
            size = int(base["size"])
            out_c = int(base["c0"])

            if ascending:
                lower = 0 if idx == 0 else int(placed[-1]["c0"]) + 1
                for prev in placed:
                    if _rows_overlap(row, size, int(prev["r0"]), int(prev["size"])):
                        lower = max(lower, int(prev["c0"]) + int(prev["size"]))
                upper = GRID_W - size
                if lower > upper:
                    ok = False
                    break
                candidates = [c0 for c0 in range(lower, upper + 1) if int(c0) != out_c]
            else:
                upper = GRID_W - size if idx == 0 else int(placed[-1]["c0"]) - 1
                for prev in placed:
                    if _rows_overlap(row, size, int(prev["r0"]), int(prev["size"])):
                        upper = min(upper, int(prev["c0"]) - size)
                lower = 0
                if lower > upper:
                    ok = False
                    break
                candidates = [c0 for c0 in range(lower, upper + 1) if int(c0) != out_c]

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
            placed.append({**base, "c0": int(c0)})

        if ok and not _any_pairwise_overlap(placed):
            return placed
    return None


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice([c for c in range(10) if c not in FRAME_COLORS])

    teal_entries = _build_side_layout(AZURE, N_AZURE_RANGE, "left", force_special=True)
    if teal_entries is None:
        return None
    red_entries = _build_side_layout(RED, N_RED_RANGE, "right", force_special=False, forbidden=teal_entries)
    if red_entries is None:
        return None

    output_entries = teal_entries + red_entries
    if _any_pairwise_overlap(output_entries):
        return None
    if 5 not in [int(e["size"]) for e in output_entries] or 7 not in [int(e["size"]) for e in output_entries]:
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
    raise RuntimeError("Failed to generate valid input for b5ca7ac4 combo D3/0+D6/0+D11/1")
