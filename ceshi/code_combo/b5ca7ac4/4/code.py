# Combo variant of b5ca7ac4 (index 4)
# combo_spec: D3/0+D6/1
# Name: 可变尺寸框架+共享边框堆叠

from __future__ import annotations

import random

import numpy as np

from arc_utils import crop_bbox, get_bg, paste


# Design analysis for the combo variant
# a) Concrete behavior:
#    - The original task had only 5x5 frames and a fixed one-frame inward shift. Here every object is
#      still a square frame with a 1-cell border in teal (8) or red (2), but the size can be 5x5 or
#      7x7. The output still pushes teal to the left and red to the right while preserving top rows
#      and sorting by input column (teal ascending, red descending).
#    - The changed stacking rule is "shared border" instead of "leave a full gap for the next frame".
#      Example on the left: if a 7x7 teal edge frame starts at col 0, an overlapping later frame does
#      not start at col 7; it starts at col 6 so the two frames share one border column. For a 5x5
#      edge frame the inward start is col 4. On the right, an inward frame of width 5 anchored to a
#      7x7 edge frame starts at W-7-5+1 instead of W-7-5.
#    - So the effective inward offset is no longer a constant 5 and no longer the raw anchor width;
#      it is (anchor width - 1), giving the visible pair of offsets 4 and 6.
# b) Complexity calibration:
#    - Only two frame sizes are used: 5 and 7. This is enough to force size reasoning without making
#      the task explode into too many possible widths.
#    - The generator forces one inward stack anchored by a 5-wide edge frame and another anchored by a
#      7-wide edge frame, so both shared-border offsets (4 and 6) appear repeatedly in the samples.
# c) Implicit assumption audit:
#    - The original 5x5-only scanner is invalid now; detection must scan both supported sizes.
#    - Hard-coded destination columns 0/5 and W-5/W-10 are invalid; inward placement now depends on
#      the overlapped edge width and subtracts one extra column because of the shared border.
#    - Row-overlap checks must use each frame's own height. The generator therefore forbids an inward
#      frame from overlapping multiple edge frames on the same side, keeping the solver's anchor
#      inference unambiguous even though the solver also has a safe "widest overlap wins" fallback.
# d) Placement legality:
#    - Input frames are generated in-bounds and with no rectangle overlap at all.
#    - Output frames on the same side may overlap only on a single shared border column between an
#      edge frame and the specific inward frame attached to it. No other overlap is allowed.
#    - Cross-side output overlap is rejected entirely, so the middle of the grid stays readable.
# e) Visual observability:
#    - All cues are visible: border color chooses the side, original column chooses processing order,
#      row overlap decides whether a frame stacks inward, and the amount of inward movement can be read
#      directly from the visible width of the edge frame it touches.
#    - Because every sample contains both 5x5 and 7x7 objects plus both offset values 4 and 6, a
#      human can infer that the second layer shares the border and that the offset depends on size.
# f) Visual cleanliness:
#    - The generator keeps only two stack depths per side, bans ambiguous multi-anchor overlaps, and
#      keeps the input scattered. That prevents the shared-border output from degenerating into a blob.
# g) Color inference:
#    - No recoloring rule is introduced. Inner colors remain uniform identity markers, and the
#      generator avoids using frame colors as inner fills so shared borders stay visually unambiguous.


AZURE = 8
RED = 2
FRAME_COLORS = {AZURE, RED}
GRID_H = 22
GRID_W = 22
SIZES = (5, 7)
N_AZURE_RANGE = (3, 4)
N_RED_RANGE = (3, 5)
MAX_INWARD = 2
SIDE_RETRIES = 250
INPUT_COL_RETRIES = 180
GEN_RETRIES = 1200


def _rows_overlap(r1: int, h1: int, r2: int, h2: int) -> bool:
    return not (int(r1 + h1 - 1) < int(r2) or int(r2 + h2 - 1) < int(r1))


def _rects_overlap(a: tuple[int, int, int], b: tuple[int, int, int]) -> bool:
    ar, ac, asz = a
    br, bc, bsz = b
    return _rows_overlap(ar, asz, br, bsz) and not (
        int(ac + asz - 1) < int(bc) or int(bc + bsz - 1) < int(ac)
    )


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

    for size in SIZES:
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
    Move mixed-size square frames to the side selected by their border color.

    Input structure:
    - The background is the most common color.
    - Teal (8) and red (2) hollow square frames are scattered around the grid.
    - Every frame has a 1-cell border and a uniform interior, but the side length can be either
      5 or 7 instead of being fixed.

    Rule:
    - Teal frames move to the left edge; red frames move to the right edge.
    - Each frame keeps its original top row, and the processing order still comes from the original
      column: teal ascending, red descending.
    - A frame with no row-overlap against an already placed edge frame sits flush on the edge.
    - If it row-overlaps an edge frame on its side, it moves inward by one less than that edge
      frame's width, so adjacent frames share a border column instead of leaving a full extra gap.

    Key insight: Border color chooses the destination side, input-column order decides which frames
    claim the outer edge first, and the visible width of the overlapped edge frame sets the shared-
    border inward offset: 4 for a 5x5 anchor and 6 for a 7x7 anchor.
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
        inward_shift = max((int(p["size"]) - 1 for p in overlaps), default=0)
        dest_c = int(inward_shift)
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
        inward_shift = max((int(p["size"]) - 1 for p in overlaps), default=0)
        dest_c = int(W - size - inward_shift)
        paste(out, frame["patch"], r0, dest_c, transparent=bg)
        if not overlaps:
            edge_red.append({"r0": r0, "size": size})

    return out


def _entry_rect(entry: dict) -> tuple[int, int, int]:
    return int(entry["r0"]), int(entry["c0"]), int(entry["size"])


def _entries_compatible(a: dict, b: dict) -> bool:
    if not _rects_overlap(_entry_rect(a), _entry_rect(b)):
        return True
    if int(a["frame"]) != int(b["frame"]) or str(a["side"]) != str(b["side"]):
        return False
    if bool(a["is_edge"]) == bool(b["is_edge"]):
        return False

    edge = a if bool(a["is_edge"]) else b
    inward = b if bool(a["is_edge"]) else a
    if int(inward["anchor"]) != int(edge["edge_idx"]):
        return False

    overlap_c0 = max(int(a["c0"]), int(b["c0"]))
    overlap_c1 = min(int(a["c0"]) + int(a["size"]) - 1, int(b["c0"]) + int(b["size"]) - 1)
    if overlap_c0 != overlap_c1:
        return False

    if str(edge["side"]) == "left":
        return overlap_c0 == int(edge["c0"]) + int(edge["size"]) - 1 == int(inward["c0"])
    return overlap_c0 == int(edge["c0"]) == int(inward["c0"]) + int(inward["size"]) - 1


def _layout_is_valid(entries: list[dict]) -> bool:
    for i in range(len(entries)):
        for j in range(i + 1, len(entries)):
            if not _entries_compatible(entries[i], entries[j]):
                return False
    return True


def _render_entries(entries: list[dict], bg: int) -> np.ndarray | None:
    out = np.full((GRID_H, GRID_W), bg, dtype=int)
    owner = np.full((GRID_H, GRID_W), -1, dtype=int)

    for idx, entry in enumerate(entries):
        patch = entry["patch"]
        size = int(entry["size"])
        r0 = int(entry["r0"])
        c0 = int(entry["c0"])
        if r0 < 0 or c0 < 0 or r0 + size > GRID_H or c0 + size > GRID_W:
            return None
        for dr in range(size):
            for dc in range(size):
                rr = r0 + dr
                cc = c0 + dc
                if owner[rr, cc] == -1:
                    continue
                other = entries[int(owner[rr, cc])]
                if not _entries_compatible(entry, other):
                    return None
                if int(out[rr, cc]) != int(patch[dr, dc]):
                    return None
        paste(out, patch, r0, c0)
        for dr in range(size):
            for dc in range(size):
                rr = r0 + dr
                cc = c0 + dc
                owner[rr, cc] = idx
    return out


def _any_input_overlap(entries: list[dict]) -> bool:
    rects = [_entry_rect(entry) for entry in entries]
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            if _rects_overlap(rects[i], rects[j]):
                return True
    return False


def _sample_side_sizes(
    n_edge: int,
    n_inward: int,
    required_anchor_size: int,
) -> tuple[list[int], list[int], list[int]]:
    edge_sizes = [random.choice(SIZES) for _ in range(n_edge)]
    inward_sizes = [random.choice(SIZES) for _ in range(n_inward)]

    if required_anchor_size not in edge_sizes:
        edge_sizes[random.randrange(n_edge)] = int(required_anchor_size)

    anchor_choices = [idx for idx, size in enumerate(edge_sizes) if int(size) == int(required_anchor_size)]
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
            if any(
                edge_idx != anchor and _rows_overlap(r0, size, er, esz)
                for edge_idx, (er, esz) in enumerate(zip(edge_rows, edge_sizes))
            ):
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
    required_anchor_size: int,
    forbidden: list[dict] | None = None,
) -> list[dict] | None:
    n = random.randint(*count_range)
    n_inward = random.randint(1, min(MAX_INWARD, n - 1))
    n_edge = n - n_inward

    for _ in range(SIDE_RETRIES):
        edge_sizes, inward_sizes, anchors = _sample_side_sizes(n_edge, n_inward, required_anchor_size)
        edge_rows = _assign_edge_rows(edge_sizes)
        if edge_rows is None:
            continue
        inward_rows = _assign_inward_rows(edge_rows, edge_sizes, inward_sizes, anchors)
        if inward_rows is None:
            continue

        entries: list[dict] = []
        for edge_idx, (r0, size) in enumerate(zip(edge_rows, edge_sizes)):
            c0 = 0 if side == "left" else GRID_W - size
            entries.append(
                {
                    "frame": frame_color,
                    "side": side,
                    "size": int(size),
                    "r0": int(r0),
                    "c0": int(c0),
                    "is_edge": True,
                    "edge_idx": int(edge_idx),
                    "anchor": int(edge_idx),
                }
            )

        for idx, (r0, size, anchor) in enumerate(zip(inward_rows, inward_sizes, anchors)):
            anchor_size = int(edge_sizes[anchor])
            if side == "left":
                c0 = anchor_size - 1
            else:
                c0 = GRID_W - anchor_size - size + 1
            entries.append(
                {
                    "frame": frame_color,
                    "side": side,
                    "size": int(size),
                    "r0": int(r0),
                    "c0": int(c0),
                    "is_edge": False,
                    "edge_idx": int(n_edge + idx),
                    "anchor": int(anchor),
                }
            )

        if not _layout_is_valid(entries):
            continue
        if forbidden is not None and not _layout_is_valid(entries + forbidden):
            continue
        return entries
    return None


def _place_output_and_colorize(entries: list[dict], bg: int) -> list[dict] | None:
    by_frame: dict[int, list[dict]] = {AZURE: [], RED: []}
    for entry in entries:
        by_frame[int(entry["frame"])].append(entry)

    for frame_color, group in by_frame.items():
        avail = [c for c in range(10) if c not in {bg, AZURE, RED, frame_color}]
        if len(avail) < len(group):
            return None
        colors = random.sample(avail, len(group))
        for entry, inner in zip(group, colors):
            patch = _make_patch(int(entry["size"]), frame_color, int(inner))
            entry["inner"] = int(inner)
            entry["patch"] = patch
    return entries


def _input_order(entries: list[dict]) -> list[dict]:
    edges = [entry for entry in entries if bool(entry["is_edge"])]
    inwards = [entry for entry in entries if not bool(entry["is_edge"])]

    edge_indices = list(range(len(edges)))
    random.shuffle(edge_indices)
    order: list[tuple[str, int]] = [("edge", idx) for idx in edge_indices]

    inward_indices = list(range(len(inwards)))
    random.shuffle(inward_indices)
    for inward_idx in inward_indices:
        anchor_idx = int(inwards[inward_idx]["anchor"])
        latest_pos = max(
            pos
            for pos, (kind, edge_idx) in enumerate(order)
            if kind == "edge" and edge_idx == anchor_idx
        )
        insert_pos = random.randint(latest_pos + 1, len(order))
        order.insert(insert_pos, ("inward", inward_idx))

    return [edges[idx] if kind == "edge" else inwards[idx] for kind, idx in order]


def _derive_input_positions(
    entries: list[dict],
    ascending: bool,
    forbidden: list[dict] | None = None,
) -> list[dict] | None:
    ordered = _input_order(entries)

    for _ in range(INPUT_COL_RETRIES):
        placed: list[dict] = []
        ok = True
        for idx, base in enumerate(ordered):
            size = int(base["size"])
            row = int(base["r0"])
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
                candidates = [c0 for c0 in range(lower, upper + 1) if c0 != out_c]
            else:
                upper = GRID_W - size if idx == 0 else int(placed[-1]["c0"]) - 1
                for prev in placed:
                    if _rows_overlap(row, size, int(prev["r0"]), int(prev["size"])):
                        upper = min(upper, int(prev["c0"]) - size)
                if 0 > upper:
                    ok = False
                    break
                candidates = [c0 for c0 in range(0, upper + 1) if c0 != out_c]

            if forbidden is not None:
                filtered: list[int] = []
                for c0 in candidates:
                    rect = (row, int(c0), size)
                    if all(not _rects_overlap(rect, _entry_rect(other)) for other in forbidden):
                        filtered.append(int(c0))
                candidates = filtered

            if not candidates:
                ok = False
                break

            c0 = random.choice(candidates)
            placed.append({**base, "c0": int(c0)})

        if ok and not _any_input_overlap(placed):
            return placed
    return None


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice([c for c in range(10) if c not in FRAME_COLORS])

    left_required = random.choice(SIZES)
    right_required = 7 if left_required == 5 else 5

    teal_entries = _build_side_layout(
        AZURE,
        N_AZURE_RANGE,
        side="left",
        required_anchor_size=left_required,
    )
    if teal_entries is None:
        return None

    red_entries = _build_side_layout(
        RED,
        N_RED_RANGE,
        side="right",
        required_anchor_size=right_required,
        forbidden=teal_entries,
    )
    if red_entries is None:
        return None

    output_entries = teal_entries + red_entries
    if not _layout_is_valid(output_entries):
        return None

    anchor_sizes = [
        int([e for e in output_entries if bool(e["is_edge"]) and int(e["edge_idx"]) == int(entry["anchor"]) and str(e["side"]) == str(entry["side"])][0]["size"])
        for entry in output_entries
        if not bool(entry["is_edge"])
    ]
    if 5 not in anchor_sizes or 7 not in anchor_sizes:
        return None

    sizes_present = [int(entry["size"]) for entry in output_entries]
    if 5 not in sizes_present or 7 not in sizes_present:
        return None

    colored_entries = _place_output_and_colorize(output_entries, bg)
    if colored_entries is None:
        return None

    output = _render_entries(colored_entries, bg)
    if output is None:
        return None

    teal_input = _derive_input_positions(
        [entry for entry in colored_entries if int(entry["frame"]) == AZURE],
        ascending=True,
    )
    if teal_input is None:
        return None

    red_input = _derive_input_positions(
        [entry for entry in colored_entries if int(entry["frame"]) == RED],
        ascending=False,
        forbidden=teal_input,
    )
    if red_input is None:
        return None

    input_entries = teal_input + red_input
    if _any_input_overlap(input_entries):
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
