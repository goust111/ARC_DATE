# Combo variant of b5ca7ac4 (index 35)
# combo_spec: D3/0+D6/1+D8/3
# Name: 边框合并+可变尺寸+共享边框

from __future__ import annotations

import random

import numpy as np

from arc_utils import crop_bbox, get_bg, paste


# Design analysis for the combo variant
# a) Concrete behavior:
#    - Input objects are still square frames identified by their INPUT border color: teal (8) frames belong
#      to the left-moving group and red (2) frames belong to the right-moving group.
#    - Unlike the base task, sizes are mixed: the generator uses both 5x5 and 7x7 frames. Each frame keeps
#      a 1-cell border and a uniform filled interior.
#    - The original stack rule had a fixed second-layer offset of one full 5-cell width. Here the offset is
#      "anchor width minus 1", so later overlapping frames share exactly one border column with the edge frame.
#      Example on the left: a 7x7 edge frame at col 0 and a later 5x5 overlapping frame move to col 6, not 7.
#      The two frames overlap on the edge frame's right border column, producing a connected jigsaw-like union.
#    - After placement, each moved frame swaps its colors: output border = original interior color, and output
#      interior = original border color. So a teal frame with inner color 3 becomes a frame with border 3 and
#      interior 8 in the output.
#    - Compared with the original task, three things change together at the same moment: the width that defines
#      the inward shift depends on the visible size, the shift is reduced by one cell to create a shared border,
#      and the visible colors change only after the frame reaches its destination.
# b) Complexity calibration:
#    - Only two sizes are used, 5 and 7, which is enough to expose variable-size reasoning without making the
#      hypothesis space too large.
#    - Every generated sample contains at least one mixed-size shared-border pair, specifically a 7-wide edge
#      anchor that receives a 5-wide inward frame. This guarantees the fusion is visible instead of degenerating
#      into "original stacking plus a color swap".
#    - The number of inward frames stays small (1-2 per side), so the output forms readable irregular clusters
#      instead of a dense pile of overlaps.
# c) Implicit assumption audit:
#    - The base solver's 5x5 scan is invalid once sizes vary, so detection must scan both 5x5 and 7x7 windows.
#    - Hard-coded destination columns 5 and W-10 are invalid once the offset depends on anchor width and shared
#      borders. The new columns are anchor_size - 1 on the left and W - anchor_size - size + 1 on the right.
#    - Transparent pasting is no longer a good default for output because shared-border fusion intentionally
#      overwrites one border column. The solver therefore uses opaque paste for moved objects.
#    - Output border colors are no longer the group colors 2/8, so grouping must be inferred only from the
#      input border colors, never from the output.
# d) Placement legality:
#    - Input frames are generated in-bounds and non-overlapping.
#    - On each side, edge frames have disjoint row spans. Each inward frame overlaps exactly one edge frame in
#      rows, which keeps the anchor for the shared-border shift unambiguous.
#    - Output-side overlap is allowed only for an inward frame and its own anchor edge frame, and only as a
#      one-column shared border. All other pairwise overlaps are rejected.
# e) Visual observability:
#    - A human can track the same object by row, see that teal still goes left and red still goes right, and
#      notice that overlapping objects now interlock by one shared column instead of leaving a full-width gap.
#    - Because at least one shared pair uses different sizes, the irregular silhouette is visible from shape
#      alone rather than from hidden metadata.
# f) Visual cleanliness:
#    - The generator keeps the intentional overlap limited to a single shared border column. No same-side pair
#      is allowed to overlap in interiors, and cross-side regions are kept disjoint.
#    - Interior colors stay uniform, so even when several frames share borders, each component remains legible.
# g) Color inference:
#    - Color change is structural rather than lookup-based: border and interior trade roles for every frame.
#    - The generator excludes bg/2/8 from interior colors so the swapped output border is clearly distinct from
#      both the background and the original grouping colors.
# 9) New implicit assumptions and generator support:
#    - main() assumes true objects are full square frames with uniform borders and uniform interiors; the
#      generator enforces that exactly.
#    - main() assumes any inward placement ambiguity can be resolved by the set of already placed edge frames;
#      the generator guarantees each inward frame row-overlaps exactly one edge anchor.
#    - main() pastes shared-border outputs opaquely; the generator builds the output with the same edge-then-
#      inward layering, so validation against main() is exact.


AZURE = 8
RED = 2
FRAME_COLORS = {AZURE, RED}
GRID_H = 22
GRID_W = 22
SIZES = (5, 7)
N_AZURE_RANGE = (3, 4)
N_RED_RANGE = (3, 5)
MAX_INWARD = 2
SIDE_RETRIES = 300
INPUT_COL_RETRIES = 220
GEN_RETRIES = 1200


def _rows_overlap(r1: int, h1: int, r2: int, h2: int) -> bool:
    return not (int(r1 + h1 - 1) < int(r2) or int(r2 + h2 - 1) < int(r1))


def _rect_intersection(a: tuple[int, int, int], b: tuple[int, int, int]) -> tuple[int, int]:
    ar, ac, asz = a
    br, bc, bsz = b
    row0 = max(int(ar), int(br))
    row1 = min(int(ar + asz - 1), int(br + bsz - 1))
    col0 = max(int(ac), int(bc))
    col1 = min(int(ac + asz - 1), int(bc + bsz - 1))
    if row0 > row1 or col0 > col1:
        return 0, 0
    return int(row1 - row0 + 1), int(col1 - col0 + 1)


def _rects_overlap(a: tuple[int, int, int], b: tuple[int, int, int]) -> bool:
    h, w = _rect_intersection(a, b)
    return h > 0 and w > 0


def _make_patch(size: int, frame_color: int, inner_color: int) -> np.ndarray:
    patch = np.full((int(size), int(size)), int(frame_color), dtype=int)
    patch[1:-1, 1:-1] = int(inner_color)
    return patch


def _swap_patch_colors(patch: np.ndarray) -> np.ndarray:
    border = np.concatenate([patch[0, :], patch[-1, :], patch[1:-1, 0], patch[1:-1, -1]])
    if int(border.min()) != int(border.max()):
        raise ValueError("patch border is not uniform")
    frame_color = int(border[0])

    inner = patch[1:-1, 1:-1]
    if int(inner.min()) != int(inner.max()):
        raise ValueError("patch interior is not uniform")
    inner_color = int(inner[0, 0])
    if inner_color == frame_color:
        raise ValueError("patch interior matches frame color")

    return _make_patch(int(patch.shape[0]), inner_color, frame_color)


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
                patch = crop_bbox(g, (int(r0), int(c0), int(r0 + size - 1), int(c0 + size - 1)))
                border = _frame_border(patch)
                if int(border.min()) != int(border.max()):
                    continue
                frame_color = int(border[0])
                if frame_color not in FRAME_COLORS:
                    continue

                inner = patch[1:-1, 1:-1]
                if inner.size == 0 or int(inner.min()) != int(inner.max()):
                    continue
                inner_color = int(inner[0, 0])
                if inner_color in {bg, frame_color}:
                    continue

                key = (int(r0), int(c0), int(size))
                if key in seen:
                    continue
                seen.add(key)
                found.append(
                    {
                        "frame": frame_color,
                        "inner": inner_color,
                        "r0": int(r0),
                        "c0": int(c0),
                        "size": int(size),
                        "patch": patch,
                    }
                )
    return found


def _output_col(side: str, is_edge: bool, anchor_size: int, size: int) -> int:
    if side == "left":
        return 0 if is_edge else int(anchor_size - 1)
    return int(GRID_W - size) if is_edge else int(GRID_W - anchor_size - size + 1)


def _shared_pair_ok(a: dict, b: dict) -> bool:
    if bool(a["is_edge"]) == bool(b["is_edge"]):
        return False

    inward = a if not bool(a["is_edge"]) else b
    edge = b if inward is a else a
    if int(inward["anchor"]) != int(edge["anchor_id"]):
        return False

    rect_a = (int(a["r0"]), int(a["c0"]), int(a["size"]))
    rect_b = (int(b["r0"]), int(b["c0"]), int(b["size"]))
    inter_h, inter_w = _rect_intersection(rect_a, rect_b)
    return inter_h > 0 and inter_w == 1


def _layout_is_valid(entries: list[dict]) -> bool:
    for i in range(len(entries)):
        for j in range(i + 1, len(entries)):
            a = entries[i]
            b = entries[j]
            rect_a = (int(a["r0"]), int(a["c0"]), int(a["size"]))
            rect_b = (int(b["r0"]), int(b["c0"]), int(b["size"]))
            if not _rects_overlap(rect_a, rect_b):
                continue
            if a["side"] == b["side"] and _shared_pair_ok(a, b):
                continue
            return False
    return True


def _sample_side_sizes(n_edge: int, n_inward: int, *, need_mixed_pair: bool) -> tuple[list[int], list[int], list[int]]:
    edge_sizes = [random.choice(SIZES) for _ in range(n_edge)]
    inward_sizes = [random.choice(SIZES) for _ in range(n_inward)]
    anchors = [random.randrange(n_edge) for _ in range(n_inward)]

    if need_mixed_pair:
        mix_anchor = random.randrange(n_edge)
        mix_inward = random.randrange(n_inward)
        edge_sizes[mix_anchor] = 7
        inward_sizes[mix_inward] = 5
        anchors[mix_inward] = mix_anchor

    return edge_sizes, inward_sizes, anchors


def _assign_edge_rows(sizes: list[int]) -> list[int] | None:
    order = list(range(len(sizes)))
    random.shuffle(order)
    rows: list[int | None] = [None] * len(sizes)
    placed: list[tuple[int, int]] = []

    for idx in order:
        size = int(sizes[idx])
        candidates = list(range(GRID_H - size + 1))
        random.shuffle(candidates)
        chosen = None
        for r0 in candidates:
            if all(not _rows_overlap(r0, size, pr, psz) for pr, psz in placed):
                chosen = int(r0)
                break
        if chosen is None:
            return None
        rows[idx] = chosen
        placed.append((chosen, size))

    return [int(r) for r in rows if r is not None]


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
        chosen = None
        for r0 in candidates:
            if not _rows_overlap(r0, size, edge_rows[anchor], edge_sizes[anchor]):
                continue
            if any(
                e_idx != anchor and _rows_overlap(r0, size, er, esz)
                for e_idx, (er, esz) in enumerate(zip(edge_rows, edge_sizes))
            ):
                continue
            if any(_rows_overlap(r0, size, ir, inward_sizes[j]) for j, ir in enumerate(inward_rows)):
                continue
            chosen = int(r0)
            break
        if chosen is None:
            return None
        inward_rows.append(chosen)
    return inward_rows


def _build_side_layout(
    frame_color: int,
    count_range: tuple[int, int],
    side: str,
    *,
    forbidden: list[dict] | None = None,
    need_mixed_pair: bool,
) -> list[dict] | None:
    n = random.randint(*count_range)
    n_inward = random.randint(1, min(MAX_INWARD, n - 1))
    n_edge = n - n_inward

    for _ in range(SIDE_RETRIES):
        edge_sizes, inward_sizes, anchors = _sample_side_sizes(
            n_edge, n_inward, need_mixed_pair=need_mixed_pair
        )
        edge_rows = _assign_edge_rows(edge_sizes)
        if edge_rows is None:
            continue
        inward_rows = _assign_inward_rows(edge_rows, edge_sizes, inward_sizes, anchors)
        if inward_rows is None:
            continue

        entries: list[dict] = []
        for idx, (r0, size) in enumerate(zip(edge_rows, edge_sizes)):
            entries.append(
                {
                    "side": side,
                    "frame": int(frame_color),
                    "r0": int(r0),
                    "c0": _output_col(side, True, int(size), int(size)),
                    "size": int(size),
                    "is_edge": True,
                    "anchor": idx,
                    "anchor_id": idx,
                }
            )
        for idx, (r0, size, anchor) in enumerate(zip(inward_rows, inward_sizes, anchors)):
            entries.append(
                {
                    "side": side,
                    "frame": int(frame_color),
                    "r0": int(r0),
                    "c0": _output_col(side, False, int(edge_sizes[anchor]), int(size)),
                    "size": int(size),
                    "is_edge": False,
                    "anchor": int(anchor),
                    "anchor_id": n_edge + idx,
                }
            )

        if not _layout_is_valid(entries):
            continue
        if forbidden is not None and not _layout_is_valid(forbidden + entries):
            continue
        if need_mixed_pair and not any(
            (not bool(e["is_edge"])) and int(e["size"]) != int(edge_sizes[int(e["anchor"])])
            for e in entries
        ):
            continue
        return entries

    return None


def _colorize_entries(entries: list[dict], bg: int) -> list[dict] | None:
    grouped: dict[int, list[dict]] = {AZURE: [], RED: []}
    for entry in entries:
        grouped[int(entry["frame"])].append(entry)

    for frame_color, group in grouped.items():
        avail = [c for c in range(10) if c not in {bg, AZURE, RED, frame_color}]
        if len(avail) < len(group):
            return None
        colors = random.sample(avail, len(group))
        for entry, inner_color in zip(group, colors):
            patch_in = _make_patch(int(entry["size"]), int(frame_color), int(inner_color))
            entry["inner"] = int(inner_color)
            entry["patch_in"] = patch_in
            entry["patch_out"] = _swap_patch_colors(patch_in)
    return entries


def _render_output(entries: list[dict], bg: int) -> np.ndarray:
    output = np.full((GRID_H, GRID_W), int(bg), dtype=int)
    for side in ("left", "right"):
        side_entries = [e for e in entries if e["side"] == side]
        for entry in side_entries:
            if bool(entry["is_edge"]):
                paste(output, entry["patch_out"], int(entry["r0"]), int(entry["c0"]))
        for entry in side_entries:
            if not bool(entry["is_edge"]):
                paste(output, entry["patch_out"], int(entry["r0"]), int(entry["c0"]))
    return output


def _derive_input_positions(
    entries: list[dict],
    *,
    ascending: bool,
    forbidden: list[dict] | None = None,
) -> list[dict] | None:
    edges = [e for e in entries if bool(e["is_edge"])]
    inwards = [e for e in entries if not bool(e["is_edge"])]

    for _ in range(INPUT_COL_RETRIES):
        edge_indices = list(range(len(edges)))
        random.shuffle(edge_indices)
        order: list[tuple[str, int]] = [("edge", idx) for idx in edge_indices]

        inward_indices = list(range(len(inwards)))
        random.shuffle(inward_indices)
        for iw_idx in inward_indices:
            anchor = int(inwards[iw_idx]["anchor"])
            latest_pos = max(
                pos
                for pos, (kind, idx) in enumerate(order)
                if kind == "edge" and idx == anchor
            )
            insert_pos = random.randint(latest_pos + 1, len(order))
            order.insert(insert_pos, ("inward", iw_idx))

        ordered = [edges[idx] if kind == "edge" else inwards[idx] for kind, idx in order]
        placed: list[dict] = []
        ok = True

        for idx, base in enumerate(ordered):
            row = int(base["r0"])
            size = int(base["size"])
            out_c = int(base["c0"])

            if ascending:
                lower = 0 if idx == 0 else int(placed[-1]["c0"]) + 1
                upper = GRID_W - size
                candidates = list(range(lower, upper + 1))
            else:
                upper = GRID_W - size if idx == 0 else int(placed[-1]["c0"]) - 1
                lower = 0
                candidates = list(range(lower, upper + 1))

            filtered: list[int] = []
            for c0 in candidates:
                if int(c0) == out_c:
                    continue
                rect = (row, int(c0), size)
                if any(
                    _rects_overlap(rect, (int(prev["r0"]), int(prev["c0"]), int(prev["size"])))
                    for prev in placed
                ):
                    continue
                if forbidden is not None and any(
                    _rects_overlap(rect, (int(other["r0"]), int(other["c0"]), int(other["size"])))
                    for other in forbidden
                ):
                    continue
                filtered.append(int(c0))

            if not filtered:
                ok = False
                break

            c0 = random.choice(filtered)
            placed.append({**base, "c0": int(c0)})

        if ok:
            return placed

    return None


def _has_mixed_shared_pair(entries: list[dict]) -> bool:
    edge_sizes = {
        (e["side"], int(e["anchor_id"])): int(e["size"])
        for e in entries
        if bool(e["is_edge"])
    }
    return any(
        (not bool(e["is_edge"])) and int(e["size"]) != edge_sizes[(e["side"], int(e["anchor"]))]
        for e in entries
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move mixed-size framed squares to side stacks, let overlapping placements share one border column, then swap frame and fill colors.

    Input structure:
    - The background is the most common color.
    - Teal (8) and red (2) square frames are scattered across the grid.
    - Frames still have a 1-cell border and a uniform filled interior, but the side length can now be either 5 or 7.

    Rule:
    - Teal-bordered frames move to the left side and red-bordered frames move to the right side, keeping their original top row.
    - Processing order still comes from the input column: teal ascending, red descending.
    - A frame with no row-overlap against already placed edge frames claims the outer edge. If it row-overlaps an edge frame, it moves inward by anchor_width - 1 instead of anchor_width, so the two frames share exactly one border column.
    - After choosing that destination, swap the frame's colors: new border = old interior, new interior = old border.

    Key insight:
    - Group membership is read only from the INPUT border color. The output combines three visible cues at once: variable frame sizes, one-column shared-border stacking, and a consistent border/interior color exchange after movement.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    out = np.full((H, W), bg, dtype=int)

    frames = _detect_frames(g, bg)

    teal_frames = sorted((f for f in frames if int(f["frame"]) == AZURE), key=lambda f: int(f["c0"]))
    edge_teal: list[dict] = []
    for frame in teal_frames:
        r0 = int(frame["r0"])
        size = int(frame["size"])
        overlaps = [
            edge for edge in edge_teal if _rows_overlap(r0, size, int(edge["r0"]), int(edge["size"]))
        ]
        anchor_width = max((int(edge["size"]) for edge in overlaps), default=0)
        dest_c = 0 if not overlaps else int(anchor_width - 1)
        paste(out, _swap_patch_colors(frame["patch"]), r0, dest_c)
        if not overlaps:
            edge_teal.append({"r0": r0, "size": size})

    red_frames = sorted((f for f in frames if int(f["frame"]) == RED), key=lambda f: -int(f["c0"]))
    edge_red: list[dict] = []
    for frame in red_frames:
        r0 = int(frame["r0"])
        size = int(frame["size"])
        overlaps = [
            edge for edge in edge_red if _rows_overlap(r0, size, int(edge["r0"]), int(edge["size"]))
        ]
        if overlaps:
            anchor_width = max(int(edge["size"]) for edge in overlaps)
            dest_c = int(W - anchor_width - size + 1)
        else:
            dest_c = int(W - size)
            edge_red.append({"r0": r0, "size": size})
        paste(out, _swap_patch_colors(frame["patch"]), r0, dest_c)

    return out


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice([c for c in range(10) if c not in FRAME_COLORS])
    mixed_side = random.choice(["left", "right"])

    teal_entries = _build_side_layout(
        AZURE,
        N_AZURE_RANGE,
        "left",
        need_mixed_pair=(mixed_side == "left"),
    )
    if teal_entries is None:
        return None

    red_entries = _build_side_layout(
        RED,
        N_RED_RANGE,
        "right",
        forbidden=teal_entries,
        need_mixed_pair=(mixed_side == "right"),
    )
    if red_entries is None:
        return None

    output_entries = teal_entries + red_entries
    if not _layout_is_valid(output_entries):
        return None
    if not _has_mixed_shared_pair(output_entries):
        return None
    if {int(e["size"]) for e in output_entries} != {5, 7}:
        return None

    colored_entries = _colorize_entries(output_entries, bg)
    if colored_entries is None:
        return None
    output = _render_output(colored_entries, bg)

    teal_input = _derive_input_positions(
        [e for e in colored_entries if int(e["frame"]) == AZURE],
        ascending=True,
    )
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
    if any(
        _rects_overlap(
            (int(a["r0"]), int(a["c0"]), int(a["size"])),
            (int(b["r0"]), int(b["c0"]), int(b["size"])),
        )
        for i, a in enumerate(input_entries)
        for b in input_entries[i + 1 :]
    ):
        return None

    inp = np.full((GRID_H, GRID_W), int(bg), dtype=int)
    for entry in input_entries:
        paste(inp, entry["patch_in"], int(entry["r0"]), int(entry["c0"]))

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
