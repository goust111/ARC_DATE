# Combo variant of edb79dae (index 19)
# combo_spec: D9/2+D11/0
# Name: 按颜色分割输出+多框架

from __future__ import annotations

import itertools
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


# ---------------------------------------------------------------------------
# Design analysis for the combo variant
#
# a) Concrete behavior and trace
#    - Input now has 2-3 separate grey frames, each containing a tiny 2-column
#      grid of solid 3x3 placeholder blocks. Outside the whole framed region are
#      one sprite template per source color and one legend glyph per source color.
#    - Original task flow:
#      one frame -> gather its placeholder colors -> find same-color external
#      sprite -> recolor by legend -> stamp everything into one output crop.
#    - Fused flow:
#      first find every frame, then recover which source colors live in which
#      frame, then read all global legends, and finally group the recolored
#      results by target color across all frames.
#    - Small trace:
#      frame A contains red and blue placeholders, frame B contains green and
#      yellow placeholders. Suppose legends say red->orange, blue->cyan,
#      green->orange, yellow->cyan. The old multi-frame-only behavior would draw
#      one union crop with orange+cyan sprites mixed together. The fused output
#      instead becomes [orange panel with both frames populated only where
#      red/green belonged] |5| [cyan panel with both frames populated only where
#      blue/yellow belonged]. Cross-frame aggregation is therefore essential.
#
# b) Complexity calibration
#    - D9 splitting multiplies output width, so every source frame is forced to
#      a compact 9-column footprint. Frames are stacked vertically, keeping the
#      shared union width fixed at 9; this allows 2 or 3 output panels while
#      staying inside ARC's 30-column limit.
#    - The generator uses 3-5 source colors but only 2-3 distinct target colors.
#      This guarantees real grouping rather than trivial "one legend -> one
#      panel", and it creates repeated evidence for the grouping rule.
#    - At least one target color is assigned to source colors from different
#      frames. Without that, the combo would collapse into independent per-frame
#      reasoning and would not visibly require cross-frame aggregation.
#
# c) Hidden assumptions audit
#    - The original `_find_frame(...)` assumption is invalid; the solver must
#      enumerate 2-3 rectangular frames with the same grey border color.
#    - Placeholder colors can no longer be read from one crop. The solver now
#      aggregates local colors across frames while remembering each color's owner
#      frame. The generator guarantees each source color belongs to exactly one
#      frame so this ownership is inferable from the input itself.
#    - The old output allocation `out = crop.copy()` is also invalid. The fused
#      solver first builds a base union-layout panel for every distinct target
#      color, then stamps only the matching recolored sprites into that panel.
#    - Panel order cannot depend on numeric color ids. The solver derives panel
#      order from the visible reading order of legend glyphs, grouped by first
#      occurrence of each target color. The generator makes this stable by
#      placing legends as a single vertical strip.
#
# d) Placement legality
#    - Frames are packed into one vertical column with fixed 2-cell background
#      gaps, so they never overlap and the union box is easy to crop.
#    - The generator reserves the whole union box before placing any external
#      patch, keeping all sprites and legends outside the framed area.
#    - Each source color gets exactly one external sprite bbox matching the 3x3
#      placeholders of its owning frame. Legend glyphs keep the original 3x4
#      two-color pattern and are separated by background gaps.
#
# e) Visual observability
#    - Multi-frame structure is visible immediately in both input and output.
#    - The output split is also visible immediately because there are always at
#      least 2 target panels and every panel preserves the full multi-frame
#      scaffold while showing only one recolor group.
#    - The critical combo signal is that one panel may contain hits in several
#      different frames because multiple source colors from different frames share
#      the same target. That is readable from a few examples without any hidden
#      numeric ordering.
#
# f) Visual cleanliness
#    - Frames remain simple rectangles, vertically aligned, with only background
#      between them. Legends are grouped into one strip outside the union box, and
#      sprites are placed as isolated patches elsewhere on the canvas.
#    - Output panels are separated by full-height grey divider columns, mirroring
#      the frame color and avoiding clutter from arbitrary separators.
#
# g) Change inferability
#    - Recoloring still comes from explicit legend glyphs, so color change is not
#      arbitrary. The new learnable step is the target-color grouping:
#      multiple legends can share the same target, and those source colors, even
#      when living in different frames, merge into the same output panel.
#    - The generator enforces repeated-target examples, making the grouped-output
#      rule recoverable from only 2-4 demonstrations instead of looking like
#      separate unrelated color changes.
# ---------------------------------------------------------------------------


FRAME_COL = 5
SPRITE_N = 3
FRAME_COLS = 2
NUM_FRAMES_CHOICES = [2, 3]
FRAME_SPEC_BASE = [(1, 2), (2, 3), (2, 4)]  # (rows, filled_count)
FRAME_GAP = 2
NUM_SOURCE_COLORS_RANGE = (3, 5)
NUM_TARGET_COLORS_CHOICES = [2, 3]
SPRITE_FILL_TARGETS = [5, 6, 7]
CANVAS_H_RANGE = (24, 31)
CANVAS_W_RANGE = (24, 31)
PATCH_PADDING = 1
GEN_RETRIES = 500


def _find_frames(g: np.ndarray, bg: int) -> list[tuple[int, tuple[int, int, int, int]]]:
    """Find all rectangular border components that behave like frames."""
    candidates: list[tuple[int, tuple[int, int, int, int]]] = []
    colors = [int(c) for c in np.unique(g) if int(c) != bg]
    for col in colors:
        for pts in components_mask(g == col, connectivity=4):
            r0, c0, r1, c1 = bbox_coords(pts)
            if r1 - r0 < 2 or c1 - c0 < 2:
                continue
            sub = g[r0 : r1 + 1, c0 : c1 + 1]
            border = np.zeros(sub.shape, dtype=bool)
            border[0, :] = True
            border[-1, :] = True
            border[:, 0] = True
            border[:, -1] = True
            if not np.all(sub[border] == col):
                continue
            if np.all(sub[~border] == col):
                continue
            candidates.append((int(col), (int(r0), int(c0), int(r1), int(c1))))
    assert 2 <= len(candidates) <= 3, f"expected 2-3 rectangular frames, found {len(candidates)}"
    frame_colors = {col for col, _ in candidates}
    assert len(frame_colors) == 1, f"expected one shared frame color, found {sorted(frame_colors)}"
    return sorted(candidates, key=lambda item: (item[1][0], item[1][1]))


def _bbox_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _bbox_union(boxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def _collect_components(
    g: np.ndarray,
    outside: np.ndarray,
    bg: int,
    frame_col: int,
) -> list[dict[str, object]]:
    comps: list[dict[str, object]] = []
    colors = [int(c) for c in np.unique(g) if int(c) not in (bg, frame_col)]
    cid = 0
    for col in colors:
        for pts in components_mask((g == col) & outside, connectivity=4):
            r0, c0, r1, c1 = bbox_coords(pts)
            pts_tuples = [(int(r), int(c)) for r, c in pts]
            comps.append(
                {
                    "id": cid,
                    "col": int(col),
                    "pts": pts_tuples,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": len(pts_tuples),
                }
            )
            cid += 1
    return comps


def _extract_legends(
    g: np.ndarray,
    comps: list[dict[str, object]],
    bg: int,
    frame_col: int,
    placeholder_colors: set[int],
) -> tuple[dict[int, int], set[int], list[dict[str, object]]]:
    color_map: dict[int, int] = {}
    legend_ids: set[int] = set()
    legends: list[dict[str, object]] = []

    for i in range(len(comps)):
        for j in range(i + 1, len(comps)):
            ci = comps[i]
            cj = comps[j]
            if ci["col"] == cj["col"]:
                continue
            if not _bbox_overlap(ci["bbox"], cj["bbox"]):
                continue

            r0 = min(ci["bbox"][0], cj["bbox"][0])
            c0 = min(ci["bbox"][1], cj["bbox"][1])
            r1 = max(ci["bbox"][2], cj["bbox"][2])
            c1 = max(ci["bbox"][3], cj["bbox"][3])
            if (r1 - r0 + 1) > 8 or (c1 - c0 + 1) > 8:
                continue

            sub = g[r0 : r1 + 1, c0 : c1 + 1]
            sub_colors = set(map(int, np.unique(sub))) - {frame_col}
            if bg in sub_colors or len(sub_colors) != 2:
                continue
            if ci["col"] not in sub_colors or cj["col"] not in sub_colors:
                continue

            if (ci["col"] in placeholder_colors) ^ (cj["col"] in placeholder_colors):
                src = ci["col"] if ci["col"] in placeholder_colors else cj["col"]
                tgt = cj["col"] if src == ci["col"] else ci["col"]
            else:
                assert ci["area"] != cj["area"], f"ambiguous equal-area legend pair: {ci['col']} and {cj['col']}"
                if ci["area"] > cj["area"]:
                    src, tgt = ci["col"], cj["col"]
                else:
                    src, tgt = cj["col"], ci["col"]

            assert src in placeholder_colors, f"legend source color {src} is not a placeholder"
            assert src not in color_map, f"duplicate legend mapping for placeholder color {src}"
            color_map[int(src)] = int(tgt)
            legend_ids.add(int(ci["id"]))
            legend_ids.add(int(cj["id"]))
            legends.append({"src": int(src), "tgt": int(tgt), "bbox": (int(r0), int(c0), int(r1), int(c1))})

    assert set(color_map) == placeholder_colors, (
        f"legend mappings {sorted(color_map)} do not match placeholders {sorted(placeholder_colors)}"
    )
    return color_map, legend_ids, legends


def _extract_sprites(
    comps: list[dict[str, object]],
    legend_ids: set[int],
    placeholder_colors: set[int],
    phw: dict[int, tuple[int, int]],
) -> dict[int, tuple[tuple[tuple[int, int], ...], int, int]]:
    sprite_coords: dict[int, tuple[tuple[tuple[int, int], ...], int, int]] = {}
    for src_col in sorted(placeholder_colors):
        all_candidates = [c for c in comps if c["col"] == src_col and c["id"] not in legend_ids]
        assert all_candidates, f"no non-legend sprite found for color {src_col}"
        want_h, want_w = phw[src_col]
        sized = [
            c
            for c in all_candidates
            if (c["bbox"][2] - c["bbox"][0] + 1) == want_h and (c["bbox"][3] - c["bbox"][1] + 1) == want_w
        ]
        assert len(sized) == 1, f"expected one size-matched sprite for color {src_col}, found {len(sized)}"
        base = sized[0]
        sr0, sc0, sr1, sc1 = base["bbox"]

        pts_set = set(base["pts"])
        for comp in all_candidates:
            if comp["id"] == base["id"]:
                continue
            r0, c0, r1, c1 = comp["bbox"]
            assert sr0 <= r0 and sr1 >= r1 and sc0 <= c0 and sc1 >= c1, (
                f"extra sprite component for color {src_col} is outside the base bbox"
            )
            pts_set.update(comp["pts"])

        coords = tuple(sorted((r - sr0, c - sc0) for r, c in pts_set))
        sprite_coords[src_col] = (coords, sr1 - sr0 + 1, sc1 - sc0 + 1)
    return sprite_coords


def _panel_order(legends: list[dict[str, object]]) -> list[int]:
    ordered = sorted(legends, key=lambda item: (item["bbox"][0], item["bbox"][1], item["bbox"][2], item["bbox"][3]))
    seen: set[int] = set()
    out: list[int] = []
    for item in ordered:
        tgt = int(item["tgt"])
        if tgt not in seen:
            seen.add(tgt)
            out.append(tgt)
    return out


def _concat_panels(panels: list[np.ndarray], divider_color: int) -> np.ndarray:
    assert panels, "no panels to concatenate"
    h = panels[0].shape[0]
    divider = np.full((h, 1), divider_color, dtype=int)
    pieces = [panels[0]]
    for panel in panels[1:]:
        assert panel.shape[0] == h, "panel heights must match"
        pieces.extend([divider, panel])
    return np.concatenate(pieces, axis=1)


def _frame_dims(rows: int) -> tuple[int, int]:
    inner_h = rows * SPRITE_N + (rows - 1)
    inner_w = FRAME_COLS * SPRITE_N + (FRAME_COLS - 1)
    return inner_h + 2, inner_w + 2


def _make_bordered_frame(h: int, w: int, bg: int) -> np.ndarray:
    frame = np.full((h, w), bg, dtype=int)
    frame[0, :] = frame[-1, :] = FRAME_COL
    frame[:, 0] = frame[:, -1] = FRAME_COL
    return frame


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split the solved multi-frame scene into target-color panels that aggregate across frames.

    Input structure:
    - A background canvas contains 2-3 disjoint grey rectangular frames, each with solid placeholder blocks inside.
    - Outside the whole framed region are same-color sprite templates and compact two-color legend glyphs.

    Rule:
    - Find every frame, recover which placeholder colors belong to each frame, and match each source color to its external sprite and legend target.
    - Build one output panel per distinct legend target color, preserving the full multi-frame layout in every panel.
    - In each panel, stamp only the sprites whose legends recolor them to that panel's target, even when those source colors come from different frames.

    Key insight: The legends now control two things at once: recoloring still comes from each source->target pair, but equal target colors must be merged into the same output panel across all frames.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    frames = _find_frames(g, bg)
    frame_col = int(frames[0][0])

    frame_infos = []
    placeholder_colors: set[int] = set()
    color_to_frame: dict[int, int] = {}
    phw: dict[int, tuple[int, int]] = {}

    for frame_idx, (_col, (fr0, fc0, fr1, fc1)) in enumerate(frames):
        crop = g[fr0 : fr1 + 1, fc0 : fc1 + 1]
        h, w = crop.shape
        assert h >= 3 and w >= 3, f"frame too small for interior: {(h, w)}"

        inner_mask = np.zeros((h, w), dtype=bool)
        inner_mask[1:-1, 1:-1] = True
        local_colors = set(map(int, np.unique(crop[inner_mask]))) - {bg, frame_col}
        assert local_colors, f"frame {frame_idx} has no placeholder colors"

        placeholders: list[tuple[int, tuple[int, int, int, int]]] = []
        for col in sorted(local_colors):
            assert col not in color_to_frame, f"placeholder color {col} appears in multiple frames"
            color_to_frame[col] = int(frame_idx)
            placeholder_colors.add(int(col))

            dims = []
            for pts in components_mask((crop == col) & inner_mask, connectivity=4):
                r0, c0, r1, c1 = bbox_coords(pts)
                bbox = (int(r0), int(c0), int(r1), int(c1))
                placeholders.append((int(col), bbox))
                dims.append((bbox[2] - bbox[0] + 1, bbox[3] - bbox[1] + 1))

            assert dims, f"no placeholder component found for color {col}"
            unique_dims = sorted(set(dims))
            assert len(unique_dims) == 1, f"ambiguous placeholder sizes for color {col}: {unique_dims}"
            phw[int(col)] = unique_dims[0]

        frame_infos.append(
            {
                "bbox": (int(fr0), int(fc0), int(fr1), int(fc1)),
                "crop": np.array(crop, dtype=int),
                "shape": (int(h), int(w)),
                "placeholders": placeholders,
            }
        )

    outside = np.ones(g.shape, dtype=bool)
    for _col, (fr0, fc0, fr1, fc1) in frames:
        outside[fr0 : fr1 + 1, fc0 : fc1 + 1] = False

    comps = _collect_components(g, outside, bg, frame_col)
    color_map, legend_ids, legends = _extract_legends(g, comps, bg, frame_col, placeholder_colors)
    sprite_coords = _extract_sprites(comps, legend_ids, placeholder_colors, phw)
    ordered_tgts = _panel_order(legends)

    union_box = _bbox_union([info["bbox"] for info in frame_infos])
    ur0, uc0, ur1, uc1 = union_box
    panel_shape = (ur1 - ur0 + 1, uc1 - uc0 + 1)
    panels = {tgt: np.full(panel_shape, bg, dtype=int) for tgt in ordered_tgts}

    for tgt in ordered_tgts:
        panel = panels[tgt]
        for info in frame_infos:
            fr0, fc0, _fr1, _fc1 = info["bbox"]
            h, w = info["shape"]
            rr0 = fr0 - ur0
            cc0 = fc0 - uc0
            panel[rr0 : rr0 + h, cc0 : cc0 + w] = info["crop"]
            panel[rr0 + 1 : rr0 + h - 1, cc0 + 1 : cc0 + w - 1] = bg

    for frame_idx, info in enumerate(frame_infos):
        fr0, fc0, _fr1, _fc1 = info["bbox"]
        rr0 = fr0 - ur0
        cc0 = fc0 - uc0
        for src_col, (pr0, pc0, pr1, pc1) in info["placeholders"]:
            assert color_to_frame[src_col] == frame_idx, f"placeholder color {src_col} assigned to wrong frame"
            assert src_col in sprite_coords and src_col in color_map, (
                f"placeholder color {src_col} missing sprite or color mapping"
            )
            tgt_col = int(color_map[src_col])
            assert tgt_col in panels, f"target color {tgt_col} has no output panel"
            coords, sh, sw = sprite_coords[src_col]
            assert (pr1 - pr0 + 1, pc1 - pc0 + 1) == (sh, sw), (
                f"placeholder size does not match sprite for color {src_col}"
            )
            off_r = rr0 + pr0
            off_c = cc0 + pc0
            for dr, dc in coords:
                panels[tgt_col][off_r + dr, off_c + dc] = tgt_col

    return _concat_panels([panels[tgt] for tgt in ordered_tgts], frame_col)


def _legend(src: int, tgt: int) -> np.ndarray:
    return np.array(
        [[src, src, tgt, tgt], [src, src, src, tgt], [src, src, tgt, tgt]],
        dtype=int,
    )


def _legend_strip(order: list[int], color_to_tgt: dict[int, int], bg: int) -> np.ndarray:
    parts = []
    gap = np.full((1, 4), bg, dtype=int)
    for i, src in enumerate(order):
        if i:
            parts.append(gap)
        parts.append(_legend(src, color_to_tgt[src]))
    return np.concatenate(parts, axis=0)


def _sprite_mask() -> np.ndarray:
    target = random.choice(SPRITE_FILL_TARGETS)
    mask = np.zeros((SPRITE_N, SPRITE_N), dtype=bool)

    c = random.randrange(SPRITE_N)
    for r in range(SPRITE_N):
        c = max(0, min(SPRITE_N - 1, c + random.choice([-1, 0, 1])))
        mask[r, c] = True
    mask[random.randrange(SPRITE_N), :] = True

    if random.random() < 0.5:
        mask = mask.T.copy()
    if random.random() < 0.5:
        mask = np.flipud(mask)
    if random.random() < 0.5:
        mask = np.fliplr(mask)

    while int(mask.sum()) < target:
        dilated = np.zeros_like(mask)
        dilated[1:, :] |= mask[:-1, :]
        dilated[:-1, :] |= mask[1:, :]
        dilated[:, 1:] |= mask[:, :-1]
        dilated[:, :-1] |= mask[:, 1:]
        frontier = list(zip(*np.where(dilated & ~mask)))
        r, c = random.choice(frontier)
        mask[r, c] = True

    assert int(mask.sum()) < SPRITE_N * SPRITE_N
    return mask


def _distinct_masks(src_cols: list[int]) -> dict[int, np.ndarray]:
    masks: dict[int, np.ndarray] = {}
    seen: set[tuple[tuple[int, ...], ...]] = set()
    for src in src_cols:
        for _ in range(100):
            mask = _sprite_mask()
            key = tuple(tuple(int(v) for v in row) for row in mask.astype(int))
            if key not in seen:
                seen.add(key)
                masks[src] = mask
                break
        else:
            return {}
    return masks


def _colored_sprite(mask: np.ndarray, color: int, bg: int) -> np.ndarray:
    out = np.full(mask.shape, bg, dtype=int)
    out[mask] = int(color)
    return out


def _choose_empty_slots(rows: int, cols: int, empty_count: int) -> tuple[set[tuple[int, int]], list[tuple[int, int]]]:
    slots = [(r, c) for r in range(rows) for c in range(cols)]
    valid = []
    for combo in itertools.combinations(slots, empty_count):
        empty = set(combo)
        filled = [s for s in slots if s not in empty]
        if len({r for r, _ in filled}) == rows and len({c for _, c in filled}) == cols:
            valid.append(empty)
    empty_slots = random.choice(valid)
    filled_slots = [s for s in slots if s not in empty_slots]
    random.shuffle(filled_slots)
    return empty_slots, filled_slots


def _choose_frame_specs(num_frames: int) -> list[tuple[int, int]]:
    options = []
    for combo in itertools.product(FRAME_SPEC_BASE, repeat=num_frames):
        total = sum(filled for _rows, filled in combo)
        union_h = sum(_frame_dims(rows)[0] for rows, _filled in combo) + FRAME_GAP * (num_frames - 1)
        if 6 <= total <= 10 and union_h <= 27:
            options.append(list(combo))
    return random.choice(options)


def _partitions(total: int, caps: list[int]) -> list[tuple[int, ...]]:
    out: list[tuple[int, ...]] = []

    def rec(idx: int, remain: int, acc: list[int]) -> None:
        if idx == len(caps):
            if remain == 0:
                out.append(tuple(acc))
            return
        left = len(caps) - idx - 1
        lo = 1
        hi = min(caps[idx], remain - left)
        for val in range(lo, hi + 1):
            acc.append(val)
            rec(idx + 1, remain - val, acc)
            acc.pop()

    rec(0, total, [])
    return out


def _vertical_frame_positions(frame_specs: list[tuple[int, int]]) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    positions = []
    cur_r = 0
    union_w = 0
    for idx, (rows, _filled) in enumerate(frame_specs):
        h, w = _frame_dims(rows)
        positions.append((cur_r, 0))
        union_w = max(union_w, w)
        cur_r += h
        if idx + 1 < len(frame_specs):
            cur_r += FRAME_GAP
    return positions, (cur_r, union_w)


def _place_patch(
    grid: np.ndarray,
    reserve: np.ndarray,
    patch: np.ndarray,
    bg: int,
    *,
    transparent: bool,
) -> tuple[int, int] | None:
    blocker = np.full(patch.shape, FRAME_COL, dtype=int)
    try:
        r, c = random_free_location(reserve, blocker, transparent=bg, padding=PATCH_PADDING)
    except ValueError:
        return None
    paste(reserve, blocker, r, c)
    paste(grid, patch, r, c, transparent=bg if transparent else None)
    return int(r), int(c)


def _assign_targets(
    frame_color_groups: list[list[int]],
    target_cols: list[int],
) -> dict[int, int] | None:
    src_cols = [src for group in frame_color_groups for src in group]
    frame_of = {src: frame_idx for frame_idx, group in enumerate(frame_color_groups) for src in group}

    for _ in range(200):
        shuffled = src_cols[:]
        random.shuffle(shuffled)
        color_to_tgt: dict[int, int] = {}
        for tgt, src in zip(target_cols, shuffled):
            color_to_tgt[src] = tgt
        for src in shuffled[len(target_cols) :]:
            color_to_tgt[src] = random.choice(target_cols)

        used = {color_to_tgt[src] for src in src_cols}
        if used != set(target_cols):
            continue

        shared = False
        for tgt in target_cols:
            owners = {frame_of[src] for src in src_cols if color_to_tgt[src] == tgt}
            if len(owners) >= 2:
                shared = True
                break
        if shared:
            return color_to_tgt
    return None


def _panel_order_from_legends(legend_order: list[int], color_to_tgt: dict[int, int]) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for src in legend_order:
        tgt = int(color_to_tgt[src])
        if tgt not in seen:
            seen.add(tgt)
            out.append(tgt)
    return out


def _build_expected_output(
    bg: int,
    frame_specs: list[tuple[int, int]],
    frame_positions: list[tuple[int, int]],
    union_shape: tuple[int, int],
    cell_color_maps: list[dict[tuple[int, int], int]],
    masks: dict[int, np.ndarray],
    color_to_tgt: dict[int, int],
    panel_order: list[int],
) -> np.ndarray:
    union_h, union_w = union_shape
    base = np.full((union_h, union_w), bg, dtype=int)

    for (rows, _filled), (rr, cc) in zip(frame_specs, frame_positions):
        frame_h, frame_w = _frame_dims(rows)
        paste(base, _make_bordered_frame(frame_h, frame_w, bg), rr, cc)

    panels = {tgt: np.array(base, dtype=int).copy() for tgt in panel_order}
    for cell_colors, (rows, _filled), (rr, cc) in zip(cell_color_maps, frame_specs, frame_positions):
        for (slot_r, slot_c), src in cell_colors.items():
            r0 = rr + 1 + slot_r * (SPRITE_N + 1)
            c0 = cc + 1 + slot_c * (SPRITE_N + 1)
            paste(
                panels[color_to_tgt[src]],
                _colored_sprite(masks[src], color_to_tgt[src], bg),
                r0,
                c0,
                transparent=bg,
            )

    out = _concat_panels([panels[tgt] for tgt in panel_order], FRAME_COL)
    assert out.shape[0] <= 30 and out.shape[1] <= 30, f"output too large: {out.shape}"
    return out


def _gen_attempt() -> np.ndarray | None:
    num_frames = random.choice(NUM_FRAMES_CHOICES)
    frame_specs = _choose_frame_specs(num_frames)
    frame_positions, union_shape = _vertical_frame_positions(frame_specs)
    union_h, union_w = union_shape

    bg = random.choice([c for c in range(10) if c != FRAME_COL])
    usable = [c for c in range(10) if c not in (bg, FRAME_COL)]

    total_blocks = sum(filled for _rows, filled in frame_specs)
    k = random.randint(max(NUM_SOURCE_COLORS_RANGE[0], num_frames), min(NUM_SOURCE_COLORS_RANGE[1], total_blocks))
    color_partition = random.choice(_partitions(k, [filled for _rows, filled in frame_specs]))

    src_cols = random.sample(usable, k)
    remaining_for_tgt = [c for c in usable if c not in src_cols]
    num_targets = random.choice([n for n in NUM_TARGET_COLORS_CHOICES if n <= k and n <= len(remaining_for_tgt)])
    target_cols = random.sample(remaining_for_tgt, num_targets)

    shuffled_srcs = src_cols[:]
    random.shuffle(shuffled_srcs)
    frame_color_groups: list[list[int]] = []
    cursor = 0
    for count in color_partition:
        group = shuffled_srcs[cursor : cursor + count]
        frame_color_groups.append(group)
        cursor += count

    color_to_tgt = _assign_targets(frame_color_groups, target_cols)
    if color_to_tgt is None:
        return None

    masks = _distinct_masks(src_cols)
    if len(masks) != k:
        return None

    input_frames: list[np.ndarray] = []
    cell_color_maps: list[dict[tuple[int, int], int]] = []
    for (rows, filled_count), local_srcs in zip(frame_specs, frame_color_groups):
        empty_count = rows * FRAME_COLS - filled_count
        _, filled_slots = _choose_empty_slots(rows, FRAME_COLS, empty_count)

        cell_colors: dict[tuple[int, int], int] = {}
        for slot, col in zip(filled_slots[: len(local_srcs)], local_srcs):
            cell_colors[slot] = int(col)
        for slot in filled_slots[len(local_srcs) :]:
            cell_colors[slot] = int(random.choice(local_srcs))
        cell_color_maps.append(cell_colors)

        frame_h, frame_w = _frame_dims(rows)
        input_frame = _make_bordered_frame(frame_h, frame_w, bg)
        for (rr, cc), src in cell_colors.items():
            r0 = 1 + rr * (SPRITE_N + 1)
            c0 = 1 + cc * (SPRITE_N + 1)
            paste(input_frame, np.full((SPRITE_N, SPRITE_N), src, dtype=int), r0, c0)
        input_frames.append(input_frame)

    legend_order = src_cols[:]
    random.shuffle(legend_order)
    panel_order = _panel_order_from_legends(legend_order, color_to_tgt)
    expected = _build_expected_output(
        bg,
        frame_specs,
        frame_positions,
        union_shape,
        cell_color_maps,
        masks,
        color_to_tgt,
        panel_order,
    )

    ext_area = len(src_cols) * (SPRITE_N * SPRITE_N + 16)
    possible_dims = [
        (h, w)
        for h in range(*CANVAS_H_RANGE)
        for w in range(*CANVAS_W_RANGE)
        if h >= union_h and w >= union_w and h * w - union_h * union_w >= ext_area
    ]
    if not possible_dims:
        return None

    h, w = random.choice(possible_dims)
    grid = np.full((h, w), bg, dtype=int)
    reserve = np.full((h, w), bg, dtype=int)

    ur0 = random.randrange(0, h - union_h + 1)
    uc0 = random.randrange(0, w - union_w + 1)
    paste(reserve, np.full((union_h, union_w), FRAME_COL, dtype=int), ur0, uc0)

    for input_frame, (rr, cc) in zip(input_frames, frame_positions):
        paste(grid, input_frame, ur0 + rr, uc0 + cc)

    legend_patch = _legend_strip(legend_order, color_to_tgt, bg)
    if _place_patch(grid, reserve, legend_patch, bg, transparent=False) is None:
        return None

    sprite_patches = [(_colored_sprite(masks[src], src, bg), True) for src in src_cols]
    random.shuffle(sprite_patches)
    for patch, transparent in sprite_patches:
        if _place_patch(grid, reserve, patch, bg, transparent=transparent) is None:
            return None

    try:
        got = main(grid)
    except Exception:
        return None
    if not np.array_equal(got, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        out = _gen_attempt()
        if out is not None:
            return out
    raise RuntimeError("failed to generate valid edb79dae combo input")
