from __future__ import annotations

import numpy as np


SIDES = ("top", "right", "bottom", "left")
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_GEN_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _tight_mask(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask, (r0, c0, r1, c1)


def _crop_bool(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox(pts)
    return np.asarray(mask[r0 : r1 + 1, c0 : c1 + 1], dtype=bool)


def _is_connected(mask: np.ndarray) -> bool:
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return False
    return len(_components(mask)) == 1


def _edge_signature(mask: np.ndarray, side: str) -> tuple[int, tuple[int, ...]] | None:
    mask = np.asarray(mask, dtype=bool)
    if side == "top":
        line = mask[0, :]
    elif side == "bottom":
        line = mask[-1, :]
    elif side == "left":
        line = mask[:, 0]
    else:
        line = mask[:, -1]
    idx = np.flatnonzero(line)
    if idx.size == 0:
        return None
    start = int(idx.min())
    end = int(idx.max())
    return start, tuple(int(v) for v in line[start : end + 1])


def _rotation_k(src_side: str, dst_side: str) -> int:
    return (SIDES.index(src_side) - SIDES.index(dst_side)) % 4


def _boundary_pixels(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    out = np.zeros((h, w), dtype=bool)
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        for dr, dc in DIRS4:
            nr, nc = rr + dr, cc + dc
            if not (0 <= nr < h and 0 <= nc < w) or not mask[nr, nc]:
                out[rr, cc] = True
                break
    return out


def _frame_boundary_mask(h: int, w: int) -> np.ndarray:
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return border


def _detect_frames(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    frames: list[dict] = []
    for color in [int(v) for v in np.unique(g) if int(v) != bg]:
        for coords in _components(g == color):
            r0, c0, r1, c1 = _bbox(coords)
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            if h < 5 or w < 5:
                continue
            local = np.zeros((h, w), dtype=bool)
            local[coords[:, 0] - r0, coords[:, 1] - c0] = True
            border = _frame_boundary_mask(h, w)
            if np.any(local & (~border)):
                continue
            missing = border & (~local)
            miss_pts = np.argwhere(missing)
            if miss_pts.size == 0 or int(miss_pts.shape[0]) > 3:
                continue

            side = None
            positions: list[int] = []
            if np.all(miss_pts[:, 0] == 0):
                side = "top"
                positions = sorted(int(c) for c in miss_pts[:, 1])
                edge_limit = w - 1
            elif np.all(miss_pts[:, 0] == h - 1):
                side = "bottom"
                positions = sorted(int(c) for c in miss_pts[:, 1])
                edge_limit = w - 1
            elif np.all(miss_pts[:, 1] == 0):
                side = "left"
                positions = sorted(int(r) for r in miss_pts[:, 0])
                edge_limit = h - 1
            elif np.all(miss_pts[:, 1] == w - 1):
                side = "right"
                positions = sorted(int(r) for r in miss_pts[:, 0])
                edge_limit = h - 1
            else:
                continue

            if positions[0] <= 0 or positions[-1] >= edge_limit:
                continue
            if positions != list(range(positions[0], positions[-1] + 1)):
                continue

            notch_cells = {(int(r0 + rr), int(c0 + cc)) for rr, cc in miss_pts}
            frames.append(
                {
                    "color": int(color),
                    "bbox": (r0, c0, r1, c1),
                    "side": side,
                    "notch_start": int(positions[0]),
                    "notch_signature": tuple(1 for _ in positions),
                    "notch_cells": notch_cells,
                    "outer_h": int(h),
                    "outer_w": int(w),
                }
            )

    frames.sort(key=lambda d: d["bbox"])
    return frames


def _attach_patterns(grid: np.ndarray, bg: int, frames: list[dict]) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    out: list[dict] = []
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        comps = _components(inner != bg)
        if len(comps) != 1:
            continue
        local_coords = comps[0]
        abs_coords = local_coords.copy()
        abs_coords[:, 0] += r0 + 1
        abs_coords[:, 1] += c0 + 1
        colors = {int(g[int(rr), int(cc)]) for rr, cc in abs_coords}
        if len(colors) != 1:
            continue
        pmask, pbbox = _tight_mask(abs_coords)
        out.append(
            {
                **frame,
                "pattern": {
                    "color": int(next(iter(colors))),
                    "bbox": pbbox,
                    "mask": pmask,
                },
            }
        )
    return out


def _try_match(
    grid: np.ndarray,
    bg: int,
    src_frame: dict,
    dst_frame: dict,
) -> dict | None:
    src_mask = np.asarray(src_frame["pattern"]["mask"], dtype=bool)
    k = _rotation_k(src_frame["side"], dst_frame["side"])
    rot_mask = np.rot90(src_mask, k)
    edge = _edge_signature(rot_mask, dst_frame["side"])
    if edge is None or edge[1] != dst_frame["notch_signature"]:
        return None

    start = int(edge[0])
    notch_start = int(dst_frame["notch_start"])
    r0, c0, r1, c1 = dst_frame["bbox"]
    h, w = rot_mask.shape
    if dst_frame["side"] == "top":
        tr = int(r0)
        tc = int(c0 + notch_start - start)
    elif dst_frame["side"] == "bottom":
        tr = int(r1 - h + 1)
        tc = int(c0 + notch_start - start)
    elif dst_frame["side"] == "left":
        tr = int(r0 + notch_start - start)
        tc = int(c0)
    else:
        tr = int(r0 + notch_start - start)
        tc = int(c1 - w + 1)

    cells = []
    for rr, cc in np.argwhere(rot_mask):
        gr = int(tr + rr)
        gc = int(tc + cc)
        if not (r0 <= gr <= r1 and c0 <= gc <= c1):
            return None
        on_border = gr in (r0, r1) or gc in (c0, c1)
        if on_border and (gr, gc) not in dst_frame["notch_cells"]:
            return None
        if int(grid[gr, gc]) != bg:
            return None
        cells.append((gr, gc))

    return {
        "src_bbox": src_frame["pattern"]["bbox"],
        "src_mask": src_mask,
        "src_color": int(src_frame["pattern"]["color"]),
        "src_index": int(src_frame["index"]),
        "dst_index": int(dst_frame["index"]),
        "rot_mask": rot_mask,
        "top": int(tr),
        "left": int(tc),
        "cells": cells,
    }


def _find_unique_match(grid: np.ndarray) -> dict | None:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    frames = _attach_patterns(g, bg, _detect_frames(g, bg))
    for idx, frame in enumerate(frames):
        frame["index"] = int(idx)

    matches: list[dict] = []
    for i, src in enumerate(frames):
        for j, dst in enumerate(frames):
            if i == j:
                continue
            found = _try_match(g, bg, src, dst)
            if found is not None:
                matches.append(found)

    if len(matches) != 1:
        return None
    return matches[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move the one key-shape whose active edge matches another frame's notch.

    The key edge is read from the side pointed to by the source frame's notch.
    The moved shape is rotated to the target notch direction, docked into that
    notch, and the source position is replaced by a 1-pixel hollow outline.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    match = _find_unique_match(g)
    if match is None:
        return g.copy()

    out = g.copy()
    src_r0, src_c0, _src_r1, _src_c1 = match["src_bbox"]
    src_mask = np.asarray(match["src_mask"], dtype=bool)
    src_color = int(match["src_color"])

    out[src_r0 : src_r0 + src_mask.shape[0], src_c0 : src_c0 + src_mask.shape[1]][src_mask] = bg
    outline = _boundary_pixels(src_mask)
    out[src_r0 : src_r0 + outline.shape[0], src_c0 : src_c0 + outline.shape[1]][outline] = src_color

    rot_mask = np.asarray(match["rot_mask"], dtype=bool)
    top = int(match["top"])
    left = int(match["left"])
    out[top : top + rot_mask.shape[0], left : left + rot_mask.shape[1]][rot_mask] = src_color
    return out


def _rotate_top_active(mask: np.ndarray, side: str) -> np.ndarray:
    return np.rot90(np.asarray(mask, dtype=bool), _rotation_k("top", side))


def _weighted_pick(rng: np.random.Generator, weights: list[float]) -> int:
    probs = np.asarray(weights, dtype=float)
    probs /= probs.sum()
    return int(rng.choice(len(weights), p=probs))


def _generate_top_shape(
    rng: np.random.Generator,
    signature: tuple[int, ...],
    max_h: int,
    max_w: int,
    min_area: int,
    max_area: int,
) -> np.ndarray | None:
    need_w = max(2, len(signature))
    if max_h < 2 or max_w < need_w:
        return None
    sig_arr = np.asarray(signature, dtype=int)

    for _ in range(200):
        h = int(rng.integers(2, max_h + 1))
        w = int(rng.integers(need_w, max_w + 1))
        offset = int(rng.integers(0, w - len(signature) + 1))

        mask = np.zeros((h, w), dtype=bool)
        for i, v in enumerate(sig_arr):
            if int(v):
                mask[0, offset + i] = True

        lo = max(int(mask.sum()) + 1, int(min_area))
        hi = min(int(max_area), int(h * w))
        if lo > hi:
            continue
        target_area = int(rng.integers(lo, hi + 1))

        while int(mask.sum()) < target_area:
            frontier: list[tuple[int, int]] = []
            weights: list[float] = []
            for rr, cc in np.argwhere(mask):
                r = int(rr)
                c = int(cc)
                for dr, dc in DIRS4:
                    nr = r + dr
                    nc = c + dc
                    if not (0 <= nr < h and 0 <= nc < w) or mask[nr, nc]:
                        continue
                    if nr == 0:
                        inside = offset <= nc < offset + len(signature) and signature[nc - offset] == 1
                        if not inside:
                            continue
                    cand = (nr, nc)
                    if cand in frontier:
                        continue
                    frontier.append(cand)
                    weights.append(3.0 if nr > 0 else 1.0)
            if not frontier:
                break
            pick = frontier[_weighted_pick(rng, weights)]
            mask[pick] = True

        tight = _crop_bool(mask)
        edge = _edge_signature(tight, "top")
        if edge is None or edge[1] != signature:
            continue
        if not _is_connected(tight):
            continue
        if int(tight.sum()) < lo:
            continue
        return tight
    return None


def _frame_patch(
    outer_h: int,
    outer_w: int,
    frame_color: int,
    bg: int,
    side: str,
    notch_start: int,
    notch_len: int,
) -> np.ndarray:
    patch = np.full((outer_h, outer_w), bg, dtype=int)
    patch[0, :] = frame_color
    patch[-1, :] = frame_color
    patch[:, 0] = frame_color
    patch[:, -1] = frame_color
    if side == "top":
        patch[0, notch_start : notch_start + notch_len] = bg
    elif side == "bottom":
        patch[-1, notch_start : notch_start + notch_len] = bg
    elif side == "left":
        patch[notch_start : notch_start + notch_len, 0] = bg
    else:
        patch[notch_start : notch_start + notch_len, -1] = bg
    return patch


def _layout_frames(
    rng: np.random.Generator,
    sizes: list[tuple[int, int]],
) -> tuple[tuple[int, int], list[tuple[int, int]]] | None:
    n = len(sizes)
    all_slots = [(0, 0), (0, 1), (1, 0), (1, 1)]
    if n == 4:
        chosen = all_slots
    else:
        drop = int(rng.integers(0, 4))
        chosen = [slot for i, slot in enumerate(all_slots) if i != drop]
    rng.shuffle(chosen)

    row_heights = [0, 0]
    col_widths = [0, 0]
    for (h, w), (sr, sc) in zip(sizes, chosen, strict=False):
        row_heights[sr] = max(row_heights[sr], int(h))
        col_widths[sc] = max(col_widths[sc], int(w))

    gap_r = int(rng.integers(2, 5))
    gap_c = int(rng.integers(2, 5))
    top_margin = int(rng.integers(1, 4))
    bottom_margin = int(rng.integers(1, 4))
    left_margin = int(rng.integers(1, 4))
    right_margin = int(rng.integers(1, 4))

    H = top_margin + row_heights[0] + gap_r + row_heights[1] + bottom_margin
    W = left_margin + col_widths[0] + gap_c + col_widths[1] + right_margin
    if H > 30 or W > 30:
        return None

    row_base = [top_margin, top_margin + row_heights[0] + gap_r]
    col_base = [left_margin, left_margin + col_widths[0] + gap_c]
    positions: list[tuple[int, int]] = []
    for (h, w), (sr, sc) in zip(sizes, chosen, strict=False):
        r_slack = row_heights[sr] - int(h)
        c_slack = col_widths[sc] - int(w)
        r0 = row_base[sr] + int(rng.integers(0, r_slack + 1))
        c0 = col_base[sc] + int(rng.integers(0, c_slack + 1))
        positions.append((r0, c0))
    return (H, W), positions


def _place_pattern(
    rng: np.random.Generator,
    grid: np.ndarray,
    frame: dict,
    mask: np.ndarray,
    color: int,
    forbidden: np.ndarray | None = None,
) -> tuple[int, int] | None:
    r0, c0, r1, c1 = frame["bbox"]
    ph, pw = mask.shape
    if ph > (r1 - r0 - 1) or pw > (c1 - c0 - 1):
        return None

    candidates: list[tuple[int, int]] = []
    for top in range(r0 + 1, r1 - ph + 1):
        for left in range(c0 + 1, c1 - pw + 1):
            ok = True
            for rr, cc in np.argwhere(mask):
                gr = int(top + rr)
                gc = int(left + cc)
                if int(grid[gr, gc]) != _get_bg(grid):
                    ok = False
                    break
                if forbidden is not None and forbidden[gr, gc]:
                    ok = False
                    break
            if ok:
                candidates.append((int(top), int(left)))
    if not candidates:
        return None
    top, left = candidates[int(rng.integers(0, len(candidates)))]
    view = grid[top : top + ph, left : left + pw]
    view[mask] = int(color)
    return int(top), int(left)


def _signature_for_notch_len(notch_len: int) -> tuple[int, ...]:
    return tuple(1 for _ in range(int(notch_len)))


def _distractor_signature(rng: np.random.Generator) -> tuple[int, ...]:
    choices = [
        (1, 0, 1),
        (1, 1, 0, 1),
        (1, 0, 1, 1),
        (1, 1, 1, 1),
    ]
    return choices[int(rng.integers(0, len(choices)))]


def _make_frame_specs(
    rng: np.random.Generator,
    n_frames: int,
    src_idx: int,
    dst_idx: int,
) -> list[dict]:
    target_len = int(rng.integers(1, 4))
    other_lengths = [1, 2, 3]
    other_lengths.remove(target_len)
    specs = []
    for idx in range(n_frames):
        outer_h = int(rng.integers(8, 11))
        outer_w = int(rng.integers(8, 11))
        side = SIDES[int(rng.integers(0, 4))]
        if idx == dst_idx:
            notch_len = target_len
        elif idx == src_idx:
            notch_len = int(rng.integers(1, 4))
        else:
            notch_len = int(other_lengths[int(rng.integers(0, len(other_lengths)))])
        if side in ("top", "bottom"):
            notch_start = int(rng.integers(1, outer_w - notch_len))
        else:
            notch_start = int(rng.integers(1, outer_h - notch_len))
        specs.append(
            {
                "outer_h": outer_h,
                "outer_w": outer_w,
                "side": side,
                "notch_len": notch_len,
                "notch_start": notch_start,
            }
        )
    return specs


def _build_input(rng: np.random.Generator) -> np.ndarray | None:
    n_frames = int(rng.choice([3, 4]))
    src_idx, dst_idx = rng.choice(n_frames, size=2, replace=False).tolist()

    specs = _make_frame_specs(rng, n_frames, int(src_idx), int(dst_idx))
    sizes = [(spec["outer_h"], spec["outer_w"]) for spec in specs]
    layout = _layout_frames(rng, sizes)
    if layout is None:
        return None
    (H, W), positions = layout

    palette = rng.choice(np.arange(10), size=n_frames + 2, replace=False)
    bg = int(palette[0])
    frame_color = int(palette[1])
    pattern_colors = [int(v) for v in palette[2:]]

    grid = np.full((H, W), bg, dtype=int)
    frames: list[dict] = []
    for idx, (spec, (r0, c0)) in enumerate(zip(specs, positions, strict=False)):
        patch = _frame_patch(
            spec["outer_h"],
            spec["outer_w"],
            frame_color,
            bg,
            spec["side"],
            spec["notch_start"],
            spec["notch_len"],
        )
        grid[r0 : r0 + spec["outer_h"], c0 : c0 + spec["outer_w"]] = patch
        frames.append(
            {
                "index": int(idx),
                "bbox": (int(r0), int(c0), int(r0 + spec["outer_h"] - 1), int(c0 + spec["outer_w"] - 1)),
                "side": spec["side"],
                "notch_start": int(spec["notch_start"]),
                "notch_signature": _signature_for_notch_len(spec["notch_len"]),
                "notch_cells": {
                    (int(r0 + rr), int(c0 + cc))
                    for rr, cc in np.argwhere(patch == bg)
                    if rr in (0, spec["outer_h"] - 1) or cc in (0, spec["outer_w"] - 1)
                },
            }
        )

    source_signature = _signature_for_notch_len(specs[int(dst_idx)]["notch_len"])
    source_top = _generate_top_shape(rng, source_signature, max_h=5, max_w=5, min_area=4, max_area=8)
    if source_top is None:
        return None
    source_mask = _rotate_top_active(source_top, frames[int(src_idx)]["side"])

    match_probe = _try_match(grid, bg, {**frames[int(src_idx)], "pattern": {"mask": source_mask, "color": pattern_colors[int(src_idx)], "bbox": (0, 0, 0, 0)}}, frames[int(dst_idx)])
    if match_probe is None:
        return None

    source_frame = frames[int(src_idx)]
    placed_source = _place_pattern(rng, grid, source_frame, source_mask, pattern_colors[int(src_idx)])
    if placed_source is None:
        return None

    reserved = np.zeros_like(grid, dtype=bool)
    for cell in match_probe["cells"]:
        reserved[cell] = True

    for idx, frame in enumerate(frames):
        if idx == src_idx:
            continue
        sig = _distractor_signature(rng)
        top_mask = _generate_top_shape(rng, sig, max_h=5, max_w=6, min_area=4, max_area=8)
        if top_mask is None:
            return None
        pat_mask = _rotate_top_active(top_mask, frame["side"])
        forbidden = reserved if idx == dst_idx else None
        if _place_pattern(rng, grid, frame, pat_mask, pattern_colors[idx], forbidden=forbidden) is None:
            return None

    if _get_bg(grid) != bg:
        return None
    match = _find_unique_match(grid)
    if match is None:
        return None
    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _build_input(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {MAX_GEN_ATTEMPTS} attempts")
