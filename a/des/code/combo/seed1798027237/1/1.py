from __future__ import annotations

import numpy as np


ORTHO_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
RNG = np.random.default_rng()
BG = 0
FRAME_THICKNESS = 3
MAX_GENERATION_ATTEMPTS = 400

# Top-oriented gap templates: outer end at row 0, inner end at row 2.
GAP_TEMPLATES = [
    {
        "name": "branch_mid",
        "cells": [(0, 0), (1, 0), (2, 0), (1, 1)],
        "trunk": [(0, 0), (1, 0), (2, 0)],
    },
    {
        "name": "branch_inner",
        "cells": [(0, 0), (1, 0), (2, 0), (2, 1)],
        "trunk": [(0, 0), (1, 0), (2, 0)],
    },
    {
        "name": "hook_inner",
        "cells": [(0, 0), (1, 0), (2, 0), (1, 1), (2, 1)],
        "trunk": [(0, 0), (1, 0), (2, 0)],
    },
    {
        "name": "hook_outer",
        "cells": [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1)],
        "trunk": [(0, 0), (1, 0), (2, 0)],
    },
    {
        "name": "fork_outer",
        "cells": [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (1, 2)],
        "trunk": [(0, 0), (1, 0), (2, 0)],
    },
    {
        "name": "fork_mid",
        "cells": [(0, 0), (1, 0), (2, 0), (1, 1), (1, 2)],
        "trunk": [(0, 0), (1, 0), (2, 0)],
    },
    {
        "name": "zigzag",
        "cells": [(0, 0), (1, 0), (2, 0), (0, 1), (2, 1), (2, 2)],
        "trunk": [(0, 0), (1, 0), (2, 0)],
    },
]

SIDE_TO_ROT = {"top": 0, "left": 1, "bottom": 2, "right": 3}


def _bbox_from_points(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in points]
    cs = [c for _, c in points]
    return min(rs), min(cs), max(rs), max(cs)


def _mask_from_points(points: list[tuple[int, int]]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_from_points(points)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in points:
        mask[r - r0, c - c0] = True
    return mask


def _rotate_coords(points: list[tuple[int, int]], k: int) -> list[tuple[int, int]]:
    mask = _mask_from_points(points).astype(int)
    rotated = np.rot90(mask, k=k) > 0
    return [(int(r), int(c)) for r, c in np.argwhere(rotated)]


def _crop_mask(mask: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return mask[:0, :0].copy(), (0, 0, -1, -1)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy(), (int(r0), int(c0), int(r1), int(c1))


def _crop_patch(patch: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cropped_mask, (r0, c0, r1, c1) = _crop_mask(mask)
    return patch[r0 : r1 + 1, c0 : c1 + 1].copy(), cropped_mask


def _normalize_points(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    cropped, _ = _crop_mask(mask)
    return tuple(map(tuple, np.argwhere(cropped)))


def _canonical_shape(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    variants = []
    cur = mask.copy()
    for _ in range(4):
        variants.append(_normalize_points(cur))
        cur = np.rot90(cur)
    return min(variants)


def _connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr, dc in ORTHO_DIRS:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _distance_map(mask: np.ndarray, start: tuple[int, int]) -> np.ndarray:
    h, w = mask.shape
    dist = np.full((h, w), -1, dtype=int)
    queue = [start]
    dist[start] = 0
    head = 0
    while head < len(queue):
        r, c = queue[head]
        head += 1
        for dr, dc in ORTHO_DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and dist[nr, nc] < 0:
                dist[nr, nc] = dist[r, c] + 1
                queue.append((nr, nc))
    return dist


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _band_mask(height: int, width: int, thickness: int) -> np.ndarray:
    rows = np.arange(height)[:, None]
    cols = np.arange(width)[None, :]
    dist = np.minimum(rows, height - 1 - rows)
    dist = np.minimum(dist, cols)
    dist = np.minimum(dist, width - 1 - cols)
    return dist < thickness


def _adjacent_to_mask(mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(mask, dtype=bool)
    h, w = mask.shape
    for r, c in np.argwhere(mask):
        for dr, dc in ORTHO_DIRS:
            nr, nc = int(r + dr), int(c + dc)
            if 0 <= nr < h and 0 <= nc < w:
                out[nr, nc] = True
    return out


def _extract_frames(grid: np.ndarray, bg: int) -> list[dict]:
    frames = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != bg):
        pts = np.argwhere(grid == color)
        if pts.size == 0:
            continue
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        r0 = int(r0)
        c0 = int(c0)
        r1 = int(r1)
        c1 = int(c1)
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        if height < 7 or width < 7:
            continue
        edge_dist = np.min(
            np.stack(
                [
                    pts[:, 0] - r0,
                    r1 - pts[:, 0],
                    pts[:, 1] - c0,
                    c1 - pts[:, 1],
                ],
                axis=1,
            ),
            axis=1,
        )
        thickness = int(edge_dist.max()) + 1
        if thickness < 2:
            continue
        band = _band_mask(height, width, thickness)
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        if np.any((sub == color) & (~band)):
            continue
        if np.any((sub != bg) & (sub != color)):
            continue
        interior = ~band
        if not interior.any() or np.any(sub[interior] != bg):
            continue
        gap = band & (sub == bg)
        if not gap.any():
            continue
        comps = _connected_components(gap)
        if len(comps) != 1:
            continue
        if int(gap.sum()) > int(band.sum()) // 3:
            continue
        boundary = np.zeros_like(gap, dtype=bool)
        boundary[0, :] = True
        boundary[-1, :] = True
        boundary[:, 0] = True
        boundary[:, -1] = True
        inner_touch = gap & _adjacent_to_mask(interior)
        outer_touch = gap & boundary
        if not outer_touch.any() or not inner_touch.any():
            continue
        gap_norm, (gr0, gc0, gr1, gc1) = _crop_mask(gap)
        outer_norm = outer_touch[gr0 : gr1 + 1, gc0 : gc1 + 1]
        inner_norm = inner_touch[gr0 : gr1 + 1, gc0 : gc1 + 1]
        frames.append(
            {
                "color": color,
                "bbox": (r0, c0, r1, c1),
                "thickness": thickness,
                "gap_mask": gap,
                "gap_norm": gap_norm,
                "gap_canon": _canonical_shape(gap_norm),
                "outer_norm": outer_norm,
                "inner_norm": inner_norm,
            }
        )
    return frames


def _extract_fragments(grid: np.ndarray, bg: int, frame_colors: set[int]) -> list[dict]:
    frame_color_mask = np.isin(grid, list(frame_colors)) if frame_colors else np.zeros_like(grid, dtype=bool)
    frag_mask = (grid != bg) & (~frame_color_mask)
    components = _connected_components(frag_mask)
    frags = []
    for pts in components:
        r0, c0, r1, c1 = _bbox_from_points(pts)
        patch = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
        local_mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in pts:
            local_mask[r - r0, c - c0] = True
        patch, local_mask = _crop_patch(patch, local_mask)
        colors, counts = np.unique(patch[local_mask], return_counts=True)
        base_color = int(colors[np.argmax(counts)])
        marker_pts = []
        marker_colors = []
        for rr, cc in np.argwhere(local_mask):
            color = int(patch[rr, cc])
            if color != base_color:
                marker_pts.append((int(rr), int(cc)))
                marker_colors.append(color)
        if not marker_pts:
            continue
        frags.append(
            {
                "patch": patch,
                "mask": local_mask,
                "bbox": (r0, c0, r1, c1),
                "base_color": base_color,
                "marker_points": marker_pts,
                "marker_colors": marker_colors,
                "canon": _canonical_shape(local_mask),
            }
        )
    return frags


def _read_marker_sequence(frame: dict, frag: dict) -> list[int]:
    matched = None
    for k in range(4):
        rot_patch = np.rot90(frag["patch"], k=k)
        rot_mask = np.rot90(frag["mask"], k=k)
        rot_patch, rot_mask = _crop_patch(rot_patch, rot_mask)
        if rot_mask.shape == frame["gap_norm"].shape and np.array_equal(rot_mask, frame["gap_norm"]):
            matched = (rot_patch, rot_mask)
            break
    assert matched is not None, "fragment rotation must align with gap"
    rot_patch, rot_mask = matched
    outer_pts = [tuple(map(int, p)) for p in np.argwhere(frame["outer_norm"] & frame["gap_norm"])]
    assert len(outer_pts) == 1, "gap must expose a unique outer end"
    dist = _distance_map(rot_mask, outer_pts[0])
    markers = []
    for rr, cc in np.argwhere(rot_mask):
        color = int(rot_patch[rr, cc])
        if color != frag["base_color"]:
            markers.append((int(dist[rr, cc]), color))
    markers.sort()
    seq = [color for _d, color in markers]
    assert seq, "fragment must contain markers"
    return seq


def _fill_rings(out: np.ndarray, frame: dict, colors: list[int]) -> None:
    r0, c0, r1, c1 = frame["bbox"]
    t = frame["thickness"]
    out[r0 : r1 + 1, c0 : c1 + 1][frame["gap_mask"]] = frame["color"]
    ih = r1 - r0 + 1 - 2 * t
    iw = c1 - c0 + 1 - 2 * t
    for rr in range(ih):
        for cc in range(iw):
            ring = min(rr, cc, ih - 1 - rr, iw - 1 - cc)
            out[r0 + t + rr, c0 + t + cc] = colors[ring % len(colors)]


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    frames = _extract_frames(g, bg)
    assert frames, "expected at least one frame"
    frame_colors = {frame["color"] for frame in frames}
    frags = _extract_fragments(g, bg, frame_colors)
    assert len(frags) == len(frames), "fragment count must match frame count"

    frame_by_canon = {frame["gap_canon"]: frame for frame in frames}
    frag_by_canon = {frag["canon"]: frag for frag in frags}
    assert set(frame_by_canon) == set(frag_by_canon), "gap shapes must uniquely match fragments"

    out = g.copy()
    out[(g != bg) & (~np.isin(g, list(frame_colors)))] = bg
    for canon, frame in frame_by_canon.items():
        frag = frag_by_canon[canon]
        seq = _read_marker_sequence(frame, frag)
        _fill_rings(out, frame, seq)
    return out


def _expand_true(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    out = mask.copy()
    for _ in range(radius):
        grown = out.copy()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                src_r0 = max(0, -dr)
                src_r1 = mask.shape[0] - max(0, dr)
                src_c0 = max(0, -dc)
                src_c1 = mask.shape[1] - max(0, dc)
                dst_r0 = max(0, dr)
                dst_r1 = dst_r0 + (src_r1 - src_r0)
                dst_c0 = max(0, dc)
                dst_c1 = dst_c0 + (src_c1 - src_c0)
                grown[dst_r0:dst_r1, dst_c0:dst_c1] |= out[src_r0:src_r1, src_c0:src_c1]
        out = grown
    return out


def _box_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + pad < br0
        or br1 + pad < ar0
        or ac1 + pad < bc0
        or bc1 + pad < ac0
    )


def _sample_frame_sizes(count: int) -> list[tuple[int, int]]:
    sizes = []
    for _ in range(count):
        height = int(RNG.integers(9, 15))
        width = int(RNG.integers(9, 15))
        sizes.append((height, width))
    sizes.sort(reverse=True)
    return sizes


def _choose_grid_size(frame_count: int) -> tuple[int, int]:
    if frame_count == 1:
        return int(RNG.integers(20, 27)), int(RNG.integers(20, 27))
    if frame_count == 2:
        return int(RNG.integers(24, 31)), int(RNG.integers(24, 31))
    return int(RNG.integers(28, 35)), int(RNG.integers(28, 35))


def _place_frame_boxes(height: int, width: int, sizes: list[tuple[int, int]]) -> list[tuple[int, int, int, int]] | None:
    boxes = []
    for box_h, box_w in sizes:
        placed = None
        for _ in range(200):
            r0 = int(RNG.integers(1, height - box_h))
            c0 = int(RNG.integers(1, width - box_w))
            cand = (r0, c0, r0 + box_h - 1, c0 + box_w - 1)
            if any(_box_overlap(cand, prev, pad=2) for prev in boxes):
                continue
            placed = cand
            break
        if placed is None:
            return None
        boxes.append(placed)
    return boxes


def _place_gap_in_frame(shape: list[tuple[int, int]], frame_h: int, frame_w: int, side: str) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    rot = SIDE_TO_ROT[side]
    cells = _rotate_coords(shape, rot)
    gh = max(r for r, _ in cells) + 1
    gw = max(c for _, c in cells) + 1
    if side == "top":
        row_off = 0
        col_off = int(RNG.integers(1, frame_w - gw))
    elif side == "bottom":
        row_off = frame_h - gh
        col_off = int(RNG.integers(1, frame_w - gw))
    elif side == "left":
        row_off = int(RNG.integers(1, frame_h - gh))
        col_off = 0
    else:
        row_off = int(RNG.integers(1, frame_h - gh))
        col_off = frame_w - gw
    placed = [(r + row_off, c + col_off) for r, c in cells]
    trunk = _rotate_coords([(0, 0), (1, 0), (2, 0)], rot)
    trunk = [(r + row_off, c + col_off) for r, c in trunk]
    return placed, trunk


def _render_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int, gap_pts: list[tuple[int, int]]) -> None:
    r0, c0, r1, c1 = bbox
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    band = _band_mask(sub.shape[0], sub.shape[1], FRAME_THICKNESS)
    sub[band] = color
    for rr, cc in gap_pts:
        sub[rr, cc] = BG


def _build_fragment(template: dict, marker_count: int, base_color: int, marker_colors: list[int], rotation: int) -> np.ndarray:
    mask = _mask_from_points(template["cells"])
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = base_color
    chosen_rows = sorted(RNG.choice(len(template["trunk"]), size=marker_count, replace=False))
    chosen_trunk = [template["trunk"][idx] for idx in chosen_rows]
    for (rr, cc), color in zip(chosen_trunk, marker_colors):
        patch[rr, cc] = color
    patch = np.rot90(patch, k=rotation)
    mask = np.rot90(mask, k=rotation)
    patch, _ = _crop_patch(patch, mask)
    return patch


def _place_patch(grid: np.ndarray, patch: np.ndarray, forbidden: np.ndarray) -> bool:
    mask = patch != BG
    ph, pw = patch.shape
    h, w = grid.shape
    choices = []
    for r0 in range(1, h - ph):
        for c0 in range(1, w - pw):
            view = forbidden[r0 : r0 + ph, c0 : c0 + pw]
            if np.any(view & mask):
                continue
            choices.append((r0, c0))
    if not choices:
        return False
    r0, c0 = choices[int(RNG.integers(len(choices)))]
    view = grid[r0 : r0 + ph, c0 : c0 + pw]
    view[mask] = patch[mask]
    patch_mask = np.zeros_like(grid, dtype=bool)
    patch_mask[r0 : r0 + ph, c0 : c0 + pw] = mask
    forbidden |= _expand_true(patch_mask, radius=1)
    return True


def _frame_reserved_mask(shape: tuple[int, int], boxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    h, w = shape
    for r0, c0, r1, c1 in boxes:
        rr0 = max(0, r0 - 1)
        cc0 = max(0, c0 - 1)
        rr1 = min(h - 1, r1 + 1)
        cc1 = min(w - 1, c1 + 1)
        mask[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
    return mask


def _gen_attempt() -> np.ndarray | None:
    frame_count = int(RNG.choice([1, 2, 2, 3, 3]))
    grid_h, grid_w = _choose_grid_size(frame_count)
    sizes = _sample_frame_sizes(frame_count)
    boxes = _place_frame_boxes(grid_h, grid_w, sizes)
    if boxes is None:
        return None

    grid = np.zeros((grid_h, grid_w), dtype=int)
    all_colors = list(range(1, 10))
    frame_colors = list(RNG.choice(all_colors, size=frame_count, replace=False))
    free_colors = [c for c in all_colors if c not in frame_colors]
    templates = list(RNG.choice(len(GAP_TEMPLATES), size=frame_count, replace=False))

    fragment_specs = []
    for idx, bbox in enumerate(boxes):
        r0, c0, r1, c1 = bbox
        frame_h = r1 - r0 + 1
        frame_w = c1 - c0 + 1
        template = GAP_TEMPLATES[int(templates[idx])]
        side = str(RNG.choice(["top", "left", "bottom", "right"]))
        gap_pts, trunk_pts = _place_gap_in_frame(template["cells"], frame_h, frame_w, side)
        _render_frame(grid, bbox, frame_colors[idx], gap_pts)

        base_color = int(RNG.choice(free_colors))
        max_markers = min(3, len(template["cells"]) - 2)
        marker_count = int(RNG.integers(1, max_markers + 1))
        marker_palette = [c for c in free_colors if c != base_color]
        marker_colors = list(RNG.choice(marker_palette, size=marker_count, replace=False))
        fragment_specs.append(
            {
                "template": template,
                "rotation": int(RNG.integers(0, 4)),
                "base_color": base_color,
                "marker_colors": marker_colors,
                "marker_count": marker_count,
                "ring_colors": [color for _, color in sorted(zip([p[0] for p in trunk_pts[:marker_count]], marker_colors))],
            }
        )

    forbidden = _frame_reserved_mask(grid.shape, boxes)
    for spec in fragment_specs:
        patch = _build_fragment(
            spec["template"],
            spec["marker_count"],
            spec["base_color"],
            spec["marker_colors"],
            spec["rotation"],
        )
        if not _place_patch(grid, patch, forbidden):
            return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if out.shape != grid.shape:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
