from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


_DIRECTIONS = ("top", "bottom", "left", "right")
_MAX_GROUPS = 4
_MAX_ATTEMPTS = 400


def _background_color(grid: np.ndarray) -> int:
    corners = [int(grid[0, 0]), int(grid[0, -1]), int(grid[-1, 0]), int(grid[-1, -1])]
    if len(set(corners)) == 1:
        return corners[0]
    return Counter(map(int, grid.flat)).most_common(1)[0][0]


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _draw_frame(grid: np.ndarray, r0: int, c0: int, h: int, w: int, color: int) -> None:
    grid[r0, c0 : c0 + w] = color
    grid[r0 + h - 1, c0 : c0 + w] = color
    grid[r0 : r0 + h, c0] = color
    grid[r0 : r0 + h, c0 + w - 1] = color


def _erase_gap(
    grid: np.ndarray,
    r0: int,
    c0: int,
    h: int,
    w: int,
    side: str,
    span: tuple[int, int],
    bg: int,
) -> None:
    a, b = span
    if side == "top":
        grid[r0, c0 + a : c0 + b + 1] = bg
    elif side == "bottom":
        grid[r0 + h - 1, c0 + a : c0 + b + 1] = bg
    elif side == "left":
        grid[r0 + a : r0 + b + 1, c0] = bg
    else:
        grid[r0 + a : r0 + b + 1, c0 + w - 1] = bg


def _frame_info(grid: np.ndarray, color: int, coords: np.ndarray, bg: int) -> dict[str, object] | None:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None

    mask = np.zeros((h, w), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    if np.any(mask & ~border):
        return None

    missing = border & ~mask
    if not np.any(missing):
        return {
            "kind": "closed",
            "color": int(color),
            "bbox": (r0, c0, r1, c1),
            "coords": coords,
        }

    side_masks = {
        "top": missing[0, :],
        "bottom": missing[-1, :],
        "left": missing[:, 0],
        "right": missing[:, -1],
    }
    sides = [side for side, line in side_masks.items() if np.any(line)]
    if len(sides) != 1:
        return None

    side = sides[0]
    idx = np.flatnonzero(side_masks[side])
    if idx.size == 0 or int(idx[-1] - idx[0] + 1) != int(idx.size):
        return None

    gap_mask = np.zeros((h, w), dtype=bool)
    if side == "top":
        gap_mask[0, idx[0] : idx[-1] + 1] = True
    elif side == "bottom":
        gap_mask[-1, idx[0] : idx[-1] + 1] = True
    elif side == "left":
        gap_mask[idx[0] : idx[-1] + 1, 0] = True
    else:
        gap_mask[idx[0] : idx[-1] + 1, -1] = True

    if not np.array_equal(missing, gap_mask):
        return None

    miss_rows, miss_cols = np.where(missing)
    if np.any(grid[r0 + miss_rows, c0 + miss_cols] != bg):
        return None

    return {
        "kind": "gap",
        "color": int(color),
        "bbox": (r0, c0, r1, c1),
        "coords": coords,
        "gap_side": side,
        "gap_span": (int(idx[0]), int(idx[-1])),
    }


def _inside_with_moat(outer_bbox: tuple[int, int, int, int], inner_bbox: tuple[int, int, int, int]) -> bool:
    or0, oc0, or1, oc1 = outer_bbox
    ir0, ic0, ir1, ic1 = inner_bbox
    return or0 + 1 < ir0 and ir1 < or1 - 1 and oc0 + 1 < ic0 and ic1 < oc1 - 1


def _gap_covers_inner(outer: dict[str, object], inner_bbox: tuple[int, int, int, int]) -> bool:
    or0, oc0, _or1, _oc1 = outer["bbox"]  # type: ignore[index]
    ir0, ic0, ir1, ic1 = inner_bbox
    gap0, gap1 = outer["gap_span"]  # type: ignore[index]
    side = outer["gap_side"]  # type: ignore[index]
    if side in {"top", "bottom"}:
        return oc0 + gap0 <= ic0 and ic1 <= oc0 + gap1
    return or0 + gap0 <= ir0 and ir1 <= or0 + gap1


def _destination_bbox(
    outer_bbox: tuple[int, int, int, int],
    inner_bbox: tuple[int, int, int, int],
    side: str,
) -> tuple[int, int, int, int]:
    or0, oc0, or1, oc1 = outer_bbox
    ir0, ic0, ir1, ic1 = inner_bbox
    ih = ir1 - ir0 + 1
    iw = ic1 - ic0 + 1
    if side == "top":
        return or0 - ih, ic0, or0 - 1, ic1
    if side == "bottom":
        return or1 + 1, ic0, or1 + ih, ic1
    if side == "left":
        return ir0, oc0 - iw, ir1, oc0 - 1
    return ir0, oc1 + 1, ir1, oc1 + iw


def _extract_groups(grid: np.ndarray) -> tuple[int, list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)

    closed_frames: list[dict[str, object]] = []
    gap_frames: list[dict[str, object]] = []
    for color in sorted(int(c) for c in np.unique(g) if int(c) != bg):
        for coords in _components(g == color):
            info = _frame_info(g, color, coords, bg)
            if info is None:
                continue
            if info["kind"] == "closed":
                closed_frames.append(info)
            else:
                gap_frames.append(info)

    groups: list[dict[str, object]] = []
    used_closed: set[int] = set()
    for outer in sorted(gap_frames, key=lambda item: item["bbox"]):  # type: ignore[index]
        candidates = [
            idx
            for idx, inner in enumerate(closed_frames)
            if idx not in used_closed
            and _inside_with_moat(outer["bbox"], inner["bbox"])  # type: ignore[index]
            and _gap_covers_inner(outer, inner["bbox"])  # type: ignore[index]
        ]
        assert len(candidates) == 1, "each gap frame must contain exactly one matching inner frame"

        inner = closed_frames[candidates[0]]
        used_closed.add(candidates[0])
        src_bbox = inner["bbox"]  # type: ignore[index]
        dst_bbox = _destination_bbox(outer["bbox"], src_bbox, outer["gap_side"])  # type: ignore[index]
        dr0, dc0, dr1, dc1 = dst_bbox
        assert 0 <= dr0 <= dr1 < g.shape[0]
        assert 0 <= dc0 <= dc1 < g.shape[1]

        sr0, sc0, sr1, sc1 = src_bbox
        groups.append(
            {
                "outer_bbox": outer["bbox"],
                "src_bbox": src_bbox,
                "dst_bbox": dst_bbox,
                "direction": outer["gap_side"],
                "patch": g[sr0 : sr1 + 1, sc0 : sc1 + 1].copy(),
            }
        )

    assert groups, "expected at least one nested-frame group"
    return bg, groups


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pop each inner rectangle frame out through the matching gap in its surrounding outer frame.

    Input structure:
    - The grid has a uniform background plus one or more nested-frame groups.
    - Each group contains a one-pixel-thick outer rectangular frame with exactly one continuous gap on one side.
    - Inside it sits a smaller closed one-pixel-thick inner rectangular frame, separated from the outer frame by background, and the inner frame may contain colored pixels.

    Rule:
    - Process each nested-frame group independently.
    - Read the outer frame's gap side as a direction: top, bottom, left, or right.
    - Move the entire inner-frame bounding patch, including the inner frame and any colored content inside it, straight through that side so it ends immediately outside the outer frame while keeping the orthogonal alignment unchanged.
    - Clear the original inner-frame location to background and keep the outer frame, including its gap, unchanged.

    Key insight:
    - The missing segment on the outer frame is not damage; it is an exit that determines both the movement direction and the exact stopping position of the whole inner patch.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg, groups = _extract_groups(g)
    out = g.copy()

    for group in groups:
        sr0, sc0, sr1, sc1 = group["src_bbox"]  # type: ignore[index]
        out[sr0 : sr1 + 1, sc0 : sc1 + 1] = bg

    for group in groups:
        dr0, dc0, dr1, dc1 = group["dst_bbox"]  # type: ignore[index]
        patch = group["patch"]  # type: ignore[index]
        mask = patch != bg
        target = out[dr0 : dr1 + 1, dc0 : dc1 + 1]
        target[mask] = patch[mask]

    return out


def _random_inner_patch(
    rng: random.Random,
    h: int,
    w: int,
    bg: int,
    frame_color: int,
) -> np.ndarray:
    patch = np.full((h, w), bg, dtype=int)
    _draw_frame(patch, 0, 0, h, w, frame_color)

    interior_cells = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
    if not interior_cells or rng.random() < 0.25:
        return patch

    max_cells = min(len(interior_cells), max(1, len(interior_cells) // 2 + 1))
    n_cells = rng.randint(1, max_cells)
    colors = [color for color in range(10) if color not in {bg, frame_color}]
    n_colors = rng.randint(1, min(3, len(colors)))
    palette = rng.sample(colors, n_colors)
    chosen = rng.sample(interior_cells, n_cells)
    for r, c in chosen:
        patch[r, c] = rng.choice(palette)
    return patch


def _sample_group_spec(
    rng: random.Random,
    bg: int,
    outer_color: int,
    inner_color: int,
    direction: str,
) -> dict[str, object]:
    inner_h = rng.randint(3, 6)
    inner_w = rng.randint(3, 6)
    outer_h = rng.randint(inner_h + 4, min(inner_h + 7, 11))
    outer_w = rng.randint(inner_w + 4, min(inner_w + 7, 11))

    inner_r = rng.randint(2, outer_h - inner_h - 2)
    inner_c = rng.randint(2, outer_w - inner_w - 2)

    if direction in {"top", "bottom"}:
        gap0 = rng.randint(1, inner_c)
        gap1 = rng.randint(inner_c + inner_w - 1, outer_w - 2)
        reserve_h = outer_h + inner_h
        reserve_w = outer_w
        outer_origin = (inner_h, 0) if direction == "top" else (0, 0)
    else:
        gap0 = rng.randint(1, inner_r)
        gap1 = rng.randint(inner_r + inner_h - 1, outer_h - 2)
        reserve_h = outer_h
        reserve_w = outer_w + inner_w
        outer_origin = (0, inner_w) if direction == "left" else (0, 0)

    return {
        "direction": direction,
        "outer_color": outer_color,
        "inner_color": inner_color,
        "outer_h": outer_h,
        "outer_w": outer_w,
        "inner_h": inner_h,
        "inner_w": inner_w,
        "inner_r": inner_r,
        "inner_c": inner_c,
        "gap_span": (gap0, gap1),
        "reserve_h": reserve_h,
        "reserve_w": reserve_w,
        "outer_origin": outer_origin,
        "inner_patch": _random_inner_patch(rng, inner_h, inner_w, bg, inner_color),
    }


def _render_group(grid: np.ndarray, top: int, left: int, spec: dict[str, object], bg: int) -> None:
    outer_r0 = top + spec["outer_origin"][0]  # type: ignore[index]
    outer_c0 = left + spec["outer_origin"][1]  # type: ignore[index]
    outer_h = spec["outer_h"]  # type: ignore[index]
    outer_w = spec["outer_w"]  # type: ignore[index]
    direction = spec["direction"]  # type: ignore[index]
    outer_color = spec["outer_color"]  # type: ignore[index]
    gap_span = spec["gap_span"]  # type: ignore[index]

    _draw_frame(grid, outer_r0, outer_c0, outer_h, outer_w, outer_color)
    _erase_gap(grid, outer_r0, outer_c0, outer_h, outer_w, direction, gap_span, bg)

    inner_r0 = outer_r0 + spec["inner_r"]  # type: ignore[index]
    inner_c0 = outer_c0 + spec["inner_c"]  # type: ignore[index]
    patch = spec["inner_patch"]  # type: ignore[index]
    mask = patch != bg
    target = grid[inner_r0 : inner_r0 + patch.shape[0], inner_c0 : inner_c0 + patch.shape[1]]
    target[mask] = patch[mask]


def _reserve_free(blocked: np.ndarray, top: int, left: int, h: int, w: int) -> bool:
    return not np.any(blocked[top - 1 : top + h + 1, left - 1 : left + w + 1])


def _mark_reserved(blocked: np.ndarray, top: int, left: int, h: int, w: int) -> None:
    blocked[top - 1 : top + h + 1, left - 1 : left + w + 1] = True


def _generate_with_count(rng: random.Random, count: int) -> np.ndarray | None:
    bg = rng.randint(0, 9)
    min_dim = {1: 12, 2: 16, 3: 20, 4: 24}[count]
    h = rng.randint(min_dim, 30)
    w = rng.randint(min_dim, 30)

    palette = [color for color in range(10) if color != bg]
    rng.shuffle(palette)
    directions = list(_DIRECTIONS)
    rng.shuffle(directions)

    grid = np.full((h, w), bg, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    specs: list[dict[str, object]] = []
    for idx in range(count):
        outer_color = palette[2 * idx]
        inner_color = palette[2 * idx + 1]
        spec = _sample_group_spec(rng, bg, outer_color, inner_color, directions[idx])
        reserve_h = spec["reserve_h"]  # type: ignore[index]
        reserve_w = spec["reserve_w"]  # type: ignore[index]
        if reserve_h > h - 2 or reserve_w > w - 2:
            return None

        placed = False
        for _ in range(120):
            top = rng.randint(1, h - reserve_h - 1)
            left = rng.randint(1, w - reserve_w - 1)
            if not _reserve_free(blocked, top, left, reserve_h, reserve_w):
                continue
            _mark_reserved(blocked, top, left, reserve_h, reserve_w)
            _render_group(grid, top, left, spec, bg)
            specs.append(spec)
            placed = True
            break
        if not placed:
            return None

    try:
        parsed_bg, groups = _extract_groups(grid)
        if parsed_bg != bg or len(groups) != count:
            return None
        _ = main(grid)
    except Exception:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    target_count = rng.randint(1, _MAX_GROUPS)
    for _ in range(_MAX_ATTEMPTS):
        grid = _generate_with_count(rng, target_count)
        if grid is not None:
            return grid
    for _ in range(_MAX_ATTEMPTS):
        grid = _generate_with_count(rng, rng.randint(1, _MAX_GROUPS))
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed to construct a valid nested-frame puzzle")
