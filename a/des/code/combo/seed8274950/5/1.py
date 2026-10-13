from __future__ import annotations

from collections import Counter

import numpy as np


RED = 2
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_RETRIES = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    r0, c0, _, _ = _bbox(coords)
    return tuple(sorted((r - r0, c - c0) for r, c in coords))


def _shape_dims(coords: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _ in coords) + 1, max(c for _, c in coords) + 1


def _all_orientations(coords: tuple[tuple[int, int], ...]) -> list[tuple[tuple[int, int], ...]]:
    variants = set()
    for flip in (False, True):
        for rot in range(4):
            cur = []
            for r, c in coords:
                rr, cc = (r, -c) if flip else (r, c)
                for _ in range(rot):
                    rr, cc = cc, -rr
                cur.append((rr, cc))
            variants.add(_normalize(cur))
    return sorted(variants)


def _canonical(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    return min(_all_orientations(_normalize(coords)))


def _paint_shape(grid: np.ndarray, top: int, left: int, coords: tuple[tuple[int, int], ...], color: int) -> None:
    for dr, dc in coords:
        grid[top + dr, left + dc] = color


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + pad < br0 - pad
        or br1 + pad < ar0 - pad
        or ac1 + pad < bc0 - pad
        or bc1 + pad < ac0 - pad
    )


def _non_bg_components(grid: np.ndarray, bg: int) -> list[dict]:
    comps: list[dict] = []
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        for pts in _components(grid == color):
            comps.append(
                {
                    "color": color,
                    "coords": pts,
                    "area": len(pts),
                    "bbox": _bbox(pts),
                }
            )
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each outside sprite to the same-shaped hole in the large template,
    fill the hole, recolor lower-half matches to red, and erase the sprites.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    comps = _non_bg_components(g, bg)
    assert len(comps) >= 2

    max_area = max(comp["area"] for comp in comps)
    frame_candidates = [comp for comp in comps if comp["area"] == max_area]
    assert len(frame_candidates) == 1
    frame = frame_candidates[0]
    frame_color = int(frame["color"])
    fr0, fc0, fr1, fc1 = frame["bbox"]

    sub = g[fr0 : fr1 + 1, fc0 : fc1 + 1]
    holes = []
    for pts in _components(sub == bg):
        if any(r in (0, sub.shape[0] - 1) or c in (0, sub.shape[1] - 1) for r, c in pts):
            continue
        abs_pts = [(fr0 + r, fc0 + c) for r, c in pts]
        holes.append(
            {
                "coords": abs_pts,
                "signature": _canonical(abs_pts),
                "bbox": _bbox(abs_pts),
            }
        )
    assert holes

    sprites = [comp for comp in comps if comp is not frame]
    assert len(sprites) == len(holes)

    hole_by_shape: dict[tuple[tuple[int, int], ...], dict] = {}
    for hole in holes:
        sig = hole["signature"]
        assert sig not in hole_by_shape
        hole_by_shape[sig] = hole

    sprite_by_shape: dict[tuple[tuple[int, int], ...], dict] = {}
    for sprite in sprites:
        sig = _canonical(sprite["coords"])
        assert sig not in sprite_by_shape
        sprite_by_shape[sig] = sprite

    assert set(hole_by_shape) == set(sprite_by_shape)

    out = np.full_like(g, bg)
    for r, c in frame["coords"]:
        out[r, c] = frame_color

    mid_row = (fr0 + fr1) / 2.0
    for sig, hole in hole_by_shape.items():
        hr0, _hc0, hr1, _hc1 = hole["bbox"]
        hole_center = (hr0 + hr1) / 2.0
        fill_color = int(sprite_by_shape[sig]["color"]) if hole_center < mid_row else RED
        for r, c in hole["coords"]:
            out[r, c] = fill_color

    return out


def _random_shape(rng: np.random.Generator, max_dim: int) -> tuple[tuple[int, int], ...] | None:
    for _ in range(100):
        target = int(rng.integers(3, min(7, max_dim * max_dim) + 1))
        cells = {(0, 0)}
        while len(cells) < target:
            frontier = set()
            for r, c in cells:
                for dr, dc in DIRS4:
                    frontier.add((r + dr, c + dc))
            frontier.difference_update(cells)
            candidates = []
            for nr, nc in frontier:
                trial = cells | {(nr, nc)}
                norm = _normalize(tuple(trial))
                h, w = _shape_dims(norm)
                if h <= max_dim and w <= max_dim:
                    candidates.append((nr, nc))
            if not candidates:
                break
            cells.add(candidates[int(rng.integers(len(candidates)))])
        if len(cells) != target:
            continue
        shape = _normalize(tuple(cells))
        h, w = _shape_dims(shape)
        if h < 2 or w < 2:
            continue
        return shape
    return None


def _spec_output(
    shape_entries: list[dict],
    grid_shape: tuple[int, int],
    bg: int,
    frame_color: int,
    frame_bbox: tuple[int, int, int, int],
) -> np.ndarray:
    out = np.full(grid_shape, bg, dtype=int)
    fr0, fc0, fr1, fc1 = frame_bbox
    out[fr0 : fr1 + 1, fc0 : fc1 + 1] = frame_color
    mid_row = (fr0 + fr1) / 2.0
    for entry in shape_entries:
        hr0, _hc0, hr1, _hc1 = entry["hole_bbox"]
        hole_center = (hr0 + hr1) / 2.0
        fill_color = entry["sprite_color"] if hole_center < mid_row else RED
        for r, c in entry["hole_abs"]:
            out[r, c] = fill_color
    return out


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    hole_count = int(rng.choice([3, 4, 5]))
    frame_h = int(rng.integers(12 + (hole_count == 5), 17))
    frame_w = int(rng.integers(14 + (hole_count >= 4), 22))
    grid_h = frame_h + int(rng.integers(8, 13))
    grid_w = frame_w + int(rng.integers(10, 15))
    if grid_h < frame_h + 6 or grid_w < frame_w + 6:
        return None

    bg_choices = [c for c in range(10) if c != RED]
    bg = int(rng.choice(bg_choices))
    frame_choices = [c for c in range(10) if c not in (bg, RED)]
    frame_color = int(rng.choice(frame_choices))
    sprite_palette = [c for c in range(10) if c not in (bg, frame_color, RED)]
    if len(sprite_palette) < hole_count:
        return None
    sprite_colors = list(map(int, rng.choice(sprite_palette, size=hole_count, replace=False)))

    fr0_lo = 2
    fr0_hi = grid_h - frame_h - 2
    fc0_lo = 2
    fc0_hi = grid_w - frame_w - 2
    if fr0_lo > fr0_hi or fc0_lo > fc0_hi:
        return None
    fr0_center = (grid_h - frame_h) // 2
    fc0_center = (grid_w - frame_w) // 2
    fr0 = int(np.clip(fr0_center + int(rng.integers(-1, 2)), fr0_lo, fr0_hi))
    fc0 = int(np.clip(fc0_center + int(rng.integers(-2, 3)), fc0_lo, fc0_hi))
    frame_bbox = (fr0, fc0, fr0 + frame_h - 1, fc0 + frame_w - 1)

    if hole_count % 2 == 0:
        top_count = hole_count // 2
    else:
        top_count = hole_count // 2 + int(rng.integers(0, 2))
    bottom_count = hole_count - top_count
    sides = ["top"] * top_count + ["bottom"] * bottom_count
    rng.shuffle(sides)

    max_dim = 3 if hole_count == 5 else 4
    seen_shapes = set()
    shape_entries = []
    for side, color in zip(sides, sprite_colors):
        shape = _random_shape(rng, max_dim=max_dim)
        if shape is None:
            return None
        sig = _canonical(shape)
        tries = 0
        while sig in seen_shapes and tries < 50:
            shape = _random_shape(rng, max_dim=max_dim)
            if shape is None:
                return None
            sig = _canonical(shape)
            tries += 1
        if sig in seen_shapes:
            return None
        seen_shapes.add(sig)
        shape_entries.append({"side": side, "sprite_color": int(color), "shape": shape, "signature": sig})

    shape_entries.sort(
        key=lambda entry: _shape_dims(entry["shape"])[0] * _shape_dims(entry["shape"])[1],
        reverse=True,
    )

    used_hole_boxes: list[tuple[int, int, int, int]] = []
    mid = (frame_h - 1) / 2.0
    top_band_end = int(np.floor(mid))
    bottom_band_start = int(np.ceil(mid))
    for entry in shape_entries:
        shape = entry["shape"]
        sh, sw = _shape_dims(shape)
        if entry["side"] == "top":
            row_lo, row_hi = 1, top_band_end - sh + 1
        else:
            row_lo, row_hi = bottom_band_start, frame_h - sh - 1
        col_lo, col_hi = 1, frame_w - sw - 1
        if row_lo > row_hi or col_lo > col_hi:
            return None
        placed = False
        for _ in range(200):
            rr = int(rng.integers(row_lo, row_hi + 1))
            cc = int(rng.integers(col_lo, col_hi + 1))
            hole_box_rel = (rr, cc, rr + sh - 1, cc + sw - 1)
            if any(_boxes_overlap(hole_box_rel, other, pad=1) for other in used_hole_boxes):
                continue
            hole_center = (hole_box_rel[0] + hole_box_rel[2]) / 2.0
            if entry["side"] == "top" and not (hole_center < mid):
                continue
            if entry["side"] == "bottom" and not (hole_center > mid):
                continue
            used_hole_boxes.append(hole_box_rel)
            hole_abs = tuple((fr0 + rr + dr, fc0 + cc + dc) for dr, dc in shape)
            entry["hole_abs"] = hole_abs
            entry["hole_bbox"] = _bbox(hole_abs)
            placed = True
            break
        if not placed:
            return None

    inp = np.full((grid_h, grid_w), bg, dtype=int)
    inp[fr0 : fr0 + frame_h, fc0 : fc0 + frame_w] = frame_color
    for entry in shape_entries:
        for r, c in entry["hole_abs"]:
            inp[r, c] = bg
    if len(_components(inp == frame_color)) != 1:
        return None

    used_sprite_boxes = [frame_bbox]
    for entry in shape_entries:
        shape = entry["shape"]
        variants = _all_orientations(shape)
        sprite_shape = variants[int(rng.integers(len(variants)))]
        sh, sw = _shape_dims(sprite_shape)
        placed = False
        for _ in range(300):
            sr = int(rng.integers(1, grid_h - sh - 1))
            sc = int(rng.integers(1, grid_w - sw - 1))
            sprite_box = (sr, sc, sr + sh - 1, sc + sw - 1)
            if any(_boxes_overlap(sprite_box, other, pad=1) for other in used_sprite_boxes):
                continue
            used_sprite_boxes.append(sprite_box)
            entry["sprite_abs"] = tuple((sr + dr, sc + dc) for dr, dc in sprite_shape)
            placed = True
            break
        if not placed:
            return None
        for r, c in entry["sprite_abs"]:
            inp[r, c] = entry["sprite_color"]

    counts = Counter(map(int, inp.ravel()))
    if counts[bg] <= max(count for color, count in counts.items() if color != bg):
        return None

    if len(_components(inp == frame_color)) != 1:
        return None

    expected = _spec_output(shape_entries, inp.shape, bg, frame_color, frame_bbox)
    try:
        solved = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
