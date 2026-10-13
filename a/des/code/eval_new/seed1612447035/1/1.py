from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GAP_TO_TURNS = {"top": 0, "right": 1, "bottom": 2, "left": 3}
MARKER_COLOR = 5
GEN_RETRIES = 400
BOX_SPEC_RETRIES = 120
PLACEMENT_RETRIES = 200


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _rotate_cw(arr: np.ndarray, turns: int) -> np.ndarray:
    turns %= 4
    if turns == 0:
        return arr.copy()
    return np.rot90(arr, -turns)


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return None
    r0 = int(pts[:, 0].min())
    c0 = int(pts[:, 1].min())
    r1 = int(pts[:, 0].max())
    c1 = int(pts[:, 1].max())
    return r0, c0, r1, c1


def _crop_non_bg(arr: np.ndarray, bg: int) -> np.ndarray:
    box = _bbox_from_mask(arr != bg)
    if box is None:
        return np.zeros((0, 0), dtype=int)
    r0, c0, r1, c1 = box
    return arr[r0 : r1 + 1, c0 : c1 + 1].copy()


def _center_interior(patch: np.ndarray, shape: tuple[int, int], bg: int) -> np.ndarray:
    h, w = shape
    canvas = np.full((h, w), bg, dtype=int)
    if patch.size == 0:
        return canvas
    ph, pw = patch.shape
    top = (h - ph) // 2
    left = (w - pw) // 2
    mask = patch != bg
    canvas[top : top + ph, left : left + pw][mask] = patch[mask]
    return canvas


def _same_color_components(grid: np.ndarray, bg: int) -> list[tuple[int, list[tuple[int, int]]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[tuple[int, list[tuple[int, int]]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c]:
                continue
            color = int(grid[r, c])
            if color == bg:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            pts = [(r, c)]
            while q:
                cr, cc = q.popleft()
                for dr, dc in DIR4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
                        pts.append((nr, nc))
            comps.append((color, pts))
    return comps


def _is_valid_frame(
    grid: np.ndarray, bg: int, color: int, pts: list[tuple[int, int]]
) -> dict | None:
    rows = [r for r, _ in pts]
    cols = [c for _, c in pts]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    outer_h = r1 - r0 + 1
    outer_w = c1 - c0 + 1
    if outer_h < 4 or outer_w < 4:
        return None

    border = set()
    for c in range(c0, c1 + 1):
        border.add((r0, c))
        border.add((r1, c))
    for r in range(r0, r1 + 1):
        border.add((r, c0))
        border.add((r, c1))

    comp = set(pts)
    if not comp.issubset(border):
        return None

    missing = [cell for cell in border if cell not in comp]
    if len(missing) != 1:
        return None
    gap_r, gap_c = missing[0]
    if gap_r in (r0, r1) and gap_c in (c0, c1):
        return None
    if gap_r == r0:
        gap_side = "top"
    elif gap_r == r1:
        gap_side = "bottom"
    elif gap_c == c0:
        gap_side = "left"
    elif gap_c == c1:
        gap_side = "right"
    else:
        return None

    marker_positions = [
        (r0 - 1, c0 - 1),
        (r0 - 1, c1 + 1),
        (r1 + 1, c0 - 1),
        (r1 + 1, c1 + 1),
    ]
    h, w = grid.shape
    if any(not (0 <= r < h and 0 <= c < w) for r, c in marker_positions):
        return None
    marker_colors = [int(grid[r, c]) for r, c in marker_positions]
    if any(col == bg for col in marker_colors):
        return None
    if len(set(marker_colors)) != 1:
        return None

    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    if not np.any(interior != bg):
        return None

    return {
        "bbox": (r0, c0, r1, c1),
        "frame_color": color,
        "marker_color": marker_colors[0],
        "gap_side": gap_side,
    }


def _find_boxes(grid: np.ndarray) -> list[dict]:
    bg = _get_bg(grid)
    boxes = []
    for color, pts in _same_color_components(grid, bg):
        cand = _is_valid_frame(grid, bg, color, pts)
        if cand is not None:
            boxes.append(cand)
    boxes.sort(key=lambda box: box["bbox"])
    return boxes


def _quadrant_slices(h: int, w: int) -> list[tuple[str, tuple[slice, slice]]]:
    mid_r = h // 2
    mid_c = w // 2
    return [
        ("tl", (slice(0, mid_r), slice(0, mid_c))),
        ("tr", (slice(0, mid_r), slice(mid_c, w))),
        ("bl", (slice(mid_r, h), slice(0, mid_c))),
        ("br", (slice(mid_r, h), slice(mid_c, w))),
    ]


def _quadrant_majorities(arr: np.ndarray, bg: int) -> dict[str, int | None] | None:
    info: dict[str, int | None] = {}
    for name, (rs, cs) in _quadrant_slices(*arr.shape):
        vals = arr[rs, cs]
        vals = vals[vals != bg]
        if vals.size == 0:
            info[name] = None
            continue
        counts = Counter(map(int, vals.tolist()))
        top = counts.most_common()
        if len(top) > 1 and top[0][1] == top[1][1]:
            return None
        info[name] = int(top[0][0])
    return info


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each boxed interior by the border-gap direction, then recolor corner markers from the rotated quadrant majorities.

    Input structure:
    - bg is the most common color.
    - Several separate hollow rectangular boxes appear on the grid; each box is a single-color border with exactly one 1-cell gap on one side.
    - Each box has four singleton corner markers placed one cell diagonally outside its corners, all in the same marker color.
    - Inside each box is a small two-color asymmetric pattern occupying part of the interior.

    Rule:
    - For each box, read the missing border side as a clockwise rotation amount for the interior pattern: top=0, right=90, bottom=180, left=270 degrees.
    - Crop the interior pattern to its tight non-background bounding box, rotate it by that amount, and place the rotated pattern back into the box interior centered on the available interior area.
    - Inspect the rotated centered pattern by interior quadrants; every quadrant that contains pattern pixels recolors the matching outer corner marker to that quadrant's majority pattern color.
    - Quadrants with no rotated pattern pixels leave their corner marker unchanged in the original marker color.

    Key insight:
    - The border gap encodes a rotation, not a destination; only after rotating and centering the interior pattern do the occupied quadrants determine which outside corner markers activate and what colors they take.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    out = g.copy()

    for box in _find_boxes(g):
        r0, c0, r1, c1 = box["bbox"]
        marker_color = int(box["marker_color"])
        turns = GAP_TO_TURNS[box["gap_side"]]

        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        patch = _crop_non_bg(interior, bg)
        rotated = _rotate_cw(patch, turns)
        centered = _center_interior(rotated, interior.shape, bg)

        out[r0 + 1 : r1, c0 + 1 : c1] = bg
        mask = centered != bg
        out[r0 + 1 : r1, c0 + 1 : c1][mask] = centered[mask]

        marker_map = {
            "tl": (r0 - 1, c0 - 1),
            "tr": (r0 - 1, c1 + 1),
            "bl": (r1 + 1, c0 - 1),
            "br": (r1 + 1, c1 + 1),
        }
        majorities = _quadrant_majorities(centered, bg)
        for name, pos in marker_map.items():
            out[pos] = marker_color
            if majorities is not None and majorities[name] is not None:
                out[pos] = int(majorities[name])

    return out


def _random_connected_mask(rng: random.Random, h: int, w: int) -> np.ndarray:
    target_lo = max(4, (h * w) // 3)
    target_hi = max(target_lo, min(h * w - 1, (2 * h * w) // 3 + 1))
    target = rng.randint(target_lo, target_hi)

    start = (rng.randrange(h), rng.randrange(w))
    cells = {start}
    frontier = set()
    for dr, dc in DIR4:
        nr = start[0] + dr
        nc = start[1] + dc
        if 0 <= nr < h and 0 <= nc < w:
            frontier.add((nr, nc))

    while len(cells) < target and frontier:
        candidates = list(frontier)
        rng.shuffle(candidates)
        best = max(
            candidates,
            key=lambda p: (
                sum((p[0] + dr, p[1] + dc) in cells for dr, dc in DIR4),
                rng.random(),
            ),
        )
        frontier.remove(best)
        cells.add(best)
        for dr, dc in DIR4:
            nr = best[0] + dr
            nc = best[1] + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                frontier.add((nr, nc))

    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    box = _bbox_from_mask(mask)
    assert box is not None
    r0, c0, r1, c1 = box
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _is_rotationally_asymmetric(mask: np.ndarray) -> bool:
    base = mask.astype(int)
    for turns in (1, 2, 3):
        rot = _rotate_cw(base, turns)
        if rot.shape == base.shape and np.array_equal(rot, base):
            return False
    return True


def _paint_mask(mask: np.ndarray, bg: int, color_a: int, color_b: int, rng: random.Random) -> np.ndarray:
    pts = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    anchor_a = pts[rng.randrange(len(pts))]
    anchor_b = pts[rng.randrange(len(pts))]
    patch = np.full(mask.shape, bg, dtype=int)
    for r, c in pts:
        da = abs(r - anchor_a[0]) + abs(c - anchor_a[1]) + rng.randint(0, 1)
        db = abs(r - anchor_b[0]) + abs(c - anchor_b[1]) + rng.randint(0, 1)
        patch[r, c] = color_a if da <= db else color_b
    vals = {int(v) for v in patch[mask]}
    if vals == {color_a} or vals == {color_b}:
        rr, cc = pts[rng.randrange(len(pts))]
        patch[rr, cc] = color_b if patch[rr, cc] == color_a else color_a
    return patch


def _all_offsets(h: int, w: int, ph: int, pw: int) -> list[tuple[int, int]]:
    return [(r, c) for r in range(h - ph + 1) for c in range(w - pw + 1)]


def _choose_offset(
    rng: random.Random, h: int, w: int, ph: int, pw: int
) -> tuple[int, int]:
    offsets = _all_offsets(h, w, ph, pw)
    center = ((h - ph) // 2, (w - pw) // 2)
    if len(offsets) > 1 and rng.random() < 0.7:
        offsets = [off for off in offsets if off != center] or offsets
    return offsets[rng.randrange(len(offsets))]


def _make_pattern(
    rng: random.Random,
    inner_h: int,
    inner_w: int,
    turns: int,
    bg: int,
    frame_color: int,
) -> tuple[np.ndarray, tuple[int, int]]:
    lim = min(5, min(inner_h, inner_w) - 1)
    colors = [c for c in range(10) if c not in (bg, MARKER_COLOR, frame_color)]
    for _ in range(BOX_SPEC_RETRIES):
        ph = rng.randint(2, lim)
        pw = rng.randint(2, lim)
        mask = _random_connected_mask(rng, ph, pw)
        if not _is_rotationally_asymmetric(mask):
            continue

        color_a, color_b = rng.sample(colors, 2)
        patch = _paint_mask(mask, bg, color_a, color_b, rng)
        patch = _rotate_cw(patch, rng.randrange(4))
        centered = _center_interior(_rotate_cw(patch, turns), (inner_h, inner_w), bg)
        majorities = _quadrant_majorities(centered, bg)
        if majorities is None:
            continue
        covered = sum(val is not None for val in majorities.values())
        if covered == 0 or covered == 4:
            continue

        off_r, off_c = _choose_offset(rng, inner_h, inner_w, patch.shape[0], patch.shape[1])
        return patch, (off_r, off_c)
    raise RuntimeError("failed to generate a valid asymmetric pattern")


def _make_box_spec(
    rng: random.Random,
    bg: int,
    frame_color: int,
    gap_side: str,
    inner_choices: list[tuple[int, int]],
) -> dict:
    inner_h, inner_w = inner_choices[rng.randrange(len(inner_choices))]
    outer_h = inner_h + 2
    outer_w = inner_w + 2
    turns = GAP_TO_TURNS[gap_side]
    patch, offset = _make_pattern(rng, inner_h, inner_w, turns, bg, frame_color)

    if gap_side in ("top", "bottom"):
        gap_index = rng.randint(1, outer_w - 2)
    else:
        gap_index = rng.randint(1, outer_h - 2)

    return {
        "inner_h": inner_h,
        "inner_w": inner_w,
        "outer_h": outer_h,
        "outer_w": outer_w,
        "frame_color": frame_color,
        "gap_side": gap_side,
        "gap_index": gap_index,
        "patch": patch,
        "offset": offset,
    }


def _draw_box(grid: np.ndarray, spec: dict, top: int, left: int, bg: int) -> None:
    outer_h = int(spec["outer_h"])
    outer_w = int(spec["outer_w"])
    frame_color = int(spec["frame_color"])
    gap_side = spec["gap_side"]
    gap_index = int(spec["gap_index"])
    patch = spec["patch"]
    off_r, off_c = spec["offset"]

    bottom = top + outer_h - 1
    right = left + outer_w - 1

    grid[top : bottom + 1, left : right + 1] = bg
    grid[top, left : right + 1] = frame_color
    grid[bottom, left : right + 1] = frame_color
    grid[top : bottom + 1, left] = frame_color
    grid[top : bottom + 1, right] = frame_color

    if gap_side == "top":
        grid[top, left + gap_index] = bg
    elif gap_side == "bottom":
        grid[bottom, left + gap_index] = bg
    elif gap_side == "left":
        grid[top + gap_index, left] = bg
    else:
        grid[top + gap_index, right] = bg

    ph, pw = patch.shape
    mask = patch != bg
    region = grid[top + 1 + off_r : top + 1 + off_r + ph, left + 1 + off_c : left + 1 + off_c + pw]
    region[mask] = patch[mask]

    grid[top - 1, left - 1] = MARKER_COLOR
    grid[top - 1, right + 1] = MARKER_COLOR
    grid[bottom + 1, left - 1] = MARKER_COLOR
    grid[bottom + 1, right + 1] = MARKER_COLOR


def _place_specs(grid: np.ndarray, specs: list[dict], bg: int, rng: random.Random) -> bool:
    h, w = grid.shape
    blocked = np.zeros((h, w), dtype=bool)
    placed = []

    order = sorted(range(len(specs)), key=lambda i: specs[i]["outer_h"] * specs[i]["outer_w"], reverse=True)
    for idx in order:
        spec = specs[idx]
        outer_h = int(spec["outer_h"])
        outer_w = int(spec["outer_w"])
        if h < outer_h + 4 or w < outer_w + 4:
            return False

        ok = False
        for _ in range(PLACEMENT_RETRIES):
            top = rng.randint(1, h - outer_h - 1)
            left = rng.randint(1, w - outer_w - 1)
            rr0 = max(0, top - 2)
            rr1 = min(h, top + outer_h + 2)
            cc0 = max(0, left - 2)
            cc1 = min(w, left + outer_w + 2)
            if blocked[rr0:rr1, cc0:cc1].any():
                continue
            blocked[rr0:rr1, cc0:cc1] = True
            placed.append((idx, top, left))
            ok = True
            break
        if not ok:
            return False

    for idx, top, left in placed:
        _draw_box(grid, specs[idx], top, left, bg)
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    bg = rng.choice([c for c in range(10) if c != MARKER_COLOR])
    n_boxes = rng.choices([1, 2, 3, 4], weights=[1, 3, 3, 2], k=1)[0]
    inner_choices = [(4, 4), (4, 6), (6, 4), (6, 6)]
    if n_boxes <= 2:
        inner_choices += [(4, 8), (8, 4), (6, 8), (8, 6)]

    frame_palette = [c for c in range(10) if c not in (bg, MARKER_COLOR)]
    rng.shuffle(frame_palette)
    frame_colors = frame_palette[:n_boxes]
    gap_sides = [rng.choice(list(GAP_TO_TURNS)) for _ in range(n_boxes)]

    specs = []
    for i in range(n_boxes):
        specs.append(_make_box_spec(rng, bg, frame_colors[i], gap_sides[i], inner_choices))

    padded_area = sum((spec["outer_h"] + 4) * (spec["outer_w"] + 4) for spec in specs)
    min_side = max(
        18,
        max(spec["outer_h"] + 4 for spec in specs),
        max(spec["outer_w"] + 4 for spec in specs),
        int(np.ceil(np.sqrt(padded_area))) + 1,
    )
    if min_side > 30:
        return None

    h = rng.randint(min_side, 30)
    w = rng.randint(min_side, 30)
    grid = np.full((h, w), bg, dtype=int)
    if not _place_specs(grid, specs, bg, rng):
        return None

    boxes = _find_boxes(grid)
    if len(boxes) != n_boxes:
        return None
    if not np.any(main(grid) != grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        rng = random.Random()
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
