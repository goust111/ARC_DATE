import numpy as np
from collections import Counter


DIR4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]
UP, RIGHT, DOWN, LEFT = range(4)
ROT_K = {UP: 0, RIGHT: 3, DOWN: 2, LEFT: 1}

GEN_BG = 0
GEN_SEPARATOR = 5
GEN_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_MAX_ATTEMPTS = 300
RNG = np.random.default_rng()


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _find_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _extract_patch(grid: np.ndarray, coords: np.ndarray, bg: int) -> np.ndarray:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    patch = np.full((int(r1 - r0 + 1), int(c1 - c0 + 1)), int(bg), dtype=int)
    for r, c in coords:
        patch[int(r - r0), int(c - c0)] = int(grid[int(r), int(c)])
    return patch


def _tight_crop_non_bg(patch: np.ndarray, bg: int) -> np.ndarray:
    pts = np.argwhere(patch != int(bg))
    assert len(pts) > 0
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return patch[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _interior_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            ok = True
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < h and 0 <= nc < w and mask[nr, nc]):
                    ok = False
                    break
            out[r, c] = ok
    return out


def _outline_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < h and 0 <= nc < w) or not mask[nr, nc]:
                    out[r, c] = True
                    break
    return out


def _find_separator_column(grid: np.ndarray) -> int:
    h, w = grid.shape
    candidates = []
    for c in range(1, w - 1):
        col = grid[:, c]
        if len(set(map(int, col))) != 1:
            continue
        color = int(col[0])
        if int(np.sum(grid == color)) == h:
            candidates.append(c)
    assert len(candidates) == 1, "expected exactly one full-height separator column"
    return int(candidates[0])


def _parse_markers(right: np.ndarray, bg: int) -> dict[int, list[dict]]:
    comps = _find_components(right != int(bg))
    markers: dict[int, list[dict]] = {}
    for coords in comps:
        colors = {int(right[int(r), int(c)]) for r, c in coords}
        assert len(colors) == 1, "each target marker must be monochrome"
        color = int(next(iter(colors)))
        assert len(coords) == 2, "each target marker must be a 2-pixel segment"
        (r0, c0), (r1, c1) = sorted((tuple(map(int, p)) for p in coords))
        if r0 == r1:
            left_c, right_c = sorted([c0, c1])
            markers[color] = [
                {"direction": RIGHT, "base": (r0, left_c)},
                {"direction": LEFT, "base": (r0, right_c)},
            ]
        else:
            top_r, bottom_r = sorted([r0, r1])
            assert c0 == c1
            markers[color] = [
                {"direction": DOWN, "base": (top_r, c0)},
                {"direction": UP, "base": (bottom_r, c0)},
            ]
    return markers


def _parse_stamps(left: np.ndarray, bg: int, marker_colors: set[int]) -> list[dict]:
    comps = _find_components(left != int(bg))
    stamps = []
    seen_anchor_colors = set()
    for coords in comps:
        patch = _extract_patch(left, coords, bg)
        mask = patch != int(bg)
        interior = _interior_mask(mask)
        per_color: dict[int, list[tuple[int, int]]] = {}
        for r, c in np.argwhere(mask):
            per_color.setdefault(int(patch[r, c]), []).append((int(r), int(c)))
        singleton_candidates = [
            (color, pts[0])
            for color, pts in per_color.items()
            if len(pts) == 1 and color in marker_colors
        ]
        interior_singletons = [
            (color, pos) for color, pos in singleton_candidates if interior[int(pos[0]), int(pos[1])]
        ]
        if len(interior_singletons) == 1:
            anchor_color, anchor_pos = interior_singletons[0]
        else:
            assert len(singleton_candidates) == 1, "each stamp must have one marker-matching singleton color"
            anchor_color, anchor_pos = singleton_candidates[0]
        assert anchor_color not in seen_anchor_colors, "anchor colors must be unique"
        seen_anchor_colors.add(int(anchor_color))
        stamps.append(
            {
                "anchor_color": int(anchor_color),
                "patch": patch,
                "area": int(mask.sum()),
                "anchor_pos": (int(anchor_pos[0]), int(anchor_pos[1])),
            }
        )
    return sorted(stamps, key=lambda d: d["anchor_color"])


def _rotated_stamp(stamp: dict, direction: int, bg: int) -> dict:
    patch = np.rot90(stamp["patch"], ROT_K[int(direction)])
    mask = patch != int(bg)
    anchor_pts = np.argwhere(patch == int(stamp["anchor_color"]))
    assert len(anchor_pts) == 1, "anchor color must remain unique under rotation"
    ar, ac = map(int, anchor_pts[0])
    outline = _outline_mask(mask)
    full_rel = [tuple(map(int, p)) for p in np.argwhere(mask)]
    outline_rel = [(int(r), int(c), int(patch[int(r), int(c)])) for r, c in np.argwhere(outline)]
    return {
        "direction": int(direction),
        "anchor_rel": (int(ar), int(ac)),
        "full_rel": full_rel,
        "outline_rel": outline_rel,
        "shape": patch.shape,
    }


def _candidate_placement(stamp: dict, variant: dict, base: tuple[int, int], canvas_shape: tuple[int, int]) -> dict | None:
    h, w = canvas_shape
    ar, ac = variant["anchor_rel"]
    top = int(base[0] - ar)
    left = int(base[1] - ac)
    full_abs = []
    for rr, cc in variant["full_rel"]:
        r = int(top + rr)
        c = int(left + cc)
        if not (0 <= r < h and 0 <= c < w):
            return None
        full_abs.append((r, c))
    outline_abs = [(int(top + r), int(left + c), int(color)) for r, c, color in variant["outline_rel"]]
    return {
        "anchor_color": int(stamp["anchor_color"]),
        "direction": int(variant["direction"]),
        "full_abs": set(full_abs),
        "outline_abs": outline_abs,
    }


def _solve_layout(stamps: list[dict], markers: dict[int, list[dict]], canvas_shape: tuple[int, int], bg: int) -> list[dict]:
    enriched = []
    for stamp in stamps:
        variants = []
        for marker_option in markers[int(stamp["anchor_color"])]:
            rotated = _rotated_stamp(stamp, int(marker_option["direction"]), bg)
            cand = _candidate_placement(stamp, rotated, marker_option["base"], canvas_shape)
            if cand is not None:
                variants.append(cand)
        assert variants, "each stamp needs at least one valid placement"
        enriched.append({"stamp": stamp, "variants": variants})

    enriched.sort(key=lambda d: (len(d["variants"]), -d["stamp"]["area"], d["stamp"]["anchor_color"]))
    solutions: list[list[dict]] = []

    def dfs(i: int, occupied: set[tuple[int, int]], chosen: list[dict]) -> None:
        if len(solutions) >= 2:
            return
        if i == len(enriched):
            solutions.append(chosen.copy())
            return
        for cand in sorted(enriched[i]["variants"], key=lambda d: d["direction"]):
            if cand["full_abs"] & occupied:
                continue
            chosen.append(cand)
            dfs(i + 1, occupied | cand["full_abs"], chosen)
            chosen.pop()

    dfs(0, set(), [])
    assert len(solutions) == 1, "marker directions must resolve to a unique non-overlapping layout"
    return solutions[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each stamped object to a same-color target marker, rotate it, and print only its outline on the right canvas.

    Input structure:
    - The grid is split into left and right halves by a single full-height separator column.
    - The left half contains several 4-connected stamp objects; each stamp has one internal singleton-color anchor pixel.
    - The right half is a blank target canvas containing one monochrome 2-pixel marker segment for each stamp, with marker color equal to that stamp's anchor color.

    Rule:
    - Extract every left-half stamp and identify its anchor color from the unique internal marker pixel.
    - Match each stamp to the right-half 2-pixel segment sharing that anchor color.
    - The segment's direction encodes the rotation: up = 0 degrees, right = clockwise 90 degrees, down = 180 degrees, left = counterclockwise 90 degrees.
    - Rotate the whole stamp, translate it so the stamp anchor lands on the marker's starting pixel, and discard the marker.
    - Output only the right canvas, drawing just the rotated stamp's boundary pixels and leaving the interior as background.

    Key insight:
    The anchor color performs the pairing, the tiny same-color segment encodes the rotation, and the output is not the full stamp but only its outline.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    sep_c = _find_separator_column(g)
    left = g[:, :sep_c]
    right = g[:, sep_c + 1 :]

    markers = _parse_markers(right, bg)
    stamps = _parse_stamps(left, bg, set(markers))
    assert {int(s["anchor_color"]) for s in stamps} == set(markers), "stamp/marker color sets must match"

    chosen = _solve_layout(stamps, markers, right.shape, bg)
    out = np.full(right.shape, int(bg), dtype=int)
    for cand in chosen:
        for r, c, color in cand["outline_abs"]:
            out[int(r), int(c)] = int(color)
    return out


def _is_connected(mask: np.ndarray) -> bool:
    comps = _find_components(mask)
    return len(comps) == 1


def _is_rotation_distinct(patch: np.ndarray) -> bool:
    for k in (1, 2, 3):
        rot = np.rot90(patch, k)
        if patch.shape == rot.shape and np.array_equal(patch, rot):
            return False
    return True


def _gen_stamp_patch(body_color: int, anchor_color: int) -> np.ndarray | None:
    for _ in range(80):
        h = int(RNG.integers(4, 7))
        w = int(RNG.integers(4, 7))
        mask = np.ones((h, w), dtype=bool)
        for _ in range(int(RNG.integers(0, 4))):
            side = int(RNG.integers(0, 4))
            if side in (UP, DOWN):
                seg = int(RNG.integers(1, w))
                start = int(RNG.integers(0, w - seg + 1))
                row = 0 if side == UP else h - 1
                mask[row, start : start + seg] = False
            else:
                seg = int(RNG.integers(1, h))
                start = int(RNG.integers(0, h - seg + 1))
                col = 0 if side == LEFT else w - 1
                mask[start : start + seg, col] = False
        if mask.sum() < 10 or not _is_connected(mask):
            continue
        interior = _interior_mask(mask)
        interior_pts = np.argwhere(interior)
        if len(interior_pts) < 2:
            continue
        anchor_r, anchor_c = map(int, interior_pts[int(RNG.integers(0, len(interior_pts)))])
        patch = np.full((h, w), int(GEN_BG), dtype=int)
        patch[mask] = int(body_color)
        patch[anchor_r, anchor_c] = int(anchor_color)
        patch = _tight_crop_non_bg(patch, GEN_BG)
        if not _is_rotation_distinct(patch):
            continue
        return patch
    return None


def _layout_patches(patches: list[np.ndarray]) -> tuple[list[tuple[int, int]], int, int] | None:
    n = len(patches)
    if n == 1:
        rows, cols = 1, 1
    elif n == 2:
        rows, cols = (1, 2) if RNG.random() < 0.5 else (2, 1)
    else:
        rows, cols = 2, 2

    slots = [(r, c) for r in range(rows) for c in range(cols)]
    if n == 3 and rows == cols == 2:
        empty = int(RNG.integers(0, 4))
        slots = [slot for idx, slot in enumerate(slots) if idx != empty]
    else:
        slots = slots[:n]
    perm = RNG.permutation(len(slots))
    assigned = [slots[int(i)] for i in perm]

    row_heights = [0] * rows
    col_widths = [0] * cols
    for patch, (sr, sc) in zip(patches, assigned):
        h, w = patch.shape
        row_heights[int(sr)] = max(row_heights[int(sr)], int(h))
        col_widths[int(sc)] = max(col_widths[int(sc)], int(w))

    if any(v == 0 for v in row_heights) or any(v == 0 for v in col_widths):
        return None

    row_starts = [0]
    for h in row_heights[:-1]:
        row_starts.append(int(row_starts[-1] + h + 1))
    col_starts = [0]
    for w in col_widths[:-1]:
        col_starts.append(int(col_starts[-1] + w + 1))

    positions = []
    for patch, (sr, sc) in zip(patches, assigned):
        h, w = patch.shape
        slot_h = row_heights[int(sr)]
        slot_w = col_widths[int(sc)]
        dr = int(RNG.integers(0, slot_h - h + 1))
        dc = int(RNG.integers(0, slot_w - w + 1))
        top = int(row_starts[int(sr)] + dr)
        left = int(col_starts[int(sc)] + dc)
        positions.append((top, left))

    panel_h = int(row_starts[-1] + row_heights[-1])
    panel_w = int(col_starts[-1] + col_widths[-1])
    return positions, panel_h, panel_w


def _paint_patch(canvas: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    for r in range(h):
        for c in range(w):
            val = int(patch[r, c])
            if val != int(GEN_BG):
                canvas[int(top + r), int(left + c)] = val


def _paint_outline(canvas: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    mask = patch != int(GEN_BG)
    outline = _outline_mask(mask)
    for r, c in np.argwhere(outline):
        canvas[int(top + r), int(left + c)] = int(patch[int(r), int(c)])


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    n = int(RNG.choice([2, 3, 4]))
    color_order = list(map(int, RNG.permutation(GEN_COLORS)))
    body_colors = color_order[:n]
    anchor_colors = color_order[n : 2 * n]

    stamps = []
    for body_color, anchor_color in zip(body_colors, anchor_colors):
        patch = _gen_stamp_patch(int(body_color), int(anchor_color))
        if patch is None:
            return None
        stamps.append({"body_color": int(body_color), "anchor_color": int(anchor_color), "patch": patch})

    directions = [int(RNG.integers(0, 4)) for _ in range(n)]
    rotated_patches = [np.rot90(stamp["patch"], ROT_K[int(direction)]) for stamp, direction in zip(stamps, directions)]

    left_layout = _layout_patches([stamp["patch"] for stamp in stamps])
    right_layout = _layout_patches(rotated_patches)
    if left_layout is None or right_layout is None:
        return None

    left_positions, left_panel_h, left_panel_w = left_layout
    right_positions, right_panel_h, right_panel_w = right_layout

    left_outer = int(RNG.integers(0, 2))
    right_outer = int(RNG.integers(0, 2))
    left_w = int(left_outer + left_panel_w + 1)
    right_w = int(1 + right_panel_w + right_outer)
    if left_w + 1 + right_w > 30:
        return None

    left_top = int(RNG.integers(0, 3))
    right_top = int(RNG.integers(0, 3))
    h = int(max(left_top + left_panel_h, right_top + right_panel_h) + RNG.integers(0, 3))
    if h > 30:
        return None

    grid = np.full((h, left_w + 1 + right_w), int(GEN_BG), dtype=int)
    grid[:, left_w] = int(GEN_SEPARATOR)

    for stamp, (top, left) in zip(stamps, left_positions):
        _paint_patch(grid, stamp["patch"], int(left_top + top), int(left_outer + left))

    expected = np.full((h, right_w), int(GEN_BG), dtype=int)
    right_origin_col = int(left_w + 1)
    for stamp, direction, rot_patch, (top, left) in zip(stamps, directions, rotated_patches, right_positions):
        abs_top = int(right_top + top)
        abs_left = int(1 + left)
        anchor_pts = np.argwhere(rot_patch == int(stamp["anchor_color"]))
        assert len(anchor_pts) == 1
        ar, ac = map(int, anchor_pts[0])
        dr, dc = DIR4[int(direction)]
        nr, nc = int(ar + dr), int(ac + dc)
        if not (0 <= nr < rot_patch.shape[0] and 0 <= nc < rot_patch.shape[1]):
            return None
        if int(rot_patch[nr, nc]) == int(GEN_BG):
            return None
        base_r = int(abs_top + ar)
        base_c = int(abs_left + ac)
        tip_r = int(base_r + dr)
        tip_c = int(base_c + dc)
        grid[base_r, right_origin_col + base_c] = int(stamp["anchor_color"])
        grid[tip_r, right_origin_col + tip_c] = int(stamp["anchor_color"])
        _paint_outline(expected, rot_patch, abs_top, abs_left)

    try:
        produced = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(produced, expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        result = _gen_attempt()
        if result is None:
            continue
        grid, _expected = result
        return grid
    raise RuntimeError("failed to generate a valid stamp-placement ARC input")
