import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
PAIR_COUNT_RANGE = (2, 4)
GRID_BASE_RANGE = (20, 28)
RECT_HEIGHT_RANGE = (4, 8)
RECT_WIDTH_RANGE = (4, 8)
PROFILE_SPAN_RANGE = (3, 7)
GAP_RANGE = (2, 7)
MAX_GENERATION_ATTEMPTS = 400
PAIR_PLACEMENT_ATTEMPTS = 120
PADDING = 1

RNG = np.random.default_rng()


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            rows = [rr for rr, _cc in coords]
            cols = [cc for _rr, cc in coords]
            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                }
            )
    return components


def _component_mask(component: dict) -> np.ndarray:
    r0, c0, r1, c1 = component["bbox"]
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in component["coords"]:
        mask[r - r0, c - c0] = True
    return mask


def _is_contiguous(indices: np.ndarray) -> bool:
    return bool(indices.size) and bool(np.all(np.diff(indices) == 1))


def _top_profile(mask: np.ndarray) -> tuple[int, ...] | None:
    h, w = mask.shape
    profile = []
    for c in range(w):
        rows = np.flatnonzero(mask[:, c])
        if rows.size == 0 or int(rows[0]) != 0 or not _is_contiguous(rows):
            return None
        profile.append(int(rows[-1]) + 1)
    return tuple(profile)


def _bottom_profile(mask: np.ndarray) -> tuple[int, ...] | None:
    h, w = mask.shape
    profile = []
    for c in range(w):
        rows = np.flatnonzero(mask[:, c])
        if rows.size == 0 or int(rows[-1]) != h - 1 or not _is_contiguous(rows):
            return None
        profile.append(int(rows.size))
    return tuple(profile)


def _left_profile(mask: np.ndarray) -> tuple[int, ...] | None:
    h, w = mask.shape
    profile = []
    for r in range(h):
        cols = np.flatnonzero(mask[r])
        if cols.size == 0 or int(cols[0]) != 0 or not _is_contiguous(cols):
            return None
        profile.append(int(cols[-1]) + 1)
    return tuple(profile)


def _right_profile(mask: np.ndarray) -> tuple[int, ...] | None:
    h, w = mask.shape
    profile = []
    for r in range(h):
        cols = np.flatnonzero(mask[r])
        if cols.size == 0 or int(cols[-1]) != w - 1 or not _is_contiguous(cols):
            return None
        profile.append(int(cols.size))
    return tuple(profile)


def _descriptors(mask: np.ndarray) -> list[tuple[str, str, tuple[int, ...]]]:
    candidates = []
    top = _top_profile(mask)
    bottom = _bottom_profile(mask)
    left = _left_profile(mask)
    right = _right_profile(mask)
    if top is not None:
        candidates.append(("vertical", "top", top))
    if bottom is not None:
        candidates.append(("vertical", "bottom", bottom))
    if left is not None:
        candidates.append(("horizontal", "left", left))
    if right is not None:
        candidates.append(("horizontal", "right", right))
    out = []
    seen = set()
    for item in candidates:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _pair_solution(mask_a: np.ndarray, mask_b: np.ndarray) -> dict:
    desc_a = _descriptors(mask_a)
    desc_b = _descriptors(mask_b)
    matches: list[tuple] = []
    for axis_a, wall_a, profile_a in desc_a:
        for axis_b, wall_b, profile_b in desc_b:
            if axis_a != axis_b:
                continue
            if axis_a == "vertical" and {wall_a, wall_b} == {"top", "bottom"} and len(profile_a) == len(profile_b):
                totals = tuple(int(a + b) for a, b in zip(profile_a, profile_b))
                if len(set(totals)) == 1:
                    top_profile = profile_a if wall_a == "top" else profile_b
                    bottom_profile = profile_a if wall_a == "bottom" else profile_b
                    matches.append(("vertical", top_profile, bottom_profile, totals[0], len(top_profile)))
            if axis_a == "horizontal" and {wall_a, wall_b} == {"left", "right"} and len(profile_a) == len(profile_b):
                totals = tuple(int(a + b) for a, b in zip(profile_a, profile_b))
                if len(set(totals)) == 1:
                    left_profile = profile_a if wall_a == "left" else profile_b
                    right_profile = profile_a if wall_a == "right" else profile_b
                    matches.append(("horizontal", left_profile, right_profile, len(left_profile), totals[0]))

    unique_matches = []
    seen = set()
    for item in matches:
        if item not in seen:
            seen.add(item)
            unique_matches.append(item)
    assert len(unique_matches) == 1, "same-color pair must admit exactly one rectangle assembly"

    axis, first_profile, second_profile, height, width = unique_matches[0]
    if axis == "vertical":
        return {
            "axis": axis,
            "top_profile": first_profile,
            "bottom_profile": second_profile,
            "height": int(height),
            "width": int(width),
        }
    return {
        "axis": axis,
        "left_profile": first_profile,
        "right_profile": second_profile,
        "height": int(height),
        "width": int(width),
    }


def _centered_top_left(bbox: tuple[int, int, int, int], height: int, width: int) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    top = (r0 + r1 - (height - 1)) // 2
    left = (c0 + c1 - (width - 1)) // 2
    return int(top), int(left)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair same-colored complementary fragments and merge each pair into a filled rectangle.

    Each same-colored pair has opposite outer walls along one axis, and their jagged inner
    profiles complement to a full rectangle. The merged rectangle is centered on the union
    of the two original fragment bounding boxes.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = get_bg(g)

    components = find_components(g, bg)
    by_color: dict[int, list[dict]] = {}
    for component in components:
        by_color.setdefault(int(component["color"]), []).append(component)

    out = np.full_like(g, bg)
    for color, pair in by_color.items():
        assert len(pair) == 2, "each foreground color must appear in exactly two components"
        mask_a = _component_mask(pair[0])
        mask_b = _component_mask(pair[1])
        solution = _pair_solution(mask_a, mask_b)

        r0 = min(pair[0]["bbox"][0], pair[1]["bbox"][0])
        c0 = min(pair[0]["bbox"][1], pair[1]["bbox"][1])
        r1 = max(pair[0]["bbox"][2], pair[1]["bbox"][2])
        c1 = max(pair[0]["bbox"][3], pair[1]["bbox"][3])
        top, left = _centered_top_left((r0, c0, r1, c1), solution["height"], solution["width"])
        bottom = top + solution["height"]
        right = left + solution["width"]
        assert 0 <= top < bottom <= g.shape[0]
        assert 0 <= left < right <= g.shape[1]
        patch = out[top:bottom, left:right]
        assert np.all(patch == bg), "merged rectangles must not overlap"
        patch[:] = int(color)
    return out


def _is_monotone(seq: tuple[int, ...]) -> bool:
    nondecreasing = all(a <= b for a, b in zip(seq, seq[1:]))
    nonincreasing = all(a >= b for a, b in zip(seq, seq[1:]))
    return nondecreasing or nonincreasing


def _random_nonmonotone_profile(length: int, total: int) -> tuple[int, ...]:
    assert length >= 3 and total >= 4
    for _ in range(200):
        profile = tuple(int(x) for x in RNG.integers(1, total, size=length))
        if len(set(profile)) < 2:
            continue
        if _is_monotone(profile):
            continue
        return profile
    raise RuntimeError("failed to sample a non-monotone profile")


def _top_mask(profile: tuple[int, ...]) -> np.ndarray:
    height = max(profile)
    width = len(profile)
    mask = np.zeros((height, width), dtype=bool)
    for c, depth in enumerate(profile):
        mask[:depth, c] = True
    return mask


def _bottom_mask(profile: tuple[int, ...]) -> np.ndarray:
    height = max(profile)
    width = len(profile)
    mask = np.zeros((height, width), dtype=bool)
    for c, depth in enumerate(profile):
        mask[height - depth :, c] = True
    return mask


def _left_mask(profile: tuple[int, ...]) -> np.ndarray:
    height = len(profile)
    width = max(profile)
    mask = np.zeros((height, width), dtype=bool)
    for r, depth in enumerate(profile):
        mask[r, :depth] = True
    return mask


def _right_mask(profile: tuple[int, ...]) -> np.ndarray:
    height = len(profile)
    width = max(profile)
    mask = np.zeros((height, width), dtype=bool)
    for r, depth in enumerate(profile):
        mask[r, width - depth :] = True
    return mask


def _sample_pair_spec(axis: str) -> dict:
    assert axis in {"vertical", "horizontal"}
    for _ in range(200):
        if axis == "vertical":
            rect_h = int(RNG.integers(RECT_HEIGHT_RANGE[0], RECT_HEIGHT_RANGE[1] + 1))
            rect_w = int(RNG.integers(PROFILE_SPAN_RANGE[0], PROFILE_SPAN_RANGE[1] + 1))
            top_profile = _random_nonmonotone_profile(rect_w, rect_h)
            bottom_profile = tuple(rect_h - v for v in top_profile)
            mask_a = _top_mask(top_profile)
            mask_b = _bottom_mask(bottom_profile)
        else:
            rect_h = int(RNG.integers(PROFILE_SPAN_RANGE[0], PROFILE_SPAN_RANGE[1] + 1))
            rect_w = int(RNG.integers(RECT_WIDTH_RANGE[0], RECT_WIDTH_RANGE[1] + 1))
            left_profile = _random_nonmonotone_profile(rect_h, rect_w)
            right_profile = tuple(rect_w - v for v in left_profile)
            mask_a = _left_mask(left_profile)
            mask_b = _right_mask(right_profile)

        solution = _pair_solution(mask_a, mask_b)
        if solution["axis"] != axis:
            continue
        gap = int(RNG.integers(GAP_RANGE[0], GAP_RANGE[1] + 1))
        return {
            "axis": axis,
            "mask_a": mask_a,
            "mask_b": mask_b,
            "height": solution["height"],
            "width": solution["width"],
            "gap": gap,
        }
    raise RuntimeError("failed to sample a valid pair spec")


def _bbox_clear(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int) -> bool:
    gh, gw = occupied.shape
    if r0 < 0 or c0 < 0 or r0 + h > gh or c0 + w > gw:
        return False
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(gh, r0 + h + pad)
    cc1 = min(gw, c0 + w + pad)
    return not bool(np.any(occupied[rr0:rr1, cc0:cc1]))


def _mark_bbox(occupied: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    occupied[r0 : r0 + h, c0 : c0 + w] = True


def _paint_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    h, w = mask.shape
    patch = grid[top : top + h, left : left + w]
    assert np.all(patch[mask] == BG)
    patch[mask] = int(color)


def _sample_orientations(pair_count: int) -> list[str]:
    vertical_count = pair_count // 2
    horizontal_count = pair_count // 2
    if pair_count % 2 == 1:
        if bool(RNG.integers(0, 2)):
            vertical_count += 1
        else:
            horizontal_count += 1
    orientations = ["vertical"] * vertical_count + ["horizontal"] * horizontal_count
    RNG.shuffle(orientations)
    return orientations


def _place_pair(grid: np.ndarray, input_occ: np.ndarray, output_occ: np.ndarray, color: int, spec: dict) -> bool:
    gh, gw = grid.shape
    axis = spec["axis"]
    mask_a = spec["mask_a"]
    mask_b = spec["mask_b"]
    gap = spec["gap"]
    ha, wa = mask_a.shape
    hb, wb = mask_b.shape

    if axis == "vertical":
        union_h = ha + gap + hb
        pre = (union_h - spec["height"]) // 2
        post = union_h - spec["height"] - pre
        row_min = pre
        row_max = gh - spec["height"] - post
        col_min = 0
        col_max = gw - spec["width"]
    else:
        union_w = wa + gap + wb
        pre = (union_w - spec["width"]) // 2
        post = union_w - spec["width"] - pre
        row_min = 0
        row_max = gh - spec["height"]
        col_min = pre
        col_max = gw - spec["width"] - post

    if row_min > row_max or col_min > col_max:
        return False

    for _ in range(PAIR_PLACEMENT_ATTEMPTS):
        out_r = int(RNG.integers(row_min, row_max + 1))
        out_c = int(RNG.integers(col_min, col_max + 1))
        if not _bbox_clear(output_occ, out_r, out_c, spec["height"], spec["width"], PADDING):
            continue

        flipped = bool(RNG.integers(0, 2))
        if axis == "vertical":
            union_r = out_r - pre
            if not flipped:
                pos_a = (union_r, out_c)
                pos_b = (union_r + ha + gap, out_c)
            else:
                pos_b = (union_r, out_c)
                pos_a = (union_r + hb + gap, out_c)
        else:
            union_c = out_c - pre
            if not flipped:
                pos_a = (out_r, union_c)
                pos_b = (out_r, union_c + wa + gap)
            else:
                pos_b = (out_r, union_c)
                pos_a = (out_r, union_c + wb + gap)

        if not _bbox_clear(input_occ, pos_a[0], pos_a[1], ha, wa, PADDING):
            continue
        if not _bbox_clear(input_occ, pos_b[0], pos_b[1], hb, wb, PADDING):
            continue

        _paint_mask(grid, mask_a, pos_a[0], pos_a[1], color)
        _paint_mask(grid, mask_b, pos_b[0], pos_b[1], color)
        _mark_bbox(input_occ, pos_a[0], pos_a[1], ha, wa)
        _mark_bbox(input_occ, pos_b[0], pos_b[1], hb, wb)
        _mark_bbox(output_occ, out_r, out_c, spec["height"], spec["width"])
        return True
    return False


def _validate_generated(grid: np.ndarray) -> bool:
    bg = get_bg(grid)
    components = find_components(grid, bg)
    by_color: dict[int, list[dict]] = {}
    for component in components:
        by_color.setdefault(int(component["color"]), []).append(component)
    if not by_color:
        return False
    if any(len(parts) != 2 for parts in by_color.values()):
        return False

    out = main(grid)
    if out.shape != grid.shape:
        return False
    for color, pair in by_color.items():
        coords = np.argwhere(out == color)
        if coords.size == 0:
            return False
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        if coords.shape[0] != (r1 - r0 + 1) * (c1 - c0 + 1):
            return False
        mask_a = _component_mask(pair[0])
        mask_b = _component_mask(pair[1])
        _pair_solution(mask_a, mask_b)
    return True


def _gen_attempt() -> np.ndarray | None:
    pair_count = int(RNG.integers(PAIR_COUNT_RANGE[0], PAIR_COUNT_RANGE[1] + 1))
    min_dim = max(GRID_BASE_RANGE[0], 14 + 3 * pair_count)
    max_dim = max(GRID_BASE_RANGE[1], min_dim + 2)
    gh = int(RNG.integers(min_dim, max_dim + 1))
    gw = int(RNG.integers(min_dim, max_dim + 1))

    colors = [int(c) for c in RNG.choice(PALETTE, size=pair_count, replace=False)]
    orientations = _sample_orientations(pair_count)

    grid = np.full((gh, gw), BG, dtype=int)
    input_occ = np.zeros((gh, gw), dtype=bool)
    output_occ = np.zeros((gh, gw), dtype=bool)

    order = RNG.permutation(pair_count)
    for idx in order:
        color = colors[idx]
        axis = orientations[idx]
        placed = False
        for _ in range(80):
            spec = _sample_pair_spec(axis)
            if _place_pair(grid, input_occ, output_occ, color, spec):
                placed = True
                break
        if not placed:
            return None

    if not _validate_generated(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
