from __future__ import annotations

from collections import deque
import random

import numpy as np


DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDES = ("top", "bottom", "left", "right")


def _background_color(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[cnts.argmax()])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q: deque[tuple[int, int]] = deque([(int(r), int(c))])
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _touches_mask_boundary(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = shape
    return bool(
        np.any(coords[:, 0] == 0)
        or np.any(coords[:, 0] == h - 1)
        or np.any(coords[:, 1] == 0)
        or np.any(coords[:, 1] == w - 1)
    )


def _normalize_coords(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[tuple[int, int], ...], int, int]:
    rs = [int(r) for r, _ in coords]
    cs = [int(c) for _, c in coords]
    r0 = min(rs)
    c0 = min(cs)
    norm = tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))
    h = max(r for r, _ in norm) + 1
    w = max(c for _, c in norm) + 1
    return norm, int(h), int(w)


def _rot90(coords: tuple[tuple[int, int], ...]) -> tuple[tuple[tuple[int, int], ...], int, int]:
    norm, h, _w = _normalize_coords(coords)
    return _normalize_coords([(c, h - 1 - r) for r, c in norm])


def _flip_h(coords: tuple[tuple[int, int], ...]) -> tuple[tuple[tuple[int, int], ...], int, int]:
    norm, _h, w = _normalize_coords(coords)
    return _normalize_coords([(r, w - 1 - c) for r, c in norm])


def _canonical_shape_from_coords(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    cur, _, _ = _normalize_coords(coords)
    variants: set[tuple[tuple[int, int], ...]] = set()

    temp = cur
    for _ in range(4):
        norm, _, _ = _normalize_coords(temp)
        variants.add(norm)
        temp, _, _ = _rot90(temp)

    temp, _, _ = _flip_h(cur)
    for _ in range(4):
        norm, _, _ = _normalize_coords(temp)
        variants.add(norm)
        temp, _, _ = _rot90(temp)

    return min(variants)


def _mask_to_coords(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    pts = np.argwhere(mask)
    return tuple((int(r), int(c)) for r, c in pts)


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert len(coords) > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _oriented_masks(mask: np.ndarray) -> list[np.ndarray]:
    coords = _mask_to_coords(mask)
    seen: set[tuple[tuple[int, int], ...]] = set()
    out: list[np.ndarray] = []

    cur, _, _ = _normalize_coords(coords)
    for flipped in (False, True):
        base = cur
        if flipped:
            base, _, _ = _flip_h(base)
        temp = base
        for _ in range(4):
            norm, h, w = _normalize_coords(temp)
            if norm not in seen:
                seen.add(norm)
                arr = np.zeros((h, w), dtype=bool)
                for r, c in norm:
                    arr[r, c] = True
                out.append(arr)
            temp, _, _ = _rot90(temp)
    return out


def _largest_rectangle(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    h, w = mask.shape
    heights = [0] * w
    best: tuple[int, int, int, int, int, int] | None = None

    for r in range(h):
        for c in range(w):
            heights[c] = heights[c] + 1 if mask[r, c] else 0

        stack: list[int] = []
        for c in range(w + 1):
            cur_h = heights[c] if c < w else 0
            while stack and heights[stack[-1]] > cur_h:
                idx = stack.pop()
                height = heights[idx]
                left = stack[-1] + 1 if stack else 0
                width = c - left
                if height <= 0 or width <= 0:
                    continue
                area = height * width
                top = r - height + 1
                bottom = r
                right = c - 1
                key = (area, min(height, width), height + width, -top, -left, -bottom)
                if best is None or key > best:
                    best = (area, min(height, width), height + width, -top, -left, -bottom)
                    best_rect = (int(top), int(left), int(bottom), int(right))
            stack.append(c)

    return best_rect if best is not None else None


def _find_frame(grid: np.ndarray, bg: int) -> tuple[int, np.ndarray, tuple[int, int, int, int]]:
    h, w = grid.shape
    candidates: list[tuple[int, int, int, np.ndarray, tuple[int, int, int, int]]] = []

    for color in sorted(int(v) for v in np.unique(grid) if int(v) != bg):
        for comp in _components(grid == color):
            r0, c0, r1, c1 = _bbox(comp)
            if r1 - r0 + 1 < 5 or c1 - c0 + 1 < 5:
                continue

            local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            local[comp[:, 0] - r0, comp[:, 1] - c0] = True
            if not (
                np.all(local[0, :])
                and np.all(local[-1, :])
                and np.all(local[:, 0])
                and np.all(local[:, -1])
            ):
                continue

            bg_local = grid[r0 : r1 + 1, c0 : c1 + 1] == bg
            enclosed = [
                pts for pts in _components(bg_local) if not _touches_mask_boundary(pts, bg_local.shape)
            ]
            if not enclosed:
                continue

            full_mask = np.zeros((h, w), dtype=bool)
            full_mask[comp[:, 0], comp[:, 1]] = True
            area = int(len(comp))
            box_area = int((r1 - r0 + 1) * (c1 - c0 + 1))
            candidates.append((area, box_area, color, full_mask, (r0, c0, r1, c1)))

    assert candidates, "expected a rectangular frame"
    candidates.sort(key=lambda item: (item[0], item[1]))
    _area, _box_area, frame_color, frame_mask, bbox = candidates[-1]
    return int(frame_color), frame_mask, bbox


def _core_and_notches(
    grid: np.ndarray,
    bg: int,
    frame_bbox: tuple[int, int, int, int],
) -> tuple[tuple[int, int, int, int], list[tuple[str, tuple[tuple[int, int], ...]]]]:
    r0, c0, r1, c1 = frame_bbox
    bg_local = grid[r0 : r1 + 1, c0 : c1 + 1] == bg
    enclosed = [
        pts for pts in _components(bg_local) if not _touches_mask_boundary(pts, bg_local.shape)
    ]
    assert enclosed, "frame must contain an enclosed background region"
    main_bg = max(enclosed, key=len)

    interior_mask = np.zeros_like(bg_local, dtype=bool)
    interior_mask[main_bg[:, 0], main_bg[:, 1]] = True

    rect = _largest_rectangle(interior_mask)
    assert rect is not None, "expected a maximal interior rectangle"
    ir0, ic0, ir1, ic1 = rect

    extras = interior_mask.copy()
    extras[ir0 : ir1 + 1, ic0 : ic1 + 1] = False

    notches: list[tuple[str, tuple[tuple[int, int], ...]]] = []
    for comp in _components(extras):
        cr0, cc0, cr1, cc1 = _bbox(comp)
        if cr1 < ir0:
            side = "top"
        elif cr0 > ir1:
            side = "bottom"
        elif cc1 < ic0:
            side = "left"
        elif cc0 > ic1:
            side = "right"
        else:
            raise AssertionError("notch must sit on exactly one side of the inner rectangle")
        rel = [(int(r), int(c)) for r, c in comp]
        notches.append((side, _canonical_shape_from_coords(rel)))

    return (r0 + ir0, c0 + ic0, r0 + ir1, c0 + ic1), notches


def _fragment_shape_colors(
    grid: np.ndarray,
    bg: int,
    frame_mask: np.ndarray,
) -> dict[tuple[tuple[int, int], ...], int]:
    other = (grid != bg) & (~frame_mask)
    mapping: dict[tuple[tuple[int, int], ...], int] = {}
    for comp in _components(other):
        colors = np.unique(grid[comp[:, 0], comp[:, 1]])
        assert len(colors) == 1, "each outside fragment must be monochrome"
        color = int(colors[0])
        sig = _canonical_shape_from_coords([(int(r), int(c)) for r, c in comp])
        assert sig not in mapping, "fragment shapes must match uniquely"
        mapping[sig] = color
    return mapping


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match the frame's inward notch shapes to outside fragments and recolor the corresponding border sides.

    Input structure:
    - The background is the dominant color.
    - A single-color rectangular frame sits near the center; its outer boundary is complete, while some sides have inward background notches cut from the inner boundary.
    - Outside the frame are several separated monochrome fragments, each shape-congruent to one notch up to rotation or reflection.

    Rule:
    - Detect the unique rectangular frame and the enclosed background region inside it.
    - Recover the frame's main inner rectangle, then treat every connected background protrusion outside that rectangle as a notch and match it to an outside fragment under the 8 dihedral orientations.
    - For each successful match, recolor the whole straight border side containing that notch, leaving the notch itself as background and deleting all outside fragments.

    Key insight:
    - The outside fragments are a shape legend for the frame's notches, and the matched fragment color applies to the entire border side rather than filling the notch.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)

    frame_color, frame_mask, frame_bbox = _find_frame(g, bg)
    core_rect, notches = _core_and_notches(g, bg, frame_bbox)
    fragment_map = _fragment_shape_colors(g, bg, frame_mask)

    side_colors: dict[str, int] = {}
    for side, shape in notches:
        assert shape in fragment_map, "missing outside fragment for a frame notch"
        color = int(fragment_map[shape])
        if side in side_colors:
            assert side_colors[side] == color, "multiple notches on one side must agree in color"
        side_colors[side] = color

    r0, c0, r1, c1 = frame_bbox
    ir0, ic0, ir1, ic1 = core_rect

    out = np.full_like(g, bg)
    out[frame_mask] = frame_color

    side_masks: dict[str, np.ndarray] = {}
    top = np.zeros_like(g, dtype=bool)
    top[r0:ir0, ic0 : ic1 + 1] = True
    side_masks["top"] = top

    bottom = np.zeros_like(g, dtype=bool)
    bottom[ir1 + 1 : r1 + 1, ic0 : ic1 + 1] = True
    side_masks["bottom"] = bottom

    left = np.zeros_like(g, dtype=bool)
    left[ir0 : ir1 + 1, c0:ic0] = True
    side_masks["left"] = left

    right = np.zeros_like(g, dtype=bool)
    right[ir0 : ir1 + 1, ic1 + 1 : c1 + 1] = True
    side_masks["right"] = right

    for side, color in side_colors.items():
        out[side_masks[side] & frame_mask] = int(color)

    return out


def _is_rect_frame_mask(mask: np.ndarray) -> bool:
    if mask.size == 0:
        return False
    coords = np.argwhere(mask)
    if len(coords) < 8:
        return False
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    sub = mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
    if sub.shape[0] < 3 or sub.shape[1] < 3:
        return False
    border = np.zeros_like(sub, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return bool(np.all(sub[border]) and np.any(~sub[1:-1, 1:-1]))


def _random_connected_mask(height: int, width: int, required_edge: str) -> np.ndarray | None:
    assert height >= 1 and width >= 1
    max_area = min(height * width, 6)
    min_area = 2 if height * width >= 2 else 1
    if max_area < min_area:
        return None

    for _ in range(80):
        target = random.randint(min_area, max_area)
        if required_edge == "top":
            start = (0, random.randrange(width))
        elif required_edge == "bottom":
            start = (height - 1, random.randrange(width))
        elif required_edge == "left":
            start = (random.randrange(height), 0)
        else:
            start = (random.randrange(height), width - 1)

        cells = {start}
        while len(cells) < target:
            frontier: list[tuple[int, int]] = []
            for r, c in cells:
                for dr, dc in DIRS4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))

        mask = np.zeros((height, width), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        mask = _trim_mask(mask)

        if required_edge == "top" and not np.any(mask[0, :]):
            continue
        if required_edge == "bottom" and not np.any(mask[-1, :]):
            continue
        if required_edge == "left" and not np.any(mask[:, 0]):
            continue
        if required_edge == "right" and not np.any(mask[:, -1]):
            continue
        if len(_components(mask)) != 1:
            continue
        if _is_rect_frame_mask(mask):
            continue
        return mask

    return None


def _draw_frame_with_notches(
    grid: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    thickness: dict[str, int],
    notch_specs: dict[str, tuple[np.ndarray, int]],
    frame_color: int,
    bg: int,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = frame_bbox
    ir0 = r0 + thickness["top"]
    ir1 = r1 - thickness["bottom"]
    ic0 = c0 + thickness["left"]
    ic1 = c1 - thickness["right"]

    grid[r0 : r1 + 1, c0 : c1 + 1] = frame_color
    grid[ir0 : ir1 + 1, ic0 : ic1 + 1] = bg

    for side, (mask, offset) in notch_specs.items():
        mh, mw = mask.shape
        if side == "top":
            rs = ir0 - mh
            cs = ic0 + offset
            patch = grid[rs:ir0, cs : cs + mw]
        elif side == "bottom":
            rs = ir1 + 1
            cs = ic0 + offset
            patch = grid[rs : rs + mh, cs : cs + mw]
        elif side == "left":
            rs = ir0 + offset
            cs = ic0 - mw
            patch = grid[rs : rs + mh, cs:ic0]
        else:
            rs = ir0 + offset
            cs = ic1 + 1
            patch = grid[rs : rs + mh, cs : cs + mw]
        patch[mask] = bg

    frame_mask = grid == frame_color
    return frame_mask, (ir0, ic0, ir1, ic1)


def _render_expected(
    shape: tuple[int, int],
    frame_mask: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    core_rect: tuple[int, int, int, int],
    frame_color: int,
    side_colors: dict[str, int],
    bg: int,
) -> np.ndarray:
    r0, c0, r1, c1 = frame_bbox
    ir0, ic0, ir1, ic1 = core_rect
    out = np.full(shape, bg, dtype=int)
    out[frame_mask] = frame_color

    masks: dict[str, np.ndarray] = {}
    masks["top"] = np.zeros(shape, dtype=bool)
    masks["top"][r0:ir0, ic0 : ic1 + 1] = True
    masks["bottom"] = np.zeros(shape, dtype=bool)
    masks["bottom"][ir1 + 1 : r1 + 1, ic0 : ic1 + 1] = True
    masks["left"] = np.zeros(shape, dtype=bool)
    masks["left"][ir0 : ir1 + 1, c0:ic0] = True
    masks["right"] = np.zeros(shape, dtype=bool)
    masks["right"][ir0 : ir1 + 1, ic1 + 1 : c1 + 1] = True

    for side, color in side_colors.items():
        out[masks[side] & frame_mask] = int(color)
    return out


def _place_fragment(
    grid: np.ndarray,
    occ: np.ndarray,
    mask: np.ndarray,
    color: int,
) -> bool:
    h, w = grid.shape
    ph, pw = mask.shape
    if ph + 2 > h or pw + 2 > w:
        return False

    for _ in range(300):
        r0 = random.randint(1, h - ph - 1)
        c0 = random.randint(1, w - pw - 1)
        rr0 = r0 - 1
        cc0 = c0 - 1
        rr1 = r0 + ph
        cc1 = c0 + pw
        if np.any(occ[rr0 : rr1 + 1, cc0 : cc1 + 1]):
            continue
        patch = grid[r0 : r0 + ph, c0 : c0 + pw]
        if np.any(patch[mask] != 0):
            continue
        patch[mask] = color
        occ[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
        return True
    return False


def _gen_attempt() -> np.ndarray | None:
    bg = 0
    h = random.randint(22, 30)
    w = random.randint(22, 30)
    frame_color = random.randint(1, 9)

    n_sides = random.randint(1, 4)
    chosen_sides = random.sample(list(SIDES), n_sides)

    inner_h = random.randint(6, 11)
    inner_w = random.randint(6, 11)
    thickness = {side: random.randint(2, 4) for side in SIDES}

    outer_h = inner_h + thickness["top"] + thickness["bottom"]
    outer_w = inner_w + thickness["left"] + thickness["right"]
    if outer_h > h - 4 or outer_w > w - 4:
        return None

    used_shapes: set[tuple[tuple[int, int], ...]] = set()
    notch_specs: dict[str, tuple[np.ndarray, int]] = {}
    side_colors: dict[str, int] = {}

    colors = [c for c in range(1, 10) if c != frame_color]
    random.shuffle(colors)
    if len(colors) < n_sides:
        return None

    for idx, side in enumerate(chosen_sides):
        for _ in range(120):
            if side in ("top", "bottom"):
                max_h = thickness[side] - 1
                max_w = min(5, inner_w - 2)
                if max_h < 1 or max_w < 2:
                    return None
                mask = _random_connected_mask(
                    random.randint(1, max_h),
                    random.randint(2, max_w),
                    "bottom" if side == "top" else "top",
                )
                if mask is None or mask.shape[1] > inner_w - 2:
                    continue
                offset = random.randint(1, inner_w - mask.shape[1] - 1)
            else:
                max_h = min(5, inner_h - 2)
                max_w = thickness[side] - 1
                if max_h < 2 or max_w < 1:
                    return None
                mask = _random_connected_mask(
                    random.randint(2, max_h),
                    random.randint(1, max_w),
                    "right" if side == "left" else "left",
                )
                if mask is None or mask.shape[0] > inner_h - 2:
                    continue
                offset = random.randint(1, inner_h - mask.shape[0] - 1)

            sig = _canonical_shape_from_coords(_mask_to_coords(mask))
            if sig in used_shapes:
                continue
            used_shapes.add(sig)
            notch_specs[side] = (mask, offset)
            side_colors[side] = colors[idx]
            break
        else:
            return None

    margin_top = random.randint(1, h - outer_h - 1)
    margin_left = random.randint(1, w - outer_w - 1)
    frame_bbox = (
        margin_top,
        margin_left,
        margin_top + outer_h - 1,
        margin_left + outer_w - 1,
    )

    inp = np.full((h, w), bg, dtype=int)
    frame_mask, core_rect = _draw_frame_with_notches(
        inp, frame_bbox, thickness, notch_specs, frame_color, bg
    )
    expected = _render_expected(inp.shape, frame_mask, frame_bbox, core_rect, frame_color, side_colors, bg)

    occ = np.zeros((h, w), dtype=bool)
    r0, c0, r1, c1 = frame_bbox
    occ[max(0, r0 - 1) : min(h, r1 + 2), max(0, c0 - 1) : min(w, c1 + 2)] = True

    for side in chosen_sides:
        mask = notch_specs[side][0]
        variants = _oriented_masks(mask)
        random.shuffle(variants)
        fragment = random.choice(variants)
        if not _place_fragment(inp, occ, fragment, side_colors[side]):
            return None

    try:
        if not np.array_equal(main(inp), expected):
            return None
    except Exception:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to sample a valid puzzle")
