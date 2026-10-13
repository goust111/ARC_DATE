from __future__ import annotations

import random
from math import isqrt

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = DIR4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]
SIDE_NAMES = ("top", "bottom", "left", "right")
BG = 0


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray, connectivity: int) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    dirs = DIR4 if connectivity == 4 else DIR8
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
                for dr, dc in dirs:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _colored_components(grid: np.ndarray, bg: int) -> list[dict]:
    comps = []
    for coords in _components(grid != bg, connectivity=8):
        rows = coords[:, 0]
        cols = coords[:, 1]
        colors = tuple(sorted(int(x) for x in np.unique(grid[rows, cols])))
        comps.append(
            {
                "coords": coords,
                "bbox": _bbox_from_coords(coords),
                "area": int(coords.shape[0]),
                "colors": colors,
            }
        )
    return comps


def _crop_patch(patch: np.ndarray, fill: int) -> np.ndarray:
    coords = np.argwhere(patch != fill)
    if coords.size == 0:
        return np.zeros((0, 0), dtype=int)
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    return patch[r0 : r1 + 1, c0 : c1 + 1].copy()


def _paste(out: np.ndarray, patch: np.ndarray, top: int, left: int, transparent: int) -> None:
    h, w = patch.shape
    mask = patch != transparent
    target = out[top : top + h, left : left + w]
    target[mask] = patch[mask]


def _frame_info(grid: np.ndarray, comp: dict, bg: int) -> dict | None:
    if len(comp["colors"]) != 1:
        return None
    r0, c0, r1, c1 = comp["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None
    color = int(comp["colors"][0])
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    if comp["area"] != 2 * (h + w) - 4:
        return None
    if not np.all(sub[0, :] == color):
        return None
    if not np.all(sub[-1, :] == color):
        return None
    if not np.all(sub[:, 0] == color):
        return None
    if not np.all(sub[:, -1] == color):
        return None
    if not np.all(sub[1:-1, 1:-1] == bg):
        return None
    return {
        "color": color,
        "bbox": (r0, c0, r1, c1),
        "inner_bbox": (r0 + 1, c0 + 1, r1 - 1, c1 - 1),
    }


def _primary_direction(fragment_bbox: tuple[int, int, int, int], frame_bbox: tuple[int, int, int, int]) -> str:
    fr0, fc0, fr1, fc1 = frame_bbox
    gr0, gc0, gr1, gc1 = fragment_bbox
    frame_center_r = (fr0 + fr1) / 2.0
    frame_center_c = (fc0 + fc1) / 2.0
    frag_center_r = (gr0 + gr1) / 2.0
    frag_center_c = (gc0 + gc1) / 2.0
    dr = frag_center_r - frame_center_r
    dc = frag_center_c - frame_center_c
    if abs(dr) >= abs(dc):
        return "bottom" if dr > 0 else "top"
    return "right" if dc > 0 else "left"


def _direction_sort_key(fragment: dict, direction: str, use_origin_tiebreak: bool) -> tuple:
    if not use_origin_tiebreak or "origin_bbox" not in fragment:
        return (-int(fragment["component_area"]), int(fragment.get("seq", 0)))
    r0, c0, r1, c1 = fragment["origin_bbox"]
    if direction == "top":
        outward = int(r0)
    elif direction == "bottom":
        outward = -int(r1)
    elif direction == "left":
        outward = int(c0)
    else:
        outward = -int(c1)
    return (-int(fragment["component_area"]), outward, int(r0), int(c0))


def _planned_positions(frame: dict, fragments: list[dict], *, use_origin_tiebreak: bool) -> list[tuple[dict, int, int]] | None:
    inner_r0, inner_c0, inner_r1, inner_c1 = frame["inner_bbox"]
    inner_h = inner_r1 - inner_r0 + 1
    inner_w = inner_c1 - inner_c0 + 1

    groups = {side: [] for side in SIDE_NAMES}
    for fragment in fragments:
        groups[str(fragment["direction"])].append(fragment)

    rects: list[tuple[int, int, int, int]] = []
    planned: list[tuple[dict, int, int]] = []

    top_cursor = inner_r0
    for fragment in sorted(groups["top"], key=lambda item: _direction_sort_key(item, "top", use_origin_tiebreak)):
        patch = fragment["content_patch"]
        ph, pw = patch.shape
        row = top_cursor
        col = inner_c0 + (inner_w - pw) // 2
        rect = (row, col, row + ph - 1, col + pw - 1)
        rects.append(rect)
        planned.append((fragment, row, col))
        top_cursor += ph

    bottom_cursor = inner_r1 + 1
    for fragment in sorted(groups["bottom"], key=lambda item: _direction_sort_key(item, "bottom", use_origin_tiebreak)):
        patch = fragment["content_patch"]
        ph, pw = patch.shape
        row = bottom_cursor - ph
        col = inner_c0 + (inner_w - pw) // 2
        rect = (row, col, row + ph - 1, col + pw - 1)
        rects.append(rect)
        planned.append((fragment, row, col))
        bottom_cursor -= ph

    left_cursor = inner_c0
    for fragment in sorted(groups["left"], key=lambda item: _direction_sort_key(item, "left", use_origin_tiebreak)):
        patch = fragment["content_patch"]
        ph, pw = patch.shape
        row = inner_r0 + (inner_h - ph) // 2
        col = left_cursor
        rect = (row, col, row + ph - 1, col + pw - 1)
        rects.append(rect)
        planned.append((fragment, row, col))
        left_cursor += pw

    right_cursor = inner_c1 + 1
    for fragment in sorted(groups["right"], key=lambda item: _direction_sort_key(item, "right", use_origin_tiebreak)):
        patch = fragment["content_patch"]
        ph, pw = patch.shape
        row = inner_r0 + (inner_h - ph) // 2
        col = right_cursor - pw
        rect = (row, col, row + ph - 1, col + pw - 1)
        rects.append(rect)
        planned.append((fragment, row, col))
        right_cursor -= pw

    for row0, col0, row1, col1 in rects:
        if row0 < inner_r0 or col0 < inner_c0 or row1 > inner_r1 or col1 > inner_c1:
            return None
    for i in range(len(rects)):
        a0, b0, a1, b1 = rects[i]
        for j in range(i + 1, len(rects)):
            c0, d0, c1, d1 = rects[j]
            if not (a1 < c0 or c1 < a0 or b1 < d0 or d1 < b0):
                return None
    return planned


def _extract_fragments(grid: np.ndarray, bg: int, frames: list[dict], other_components: list[dict]) -> list[dict]:
    frame_colors = {int(frame["color"]) for frame in frames}
    frames_by_color = {int(frame["color"]): frame for frame in frames}
    fragments: list[dict] = []
    for seq, comp in enumerate(other_components):
        colors = set(int(x) for x in comp["colors"])
        owner_colors = colors & frame_colors
        if len(owner_colors) != 1 or len(colors) != 2:
            continue
        owner_color = int(next(iter(owner_colors)))
        content_color = int(next(iter(colors - {owner_color})))
        r0, c0, r1, c1 = comp["bbox"]
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        content_only = np.where(sub == content_color, content_color, bg)
        content_patch = _crop_patch(content_only, bg)
        if content_patch.size == 0:
            continue
        frame = frames_by_color[owner_color]
        fragments.append(
            {
                "owner_color": owner_color,
                "content_color": content_color,
                "direction": _primary_direction(comp["bbox"], frame["bbox"]),
                "origin_bbox": comp["bbox"],
                "component_area": int(comp["area"]),
                "content_patch": content_patch,
                "seq": seq,
            }
        )
    return fragments


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group each two-color fragment into the matching hollow frame, strip its border,
    then place the remaining content against the interior side indicated by the
    fragment's original direction relative to that frame.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)

    components = _colored_components(g, bg)
    frames: list[dict] = []
    others: list[dict] = []
    for comp in components:
        frame = _frame_info(g, comp, bg)
        if frame is None:
            others.append(comp)
        else:
            frames.append(frame)

    out = g.copy()
    for comp in others:
        rows = comp["coords"][:, 0]
        cols = comp["coords"][:, 1]
        out[rows, cols] = bg

    fragments = _extract_fragments(g, bg, frames, others)
    by_frame: dict[int, list[dict]] = {int(frame["color"]): [] for frame in frames}
    for fragment in fragments:
        by_frame[int(fragment["owner_color"])].append(fragment)

    for frame in frames:
        planned = _planned_positions(frame, by_frame[int(frame["color"])], use_origin_tiebreak=True)
        assert planned is not None
        for fragment, row, col in planned:
            _paste(out, fragment["content_patch"], int(row), int(col), bg)
    return out


GEN_ATTEMPTS = 400
FENCE_COUNT_RANGE = (2, 3)
INNER_SIZE_MIN = 3
INNER_SIZE_MAX = 14
CONTENT_DIM_MIN = 1
CONTENT_DIM_MAX = 4
FRAGMENT_RETRIES = 40
LAYOUT_RETRIES = 120
CANVAS_EXTRA_MIN = 4
CANVAS_EXTRA_MAX = 10
OUTER_GAP_MIN = 1
OUTER_GAP_MAX = 4
RESERVE_PADDING = 2

PATTERN_BANK = [
    ("top",),
    ("bottom",),
    ("left",),
    ("right",),
    ("top", "top"),
    ("bottom", "bottom"),
    ("left", "left"),
    ("right", "right"),
    ("top", "bottom"),
    ("left", "right"),
    ("top", "left"),
    ("top", "right"),
    ("bottom", "left"),
    ("bottom", "right"),
    ("top", "bottom", "left", "right"),
]


def _dilate(mask: np.ndarray, connectivity: int) -> np.ndarray:
    out = mask.copy()
    dirs = DIR4 if connectivity == 4 else DIR8
    h, w = mask.shape
    for dr, dc in dirs:
        src_r0 = max(0, -dr)
        src_r1 = h - max(0, dr)
        src_c0 = max(0, -dc)
        src_c1 = w - max(0, dc)
        dst_r0 = max(0, dr)
        dst_r1 = h - max(0, -dr)
        dst_c0 = max(0, dc)
        dst_c1 = w - max(0, -dc)
        out[dst_r0:dst_r1, dst_c0:dst_c1] |= mask[src_r0:src_r1, src_c0:src_c1]
    return out


def _random_connected_mask(h: int, w: int) -> np.ndarray:
    area = random.randint(max(1, (h * w + 1) // 2), h * w)
    start = (random.randrange(h), random.randrange(w))
    cells = {start}
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for dr, dc in DIR4:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            break
        cells.add(random.choice(frontier))
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return _crop_patch(mask.astype(int), 0).astype(bool)


def _make_fragment(owner_color: int, content_color: int) -> dict:
    ch = random.randint(CONTENT_DIM_MIN, CONTENT_DIM_MAX)
    cw = random.randint(CONTENT_DIM_MIN, CONTENT_DIM_MAX)
    content_mask = _random_connected_mask(ch, cw)
    content_h, content_w = content_mask.shape
    canvas = np.zeros((content_h + 2, content_w + 2), dtype=bool)
    canvas[1 : 1 + content_h, 1 : 1 + content_w] = content_mask
    outline_mask = _dilate(canvas, connectivity=8 if random.random() < 0.5 else 4) & (~canvas)
    whole = np.full(canvas.shape, BG, dtype=int)
    whole[outline_mask] = owner_color
    whole[canvas] = content_color
    whole = _crop_patch(whole, BG)
    content_patch = np.where(whole == content_color, content_color, BG)
    content_patch = _crop_patch(content_patch, BG)
    return {
        "owner_color": owner_color,
        "content_color": content_color,
        "whole_patch": whole,
        "content_patch": content_patch,
        "whole_bbox_area": int(whole.shape[0] * whole.shape[1]),
        "component_area": int(np.count_nonzero(whole != BG)),
    }


def _factor_pairs(area: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for h in range(1, isqrt(area) + 1):
        if area % h != 0:
            continue
        w = area // h
        pairs.append((h, w))
        if h != w:
            pairs.append((w, h))
    return pairs


def _candidate_inner_shapes(fragments: list[dict]) -> list[tuple[int, int]]:
    total_area = sum(int(fragment["whole_bbox_area"]) for fragment in fragments)
    candidates: list[tuple[int, int]] = []
    pseudo_frame = {"inner_bbox": (0, 0, 0, 0)}
    for inner_h, inner_w in _factor_pairs(total_area):
        if not (INNER_SIZE_MIN <= inner_h <= INNER_SIZE_MAX and INNER_SIZE_MIN <= inner_w <= INNER_SIZE_MAX):
            continue
        pseudo_frame["inner_bbox"] = (0, 0, inner_h - 1, inner_w - 1)
        if _planned_positions(pseudo_frame, fragments, use_origin_tiebreak=False) is not None:
            candidates.append((inner_h, inner_w))
    return candidates


def _build_fence_config(frame_color: int, content_palette: list[int], seq_base: int) -> dict | None:
    directions = list(random.choice(PATTERN_BANK))
    random.shuffle(directions)
    used_areas = {side: set() for side in SIDE_NAMES}
    fragments: list[dict] = []
    seq = seq_base
    for direction in directions:
        built = None
        for _ in range(FRAGMENT_RETRIES):
            fragment = _make_fragment(frame_color, random.choice(content_palette))
            if int(fragment["component_area"]) in used_areas[direction]:
                continue
            fragment["direction"] = direction
            fragment["seq"] = seq
            built = fragment
            used_areas[direction].add(int(fragment["component_area"]))
            seq += 1
            break
        if built is None:
            return None
        fragments.append(built)

    shapes = _candidate_inner_shapes(fragments)
    if not shapes:
        return None
    inner_h, inner_w = random.choice(shapes)
    frame_h = inner_h + 2
    frame_w = inner_w + 2

    top_margin = max(
        [fragment["whole_patch"].shape[0] + OUTER_GAP_MAX + 1 for fragment in fragments if fragment["direction"] == "top"] + [1]
    )
    bottom_margin = max(
        [fragment["whole_patch"].shape[0] + OUTER_GAP_MAX + 1 for fragment in fragments if fragment["direction"] == "bottom"] + [1]
    )
    left_margin = max(
        [fragment["whole_patch"].shape[1] + OUTER_GAP_MAX + 1 for fragment in fragments if fragment["direction"] == "left"] + [1]
    )
    right_margin = max(
        [fragment["whole_patch"].shape[1] + OUTER_GAP_MAX + 1 for fragment in fragments if fragment["direction"] == "right"] + [1]
    )

    widest = max(fragment["whole_patch"].shape[1] for fragment in fragments)
    tallest = max(fragment["whole_patch"].shape[0] for fragment in fragments)
    extra_w = max(0, widest - frame_w)
    extra_h = max(0, tallest - frame_h)
    left_margin += extra_w // 2
    right_margin += extra_w - extra_w // 2
    top_margin += extra_h // 2
    bottom_margin += extra_h - extra_h // 2
    return {
        "color": int(frame_color),
        "fragments": fragments,
        "inner_shape": (int(inner_h), int(inner_w)),
        "frame_shape": (int(frame_h), int(frame_w)),
        "top_margin": int(top_margin),
        "bottom_margin": int(bottom_margin),
        "left_margin": int(left_margin),
        "right_margin": int(right_margin),
    }


def _mark_box(occ: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> None:
    occ[max(0, r0) : min(occ.shape[0], r1 + 1), max(0, c0) : min(occ.shape[1], c1 + 1)] = True


def _clear_box(occ: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> bool:
    rr0 = max(0, r0)
    cc0 = max(0, c0)
    rr1 = min(occ.shape[0] - 1, r1)
    cc1 = min(occ.shape[1] - 1, c1)
    if rr0 > rr1 or cc0 > cc1:
        return True
    return not bool(occ[rr0 : rr1 + 1, cc0 : cc1 + 1].any())


def _place_reserves(fences: list[dict], height: int, width: int) -> bool:
    occ = np.zeros((height, width), dtype=bool)
    order = list(range(len(fences)))
    random.shuffle(order)
    for idx in order:
        fence = fences[idx]
        frame_h, frame_w = fence["frame_shape"]
        reserve_h = frame_h + fence["top_margin"] + fence["bottom_margin"]
        reserve_w = frame_w + fence["left_margin"] + fence["right_margin"]
        if reserve_h + 2 * RESERVE_PADDING > height or reserve_w + 2 * RESERVE_PADDING > width:
            return False
        placed = False
        for _ in range(LAYOUT_RETRIES):
            top = random.randint(RESERVE_PADDING, height - reserve_h - RESERVE_PADDING)
            left = random.randint(RESERVE_PADDING, width - reserve_w - RESERVE_PADDING)
            if not _clear_box(occ, top - RESERVE_PADDING, left - RESERVE_PADDING, top + reserve_h - 1 + RESERVE_PADDING, left + reserve_w - 1 + RESERVE_PADDING):
                continue
            _mark_box(occ, top - RESERVE_PADDING, left - RESERVE_PADDING, top + reserve_h - 1 + RESERVE_PADDING, left + reserve_w - 1 + RESERVE_PADDING)
            fence["reserve_bbox"] = (top, left, top + reserve_h - 1, left + reserve_w - 1)
            frame_top = top + fence["top_margin"]
            frame_left = left + fence["left_margin"]
            fence["bbox"] = (frame_top, frame_left, frame_top + frame_h - 1, frame_left + frame_w - 1)
            fence["inner_bbox"] = (frame_top + 1, frame_left + 1, frame_top + frame_h - 2, frame_left + frame_w - 2)
            placed = True
            break
        if not placed:
            return False
    return True


def _fragment_outside_ok(direction: str, fragment_bbox: tuple[int, int, int, int], frame_bbox: tuple[int, int, int, int]) -> bool:
    if _primary_direction(fragment_bbox, frame_bbox) != direction:
        return False
    fr0, fc0, fr1, fc1 = frame_bbox
    gr0, gc0, gr1, gc1 = fragment_bbox
    if direction == "top":
        return gr1 < fr0 - 1
    if direction == "bottom":
        return gr0 > fr1 + 1
    if direction == "left":
        return gc1 < fc0 - 1
    return gc0 > fc1 + 1


def _place_fragments_for_fence(fence: dict) -> bool:
    top, left, bottom, right = fence["reserve_bbox"]
    reserve_h = bottom - top + 1
    reserve_w = right - left + 1
    local_occ = np.zeros((reserve_h, reserve_w), dtype=bool)

    fr0, fc0, fr1, fc1 = fence["bbox"]
    local_frame = (fr0 - top, fc0 - left, fr1 - top, fc1 - left)
    _mark_box(local_occ, local_frame[0] - 1, local_frame[1] - 1, local_frame[2] + 1, local_frame[3] + 1)

    fragments = sorted(
        fence["fragments"],
        key=lambda item: (-int(item["whole_bbox_area"]), -int(item["component_area"]), int(item["seq"])),
    )
    for fragment in fragments:
        patch = fragment["whole_patch"]
        ph, pw = patch.shape
        candidates: list[tuple[int, int]] = []
        for local_r in range(reserve_h - ph + 1):
            for local_c in range(reserve_w - pw + 1):
                if not _clear_box(local_occ, local_r - 1, local_c - 1, local_r + ph, local_c + pw):
                    continue
                global_bbox = (top + local_r, left + local_c, top + local_r + ph - 1, left + local_c + pw - 1)
                if _fragment_outside_ok(fragment["direction"], global_bbox, fence["bbox"]):
                    candidates.append((local_r, local_c))
        if not candidates:
            return False
        local_r, local_c = random.choice(candidates)
        fragment["origin_bbox"] = (top + local_r, left + local_c, top + local_r + ph - 1, left + local_c + pw - 1)
        _mark_box(local_occ, local_r - 1, local_c - 1, local_r + ph, local_c + pw)
    return True


def _draw_frame(grid: np.ndarray, frame: dict) -> None:
    r0, c0, r1, c1 = frame["bbox"]
    color = int(frame["color"])
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _render_expected(height: int, width: int, fences: list[dict]) -> np.ndarray:
    out = np.full((height, width), BG, dtype=int)
    for fence in fences:
        _draw_frame(out, fence)
        planned = _planned_positions(fence, fence["fragments"], use_origin_tiebreak=True)
        if planned is None:
            raise ValueError("invalid fence layout")
        for fragment, row, col in planned:
            _paste(out, fragment["content_patch"], int(row), int(col), BG)
    return out


def _render_input(height: int, width: int, fences: list[dict]) -> np.ndarray:
    inp = np.full((height, width), BG, dtype=int)
    for fence in fences:
        _draw_frame(inp, fence)
    for fence in fences:
        for fragment in fence["fragments"]:
            r0, c0, _r1, _c1 = fragment["origin_bbox"]
            _paste(inp, fragment["whole_patch"], int(r0), int(c0), BG)
    return inp


def _gen_attempt() -> np.ndarray | None:
    n_fences = random.randint(*FENCE_COUNT_RANGE)
    frame_colors = random.sample(list(range(1, 10)), n_fences)
    content_palette = [color for color in range(1, 10) if color not in frame_colors]
    if not content_palette:
        return None

    fences: list[dict] = []
    seq_base = 0
    for frame_color in frame_colors:
        built = None
        for _ in range(FRAGMENT_RETRIES):
            built = _build_fence_config(frame_color, content_palette, seq_base)
            if built is not None:
                seq_base += len(built["fragments"])
                break
        if built is None:
            return None
        fences.append(built)

    reserve_dims = []
    for fence in fences:
        frame_h, frame_w = fence["frame_shape"]
        reserve_dims.append(
            (
                frame_h + fence["top_margin"] + fence["bottom_margin"],
                frame_w + fence["left_margin"] + fence["right_margin"],
            )
        )

    total_area = sum(h * w for h, w in reserve_dims)
    min_h = max(h for h, _ in reserve_dims) + 2 * RESERVE_PADDING
    min_w = max(w for _, w in reserve_dims) + 2 * RESERVE_PADDING
    base = max(int(np.sqrt(total_area) * 1.35) + CANVAS_EXTRA_MIN, 24)
    height = random.randint(max(min_h, base), max(min_h, base) + CANVAS_EXTRA_MAX)
    width = random.randint(max(min_w, base), max(min_w, base) + CANVAS_EXTRA_MAX)

    if not _place_reserves(fences, height, width):
        return None
    for fence in fences:
        if not _place_fragments_for_fence(fence):
            return None

    inp = _render_input(height, width, fences)
    expected = _render_expected(height, width, fences)
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC instance")
