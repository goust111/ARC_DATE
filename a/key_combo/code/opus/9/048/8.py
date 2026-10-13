from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


MAX_ATTEMPTS = 400
BG = 0
PALETTE = tuple(range(1, 10))
SEGMENT_WIDTH_CHOICES = (3, 5, 7)


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _top_band_height(grid: np.ndarray, bg: int) -> int:
    h = 0
    for r in range(grid.shape[0]):
        if np.all(grid[r] != int(bg)):
            h += 1
        else:
            break
    if h <= 0:
        raise AssertionError("expected a non-empty full-width top band")
    return int(h)


def _extract_segments(row: np.ndarray) -> list[dict]:
    row = np.asarray(row, dtype=int)
    segments = []
    c = 0
    while c < row.size:
        color = int(row[c])
        c1 = c
        while c1 + 1 < row.size and int(row[c1 + 1]) == color:
            c1 += 1
        segments.append({"color": color, "start": int(c), "end": int(c1)})
        c = c1 + 1
    return segments


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _component_patch(
    grid: np.ndarray, comp: dict, bg: int
) -> np.ndarray:
    r0, c0, r1, c1 = comp["bbox"]
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
    for r, c in comp["coords"]:
        patch[r - r0, c - c0] = int(comp["color"])
    return patch


def _find_components(
    grid: np.ndarray, bg: int, *, connectivity: int = 4
) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    visited = np.zeros((H, W), dtype=bool)
    comps = []
    if int(connectivity) == 4:
        nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        nbrs = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ]
    for r in range(H):
        for c in range(W):
            color = int(g[r, c])
            if color == int(bg) or visited[r, c]:
                continue
            q = deque([(r, c)])
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((int(rr), int(cc)))
                for dr, dc in nbrs:
                    nr = rr + dr
                    nc = cc + dc
                    if not (0 <= nr < H and 0 <= nc < W):
                        continue
                    if visited[nr, nc] or int(g[nr, nc]) != color:
                        continue
                    visited[nr, nc] = True
                    q.append((nr, nc))
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox_from_coords(coords),
                    "area": len(coords),
                }
            )
    return comps


def _paste_patch(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, bg: int) -> None:
    h, w = patch.shape
    view = dst[r0 : r0 + h, c0 : c0 + w]
    mask = patch != int(bg)
    view[mask] = patch[mask]


def _overlaps(occupied: np.ndarray, patch: np.ndarray, r0: int, c0: int, bg: int) -> bool:
    h, w = patch.shape
    if r0 < 0 or c0 < 0 or r0 + h > occupied.shape[0] or c0 + w > occupied.shape[1]:
        return True
    return bool((occupied[r0 : r0 + h, c0 : c0 + w] & (patch != int(bg))).any())


def _fits_with_padding(
    occupied: np.ndarray, patch: np.ndarray, r0: int, c0: int, *, bg: int, padding: int
) -> bool:
    h, w = patch.shape
    H, W = occupied.shape
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False
    for pr, pc in np.argwhere(patch != int(bg)):
        rr = int(r0 + pr)
        cc = int(c0 + pc)
        r1 = max(0, rr - padding)
        r2 = min(H, rr + padding + 1)
        c1 = max(0, cc - padding)
        c2 = min(W, cc + padding + 1)
        if occupied[r1:r2, c1:c2].any():
            return False
    return True


def _patch_from_lines(lines: tuple[str, ...]) -> np.ndarray:
    arr = np.array([[1 if ch == "#" else 0 for ch in row] for row in lines], dtype=int)
    return arr


_BASE_TEMPLATES = {
    "v3": _patch_from_lines(("#", "#", "#")),
    "v4": _patch_from_lines(("#", "#", "#", "#")),
    "h3": _patch_from_lines(("###",)),
    "t4": _patch_from_lines(("###", ".#.")),
    "l4": _patch_from_lines(("#..", "###")),
    "j4": _patch_from_lines(("..#", "###")),
    "z4": _patch_from_lines(("##.", ".##")),
    "u5": _patch_from_lines(("#.#", "###")),
    "plus5": _patch_from_lines((".#.", "###", ".#.")),
}

_GENERAL_TEMPLATE_NAMES = ("v3", "v4", "h3", "t4", "l4", "j4", "z4", "u5", "plus5")


def _colorize(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask != 0] = int(color)
    return patch


def _segment_center_start(seg: dict, patch_width: int) -> int:
    seg_w = int(seg["end"]) - int(seg["start"]) + 1
    if patch_width > seg_w:
        raise AssertionError("sprite wider than its target segment")
    return int(seg["start"] + (seg_w - patch_width) // 2)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror the top color band to the bottom, then center each same-color sprite inside that band's segment.

    Input structure:
    - Black background surrounds a full-width top band of 1-2 non-black rows.
    - The band is split into contiguous horizontal color segments, each color appearing in exactly one segment.
    - Below the band are several 4-connected monochrome sprites whose colors match band colors; the bottom band rows are empty.

    Rule:
    - Copy the top band to the bottom rows in vertically reversed row order.
    - Read the top band's left-to-right contiguous color segments as closed column intervals.
    - For every middle sprite, find the segment with the same color and translate the sprite horizontally so its bounding box is centered in that interval while keeping its original row.
    - If multiple sprites of one color would overlap after centering, process them from top to bottom and push each later sprite downward one row at a time until no pixels overlap.

    Key insight:
    - The mirrored band is a functional coordinate system: its color boundaries, not another template object, encode the horizontal destination zone for each sprite.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    H, W = g.shape

    band_h = _top_band_height(g, bg)
    top_band = g[:band_h, :].copy()
    if not all(np.array_equal(top_band[r], top_band[0]) for r in range(band_h)):
        raise AssertionError("all band rows must share the same segment pattern")

    segments = _extract_segments(top_band[0])
    if any(seg["color"] == int(bg) for seg in segments):
        raise AssertionError("band segments must be non-background")
    color_to_segment = {}
    for seg in segments:
        color = int(seg["color"])
        if color in color_to_segment:
            raise AssertionError("each band color must occupy exactly one segment")
        color_to_segment[color] = seg

    middle = g.copy()
    middle[:band_h, :] = int(bg)
    middle[H - band_h :, :] = int(bg)
    comps = _find_components(middle, bg, connectivity=4)

    out = np.full_like(g, int(bg))
    out[:band_h, :] = top_band
    out[H - band_h :, :] = top_band[::-1, :]
    occupied = out != int(bg)

    groups: dict[int, list[dict]] = {}
    for comp in comps:
        groups.setdefault(int(comp["color"]), []).append(comp)

    for color in [int(seg["color"]) for seg in segments]:
        seg = color_to_segment[color]
        comps_of_color = sorted(groups.get(color, []), key=lambda comp: (comp["bbox"][0], comp["bbox"][1]))
        for comp in comps_of_color:
            patch = _component_patch(middle, comp, bg)
            h, w = patch.shape
            dest_c0 = _segment_center_start(seg, w)
            dest_r0 = int(comp["bbox"][0])
            while _overlaps(occupied, patch, dest_r0, dest_c0, bg):
                dest_r0 += 1
                if dest_r0 + h > H - band_h:
                    raise AssertionError("sprite cannot be placed without leaving the content area")
            _paste_patch(out, patch, dest_r0, dest_c0, bg)
            occupied[dest_r0 : dest_r0 + h, dest_c0 : dest_c0 + w] |= patch != int(bg)

    return out


def _pick_segments() -> list[dict] | None:
    n = random.choice((3, 4))
    colors = random.sample(PALETTE, n)
    widths = [random.choice(SEGMENT_WIDTH_CHOICES) for _ in range(n)]
    total_w = sum(widths)
    if total_w > 24:
        return None
    segments = []
    c = 0
    for color, width in zip(colors, widths):
        segments.append({"color": int(color), "start": int(c), "end": int(c + width - 1)})
        c += width
    return segments


def _pick_counts(n_segments: int) -> list[int]:
    counts = [1] * n_segments
    if n_segments == 3:
        extra = random.choice((1, 1, 2))
    else:
        extra = random.choice((0, 1, 1, 2))
    for idx in random.sample(range(n_segments), extra):
        counts[idx] += 1
    return counts


def _general_template(seg_w: int) -> np.ndarray:
    candidates = [
        _BASE_TEMPLATES[name]
        for name in _GENERAL_TEMPLATE_NAMES
        if _BASE_TEMPLATES[name].shape[1] <= seg_w
    ]
    return random.choice(candidates).copy()


def _choose_group_plan(
    seg: dict, count: int, band_h: int, H: int, collision: bool
) -> list[dict] | None:
    max_row = H - band_h - 1
    objects = []

    if count == 2 and collision:
        patch1 = _colorize(_BASE_TEMPLATES["plus5"], seg["color"])
        patch2 = _colorize(_BASE_TEMPLATES["u5"], seg["color"])
        h1 = patch1.shape[0]
        h2 = patch2.shape[0]
        upper = max_row - (h1 + h2 - 1)
        if upper < band_h + 1:
            return None
        row1 = random.randint(band_h + 1, upper)
        row2_out = row1 + h1 - 1
        row2_in = row1 + 1
        objects.append(
            {
                "color": int(seg["color"]),
                "segment": seg,
                "patch": patch1,
                "row_in": int(row1),
                "row_out": int(row1),
                "collision_expected": False,
            }
        )
        objects.append(
            {
                "color": int(seg["color"]),
                "segment": seg,
                "patch": patch2,
                "row_in": int(row2_in),
                "row_out": int(row2_out),
                "collision_expected": True,
            }
        )
        return objects

    prev_end = None
    for i in range(count):
        patch = _colorize(_general_template(seg["end"] - seg["start"] + 1), seg["color"])
        h = patch.shape[0]
        if prev_end is None:
            row = random.randint(band_h + 1, max_row - h + 1)
        else:
            min_row = prev_end + 2
            if min_row > max_row - h + 1:
                return None
            row = random.randint(min_row, max_row - h + 1)
        prev_end = row + h - 1
        objects.append(
            {
                "color": int(seg["color"]),
                "segment": seg,
                "patch": patch,
                "row_in": int(row),
                "row_out": int(row),
                "collision_expected": False,
            }
        )
    return objects


def _build_band(segments: list[dict], band_h: int) -> np.ndarray:
    W = segments[-1]["end"] + 1
    row = np.zeros(W, dtype=int)
    for seg in segments:
        row[seg["start"] : seg["end"] + 1] = int(seg["color"])
    return np.repeat(row[None, :], band_h, axis=0)


def _expected_output(shape: tuple[int, int], band: np.ndarray, objects: list[dict]) -> np.ndarray:
    H, W = shape
    band_h = band.shape[0]
    out = np.zeros((H, W), dtype=int)
    out[:band_h, :] = band
    out[H - band_h :, :] = band[::-1, :]
    for obj in objects:
        patch = obj["patch"]
        c0 = _segment_center_start(obj["segment"], patch.shape[1])
        _paste_patch(out, patch, int(obj["row_out"]), int(c0), BG)
    return out


def _choose_cross_flags(objects: list[dict]) -> list[bool]:
    flags = [False] * len(objects)
    must = random.sample(range(len(objects)), random.randint(1, max(1, len(objects) // 2)))
    for idx in must:
        flags[idx] = True
    return flags


def _crosses_segment(seg: dict, c0: int, width: int) -> bool:
    return c0 < int(seg["start"]) or c0 + width - 1 > int(seg["end"])


def _place_input_objects(shape: tuple[int, int], objects: list[dict], band: np.ndarray) -> np.ndarray | None:
    H, W = shape
    band_h = band.shape[0]
    grid = np.zeros((H, W), dtype=int)
    grid[:band_h, :] = band
    occupied = grid != BG
    cross_flags = _choose_cross_flags(objects)

    order = list(range(len(objects)))
    random.shuffle(order)
    order.sort(key=lambda idx: (-int(cross_flags[idx]), -objects[idx]["patch"].shape[1], objects[idx]["row_in"]))

    actual_crosses = 0
    for idx in order:
        obj = objects[idx]
        patch = obj["patch"]
        r0 = int(obj["row_in"])
        h, w = patch.shape
        center_c0 = _segment_center_start(obj["segment"], w)
        candidates = []
        for c0 in range(W - w + 1):
            if c0 == center_c0:
                continue
            cross = _crosses_segment(obj["segment"], c0, w)
            if cross_flags[idx] and not cross:
                continue
            if not _fits_with_padding(occupied, patch, r0, c0, bg=BG, padding=1):
                continue
            score = (
                int(cross),
                abs(c0 - center_c0),
                random.random(),
            )
            candidates.append((score, c0))
        if not candidates and cross_flags[idx]:
            for c0 in range(W - w + 1):
                if c0 == center_c0:
                    continue
                if not _fits_with_padding(occupied, patch, r0, c0, bg=BG, padding=1):
                    continue
                score = (
                    int(_crosses_segment(obj["segment"], c0, w)),
                    abs(c0 - center_c0),
                    random.random(),
                )
                candidates.append((score, c0))
        if not candidates:
            return None
        candidates.sort(reverse=True)
        _, c0 = candidates[0]
        obj["col_in"] = int(c0)
        obj["col_out"] = int(center_c0)
        if _crosses_segment(obj["segment"], c0, w):
            actual_crosses += 1
        _paste_patch(grid, patch, r0, c0, BG)
        occupied[r0 : r0 + h, c0 : c0 + w] |= patch != BG

    if actual_crosses == 0:
        return None
    return grid


def _mask_key(patch: np.ndarray) -> tuple[bytes, tuple[int, int]]:
    mask = (patch != BG).astype(np.uint8)
    return mask.tobytes(), mask.shape


def _collect_content_components(grid: np.ndarray) -> tuple[int, list[dict], list[dict]]:
    bg = _get_bg(grid)
    band_h = _top_band_height(grid, bg)
    content = np.array(grid, copy=True)
    content[:band_h, :] = int(bg)
    content[grid.shape[0] - band_h :, :] = int(bg)
    comps = _find_components(content, bg, connectivity=4)
    return band_h, _extract_segments(grid[0]), comps


def _validate_scene(inp: np.ndarray, expected: np.ndarray, objects: list[dict]) -> None:
    assert np.array_equal(main(inp), expected)

    band_h_in, segments_in, comps_in = _collect_content_components(inp)
    band_h_out, segments_out, _comps_out = _collect_content_components(expected)
    assert band_h_in == band_h_out
    assert segments_in == segments_out

    expected_bottom = inp[:band_h_in, :][::-1, :]
    assert np.array_equal(expected[-band_h_in:, :], expected_bottom)

    plan_by_color: dict[int, list[dict]] = {}
    in_by_color: dict[int, list[dict]] = {}
    for obj in objects:
        plan_by_color.setdefault(int(obj["color"]), []).append(obj)
    for comp in comps_in:
        in_by_color.setdefault(int(comp["color"]), []).append(comp)

    moved_count = 0
    cross_count = 0
    downshift_count = 0

    for color, planned in plan_by_color.items():
        planned = sorted(planned, key=lambda obj: (obj["row_in"], obj["col_in"]))
        got_in = sorted(in_by_color.get(color, []), key=lambda comp: (comp["bbox"][0], comp["bbox"][1]))
        assert len(planned) == len(got_in)
        for obj, comp_in in zip(planned, got_in):
            patch = obj["patch"]
            patch_in = _component_patch(inp, comp_in, BG)
            assert _mask_key(patch_in) == _mask_key(patch)
            assert comp_in["bbox"][0] == int(obj["row_in"])
            assert comp_in["bbox"][1] == int(obj["col_in"])
            h, w = patch.shape
            view = expected[
                int(obj["row_out"]) : int(obj["row_out"]) + h,
                int(obj["col_out"]) : int(obj["col_out"]) + w,
            ]
            assert np.all(view[patch != BG] == int(obj["color"]))
            assert int(obj["row_out"]) >= comp_in["bbox"][0]
            if int(obj["col_in"]) != int(obj["col_out"]):
                moved_count += 1
            if _crosses_segment(obj["segment"], int(obj["col_in"]), patch.shape[1]):
                cross_count += 1
            if int(obj["row_out"]) > int(obj["row_in"]):
                downshift_count += 1

    assert moved_count >= 1
    assert cross_count >= 1
    collision_planned = any(bool(obj["collision_expected"]) for obj in objects)
    if collision_planned:
        assert downshift_count >= 1


def _objects_stay_distinct(grid: np.ndarray, objects: list[dict]) -> bool:
    _, _segments, comps = _collect_content_components(grid)
    counts_by_color: dict[int, int] = {}
    for obj in objects:
        counts_by_color[int(obj["color"])] = counts_by_color.get(int(obj["color"]), 0) + 1
    seen_by_color: dict[int, int] = {}
    for comp in comps:
        color = int(comp["color"])
        seen_by_color[color] = seen_by_color.get(color, 0) + 1
    return seen_by_color == counts_by_color


def _gen_attempt() -> np.ndarray | None:
    segments = _pick_segments()
    if segments is None:
        return None
    band_h = random.choice((1, 2))
    H = random.randint(16, 24)
    W = segments[-1]["end"] + 1
    counts = _pick_counts(len(segments))
    two_sprite_idxs = [idx for idx, count in enumerate(counts) if count == 2]
    collision_idx = None
    if two_sprite_idxs and random.random() < 0.65:
        collision_idx = random.choice(two_sprite_idxs)

    objects: list[dict] = []
    for idx, (seg, count) in enumerate(zip(segments, counts)):
        group = _choose_group_plan(seg, count, band_h, H, collision=idx == collision_idx)
        if group is None:
            return None
        objects.extend(group)

    band = _build_band(segments, band_h)
    expected = _expected_output((H, W), band, objects)
    if not _objects_stay_distinct(expected, objects):
        return None
    inp = _place_input_objects((H, W), objects, band)
    if inp is None:
        return None

    if np.any(inp[H - band_h :, :] != BG):
        return None

    try:
        _validate_scene(inp, expected, objects)
    except AssertionError:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirrored-band alignment sample")
