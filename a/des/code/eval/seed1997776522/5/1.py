from __future__ import annotations

import itertools
import random
from collections import deque

import numpy as np


BG = 0
FRAME = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
MAX_GENERATION_ATTEMPTS = 300
TEMPLATE_BUILD_ATTEMPTS = 120


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            parts.append(coords)
    return parts


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[counts.argmax()])


def _descriptor(coords: list[tuple[int, int]], shape: tuple[int, int]) -> str:
    r0, c0, r1, c1 = _bbox(coords)
    cy = (shape[0] - 1) / 2.0
    cx = (shape[1] - 1) / 2.0
    dy = (((r0 + r1) / 2.0) - cy) / max(cy, 1.0)
    dx = (((c0 + c1) / 2.0) - cx) / max(cx, 1.0)
    threshold = 0.18
    if abs(dx) <= threshold and abs(dy) > threshold:
        return "top" if dy < 0 else "bottom"
    if abs(dy) <= threshold and abs(dx) > threshold:
        return "left" if dx < 0 else "right"
    return ("t" if dy < 0 else "b") + ("l" if dx < 0 else "r")


def _ranked_bg_regions(inner: np.ndarray, bg: int = BG, frame_color: int = FRAME) -> list[dict[str, object]]:
    parts = _components(inner == bg)
    ranked: list[dict[str, object]] = []
    for coords in parts:
        ranked.append(
            {
                "coords": coords,
                "area": len(coords),
                "desc": _descriptor(coords, inner.shape),
            }
        )
    ranked.sort(key=lambda item: -int(item["area"]))
    areas = [int(item["area"]) for item in ranked]
    assert len(areas) == len(set(areas)), "partition region areas must be distinct"
    return ranked


def _ranked_color_parts(block: np.ndarray, bg: int = BG, frame_color: int = FRAME) -> list[dict[str, object]]:
    ranked: list[dict[str, object]] = []
    for color in sorted(int(v) for v in np.unique(block)):
        if color in (bg, frame_color):
            continue
        coords = list(map(tuple, np.argwhere(block == color)))
        ranked.append(
            {
                "color": color,
                "coords": coords,
                "area": len(coords),
                "desc": _descriptor(coords, block.shape),
            }
        )
    ranked.sort(key=lambda item: -int(item["area"]))
    areas = [int(item["area"]) for item in ranked]
    assert len(areas) == len(set(areas)), "template color areas must be distinct"
    return ranked


def _is_frame_component(grid: np.ndarray, color: int, coords: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h != w or h < 5:
        return False
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    if not (
        np.all(sub[0, :] == color)
        and np.all(sub[-1, :] == color)
        and np.all(sub[:, 0] == color)
        and np.all(sub[:, -1] == color)
    ):
        return False
    return True


def _extract_frames(grid: np.ndarray, bg: int, frame_color: int) -> list[dict[str, object]]:
    frames: list[dict[str, object]] = []
    for coords in _components(grid == frame_color):
        if not _is_frame_component(grid, frame_color, coords):
            continue
        r0, c0, r1, c1 = _bbox(coords)
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        inner = sub[1:-1, 1:-1]
        colors = {int(v) for v in np.unique(inner) if int(v) not in (bg, frame_color)}
        frames.append(
            {
                "bbox": (r0, c0, r1, c1),
                "inner": inner.copy(),
                "color_set": frozenset(colors),
            }
        )
    frames.sort(key=lambda item: tuple(int(v) for v in item["bbox"]))
    assert frames, "no frames found"
    side_lengths = {(int(r1) - int(r0) - 1) for r0, _c0, r1, _c1 in (frame["bbox"] for frame in frames)}
    assert len(side_lengths) == 1, "all frames must share one interior size"
    return frames


def _extract_templates(grid: np.ndarray, bg: int, frame_boxes: list[tuple[int, int, int, int]]) -> list[dict[str, object]]:
    mask = grid != bg
    for r0, c0, r1, c1 in frame_boxes:
        mask[r0 : r1 + 1, c0 : c1 + 1] = False
    templates: list[dict[str, object]] = []
    for coords in _components(mask):
        r0, c0, r1, c1 = _bbox(coords)
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        if np.any(sub == bg) or np.any(sub == FRAME):
            continue
        if len(coords) != sub.size:
            continue
        templates.append(
            {
                "bbox": (r0, c0, r1, c1),
                "grid": sub.copy(),
                "color_set": frozenset(int(v) for v in np.unique(sub)),
            }
        )
    templates.sort(key=lambda item: tuple(int(v) for v in item["bbox"]))
    assert templates, "no templates found"
    return templates


def _target_rotation(empty_inner: np.ndarray, template: np.ndarray, bg: int, frame_color: int) -> int:
    region_sig = [str(part["desc"]) for part in _ranked_bg_regions(empty_inner, bg=bg, frame_color=frame_color)]
    hits: list[int] = []
    for k in range(4):
        rotated = np.rot90(template, k)
        parts = _ranked_color_parts(rotated, bg=bg, frame_color=frame_color)
        if len(parts) != len(region_sig):
            continue
        if [str(part["desc"]) for part in parts] == region_sig:
            hits.append(k)
    assert len(hits) == 1, "rotation must be unique"
    return hits[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match templates to frames by color-set elimination, then orient the leftover template
    by comparing template color-area ranking with the empty frame's partition-area ranking.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    frame_color = FRAME

    frames = _extract_frames(g, bg, frame_color)
    templates = _extract_templates(g, bg, [tuple(frame["bbox"]) for frame in frames])

    colored_frames = [frame for frame in frames if frame["color_set"]]
    empty_frames = [frame for frame in frames if not frame["color_set"]]
    assert len(empty_frames) == 1, "there must be exactly one empty frame"
    empty_frame = empty_frames[0]

    used_template_ids: set[int] = set()
    for frame in colored_frames:
        matches = [i for i, tmpl in enumerate(templates) if tmpl["color_set"] == frame["color_set"]]
        assert len(matches) == 1, "colored frame must match exactly one template by color set"
        used_template_ids.add(matches[0])

    leftover = [i for i in range(len(templates)) if i not in used_template_ids]
    assert len(leftover) == 1, "exactly one template must remain after elimination"
    target_template = templates[leftover[0]]

    r0, c0, r1, c1 = empty_frame["bbox"]
    inner = empty_frame["inner"]
    assert inner.shape == target_template["grid"].shape, "template size must match frame interior size"

    rot = _target_rotation(inner, target_template["grid"], bg=bg, frame_color=frame_color)
    oriented = np.rot90(target_template["grid"], rot)

    out = g.copy()
    patch = out[r0 + 1 : r1, c0 + 1 : c1]
    patch[patch == bg] = oriented[patch == bg]
    out[r0 + 1 : r1, c0 + 1 : c1] = patch
    return out


def _canonical_frame_lines(kind: int, inner_size: int) -> np.ndarray:
    lines = np.zeros((inner_size, inner_size), dtype=bool)
    if kind == 2:
        lines[:, 1] = True
    elif kind == 3:
        lines[:, 1] = True
        lines[1, 2:] = True
    elif kind == 4:
        lines[:, 1] = True
        lines[2, :] = True
    else:
        raise AssertionError("unsupported region count")
    return lines


def _random_stair(length: int, base: int, lo: int, hi: int) -> list[int]:
    vals = [max(lo, min(hi, base + random.choice([-1, 0, 1])))]
    for _ in range(length - 1):
        vals.append(max(lo, min(hi, vals[-1] + random.choice([-1, 0, 1]))))
    return vals


def _template_roles(kind: int, inner_size: int) -> np.ndarray:
    vb = _random_stair(inner_size, 1, 1, 2)
    if kind == 2:
        roles = np.zeros((inner_size, inner_size), dtype=int)
        for r in range(inner_size):
            for c in range(inner_size):
                roles[r, c] = 0 if c <= vb[r] else 1
        return roles

    hb_base = 1 if kind == 3 else 2
    hb = _random_stair(inner_size, hb_base, hb_base, min(inner_size - 2, hb_base + 1))
    roles = np.zeros((inner_size, inner_size), dtype=int)
    for r in range(inner_size):
        for c in range(inner_size):
            if kind == 3:
                if c <= vb[r]:
                    roles[r, c] = 0
                elif r <= hb[c]:
                    roles[r, c] = 1
                else:
                    roles[r, c] = 2
            else:
                if c <= vb[r] and r <= hb[c]:
                    roles[r, c] = 0
                elif c > vb[r] and r <= hb[c]:
                    roles[r, c] = 1
                elif c <= vb[r] and r > hb[c]:
                    roles[r, c] = 2
                else:
                    roles[r, c] = 3
    return roles


def _ranked_descs_from_frame_lines(lines: np.ndarray) -> list[str]:
    inner = np.full(lines.shape, BG, dtype=int)
    inner[lines] = FRAME
    return [str(part["desc"]) for part in _ranked_bg_regions(inner)]


def _ranked_descs_from_roles(roles: np.ndarray) -> list[str]:
    out: list[tuple[int, str]] = []
    for role in range(int(roles.max()) + 1):
        coords = list(map(tuple, np.argwhere(roles == role)))
        out.append((len(coords), _descriptor(coords, roles.shape)))
    out.sort(key=lambda item: -item[0])
    areas = [area for area, _desc in out]
    if len(areas) != len(set(areas)):
        return []
    return [desc for _area, desc in out]


def _render_frame(lines: np.ndarray, fill_colors: list[int] | None) -> np.ndarray:
    inner_size = lines.shape[0]
    frame = np.full((inner_size + 2, inner_size + 2), BG, dtype=int)
    frame[0, :] = FRAME
    frame[-1, :] = FRAME
    frame[:, 0] = FRAME
    frame[:, -1] = FRAME
    inner = frame[1:-1, 1:-1]
    inner[lines] = FRAME
    if fill_colors is not None:
        regions = _ranked_bg_regions(inner)
        assert len(regions) == len(fill_colors)
        for color, region in zip(fill_colors, regions, strict=False):
            for r, c in region["coords"]:
                inner[r, c] = int(color)
    frame[1:-1, 1:-1] = inner
    return frame


def _map_roles_to_colors(roles: np.ndarray, role_colors: list[int]) -> np.ndarray:
    out = np.empty_like(roles, dtype=int)
    for role, color in enumerate(role_colors):
        out[roles == role] = int(color)
    return out


def _choose_template_and_colors(kind: int, inner_size: int, colors: tuple[int, ...]) -> tuple[np.ndarray, list[int]]:
    frame_sig = _ranked_descs_from_frame_lines(_canonical_frame_lines(kind, inner_size))
    role_count = kind
    for _ in range(TEMPLATE_BUILD_ATTEMPTS):
        roles = _template_roles(kind, inner_size)
        if _ranked_descs_from_roles(roles) != frame_sig:
            continue
        role_colors = random.sample(list(colors), role_count)
        colored = _map_roles_to_colors(roles, role_colors)
        ranked = _ranked_color_parts(colored)
        ranked_colors = [int(part["color"]) for part in ranked]
        return colored, ranked_colors
    raise RuntimeError("failed to build a template with matching ranked descriptors")


def _sample_unique_color_sets(pair_count: int, sizes: list[int]) -> list[tuple[int, ...]]:
    chosen: list[tuple[int, ...]] = []
    used: set[tuple[int, ...]] = set()
    for size in sizes:
        choices = [combo for combo in itertools.combinations(COLOR_POOL, size) if combo not in used]
        picked = random.choice(choices)
        chosen.append(tuple(int(v) for v in picked))
        used.add(picked)
    return chosen


def _row_counts(total: int, rows: int) -> list[int]:
    if rows == 1:
        return [total]
    cut = random.randint(1, total - 1)
    return [cut, total - cut]


def _place_block(
    canvas: np.ndarray,
    arrays: list[np.ndarray],
    *,
    top: int,
    left: int,
    counts: list[int],
    gap_min: int,
    gap_max: int,
) -> list[tuple[int, int]]:
    positions: list[tuple[int, int]] = []
    idx = 0
    cur_r = top
    for count in counts:
        row_arrays = arrays[idx : idx + count]
        heights = [arr.shape[0] for arr in row_arrays]
        widths = [arr.shape[1] for arr in row_arrays]
        gaps = [random.randint(gap_min, gap_max) for _ in range(max(0, count - 1))]
        total_w = sum(widths) + sum(gaps)
        max_h = max(heights)
        max_start = canvas.shape[1] - total_w - 1
        start_c = random.randint(min(left, max_start), max_start)
        c = start_c
        for arr, gap in itertools.zip_longest(row_arrays, gaps, fillvalue=0):
            if arr is None:
                continue
            h, w = arr.shape
            canvas[cur_r : cur_r + h, c : c + w] = arr
            positions.append((cur_r, c))
            c += w + int(gap)
        idx += count
        cur_r += max_h + random.randint(gap_min, gap_max)
    return positions


def _expected_output(inp: np.ndarray, empty_bbox: tuple[int, int, int, int], fill_template: np.ndarray) -> np.ndarray:
    out = inp.copy()
    r0, c0, r1, c1 = empty_bbox
    patch = out[r0 + 1 : r1, c0 + 1 : c1]
    patch[patch == BG] = fill_template[patch == BG]
    out[r0 + 1 : r1, c0 + 1 : c1] = patch
    return out


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.choice([3, 4])
    kinds = [random.choice([2, 3, 4]) for _ in range(pair_count)]
    inner_size = random.choice([7, 8])

    color_sets = _sample_unique_color_sets(pair_count, kinds)
    empty_index = random.randrange(pair_count)

    frames: list[np.ndarray] = []
    templates: list[np.ndarray] = []
    fill_templates: list[np.ndarray] = []

    for idx, (kind, color_set) in enumerate(zip(kinds, color_sets, strict=False)):
        canonical_lines = _canonical_frame_lines(kind, inner_size)
        canonical_template, ranked_colors = _choose_template_and_colors(kind, inner_size, color_set)

        frame_rot = random.randrange(4)
        template_rot = random.randrange(4)
        rot_lines = np.rot90(canonical_lines, frame_rot)

        frame = _render_frame(rot_lines, None if idx == empty_index else ranked_colors)
        template = np.rot90(canonical_template, template_rot)

        frames.append(frame)
        templates.append(template)
        fill_templates.append(np.rot90(canonical_template, frame_rot))

    frame_rows = 2 if pair_count == 4 and random.random() < 0.7 else random.choice([1, 2])
    template_rows = 2 if pair_count == 4 and random.random() < 0.7 else random.choice([1, 2])
    if pair_count == 3:
        frame_rows = random.choice([1, 2])
        template_rows = random.choice([1, 2])

    frame_counts = _row_counts(pair_count, frame_rows)
    template_counts = _row_counts(pair_count, template_rows)

    frame_h = max(arr.shape[0] for arr in frames)
    frame_w = max(arr.shape[1] for arr in frames)
    template_h = max(arr.shape[0] for arr in templates)
    template_w = max(arr.shape[1] for arr in templates)

    max_frame_row = max(frame_counts)
    max_template_row = max(template_counts)
    width = max_frame_row * frame_w + max(0, max_frame_row - 1) * 3
    width = max(width, max_template_row * template_w + max(0, max_template_row - 1) * 3)
    width += random.randint(6, 10)

    frame_band_h = len(frame_counts) * frame_h + (len(frame_counts) - 1) * 3
    template_band_h = len(template_counts) * template_h + (len(template_counts) - 1) * 3
    height = frame_band_h + template_band_h + random.randint(8, 12)

    grid = np.full((height, width), BG, dtype=int)
    frame_top = random.randint(1, 3)
    frame_positions = _place_block(
        grid,
        frames,
        top=frame_top,
        left=1,
        counts=frame_counts,
        gap_min=2,
        gap_max=3,
    )
    template_top = height - template_band_h - random.randint(1, 3)
    template_positions = _place_block(
        grid,
        templates,
        top=template_top,
        left=max(1, width // 3),
        counts=template_counts,
        gap_min=2,
        gap_max=3,
    )

    empty_pos = frame_positions[empty_index]
    empty_h, empty_w = frames[empty_index].shape
    empty_bbox = (empty_pos[0], empty_pos[1], empty_pos[0] + empty_h - 1, empty_pos[1] + empty_w - 1)
    expected = _expected_output(grid, empty_bbox, fill_templates[empty_index])

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None

    del template_positions
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
