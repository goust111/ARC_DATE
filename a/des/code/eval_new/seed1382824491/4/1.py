from __future__ import annotations

import random

import numpy as np


_DIR_TO_ROT = {"up": 0, "right": 3, "down": 2, "left": 1}
_DIRECTIONS = ("up", "right", "down", "left")

# Default-up legend templates. Each one is connected, asymmetric, and spans
# the full 3x3 tile so the cropped patch preserves the intended rotation cue.
_PATTERN_LIBRARY = (
    np.array([[0, 1, 0], [1, 1, 1], [1, 1, 1]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1], [1, 0, 1]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1], [1, 1, 0]], dtype=bool),
)


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return out


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = map(int, grid.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))

            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(arr.shape[0]),
                }
            )
    return comps


def _frame_info(grid: np.ndarray, comp: dict) -> dict | None:
    color = int(comp["color"])
    r0, c0, r1, c1 = map(int, comp["bbox"])
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None

    sub = grid[r0 : r1 + 1, c0 : c1 + 1] == color
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    if int(sub.sum()) != int(border.sum()) - 1:
        return None
    if bool(np.any(sub & ~border)):
        return None

    missing = np.argwhere(border & ~sub)
    if int(missing.shape[0]) != 1:
        return None
    mr, mc = map(int, missing[0])
    if (mr in (0, h - 1)) and (mc in (0, w - 1)):
        return None

    if mr == 0:
        direction = "up"
    elif mr == h - 1:
        direction = "down"
    elif mc == 0:
        direction = "left"
    else:
        direction = "right"

    return {
        "color": color,
        "bbox": (r0, c0, r1, c1),
        "direction": direction,
        "area": int(comp["area"]),
    }


def _legend_patterns_from_band(band: np.ndarray, bg: int) -> list[np.ndarray]:
    occupied = np.any(band != bg, axis=0)
    runs: list[tuple[int, int]] = []
    c = 0
    while c < occupied.size:
        if not occupied[c]:
            c += 1
            continue
        start = c
        while c + 1 < occupied.size and occupied[c + 1]:
            c += 1
        runs.append((start, c))
        c += 1

    return [band[:, c0 : c1 + 1].copy() for c0, c1 in runs if np.any(band[:, c0 : c1 + 1] != bg)]


def _extract_legend_patterns(grid: np.ndarray, bg: int) -> list[np.ndarray]:
    h, w = map(int, grid.shape)
    occupied_rows = [r for r in range(h) if np.any(grid[r, :] != bg)]
    assert occupied_rows, "grid must contain non-background pixels"
    last_non_bg = int(max(occupied_rows))

    for split in range(last_non_bg - 1, -1, -1):
        if not bool(np.all(grid[split, :] == bg)):
            continue
        band = grid[split + 1 : last_non_bg + 1, :]
        if band.size == 0 or not np.any(band != bg):
            continue
        patterns = _legend_patterns_from_band(band, bg)
        if patterns:
            return patterns

    # Fallback for more tightly packed variants: use bottom-most non-frame pieces.
    comps = _find_components(grid, bg)
    frame_bboxes = {info["bbox"] for info in (_frame_info(grid, comp) for comp in comps) if info is not None}
    non_frames = [
        comp for comp in comps if tuple(map(int, comp["bbox"])) not in frame_bboxes
    ]
    non_frames.sort(key=lambda comp: (int(comp["bbox"][1]), int(comp["bbox"][0])))
    patterns = []
    for comp in non_frames:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        patterns.append(grid[r0 : r1 + 1, c0 : c1 + 1].copy())
    return patterns


def _contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    or0, oc0, or1, oc1 = map(int, outer)
    ir0, ic0, ir1, ic1 = map(int, inner)
    return or0 + 2 <= ir0 and oc0 + 2 <= ic0 and or1 - 2 >= ir1 and oc1 - 2 >= ic1


def _tile_patch(patch: np.ndarray, h: int, w: int) -> np.ndarray:
    ph, pw = map(int, patch.shape)
    reps_r = (int(h) + ph - 1) // ph
    reps_c = (int(w) + pw - 1) // pw
    return np.tile(patch, (reps_r, reps_c))[:h, :w]


def _fill_innermost_region(
    out: np.ndarray,
    bbox: tuple[int, int, int, int],
    direction: str,
    pattern: np.ndarray,
    bg: int,
) -> None:
    r0, c0, r1, c1 = map(int, bbox)
    interior = out[r0 + 1 : r1, c0 + 1 : c1]
    rotated = np.rot90(pattern, _DIR_TO_ROT[direction])
    tiled = _tile_patch(rotated, int(interior.shape[0]), int(interior.shape[1]))
    interior[interior == bg] = tiled[interior == bg]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each nested-frame center with the legend pattern selected by depth and rotated toward the innermost gap.

    Input structure:
    - A dominant background color covers the grid.
    - Several monochrome rectangular frame components appear on the canvas; each frame is one pixel thick and misses exactly one non-corner border pixel.
    - Some frames are nested, forming groups with depth 1, 2, 3, ...; a separate legend band at the bottom contains one square pattern per possible depth, ordered left to right.
    - Each legend pattern is drawn with foreground color(s) on the same background and is defined in its default "opening upward" orientation.

    Rule:
    - Identify every frame and group nested frames by rectangle containment.
    - For each group, take its innermost frame and count how many frames lie in that nesting chain; use the depth-K legend pattern as the fill template.
    - Detect the side of the innermost frame's missing border pixel and rotate the chosen pattern so its default upward opening points toward that side.
    - Tile the rotated pattern from the inner top-left corner across the background cells inside the innermost frame, clipping at the interior boundary; leave frames and the legend unchanged.

    Key insight:
    The fill choice depends on two separate latent attributes: nesting depth chooses which legend tile to use, and the innermost frame's gap direction chooses how that tile must be rotated before tiling.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _most_common_color(g)
    comps = _find_components(g, bg)
    frames = [info for info in (_frame_info(g, comp) for comp in comps) if info is not None]
    assert frames, "at least one frame is required"

    patterns = _extract_legend_patterns(g, bg)
    assert patterns, "legend patterns must be present"

    n = len(frames)
    parents: list[int | None] = [None] * n
    child_count = [0] * n
    for i, frame in enumerate(frames):
        bbox = frame["bbox"]
        candidates = [
            (j, other)
            for j, other in enumerate(frames)
            if i != j and _contains(other["bbox"], bbox)
        ]
        if not candidates:
            continue
        parent_idx = min(
            candidates,
            key=lambda item: (
                (item[1]["bbox"][2] - item[1]["bbox"][0] + 1)
                * (item[1]["bbox"][3] - item[1]["bbox"][1] + 1)
            ),
        )[0]
        parents[i] = int(parent_idx)
        child_count[int(parent_idx)] += 1

    out = g.copy()
    for i, frame in enumerate(frames):
        if child_count[i] != 0:
            continue
        depth = 1
        parent = parents[i]
        while parent is not None:
            depth += 1
            parent = parents[parent]
        assert depth <= len(patterns), "legend must provide a pattern for every nesting depth"
        _fill_innermost_region(out, frame["bbox"], frame["direction"], patterns[depth - 1], bg)

    return out


def _pattern_patch(mask: np.ndarray, color: int, bg: int) -> np.ndarray:
    patch = np.full(mask.shape, bg, dtype=int)
    patch[mask] = int(color)
    return patch


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int, direction: str, bg: int) -> None:
    r0, c0, r1, c1 = map(int, bbox)
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color

    if direction == "up":
        gc = random.randint(c0 + 1, c1 - 1)
        grid[r0, gc] = bg
    elif direction == "down":
        gc = random.randint(c0 + 1, c1 - 1)
        grid[r1, gc] = bg
    elif direction == "left":
        gr = random.randint(r0 + 1, r1 - 1)
        grid[gr, c0] = bg
    else:
        gr = random.randint(r0 + 1, r1 - 1)
        grid[gr, c1] = bg


def _rect_overlaps(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = map(int, a)
    br0, bc0, br1, bc1 = map(int, b)
    return not (
        ar1 + gap < br0
        or br1 + gap < ar0
        or ac1 + gap < bc0
        or bc1 + gap < ac0
    )


def _place_groups(
    scene_h: int, scene_w: int, group_dims: list[tuple[int, int]]
) -> list[tuple[int, int, int, int]] | None:
    order = sorted(range(len(group_dims)), key=lambda i: group_dims[i][0] * group_dims[i][1], reverse=True)
    placed: list[tuple[int, int, int, int] | None] = [None] * len(group_dims)
    occupied: list[tuple[int, int, int, int]] = []

    for idx in order:
        gh, gw = map(int, group_dims[idx])
        if gh + 2 > scene_h or gw + 2 > scene_w:
            return None
        found = None
        for _ in range(300):
            r0 = random.randint(1, scene_h - gh - 1)
            c0 = random.randint(1, scene_w - gw - 1)
            rect = (r0, c0, r0 + gh - 1, c0 + gw - 1)
            if any(_rect_overlaps(rect, other, gap=1) for other in occupied):
                continue
            found = rect
            occupied.append(rect)
            placed[idx] = rect
            break
        if found is None:
            return None

    return [rect for rect in placed if rect is not None]


def _expected_from_spec(
    inp: np.ndarray,
    bg: int,
    patterns: list[np.ndarray],
    groups: list[dict],
) -> np.ndarray:
    out = inp.copy()
    for group in groups:
        depth = int(group["depth"])
        inner = group["layers"][-1]
        _fill_innermost_region(
            out,
            inner["bbox"],
            inner["direction"],
            patterns[depth - 1],
            bg,
        )
    return out


def _attempt_generate() -> np.ndarray | None:
    bg = 0
    depth_count = random.choice([2, 3])
    depths = list(range(1, depth_count + 1))
    random.shuffle(depths)

    tile_masks = random.sample(list(_PATTERN_LIBRARY), depth_count)
    pattern_count = depth_count
    frame_color_count = sum(depths)
    colors = random.sample(list(range(1, 10)), pattern_count + frame_color_count)
    pattern_colors = colors[:pattern_count]
    frame_colors = colors[pattern_count:]

    patterns = [
        _pattern_patch(mask, color, bg) for mask, color in zip(tile_masks, pattern_colors, strict=False)
    ]

    groups: list[dict] = []
    dims: list[tuple[int, int]] = []
    color_at = 0
    for depth in depths:
        max_inner = {1: 7, 2: 6, 3: 5}[depth]
        inner_h = random.randint(3, max_inner)
        inner_w = random.randint(3, max_inner)
        outer_h = inner_h + 2 + 4 * (depth - 1)
        outer_w = inner_w + 2 + 4 * (depth - 1)
        dims.append((outer_h, outer_w))

        layers = []
        for level in range(depth):
            layers.append(
                {
                    "color": int(frame_colors[color_at + level]),
                    "direction": random.choice(_DIRECTIONS),
                }
            )
        color_at += depth
        groups.append(
            {
                "depth": int(depth),
                "outer_h": int(outer_h),
                "outer_w": int(outer_w),
                "inner_h": int(inner_h),
                "inner_w": int(inner_w),
                "layers": layers,
            }
        )

    legend_h = int(patterns[0].shape[0])
    legend_total_w = pattern_count * legend_h + (pattern_count - 1)
    min_scene_h = max(max(h for h, _ in dims) + 2, 12)
    min_scene_w = max(max(w for _, w in dims) + 2, 14)
    total_area = sum((h + 2) * (w + 2) for h, w in dims)
    area_side = int(np.ceil(np.sqrt(total_area * 1.5)))
    min_scene_h = max(min_scene_h, min(26, area_side))
    min_scene_w = max(min_scene_w, min(30, area_side))
    min_scene_w = max(min_scene_w, legend_total_w + 2)
    if min_scene_h > 26 or min_scene_w > 30:
        return None

    scene_h = random.randint(min_scene_h, 26)
    scene_w = random.randint(min_scene_w, 30)
    placements = _place_groups(scene_h, scene_w, dims)
    if placements is None:
        return None

    grid = np.full((scene_h + 1 + legend_h, scene_w), bg, dtype=int)
    for group, rect in zip(groups, placements, strict=False):
        or0, oc0, or1, oc1 = map(int, rect)
        layer_specs = []
        for level, layer in enumerate(group["layers"]):
            bbox = (or0 + 2 * level, oc0 + 2 * level, or1 - 2 * level, oc1 - 2 * level)
            layer_specs.append(
                {
                    "bbox": bbox,
                    "direction": str(layer["direction"]),
                    "color": int(layer["color"]),
                }
            )
            _draw_frame(grid, bbox, int(layer["color"]), str(layer["direction"]), bg)
        group["layers"] = layer_specs

    legend_row = scene_h + 1
    start_c = random.randint(1, scene_w - legend_total_w - 1)
    c = start_c
    for patch in patterns:
        h, w = map(int, patch.shape)
        grid[legend_row : legend_row + h, c : c + w] = patch
        c += w + 1

    expected = _expected_from_spec(grid, bg, patterns, groups)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-frame input")
