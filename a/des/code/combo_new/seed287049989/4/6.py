from __future__ import annotations

import numpy as np


BG = 0
FRAME_COLOR = 3
SEED_COLOR = 2
STAMP_COLORS = (1, 4, 5, 6, 7, 8, 9)

TEMPLATE_COUNT_RANGE = (3, 6)
TEMPLATE_H_RANGE = (3, 7)
TEMPLATE_W_RANGE = (3, 6)
FRAME_INNER_H_RANGE = (4, 7)
FRAME_INNER_W_RANGE = (4, 8)
REGION_COUNT_RANGE = (2, 5)
GEN_ATTEMPTS = 400

TEMPLATE_GAP = 1
FRAME_GAP_X = 2
FRAME_GAP_Y = 2
TOP_MARGIN = 1
BOTTOM_MARGIN = 1
SIDE_MARGIN = 1
TEMPLATE_TO_FRAME_GAP = 2


def _boundary_bg(grid: np.ndarray) -> int:
    boundary = np.concatenate((grid[0, :], grid[-1, :], grid[:, 0], grid[:, -1]))
    colors, counts = np.unique(boundary, return_counts=True)
    return int(colors[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _rect_frame_bbox(coords: np.ndarray) -> tuple[int, int, int, int] | None:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None
    local = np.zeros((h, w), dtype=bool)
    shifted = coords - np.array([r0, c0], dtype=int)
    local[shifted[:, 0], shifted[:, 1]] = True
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    if np.array_equal(local, border):
        return (r0, c0, r1, c1)
    return None


def _find_frames(grid: np.ndarray, bg: int) -> tuple[int, list[tuple[int, int, int, int]]]:
    by_color: dict[int, list[tuple[int, int, int, int]]] = {}
    for color0 in np.unique(grid):
        color = int(color0)
        if color == bg:
            continue
        frames = []
        for comp in _components(grid == color):
            bbox = _rect_frame_bbox(comp)
            if bbox is not None:
                frames.append(bbox)
        if frames:
            by_color[color] = sorted(frames)

    assert by_color, "expected at least one rectangular frame color"
    max_count = max(len(frames) for frames in by_color.values())
    frame_colors = [color for color, frames in by_color.items() if len(frames) == max_count]
    assert len(frame_colors) == 1, "rectangular frames must identify a unique color"
    frame_color = int(frame_colors[0])
    return frame_color, by_color[frame_color]


def _extract_templates(grid: np.ndarray, bg: int, frame_top: int) -> list[np.ndarray]:
    top = grid[:frame_top, :]
    active_cols = np.flatnonzero(np.any(top != bg, axis=0))
    assert active_cols.size > 0, "expected template stamps above the frames"

    intervals: list[tuple[int, int]] = []
    start = end = int(active_cols[0])
    for col0 in active_cols[1:]:
        col = int(col0)
        if col == end + 1:
            end = col
        else:
            intervals.append((start, end))
            start = end = col
    intervals.append((start, end))

    templates: list[np.ndarray] = []
    for c0, c1 in intervals:
        window = top[:, c0 : c1 + 1]
        active_rows = np.flatnonzero(np.any(window != bg, axis=1))
        assert active_rows.size > 0
        r0 = int(active_rows[0])
        r1 = int(active_rows[-1])
        template = top[r0 : r1 + 1, c0 : c1 + 1].copy()
        assert np.any(template != bg)
        templates.append(template)

    return templates


def _seed_color(grid: np.ndarray, frames: list[tuple[int, int, int, int]], bg: int, frame_color: int) -> int:
    colors: set[int] = set()
    for r0, c0, r1, c1 in frames:
        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        colors.update(map(int, np.unique(interior)))
    colors -= {bg, frame_color}
    assert len(colors) == 1, "frame interiors must contain one seed color"
    return int(next(iter(colors)))


def _seed_anchor(interior: np.ndarray, seed_color: int) -> tuple[int, int, int]:
    seeds = np.argwhere(interior == seed_color)
    assert len(seeds) > 0, "each region must contain at least one seed pixel"
    r0, c0 = seeds.min(axis=0)
    r1, c1 = seeds.max(axis=0)
    anchor_r = int((int(r0) + int(r1)) // 2)
    anchor_c = int((int(c0) + int(c1)) // 2)
    return anchor_r, anchor_c, int(len(seeds))


def _stamp_into_region(
    out: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    template: np.ndarray,
    seed_color: int,
    bg: int,
) -> None:
    r0, c0, r1, c1 = frame_bbox
    interior = out[r0 + 1 : r1, c0 + 1 : c1]
    anchor_r, anchor_c, _count = _seed_anchor(interior, seed_color)
    interior[interior == seed_color] = bg

    center_r = template.shape[0] // 2
    center_c = template.shape[1] // 2
    for tr, tc in np.argwhere(template != bg):
        rr = anchor_r + int(tr) - center_r
        cc = anchor_c + int(tc) - center_c
        if 0 <= rr < interior.shape[0] and 0 <= cc < interior.shape[1]:
            interior[rr, cc] = int(template[int(tr), int(tc)])


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _boundary_bg(g)
    frame_color, frames = _find_frames(g, bg)
    seed_color = _seed_color(g, frames, bg, frame_color)

    top_limit = min(r0 for r0, _c0, _r1, _c1 in frames)
    templates = _extract_templates(g, bg, top_limit)

    out = g.copy()
    for frame_bbox in frames:
        r0, c0, r1, c1 = frame_bbox
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        anchor_r, anchor_c, count = _seed_anchor(interior, seed_color)
        _ = (anchor_r, anchor_c)
        assert 1 <= count <= len(templates), "seed count must select an existing stamp"
        _stamp_into_region(out, frame_bbox, templates[count - 1], seed_color, bg)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Select the Nth upper stamp by seed count and print it inside each framed region at the seed cluster center.

    Input structure:
    - The background is black.
    - Several green one-cell-thick rectangular borders enclose empty interiors that contain scattered red seed pixels.
    - Above the framed regions, small colored stamp templates are arranged left to right with black gaps between them.

    Rule:
    - For each closed rectangular region, count the red seed pixels inside it and choose the stamp with the same left-to-right index.
    - Compute the seed anchor from the midpoint of the seeds' row span and column span, using floor division in both directions.
    - Erase the red seeds in that region, align the chosen stamp's bbox center to the anchor, and paint only the stamp's non-black pixels inside the region; clip any part that would cross the green border.

    Key insight:
    - The seed pixels encode two independent pieces of information at once: how many of them there are picks the stamp, and how far they spread sets the stamp's placement center.
    """
    return _solve(grid)


def _random_template(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    palette = list(map(int, rng.choice(STAMP_COLORS, size=int(rng.integers(1, 4)), replace=False)))
    for _ in range(120):
        patch = np.zeros((h, w), dtype=int)
        n_parts = int(rng.integers(3, 7))
        for _ in range(n_parts):
            color = palette[int(rng.integers(0, len(palette)))]
            kind = int(rng.integers(0, 3))
            if kind == 0:
                r0 = int(rng.integers(0, h))
                r1 = int(rng.integers(r0, h))
                c0 = int(rng.integers(0, w))
                c1 = int(rng.integers(c0, w))
                patch[r0 : r1 + 1, c0 : c1 + 1] = color
            elif kind == 1:
                row = int(rng.integers(0, h))
                c0 = int(rng.integers(0, w))
                c1 = int(rng.integers(c0, w))
                patch[row, c0 : c1 + 1] = color
            else:
                col = int(rng.integers(0, w))
                r0 = int(rng.integers(0, h))
                r1 = int(rng.integers(r0, h))
                patch[r0 : r1 + 1, col] = color

        for row in np.flatnonzero(~np.any(patch != BG, axis=1)):
            patch[int(row), int(rng.integers(0, w))] = palette[int(rng.integers(0, len(palette)))]
        for col in np.flatnonzero(~np.any(patch != BG, axis=0)):
            patch[int(rng.integers(0, h)), int(col)] = palette[int(rng.integers(0, len(palette)))]

        density = float(np.mean(patch != BG))
        if 0.30 <= density <= 0.85:
            return patch
    raise RuntimeError("failed to generate template")


def _template_specs(rng: np.random.Generator) -> list[np.ndarray]:
    for _ in range(120):
        count = int(rng.integers(*TEMPLATE_COUNT_RANGE))
        dims = [
            (
                int(rng.integers(*TEMPLATE_H_RANGE)),
                int(rng.integers(*TEMPLATE_W_RANGE)),
            )
            for _ in range(count)
        ]
        total_w = sum(w for _h, w in dims) + TEMPLATE_GAP * (count - 1)
        if total_w > 28:
            continue

        templates: list[np.ndarray] = []
        seen: set[bytes] = set()
        ok = True
        for h, w in dims:
            template = _random_template(rng, h, w)
            key = template.tobytes() + bytes(template.shape)
            if key in seen:
                ok = False
                break
            seen.add(key)
            templates.append(template)
        if ok:
            return templates
    raise RuntimeError("failed to generate templates")


def _seed_box_options(center: int, limit: int) -> list[tuple[int, int]]:
    options: list[tuple[int, int]] = []
    for up in range(center + 1):
        for delta in (0, 1):
            down = up + delta
            if center + down < limit:
                options.append((center - up, center + down))
    return options


def _seed_positions(
    rng: np.random.Generator,
    inner_h: int,
    inner_w: int,
    count: int,
    anchor: tuple[int, int],
) -> np.ndarray | None:
    ar, ac = anchor
    if count == 1:
        return np.array([[ar, ac]], dtype=int)

    row_options = _seed_box_options(ar, inner_h)
    col_options = _seed_box_options(ac, inner_w)
    for ri in rng.permutation(len(row_options)):
        r0, r1 = row_options[int(ri)]
        for ci in rng.permutation(len(col_options)):
            c0, c1 = col_options[int(ci)]
            cells = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
            if len(cells) < count:
                continue
            chosen = [(r0, c0)]
            if (r1, c1) not in chosen:
                chosen.append((r1, c1))
            if len(chosen) > count:
                continue
            extras = [cell for cell in cells if cell not in chosen]
            if len(chosen) < count:
                pick = rng.choice(len(extras), size=count - len(chosen), replace=False)
                chosen.extend(extras[int(i)] for i in np.atleast_1d(pick))
            return np.array(chosen, dtype=int)
    return None


def _safe_anchor_box(template_shape: tuple[int, int], inner_h: int, inner_w: int) -> tuple[int, int, int, int]:
    th, tw = template_shape
    up = th // 2
    down = th - 1 - up
    left = tw // 2
    right = tw - 1 - left
    return up, inner_h - 1 - down, left, inner_w - 1 - right


def _is_clipped(anchor: tuple[int, int], template_shape: tuple[int, int], inner_h: int, inner_w: int) -> bool:
    ar, ac = anchor
    r0, r1, c0, c1 = _safe_anchor_box(template_shape, inner_h, inner_w)
    return not (r0 <= ar <= r1 and c0 <= ac <= c1)


def _region_spec(
    rng: np.random.Generator,
    template: np.ndarray,
    count: int,
    want_clipped: bool,
) -> dict[str, object]:
    th, tw = map(int, template.shape)
    for _ in range(160):
        inner_h = int(rng.integers(*FRAME_INNER_H_RANGE))
        inner_w = int(rng.integers(*FRAME_INNER_W_RANGE))
        if inner_h * inner_w < count:
            continue

        sr0, sr1, sc0, sc1 = _safe_anchor_box((th, tw), inner_h, inner_w)
        has_safe = sr0 <= sr1 and sc0 <= sc1
        if not want_clipped and not has_safe:
            continue

        anchors: list[tuple[int, int]] = []
        for r in range(inner_h):
            for c in range(inner_w):
                clipped = _is_clipped((r, c), (th, tw), inner_h, inner_w)
                if clipped == want_clipped:
                    anchors.append((r, c))
        if not anchors:
            continue

        anchor = anchors[int(rng.integers(0, len(anchors)))]
        seeds = _seed_positions(rng, inner_h, inner_w, count, anchor)
        if seeds is None:
            continue
        assert _is_clipped(anchor, (th, tw), inner_h, inner_w) == want_clipped
        return {
            "count": int(count),
            "template": template,
            "inner_h": int(inner_h),
            "inner_w": int(inner_w),
            "frame_h": int(inner_h + 2),
            "frame_w": int(inner_w + 2),
            "seeds": seeds,
        }
    raise RuntimeError("failed to generate region")


def _group_regions(
    rng: np.random.Generator,
    regions: list[dict[str, object]],
) -> list[list[dict[str, object]]]:
    ordered = [regions[int(i)] for i in rng.permutation(len(regions))]
    n = len(ordered)
    options: list[list[list[dict[str, object]]]] = []
    if n == 2:
        options = [[ordered], [ordered[:1], ordered[1:]]]
    elif n == 3:
        options = [[ordered], [ordered[:2], ordered[2:]], [ordered[:1], ordered[1:]]]
    else:
        options = [[ordered[:2], ordered[2:]]]

    fit_options = []
    for rows in options:
        row_widths = [
            sum(int(region["frame_w"]) for region in row) + FRAME_GAP_X * (len(row) - 1)
            for row in rows
        ]
        req_w = SIDE_MARGIN * 2 + max(row_widths)
        row_heights = [max(int(region["frame_h"]) for region in row) for row in rows]
        req_h = (
            TOP_MARGIN
            + max(int(template.shape[0]) for template in _template_specs_for_rows(regions))
            + TEMPLATE_TO_FRAME_GAP
            + sum(row_heights)
            + FRAME_GAP_Y * (len(rows) - 1)
            + BOTTOM_MARGIN
        )
        if req_w <= 30 and req_h <= 30:
            fit_options.append(rows)

    assert fit_options, "expected at least one valid region grouping"
    return fit_options[int(rng.integers(0, len(fit_options)))]


def _template_specs_for_rows(regions: list[dict[str, object]]) -> list[np.ndarray]:
    return [np.asarray(region["template"], dtype=int) for region in regions]


def _draw_frame(grid: np.ndarray, top: int, left: int, inner_h: int, inner_w: int) -> tuple[int, int, int, int]:
    bottom = top + inner_h + 1
    right = left + inner_w + 1
    grid[top, left : right + 1] = FRAME_COLOR
    grid[bottom, left : right + 1] = FRAME_COLOR
    grid[top : bottom + 1, left] = FRAME_COLOR
    grid[top : bottom + 1, right] = FRAME_COLOR
    return top, left, bottom, right


def _expected_output(
    grid: np.ndarray,
    region_boxes: list[tuple[int, int, int, int]],
    regions: list[dict[str, object]],
) -> np.ndarray:
    out = grid.copy()
    for bbox, region in zip(region_boxes, regions):
        interior = out[bbox[0] + 1 : bbox[2], bbox[1] + 1 : bbox[3]]
        interior[interior == SEED_COLOR] = BG
        template = np.asarray(region["template"], dtype=int)
        center_r = template.shape[0] // 2
        center_c = template.shape[1] // 2
        seeds = np.asarray(region["seeds"], dtype=int)
        sr0, sc0 = seeds.min(axis=0)
        sr1, sc1 = seeds.max(axis=0)
        anchor = (int((int(sr0) + int(sr1)) // 2), int((int(sc0) + int(sc1)) // 2))
        for tr, tc in np.argwhere(template != BG):
            rr = anchor[0] + int(tr) - center_r
            cc = anchor[1] + int(tc) - center_c
            if 0 <= rr < interior.shape[0] and 0 <= cc < interior.shape[1]:
                interior[rr, cc] = int(template[int(tr), int(tc)])
    return out


def _build_example(rng: np.random.Generator) -> np.ndarray | None:
    templates = _template_specs(rng)
    template_count = len(templates)

    counts = list(range(1, min(template_count, int(rng.integers(*REGION_COUNT_RANGE))) + 1))
    region_count = max(2, len(counts))
    while len(counts) < region_count:
        counts.append(int(rng.integers(1, template_count + 1)))
    counts = counts[:region_count]
    rng.shuffle(counts)

    clip_targets = [False, True] + [bool(rng.integers(0, 2)) for _ in range(max(0, region_count - 2))]
    regions = [
        _region_spec(rng, templates[count - 1], count, clip_targets[i])
        for i, count in enumerate(counts)
    ]
    rows = _group_regions(rng, regions)

    template_total_w = sum(template.shape[1] for template in templates) + TEMPLATE_GAP * (len(templates) - 1)
    top_h = max(template.shape[0] for template in templates)
    row_widths = [
        sum(int(region["frame_w"]) for region in row) + FRAME_GAP_X * (len(row) - 1)
        for row in rows
    ]
    row_heights = [max(int(region["frame_h"]) for region in row) for row in rows]

    req_w = SIDE_MARGIN * 2 + max(template_total_w, max(row_widths))
    req_h = (
        TOP_MARGIN
        + top_h
        + TEMPLATE_TO_FRAME_GAP
        + sum(row_heights)
        + FRAME_GAP_Y * (len(rows) - 1)
        + BOTTOM_MARGIN
    )
    if req_w > 30 or req_h > 30:
        return None

    grid_h = int(rng.integers(req_h, 31))
    grid_w = int(rng.integers(req_w, 31))
    grid = np.full((grid_h, grid_w), BG, dtype=int)

    template_left = int(rng.integers(SIDE_MARGIN, grid_w - SIDE_MARGIN - template_total_w + 1))
    current_c = template_left
    for template in templates:
        top = TOP_MARGIN
        h, w = template.shape
        grid[top : top + h, current_c : current_c + w] = template
        current_c += w + TEMPLATE_GAP

    first_frame_top = TOP_MARGIN + top_h + TEMPLATE_TO_FRAME_GAP
    extra_vertical = grid_h - req_h
    if extra_vertical:
        first_frame_top += int(rng.integers(0, extra_vertical + 1))

    region_boxes: list[tuple[int, int, int, int]] = []
    placed_regions: list[dict[str, object]] = []
    row_top = first_frame_top
    for row, row_height, row_width in zip(rows, row_heights, row_widths):
        row_left = int(rng.integers(SIDE_MARGIN, grid_w - SIDE_MARGIN - row_width + 1))
        col = row_left
        for region in row:
            frame_h = int(region["frame_h"])
            frame_w = int(region["frame_w"])
            inner_h = int(region["inner_h"])
            inner_w = int(region["inner_w"])
            box = _draw_frame(grid, row_top, col, inner_h, inner_w)
            seeds = np.asarray(region["seeds"], dtype=int)
            for sr, sc in seeds:
                grid[box[0] + 1 + int(sr), box[1] + 1 + int(sc)] = SEED_COLOR
            region_boxes.append(box)
            placed_regions.append(region)
            col += frame_w + FRAME_GAP_X
        row_top += row_height + FRAME_GAP_Y

    expected = _expected_output(grid, region_boxes, placed_regions)
    if np.array_equal(_solve(grid), expected):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        rng = np.random.default_rng()
        grid = _build_example(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
