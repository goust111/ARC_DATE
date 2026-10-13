from __future__ import annotations

from collections import Counter

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

TEMPLATE_LIBRARY = [
    np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=bool),
]

GEN_BASE_SIZE = 30
GEN_ATTEMPTS = 400
GEN_GAP = 2
GEN_PALETTE_COUNTS = (3, 4)
GEN_FRAME_COUNTS = (2, 3)


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in DIR4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _touches_border(coords: np.ndarray, h: int, w: int) -> bool:
    return bool(
        np.any(coords[:, 0] == 0)
        or np.any(coords[:, 0] == h - 1)
        or np.any(coords[:, 1] == 0)
        or np.any(coords[:, 1] == w - 1)
    )


def _enclosed_regions_for_color(grid: np.ndarray, wall_color: int, bg: int) -> list[dict]:
    h, w = grid.shape
    regions: list[dict] = []
    for coords in _components(grid != wall_color):
        if _touches_border(coords, h, w):
            continue
        rows = coords[:, 0]
        cols = coords[:, 1]
        bg_count = int(np.count_nonzero(grid[rows, cols] == bg))
        if bg_count == 0:
            continue
        r0, c0, r1, c1 = _bbox(coords)
        local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        local[rows - r0, cols - c0] = True
        regions.append(
            {
                "coords": coords,
                "bbox": (r0, c0, r1, c1),
                "mask": local,
                "bg_count": bg_count,
            }
        )
    return regions


def _find_frame_color_and_regions(grid: np.ndarray, bg: int) -> tuple[int, list[dict]]:
    best_color = None
    best_regions: list[dict] | None = None
    best_score = None
    for color0 in np.unique(grid):
        color = int(color0)
        if color == bg:
            continue
        regions = _enclosed_regions_for_color(grid, color, bg)
        if not regions:
            continue
        score = (
            len(regions),
            sum(region["bg_count"] for region in regions),
            sum(int(len(region["coords"])) for region in regions),
        )
        if best_score is None or score > best_score:
            best_color = color
            best_regions = regions
            best_score = score
    assert best_color is not None and best_regions is not None, "expected enclosed framed regions"
    return int(best_color), best_regions


def _extract_palette_templates(
    grid: np.ndarray,
    bg: int,
    frame_color: int,
    regions: list[dict],
) -> dict[int, np.ndarray]:
    inside_mask = np.zeros(grid.shape, dtype=bool)
    for region in regions:
        coords = region["coords"]
        inside_mask[coords[:, 0], coords[:, 1]] = True

    templates: dict[int, np.ndarray] = {}
    for color0 in np.unique(grid):
        color = int(color0)
        if color in (bg, frame_color):
            continue
        mask = (grid == color) & (~inside_mask)
        if not np.any(mask):
            continue
        comps = _components(mask)
        largest = max(comps, key=len)
        r0, c0, r1, c1 = _bbox(largest)
        crop = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        crop[largest[:, 0] - r0, largest[:, 1] - c0] = True
        templates[color] = crop
    assert templates, "expected palette templates outside the framed regions"
    return templates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each closed region with the palette template selected by the majority marker color and scaled by the marker count.

    Input structure:
    - A dominant background contains several same-color closed frames, each enclosing a rectangular interior region.
    - Inside each framed region there are 1-3 scattered marker pixels, possibly using more than one palette color.
    - Along the bottom or right side, the grid also shows one template component per palette color on the background.

    Rule:
    - Detect the frame color as the one whose pixels enclose the non-border interior regions.
    - For each enclosed region, count all marker pixels to get the scale factor N and find the uniquely most frequent marker color.
    - Extract the template drawn elsewhere in that color, enlarge it by replacing each template pixel with an N x N block, and center the scaled template inside the region.
    - Clear the old markers while keeping the frame outlines and the palette area unchanged.

    Key insight:
    The markers encode two independent controls at once: the winning color selects which external template to copy, and the total marker count selects how much to scale it before centering.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    frame_color, regions = _find_frame_color_and_regions(g, bg)
    templates = _extract_palette_templates(g, bg, frame_color, regions)

    out = g.copy()
    for region in regions:
        r0, c0, r1, c1 = region["bbox"]
        local_mask = region["mask"]
        sub = g[r0 : r1 + 1, c0 : c1 + 1]

        marker_values = [int(v) for v in sub[local_mask] if int(v) != bg]
        assert marker_values, "each region must contain marker pixels"
        counts = Counter(marker_values)
        total_markers = sum(counts.values())
        winners = [color for color, count in counts.items() if count == max(counts.values())]
        assert len(winners) == 1, "majority marker color must be unique"
        marker_color = int(winners[0])
        assert marker_color in templates, "marker color must map to a palette template"

        scaled = np.kron(templates[marker_color], np.ones((total_markers, total_markers), dtype=bool))
        box_h, box_w = local_mask.shape
        sh, sw = scaled.shape
        assert sh <= box_h and sw <= box_w, "scaled template must fit in the region"
        off_r = (box_h - sh) // 2
        off_c = (box_w - sw) // 2

        placement = np.zeros_like(local_mask)
        placement[off_r : off_r + sh, off_c : off_c + sw] = scaled
        assert np.all(~placement | local_mask), "centered template must stay inside the enclosed region"

        out_sub = out[r0 : r1 + 1, c0 : c1 + 1]
        out_sub[local_mask] = bg
        out_sub[placement] = marker_color

    return out


def _paste(canvas: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    canvas[r0 : r0 + h, c0 : c0 + w] = patch


def _reserve(occ: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> None:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(occ.shape[0], r0 + h + pad)
    cc1 = min(occ.shape[1], c0 + w + pad)
    occ[rr0:rr1, cc0:cc1] = True


def _try_place_rect(
    occ: np.ndarray,
    patch_shape: tuple[int, int],
    row_range: tuple[int, int],
    col_range: tuple[int, int],
    rng: np.random.Generator,
    attempts: int = 100,
) -> tuple[int, int] | None:
    h, w = patch_shape
    r_lo, r_hi = row_range
    c_lo, c_hi = col_range
    if r_hi - r_lo < h or c_hi - c_lo < w:
        return None
    for _ in range(attempts):
        r0 = int(rng.integers(r_lo, r_hi - h + 1))
        c0 = int(rng.integers(c_lo, c_hi - w + 1))
        rr0 = max(0, r0 - 1)
        cc0 = max(0, c0 - 1)
        rr1 = min(occ.shape[0], r0 + h + 1)
        cc1 = min(occ.shape[1], c0 + w + 1)
        if not np.any(occ[rr0:rr1, cc0:cc1]):
            return r0, c0
    return None


def _choose_marker_positions(h: int, w: int, count: int, rng: np.random.Generator) -> list[tuple[int, int]] | None:
    cells = [(r, c) for r in range(h) for c in range(w)]
    for _ in range(60):
        rng.shuffle(cells)
        chosen: list[tuple[int, int]] = []
        for r, c in cells:
            if all(abs(r - rr) + abs(c - cc) > 1 for rr, cc in chosen):
                chosen.append((r, c))
                if len(chosen) == count:
                    return chosen
    if len(cells) >= count:
        picks = rng.choice(len(cells), size=count, replace=False)
        return [cells[int(i)] for i in picks]
    return None


def _make_irregular_frame(
    interior_h: int,
    interior_w: int,
    frame_color: int,
    bg: int,
    rng: np.random.Generator,
) -> np.ndarray:
    patch = np.full((interior_h + 4, interior_w + 4), bg, dtype=int)
    r0 = 1
    c0 = 1
    r1 = interior_h + 2
    c1 = interior_w + 2
    patch[r0, c0 : c1 + 1] = frame_color
    patch[r1, c0 : c1 + 1] = frame_color
    patch[r0 : r1 + 1, c0] = frame_color
    patch[r0 : r1 + 1, c1] = frame_color

    side_order = ["top", "bottom", "left", "right"]
    rng.shuffle(side_order)
    for side in side_order[: int(rng.integers(1, 5))]:
        if side in ("top", "bottom"):
            span_max = max(1, min(3, interior_w + 2))
            span = int(rng.integers(1, span_max + 1))
            start = int(rng.integers(c0, c1 - span + 2))
            row = 0 if side == "top" else patch.shape[0] - 1
            patch[row, start : start + span] = frame_color
        else:
            span_max = max(1, min(3, interior_h + 2))
            span = int(rng.integers(1, span_max + 1))
            start = int(rng.integers(r0, r1 - span + 2))
            col = 0 if side == "left" else patch.shape[1] - 1
            patch[start : start + span, col] = frame_color
    return patch


def _make_frame_pair(
    template: np.ndarray,
    palette_colors: list[int],
    frame_color: int,
    bg: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict] | None:
    total_markers = int(rng.integers(1, 4))
    majority_color = int(palette_colors[int(rng.integers(0, len(palette_colors)))])

    if total_markers == 1:
        marker_colors = [majority_color]
    elif total_markers == 2:
        marker_colors = [majority_color, majority_color]
    else:
        if len(palette_colors) > 1 and bool(rng.integers(0, 2)):
            minority_choices = [color for color in palette_colors if color != majority_color]
            minority_color = int(minority_choices[int(rng.integers(0, len(minority_choices)))])
            marker_colors = [majority_color, majority_color, minority_color]
        else:
            marker_colors = [majority_color, majority_color, majority_color]
    rng.shuffle(marker_colors)

    scaled = np.kron(template, np.ones((total_markers, total_markers), dtype=bool))
    pad_r = int(rng.integers(0, 2))
    pad_c = int(rng.integers(0, 2))
    interior_h = int(scaled.shape[0] + 2 * pad_r)
    interior_w = int(scaled.shape[1] + 2 * pad_c)

    markers = _choose_marker_positions(interior_h, interior_w, total_markers, rng)
    if markers is None:
        return None

    inp = _make_irregular_frame(interior_h, interior_w, frame_color, bg, rng)
    out = inp.copy()
    in_r0 = 2
    in_c0 = 2
    for (mr, mc), color in zip(markers, marker_colors):
        inp[in_r0 + mr, in_c0 + mc] = int(color)

    out[in_r0 : in_r0 + interior_h, in_c0 : in_c0 + interior_w] = bg
    out[
        in_r0 + pad_r : in_r0 + pad_r + scaled.shape[0],
        in_c0 + pad_c : in_c0 + pad_c + scaled.shape[1],
    ][scaled] = majority_color

    info = {
        "majority_color": majority_color,
        "total_markers": total_markers,
    }
    return inp, out, info


def _make_palette_patch(
    templates: list[np.ndarray],
    colors: list[int],
    orientation: str,
    bg: int,
) -> np.ndarray:
    gap = GEN_GAP
    if orientation == "bottom":
        height = max(template.shape[0] for template in templates)
        width = sum(template.shape[1] for template in templates) + gap * (len(templates) - 1)
        patch = np.full((height, width), bg, dtype=int)
        c = 0
        for template, color in zip(templates, colors):
            th, tw = template.shape
            r0 = (height - th) // 2
            patch[r0 : r0 + th, c : c + tw][template] = int(color)
            c += tw + gap
        return patch

    width = max(template.shape[1] for template in templates)
    height = sum(template.shape[0] for template in templates) + gap * (len(templates) - 1)
    patch = np.full((height, width), bg, dtype=int)
    r = 0
    for template, color in zip(templates, colors):
        th, tw = template.shape
        c0 = (width - tw) // 2
        patch[r : r + th, c0 : c0 + tw][template] = int(color)
        r += th + gap
    return patch


def _crop_flush(
    inp: np.ndarray,
    out: np.ndarray,
    bg: int,
    orientation: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    coords = np.argwhere(inp != bg)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)

    if orientation == "bottom":
        top = max(0, int(r0) - int(rng.integers(1, 4)))
        left = max(0, int(c0) - int(rng.integers(1, 4)))
        right = min(inp.shape[1], int(c1) + 1 + int(rng.integers(1, 4)))
        return inp[top:, left:right], out[top:, left:right]

    top = max(0, int(r0) - int(rng.integers(1, 4)))
    left = max(0, int(c0) - int(rng.integers(1, 4)))
    bottom = min(inp.shape[0], int(r1) + 1 + int(rng.integers(1, 4)))
    return inp[top:bottom, left:], out[top:bottom, left:]


def _bg_is_unique_mode(grid: np.ndarray, bg: int) -> bool:
    values, counts = np.unique(grid, return_counts=True)
    top = int(counts.max())
    winners = [int(v) for v, c in zip(values, counts) if int(c) == top]
    return len(winners) == 1 and winners[0] == bg


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    bg = int(rng.integers(0, 10))
    remaining = [color for color in range(10) if color != bg]
    rng.shuffle(remaining)
    frame_color = int(remaining[0])
    palette_count = int(rng.integers(GEN_PALETTE_COUNTS[0], GEN_PALETTE_COUNTS[1] + 1))
    palette_colors = [int(color) for color in remaining[1 : 1 + palette_count]]
    frame_count = int(rng.integers(GEN_FRAME_COUNTS[0], GEN_FRAME_COUNTS[1] + 1))

    template_ids = rng.choice(len(TEMPLATE_LIBRARY), size=palette_count, replace=False)
    templates = [TEMPLATE_LIBRARY[int(i)].copy() for i in template_ids]
    order = rng.permutation(palette_count)
    templates = [templates[int(i)] for i in order]
    palette_colors = [palette_colors[int(i)] for i in order]
    template_by_color = {color: template for color, template in zip(palette_colors, templates)}

    frame_inputs: list[np.ndarray] = []
    frame_outputs: list[np.ndarray] = []
    frame_specs: list[dict] = []
    for _ in range(frame_count):
        majority_color = int(palette_colors[int(rng.integers(0, len(palette_colors)))])
        pair = _make_frame_pair(template_by_color[majority_color], palette_colors, frame_color, bg, rng)
        if pair is None:
            return None
        inp_patch, out_patch, info = pair
        frame_inputs.append(inp_patch)
        frame_outputs.append(out_patch)
        frame_specs.append(info)

    if frame_count > 1:
        signatures = {(spec["majority_color"], spec["total_markers"]) for spec in frame_specs}
        if len(signatures) == 1:
            return None

    orientation = "bottom" if bool(rng.integers(0, 2)) else "right"
    palette_patch = _make_palette_patch(templates, palette_colors, orientation, bg)

    inp = np.full((GEN_BASE_SIZE, GEN_BASE_SIZE), bg, dtype=int)
    out = inp.copy()
    occ = np.zeros_like(inp, dtype=bool)

    if orientation == "bottom":
        pr0 = GEN_BASE_SIZE - palette_patch.shape[0]
        pc0 = int(rng.integers(1, GEN_BASE_SIZE - palette_patch.shape[1]))
        _paste(inp, palette_patch, pr0, pc0)
        _paste(out, palette_patch, pr0, pc0)
        _reserve(occ, pr0, pc0, *palette_patch.shape, pad=1)
        row_range = (1, pr0 - 1)
        col_range = (1, GEN_BASE_SIZE - 1)
    else:
        pr0 = int(rng.integers(1, GEN_BASE_SIZE - palette_patch.shape[0]))
        pc0 = GEN_BASE_SIZE - palette_patch.shape[1]
        _paste(inp, palette_patch, pr0, pc0)
        _paste(out, palette_patch, pr0, pc0)
        _reserve(occ, pr0, pc0, *palette_patch.shape, pad=1)
        row_range = (1, GEN_BASE_SIZE - 1)
        col_range = (1, pc0 - 1)

    place_order = list(range(frame_count))
    place_order.sort(key=lambda i: frame_inputs[i].shape[0] * frame_inputs[i].shape[1], reverse=True)
    placements: dict[int, tuple[int, int]] = {}
    for idx in place_order:
        patch = frame_inputs[idx]
        loc = _try_place_rect(occ, patch.shape, row_range, col_range, rng, attempts=140)
        if loc is None:
            return None
        r0, c0 = loc
        placements[idx] = (r0, c0)
        _reserve(occ, r0, c0, *patch.shape, pad=1)

    for idx, (r0, c0) in placements.items():
        _paste(inp, frame_inputs[idx], r0, c0)
        _paste(out, frame_outputs[idx], r0, c0)

    inp, out = _crop_flush(inp, out, bg, orientation, rng)
    if max(inp.shape) > 30 or max(out.shape) > 30:
        return None
    if not _bg_is_unique_mode(inp, bg):
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC input for this task")
