from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
SEPARATOR = 5
FRAME = 9
TEMPLATE_COLORS = (1, 2, 3, 4, 6, 7, 8)
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q: deque[tuple[int, int]] = deque([(r, c)])
            cells: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 0) -> bool:
    return not (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _find_separator(g: np.ndarray) -> tuple[str, int]:
    row_hits = [r for r in range(g.shape[0]) if np.all(g[r, :] == SEPARATOR)]
    col_hits = [c for c in range(g.shape[1]) if np.all(g[:, c] == SEPARATOR)]
    assert len(row_hits) + len(col_hits) == 1, "expected one full grey separator"
    if row_hits:
        return "horizontal", int(row_hits[0])
    return "vertical", int(col_hits[0])


def _split_regions(g: np.ndarray, orient: str, axis: int):
    if orient == "horizontal":
        return ((g[:axis, :], (0, 0)), (g[axis + 1 :, :], (axis + 1, 0)))
    return ((g[:, :axis], (0, 0)), (g[:, axis + 1 :], (0, axis + 1)))


def _frame_bboxes(region: np.ndarray) -> list[tuple[int, int, int, int]]:
    infos: list[tuple[int, int, int, int]] = []
    for comp in _components(region == FRAME):
        r0, c0, r1, c1 = _bbox(comp)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 3 or w < 3:
            continue
        sub = region[r0 : r1 + 1, c0 : c1 + 1] == FRAME
        border = np.zeros((h, w), dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        if np.array_equal(sub, border):
            infos.append((r0, c0, r1, c1))
    return infos


def _extract_templates(region: np.ndarray) -> dict[int, np.ndarray]:
    templates: dict[int, np.ndarray] = {}
    for color0 in np.unique(region):
        color = int(color0)
        if color == BG:
            continue
        mask = region == color
        comps = _components(mask)
        assert len(comps) == 1, "each template color must form one connected object"
        r0, c0, r1, c1 = _bbox(comps[0])
        templates[color] = mask[r0 : r1 + 1, c0 : c1 + 1].copy()
    assert templates, "expected legend templates"
    return templates


def _stamp(
    out: np.ndarray,
    color: int,
    mask: np.ndarray,
    seed_r: int,
    seed_c: int,
    interior_bbox: tuple[int, int, int, int],
) -> None:
    center_r = mask.shape[0] // 2
    center_c = mask.shape[1] // 2
    ir0, ic0, ir1, ic1 = interior_bbox
    for mr, mc in np.argwhere(mask):
        tr = seed_r + int(mr) - center_r
        tc = seed_c + int(mc) - center_c
        if ir0 <= tr <= ir1 and ic0 <= tc <= ic1:
            out[tr, tc] = color


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    orient, axis = _find_separator(g)
    (region_a, offset_a), (region_b, offset_b) = _split_regions(g, orient, axis)

    frames_a = _frame_bboxes(region_a)
    frames_b = _frame_bboxes(region_b)
    assert bool(frames_a) ^ bool(frames_b), "exactly one side must contain the maroon frames"

    if frames_a:
        canvas, canvas_offset, frames = region_a, offset_a, frames_a
        legend = region_b
    else:
        canvas, canvas_offset, frames = region_b, offset_b, frames_b
        legend = region_a

    templates = _extract_templates(legend)
    out = g.copy()

    for fr0, fc0, fr1, fc1 in frames:
        ir0, ic0, ir1, ic1 = fr0 + 1, fc0 + 1, fr1 - 1, fc1 - 1
        assert ir0 <= ir1 and ic0 <= ic1, "frame must have a non-empty interior"
        interior = canvas[ir0 : ir1 + 1, ic0 : ic1 + 1]
        seeds = np.argwhere((interior != BG) & (interior != FRAME))
        assert len(seeds) == 1, "each frame interior must contain exactly one colored seed"
        sr, sc = map(int, seeds[0])
        color = int(interior[sr, sc])
        assert color in templates, "seed color must appear in the legend"

        global_seed_r = canvas_offset[0] + ir0 + sr
        global_seed_c = canvas_offset[1] + ic0 + sc
        global_interior = (
            canvas_offset[0] + ir0,
            canvas_offset[1] + ic0,
            canvas_offset[0] + ir1,
            canvas_offset[1] + ic1,
        )
        _stamp(out, color, templates[color], global_seed_r, global_seed_c, global_interior)

    return out


def _random_path(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = start
    tr, tc = end
    cells = [(r, c)]
    while (r, c) != (tr, tc):
        move_row = r != tr and (c == tc or random.random() < 0.5)
        if move_row:
            r += 1 if tr > r else -1
        else:
            c += 1 if tc > c else -1
        cells.append((r, c))
    return cells


def _sample_template_mask() -> np.ndarray | None:
    dims = random.choice(
        ((1, 3), (3, 1), (3, 3), (3, 5), (5, 3), (5, 5))
    )
    h, w = dims
    center = (h // 2, w // 2)
    cells = {center}
    targets = []
    if h > 1:
        targets.extend([(0, random.randrange(w)), (h - 1, random.randrange(w))])
    if w > 1:
        targets.extend([(random.randrange(h), 0), (random.randrange(h), w - 1)])
    for target in targets:
        cells.update(_random_path(center, target))

    max_area = h * w
    target_area = random.randint(len(cells), max_area)
    for _ in range(100):
        if len(cells) >= target_area:
            break
        frontier = []
        for r, c in cells:
            for nr, nc in _neighbors4(r, c, h, w):
                if (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            break
        cells.add(random.choice(frontier))

    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return None
    if rows[0] != 0 or rows[-1] != h - 1 or cols[0] != 0 or cols[-1] != w - 1:
        return None
    return mask


def _place_patches(
    region_shape: tuple[int, int], patches: list[tuple[int, np.ndarray]]
) -> dict[int, tuple[np.ndarray, int, int]] | None:
    rh, rw = region_shape
    placements: dict[int, tuple[np.ndarray, int, int]] = {}
    placed_boxes: list[tuple[int, int, int, int]] = []
    order = sorted(patches, key=lambda item: int(item[1].sum()), reverse=True)
    for color, patch in order:
        ph, pw = map(int, patch.shape)
        if rh < ph + 2 or rw < pw + 2:
            return None
        for _ in range(200):
            r0 = random.randint(1, rh - ph - 1)
            c0 = random.randint(1, rw - pw - 1)
            box = (r0, c0, r0 + ph - 1, c0 + pw - 1)
            if any(_boxes_overlap(box, other, gap=1) for other in placed_boxes):
                continue
            placed_boxes.append(box)
            placements[color] = (patch, r0, c0)
            break
        else:
            return None
    return placements


def _seed_position_sets(mask_shape: tuple[int, int], interior_shape: tuple[int, int]):
    th, tw = mask_shape
    ih, iw = interior_shape
    center_r = th // 2
    center_c = tw // 2
    full_rows = range(center_r, ih - (th - 1 - center_r))
    full_cols = range(center_c, iw - (tw - 1 - center_c))
    full = {(r, c) for r in full_rows for c in full_cols}
    all_pos = {(r, c) for r in range(ih) for c in range(iw)}
    crop = all_pos - full
    return sorted(full), sorted(crop)


def _sample_frame_spec(
    color: int,
    mask: np.ndarray,
    want_crop: bool,
    canvas_shape: tuple[int, int],
) -> dict[str, object] | None:
    th, tw = map(int, mask.shape)
    max_ih = canvas_shape[0] - 4
    max_iw = canvas_shape[1] - 4
    if max_ih < 1 or max_iw < 1:
        return None

    for _ in range(120):
        if want_crop:
            ih_low = max(1, th - 2)
            ih_high = min(max_ih, th + 2)
            iw_low = max(1, tw - 2)
            iw_high = min(max_iw, tw + 2)
        else:
            if max_ih < th or max_iw < tw:
                continue
            ih_low = th
            ih_high = min(max_ih, th + 3)
            iw_low = tw
            iw_high = min(max_iw, tw + 3)
        if ih_low > ih_high or iw_low > iw_high:
            continue
        ih = random.randint(ih_low, ih_high)
        iw = random.randint(iw_low, iw_high)
        full_positions, crop_positions = _seed_position_sets(mask.shape, (ih, iw))
        if want_crop and crop_positions:
            seed = random.choice(crop_positions)
            clipped = True
        elif (not want_crop) and full_positions:
            seed = random.choice(full_positions)
            clipped = False
        elif crop_positions:
            seed = random.choice(crop_positions)
            clipped = True
        elif full_positions:
            seed = random.choice(full_positions)
            clipped = False
        else:
            continue
        return {
            "color": color,
            "mask": mask,
            "frame_h": ih + 2,
            "frame_w": iw + 2,
            "seed_local": seed,
            "clipped": clipped,
        }
    return None


def _place_frames(
    canvas_shape: tuple[int, int], specs: list[dict[str, object]]
) -> list[dict[str, object]] | None:
    ch, cw = canvas_shape
    placed_boxes: list[tuple[int, int, int, int]] = []
    placed: list[dict[str, object]] = []
    order = sorted(specs, key=lambda spec: int(spec["frame_h"]) * int(spec["frame_w"]), reverse=True)
    for spec in order:
        fh = int(spec["frame_h"])
        fw = int(spec["frame_w"])
        if ch < fh + 2 or cw < fw + 2:
            return None
        for _ in range(250):
            r0 = random.randint(1, ch - fh - 1)
            c0 = random.randint(1, cw - fw - 1)
            box = (r0, c0, r0 + fh - 1, c0 + fw - 1)
            if any(_boxes_overlap(box, other, gap=1) for other in placed_boxes):
                continue
            placed_boxes.append(box)
            placed.append({**spec, "r0": r0, "c0": c0})
            break
        else:
            return None
    return placed


def _draw_frame(canvas: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    canvas[r0, c0 : c0 + w] = FRAME
    canvas[r0 + h - 1, c0 : c0 + w] = FRAME
    canvas[r0 : r0 + h, c0] = FRAME
    canvas[r0 : r0 + h, c0 + w - 1] = FRAME


def _render_input(
    shape: tuple[int, int],
    orient: str,
    axis: int,
    legend_is_first: bool,
    legend_shape: tuple[int, int],
    canvas_shape: tuple[int, int],
    template_layout: dict[int, tuple[np.ndarray, int, int]],
    frame_layout: list[dict[str, object]],
) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    if orient == "horizontal":
        grid[axis, :] = SEPARATOR
        if legend_is_first:
            legend_slice = (slice(0, legend_shape[0]), slice(None))
            canvas_slice = (slice(axis + 1, axis + 1 + canvas_shape[0]), slice(None))
        else:
            canvas_slice = (slice(0, canvas_shape[0]), slice(None))
            legend_slice = (slice(axis + 1, axis + 1 + legend_shape[0]), slice(None))
    else:
        grid[:, axis] = SEPARATOR
        if legend_is_first:
            legend_slice = (slice(None), slice(0, legend_shape[1]))
            canvas_slice = (slice(None), slice(axis + 1, axis + 1 + canvas_shape[1]))
        else:
            canvas_slice = (slice(None), slice(0, canvas_shape[1]))
            legend_slice = (slice(None), slice(axis + 1, axis + 1 + legend_shape[1]))

    legend = grid[legend_slice]
    for color, (mask, r0, c0) in template_layout.items():
        ph, pw = mask.shape
        legend[r0 : r0 + ph, c0 : c0 + pw][mask] = int(color)

    canvas = grid[canvas_slice]
    for spec in frame_layout:
        r0 = int(spec["r0"])
        c0 = int(spec["c0"])
        fh = int(spec["frame_h"])
        fw = int(spec["frame_w"])
        seed_r, seed_c = spec["seed_local"]
        color = int(spec["color"])
        _draw_frame(canvas, r0, c0, fh, fw)
        canvas[r0 + 1 + seed_r, c0 + 1 + seed_c] = color
    return grid


def _gen_attempt() -> np.ndarray | None:
    orient = random.choice(("horizontal", "vertical"))
    legend_is_first = bool(random.getrandbits(1))

    if orient == "horizontal":
        legend_h = random.randint(8, 11)
        canvas_h = random.randint(12, 18)
        width = random.randint(19, 28)
        height = legend_h + 1 + canvas_h
        axis = legend_h if legend_is_first else canvas_h
        shape = (height, width)
        legend_shape = (legend_h, width)
        canvas_shape = (canvas_h, width)
    else:
        legend_w = random.randint(8, 11)
        canvas_w = random.randint(12, 18)
        height = random.randint(19, 28)
        width = legend_w + 1 + canvas_w
        axis = legend_w if legend_is_first else canvas_w
        shape = (height, width)
        legend_shape = (height, legend_w)
        canvas_shape = (height, canvas_w)

    template_count = random.randint(2, 5)
    colors = random.sample(TEMPLATE_COLORS, template_count)
    templates: dict[int, np.ndarray] = {}
    for color in colors:
        mask = _sample_template_mask()
        if mask is None:
            return None
        templates[color] = mask

    template_layout = _place_patches(legend_shape, list(templates.items()))
    if template_layout is None:
        return None

    region_count = random.randint(2, 4)
    region_colors = [random.choice(colors) for _ in range(region_count)]
    if len(set(region_colors)) < 2:
        return None

    want_crop_flags = [i % 2 == 0 for i in range(region_count)]
    random.shuffle(want_crop_flags)

    specs: list[dict[str, object]] = []
    for color, want_crop in zip(region_colors, want_crop_flags):
        spec = _sample_frame_spec(color, templates[color], want_crop, canvas_shape)
        if spec is None:
            return None
        specs.append(spec)

    if not any(bool(spec["clipped"]) for spec in specs):
        return None
    if not any(not bool(spec["clipped"]) for spec in specs):
        return None

    frame_layout = _place_frames(canvas_shape, specs)
    if frame_layout is None:
        return None

    inp = _render_input(
        shape,
        orient,
        axis,
        legend_is_first,
        legend_shape,
        canvas_shape,
        template_layout,
        frame_layout,
    )
    try:
        out = main(inp)
    except Exception:
        return None

    canvas_only = out.copy()
    if orient == "horizontal":
        if legend_is_first:
            canvas_only[: legend_shape[0], :] = BG
        else:
            canvas_only[canvas_shape[0] + 1 :, :] = BG
    else:
        if legend_is_first:
            canvas_only[:, : legend_shape[1]] = BG
        else:
            canvas_only[:, canvas_shape[1] + 1 :] = BG
    if np.array_equal(inp, out):
        return None
    if np.count_nonzero(canvas_only == FRAME) == 0:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
