from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


FRAME_COLORS = (1, 4, 6, 7)
TEMPLATE_COLORS = (2, 3, 5, 8, 9)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    extra = [(-1, -1), (-1, 1), (1, -1), (1, 1)] if connectivity == 8 else []
    dirs = list(DIR4) + extra
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in dirs:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _find_monochrome_components(grid: np.ndarray, bg: int) -> list[dict]:
    comps: list[dict] = []
    for color in sorted(set(map(int, grid.ravel())) - {int(bg)}):
        for coords in _components_mask(grid == color, connectivity=4):
            comps.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": _bbox_coords(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return comps


def _is_hollow_frame(grid: np.ndarray, comp: dict, bg: int) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    color = int(comp["color"])
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    if comp["area"] != 2 * (h + w) - 4:
        return False
    if not (
        np.all(grid[r0, c0 : c1 + 1] == color)
        and np.all(grid[r1, c0 : c1 + 1] == color)
        and np.all(grid[r0 : r1 + 1, c0] == color)
        and np.all(grid[r0 : r1 + 1, c1] == color)
    ):
        return False
    return bool(np.all(grid[r0 + 1 : r1, c0 + 1 : c1] == bg))


def _find_frames(grid: np.ndarray, bg: int) -> tuple[list[dict], np.ndarray]:
    frames = [comp for comp in _find_monochrome_components(grid, bg) if _is_hollow_frame(grid, comp, bg)]
    assert frames, "expected at least one hollow frame"
    mask = np.zeros(grid.shape, dtype=bool)
    for frame in frames:
        rr, cc = frame["coords"][:, 0], frame["coords"][:, 1]
        mask[rr, cc] = True
    return frames, mask


def _extract_template(grid: np.ndarray, frame_mask: np.ndarray, bg: int) -> np.ndarray:
    leftover = (grid != bg) & (~frame_mask)
    comps = _components_mask(leftover, connectivity=4)
    candidates = []
    for coords in comps:
        r0, c0, r1, c1 = _bbox_coords(coords)
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        colors = {int(v) for v in patch.ravel() if int(v) != bg}
        if len(colors) < 2:
            continue
        if np.any(patch == bg):
            continue
        candidates.append(patch.copy())
    assert len(candidates) == 1, "expected exactly one multicolor rectangular template"
    return candidates[0]


def _scale_nn(template: np.ndarray, factor: int) -> np.ndarray:
    return np.repeat(np.repeat(template, factor, axis=0), factor, axis=1)


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    dst[r0 : r0 + h, c0 : c0 + w] = patch


def _template_majority_color(template: np.ndarray, bg: int) -> int:
    counts = Counter(int(v) for v in template.ravel() if int(v) != bg)
    ranked = counts.most_common()
    assert ranked, "template must contain non-background colors"
    assert len(ranked) == 1 or ranked[0][1] > ranked[1][1], "template must have a unique majority color"
    return int(ranked[0][0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Scale a multicolor template to each hollow frame's interior, recolor its majority color to the frame color, and stamp it inside.

    Input structure:
    - A black-dominant background contains one compact, fully occupied multicolor template rectangle.
    - Elsewhere there are 2-4 hollow single-color rectangular frames with 1-pixel borders and black interiors.
    - Each frame's interior size is an integer isotropic multiple of the template size.

    Rule:
    - Keep the original template and every frame border unchanged.
    - For each hollow frame, compute the scale factor from interior size / template size, enlarge the template by nearest-neighbor replication, then replace every pixel of the template's unique majority color with that frame's border color.
    - Stamp the recolored scaled copy into the frame interior aligned to the interior's top-left corner.

    Key insight:
    - Each frame simultaneously encodes where to place the copy, how much to scale it, and which color should replace the template's most frequent color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    frames, frame_mask = _find_frames(g, bg)
    template = _extract_template(g, frame_mask, bg)
    th, tw = template.shape
    majority = _template_majority_color(template, bg)

    out = g.copy()
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        inner_h = r1 - r0 - 1
        inner_w = c1 - c0 - 1
        assert inner_h > 0 and inner_w > 0, "frame interior must be non-empty"
        assert inner_h % th == 0 and inner_w % tw == 0, "frame interior must be an integer multiple of template size"
        scale_r = inner_h // th
        scale_c = inner_w // tw
        assert scale_r == scale_c, "frame interior must encode isotropic scaling"
        scaled = _scale_nn(template, scale_r)
        stamped = scaled.copy()
        stamped[stamped == majority] = int(frame["color"])
        _paste(out, stamped, r0 + 1, c0 + 1)

    return out


def _make_template(rng: random.Random, h: int, w: int) -> np.ndarray:
    n = h * w
    color_count = 2 if n <= 6 else rng.choice([2, 3])
    colors = rng.sample(TEMPLATE_COLORS, color_count)
    weights = [6] + [2] * (color_count - 1)
    for _ in range(200):
        flat = [rng.choices(colors, weights=weights, k=1)[0] for _ in range(n)]
        counts = Counter(flat)
        ranked = counts.most_common()
        if any(counts[c] == 0 for c in colors):
            continue
        if ranked[0][1] == ranked[1][1]:
            continue
        template = np.array(flat, dtype=int).reshape(h, w)
        if np.array_equal(template, np.full((h, w), template[0, 0], dtype=int)):
            continue
        if np.array_equal(template, np.fliplr(template)) and np.array_equal(template, np.flipud(template)):
            continue
        return template
    raise RuntimeError("failed to create a valid template")


def _choose_scales(rng: random.Random, frame_count: int) -> list[int]:
    if frame_count == 4:
        scales = [1, 1, 2, 2]
        rng.shuffle(scales)
        return scales
    if frame_count == 3:
        if rng.random() < 0.6:
            scales = [1, 2, 3]
        else:
            scales = [1, 2, rng.choice([1, 2, 3])]
        rng.shuffle(scales)
        return scales
    scales = [1, rng.choice([2, 3])]
    if rng.random() < 0.35:
        scales[1] = 1
    rng.shuffle(scales)
    return scales


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _boxes_with_gap_do_not_overlap(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    gap: int = 1,
) -> bool:
    return a[2] + gap < b[0] or b[2] + gap < a[0] or a[3] + gap < b[1] or b[3] + gap < a[1]


def _layout_scene(
    rng: random.Random,
    template_shape: tuple[int, int],
    frame_dims: list[tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int], list[tuple[int, int, int, int]]] | None:
    th, tw = template_shape
    frame_count = len(frame_dims)
    order = list(range(frame_count))
    rng.shuffle(order)

    if frame_count == 4:
        top_count = 2
    elif frame_count == 3:
        top_count = rng.choice([1, 2])
    else:
        top_count = 1

    top_ids = order[:top_count]
    bottom_ids = order[top_count:]

    template_rc = (rng.randint(1, 2), rng.randint(1, 2))
    template_box = (
        template_rc[0],
        template_rc[1],
        template_rc[0] + th - 1,
        template_rc[1] + tw - 1,
    )

    boxes: list[tuple[int, int, int, int] | None] = [None] * frame_count
    top_y = rng.randint(1, 3)
    x = template_rc[1] + tw + rng.randint(3, 5)
    top_bottom = template_box[2]
    for idx in top_ids:
        fh, fw = frame_dims[idx]
        y = top_y + rng.randint(0, 1)
        boxes[idx] = (y, x, y + fh - 1, x + fw - 1)
        top_bottom = max(top_bottom, y + fh - 1)
        x += fw + rng.randint(2, 4)

    bottom_start = top_bottom + rng.randint(3, 5)
    x = rng.randint(1, 3)
    bottom_bottom = template_box[2]
    for idx in bottom_ids:
        fh, fw = frame_dims[idx]
        y = bottom_start + rng.randint(0, 1)
        boxes[idx] = (y, x, y + fh - 1, x + fw - 1)
        bottom_bottom = max(bottom_bottom, y + fh - 1)
        x += fw + rng.randint(2, 4)

    final_boxes = [box for box in boxes if box is not None]
    assert len(final_boxes) == frame_count
    all_boxes = [template_box] + final_boxes
    for i in range(len(all_boxes)):
        for j in range(i + 1, len(all_boxes)):
            if not _boxes_with_gap_do_not_overlap(all_boxes[i], all_boxes[j], gap=1):
                return None

    max_r = max(box[2] for box in all_boxes)
    max_c = max(box[3] for box in all_boxes)
    h = max_r + 1 + rng.randint(1, 3)
    w = max_c + 1 + rng.randint(1, 3)
    if h > 30 or w > 30:
        return None
    return (h, w), template_rc, final_boxes


def _render_expected_output(
    inp: np.ndarray,
    template: np.ndarray,
    frames: list[dict],
    bg: int,
) -> np.ndarray:
    out = inp.copy()
    majority = _template_majority_color(template, bg)
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        k = (r1 - r0 - 1) // template.shape[0]
        patch = _scale_nn(template, k)
        patch = patch.copy()
        patch[patch == majority] = int(frame["color"])
        _paste(out, patch, r0 + 1, c0 + 1)
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    frame_count = rng.randint(2, 4)
    th = rng.randint(2, 4)
    tw = rng.randint(2, 4)
    template = _make_template(rng, th, tw)
    scales = _choose_scales(rng, frame_count)
    frame_colors = rng.sample(FRAME_COLORS, frame_count)
    frame_dims = [(th * k + 2, tw * k + 2) for k in scales]

    layout = _layout_scene(rng, (th, tw), frame_dims)
    if layout is None:
        return None
    shape, template_rc, frame_boxes = layout

    inp = np.zeros(shape, dtype=int)
    _paste(inp, template, template_rc[0], template_rc[1])

    frame_records: list[dict] = []
    for idx, box in enumerate(frame_boxes):
        color = int(frame_colors[idx])
        _draw_frame(inp, box, color)
        frame_records.append(
            {
                "color": color,
                "bbox": box,
                "scale": int(scales[idx]),
            }
        )

    if _get_bg(inp) != 0:
        return None

    expected = _render_expected_output(inp, template, frame_records, bg=0)
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
