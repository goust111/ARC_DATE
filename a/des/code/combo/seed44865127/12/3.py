from __future__ import annotations

import random

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
BRANCHES = ("translate", "rotate", "dilate")
PANEL_COUNT_OPTIONS = (3, 4)
PANEL_HEIGHT_RANGE = (8, 13)
PANEL_WIDTH_RANGE = (8, 13)
MAX_GENERATION_ATTEMPTS = 500


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_mask(mask: np.ndarray) -> tuple[tuple[int, int, int, int], np.ndarray]:
    box = _bbox(mask)
    assert box is not None, "each panel must contain a foreground pattern"
    r0, c0, r1, c1 = box
    return box, mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _dilate4(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _render_panel(mask: np.ndarray, bg: int, fg: int) -> np.ndarray:
    panel = np.full(mask.shape, bg, dtype=int)
    panel[mask] = fg
    return panel


def _split_panels(grid: np.ndarray) -> tuple[list[np.ndarray], int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    candidates: list[tuple[int, list[int]]] = []

    for color in map(int, np.unique(g)):
        cols = [c for c in range(w) if np.all(g[:, c] == color)]
        if len(cols) not in (2, 3):
            continue
        if cols[0] == 0 or cols[-1] == w - 1:
            continue

        widths = []
        start = 0
        valid = True
        for c in cols:
            width = c - start
            if width <= 0:
                valid = False
                break
            widths.append(width)
            panel = g[:, start:c]
            if np.any(panel == color):
                valid = False
                break
            start = c + 1

        if not valid:
            continue
        width = w - start
        if width <= 0:
            continue
        widths.append(width)
        last_panel = g[:, start:w]
        if np.any(last_panel == color):
            continue
        if len(set(widths)) != 1:
            continue
        candidates.append((color, cols))

    assert len(candidates) == 1, "expected one regular separator pattern"
    sep_color, sep_cols = candidates[0]

    panels = []
    start = 0
    for c in sep_cols:
        panels.append(g[:, start:c].copy())
        start = c + 1
    panels.append(g[:, start:w].copy())

    assert len(panels) in PANEL_COUNT_OPTIONS
    assert len({panel.shape for panel in panels}) == 1, "panels must have equal size"
    return panels, int(sep_color)


def _panel_info(panel: np.ndarray, bg: int) -> dict[str, object]:
    mask = panel != bg
    colors = {int(v) for v in np.unique(panel[mask])}
    assert len(colors) == 1, "each panel should contain exactly one foreground color"
    fg = colors.pop()
    box, crop = _crop_mask(mask)
    return {"mask": mask, "bbox": box, "crop": crop, "fg": fg}


def _infer_bg(panels: list[np.ndarray]) -> int:
    corners = {
        int(panel[0, 0])
        for panel in panels
    }
    corners |= {
        int(panel[0, -1])
        for panel in panels
    }
    corners |= {
        int(panel[-1, 0])
        for panel in panels
    }
    corners |= {
        int(panel[-1, -1])
        for panel in panels
    }
    assert len(corners) == 1, "all panel corners should share the background color"
    return corners.pop()


def _predict_translation(infos: list[dict[str, object]], panel_shape: tuple[int, int]) -> np.ndarray | None:
    crops = [info["crop"] for info in infos]
    boxes = [info["bbox"] for info in infos]
    first = crops[0]
    for crop in crops[1:]:
        if crop.shape != first.shape or not np.array_equal(crop, first):
            return None

    positions = [(box[0], box[1]) for box in boxes]
    deltas = [
        (positions[i + 1][0] - positions[i][0], positions[i + 1][1] - positions[i][1])
        for i in range(len(positions) - 1)
    ]
    if len(set(deltas)) != 1:
        return None
    dr, dc = deltas[0]
    if dr == 0 and dc == 0:
        return None

    h, w = first.shape
    r0 = positions[-1][0] + dr
    c0 = positions[-1][1] + dc
    if not (0 <= r0 and 0 <= c0 and r0 + h <= panel_shape[0] and c0 + w <= panel_shape[1]):
        return None

    pred = np.zeros(panel_shape, dtype=bool)
    pred[r0 : r0 + h, c0 : c0 + w] = first
    return pred


def _predict_rotation(infos: list[dict[str, object]], panel_shape: tuple[int, int]) -> np.ndarray | None:
    crops = [info["crop"] for info in infos]
    boxes = [info["bbox"] for info in infos]
    anchors = {(box[0], box[1]) for box in boxes}
    if len(anchors) != 1:
        return None
    r0, c0 = next(iter(anchors))

    candidates = []
    for k in (1, 2, 3):
        ok = True
        for i in range(len(crops) - 1):
            if not np.array_equal(np.rot90(crops[i], k), crops[i + 1]):
                ok = False
                break
        if not ok:
            continue

        pred_crop = np.rot90(crops[-1], k)
        if r0 + pred_crop.shape[0] > panel_shape[0] or c0 + pred_crop.shape[1] > panel_shape[1]:
            continue
        pred = np.zeros(panel_shape, dtype=bool)
        pred[r0 : r0 + pred_crop.shape[0], c0 : c0 + pred_crop.shape[1]] = pred_crop
        candidates.append(pred)

    if len(candidates) != 1:
        return None
    return candidates[0]


def _predict_dilation(infos: list[dict[str, object]], _panel_shape: tuple[int, int]) -> np.ndarray | None:
    masks = [info["mask"] for info in infos]
    for i in range(len(masks) - 1):
        nxt = _dilate4(masks[i])
        if np.array_equal(nxt, masks[i]):
            return None
        if not np.array_equal(nxt, masks[i + 1]):
            return None
    pred = _dilate4(masks[-1])
    if np.array_equal(pred, masks[-1]):
        return None
    return pred


def _infer_next_mask(panels: list[np.ndarray]) -> tuple[np.ndarray, int, int]:
    bg = _infer_bg(panels)
    infos = [_panel_info(panel, bg) for panel in panels]
    fg_colors = {int(info["fg"]) for info in infos}
    assert len(fg_colors) == 1, "foreground color must stay constant across panels"
    fg = fg_colors.pop()
    panel_shape = panels[0].shape

    candidates = []
    for name, fn in (
        ("translate", _predict_translation),
        ("rotate", _predict_rotation),
        ("dilate", _predict_dilation),
    ):
        pred = fn(infos, panel_shape)
        if pred is not None:
            candidates.append((name, pred))

    assert len(candidates) == 1, "trend must be uniquely identifiable"
    return candidates[0][1], bg, fg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find the geometric trend across horizontal panels and extrapolate one more panel.

    The generator uses three trend families:
    - constant translation by a fixed vector,
    - repeated 90-degree rotation of the cropped motif,
    - repeated 4-neighbour dilation on the full panel canvas.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    panels, _sep = _split_panels(g)
    next_mask, bg, fg = _infer_next_mask(panels)
    return _render_panel(next_mask, bg, fg)


def _normalize_cells(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return {(r - min_r, c - min_c) for r, c in cells}


def _cells_to_mask(cells: set[tuple[int, int]]) -> np.ndarray:
    max_r = max(r for r, _ in cells)
    max_c = max(c for _, c in cells)
    mask = np.zeros((max_r + 1, max_c + 1), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _random_polyomino(
    max_h: int,
    max_w: int,
    min_cells: int,
    max_cells: int,
    require_unique_rotations: bool = False,
) -> np.ndarray | None:
    max_cells = min(max_cells, max_h * max_w)
    if min_cells > max_cells:
        return None

    for _ in range(120):
        target = random.randint(min_cells, max_cells)
        cells: set[tuple[int, int]] = {(0, 0)}

        while len(cells) < target:
            frontier = set()
            for r, c in cells:
                for dr, dc in DIR4:
                    nxt = (r + dr, c + dc)
                    if nxt not in cells:
                        frontier.add(nxt)

            options = list(frontier)
            random.shuffle(options)
            placed = False
            for nxt in options:
                trial = _normalize_cells(set(cells) | {nxt})
                mask = _cells_to_mask(trial)
                if mask.shape[0] <= max_h and mask.shape[1] <= max_w:
                    cells = trial
                    placed = True
                    break
            if not placed:
                break

        if len(cells) != target:
            continue
        mask = _cells_to_mask(_normalize_cells(cells))
        if require_unique_rotations:
            rots = [np.rot90(mask, k) for k in range(4)]
            unique = []
            for rot in rots:
                if not any(rot.shape == seen.shape and np.array_equal(rot, seen) for seen in unique):
                    unique.append(rot)
            if len(unique) != 4:
                continue
        return mask
    return None


def _place_crop(
    panel_h: int,
    panel_w: int,
    crop: np.ndarray,
    top: int,
    left: int,
) -> np.ndarray:
    mask = np.zeros((panel_h, panel_w), dtype=bool)
    h, w = crop.shape
    mask[top : top + h, left : left + w] = crop
    return mask


def _build_translation_sequence(panel_h: int, panel_w: int, n_panels: int) -> list[np.ndarray] | None:
    step_count = n_panels
    delta_choices = [
        (-2, 0), (-1, 0), (1, 0), (2, 0),
        (0, -2), (0, -1), (0, 1), (0, 2),
        (-1, -1), (-1, 1), (1, -1), (1, 1),
    ]

    for _ in range(120):
        dr, dc = random.choice(delta_choices)
        max_h = panel_h - 2 - abs(dr) * step_count
        max_w = panel_w - 2 - abs(dc) * step_count
        if max_h < 2 or max_w < 2:
            continue

        crop = _random_polyomino(
            max_h=min(5, max_h),
            max_w=min(5, max_w),
            min_cells=3,
            max_cells=6,
        )
        if crop is None:
            continue
        h, w = crop.shape

        if dr >= 0:
            r_min, r_max = 1, panel_h - h - 1 - dr * step_count
        else:
            r_min, r_max = 1 - dr * step_count, panel_h - h - 1
        if dc >= 0:
            c_min, c_max = 1, panel_w - w - 1 - dc * step_count
        else:
            c_min, c_max = 1 - dc * step_count, panel_w - w - 1
        if r_min > r_max or c_min > c_max:
            continue

        top = random.randint(r_min, r_max)
        left = random.randint(c_min, c_max)
        return [
            _place_crop(panel_h, panel_w, crop, top + i * dr, left + i * dc)
            for i in range(n_panels + 1)
        ]
    return None


def _build_rotation_sequence(panel_h: int, panel_w: int, n_panels: int) -> list[np.ndarray] | None:
    for _ in range(120):
        crop = _random_polyomino(
            max_h=4,
            max_w=4,
            min_cells=4,
            max_cells=6,
            require_unique_rotations=True,
        )
        if crop is None:
            continue

        k = random.choice((1, 2, 3))
        span = max(crop.shape)
        if panel_h < span + 2 or panel_w < span + 2:
            continue
        top = random.randint(1, panel_h - span - 1)
        left = random.randint(1, panel_w - span - 1)

        masks = []
        for i in range(n_panels + 1):
            rot = np.rot90(crop, k * i)
            masks.append(_place_crop(panel_h, panel_w, rot, top, left))
        return masks
    return None


def _build_dilation_sequence(panel_h: int, panel_w: int, n_panels: int) -> list[np.ndarray] | None:
    margin = n_panels + 1
    for _ in range(120):
        crop = _random_polyomino(
            max_h=3,
            max_w=3,
            min_cells=1,
            max_cells=3,
        )
        if crop is None:
            continue
        h, w = crop.shape
        if panel_h < h + 2 * margin or panel_w < w + 2 * margin:
            continue

        top = random.randint(margin, panel_h - h - margin)
        left = random.randint(margin, panel_w - w - margin)
        masks = [_place_crop(panel_h, panel_w, crop, top, left)]
        ok = True
        for _step in range(n_panels):
            nxt = _dilate4(masks[-1])
            if np.array_equal(nxt, masks[-1]):
                ok = False
                break
            masks.append(nxt)
        if ok:
            return masks
    return None


def _render_input(panels: list[np.ndarray], sep: int) -> np.ndarray:
    h, w = panels[0].shape
    out_w = len(panels) * w + (len(panels) - 1)
    grid = np.full((h, out_w), sep, dtype=int)
    for i, panel in enumerate(panels):
        c0 = i * (w + 1)
        grid[:, c0 : c0 + w] = panel
    return grid


def _gen_attempt(branch: str) -> tuple[np.ndarray, np.ndarray] | None:
    n_panels = random.choice(PANEL_COUNT_OPTIONS)
    panel_h = random.randint(*PANEL_HEIGHT_RANGE)
    panel_w = random.randint(*PANEL_WIDTH_RANGE)

    if branch == "translate":
        masks = _build_translation_sequence(panel_h, panel_w, n_panels)
    elif branch == "rotate":
        masks = _build_rotation_sequence(panel_h, panel_w, n_panels)
    else:
        masks = _build_dilation_sequence(panel_h, panel_w, n_panels)
    if masks is None:
        return None

    bg, fg, sep = random.sample(range(10), 3)
    input_panels = [_render_panel(mask, bg, fg) for mask in masks[:-1]]
    expected = _render_panel(masks[-1], bg, fg)
    grid = _render_input(input_panels, sep)

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    branch = random.choice(BRANCHES)
    for _ in range(MAX_GENERATION_ATTEMPTS):
        sample = _gen_attempt(branch)
        if sample is not None:
            return sample[0]
    raise RuntimeError(f"failed to generate a valid input for branch {branch}")
