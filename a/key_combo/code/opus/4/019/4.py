from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, find_components, get_bg, paste, translate


def _runs(values: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx, value in enumerate(values.tolist()):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            runs.append((start, idx - 1))
            start = None
    if start is not None:
        runs.append((start, len(values) - 1))
    return runs


def _extract_panels(grid: np.ndarray) -> tuple[int, list[np.ndarray]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    separator = int(g[0, 0])
    assert np.all(g[0, :] == separator) and np.all(g[-1, :] == separator)
    assert np.all(g[:, 0] == separator) and np.all(g[:, -1] == separator)

    row_runs = _runs(np.any(g != separator, axis=1))
    col_runs = _runs(np.any(g != separator, axis=0))
    assert len(row_runs) == 1, "expected one horizontal band of panel interiors"
    assert len(col_runs) == 3, "expected three side-by-side panels"

    r0, r1 = row_runs[0]
    panels = [g[r0 : r1 + 1, c0 : c1 + 1].copy() for c0, c1 in col_runs]

    shapes = {panel.shape for panel in panels}
    assert len(shapes) == 1, "panels must have the same size"
    for panel in panels:
        assert not np.any(panel == separator), "separator color must stay outside panels"

    return separator, panels


def _single_object(panel: np.ndarray) -> tuple[int, int, np.ndarray]:
    bg = int(get_bg(panel))
    comps = find_components(panel, bg=bg, connectivity=4, monochromatic=True)
    assert len(comps) == 1, "each panel must contain exactly one object"

    comp = comps[0]
    obj = np.full(panel.shape, bg, dtype=int)
    coords = comp["coords"]
    obj[coords[:, 0], coords[:, 1]] = int(comp["color"])
    return int(comp["color"]), int(comp["area"]), obj


def _overlay_layers(layers: list[np.ndarray], areas: list[int], bg: int) -> np.ndarray:
    out = np.full(layers[0].shape, int(bg), dtype=int)
    order = sorted(range(len(layers)), key=lambda idx: (areas[idx], -idx))
    for idx in order:
        mask = layers[idx] != bg
        out[mask] = layers[idx][mask]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Shift each panel object upward by its area, then overlay the three panels with area-based priority.

    Input structure:
    - A grey separator frame and two grey separator columns split the grid into three same-sized horizontal panels.
    - Each panel interior has a black background and exactly one 4-connected monochrome object.
    - The three panel objects have distinct areas, and only the panel interiors matter for the output.

    Rule:
    - Inside each panel, translate the object straight upward by a number of rows equal to its pixel count; pixels that move above the top edge are clipped away.
    - Remove the separator structure and transparently overlay the three shifted panels into one panel-sized output.
    - At empty cells output black; at conflicts choose the color of the object with the largest area, breaking equal-area ties in favor of the leftmost panel.

    Key insight:
    - The same quantity, object area, controls both the vertical displacement and the conflict winner during overlay, so motion and priority must be solved together.
    """
    _separator, panels = _extract_panels(np.asarray(grid, dtype=int))

    bg = int(get_bg(panels[0]))
    assert all(int(get_bg(panel)) == bg for panel in panels), "panel backgrounds must agree"

    shifted_layers: list[np.ndarray] = []
    areas: list[int] = []
    for panel in panels:
        _color, area, obj = _single_object(panel)
        shifted_layers.append(translate(obj, -area, 0, background=bg))
        areas.append(area)

    return _overlay_layers(shifted_layers, areas, bg)


_GEN_SEPARATOR = Color.GREY
_GEN_BG = Color.BLACK
_GEN_COLORS = [Color.BLUE, Color.RED, Color.GREEN]
_GEN_PANEL_H_RANGE = (9, 12)
_GEN_PANEL_W_RANGE = (5, 8)
_GEN_AREA_CHOICES = [3, 4, 5, 6, 7]
_GEN_MAX_ATTEMPTS = 400
_GEN_CLIP_PROB = 0.6


def _tight_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].astype(bool)


def _random_connected_mask(area: int) -> np.ndarray | None:
    candidates = [
        (h, w)
        for h in range(2, min(4, area) + 1)
        for w in range(2, min(4, area) + 1)
        if area <= h * w
    ]
    for _ in range(80):
        h, w = random.choice(candidates)
        cells = {(random.randrange(h), random.randrange(w))}
        while len(cells) < area:
            frontier = set()
            for r, c in cells:
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(tuple(frontier)))

        if len(cells) != area:
            continue

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        mask = _tight_mask(mask)
        if 2 <= mask.shape[0] <= 4 and 2 <= mask.shape[1] <= 4:
            return mask
    return None


def _render_mask(
    mask: np.ndarray,
    color: int,
    top: int,
    left: int,
    panel_h: int,
    panel_w: int,
) -> np.ndarray:
    patch = np.full(mask.shape, _GEN_BG, dtype=int)
    patch[mask] = int(color)
    canvas = np.full((panel_h, panel_w), _GEN_BG, dtype=int)
    paste(canvas, patch, top, left, transparent=_GEN_BG, clip=True)
    return canvas


def _visible_mask(mask: np.ndarray, top: int, left: int, panel_h: int, panel_w: int) -> np.ndarray:
    return _render_mask(mask, 1, top, left, panel_h, panel_w) != _GEN_BG


def _feasible_target_placements(
    mask: np.ndarray,
    area: int,
    panel_h: int,
    panel_w: int,
    target_r: int,
    target_c: int,
    *,
    require_clip: bool,
) -> list[tuple[int, int]]:
    h, w = mask.shape
    options: list[tuple[int, int]] = []
    for mr, mc in np.argwhere(mask):
        sr = int(target_r - int(mr))
        sc = int(target_c - int(mc))
        input_top = sr + area
        if sc < 0 or sc + w > panel_w:
            continue
        if input_top < 0 or input_top + h > panel_h:
            continue
        if require_clip and sr >= 0:
            continue
        if not require_clip and sr < 0:
            continue
        if input_top + h <= panel_h // 2:
            continue
        options.append((sr, sc))
    return options


def _random_shift(
    mask: np.ndarray,
    area: int,
    panel_h: int,
    panel_w: int,
    occupied: np.ndarray,
    banned: set[tuple[int, int]],
) -> tuple[int, int] | None:
    h, w = mask.shape
    rows = list(range(-min(area - 1, 2), panel_h - h - area + 1))
    cols = list(range(panel_w - w + 1))
    random.shuffle(rows)
    random.shuffle(cols)

    for sr in rows:
        input_top = sr + area
        if input_top < 0 or input_top + h > panel_h:
            continue
        if input_top + h <= panel_h // 2:
            continue
        for sc in cols:
            visible = _visible_mask(mask, sr, sc, panel_h, panel_w)
            if not np.any(visible):
                continue
            if np.any(visible & occupied):
                continue
            coords = set(map(tuple, np.argwhere(visible)))
            if coords & banned:
                continue
            return sr, sc
    return None


def _expected_from_spec(objects: list[dict], panel_h: int, panel_w: int) -> np.ndarray:
    layers = []
    areas = []
    for obj in objects:
        layers.append(
            _render_mask(
                obj["mask"],
                obj["color"],
                obj["shifted_top"],
                obj["left"],
                panel_h,
                panel_w,
            )
        )
        areas.append(int(obj["area"]))
    return _overlay_layers(layers, areas, _GEN_BG)


def _render_input(objects: list[dict], panel_h: int, panel_w: int) -> np.ndarray:
    full_h = panel_h + 2
    full_w = panel_w * 3 + 4
    grid = np.full((full_h, full_w), int(_GEN_SEPARATOR), dtype=int)

    r0 = 1
    for idx, obj in enumerate(objects):
        c0 = 1 + idx * (panel_w + 1)
        panel = np.full((panel_h, panel_w), int(_GEN_BG), dtype=int)
        patch = np.full(obj["mask"].shape, int(_GEN_BG), dtype=int)
        patch[obj["mask"]] = int(obj["color"])
        paste(panel, patch, int(obj["input_top"]), int(obj["left"]), transparent=_GEN_BG)
        grid[r0 : r0 + panel_h, c0 : c0 + panel_w] = panel

    return grid


def _diagnostic_conflict(expected_layers: list[np.ndarray], areas: list[int]) -> bool:
    coverage = np.zeros(expected_layers[0].shape, dtype=int)
    for layer in expected_layers:
        coverage += (layer != _GEN_BG).astype(int)
    conflict = coverage >= 2
    if not np.any(conflict):
        return False

    for r, c in np.argwhere(conflict):
        covering = [idx for idx, layer in enumerate(expected_layers) if int(layer[r, c]) != _GEN_BG]
        winner = max(covering, key=lambda idx: (areas[idx], -idx))
        if winner != min(covering):
            return True
    return False


def _valid_overlap_size(expected_layers: list[np.ndarray]) -> bool:
    coverage = np.zeros(expected_layers[0].shape, dtype=int)
    for layer in expected_layers:
        coverage += (layer != _GEN_BG).astype(int)
    overlap = int((coverage >= 2).sum())
    visible = int((coverage >= 1).sum())
    return 1 <= overlap <= max(4, visible // 3)


def _choose_areas() -> list[int]:
    for _ in range(50):
        areas = random.sample(_GEN_AREA_CHOICES, 3)
        if not (areas[0] > areas[1] > areas[2]):
            return areas
    raise RuntimeError("failed to choose a diagnostic area ordering")


def _choose_participants(areas: list[int]) -> tuple[int, ...]:
    pair_choices = [
        (i, j)
        for i in range(3)
        for j in range(i + 1, 3)
        if areas[j] > areas[i]
    ]
    triple_ok = int(np.argmax(areas)) != 0

    options: list[tuple[int, ...]] = pair_choices[:]
    if triple_ok:
        options.extend([(0, 1, 2)] * 2)
    assert options
    return random.choice(options)


def _sample_objects() -> tuple[np.ndarray, np.ndarray] | None:
    panel_h = random.randint(*_GEN_PANEL_H_RANGE)
    panel_w = random.randint(*_GEN_PANEL_W_RANGE)
    colors = random.sample(_GEN_COLORS, 3)
    areas = _choose_areas()
    participants = _choose_participants(areas)

    masks = []
    for area in areas:
        mask = _random_connected_mask(area)
        if mask is None or mask.shape[1] > panel_w or mask.shape[0] > panel_h - 2:
            return None
        masks.append(mask)

    use_clip = random.random() < _GEN_CLIP_PROB
    clip_candidates = [idx for idx in participants if masks[idx].shape[0] > 1]
    clip_idx = random.choice(clip_candidates) if use_clip and clip_candidates else None

    overlap_target: tuple[int, int] | None = None
    placements: dict[int, tuple[int, int]] = {}
    for _ in range(120):
        target_r = random.randint(0, min(2, panel_h - 1))
        target_c = random.randint(1, panel_w - 2)
        attempt: dict[int, tuple[int, int]] = {}
        ok = True
        for idx in participants:
            options = _feasible_target_placements(
                masks[idx],
                areas[idx],
                panel_h,
                panel_w,
                target_r,
                target_c,
                require_clip=(idx == clip_idx),
            )
            if not options:
                ok = False
                break
            attempt[idx] = random.choice(options)
        if not ok:
            continue

        layers = [
            _render_mask(masks[idx], colors[idx], attempt[idx][0], attempt[idx][1], panel_h, panel_w)
            for idx in participants
        ]
        coverage = sum((layer != _GEN_BG).astype(int) for layer in layers)
        if int(coverage[target_r, target_c]) != len(participants):
            continue
        if not _valid_overlap_size(layers):
            continue

        placements = attempt
        overlap_target = (target_r, target_c)
        break
    if overlap_target is None:
        return None

    occupied = np.zeros((panel_h, panel_w), dtype=bool)
    banned = {overlap_target}
    for idx in participants:
        sr, sc = placements[idx]
        visible = _visible_mask(masks[idx], sr, sc, panel_h, panel_w)
        occupied |= visible

    others = [idx for idx in range(3) if idx not in participants]
    for idx in others:
        placement = _random_shift(masks[idx], areas[idx], panel_h, panel_w, occupied, banned)
        if placement is None:
            return None
        placements[idx] = placement
        occupied |= _visible_mask(masks[idx], placement[0], placement[1], panel_h, panel_w)

    objects = []
    expected_layers = []
    for idx in range(3):
        sr, sc = placements[idx]
        input_top = sr + areas[idx]
        objects.append(
            {
                "color": int(colors[idx]),
                "area": int(areas[idx]),
                "mask": masks[idx],
                "shifted_top": int(sr),
                "input_top": int(input_top),
                "left": int(sc),
            }
        )
        expected_layers.append(_render_mask(masks[idx], colors[idx], sr, sc, panel_h, panel_w))

    expected = _expected_from_spec(objects, panel_h, panel_w)
    if set(map(int, np.unique(expected))) - {_GEN_BG} != set(colors):
        return None
    if not _diagnostic_conflict(expected_layers, areas):
        return None

    inp = _render_input(objects, panel_h, panel_w)
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_ATTEMPTS):
        sampled = _sample_objects()
        if sampled is None:
            continue
        inp, expected = sampled

        try:
            _separator, panels = _extract_panels(inp)
            panel_areas = []
            for panel in panels:
                _color, area, _obj = _single_object(panel)
                panel_areas.append(area)
            if len(set(panel_areas)) != 3:
                continue
            if not np.array_equal(main(inp), expected):
                continue
        except AssertionError:
            continue

        return inp
    raise RuntimeError("failed to generate a valid input for ARC new puzzle 4")
