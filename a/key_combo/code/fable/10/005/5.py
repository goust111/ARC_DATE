from __future__ import annotations

import numpy as np


BG = 0
ANCHOR = 9
PANEL_NAMES = ("TL", "TR", "BL", "BR")
FULL_ORDER = PANEL_NAMES
GEN_ATTEMPTS = 400


def _find_full_cross(grid: np.ndarray) -> tuple[int, int, int]:
    candidates: list[tuple[int, int, int]] = []
    for color in np.unique(grid):
        rows = np.where(np.all(grid == int(color), axis=1))[0]
        cols = np.where(np.all(grid == int(color), axis=0))[0]
        for row in rows:
            for col in cols:
                if row == 0 or col == 0 or row == grid.shape[0] - 1 or col == grid.shape[1] - 1:
                    continue
                if row * 2 + 1 != grid.shape[0] or col * 2 + 1 != grid.shape[1]:
                    continue
                candidates.append((int(color), int(row), int(col)))
    assert len(candidates) == 1, "expected exactly one full cross"
    return candidates[0]


def _split_panels(grid: np.ndarray, cross_r: int, cross_c: int) -> dict[str, np.ndarray]:
    return {
        "TL": grid[:cross_r, :cross_c],
        "TR": grid[:cross_r, cross_c + 1 :],
        "BL": grid[cross_r + 1 :, :cross_c],
        "BR": grid[cross_r + 1 :, cross_c + 1 :],
    }


def _extract_panel_info(panels: dict[str, np.ndarray]) -> dict[str, dict[str, object]]:
    common_colors = None
    for panel in panels.values():
        colors = set(map(int, np.unique(panel)))
        common_colors = colors if common_colors is None else common_colors & colors
    assert common_colors is not None
    assert BG in common_colors
    common_colors.discard(BG)
    assert len(common_colors) == 1, "expected one shared anchor color"
    anchor_color = common_colors.pop()

    info: dict[str, dict[str, object]] = {}
    for name, panel in panels.items():
        anchor_coords = np.argwhere(panel == anchor_color)
        assert anchor_coords.shape == (1, 2), "each panel must have one anchor"
        other_colors = set(map(int, np.unique(panel))) - {BG, anchor_color}
        assert len(other_colors) == 1, "each panel must have one foreground color"
        fg_color = other_colors.pop()
        fg_mask = panel == fg_color
        info[name] = {
            "anchor": (int(anchor_coords[0, 0]), int(anchor_coords[0, 1])),
            "fg_color": int(fg_color),
            "count": int(fg_mask.sum()),
            "mask": fg_mask,
        }
    counts = [int(item["count"]) for item in info.values()]
    assert len(set(counts)) == 4, "foreground counts must be pairwise distinct"
    return info


def _shift_mask(mask: np.ndarray, delta: tuple[int, int], shape: tuple[int, int]) -> np.ndarray:
    dr, dc = map(int, delta)
    rows, cols = np.argwhere(mask).T if np.any(mask) else (np.array([], dtype=int), np.array([], dtype=int))
    rr = rows + dr
    cc = cols + dc
    out = np.zeros(shape, dtype=bool)
    keep = (0 <= rr) & (rr < shape[0]) & (0 <= cc) & (cc < shape[1])
    out[rr[keep], cc[keep]] = True
    return out


def _render_layers(
    panel_info: dict[str, dict[str, object]],
    shape: tuple[int, int],
    order: tuple[str, ...],
    deltas: dict[str, tuple[int, int]],
) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for name in order:
        mask = _shift_mask(np.asarray(panel_info[name]["mask"], dtype=bool), deltas[name], shape)
        out[mask] = int(panel_info[name]["fg_color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Align the four cross-separated panels by their white anchors, then overlay them from densest to sparsest.

    Input structure:
    - One full row and one full column form a complete cross that splits the grid into four equal panels.
    - Every panel has a black background, one foreground color, and exactly one shared white anchor pixel.
    - The four foreground pixel counts are all different.

    Rule:
    - Remove the cross and use the panel with the most foreground pixels as the output canvas and anchor reference.
    - Translate every other panel so its white anchor lands on the densest panel's anchor; crop translated foreground pixels that leave the canvas.
    - Treat the anchor as transparent, then paint panel foregrounds in descending foreground-count order so later, sparser panels overwrite earlier denser ones.

    Key insight:
    - Panel position does not determine the stack order; the white anchor only defines translation, while density rank decides which color ends up on top.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    _cross_color, cross_r, cross_c = _find_full_cross(g)
    panels = _split_panels(g, cross_r, cross_c)
    shapes = {panel.shape for panel in panels.values()}
    assert len(shapes) == 1
    shape = next(iter(shapes))

    panel_info = _extract_panel_info(panels)
    base_name = max(FULL_ORDER, key=lambda name: int(panel_info[name]["count"]))
    base_anchor = panel_info[base_name]["anchor"]

    deltas = {
        name: (
            int(base_anchor[0]) - int(panel_info[name]["anchor"][0]),
            int(base_anchor[1]) - int(panel_info[name]["anchor"][1]),
        )
        for name in FULL_ORDER
    }
    order = tuple(sorted(FULL_ORDER, key=lambda name: int(panel_info[name]["count"]), reverse=True))
    return _render_layers(panel_info, shape, order, deltas)


def _neighbor_mask(mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(mask, dtype=bool)
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _weighted_pick(coords: np.ndarray, center: tuple[int, int]) -> tuple[int, int] | None:
    if len(coords) == 0:
        return None
    deltas = coords - np.asarray(center, dtype=float)
    dist2 = np.sum(deltas * deltas, axis=1)
    weights = 1.0 / (1.0 + dist2)
    weights = weights / weights.sum()
    idx = int(np.random.choice(len(coords), p=weights))
    return int(coords[idx, 0]), int(coords[idx, 1])


def _clip_zone(shape: tuple[int, int], delta: tuple[int, int]) -> np.ndarray:
    h, w = map(int, shape)
    dr, dc = map(int, delta)
    rows = np.arange(h)[:, None]
    cols = np.arange(w)[None, :]
    rr = rows + dr
    cc = cols + dc
    return (rr < 0) | (rr >= h) | (cc < 0) | (cc >= w)


def _valid_deltas(shape: tuple[int, int], base_anchor: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = map(int, shape)
    ar, ac = map(int, base_anchor)
    return [
        (dr, dc)
        for dr in range(ar - (h - 1), ar + 1)
        for dc in range(ac - (w - 1), ac + 1)
        if abs(dr) <= 3 and abs(dc) <= 3
    ]


def _pick_focus(shape: tuple[int, int], deltas: dict[str, tuple[int, int]], base_anchor: tuple[int, int]) -> tuple[int, int] | None:
    h, w = map(int, shape)
    r0 = max(0, max(int(delta[0]) for delta in deltas.values()))
    r1 = min(h - 1, min(int(delta[0]) + h - 1 for delta in deltas.values()))
    c0 = max(0, max(int(delta[1]) for delta in deltas.values()))
    c1 = min(w - 1, min(int(delta[1]) + w - 1 for delta in deltas.values()))
    coords = [
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if (r, c) != base_anchor
    ]
    if not coords:
        return None
    coords = sorted(coords, key=lambda rc: (abs(rc[0] - base_anchor[0]) + abs(rc[1] - base_anchor[1]), rc))
    return coords[int(np.random.randint(min(len(coords), 6)))]


def _grow_mask(
    shape: tuple[int, int],
    count: int,
    anchor: tuple[int, int],
    preferred: tuple[int, int],
    forced_cells: list[tuple[int, int]],
) -> np.ndarray | None:
    h, w = map(int, shape)
    mask = np.zeros((h, w), dtype=bool)
    forbidden = np.zeros((h, w), dtype=bool)
    forbidden[anchor] = True

    for cell in forced_cells:
        if forbidden[cell] or mask[cell]:
            return None
        mask[cell] = True
    if int(mask.sum()) > count:
        return None

    while int(mask.sum()) < count:
        frontier = _neighbor_mask(mask) & ~mask & ~forbidden
        use_frontier = frontier.any() and (np.random.rand() < 0.88 or not np.any(mask))
        coords = np.argwhere(frontier if use_frontier else (~mask & ~forbidden))
        picked = _weighted_pick(coords, preferred)
        if picked is None:
            return None
        mask[picked] = True
    return mask


def _sample_counts(n_cells: int, want_visible_mismatch: bool) -> list[int] | None:
    low = max(4, int(np.ceil(0.16 * n_cells)))
    high = min(n_cells - 2, max(low + 5, int(np.floor(0.44 * n_cells))))
    if high - low + 1 < 4:
        return None
    if want_visible_mismatch:
        window = min(7, high - low + 1)
        start = int(np.random.randint(low, high - window + 2))
        pool = np.arange(start, start + window)
    else:
        pool = np.arange(low, high + 1)
    picks = np.random.choice(pool, size=4, replace=False)
    return sorted([int(v) for v in picks], reverse=True)


def _choose_clip_cells(
    shape: tuple[int, int],
    delta: tuple[int, int],
    forbidden: set[tuple[int, int]],
    loss_target: int,
) -> list[tuple[int, int]] | None:
    if loss_target <= 0:
        return []
    clip_mask = _clip_zone(shape, delta)
    coords = [tuple(map(int, cell)) for cell in np.argwhere(clip_mask)]
    coords = [cell for cell in coords if cell not in forbidden]
    if len(coords) < loss_target:
        return None
    chosen_idx = np.random.choice(len(coords), size=loss_target, replace=False)
    return [coords[int(i)] for i in chosen_idx]


def _assemble_input(
    shape: tuple[int, int],
    cross_color: int,
    panel_records: dict[str, dict[str, object]],
) -> np.ndarray:
    h, w = map(int, shape)
    grid = np.full((2 * h + 1, 2 * w + 1), cross_color, dtype=int)
    for name, (r0, c0) in {
        "TL": (0, 0),
        "TR": (0, w + 1),
        "BL": (h + 1, 0),
        "BR": (h + 1, w + 1),
    }.items():
        grid[r0 : r0 + h, c0 : c0 + w] = panel_records[name]["panel"]
    return grid


def _panel_domains(
    shape: tuple[int, int],
    panel_records: dict[str, dict[str, object]],
    deltas: dict[str, tuple[int, int]],
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    shifted_masks: dict[str, np.ndarray] = {}
    visible_counts: dict[str, int] = {}
    for name in FULL_ORDER:
        shifted = _shift_mask(np.asarray(panel_records[name]["mask"], dtype=bool), deltas[name], shape)
        shifted_masks[name] = shifted
        visible_counts[name] = int(shifted.sum())
    return shifted_masks, visible_counts


def _validate_attempt(
    grid: np.ndarray,
    shape: tuple[int, int],
    panel_records: dict[str, dict[str, object]],
    local_counts: dict[str, int],
    anchors: dict[str, tuple[int, int]],
    deltas: dict[str, tuple[int, int]],
    base_name: str,
    focus: tuple[int, int],
    want_clip: bool,
    want_visible_mismatch: bool,
) -> bool:
    correct_order = tuple(sorted(FULL_ORDER, key=lambda name: local_counts[name], reverse=True))
    expected = _render_layers(panel_records, shape, correct_order, deltas)
    try:
        actual = main(grid)
    except AssertionError:
        return False
    if not np.array_equal(actual, expected):
        return False
    if np.any(actual == ANCHOR):
        return False

    shifted_masks, visible_counts = _panel_domains(shape, panel_records, deltas)
    coverage = np.zeros(shape, dtype=int)
    clipped_panels = 0
    for name in FULL_ORDER:
        coverage += shifted_masks[name].astype(int)
        if visible_counts[name] < local_counts[name]:
            clipped_panels += 1
    if coverage[focus] != 4:
        return False
    if int(coverage.max()) < 2:
        return False
    if not any(deltas[name] != (0, 0) for name in FULL_ORDER if name != base_name):
        return False
    if want_clip and clipped_panels == 0:
        return False
    if len({int(local_counts[name]) for name in FULL_ORDER}) != 4:
        return False
    if any(not np.any(actual == int(panel_records[name]["fg_color"])) for name in FULL_ORDER):
        return False

    no_translate = _render_layers(panel_records, shape, correct_order, {name: (0, 0) for name in FULL_ORDER})
    if np.array_equal(actual, no_translate):
        return False

    for wrong_order in (
        ("TL", "TR", "BL", "BR"),
        ("BL", "TR", "TL", "BR"),
        tuple(reversed(correct_order)),
    ):
        if wrong_order == correct_order:
            continue
        if np.array_equal(actual, _render_layers(panel_records, shape, wrong_order, deltas)):
            return False

    if want_visible_mismatch:
        if len(set(visible_counts.values())) != 4:
            return False
        visible_order = tuple(sorted(FULL_ORDER, key=lambda name: visible_counts[name], reverse=True))
        if visible_order == correct_order:
            return False
        if np.array_equal(actual, _render_layers(panel_records, shape, visible_order, deltas)):
            return False

    if clipped_panels > 0:
        for alt_name in FULL_ORDER:
            if alt_name == base_name:
                continue
            alt_deltas = {
                name: (
                    int(anchors[alt_name][0]) - int(anchors[name][0]),
                    int(anchors[alt_name][1]) - int(anchors[name][1]),
                )
                for name in FULL_ORDER
            }
            if np.array_equal(actual, _render_layers(panel_records, shape, correct_order, alt_deltas)):
                return False
    return True


def _generate_attempt() -> np.ndarray | None:
    panel_h = int(np.random.randint(5, 9))
    panel_w = int(np.random.randint(5, 9))
    shape = (panel_h, panel_w)
    n_cells = panel_h * panel_w

    want_visible_mismatch = bool(np.random.rand() < 0.5)
    count_values = _sample_counts(n_cells, want_visible_mismatch)
    if count_values is None:
        return None
    shuffled_counts = count_values.copy()
    np.random.shuffle(shuffled_counts)
    local_counts = {name: int(shuffled_counts[i]) for i, name in enumerate(FULL_ORDER)}
    base_name = max(FULL_ORDER, key=lambda name: local_counts[name])

    color_pool = np.arange(1, 9)
    picked = [int(v) for v in np.random.choice(color_pool, size=5, replace=False)]
    cross_color = picked[0]
    fg_colors = {name: picked[i + 1] for i, name in enumerate(FULL_ORDER)}

    margin_r = 2 if panel_h >= 6 else 1
    margin_c = 2 if panel_w >= 6 else 1
    base_anchor = (
        int(np.random.randint(margin_r, panel_h - margin_r)),
        int(np.random.randint(margin_c, panel_w - margin_c)),
    )

    candidates = _valid_deltas(shape, base_anchor)
    nonzero = [delta for delta in candidates if delta != (0, 0)]
    if not nonzero:
        return None

    deltas = {base_name: (0, 0)}
    want_clip = bool(want_visible_mismatch or np.random.rand() < 0.55)
    clip_name = None
    clip_loss_target = 0
    if want_clip:
        clip_name = str(np.random.choice([name for name in FULL_ORDER if name != base_name]))
        clip_loss_target = int(np.random.randint(2, 5)) if want_visible_mismatch else int(np.random.randint(1, 4))

    used_nonzero = False
    for name in FULL_ORDER:
        if name == base_name:
            continue
        options = nonzero if (name == clip_name or np.random.rand() < 0.8) else candidates
        if name == clip_name:
            options = [
                delta
                for delta in options
                if delta != (0, 0)
                and int(_clip_zone(shape, delta).sum()) >= clip_loss_target
            ]
        if not options:
            return None
        delta = options[int(np.random.randint(len(options)))]
        deltas[name] = delta
        used_nonzero |= delta != (0, 0)
    if not used_nonzero:
        return None

    focus = _pick_focus(shape, deltas, base_anchor)
    if focus is None:
        return None

    anchors = {name: (base_anchor[0] - deltas[name][0], base_anchor[1] - deltas[name][1]) for name in FULL_ORDER}
    panel_records: dict[str, dict[str, object]] = {}
    for name in FULL_ORDER:
        anchor = anchors[name]
        local_focus = (focus[0] - deltas[name][0], focus[1] - deltas[name][1])
        if local_focus == anchor:
            return None
        forbidden = {local_focus}
        clip_cells = _choose_clip_cells(shape, deltas[name], forbidden | {anchor}, clip_loss_target if name == clip_name else 0)
        if clip_cells is None:
            return None
        forced_cells = [local_focus, *clip_cells]
        preferred = (
            int(np.clip(local_focus[0] + np.random.randint(-1, 2), 0, panel_h - 1)),
            int(np.clip(local_focus[1] + np.random.randint(-1, 2), 0, panel_w - 1)),
        )
        mask = _grow_mask(shape, local_counts[name], anchor, preferred, forced_cells)
        if mask is None or mask[anchor]:
            return None

        panel = np.full(shape, BG, dtype=int)
        panel[mask] = fg_colors[name]
        panel[anchor] = ANCHOR
        panel_records[name] = {
            "panel": panel,
            "mask": mask,
            "anchor": anchor,
            "fg_color": fg_colors[name],
        }

    grid = _assemble_input(shape, cross_color, panel_records)
    if not _validate_attempt(
        grid,
        shape,
        panel_records,
        local_counts,
        anchors,
        deltas,
        base_name,
        focus,
        want_clip,
        want_visible_mismatch,
    ):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
