from __future__ import annotations

import numpy as np


BACKGROUND = 0
RED = 2
SEPARATOR = 5
DISTRACTOR = 6
CYAN = 8

PANELS = ("UL", "UR", "LL", "LR")
LAYER_ORDER = ("LL", "UR", "UL", "LR")
WRONG_LAYER_ORDER_TOP = ("LL", "UR", "LR", "UL")
WRONG_LAYER_ORDER_MID = ("UR", "LL", "UL", "LR")

PANEL_SIZE_CHOICES = tuple(range(8, 15))
MAX_GENERATION_ATTEMPTS = 500
_RNG = np.random.default_rng()


def _find_components_of_color(grid: np.ndarray, color: int) -> list[dict]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    components = []

    for r in range(h):
        for c in range(w):
            if visited[r, c] or int(grid[r, c]) != color:
                continue

            stack = [(r, c)]
            visited[r, c] = True
            cells = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        stack.append((nr, nc))

            coords = np.array(cells, dtype=int)
            r0 = int(coords[:, 0].min())
            c0 = int(coords[:, 1].min())
            r1 = int(coords[:, 0].max())
            c1 = int(coords[:, 1].max())
            components.append(
                {
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "area": int(coords.shape[0]),
                }
            )

    return components


def _anchor_order(anchors: tuple[tuple[int, int], ...], tie_mode: str) -> list[tuple[int, int]]:
    assert len(set(anchors)) == len(anchors), "anchors within one panel must be unique"
    if tie_mode == "row_asc":
        return sorted(anchors, key=lambda rc: (rc[1], rc[0]))
    if tie_mode == "row_desc":
        return sorted(anchors, key=lambda rc: (rc[1], -rc[0]))
    raise ValueError(f"unsupported tie mode: {tie_mode}")


def _anchor_colors(
    anchors: tuple[tuple[int, int], ...],
    *,
    from_right: bool,
    tie_mode: str,
) -> dict[tuple[int, int], int]:
    ordered = _anchor_order(anchors, tie_mode)
    visit_order = ordered[::-1] if from_right else ordered
    colors = {}
    for rank, anchor in enumerate(visit_order):
        colors[anchor] = CYAN if rank % 2 == 0 else RED
    return colors


def _render_panel(
    size: int,
    anchors: tuple[tuple[int, int], ...],
    distractors: tuple[tuple[int, int], ...],
    *,
    from_right: bool = True,
    tie_mode: str = "row_asc",
    keep_distractors: bool = False,
    raw: bool = False,
    explicit_colors: dict[tuple[int, int], int] | None = None,
) -> np.ndarray:
    panel = np.zeros((size, size), dtype=int)
    if explicit_colors is None:
        if raw:
            colors = {anchor: RED for anchor in anchors}
        else:
            colors = _anchor_colors(anchors, from_right=from_right, tie_mode=tie_mode)
    else:
        colors = explicit_colors

    for r, c in anchors:
        panel[r : r + 2, c : c + 2] = int(colors[(r, c)])
    if keep_distractors:
        for r, c in distractors:
            panel[r, c] = DISTRACTOR
    return panel


def _overlay_panels(panels: dict[str, np.ndarray], order: tuple[str, ...]) -> np.ndarray:
    out = np.zeros_like(next(iter(panels.values())))
    for panel_id in order:
        panel = panels[panel_id]
        mask = panel != BACKGROUND
        out[mask] = panel[mask]
    return out


def _extract_panels(grid: np.ndarray) -> dict[str, np.ndarray]:
    sep_rows = np.flatnonzero(np.all(grid == SEPARATOR, axis=1))
    sep_cols = np.flatnonzero(np.all(grid == SEPARATOR, axis=0))
    assert len(sep_rows) == 1 and len(sep_cols) == 1, "expected exactly one separator row and one separator column"

    mid_r = int(sep_rows[0])
    mid_c = int(sep_cols[0])
    expected_sep = np.zeros_like(grid, dtype=bool)
    expected_sep[mid_r, :] = True
    expected_sep[:, mid_c] = True
    assert np.array_equal(grid == SEPARATOR, expected_sep), "grey pixels must form the full cross separator only"

    top_h = mid_r
    left_w = mid_c
    bottom_h = grid.shape[0] - mid_r - 1
    right_w = grid.shape[1] - mid_c - 1
    assert top_h == bottom_h and left_w == right_w, "separator must split the grid into equal panels"

    panels = {
        "UL": grid[:mid_r, :mid_c],
        "UR": grid[:mid_r, mid_c + 1 :],
        "LL": grid[mid_r + 1 :, :mid_c],
        "LR": grid[mid_r + 1 :, mid_c + 1 :],
    }
    shapes = {panel.shape for panel in panels.values()}
    assert len(shapes) == 1, "all panels must share the same shape"
    return panels


def _process_input_panel(panel: np.ndarray) -> np.ndarray:
    allowed = {BACKGROUND, RED, DISTRACTOR}
    assert set(map(int, np.unique(panel))).issubset(allowed), "unexpected panel colors"

    magenta_parts = _find_components_of_color(panel, DISTRACTOR)
    assert all(part["area"] == 1 for part in magenta_parts), "distractors must be isolated single pixels"

    blocks = _find_components_of_color(panel, RED)
    assert 3 <= len(blocks) <= 5, "each panel must contain 3-5 red blocks"

    anchors = []
    for block in blocks:
        r0, c0, r1, c1 = block["bbox"]
        assert block["area"] == 4, "red objects must be 2x2 blocks"
        assert r1 - r0 == 1 and c1 - c0 == 1, "red objects must be 2x2 blocks"
        assert np.all(panel[r0 : r1 + 1, c0 : c1 + 1] == RED), "red blocks must be solid"
        anchors.append((r0, c0))

    return _render_panel(panel.shape[0], tuple(anchors), (), from_right=True, tie_mode="row_asc")


def main(grid: np.ndarray) -> np.ndarray:
    """Filter each quadrant's red 2x2 blocks by local right-anchored parity, then overlay the four processed panels.

    Input structure:
    - A black grid is split by one full grey row and one full grey column into four equal panels.
    - Each panel contains 3-5 separated 2x2 red blocks and may also contain isolated magenta singleton distractors.
    - The red blocks stay fully inside their panel and do not touch the grey cross.

    Rule:
    - In each panel independently, erase magenta singleton distractors to background.
    - Sort that panel's red 2x2 blocks by the local top-left column from left to right, breaking ties by the local top-left row.
    - Starting from the rightmost block of that local order, recolor ranks 0, 2, 4, ... to cyan and leave ranks 1, 3, 5, ... red.
    - Remove the grey cross and align the four processed panels onto one shared canvas.
    - Transparent-overlay the aligned panels from bottom to top in the order lower-left, upper-right, upper-left, lower-right.

    Key insight:
    - The alternating recolor is recomputed separately inside each quadrant before overlay, so visible overlap colors depend on both local rank parity and the fixed quadrant layer order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert set(map(int, np.unique(g))).issubset({BACKGROUND, RED, SEPARATOR, DISTRACTOR}), "unexpected grid colors"

    raw_panels = _extract_panels(g)
    processed = {panel_id: _process_input_panel(panel) for panel_id, panel in raw_panels.items()}
    return _overlay_panels(processed, LAYER_ORDER)


def _candidate_anchors(size: int) -> list[tuple[int, int]]:
    starts = list(range(0, size - 1, 3))
    return [(int(r), int(c)) for r in starts for c in starts]


def _block_mask(size: int, anchors: tuple[tuple[int, int], ...]) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    for r, c in anchors:
        mask[r : r + 2, c : c + 2] = True
    return mask


def _expand_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    expanded = mask.copy()
    for r, c in np.argwhere(mask):
        expanded[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)] = True
    return expanded


def _pick_isolated_points(
    allowed: np.ndarray,
    count: int,
    used_points: set[tuple[int, int]],
) -> tuple[tuple[int, int], ...] | None:
    cells = [(int(r), int(c)) for r, c in np.argwhere(allowed)]
    _RNG.shuffle(cells)

    blocked = np.zeros_like(allowed, dtype=bool)
    for r, c in used_points:
        blocked[max(0, r - 1) : min(allowed.shape[0], r + 2), max(0, c - 1) : min(allowed.shape[1], c + 2)] = True

    chosen = []
    for r, c in cells:
        if blocked[r, c]:
            continue
        chosen.append((r, c))
        blocked[max(0, r - 1) : min(allowed.shape[0], r + 2), max(0, c - 1) : min(allowed.shape[1], c + 2)] = True
        if len(chosen) == count:
            return tuple(chosen)
    return None


def _panel_offsets(size: int) -> dict[str, tuple[int, int]]:
    return {
        "UL": (0, 0),
        "UR": (0, size + 1),
        "LL": (size + 1, 0),
        "LR": (size + 1, size + 1),
    }


def _render_output_from_specs(
    size: int,
    specs: dict[str, dict[str, tuple[tuple[int, int], ...]]],
    *,
    order: tuple[str, ...] = LAYER_ORDER,
    from_right: bool = True,
    tie_mode: str = "row_asc",
    keep_distractors: bool = False,
    raw: bool = False,
    explicit_colors: dict[str, dict[tuple[int, int], int]] | None = None,
) -> np.ndarray:
    panels = {}
    for panel_id in PANELS:
        panel_colors = None if explicit_colors is None else explicit_colors[panel_id]
        panels[panel_id] = _render_panel(
            size,
            specs[panel_id]["anchors"],
            specs[panel_id]["distractors"],
            from_right=from_right,
            tie_mode=tie_mode,
            keep_distractors=keep_distractors,
            raw=raw,
            explicit_colors=panel_colors,
        )
    return _overlay_panels(panels, order)


def _render_global_sort_output(size: int, specs: dict[str, dict[str, tuple[tuple[int, int], ...]]]) -> np.ndarray:
    instances = []
    offsets = _panel_offsets(size)
    for panel_id in PANELS:
        dr, dc = offsets[panel_id]
        for anchor in specs[panel_id]["anchors"]:
            instances.append((dc + anchor[1], dr + anchor[0], panel_id, anchor))

    ordered = sorted(instances, key=lambda item: (item[0], item[1]))
    color_maps = {panel_id: {} for panel_id in PANELS}
    for rank, (_gc, _gr, panel_id, anchor) in enumerate(ordered[::-1]):
        color_maps[panel_id][anchor] = CYAN if rank % 2 == 0 else RED

    return _render_output_from_specs(size, specs, explicit_colors=color_maps)


def _assemble_input(size: int, specs: dict[str, dict[str, tuple[tuple[int, int], ...]]]) -> np.ndarray:
    grid = np.full((2 * size + 1, 2 * size + 1), BACKGROUND, dtype=int)
    grid[size, :] = SEPARATOR
    grid[:, size] = SEPARATOR

    offsets = _panel_offsets(size)
    for panel_id, (r0, c0) in offsets.items():
        panel = np.zeros((size, size), dtype=int)
        for r, c in specs[panel_id]["anchors"]:
            panel[r : r + 2, c : c + 2] = RED
        for r, c in specs[panel_id]["distractors"]:
            panel[r, c] = DISTRACTOR
        grid[r0 : r0 + size, c0 : c0 + size] = panel

    return grid


def _semantic_instance_ok(
    size: int,
    specs: dict[str, dict[str, tuple[tuple[int, int], ...]]],
    expected: np.ndarray,
) -> bool:
    counts = [len(specs[panel_id]["anchors"]) for panel_id in PANELS]
    if len(set(counts)) < 2:
        return False

    color_maps = {
        panel_id: _anchor_colors(specs[panel_id]["anchors"], from_right=True, tie_mode="row_asc")
        for panel_id in PANELS
    }
    shared_top = set(specs["UL"]["anchors"]) & set(specs["LR"]["anchors"])
    shared_mid = set(specs["LL"]["anchors"]) & set(specs["UR"]["anchors"])
    if not shared_top or all(color_maps["UL"][anchor] == color_maps["LR"][anchor] for anchor in shared_top):
        return False
    if not shared_mid or all(color_maps["LL"][anchor] == color_maps["UR"][anchor] for anchor in shared_mid):
        return False

    wrong_outputs = (
        _render_output_from_specs(size, specs, raw=True),
        _render_output_from_specs(size, specs, from_right=False),
        _render_output_from_specs(size, specs, keep_distractors=True),
        _render_output_from_specs(size, specs, tie_mode="row_desc"),
        _render_output_from_specs(size, specs, order=WRONG_LAYER_ORDER_TOP),
        _render_output_from_specs(size, specs, order=WRONG_LAYER_ORDER_MID),
        _render_global_sort_output(size, specs),
    )
    if any(np.array_equal(expected, wrong) for wrong in wrong_outputs):
        return False

    own_masks = {panel_id: _block_mask(size, specs[panel_id]["anchors"]) for panel_id in PANELS}
    union_mask = np.zeros((size, size), dtype=bool)
    total_area = 0
    for mask in own_masks.values():
        union_mask |= mask
        total_area += int(mask.sum())

    overlap_area = total_area - int(union_mask.sum())
    density = np.count_nonzero(expected) / expected.size
    if overlap_area < 8 or overlap_area > total_area // 2:
        return False
    if not (0.12 <= density <= 0.5):
        return False
    return True


def _sample_specs(size: int) -> dict[str, dict[str, tuple[tuple[int, int], ...]]] | None:
    candidates = _candidate_anchors(size)
    rows = sorted({r for r, _ in candidates})
    cols = sorted({c for _, c in candidates})
    if len(rows) < 3 or len(candidates) < 9:
        return None

    forced = {panel_id: set() for panel_id in PANELS}

    tie_col = int(_RNG.choice(cols))
    tie_rows = [int(v) for v in _RNG.choice(rows, size=2, replace=False)]
    for row in tie_rows:
        forced["LR"].add((row, tie_col))

    remaining = [anchor for anchor in candidates if anchor not in forced["LR"]]
    _RNG.shuffle(remaining)
    if len(remaining) < 4:
        return None

    ll_lr, ul_lr, ll_ur, ul_ur = remaining[:4]
    forced["LL"].add(ll_lr)
    forced["LR"].add(ll_lr)
    forced["UL"].add(ul_lr)
    forced["LR"].add(ul_lr)
    forced["LL"].add(ll_ur)
    forced["UR"].add(ll_ur)
    forced["UL"].add(ul_ur)
    forced["UR"].add(ul_ur)

    counts = {
        "LR": int(_RNG.integers(max(4, len(forced["LR"])), 6)),
        "UL": int(_RNG.integers(max(3, len(forced["UL"])), 6)),
        "UR": int(_RNG.integers(max(3, len(forced["UR"])), 6)),
        "LL": int(_RNG.integers(max(3, len(forced["LL"])), 6)),
    }
    if len(set(counts.values())) < 2:
        counts["LL"] = 3 if counts["LL"] != 3 else 5

    specs = {}
    for panel_id in PANELS:
        chosen = set(forced[panel_id])
        needed = counts[panel_id] - len(chosen)
        if needed < 0:
            return None
        pool = [anchor for anchor in candidates if anchor not in chosen]
        _RNG.shuffle(pool)
        chosen.update(pool[:needed])
        anchors = tuple(sorted((int(r), int(c)) for r, c in chosen))
        specs[panel_id] = {"anchors": anchors, "distractors": ()}

    own_masks = {panel_id: _block_mask(size, specs[panel_id]["anchors"]) for panel_id in PANELS}
    expanded_masks = {panel_id: _expand_mask(mask) for panel_id, mask in own_masks.items()}
    union_mask = np.zeros((size, size), dtype=bool)
    for mask in own_masks.values():
        union_mask |= mask

    used_points: set[tuple[int, int]] = set()
    lr_count = int(_RNG.integers(1, 3))
    lr_points = _pick_isolated_points((~union_mask) & (~expanded_masks["LR"]), lr_count, used_points)
    if lr_points is None:
        return None
    specs["LR"]["distractors"] = lr_points
    used_points.update(lr_points)

    for panel_id in ("UL", "UR", "LL"):
        if _RNG.random() < 0.7:
            count = 1 + int(_RNG.random() < 0.35)
            points = _pick_isolated_points((~union_mask) & (~expanded_masks[panel_id]), count, used_points)
            if points is None:
                return None
            specs[panel_id]["distractors"] = points
            used_points.update(points)

    return specs


def _gen_attempt() -> np.ndarray | None:
    size = int(_RNG.choice(PANEL_SIZE_CHOICES))
    specs = _sample_specs(size)
    if specs is None:
        return None

    expected = _render_output_from_specs(size, specs)
    if not _semantic_instance_ok(size, specs, expected):
        return None

    inp = _assemble_input(size, specs)
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
