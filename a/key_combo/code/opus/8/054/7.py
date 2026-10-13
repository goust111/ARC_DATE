from __future__ import annotations

import random

import numpy as np


GREY = 5
BLACK = 0
BASE_PALETTE = (1, 2, 3, 4, 6, 7)
MOTIF_PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_ATTEMPTS = 500


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _bands(mask: np.ndarray) -> list[tuple[int, int]]:
    idx = [int(i) for i in np.flatnonzero(mask)]
    bands: list[tuple[int, int]] = []
    if not idx:
        return bands
    start = prev = idx[0]
    for x in idx[1:]:
        if x == prev + 1:
            prev = x
        else:
            bands.append((start, prev))
            start = prev = x
    bands.append((start, prev))
    return bands


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected non-empty mask"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _separator_threshold(length: int) -> int:
    return max(1, int(np.floor(length * 0.6)))


def _parse_panels(grid: np.ndarray) -> dict:
    fg = grid != BLACK
    r0, c0, r1, c1 = _bbox(fg)
    box = grid[r0 : r1 + 1, c0 : c1 + 1]

    row_sep = np.count_nonzero(box == GREY, axis=1) >= _separator_threshold(box.shape[1])
    col_sep = np.count_nonzero(box == GREY, axis=0) >= _separator_threshold(box.shape[0])
    row_bands = _bands(row_sep)
    col_bands = _bands(col_sep)
    assert len(row_bands) == 3, f"expected 3 separator-row bands, got {row_bands}"
    assert len(col_bands) == 4, f"expected 4 separator-col bands, got {col_bands}"

    panel_rows = [(row_bands[i][1] + 1, row_bands[i + 1][0] - 1) for i in range(2)]
    panel_cols = [(col_bands[i][1] + 1, col_bands[i + 1][0] - 1) for i in range(3)]
    assert all(a <= b for a, b in panel_rows), "invalid panel row gaps"
    assert all(a <= b for a, b in panel_cols), "invalid panel col gaps"

    panels: list[list[tuple[int, int, int, int]]] = []
    for pr0, pr1 in panel_rows:
        row_panels = []
        for pc0, pc1 in panel_cols:
            row_panels.append((r0 + pr0, c0 + pc0, r0 + pr1, c0 + pc1))
        panels.append(row_panels)

    return {
        "bbox": (r0, c0, r1, c1),
        "row_bands": row_bands,
        "col_bands": col_bands,
        "panel_rows": panel_rows,
        "panel_cols": panel_cols,
        "panels": panels,
    }


def _locate_panel(panels: list[list[tuple[int, int, int, int]]], r: int, c: int) -> tuple[int, int] | None:
    for pr in range(2):
        for pc in range(3):
            r0, c0, r1, c1 = panels[pr][pc]
            if r0 <= r <= r1 and c0 <= c <= c1:
                return pr, pc
    return None


def _panel_states(
    grid: np.ndarray, panels: list[list[tuple[int, int, int, int]]]
) -> tuple[list[int], list[np.ndarray], list[int]]:
    bases: list[int] = []
    masks: list[np.ndarray] = []
    motif_colors: list[int] = []
    for pr in range(2):
        for pc in range(3):
            r0, c0, r1, c1 = panels[pr][pc]
            panel = grid[r0 : r1 + 1, c0 : c1 + 1]
            colors, counts = np.unique(panel, return_counts=True)
            assert len(colors) == 2, f"expected a two-color panel, got {colors.tolist()}"
            max_count = int(counts.max())
            assert int(np.sum(counts == max_count)) == 1, "panel base must be a unique majority color"
            base = int(colors[int(np.argmax(counts))])
            motif = int([int(color) for color in colors if int(color) != base][0])
            bases.append(base)
            masks.append(panel != base)
            motif_colors.append(motif)
    return bases, masks, motif_colors


def _find_protrusions(
    grid: np.ndarray, parsed: dict
) -> tuple[list[dict], set[int], set[int]]:
    r0, c0, r1, c1 = parsed["bbox"]
    panels = parsed["panels"]

    protrusions: list[dict] = []
    selected_rows: set[int] = set()
    selected_cols: set[int] = set()

    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            v = int(grid[r, c])
            if v in (BLACK, GREY):
                continue
            if _locate_panel(panels, r, c) is not None:
                continue

            same_neighbors = []
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and int(grid[nr, nc]) == v:
                    same_neighbors.append((nr, nc))
            assert len(same_neighbors) == 1, "protrusion must have exactly one same-color neighbor"

            nr, nc = same_neighbors[0]
            panel_rc = _locate_panel(panels, nr, nc)
            assert panel_rc is not None, "protrusion neighbor must point into a panel"
            panel_r, panel_c = panel_rc

            if nr != r:
                selected_rows.add(panel_r)
                axis = "row"
            else:
                selected_cols.add(panel_c)
                axis = "col"

            protrusions.append(
                {
                    "pixel": (r, c),
                    "neighbor": (nr, nc),
                    "panel": panel_rc,
                    "axis": axis,
                }
            )

    return protrusions, selected_rows, selected_cols


def _swap_states(
    bases: list[int],
    masks: list[np.ndarray],
    motif_colors: list[int],
    selected_rows: set[int],
    selected_cols: set[int],
    *,
    row_first: bool = False,
    use_rows: bool = True,
    use_cols: bool = True,
) -> tuple[list[np.ndarray], list[int]]:
    cur_masks = [mask.copy() for mask in masks]
    cur_colors = [int(color) for color in motif_colors]

    def idx(pr: int, pc: int) -> int:
        return pr * 3 + pc

    def do_col_round() -> None:
        for pc in sorted(selected_cols):
            top = idx(0, pc)
            bot = idx(1, pc)
            top_mask = cur_masks[top].copy()
            bot_mask = cur_masks[bot].copy()
            cur_masks[top] = bot_mask
            cur_masks[bot] = top_mask
            cur_colors[top] = int(bases[bot])
            cur_colors[bot] = int(bases[top])

    def do_row_round() -> None:
        for pr in sorted(selected_rows):
            left = idx(pr, 0)
            mid = idx(pr, 1)
            left_mask = cur_masks[left].copy()
            mid_mask = cur_masks[mid].copy()
            cur_masks[left] = mid_mask
            cur_masks[mid] = left_mask
            cur_colors[left] = int(bases[mid])
            cur_colors[mid] = int(bases[left])

    if row_first:
        if use_rows:
            do_row_round()
        if use_cols:
            do_col_round()
    else:
        if use_cols:
            do_col_round()
        if use_rows:
            do_row_round()

    return cur_masks, cur_colors


def _render(
    grid: np.ndarray,
    panels: list[list[tuple[int, int, int, int]]],
    bases: list[int],
    masks: list[np.ndarray],
    motif_colors: list[int],
) -> np.ndarray:
    out = grid.copy()
    flat_panels = [panels[pr][pc] for pr in range(2) for pc in range(3)]
    for idx, (r0, c0, r1, c1) in enumerate(flat_panels):
        block = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bases[idx]), dtype=int)
        block[masks[idx]] = int(motif_colors[idx])
        out[r0 : r1 + 1, c0 : c1 + 1] = block
    return out


def _solve_variant(
    grid: np.ndarray,
    *,
    row_first: bool = False,
    use_rows: bool = True,
    use_cols: bool = True,
) -> np.ndarray:
    parsed = _parse_panels(grid)
    panels = parsed["panels"]
    bases, masks, motif_colors = _panel_states(grid, panels)
    _protrusions, selected_rows, selected_cols = _find_protrusions(grid, parsed)
    new_masks, new_colors = _swap_states(
        bases,
        masks,
        motif_colors,
        selected_rows,
        selected_cols,
        row_first=row_first,
        use_rows=use_rows,
        use_cols=use_cols,
    )
    return _render(grid, panels, bases, new_masks, new_colors)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap panel motifs along leaf-selected panel columns and panel rows, with column swaps happening first.

    Input structure:
    - A black background surrounds one grey separator frame that partitions the non-black area into a 2x3 grid of rectangular panels.
    - Each panel has a unique dominant base color and one minority motif color forming an irregular mask.
    - Single motif-colored leaf pixels protrude from some panel edges into the grey separator area.

    Rule:
    - A protrusion whose only same-color neighbor lies above or below selects that panel's panel-row; a protrusion whose neighbor lies left or right selects that panel's panel-column.
    - First, every selected panel-column swaps the top and bottom motif masks; each received mask is recolored with the donating panel's base color after the receiver is reset to its own base color.
    - Second, every selected panel-row swaps its leftmost two panels in the same mask-only way, using the current masks produced by the first round.
    - Grey separators, black background, and the protrusion pixels themselves stay unchanged.

    Key insight:
    - The leaf direction does not name a pixel row or column inside one panel; it names a panel row or panel column, and the required output depends on applying the column-triggered exchanges before the row-triggered ones.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be 2D"
    return _solve_variant(g, row_first=False, use_rows=True, use_cols=True)


def _grow_mask(
    height: int,
    width: int,
    target: int,
    required: tuple[int, int] | None,
) -> np.ndarray | None:
    cells: set[tuple[int, int]] = set()
    if required is not None:
        cells.add(required)
    else:
        cells.add((random.randrange(height), random.randrange(width)))

    while len(cells) < target:
        frontier = set()
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        if not frontier:
            return None

        choices = list(frontier)
        random.shuffle(choices)
        pick = None
        for cand in choices:
            rr, cc = cand
            border_penalty = int(rr in (0, height - 1)) + int(cc in (0, width - 1))
            if required is None and border_penalty == 2:
                continue
            pick = cand
            break
        if pick is None:
            pick = choices[0]
        cells.add(pick)

    mask = np.zeros((height, width), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _panel_bounds(
    margin_r: int, margin_c: int, sep: int, panel_h: int, panel_w: int
) -> list[list[tuple[int, int, int, int]]]:
    panels: list[list[tuple[int, int, int, int]]] = []
    for pr in range(2):
        row_panels = []
        r0 = margin_r + sep + pr * (panel_h + sep)
        for pc in range(3):
            c0 = margin_c + sep + pc * (panel_w + sep)
            row_panels.append((r0, c0, r0 + panel_h - 1, c0 + panel_w - 1))
        panels.append(row_panels)
    return panels


def _choose_protrusions() -> list[dict]:
    selected_row = random.randrange(2)
    selected_col = random.randrange(2)

    col_panel_row = random.randrange(2)
    col_panel = (col_panel_row, selected_col)

    row_panel_choices = [(selected_row, c) for c in range(3) if (selected_row, c) != col_panel]
    row_panel = random.choice(row_panel_choices)

    protrusions = [
        {
            "panel": row_panel,
            "axis": "row",
            "side": random.choice(("left", "right")),
            "selected_row": selected_row,
        },
        {
            "panel": col_panel,
            "axis": "col",
            "side": random.choice(("top", "bottom")),
            "selected_col": selected_col,
        },
    ]

    if random.random() < 0.55:
        extra_by_type: dict[str, list[dict]] = {}
        other_row = 1 - selected_row
        extra_row_panels = [(other_row, c) for c in range(3) if (other_row, c) not in {row_panel, col_panel}]
        if extra_row_panels:
            extra_by_type["row"] = [
                {
                    "panel": random.choice(extra_row_panels),
                    "axis": "row",
                    "side": random.choice(("left", "right")),
                    "selected_row": other_row,
                }
            ]

        remaining_cols = [c for c in range(3) if c != selected_col]
        random.shuffle(remaining_cols)
        extra_col_options = []
        for col in remaining_cols:
            panel_choices = [(r, col) for r in range(2) if (r, col) not in {row_panel, col_panel}]
            if panel_choices:
                extra_col_options.append(
                    {
                        "panel": random.choice(panel_choices),
                        "axis": "col",
                        "side": random.choice(("top", "bottom")),
                        "selected_col": col,
                    }
                )
        if extra_col_options:
            extra_by_type["col"] = extra_col_options

        if extra_by_type:
            extra_type = random.choice(sorted(extra_by_type))
            protrusions.append(random.choice(extra_by_type[extra_type]))

    return protrusions


def _anchor_for_side(panel_h: int, panel_w: int, side: str) -> tuple[tuple[int, int], tuple[int, int]]:
    if side == "left":
        rr = random.randint(1, panel_h - 2)
        return (rr, 0), (0, -1)
    if side == "right":
        rr = random.randint(1, panel_h - 2)
        return (rr, panel_w - 1), (0, 1)
    if side == "top":
        cc = random.randint(1, panel_w - 2)
        return (0, cc), (-1, 0)
    if side == "bottom":
        cc = random.randint(1, panel_w - 2)
        return (panel_h - 1, cc), (1, 0)
    raise ValueError(side)


def _build_candidate() -> np.ndarray | None:
    sep = random.choice((1, 2))
    panel_h = random.randint(4, 6)
    panel_w = random.randint(4, 6)
    margin_r = random.randint(2, 4)
    margin_c = random.randint(2, 4)

    height = 2 * panel_h + 3 * sep + 2 * margin_r
    width = 3 * panel_w + 4 * sep + 2 * margin_c
    if height > 30 or width > 30:
        return None

    grid = np.zeros((height, width), dtype=int)
    box_r0 = margin_r
    box_c0 = margin_c
    box_r1 = height - margin_r - 1
    box_c1 = width - margin_c - 1
    grid[box_r0 : box_r1 + 1, box_c0 : box_c1 + 1] = GREY

    panels = _panel_bounds(margin_r, margin_c, sep, panel_h, panel_w)
    bases = random.sample(list(BASE_PALETTE), 6)
    protrusions = _choose_protrusions()

    protrusion_by_panel = {tuple(item["panel"]): item for item in protrusions}
    masks: list[np.ndarray] = []
    mask_keys: set[bytes] = set()
    used_anchors: set[tuple[int, int, int, int]] = set()

    for pr in range(2):
        for pc in range(3):
            panel_idx = pr * 3 + pc
            r0, c0, r1, c1 = panels[pr][pc]
            base = int(bases[panel_idx])
            motif_choices = [color for color in MOTIF_PALETTE if color != base]
            motif_color = int(random.choice(motif_choices))
            grid[r0 : r1 + 1, c0 : c1 + 1] = base

            required = None
            protrusion = protrusion_by_panel.get((pr, pc))
            if protrusion is not None:
                anchor, step = _anchor_for_side(panel_h, panel_w, protrusion["side"])
                if (pr, pc, anchor[0], anchor[1]) in used_anchors:
                    return None
                used_anchors.add((pr, pc, anchor[0], anchor[1]))
                required = anchor
                protrusion["anchor"] = anchor
                protrusion["step"] = step

            area = panel_h * panel_w
            lo = max(1, int(np.ceil(area * 0.2)))
            hi = max(lo, int(np.floor(area * 0.35)))
            target = random.randint(lo, hi)

            mask = None
            for _ in range(80):
                candidate = _grow_mask(panel_h, panel_w, target, required)
                if candidate is None:
                    continue
                key = candidate.tobytes()
                if key in mask_keys:
                    continue
                mask = candidate
                mask_keys.add(key)
                break
            if mask is None:
                return None

            masks.append(mask)
            grid[r0 : r1 + 1, c0 : c1 + 1][mask] = motif_color

            if protrusion is not None:
                ar, ac = protrusion["anchor"]
                dr, dc = protrusion["step"]
                leaf_r = r0 + ar + dr
                leaf_c = c0 + ac + dc
                if not (box_r0 <= leaf_r <= box_r1 and box_c0 <= leaf_c <= box_c1):
                    return None
                if grid[leaf_r, leaf_c] != GREY:
                    return None
                grid[leaf_r, leaf_c] = motif_color

    return grid


def _validate_grid(grid: np.ndarray) -> bool:
    parsed = _parse_panels(grid)
    panels = parsed["panels"]
    bases, masks, motif_colors = _panel_states(grid, panels)
    protrusions, selected_rows, selected_cols = _find_protrusions(grid, parsed)

    if len(protrusions) not in (2, 3):
        return False
    if len({tuple(item["panel"]) for item in protrusions}) != len(protrusions):
        return False
    if not selected_rows or not selected_cols:
        return False
    if not any(col in (0, 1) for col in selected_cols):
        return False
    if len(set(bases)) != 6:
        return False
    if set(bases) != set(BASE_PALETTE):
        return False
    if len({mask.tobytes() for mask in masks}) != 6:
        return False
    if any(len(np.unique(grid[r0 : r1 + 1, c0 : c1 + 1])) != 2 for row in panels for (r0, c0, r1, c1) in row):
        return False

    for idx, mask in enumerate(masks):
        frac = float(mask.mean())
        if not (0.18 <= frac <= 0.36):
            return False
        if motif_colors[idx] in (BLACK, GREY):
            return False

    correct = _solve_variant(grid, row_first=False, use_rows=True, use_cols=True)
    wrong_order = _solve_variant(grid, row_first=True, use_rows=True, use_cols=True)
    cols_only = _solve_variant(grid, row_first=False, use_rows=False, use_cols=True)
    rows_only = _solve_variant(grid, row_first=False, use_rows=True, use_cols=False)

    if np.array_equal(correct, grid):
        return False
    if np.array_equal(correct, wrong_order):
        return False
    if np.array_equal(correct, cols_only):
        return False
    if np.array_equal(correct, rows_only):
        return False

    double_swapped = {(pr, pc) for pr in selected_rows for pc in selected_cols if pc in (0, 1)}
    if not double_swapped:
        return False
    if not any(
        not np.array_equal(
            grid[panels[pr][pc][0] : panels[pr][pc][2] + 1, panels[pr][pc][1] : panels[pr][pc][3] + 1],
            correct[panels[pr][pc][0] : panels[pr][pc][2] + 1, panels[pr][pc][1] : panels[pr][pc][3] + 1],
        )
        for pr, pc in double_swapped
    ):
        return False

    if not any(
        np.array_equal(
            grid[panels[pr][pc][0] : panels[pr][pc][2] + 1, panels[pr][pc][1] : panels[pr][pc][3] + 1],
            correct[panels[pr][pc][0] : panels[pr][pc][2] + 1, panels[pr][pc][1] : panels[pr][pc][3] + 1],
        )
        for pr in range(2)
        for pc in range(3)
    ):
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _build_candidate()
        if grid is None:
            continue
        try:
            if _validate_grid(grid):
                return grid
        except Exception:
            continue
    raise RuntimeError("failed to generate a valid input for puzzle 7")
