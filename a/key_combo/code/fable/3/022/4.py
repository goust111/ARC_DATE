from __future__ import annotations

import random

import numpy as np


NEIGHBORS8 = tuple(
    (dr, dc)
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
    if not (dr == 0 and dc == 0)
)
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
ARC_COLORS = tuple(range(10))


def _most_common_color(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    max_count = int(counts.max())
    assert int(np.count_nonzero(counts == max_count)) == 1, "background color must be uniquely dominant"
    return int(colors[int(np.argmax(counts))])


def _colors_present(grid: np.ndarray, *, exclude: set[int] | None = None) -> list[int]:
    excluded = set() if exclude is None else {int(c) for c in exclude}
    return [int(c) for c in np.unique(grid) if int(c) not in excluded]


def _split_panels(grid: np.ndarray) -> tuple[list[np.ndarray], int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height > 0 and width > 0, "expected a non-empty grid"

    bg = _most_common_color(g)
    solid_non_bg_cols = {
        c
        for c in range(width)
        if int(g[0, c]) != bg and bool(np.all(g[:, c] == int(g[0, c])))
    }

    valid: list[list[np.ndarray]] = []
    for panel_count in range(4, width + 1):
        payload_width = width - (panel_count - 1)
        if payload_width <= 0 or payload_width % panel_count != 0:
            continue
        panel_width = payload_width // panel_count
        if panel_width <= 0:
            continue

        sep_cols = [(i + 1) * panel_width + i for i in range(panel_count - 1)]
        if not all(c in solid_non_bg_cols for c in sep_cols):
            continue

        panels = []
        start = 0
        for i in range(panel_count):
            panels.append(g[:, start : start + panel_width])
            start += panel_width + (1 if i < panel_count - 1 else 0)

        panel_colors = []
        ok = True
        for panel in panels:
            colors = _colors_present(panel, exclude={bg})
            if len(colors) != 1:
                ok = False
                break
            if not np.any(panel != bg):
                ok = False
                break
            panel_colors.append(colors[0])
        if ok and len(set(panel_colors)) == len(panel_colors):
            valid.append(panels)

    assert len(valid) == 1, "expected a unique split into four or more equal panels"
    return valid[0], bg


def _touches8(mask: np.ndarray, canvas: np.ndarray) -> bool:
    height, width = mask.shape
    for r_raw, c_raw in np.argwhere(mask):
        r = int(r_raw)
        c = int(c_raw)
        for dr, dc in NEIGHBORS8:
            rr = r + dr
            cc = c + dc
            if 0 <= rr < height and 0 <= cc < width and bool(canvas[rr, cc]):
                return True
    return False


def _solve_panels(panels: list[np.ndarray], bg: int) -> tuple[np.ndarray, list[dict[str, bool]]]:
    out = panels[0].copy()
    canvas_mask = out != bg
    decisions: list[dict[str, bool]] = []

    for panel in panels[1:]:
        mask = panel != bg
        overlap = bool(np.any(canvas_mask & mask))
        touch = _touches8(mask, canvas_mask)
        accepted = (not overlap) and touch
        decisions.append({"overlap": overlap, "touch": touch, "accepted": accepted})
        if accepted:
            out[mask] = panel[mask]
            canvas_mask |= mask

    return out, decisions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Chain together left-to-right panel masks when each new mask fits and touches the growing canvas.

    Input structure:
    - The input is split by solid non-background separator columns into four or more equal-size panels.
    - All panels share one background color.
    - Each panel contains exactly one foreground color, and the panel foreground colors are mutually distinct.

    Rule:
    - Remove the separator columns and use the leftmost panel as the initial output canvas.
    - Process the remaining panels from left to right.
    - Accept a panel only if none of its foreground cells overlap the current canvas foreground and at least one of
      its foreground cells is 8-adjacent to the current canvas foreground.
    - Copy accepted foreground cells into the canvas; discard any panel that overlaps or fails to touch.

    Key insight:
    - Fitting is checked against the continually growing canvas, so earlier accepts or rejects change the fate of
      later panels and the sequence cannot be treated as an unordered overlay.
    """
    panels, bg = _split_panels(grid)
    out, _ = _solve_panels(panels, bg)
    return out


# --------------- generator ---------------


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _mask_from_cells(shape: tuple[int, int], cells: set[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r, c in cells:
        mask[int(r), int(c)] = True
    return mask


def _dilate8(mask: np.ndarray, *, include_self: bool = False) -> np.ndarray:
    out = mask.copy() if include_self else np.zeros_like(mask, dtype=bool)
    shape = mask.shape
    for r_raw, c_raw in np.argwhere(mask):
        r = int(r_raw)
        c = int(c_raw)
        for dr, dc in NEIGHBORS8:
            rr = r + dr
            cc = c + dc
            if _in_bounds(shape, rr, cc):
                out[rr, cc] = True
    return out


def _sample_blob(
    shape: tuple[int, int],
    allowed: np.ndarray,
    size_range: tuple[int, int],
    *,
    predicate=None,
    attempts: int = 120,
) -> np.ndarray | None:
    allowed_cells = [tuple(map(int, rc)) for rc in np.argwhere(allowed)]
    if not allowed_cells:
        return None

    lo, hi = map(int, size_range)
    hi = min(hi, len(allowed_cells))
    if lo > hi:
        return None

    allowed_set = set(allowed_cells)
    for _ in range(attempts):
        size = random.randint(lo, hi)
        seed = random.choice(allowed_cells)
        cells = {seed}
        frontier = {
            (seed[0] + dr, seed[1] + dc)
            for dr, dc in NEIGHBORS4
            if (seed[0] + dr, seed[1] + dc) in allowed_set
        }

        while len(cells) < size and frontier:
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            cells.add(cell)
            for dr, dc in NEIGHBORS4:
                nb = (cell[0] + dr, cell[1] + dc)
                if nb in allowed_set and nb not in cells:
                    frontier.add(nb)

        if len(cells) != size:
            continue
        mask = _mask_from_cells(shape, cells)
        if predicate is None or predicate(mask):
            return mask
    return None


def _sample_overlap_blob(shape: tuple[int, int], current: np.ndarray) -> np.ndarray | None:
    current_cells = [tuple(map(int, rc)) for rc in np.argwhere(current)]
    if not current_cells:
        return None
    seed = random.choice(current_cells)
    size = random.randint(1, 3)
    cells = {seed}
    frontier = {
        (seed[0] + dr, seed[1] + dc)
        for dr, dc in NEIGHBORS4
        if _in_bounds(shape, seed[0] + dr, seed[1] + dc)
    }
    while len(cells) < size and frontier:
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        cells.add(cell)
        for dr, dc in NEIGHBORS4:
            nb = (cell[0] + dr, cell[1] + dc)
            if _in_bounds(shape, *nb) and nb not in cells:
                frontier.add(nb)
    return _mask_from_cells(shape, cells)


def _build_masks(shape: tuple[int, int], panel_count: int) -> tuple[list[np.ndarray], list[str]] | None:
    height, width = shape
    inner = np.zeros(shape, dtype=bool)
    if height > 2 and width > 2:
        inner[1 : height - 1, 1 : width - 1] = True
    else:
        inner[:, :] = True

    seed = _sample_blob(shape, inner, (2, 4))
    if seed is None:
        return None

    a_allowed = ~seed
    accept_a = _sample_blob(
        shape,
        a_allowed,
        (1, 3),
        predicate=lambda m: (not np.any(m & seed)) and _touches8(m, seed),
    )
    if accept_a is None:
        return None

    seed_forbidden = _dilate8(seed, include_self=True)
    b_allowed = ~(seed | accept_a) & ~seed_forbidden
    accept_b = _sample_blob(
        shape,
        b_allowed,
        (1, 3),
        predicate=lambda m: _touches8(m, accept_a) and not _touches8(m, seed),
    )
    if accept_b is None:
        return None

    current = seed | accept_a | accept_b
    overlap_reject = _sample_overlap_blob(shape, current)
    if overlap_reject is None:
        return None

    safe_from_current = ~_dilate8(current, include_self=True)
    isolated_reject = _sample_blob(
        shape,
        safe_from_current,
        (1, 3),
        predicate=lambda m: (not np.any(m & current)) and not _touches8(m, current),
    )
    if isolated_reject is None:
        return None

    masks = [seed, accept_a, accept_b, overlap_reject, isolated_reject]
    roles = ["seed", "accept", "chain", "overlap", "isolated"]

    if panel_count == 6:
        ghost_allowed = safe_from_current & ~isolated_reject
        ghost_reject = _sample_blob(
            shape,
            ghost_allowed,
            (1, 2),
            predicate=lambda m: _touches8(m, isolated_reject) and not _touches8(m, current),
        )
        if ghost_reject is None:
            return None
        masks.append(ghost_reject)
        roles.append("ghost")

    return masks, roles


def _render_input(
    masks: list[np.ndarray], bg: int, separator: int, foregrounds: list[int]
) -> np.ndarray:
    panel_h, panel_w = masks[0].shape
    panel_count = len(masks)
    grid = np.full((panel_h, panel_count * panel_w + panel_count - 1), int(bg), dtype=int)
    for i, mask in enumerate(masks):
        c0 = i * (panel_w + 1)
        grid[:, c0 : c0 + panel_w][mask] = int(foregrounds[i])
        if i < panel_count - 1:
            grid[:, c0 + panel_w] = int(separator)
    return grid


def _semantic_case_ok(grid: np.ndarray, roles: list[str], foregrounds: list[int]) -> bool:
    try:
        panels, bg = _split_panels(grid)
        out, decisions = _solve_panels(panels, bg)
    except AssertionError:
        return False

    if len(panels) != len(roles) or len(decisions) != len(roles) - 1:
        return False
    if not np.array_equal(main(grid), out):
        return False
    if max(grid.shape) > 30 or max(out.shape) > 30:
        return False

    accepted_indices = {0}
    for panel_idx, decision in enumerate(decisions, start=1):
        if decision["accepted"]:
            accepted_indices.add(panel_idx)

    if roles[2] != "chain" or not decisions[1]["accepted"]:
        return False
    seed_mask = panels[0] != bg
    chain_mask = panels[2] != bg
    if _touches8(chain_mask, seed_mask):
        return False

    has_overlap_reject = any(d["overlap"] and not d["accepted"] for d in decisions)
    has_notouch_reject = any((not d["overlap"]) and (not d["touch"]) for d in decisions)
    if not has_overlap_reject or not has_notouch_reject:
        return False

    if "ghost" in roles:
        ghost_idx = roles.index("ghost")
        isolated_idx = roles.index("isolated")
        ghost_decision = decisions[ghost_idx - 1]
        if ghost_decision["accepted"] or ghost_decision["touch"]:
            return False
        if not _touches8(panels[ghost_idx] != bg, panels[isolated_idx] != bg):
            return False

    out_colors = set(_colors_present(out, exclude={bg}))
    expected_colors = {int(foregrounds[i]) for i in accepted_indices}
    return out_colors == expected_colors


def _gen_attempt() -> np.ndarray | None:
    panel_count = random.choice((5, 6))
    panel_width = random.choice((4, 5)) if panel_count == 5 else 4
    panel_height = random.randint(6, 9)
    shape = (panel_height, panel_width)

    built = _build_masks(shape, panel_count)
    if built is None:
        return None
    masks, roles = built

    palette = random.sample(ARC_COLORS, panel_count + 2)
    bg = palette[0]
    separator = palette[1]
    foregrounds = palette[2:]
    grid = _render_input(masks, bg, separator, foregrounds)

    if _semantic_case_ok(grid, roles, foregrounds):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid chained panel overlay input")
