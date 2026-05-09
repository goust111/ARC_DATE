# Combo variant of 7b80bb43 (index 8)
# combo_spec: D8/2+D9/2
# Name: 全连通边框锚定网络

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask, draw_line, get_bg


# Design analysis for the fused variant:
# a) Concrete behavior:
#    Small example: a horizontal fg spine runs from the left border marker to the
#    right border marker, with one top marker and two bottom markers attached by
#    vertical branches. A 3-cell gap inside the spine is accompanied by a 2-pixel
#    diagonal staircase. In the original task the solver could treat each broken
#    segment locally. Here the clean target is one border-to-border network, so
#    after removing diagonal-only hints the solver fills only those short gaps
#    that merge disconnected anchored pieces, then prunes any leaf that is not
#    attached to a control-colored border marker.
# b) Complexity calibration:
#    The generator uses 5-6 anchors and 3-4 internal branches, so the visual
#    change is obvious: samples now have a dedicated border control color and
#    the repaired fg always becomes a single connected pipe-like structure.
#    The topology family is still narrow enough that 2-4 examples support the
#    inference.
# c) Broken original assumptions:
#    - "all non-bg pixels share one color" is false: we now have fg + ctrl.
#    - "independent components are acceptable" is false: valid output must be
#      one 4-connected fg component.
#    - "leaf pruning only matters near former diagonal hints" is too weak:
#      because every true endpoint is marker-anchored, any remaining non-anchor
#      fg leaf is globally invalid and can be pruned.
# d) Placement legality:
#    The generator keeps ctrl on the border only, keeps fg off the border,
#    spaces branch coordinates apart, and cuts gaps away from anchors/junctions.
#    Diagonal hints and distractors are validated to avoid overlap and orthogonal
#    contact with the network.
# e) Visual observability:
#    Humans can see one special border color, a broken interior network, and the
#    final repaired border-to-border structure. The role of ctrl is spatial, not
#    numeric: it marks the only legal endpoints.
# f) Visual cleanliness:
#    The topology is a spine with spaced branches, so markers, hints, and gaps do
#    not stack on top of each other. Distractors are placed in open background.
# g) New solver assumptions guaranteed by the generator:
#    - ctrl is the only non-bg color on the border.
#    - every true fg endpoint is orthogonally adjacent to ctrl.
#    - every intended gap is a 1-3 cell run between two fg pieces with diagonal
#      hints nearby, and filling it reduces the number of fg components.


GRID_H_RANGE = (18, 28)
GRID_W_RANGE = (18, 30)
N_COLORS = 10

GAP_SIZE = 3
CHAIN_LENGTH = 2
MAX_GAP = 3
BRANCH_GAP = 3
GAP_END_INSET = 2
GAP_BLOCK_RADIUS = 1
DISTRACTOR_LENGTHS = (3, 2)
GEN_RETRIES = 400


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            out.append((r + dr, c + dc))
    return out


def _deg4(mask: np.ndarray, r: int, c: int) -> int:
    h, w = mask.shape
    d = 0
    for rr, cc in _neighbors4(int(r), int(c)):
        if 0 <= rr < h and 0 <= cc < w and bool(mask[rr, cc]):
            d += 1
    return int(d)


def _has_diag(mask: np.ndarray, r: int, c: int) -> bool:
    h, w = mask.shape
    for rr, cc in ((r - 1, c - 1), (r - 1, c + 1), (r + 1, c - 1), (r + 1, c + 1)):
        if 0 <= rr < h and 0 <= cc < w and bool(mask[rr, cc]):
            return True
    return False


def _label4(mask: np.ndarray) -> tuple[np.ndarray, int]:
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=int)
    comp_id = 0
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or labels[r, c] != 0:
                continue
            comp_id += 1
            stack = [(int(r), int(c))]
            labels[r, c] = int(comp_id)
            while stack:
                rr, cc = stack.pop()
                for nr, nc in _neighbors4(int(rr), int(cc)):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and labels[nr, nc] == 0:
                        labels[nr, nc] = int(comp_id)
                        stack.append((int(nr), int(nc)))
    return labels, int(comp_id)


def _contiguous_runs(values: np.ndarray) -> list[tuple[int, int]]:
    if values.size == 0:
        return []
    start = prev = int(values[0])
    runs: list[tuple[int, int]] = []
    for raw in values[1:]:
        v = int(raw)
        if v == prev + 1:
            prev = v
            continue
        runs.append((int(start), int(prev)))
        start = prev = v
    runs.append((int(start), int(prev)))
    return runs


def _anchor_adjacent_mask(ctrl_mask: np.ndarray) -> np.ndarray:
    h, w = ctrl_mask.shape
    out = np.zeros((h, w), dtype=bool)
    for r, c in np.argwhere(ctrl_mask):
        r = int(r)
        c = int(c)
        for rr, cc in _neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc < w:
                out[rr, cc] = True
    return out


def _anchored_labels(labels: np.ndarray, anchor_adj: np.ndarray) -> set[int]:
    ids = {int(labels[r, c]) for r, c in np.argwhere((labels > 0) & anchor_adj)}
    ids.discard(0)
    return ids


def _candidate_gaps(mask: np.ndarray, diag: np.ndarray, max_gap: int) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    candidates: list[list[tuple[int, int]]] = []
    seen: set[tuple[tuple[int, int], ...]] = set()

    for r in range(h):
        runs = _contiguous_runs(np.where(mask[r, :])[0])
        for (_a1, b1), (a2, _b2) in zip(runs, runs[1:]):
            gap0 = int(b1 + 1)
            gap1 = int(a2 - 1)
            gap_len = int(gap1 - gap0 + 1)
            if gap_len < 1 or gap_len > int(max_gap):
                continue
            rr0 = max(0, int(r - 1))
            rr1 = min(int(h - 1), int(r + 1))
            cc0 = max(0, int(gap0 - 1))
            cc1 = min(int(w - 1), int(gap1 + 1))
            if not bool(diag[rr0 : rr1 + 1, cc0 : cc1 + 1].any()):
                continue
            cells = tuple((int(r), int(c)) for c in range(gap0, gap1 + 1))
            if cells not in seen:
                seen.add(cells)
                candidates.append(list(cells))

    for c in range(w):
        runs = _contiguous_runs(np.where(mask[:, c])[0])
        for (_a1, b1), (a2, _b2) in zip(runs, runs[1:]):
            gap0 = int(b1 + 1)
            gap1 = int(a2 - 1)
            gap_len = int(gap1 - gap0 + 1)
            if gap_len < 1 or gap_len > int(max_gap):
                continue
            rr0 = max(0, int(gap0 - 1))
            rr1 = min(int(h - 1), int(gap1 + 1))
            cc0 = max(0, int(c - 1))
            cc1 = min(int(w - 1), int(c + 1))
            if not bool(diag[rr0 : rr1 + 1, cc0 : cc1 + 1].any()):
                continue
            cells = tuple((int(r), int(c)) for r in range(gap0, gap1 + 1))
            if cells not in seen:
                seen.add(cells)
                candidates.append(list(cells))

    return candidates


def _fill_connectivity_gaps(mask: np.ndarray, diag: np.ndarray, anchor_adj: np.ndarray, max_gap: int) -> np.ndarray:
    m = mask.copy()
    while True:
        labels, comp_count = _label4(m)
        old_anchor_count = len(_anchored_labels(labels, anchor_adj))
        best_cells: list[tuple[int, int]] | None = None
        best_score: tuple[int, int, int, int] | None = None

        for cells in _candidate_gaps(m, diag, max_gap):
            neighbor_ids: set[int] = set()
            for r, c in cells:
                for rr, cc in _neighbors4(int(r), int(c)):
                    if 0 <= rr < m.shape[0] and 0 <= cc < m.shape[1] and labels[rr, cc] > 0:
                        neighbor_ids.add(int(labels[rr, cc]))
            if len(neighbor_ids) < 2:
                continue

            trial = m.copy()
            for r, c in cells:
                trial[int(r), int(c)] = True
            trial_labels, trial_comp_count = _label4(trial)
            trial_anchor_count = len(_anchored_labels(trial_labels, anchor_adj))
            score = (
                int(old_anchor_count - trial_anchor_count),
                int(comp_count - trial_comp_count),
                int(len(neighbor_ids)),
                -int(len(cells)),
            )
            if score[:2] <= (0, 0):
                continue
            if best_score is None or score > best_score:
                best_score = score
                best_cells = cells

        if best_cells is None:
            break
        for r, c in best_cells:
            m[int(r), int(c)] = True
    return m


def _prune_non_anchor_leaves(mask: np.ndarray, anchor_adj: np.ndarray) -> np.ndarray:
    m = mask.copy()
    while True:
        to_clear: list[tuple[int, int]] = []
        for r, c in np.argwhere(m):
            r = int(r)
            c = int(c)
            if bool(anchor_adj[r, c]):
                continue
            if _deg4(m, r, c) == 1:
                to_clear.append((r, c))
        if not to_clear:
            break
        for r, c in to_clear:
            m[int(r), int(c)] = False
    return m


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair a broken border-anchored line network by using diagonal hints and control markers.

    Input structure:
    - bg is the background, fg is the network color, and ctrl is a distinct border-only color.
    - ctrl pixels mark the only legal endpoints; the intended clean fg network is a single 4-connected component.
    - Gaps and diagonal staircase hints appear only inside the network, while extra diagonal distractors may float in open background.

    Rule:
    - Separate ctrl from fg, then remove diagonal-only fg pixels.
    - Fill short horizontal or vertical gaps near those removed hints, but only when the fill improves global fg connectivity between anchored pieces.
    - Keep ctrl fixed and iteratively prune any remaining fg leaf that is not adjacent to a ctrl marker.

    Key insight: the diagonal pixels still indicate where a straight segment was interrupted, but the correct repairs are the ones that restore one globally connected network whose only surviving leaves are the border-marked anchors.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    h, w = g.shape

    colors = {int(v) for v in np.unique(g)}
    colors.discard(int(bg))
    assert len(colors) == 2

    border_colors = {int(v) for v in np.concatenate((g[0, :], g[-1, :], g[:, 0], g[:, -1])) if int(v) != int(bg)}
    assert len(border_colors) == 1
    ctrl = int(next(iter(border_colors)))
    fg = int(next(iter(colors - {ctrl})))

    ctrl_mask = g == int(ctrl)
    fg_mask = g == int(fg)

    diag = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if bool(fg_mask[r, c]) and _deg4(fg_mask, r, c) == 0 and _has_diag(fg_mask, r, c):
                diag[r, c] = True

    cleaned = fg_mask & (~diag)
    anchor_adj = _anchor_adjacent_mask(ctrl_mask)
    repaired = _fill_connectivity_gaps(cleaned, diag, anchor_adj, MAX_GAP)
    pruned = _prune_non_anchor_leaves(repaired, anchor_adj)

    out = np.full((h, w), int(bg), dtype=int)
    out[ctrl_mask] = int(ctrl)
    out[pruned] = int(fg)
    return out


def _pick_distinct(rng: random.Random, values: list[int], k: int, gap: int) -> list[int]:
    vals = values[:]
    rng.shuffle(vals)
    out: list[int] = []
    for v in vals:
        if all(abs(int(v) - int(u)) >= int(gap) for u in out):
            out.append(int(v))
            if len(out) == k:
                return sorted(out)
    return []


def _h_chain_coords(row: int, gap_start: int, anchor: str, dr: int) -> list[tuple[int, int]]:
    if anchor == "left":
        root_c = int(gap_start - 1)
        return [(int(row + dr * i), int(root_c + i)) for i in range(1, CHAIN_LENGTH + 1)]
    root_c = int(gap_start + GAP_SIZE)
    return [(int(row + dr * i), int(root_c - i)) for i in range(1, CHAIN_LENGTH + 1)]


def _v_chain_coords(col: int, gap_start: int, anchor: str, dc: int) -> list[tuple[int, int]]:
    if anchor == "top":
        root_r = int(gap_start - 1)
        return [(int(root_r + i), int(col + dc * i)) for i in range(1, CHAIN_LENGTH + 1)]
    root_r = int(gap_start + GAP_SIZE)
    return [(int(root_r - i), int(col + dc * i)) for i in range(1, CHAIN_LENGTH + 1)]


def _validate_diag_chain(
    grid: np.ndarray,
    bg: int,
    fg: int,
    ctrl: int,
    coords: list[tuple[int, int]],
    forbidden: set[tuple[int, int]],
) -> bool:
    h, w = grid.shape
    coord_set = set(coords)
    for r, c in coords:
        if not (1 <= int(r) < h - 1 and 1 <= int(c) < w - 1):
            return False
        if (int(r), int(c)) in forbidden:
            return False
        if int(grid[int(r), int(c)]) != int(bg):
            return False
        for rr, cc in _neighbors4(int(r), int(c)):
            if (rr, cc) in coord_set:
                return False
            if 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) in {int(fg), int(ctrl)}:
                return False
        if not any(
            ((rr, cc) in coord_set) or (0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) == int(fg))
            for rr, cc in _neighbors8(int(r), int(c))
            if abs(rr - int(r)) == 1 and abs(cc - int(c)) == 1
        ):
            return False
    return True


def _reserve_neighborhood(forbidden: set[tuple[int, int]], coords: list[tuple[int, int]]) -> None:
    for r, c in coords:
        for rr, cc in _neighbors8(int(r), int(c)) + [(int(r), int(c))]:
            forbidden.add((int(rr), int(cc)))


def _build_output(
    rng: random.Random, h: int, w: int, bg: int, fg: int, ctrl: int
) -> tuple[np.ndarray, list[dict], list[dict]] | None:
    out = np.full((h, w), int(bg), dtype=int)
    h_segments: list[dict] = []
    v_segments: list[dict] = []

    horizontal = bool(rng.choice([0, 1]))
    branch_count = int(rng.choice([3, 4]))

    if horizontal:
        if h < 15 or w < 18:
            return None
        spine_row = rng.randint(6, h - 7)
        branch_cols = _pick_distinct(rng, list(range(3, w - 3)), branch_count, BRANCH_GAP)
        if len(branch_cols) != branch_count:
            return None
        top_count = rng.randint(1, branch_count - 1)
        rng.shuffle(branch_cols)
        top_cols = sorted(branch_cols[:top_count])
        bottom_cols = sorted(branch_cols[top_count:])
        if not bottom_cols:
            return None

        out[spine_row, 1 : w - 1] = int(fg)
        out[spine_row, 0] = int(ctrl)
        out[spine_row, w - 1] = int(ctrl)
        h_segments.append({"row": int(spine_row), "c0": 1, "c1": w - 2, "blocked": set(branch_cols)})

        for col in top_cols:
            draw_line(out, 1, int(col), end_r=spine_row, end_c=int(col), color=int(fg))
            out[0, int(col)] = int(ctrl)
            v_segments.append({"col": int(col), "r0": 1, "r1": int(spine_row), "blocked": {int(spine_row)}})
        for col in bottom_cols:
            draw_line(out, int(spine_row), int(col), end_r=h - 2, end_c=int(col), color=int(fg))
            out[h - 1, int(col)] = int(ctrl)
            v_segments.append({"col": int(col), "r0": int(spine_row), "r1": h - 2, "blocked": {int(spine_row)}})
    else:
        if h < 18 or w < 15:
            return None
        spine_col = rng.randint(6, w - 7)
        branch_rows = _pick_distinct(rng, list(range(3, h - 3)), branch_count, BRANCH_GAP)
        if len(branch_rows) != branch_count:
            return None
        left_count = rng.randint(1, branch_count - 1)
        rng.shuffle(branch_rows)
        left_rows = sorted(branch_rows[:left_count])
        right_rows = sorted(branch_rows[left_count:])
        if not right_rows:
            return None

        out[1 : h - 1, spine_col] = int(fg)
        out[0, spine_col] = int(ctrl)
        out[h - 1, spine_col] = int(ctrl)
        v_segments.append({"col": int(spine_col), "r0": 1, "r1": h - 2, "blocked": set(branch_rows)})

        for row in left_rows:
            draw_line(out, int(row), 1, end_r=int(row), end_c=spine_col, color=int(fg))
            out[int(row), 0] = int(ctrl)
            h_segments.append({"row": int(row), "c0": 1, "c1": int(spine_col), "blocked": {int(spine_col)}})
        for row in right_rows:
            draw_line(out, int(row), int(spine_col), end_r=int(row), end_c=w - 2, color=int(fg))
            out[int(row), w - 1] = int(ctrl)
            h_segments.append({"row": int(row), "c0": int(spine_col), "c1": w - 2, "blocked": {int(spine_col)}})

    fg_only = out == int(fg)
    comps = components_mask(fg_only, connectivity=4)
    if len(comps) != 1:
        return None
    return out, h_segments, v_segments


def _h_gap_options(
    seg: dict,
    out: np.ndarray,
    bg: int,
    fg: int,
    ctrl: int,
    forbidden: set[tuple[int, int]],
) -> list[tuple[int, str, int]]:
    row = int(seg["row"])
    c0 = int(seg["c0"])
    c1 = int(seg["c1"])
    blocked = {int(x) for x in seg["blocked"]}
    options: list[tuple[int, str, int]] = []
    for g in range(c0 + GAP_END_INSET, c1 - GAP_SIZE):
        gap_cols = range(int(g), int(g + GAP_SIZE))
        if any(abs(int(cell) - int(b)) <= GAP_BLOCK_RADIUS for b in blocked for cell in gap_cols):
            continue
        temp = out.copy()
        temp[row, g : g + GAP_SIZE] = int(bg)
        for anchor in ("left", "right"):
            for dr in (-1, 1):
                coords = _h_chain_coords(row, int(g), str(anchor), int(dr))
                if _validate_diag_chain(temp, bg, fg, ctrl, coords, forbidden):
                    options.append((int(g), str(anchor), int(dr)))
    return options


def _v_gap_options(
    seg: dict,
    out: np.ndarray,
    bg: int,
    fg: int,
    ctrl: int,
    forbidden: set[tuple[int, int]],
) -> list[tuple[int, str, int]]:
    col = int(seg["col"])
    r0 = int(seg["r0"])
    r1 = int(seg["r1"])
    blocked = {int(x) for x in seg["blocked"]}
    options: list[tuple[int, str, int]] = []
    for g in range(r0 + GAP_END_INSET, r1 - GAP_SIZE):
        gap_rows = range(int(g), int(g + GAP_SIZE))
        if any(abs(int(cell) - int(b)) <= GAP_BLOCK_RADIUS for b in blocked for cell in gap_rows):
            continue
        temp = out.copy()
        temp[g : g + GAP_SIZE, col] = int(bg)
        for anchor in ("top", "bottom"):
            for dc in (-1, 1):
                coords = _v_chain_coords(col, int(g), str(anchor), int(dc))
                if _validate_diag_chain(temp, bg, fg, ctrl, coords, forbidden):
                    options.append((int(g), str(anchor), int(dc)))
    return options


def _distractor_options(
    inp: np.ndarray,
    bg: int,
    fg: int,
    ctrl: int,
    forbidden: set[tuple[int, int]],
) -> list[list[tuple[int, int]]]:
    h, w = inp.shape
    options: list[list[tuple[int, int]]] = []
    for length in DISTRACTOR_LENGTHS:
        for r in range(1, h - 1):
            for c in range(1, w - 1):
                for dr, dc in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                    coords = [(int(r + dr * i), int(c + dc * i)) for i in range(length)]
                    if _validate_diag_chain(inp, bg, fg, ctrl, coords, forbidden):
                        options.append(coords)
    return options


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    colors = rng.sample(list(range(N_COLORS)), 3)
    bg, fg, ctrl = map(int, colors)

    built = _build_output(rng, h, w, bg, fg, ctrl)
    if built is None:
        return None
    out, h_segments, v_segments = built
    inp = out.copy()

    forbidden: set[tuple[int, int]] = set()
    gap_count = int(rng.choice([2, 3]))
    distractor_count = int(rng.choice([0, 1, 2]))

    h_ready = [(seg, _h_gap_options(seg, out, bg, fg, ctrl, forbidden)) for seg in h_segments]
    h_ready = [(seg, opts) for seg, opts in h_ready if opts]
    v_ready = [(seg, _v_gap_options(seg, out, bg, fg, ctrl, forbidden)) for seg in v_segments]
    v_ready = [(seg, opts) for seg, opts in v_ready if opts]
    if not h_ready or not v_ready:
        return None

    gap_specs: list[tuple[str, dict]] = []
    if rng.choice([True, False]):
        gap_specs.append(("h", rng.choice(h_ready)[0]))
    else:
        gap_specs.append(("v", rng.choice(v_ready)[0]))
    gap_specs.append(("v", rng.choice(v_ready)[0]) if gap_specs[0][0] == "h" else ("h", rng.choice(h_ready)[0]))

    extra_pool = [("h", seg) for seg, _opts in h_ready] + [("v", seg) for seg, _opts in v_ready]
    rng.shuffle(extra_pool)
    for kind, seg in extra_pool:
        if len(gap_specs) >= gap_count:
            break
        if any(seg is chosen for _k, chosen in gap_specs):
            continue
        gap_specs.append((kind, seg))

    if len(gap_specs) != gap_count:
        return None

    for kind, seg in gap_specs:
        if kind == "h":
            opts = _h_gap_options(seg, out, bg, fg, ctrl, forbidden)
            if not opts:
                return None
            gap_start, anchor, dr = rng.choice(opts)
            row = int(seg["row"])
            inp[row, gap_start : gap_start + GAP_SIZE] = int(bg)
            coords = _h_chain_coords(row, int(gap_start), str(anchor), int(dr))
        else:
            opts = _v_gap_options(seg, out, bg, fg, ctrl, forbidden)
            if not opts:
                return None
            gap_start, anchor, dc = rng.choice(opts)
            col = int(seg["col"])
            inp[gap_start : gap_start + GAP_SIZE, col] = int(bg)
            coords = _v_chain_coords(col, int(gap_start), str(anchor), int(dc))
        for r, c in coords:
            inp[int(r), int(c)] = int(fg)
        _reserve_neighborhood(forbidden, coords)

    net_forbidden: set[tuple[int, int]] = set()
    for r, c in np.argwhere((inp == int(fg)) | (inp == int(ctrl))):
        _reserve_neighborhood(net_forbidden, [(int(r), int(c))])
    for cell in list(forbidden):
        net_forbidden.add((int(cell[0]), int(cell[1])))

    distractors = _distractor_options(inp, bg, fg, ctrl, net_forbidden)
    rng.shuffle(distractors)
    placed = 0
    for coords in distractors:
        if placed >= distractor_count:
            break
        if any((int(r), int(c)) in net_forbidden for r, c in coords):
            continue
        for r, c in coords:
            inp[int(r), int(c)] = int(fg)
        _reserve_neighborhood(net_forbidden, coords)
        placed += 1

    try:
        if not np.array_equal(main(inp), out):
            return None
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for 7b80bb43 combo variant")
