from __future__ import annotations

import random

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIR8 = tuple(
    (dr, dc)
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
    if not (dr == 0 and dc == 0)
)


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert len(coords) > 0
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


def _crop_group(grid: np.ndarray, mask: np.ndarray, bg: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    bbox = _bbox_from_mask(mask)
    r0, c0, r1, c1 = bbox
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    submask = mask[r0 : r1 + 1, c0 : c1 + 1]
    patch[submask] = grid[r0 : r1 + 1, c0 : c1 + 1][submask]
    return patch, bbox


def _crop_non_bg(grid: np.ndarray, bg: int) -> np.ndarray:
    mask = grid != bg
    if not mask.any():
        return np.full((0, 0), bg, dtype=int)
    r0, c0, r1, c1 = _bbox_from_mask(mask)
    return np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int)


def _true_runs(flags: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    i = 0
    n = int(len(flags))
    while i < n:
        if not bool(flags[i]):
            i += 1
            continue
        j = i
        while j + 1 < n and bool(flags[j + 1]):
            j += 1
        runs.append((i, j))
        i = j + 1
    return runs


def _mirror_patch(patch: np.ndarray, kind: str) -> np.ndarray:
    if kind == "cols":
        return np.fliplr(patch)
    return np.flipud(patch)


def _mirror_cell(kind: str, h: int, w: int, r: int, c: int) -> tuple[int, int]:
    if kind == "cols":
        return r, w - 1 - c
    return h - 1 - r, c


def _candidate_from_gap(
    grid: np.ndarray,
    bg: int,
    kind: str,
    start: int,
    end: int,
) -> dict | None:
    fg = grid != bg
    if kind == "cols":
        left = fg.copy()
        left[:, start:] = False
        right = fg.copy()
        right[:, : end + 1] = False
    else:
        left = fg.copy()
        left[start:, :] = False
        right = fg.copy()
        right[: end + 1, :] = False

    if not left.any() or not right.any():
        return None

    patch_a, bbox_a = _crop_group(grid, left, bg)
    patch_b, bbox_b = _crop_group(grid, right, bg)
    if patch_a.shape != patch_b.shape:
        return None
    if not np.array_equal(patch_a, _mirror_patch(patch_b, kind)):
        return None

    ra0, ca0, ra1, ca1 = bbox_a
    rb0, cb0, rb1, cb1 = bbox_b
    center2_a = (ra0 + ra1, ca0 + ca1)
    center2_b = (rb0 + rb1, cb0 + cb1)

    if kind == "cols":
        if center2_a[0] != center2_b[0]:
            return None
        if (center2_b[1] - center2_a[1]) % 4 != 0:
            return None
    else:
        if center2_a[1] != center2_b[1]:
            return None
        if (center2_b[0] - center2_a[0]) % 4 != 0:
            return None

    return {
        "kind": kind,
        "bbox_a": bbox_a,
        "bbox_b": bbox_b,
        "coords_a": np.argwhere(left),
        "coords_b": np.argwhere(right),
        "gap": end - start + 1,
    }


def _find_candidate(grid: np.ndarray, bg: int) -> dict:
    fg = grid != bg
    assert fg.any(), "expected at least one foreground pixel"

    candidates: list[dict] = []

    empty_cols = ~fg.any(axis=0)
    for start, end in _true_runs(empty_cols):
        if fg[:, :start].any() and fg[:, end + 1 :].any():
            cand = _candidate_from_gap(grid, bg, "cols", start, end)
            if cand is not None:
                candidates.append(cand)

    empty_rows = ~fg.any(axis=1)
    for start, end in _true_runs(empty_rows):
        if fg[:start, :].any() and fg[end + 1 :, :].any():
            cand = _candidate_from_gap(grid, bg, "rows", start, end)
            if cand is not None:
                candidates.append(cand)

    assert candidates, "failed to find a valid mirrored split"
    candidates.sort(key=lambda item: (item["gap"], item["kind"] == "cols"), reverse=True)
    best = candidates[0]
    if len(candidates) > 1:
        assert (
            candidates[0]["gap"] > candidates[1]["gap"]
            or candidates[0]["kind"] == candidates[1]["kind"]
        ), "split is ambiguous"
    return best


def _merge_candidate(grid: np.ndarray, bg: int, candidate: dict) -> np.ndarray:
    ra0, ca0, ra1, ca1 = candidate["bbox_a"]
    rb0, cb0, rb1, cb1 = candidate["bbox_b"]
    kind = candidate["kind"]

    center2_a = (ra0 + ra1, ca0 + ca1)
    center2_b = (rb0 + rb1, cb0 + cb1)
    diff_r = center2_b[0] - center2_a[0]
    diff_c = center2_b[1] - center2_a[1]
    assert diff_r % 4 == 0 and diff_c % 4 == 0, "group centers must align on the cell lattice"

    shift_a = (diff_r // 4, diff_c // 4)
    shift_b = (-shift_a[0], -shift_a[1])

    if kind == "cols":
        axis4 = center2_a[1] + center2_b[1]
    else:
        axis4 = center2_a[0] + center2_b[0]

    placed: list[tuple[int, int, int, int, int]] = []
    for priority, (coords, shift) in enumerate(
        ((candidate["coords_a"], shift_a), (candidate["coords_b"], shift_b))
    ):
        dr, dc = shift
        for r_raw, c_raw in coords:
            r = int(r_raw)
            c = int(c_raw)
            rr = r + dr
            cc = c + dc
            color = int(grid[r, c])
            if kind == "cols":
                dist = abs(4 * c - axis4)
            else:
                dist = abs(4 * r - axis4)
            placed.append((rr, cc, color, dist, priority))

    min_r = min(item[0] for item in placed)
    min_c = min(item[1] for item in placed)
    max_r = max(item[0] for item in placed)
    max_c = max(item[1] for item in placed)

    out = np.full((max_r - min_r + 1, max_c - min_c + 1), bg, dtype=int)
    best: dict[tuple[int, int], tuple[int, int, int]] = {}
    for rr, cc, color, dist, priority in placed:
        key = (rr - min_r, cc - min_c)
        prev = best.get(key)
        if prev is None or dist < prev[1] or (dist == prev[1] and priority < prev[2]):
            best[key] = (color, dist, priority)

    for (r, c), (color, _dist, _priority) in best.items():
        out[r, c] = color
    return _crop_non_bg(out, bg)


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    bg = _get_bg(g)
    candidate = _find_candidate(g, bg)
    return _merge_candidate(g, bg, candidate)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Merge two mirrored sprite groups by aligning their bbox centers onto the shared midpoint.

    The groups are separated by a full background gap along one axis. That gap
    identifies the two halves; their tight crops must be mirror images of each
    other. Each group's pixels are then translated so the two bbox centers meet
    at their midpoint, and collisions keep the source that was closer to the
    inferred reflection axis in the input.
    """
    return _solve(grid)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(comp)
    return out


def _component_sizes(total: int, count: int, lo: int, hi: int) -> list[int] | None:
    sizes: list[int] = []
    remaining = total
    for i in range(count):
        left = count - i - 1
        cur_lo = max(lo, remaining - left * hi)
        cur_hi = min(hi, remaining - left * lo)
        if cur_lo > cur_hi:
            return None
        sizes.append(random.randint(cur_lo, cur_hi))
        remaining -= sizes[-1]
    random.shuffle(sizes)
    return sizes


def _grow_component(
    h: int,
    w: int,
    occupied: np.ndarray,
    blocked: np.ndarray,
    kind: str,
    forbid_overlap: bool,
    allow_overlap_bias: bool,
    size: int,
) -> list[tuple[int, int]] | None:
    covered_rows = {int(r) for r in np.flatnonzero(occupied.any(axis=1))}
    covered_cols = {int(c) for c in np.flatnonzero(occupied.any(axis=0))}
    touched_edges = {
        name
        for name, cond in (
            ("top", bool(occupied[0, :].any())),
            ("bottom", bool(occupied[-1, :].any())),
            ("left", bool(occupied[:, 0].any())),
            ("right", bool(occupied[:, -1].any())),
        )
        if cond
    }

    selected: set[tuple[int, int]] = set()
    sel_rows: set[int] = set()
    sel_cols: set[int] = set()
    sel_edges: set[str] = set()

    def cell_ok(r: int, c: int) -> bool:
        if bool(blocked[r, c]) or (r, c) in selected:
            return False
        if forbid_overlap:
            mr, mc = _mirror_cell(kind, h, w, r, c)
            if (mr, mc) == (r, c):
                return False
            if bool(occupied[mr, mc]) or (mr, mc) in selected:
                return False
        return True

    def score(r: int, c: int) -> float:
        value = 1.0
        if r not in covered_rows and r not in sel_rows:
            value += 3.0
        if c not in covered_cols and c not in sel_cols:
            value += 3.0
        if r == 0 and "top" not in touched_edges and "top" not in sel_edges:
            value += 4.0
        if r == h - 1 and "bottom" not in touched_edges and "bottom" not in sel_edges:
            value += 4.0
        if c == 0 and "left" not in touched_edges and "left" not in sel_edges:
            value += 4.0
        if c == w - 1 and "right" not in touched_edges and "right" not in sel_edges:
            value += 4.0
        if allow_overlap_bias:
            mr, mc = _mirror_cell(kind, h, w, r, c)
            if (mr, mc) != (r, c) and (bool(occupied[mr, mc]) or (mr, mc) in selected):
                value += 5.0
        return value

    starts = [(r, c) for r in range(h) for c in range(w) if cell_ok(r, c)]
    if not starts:
        return None
    weights = [score(r, c) for r, c in starts]
    start_r, start_c = random.choices(starts, weights=weights, k=1)[0]
    selected.add((start_r, start_c))
    sel_rows.add(start_r)
    sel_cols.add(start_c)
    if start_r == 0:
        sel_edges.add("top")
    if start_r == h - 1:
        sel_edges.add("bottom")
    if start_c == 0:
        sel_edges.add("left")
    if start_c == w - 1:
        sel_edges.add("right")

    while len(selected) < size:
        frontier: set[tuple[int, int]] = set()
        for r, c in selected:
            for dr, dc in DIR4:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w and cell_ok(nr, nc):
                    frontier.add((nr, nc))
        if not frontier:
            return None
        frontier_list = list(frontier)
        frontier_weights = [score(r, c) for r, c in frontier_list]
        pick_r, pick_c = random.choices(frontier_list, weights=frontier_weights, k=1)[0]
        selected.add((pick_r, pick_c))
        sel_rows.add(pick_r)
        sel_cols.add(pick_c)
        if pick_r == 0:
            sel_edges.add("top")
        if pick_r == h - 1:
            sel_edges.add("bottom")
        if pick_c == 0:
            sel_edges.add("left")
        if pick_c == w - 1:
            sel_edges.add("right")

    return list(selected)


def _mark_blocked(blocked: np.ndarray, cells: list[tuple[int, int]]) -> None:
    h, w = blocked.shape
    for r, c in cells:
        for dr, dc in DIR8 + ((0, 0),):
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w:
                blocked[nr, nc] = True


def _nontrivial_overlap_pairs(mask: np.ndarray, kind: str) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    h, w = mask.shape
    pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        mr, mc = _mirror_cell(kind, h, w, rr, cc)
        if (mr, mc) == (rr, cc):
            continue
        if bool(mask[mr, mc]) and (rr, cc) < (mr, mc):
            pairs.append(((rr, cc), (mr, mc)))
    return pairs


def _build_patch(kind: str, overlap_mode: bool) -> tuple[np.ndarray, list[list[tuple[int, int]]], int] | None:
    h = random.randint(6, 9)
    w = random.randint(6, 9)
    n_components = random.randint(2, 4)

    min_total = max(h, w) + 2
    max_total = min(h * w // 2, max(h, w) + 8, 18)
    if min_total > max_total:
        return None
    total = random.randint(min_total, max_total)
    sizes = _component_sizes(total, n_components, 2, 6)
    if sizes is None:
        return None

    occupied = np.zeros((h, w), dtype=bool)
    blocked = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for size in sizes:
        comp = _grow_component(
            h=h,
            w=w,
            occupied=occupied,
            blocked=blocked,
            kind=kind,
            forbid_overlap=not overlap_mode,
            allow_overlap_bias=overlap_mode,
            size=size,
        )
        if comp is None:
            return None
        for r, c in comp:
            occupied[r, c] = True
        _mark_blocked(blocked, comp)
        components.append(comp)

    if len(_components(occupied)) != n_components:
        return None
    if not occupied.any(axis=0).all() or not occupied.any(axis=1).all():
        return None
    if not (occupied[0, :].any() and occupied[-1, :].any() and occupied[:, 0].any() and occupied[:, -1].any()):
        return None

    pairs = _nontrivial_overlap_pairs(occupied, kind)
    if overlap_mode:
        if not pairs:
            return None
    else:
        if pairs:
            return None

    mirrored_occ = _mirror_patch(occupied.astype(int), kind).astype(bool)
    if np.array_equal(occupied, mirrored_occ):
        return None

    bg = random.randrange(10)
    palette = [color for color in range(10) if color != bg]
    random.shuffle(palette)
    patch = np.full((h, w), bg, dtype=int)
    for comp in components:
        comp_cells = list(comp)
        random.shuffle(comp_cells)
        n_cols = random.randint(2, 3)
        comp_palette = random.sample(palette, n_cols)
        patch[comp_cells[0]] = comp_palette[0]
        patch[comp_cells[1]] = comp_palette[1]
        for cell in comp_cells[2:]:
            patch[cell] = random.choice(comp_palette)

    if overlap_mode:
        pair_a, pair_b = random.choice(pairs)
        if int(patch[pair_a]) == int(patch[pair_b]):
            choices = [color for color in palette if color != int(patch[pair_a])]
            patch[pair_b] = random.choice(choices)

    for comp in components:
        colors = {int(patch[r, c]) for r, c in comp}
        if len(colors) < 2:
            return None

    return patch, components, bg


def _place_groups(patch: np.ndarray, kind: str, bg: int) -> np.ndarray | None:
    h, w = patch.shape
    patch_b = _mirror_patch(patch, kind)

    if kind == "cols":
        gap_choices = [gap for gap in range(2, 7) if gap % 2 == w % 2]
        if not gap_choices:
            return None
        gap = random.choice(gap_choices)
        pad_top = random.randint(0, 4)
        pad_bottom = random.randint(0, 4)
        pad_left = random.randint(0, 4)
        pad_right = random.randint(0, 4)
        grid = np.full((h + pad_top + pad_bottom, 2 * w + gap + pad_left + pad_right), bg, dtype=int)
        r0 = pad_top
        c0 = pad_left
        c1 = pad_left + w + gap
        grid[r0 : r0 + h, c0 : c0 + w] = patch
        grid[r0 : r0 + h, c1 : c1 + w] = patch_b
        return grid

    gap_choices = [gap for gap in range(2, 7) if gap % 2 == h % 2]
    if not gap_choices:
        return None
    gap = random.choice(gap_choices)
    pad_top = random.randint(0, 4)
    pad_bottom = random.randint(0, 4)
    pad_left = random.randint(0, 4)
    pad_right = random.randint(0, 4)
    grid = np.full((2 * h + gap + pad_top + pad_bottom, w + pad_left + pad_right), bg, dtype=int)
    r0 = pad_top
    r1 = pad_top + h + gap
    c0 = pad_left
    grid[r0 : r0 + h, c0 : c0 + w] = patch
    grid[r1 : r1 + h, c0 : c0 + w] = patch_b
    return grid


def _valid_input(grid: np.ndarray) -> bool:
    bg = _get_bg(grid)
    fg = grid != bg
    candidate = _find_candidate(grid, bg)
    comps = _components(fg)
    if len(comps) < 4:
        return False
    return candidate["kind"] in {"cols", "rows"}


def _gen_attempt() -> np.ndarray | None:
    kind = random.choice(("cols", "rows"))
    overlap_mode = bool(random.getrandbits(1))
    built = _build_patch(kind, overlap_mode)
    if built is None:
        return None
    patch, components, bg = built
    if len(components) < 2:
        return None
    grid = _place_groups(patch, kind, bg)
    if grid is None:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.size == 0 or not _valid_input(grid):
        return None
    if np.array_equal(out, _crop_non_bg(grid, bg)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirrored-group input")
