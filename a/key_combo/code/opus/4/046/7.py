from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _load_arc_utils():
    try:
        from arc_utils import Color, find_components, get_bg, in_bounds  # type: ignore

        return Color, find_components, get_bg, in_bounds
    except ModuleNotFoundError:
        utils_path = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
        spec = importlib.util.spec_from_file_location("_arc_utils_local", utils_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load arc_utils from {utils_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.Color, module.find_components, module.get_bg, module.in_bounds


Color, find_components, get_bg, in_bounds = _load_arc_utils()

BG = Color.BLACK
ANCHOR = Color.GREY
PALETTE = [Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON]
GEN_RETRIES = 400


def _neighbors4(rc: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = rc
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _is_connected(nodes: set[tuple[int, int]]) -> bool:
    if not nodes:
        return False
    todo = [next(iter(nodes))]
    seen = set()
    while todo:
        cur = todo.pop()
        if cur in seen:
            continue
        seen.add(cur)
        for nxt in _neighbors4(cur):
            if nxt in nodes and nxt not in seen:
                todo.append(nxt)
    return seen == nodes


def _component_records(grid: np.ndarray) -> list[dict]:
    comps = find_components(grid, bg=BG, connectivity=4, monochromatic=False, return_format="dict")
    records = []
    for comp in comps:
        coords = comp["coords"]
        vals = grid[coords[:, 0], coords[:, 1]]
        gray = coords[vals == ANCHOR]
        if gray.shape[0] == 0:
            continue
        assert gray.shape[0] == 1, "each sprite region must contain exactly one gray anchor"
        anchor_r, anchor_c = map(int, gray[0])
        colored = coords[(vals != BG) & (vals != ANCHOR)]
        assert colored.shape[0] >= 1, "each anchor region must contain at least one colored pixel"
        offsets = {}
        for r, c in colored:
            rr = int(r)
            cc = int(c)
            offsets[(rr - anchor_r, cc - anchor_c)] = int(grid[rr, cc])
        records.append(
            {
                "anchor": (anchor_r, anchor_c),
                "offsets": offsets,
                "size": len(offsets),
                "coords": {tuple(map(int, rc)) for rc in coords},
            }
        )
    return records


def _is_consistent_subset(part: dict[tuple[int, int], int], whole: dict[tuple[int, int], int]) -> bool:
    return all(whole.get(offset) == color for offset, color in part.items())


def _source_index(records: list[dict]) -> int:
    candidates = []
    for i, rec in enumerate(records):
        if all(i == j or _is_consistent_subset(other["offsets"], rec["offsets"]) for j, other in enumerate(records)):
            candidates.append(i)
    assert len(candidates) == 1, "expected exactly one full source template"
    src = candidates[0]
    source_size = records[src]["size"]
    assert all(i == src or rec["size"] < source_size for i, rec in enumerate(records)), "targets must be strict partial copies"
    return src


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each partial anchored copy by translating the full source sprite to every gray target anchor.

    Input structure:
    - The background is black.
    - One 4-connected multicolor source sprite contains exactly one adjacent gray anchor pixel.
    - Two to four other 4-connected target sprites each contain one gray anchor and a strict non-empty subset of the source pixels at matching anchor-relative offsets.

    Rule:
    - Find every connected region containing a gray pixel and record its non-gray pixels as anchor-relative offsets.
    - The unique region whose offset-color map is a consistent superset of all the others is the full source template.
    - For every other gray anchor, paint each missing source offset onto black cells only, leave already present matching pixels unchanged, and erase every gray anchor to black.

    Key insight:
    - The gray pixels define a shared local coordinate system: the source anchor provides the complete offset template, while each target anchor tells where to translate that template and which pixels are still missing.
    """
    g = np.asarray(grid, dtype=int)
    records = _component_records(g)
    assert 3 <= len(records) <= 5, "expected one source and two to four targets"

    src_idx = _source_index(records)
    source = records[src_idx]["offsets"]

    out = g.copy()
    gray_coords = np.argwhere(out == ANCHOR)
    out[gray_coords[:, 0], gray_coords[:, 1]] = BG

    for i, rec in enumerate(records):
        if i == src_idx:
            continue
        ar, ac = rec["anchor"]
        for (dr, dc), color in source.items():
            rr = ar + dr
            cc = ac + dc
            assert in_bounds(g.shape, rr, cc), "translated template must stay in bounds"
            current = int(out[rr, cc])
            assert current in (BG, color), "partial copies must agree with the source template"
            if current == BG:
                out[rr, cc] = color

    return out


def _random_mask(rng: np.random.Generator) -> np.ndarray | None:
    for _ in range(80):
        h = int(rng.integers(2, 5))
        w = int(rng.integers(2, 5))
        total = int(rng.integers(4, min(8, h * w) + 1))
        start = (int(rng.integers(h)), int(rng.integers(w)))
        cells = {start}
        while len(cells) < total:
            frontier = []
            for cell in list(cells):
                for nbr in _neighbors4(cell):
                    r, c = nbr
                    if 0 <= r < h and 0 <= c < w and nbr not in cells:
                        frontier.append(nbr)
            if not frontier:
                break
            cells.add(frontier[int(rng.integers(len(frontier)))])
        if len(cells) != total:
            continue
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        cropped = {(r - r0, c - c0) for r, c in cells}
        if len({r for r, _ in cropped}) == 1 or len({c for _, c in cropped}) == 1:
            continue
        hh = r1 - r0 + 1
        ww = c1 - c0 + 1
        mask = np.zeros((hh, ww), dtype=bool)
        for r, c in cropped:
            mask[r, c] = True
        if _is_connected(cropped):
            return mask
    return None


def _sample_template(rng: np.random.Generator) -> tuple[dict[tuple[int, int], int], set[tuple[int, int]]] | None:
    mask = _random_mask(rng)
    if mask is None:
        return None

    cells = {tuple(map(int, rc)) for rc in np.argwhere(mask)}
    anchor_candidates = set()
    for cell in cells:
        for nbr in _neighbors4(cell):
            if nbr not in cells:
                anchor_candidates.add(nbr)
    anchor_candidates = {cell for cell in anchor_candidates if any(nbr in cells for nbr in _neighbors4(cell))}
    if not anchor_candidates:
        return None

    anchor = sorted(anchor_candidates)[int(rng.integers(len(anchor_candidates)))]
    color_count = 1 if rng.random() < 0.45 else 2
    colors = list(map(int, rng.choice(PALETTE, size=color_count, replace=False)))
    assignments = []
    if color_count == 1:
        assignments = [colors[0]] * len(cells)
    else:
        assignments = [colors[i % 2] for i in range(len(cells))]
        rng.shuffle(assignments)
        if len(set(assignments)) < 2:
            return None

    offsets = {}
    for (r, c), color in zip(sorted(cells), assignments):
        offsets[(r - anchor[0], c - anchor[1])] = int(color)

    if not any(abs(dr) + abs(dc) == 1 for dr, dc in offsets):
        return None
    return offsets, cells | {anchor}


def _connected_subset(
    full_offsets: dict[tuple[int, int], int],
    rng: np.random.Generator,
) -> set[tuple[int, int]] | None:
    full = set(full_offsets)
    anchor_neighbors = [offset for offset in full if abs(offset[0]) + abs(offset[1]) == 1]
    if not anchor_neighbors:
        return None
    for _ in range(80):
        keep_size = int(rng.integers(1, len(full)))
        kept = {anchor_neighbors[int(rng.integers(len(anchor_neighbors)))]}
        while len(kept) < keep_size:
            frontier = []
            for cell in kept:
                for nbr in _neighbors4(cell):
                    if nbr in full and nbr not in kept:
                        frontier.append(nbr)
            frontier = sorted(set(frontier))
            if not frontier:
                break
            kept.add(frontier[int(rng.integers(len(frontier)))])
        nodes = {(0, 0)} | kept
        if len(kept) == keep_size and kept != full and _is_connected(nodes):
            return kept
    return None


def _placement_cells(anchor: tuple[int, int], offsets: dict[tuple[int, int], int]) -> set[tuple[int, int]]:
    ar, ac = anchor
    cells = {anchor}
    for dr, dc in offsets:
        cells.add((ar + dr, ac + dc))
    return cells


def _expanded_cells(shape: tuple[int, int], cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    H, W = shape
    expanded = set()
    for r, c in cells:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if 0 <= rr < H and 0 <= cc < W:
                    expanded.add((rr, cc))
    return expanded


def _sample_anchors(
    shape: tuple[int, int],
    offsets: dict[tuple[int, int], int],
    count: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]] | None:
    H, W = shape
    row_offsets = [0] + [dr for dr, _ in offsets]
    col_offsets = [0] + [dc for _, dc in offsets]
    min_r, max_r = min(row_offsets), max(row_offsets)
    min_c, max_c = min(col_offsets), max(col_offsets)

    candidates = [(r, c) for r in range(-min_r, H - max_r) for c in range(-min_c, W - max_c)]
    rng.shuffle(candidates)

    anchors = []
    blocked = set()
    for anchor in candidates:
        full_cells = _placement_cells(anchor, offsets)
        if full_cells & blocked:
            continue
        anchors.append(anchor)
        blocked |= _expanded_cells(shape, full_cells)
        if len(anchors) == count:
            return anchors
    return None


def _render(
    shape: tuple[int, int],
    offsets: dict[tuple[int, int], int],
    source_anchor: tuple[int, int],
    targets: list[tuple[tuple[int, int], set[tuple[int, int]]]],
) -> tuple[np.ndarray, np.ndarray]:
    inp = np.zeros(shape, dtype=int)
    out = np.zeros(shape, dtype=int)

    sr, sc = source_anchor
    inp[sr, sc] = ANCHOR
    for (dr, dc), color in offsets.items():
        rr = sr + dr
        cc = sc + dc
        inp[rr, cc] = color
        out[rr, cc] = color

    for anchor, subset in targets:
        ar, ac = anchor
        inp[ar, ac] = ANCHOR
        for (dr, dc), color in offsets.items():
            rr = ar + dr
            cc = ac + dc
            if (dr, dc) in subset:
                inp[rr, cc] = color
            out[rr, cc] = color

    return inp, out


def _semantic_checks(
    inp: np.ndarray,
    out: np.ndarray,
    offsets: dict[tuple[int, int], int],
    source_anchor: tuple[int, int],
    targets: list[tuple[tuple[int, int], set[tuple[int, int]]]],
) -> bool:
    if not np.array_equal(main(inp), out):
        return False
    if np.array_equal(inp, out):
        return False
    if int(np.count_nonzero(inp == ANCHOR)) != 1 + len(targets):
        return False
    if np.any(out == ANCHOR):
        return False

    full_cells = []
    seen_subsets = set()
    for anchor, subset in [(source_anchor, set(offsets))] + targets:
        cells = _placement_cells(anchor, offsets)
        full_cells.append(cells)
        if subset != set(offsets):
            if not subset or subset == set(offsets):
                return False
            if not _is_connected({(0, 0)} | subset):
                return False
            seen_subsets.add(frozenset(subset))
    for i in range(len(full_cells)):
        for j in range(i + 1, len(full_cells)):
            if full_cells[i] & full_cells[j]:
                return False
    if len(seen_subsets) != len(targets):
        return False

    records = _component_records(inp)
    if len(records) != 1 + len(targets):
        return False
    try:
        src_idx = _source_index(records)
    except AssertionError:
        return False
    if records[src_idx]["size"] != len(offsets):
        return False
    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    made = _sample_template(rng)
    if made is None:
        return None
    offsets, _template_cells = made

    H = int(rng.integers(13, 20))
    W = int(rng.integers(13, 20))
    target_count = int(rng.integers(2, 5))
    anchors = _sample_anchors((H, W), offsets, 1 + target_count, rng)
    if anchors is None:
        return None

    subsets = []
    used = set()
    for _ in range(target_count):
        subset = _connected_subset(offsets, rng)
        if subset is None or frozenset(subset) in used:
            return None
        used.add(frozenset(subset))
        subsets.append(subset)

    source_anchor = anchors[0]
    targets = list(zip(anchors[1:], subsets))
    inp, out = _render((H, W), offsets, source_anchor, targets)

    if not _semantic_checks(inp, out, offsets, source_anchor, targets):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid anchored-completion puzzle")
