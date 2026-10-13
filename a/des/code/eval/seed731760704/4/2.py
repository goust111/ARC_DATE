from __future__ import annotations

import random
from collections import Counter, defaultdict, deque

import numpy as np


BG = 0
SIDES = ("N", "E", "S", "W")
OPPOSITE = {"N": "S", "S": "N", "E": "W", "W": "E"}


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = [(r, c)]
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.pop()
                comp.append((cr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _legend_map(grid: np.ndarray, bg: int) -> tuple[dict[int, int], list[tuple[int, int, int, int]], list[list[tuple[int, int]]]]:
    pairs: dict[int, int] = {}
    pair_boxes: list[tuple[int, int, int, int]] = []
    piece_like: list[list[tuple[int, int]]] = []
    for comp in _components(grid != bg):
        r0, c0, r1, c1 = _bbox(comp)
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        colors = set(map(int, sub[sub != bg].ravel()))
        if len(comp) == 2 and r0 == r1 and c1 == c0 + 1 and len(colors) == 2:
            left = int(grid[r0, c0])
            right = int(grid[r0, c1])
            pairs[left] = right
            pair_boxes.append((r0, c0, r1, c1))
        else:
            piece_like.append(comp)
    assert pairs, "expected at least one legend pair"
    return pairs, pair_boxes, piece_like


def _parse_piece(grid: np.ndarray, comp: list[tuple[int, int]], bg: int) -> dict:
    r0, c0, r1, c1 = _bbox(comp)
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    colors = [int(v) for v in np.unique(sub) if int(v) != bg]
    assert len(colors) == 2, "piece must contain one body color and one marker color"

    counts = {color: int(np.sum(sub == color)) for color in colors}
    marker_candidates = [color for color, count in counts.items() if count == 1]
    assert len(marker_candidates) == 1, "piece marker must be a unique pixel"
    marker = marker_candidates[0]
    body = next(color for color in colors if color != marker)

    occ = sub != bg
    top_tab = int(np.sum(occ[0, :])) == 1
    bottom_tab = int(np.sum(occ[-1, :])) == 1
    left_tab = int(np.sum(occ[:, 0])) == 1
    right_tab = int(np.sum(occ[:, -1])) == 1

    core_r0 = 1 if top_tab else 0
    core_r1 = occ.shape[0] - 2 if bottom_tab else occ.shape[0] - 1
    core_c0 = 1 if left_tab else 0
    core_c1 = occ.shape[1] - 2 if right_tab else occ.shape[1] - 1
    assert core_r0 <= core_r1 and core_c0 <= core_c1

    core = occ[core_r0 : core_r1 + 1, core_c0 : core_c1 + 1]
    core_h, core_w = map(int, core.shape)

    mr, mc = map(int, np.argwhere(sub == marker)[0])
    assert core_r0 <= mr <= core_r1 and core_c0 <= mc <= core_c1

    sides: dict[str, tuple[int, int | None, int]] = {}

    if top_tab:
        cols = np.flatnonzero(occ[0, :])
        assert len(cols) == 1
        sides["N"] = (1, int(cols[0] - core_c0), core_w)
    else:
        miss = np.flatnonzero(~core[0, :])
        assert len(miss) in (0, 1)
        sides["N"] = (-1, int(miss[0]), core_w) if len(miss) == 1 else (0, None, core_w)

    if bottom_tab:
        cols = np.flatnonzero(occ[-1, :])
        assert len(cols) == 1
        sides["S"] = (1, int(cols[0] - core_c0), core_w)
    else:
        miss = np.flatnonzero(~core[-1, :])
        assert len(miss) in (0, 1)
        sides["S"] = (-1, int(miss[0]), core_w) if len(miss) == 1 else (0, None, core_w)

    if left_tab:
        rows = np.flatnonzero(occ[:, 0])
        assert len(rows) == 1
        sides["W"] = (1, int(rows[0] - core_r0), core_h)
    else:
        miss = np.flatnonzero(~core[:, 0])
        assert len(miss) in (0, 1)
        sides["W"] = (-1, int(miss[0]), core_h) if len(miss) == 1 else (0, None, core_h)

    if right_tab:
        rows = np.flatnonzero(occ[:, -1])
        assert len(rows) == 1
        sides["E"] = (1, int(rows[0] - core_r0), core_h)
    else:
        miss = np.flatnonzero(~core[:, -1])
        assert len(miss) in (0, 1)
        sides["E"] = (-1, int(miss[0]), core_h) if len(miss) == 1 else (0, None, core_h)

    center_r = core_r0 + core_h // 2
    center_c = core_c0 + core_w // 2
    assert (mr, mc) == (center_r, center_c), "marker must sit at the core center"

    return {
        "body": body,
        "marker": marker,
        "core_h": core_h,
        "core_w": core_w,
        "sides": sides,
    }


def _adjacency(pieces: list[dict]) -> list[list[tuple[int, str]]]:
    by_key: dict[tuple[str, int, int, int], list[int]] = defaultdict(list)
    for idx, piece in enumerate(pieces):
        for side in SIDES:
            sign, offset, span = piece["sides"][side]
            if sign != 0:
                assert offset is not None
                by_key[(side, span, int(offset), sign)].append(idx)

    graph: list[list[tuple[int, str]]] = [[] for _ in pieces]
    for idx, piece in enumerate(pieces):
        for side in SIDES:
            sign, offset, span = piece["sides"][side]
            if sign == 0:
                continue
            opp = OPPOSITE[side]
            candidates = [j for j in by_key[(opp, span, int(offset), -sign)] if j != idx]
            assert len(candidates) == 1, "every connector must have a unique complement"
            graph[idx].append((candidates[0], side))
    return graph


def _place_pieces(pieces: list[dict], graph: list[list[tuple[int, str]]]) -> dict[int, tuple[int, int]]:
    positions: dict[int, tuple[int, int]] = {0: (0, 0)}
    q = deque([0])
    while q:
        idx = q.popleft()
        base_r, base_c = positions[idx]
        piece = pieces[idx]
        for nxt, side in graph[idx]:
            other = pieces[nxt]
            sign, offset, span = piece["sides"][side]
            other_sign, other_offset, other_span = other["sides"][OPPOSITE[side]]
            assert sign == -other_sign and offset == other_offset and span == other_span

            if side == "E":
                assert piece["core_h"] == other["core_h"]
                pos = (base_r, base_c + piece["core_w"])
            elif side == "W":
                assert piece["core_h"] == other["core_h"]
                pos = (base_r, base_c - other["core_w"])
            elif side == "S":
                assert piece["core_w"] == other["core_w"]
                pos = (base_r + piece["core_h"], base_c)
            else:
                assert piece["core_w"] == other["core_w"]
                pos = (base_r - other["core_h"], base_c)

            if nxt in positions:
                assert positions[nxt] == pos, "connector graph must imply consistent placement"
            else:
                positions[nxt] = pos
                q.append(nxt)

    assert len(positions) == len(pieces), "assembled puzzle must be connected"
    return positions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble scattered jigsaw-like rectangular pieces by complementary tabs/notches,
    then recolor each piece according to the legend that maps its center marker color
    to an output color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _most_common_color(g)

    legend, _pair_boxes, piece_comps = _legend_map(g, bg)
    pieces = [_parse_piece(g, comp, bg) for comp in piece_comps]
    assert pieces, "expected at least one puzzle piece"

    for piece in pieces:
        assert piece["marker"] in legend, "every piece marker must appear in the legend"

    graph = _adjacency(pieces)
    positions = _place_pieces(pieces, graph)

    min_r = min(r for r, _ in positions.values())
    min_c = min(c for _, c in positions.values())
    max_r = max(positions[i][0] + pieces[i]["core_h"] for i in range(len(pieces)))
    max_c = max(positions[i][1] + pieces[i]["core_w"] for i in range(len(pieces)))

    out = np.full((max_r - min_r, max_c - min_c), -1, dtype=int)
    for idx, piece in enumerate(pieces):
        r0, c0 = positions[idx]
        rr = r0 - min_r
        cc = c0 - min_c
        region = out[rr : rr + piece["core_h"], cc : cc + piece["core_w"]]
        assert np.all(region == -1), "assembled core rectangles must not overlap"
        region[:, :] = int(legend[piece["marker"]])

    assert np.all(out != -1), "assembled core rectangles must tile the final rectangle"
    return out


BODY_COLORS = [1, 2, 3]
TARGET_COLORS = [4, 5, 6, 7]
MARKER_COLORS = [1, 2, 3, 4, 5, 6, 7, 8, 9]
LAYOUTS = [(1, 3), (1, 4), (1, 5), (2, 2), (2, 3), (3, 2)]
DIM_CHOICES = [7, 9, 11]
GRID_MIN = 28
GRID_MAX = 40
GEN_ATTEMPTS = 400


def _make_piece_patch(h: int, w: int, body: int, marker: int, sides: dict[str, tuple[int, int | None]]) -> np.ndarray:
    top_tab = sides["N"][0] == 1
    bottom_tab = sides["S"][0] == 1
    left_tab = sides["W"][0] == 1
    right_tab = sides["E"][0] == 1

    patch_h = h + int(top_tab) + int(bottom_tab)
    patch_w = w + int(left_tab) + int(right_tab)
    patch = np.full((patch_h, patch_w), BG, dtype=int)
    r0 = 1 if top_tab else 0
    c0 = 1 if left_tab else 0
    patch[r0 : r0 + h, c0 : c0 + w] = body

    if sides["N"][0] == -1:
        patch[r0, c0 + int(sides["N"][1])] = BG
    elif sides["N"][0] == 1:
        patch[0, c0 + int(sides["N"][1])] = body

    if sides["S"][0] == -1:
        patch[r0 + h - 1, c0 + int(sides["S"][1])] = BG
    elif sides["S"][0] == 1:
        patch[patch_h - 1, c0 + int(sides["S"][1])] = body

    if sides["W"][0] == -1:
        patch[r0 + int(sides["W"][1]), c0] = BG
    elif sides["W"][0] == 1:
        patch[r0 + int(sides["W"][1]), 0] = body

    if sides["E"][0] == -1:
        patch[r0 + int(sides["E"][1]), c0 + w - 1] = BG
    elif sides["E"][0] == 1:
        patch[r0 + int(sides["E"][1]), patch_w - 1] = body

    patch[r0 + h // 2, c0 + w // 2] = marker
    return patch


def _expected_output(row_heights: list[int], col_widths: list[int], targets: dict[tuple[int, int], int]) -> np.ndarray:
    h = sum(row_heights)
    w = sum(col_widths)
    out = np.full((h, w), BG, dtype=int)
    r = 0
    for rr, row_h in enumerate(row_heights):
        c = 0
        for cc, col_w in enumerate(col_widths):
            out[r : r + row_h, c : c + col_w] = targets[(rr, cc)]
            c += col_w
        r += row_h
    return out


def _choose_offsets(counts: Counter) -> dict[int, list[int]] | None:
    pools: dict[int, list[int]] = {}
    for span, needed in counts.items():
        choices = list(range(1, span - 1))
        if needed > len(choices):
            return None
        random.shuffle(choices)
        pools[int(span)] = choices[:needed]
    return pools


def _rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0)


def _place_patch(grid: np.ndarray, patch: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> bool:
    h, w = grid.shape
    ph, pw = patch.shape
    for _ in range(300):
        r0 = random.randint(0, h - ph)
        c0 = random.randint(0, w - pw)
        box = (r0, c0, r0 + ph - 1, c0 + pw - 1)
        if any(_rects_overlap(box, other, gap=1) for other in boxes):
            continue
        sub = grid[r0 : r0 + ph, c0 : c0 + pw]
        if np.any((patch != BG) & (sub != BG)):
            continue
        sub[patch != BG] = patch[patch != BG]
        boxes.append(box)
        return True
    return False


def _legend_patch(mapping: list[tuple[int, int]]) -> np.ndarray:
    patch = np.full((2 * len(mapping) - 1, 2), BG, dtype=int)
    for i, (marker, target) in enumerate(mapping):
        patch[2 * i, 0] = marker
        patch[2 * i, 1] = target
    return patch


def _corner_pos(grid_h: int, grid_w: int, patch_h: int, patch_w: int, corner: str) -> tuple[int, int]:
    if corner == "TL":
        return 0, 0
    if corner == "TR":
        return 0, grid_w - patch_w
    if corner == "BL":
        return grid_h - patch_h, 0
    return grid_h - patch_h, grid_w - patch_w


def _paint_at(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    sub = grid[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]]
    sub[patch != BG] = patch[patch != BG]


def _generate_case() -> tuple[np.ndarray, np.ndarray] | None:
    rows, cols = random.choice(LAYOUTS)
    row_heights = [random.choice(DIM_CHOICES) for _ in range(rows)]
    col_widths = [random.choice(DIM_CHOICES) for _ in range(cols)]
    piece_count = rows * cols

    marker_colors = random.sample(MARKER_COLORS, piece_count)
    remaining = [c for c in range(1, 10) if c not in marker_colors]
    if len(remaining) < 2:
        return None
    body_palette = random.sample(remaining, k=min(2, len(remaining)))
    target_palette = remaining[:]

    vert_counts = Counter(row_heights[r] for r in range(rows) for _ in range(cols - 1))
    horiz_counts = Counter(col_widths[c] for c in range(cols) for _ in range(rows - 1))
    vert_offsets = _choose_offsets(vert_counts)
    horiz_offsets = _choose_offsets(horiz_counts)
    if vert_offsets is None or horiz_offsets is None:
        return None

    pieces: dict[tuple[int, int], dict[str, tuple[int, int | None]]] = {}
    for r in range(rows):
        for c in range(cols):
            pieces[(r, c)] = {side: (0, None) for side in SIDES}

    used_vert = defaultdict(int)
    for r in range(rows):
        span = row_heights[r]
        for c in range(cols - 1):
            idx = used_vert[span]
            used_vert[span] += 1
            offset = vert_offsets[span][idx]
            if random.random() < 0.5:
                pieces[(r, c)]["E"] = (1, offset)
                pieces[(r, c + 1)]["W"] = (-1, offset)
            else:
                pieces[(r, c)]["E"] = (-1, offset)
                pieces[(r, c + 1)]["W"] = (1, offset)

    used_horiz = defaultdict(int)
    for c in range(cols):
        span = col_widths[c]
        for r in range(rows - 1):
            idx = used_horiz[span]
            used_horiz[span] += 1
            offset = horiz_offsets[span][idx]
            if random.random() < 0.5:
                pieces[(r, c)]["S"] = (1, offset)
                pieces[(r + 1, c)]["N"] = (-1, offset)
            else:
                pieces[(r, c)]["S"] = (-1, offset)
                pieces[(r + 1, c)]["N"] = (1, offset)

    mapping_items: list[tuple[int, int]] = []
    patches: list[np.ndarray] = []
    targets: dict[tuple[int, int], int] = {}
    idx = 0
    for r in range(rows):
        for c in range(cols):
            marker = marker_colors[idx]
            body = random.choice(body_palette)
            target = random.choice(target_palette)
            targets[(r, c)] = target
            mapping_items.append((marker, target))
            patch = _make_piece_patch(row_heights[r], col_widths[c], body, marker, pieces[(r, c)])
            patches.append(patch)
            idx += 1

    expected = _expected_output(row_heights, col_widths, targets)

    max_piece_h = max(p.shape[0] for p in patches)
    max_piece_w = max(p.shape[1] for p in patches)
    legend = _legend_patch(mapping_items)
    min_h = max(GRID_MIN, legend.shape[0] + 2, max_piece_h + 2)
    min_w = max(GRID_MIN, legend.shape[1] + 2, max_piece_w + 2)
    grid_h = random.randint(min_h, GRID_MAX)
    grid_w = random.randint(min_w, GRID_MAX)
    grid = np.full((grid_h, grid_w), BG, dtype=int)

    corner = random.choice(["TL", "TR", "BL", "BR"])
    lr, lc = _corner_pos(grid_h, grid_w, legend.shape[0], legend.shape[1], corner)
    _paint_at(grid, legend, lr, lc)
    boxes = [(lr, lc, lr + legend.shape[0] - 1, lc + legend.shape[1] - 1)]

    random.shuffle(patches)
    for patch in patches:
        if not _place_patch(grid, patch, boxes):
            return None

    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        case = _generate_case()
        if case is None:
            continue
        grid, expected = case
        try:
            actual = main(grid)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
