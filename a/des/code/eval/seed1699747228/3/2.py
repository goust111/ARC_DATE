from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

FRAGMENT_COLORS = (BLUE, RED, GREEN)
DIRS = {
    "up": (-1, 0),
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
}
DIR_ORDER = ("up", "right", "down", "left")
OPPOSITE = {
    "up": "down",
    "right": "left",
    "down": "up",
    "left": "right",
}

# The colored road sign sits in a corner-wall next to the junction.
SIGN_OFFSETS = {
    "up": (-1, -1),
    "right": (-1, 1),
    "down": (1, 1),
    "left": (1, -1),
}

GEN_ATTEMPTS = 600


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _open_neighbors(mask: np.ndarray, r: int, c: int) -> list[str]:
    out: list[str] = []
    h, w = mask.shape
    for name, (dr, dc) in DIRS.items():
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]):
            out.append(name)
    return out


def _components(mask: np.ndarray, connectivity: int = 4) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 4:
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        steps = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            q = [(r, c)]
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.pop()
                comp.append((rr, cc))
                for dr, dc in steps:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _boundary_side(h: int, w: int, r: int, c: int) -> str | None:
    if r == 0:
        return "top"
    if r == h - 1:
        return "bottom"
    if c == 0:
        return "left"
    if c == w - 1:
        return "right"
    return None


def _arrow_footprints(h: int, w: int, r: int, c: int) -> list[list[tuple[int, int]]]:
    side = _boundary_side(h, w, r, c)
    options: list[list[tuple[int, int]]] = []
    raw: list[list[tuple[int, int]]] = []

    if side == "top":
        raw = [
            [(0, c - 2), (0, c - 1), (1, c - 2)],
            [(0, c + 1), (0, c + 2), (1, c + 2)],
        ]
    elif side == "bottom":
        raw = [
            [(h - 1, c - 2), (h - 1, c - 1), (h - 2, c - 2)],
            [(h - 1, c + 1), (h - 1, c + 2), (h - 2, c + 2)],
        ]
    elif side == "left":
        raw = [
            [(r - 2, 0), (r - 1, 0), (r - 2, 1)],
            [(r + 1, 0), (r + 2, 0), (r + 2, 1)],
        ]
    elif side == "right":
        raw = [
            [(r - 2, w - 1), (r - 1, w - 1), (r - 2, w - 2)],
            [(r + 1, w - 1), (r + 2, w - 1), (r + 2, w - 2)],
        ]

    for cells in raw:
        if all(_in_bounds((h, w), rr, cc) for rr, cc in cells):
            options.append(cells)
    return options


def _find_fragments(grid: np.ndarray) -> list[dict[str, int | str]]:
    h, w = grid.shape
    found: list[dict[str, int | str]] = []

    boundary_cells: list[tuple[int, int]] = []
    for c in range(w):
        boundary_cells.append((0, c))
        boundary_cells.append((h - 1, c))
    for r in range(1, h - 1):
        boundary_cells.append((r, 0))
        boundary_cells.append((r, w - 1))

    for r, c in boundary_cells:
        color = int(grid[r, c])
        if color not in FRAGMENT_COLORS:
            continue
        inward = []
        for name, (dr, dc) in DIRS.items():
            nr, nc = r + dr, c + dc
            if _in_bounds((h, w), nr, nc) and int(grid[nr, nc]) == BLACK:
                inward.append(name)
        assert len(inward) == 1, "each fragment must have exactly one inward track neighbor"
        found.append({"row": r, "col": c, "color": color, "dir": inward[0]})

    found.sort(key=lambda item: (int(item["row"]), int(item["col"]), int(item["color"])))
    assert found, "expected at least one boundary fragment"
    return found


def _build_open_mask(grid: np.ndarray, fragments: list[dict[str, int | str]]) -> np.ndarray:
    mask = grid == BLACK
    for frag in fragments:
        mask[int(frag["row"]), int(frag["col"])] = True
    return mask


def _find_signs(grid: np.ndarray, open_mask: np.ndarray) -> dict[tuple[int, int], tuple[int, str]]:
    h, w = grid.shape
    signs: dict[tuple[int, int], tuple[int, str]] = {}
    for r, c in np.argwhere(open_mask):
        rr, cc = int(r), int(c)
        if len(_open_neighbors(open_mask, rr, cc)) < 3:
            continue
        found: list[tuple[int, str]] = []
        for direction, (dr, dc) in SIGN_OFFSETS.items():
            sr, sc = rr + dr, cc + dc
            if not _in_bounds((h, w), sr, sc):
                continue
            value = int(grid[sr, sc])
            if value in FRAGMENT_COLORS and not bool(open_mask[sr, sc]):
                found.append((value, direction))
        assert len(found) <= 1, "each junction may have at most one road sign"
        if found:
            signs[(rr, cc)] = found[0]
    return signs


def _simulate_layout(
    open_mask: np.ndarray,
    fragments: list[dict[str, int | str]],
    signs: dict[tuple[int, int], tuple[int, str]],
) -> tuple[list[tuple[int, int, int]], list[dict[str, object]]]:
    h, w = open_mask.shape
    finals: list[tuple[int, int, int]] = []
    traces: list[dict[str, object]] = []

    for frag in fragments:
        r = int(frag["row"])
        c = int(frag["col"])
        color = int(frag["color"])
        direction = str(frag["dir"])
        seen: set[tuple[int, int, str]] = set()
        junctions: list[dict[str, object]] = []

        while True:
            state = (r, c, direction)
            if state in seen:
                raise AssertionError("fragment entered a loop")
            seen.add(state)

            neighbors = _open_neighbors(open_mask, r, c)
            if len(neighbors) >= 3:
                turn_options = tuple(
                    cand
                    for cand in DIR_ORDER
                    if cand in neighbors and cand not in (direction, OPPOSITE[direction])
                )
                event: dict[str, object] = {
                    "pos": (r, c),
                    "incoming": direction,
                    "turn_options": turn_options,
                    "has_sign": (r, c) in signs,
                    "matched": None,
                    "turned": False,
                }
                sign = signs.get((r, c))
                if sign is not None:
                    sign_color, sign_dir = sign
                    event["sign_color"] = int(sign_color)
                    event["sign_dir"] = sign_dir
                    matched = int(sign_color) == color
                    event["matched"] = matched
                    if matched and sign_dir in turn_options:
                        direction = sign_dir
                        event["turned"] = True
                junctions.append(event)

            dr, dc = DIRS[direction]
            nr, nc = r + dr, c + dc
            if not _in_bounds((h, w), nr, nc) or not bool(open_mask[nr, nc]):
                finals.append((r, c, color))
                traces.append(
                    {
                        "start": (int(frag["row"]), int(frag["col"])),
                        "color": color,
                        "final": (r, c, color),
                        "junctions": junctions,
                    }
                )
                break
            r, c = nr, nc

    return finals, traces


def _apply_finals(
    grid: np.ndarray,
    fragments: list[dict[str, int | str]],
    finals: list[tuple[int, int, int]],
) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for frag in fragments:
        out[int(frag["row"]), int(frag["col"])] = BLACK
    for r, c, color in finals:
        out[int(r), int(c)] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move boundary fragments through a rectilinear channel network.

    Grey cells are walls, black cells are channels, yellow elbow triominoes mark
    boundary entries, and colored diagonal wall pixels next to junctions act as
    road signs. A matching fragment turns toward that sign when it reaches the
    junction; otherwise it keeps moving straight until the next wall or edge.
    """
    g = np.asarray(grid, dtype=int)
    fragments = _find_fragments(g)
    open_mask = _build_open_mask(g, fragments)
    signs = _find_signs(g, open_mask)
    finals, _ = _simulate_layout(open_mask, fragments, signs)
    return _apply_finals(g, fragments, finals)


def _build_network() -> np.ndarray | None:
    h = random.choice(range(15, 26, 2))
    w = random.choice(range(15, 26, 2))
    # Keep junctions four cells apart so a diagonal sign belongs to only one junction.
    candidate_rows = list(range(3, h - 3, 4))
    candidate_cols = list(range(3, w - 3, 4))
    if len(candidate_rows) < 2 or len(candidate_cols) < 2:
        return None

    backbone_row = random.choice(candidate_rows)
    backbone_col = random.choice(candidate_cols)
    horiz: dict[int, tuple[int, int]] = {backbone_row: (0, w - 1)}
    vert: dict[int, tuple[int, int]] = {backbone_col: (0, h - 1)}

    target_h = random.randint(2, min(4, len(candidate_rows)))
    target_v = random.randint(2, min(4, len(candidate_cols)))

    while len(horiz) < target_h or len(vert) < target_v:
        choices = []
        if len(horiz) < target_h:
            choices.append("h")
        if len(vert) < target_v:
            choices.append("v")
        orientation = random.choice(choices)

        if orientation == "h":
            row = random.choice([r for r in candidate_rows if r not in horiz])
            anchors = [0] + sorted(vert) + [w - 1]
            pairs = []
            for i, left in enumerate(anchors[:-1]):
                for right in anchors[i + 1 :]:
                    if any(left <= col <= right for col in vert):
                        pairs.append((int(left), int(right)))
            horiz[row] = random.choice(pairs)
        else:
            col = random.choice([c for c in candidate_cols if c not in vert])
            anchors = [0] + sorted(horiz) + [h - 1]
            pairs = []
            for i, top in enumerate(anchors[:-1]):
                for bottom in anchors[i + 1 :]:
                    if any(top <= row <= bottom for row in horiz):
                        pairs.append((int(top), int(bottom)))
            vert[col] = random.choice(pairs)

    open_mask = np.zeros((h, w), dtype=bool)
    for row, (c0, c1) in horiz.items():
        open_mask[int(row), int(c0) : int(c1) + 1] = True
    for col, (r0, r1) in vert.items():
        open_mask[int(r0) : int(r1) + 1, int(col)] = True

    if sum(len(_open_neighbors(open_mask, int(r), int(c))) >= 3 for r, c in np.argwhere(open_mask)) < 2:
        return None
    return open_mask


def _entry_candidates(open_mask: np.ndarray) -> list[dict[str, object]]:
    h, w = open_mask.shape
    candidates: list[dict[str, object]] = []

    boundary_cells: list[tuple[int, int]] = []
    for c in range(w):
        boundary_cells.append((0, c))
        boundary_cells.append((h - 1, c))
    for r in range(1, h - 1):
        boundary_cells.append((r, 0))
        boundary_cells.append((r, w - 1))

    for r, c in boundary_cells:
        if not bool(open_mask[r, c]):
            continue
        neighbors = _open_neighbors(open_mask, r, c)
        if len(neighbors) != 1:
            continue
        arrow_options = []
        for cells in _arrow_footprints(h, w, r, c):
            if all(not bool(open_mask[rr, cc]) for rr, cc in cells):
                arrow_options.append(cells)
        if not arrow_options:
            continue
        candidates.append(
            {
                "row": r,
                "col": c,
                "side": _boundary_side(h, w, r, c),
                "dir": neighbors[0],
                "arrow_options": arrow_options,
            }
        )

    return candidates


def _choose_entries(open_mask: np.ndarray) -> list[dict[str, object]] | None:
    candidates = _entry_candidates(open_mask)
    if len(candidates) < 3:
        return None

    random.shuffle(candidates)
    target = random.randint(3, min(5, len(candidates)))
    chosen: list[dict[str, object]] = []
    used_arrow_cells: set[tuple[int, int]] = set()

    for cand in candidates:
        options = list(cand["arrow_options"])
        random.shuffle(options)
        picked = None
        for cells in options:
            cell_set = set(cells)
            if cell_set & used_arrow_cells:
                continue
            picked = cells
            break
        if picked is None:
            continue

        info = dict(cand)
        info["arrow"] = picked
        chosen.append(info)
        used_arrow_cells |= set(picked)
        if len(chosen) == target:
            break

    if len(chosen) < target:
        return None
    if len({str(item["side"]) for item in chosen}) < 2:
        return None
    return chosen


def _fragments_from_entries(entries: list[dict[str, object]], colors: list[int]) -> list[dict[str, int | str]]:
    frags: list[dict[str, int | str]] = []
    for entry, color in zip(entries, colors):
        frags.append(
            {
                "row": int(entry["row"]),
                "col": int(entry["col"]),
                "dir": str(entry["dir"]),
                "color": int(color),
            }
        )
    return frags


def _intersection_positions(open_mask: np.ndarray) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for r, c in np.argwhere(open_mask):
        rr, cc = int(r), int(c)
        if len(_open_neighbors(open_mask, rr, cc)) >= 3:
            out.append((rr, cc))
    return out


def _render_input(
    open_mask: np.ndarray,
    entries: list[dict[str, object]],
    colors: list[int],
    signs: dict[tuple[int, int], tuple[int, str]],
) -> np.ndarray:
    h, w = open_mask.shape
    grid = np.full((h, w), GREY, dtype=int)
    grid[open_mask] = BLACK

    for entry in entries:
        for r, c in entry["arrow"]:
            assert not bool(open_mask[int(r), int(c)])
            grid[int(r), int(c)] = YELLOW

    for (r, c), (color, direction) in signs.items():
        dr, dc = SIGN_OFFSETS[direction]
        sr, sc = r + dr, c + dc
        assert not bool(open_mask[sr, sc])
        assert grid[sr, sc] == GREY
        grid[sr, sc] = int(color)

    for entry, color in zip(entries, colors):
        grid[int(entry["row"]), int(entry["col"])] = int(color)

    return grid


def _is_yellow_elbow(comp: list[tuple[int, int]]) -> bool:
    if len(comp) != 3:
        return False
    rows = [r for r, _ in comp]
    cols = [c for _, c in comp]
    return max(rows) - min(rows) == 1 and max(cols) - min(cols) == 1


def _validate_generated(grid: np.ndarray) -> bool:
    fragments = _find_fragments(grid)
    if len(set(int(f["color"]) for f in fragments)) < 2:
        return False

    open_mask = _build_open_mask(grid, fragments)
    signs = _find_signs(grid, open_mask)
    if len(signs) < 2:
        return False

    try:
        finals, traces = _simulate_layout(open_mask, fragments, signs)
    except AssertionError:
        return False

    final_positions = [(r, c) for r, c, _ in finals]
    if len(final_positions) != len(set(final_positions)):
        return False

    turned = False
    mismatched = False
    for trace in traces:
        for event in trace["junctions"]:
            if not bool(event["has_sign"]):
                continue
            if bool(event["turned"]):
                turned = True
            if event["matched"] is False:
                mismatched = True
    if not turned or not mismatched:
        return False

    arrow_comps = _components(grid == YELLOW, connectivity=4)
    if len(arrow_comps) != len(fragments):
        return False
    if any(not _is_yellow_elbow(comp) for comp in arrow_comps):
        return False

    expected = _apply_finals(grid, fragments, finals)
    return np.array_equal(main(grid), expected)


def _gen_attempt() -> np.ndarray | None:
    open_mask = _build_network()
    if open_mask is None:
        return None

    entries = _choose_entries(open_mask)
    if entries is None:
        return None

    placeholder = _fragments_from_entries(entries, [BLUE] * len(entries))
    try:
        _, base_traces = _simulate_layout(open_mask, placeholder, {})
    except AssertionError:
        return None

    turn_candidates: list[tuple[int, dict[str, object]]] = []
    for frag_idx, trace in enumerate(base_traces):
        for event in trace["junctions"]:
            if event["turn_options"]:
                turn_candidates.append((frag_idx, event))
    if not turn_candidates:
        return None

    base_colors = [random.choice(FRAGMENT_COLORS) for _ in entries]
    random.shuffle(turn_candidates)
    intersections = _intersection_positions(open_mask)

    for frag_idx, event in turn_candidates:
        signs: dict[tuple[int, int], tuple[int, str]] = {}
        colors = list(base_colors)
        turn_color = random.choice(FRAGMENT_COLORS)
        colors[frag_idx] = turn_color
        turn_dir = random.choice(list(event["turn_options"]))
        signs[tuple(event["pos"])] = (turn_color, turn_dir)

        frags = _fragments_from_entries(entries, colors)
        try:
            _, traces_after_turn = _simulate_layout(open_mask, frags, signs)
        except AssertionError:
            continue

        mismatch_candidates: list[tuple[int, dict[str, object]]] = []
        for idx, trace in enumerate(traces_after_turn):
            for junction in trace["junctions"]:
                if junction["pos"] in signs:
                    continue
                if junction["turn_options"]:
                    mismatch_candidates.append((idx, junction))
        if not mismatch_candidates:
            continue

        random.shuffle(mismatch_candidates)
        picked_mismatch = False
        for mismatch_idx, mismatch_event in mismatch_candidates:
            mismatch_choices = [c for c in FRAGMENT_COLORS if c != colors[mismatch_idx]]
            if not mismatch_choices:
                continue
            mismatch_color = random.choice(mismatch_choices)
            mismatch_dir = random.choice(list(mismatch_event["turn_options"]))
            signs[tuple(mismatch_event["pos"])] = (mismatch_color, mismatch_dir)
            picked_mismatch = True
            break
        if not picked_mismatch:
            continue

        remaining = [pos for pos in intersections if pos not in signs]
        if remaining and random.random() < 0.55:
            extra_pos = random.choice(remaining)
            extra_dirs = [d for d in DIR_ORDER if d in _open_neighbors(open_mask, extra_pos[0], extra_pos[1])]
            signs[extra_pos] = (random.choice(FRAGMENT_COLORS), random.choice(extra_dirs))

        if len(set(colors)) < 2:
            tweak_idx = random.randrange(len(colors))
            choices = [c for c in FRAGMENT_COLORS if c != colors[tweak_idx]]
            colors[tweak_idx] = random.choice(choices)

        grid = _render_input(open_mask, entries, colors, signs)
        if _validate_generated(grid):
            return grid

    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid channel-routing sample")
