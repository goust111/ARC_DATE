import random
from typing import Any

import numpy as np


BLACK = 0
MAROON = 9
PATH_COLORS = [1, 2, 3, 4, 5, 6, 7, 8]
MAX_GENERATION_ATTEMPTS = 400


def _perimeter_index(r: int, c: int, h: int, w: int) -> int:
    if r == 0:
        return c
    if c == w - 1:
        return w + r - 1
    if r == h - 1:
        return w + h - 1 + (w - 1 - c)
    assert c == 0
    return 2 * w + h - 2 + (h - 1 - r)


def _indicator_sequence(grid: np.ndarray) -> tuple[list[int], int]:
    h, w = grid.shape
    assert h >= 3 and w >= 3

    seq: list[int] = []
    c = 0
    while c < w and int(grid[0, c]) not in (BLACK, MAROON):
        seq.append(int(grid[0, c]))
        c += 1

    assert seq, "missing top-left color strip"
    assert len(seq) == len(set(seq)), "indicator colors must be distinct"
    return seq, c


def _find_starts(grid: np.ndarray, seq: list[int], strip_len: int) -> list[dict[str, Any]]:
    h, w = grid.shape
    strip_mask = np.zeros((h, w), dtype=bool)
    strip_mask[0, :strip_len] = True

    starts: list[dict[str, Any]] = []
    for color in seq:
        matches: list[dict[str, Any]] = []
        for r in range(h):
            for c in range(w):
                if not (r in (0, h - 1) or c in (0, w - 1)):
                    continue
                if strip_mask[r, c]:
                    continue
                if int(grid[r, c]) != int(color):
                    continue

                inward: list[tuple[int, int]] = []
                if r == 0:
                    inward.append((1, 0))
                if r == h - 1:
                    inward.append((-1, 0))
                if c == 0:
                    inward.append((0, 1))
                if c == w - 1:
                    inward.append((0, -1))

                good = []
                for dr, dc in inward:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) == int(color):
                        good.append((dr, dc))
                if len(good) != 1:
                    continue

                dr, dc = good[0]
                matches.append(
                    {
                        "color": int(color),
                        "border": (int(r), int(c)),
                        "inner": (int(r + dr), int(c + dc)),
                        "dir": (int(dr), int(dc)),
                        "order": _perimeter_index(int(r), int(c), h, w),
                    }
                )

        assert len(matches) == 1, f"expected one start for color {color}, found {len(matches)}"
        starts.append(matches[0])

    starts.sort(key=lambda item: int(item["order"]))
    return starts


def _find_portal_pairs(grid: np.ndarray) -> dict[tuple[int, int], tuple[int, int]]:
    points = [tuple(map(int, pt)) for pt in np.argwhere(grid == MAROON)]
    if not points:
        return {}

    neighbors: dict[tuple[int, int], list[tuple[int, int]]] = {pt: [] for pt in points}

    by_row: dict[int, list[int]] = {}
    by_col: dict[int, list[int]] = {}
    for r, c in points:
        by_row.setdefault(r, []).append(c)
        by_col.setdefault(c, []).append(r)

    for r, cols in by_row.items():
        cols.sort()
        for c1, c2 in zip(cols, cols[1:]):
            a = (int(r), int(c1))
            b = (int(r), int(c2))
            neighbors[a].append(b)
            neighbors[b].append(a)

    for c, rows in by_col.items():
        rows.sort()
        for r1, r2 in zip(rows, rows[1:]):
            a = (int(r1), int(c))
            b = (int(r2), int(c))
            neighbors[a].append(b)
            neighbors[b].append(a)

    assert all(len(nbrs) == 1 for nbrs in neighbors.values()), "portal pairing must be unambiguous"

    pairs: dict[tuple[int, int], tuple[int, int]] = {}
    for pt, nbrs in neighbors.items():
        pairs[pt] = nbrs[0]
    return pairs


def _simulate(grid: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    seq, strip_len = _indicator_sequence(g)
    starts = _find_starts(g, seq, strip_len)
    assert len(starts) == len(seq), "indicator colors must match the start set"

    portal_pairs = _find_portal_pairs(g)
    color_index = {int(color): idx for idx, color in enumerate(seq)}

    out = g.copy()
    path_infos: list[dict[str, Any]] = []

    for start in starts:
        cur_r, cur_c = map(int, start["inner"])
        dr, dc = map(int, start["dir"])
        cur_color = int(start["color"])
        painted = 0
        teleports = 0
        stop_reason = "boundary"
        seen: set[tuple[int, int, int]] = set()

        while True:
            state = (int(cur_r), int(cur_c), int(cur_color))
            if state in seen:
                stop_reason = "loop"
                break
            seen.add(state)

            nr, nc = int(cur_r + dr), int(cur_c + dc)
            if not (0 <= nr < h and 0 <= nc < w):
                stop_reason = "boundary"
                break

            cell = int(out[nr, nc])
            if cell == BLACK:
                out[nr, nc] = int(cur_color)
                cur_r, cur_c = nr, nc
                painted += 1
                continue

            if cell == MAROON:
                assert (nr, nc) in portal_pairs, "unpaired portal"
                cur_r, cur_c = portal_pairs[(nr, nc)]
                cur_color = int(seq[(color_index[cur_color] + 1) % len(seq)])
                teleports += 1
                continue

            stop_reason = "dynamic_block" if int(g[nr, nc]) == BLACK else "static_block"
            break

        path_infos.append(
            {
                "color": int(start["color"]),
                "painted": int(painted),
                "teleports": int(teleports),
                "stop_reason": stop_reason,
            }
        )

    meta = {
        "sequence": seq,
        "strip_len": int(strip_len),
        "starts": starts,
        "paths": path_infos,
        "portal_count": int(len(portal_pairs) // 2),
        "total_teleports": int(sum(info["teleports"] for info in path_infos)),
        "total_painted": int(sum(info["painted"] for info in path_infos)),
    }
    return out, meta


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend edge-started paths across the grid, teleporting through paired maroon portals.

    Input structure
    - Background is black.
    - The top-left corner contains a 1xN horizontal color strip whose distinct
      colors are exactly the path-start colors.
    - Each start is two same-color cells: one on the border and one immediately
      inward, which fixes the straight-line direction.
    - Interior maroon pixels form unambiguous portal pairs by consecutive
      alignment in one row or one column.

    Rule
    - Draw each path in clockwise border order, extending straight ahead.
    - Black cells are painted in the current path color.
    - When the next cell is a maroon portal, jump to its partner and continue
      from the next cell in the same direction; after every teleport, rotate to
      the next color in the strip cyclically.
    - A path stops at the grid boundary or before any non-black non-maroon cell.
    - Portals and the indicator strip remain unchanged.
    """
    out, _ = _simulate(np.asarray(grid, dtype=int))
    return out


def _start_spec(side: str, coord: int, color: int, h: int, w: int) -> dict[str, Any]:
    if side == "top":
        border = (0, int(coord))
        inner = (1, int(coord))
        direction = (1, 0)
    elif side == "bottom":
        border = (h - 1, int(coord))
        inner = (h - 2, int(coord))
        direction = (-1, 0)
    elif side == "left":
        border = (int(coord), 0)
        inner = (int(coord), 1)
        direction = (0, 1)
    else:
        assert side == "right"
        border = (int(coord), w - 1)
        inner = (int(coord), w - 2)
        direction = (0, -1)

    return {
        "side": side,
        "coord": int(coord),
        "color": int(color),
        "border": border,
        "inner": inner,
        "dir": direction,
    }


def _paint_start(grid: np.ndarray, spec: dict[str, Any]) -> None:
    color = int(spec["color"])
    br, bc = spec["border"]
    ir, ic = spec["inner"]
    grid[br, bc] = color
    grid[ir, ic] = color


def _dot_along(a: tuple[int, int], b: tuple[int, int], direction: tuple[int, int]) -> int:
    dr, dc = direction
    return int((a[0] - b[0]) * dr + (a[1] - b[1]) * dc)


def _entry_candidates(
    spec: dict[str, Any],
    shape: tuple[int, int],
    cross: tuple[int, int] | None,
    before_cross: bool | None,
) -> list[tuple[int, int]]:
    h, w = shape
    dr, dc = spec["dir"]
    r, c = spec["inner"]
    r += dr
    c += dc
    entries: list[tuple[int, int]] = []
    while 1 < r < h - 2 and 1 < c < w - 2:
        if cross is not None and before_cross is not None:
            rel = _dot_along((int(r), int(c)), cross, (int(dr), int(dc)))
            if before_cross and rel >= 0:
                r += dr
                c += dc
                continue
            if (not before_cross) and rel <= 0:
                r += dr
                c += dc
                continue
        entries.append((int(r), int(c)))
        r += dr
        c += dc
    return entries


def _choose_partner(
    entry: tuple[int, int],
    spec: dict[str, Any],
    shape: tuple[int, int],
    used_rows: set[int],
    used_cols: set[int],
    prefer_parallel: bool,
) -> tuple[tuple[int, int], str] | None:
    h, w = shape
    dr, dc = spec["dir"]
    er, ec = entry

    modes = ["parallel", "perpendicular"] if prefer_parallel else ["perpendicular", "parallel"]
    for mode in modes:
        if mode == "parallel":
            if dr != 0:
                if ec in used_cols or er in used_rows:
                    continue
                candidates = []
                for rr in range(2, h - 2):
                    if rr == er or rr in used_rows:
                        continue
                    if (rr - er) * dr <= 0:
                        continue
                    nr = rr + dr
                    if not (1 < nr < h - 1):
                        continue
                    candidates.append((int(rr), int(ec)))
                if candidates:
                    return random.choice(candidates), mode
            else:
                if er in used_rows or ec in used_cols:
                    continue
                candidates = []
                for cc in range(2, w - 2):
                    if cc == ec or cc in used_cols:
                        continue
                    if (cc - ec) * dc <= 0:
                        continue
                    nc = cc + dc
                    if not (1 < nc < w - 1):
                        continue
                    candidates.append((int(er), int(cc)))
                if candidates:
                    return random.choice(candidates), mode
        else:
            if dr != 0:
                if er in used_rows or ec in used_cols:
                    continue
                candidates = []
                for cc in range(2, w - 2):
                    if cc == ec or cc in used_cols:
                        continue
                    nr = er + dr
                    if not (1 <= nr < h - 1):
                        continue
                    candidates.append((int(er), int(cc)))
                if candidates:
                    return random.choice(candidates), mode
            else:
                if er in used_rows or ec in used_cols:
                    continue
                candidates = []
                for rr in range(2, h - 2):
                    if rr == er or rr in used_rows:
                        continue
                    nc = ec + dc
                    if not (1 <= nc < w - 1):
                        continue
                    candidates.append((int(rr), int(ec)))
                if candidates:
                    return random.choice(candidates), mode
    return None


def _reserve_pair(a: tuple[int, int], b: tuple[int, int], used_rows: set[int], used_cols: set[int]) -> None:
    ar, ac = a
    br, bc = b
    if ar == br:
        used_rows.add(int(ar))
        used_cols.add(int(ac))
        used_cols.add(int(bc))
    else:
        assert ac == bc
        used_cols.add(int(ac))
        used_rows.add(int(ar))
        used_rows.add(int(br))


def _place_targeted_pair(
    spec: dict[str, Any],
    shape: tuple[int, int],
    used_rows: set[int],
    used_cols: set[int],
    cross: tuple[int, int] | None = None,
    before_cross: bool | None = None,
    prefer_parallel: bool = False,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    entries = _entry_candidates(spec, shape, cross, before_cross)
    random.shuffle(entries)
    for entry in entries:
        partner_info = _choose_partner(entry, spec, shape, used_rows, used_cols, prefer_parallel)
        if partner_info is None:
            continue
        partner, _mode = partner_info
        _reserve_pair(entry, partner, used_rows, used_cols)
        return entry, partner
    return None


def _place_random_pair(
    shape: tuple[int, int],
    used_rows: set[int],
    used_cols: set[int],
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    h, w = shape
    modes = ["horizontal", "vertical"]
    random.shuffle(modes)
    for mode in modes:
        if mode == "horizontal":
            rows = [r for r in range(2, h - 2) if r not in used_rows]
            random.shuffle(rows)
            for r in rows:
                cols = [c for c in range(2, w - 2) if c not in used_cols]
                if len(cols) < 2:
                    continue
                c1, c2 = sorted(random.sample(cols, 2))
                if c2 - c1 < 2:
                    continue
                a = (int(r), int(c1))
                b = (int(r), int(c2))
                _reserve_pair(a, b, used_rows, used_cols)
                return a, b
        else:
            cols = [c for c in range(2, w - 2) if c not in used_cols]
            random.shuffle(cols)
            for c in cols:
                rows = [r for r in range(2, h - 2) if r not in used_rows]
                if len(rows) < 2:
                    continue
                r1, r2 = sorted(random.sample(rows, 2))
                if r2 - r1 < 2:
                    continue
                a = (int(r1), int(c))
                b = (int(r2), int(c))
                _reserve_pair(a, b, used_rows, used_cols)
                return a, b
    return None


def _extra_start(side: str, coord: int, occupied: set[tuple[int, int]], h: int, w: int) -> bool:
    spec = _start_spec(side, coord, color=1, h=h, w=w)
    for cell in (spec["border"], spec["inner"]):
        if cell in occupied:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    num_colors = random.choice([2, 3, 4])
    strip_colors = random.sample(PATH_COLORS, num_colors)

    h = random.randint(11, 17)
    w = random.randint(max(11, num_colors + 7), 17)
    if h < 11 or w < num_colors + 7:
        return None

    grid = np.zeros((h, w), dtype=int)
    grid[0, :num_colors] = np.array(strip_colors, dtype=int)

    color_order = strip_colors[:]
    random.shuffle(color_order)

    patterns = [
        ("top", "left"),
        ("top", "right"),
        ("right", "bottom"),
        ("bottom", "left"),
    ]
    early_side, late_side = random.choice(patterns)

    top_cols = list(range(num_colors + 1, w - 1))
    bottom_cols = list(range(2, w - 2))
    side_rows = list(range(2, h - 2))
    if not top_cols or not bottom_cols or not side_rows:
        return None

    occupied: set[tuple[int, int]] = set((0, c) for c in range(num_colors))

    if early_side == "top":
        early_coord = random.choice(top_cols)
        cross_col = int(early_coord)
    elif early_side == "bottom":
        early_coord = random.choice(bottom_cols)
        cross_col = int(early_coord)
    elif early_side == "left":
        early_coord = random.choice(side_rows)
        cross_row = int(early_coord)
    else:
        early_coord = random.choice(side_rows)
        cross_row = int(early_coord)

    if late_side == "left":
        late_coord = random.choice(side_rows)
        cross_row = int(late_coord)
    elif late_side == "right":
        late_coord = random.choice(side_rows)
        cross_row = int(late_coord)
    elif late_side == "top":
        late_coord = random.choice(top_cols)
        cross_col = int(late_coord)
    else:
        late_coord = random.choice(bottom_cols)
        cross_col = int(late_coord)

    if "cross_row" not in locals() or "cross_col" not in locals():
        return None
    cross = (int(cross_row), int(cross_col))

    starts: list[dict[str, Any]] = []
    starts.append(_start_spec(early_side, early_coord, color_order[0], h, w))
    starts.append(_start_spec(late_side, late_coord, color_order[1], h, w))

    for spec in starts:
        for cell in (spec["border"], spec["inner"]):
            if cell in occupied:
                return None
        occupied.add(spec["border"])
        occupied.add(spec["inner"])

    side_pool = ["top", "right", "bottom", "left"]
    for color in color_order[2:]:
        candidates: list[tuple[str, int]] = []
        random.shuffle(side_pool)
        for side in side_pool:
            if side == "top":
                coords = top_cols[:]
            elif side == "bottom":
                coords = bottom_cols[:]
            else:
                coords = side_rows[:]
            random.shuffle(coords)
            for coord in coords:
                if _extra_start(side, int(coord), occupied, h, w):
                    candidates.append((side, int(coord)))
                    break
        if not candidates:
            return None
        side, coord = random.choice(candidates)
        spec = _start_spec(side, coord, color, h, w)
        starts.append(spec)
        occupied.add(spec["border"])
        occupied.add(spec["inner"])

    for spec in starts:
        _paint_start(grid, spec)

    require_dynamic_block = bool(random.getrandbits(1))
    before_cross = not require_dynamic_block
    prefer_parallel = bool(random.getrandbits(1))

    used_rows: set[int] = set()
    used_cols: set[int] = set()
    portal_pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []

    primary_pair = _place_targeted_pair(
        starts[0],
        grid.shape,
        used_rows,
        used_cols,
        cross=cross,
        before_cross=before_cross,
        prefer_parallel=prefer_parallel,
    )
    if primary_pair is None:
        return None
    portal_pairs.append(primary_pair)

    if len(starts) > 2 and random.random() < 0.6:
        secondary = random.choice(starts[1:])
        extra_pair = _place_targeted_pair(
            secondary,
            grid.shape,
            used_rows,
            used_cols,
            cross=None,
            before_cross=None,
            prefer_parallel=bool(random.getrandbits(1)),
        )
        if extra_pair is not None:
            portal_pairs.append(extra_pair)

    if random.random() < 0.5:
        spare_pair = _place_random_pair(grid.shape, used_rows, used_cols)
        if spare_pair is not None:
            portal_pairs.append(spare_pair)

    for a, b in portal_pairs:
        if a in occupied or b in occupied:
            return None
        grid[a] = MAROON
        grid[b] = MAROON

    try:
        out, meta = _simulate(grid)
    except AssertionError:
        return None

    if meta["portal_count"] != len(portal_pairs):
        return None
    if meta["total_teleports"] < 1:
        return None
    if meta["total_painted"] < num_colors + 2:
        return None
    if np.array_equal(out, grid):
        return None

    if require_dynamic_block:
        if not any(info["stop_reason"] == "dynamic_block" for info in meta["paths"]):
            return None

    if not any(info["teleports"] > 0 and info["painted"] > 0 for info in meta["paths"]):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid portal-path ARC input")
