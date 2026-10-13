import math
from collections import deque

import numpy as np


BG = 0
MAROON = 9
ORTHO = [(-1, 0), (1, 0), (0, -1), (0, 1)]
LAYOUTS = [(1, 2), (1, 3), (1, 4), (2, 1), (3, 1), (4, 1), (2, 2), (2, 3), (3, 2)]
BODY_COLORS = [1, 2, 3, 4, 5, 6, 7, 8]


def _dominant_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for dr, dc in ORTHO:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(pts)
    return out


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _single_or_none(values: list[int]) -> int | None:
    if not values:
        return None
    assert len(values) == 1, "each side should have at most one interface mark"
    return int(values[0])


def _piece_side_data(
    grid: np.ndarray,
    body_set: set[tuple[int, int]],
    r0: int,
    c0: int,
    h: int,
    w: int,
) -> dict[str, dict[str, int | None]]:
    r1 = r0 + h - 1
    c1 = c0 + w - 1

    top_notch = [k for k in range(w) if (r0, c0 + k) not in body_set]
    bot_notch = [k for k in range(w) if (r1, c0 + k) not in body_set]
    left_notch = [k for k in range(h) if (r0 + k, c0) not in body_set]
    right_notch = [k for k in range(h) if (r0 + k, c1) not in body_set]

    top_pro = [k for k in range(w) if r0 > 0 and int(grid[r0 - 1, c0 + k]) == MAROON]
    bot_pro = [k for k in range(w) if r1 + 1 < grid.shape[0] and int(grid[r1 + 1, c0 + k]) == MAROON]
    left_pro = [k for k in range(h) if c0 > 0 and int(grid[r0 + k, c0 - 1]) == MAROON]
    right_pro = [k for k in range(h) if c1 + 1 < grid.shape[1] and int(grid[r0 + k, c1 + 1]) == MAROON]

    sides = {
        "N": {"length": w, "notch": _single_or_none(top_notch), "protrusion": _single_or_none(top_pro)},
        "S": {"length": w, "notch": _single_or_none(bot_notch), "protrusion": _single_or_none(bot_pro)},
        "W": {"length": h, "notch": _single_or_none(left_notch), "protrusion": _single_or_none(left_pro)},
        "E": {"length": h, "notch": _single_or_none(right_notch), "protrusion": _single_or_none(right_pro)},
    }
    for data in sides.values():
        assert not (data["notch"] is not None and data["protrusion"] is not None), "one side cannot be both notch and protrusion"
    return sides


def _extract_pieces(grid: np.ndarray, bg: int) -> list[dict]:
    pieces: list[dict] = []
    for pts in _components(grid != bg):
        colors = {int(grid[r, c]) for r, c in pts}
        body_colors = sorted(colors - {bg, MAROON})
        assert len(body_colors) == 1, "each component should contain exactly one body color"
        color = int(body_colors[0])
        body_pts = [(r, c) for r, c in pts if int(grid[r, c]) == color]
        r0, c0, r1, c1 = _bbox(body_pts)
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        body_set = set(body_pts)
        pieces.append(
            {
                "color": color,
                "h": h,
                "w": w,
                "sides": _piece_side_data(grid, body_set, r0, c0, h, w),
            }
        )
    assert len(pieces) >= 2, "expected multiple pieces"
    return pieces


def _store_unique(mapping: dict[tuple[int, int], int], key: tuple[int, int], piece_idx: int) -> None:
    assert key not in mapping, "interface signature should identify a single side"
    mapping[key] = int(piece_idx)


def _match_adjacencies(pieces: list[dict]) -> list[dict[str, int | None]]:
    neighbors = [{d: None for d in "NSEW"} for _ in pieces]

    e_pro: dict[tuple[int, int], int] = {}
    e_not: dict[tuple[int, int], int] = {}
    w_pro: dict[tuple[int, int], int] = {}
    w_not: dict[tuple[int, int], int] = {}
    s_pro: dict[tuple[int, int], int] = {}
    s_not: dict[tuple[int, int], int] = {}
    n_pro: dict[tuple[int, int], int] = {}
    n_not: dict[tuple[int, int], int] = {}

    for i, piece in enumerate(pieces):
        east = piece["sides"]["E"]
        west = piece["sides"]["W"]
        south = piece["sides"]["S"]
        north = piece["sides"]["N"]

        if east["protrusion"] is not None:
            _store_unique(e_pro, (int(east["length"]), int(east["protrusion"])), i)
        if east["notch"] is not None:
            _store_unique(e_not, (int(east["length"]), int(east["notch"])), i)
        if west["protrusion"] is not None:
            _store_unique(w_pro, (int(west["length"]), int(west["protrusion"])), i)
        if west["notch"] is not None:
            _store_unique(w_not, (int(west["length"]), int(west["notch"])), i)
        if south["protrusion"] is not None:
            _store_unique(s_pro, (int(south["length"]), int(south["protrusion"])), i)
        if south["notch"] is not None:
            _store_unique(s_not, (int(south["length"]), int(south["notch"])), i)
        if north["protrusion"] is not None:
            _store_unique(n_pro, (int(north["length"]), int(north["protrusion"])), i)
        if north["notch"] is not None:
            _store_unique(n_not, (int(north["length"]), int(north["notch"])), i)

    assert set(e_pro) == set(w_not), "every east protrusion should meet one west notch"
    assert set(e_not) == set(w_pro), "every east notch should meet one west protrusion"
    assert set(s_pro) == set(n_not), "every south protrusion should meet one north notch"
    assert set(s_not) == set(n_pro), "every south notch should meet one north protrusion"

    for key, a in e_pro.items():
        b = w_not[key]
        neighbors[a]["E"] = b
        neighbors[b]["W"] = a
    for key, a in e_not.items():
        b = w_pro[key]
        neighbors[a]["E"] = b
        neighbors[b]["W"] = a
    for key, a in s_pro.items():
        b = n_not[key]
        neighbors[a]["S"] = b
        neighbors[b]["N"] = a
    for key, a in s_not.items():
        b = n_pro[key]
        neighbors[a]["S"] = b
        neighbors[b]["N"] = a

    return neighbors


def _assemble(pieces: list[dict], neighbors: list[dict[str, int | None]]) -> np.ndarray:
    coords: dict[int, tuple[int, int]] = {0: (0, 0)}
    q = deque([0])
    while q:
        i = q.popleft()
        r, c = coords[i]
        piece = pieces[i]

        east = neighbors[i]["E"]
        if east is not None:
            cand = (r, c + int(piece["w"]))
            if east in coords:
                assert coords[east] == cand, "inconsistent east placement"
            else:
                coords[east] = cand
                q.append(east)

        west = neighbors[i]["W"]
        if west is not None:
            cand = (r, c - int(pieces[west]["w"]))
            if west in coords:
                assert coords[west] == cand, "inconsistent west placement"
            else:
                coords[west] = cand
                q.append(west)

        south = neighbors[i]["S"]
        if south is not None:
            cand = (r + int(piece["h"]), c)
            if south in coords:
                assert coords[south] == cand, "inconsistent south placement"
            else:
                coords[south] = cand
                q.append(south)

        north = neighbors[i]["N"]
        if north is not None:
            cand = (r - int(pieces[north]["h"]), c)
            if north in coords:
                assert coords[north] == cand, "inconsistent north placement"
            else:
                coords[north] = cand
                q.append(north)

    assert len(coords) == len(pieces), "all pieces should belong to one assembly"

    min_r = min(r for r, _ in coords.values())
    min_c = min(c for _, c in coords.values())
    norm = {i: (r - min_r, c - min_c) for i, (r, c) in coords.items()}

    out_h = max(norm[i][0] + int(pieces[i]["h"]) for i in range(len(pieces)))
    out_w = max(norm[i][1] + int(pieces[i]["w"]) for i in range(len(pieces)))
    out = np.full((out_h, out_w), -1, dtype=int)

    for i, piece in enumerate(pieces):
        r, c = norm[i]
        h = int(piece["h"])
        w = int(piece["w"])
        block = out[r : r + h, c : c + w]
        assert np.all(block == -1), "assembled bodies should not overlap"
        block[:] = int(piece["color"])

    assert np.all(out != -1), "assembled pieces should fill a solid rectangle"
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble scattered keyed fragments into their hidden full rectangle.

    Input structure:
    - The background is black.
    - Several disconnected fragments are scattered on the grid; each fragment has one unique body color.
    - A fragment's body is a rectangle, but some side cells may be missing as 1-pixel notches.
    - Some sides also carry 1-pixel maroon protrusions just outside the body; a protrusion and a notch match when they sit on opposite sides with the same offset along equally long edges.

    Rule:
    - For every fragment, recover its full body rectangle and read the notch/protrusion signature on each of the four sides.
    - Match complementary east-west and north-south side signatures to determine which fragments are adjacent in the final assembly.
    - Translate the fragments so their body rectangles tile one tight rectangle with no gaps.
    - Output that completed rectangle using only each fragment's body color; maroon connector pixels disappear and filled notch cells take the owning fragment's body color.

    Key insight:
    - The scattered positions are irrelevant: the maroon protrusions and missing edge cells act as lock-and-key side codes that uniquely determine the 2D packing.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _dominant_color(g)
    pieces = _extract_pieces(g, bg)
    neighbors = _match_adjacencies(pieces)
    return _assemble(pieces, neighbors)


def _sample_distinct(rng: np.random.Generator, lo: int, hi: int, count: int) -> list[int]:
    values = np.arange(lo, hi + 1)
    assert len(values) >= count, "not enough distinct sizes available"
    pick = rng.choice(values, size=count, replace=False)
    return [int(x) for x in pick]


def _blank_piece_spec(h: int, w: int, color: int) -> dict:
    return {
        "h": int(h),
        "w": int(w),
        "color": int(color),
        "N": {"notch": None, "protrusion": None},
        "S": {"notch": None, "protrusion": None},
        "W": {"notch": None, "protrusion": None},
        "E": {"notch": None, "protrusion": None},
    }


def _piece_patch(spec: dict) -> np.ndarray:
    h = int(spec["h"])
    w = int(spec["w"])
    top = 1 if spec["N"]["protrusion"] is not None else 0
    bottom = 1 if spec["S"]["protrusion"] is not None else 0
    left = 1 if spec["W"]["protrusion"] is not None else 0
    right = 1 if spec["E"]["protrusion"] is not None else 0

    patch = np.full((h + top + bottom, w + left + right), BG, dtype=int)
    r0 = top
    c0 = left
    patch[r0 : r0 + h, c0 : c0 + w] = int(spec["color"])

    if spec["N"]["notch"] is not None:
        patch[r0, c0 + int(spec["N"]["notch"])] = BG
    if spec["S"]["notch"] is not None:
        patch[r0 + h - 1, c0 + int(spec["S"]["notch"])] = BG
    if spec["W"]["notch"] is not None:
        patch[r0 + int(spec["W"]["notch"]), c0] = BG
    if spec["E"]["notch"] is not None:
        patch[r0 + int(spec["E"]["notch"]), c0 + w - 1] = BG

    if spec["N"]["protrusion"] is not None:
        patch[r0 - 1, c0 + int(spec["N"]["protrusion"])] = MAROON
    if spec["S"]["protrusion"] is not None:
        patch[r0 + h, c0 + int(spec["S"]["protrusion"])] = MAROON
    if spec["W"]["protrusion"] is not None:
        patch[r0 + int(spec["W"]["protrusion"]), c0 - 1] = MAROON
    if spec["E"]["protrusion"] is not None:
        patch[r0 + int(spec["E"]["protrusion"]), c0 + w] = MAROON

    return patch


def _mask_connected(mask: np.ndarray) -> bool:
    points = np.argwhere(mask)
    if len(points) == 0:
        return False
    seen = {tuple(map(int, points[0]))}
    q = deque([tuple(map(int, points[0]))])
    while q:
        r, c = q.popleft()
        for dr, dc in ORTHO:
            nr, nc = r + dr, c + dc
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == int(mask.sum())


def _body_connected(spec: dict) -> bool:
    mask = np.ones((int(spec["h"]), int(spec["w"])), dtype=bool)
    if spec["N"]["notch"] is not None:
        mask[0, int(spec["N"]["notch"])] = False
    if spec["S"]["notch"] is not None:
        mask[-1, int(spec["S"]["notch"])] = False
    if spec["W"]["notch"] is not None:
        mask[int(spec["W"]["notch"]), 0] = False
    if spec["E"]["notch"] is not None:
        mask[int(spec["E"]["notch"]), -1] = False
    return _mask_connected(mask)


def _expected_output(rows: list[int], cols: list[int], specs: list[list[dict]]) -> np.ndarray:
    out = np.full((sum(rows), sum(cols)), BG, dtype=int)
    rr = 0
    for i, h in enumerate(rows):
        cc = 0
        for j, w in enumerate(cols):
            out[rr : rr + h, cc : cc + w] = int(specs[i][j]["color"])
            cc += w
        rr += h
    return out


def _scatter_patches(rng: np.random.Generator, patches: list[np.ndarray]) -> np.ndarray | None:
    max_h = max(int(p.shape[0]) for p in patches)
    max_w = max(int(p.shape[1]) for p in patches)
    total = sum(int((p.shape[0] + 2) * (p.shape[1] + 2)) for p in patches)
    side = int(math.ceil(math.sqrt(total)))
    h_lo = max(18, max_h + 2, side - 2)
    w_lo = max(18, max_w + 2, side - 2)
    if h_lo > 30 or w_lo > 30:
        return None

    h = int(rng.integers(h_lo, 31))
    w = int(rng.integers(w_lo, 31))
    grid = np.full((h, w), BG, dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    order = list(rng.permutation(len(patches)))
    for idx in order:
        patch = patches[idx]
        ph, pw = map(int, patch.shape)
        if ph + 2 > h or pw + 2 > w:
            return None
        placed = False
        row_hi = h - ph - 1
        col_hi = w - pw - 1
        if row_hi < 1 or col_hi < 1:
            return None
        for _ in range(300):
            r = int(rng.integers(1, row_hi + 1))
            c = int(rng.integers(1, col_hi + 1))
            if np.any(reserved[r - 1 : r + ph + 1, c - 1 : c + pw + 1]):
                continue
            grid[r : r + ph, c : c + pw] = np.where(patch != BG, patch, grid[r : r + ph, c : c + pw])
            reserved[r - 1 : r + ph + 1, c - 1 : c + pw + 1] = True
            placed = True
            break
        if not placed:
            return None

    return grid


def _generate_candidate(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    rows_n, cols_n = LAYOUTS[int(rng.integers(len(LAYOUTS)))]
    piece_count = rows_n * cols_n
    colors = [int(x) for x in rng.choice(np.array(BODY_COLORS), size=piece_count, replace=False)]

    min_h = max(4, cols_n + 1)
    min_w = max(4, rows_n + 1)
    row_heights = _sample_distinct(rng, min_h, 8, rows_n)
    col_widths = _sample_distinct(rng, min_w, 8, cols_n)

    specs: list[list[dict]] = []
    color_idx = 0
    for i in range(rows_n):
        row_specs = []
        for j in range(cols_n):
            row_specs.append(_blank_piece_spec(row_heights[i], col_widths[j], colors[color_idx]))
            color_idx += 1
        specs.append(row_specs)

    for i in range(rows_n):
        offsets = [int(x) for x in rng.choice(np.arange(1, row_heights[i] - 1), size=cols_n - 1, replace=False)]
        for j, offset in enumerate(offsets):
            left = specs[i][j]
            right = specs[i][j + 1]
            if bool(rng.integers(0, 2)):
                left["E"]["protrusion"] = offset
                right["W"]["notch"] = offset
            else:
                left["E"]["notch"] = offset
                right["W"]["protrusion"] = offset

    for j in range(cols_n):
        offsets = [int(x) for x in rng.choice(np.arange(1, col_widths[j] - 1), size=rows_n - 1, replace=False)]
        for i, offset in enumerate(offsets):
            top = specs[i][j]
            bottom = specs[i + 1][j]
            if bool(rng.integers(0, 2)):
                top["S"]["protrusion"] = offset
                bottom["N"]["notch"] = offset
            else:
                top["S"]["notch"] = offset
                bottom["N"]["protrusion"] = offset

    for row in specs:
        for spec in row:
            if not _body_connected(spec):
                return None

    patches = [_piece_patch(specs[i][j]) for i in range(rows_n) for j in range(cols_n)]
    grid = _scatter_patches(rng, patches)
    if grid is None:
        return None

    expected = _expected_output(row_heights, col_widths, specs)
    if not np.array_equal(main(grid), expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(400):
        candidate = _generate_candidate(rng)
        if candidate is None:
            continue
        grid, _ = candidate
        assert max(grid.shape) <= 30
        return grid
    raise RuntimeError("failed to generate a valid jigsaw assembly sample")
