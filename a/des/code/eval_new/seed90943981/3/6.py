from __future__ import annotations

import itertools
import random
from collections import deque

import numpy as np


SIDES = ("top", "right", "bottom", "left")
GEN_BG = 0
GEN_WALL = 3
PALETTE_CHOICES = (1, 2, 4, 5, 6, 7, 8, 9)
GRID_H_RANGE = (19, 29)
GRID_W_RANGE = (19, 29)
ROOM_COUNT_RANGE = (2, 4)
ROOM_SIDE_RANGE = (4, 7)
GEN_RETRIES = 400
ROOM_SHAPE_RETRIES = 120
PLACEMENT_RETRIES = 120
ROOM_CLEARANCE = 1
PALETTE_GAP_ROWS = 1


def _most_common_color(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    order = np.argsort(-cnts)
    assert len(order) == 1 or cnts[order[0]] > cnts[order[1]], "expected a unique background color"
    return int(vals[order[0]])


def _neighbors(r: int, c: int, connectivity: int) -> list[tuple[int, int]]:
    if connectivity == 4:
        return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
    return [
        (r - 1, c - 1),
        (r - 1, c),
        (r - 1, c + 1),
        (r, c - 1),
        (r, c + 1),
        (r + 1, c - 1),
        (r + 1, c),
        (r + 1, c + 1),
    ]


def _components(mask: np.ndarray, connectivity: int) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors(cr, cc, connectivity):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_palette_row(grid: np.ndarray, bg: int) -> tuple[int, list[int]]:
    h, _w = grid.shape
    for r in range(h - 1, -1, -1):
        cols = np.flatnonzero(grid[r] != bg)
        if len(cols) != 4:
            continue
        colors = [int(grid[r, c]) for c in cols]
        if len(set(colors)) != 4:
            continue
        return int(r), colors
    raise AssertionError("expected a bottom palette row with four distinct colors")


def _gap_candidates(
    grid: np.ndarray, wall_mask: np.ndarray, bg: int, row_limit: int
) -> list[tuple[int, int]]:
    _h, w = wall_mask.shape
    gaps: list[tuple[int, int]] = []
    for r in range(row_limit):
        for c in range(w):
            if wall_mask[r, c] or int(grid[r, c]) != bg:
                continue
            horizontal_bridge = 0 < c < w - 1 and wall_mask[r, c - 1] and wall_mask[r, c + 1]
            vertical_bridge = 0 < r < row_limit - 1 and wall_mask[r - 1, c] and wall_mask[r + 1, c]
            if horizontal_bridge ^ vertical_bridge:
                gaps.append((r, c))
    return gaps


def _rooms_from_wall_mask(
    grid: np.ndarray, wall_mask: np.ndarray, bg: int, row_limit: int
) -> list[tuple[list[tuple[int, int]], dict[str, int], tuple[int, int, int, int]]]:
    gaps = _gap_candidates(grid, wall_mask, bg, row_limit)
    sealed = wall_mask[:row_limit, :].copy()
    for r, c in gaps:
        sealed[r, c] = True

    rooms: list[tuple[list[tuple[int, int]], dict[str, int], tuple[int, int, int, int]]] = []
    for comp in _components(~sealed, connectivity=4):
        if any(r == 0 or r == row_limit - 1 or c == 0 or c == sealed.shape[1] - 1 for r, c in comp):
            continue
        comp_set = set(comp)
        side_counts = {side: 0 for side in SIDES}
        for r, c in gaps:
            touching_sides: list[str] = []
            if (r + 1, c) in comp_set:
                touching_sides.append("top")
            if (r, c - 1) in comp_set:
                touching_sides.append("right")
            if (r - 1, c) in comp_set:
                touching_sides.append("bottom")
            if (r, c + 1) in comp_set:
                touching_sides.append("left")
            if len(touching_sides) == 1:
                side_counts[touching_sides[0]] += 1
            else:
                assert len(touching_sides) == 0, "candidate gap must touch at most one room"

        gap_total = sum(side_counts.values())
        if gap_total == 0:
            continue
        assert 1 <= gap_total <= 4, "each room must expose one to four detectable gaps"
        rooms.append((comp, side_counts, _bbox(comp)))

    assert rooms, "expected at least one enclosed room"
    return rooms


def _extract_structure(
    grid: np.ndarray,
) -> tuple[int, int, list[int], int, list[tuple[list[tuple[int, int]], dict[str, int], tuple[int, int, int, int]]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _most_common_color(g)
    palette_row, palette = _find_palette_row(g, bg)

    trimmed = g.copy()
    trimmed[palette_row, :] = bg
    vals, cnts = np.unique(trimmed, return_counts=True)
    non_bg = [(int(v), int(c)) for v, c in zip(vals, cnts, strict=False) if int(v) != bg]
    assert non_bg, "expected wall color above the palette row"
    non_bg.sort(key=lambda item: (-item[1], item[0]))
    wall_color = non_bg[0][0]

    wall_mask = trimmed == wall_color
    rooms = _rooms_from_wall_mask(g, wall_mask, bg, palette_row)
    return bg, palette_row, palette, wall_color, rooms


def _major_side(side_counts: dict[str, int]) -> str:
    best = SIDES[0]
    best_count = side_counts[best]
    for side in SIDES[1:]:
        if side_counts[side] > best_count:
            best = side
            best_count = side_counts[side]
    return best


def _paint_stripes(
    out: np.ndarray, interior: list[tuple[int, int]], fill_color: int, side: str
) -> None:
    if side in ("top", "bottom"):
        stripe_rows = set(sorted({r for r, _ in interior})[::2])
        for r, c in interior:
            if r in stripe_rows:
                out[r, c] = fill_color
    else:
        stripe_cols = set(sorted({c for _, c in interior})[::2])
        for r, c in interior:
            if c in stripe_cols:
                out[r, c] = fill_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each gapped wall-bounded region with stripes chosen from a bottom color pool.

    Input structure:
    - bg = black. Several green wall components form almost-closed regions, with
      each region having one to four single-cell gaps on its boundary.
    - The bottom of the grid contains a palette row of four distinct single-pixel
      colors ordered from left to right.

    Rule:
    - For each wall component, detect its gap cells and temporarily seal them to
      recover the enclosed interior region.
    - Use the number of gaps N to select the N-th palette color.
    - Count which side of the region the gaps belong to; ties are broken by the
      fixed priority top -> right -> bottom -> left.
    - A top/bottom winner produces horizontal stripes, and a left/right winner
      produces vertical stripes. Starting from the first interior row or column,
      fill every other row or column of the interior with the selected color.
    - Keep the walls, gaps, and palette row unchanged.

    Key insight:
    - The same gap pixels encode two independent signals: their count chooses the
      fill color from the palette, and their side distribution chooses the stripe
      orientation.
    """
    g = np.asarray(grid, dtype=int)
    _bg, _palette_row, palette, _wall_color, rooms = _extract_structure(g)
    out = g.copy()

    for interior, side_counts, _bbox_ in rooms:
        n_gaps = sum(side_counts.values())
        assert 1 <= n_gaps <= 4, "gap count must match the four-color palette"
        fill_color = int(palette[n_gaps - 1])
        side = _major_side(side_counts)
        _paint_stripes(out, interior, fill_color, side)

    return out


def _count_major_side(counts: tuple[int, int, int, int]) -> str:
    best_idx = 0
    best_value = counts[0]
    for idx in range(1, 4):
        if counts[idx] > best_value:
            best_idx = idx
            best_value = counts[idx]
    return SIDES[best_idx]


def _is_tied(counts: tuple[int, int, int, int]) -> bool:
    max_count = max(counts)
    return sum(v == max_count for v in counts) > 1


COUNT_PATTERNS: dict[tuple[int, str], list[tuple[int, int, int, int]]] = {}
for total in range(1, 5):
    for counts in itertools.product(range(total + 1), repeat=4):
        if sum(counts) != total:
            continue
        COUNT_PATTERNS.setdefault((total, _count_major_side(counts)), []).append(counts)


def _row_spans(
    inner_h: int, inner_w: int, tl: int, tr: int, bl: int, br: int
) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for r in range(inner_h):
        if inner_h == 1:
            left = tl
            right = tr
        else:
            left = int(round(tl + (bl - tl) * r / (inner_h - 1)))
            right = int(round(tr + (br - tr) * r / (inner_h - 1)))
        width = inner_w - left - right
        if width < 3:
            return []
        spans.append((left, inner_w - right))
    return spans


def _build_room_geometry(
    inner_h: int, inner_w: int, tl: int, tr: int, bl: int, br: int
) -> dict[str, object] | None:
    room_h = inner_h + 2
    room_w = inner_w + 2
    interior = {
        (r, c)
        for r in range(1, room_h - 1)
        for c in range(1, room_w - 1)
    }
    wall = {
        (r, c)
        for r in range(room_h)
        for c in range(room_w)
        if r in (0, room_h - 1) or c in (0, room_w - 1)
    }

    gap_candidates: dict[str, list[tuple[int, int]]] = {side: [] for side in SIDES}
    for c in range(1, room_w - 1):
        gap_candidates["top"].append((0, c))
        gap_candidates["bottom"].append((room_h - 1, c))
    for r in range(1, room_h - 1):
        gap_candidates["left"].append((r, 0))
        gap_candidates["right"].append((r, room_w - 1))

    return {
        "shape": (room_h, room_w),
        "interior": interior,
        "wall": wall,
        "gap_candidates": gap_candidates,
    }


def _pick_nonadjacent(
    candidates: list[tuple[int, int]], count: int, rng: random.Random
) -> list[tuple[int, int]] | None:
    if count == 0:
        return []
    valid: list[tuple[tuple[int, int], ...]] = []
    for combo in itertools.combinations(candidates, count):
        ok = True
        for i in range(len(combo)):
            for j in range(i + 1, len(combo)):
                if abs(combo[i][0] - combo[j][0]) + abs(combo[i][1] - combo[j][1]) == 1:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            valid.append(combo)
    if not valid:
        return None
    return list(rng.choice(valid))


def _sample_gap_pattern(
    gap_candidates: dict[str, list[tuple[int, int]]],
    total_gaps: int,
    major_side: str,
    want_tie: bool,
    rng: random.Random,
) -> set[tuple[int, int]] | None:
    patterns = COUNT_PATTERNS[(total_gaps, major_side)]
    feasible: list[tuple[int, int, int, int]] = []
    tied: list[tuple[int, int, int, int]] = []
    untied: list[tuple[int, int, int, int]] = []
    for pattern in patterns:
        chosen: list[tuple[int, int]] = []
        ok = True
        for side, need in zip(SIDES, pattern, strict=False):
            picked = _pick_nonadjacent(gap_candidates[side], need, rng)
            if picked is None:
                ok = False
                break
            chosen.extend(picked)
        if ok:
            feasible.append(pattern)
            if _is_tied(pattern):
                tied.append(pattern)
            else:
                untied.append(pattern)

    if not feasible:
        return None
    if want_tie and tied:
        pattern = rng.choice(tied)
    elif not want_tie and untied:
        pattern = rng.choice(untied)
    else:
        pattern = rng.choice(feasible)

    gaps: set[tuple[int, int]] = set()
    for side, need in zip(SIDES, pattern, strict=False):
        picked = _pick_nonadjacent(gap_candidates[side], need, rng)
        assert picked is not None
        gaps.update(picked)
    return gaps


def _sample_room(rng: random.Random) -> dict[str, object] | None:
    total_gaps = rng.randint(1, 4)
    major_side = rng.choice(SIDES)
    want_tie = total_gaps >= 2 and major_side != "left" and rng.random() < 0.35

    for _ in range(ROOM_SHAPE_RETRIES):
        inner_h = rng.randint(*ROOM_SIDE_RANGE)
        inner_w = rng.randint(*ROOM_SIDE_RANGE)
        geometry = _build_room_geometry(inner_h, inner_w, 0, 0, 0, 0)
        if geometry is None:
            continue
        gaps = _sample_gap_pattern(
            geometry["gap_candidates"], total_gaps, major_side, want_tie, rng
        )
        if gaps is None:
            continue

        room_h, room_w = geometry["shape"]
        room_grid = np.full((room_h, room_w), GEN_BG, dtype=int)
        wall = set(geometry["wall"]) - gaps
        for r, c in wall:
            room_grid[r, c] = GEN_WALL

        try:
            rooms = _rooms_from_wall_mask(room_grid, room_grid == GEN_WALL, GEN_BG, room_grid.shape[0])
        except AssertionError:
            continue
        if len(rooms) != 1:
            continue
        _interior, side_counts, _bbox_ = rooms[0]
        if sum(side_counts.values()) != total_gaps:
            continue
        if _major_side(side_counts) != major_side:
            continue

        geometry["grid"] = room_grid
        geometry["gaps"] = gaps
        geometry["wall"] = wall
        geometry["gap_total"] = total_gaps
        geometry["major_side"] = major_side
        return geometry
    return None


def _sample_palette_columns(width: int, rng: random.Random) -> list[int] | None:
    positions = list(range(1, width - 1))
    valid = [
        cols
        for cols in itertools.combinations(positions, 4)
        if min(b - a for a, b in zip(cols, cols[1:], strict=False)) >= 2
    ]
    if not valid:
        return None
    return list(rng.choice(valid))


def _place_rooms(
    grid: np.ndarray, rooms: list[dict[str, object]], usable_h: int, rng: random.Random
) -> bool:
    h, w = grid.shape
    blocked = np.zeros((h, w), dtype=bool)
    rooms = sorted(rooms, key=lambda room: room["shape"][0] * room["shape"][1], reverse=True)
    for room in rooms:
        room_h, room_w = room["shape"]
        min_r = 1
        max_r = usable_h - room_h - 1
        min_c = 1
        max_c = w - room_w - 1
        if max_r < min_r or max_c < min_c:
            return False

        placed = False
        for _ in range(PLACEMENT_RETRIES):
            top = rng.randint(min_r, max_r)
            left = rng.randint(min_c, max_c)
            r0 = max(0, top - ROOM_CLEARANCE)
            r1 = min(h, top + room_h + ROOM_CLEARANCE)
            c0 = max(0, left - ROOM_CLEARANCE)
            c1 = min(w, left + room_w + ROOM_CLEARANCE)
            if blocked[r0:r1, c0:c1].any():
                continue
            room_grid = room["grid"]
            grid[top : top + room_h, left : left + room_w] = room_grid
            blocked[r0:r1, c0:c1] = True
            placed = True
            break
        if not placed:
            return False
    return True


def _structure_is_valid(grid: np.ndarray, expected_rooms: int) -> bool:
    try:
        bg, palette_row, palette, wall_color, rooms = _extract_structure(grid)
        out = main(grid)
    except Exception:
        return False

    if bg != GEN_BG or palette_row != grid.shape[0] - 1:
        return False
    if wall_color != GEN_WALL:
        return False
    if len(palette) != 4 or len(set(palette)) != 4:
        return False
    if len(rooms) != expected_rooms:
        return False
    if out.shape != grid.shape:
        return False
    if np.array_equal(out, grid):
        return False
    return True


def generate_input() -> np.ndarray:
    rng = random.Random()

    for _ in range(GEN_RETRIES):
        height = rng.randint(*GRID_H_RANGE)
        width = rng.randint(*GRID_W_RANGE)
        usable_h = height - (PALETTE_GAP_ROWS + 1)
        if usable_h < 8:
            continue

        grid = np.full((height, width), GEN_BG, dtype=int)
        max_rooms = ROOM_COUNT_RANGE[1] if height >= 24 and width >= 24 else ROOM_COUNT_RANGE[1] - 1
        room_count = rng.randint(ROOM_COUNT_RANGE[0], max_rooms)
        rooms: list[dict[str, object]] = []
        ok = True
        for _room_idx in range(room_count):
            room = _sample_room(rng)
            if room is None:
                ok = False
                break
            rooms.append(room)
        if not ok:
            continue

        if not _place_rooms(grid, rooms, usable_h, rng):
            continue

        palette = rng.sample([c for c in PALETTE_CHOICES if c != GEN_WALL], 4)
        palette_cols = _sample_palette_columns(width, rng)
        if palette_cols is None:
            continue
        for c, color in zip(palette_cols, palette, strict=False):
            grid[height - 1, c] = color

        counts = np.bincount(grid.ravel(), minlength=10)
        if int(np.argmax(counts)) != GEN_BG:
            continue
        if counts[GEN_WALL] <= 4:
            continue

        if _structure_is_valid(grid, room_count):
            return grid

    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
