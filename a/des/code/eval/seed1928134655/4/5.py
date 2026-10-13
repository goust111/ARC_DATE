from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import math
import random

import numpy as np


BG = 0
GREEN = 3
USABLE_COLORS = [1, 2, 4, 5, 6, 7, 8, 9]

GEN_RETRIES = 400
PROFILE_RETRIES = 200
PLACEMENT_RETRIES = 400

PIECE_COUNT_RANGE = (2, 4)
COMMON_DIM_RANGE = (5, 8)
SEGMENT_LEN_RANGE = (3, 6)


@dataclass(frozen=True)
class ParsedPiece:
    length: int
    cross_dim: int
    lead: tuple[int, ...]
    trail: tuple[int, ...]
    coords: tuple[tuple[int, int], ...]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q: deque[tuple[int, int]] = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_mask(coords: list[tuple[int, int]]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask, (r0, c0, r1, c1)


def _neg(profile: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(-x for x in profile)


def _profile_key(profile: tuple[int, ...]) -> tuple[int, ...]:
    neg = _neg(profile)
    return profile if profile < neg else neg


def _find_palette(grid: np.ndarray) -> tuple[tuple[int, int, int, int], tuple[int, ...]]:
    green_comps = _components(grid == GREEN)
    candidates: list[tuple[int, tuple[int, int, int, int], tuple[int, ...]]] = []
    for comp in green_comps:
        r0, c0, r1, c1 = _bbox(comp)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        box = grid[r0 : r1 + 1, c0 : c1 + 1]
        border = np.zeros_like(box, dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        if not np.all(box[border] == GREEN):
            continue
        if np.any(box[~border] == GREEN):
            continue
        inner = box[1:-1, 1:-1]
        pts = np.argwhere((inner != BG) & (inner != GREEN))
        if pts.size == 0:
            continue
        cols = sorted({int(c) for _, c in pts})
        if len(cols) != 1:
            continue
        col = cols[0]
        rows = sorted(int(r) for r, _ in pts)
        if rows != list(range(rows[0], rows[-1] + 1)):
            continue
        colors = tuple(int(inner[r, col]) for r in rows)
        score = r1 + c1
        candidates.append((score, (r0, c0, r1, c1), colors))
    assert candidates, "expected a green palette frame"
    _, bbox, colors = max(candidates, key=lambda item: item[0])
    return bbox, colors


def _parse_row_piece(mask: np.ndarray) -> ParsedPiece | None:
    h, w = mask.shape
    if h == 0 or w == 0:
        return None

    lefts: list[int] = []
    rights: list[int] = []
    for r in range(h):
        cols = np.flatnonzero(mask[r])
        if cols.size == 0:
            return None
        if int(cols[-1]) - int(cols[0]) + 1 != int(cols.size):
            return None
        lefts.append(int(cols[0]))
        rights.append(int(cols[-1]))

    left_vals = sorted(set(lefts))
    if len(left_vals) == 1:
        base_left = left_vals[0]
        lead = (0,) * h
    elif len(left_vals) == 2 and left_vals[1] - left_vals[0] == 2:
        base_left = left_vals[0] + 1
        lead = tuple(1 if x == left_vals[0] else -1 for x in lefts)
        if 1 not in lead or -1 not in lead:
            return None
    else:
        return None

    right_vals = sorted(set(rights))
    if len(right_vals) == 1:
        base_right = right_vals[0]
        trail = (0,) * h
    elif len(right_vals) == 2 and right_vals[1] - right_vals[0] == 2:
        base_right = right_vals[0] + 1
        trail = tuple(-1 if x == right_vals[0] else 1 for x in rights)
        if 1 not in trail or -1 not in trail:
            return None
    else:
        return None

    if base_right < base_left:
        return None

    coords: list[tuple[int, int]] = []
    cell_count = 0
    for r in range(h):
        start = base_left + (-1 if lead[r] == 1 else 1 if lead[r] == -1 else 0)
        end = base_right + (1 if trail[r] == 1 else -1 if trail[r] == -1 else 0)
        if start != lefts[r] or end != rights[r] or end < start:
            return None
        for c in range(start, end + 1):
            if not mask[r, c]:
                return None
            coords.append((r, c - base_left))
            cell_count += 1
    if cell_count != int(mask.sum()):
        return None

    return ParsedPiece(
        length=int(base_right - base_left + 1),
        cross_dim=int(h),
        lead=lead,
        trail=trail,
        coords=tuple(coords),
    )


def _parse_piece(mask: np.ndarray, orientation: str) -> ParsedPiece | None:
    if orientation == "horizontal":
        return _parse_row_piece(mask)

    parsed = _parse_row_piece(mask.T)
    if parsed is None:
        return None
    return ParsedPiece(
        length=parsed.length,
        cross_dim=parsed.cross_dim,
        lead=parsed.lead,
        trail=parsed.trail,
        coords=tuple((col_rel, row_idx) for row_idx, col_rel in parsed.coords),
    )


def _chain_order(pieces: list[ParsedPiece]) -> list[int] | None:
    if not pieces:
        return None
    cross_dim = pieces[0].cross_dim
    if any(piece.cross_dim != cross_dim for piece in pieces):
        return None

    zero = (0,) * cross_dim
    starts = [i for i, piece in enumerate(pieces) if piece.lead == zero]
    ends = [i for i, piece in enumerate(pieces) if piece.trail == zero]
    if len(starts) != 1 or len(ends) != 1:
        return None

    lead_map: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for idx, piece in enumerate(pieces):
        lead_map[piece.lead].append(idx)

    order = [starts[0]]
    used = {starts[0]}
    cur = starts[0]
    while len(order) < len(pieces):
        trail = pieces[cur].trail
        if trail == zero:
            return None
        want = _neg(trail)
        candidates = [idx for idx in lead_map.get(want, []) if idx not in used]
        if len(candidates) != 1:
            return None
        nxt = candidates[0]
        order.append(nxt)
        used.add(nxt)
        cur = nxt

    if pieces[cur].trail != zero:
        return None
    if order[-1] != ends[0]:
        return None
    return order


def _assemble(pieces: list[ParsedPiece], order: list[int], colors: tuple[int, ...], orientation: str) -> np.ndarray:
    assert len(order) == len(colors)
    cross_dim = pieces[0].cross_dim
    total_length = sum(pieces[idx].length for idx in order)

    if orientation == "horizontal":
        out = np.zeros((cross_dim, total_length), dtype=int)
        offset = 0
        for idx, color in zip(order, colors):
            for r, c in pieces[idx].coords:
                rr = r
                cc = offset + c
                assert 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]
                assert out[rr, cc] == 0
                out[rr, cc] = int(color)
            offset += pieces[idx].length
    else:
        out = np.zeros((total_length, cross_dim), dtype=int)
        offset = 0
        for idx, color in zip(order, colors):
            for r, c in pieces[idx].coords:
                rr = offset + r
                cc = c
                assert 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]
                assert out[rr, cc] == 0
                out[rr, cc] = int(color)
            offset += pieces[idx].length

    assert np.all(out != 0)
    return out


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    palette_bbox, palette_colors = _find_palette(g)
    pr0, pc0, pr1, pc1 = palette_bbox

    piece_mask = g != BG
    piece_mask[pr0 : pr1 + 1, pc0 : pc1 + 1] = False
    components = _components(piece_mask)
    assert components, "expected puzzle pieces"
    assert len(components) == len(palette_colors), "palette size must match piece count"

    piece_masks: list[np.ndarray] = []
    for comp in components:
        colors = {int(g[r, c]) for r, c in comp}
        assert len(colors) == 1 and GREEN not in colors, "each piece must be monochrome"
        mask, _ = _crop_mask(comp)
        piece_masks.append(mask)

    solutions: list[np.ndarray] = []
    for orientation in ("horizontal", "vertical"):
        parsed: list[ParsedPiece] = []
        for mask in piece_masks:
            piece = _parse_piece(mask, orientation)
            if piece is None:
                parsed = []
                break
            parsed.append(piece)
        if not parsed:
            continue
        order = _chain_order(parsed)
        if order is None:
            continue
        solutions.append(_assemble(parsed, order, palette_colors, orientation))

    unique: list[np.ndarray] = []
    for sol in solutions:
        if not any(sol.shape == other.shape and np.array_equal(sol, other) for other in unique):
            unique.append(sol)
    assert len(unique) == 1, "expected exactly one valid assembly"
    return unique[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble the toothed fragments into a single rectangle, then recolor them by the palette order.
    """
    return _solve(grid)


def _random_profile(dim: int) -> tuple[int, ...]:
    run_count = random.randint(2, min(5, dim))
    cuts = sorted(random.sample(range(1, dim), run_count - 1))
    vals: list[int] = []
    last = 0
    sign = random.choice([-1, 1])
    for cut in cuts + [dim]:
        vals.extend([sign] * (cut - last))
        sign *= -1
        last = cut
    profile = tuple(vals)
    assert 1 in profile and -1 in profile
    return profile


def _build_horizontal_piece(length: int, height: int, lead: tuple[int, ...], trail: tuple[int, ...]) -> tuple[np.ndarray, ParsedPiece]:
    left_pad = 1 if 1 in lead else 0
    right_pad = 1 if 1 in trail else 0
    base_left = left_pad
    mask = np.zeros((height, left_pad + length + right_pad), dtype=bool)
    coords: list[tuple[int, int]] = []
    for r in range(height):
        start = base_left + (-1 if lead[r] == 1 else 1 if lead[r] == -1 else 0)
        end = base_left + length - 1 + (1 if trail[r] == 1 else -1 if trail[r] == -1 else 0)
        mask[r, start : end + 1] = True
        for c in range(start, end + 1):
            coords.append((r, c - base_left))
    piece = ParsedPiece(
        length=int(length),
        cross_dim=int(height),
        lead=tuple(int(x) for x in lead),
        trail=tuple(int(x) for x in trail),
        coords=tuple(coords),
    )
    return mask, piece


def _build_piece(orientation: str, length: int, common_dim: int, lead: tuple[int, ...], trail: tuple[int, ...]) -> tuple[np.ndarray, ParsedPiece]:
    if orientation == "horizontal":
        return _build_horizontal_piece(length, common_dim, lead, trail)

    temp_mask, temp_piece = _build_horizontal_piece(length, common_dim, lead, trail)
    mask = temp_mask.T
    piece = ParsedPiece(
        length=temp_piece.length,
        cross_dim=temp_piece.cross_dim,
        lead=temp_piece.lead,
        trail=temp_piece.trail,
        coords=tuple((col_rel, row_idx) for row_idx, col_rel in temp_piece.coords),
    )
    return mask, piece


def _place_blocked(blocked: np.ndarray, r0: int, c0: int, h: int, w: int, margin: int = 1) -> None:
    rr0 = max(0, r0 - margin)
    cc0 = max(0, c0 - margin)
    rr1 = min(blocked.shape[0], r0 + h + margin)
    cc1 = min(blocked.shape[1], c0 + w + margin)
    blocked[rr0:rr1, cc0:cc1] = True


def _gen_attempt() -> np.ndarray | None:
    orientation = random.choice(["horizontal", "vertical"])
    piece_count = random.randint(*PIECE_COUNT_RANGE)
    common_dim = random.randint(*COMMON_DIM_RANGE)
    lengths = [random.randint(*SEGMENT_LEN_RANGE) for _ in range(piece_count)]

    if 2 * piece_count > len(USABLE_COLORS):
        return None
    chosen_colors = random.sample(USABLE_COLORS, 2 * piece_count)
    piece_colors = chosen_colors[:piece_count]
    palette_colors = tuple(chosen_colors[piece_count:])

    profiles: list[tuple[int, ...]] = []
    used_profile_keys: set[tuple[int, ...]] = set()
    for _ in range(piece_count - 1):
        profile = None
        for _attempt in range(PROFILE_RETRIES):
            candidate = _random_profile(common_dim)
            key = _profile_key(candidate)
            if key in used_profile_keys:
                continue
            used_profile_keys.add(key)
            profile = candidate
            break
        if profile is None:
            return None
        profiles.append(profile)

    piece_specs: list[ParsedPiece] = []
    piece_arrays: list[np.ndarray] = []
    zero = (0,) * common_dim
    for idx in range(piece_count):
        lead = zero if idx == 0 else _neg(profiles[idx - 1])
        trail = zero if idx == piece_count - 1 else profiles[idx]
        mask, spec = _build_piece(orientation, lengths[idx], common_dim, lead, trail)
        arr = np.zeros(mask.shape, dtype=int)
        arr[mask] = int(piece_colors[idx])
        piece_specs.append(spec)
        piece_arrays.append(arr)

    expected = _assemble(piece_specs, list(range(piece_count)), palette_colors, orientation)

    palette_h = piece_count + 2
    palette_w = 3
    palette = np.zeros((palette_h, palette_w), dtype=int)
    palette[0, :] = GREEN
    palette[-1, :] = GREEN
    palette[:, 0] = GREEN
    palette[:, -1] = GREEN
    for i, color in enumerate(palette_colors):
        palette[1 + i, 1] = int(color)

    piece_boxes = [(arr.shape[0], arr.shape[1]) for arr in piece_arrays]
    total_box_area = sum((h + 2) * (w + 2) for h, w in piece_boxes) + (palette_h + 2) * (palette_w + 2)
    side = int(math.ceil(math.sqrt(total_box_area))) + 2
    min_h = max(side, max(h for h, _ in piece_boxes) + 2, palette_h + 2)
    min_w = max(side, max(w for _, w in piece_boxes) + 2, palette_w + 2)
    grid_h = min_h + random.randint(2, 7)
    grid_w = min_w + random.randint(2, 9)

    bottom_margin = random.randint(0, 2)
    right_margin = random.randint(0, 2)
    palette_top = grid_h - palette_h - bottom_margin
    palette_left = grid_w - palette_w - right_margin
    if palette_top < 0 or palette_left < 0:
        return None

    grid = np.zeros((grid_h, grid_w), dtype=int)
    blocked = np.zeros((grid_h, grid_w), dtype=bool)
    grid[palette_top : palette_top + palette_h, palette_left : palette_left + palette_w] = palette
    _place_blocked(blocked, palette_top, palette_left, palette_h, palette_w)

    order = list(range(piece_count))
    order.sort(key=lambda idx: piece_arrays[idx].shape[0] * piece_arrays[idx].shape[1], reverse=True)
    random.shuffle(order)
    placements: dict[int, tuple[int, int]] = {}
    for idx in order:
        arr = piece_arrays[idx]
        h, w = arr.shape
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            r0 = random.randint(0, grid_h - h)
            c0 = random.randint(0, grid_w - w)
            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(grid_h, r0 + h + 1)
            cc1 = min(grid_w, c0 + w + 1)
            if np.any(blocked[rr0:rr1, cc0:cc1]):
                continue
            placements[idx] = (r0, c0)
            grid[r0 : r0 + h, c0 : c0 + w][arr != 0] = arr[arr != 0]
            _place_blocked(blocked, r0, c0, h, w)
            placed = True
            break
        if not placed:
            return None

    solved = main(grid)
    if solved.shape != expected.shape or not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
