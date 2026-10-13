from __future__ import annotations

from functools import lru_cache
import numpy as np


AZURE = 8
BLACK = 0
COLOR_POOL = [1, 2, 3, 4, 5, 6, 7, 9]
MIN_COLORS = 2
MAX_COLORS = 4
MIN_PIECES = 2
MAX_PIECES = 4
MIN_RECT_H = 2
MAX_RECT_H = 6
MIN_RECT_W = 3
MAX_RECT_W = 7
MIN_RECT_AREA = 6
MIN_PIECE_AREA = 2
GRID_MAX = 30
MAX_GENERATION_ATTEMPTS = 300

_RNG = np.random.default_rng()
_NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in _NEIGHBORS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(coords)
    return components


def _normalize_coords(coords: list[tuple[int, int]]) -> tuple[tuple[tuple[int, int], ...], int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0 = min(rows)
    c0 = min(cols)
    normalized = tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))
    h = max(rows) - r0 + 1
    w = max(cols) - c0 + 1
    return normalized, int(h), int(w)


def _piece_from_coords(coords: list[tuple[int, int]]) -> dict[str, object]:
    normalized, h, w = _normalize_coords(coords)
    return {
        "coords": normalized,
        "h": h,
        "w": w,
        "area": len(normalized),
    }


def _factor_pairs(area: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for h in range(1, int(area ** 0.5) + 1):
        if area % h != 0:
            continue
        w = area // h
        pairs.append((int(h), int(w)))
        if h != w:
            pairs.append((int(w), int(h)))
    return pairs


def _can_tile_rectangle(pieces: list[dict[str, object]], rect_h: int, rect_w: int) -> bool:
    if rect_h <= 0 or rect_w <= 0:
        return False
    if sum(int(piece["area"]) for piece in pieces) != rect_h * rect_w:
        return False
    if any(int(piece["h"]) > rect_h or int(piece["w"]) > rect_w for piece in pieces):
        return False

    n = len(pieces)
    full_mask = (1 << (rect_h * rect_w)) - 1
    coverers: list[list[tuple[int, int]]] = [[] for _ in range(rect_h * rect_w)]

    for piece_id, piece in enumerate(pieces):
        seen_masks: set[int] = set()
        coords = piece["coords"]
        piece_h = int(piece["h"])
        piece_w = int(piece["w"])
        for dr in range(rect_h - piece_h + 1):
            for dc in range(rect_w - piece_w + 1):
                mask = 0
                for r, c in coords:
                    idx = (int(r) + dr) * rect_w + int(c) + dc
                    mask |= 1 << idx
                if mask in seen_masks:
                    continue
                seen_masks.add(mask)
                m = mask
                while m:
                    bit = m & -m
                    coverers[bit.bit_length() - 1].append((piece_id, mask))
                    m ^= bit
        if not seen_masks:
            return False

    @lru_cache(maxsize=None)
    def search(occupied: int, remaining: int) -> bool:
        if remaining == 0:
            return occupied == full_mask
        uncovered = full_mask & ~occupied
        if uncovered == 0:
            return False
        bit = uncovered & -uncovered
        cell_index = bit.bit_length() - 1
        for piece_id, placement in coverers[cell_index]:
            if not (remaining & (1 << piece_id)):
                continue
            if placement & occupied:
                continue
            if search(occupied | placement, remaining ^ (1 << piece_id)):
                return True
        return False

    return bool(search(0, (1 << n) - 1))


def _rectangle_candidates(pieces: list[dict[str, object]]) -> list[tuple[int, int]]:
    total_area = sum(int(piece["area"]) for piece in pieces)
    candidates: list[tuple[int, int]] = []
    for rect_h, rect_w in _factor_pairs(total_area):
        if _can_tile_rectangle(pieces, rect_h, rect_w):
            candidates.append((int(rect_h), int(rect_w)))
    candidates = sorted(set(candidates))
    return candidates


def _extract_rectangles(grid: np.ndarray) -> list[dict[str, int]]:
    bg = _most_common_color(grid)
    rectangles: list[dict[str, int]] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != bg):
        components = _connected_components(grid == color)
        assert MIN_PIECES <= len(components) <= MAX_PIECES
        pieces = [_piece_from_coords(component) for component in components]
        candidates = _rectangle_candidates(pieces)
        assert len(candidates) == 1
        rect_h, rect_w = candidates[0]
        rectangles.append(
            {
                "color": int(color),
                "h": int(rect_h),
                "w": int(rect_w),
                "area": int(rect_h * rect_w),
            }
        )
    assert rectangles
    areas = [rect["area"] for rect in rectangles]
    assert len(set(areas)) == len(areas)
    return rectangles


def main(grid: np.ndarray) -> np.ndarray:
    """Assemble each color's fragments into a solid rectangle, then stack rectangles by area."""
    g = np.asarray(grid, dtype=int)
    rectangles = _extract_rectangles(g)
    largest = max(rectangles, key=lambda rect: rect["area"])
    out = np.full((largest["h"], largest["w"]), BLACK, dtype=int)

    # Paint largest first so smaller rectangles remain visible on top.
    for rect in sorted(rectangles, key=lambda item: item["area"], reverse=True):
        out[: rect["h"], : rect["w"]] = rect["color"]
    return out


def _sample_rectangle_chain(n_colors: int) -> list[tuple[int, int]]:
    for _ in range(200):
        rects: list[tuple[int, int]] = []
        cur_h = int(_RNG.integers(MIN_RECT_H, MAX_RECT_H - 1))
        cur_w = int(_RNG.integers(MIN_RECT_W, MAX_RECT_W - 1))
        if cur_h * cur_w < MIN_RECT_AREA:
            cur_w = max(cur_w, (MIN_RECT_AREA + cur_h - 1) // cur_h)
        rects.append((cur_h, cur_w))

        while len(rects) < n_colors:
            prev_h, prev_w = rects[-1]
            next_h = prev_h + int(_RNG.integers(0, 3))
            next_w = prev_w + int(_RNG.integers(0, 3))
            if next_h == prev_h and next_w == prev_w:
                continue
            if next_h > MAX_RECT_H or next_w > MAX_RECT_W:
                break
            area = next_h * next_w
            if area in {h * w for h, w in rects}:
                continue
            rects.append((int(next_h), int(next_w)))

        if len(rects) != n_colors:
            continue
        areas = [h * w for h, w in rects]
        if min(areas) < MIN_RECT_AREA or len(set(areas)) != n_colors:
            continue
        return rects
    raise RuntimeError("failed to sample rectangle chain")


def _sample_piece_sizes(total_area: int, n_pieces: int) -> np.ndarray:
    extras = _RNG.multinomial(total_area - MIN_PIECE_AREA * n_pieces, np.ones(n_pieces) / n_pieces)
    sizes = extras + MIN_PIECE_AREA
    _RNG.shuffle(sizes)
    return sizes


def _partition_rectangle(rect_h: int, rect_w: int, n_pieces: int) -> list[dict[str, object]] | None:
    targets = _sample_piece_sizes(rect_h * rect_w, n_pieces)
    for _ in range(120):
        labels = np.full((rect_h, rect_w), -1, dtype=int)
        seed_ids = _RNG.choice(rect_h * rect_w, size=n_pieces, replace=False)
        areas = np.ones(n_pieces, dtype=int)
        for piece_id, seed in enumerate(seed_ids):
            rr = int(seed // rect_w)
            cc = int(seed % rect_w)
            labels[rr, cc] = piece_id

        while np.any(labels < 0):
            choices: list[int] = []
            weights: list[int] = []
            frontiers: list[list[tuple[int, int]]] = [[] for _ in range(n_pieces)]
            for piece_id in range(n_pieces):
                if areas[piece_id] >= targets[piece_id]:
                    continue
                frontier: set[tuple[int, int]] = set()
                coords = np.argwhere(labels == piece_id)
                for rr, cc in coords:
                    for dr, dc in _NEIGHBORS4:
                        nr = int(rr + dr)
                        nc = int(cc + dc)
                        if 0 <= nr < rect_h and 0 <= nc < rect_w and labels[nr, nc] < 0:
                            frontier.add((nr, nc))
                if frontier:
                    choices.append(piece_id)
                    weights.append(int(targets[piece_id] - areas[piece_id]))
                    frontiers[piece_id] = list(frontier)
            if not choices:
                break
            probs = np.asarray(weights, dtype=float)
            probs /= probs.sum()
            piece_id = int(_RNG.choice(np.asarray(choices), p=probs))
            cell = frontiers[piece_id][int(_RNG.integers(len(frontiers[piece_id])))]
            labels[cell] = piece_id
            areas[piece_id] += 1

        if np.any(labels < 0):
            continue
        if any(int(areas[piece_id]) != int(targets[piece_id]) for piece_id in range(n_pieces)):
            continue

        pieces = []
        for piece_id in range(n_pieces):
            coords = [tuple(map(int, rc)) for rc in np.argwhere(labels == piece_id)]
            piece = _piece_from_coords(coords)
            if int(piece["area"]) < MIN_PIECE_AREA:
                break
            pieces.append(piece)
        else:
            candidates = _rectangle_candidates(pieces)
            if candidates == [(rect_h, rect_w)]:
                return pieces
    return None


def _piece_patch(piece: dict[str, object], color: int) -> np.ndarray:
    patch = np.full((int(piece["h"]), int(piece["w"])), AZURE, dtype=int)
    for r, c in piece["coords"]:
        patch[int(r), int(c)] = int(color)
    return patch


def _dilate8(mask: np.ndarray) -> np.ndarray:
    dilated = mask.copy()
    dilated[1:, :] |= mask[:-1, :]
    dilated[:-1, :] |= mask[1:, :]
    dilated[:, 1:] |= mask[:, :-1]
    dilated[:, :-1] |= mask[:, 1:]
    dilated[1:, 1:] |= mask[:-1, :-1]
    dilated[1:, :-1] |= mask[:-1, 1:]
    dilated[:-1, 1:] |= mask[1:, :-1]
    dilated[:-1, :-1] |= mask[1:, 1:]
    return dilated


def _place_patches(patches: list[np.ndarray]) -> np.ndarray | None:
    total_area = sum(int(np.count_nonzero(patch != AZURE)) for patch in patches)
    max_h = max(int(patch.shape[0]) for patch in patches)
    max_w = max(int(patch.shape[1]) for patch in patches)
    n_patches = len(patches)

    side = int(np.ceil(np.sqrt(total_area * 3.5))) + n_patches
    min_h = max(14, max_h + 2, side - 2)
    min_w = max(14, max_w + 2, side - 1)
    if min_h > GRID_MAX or min_w > GRID_MAX:
        return None

    grid_h = int(_RNG.integers(min_h, GRID_MAX + 1))
    grid_w = int(_RNG.integers(min_w, GRID_MAX + 1))
    grid = np.full((grid_h, grid_w), AZURE, dtype=int)
    occupied = np.zeros((grid_h, grid_w), dtype=bool)

    order = list(range(n_patches))
    _RNG.shuffle(order)
    order.sort(key=lambda idx: int(np.count_nonzero(patches[idx] != AZURE)), reverse=True)

    for idx in order:
        patch = patches[idx]
        mask = patch != AZURE
        ph, pw = patch.shape
        dilated = _dilate8(occupied)
        positions: list[tuple[int, int]] = []
        for r0 in range(grid_h - ph + 1):
            for c0 in range(grid_w - pw + 1):
                if bool((dilated[r0 : r0 + ph, c0 : c0 + pw] & mask).any()):
                    continue
                positions.append((int(r0), int(c0)))
        if not positions:
            return None
        r0, c0 = positions[int(_RNG.integers(len(positions)))]
        grid[r0 : r0 + ph, c0 : c0 + pw][mask] = patch[mask]
        occupied[r0 : r0 + ph, c0 : c0 + pw] |= mask
    return grid


def _expected_output(rectangles: list[dict[str, int]]) -> np.ndarray:
    largest = max(rectangles, key=lambda rect: rect["area"])
    out = np.full((largest["h"], largest["w"]), BLACK, dtype=int)
    for rect in sorted(rectangles, key=lambda item: item["area"], reverse=True):
        out[: rect["h"], : rect["w"]] = rect["color"]
    return out


def _gen_attempt() -> np.ndarray | None:
    n_colors = int(_RNG.integers(MIN_COLORS, MAX_COLORS + 1))
    colors = [int(c) for c in _RNG.choice(np.asarray(COLOR_POOL), size=n_colors, replace=False)]
    dims = _sample_rectangle_chain(n_colors)

    rectangles: list[dict[str, int]] = []
    patches: list[np.ndarray] = []
    for color, (rect_h, rect_w) in zip(colors, dims):
        max_piece_count = min(MAX_PIECES, (rect_h * rect_w) // MIN_PIECE_AREA)
        if max_piece_count < MIN_PIECES:
            return None
        n_pieces = int(_RNG.integers(MIN_PIECES, max_piece_count + 1))
        pieces = _partition_rectangle(rect_h, rect_w, n_pieces)
        if pieces is None:
            return None
        rectangles.append({"color": color, "h": rect_h, "w": rect_w, "area": rect_h * rect_w})
        patches.extend(_piece_patch(piece, color) for piece in pieces)

    grid = _place_patches(patches)
    if grid is None:
        return None

    expected = _expected_output(rectangles)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if np.array_equal(actual, expected):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
