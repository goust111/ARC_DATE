from __future__ import annotations

from collections import Counter
import random

import numpy as np


GREY = 5
ALL_COLORS = tuple(range(10))
GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _crop_component_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_coords(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _dihedral_masks(mask: np.ndarray) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    seen: set[tuple[int, int, bytes]] = set()
    for k in range(4):
        rot = np.rot90(mask, k)
        for variant in (rot, np.fliplr(rot)):
            key = (int(variant.shape[0]), int(variant.shape[1]), variant.tobytes())
            if key in seen:
                continue
            seen.add(key)
            out.append(variant.copy())
    return out


def _canonical_key(mask: np.ndarray) -> tuple[int, int, bytes]:
    return min(
        (int(variant.shape[0]), int(variant.shape[1]), variant.tobytes())
        for variant in _dihedral_masks(mask)
    )


def _most_common_color(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    max_count = int(counts.max())
    winners = [int(c) for c, n in zip(colors, counts, strict=False) if int(n) == max_count]
    assert len(winners) == 1, "background must be uniquely most common"
    return winners[0]


def _find_frame(grid: np.ndarray, bg: int) -> tuple[int, tuple[int, int, int, int], list[dict]]:
    candidates: list[tuple[int, int, int, tuple[int, int, int, int], list[dict]]] = []
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        for coords in _components_mask(grid == color):
            r0, c0, r1, c1 = _bbox_coords(coords)
            if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 3:
                continue
            sub = grid[r0 : r1 + 1, c0 : c1 + 1]
            if not set(map(int, np.unique(sub))).issubset({bg, color}):
                continue
            if not (
                np.all(sub[0, :] == color)
                and np.all(sub[-1, :] == color)
                and np.all(sub[:, 0] == color)
                and np.all(sub[:, -1] == color)
            ):
                continue

            holes = []
            for hole_coords in _components_mask(sub == bg):
                if (
                    np.any(hole_coords[:, 0] == 0)
                    or np.any(hole_coords[:, 0] == sub.shape[0] - 1)
                    or np.any(hole_coords[:, 1] == 0)
                    or np.any(hole_coords[:, 1] == sub.shape[1] - 1)
                ):
                    continue
                hole_mask = _crop_component_mask(hole_coords)
                holes.append(
                    {
                        "coords": hole_coords,
                        "mask": hole_mask,
                        "key": _canonical_key(hole_mask),
                    }
                )
            if not holes:
                continue
            bbox_area = int((r1 - r0 + 1) * (c1 - c0 + 1))
            candidates.append((int(len(coords)), bbox_area, color, (r0, c0, r1, c1), holes))

    assert candidates, "expected one framed region with internal holes"
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    _area, _bbox_area, color, bbox, holes = candidates[0]
    holes.sort(key=lambda hole: (int(hole["coords"][:, 0].min()), int(hole["coords"][:, 1].min())))
    return int(color), bbox, holes


def _extract_pieces(grid: np.ndarray, bg: int, frame_color: int) -> list[dict]:
    pieces = []
    for coords in _components_mask((grid != bg) & (grid != frame_color)):
        vals = [int(grid[r, c]) for r, c in coords]
        counts = Counter(vals)
        assert len(counts) == 2, "each fragment must contain one body color and one marker color"
        marker_colors = [color for color, n in counts.items() if int(n) == 1]
        assert len(marker_colors) == 1, "each fragment must contain exactly one marker pixel"
        marker_color = int(marker_colors[0])
        body_counts = [n for color, n in counts.items() if int(color) != marker_color]
        assert len(body_counts) == 1 and int(body_counts[0]) >= 2, "fragment body must have area >= 2"
        piece_mask = _crop_component_mask(coords)
        pieces.append(
            {
                "mask": piece_mask,
                "key": _canonical_key(piece_mask),
                "marker": marker_color,
            }
        )
    assert pieces, "expected at least one fragment outside the frame"
    return pieces


def _unique_assignment(edges: list[list[int]], hole_count: int) -> list[int]:
    order = sorted(range(len(edges)), key=lambda idx: (len(edges[idx]), idx))
    used = [False] * hole_count
    assign = [-1] * len(edges)
    solutions: list[list[int]] = []

    def dfs(pos: int) -> None:
        if len(solutions) > 1:
            return
        if pos == len(order):
            solutions.append(assign.copy())
            return
        piece_idx = order[pos]
        for hole_idx in edges[piece_idx]:
            if used[hole_idx]:
                continue
            used[hole_idx] = True
            assign[piece_idx] = hole_idx
            dfs(pos + 1)
            assign[piece_idx] = -1
            used[hole_idx] = False

    dfs(0)
    assert len(solutions) == 1, "piece-to-hole matching must be unique"
    return solutions[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each outside fragment to a same-shaped hole inside the grey frame, then
    fill that hole with the fragment's marker color. The output is the cropped
    frame region, including the grey boundary and inner walls.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _most_common_color(g)
    frame_color, (r0, c0, r1, c1), holes = _find_frame(g, bg)
    pieces = _extract_pieces(g, bg, frame_color)
    assert len(pieces) == len(holes), "hole and fragment counts must match"

    edges: list[list[int]] = []
    for piece in pieces:
        matches = [idx for idx, hole in enumerate(holes) if piece["key"] == hole["key"]]
        assert matches, "every fragment must match at least one hole"
        edges.append(matches)

    assignment = _unique_assignment(edges, len(holes))

    out = g[r0 : r1 + 1, c0 : c1 + 1].copy()
    for piece_idx, hole_idx in enumerate(assignment):
        coords = holes[hole_idx]["coords"]
        out[coords[:, 0], coords[:, 1]] = int(pieces[piece_idx]["marker"])
    return out


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[:-1, :] |= mask[1:, :]
    out[1:, :] |= mask[:-1, :]
    out[:, :-1] |= mask[:, 1:]
    out[:, 1:] |= mask[:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    return out


def _grow_connected_shape(free: np.ndarray, area: int, rng: random.Random) -> np.ndarray | None:
    free_cells = [tuple(map(int, cell)) for cell in np.argwhere(free)]
    if not free_cells:
        return None
    rng.shuffle(free_cells)
    for start in free_cells[: min(16, len(free_cells))]:
        chosen = {start}
        frontier = {
            (nr, nc)
            for nr, nc in _neighbors4(*start)
            if 0 <= nr < free.shape[0] and 0 <= nc < free.shape[1] and free[nr, nc]
        }
        while len(chosen) < area and frontier:
            cell = rng.choice(tuple(frontier))
            frontier.remove(cell)
            if cell in chosen or not free[cell]:
                continue
            chosen.add(cell)
            for nr, nc in _neighbors4(*cell):
                if 0 <= nr < free.shape[0] and 0 <= nc < free.shape[1] and free[nr, nc] and (nr, nc) not in chosen:
                    frontier.add((nr, nc))
        if len(chosen) == area:
            return np.array(sorted(chosen), dtype=int)
    return None


def _transform_mask(mask: np.ndarray, code: int) -> np.ndarray:
    out = np.rot90(mask, code % 4)
    if code >= 4:
        out = np.fliplr(out)
    return out.copy()


def _place_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int, bg: int) -> None:
    sub = grid[top : top + patch.shape[0], left : left + patch.shape[1]]
    mask = patch != bg
    assert np.all(sub[mask] == bg)
    sub[mask] = patch[mask]


def _mark_blocked(blocked: np.ndarray, top: int, left: int, h: int, w: int, pad: int = 1) -> None:
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(blocked.shape[0], top + h + pad)
    c1 = min(blocked.shape[1], left + w + pad)
    blocked[r0:r1, c0:c1] = True


def _gen_layout(interior_h: int, interior_w: int, hole_count: int, rng: random.Random) -> list[dict] | None:
    forbidden = np.zeros((interior_h, interior_w), dtype=bool)
    holes: list[dict] = []
    used_keys: set[tuple[int, int, bytes]] = set()

    max_area = min(9, interior_h * interior_w)
    areas = [rng.randint(3, max_area) for _ in range(hole_count)]
    areas.sort(reverse=True)

    for area in areas:
        placed = None
        for _ in range(50):
            coords = _grow_connected_shape(~forbidden, area, rng)
            if coords is None:
                continue
            mask = _crop_component_mask(coords)
            key = _canonical_key(mask)
            if key in used_keys:
                continue
            if int(mask.sum()) == int(mask.size) and mask.shape[0] >= 3 and mask.shape[1] >= 3 and rng.random() < 0.6:
                continue
            placed = {"coords": coords, "mask": mask, "key": key}
            used_keys.add(key)
            cell_mask = np.zeros_like(forbidden)
            cell_mask[coords[:, 0], coords[:, 1]] = True
            forbidden |= _dilate8(cell_mask)
            break
        if placed is None:
            return None
        holes.append(placed)

    density = sum(int(hole["mask"].sum()) for hole in holes) / float(interior_h * interior_w)
    if not (0.18 <= density <= 0.55):
        return None
    return holes


def _build_example(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    bg = rng.choice([color for color in ALL_COLORS if color != GREY])
    hole_count = rng.randint(2, 4)
    frame_h = rng.randint(8, 15)
    frame_w = rng.randint(8, 15)
    interior_h = frame_h - 2
    interior_w = frame_w - 2

    holes = _gen_layout(interior_h, interior_w, hole_count, rng)
    if holes is None:
        return None

    available = [color for color in ALL_COLORS if color not in (bg, GREY)]
    rng.shuffle(available)
    if len(available) < 2 * hole_count:
        return None
    marker_colors = available[:hole_count]
    body_colors = available[hole_count : 2 * hole_count]

    frame_input = np.full((frame_h, frame_w), GREY, dtype=int)
    frame_output = frame_input.copy()
    pieces: list[np.ndarray] = []

    for hole, marker_color, body_color in zip(holes, marker_colors, body_colors, strict=False):
        global_coords = hole["coords"] + np.array([1, 1], dtype=int)
        frame_input[global_coords[:, 0], global_coords[:, 1]] = bg
        frame_output[global_coords[:, 0], global_coords[:, 1]] = int(marker_color)

        transformed = _transform_mask(hole["mask"], rng.randrange(8))
        patch = np.full(transformed.shape, bg, dtype=int)
        patch[transformed] = int(body_color)
        cells = [tuple(map(int, cell)) for cell in np.argwhere(transformed)]
        mr, mc = rng.choice(cells)
        patch[mr, mc] = int(marker_color)
        pieces.append(patch)

    min_h = max(22, frame_h + 6)
    min_w = max(22, frame_w + 6)
    if min_h > 30 or min_w > 30:
        return None
    grid_h = rng.randint(min_h, 30)
    grid_w = rng.randint(min_w, 30)
    grid = np.full((grid_h, grid_w), bg, dtype=int)

    frame_r0 = rng.randint(2, grid_h - frame_h - 2)
    frame_c0 = rng.randint(2, grid_w - frame_w - 2)
    _place_patch(grid, frame_input, frame_r0, frame_c0, bg)

    blocked = np.zeros((grid_h, grid_w), dtype=bool)
    _mark_blocked(blocked, frame_r0, frame_c0, frame_h, frame_w, pad=1)

    pieces_sorted = sorted(pieces, key=lambda patch: int(np.sum(patch != bg)), reverse=True)
    for patch in pieces_sorted:
        ph, pw = map(int, patch.shape)
        candidates = []
        for r0 in range(grid_h - ph + 1):
            for c0 in range(grid_w - pw + 1):
                rr0 = max(0, r0 - 1)
                cc0 = max(0, c0 - 1)
                rr1 = min(grid_h, r0 + ph + 1)
                cc1 = min(grid_w, c0 + pw + 1)
                if blocked[rr0:rr1, cc0:cc1].any():
                    continue
                candidates.append((r0, c0))
        if not candidates:
            return None
        top, left = rng.choice(candidates)
        _place_patch(grid, patch, top, left, bg)
        _mark_blocked(blocked, top, left, ph, pw, pad=1)

    colors, counts = np.unique(grid, return_counts=True)
    winners = [int(c) for c, n in zip(colors, counts, strict=False) if int(n) == int(counts.max())]
    if winners != [int(bg)]:
        return None

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, frame_output):
        return None
    return grid, frame_output


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        built = _build_example(rng)
        if built is not None:
            grid, _out = built
            return grid
    raise RuntimeError("failed to generate a valid input")
