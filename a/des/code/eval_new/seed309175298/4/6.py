from __future__ import annotations

import random
from collections import Counter

import numpy as np


UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

DIR_ORDER = [UP, RIGHT, DOWN, LEFT]
FOREGROUND_COLORS = list(range(1, 10))
N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
N8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]


def _mask_from_rows(rows: list[str]) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)


ARROW_MASKS = {
    UP: _mask_from_rows(["111", "010", "010"]),
    RIGHT: _mask_from_rows(["001", "111", "001"]),
    DOWN: _mask_from_rows(["010", "010", "111"]),
    LEFT: _mask_from_rows(["100", "111", "100"]),
}
ARROW_CELLS = {
    direction: [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    for direction, mask in ARROW_MASKS.items()
}
ARROW_KEYS = {
    tuple(map(tuple, np.argwhere(mask))): direction for direction, mask in ARROW_MASKS.items()
}


def _rotate_mask(mask: np.ndarray, cw_steps: int) -> np.ndarray:
    return np.rot90(np.asarray(mask, dtype=bool), -int(cw_steps) % 4)


def _rotate_dir(direction: int, cw_steps: int) -> int:
    return (int(direction) + int(cw_steps)) % 4


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _mask_from_coords(coords: list[tuple[int, int]]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask


def _coords_from_mask(mask: np.ndarray) -> list[tuple[int, int]]:
    return [tuple(map(int, rc)) for rc in np.argwhere(mask)]


def _component_masks(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    H, W = grid.shape
    seen = np.zeros((H, W), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(H):
        for c in range(W):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in N4:
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < H and 0 <= nc < W):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != int(color):
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            comps.append(coords)
    return comps


def _arrow_direction(mask: np.ndarray) -> int | None:
    key = tuple(map(tuple, np.argwhere(mask)))
    return ARROW_KEYS.get(key)


def _natural_direction(mask: np.ndarray) -> int:
    """
    Estimate the shape's natural facing by quadrant density around the bbox center.
    Axis ties are resolved by the centroid offset, preferring horizontal on exact ties.
    """
    pts = np.argwhere(mask)
    h, w = mask.shape
    center_r = (h - 1) / 2.0
    center_c = (w - 1) / 2.0

    nw = ne = se = sw = 0
    for r, c in pts:
        rr = float(r) - center_r
        cc = float(c) - center_c
        if rr == 0.0 or cc == 0.0:
            continue
        if rr < 0 and cc < 0:
            nw += 1
        elif rr < 0 and cc > 0:
            ne += 1
        elif rr > 0 and cc > 0:
            se += 1
        else:
            sw += 1

    scores = {
        UP: nw + ne,
        RIGHT: ne + se,
        DOWN: sw + se,
        LEFT: nw + sw,
    }
    best = max(scores.values())
    candidates = [direction for direction in DIR_ORDER if scores[direction] == best]

    mean_r = float(pts[:, 0].mean()) - center_r
    mean_c = float(pts[:, 1].mean()) - center_c
    if abs(mean_c) >= abs(mean_r):
        preferred = RIGHT if mean_c >= 0 else LEFT
    else:
        preferred = DOWN if mean_r >= 0 else UP
    if preferred in candidates:
        return preferred

    alignment = {
        UP: -mean_r,
        RIGHT: mean_c,
        DOWN: mean_r,
        LEFT: -mean_c,
    }
    best_align = max(alignment[d] for d in candidates)
    for direction in DIR_ORDER:
        if direction in candidates and alignment[direction] == best_align:
            return direction
    return candidates[0]


def _stamp_coords(mask: np.ndarray, center: tuple[int, int], shape: tuple[int, int]) -> list[tuple[int, int]]:
    H, W = shape
    h, w = mask.shape
    r0 = int(center[0]) - h // 2
    c0 = int(center[1]) - w // 2
    coords: list[tuple[int, int]] = []
    for rr, cc in np.argwhere(mask):
        r = r0 + int(rr)
        c = c0 + int(cc)
        if 0 <= r < H and 0 <= c < W:
            coords.append((r, c))
    return coords


def _stamp_cells(
    cells: list[tuple[int, int]],
    mask_shape: tuple[int, int],
    center: tuple[int, int],
    grid_shape: tuple[int, int],
) -> tuple[list[tuple[int, int]], bool]:
    H, W = grid_shape
    h, w = mask_shape
    r0 = int(center[0]) - h // 2
    c0 = int(center[1]) - w // 2
    coords: list[tuple[int, int]] = []
    clipped = False
    for rr, cc in cells:
        r = r0 + int(rr)
        c = c0 + int(cc)
        if 0 <= r < H and 0 <= c < W:
            coords.append((r, c))
        else:
            clipped = True
    return coords, clipped


def _paste_shape(grid: np.ndarray, mask: np.ndarray, center: tuple[int, int], color: int) -> None:
    for r, c in _stamp_coords(mask, center, grid.shape):
        grid[r, c] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each colored arrow with the same-colored shape rotated to the arrow's facing.

    Input structure:
    - Background is black.
    - Each non-black color appears in exactly two monochrome 4-connected components: one 5-cell arrow sprite and one small shape region.
    - Arrows are 3x3 sprites whose 5 occupied cells form a shaft plus a two-pixel arrowhead, facing up/right/down/left.
    - Shape regions are arbitrary connected blobs of 3 to 8 pixels.

    Rule:
    - Pair every arrow with the shape of the same color.
    - Infer the shape's natural facing from which side of its tight bounding box contains more mass, using quadrant counts around the bbox center and a centroid tie-break.
    - Rotate the shape from its natural facing to the arrow's facing.
    - Erase the original arrow and shape, then stamp the rotated shape back at the arrow position, aligning the shape bbox center to the arrow shaft's center pixel; clip anything outside the grid.

    Key insight:
    - Color gives the pairing, while arrow direction specifies the target rotation rather than a translation; the solver must detect arrow sprites, estimate each blob's inherent facing, and then re-stamp the rotated blob in place.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    pairs: list[dict] = []
    for color in sorted(int(c) for c in np.unique(g) if int(c) != 0):
        comps = _component_masks(g, color)
        arrow = None
        shape = None
        for coords in comps:
            mask = _mask_from_coords(coords)
            direction = _arrow_direction(mask)
            if direction is not None and len(coords) == 5:
                arrow = {
                    "coords": coords,
                    "center": (int(_bbox(coords)[0] + 1), int(_bbox(coords)[1] + 1)),
                    "direction": int(direction),
                }
            else:
                shape = {
                    "coords": coords,
                    "mask": mask,
                    "natural": int(_natural_direction(mask)),
                }
        assert arrow is not None and shape is not None and len(comps) == 2
        for r, c in arrow["coords"]:
            out[r, c] = 0
        for r, c in shape["coords"]:
            out[r, c] = 0
        pairs.append({"color": color, "arrow": arrow, "shape": shape})

    for pair in pairs:
        arrow = pair["arrow"]
        shape = pair["shape"]
        cw_steps = (int(arrow["direction"]) - int(shape["natural"])) % 4
        rotated = _rotate_mask(shape["mask"], cw_steps)
        _paste_shape(out, rotated, arrow["center"], pair["color"])

    return out


_BASE_SHAPE_ROWS = [
    ["001", "011", "000"],
    ["011", "001", "001"],
    ["011", "001", "011"],
    ["001", "001", "011"],
    ["01111", "00001", "00001"],
    ["00111", "00001", "00111"],
    ["00111", "00001", "00011"],
    ["01111", "00001", "00111"],
    ["001", "001", "011", "001", "001"],
    ["001", "001", "011", "011", "001"],
    ["001", "001", "011", "001", "011"],
]


def _build_shape_library() -> list[np.ndarray]:
    library: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    for rows in _BASE_SHAPE_ROWS:
        mask = _mask_from_rows(rows)
        if mask.shape[0] % 2 == 0 or mask.shape[1] % 2 == 0:
            continue
        key = tuple(map(tuple, np.argwhere(mask)))
        if key in ARROW_KEYS:
            continue
        if _natural_direction(mask) != RIGHT:
            continue
        rotations = [_natural_direction(_rotate_mask(mask, k)) for k in range(4)]
        if rotations != [RIGHT, DOWN, LEFT, UP]:
            continue
        if key not in seen:
            seen.add(key)
            library.append(mask)
    assert library
    return library


SHAPE_LIBRARY = _build_shape_library()


def _dilate_coords(coords: list[tuple[int, int]], shape: tuple[int, int]) -> np.ndarray:
    H, W = shape
    mask = np.zeros((H, W), dtype=bool)
    for r, c in coords:
        for dr, dc in [(0, 0), *N8]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W:
                mask[nr, nc] = True
    return mask


def _placeable(coords: list[tuple[int, int]], forbidden: np.ndarray) -> bool:
    H, W = forbidden.shape
    for r, c in coords:
        if not (0 <= r < H and 0 <= c < W):
            return False
        if forbidden[r, c]:
            return False
    return True


def _translated_coords(mask: np.ndarray, r0: int, c0: int) -> list[tuple[int, int]]:
    return [(int(r0 + r), int(c0 + c)) for r, c in np.argwhere(mask)]


def _arrow_coords(direction: int, center: tuple[int, int]) -> list[tuple[int, int]]:
    r0 = int(center[0]) - 1
    c0 = int(center[1]) - 1
    return [(r0 + r, c0 + c) for r, c in ARROW_CELLS[int(direction)]]


def _expected_output(pairs: list[dict], H: int, W: int) -> np.ndarray:
    out = np.zeros((H, W), dtype=int)
    for pair in pairs:
        _paste_shape(out, pair["output_mask"], pair["arrow_center"], pair["color"])
    return out


def _pick_arrow_centers(pairs: list[dict], H: int, W: int) -> bool:
    order = list(range(len(pairs)))
    order.sort(
        key=lambda idx: (
            not pairs[idx]["require_clipped"],
            -int(pairs[idx]["output_mask"].sum()),
        )
    )
    def search(
        depth: int,
        input_forbidden: np.ndarray,
        output_forbidden: np.ndarray,
    ) -> bool:
        if depth == len(order):
            return True
        idx = order[depth]
        pair = pairs[idx]
        candidates: list[tuple[tuple[int, int], list[tuple[int, int]], list[tuple[int, int]]]] = []
        for r in range(1, H - 1):
            for c in range(1, W - 1):
                center = (r, c)
                arrow_cells = _arrow_coords(pair["arrow_dir"], center)
                if not _placeable(arrow_cells, input_forbidden):
                    continue
                stamp_cells, is_clipped = _stamp_cells(
                    pair["output_cells"],
                    pair["output_mask"].shape,
                    center,
                    (H, W),
                )
                if pair["require_clipped"] != is_clipped:
                    continue
                if len(stamp_cells) < 2:
                    continue
                if not _placeable(stamp_cells, output_forbidden):
                    continue
                candidates.append((center, arrow_cells, stamp_cells))
        random.shuffle(candidates)
        for center, arrow_cells, stamp_cells in candidates:
            next_input = input_forbidden.copy()
            next_input |= _dilate_coords(arrow_cells, (H, W))
            next_output = output_forbidden.copy()
            next_output |= _dilate_coords(stamp_cells, (H, W))
            pair["arrow_center"] = center
            if search(depth + 1, next_input, next_output):
                return True
        pair.pop("arrow_center", None)
        return False

    return search(0, np.zeros((H, W), dtype=bool), np.zeros((H, W), dtype=bool))


def _pick_shape_locations(pairs: list[dict], H: int, W: int) -> bool:
    forbidden = np.zeros((H, W), dtype=bool)
    for pair in pairs:
        forbidden |= _dilate_coords(_arrow_coords(pair["arrow_dir"], pair["arrow_center"]), (H, W))

    remaining = set(range(len(pairs)))

    def candidates_for(idx: int, current_forbidden: np.ndarray) -> list[tuple[tuple[int, int], list[tuple[int, int]]]]:
        pair = pairs[idx]
        h, w = pair["shape_mask"].shape
        options: list[tuple[tuple[int, int], list[tuple[int, int]]]] = []
        for r0 in range(0, H - h + 1):
            for c0 in range(0, W - w + 1):
                coords = [(r0 + r, c0 + c) for r, c in pair["shape_cells"]]
                if _placeable(coords, current_forbidden):
                    options.append(((r0, c0), coords))
        random.shuffle(options)
        return options

    def search(current_forbidden: np.ndarray) -> bool:
        if not remaining:
            return True
        best_idx = None
        best_options = None
        for idx in sorted(remaining):
            options = candidates_for(idx, current_forbidden)
            if not options:
                return False
            if best_options is None or len(options) < len(best_options):
                best_idx = idx
                best_options = options
        assert best_idx is not None and best_options is not None
        remaining.remove(best_idx)
        pair = pairs[best_idx]
        for top_left, coords in best_options:
            next_forbidden = current_forbidden.copy()
            next_forbidden |= _dilate_coords(coords, (H, W))
            pair["shape_top_left"] = top_left
            if search(next_forbidden):
                return True
        pair.pop("shape_top_left", None)
        remaining.add(best_idx)
        return False

    return search(forbidden)


def _build_input_grid(pairs: list[dict], H: int, W: int) -> np.ndarray:
    grid = np.zeros((H, W), dtype=int)
    for pair in pairs:
        for r, c in _arrow_coords(pair["arrow_dir"], pair["arrow_center"]):
            grid[r, c] = int(pair["color"])
        r0, c0 = pair["shape_top_left"]
        for r, c in [(r0 + r, c0 + c) for r, c in pair["shape_cells"]]:
            grid[r, c] = int(pair["color"])
    return grid


def _valid_component_structure(grid: np.ndarray, n_pairs: int) -> bool:
    colors = [int(c) for c in np.unique(grid) if int(c) != 0]
    if len(colors) != n_pairs:
        return False
    for color in colors:
        comps = _component_masks(grid, color)
        if len(comps) != 2:
            return False
        arrow_count = 0
        for coords in comps:
            if len(coords) == 5 and _arrow_direction(_mask_from_coords(coords)) is not None:
                arrow_count += 1
        if arrow_count != 1:
            return False
    return True


def _make_pairs(n_pairs: int) -> list[dict]:
    colors = random.sample(FOREGROUND_COLORS, n_pairs)
    clip_count = random.randint(0, min(2, n_pairs))
    clip_indices = set(random.sample(range(n_pairs), clip_count))
    pairs: list[dict] = []
    for idx, color in enumerate(colors):
        base = random.choice(SHAPE_LIBRARY)
        natural_rot = random.randrange(4)
        shape_mask = _rotate_mask(base, natural_rot)
        natural_dir = _natural_direction(shape_mask)
        arrow_dir = random.randrange(4)
        output_mask = _rotate_mask(shape_mask, (arrow_dir - natural_dir) % 4)
        pairs.append(
            {
                "color": int(color),
                "shape_mask": shape_mask,
                "shape_cells": _coords_from_mask(shape_mask),
                "natural_dir": int(natural_dir),
                "arrow_dir": int(arrow_dir),
                "output_mask": output_mask,
                "output_cells": _coords_from_mask(output_mask),
                "require_clipped": idx in clip_indices,
            }
        )
    return pairs


def _gen_attempt() -> np.ndarray | None:
    n_pairs = random.randint(2, 4)
    min_side = {2: 15, 3: 18, 4: 21}[n_pairs]
    H = random.randint(min_side, 30)
    W = random.randint(min_side, 30)
    pairs = _make_pairs(n_pairs)

    if not _pick_arrow_centers(pairs, H, W):
        return None
    if not _pick_shape_locations(pairs, H, W):
        return None

    grid = _build_input_grid(pairs, H, W)
    if not _valid_component_structure(grid, n_pairs):
        return None

    expected = _expected_output(pairs, H, W)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None

    if not (grid.ndim == 2 and expected.ndim == 2):
        return None
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(600):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample for puzzle 6")
