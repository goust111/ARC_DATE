from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
ARC_MAX_DIM = 30
GEN_ATTEMPTS = 500
PAIR_COUNT_CHOICES = (2, 3, 4)
INNER_SIDE_CHOICES = (3, 5)
GAP_SIDES = ("top", "right", "bottom", "left")
COLOR_POOL = tuple(range(1, 10))
PLACEMENT_PAD = 1


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = map(int, mask.shape)
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            queue: deque[tuple[int, int]] = deque([(int(r), int(c))])
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(int(rr), int(cc), h, w):
                    if bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        queue.append((int(nr), int(nc)))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _coords_to_patch(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    patch[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return patch, (int(r0), int(c0), int(r1), int(c1))


def _classify_component(mask_patch: np.ndarray) -> tuple[str, str | None] | None:
    h, w = map(int, mask_patch.shape)
    if bool(mask_patch.all()):
        return ("canvas", None)
    if h < 3 or w < 3:
        return None

    border = np.zeros_like(mask_patch, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    if bool(np.any(mask_patch & (~border))):
        return None
    if int(mask_patch.sum()) != int(border.sum()) - 1:
        return None

    candidates: list[str] = []

    def matches(gap_r: int, gap_c: int) -> bool:
        want = border.copy()
        want[int(gap_r), int(gap_c)] = False
        return bool(np.array_equal(mask_patch, want))

    if w % 2 == 1:
        mid_c = w // 2
        if matches(0, mid_c):
            candidates.append("top")
        if matches(h - 1, mid_c):
            candidates.append("bottom")
    if h % 2 == 1:
        mid_r = h // 2
        if matches(mid_r, 0):
            candidates.append("left")
        if matches(mid_r, w - 1):
            candidates.append("right")

    if len(candidates) == 1:
        return ("stamp", candidates[0])
    return None


def _rotation_k(gap_side: str) -> int:
    return {
        "top": 0,
        "right": 3,   # clockwise 90
        "bottom": 2,
        "left": 1,    # counterclockwise 90
    }[gap_side]


def _detect_stamps_and_canvases(grid: np.ndarray) -> tuple[dict[int, dict], dict[int, dict]]:
    g = np.asarray(grid, dtype=int)
    stamps: dict[int, dict] = {}
    canvases: dict[int, dict] = {}

    for color in sorted(set(map(int, np.unique(g))) - {BG}):
        for coords in _components(g == int(color)):
            patch, bbox = _coords_to_patch(coords)
            info = _classify_component(patch)
            if info is None:
                continue
            kind, gap_side = info
            r0, c0, r1, c1 = bbox
            if kind == "stamp":
                stamps[int(color)] = {
                    "bbox": bbox,
                    "gap_side": gap_side,
                    "interior": g[r0 + 1 : r1, c0 + 1 : c1].copy(),
                }
            else:
                canvases[int(color)] = {
                    "bbox": bbox,
                    "shape": patch.shape,
                }

    return stamps, canvases


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode each stamp's border-gap rotation and print its rotated interior pattern onto the color-matched canvas.

    Input structure:
    - The background is black.
    - Several colored regions are scattered on the grid, of two kinds: hollow 1-pixel-thick rectangular stamp borders and solid rectangular canvases.
    - Each stamp border uses a unique color C, contains a non-black interior pattern, and has exactly one single-cell gap at the center of one border side.
    - Each canvas is a solid rectangle whose color is the same unique color C as exactly one stamp border.

    Rule:
    - Pair every stamp with the canvas of the same border color.
    - Read the gap side as a rotation code for the whole stamp interior: top = 0 degrees, right = clockwise 90 degrees, bottom = 180 degrees, left = counterclockwise 90 degrees.
    - Rotate the stamp interior by that angle and overlay its non-black pattern pixels onto the matched solid canvas.
    - Erase every stamp border and keep only the transformed canvases on the original black background.

    Key insight:
    - The border color identifies which canvas to use, while the missing center pixel on the border encodes how the interior pattern must be rotated before printing.
    """
    g = np.asarray(grid, dtype=int)
    out = np.zeros_like(g)
    stamps, canvases = _detect_stamps_and_canvases(g)

    paired_colors = sorted(set(stamps) & set(canvases))
    assert paired_colors and set(stamps) == set(canvases)

    for color in paired_colors:
        stamp = stamps[int(color)]
        canvas = canvases[int(color)]
        interior = np.asarray(stamp["interior"], dtype=int)
        rotated = np.rot90(interior, _rotation_k(str(stamp["gap_side"])))
        ch, cw = canvas["shape"]
        assert rotated.shape == (int(ch), int(cw))

        patch = np.full((int(ch), int(cw)), int(color), dtype=int)
        patch[rotated != BG] = rotated[rotated != BG]

        r0, c0, r1, c1 = canvas["bbox"]
        assert (r1 - r0 + 1, c1 - c0 + 1) == patch.shape
        out[r0 : r1 + 1, c0 : c1 + 1] = patch

    return out


def _mask_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    h, w = map(int, mask.shape)
    start = tuple(map(int, coords[0]))
    queue: deque[tuple[int, int]] = deque([start])
    seen = {start}
    while queue:
        r, c = queue.popleft()
        for nr, nc in _neighbors4(int(r), int(c), h, w):
            if bool(mask[nr, nc]) and (int(nr), int(nc)) not in seen:
                seen.add((int(nr), int(nc)))
                queue.append((int(nr), int(nc)))
    return len(seen) == int(mask.sum())


def _rotations_unique(mask: np.ndarray) -> bool:
    return len({np.rot90(mask, k).tobytes() for k in range(4)}) == 4


def _random_pattern_mask(side: int) -> np.ndarray | None:
    min_cells = max(3, side)
    max_cells = side * side - 2
    for _ in range(200):
        target = random.randint(min_cells, max_cells)
        start = (random.randrange(side), random.randrange(side))
        cells = {start}
        while len(cells) < target:
            frontier: list[tuple[int, int]] = []
            for r, c in cells:
                for nr, nc in _neighbors4(int(r), int(c), side, side):
                    if (int(nr), int(nc)) not in cells:
                        frontier.append((int(nr), int(nc)))
            if not frontier:
                break
            cells.add(random.choice(frontier))
        if len(cells) != target:
            continue

        mask = np.zeros((side, side), dtype=bool)
        for r, c in cells:
            mask[int(r), int(c)] = True

        rows = np.flatnonzero(mask.any(axis=1))
        cols = np.flatnonzero(mask.any(axis=0))
        if rows.size < 2 or cols.size < 2:
            continue
        if not _mask_connected(mask):
            continue
        if not _rotations_unique(mask):
            continue
        tight_patch, _ = _coords_to_patch(np.argwhere(mask))
        if _classify_component(tight_patch) is not None:
            continue
        return mask
    return None


def _build_stamp(side: int, border_color: int, pattern_color: int, gap_side: str, mask: np.ndarray) -> np.ndarray:
    outer = side + 2
    patch = np.zeros((outer, outer), dtype=int)
    patch[0, :] = int(border_color)
    patch[-1, :] = int(border_color)
    patch[:, 0] = int(border_color)
    patch[:, -1] = int(border_color)

    mid = outer // 2
    if gap_side == "top":
        patch[0, mid] = BG
    elif gap_side == "right":
        patch[mid, outer - 1] = BG
    elif gap_side == "bottom":
        patch[outer - 1, mid] = BG
    else:
        patch[mid, 0] = BG

    patch[1:-1, 1:-1][mask] = int(pattern_color)
    return patch


def _build_canvas(side: int, color: int) -> np.ndarray:
    return np.full((side, side), int(color), dtype=int)


def _render_transformed_canvas(side: int, border_color: int, pattern_color: int, gap_side: str, mask: np.ndarray) -> np.ndarray:
    interior = np.zeros((side, side), dtype=int)
    interior[mask] = int(pattern_color)
    rotated = np.rot90(interior, _rotation_k(gap_side))
    out = np.full((side, side), int(border_color), dtype=int)
    out[rotated != BG] = rotated[rotated != BG]
    return out


def _mark_occupied(occ: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    rr0 = max(0, int(r0) - PLACEMENT_PAD)
    cc0 = max(0, int(c0) - PLACEMENT_PAD)
    rr1 = min(int(occ.shape[0]), int(r0) + int(h) + PLACEMENT_PAD)
    cc1 = min(int(occ.shape[1]), int(c0) + int(w) + PLACEMENT_PAD)
    occ[rr0:rr1, cc0:cc1] = True


def _place_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = map(int, patch.shape)
    grid[r0 : r0 + h, c0 : c0 + w] = patch


def _place_objects(grid: np.ndarray, objects: list[dict], h: int, w: int) -> dict[str, tuple[int, int]] | None:
    occ = np.zeros((h, w), dtype=bool)
    placements: dict[str, tuple[int, int]] = {}

    order = list(range(len(objects)))
    random.shuffle(order)
    order.sort(key=lambda idx: objects[idx]["patch"].shape[0] * objects[idx]["patch"].shape[1], reverse=True)

    for idx in order:
        obj = objects[idx]
        patch = obj["patch"]
        ph, pw = map(int, patch.shape)
        if ph + 2 > h or pw + 2 > w:
            return None

        candidates = [
            (int(r0), int(c0))
            for r0 in range(1, h - ph)
            for c0 in range(1, w - pw)
            if not bool(occ[r0 : r0 + ph, c0 : c0 + pw].any())
        ]
        random.shuffle(candidates)
        if not candidates:
            return None

        chosen = None
        for r0, c0 in candidates:
            rr0 = max(0, r0 - PLACEMENT_PAD)
            cc0 = max(0, c0 - PLACEMENT_PAD)
            rr1 = min(h, r0 + ph + PLACEMENT_PAD)
            cc1 = min(w, c0 + pw + PLACEMENT_PAD)
            if bool(occ[rr0:rr1, cc0:cc1].any()):
                continue
            chosen = (int(r0), int(c0))
            break
        if chosen is None:
            return None

        r0, c0 = chosen
        _place_patch(grid, patch, int(r0), int(c0))
        _mark_occupied(occ, int(r0), int(c0), int(ph), int(pw))
        placements[str(obj["name"])] = (int(r0), int(c0))

    return placements


def _gen_attempt() -> np.ndarray | None:
    n_pairs = random.choice(PAIR_COUNT_CHOICES)
    colors = random.sample(COLOR_POOL, 2 * n_pairs)
    border_colors = colors[:n_pairs]
    pattern_colors = colors[n_pairs:]
    gap_sides = random.sample(GAP_SIDES, k=n_pairs)

    pair_specs: list[dict] = []
    objects: list[dict] = []
    max_outer = 0
    max_inner = 0

    for idx in range(n_pairs):
        side = random.choice(INNER_SIDE_CHOICES)
        mask = _random_pattern_mask(side)
        if mask is None:
            return None

        border_color = int(border_colors[idx])
        pattern_color = int(pattern_colors[idx])
        gap_side = str(gap_sides[idx])

        stamp = _build_stamp(side, border_color, pattern_color, gap_side, mask)
        canvas = _build_canvas(side, border_color)
        transformed = _render_transformed_canvas(side, border_color, pattern_color, gap_side, mask)

        pair_specs.append(
            {
                "name": f"pair{idx}",
                "color": border_color,
                "canvas_patch": canvas,
                "output_patch": transformed,
            }
        )
        objects.append({"name": f"stamp{idx}", "patch": stamp})
        objects.append({"name": f"canvas{idx}", "patch": canvas})

        max_outer = max(max_outer, int(stamp.shape[0]))
        max_inner = max(max_inner, int(canvas.shape[0]))

    min_dim = max(14, max_outer + 8, 5 * n_pairs + 2)
    if min_dim > ARC_MAX_DIM:
        return None
    h = random.randint(min_dim, ARC_MAX_DIM)
    w = random.randint(min_dim, ARC_MAX_DIM)

    grid = np.zeros((h, w), dtype=int)
    placements = _place_objects(grid, objects, h, w)
    if placements is None:
        return None

    expected = np.zeros((h, w), dtype=int)
    for idx, spec in enumerate(pair_specs):
        r0, c0 = placements[f"canvas{idx}"]
        patch = np.asarray(spec["output_patch"], dtype=int)
        ph, pw = map(int, patch.shape)
        expected[r0 : r0 + ph, c0 : c0 + pw] = patch

    try:
        produced = main(grid)
    except Exception:
        return None
    if not np.array_equal(produced, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_ATTEMPTS} attempts")
