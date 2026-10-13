from __future__ import annotations

from collections import Counter
import random

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _neighbors4(r: int, c: int) -> tuple[Coord, Coord, Coord, Coord]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for sr, sc in np.argwhere(mask):
        sr = int(sr)
        sc = int(sc)
        if seen[sr, sc]:
            continue
        stack = [(sr, sc)]
        seen[sr, sc] = True
        cells: list[Coord] = []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(np.array(cells, dtype=int))
    return comps


def _mask_from_coords(coords: np.ndarray) -> tuple[np.ndarray, BBox]:
    r0, c0, r1, c1 = _bbox_coords(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel = coords - np.array([r0, c0], dtype=int)
    mask[rel[:, 0], rel[:, 1]] = True
    return mask, (r0, c0, r1, c1)


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return (tuple(map(int, mask.shape)), mask.tobytes())


def _coords_key(coords: np.ndarray) -> tuple[tuple[int, int], bytes]:
    mask, _ = _mask_from_coords(coords)
    return _mask_key(mask)


def _enclosed_holes(subgrid: np.ndarray, bg: int) -> list[np.ndarray]:
    holes: list[np.ndarray] = []
    h, w = subgrid.shape
    for comp in _components_mask(subgrid == bg):
        touches_edge = any(
            int(r) == 0 or int(c) == 0 or int(r) == h - 1 or int(c) == w - 1
            for r, c in comp
        )
        if not touches_edge:
            holes.append(comp)
    return holes


def _frame_hole_info(
    grid: np.ndarray, color: int, coords: np.ndarray, bg: int
) -> tuple[BBox, np.ndarray, tuple[tuple[int, int], bytes]] | None:
    mask, bbox = _mask_from_coords(coords)
    if mask.shape[0] < 3 or mask.shape[1] < 3:
        return None
    if not (mask[0, :].all() and mask[-1, :].all() and mask[:, 0].all() and mask[:, -1].all()):
        return None

    r0, c0, r1, c1 = bbox
    subgrid = grid[r0 : r1 + 1, c0 : c1 + 1]
    holes = _enclosed_holes(subgrid, bg)
    if len(holes) != 1:
        return None

    hole = holes[0]
    hole_mask, _ = _mask_from_coords(hole)
    return bbox, hole, _mask_key(hole_mask)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each outside monochrome fragment into the same-color hole inside its broken frame.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    out = g.copy()

    for color in (int(c) for c in np.unique(g) if int(c) != bg):
        components = _components_mask(g == color)
        frames: list[tuple[BBox, np.ndarray, tuple[tuple[int, int], bytes]]] = []
        fragments_by_key: dict[tuple[tuple[int, int], bytes], list[np.ndarray]] = {}

        for comp in components:
            hole_info = _frame_hole_info(g, color, comp, bg)
            if hole_info is None:
                fragments_by_key.setdefault(_coords_key(comp), []).append(comp)
            else:
                frames.append(hole_info)

        for bbox, hole, key in frames:
            matches = fragments_by_key.get(key)
            assert matches, f"missing fragment for color {color}"
            frag = matches.pop()
            out[frag[:, 0], frag[:, 1]] = bg
            r0, c0, _r1, _c1 = bbox
            out[r0 + hole[:, 0], c0 + hole[:, 1]] = color

    return out


# --------------- generator ---------------

BG = 0
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
GRID_H_RANGE = (24, 30)
GRID_W_RANGE = (24, 30)
NUM_FRAMES_RANGE = (2, 4)
MIN_FRAME_DIM = 6
BASE_MAX_FRAME_DIM = 10
HOLE_AREA_RANGE = (2, 7)
OBJECT_PADDING = 1
POLY_ATTEMPTS = 60
GEN_ATTEMPTS = 400


def _frame_dim_limit(num_frames: int) -> int:
    return BASE_MAX_FRAME_DIM - max(0, num_frames - 2)


def _normalize_cells(cells: set[Coord]) -> tuple[list[Coord], int, int]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    normalized = sorted((r - min_r, c - min_c) for r, c in cells)
    height = max(r for r, _ in normalized) + 1
    width = max(c for _, c in normalized) + 1
    return normalized, height, width


def _cells_to_mask(cells: set[Coord]) -> np.ndarray:
    normalized, h, w = _normalize_cells(cells)
    mask = np.zeros((h, w), dtype=bool)
    for r, c in normalized:
        mask[r, c] = True
    return mask


def _random_polyomino(max_h: int, max_w: int, min_area: int, max_area: int) -> np.ndarray | None:
    for _ in range(POLY_ATTEMPTS):
        target = random.randint(min_area, max_area)
        cells: set[Coord] = {(0, 0)}

        while len(cells) < target:
            frontier: list[Coord] = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if (nr, nc) in cells:
                        continue
                    trial = set(cells)
                    trial.add((nr, nc))
                    _, h, w = _normalize_cells(trial)
                    if h <= max_h and w <= max_w:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))

        if len(cells) != target:
            continue

        mask = _cells_to_mask(cells)
        if random.random() < 0.5:
            mask = np.fliplr(mask)
        mask = np.rot90(mask, random.randrange(4))
        if mask.shape[0] <= max_h and mask.shape[1] <= max_w:
            return mask
    return None


def _make_frame_spec(color: int, num_frames: int) -> dict | None:
    max_dim = _frame_dim_limit(num_frames)
    frame_h = random.randint(MIN_FRAME_DIM, max_dim)
    frame_w = random.randint(MIN_FRAME_DIM, max_dim)
    inner_h = frame_h - 2
    inner_w = frame_w - 2

    hole_mask = _random_polyomino(
        inner_h,
        inner_w,
        HOLE_AREA_RANGE[0],
        min(HOLE_AREA_RANGE[1], inner_h * inner_w - 1),
    )
    if hole_mask is None:
        return None

    hole_h, hole_w = map(int, hole_mask.shape)
    hole_r = random.randint(1, frame_h - hole_h - 1)
    hole_c = random.randint(1, frame_w - hole_w - 1)

    frame_patch = np.full((frame_h, frame_w), color, dtype=int)
    window = frame_patch[hole_r : hole_r + hole_h, hole_c : hole_c + hole_w]
    window[hole_mask] = BG

    fragment_patch = np.full((hole_h, hole_w), BG, dtype=int)
    fragment_patch[hole_mask] = color

    return {
        "color": color,
        "frame_patch": frame_patch,
        "fragment_patch": fragment_patch,
        "hole_mask": hole_mask,
        "hole_rc": (hole_r, hole_c),
    }


def _can_place(reserve: np.ndarray, r0: int, c0: int, h: int, w: int, padding: int) -> bool:
    grid_h, grid_w = reserve.shape
    if r0 < 0 or c0 < 0 or r0 + h > grid_h or c0 + w > grid_w:
        return False
    rr0 = max(0, r0 - padding)
    cc0 = max(0, c0 - padding)
    rr1 = min(grid_h, r0 + h + padding)
    cc1 = min(grid_w, c0 + w + padding)
    return not reserve[rr0:rr1, cc0:cc1].any()


def _random_location(reserve: np.ndarray, shape: tuple[int, int], padding: int) -> Coord | None:
    h, w = map(int, shape)
    grid_h, grid_w = reserve.shape
    positions: list[Coord] = []
    for r0 in range(grid_h - h + 1):
        for c0 in range(grid_w - w + 1):
            if _can_place(reserve, r0, c0, h, w, padding):
                positions.append((r0, c0))
    if not positions:
        return None
    return random.choice(positions)


def _paste_non_bg(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    view = dst[r0 : r0 + h, c0 : c0 + w]
    view[patch != BG] = patch[patch != BG]


def _mark_reserve(reserve: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    reserve[r0 : r0 + h, c0 : c0 + w] = True


def _build_expected(inp: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = inp.copy()
    for spec in specs:
        color = int(spec["color"])
        hole_mask = spec["hole_mask"]
        hole_h, hole_w = hole_mask.shape

        frag_r, frag_c = spec["fragment_pos"]
        frag_view = out[frag_r : frag_r + hole_h, frag_c : frag_c + hole_w]
        frag_view[hole_mask] = BG

        frame_r, frame_c = spec["frame_pos"]
        hole_r, hole_c = spec["hole_rc"]
        hole_view = out[
            frame_r + hole_r : frame_r + hole_r + hole_h,
            frame_c + hole_c : frame_c + hole_c + hole_w,
        ]
        hole_view[hole_mask] = color
    return out


def _gen_attempt() -> np.ndarray | None:
    num_frames = random.randint(*NUM_FRAMES_RANGE)
    colors = random.sample(PALETTE, num_frames)
    specs: list[dict] = []
    for color in colors:
        spec = _make_frame_spec(color, num_frames)
        if spec is None:
            return None
        specs.append(spec)

    grid_h = random.randint(*GRID_H_RANGE)
    grid_w = random.randint(*GRID_W_RANGE)
    if max(spec["frame_patch"].shape[0] for spec in specs) > grid_h:
        return None
    if max(spec["frame_patch"].shape[1] for spec in specs) > grid_w:
        return None

    inp = np.full((grid_h, grid_w), BG, dtype=int)
    reserve = np.zeros((grid_h, grid_w), dtype=bool)

    frame_order = sorted(range(num_frames), key=lambda i: specs[i]["frame_patch"].size, reverse=True)
    for idx in frame_order:
        patch = specs[idx]["frame_patch"]
        pos = _random_location(reserve, patch.shape, OBJECT_PADDING)
        if pos is None:
            return None
        r0, c0 = pos
        specs[idx]["frame_pos"] = pos
        _paste_non_bg(inp, patch, r0, c0)
        _mark_reserve(reserve, r0, c0, patch.shape[0], patch.shape[1])

    frag_order = sorted(range(num_frames), key=lambda i: specs[i]["fragment_patch"].size, reverse=True)
    for idx in frag_order:
        patch = specs[idx]["fragment_patch"]
        pos = _random_location(reserve, patch.shape, OBJECT_PADDING)
        if pos is None:
            return None
        r0, c0 = pos
        specs[idx]["fragment_pos"] = pos
        _paste_non_bg(inp, patch, r0, c0)
        _mark_reserve(reserve, r0, c0, patch.shape[0], patch.shape[1])

    expected = _build_expected(inp, specs)
    try:
        if np.array_equal(main(inp), expected):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate a valid input")
