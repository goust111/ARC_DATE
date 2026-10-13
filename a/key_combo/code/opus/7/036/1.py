from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


Corner = str
BBox = tuple[int, int, int, int]

_CLOCKWISE_NEXT: dict[Corner, Corner] = {
    "TL": "TR",
    "TR": "BR",
    "BR": "BL",
    "BL": "TL",
}
_CLOCKWISE_ORDER: tuple[Corner, ...] = ("TL", "TR", "BR", "BL")
_CORNER_COLORS: tuple[int, ...] = (1, 2, 3, 4)
_NOISE_COLOR = 5
_BG = 0
_NEIGHBORS4: tuple[tuple[int, int], ...] = ((1, 0), (-1, 0), (0, 1), (0, -1))


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox_from_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue

            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in _NEIGHBORS4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            coords = np.array(cells, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox_from_coords(coords),
                }
            )

    return comps


def _bbox_corner(box: BBox, height: int, width: int) -> Corner | None:
    r0, c0, r1, c1 = box
    if r0 == 0 and c0 == 0:
        return "TL"
    if r0 == 0 and c1 == width - 1:
        return "TR"
    if r1 == height - 1 and c1 == width - 1:
        return "BR"
    if r1 == height - 1 and c0 == 0:
        return "BL"
    return None


def _corner_anchor(corner: Corner, height: int, width: int, obj_h: int, obj_w: int) -> tuple[int, int]:
    if corner == "TL":
        return 0, 0
    if corner == "TR":
        return 0, width - obj_w
    if corner == "BR":
        return height - obj_h, width - obj_w
    if corner == "BL":
        return height - obj_h, 0
    raise AssertionError(f"unknown corner {corner}")


def _extract_corner_objects(grid: np.ndarray) -> dict[Corner, dict[str, object]]:
    bg = get_bg(grid)
    height, width = grid.shape
    by_corner: dict[Corner, dict[str, object]] = {}
    for comp in _find_components(grid, bg):
        corner = _bbox_corner(comp["bbox"], height, width)
        if corner is None:
            continue
        assert corner not in by_corner, "expected at most one corner object per corner"
        by_corner[corner] = comp
    return by_corner


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy each corner object clockwise, and fill the copied object's bbox gaps with the evicted corner color.

    Input structure:
    - The background is black.
    - Relevant foreground objects are monochromatic 4-connected corner components whose tight bboxes touch two adjacent borders at one true grid corner.
    - Interior grey distractor pixels or blobs do not touch a true corner and must be ignored.

    Rule:
    - Keep the input grid as the starting canvas and identify the original corner object, if any, at each of TL, TR, BR, and BL.
    - Process source corners in clockwise order TL -> TR -> BR -> BL; each source stamps its mask and source color into the next clockwise corner, aligned by bbox.
    - If the destination corner originally contained an object, treat that original resident as evicted and first fill the copied bbox's non-mask cells with the evicted object's color; otherwise those copied bbox gaps stay black.
    - Later clockwise stamps overwrite earlier pixels, but source objects are never removed from their original locations.

    Key insight:
    - The destination fill color is not fixed by adjacency alone; it is the original color of the object displaced from that destination corner, so spatial copying and color transfer must be solved together.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    assert int(bg) == _BG, "expected black background"
    height, width = map(int, g.shape)
    by_corner = _extract_corner_objects(g)
    assert len(by_corner) >= 2, "expected at least two occupied corners"

    out = g.copy()
    for src_corner in _CLOCKWISE_ORDER:
        if src_corner not in by_corner:
            continue

        comp = by_corner[src_corner]
        color = int(comp["color"])
        coords = np.asarray(comp["coords"], dtype=int)
        r0, c0, r1, c1 = comp["bbox"]
        obj_h = int(r1 - r0 + 1)
        obj_w = int(c1 - c0 + 1)

        dst_corner = _CLOCKWISE_NEXT[src_corner]
        dst_r0, dst_c0 = _corner_anchor(dst_corner, height, width, obj_h, obj_w)
        fill_color = int(by_corner[dst_corner]["color"]) if dst_corner in by_corner else bg

        out[dst_r0 : dst_r0 + obj_h, dst_c0 : dst_c0 + obj_w] = fill_color
        rel = coords - np.array([r0, c0], dtype=int)
        out[dst_r0 + rel[:, 0], dst_c0 + rel[:, 1]] = color

    return out


def _is_mask_connected(mask: np.ndarray) -> bool:
    points = np.argwhere(mask)
    if len(points) == 0:
        return False

    seen = {tuple(map(int, points[0]))}
    q = deque([tuple(map(int, points[0]))])
    h, w = mask.shape
    while q:
        r, c = q.popleft()
        for dr, dc in _NEIGHBORS4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == len(points)


def _build_mask_library() -> dict[tuple[int, int, int], list[np.ndarray]]:
    library: dict[tuple[int, int, int], list[np.ndarray]] = {}
    dims = ((2, 2), (2, 3), (3, 2), (2, 4), (4, 2), (3, 3))

    for h, w in dims:
        cell_count = h * w
        for bits in range(1 << cell_count):
            area = bits.bit_count()
            if not 3 <= area <= 8:
                continue

            mask = np.zeros((h, w), dtype=bool)
            for idx in range(cell_count):
                if bits & (1 << idx):
                    mask[idx // w, idx % w] = True

            if not mask[0].any() or not mask[-1].any() or not mask[:, 0].any() or not mask[:, -1].any():
                continue
            if not _is_mask_connected(mask):
                continue

            gaps = cell_count - area
            if 0 <= gaps <= 3:
                library.setdefault((h, w, gaps), []).append(mask)

    assert any(key[2] > 0 for key in library), "expected masks with bbox gaps"
    return library


_MASK_LIBRARY = _build_mask_library()


def _choose_corner_mask(require_gap: bool) -> np.ndarray:
    choices: list[tuple[int, int, int]] = []
    for key, masks in _MASK_LIBRARY.items():
        if not masks:
            continue
        gaps = key[2]
        if require_gap and gaps == 0:
            continue
        if not require_gap and gaps > 0 and random.random() < 0.35:
            continue
        choices.append(key)

    if not choices:
        choices = [key for key in _MASK_LIBRARY if (not require_gap) or key[2] > 0]

    key = random.choice(choices)
    return random.choice(_MASK_LIBRARY[key]).copy()


def _stamp_corner_object(grid: np.ndarray, corner: Corner, mask: np.ndarray, color: int) -> None:
    h, w = grid.shape
    obj_h, obj_w = mask.shape
    r0, c0 = _corner_anchor(corner, h, w, obj_h, obj_w)
    rr, cc = np.where(mask)
    grid[r0 + rr, c0 + cc] = int(color)


def _occupied_corners() -> tuple[Corner, ...]:
    count = random.choices([2, 3, 4], weights=[4, 3, 2], k=1)[0]
    if count == 4:
        return _CLOCKWISE_ORDER
    if count == 3:
        return tuple(sorted(random.sample(_CLOCKWISE_ORDER, 3), key=_CLOCKWISE_ORDER.index))

    adjacent_pairs = [("TL", "TR"), ("TR", "BR"), ("BR", "BL"), ("BL", "TL")]
    opposite_pairs = [("TL", "BR"), ("TR", "BL")]
    if random.random() < 0.75:
        return random.choice(adjacent_pairs)
    return random.choice(opposite_pairs)


def _place_noise(grid: np.ndarray) -> None:
    h, w = grid.shape
    if h <= 8 or w <= 8:
        return

    candidates = [
        (r, c)
        for r in range(4, h - 4)
        for c in range(4, w - 4)
        if int(grid[r, c]) == _BG
    ]
    if not candidates:
        return

    noise_count = random.randint(0, min(5, len(candidates)))
    for r, c in random.sample(candidates, noise_count):
        grid[r, c] = _NOISE_COLOR


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(12, 16)
    width = random.randint(12, 16)
    grid = np.zeros((height, width), dtype=int)

    occupied = _occupied_corners()
    colors = random.sample(list(_CORNER_COLORS), k=len(occupied))
    targets_with_eviction = {corner for corner in occupied if _CLOCKWISE_NEXT[corner] in occupied}
    must_show_gap = set(random.sample(list(targets_with_eviction), k=1)) if targets_with_eviction else set()

    for corner, color in zip(occupied, colors):
        require_gap = corner in must_show_gap or random.random() < 0.55
        mask = _choose_corner_mask(require_gap=require_gap)
        _stamp_corner_object(grid, corner, mask, color)

    _place_noise(grid)

    by_corner = _extract_corner_objects(grid)
    if set(by_corner) != set(occupied):
        return None
    if len({int(by_corner[corner]["color"]) for corner in occupied}) != len(occupied):
        return None

    visible_gap_case = False
    for corner in occupied:
        comp = by_corner[corner]
        r0, c0, r1, c1 = comp["bbox"]
        area = len(np.asarray(comp["coords"]))
        gaps = (r1 - r0 + 1) * (c1 - c0 + 1) - area
        if gaps < 0 or gaps > 3:
            return None
        if _CLOCKWISE_NEXT[corner] in by_corner and gaps > 0:
            visible_gap_case = True

    if targets_with_eviction and not visible_gap_case:
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None

    unique_values = set(map(int, np.unique(grid)))
    if not unique_values <= {_BG, _NOISE_COLOR, *_CORNER_COLORS}:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
