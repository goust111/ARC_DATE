from __future__ import annotations

from collections import deque

import numpy as np


BG = 0
SEPARATOR_COLOR = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_GENERATION_ATTEMPTS = 400
_RNG = np.random.default_rng()


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert len(coords) > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return np.asarray(mask[r0 : r1 + 1, c0 : c1 + 1], dtype=bool)


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False
    seen = np.zeros_like(mask, dtype=bool)
    q: deque[tuple[int, int]] = deque([tuple(map(int, coords[0]))])
    seen[q[0]] = True
    count = 0
    while q:
        r, c = q.popleft()
        count += 1
        for dr, dc in DIR4:
            rr, cc = r + dr, c + dc
            if 0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1] and mask[rr, cc] and not seen[rr, cc]:
                seen[rr, cc] = True
                q.append((rr, cc))
    return count == int(mask.sum())


_BASE_SHAPES = [
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1], [1], [1]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 0], [1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 1], [1, 1], [1, 0]], dtype=bool),
    np.array([[1, 0, 1], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[1, 0], [1, 0], [1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1], [1, 1], [1, 1]], dtype=bool),
]


def _build_shape_variants() -> list[np.ndarray]:
    variants: dict[tuple[tuple[int, int], bytes], np.ndarray] = {}
    for base in _BASE_SHAPES:
        for flip in range(2):
            flipped = np.fliplr(base) if flip else base
            for k in range(4):
                mask = _trim_mask(np.rot90(flipped, k))
                area = int(mask.sum())
                if not 3 <= area <= 6:
                    continue
                if not bool(mask[0, 0]):
                    continue
                if mask.shape[0] == 1 and mask.shape[1] in (2, 3):
                    continue
                if not _is_connected(mask):
                    continue
                key = (tuple(map(int, mask.shape)), mask.tobytes())
                variants[key] = mask.copy()
    out = list(variants.values())
    out.sort(key=lambda m: (int(m.sum()), int(m.shape[0]), int(m.shape[1]), tuple(int(x) for x in m.ravel())))
    return out


SHAPES = _build_shape_variants()


def _find_components(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    seen = np.zeros_like(g, dtype=bool)
    comps: list[dict] = []
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            color = int(g[r, c])
            if color == BG or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if (
                        0 <= nr < g.shape[0]
                        and 0 <= nc < g.shape[1]
                        and not seen[nr, nc]
                        and int(g[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            mask[arr[:, 0] - r0, arr[:, 1] - c0] = True
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "mask": mask,
                }
            )
    comps.sort(key=lambda comp: (comp["bbox"][0], comp["bbox"][1], comp["color"]))
    return comps


def _separator_row(grid: np.ndarray) -> int:
    g = np.asarray(grid, dtype=int)
    candidates = []
    for r in range(1, g.shape[0] - 1):
        if int(g[r, 0]) != BG and bool(np.all(g[r] == g[r, 0])):
            candidates.append(r)
    assert len(candidates) == 1, "expected a unique solid separator row"
    return int(candidates[0])


def _is_ruler(comp: dict) -> bool:
    mask = np.asarray(comp["mask"], dtype=bool)
    h, w = mask.shape
    return h == 1 and w in (2, 3) and int(mask.sum()) == w


def _scaled_mask(mask: np.ndarray, factor: int) -> np.ndarray:
    block = np.ones((factor, factor), dtype=np.uint8)
    return np.kron(mask.astype(np.uint8), block).astype(bool)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use each colored ruler's length as the scale factor for the same-colored object,
    then place the scaled object onto the lower canvas with its bounding-box top-left
    aligned to the same-colored anchor pixel.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    sep = _separator_row(g)
    top = g[:sep]
    lower = g[sep + 1 :]
    assert top.size > 0 and lower.size > 0

    top_comps = _find_components(top)
    by_color: dict[int, list[dict]] = {}
    for comp in top_comps:
        by_color.setdefault(int(comp["color"]), []).append(comp)

    objects: dict[int, np.ndarray] = {}
    scales: dict[int, int] = {}
    for color, comps in by_color.items():
        rulers = [comp for comp in comps if _is_ruler(comp)]
        shapes = [comp for comp in comps if not _is_ruler(comp)]
        assert len(rulers) == 1, "each color needs exactly one ruler"
        assert len(shapes) == 1, "each color needs exactly one object"
        obj_mask = np.asarray(shapes[0]["mask"], dtype=bool)
        assert 3 <= int(obj_mask.sum()) <= 6
        objects[color] = obj_mask
        scales[color] = int(rulers[0]["mask"].shape[1])

    out = g.copy()
    for color, obj_mask in objects.items():
        anchor_coords = np.argwhere(lower == color)
        assert len(anchor_coords) == 1, "each object color needs exactly one lower anchor"
        anchor_r, anchor_c = map(int, anchor_coords[0])
        scaled = _scaled_mask(obj_mask, scales[color])
        sh, sw = scaled.shape
        assert anchor_r + sh <= lower.shape[0] and anchor_c + sw <= lower.shape[1]

        for rr, cc in np.argwhere(scaled):
            gr = sep + 1 + anchor_r + int(rr)
            gc = anchor_c + int(cc)
            existing = int(out[gr, gc])
            assert existing in (BG, color), "scaled objects must not overlap other anchors or objects"
            out[gr, gc] = color

    return out


def _rect_is_free(occ: np.ndarray, top: int, left: int, height: int, width: int, *, pad: int = 1) -> bool:
    r0 = max(0, top - pad)
    r1 = min(occ.shape[0], top + height + pad)
    c0 = max(0, left - pad)
    c1 = min(occ.shape[1], left + width + pad)
    return not bool(occ[r0:r1, c0:c1].any())


def _place_rectangles(dims: list[tuple[int, int]], height: int, width: int) -> list[tuple[int, int]] | None:
    occ = np.zeros((height, width), dtype=bool)
    positions: list[tuple[int, int] | None] = [None] * len(dims)
    order = list(range(len(dims)))
    _RNG.shuffle(order)
    order.sort(key=lambda idx: dims[idx][0] * dims[idx][1], reverse=True)

    for idx in order:
        box_h, box_w = dims[idx]
        if box_h > height or box_w > width:
            return None
        candidates: list[tuple[int, int]] = []
        for _ in range(120):
            r = int(_RNG.integers(0, height - box_h + 1))
            c = int(_RNG.integers(0, width - box_w + 1))
            if _rect_is_free(occ, r, c, box_h, box_w):
                candidates.append((r, c))
                if len(candidates) >= 16:
                    break
        if not candidates:
            for r in range(height - box_h + 1):
                for c in range(width - box_w + 1):
                    if _rect_is_free(occ, r, c, box_h, box_w):
                        candidates.append((r, c))
            if not candidates:
                return None
        top, left = candidates[int(_RNG.integers(len(candidates)))]
        positions[idx] = (top, left)
        occ[top : top + box_h, left : left + box_w] = True

    return [pos for pos in positions if pos is not None]


def _stamp_mask(grid: np.ndarray, mask: np.ndarray, color: int, top: int, left: int) -> None:
    coords = np.argwhere(mask)
    for r, c in coords:
        grid[top + int(r), left + int(c)] = color


def _gen_attempt() -> np.ndarray | None:
    n_objects = int(_RNG.integers(2, 5))
    colors = list(_RNG.choice(COLOR_POOL, size=n_objects, replace=False))

    specs: list[dict] = []
    scaled_dims: list[tuple[int, int]] = []
    for color in colors:
        mask = SHAPES[int(_RNG.integers(len(SHAPES)))]
        scale = 2 if bool(_RNG.integers(0, 2) == 0) else 3
        specs.append({"color": int(color), "mask": mask.copy(), "scale": int(scale)})
        scaled_dims.append((int(mask.shape[0] * scale), int(mask.shape[1] * scale)))

    min_width = max(18, max(w for _h, w in scaled_dims) + 2)
    max_width = max(min_width, 32)
    width = int(_RNG.integers(min_width, max_width + 1))

    min_bottom_h = max(12, max(h for h, _w in scaled_dims) + 1)
    max_bottom_h = max(min_bottom_h, min_bottom_h + 8)
    bottom_h = int(_RNG.integers(min_bottom_h, max_bottom_h + 1))

    bottom_positions = _place_rectangles(scaled_dims, bottom_h, width)
    if bottom_positions is None:
        return None

    object_dims = [(int(spec["mask"].shape[0]), int(spec["mask"].shape[1])) for spec in specs]
    ruler_dims = [(1, int(spec["scale"])) for spec in specs]
    top_dims = object_dims + ruler_dims
    min_top_h = max(6, max(h for h, _w in object_dims) + 2)
    max_top_h = max(min_top_h, min_top_h + 3)
    top_h = int(_RNG.integers(min_top_h, max_top_h + 1))

    top_positions = _place_rectangles(top_dims, top_h, width)
    if top_positions is None:
        return None

    grid = np.zeros((top_h + 1 + bottom_h, width), dtype=int)
    sep = top_h
    grid[sep, :] = SEPARATOR_COLOR

    for idx, spec in enumerate(specs):
        obj_r, obj_c = top_positions[idx]
        ruler_r, ruler_c = top_positions[n_objects + idx]
        anchor_r, anchor_c = bottom_positions[idx]

        _stamp_mask(grid, spec["mask"], int(spec["color"]), obj_r, obj_c)
        grid[ruler_r, ruler_c : ruler_c + int(spec["scale"])] = int(spec["color"])
        grid[sep + 1 + anchor_r, anchor_c] = int(spec["color"])

    try:
        out = main(grid)
    except AssertionError:
        return None

    assert np.array_equal(out[: sep + 1], grid[: sep + 1])
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input after repeated attempts")
