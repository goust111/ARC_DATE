import random
from collections import deque

import numpy as np


CORNERS = ("tl", "tr", "bl", "br")
EDGES = ("top", "bottom", "left", "right")
GEN_RETRIES = 400


def _mode_color(grid: np.ndarray) -> int:
    flat = np.asarray(grid, dtype=int).ravel()
    assert flat.size > 0
    return int(np.bincount(flat).argmax())


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _touches_edge(cell: tuple[int, int], h: int, w: int) -> bool:
    r, c = cell
    return r == 0 or c == 0 or r == h - 1 or c == w - 1


def _analyze_l_piece(grid: np.ndarray, color: int, points: list[tuple[int, int]]) -> dict:
    r0, c0, r1, c1 = _bbox(points)
    sub = grid[r0 : r1 + 1, c0 : c1 + 1] == color
    miss = ~sub
    assert miss.any() and not miss.all(), "piece must be a rectangle with one missing corner block"

    miss_pts = [tuple(map(int, pt)) for pt in np.argwhere(miss)]
    mr0, mc0, mr1, mc1 = _bbox(miss_pts)
    miss_h = mr1 - mr0 + 1
    miss_w = mc1 - mc0 + 1
    assert int(miss.sum()) == miss_h * miss_w, "missing cells must form one solid rectangle"
    assert np.all(miss[mr0 : mr1 + 1, mc0 : mc1 + 1])

    top = mr0 == 0
    bottom = mr1 == sub.shape[0] - 1
    left = mc0 == 0
    right = mc1 == sub.shape[1] - 1
    assert (top ^ bottom) and (left ^ right), "missing rectangle must occupy exactly one corner"

    if top and left:
        corner = "tl"
    elif top and right:
        corner = "tr"
    elif bottom and left:
        corner = "bl"
    else:
        corner = "br"

    return {
        "bbox": (r0, c0, r1, c1),
        "height": r1 - r0 + 1,
        "width": c1 - c0 + 1,
        "missing_corner": corner,
    }


def _anchor_origin(marker: tuple[int, int], rect_h: int, rect_w: int, grid_h: int, grid_w: int) -> tuple[int, int]:
    r, c = marker
    top = r == 0
    bottom = r == grid_h - 1
    left = c == 0
    right = c == grid_w - 1
    assert top + bottom + left + right == 1, "marker must lie on exactly one edge"

    if top:
        origin = (0, c)
    elif bottom:
        origin = (grid_h - rect_h, c)
    elif left:
        origin = (r, 0)
    else:
        origin = (r, grid_w - rect_w)

    r0, c0 = origin
    assert 0 <= r0 <= grid_h - rect_h
    assert 0 <= c0 <= grid_w - rect_w
    return origin


def _extract_problem_specs(grid: np.ndarray) -> tuple[int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    h, w = g.shape
    bg = 0 if np.any(g == 0) else _mode_color(g)

    specs: list[dict] = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) != bg):
        comps = _components(g == color)
        assert len(comps) == 3, "each color must form exactly two L pieces and one marker"

        marker: tuple[int, int] | None = None
        pieces = []
        for comp in comps:
            if len(comp) == 1:
                assert marker is None, "each color must have a unique marker"
                cell = comp[0]
                assert _touches_edge(cell, h, w), "marker must lie on the grid edge"
                marker = cell
            else:
                pieces.append(_analyze_l_piece(g, color, comp))

        assert marker is not None
        assert len(pieces) == 2

        dims = {(piece["height"], piece["width"]) for piece in pieces}
        assert len(dims) == 1, "same-color pieces must share the same outer rectangle"
        missing_corners = {piece["missing_corner"] for piece in pieces}
        assert len(missing_corners) == 2, "same-color pieces must miss different corners"

        rect_h, rect_w = next(iter(dims))
        origin = _anchor_origin(marker, rect_h, rect_w, h, w)
        specs.append(
            {
                "color": color,
                "origin": origin,
                "height": rect_h,
                "width": rect_w,
            }
        )

    return bg, specs


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg, specs = _extract_problem_specs(g)
    out = np.full(g.shape, bg, dtype=int)

    for spec in specs:
        r0, c0 = spec["origin"]
        h = spec["height"]
        w = spec["width"]
        patch = out[r0 : r0 + h, c0 : c0 + w]
        assert np.all(patch == bg), "anchored rectangles must not overlap"
        patch[:] = int(spec["color"])

    return out


def _l_mask(rect_h: int, rect_w: int, cut_h: int, cut_w: int, missing_corner: str) -> np.ndarray:
    mask = np.ones((rect_h, rect_w), dtype=bool)
    if missing_corner == "tl":
        mask[:cut_h, :cut_w] = False
    elif missing_corner == "tr":
        mask[:cut_h, rect_w - cut_w :] = False
    elif missing_corner == "bl":
        mask[rect_h - cut_h :, :cut_w] = False
    else:
        mask[rect_h - cut_h :, rect_w - cut_w :] = False
    return mask


def _bbox_is_free(occupied: np.ndarray, bbox: tuple[int, int, int, int], pad: int) -> bool:
    h, w = occupied.shape
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    return not bool(occupied[rr0 : rr1 + 1, cc0 : cc1 + 1].any())


def _mark_bbox(occupied: np.ndarray, bbox: tuple[int, int, int, int], pad: int) -> None:
    h, w = occupied.shape
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    occupied[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _sample_anchor(rng: random.Random, grid_h: int, grid_w: int, rect_h: int, rect_w: int, side: str):
    if side in ("top", "bottom"):
        if grid_w - rect_w < 1:
            return None
        c0 = rng.randint(1, grid_w - rect_w)
        if side == "top":
            return (0, c0, rect_h - 1, c0 + rect_w - 1), (0, c0)
        return (grid_h - rect_h, c0, grid_h - 1, c0 + rect_w - 1), (grid_h - 1, c0)

    if grid_h - rect_h < 1:
        return None
    r0 = rng.randint(1, grid_h - rect_h)
    if side == "left":
        return (r0, 0, r0 + rect_h - 1, rect_w - 1), (r0, 0)
    return (r0, grid_w - rect_w, r0 + rect_h - 1, grid_w - 1), (r0, grid_w - 1)


def _draw_l_piece(grid: np.ndarray, top: int, left: int, color: int, mask: np.ndarray) -> bool:
    h, w = mask.shape
    patch = grid[top : top + h, left : left + w]
    if np.any(patch[mask] != 0):
        return False
    patch[mask] = color
    return True


def _build_expected(shape: tuple[int, int], plans: list[dict]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for plan in plans:
        r0, c0 = plan["target_origin"]
        rect_h = plan["rect_h"]
        rect_w = plan["rect_w"]
        out[r0 : r0 + rect_h, c0 : c0 + rect_w] = int(plan["color"])
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    grid_h = rng.randint(16, 22)
    grid_w = rng.randint(16, 22)
    max_colors = 4 if grid_h * grid_w >= 300 else 3
    color_count = rng.choice([1, 2, 2, 3, 3, max_colors])
    color_count = max(1, min(color_count, max_colors, 9))
    colors = rng.sample(list(range(1, 10)), color_count)

    if color_count <= len(EDGES) and rng.random() < 0.5:
        side_hints = rng.sample(list(EDGES), color_count)
    else:
        side_hints = [rng.choice(EDGES) for _ in range(color_count)]

    target_reserved = np.zeros((grid_h, grid_w), dtype=bool)
    plans: list[dict] = []

    for color, side_hint in zip(colors, side_hints):
        placed = False
        for _ in range(250):
            rect_h = rng.randint(3, min(6, grid_h - 3))
            rect_w = rng.randint(3, min(6, grid_w - 3))
            cut_h = rng.randint(1, rect_h - 1)
            cut_w = rng.randint(1, rect_w - 1)
            missing_pair = rng.sample(list(CORNERS), 2)
            side = side_hint if rng.random() < 0.75 else rng.choice(EDGES)

            anchor = _sample_anchor(rng, grid_h, grid_w, rect_h, rect_w, side)
            if anchor is None:
                continue
            bbox, marker = anchor
            if not _bbox_is_free(target_reserved, bbox, pad=1):
                continue

            _mark_bbox(target_reserved, bbox, pad=1)
            plans.append(
                {
                    "color": color,
                    "rect_h": rect_h,
                    "rect_w": rect_w,
                    "cut_h": cut_h,
                    "cut_w": cut_w,
                    "missing_pair": missing_pair,
                    "marker": marker,
                    "target_origin": (bbox[0], bbox[1]),
                    "pieces": [],
                }
            )
            placed = True
            break
        if not placed:
            return None

    source_reserved = target_reserved.copy()
    for plan in plans:
        rect_h = plan["rect_h"]
        rect_w = plan["rect_w"]
        max_r0 = grid_h - rect_h - 1
        max_c0 = grid_w - rect_w - 1
        if max_r0 < 1 or max_c0 < 1:
            return None

        for missing_corner in plan["missing_pair"]:
            piece_placed = False
            for _ in range(300):
                r0 = rng.randint(1, max_r0)
                c0 = rng.randint(1, max_c0)
                bbox = (r0, c0, r0 + rect_h - 1, c0 + rect_w - 1)
                if not _bbox_is_free(source_reserved, bbox, pad=1):
                    continue
                _mark_bbox(source_reserved, bbox, pad=1)
                plan["pieces"].append((r0, c0, missing_corner))
                piece_placed = True
                break
            if not piece_placed:
                return None

    grid = np.zeros((grid_h, grid_w), dtype=int)
    for plan in plans:
        mr, mc = plan["marker"]
        grid[mr, mc] = int(plan["color"])

    for plan in plans:
        mask_cache = {
            corner: _l_mask(plan["rect_h"], plan["rect_w"], plan["cut_h"], plan["cut_w"], corner)
            for corner in plan["missing_pair"]
        }
        for r0, c0, missing_corner in plan["pieces"]:
            if not _draw_l_piece(grid, r0, c0, int(plan["color"]), mask_cache[missing_corner]):
                return None

    expected = _build_expected((grid_h, grid_w), plans)
    try:
        _, parsed_specs = _extract_problem_specs(grid)
    except AssertionError:
        return None
    if len(parsed_specs) != color_count:
        return None
    if not np.array_equal(main(grid), expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a valid puzzle instance")
