import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_RETRIES = 400

SHAPES: dict[str, np.ndarray] = {
    "dot": np.array([[1]], dtype=bool),
    "diag_main": np.array([[1, 0], [0, 1]], dtype=bool),
    "diag_anti": np.array([[0, 1], [1, 0]], dtype=bool),
}


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    rows, cols = np.where(mask)
    if rows.size == 0:
        return np.zeros((0, 0), dtype=bool)
    return mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    trimmed = _trim_mask(mask)
    return tuple(map(int, trimmed.shape)), trimmed.astype(np.uint8).tobytes()


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


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
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _is_palette_tile(sub: np.ndarray) -> tuple[bool, int | None, np.ndarray | None]:
    colors = {int(v) for v in np.unique(sub)}
    if GREY in colors:
        return False, None, None
    non_black = colors - {BLACK}
    if len(non_black) != 1:
        return False, None, None
    base = next(iter(non_black))
    if not np.any(sub == BLACK):
        return False, None, None
    if not np.all((sub == BLACK) | (sub == base)):
        return False, None, None
    color_mask = sub == base
    if not (np.any(color_mask[0, :]) and np.any(color_mask[1, :])):
        return False, None, None
    if not (np.any(color_mask[:, 0]) and np.any(color_mask[:, 1])):
        return False, None, None
    return True, base, sub == BLACK


def _find_bottom_palette(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    if h < 2:
        return []
    tiles = []
    for c in range(w - 1):
        sub = grid[h - 2 : h, c : c + 2]
        ok, base, shape = _is_palette_tile(sub)
        if not ok:
            continue
        if c > 0 and np.any(grid[h - 2 : h, c - 1] != BLACK):
            continue
        if c + 2 < w and np.any(grid[h - 2 : h, c + 2] != BLACK):
            continue
        tiles.append({"bbox": (h - 2, c, h - 1, c + 1), "color": int(base), "shape": shape.copy()})
    return tiles


def _find_right_palette(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    if w < 2:
        return []
    tiles = []
    for r in range(h - 1):
        sub = grid[r : r + 2, w - 2 : w]
        ok, base, shape = _is_palette_tile(sub)
        if not ok:
            continue
        if r > 0 and np.any(grid[r - 1, w - 2 : w] != BLACK):
            continue
        if r + 2 < h and np.any(grid[r + 2, w - 2 : w] != BLACK):
            continue
        tiles.append({"bbox": (r, w - 2, r + 1, w - 1), "color": int(base), "shape": shape.copy()})
    return tiles


def _palette_mapping(grid: np.ndarray) -> tuple[dict[tuple[tuple[int, int], bytes], int], list[dict]]:
    bottom_tiles = _find_bottom_palette(grid)
    right_tiles = _find_right_palette(grid)
    tiles = bottom_tiles if len(bottom_tiles) >= len(right_tiles) else right_tiles
    assert len(tiles) >= 2, "expected a bottom or right palette with multiple 2x2 tiles"

    mapping: dict[tuple[tuple[int, int], bytes], int] = {}
    for tile in tiles:
        key = _mask_key(tile["shape"])
        assert key not in mapping, "palette shape keys must be unique"
        mapping[key] = int(tile["color"])
    return mapping, tiles


def _find_fences(grid: np.ndarray) -> list[dict]:
    fences = []
    for comp in _components(grid == GREY):
        r0, c0, r1, c1 = _bbox(comp)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 4 or w < 4:
            continue
        local = np.zeros((h, w), dtype=bool)
        for r, c in comp:
            local[r - r0, c - c0] = True
        ring = np.zeros((h, w), dtype=bool)
        ring[0, :] = True
        ring[-1, :] = True
        ring[:, 0] = True
        ring[:, -1] = True
        if np.array_equal(local, ring):
            fences.append({"bbox": (r0, c0, r1, c1), "coords": comp})
    assert fences, "expected at least one grey fence"
    return fences


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    mapping, tiles = _palette_mapping(g)
    fences = _find_fences(g)

    out = g.copy()
    for fence in fences:
        r0, c0, r1, c1 = fence["bbox"]
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        shape_mask = (interior != BLACK) & (interior != GREY)
        if not np.any(shape_mask):
            continue
        key = _mask_key(shape_mask)
        assert key in mapping, "interior shape must exist in palette mapping"
        color = mapping[key]
        for r, c in fence["coords"]:
            out[r, c] = color

    for tile in tiles:
        r0, c0, r1, c1 = tile["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = BLACK
    return out


def _embed_mask(mask: np.ndarray, box_h: int, box_w: int) -> np.ndarray:
    mh, mw = mask.shape
    assert mh <= box_h and mw <= box_w
    out = np.zeros((box_h, box_w), dtype=bool)
    top = random.randint(0, box_h - mh)
    left = random.randint(0, box_w - mw)
    out[top : top + mh, left : left + mw] = mask
    return out


def _palette_layout(count: int, limit: int) -> list[int] | None:
    total = count * 2 + (count - 1)
    if total > limit:
        return None
    start = random.randint(0, limit - total)
    return [start + i * 3 for i in range(count)]


def _sample_dimensions(orientation: str, palette_count: int) -> tuple[int, int] | None:
    for _ in range(80):
        if orientation == "bottom":
            h = random.randint(16, 27)
            w = random.randint(max(15, palette_count * 3), 29)
            if _palette_layout(palette_count, w) is None:
                continue
            if h >= 16:
                return h, w
        else:
            h = random.randint(max(15, palette_count * 3), 29)
            w = random.randint(16, 27)
            if _palette_layout(palette_count, h) is None:
                continue
            if w >= 16:
                return h, w
    return None


def _available_box(
    occupied: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int = 1
) -> bool:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(occupied.shape[0] - 1, r1 + pad)
    cc1 = min(occupied.shape[1] - 1, c1 + pad)
    return not bool(np.any(occupied[rr0 : rr1 + 1, cc0 : cc1 + 1]))


def _mark_box(occupied: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int = 1) -> None:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(occupied.shape[0] - 1, r1 + pad)
    cc1 = min(occupied.shape[1] - 1, c1 + pad)
    occupied[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _draw_palette(grid: np.ndarray, entries: list[dict], orientation: str) -> None:
    h, w = grid.shape
    if orientation == "bottom":
        cols = _palette_layout(len(entries), w)
        assert cols is not None
        row = h - 2
        for entry, c0 in zip(entries, cols):
            tile = np.full((2, 2), entry["color"], dtype=int)
            tile[entry["tile_mask"]] = BLACK
            grid[row : row + 2, c0 : c0 + 2] = tile
    else:
        rows = _palette_layout(len(entries), h)
        assert rows is not None
        col = w - 2
        for entry, r0 in zip(entries, rows):
            tile = np.full((2, 2), entry["color"], dtype=int)
            tile[entry["tile_mask"]] = BLACK
            grid[r0 : r0 + 2, col : col + 2] = tile


def _make_case() -> tuple[np.ndarray, np.ndarray] | None:
    shape_names = list(SHAPES)
    random.shuffle(shape_names)
    palette_count = 3
    chosen_names = shape_names[:palette_count]
    orientation = random.choice(("bottom", "right"))
    dims = _sample_dimensions(orientation, palette_count)
    if dims is None:
        return None
    h, w = dims

    grid = np.full((h, w), BLACK, dtype=int)
    out = np.full((h, w), BLACK, dtype=int)

    palette_colors = random.sample(COLOR_POOL, palette_count)
    entries = []
    shape_to_palette: dict[str, int] = {}
    for name, color in zip(chosen_names, palette_colors):
        tile_mask = _embed_mask(SHAPES[name], 2, 2)
        entries.append({"name": name, "color": color, "tile_mask": tile_mask})
        shape_to_palette[name] = color
    random.shuffle(entries)

    _draw_palette(grid, entries, orientation)

    occupied = np.zeros((h, w), dtype=bool)
    if orientation == "bottom":
        occupied[h - 3 :, :] = True
        max_r1 = h - 4
        max_c1 = w - 2
    else:
        occupied[:, w - 3 :] = True
        max_r1 = h - 2
        max_c1 = w - 4
    occupied[0, :] = True
    occupied[-1, :] = True
    occupied[:, 0] = True
    occupied[:, -1] = True

    fence_count = random.randint(3, 6)
    include_empty = fence_count > 1 and random.random() < 0.7
    empty_count = random.randint(1, max(1, fence_count // 2)) if include_empty else 0
    filled_count = fence_count - empty_count
    if filled_count <= 0:
        filled_count = 1
        empty_count = fence_count - 1

    filled_assignments = [random.choice(chosen_names) for _ in range(filled_count)]
    fence_specs = [{"shape": name} for name in filled_assignments] + [{"shape": None} for _ in range(empty_count)]
    random.shuffle(fence_specs)

    for spec in fence_specs:
        placed = False
        for _ in range(120):
            outer_h = random.randint(4, 8)
            outer_w = random.randint(4, 8)
            if outer_h > max_r1 - 1 or outer_w > max_c1 - 1:
                continue
            r0 = random.randint(1, max_r1 - outer_h + 1)
            c0 = random.randint(1, max_c1 - outer_w + 1)
            r1 = r0 + outer_h - 1
            c1 = c0 + outer_w - 1
            if not _available_box(occupied, r0, c0, r1, c1, pad=1):
                continue

            grid[r0 : r1 + 1, c0] = GREY
            grid[r0 : r1 + 1, c1] = GREY
            grid[r0, c0 : c1 + 1] = GREY
            grid[r1, c0 : c1 + 1] = GREY
            out[r0 : r1 + 1, c0] = GREY
            out[r0 : r1 + 1, c1] = GREY
            out[r0, c0 : c1 + 1] = GREY
            out[r1, c0 : c1 + 1] = GREY

            if spec["shape"] is not None:
                interior_h = outer_h - 2
                interior_w = outer_w - 2
                name = spec["shape"]
                mask = _embed_mask(SHAPES[name], interior_h, interior_w)
                shape_color = random.choice(COLOR_POOL)
                inner = grid[r0 + 1 : r1, c0 + 1 : c1]
                inner[mask] = shape_color
                out_inner = out[r0 + 1 : r1, c0 + 1 : c1]
                out_inner[mask] = shape_color
                recolor = shape_to_palette[name]
                out[r0 : r1 + 1, c0] = recolor
                out[r0 : r1 + 1, c1] = recolor
                out[r0, c0 : c1 + 1] = recolor
                out[r1, c0 : c1 + 1] = recolor

            _mark_box(occupied, r0, c0, r1, c1, pad=1)
            placed = True
            break
        if not placed:
            return None

    if not np.any(out != BLACK):
        return None
    return grid, out


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        case = _make_case()
        if case is not None:
            return case[0]
    raise RuntimeError("failed to generate a valid input")
