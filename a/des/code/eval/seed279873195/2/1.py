from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


SPRITE_FILL_MASKS = (
    np.array([[0, 0, 0], [0, 1, 1], [0, 0, 0]], dtype=bool),
    np.array([[0, 1, 0], [0, 1, 0], [0, 0, 0]], dtype=bool),
    np.array([[0, 0, 0], [1, 1, 0], [0, 0, 0]], dtype=bool),
    np.array([[0, 0, 0], [0, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[0, 1, 0], [0, 1, 1], [0, 0, 0]], dtype=bool),
    np.array([[0, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=bool),
)

GEN_RETRIES = 300


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _sprite_info(sprite: np.ndarray) -> tuple[int, int] | None:
    if sprite.shape != (3, 3):
        return None
    colors = {int(v) for v in np.unique(sprite)}
    if len(colors) != 2 or BLACK in colors or GREY in colors:
        return None
    fill = int(sprite[1, 1])
    border_candidates = list(colors - {fill})
    if len(border_candidates) != 1:
        return None
    border = border_candidates[0]
    perimeter = np.array(
        [
            sprite[0, 0],
            sprite[0, 1],
            sprite[0, 2],
            sprite[1, 0],
            sprite[1, 2],
            sprite[2, 0],
            sprite[2, 1],
            sprite[2, 2],
        ],
        dtype=int,
    )
    if not np.any(perimeter == border):
        return None
    return border, fill


def _parse_horizontal_gallery(grid: np.ndarray, row0: int, edge: str) -> dict | None:
    if grid.shape[0] < row0 + 3:
        return None
    band = grid[row0 : row0 + 3, :]
    occupied = np.where(np.any(band != BLACK, axis=0))[0]
    if occupied.size == 0:
        return None
    if not np.array_equal(occupied, np.arange(int(occupied[0]), int(occupied[-1]) + 1)):
        return None
    start, end = int(occupied[0]), int(occupied[-1])
    width = end - start + 1
    if width < 3 or (width + 1) % 4 != 0:
        return None
    count = (width + 1) // 4
    sprites = []
    for idx in range(count):
        c0 = start + 4 * idx
        sprite = band[:, c0 : c0 + 3]
        info = _sprite_info(sprite)
        if info is None:
            return None
        border, fill = info
        sprites.append(
            {
                "pair": frozenset((border, fill)),
                "sprite": np.array(sprite, copy=True),
                "origin": (row0, c0),
            }
        )
        if idx < count - 1 and not np.all(band[:, c0 + 3] == GREY):
            return None
    return {
        "edge": edge,
        "count": count,
        "bbox": (row0, start, row0 + 2, end),
        "sprites": sprites,
    }


def _parse_vertical_gallery(grid: np.ndarray, col0: int, edge: str) -> dict | None:
    if grid.shape[1] < col0 + 3:
        return None
    band = grid[:, col0 : col0 + 3]
    occupied = np.where(np.any(band != BLACK, axis=1))[0]
    if occupied.size == 0:
        return None
    if not np.array_equal(occupied, np.arange(int(occupied[0]), int(occupied[-1]) + 1)):
        return None
    start, end = int(occupied[0]), int(occupied[-1])
    height = end - start + 1
    if height < 3 or (height + 1) % 4 != 0:
        return None
    count = (height + 1) // 4
    sprites = []
    for idx in range(count):
        r0 = start + 4 * idx
        sprite = band[r0 : r0 + 3, :]
        info = _sprite_info(sprite)
        if info is None:
            return None
        border, fill = info
        sprites.append(
            {
                "pair": frozenset((border, fill)),
                "sprite": np.array(sprite, copy=True),
                "origin": (r0, col0),
            }
        )
        if idx < count - 1 and not np.all(band[r0 + 3, :] == GREY):
            return None
    return {
        "edge": edge,
        "count": count,
        "bbox": (start, col0, end, col0 + 2),
        "sprites": sprites,
    }


def _find_gallery(grid: np.ndarray) -> dict:
    h, w = grid.shape
    candidates: list[dict] = []
    if h >= 4 and np.all(grid[3, :] == BLACK):
        cand = _parse_horizontal_gallery(grid, 0, "top")
        if cand is not None:
            candidates.append(cand)
    if h >= 4 and np.all(grid[h - 4, :] == BLACK):
        cand = _parse_horizontal_gallery(grid, h - 3, "bottom")
        if cand is not None:
            candidates.append(cand)
    if w >= 4 and np.all(grid[:, 3] == BLACK):
        cand = _parse_vertical_gallery(grid, 0, "left")
        if cand is not None:
            candidates.append(cand)
    if w >= 4 and np.all(grid[:, w - 4] == BLACK):
        cand = _parse_vertical_gallery(grid, w - 3, "right")
        if cand is not None:
            candidates.append(cand)
    assert candidates, "gallery not found"
    candidates.sort(key=lambda item: (item["count"], item["bbox"][2] - item["bbox"][0], item["bbox"][3] - item["bbox"][1]), reverse=True)
    gallery = candidates[0]
    seen_pairs: set[frozenset[int]] = set()
    for item in gallery["sprites"]:
        pair = item["pair"]
        assert pair not in seen_pairs, "gallery color pairs must be unique"
        seen_pairs.add(pair)
    return gallery


def _line_records(grid: np.ndarray, gallery_bbox: tuple[int, int, int, int]) -> list[dict]:
    r0, c0, r1, c1 = gallery_bbox
    gallery_mask = np.zeros_like(grid, dtype=bool)
    gallery_mask[r0 : r1 + 1, c0 : c1 + 1] = True

    anchors = np.argwhere((grid != BLACK) & (grid != GREY) & ~gallery_mask)
    assert anchors.size > 0, "no anchors found"
    anchor_by_pos = {(int(r), int(c)): int(grid[r, c]) for r, c in anchors}

    grey_components = _components((grid == GREY) & ~gallery_mask)
    records: list[dict] = []
    for comp in grey_components:
        rows = comp[:, 0]
        cols = comp[:, 1]
        rmin, rmax = int(rows.min()), int(rows.max())
        cmin, cmax = int(cols.min()), int(cols.max())
        if rmin == rmax:
            orientation = "H"
            assert comp.shape[0] == cmax - cmin + 1
        else:
            assert cmin == cmax, "grey connectors must be axis-aligned"
            orientation = "V"
            assert comp.shape[0] == rmax - rmin + 1

        adjacent: dict[tuple[int, int], int] = {}
        for r, c in comp:
            for dr, dc in NEIGHBORS4:
                nr, nc = int(r + dr), int(c + dc)
                if (nr, nc) in anchor_by_pos:
                    adjacent[(nr, nc)] = anchor_by_pos[(nr, nc)]
        assert len(adjacent) == 2, "each connector must touch exactly two anchors"
        anchor_items = [((r, c), color) for (r, c), color in adjacent.items()]
        ((ar0, ac0), color0), ((ar1, ac1), color1) = anchor_items
        assert orientation == "H" and ar0 == ar1 or orientation == "V" and ac0 == ac1
        mid_r = (ar0 + ar1) // 2
        mid_c = (ac0 + ac1) // 2
        records.append(
            {
                "pair": frozenset((color0, color1)),
                "orientation": orientation,
                "mid": (mid_r, mid_c),
            }
        )
    return records


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    gallery = _find_gallery(g)
    pair_to_sprite = {item["pair"]: item["sprite"] for item in gallery["sprites"]}
    records = _line_records(g, gallery["bbox"])

    out = np.array(g, copy=True)
    r0, c0, r1, c1 = gallery["bbox"]
    out[r0 : r1 + 1, c0 : c1 + 1] = BLACK

    anchor_cells = set(map(tuple, np.argwhere((g != BLACK) & (g != GREY))))
    for record in records:
        sprite = pair_to_sprite[record["pair"]]
        if record["orientation"] == "V":
            sprite = np.rot90(sprite, -1)
        mr, mc = record["mid"]
        for sr in range(3):
            for sc in range(3):
                rr = mr + sr - 1
                cc = mc + sc - 1
                if not (0 <= rr < g.shape[0] and 0 <= cc < g.shape[1]):
                    continue
                if (rr, cc) in anchor_cells:
                    continue
                out[rr, cc] = int(sprite[sr, sc])
    return out


def _sprite_from_colors(border: int, fill: int) -> np.ndarray:
    fill_mask = random.choice(SPRITE_FILL_MASKS)
    return np.where(fill_mask, fill, border).astype(int)


def _gallery_strip_mask(shape: tuple[int, int], edge: str) -> np.ndarray:
    h, w = shape
    mask = np.zeros(shape, dtype=bool)
    if edge == "top":
        mask[:4, :] = True
    elif edge == "bottom":
        mask[h - 4 :, :] = True
    elif edge == "left":
        mask[:, :4] = True
    else:
        mask[:, w - 4 :] = True
    return mask


def _reserve_rect(occupied: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> bool:
    h, w = occupied.shape
    rr0 = max(0, r0)
    cc0 = max(0, c0)
    rr1 = min(h - 1, r1)
    cc1 = min(w - 1, c1)
    if occupied[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
        return False
    occupied[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
    return True


def _sample_line(occupied: np.ndarray, orientation: str) -> dict | None:
    h, w = occupied.shape
    if orientation == "H":
        lengths = [d for d in (4, 6, 8, 10) if d <= w - 3]
        if not lengths:
            return None
    else:
        lengths = [d for d in (4, 6, 8, 10) if d <= h - 3]
        if not lengths:
            return None

    for _ in range(200):
        dist = random.choice(lengths)
        if orientation == "H":
            r = random.randint(1, h - 2)
            c0 = random.randint(1, w - dist - 2)
            c1 = c0 + dist
            if not _reserve_rect(occupied, r - 2, c0 - 1, r + 2, c1 + 1):
                continue
            return {"orientation": "H", "a": (r, c0), "b": (r, c1)}

        c = random.randint(1, w - 2)
        r0 = random.randint(1, h - dist - 2)
        r1 = r0 + dist
        if not _reserve_rect(occupied, r0 - 1, c - 2, r1 + 1, c + 2):
            continue
        return {"orientation": "V", "a": (r0, c), "b": (r1, c)}
    return None


def _place_gallery(grid: np.ndarray, edge: str, sprites: list[np.ndarray]) -> None:
    h, w = grid.shape
    run = 4 * len(sprites) - 1
    if edge in {"top", "bottom"}:
        row0 = 0 if edge == "top" else h - 3
        start = random.randint(1, w - run - 1)
        for idx, sprite in enumerate(sprites):
            c0 = start + 4 * idx
            grid[row0 : row0 + 3, c0 : c0 + 3] = sprite
            if idx < len(sprites) - 1:
                grid[row0 : row0 + 3, c0 + 3] = GREY
        return

    col0 = 0 if edge == "left" else w - 3
    start = random.randint(1, h - run - 1)
    for idx, sprite in enumerate(sprites):
        r0 = start + 4 * idx
        grid[r0 : r0 + 3, col0 : col0 + 3] = sprite
        if idx < len(sprites) - 1:
            grid[r0 + 3, col0 : col0 + 3] = GREY


def _gen_attempt() -> np.ndarray | None:
    line_count = random.randint(2, 4)
    edge = random.choice(("top", "bottom", "left", "right"))
    run = 4 * line_count - 1
    if edge in {"top", "bottom"}:
        h = random.randint(18, 26)
        w = random.randint(max(18, run + 4), 26)
    else:
        h = random.randint(max(18, run + 4), 26)
        w = random.randint(18, 26)

    grid = np.full((h, w), BLACK, dtype=int)
    occupied = _gallery_strip_mask((h, w), edge)

    orientations = ["H", "V"]
    while len(orientations) < line_count:
        orientations.append(random.choice(("H", "V")))
    random.shuffle(orientations)

    line_specs = []
    for orientation in orientations:
        spec = _sample_line(occupied, orientation)
        if spec is None:
            return None
        line_specs.append(spec)

    colors = list(random.sample(COLOR_POOL, 2 * line_count))
    random.shuffle(colors)
    pair_colors = [(colors[2 * i], colors[2 * i + 1]) for i in range(line_count)]

    gallery_sprites = []
    for spec, (color_a, color_b) in zip(line_specs, pair_colors):
        border, fill = (color_a, color_b) if random.random() < 0.5 else (color_b, color_a)
        gallery_sprites.append(_sprite_from_colors(border, fill))
        spec["colors"] = (color_a, color_b)

    shuffled_sprites = list(gallery_sprites)
    random.shuffle(shuffled_sprites)
    _place_gallery(grid, edge, shuffled_sprites)

    for spec in line_specs:
        (ar, ac), (br, bc) = spec["a"], spec["b"]
        color_a, color_b = spec["colors"]
        grid[ar, ac] = color_a
        grid[br, bc] = color_b
        if spec["orientation"] == "H":
            left, right = sorted((ac, bc))
            grid[ar, left + 1 : right] = GREY
        else:
            top, bottom = sorted((ar, br))
            grid[top + 1 : bottom, ac] = GREY

    try:
        gallery = _find_gallery(grid)
        out = main(grid)
    except Exception:
        return None

    gr0, gc0, gr1, gc1 = gallery["bbox"]
    if not np.all(out[gr0 : gr1 + 1, gc0 : gc1 + 1] == BLACK):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
