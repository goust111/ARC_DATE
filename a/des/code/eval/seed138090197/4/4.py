from __future__ import annotations

import random
from collections import deque

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[counts.argmax()])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.asarray(coords, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _has_hole(mask: np.ndarray) -> bool:
    h, w = mask.shape
    bg = ~mask
    seen = np.zeros((h, w), dtype=bool)
    q = deque()
    for r in range(h):
        for c in (0, w - 1):
            if bg[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if bg[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    while q:
        rr, cc = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr = rr + dr
            nc = cc + dc
            if 0 <= nr < h and 0 <= nc < w and bg[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
    return bool(np.any(bg & (~seen)))


def _border_color(patch: np.ndarray, depth: int) -> int | None:
    n = patch.shape[0]
    r0 = depth
    c0 = depth
    r1 = n - 1 - depth
    c1 = n - 1 - depth
    vals = set(map(int, patch[r0, c0 : c1 + 1]))
    vals.update(map(int, patch[r1, c0 : c1 + 1]))
    vals.update(map(int, patch[r0 : r1 + 1, c0]))
    vals.update(map(int, patch[r0 : r1 + 1, c1]))
    if len(vals) != 1:
        return None
    return next(iter(vals))


def _template_layers(patch: np.ndarray, bg: int) -> list[int] | None:
    if patch.ndim != 2 or patch.shape[0] != patch.shape[1] or patch.shape[0] < 3:
        return None
    if np.any(patch == bg):
        return None
    n = patch.shape[0]
    layers: list[int] = []
    for depth in range((n + 1) // 2):
        color = _border_color(patch, depth)
        if color is None:
            return None
        if layers and color == layers[-1]:
            return None
        layers.append(int(color))
    if len(layers) < 2:
        return None
    return layers


def _parse_objects_and_template(grid: np.ndarray) -> tuple[list[dict[str, object]], list[int], int]:
    bg = _get_bg(grid)
    comps = _components(grid != bg)
    template_layers = None
    objects: list[dict[str, object]] = []
    for coords in comps:
        values = set(map(int, grid[tuple(coords.T)]))
        r0, c0, r1, c1 = _bbox(coords)
        if len(values) > 1:
            patch = grid[r0 : r1 + 1, c0 : c1 + 1]
            layers = _template_layers(patch, bg)
            assert layers is not None, "expected the only multicolor component to be the nested template"
            assert template_layers is None, "expected a unique template"
            template_layers = layers
            continue
        assert len(values) == 1, "main objects must be monochromatic"
        color = int(next(iter(values)))
        objects.append(
            {
                "color": color,
                "coords": coords.copy(),
                "bbox": (r0, c0, r1, c1),
                "anchor": ((r0 + r1) // 2, (c0 + c1) // 2),
            }
        )
    assert template_layers is not None, "template not found"
    assert objects, "expected at least one main object"
    return objects, template_layers, bg


def _compose_output(objects: list[dict[str, object]], layers: list[int], bg: int) -> np.ndarray:
    rank = {color: i for i, color in enumerate(layers)}
    ordered = sorted(
        objects,
        key=lambda obj: (
            rank[int(obj["color"])],
            int(obj["bbox"][0]),
            int(obj["bbox"][1]),
        ),
    )
    pixels: dict[tuple[int, int], int] = {}
    for obj in ordered:
        color = int(obj["color"])
        assert color in rank, "object color must appear in the template"
        coords = np.asarray(obj["coords"], dtype=int)
        ar, ac = obj["anchor"]
        rel = coords - np.array([[ar, ac]], dtype=int)
        for rr, cc in rel:
            pixels[(int(rr), int(cc))] = color

    all_coords = np.asarray(list(pixels.keys()), dtype=int)
    min_r, min_c = all_coords.min(axis=0)
    max_r, max_c = all_coords.max(axis=0)
    out = np.full((max_r - min_r + 1, max_c - min_c + 1), int(bg), dtype=int)
    for (rr, cc), color in pixels.items():
        out[rr - min_r, cc - min_c] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the color order from a concentric square template and stack matching shapes by layer.

    Input structure:
    - Background occupies most cells.
    - One corner contains a small square template made of nested rectangular color rings.
    - Elsewhere, separate monochrome 4-connected shapes use only colors seen in the template.

    Rule:
    - Parse the template from outer ring to inner core; this gives bottom-to-top layer order.
    - For every main-region object, use the integer bbox center as its alignment anchor.
    - Overlay all objects on one shared anchor: outer-template colors first, inner-template colors last.
    - Multiple objects of the same color are overlaid in reading order.
    - Return the tight crop of the composite, excluding the template.
    """
    g = np.asarray(grid, dtype=int)
    objects, layers, bg = _parse_objects_and_template(g)
    return _compose_output(objects, layers, bg)


def _transform_mask(mask: np.ndarray) -> np.ndarray:
    out = np.array(mask, dtype=bool)
    out = np.rot90(out, random.randrange(4))
    if random.random() < 0.5:
        out = np.fliplr(out)
    if random.random() < 0.5:
        out = np.flipud(out)
    return out.copy()


def _polyomino_mask() -> np.ndarray:
    h = random.randint(2, 6)
    w = random.randint(2, 6)
    total = h * w
    target = random.randint(max(3, total // 3), max(3, min(total - 1, int(np.ceil(total * 0.75)))))
    mask = np.zeros((h, w), dtype=bool)
    seed = (random.randrange(h), random.randrange(w))
    mask[seed] = True
    while int(mask.sum()) < target:
        frontier: set[tuple[int, int]] = set()
        filled = np.argwhere(mask)
        for r, c in filled:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr = int(r + dr)
                nc = int(c + dc)
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    frontier.add((nr, nc))
        mask[random.choice(tuple(frontier))] = True
    return mask


def _stair_mask() -> np.ndarray:
    h = random.randint(3, 6)
    w = random.randint(3, 6)
    mask = np.zeros((h, w), dtype=bool)
    direction = random.choice(["left", "right"])
    for r in range(h):
        span = max(1, 1 + (r * w) // h)
        if direction == "left":
            mask[r, :span] = True
        else:
            mask[r, w - span : w] = True
    return mask


def _plus_mask() -> np.ndarray:
    h = random.randint(3, 6)
    w = random.randint(3, 6)
    mask = np.zeros((h, w), dtype=bool)
    row = random.randrange(h)
    col = random.randrange(w)
    row_thick = random.randint(1, min(2, h))
    col_thick = random.randint(1, min(2, w))
    r0 = max(0, row - row_thick // 2)
    c0 = max(0, col - col_thick // 2)
    mask[r0 : min(h, r0 + row_thick), :] = True
    mask[:, c0 : min(w, c0 + col_thick)] = True
    return mask


def _t_mask() -> np.ndarray:
    h = random.randint(3, 6)
    w = random.randint(3, 6)
    mask = np.zeros((h, w), dtype=bool)
    bar_h = random.randint(1, min(2, h))
    stem_w = random.randint(1, min(2, w))
    mask[:bar_h, :] = True
    col = random.randrange(w - stem_w + 1)
    mask[:, col : col + stem_w] = True
    return mask


def _l_mask() -> np.ndarray:
    h = random.randint(3, 6)
    w = random.randint(3, 6)
    mask = np.zeros((h, w), dtype=bool)
    vh = random.randint(1, min(2, w))
    hh = random.randint(1, min(2, h))
    mask[:, :vh] = True
    mask[h - hh : h, :] = True
    return mask


def _rect_mask() -> np.ndarray:
    h = random.randint(2, 6)
    w = random.randint(2, 6)
    return np.ones((h, w), dtype=bool)


def _random_shape_mask() -> np.ndarray:
    builders = [_rect_mask, _l_mask, _t_mask, _plus_mask, _stair_mask, _polyomino_mask]
    for _ in range(50):
        mask = builders[random.randrange(len(builders))]()
        mask = _transform_mask(mask)
        mask = _crop_mask(mask)
        if int(mask.sum()) < 3:
            continue
        if _has_hole(mask):
            continue
        return mask
    raise RuntimeError("failed to build a connected shape")


def _make_template(size: int, colors: list[int]) -> np.ndarray:
    patch = np.full((size, size), int(colors[0]), dtype=int)
    for depth, color in enumerate(colors[1:], start=1):
        patch[depth : size - depth, depth : size - depth] = int(color)
    return patch


def _template_position(h: int, w: int, size: int, corner: str) -> tuple[int, int]:
    if corner == "tl":
        return 0, 0
    if corner == "tr":
        return 0, w - size
    if corner == "bl":
        return h - size, 0
    return h - size, w - size


def _separator_limits(h: int, w: int, size: int, corner: str, mode: str) -> tuple[int, int, int, int]:
    r_lo = 0
    r_hi = h - 1
    c_lo = 0
    c_hi = w - 1
    if mode in {"row", "both"}:
        if corner in {"tl", "tr"}:
            r_lo = size + 1
        else:
            r_hi = h - size - 2
    if mode in {"col", "both"}:
        if corner in {"tl", "bl"}:
            c_lo = size + 1
        else:
            c_hi = w - size - 2
    return r_lo, r_hi, c_lo, c_hi


def _overlay_visible_colors(out: np.ndarray, bg: int) -> set[int]:
    return set(map(int, np.unique(out))) - {int(bg)}


def _place_objects(
    h: int,
    w: int,
    bg: int,
    template: np.ndarray,
    corner: str,
    mode: str,
    layer_colors: list[int],
    counts: dict[int, int],
) -> tuple[np.ndarray, list[dict[str, object]]] | None:
    grid = np.full((h, w), int(bg), dtype=int)
    tr, tc = _template_position(h, w, template.shape[0], corner)
    size = template.shape[0]
    grid[tr : tr + size, tc : tc + size] = template

    blocked = np.zeros((h, w), dtype=bool)
    blocked[tr : tr + size, tc : tc + size] = True
    if mode in {"row", "both"}:
        if corner in {"tl", "tr"}:
            blocked[size, :] = True
        else:
            blocked[h - size - 1, :] = True
    if mode in {"col", "both"}:
        if corner in {"tl", "bl"}:
            blocked[:, size] = True
        else:
            blocked[:, w - size - 1] = True

    r_lo, r_hi, c_lo, c_hi = _separator_limits(h, w, size, corner, mode)
    if r_lo > r_hi or c_lo > c_hi:
        return None

    objects: list[dict[str, object]] = []
    for color in layer_colors:
        for _ in range(counts[color]):
            placed = False
            for _ in range(200):
                mask = _random_shape_mask()
                sh, sw = mask.shape
                rr_lo = r_lo
                rr_hi = r_hi - sh + 1
                cc_lo = c_lo
                cc_hi = c_hi - sw + 1
                if rr_lo > rr_hi or cc_lo > cc_hi:
                    break
                r0 = random.randint(rr_lo, rr_hi)
                c0 = random.randint(cc_lo, cc_hi)
                br0 = max(0, r0 - 1)
                bc0 = max(0, c0 - 1)
                br1 = min(h, r0 + sh + 1)
                bc1 = min(w, c0 + sw + 1)
                if np.any(blocked[br0:br1, bc0:bc1]):
                    continue
                coords = np.argwhere(mask) + np.array([[r0, c0]], dtype=int)
                grid[tuple(coords.T)] = int(color)
                blocked[br0:br1, bc0:bc1] = True
                bbox = (r0, c0, r0 + sh - 1, c0 + sw - 1)
                objects.append(
                    {
                        "color": int(color),
                        "coords": coords.copy(),
                        "bbox": bbox,
                        "anchor": ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2),
                    }
                )
                placed = True
                break
            if not placed:
                return None
    return grid, objects


def _generation_counts(colors: list[int]) -> dict[int, int]:
    counts = {int(color): 1 for color in colors}
    extra = random.randint(0, len(colors))
    if extra:
        for color in random.sample(colors, extra):
            counts[int(color)] += 1
    return counts


def _gen_attempt() -> np.ndarray | None:
    size = random.choice([5, 6, 7])
    layer_count = (size + 1) // 2
    palette = random.sample(range(10), layer_count + 1)
    bg = int(palette[0])
    layer_colors = [int(x) for x in palette[1:]]
    random.shuffle(layer_colors)
    template = _make_template(size, layer_colors)

    corner = random.choice(["tl", "tr", "bl", "br"])
    mode = random.choice(["row", "col", "both"])
    counts = _generation_counts(layer_colors)

    min_h = max(18, size + 8)
    min_w = max(18, size + 8)
    h = random.randint(min_h, 28)
    w = random.randint(min_w, 28)

    placed = _place_objects(h, w, bg, template, corner, mode, layer_colors, counts)
    if placed is None:
        return None
    grid, objects = placed

    out = _compose_output(objects, layer_colors, bg)
    rev = _compose_output(objects, list(reversed(layer_colors)), bg)
    if np.array_equal(out, rev):
        return None
    if _overlay_visible_colors(out, bg) != set(layer_colors):
        return None
    if out.shape[0] < 3 or out.shape[1] < 3:
        return None
    if np.count_nonzero(grid != bg) >= grid.size // 2:
        return None

    fg_comps = _components(grid != bg)
    multicolor = 0
    for comp in fg_comps:
        values = set(map(int, grid[tuple(comp.T)]))
        if len(values) > 1:
            multicolor += 1
    if multicolor != 1:
        return None

    try:
        solved = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(solved, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
