from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


GRAY = 5
CARD_SIZE = 4
TEXTURE_SIZE = 4
SHAPE_LIBRARY = [
    np.array([[1, 1]], dtype=bool),
    np.array([[1], [1]], dtype=bool),
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1], [1], [1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 0], [1, 0], [1, 1]], dtype=bool),
]


def _bg_color(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(cnts))])


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return out


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _mask_from_cells(cells: list[tuple[int, int]], bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in cells:
        mask[r - r0, c - c0] = True
    return mask


def _hole_mask(border_mask: np.ndarray) -> np.ndarray:
    h, w = border_mask.shape
    outside = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()
    for r in range(h):
        for c in (0, w - 1):
            if not border_mask[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not border_mask[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c, h, w):
            if not border_mask[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))
    return (~border_mask) & (~outside)


def _normalize_mask(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    pts = np.argwhere(mask)
    assert pts.size > 0
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    cropped = mask[r0 : r1 + 1, c0 : c1 + 1].astype(int)
    return tuple(tuple(int(v) for v in row) for row in cropped)


def _find_regions(grid: np.ndarray, bg: int) -> tuple[int, list[dict]]:
    candidates: list[dict] = []
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        for comp in _components(grid == color):
            box = _bbox(comp)
            local_mask = _mask_from_cells(comp, box)
            inside = _hole_mask(local_mask)
            if not bool(inside.any()):
                continue
            candidates.append(
                {
                    "color": color,
                    "bbox": box,
                    "border_mask": local_mask,
                    "interior_local": inside,
                }
            )
    assert candidates, "no enclosed region found"
    score = Counter(item["color"] for item in candidates)
    border_color = max(
        score,
        key=lambda color: (
            score[color],
            sum(int(item["interior_local"].sum()) for item in candidates if item["color"] == color),
        ),
    )
    regions = [item for item in candidates if item["color"] == border_color]
    return int(border_color), regions


def _extract_template_mapping(area: np.ndarray, bg: int) -> dict[tuple[tuple[int, ...], ...], np.ndarray]:
    pts = np.argwhere(area != bg)
    if pts.size == 0:
        return {}
    _r0, c0 = pts.min(axis=0)
    _r1, c1 = pts.max(axis=0)
    sub = area[:, c0 : c1 + 1]
    h, w = sub.shape
    if h < CARD_SIZE + TEXTURE_SIZE:
        return {}
    mapping: dict[tuple[tuple[int, ...], ...], np.ndarray] = {}
    for r in range(CARD_SIZE, h - TEXTURE_SIZE + 1):
        tex = sub[r : r + TEXTURE_SIZE, :]
        if tex.shape != (TEXTURE_SIZE, w):
            continue
        if not bool(np.all(tex != bg)):
            continue
        card = sub[r - CARD_SIZE : r, :]
        card_mask = card != bg
        if not bool(card_mask.any()):
            continue
        mapping[_normalize_mask(card_mask)] = tex.copy()
    return mapping


def _find_templates(
    grid: np.ndarray,
    bg: int,
    regions: list[dict],
    needed_keys: set[tuple[tuple[int, ...], ...]],
) -> dict[tuple[tuple[int, ...], ...], np.ndarray]:
    min_c = min(int(item["bbox"][1]) for item in regions)
    max_c = max(int(item["bbox"][3]) for item in regions)
    areas: list[np.ndarray] = []
    if min_c > 0:
        areas.append(grid[:, :min_c])
    if max_c + 1 < grid.shape[1]:
        areas.append(grid[:, max_c + 1 :])
    best: dict[tuple[tuple[int, ...], ...], np.ndarray] = {}
    for area in areas:
        mapping = _extract_template_mapping(area, bg)
        if not mapping:
            continue
        if needed_keys.issubset(mapping.keys()):
            if len(mapping) > len(best):
                best = mapping
    assert best, "template mapping not found"
    return best


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each interior seed shape to a template card, then tile the paired texture
    inside the corresponding enclosed region.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    border_color, regions = _find_regions(g, bg)

    seeds: list[dict] = []
    needed_keys: set[tuple[tuple[int, ...], ...]] = set()
    for region in regions:
        r0, c0, r1, c1 = region["bbox"]
        inside = region["interior_local"]
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        seed_mask = inside & (patch != bg) & (patch != border_color)
        assert bool(seed_mask.any()), "missing seed"
        shape_key = _normalize_mask(seed_mask)
        needed_keys.add(shape_key)
        seeds.append(
            {
                "bbox": (r0, c0, r1, c1),
                "interior_local": inside,
                "shape_key": shape_key,
            }
        )

    template_map = _find_templates(g, bg, regions, needed_keys)
    out = g.copy()
    for item in seeds:
        r0, c0, r1, c1 = item["bbox"]
        inside = item["interior_local"]
        texture = template_map[item["shape_key"]]
        th, tw = texture.shape
        inside_pts = np.argwhere(inside)
        ir0, ic0 = inside_pts.min(axis=0)
        for rr in range(r1 - r0 + 1):
            for cc in range(c1 - c0 + 1):
                if inside[rr, cc]:
                    out[r0 + rr, c0 + cc] = int(texture[(rr - ir0) % th, (cc - ic0) % tw])
    return out


def _make_texture(rng: random.Random, colors: list[int]) -> np.ndarray:
    style = rng.choice(["checker", "vstripe", "hstripe", "diag", "quad"])
    if style == "quad":
        cols = rng.sample(colors, k=min(4, len(colors)))
        while len(cols) < 4:
            cols.append(rng.choice(colors))
        tex = np.zeros((TEXTURE_SIZE, TEXTURE_SIZE), dtype=int)
        tex[:2, :2] = cols[0]
        tex[:2, 2:] = cols[1]
        tex[2:, :2] = cols[2]
        tex[2:, 2:] = cols[3]
        return tex
    a, b = rng.sample(colors, 2)
    tex = np.zeros((TEXTURE_SIZE, TEXTURE_SIZE), dtype=int)
    for r in range(TEXTURE_SIZE):
        for c in range(TEXTURE_SIZE):
            if style == "checker":
                tex[r, c] = a if (r + c) % 2 == 0 else b
            elif style == "vstripe":
                tex[r, c] = a if c % 2 == 0 else b
            elif style == "hstripe":
                tex[r, c] = a if r % 2 == 0 else b
            else:
                tex[r, c] = a if (r - c) % 3 == 0 else b
    return tex


def _draw_shape_card(rng: random.Random, shape_mask: np.ndarray, bg: int) -> np.ndarray:
    card = np.full((CARD_SIZE, CARD_SIZE), bg, dtype=int)
    sh, sw = shape_mask.shape
    off_r = rng.randint(0, CARD_SIZE - sh)
    off_c = rng.randint(0, CARD_SIZE - sw)
    for r in range(sh):
        for c in range(sw):
            if shape_mask[r, c]:
                card[off_r + r, off_c + c] = GRAY
    return card


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    palette = list(range(10))
    bg = rng.choice([color for color in palette if color != GRAY])
    border_choices = [color for color in palette if color not in (bg, GRAY)]
    border_color = rng.choice(border_choices)
    active_colors = [color for color in palette if color not in (bg, border_color, GRAY)]

    template_count = rng.randint(3, 4)
    template_ids = rng.sample(range(len(SHAPE_LIBRARY)), template_count)
    region_count = rng.randint(2, 4)
    region_shape_ids = [rng.choice(template_ids) for _ in range(region_count)]
    if region_count >= 2 and len(set(region_shape_ids)) == 1:
        region_shape_ids[-1] = rng.choice([sid for sid in template_ids if sid != region_shape_ids[0]])

    texture_by_shape: dict[int, np.ndarray] = {}
    seen_textures: set[bytes] = set()
    for sid in template_ids:
        for _ in range(50):
            tex = _make_texture(rng, active_colors)
            key = tex.tobytes()
            if key not in seen_textures:
                seen_textures.add(key)
                texture_by_shape[sid] = tex
                break
        else:
            return None

    template_order = template_ids[:]
    rng.shuffle(template_order)
    template_height = template_count * (CARD_SIZE + TEXTURE_SIZE) + (template_count - 1)

    region_specs = []
    for sid in region_shape_ids:
        inner_h = rng.randint(5, 9)
        inner_w = rng.randint(5, 9)
        region_specs.append(
            {
                "shape_id": sid,
                "inner_h": inner_h,
                "inner_w": inner_w,
                "outer_h": inner_h + 2,
                "outer_w": inner_w + 2,
            }
        )

    h = max(template_height + rng.randint(0, 3), rng.randint(20, 28))
    main_w = rng.randint(16, 22)
    gap = rng.randint(2, 4)
    w = main_w + gap + CARD_SIZE
    grid = np.full((h, w), bg, dtype=int)

    top_offset = rng.randint(0, h - template_height)
    template_c = w - CARD_SIZE
    row = top_offset
    for idx, sid in enumerate(template_order):
        card = _draw_shape_card(rng, SHAPE_LIBRARY[sid], bg)
        grid[row : row + CARD_SIZE, template_c:] = card
        grid[row + CARD_SIZE : row + CARD_SIZE + TEXTURE_SIZE, template_c:] = texture_by_shape[sid]
        row += CARD_SIZE + TEXTURE_SIZE
        if idx + 1 < len(template_order):
            row += 1

    occupancy = np.zeros((h, main_w), dtype=bool)
    for spec in sorted(region_specs, key=lambda item: item["outer_h"] * item["outer_w"], reverse=True):
        oh = spec["outer_h"]
        ow = spec["outer_w"]
        placed = False
        for _ in range(300):
            r0 = rng.randint(1, h - oh - 1)
            c0 = rng.randint(1, main_w - ow - 1)
            rr0 = max(0, r0 - 2)
            rr1 = min(h, r0 + oh + 2)
            cc0 = max(0, c0 - 2)
            cc1 = min(main_w, c0 + ow + 2)
            if bool(occupancy[rr0:rr1, cc0:cc1].any()):
                continue
            grid[r0, c0 : c0 + ow] = border_color
            grid[r0 + oh - 1, c0 : c0 + ow] = border_color
            grid[r0 : r0 + oh, c0] = border_color
            grid[r0 : r0 + oh, c0 + ow - 1] = border_color

            shape = SHAPE_LIBRARY[spec["shape_id"]]
            sh, sw = shape.shape
            sr = rng.randint(0, spec["inner_h"] - sh)
            sc = rng.randint(0, spec["inner_w"] - sw)
            seed_color = rng.choice(active_colors)
            for rr in range(sh):
                for cc in range(sw):
                    if shape[rr, cc]:
                        grid[r0 + 1 + sr + rr, c0 + 1 + sc + cc] = seed_color

            occupancy[r0 : r0 + oh, c0 : c0 + ow] = True
            placed = True
            break
        if not placed:
            return None

    return grid


def generate_input() -> np.ndarray:
    """
    Randomly generate enclosed regions with seed shapes and a right-side template strip.
    """
    for _ in range(300):
        rng = random.Random()
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
