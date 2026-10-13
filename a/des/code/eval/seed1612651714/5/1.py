import random
from collections import deque

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
MAX_GEN_RETRIES = 400
MAX_SHAPE_RETRIES = 120
MAX_LAYOUT_RETRIES = 150


def _bg_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _tight_mask(coords: np.ndarray) -> tuple[tuple[int, int, int, int], np.ndarray]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel = coords - np.array([r0, c0])
    mask[rel[:, 0], rel[:, 1]] = True
    return (r0, c0, r1, c1), mask


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _components_by_color(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            coords = np.array(pts, dtype=int)
            bbox, mask = _tight_mask(coords)
            comps.append({"color": color, "coords": coords, "bbox": bbox, "mask": mask})
    return comps


def _symmetry_masks(mask: np.ndarray) -> list[np.ndarray]:
    seen: set[tuple[tuple[int, int], bytes]] = set()
    out: list[np.ndarray] = []
    base = mask.astype(bool)
    for flipped in (False, True):
        start = np.fliplr(base) if flipped else base
        for k in range(4):
            cur = np.rot90(start, k)
            key = (cur.shape, cur.tobytes())
            if key not in seen:
                seen.add(key)
                out.append(cur.copy())
    return out


def _canonical_key(mask: np.ndarray) -> tuple[int, int, bytes]:
    return min((m.shape[0], m.shape[1], m.tobytes()) for m in _symmetry_masks(mask))


def _is_rectangular_frame(mask: np.ndarray) -> bool:
    h, w = mask.shape
    if h < 3 or w < 3:
        return False
    expected = np.zeros((h, w), dtype=bool)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    return np.array_equal(mask, expected)


def _scaled_mask(mask: np.ndarray, scale: int) -> np.ndarray:
    return np.repeat(np.repeat(mask.astype(bool), scale, axis=0), scale, axis=1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each fence seed to an outside template by shape (allowing rotation/flip),
    erase the seed, and enlarge the matched template in its own orientation to fill
    the fence interior.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    bg = _bg_color(g)

    comps = _components_by_color(g, bg)
    fences: list[dict] = []
    fence_ids: set[int] = set()
    for idx, comp in enumerate(comps):
        if not _is_rectangular_frame(comp["mask"]):
            continue
        r0, c0, r1, c1 = comp["bbox"]
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        if inner.size == 0:
            continue
        seed_mask0 = inner != bg
        if not seed_mask0.any():
            continue
        if len(np.unique(inner[seed_mask0])) != 1:
            continue
        seed_parts = _components_mask(seed_mask0)
        if len(seed_parts) != 1:
            continue
        _, seed_mask = _tight_mask(seed_parts[0])
        fences.append(
            {
                "bbox": comp["bbox"],
                "interior_shape": inner.shape,
                "seed_key": _canonical_key(seed_mask),
            }
        )
        fence_ids.add(idx)

    assert fences, "expected at least one valid fence"

    def _inside_any_fence(coords: np.ndarray) -> bool:
        for fence in fences:
            r0, c0, r1, c1 = fence["bbox"]
            if np.all((coords[:, 0] > r0) & (coords[:, 0] < r1) & (coords[:, 1] > c0) & (coords[:, 1] < c1)):
                return True
        return False

    templates: dict[tuple[int, int, bytes], tuple[np.ndarray, int]] = {}
    for idx, comp in enumerate(comps):
        if idx in fence_ids or _inside_any_fence(comp["coords"]):
            continue
        key = _canonical_key(comp["mask"])
        assert key not in templates, "template shapes must be unique up to rotation/flip"
        templates[key] = (comp["mask"], int(comp["color"]))

    assert templates, "expected at least one outside template"

    out = g.copy()
    for fence in fences:
        r0, c0, r1, c1 = fence["bbox"]
        inner = out[r0 + 1 : r1, c0 + 1 : c1]
        inner[:] = bg
        template_mask, template_color = templates[fence["seed_key"]]
        ih, iw = map(int, inner.shape)
        th, tw = map(int, template_mask.shape)
        assert ih % th == 0 and iw % tw == 0, "fence interior must be an integer scaling of template bbox"
        scale_h = ih // th
        scale_w = iw // tw
        assert scale_h == scale_w and scale_h >= 1, "template scaling must be uniform"
        scaled = _scaled_mask(template_mask, scale_h)
        inner[scaled] = template_color
    return out


def _random_connected_mask(rng: random.Random) -> np.ndarray | None:
    for _ in range(MAX_SHAPE_RETRIES):
        h = rng.randint(2, 4)
        w = rng.randint(2, 4)
        target = rng.randint(3, min(h * w, 6))
        mask = np.zeros((h, w), dtype=bool)
        start = (rng.randrange(h), rng.randrange(w))
        mask[start] = True
        while int(mask.sum()) < target:
            frontier: list[tuple[int, int]] = []
            for r, c in np.argwhere(mask):
                for dr, dc in DIR4:
                    nr, nc = int(r) + dr, int(c) + dc
                    if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                        frontier.append((nr, nc))
            if not frontier:
                break
            mask[rng.choice(frontier)] = True
        coords = np.argwhere(mask)
        _, tight = _tight_mask(coords)
        th, tw = tight.shape
        if th < 2 or tw < 2:
            continue
        orbit = len(_symmetry_masks(tight))
        if orbit <= 2 and rng.random() < 0.7:
            continue
        return tight
    return None


def _pick_offset(rng: random.Random, room: int) -> int:
    if room <= 0:
        return 0
    if room >= 2:
        return rng.randint(1, room - 1)
    return rng.randint(0, room)


def _try_place_rect(
    occ: np.ndarray, rect_h: int, rect_w: int, rng: random.Random, pad: int = 1
) -> tuple[int, int] | None:
    h, w = occ.shape
    if rect_h + 2 > h or rect_w + 2 > w:
        return None
    r_lo, r_hi = 1, h - rect_h - 1
    c_lo, c_hi = 1, w - rect_w - 1
    if r_lo > r_hi or c_lo > c_hi:
        return None
    for _ in range(MAX_LAYOUT_RETRIES):
        r0 = rng.randint(r_lo, r_hi)
        c0 = rng.randint(c_lo, c_hi)
        rr0 = max(0, r0 - pad)
        cc0 = max(0, c0 - pad)
        rr1 = min(h, r0 + rect_h + pad)
        cc1 = min(w, c0 + rect_w + pad)
        if occ[rr0:rr1, cc0:cc1].any():
            continue
        occ[rr0:rr1, cc0:cc1] = True
        return r0, c0
    return None


def _draw_mask(grid: np.ndarray, mask: np.ndarray, color: int, top: int, left: int) -> None:
    h, w = mask.shape
    view = grid[top : top + h, left : left + w]
    view[mask] = color


def _draw_frame(grid: np.ndarray, top: int, left: int, inner_h: int, inner_w: int, color: int) -> None:
    bottom = top + inner_h + 1
    right = left + inner_w + 1
    grid[top, left : right + 1] = color
    grid[bottom, left : right + 1] = color
    grid[top : bottom + 1, left] = color
    grid[top : bottom + 1, right] = color


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    pair_count = rng.choice([2, 3])
    colors = list(range(10))
    rng.shuffle(colors)
    bg = colors[0]
    template_colors = colors[1 : 1 + pair_count]
    fence_colors = colors[1 + pair_count : 1 + 2 * pair_count]
    seed_color = colors[1 + 2 * pair_count]

    bases: list[np.ndarray] = []
    seen_keys: set[tuple[int, int, bytes]] = set()
    for _ in range(pair_count):
        mask = _random_connected_mask(rng)
        if mask is None:
            return None
        key = _canonical_key(mask)
        if key in seen_keys:
            return None
        seen_keys.add(key)
        bases.append(mask)

    specs: list[dict] = []
    for i, base in enumerate(bases):
        template_mask = rng.choice(_symmetry_masks(base))
        seed_mask = rng.choice(_symmetry_masks(base))
        th, tw = template_mask.shape
        sh, sw = seed_mask.shape
        min_scale = max(2, (sh + th - 1) // th, (sw + tw - 1) // tw)
        scale_choices = [s for s in (2, 3, 4) if s >= min_scale]
        if not scale_choices:
            return None
        scale = rng.choice(scale_choices)
        inner_h = th * scale
        inner_w = tw * scale
        seed_top = _pick_offset(rng, inner_h - sh)
        seed_left = _pick_offset(rng, inner_w - sw)
        specs.append(
            {
                "template_mask": template_mask,
                "template_color": template_colors[i],
                "fence_color": fence_colors[i],
                "seed_mask": seed_mask,
                "inner_h": inner_h,
                "inner_w": inner_w,
                "seed_top": seed_top,
                "seed_left": seed_left,
            }
        )

    max_h = max(max(spec["inner_h"] + 2, spec["template_mask"].shape[0]) for spec in specs)
    max_w = max(max(spec["inner_w"] + 2, spec["template_mask"].shape[1]) for spec in specs)
    footprint = sum((spec["inner_h"] + 4) * (spec["inner_w"] + 4) for spec in specs)
    footprint += sum((spec["template_mask"].shape[0] + 2) * (spec["template_mask"].shape[1] + 2) for spec in specs)
    base_side = max(max_h + 4, max_w + 4, 20, int(np.ceil(np.sqrt(footprint / 0.55))))
    h = base_side + rng.randint(0, 4)
    w = base_side + rng.randint(0, 5)
    if h > 30 or w > 30:
        return None

    grid = np.full((h, w), bg, dtype=int)
    occ = np.zeros((h, w), dtype=bool)

    fence_positions: list[tuple[int, int]] = []
    for idx in sorted(range(pair_count), key=lambda j: specs[j]["inner_h"] * specs[j]["inner_w"], reverse=True):
        spec = specs[idx]
        pos = _try_place_rect(occ, spec["inner_h"] + 2, spec["inner_w"] + 2, rng, pad=1)
        if pos is None:
            return None
        fence_positions.append((idx, pos[0], pos[1]))
    template_positions: list[tuple[int, int]] = []
    for idx in sorted(range(pair_count), key=lambda j: specs[j]["template_mask"].sum(), reverse=True):
        mask = specs[idx]["template_mask"]
        pos = _try_place_rect(occ, mask.shape[0], mask.shape[1], rng, pad=1)
        if pos is None:
            return None
        template_positions.append((idx, pos[0], pos[1]))

    for idx, top, left in fence_positions:
        spec = specs[idx]
        _draw_frame(grid, top, left, spec["inner_h"], spec["inner_w"], spec["fence_color"])
        _draw_mask(
            grid,
            spec["seed_mask"],
            seed_color,
            top + 1 + spec["seed_top"],
            left + 1 + spec["seed_left"],
        )

    for idx, top, left in template_positions:
        spec = specs[idx]
        _draw_mask(grid, spec["template_mask"], spec["template_color"], top, left)

    if _bg_color(grid) != bg:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
