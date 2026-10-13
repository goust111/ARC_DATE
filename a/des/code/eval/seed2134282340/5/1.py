import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
NON_SPECIAL_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
SIDES = ("top", "bottom", "left", "right")
GEN_RETRIES = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q: deque[tuple[int, int]] = deque([(r, c)])
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _touches_boundary(coords: np.ndarray, h: int, w: int) -> bool:
    return bool(
        np.any(coords[:, 0] == 0)
        or np.any(coords[:, 0] == h - 1)
        or np.any(coords[:, 1] == 0)
        or np.any(coords[:, 1] == w - 1)
    )


def _tight_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_coords(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _is_rect_frame(mask: np.ndarray) -> bool:
    h, w = mask.shape
    if h < 3 or w < 3:
        return False
    expected = np.zeros((h, w), dtype=bool)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    return bool(np.array_equal(mask, expected))


def _shape_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[tuple[int, int, bytes]] = set()
    for k in range(4):
        rotated = np.rot90(mask, k)
        for variant in (rotated, np.fliplr(rotated)):
            key = (int(variant.shape[0]), int(variant.shape[1]), variant.tobytes())
            if key in seen:
                continue
            seen.add(key)
            variants.append(variant.copy())
    return variants


def _canonical_shape_key(mask: np.ndarray) -> tuple[int, int, bytes]:
    return min(
        (int(variant.shape[0]), int(variant.shape[1]), variant.tobytes())
        for variant in _shape_variants(mask)
    )


def _apply_transform(mask: np.ndarray, code: int) -> np.ndarray:
    out = np.rot90(mask, code % 4)
    if code >= 4:
        out = np.fliplr(out)
    return out.copy()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each frame's inner fragment to a grey edge legend shape under D4 symmetry,
    recolor the frame border with the paired swatch color, and erase the legend.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    frames: list[dict[str, object]] = []
    for color in [int(x) for x in np.unique(g) if int(x) != BG]:
        for coords in _components(g == color):
            if _touches_boundary(coords, h, w):
                continue
            mask = _tight_mask(coords)
            if _is_rect_frame(mask):
                r0, c0, r1, c1 = _bbox_coords(coords)
                frames.append(
                    {
                        "coords": coords,
                        "bbox": (r0, c0, r1, c1),
                    }
                )

    assert frames, "expected at least one rectangular frame"

    legend_map: dict[tuple[int, int, bytes], int] = {}
    legend_cells = np.zeros((h, w), dtype=bool)
    for coords in _components(g != BG):
        if not _touches_boundary(coords, h, w):
            continue
        colors = [int(g[r, c]) for r, c in coords]
        if GREY not in colors:
            continue
        non_grey = [(int(r), int(c), int(g[r, c])) for r, c in coords if int(g[r, c]) != GREY]
        grey_coords = np.array([(int(r), int(c)) for r, c in coords if int(g[r, c]) == GREY], dtype=int)
        if grey_coords.size == 0 or len(non_grey) != 1:
            continue
        swatch_r, swatch_c, swatch_color = non_grey[0]
        grey_mask = _tight_mask(grey_coords)
        assert any(abs(swatch_r - int(r)) + abs(swatch_c - int(c)) == 1 for r, c in grey_coords), "swatch must touch grey shape"
        key = _canonical_shape_key(grey_mask)
        if key in legend_map:
            assert legend_map[key] == swatch_color, "shape->color legend mapping must be unambiguous"
        legend_map[key] = swatch_color
        legend_cells[coords[:, 0], coords[:, 1]] = True

    assert legend_map, "expected at least one border legend entry"

    out = g.copy()
    out[legend_cells] = BG

    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        inner_mask = g[r0 + 1 : r1, c0 + 1 : c1] != BG
        inner_comps = _components(inner_mask)
        assert len(inner_comps) == 1, "each frame must contain exactly one inner fragment"
        fragment_mask = _tight_mask(inner_comps[0])
        target_color = legend_map[_canonical_shape_key(fragment_mask)]
        coords = frame["coords"]
        out[coords[:, 0], coords[:, 1]] = int(target_color)

    return out


def _random_connected_shape() -> np.ndarray | None:
    for _ in range(80):
        h = random.randint(2, 4)
        w = random.randint(2, 4)
        area = random.randint(3, min(6, h * w))
        start = (random.randrange(h), random.randrange(w))
        chosen = {start}
        frontier = set(_neighbors4(*start))
        while len(chosen) < area and frontier:
            candidates = [
                p for p in frontier if 0 <= p[0] < h and 0 <= p[1] < w and p not in chosen
            ]
            if not candidates:
                break
            nr, nc = random.choice(candidates)
            chosen.add((nr, nc))
            frontier.discard((nr, nc))
            frontier.update(_neighbors4(nr, nc))
        if len(chosen) != area:
            continue
        coords = np.array(sorted(chosen), dtype=int)
        coords[:, 0] -= int(coords[:, 0].min())
        coords[:, 1] -= int(coords[:, 1].min())
        mask = np.zeros((int(coords[:, 0].max()) + 1, int(coords[:, 1].max()) + 1), dtype=bool)
        mask[coords[:, 0], coords[:, 1]] = True
        if mask.shape[0] > 4 or mask.shape[1] > 4:
            continue
        return mask
    return None


def _sample_unique_shapes(count: int) -> list[np.ndarray] | None:
    shapes: list[np.ndarray] = []
    seen: set[tuple[int, int, bytes]] = set()
    for _ in range(200):
        mask = _random_connected_shape()
        if mask is None:
            continue
        key = _canonical_shape_key(mask)
        if key in seen:
            continue
        seen.add(key)
        shapes.append(mask)
        if len(shapes) == count:
            return shapes
    return None


def _make_entry_patch(shape_mask: np.ndarray, swatch_color: int, side: str) -> np.ndarray:
    sh, sw = shape_mask.shape
    if side in ("top", "bottom"):
        patch = np.full((sh, sw + 1), BG, dtype=int)
        swatch_on_left = random.random() < 0.5
        if swatch_on_left:
            patch[:, 1:] = np.where(shape_mask, GREY, BG)
            rows = [r for r in range(sh) if shape_mask[r, 0]]
            patch[random.choice(rows), 0] = swatch_color
        else:
            patch[:, :sw] = np.where(shape_mask, GREY, BG)
            rows = [r for r in range(sh) if shape_mask[r, sw - 1]]
            patch[random.choice(rows), sw] = swatch_color
        return patch

    patch = np.full((sh + 1, sw), BG, dtype=int)
    swatch_on_top = random.random() < 0.5
    if swatch_on_top:
        patch[1:, :] = np.where(shape_mask, GREY, BG)
        cols = [c for c in range(sw) if shape_mask[0, c]]
        patch[0, random.choice(cols)] = swatch_color
    else:
        patch[:sh, :] = np.where(shape_mask, GREY, BG)
        cols = [c for c in range(sw) if shape_mask[sh - 1, c]]
        patch[sh, random.choice(cols)] = swatch_color
    return patch


def _place_legend_entries(
    grid: np.ndarray, side: str, patches: list[np.ndarray], blocked: np.ndarray
) -> np.ndarray | None:
    h, w = grid.shape
    legend_cells = np.zeros((h, w), dtype=bool)
    min_gap = 1

    if side in ("top", "bottom"):
        total = sum(int(p.shape[1]) for p in patches) + min_gap * (len(patches) - 1)
        if total > w:
            return None
        slack = w - total
        extras = [0] * (len(patches) + 1)
        for _ in range(slack):
            extras[random.randrange(len(extras))] += 1
        c = extras[0]
        for i, patch in enumerate(patches):
            ph, pw = patch.shape
            r = 0 if side == "top" else h - ph
            grid[r : r + ph, c : c + pw] = np.maximum(grid[r : r + ph, c : c + pw], patch)
            legend_cells[r : r + ph, c : c + pw] |= patch != BG
            rr0 = max(0, r - 1)
            rr1 = min(h, r + ph + 1)
            cc0 = max(0, c - 1)
            cc1 = min(w, c + pw + 1)
            blocked[rr0:rr1, cc0:cc1] = True
            c += pw + min_gap + extras[i + 1]
        return legend_cells

    total = sum(int(p.shape[0]) for p in patches) + min_gap * (len(patches) - 1)
    if total > h:
        return None
    slack = h - total
    extras = [0] * (len(patches) + 1)
    for _ in range(slack):
        extras[random.randrange(len(extras))] += 1
    r = extras[0]
    for i, patch in enumerate(patches):
        ph, pw = patch.shape
        c = 0 if side == "left" else w - pw
        grid[r : r + ph, c : c + pw] = np.maximum(grid[r : r + ph, c : c + pw], patch)
        legend_cells[r : r + ph, c : c + pw] |= patch != BG
        rr0 = max(0, r - 1)
        rr1 = min(h, r + ph + 1)
        cc0 = max(0, c - 1)
        cc1 = min(w, c + pw + 1)
        blocked[rr0:rr1, cc0:cc1] = True
        r += ph + min_gap + extras[i + 1]
    return legend_cells


def _draw_frame(grid: np.ndarray, r0: int, c0: int, fh: int, fw: int, color: int) -> None:
    grid[r0, c0 : c0 + fw] = color
    grid[r0 + fh - 1, c0 : c0 + fw] = color
    grid[r0 : r0 + fh, c0] = color
    grid[r0 : r0 + fh, c0 + fw - 1] = color


def _place_frames(
    grid: np.ndarray, blocked: np.ndarray, specs: list[dict[str, object]]
) -> list[dict[str, object]] | None:
    h, w = grid.shape
    placed: list[dict[str, object]] = []
    order = sorted(specs, key=lambda spec: -int(spec["fh"]) * int(spec["fw"]))
    for spec in order:
        fh = int(spec["fh"])
        fw = int(spec["fw"])
        positions = [(r, c) for r in range(1, h - fh) for c in range(1, w - fw)]
        random.shuffle(positions)
        found = None
        for r0, c0 in positions:
            rr0 = r0 - 1
            rr1 = r0 + fh + 1
            cc0 = c0 - 1
            cc1 = c0 + fw + 1
            if np.any(blocked[rr0:rr1, cc0:cc1]):
                continue
            found = (r0, c0)
            blocked[rr0:rr1, cc0:cc1] = True
            break
        if found is None:
            return None

        r0, c0 = found
        border_color = int(spec["border_color"])
        fragment_color = int(spec["fragment_color"])
        fragment_mask = spec["fragment_mask"]
        sh, sw = fragment_mask.shape
        ir = random.randint(1, fh - 2 - sh - 1)
        ic = random.randint(1, fw - 2 - sw - 1)

        _draw_frame(grid, r0, c0, fh, fw, border_color)
        patch = np.where(fragment_mask, fragment_color, BG)
        fr = r0 + 1 + ir
        fc = c0 + 1 + ic
        grid[fr : fr + sh, fc : fc + sw][fragment_mask] = patch[fragment_mask]
        placed.append(
            {
                "bbox": (r0, c0, r0 + fh - 1, c0 + fw - 1),
                "border_color": border_color,
                "target_color": int(spec["target_color"]),
            }
        )
    return placed


def _build_case() -> tuple[np.ndarray, np.ndarray] | None:
    n_objects = random.randint(2, 3)
    base_shapes = _sample_unique_shapes(n_objects)
    if base_shapes is None:
        return None

    colors = NON_SPECIAL_COLORS[:]
    random.shuffle(colors)
    border_colors = colors[:n_objects]
    target_colors = colors[n_objects : 2 * n_objects]
    fragment_pool = [c for c in colors[2 * n_objects :] if c not in target_colors]
    if not fragment_pool:
        return None
    fragment_colors = [random.choice(fragment_pool) for _ in range(n_objects)]

    side = random.choice(SIDES)

    entry_patches: list[np.ndarray] = []
    frame_specs: list[dict[str, object]] = []
    legend_span = 0
    legend_depth = 0
    for i, base_shape in enumerate(base_shapes):
        fragment_mask = _apply_transform(base_shape, random.randrange(8))
        legend_shape = _apply_transform(base_shape, random.randrange(8))
        entry_patch = _make_entry_patch(legend_shape, int(target_colors[i]), side)
        entry_patches.append(entry_patch)
        legend_depth = max(
            legend_depth,
            int(entry_patch.shape[0] if side in ("top", "bottom") else entry_patch.shape[1]),
        )
        legend_span += int(entry_patch.shape[1] if side in ("top", "bottom") else entry_patch.shape[0])

        sh, sw = fragment_mask.shape
        fh = random.randint(sh + 4, min(sh + 7, 10))
        fw = random.randint(sw + 4, min(sw + 7, 10))
        frame_specs.append(
            {
                "border_color": int(border_colors[i]),
                "target_color": int(target_colors[i]),
                "fragment_color": int(fragment_colors[i]),
                "fragment_mask": fragment_mask,
                "fh": fh,
                "fw": fw,
            }
        )

    extra_gap_need = n_objects - 1
    if side in ("top", "bottom"):
        min_h = max(18, legend_depth + 10)
        min_w = max(18, legend_span + extra_gap_need + 4)
    else:
        min_h = max(18, legend_span + extra_gap_need + 4)
        min_w = max(18, legend_depth + 10)
    max_h = max(min_h, 28)
    max_w = max(min_w, 28)
    if min_h > max_h or min_w > max_w:
        return None

    h = random.randint(min_h, max_h)
    w = random.randint(min_w, max_w)
    grid = np.full((h, w), BG, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    patch_order = list(range(n_objects))
    random.shuffle(patch_order)
    shuffled_patches = [entry_patches[i] for i in patch_order]
    legend_cells = _place_legend_entries(grid, side, shuffled_patches, blocked)
    if legend_cells is None:
        return None

    placed_frames = _place_frames(grid, blocked, frame_specs)
    if placed_frames is None:
        return None

    expected = grid.copy()
    expected[legend_cells] = BG
    for frame in placed_frames:
        r0, c0, r1, c1 = frame["bbox"]
        target_color = int(frame["target_color"])
        expected[r0, c0 : c1 + 1] = target_color
        expected[r1, c0 : c1 + 1] = target_color
        expected[r0 : r1 + 1, c0] = target_color
        expected[r0 : r1 + 1, c1] = target_color

    try:
        if np.array_equal(main(grid), expected):
            return grid, expected
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        case = _build_case()
        if case is not None:
            return case[0]
    raise RuntimeError("failed to generate a valid puzzle instance")
