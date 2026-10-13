import random
from collections import Counter, deque
from itertools import permutations

import numpy as np


ARROW_PATTERNS = {
    "up": frozenset({(0, 0), (0, 1), (0, 2), (1, 1), (2, 1)}),
    "down": frozenset({(0, 1), (1, 1), (2, 0), (2, 1), (2, 2)}),
    "left": frozenset({(0, 0), (1, 0), (1, 1), (1, 2), (2, 0)}),
    "right": frozenset({(0, 2), (1, 0), (1, 1), (1, 2), (2, 2)}),
}
DIR_ORDER = ("up", "down", "left", "right")
SIDE_ORDER = ("top", "bottom", "left", "right")

GRID_MIN = 24
GRID_MAX = 30
FRAME_COUNT_MIN = 2
FRAME_COUNT_MAX = 4
FRAME_OUTER_MIN = 5
FRAME_OUTER_MAX = 9
EDGE_BAND = 5
GEN_RETRIES = 300


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> frozenset[tuple[int, int]]:
    r0, c0, _, _ = _bbox(coords)
    return frozenset((r - r0, c - c0) for r, c in coords)


def _arrow_orientation(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> str | None:
    norm = _normalize(coords)
    for name, pattern in ARROW_PATTERNS.items():
        if norm == pattern:
            return name
    return None


def _is_rect_frame(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return False
    want = set()
    for r in range(r0, r1 + 1):
        want.add((r, c0))
        want.add((r, c1))
    for c in range(c0, c1 + 1):
        want.add((r0, c))
        want.add((r1, c))
    return set(coords) == want


def _is_filled_rectangle(grid: np.ndarray, coords: list[tuple[int, int]] | tuple[tuple[int, int], ...], bg: int) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    if len(coords) != (r1 - r0 + 1) * (c1 - c0 + 1):
        return False
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    return np.all(sub != bg)


def _interval_gap(a0: int, a1: int, b0: int, b1: int) -> int:
    if a1 < b0:
        return b0 - a1 - 1
    if b1 < a0:
        return a0 - b1 - 1
    return 0


def _pair_cost(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int | None:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    row_gap = _interval_gap(ar0, ar1, br0, br1)
    col_gap = _interval_gap(ac0, ac1, bc0, bc1)
    row_overlap = row_gap == 0
    col_overlap = col_gap == 0
    if row_overlap and col_gap <= 2:
        return col_gap
    if col_overlap and row_gap <= 2:
        return row_gap
    return None


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _tile_into(out: np.ndarray, r0: int, c0: int, height: int, width: int, tile: np.ndarray) -> None:
    th, tw = tile.shape
    for dr in range(height):
        for dc in range(width):
            out[r0 + dr, c0 + dc] = int(tile[dr % th, dc % tw])


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    h, w = g.shape

    frames: list[dict[str, object]] = []
    for color in map(int, np.unique(g)):
        if color == bg:
            continue
        for comp in _components(g == color):
            if _is_rect_frame(comp):
                frames.append({"color": color, "bbox": _bbox(comp)})

    assert frames, "expected at least one closed rectangular frame"

    frame_mask = np.zeros((h, w), dtype=bool)
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        frame_mask[r0 : r1 + 1, c0 : c1 + 1] = True

    outside_fg = (g != bg) & ~frame_mask
    arrows: list[dict[str, object]] = []
    templates: list[dict[str, object]] = []
    for comp in _components(outside_fg):
        orient = _arrow_orientation(comp)
        if orient is not None:
            arrows.append({"orientation": orient, "bbox": _bbox(comp)})
            continue
        if _is_filled_rectangle(g, comp, bg):
            r0, c0, r1, c1 = _bbox(comp)
            patch = g[r0 : r1 + 1, c0 : c1 + 1]
            if len(np.unique(patch)) >= 2:
                templates.append({"bbox": (r0, c0, r1, c1), "patch": patch.copy()})

    assert len(arrows) == 4, f"expected 4 legend arrows, found {len(arrows)}"
    assert len(templates) == 4, f"expected 4 legend templates, found {len(templates)}"
    assert {entry["orientation"] for entry in arrows} == set(DIR_ORDER), "legend must cover all four directions"

    best_perm = None
    best_cost = None
    for perm in permutations(range(len(templates))):
        total = 0
        ok = True
        for arrow_idx, template_idx in enumerate(perm):
            cost = _pair_cost(arrows[arrow_idx]["bbox"], templates[template_idx]["bbox"])
            if cost is None:
                ok = False
                break
            total += cost
        if ok and (best_cost is None or total < best_cost):
            best_cost = total
            best_perm = perm
    assert best_perm is not None, "could not pair legend arrows with templates"

    legend_map: dict[str, np.ndarray] = {}
    for arrow_idx, template_idx in enumerate(best_perm):
        legend_map[arrows[arrow_idx]["orientation"]] = templates[template_idx]["patch"]

    out = g.copy()
    for frame in frames:
        frame_color = int(frame["color"])
        r0, c0, r1, c1 = frame["bbox"]
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        seed_mask = (inner != bg) & (inner != frame_color)
        seed_comps = _components(seed_mask)
        seed_dirs = [d for d in (_arrow_orientation(comp) for comp in seed_comps) if d is not None]
        assert len(seed_dirs) == 1, "each frame must contain exactly one arrow seed"
        tile = legend_map[seed_dirs[0]]
        _tile_into(out, r0 + 1, c0 + 1, r1 - r0 - 1, c1 - c0 - 1, tile)

    return out


def _rotated_arrow_patch(direction: str, color: int) -> np.ndarray:
    patch = np.full((3, 3), -1, dtype=int)
    for r, c in ARROW_PATTERNS[direction]:
        patch[r, c] = color
    return patch


def _paste_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    mask = patch >= 0
    view = grid[r0 : r0 + h, c0 : c0 + w]
    view[mask] = patch[mask]


def _template_signature(template: np.ndarray) -> tuple[tuple[int, int], tuple[int, ...]]:
    return template.shape, tuple(int(v) for v in template.flatten())


def _make_template(rng: random.Random, palette: list[int]) -> np.ndarray:
    kinds = ("checker", "hstripe", "vstripe", "blocks", "random")
    for _ in range(100):
        h = rng.randint(2, 4)
        w = rng.randint(2, 4)
        colors = rng.sample(palette, rng.randint(2, min(4, len(palette))))
        kind = rng.choice(kinds)
        arr = np.empty((h, w), dtype=int)
        if kind == "checker":
            for r in range(h):
                for c in range(w):
                    arr[r, c] = colors[(r + c) % len(colors)]
        elif kind == "hstripe":
            for r in range(h):
                arr[r, :] = colors[r % len(colors)]
        elif kind == "vstripe":
            for c in range(w):
                arr[:, c] = colors[c % len(colors)]
        elif kind == "blocks":
            for r in range(h):
                for c in range(w):
                    idx = (r >= h // 2) + 2 * (c >= w // 2)
                    arr[r, c] = colors[idx % len(colors)]
        else:
            for r in range(h):
                for c in range(w):
                    if r > 0 and c > 0 and rng.random() < 0.5:
                        arr[r, c] = arr[r - 1, c] if rng.random() < 0.5 else arr[r, c - 1]
                    elif r > 0 and rng.random() < 0.35:
                        arr[r, c] = arr[r - 1, c]
                    elif c > 0 and rng.random() < 0.35:
                        arr[r, c] = arr[r, c - 1]
                    else:
                        arr[r, c] = rng.choice(colors)
        if len(np.unique(arr)) >= 2:
            return arr
    raise RuntimeError("failed to create a distinct multicolor template")


def _place_top_or_bottom_legend(
    grid: np.ndarray,
    rng: random.Random,
    side: str,
    direction: str,
    template: np.ndarray,
    arrow_color: int,
) -> None:
    h, w = grid.shape
    th, tw = template.shape
    total_w = tw + 1 + 3
    start_c = rng.randint(EDGE_BAND, w - EDGE_BAND - total_w)
    arrow_left = rng.random() < 0.5
    template_r = 0 if side == "top" else h - th
    arrow_r = 0 if side == "top" else h - 3
    if arrow_left:
        arrow_c = start_c
        template_c = start_c + 4
    else:
        template_c = start_c
        arrow_c = start_c + tw + 1
    grid[template_r : template_r + th, template_c : template_c + tw] = template
    _paste_patch(grid, _rotated_arrow_patch(direction, arrow_color), arrow_r, arrow_c)


def _place_left_or_right_legend(
    grid: np.ndarray,
    rng: random.Random,
    side: str,
    direction: str,
    template: np.ndarray,
    arrow_color: int,
) -> None:
    h, w = grid.shape
    th, tw = template.shape
    total_h = th + 1 + 3
    start_r = rng.randint(EDGE_BAND, h - EDGE_BAND - total_h)
    arrow_top = rng.random() < 0.5
    template_c = 0 if side == "left" else w - tw
    arrow_c = 0 if side == "left" else w - 3
    if arrow_top:
        arrow_r = start_r
        template_r = start_r + 4
    else:
        template_r = start_r
        arrow_r = start_r + th + 1
    grid[template_r : template_r + th, template_c : template_c + tw] = template
    _paste_patch(grid, _rotated_arrow_patch(direction, arrow_color), arrow_r, arrow_c)


def _sample_frame_directions(rng: random.Random, count: int) -> list[str]:
    base = list(DIR_ORDER)
    rng.shuffle(base)
    if count <= 4:
        return base[:count]
    out = base[:]
    while len(out) < count:
        out.append(rng.choice(DIR_ORDER))
    return out


def _place_frames(
    grid: np.ndarray,
    rng: random.Random,
    frame_color: int,
    arrow_color: int,
    frame_count: int,
) -> list[dict[str, object]] | None:
    h, w = grid.shape
    blocked = np.zeros((h, w), dtype=bool)
    blocked[:EDGE_BAND, :] = True
    blocked[h - EDGE_BAND :, :] = True
    blocked[:, :EDGE_BAND] = True
    blocked[:, w - EDGE_BAND :] = True

    placed: list[dict[str, object]] = []
    directions = _sample_frame_directions(rng, frame_count)
    for direction in directions:
        ok = False
        for _ in range(200):
            fh = rng.randint(FRAME_OUTER_MIN, FRAME_OUTER_MAX)
            fw = rng.randint(FRAME_OUTER_MIN, FRAME_OUTER_MAX)
            if fh < 5 or fw < 5:
                continue
            if h - EDGE_BAND - fh - 1 < EDGE_BAND or w - EDGE_BAND - fw - 1 < EDGE_BAND:
                return None
            r0 = rng.randint(EDGE_BAND, h - EDGE_BAND - fh)
            c0 = rng.randint(EDGE_BAND, w - EDGE_BAND - fw)
            r1 = r0 + fh - 1
            c1 = c0 + fw - 1
            mr0 = max(0, r0 - 1)
            mc0 = max(0, c0 - 1)
            mr1 = min(h, r1 + 2)
            mc1 = min(w, c1 + 2)
            if np.any(blocked[mr0:mr1, mc0:mc1]):
                continue

            grid[r0, c0 : c1 + 1] = frame_color
            grid[r1, c0 : c1 + 1] = frame_color
            grid[r0 : r1 + 1, c0] = frame_color
            grid[r0 : r1 + 1, c1] = frame_color

            seed_r = rng.randint(r0 + 1, r1 - 3)
            seed_c = rng.randint(c0 + 1, c1 - 3)
            _paste_patch(grid, _rotated_arrow_patch(direction, arrow_color), seed_r, seed_c)

            blocked[r0 : r1 + 1, c0 : c1 + 1] = True
            placed.append(
                {
                    "bbox": (r0, c0, r1, c1),
                    "direction": direction,
                }
            )
            ok = True
            break
        if not ok:
            return None
    return placed


def _expected_from_meta(grid: np.ndarray, meta: dict[str, object]) -> np.ndarray:
    out = np.array(grid, copy=True)
    for frame in meta["frames"]:
        r0, c0, r1, c1 = frame["bbox"]
        tile = meta["templates"][frame["direction"]]
        _tile_into(out, r0 + 1, c0 + 1, r1 - r0 - 1, c1 - c0 - 1, tile)
    return out


def _build_case(rng: random.Random, with_meta: bool = False):
    h = rng.randint(GRID_MIN, GRID_MAX)
    w = rng.randint(GRID_MIN, GRID_MAX)
    colors = list(range(10))
    rng.shuffle(colors)
    bg, frame_color, arrow_color = colors[:3]
    palette = colors[3:]
    if len(palette) < 4:
        return None

    grid = np.full((h, w), bg, dtype=int)
    directions = list(DIR_ORDER)
    rng.shuffle(directions)

    templates: dict[str, np.ndarray] = {}
    used_templates: set[tuple[tuple[int, int], tuple[int, ...]]] = set()
    for direction in directions:
        for _ in range(100):
            template = _make_template(rng, palette)
            sig = _template_signature(template)
            if sig not in used_templates:
                used_templates.add(sig)
                templates[direction] = template
                break
        else:
            return None

    for side, direction in zip(SIDE_ORDER, directions):
        template = templates[direction]
        if side in {"top", "bottom"}:
            _place_top_or_bottom_legend(grid, rng, side, direction, template, arrow_color)
        else:
            _place_left_or_right_legend(grid, rng, side, direction, template, arrow_color)

    frame_count = rng.randint(FRAME_COUNT_MIN, FRAME_COUNT_MAX)
    frames = _place_frames(grid, rng, frame_color, arrow_color, frame_count)
    if frames is None:
        return None

    meta = {
        "bg": bg,
        "frame_color": frame_color,
        "arrow_color": arrow_color,
        "templates": templates,
        "frames": frames,
    }
    return (grid, meta) if with_meta else grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        built = _build_case(rng, with_meta=False)
        if built is not None:
            return built
    raise RuntimeError("generate_input: exceeded retries")
