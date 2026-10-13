from __future__ import annotations

import random

import numpy as np


BG = 0
FRAME_COLOR = 5
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
QUADRANTS = ("tl", "tr", "br", "bl")
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
TEMPLATE_LIBRARY = (
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 0], [1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0]], dtype=bool),
    np.array([[1, 1], [1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 1]], dtype=bool),
    np.array([[1, 1, 0], [1, 0, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0], [1, 1, 0]], dtype=bool),
)
GEN_RETRIES = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_mask(coords: list[tuple[int, int]]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        out[r - r0, c - c0] = True
    return out


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return mask.shape, mask.astype(np.uint8).tobytes()


def _rotation_count(mask: np.ndarray) -> int:
    seen = {_mask_key(np.rot90(mask, k)) for k in range(4)}
    return len(seen)


def _is_connected(mask: np.ndarray) -> bool:
    coords = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    return len(coords) > 0 and len(_components(mask)) == 1


def _validate_templates() -> None:
    for mask in TEMPLATE_LIBRARY:
        assert mask.dtype == bool
        assert mask[0, 0]
        assert 3 <= int(mask.sum()) <= 6
        assert _is_connected(mask)
        assert _rotation_count(mask) == 4


_validate_templates()


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _is_rect_frame(coords: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    want = np.zeros((h, w), dtype=bool)
    want[0, :] = True
    want[-1, :] = True
    want[:, 0] = True
    want[:, -1] = True
    return np.array_equal(_crop_mask(coords), want)


def _rotation_steps(quadrant: str) -> int:
    if quadrant == "tl":
        return 0
    if quadrant == "tr":
        return 3
    if quadrant == "br":
        return 2
    return 1


def _quadrant_from_seed(seed_r: int, seed_c: int, inner_h: int, inner_w: int) -> str:
    top = seed_r < inner_h // 2
    left = seed_c < inner_w // 2
    if top and left:
        return "tl"
    if top and not left:
        return "tr"
    if not top and not left:
        return "br"
    return "bl"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp the matching outer template into each framed region, rotating it by the seed's quadrant.

    Input structure:
    - The background is black.
    - Several gray 1-pixel-thick rectangular frames enclose black interiors.
    - Each frame interior contains exactly one non-black, non-gray single-pixel seed.
    - Outside the frames, the grid edge area contains one monochrome 4-connected template per used seed color.

    Rule:
    - Detect every hollow rectangular frame and locate its unique seed pixel.
    - Match the seed color to the outer template component of the same color.
    - Determine which interior quadrant contains the seed: top-left, top-right, bottom-right, or bottom-left.
    - Rotate the template by 0, 90 clockwise, 180, or 270 clockwise for those quadrants respectively, then place its bbox with top-left aligned to the seed position.

    Key insight:
    - The seed color chooses which template to copy, while the seed position inside the frame chooses the template orientation.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    out = g.copy()

    frames: list[dict[str, int]] = []
    for color in np.unique(g):
        color = int(color)
        if color == bg:
            continue
        for comp in _components(g == color):
            if _is_rect_frame(comp):
                r0, c0, r1, c1 = _bbox(comp)
                frames.append({"color": color, "r0": r0, "c0": c0, "r1": r1, "c1": c1})

    templates: dict[int, np.ndarray] = {}
    for color in np.unique(g):
        color = int(color)
        if color == bg:
            continue
        best_mask = None
        best_area = -1
        for comp in _components(g == color):
            if len(comp) <= 1 or _is_rect_frame(comp):
                continue
            inside_frame = any(
                frame["r0"] < r < frame["r1"] and frame["c0"] < c < frame["c1"]
                for r, c in comp
                for frame in frames
            )
            if inside_frame:
                continue
            area = len(comp)
            if area > best_area:
                best_area = area
                best_mask = _crop_mask(comp)
        if best_mask is not None:
            templates[color] = best_mask

    for frame in frames:
        r0, c0, r1, c1 = frame["r0"], frame["c0"], frame["r1"], frame["c1"]
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        seed_locs = np.argwhere((inner != bg) & (inner != frame["color"]))
        assert len(seed_locs) == 1
        seed_r, seed_c = map(int, seed_locs[0])
        seed_color = int(inner[seed_r, seed_c])
        quadrant = _quadrant_from_seed(seed_r, seed_c, inner.shape[0], inner.shape[1])
        template = np.rot90(templates[seed_color], _rotation_steps(quadrant))
        for dr, dc in np.argwhere(template):
            rr = r0 + 1 + seed_r + int(dr)
            cc = c0 + 1 + seed_c + int(dc)
            if r0 < rr < r1 and c0 < cc < c1:
                out[rr, cc] = seed_color
    return out


def _choose_counts() -> tuple[int, int]:
    n_templates = random.choice((2, 3, 4))
    n_frames = random.randint(n_templates, 4)
    return n_templates, n_frames


def _anchor_choices(length: int, patch_len: int, first_half: bool) -> list[int]:
    fit_max = length - patch_len
    half = length // 2
    if first_half:
        hi = min(half - 1, fit_max)
        return list(range(0, hi + 1))
    return list(range(half, fit_max + 1))


def _make_frame_spec(color: int, template: np.ndarray, quadrant: str, max_inner: int) -> dict[str, object] | None:
    rotated = np.rot90(template, _rotation_steps(quadrant))
    th, tw = rotated.shape
    inner_h_choices = [x for x in range(max(6, 2 * th), max_inner + 1, 2)]
    inner_w_choices = [x for x in range(max(6, 2 * tw), max_inner + 1, 2)]
    if not inner_h_choices or not inner_w_choices:
        return None
    inner_h = random.choice(inner_h_choices)
    inner_w = random.choice(inner_w_choices)
    top = quadrant in {"tl", "tr"}
    left = quadrant in {"tl", "bl"}
    seed_rows = _anchor_choices(inner_h, th, top)
    seed_cols = _anchor_choices(inner_w, tw, left)
    if not seed_rows or not seed_cols:
        return None
    return {
        "color": color,
        "quadrant": quadrant,
        "inner_h": inner_h,
        "inner_w": inner_w,
        "outer_h": inner_h + 2,
        "outer_w": inner_w + 2,
        "seed_r": random.choice(seed_rows),
        "seed_c": random.choice(seed_cols),
    }


def _place_rectangles(specs: list[dict[str, object]], h: int, w: int) -> bool:
    occupied = np.zeros((h, w), dtype=bool)
    for spec in sorted(specs, key=lambda item: int(item["outer_h"]) * int(item["outer_w"]), reverse=True):
        oh = int(spec["outer_h"])
        ow = int(spec["outer_w"])
        candidates: list[tuple[int, int]] = []
        for r0 in range(2, h - oh - 1):
            for c0 in range(2, w - ow - 1):
                pad = occupied[r0 - 1 : r0 + oh + 1, c0 - 1 : c0 + ow + 1]
                if not pad.any():
                    candidates.append((r0, c0))
        if not candidates:
            return False
        r0, c0 = random.choice(candidates)
        spec["r0"] = r0
        spec["c0"] = c0
        occupied[r0 - 1 : r0 + oh + 1, c0 - 1 : c0 + ow + 1] = True
    return True


def _edge_candidates(mask: np.ndarray, h: int, w: int) -> list[tuple[int, int]]:
    mh, mw = mask.shape
    candidates: list[tuple[int, int]] = []
    for c0 in range(0, w - mw + 1):
        candidates.append((0, c0))
        candidates.append((h - mh, c0))
    for r0 in range(1, h - mh):
        candidates.append((r0, 0))
        candidates.append((r0, w - mw))
    return candidates


def _place_templates(
    templates: dict[int, np.ndarray], h: int, w: int, specs: list[dict[str, object]]
) -> dict[int, tuple[int, int]] | None:
    occupied = np.zeros((h, w), dtype=bool)
    for spec in specs:
        r0 = int(spec["r0"])
        c0 = int(spec["c0"])
        oh = int(spec["outer_h"])
        ow = int(spec["outer_w"])
        occupied[r0 - 1 : r0 + oh + 1, c0 - 1 : c0 + ow + 1] = True

    placements: dict[int, tuple[int, int]] = {}
    items = list(templates.items())
    random.shuffle(items)
    for color, mask in items:
        mh, mw = mask.shape
        candidates = _edge_candidates(mask, h, w)
        random.shuffle(candidates)
        placed = None
        for r0, c0 in candidates:
            rr0 = max(0, r0 - 1)
            rr1 = min(h, r0 + mh + 1)
            cc0 = max(0, c0 - 1)
            cc1 = min(w, c0 + mw + 1)
            if occupied[rr0:rr1, cc0:cc1].any():
                continue
            placed = (r0, c0)
            occupied[rr0:rr1, cc0:cc1] = True
            break
        if placed is None:
            return None
        placements[color] = placed
    return placements


def _render_input(
    h: int,
    w: int,
    templates: dict[int, np.ndarray],
    specs: list[dict[str, object]],
    template_positions: dict[int, tuple[int, int]],
) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)

    for spec in specs:
        r0 = int(spec["r0"])
        c0 = int(spec["c0"])
        oh = int(spec["outer_h"])
        ow = int(spec["outer_w"])
        color = int(spec["color"])
        grid[r0, c0 : c0 + ow] = FRAME_COLOR
        grid[r0 + oh - 1, c0 : c0 + ow] = FRAME_COLOR
        grid[r0 : r0 + oh, c0] = FRAME_COLOR
        grid[r0 : r0 + oh, c0 + ow - 1] = FRAME_COLOR
        grid[r0 + 1 + int(spec["seed_r"]), c0 + 1 + int(spec["seed_c"])] = color

    for color, mask in templates.items():
        r0, c0 = template_positions[color]
        for dr, dc in np.argwhere(mask):
            grid[r0 + int(dr), c0 + int(dc)] = color

    return grid


def _render_expected(
    grid: np.ndarray,
    templates: dict[int, np.ndarray],
    specs: list[dict[str, object]],
) -> np.ndarray:
    out = grid.copy()
    for spec in specs:
        color = int(spec["color"])
        quadrant = str(spec["quadrant"])
        r0 = int(spec["r0"])
        c0 = int(spec["c0"])
        seed_r = int(spec["seed_r"])
        seed_c = int(spec["seed_c"])
        mask = np.rot90(templates[color], _rotation_steps(quadrant))
        for dr, dc in np.argwhere(mask):
            out[r0 + 1 + seed_r + int(dr), c0 + 1 + seed_c + int(dc)] = color
    return out


def _build_assignments(colors: list[int], n_frames: int) -> list[int]:
    assignments = colors[:]
    while len(assignments) < n_frames:
        assignments.append(random.choice(colors))
    random.shuffle(assignments)
    return assignments


def _gen_attempt() -> np.ndarray | None:
    n_templates, n_frames = _choose_counts()
    colors = random.sample(COLOR_POOL, n_templates)
    shapes = random.sample(TEMPLATE_LIBRARY, n_templates)
    templates = {color: shape.copy() for color, shape in zip(colors, shapes)}

    assignments = _build_assignments(colors, n_frames)
    quadrants = random.sample(QUADRANTS, n_frames)
    max_inner = 10 if n_frames <= 3 else 8
    specs: list[dict[str, object]] = []
    for color, quadrant in zip(assignments, quadrants):
        spec = _make_frame_spec(color, templates[color], quadrant, max_inner)
        if spec is None:
            return None
        specs.append(spec)

    side_lo = {2: 20, 3: 23, 4: 26}[n_frames]
    side_hi = {2: 26, 3: 28, 4: 30}[n_frames]
    h = random.randint(side_lo, side_hi)
    w = random.randint(side_lo, side_hi)

    if not _place_rectangles(specs, h, w):
        return None
    template_positions = _place_templates(templates, h, w, specs)
    if template_positions is None:
        return None

    grid = _render_input(h, w, templates, specs, template_positions)
    expected = _render_expected(grid, templates, specs)

    try:
        actual = main(grid)
    except Exception:
        return None
    if np.array_equal(actual, expected) and not np.array_equal(actual, grid):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exceeded retry budget")
