from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, bbox_w, colors_present, find_components, get_bg, paste


DIVIDER_COLOR = Color.GREY
QUERY_COLOR = Color.ORANGE
BACKGROUND = Color.BLACK
TEMPLATE_COLORS = (
    Color.BLUE,
    Color.RED,
    Color.GREEN,
    Color.YELLOW,
    Color.PINK,
    Color.TEAL,
    Color.MAROON,
)

DIHEDRAL_TRANSFORMS = (
    (1, 0, 0, 1),
    (1, 0, 0, -1),
    (-1, 0, 0, 1),
    (-1, 0, 0, -1),
    (0, 1, 1, 0),
    (0, 1, -1, 0),
    (0, -1, 1, 0),
    (0, -1, -1, 0),
)

SHAPE_CATALOG = {
    "I3": ((0, 0), (0, 1), (0, 2)),
    "L3": ((0, 0), (1, 0), (1, 1)),
    "O4": ((0, 0), (0, 1), (1, 0), (1, 1)),
    "T4": ((0, 0), (0, 1), (0, 2), (1, 1)),
    "Z4": ((0, 0), (0, 1), (1, 1), (1, 2)),
    "L4": ((0, 0), (1, 0), (2, 0), (2, 1)),
}
MANDATORY_TEMPLATES = ("I3", "T4", "Z4")
OPTIONAL_TEMPLATES = ("L3", "O4", "L4")
GRID_HEIGHT_RANGE = (15, 22)
GRID_WIDTH_RANGE = (16, 24)
DIVIDER_RC_RANGE = (7, 8)
MAX_GENERATION_ATTEMPTS = 400


def _normalize(coords) -> tuple[tuple[int, int], ...]:
    pts = [(int(r), int(c)) for r, c in coords]
    min_r = min(r for r, _ in pts)
    min_c = min(c for _, c in pts)
    return tuple(sorted((r - min_r, c - min_c) for r, c in pts))


def _variants(coords) -> list[tuple[tuple[int, int], ...]]:
    pts = [(int(r), int(c)) for r, c in coords]
    out = []
    for rr_r, rr_c, cc_r, cc_c in DIHEDRAL_TRANSFORMS:
        transformed = []
        for r, c in pts:
            transformed.append((rr_r * r + rr_c * c, cc_r * r + cc_c * c))
        out.append(_normalize(transformed))
    return sorted(set(out))


def _canonical_shape(coords) -> tuple[tuple[int, int], ...]:
    return min(_variants(coords))


def _bbox_from_coords(coords) -> tuple[int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return max(rows) + 1, max(cols) + 1


def _component_local_coords(component: dict) -> tuple[tuple[int, int], ...]:
    r0, c0, _r1, _c1 = component["bbox"]
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in component["coords"]))


def _render_patch(coords, color: int) -> np.ndarray:
    h, w = _bbox_from_coords(coords)
    patch = np.full((h, w), BACKGROUND, dtype=int)
    rr, cc = np.array(coords, dtype=int).T
    patch[rr, cc] = int(color)
    return patch


def _paint_cells(grid: np.ndarray, coords, color: int) -> None:
    for r, c in coords:
        grid[int(r), int(c)] = int(color)


def _absolute_coords(local_coords, r0: int, c0: int) -> list[tuple[int, int]]:
    return [(int(r0 + r), int(c0 + c)) for r, c in local_coords]


def _mark_blocked(blocked: np.ndarray, coords) -> None:
    h, w = blocked.shape
    for r, c in coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < h and 0 <= cc < w:
                    blocked[rr, cc] = True


def _find_divider(grid: np.ndarray, bg: int) -> tuple[int, int, int]:
    h, w = grid.shape
    candidates = []
    for color in colors_present(grid, exclude={bg}):
        mask = grid == int(color)
        for r in range(h):
            c = -1
            while c + 1 < w and mask[r, c + 1]:
                c += 1
            if r >= 3 and c >= 3 and mask[: r + 1, c].all():
                candidates.append((int(color), int(r), int(c)))
    assert len(candidates) == 1, "expected one L-shaped divider"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """Match each query sprite to the legend color, then shift it right by its own width before repainting.

    Input structure:
    - A black background contains a one-pixel grey L-shaped divider touching the top and left edges.
    - The upper-left legend region inside that divider holds 3-4 monochrome template sprites with distinct colors.
    - Outside the legend, several orange 4-connected query sprites appear with shapes matching legend templates up to rotation/reflection.

    Rule:
    - Detect the L-divider and read the upper-left region as a shape-to-color legend.
    - For each orange query sprite, match its cropped 4-connected shape to a legend template under the 8 dihedral variants.
    - Clear the query from its original cells, then repaint the same shape shifted right by its own bounding-box width using the matched legend color.
    - Leave the divider and all legend pixels unchanged.

    Key insight:
    - Each query contributes two independent facts: its shape chooses the output color from the legend, while its own current bbox width determines the horizontal shift distance.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    divider_color, divider_r, divider_c = _find_divider(g, bg)

    legend = g[:divider_r, :divider_c]
    template_by_shape: dict[tuple[tuple[int, int], ...], int] = {}
    template_colors: set[int] = set()
    for comp in find_components(legend, bg=bg, connectivity=4, monochromatic=True):
        color = int(comp["color"])
        assert color != divider_color
        local = _component_local_coords(comp)
        key = _canonical_shape(local)
        assert key not in template_by_shape
        template_by_shape[key] = color
        template_colors.add(color)
    assert 3 <= len(template_by_shape) <= 4

    role_colors = set(colors_present(g, exclude={bg, divider_color}))
    query_colors = role_colors - template_colors
    assert len(query_colors) == 1
    query_color = next(iter(query_colors))

    queries = find_components(g, bg=bg, connectivity=4, monochromatic=True, colors=[query_color])
    assert queries

    out = g.copy()
    for comp in queries:
        coords = comp["coords"]
        out[coords[:, 0], coords[:, 1]] = bg

    static_occupied = out != bg
    moved_occupied = np.zeros_like(g, dtype=bool)
    h, w = g.shape

    for comp in queries:
        local = _component_local_coords(comp)
        key = _canonical_shape(local)
        assert key in template_by_shape
        target_color = template_by_shape[key]
        shift = int(bbox_w(comp["bbox"]))
        src = comp["coords"]
        dst = src + np.array([0, shift], dtype=int)

        assert bool((0 <= dst[:, 0]).all() and (dst[:, 0] < h).all())
        assert bool((0 <= dst[:, 1]).all() and (dst[:, 1] < w).all())
        assert not bool(static_occupied[dst[:, 0], dst[:, 1]].any())
        assert not bool(moved_occupied[dst[:, 0], dst[:, 1]].any())

        out[dst[:, 0], dst[:, 1]] = int(target_color)
        moved_occupied[dst[:, 0], dst[:, 1]] = True

    return out


def _selected_templates() -> list[str]:
    count = random.randint(3, 4)
    optional = random.sample(OPTIONAL_TEMPLATES, count - len(MANDATORY_TEMPLATES))
    names = list(MANDATORY_TEMPLATES) + optional
    random.shuffle(names)
    return names


def _legend_slots(divider_r: int, divider_c: int) -> list[tuple[int, int]]:
    top = random.randint(0, divider_r - 7)
    left = random.randint(0, divider_c - 7)
    return [(top + dr, left + dc) for dr in (0, 4) for dc in (0, 4)]


def _zone_candidates(
    blocked: np.ndarray,
    fixed_grid: np.ndarray,
    local_coords,
    divider_r: int,
    divider_c: int,
) -> list[tuple[int, int, str]]:
    h, w = fixed_grid.shape
    sprite_h, sprite_w = _bbox_from_coords(local_coords)
    shift = sprite_w
    zones = [
        ("top", 1, divider_r - sprite_h, divider_c + 2, w - (2 * shift)),
        ("bottom", divider_r + 2, h - sprite_h - 1, 1, w - (2 * shift)),
    ]
    candidates = []
    for name, r_lo, r_hi, c_lo, c_hi in zones:
        if r_lo > r_hi or c_lo > c_hi:
            continue
        for r0 in range(r_lo, r_hi + 1):
            for c0 in range(c_lo, c_hi + 1):
                src = _absolute_coords(local_coords, r0, c0)
                dst = [(r, c + shift) for r, c in src]
                if any(blocked[r, c] for r, c in src):
                    continue
                if any(blocked[r, c] for r, c in dst):
                    continue
                if any(fixed_grid[r, c] != BACKGROUND for r, c in src):
                    continue
                if any(fixed_grid[r, c] != BACKGROUND for r, c in dst):
                    continue
                candidates.append((r0, c0, name))
    return candidates


def _choose_variant(name: str, *, desired_width: int | None = None) -> tuple[tuple[int, int], ...]:
    variants = _variants(SHAPE_CATALOG[name])
    if desired_width is not None:
        variants = [coords for coords in variants if _bbox_from_coords(coords)[1] == desired_width]
    assert variants
    return random.choice(variants)


def _query_specs(template_names: list[str]) -> list[tuple[str, int | None]]:
    first_i3_width = random.choice((1, 3))
    specs = [(name, None) for name in template_names if name != "I3"]
    specs.append(("I3", first_i3_width))
    specs.append(("I3", 4 - first_i3_width))
    extra_count = random.randint(0, 2)
    for _ in range(extra_count):
        specs.append((random.choice(template_names), None))
    random.shuffle(specs)
    return specs


def _semantic_check(inp: np.ndarray) -> bool:
    try:
        out = main(inp)
    except Exception:
        return False

    if inp.ndim != 2 or out.ndim != 2 or max(inp.shape) > 30 or max(out.shape) > 30:
        return False

    bg = int(get_bg(inp))
    if bg != BACKGROUND:
        return False

    try:
        divider_color, divider_r, divider_c = _find_divider(inp, bg)
    except Exception:
        return False
    if divider_color != DIVIDER_COLOR:
        return False

    legend = inp[:divider_r, :divider_c]
    templates = find_components(legend, bg=bg, connectivity=4, monochromatic=True)
    if not (3 <= len(templates) <= 4):
        return False

    template_by_shape = {}
    template_names = set()
    for comp in templates:
        local = _component_local_coords(comp)
        key = _canonical_shape(local)
        if key in template_by_shape:
            return False
        template_by_shape[key] = int(comp["color"])
        for name, coords in SHAPE_CATALOG.items():
            if _canonical_shape(coords) == key:
                template_names.add(name)
                break

    if not {"I3", "T4", "Z4"} <= template_names:
        return False

    role_colors = set(colors_present(inp, exclude={bg, divider_color}))
    query_colors = role_colors - set(template_by_shape.values())
    if query_colors != {QUERY_COLOR}:
        return False

    queries = find_components(inp, bg=bg, connectivity=4, monochromatic=True, colors=[QUERY_COLOR])
    if len(queries) < len(templates) + 1:
        return False

    widths = []
    i3_widths = set()
    matched_keys = set()
    if not np.array_equal(inp[: divider_r + 1, : divider_c + 1][inp[: divider_r + 1, : divider_c + 1] != QUERY_COLOR],
                          out[: divider_r + 1, : divider_c + 1][inp[: divider_r + 1, : divider_c + 1] != QUERY_COLOR]):
        return False

    for comp in queries:
        local = _component_local_coords(comp)
        key = _canonical_shape(local)
        if key not in template_by_shape:
            return False
        matched_keys.add(key)
        shift = int(bbox_w(comp["bbox"]))
        widths.append(shift)
        if key == _canonical_shape(SHAPE_CATALOG["I3"]):
            i3_widths.add(shift)
        src = comp["coords"]
        dst = src + np.array([0, shift], dtype=int)
        if not np.all(out[src[:, 0], src[:, 1]] == bg):
            return False
        colors = out[dst[:, 0], dst[:, 1]]
        if not np.all(colors == colors[0]):
            return False
        if int(colors[0]) != template_by_shape[key]:
            return False

    if len(set(widths)) < 2 or i3_widths != {1, 3}:
        return False
    if matched_keys != set(template_by_shape):
        return False
    if np.any(out == QUERY_COLOR):
        return False
    return True


def _build_input() -> np.ndarray | None:
    h = random.randint(*GRID_HEIGHT_RANGE)
    w = random.randint(*GRID_WIDTH_RANGE)
    divider_r = random.randint(*DIVIDER_RC_RANGE)
    divider_c = random.randint(*DIVIDER_RC_RANGE)
    if divider_r + 6 >= h or divider_c + 8 >= w:
        return None

    grid = np.full((h, w), BACKGROUND, dtype=int)
    grid[: divider_r + 1, divider_c] = DIVIDER_COLOR
    grid[divider_r, : divider_c + 1] = DIVIDER_COLOR

    template_names = _selected_templates()
    colors = random.sample(TEMPLATE_COLORS, len(template_names))
    slots = _legend_slots(divider_r, divider_c)
    random.shuffle(slots)

    for name, color, slot in zip(template_names, colors, slots):
        coords = _choose_variant(name)
        patch = _render_patch(coords, color)
        slot_r, slot_c = slot
        dr = random.randint(0, 3 - patch.shape[0])
        dc = random.randint(0, 3 - patch.shape[1])
        paste(grid, patch, slot_r + dr, slot_c + dc, transparent=BACKGROUND)

    fixed_grid = grid.copy()
    blocked = fixed_grid != BACKGROUND
    _mark_blocked(blocked, np.argwhere(fixed_grid != BACKGROUND))

    for name, desired_width in _query_specs(template_names):
        coords = _choose_variant(name, desired_width=desired_width)
        candidates = _zone_candidates(blocked, fixed_grid, coords, divider_r, divider_c)
        if not candidates:
            return None
        top = [candidate for candidate in candidates if candidate[2] == "top"]
        bottom = [candidate for candidate in candidates if candidate[2] == "bottom"]
        if top and bottom:
            pool = top if random.random() < 0.45 else bottom
        else:
            pool = top or bottom
        r0, c0, _zone = random.choice(pool)
        src = _absolute_coords(coords, r0, c0)
        _paint_cells(grid, src, QUERY_COLOR)
        dst = [(r, c + _bbox_from_coords(coords)[1]) for r, c in src]
        _mark_blocked(blocked, src)
        _mark_blocked(blocked, dst)
        for r, c in src:
            fixed_grid[r, c] = QUERY_COLOR

    return grid if _semantic_check(grid) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_input()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC sample")


_CATALOG_KEYS = [_canonical_shape(coords) for coords in SHAPE_CATALOG.values()]
assert len(_CATALOG_KEYS) == len(set(_CATALOG_KEYS))
