import random
from collections import Counter
from dataclasses import dataclass

import numpy as np


RED = 2
GREEN = 3
YELLOW = 4
ARROW_COLORS = {RED, GREEN, YELLOW}
ALL_COLORS = list(range(10))
BG_CHOICES = [0, 1, 5]
OBJECT_COLORS = [0, 1, 5, 6, 7, 8, 9]
NEIGHBORS8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

DIR_VECS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
ARROW_PATTERNS = {
    "right": {(0, 1), (1, 0), (2, 1)},
    "left": {(0, 0), (1, 1), (2, 0)},
    "up": {(0, 0), (0, 2), (1, 1)},
    "down": {(0, 1), (1, 0), (1, 2)},
}

GEN_ATTEMPTS = 400
MASK_ATTEMPTS = 200
PAIR_ATTEMPTS = 120
GRID_MIN = 20
GRID_MAX = 30


@dataclass(frozen=True)
class Component:
    color: int
    coords: np.ndarray
    r0: int
    c0: int
    r1: int
    c1: int


@dataclass(frozen=True)
class Arrow:
    color: int
    direction: str
    coords: np.ndarray
    r0: int
    c0: int
    r1: int
    c1: int


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_components(grid: np.ndarray, bg: int) -> list[Component]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[Component] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == bg:
                continue
            color = int(g[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in NEIGHBORS8:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            arr = np.array(cells, dtype=int)
            r0 = int(arr[:, 0].min())
            c0 = int(arr[:, 1].min())
            r1 = int(arr[:, 0].max())
            c1 = int(arr[:, 1].max())
            comps.append(Component(color, arr, r0, c0, r1, c1))
    return comps


def _detect_arrow(comp: Component) -> Arrow | None:
    if comp.color not in ARROW_COLORS or len(comp.coords) != 3:
        return None
    norm = {(int(r - comp.r0), int(c - comp.c0)) for r, c in comp.coords}
    for direction, pattern in ARROW_PATTERNS.items():
        if norm == pattern:
            return Arrow(comp.color, direction, comp.coords, comp.r0, comp.c0, comp.r1, comp.c1)
    return None


def _overlap_len(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0) + 1)


def _side_gap(arrow: Arrow, comp: Component, forward: bool) -> tuple[int, int] | None:
    if arrow.direction == "right":
        overlap = _overlap_len(arrow.r0, arrow.r1, comp.r0, comp.r1)
        if not overlap:
            return None
        if forward and comp.c0 > arrow.c1:
            return comp.c0 - arrow.c1, overlap
        if not forward and comp.c1 < arrow.c0:
            return arrow.c0 - comp.c1, overlap
        return None
    if arrow.direction == "left":
        overlap = _overlap_len(arrow.r0, arrow.r1, comp.r0, comp.r1)
        if not overlap:
            return None
        if forward and comp.c1 < arrow.c0:
            return arrow.c0 - comp.c1, overlap
        if not forward and comp.c0 > arrow.c1:
            return comp.c0 - arrow.c1, overlap
        return None
    if arrow.direction == "down":
        overlap = _overlap_len(arrow.c0, arrow.c1, comp.c0, comp.c1)
        if not overlap:
            return None
        if forward and comp.r0 > arrow.r1:
            return comp.r0 - arrow.r1, overlap
        if not forward and comp.r1 < arrow.r0:
            return arrow.r0 - comp.r1, overlap
        return None
    overlap = _overlap_len(arrow.c0, arrow.c1, comp.c0, comp.c1)
    if not overlap:
        return None
    if forward and comp.r1 < arrow.r0:
        return arrow.r0 - comp.r1, overlap
    if not forward and comp.r0 > arrow.r1:
        return comp.r0 - arrow.r1, overlap
    return None


def _nearest_component(arrow: Arrow, objects: list[Component], forward: bool, exclude: set[int] | None = None) -> tuple[int, int] | None:
    exclude = exclude or set()
    candidates: list[tuple[int, int, int, int, int]] = []
    for idx, comp in enumerate(objects):
        if idx in exclude:
            continue
        gap_info = _side_gap(arrow, comp, forward=forward)
        if gap_info is None:
            continue
        gap, overlap = gap_info
        if 1 <= gap <= 4:
            candidates.append((gap, -overlap, comp.r0, comp.c0, idx))
    if not candidates:
        return None
    candidates.sort()
    gap, _, _, _, idx = candidates[0]
    return gap, idx


def _copy_coords(comp: Component, direction: str, shift: int, mirror: bool) -> np.ndarray:
    rel = comp.coords - np.array([[comp.r0, comp.c0]])
    h = comp.r1 - comp.r0 + 1
    w = comp.c1 - comp.c0 + 1
    rel = rel.copy()
    if mirror:
        if direction in {"left", "right"}:
            rel[:, 1] = (w - 1) - rel[:, 1]
        else:
            rel[:, 0] = (h - 1) - rel[:, 0]
    dr, dc = DIR_VECS[direction]
    rel[:, 0] += dr * shift
    rel[:, 1] += dc * shift
    return rel + np.array([[comp.r0, comp.c0]])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read each colored chevron arrow and stamp a shifted copy of the object on the side it points toward.

    Input structure:
    - The background is the dominant color.
    - Foreground objects are small single-color connected components with 3x3 to 5x5 bounding boxes.
    - Some 3-pixel red, green, or yellow 8-connected chevrons form arrows pointing up, down, left, or right.
    - For each arrow, one object sits on the pointed-to side at a gap of 1-4 pixels from the arrow tip; another object may also sit on the opposite side.

    Rule:
    - Detect every arrow from its 3-pixel chevron shape and direction.
    - Let D be the gap from the arrow tip side to the nearest object on the pointed-to side, and draw a copy of that object shifted by D cells in the same direction.
    - Red arrows copy the object unchanged; green arrows mirror it across the axis perpendicular to motion before shifting; yellow arrows recolor the shifted copy to the color of the nearest opposite-side object when one exists, otherwise they keep the source color.
    - Keep all original objects and all arrows unchanged.

    Key insight:
    - The chevron geometry gives direction, the source-side gap encodes the translation distance, and the arrow color selects which copy transformation to apply.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    bg = _bg_color(g)
    comps = _find_components(g, bg)

    arrows: list[Arrow] = []
    objects: list[Component] = []
    for comp in comps:
        arrow = _detect_arrow(comp)
        if arrow is not None:
            arrows.append(arrow)
        else:
            objects.append(comp)

    assert arrows, "expected at least one arrow"
    out = g.copy()
    used_sources: set[int] = set()

    arrows.sort(key=lambda a: (a.r0, a.c0, a.direction))
    for arrow in arrows:
        source_match = _nearest_component(arrow, objects, forward=True)
        assert source_match is not None, "each arrow needs one pointed-to object"
        shift, src_idx = source_match
        assert src_idx not in used_sources, "each source object should be controlled by one arrow"
        used_sources.add(src_idx)

        donor_match = _nearest_component(arrow, objects, forward=False, exclude={src_idx})
        donor = objects[donor_match[1]] if donor_match is not None else None
        source = objects[src_idx]

        mirror = arrow.color == GREEN
        copy_color = source.color
        if arrow.color == YELLOW and donor is not None:
            copy_color = donor.color

        shifted = _copy_coords(source, arrow.direction, shift, mirror=mirror)
        assert np.all((0 <= shifted[:, 0]) & (shifted[:, 0] < h) & (0 <= shifted[:, 1]) & (shifted[:, 1] < w))
        out[shifted[:, 0], shifted[:, 1]] = copy_color

    return out


def _mask_coords(mask: np.ndarray, top: int, left: int) -> np.ndarray:
    coords = np.argwhere(mask)
    return coords + np.array([[top, left]])


def _random_mask(rng: random.Random, h: int, w: int, asym: str | None = None) -> np.ndarray | None:
    area_min = max(5, (h * w + 2) // 3)
    area_max = min(h * w - 1, (2 * h * w) // 3 + 1)
    if area_min > area_max:
        area_min = area_max = max(4, min(h * w - 1, h * w))
    for _ in range(MASK_ATTEMPTS):
        target = rng.randint(area_min, area_max)
        cells = {(rng.randrange(h), rng.randrange(w))}
        while len(cells) < target:
            frontier = []
            for r, c in cells:
                for dr, dc in NEIGHBORS4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(rng.choice(frontier))
        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue
        if asym == "h" and np.array_equal(mask, np.fliplr(mask)):
            continue
        if asym == "v" and np.array_equal(mask, np.flipud(mask)):
            continue
        return mask
    return None


def _sample_source_mask(rng: random.Random, direction: str, arrow_color: int, shift: int, clear_copy: bool) -> np.ndarray | None:
    asym = None
    if arrow_color == GREEN:
        asym = "h" if direction in {"left", "right"} else "v"

    if direction in {"left", "right"}:
        w_hi = min(4, shift) if clear_copy else 5
        if w_hi < 3:
            return None
        h = rng.randint(3, 5)
        w = rng.randint(3, w_hi)
    else:
        h_hi = min(4, shift) if clear_copy else 5
        if h_hi < 3:
            return None
        h = rng.randint(3, h_hi)
        w = rng.randint(3, 5)
    return _random_mask(rng, h, w, asym=asym)


def _mirror_mask(mask: np.ndarray, direction: str) -> np.ndarray:
    if direction in {"left", "right"}:
        return np.fliplr(mask)
    return np.flipud(mask)


def _shifted_mask_coords(mask: np.ndarray, top: int, left: int, direction: str, shift: int, mirror: bool) -> np.ndarray:
    patch = _mirror_mask(mask, direction) if mirror else mask
    dr, dc = DIR_VECS[direction]
    return _mask_coords(patch, top + dr * shift, left + dc * shift)


def _arrow_local_coords(direction: str, top: int, left: int) -> np.ndarray:
    cells = np.array(sorted(ARROW_PATTERNS[direction]), dtype=int)
    return cells + np.array([[top, left]])


def _build_pair(rng: random.Random, direction: str, arrow_color: int, source_color: int, donor_color: int | None) -> dict | None:
    has_donor = donor_color is not None
    clear_copy = arrow_color == GREEN or has_donor
    shift = rng.randint(3, 4) if clear_copy else rng.randint(1, 4)
    source_mask = _sample_source_mask(rng, direction, arrow_color, shift, clear_copy)
    if source_mask is None:
        return None

    src_h, src_w = map(int, source_mask.shape)
    source_top = 0
    source_left = 0

    if direction == "right":
        arrow_top = rng.randint(0, src_h - 3)
        arrow_left = source_left - shift - 1
        donor_gap = rng.randint(1, 4) if has_donor else None
        donor_mask = _random_mask(rng, rng.randint(3, 5), rng.randint(3, 5)) if has_donor else None
        donor_top = arrow_top if has_donor else None
        donor_left = arrow_left - donor_gap - donor_mask.shape[1] + 1 if has_donor else None
    elif direction == "left":
        arrow_top = rng.randint(0, src_h - 3)
        arrow_left = source_left + src_w - 1 + shift
        donor_gap = rng.randint(1, 4) if has_donor else None
        donor_mask = _random_mask(rng, rng.randint(3, 5), rng.randint(3, 5)) if has_donor else None
        donor_top = arrow_top if has_donor else None
        donor_left = arrow_left + 1 + donor_gap if has_donor else None
    elif direction == "down":
        arrow_left = rng.randint(0, src_w - 3)
        arrow_top = source_top - shift - 1
        donor_gap = rng.randint(1, 4) if has_donor else None
        donor_mask = _random_mask(rng, rng.randint(3, 5), rng.randint(3, 5)) if has_donor else None
        donor_top = arrow_top - donor_gap - donor_mask.shape[0] + 1 if has_donor else None
        donor_left = arrow_left if has_donor else None
    else:
        arrow_left = rng.randint(0, src_w - 3)
        arrow_top = source_top + src_h - 1 + shift
        donor_gap = rng.randint(1, 4) if has_donor else None
        donor_mask = _random_mask(rng, rng.randint(3, 5), rng.randint(3, 5)) if has_donor else None
        donor_top = arrow_top + 1 + donor_gap if has_donor else None
        donor_left = arrow_left if has_donor else None

    source_coords = _mask_coords(source_mask, source_top, source_left)
    arrow_coords = _arrow_local_coords(direction, arrow_top, arrow_left)
    copy_coords = _shifted_mask_coords(
        source_mask,
        source_top,
        source_left,
        direction,
        shift,
        mirror=(arrow_color == GREEN),
    )
    donor_coords = _mask_coords(donor_mask, donor_top, donor_left) if has_donor else None

    if clear_copy and set(map(tuple, copy_coords.tolist())) & set(map(tuple, source_coords.tolist())):
        return None

    parts = [source_coords, arrow_coords, copy_coords]
    if donor_coords is not None:
        parts.append(donor_coords)
    min_r = min(int(coords[:, 0].min()) for coords in parts)
    min_c = min(int(coords[:, 1].min()) for coords in parts)
    shifted_parts = [coords - np.array([[min_r, min_c]]) for coords in parts]
    source_coords, arrow_coords, copy_coords = shifted_parts[:3]
    donor_coords = shifted_parts[3] if donor_coords is not None else None

    max_r = max(int(coords[:, 0].max()) for coords in shifted_parts)
    max_c = max(int(coords[:, 1].max()) for coords in shifted_parts)
    rect_h = max_r + 1
    rect_w = max_c + 1

    return {
        "rect_h": rect_h,
        "rect_w": rect_w,
        "source_coords": source_coords,
        "source_color": source_color,
        "arrow_coords": arrow_coords,
        "arrow_color": arrow_color,
        "copy_coords": copy_coords,
        "copy_color": donor_color if arrow_color == YELLOW and donor_color is not None else source_color,
        "donor_coords": donor_coords,
        "donor_color": donor_color,
    }


def _place_rectangles(rng: random.Random, h: int, w: int, pairs: list[dict]) -> list[tuple[int, int]] | None:
    reserved = np.zeros((h, w), dtype=bool)
    placements: list[tuple[int, int] | None] = [None] * len(pairs)
    order = sorted(range(len(pairs)), key=lambda idx: pairs[idx]["rect_h"] * pairs[idx]["rect_w"], reverse=True)
    for idx in order:
        ph = int(pairs[idx]["rect_h"])
        pw = int(pairs[idx]["rect_w"])
        options = []
        for top in range(h - ph + 1):
            for left in range(w - pw + 1):
                if reserved[top : top + ph, left : left + pw].any():
                    continue
                options.append((top, left))
        if not options:
            return None
        top, left = rng.choice(options)
        placements[idx] = (top, left)
        r0 = max(0, top - 1)
        c0 = max(0, left - 1)
        r1 = min(h, top + ph + 1)
        c1 = min(w, left + pw + 1)
        reserved[r0:r1, c0:c1] = True
    return [p for p in placements if p is not None]


def _build_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(GRID_MIN, GRID_MAX)
    w = rng.randint(GRID_MIN, GRID_MAX)
    bg = rng.choice(BG_CHOICES)

    num_pairs = rng.choice([3, 4])
    arrow_colors = [RED, GREEN, YELLOW]
    if num_pairs == 4:
        arrow_colors.append(rng.choice([RED, GREEN, YELLOW]))
    rng.shuffle(arrow_colors)

    directions = [rng.choice(list(DIR_VECS)) for _ in range(num_pairs)]
    available_colors = [c for c in OBJECT_COLORS if c not in {bg, RED, GREEN, YELLOW}]
    rng.shuffle(available_colors)
    needed_colors = num_pairs + sum(1 for color in arrow_colors if color == YELLOW and rng.random() < 0.5)
    if needed_colors > len(available_colors):
        return None

    pairs = []
    color_idx = 0
    for direction, arrow_color in zip(directions, arrow_colors):
        source_color = available_colors[color_idx]
        color_idx += 1
        donor_color = None
        if arrow_color == YELLOW and rng.random() < 0.5:
            donor_color = available_colors[color_idx]
            color_idx += 1
        placed = None
        for _ in range(PAIR_ATTEMPTS):
            placed = _build_pair(rng, direction, arrow_color, source_color, donor_color)
            if placed is not None:
                break
        if placed is None:
            return None
        pairs.append(placed)

    placements = _place_rectangles(rng, h, w, pairs)
    if placements is None:
        return None

    inp = np.full((h, w), bg, dtype=int)
    expected = np.full((h, w), bg, dtype=int)

    for pair, (top, left) in zip(pairs, placements):
        for coords, color in (
            (pair["source_coords"], pair["source_color"]),
            (pair["arrow_coords"], pair["arrow_color"]),
        ):
            rr = coords[:, 0] + top
            cc = coords[:, 1] + left
            if np.any(inp[rr, cc] != bg):
                return None
            inp[rr, cc] = color
            expected[rr, cc] = color

        if pair["donor_coords"] is not None:
            rr = pair["donor_coords"][:, 0] + top
            cc = pair["donor_coords"][:, 1] + left
            if np.any(inp[rr, cc] != bg):
                return None
            inp[rr, cc] = int(pair["donor_color"])
            expected[rr, cc] = int(pair["donor_color"])

        rr = pair["copy_coords"][:, 0] + top
        cc = pair["copy_coords"][:, 1] + left
        copy_color = int(pair["copy_color"])
        expected[rr, cc] = copy_color

    if _bg_color(inp) != bg:
        return None
    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _build_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid arrow-copy ARC input")
