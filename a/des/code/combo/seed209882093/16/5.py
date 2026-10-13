import random
from itertools import combinations

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
YELLOW = 4
MARKER_COLORS = (BLUE, RED, YELLOW)
TEXTURES = ("horizontal", "vertical", "checker")

SAMPLE_SIZE_CHOICES = (4, 5)
GRID_H_RANGE = (20, 30)
GRID_W_RANGE = (20, 30)
RECT_COUNT_RANGE = (2, 5)
RECT_H_RANGE = (3, 8)
RECT_W_RANGE = (3, 8)
GEN_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(pts)
    return parts


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), max(rows), min(cols), max(cols)


def _texture_mask(kind: str, h: int, w: int) -> np.ndarray:
    rr = np.arange(h)[:, None]
    cc = np.arange(w)[None, :]
    if kind == "horizontal":
        return np.broadcast_to((rr % 2) == 0, (h, w))
    if kind == "vertical":
        return np.broadcast_to((cc % 2) == 0, (h, w))
    if kind == "checker":
        return ((rr + cc) % 2) == 0
    raise ValueError(kind)


def _apply_texture_patch(color: int, kind: str, h: int, w: int) -> np.ndarray:
    patch = np.full((h, w), BLACK, dtype=int)
    patch[_texture_mask(kind, h, w)] = int(color)
    return patch


def _iter_color_components(grid: np.ndarray) -> list[dict]:
    comps: list[dict] = []
    for color in np.unique(grid):
        color = int(color)
        if color == BLACK:
            continue
        for pts in _components(grid == color):
            r0, r1, c0, c1 = _bbox(pts)
            comps.append(
                {
                    "color": color,
                    "points": pts,
                    "bbox": (r0, r1, c0, c1),
                    "size": len(pts),
                    "area": (r1 - r0 + 1) * (c1 - c0 + 1),
                }
            )
    return comps


def _classify_sample_texture(grid: np.ndarray, comp: dict) -> str:
    r0, r1, c0, c1 = comp["bbox"]
    color = int(comp["color"])
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    mask = patch == color
    if patch.shape[0] < 3 or patch.shape[1] < 3:
        raise ValueError("sample too small")
    interior = mask[1:-1, 1:-1]
    for kind in TEXTURES:
        if np.array_equal(interior, _texture_mask(kind, *interior.shape)):
            return kind
    raise ValueError("unknown texture")


def _legend_triplet(grid: np.ndarray) -> tuple[list[dict], dict[tuple[int, int], int]]:
    h, w = grid.shape
    comps = _iter_color_components(grid)
    labels = {
        comp["points"][0]: int(comp["color"])
        for comp in comps
        if comp["size"] == 1 and int(comp["color"]) in MARKER_COLORS
    }
    samples = [
        comp
        for comp in comps
        if comp["size"] > 1
        and comp["size"] < comp["area"]
        and comp["bbox"][1] - comp["bbox"][0] + 1 >= 3
        and comp["bbox"][3] - comp["bbox"][2] + 1 >= 3
    ]

    best = None
    best_score = None
    for triplet in combinations(samples, 3):
        ordered = sorted(triplet, key=lambda comp: comp["bbox"][2])
        dims = {
            (
                comp["bbox"][1] - comp["bbox"][0] + 1,
                comp["bbox"][3] - comp["bbox"][2] + 1,
            )
            for comp in ordered
        }
        top_rows = {comp["bbox"][0] for comp in ordered}
        if len(dims) != 1 or len(top_rows) != 1:
            continue

        matched: dict[tuple[int, int], int] = {}
        used_positions: set[tuple[int, int]] = set()
        ok = True
        for comp in ordered:
            r0, _, c0, c1 = comp["bbox"]
            candidates = [
                (pos, color)
                for pos, color in labels.items()
                if pos[0] == r0 - 1 and c0 <= pos[1] <= c1 and pos not in used_positions
            ]
            if len(candidates) != 1:
                ok = False
                break
            pos, color = candidates[0]
            matched[pos] = color
            used_positions.add(pos)
        if not ok or set(matched.values()) != set(MARKER_COLORS):
            continue

        r0 = min(comp["bbox"][0] for comp in ordered) - 1
        r1 = max(comp["bbox"][1] for comp in ordered)
        c0 = min(comp["bbox"][2] for comp in ordered)
        c1 = max(comp["bbox"][3] for comp in ordered)
        score = min(
            r0 + c0,
            r0 + (w - 1 - c1),
            (h - 1 - r1) + c0,
            (h - 1 - r1) + (w - 1 - c1),
        )
        if best is None or score < best_score:
            best = ordered
            best_score = score

    if best is None:
        raise ValueError("legend not found")
    return best, labels


def _read_legend(grid: np.ndarray) -> tuple[dict[int, str], np.ndarray]:
    samples, labels = _legend_triplet(grid)
    legend_mask = np.zeros_like(grid, dtype=bool)
    mapping: dict[int, str] = {}
    for comp in samples:
        r0, _, c0, c1 = comp["bbox"]
        label_pos = None
        for pos, color in labels.items():
            if pos[0] == r0 - 1 and c0 <= pos[1] <= c1:
                label_pos = pos
                mapping[color] = _classify_sample_texture(grid, comp)
                break
        if label_pos is None:
            raise ValueError("sample missing label")
        for r, c in comp["points"]:
            legend_mask[r, c] = True
        legend_mask[label_pos] = True
    if set(mapping) != set(MARKER_COLORS):
        raise ValueError("incomplete legend mapping")
    return mapping, legend_mask


def _pick_side_marker(marker_positions: dict[tuple[int, int], int], cells: list[tuple[int, int]]) -> int | None:
    colors = []
    for cell in cells:
        if cell in marker_positions:
            colors.append(marker_positions[cell])
    if not colors:
        return None
    unique = list(dict.fromkeys(colors))
    return unique[0]


def _choose_marker_color(side_colors: dict[str, int | None]) -> int | None:
    present = [color for color in side_colors.values() if color is not None]
    if not present:
        return None
    if len(set(present)) == 1:
        return present[0]
    for side in ("top", "left", "bottom", "right"):
        color = side_colors[side]
        if color is not None:
            return color
    return None


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    mapping, legend_mask = _read_legend(g)

    marker_positions: dict[tuple[int, int], int] = {}
    rectangles: list[tuple[int, int, int, int, int]] = []
    for comp in _iter_color_components(g):
        points = comp["points"]
        if any(legend_mask[r, c] for r, c in points):
            continue
        color = int(comp["color"])
        r0, r1, c0, c1 = comp["bbox"]
        if comp["size"] == 1 and color in MARKER_COLORS:
            marker_positions[points[0]] = color
            continue
        if comp["size"] == comp["area"]:
            rectangles.append((r0, r1, c0, c1, color))

    out = g.copy()
    for r0, r1, c0, c1, color in rectangles:
        side_colors = {
            "top": _pick_side_marker(marker_positions, [(r0 - 1, c) for c in range(c0, c1 + 1)] if r0 > 0 else []),
            "left": _pick_side_marker(marker_positions, [(r, c0 - 1) for r in range(r0, r1 + 1)] if c0 > 0 else []),
            "bottom": _pick_side_marker(marker_positions, [(r1 + 1, c) for c in range(c0, c1 + 1)] if r1 + 1 < h else []),
            "right": _pick_side_marker(marker_positions, [(r, c1 + 1) for r in range(r0, r1 + 1)] if c1 + 1 < w else []),
        }
        chosen = _choose_marker_color(side_colors)
        if chosen is None:
            continue
        kind = mapping[chosen]
        out[r0 : r1 + 1, c0 : c1 + 1] = _apply_texture_patch(color, kind, r1 - r0 + 1, c1 - c0 + 1)
    return out


def _build_sample_patch(size: int, texture: str, color: int) -> np.ndarray:
    patch = np.full((size, size), BLACK, dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    patch[1:-1, 1:-1][_texture_mask(texture, size - 2, size - 2)] = color
    return patch


def _legend_geometry(h: int, w: int, corner: str, sample_size: int) -> tuple[int, int, int, int]:
    legend_h = sample_size + 1
    legend_w = 3 * sample_size + 2
    if corner == "tl":
        return 0, 0, legend_h, legend_w
    if corner == "tr":
        return 0, w - legend_w, legend_h, legend_w
    if corner == "bl":
        return h - legend_h, 0, legend_h, legend_w
    if corner == "br":
        return h - legend_h, w - legend_w, legend_h, legend_w
    raise ValueError(corner)


def _reserve(mask: np.ndarray, r0: int, r1: int, c0: int, c1: int) -> None:
    h, w = mask.shape
    mask[max(0, r0) : min(h, r1 + 1), max(0, c0) : min(w, c1 + 1)] = True


def _place_legend(
    grid: np.ndarray,
    reserved: np.ndarray,
    rng: random.Random,
    sample_size: int,
    legend_fg: int,
) -> dict[int, str]:
    h, w = grid.shape
    textures = list(TEXTURES)
    rng.shuffle(textures)
    mapping = {RED: textures[0], BLUE: textures[1], YELLOW: textures[2]}
    order = [RED, BLUE, YELLOW]
    rng.shuffle(order)
    corner = rng.choice(["tl", "tr", "bl", "br"])
    top, left, legend_h, legend_w = _legend_geometry(h, w, corner, sample_size)
    sample_top = top + 1
    label_row = top

    for idx, label_color in enumerate(order):
        sc = left + idx * (sample_size + 1)
        patch = _build_sample_patch(sample_size, mapping[label_color], legend_fg)
        grid[sample_top : sample_top + sample_size, sc : sc + sample_size] = patch
        grid[label_row, sc + sample_size // 2] = label_color

    _reserve(reserved, top - 1, top + legend_h, left - 1, left + legend_w)
    return mapping


def _try_place_rectangle(
    reserved: np.ndarray,
    rng: random.Random,
    rect_h: int,
    rect_w: int,
) -> tuple[int, int] | None:
    h, w = reserved.shape
    if rect_h + 2 > h or rect_w + 2 > w:
        return None
    for _ in range(80):
        r0 = rng.randint(1, h - rect_h - 1)
        c0 = rng.randint(1, w - rect_w - 1)
        r1 = r0 + rect_h - 1
        c1 = c0 + rect_w - 1
        if reserved[r0 - 1 : r1 + 2, c0 - 1 : c1 + 2].any():
            continue
        return r0, c0
    return None


def _sample_marker_spec(rng: random.Random, require_active: bool = False) -> dict[str, int]:
    sides = ["top", "left", "bottom", "right"]
    if require_active:
        mode = rng.choice(["single", "multi"])
    else:
        mode = rng.choices(["none", "single", "multi"], weights=[2, 4, 4], k=1)[0]

    if mode == "none":
        return {}

    if mode == "single":
        chosen_sides = rng.sample(sides, rng.randint(1, 4))
        color = rng.choice(MARKER_COLORS)
        return {side: color for side in chosen_sides}

    chosen_sides = rng.sample(sides, rng.randint(2, 4))
    side_colors = {side: rng.choice(MARKER_COLORS) for side in chosen_sides}
    if len(set(side_colors.values())) == 1:
        force_side = rng.choice(chosen_sides)
        alternatives = [color for color in MARKER_COLORS if color != side_colors[force_side]]
        side_colors[force_side] = rng.choice(alternatives)
    return side_colors


def _marker_position(
    rng: random.Random,
    side: str,
    r0: int,
    r1: int,
    c0: int,
    c1: int,
) -> tuple[int, int]:
    if side == "top":
        return r0 - 1, rng.randint(c0, c1)
    if side == "bottom":
        return r1 + 1, rng.randint(c0, c1)
    if side == "left":
        return rng.randint(r0, r1), c0 - 1
    if side == "right":
        return rng.randint(r0, r1), c1 + 1
    raise ValueError(side)


def _rect_output_color(marker_spec: dict[str, int], mapping: dict[int, str]) -> str | None:
    side_colors = {side: marker_spec.get(side) for side in ("top", "left", "bottom", "right")}
    chosen = _choose_marker_color(side_colors)
    if chosen is None:
        return None
    return mapping[chosen]


def _paint_expected_rectangle(
    out: np.ndarray,
    r0: int,
    r1: int,
    c0: int,
    c1: int,
    color: int,
    kind: str | None,
) -> None:
    if kind is None:
        return
    out[r0 : r1 + 1, c0 : c1 + 1] = _apply_texture_patch(color, kind, r1 - r0 + 1, c1 - c0 + 1)


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    sample_size = rng.choice(SAMPLE_SIZE_CHOICES)
    min_h = max(GRID_H_RANGE[0], sample_size + 6)
    min_w = max(GRID_W_RANGE[0], 3 * sample_size + 6)
    h = rng.randint(min_h, GRID_H_RANGE[1])
    w = rng.randint(min_w, GRID_W_RANGE[1])
    grid = np.full((h, w), BLACK, dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    palette = [c for c in range(10) if c not in (BLACK, BLUE, RED, YELLOW)]
    rng.shuffle(palette)
    legend_fg = palette[0]
    rect_colors = palette[1:]
    rect_count = rng.randint(RECT_COUNT_RANGE[0], min(RECT_COUNT_RANGE[1], len(rect_colors)))

    legend_mapping = _place_legend(grid, reserved, rng, sample_size, legend_fg)

    rects = []
    for idx in range(rect_count):
        rect_h = rng.randint(RECT_H_RANGE[0], min(RECT_H_RANGE[1], h - 2))
        rect_w = rng.randint(RECT_W_RANGE[0], min(RECT_W_RANGE[1], w - 2))
        pos = _try_place_rectangle(reserved, rng, rect_h, rect_w)
        if pos is None:
            return None
        r0, c0 = pos
        r1 = r0 + rect_h - 1
        c1 = c0 + rect_w - 1
        color = rect_colors[idx]
        grid[r0 : r1 + 1, c0 : c1 + 1] = color
        _reserve(reserved, r0 - 1, r1 + 1, c0 - 1, c1 + 1)
        rects.append((r0, r1, c0, c1, color))

    marker_specs = [_sample_marker_spec(rng, require_active=(i == 0)) for i in range(rect_count)]
    if rect_count >= 2 and rng.random() < 0.5 and not any(not spec for spec in marker_specs):
        marker_specs[rng.randrange(1, rect_count)] = {}
    if rect_count >= 2 and rng.random() < 0.6 and not any(len(set(spec.values())) >= 2 for spec in marker_specs if spec):
        idx = rng.randrange(rect_count)
        marker_specs[idx] = _sample_marker_spec(rng, require_active=True)
        while len(set(marker_specs[idx].values())) < 2:
            marker_specs[idx] = _sample_marker_spec(rng, require_active=True)

    used_marker_positions: set[tuple[int, int]] = set()
    for spec, (r0, r1, c0, c1, _) in zip(marker_specs, rects):
        for side, marker_color in spec.items():
            for _ in range(20):
                pos = _marker_position(rng, side, r0, r1, c0, c1)
                if pos not in used_marker_positions and grid[pos] == BLACK:
                    grid[pos] = marker_color
                    used_marker_positions.add(pos)
                    break
            else:
                return None

    out = grid.copy()
    for spec, (r0, r1, c0, c1, color) in zip(marker_specs, rects):
        kind = _rect_output_color(spec, legend_mapping)
        _paint_expected_rectangle(out, r0, r1, c0, c1, color, kind)

    try:
        if not np.array_equal(main(grid), out):
            return None
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
