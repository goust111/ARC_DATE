from __future__ import annotations

import random

import numpy as np


BG = 0
TEMPLATE_COLOR = 5
PIECE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
PATTERN_TEMPLATES = (
    (1, 0, 1),
    (1, 1, 0),
    (0, 1, 1),
    (1, 0, 0),
    (0, 1, 0),
    (1, 1, 1),
)
UNIQUE_SHAPES = (
    (2, 2),
    (2, 3),
    (2, 4),
    (3, 3),
    (3, 4),
)
SIDES = ("top", "bottom", "left", "right")
MAX_GEN_ATTEMPTS = 400


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        stack = [(r, c)]
        seen[r, c] = True
        coords: list[tuple[int, int]] = []
        while stack:
            rr, cc = stack.pop()
            coords.append((rr, cc))
            if rr > 0 and mask[rr - 1, cc] and not seen[rr - 1, cc]:
                seen[rr - 1, cc] = True
                stack.append((rr - 1, cc))
            if rr + 1 < h and mask[rr + 1, cc] and not seen[rr + 1, cc]:
                seen[rr + 1, cc] = True
                stack.append((rr + 1, cc))
            if cc > 0 and mask[rr, cc - 1] and not seen[rr, cc - 1]:
                seen[rr, cc - 1] = True
                stack.append((rr, cc - 1))
            if cc + 1 < w and mask[rr, cc + 1] and not seen[rr, cc + 1]:
                seen[rr, cc + 1] = True
                stack.append((rr, cc + 1))
        comps.append(np.array(coords, dtype=int))
    return comps


def _coords_bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _coords_to_mask(coords: np.ndarray, bbox: tuple[int, int, int, int] | None = None) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    if bbox is None:
        bbox = _coords_bbox(coords)
    r0, c0, r1, c1 = bbox
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel = coords - np.array([r0, c0], dtype=int)
    mask[rel[:, 0], rel[:, 1]] = True
    return mask, bbox


def _canonical_rotation_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    best: tuple[tuple[int, int], bytes] | None = None
    for k in range(4):
        rot = np.rot90(mask, k)
        key = (tuple(map(int, rot.shape)), rot.tobytes())
        if best is None or key < best:
            best = key
    assert best is not None
    return best


def _largest_monochrome_component(grid: np.ndarray, bg: int) -> dict:
    best: dict | None = None
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        for coords in _components(grid == color):
            area = int(len(coords))
            if best is None or area > best["area"]:
                mask, bbox = _coords_to_mask(coords)
                best = {
                    "color": color,
                    "coords": coords,
                    "bbox": bbox,
                    "mask": mask,
                    "area": area,
                }
    assert best is not None
    return best


def _legend_row_valid(row: np.ndarray, bg: int, template_color: int) -> bool:
    color = int(row[0])
    if color == bg or color == template_color:
        return False
    sample = [int(v) for v in row[1:4]]
    if not any(v == color for v in sample):
        return False
    return all(v in (bg, color) for v in sample)


def _find_legend(grid: np.ndarray, bg: int, template_color: int) -> tuple[dict[int, tuple[int, int, int]], np.ndarray]:
    h, w = grid.shape
    best_score: tuple[int, int, int, int] | None = None
    best_data: tuple[int, list[int]] | None = None
    start_col = max(0, w - 8)
    for c0 in range(start_col, w - 3):
        rows: list[int] = []
        r = h - 1
        while r >= 0 and _legend_row_valid(grid[r, c0 : c0 + 4], bg, template_color):
            if c0 + 4 < w and np.any(grid[r, c0 + 4 :] != bg):
                break
            rows.append(r)
            r -= 1
        rows.reverse()
        if len(rows) < 2:
            continue
        colors = [int(grid[r0, c0]) for r0 in rows]
        if len(set(colors)) != len(colors):
            continue
        score = (len(rows), c0, rows[0], rows[-1])
        if best_score is None or score > best_score:
            best_score = score
            best_data = (c0, rows)
    assert best_data is not None, "failed to locate legend"
    c0, rows = best_data
    mapping: dict[int, tuple[int, int, int]] = {}
    legend_mask = np.zeros_like(grid, dtype=bool)
    for r in rows:
        color = int(grid[r, c0])
        pattern = tuple(int(v) for v in grid[r, c0 + 1 : c0 + 4].tolist())
        mapping[color] = pattern
        legend_mask[r, c0 : c0 + 4] = True
    return mapping, legend_mask


def _template_mask(grid: np.ndarray, template: dict) -> np.ndarray:
    mask = np.zeros_like(grid, dtype=bool)
    coords = template["coords"]
    mask[coords[:, 0], coords[:, 1]] = True
    return mask


def _find_notches(grid: np.ndarray, template: dict, bg: int) -> list[dict]:
    r0, c0, r1, c1 = template["bbox"]
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    notch_mask = patch == bg
    notches: list[dict] = []
    for coords in _components(notch_mask):
        touches = set()
        if np.any(coords[:, 0] == 0):
            touches.add("top")
        if np.any(coords[:, 0] == patch.shape[0] - 1):
            touches.add("bottom")
        if np.any(coords[:, 1] == 0):
            touches.add("left")
        if np.any(coords[:, 1] == patch.shape[1] - 1):
            touches.add("right")
        if not touches:
            continue
        assert len(touches) == 1, "each notch must touch exactly one template edge"
        local_mask, local_bbox = _coords_to_mask(coords)
        nr0, nc0, nr1, nc1 = local_bbox
        notches.append(
            {
                "side": next(iter(touches)),
                "bbox": (r0 + nr0, c0 + nc0, r0 + nr1, c0 + nc1),
                "mask": local_mask,
                "key": _canonical_rotation_key(local_mask),
            }
        )
    assert notches, "expected at least one notch"
    return notches


def _find_pieces(
    grid: np.ndarray,
    bg: int,
    template_color: int,
    legend_mask: np.ndarray,
    template_mask: np.ndarray,
    colors: list[int],
) -> dict[int, dict]:
    pieces: dict[int, dict] = {}
    blocked = legend_mask | template_mask
    for color in colors:
        mask = (grid == int(color)) & (~blocked)
        comps = _components(mask)
        assert len(comps) == 1, "each legend color must have exactly one piece"
        coords = comps[0]
        piece_mask, bbox = _coords_to_mask(coords)
        pieces[int(color)] = {
            "bbox": bbox,
            "coords": coords,
            "mask": piece_mask,
            "key": _canonical_rotation_key(piece_mask),
        }
    other_fg = (grid != bg) & (~blocked)
    for color in colors:
        other_fg &= grid != int(color)
    assert not bool(other_fg.any()), "unexpected extra foreground outside template and legend"
    return pieces


def _fill_notch(out: np.ndarray, notch: dict, pattern: tuple[int, int, int]) -> None:
    r0, c0, _r1, _c1 = notch["bbox"]
    mask = notch["mask"]
    h, w = mask.shape
    if notch["side"] in {"top", "bottom"}:
        for rr in range(h):
            for cc in range(w):
                if mask[rr, cc]:
                    out[r0 + rr, c0 + cc] = int(pattern[cc % 3])
    else:
        for rr in range(h):
            for cc in range(w):
                if mask[rr, cc]:
                    out[r0 + rr, c0 + cc] = int(pattern[rr % 3])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match colored pieces to template notches by shape, then fill the notches using
    the legend's 3-cell texture sample for each piece color.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    template = _largest_monochrome_component(g, bg)
    template_color = int(template["color"])

    legend, legend_mask = _find_legend(g, bg, template_color)
    tmask = _template_mask(g, template)
    notches = _find_notches(g, template, bg)
    notch_by_key = {notch["key"]: notch for notch in notches}
    assert len(notch_by_key) == len(notches), "notch shapes must be unique up to rotation"

    piece_colors = list(legend.keys())
    pieces = _find_pieces(g, bg, template_color, legend_mask, tmask, piece_colors)

    out = g.copy()
    for piece in pieces.values():
        coords = piece["coords"]
        out[coords[:, 0], coords[:, 1]] = bg

    for color, piece in pieces.items():
        notch = notch_by_key.get(piece["key"])
        assert notch is not None, "piece shape must match a template notch"
        _fill_notch(out, notch, legend[int(color)])

    return out


def _pattern_from_template(color: int, template: tuple[int, int, int]) -> tuple[int, int, int]:
    return tuple(color if bit else BG for bit in template)


def _shape_options_for_pair(pair: tuple[int, int]) -> list[tuple[int, int]]:
    a, b = pair
    options: list[tuple[int, int]] = []
    if 2 <= a <= 3 and 2 <= b <= 4:
        options.append((a, b))
    if (a, b) != (b, a) and 2 <= b <= 3 and 2 <= a <= 4:
        options.append((b, a))
    return options


def _carve_notch(template_patch: np.ndarray, side: str, start: int, span: int, depth: int) -> tuple[np.ndarray, dict]:
    patch = template_patch.copy()
    h, w = patch.shape
    if side == "top":
        patch[0:depth, start : start + span] = BG
        bbox = (0, start, depth - 1, start + span - 1)
    elif side == "bottom":
        patch[h - depth : h, start : start + span] = BG
        bbox = (h - depth, start, h - 1, start + span - 1)
    elif side == "left":
        patch[start : start + span, 0:depth] = BG
        bbox = (start, 0, start + span - 1, depth - 1)
    else:
        patch[start : start + span, w - depth : w] = BG
        bbox = (start, w - depth, start + span - 1, w - 1)
    mask = np.ones((bbox[2] - bbox[0] + 1, bbox[3] - bbox[1] + 1), dtype=bool)
    return patch, {"side": side, "bbox": bbox, "mask": mask, "key": _canonical_rotation_key(mask)}


def _reserve_box(occupied: np.ndarray, box: tuple[int, int, int, int], pad: int = 1) -> None:
    h, w = occupied.shape
    r0, c0, r1, c1 = box
    occupied[max(0, r0 - pad) : min(h, r1 + pad + 1), max(0, c0 - pad) : min(w, c1 + pad + 1)] = True


def _can_place_box(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> bool:
    gh, gw = occupied.shape
    if r0 < 0 or c0 < 0 or r0 + h > gh or c0 + w > gw:
        return False
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(gh, r0 + h + pad)
    cc1 = min(gw, c0 + w + pad)
    return not bool(occupied[rr0:rr1, cc0:cc1].any())


def _sample_case() -> tuple[np.ndarray, np.ndarray] | None:
    n_notches = random.randint(2, 4)
    sides = random.sample(SIDES, n_notches)
    shape_pairs = random.sample(UNIQUE_SHAPES, n_notches)
    template_h = random.randint(10, 16)
    template_w = random.randint(10, 14)
    grid_h = random.randint(max(18, template_h + 4), 28)
    grid_w = random.randint(max(24, template_w + 10), 32)
    template_r = random.randint(1, grid_h - template_h - 2)
    template_c = random.randint(1, 3)

    legend_r = grid_h - n_notches
    legend_c = grid_w - 4
    min_piece_c = template_c + template_w + 2
    if min_piece_c >= legend_c - 1:
        return None

    template_patch = np.full((template_h, template_w), TEMPLATE_COLOR, dtype=int)
    notch_specs: list[dict] = []
    for side, pair in zip(sides, shape_pairs, strict=False):
        depth, span = random.choice(_shape_options_for_pair(pair))
        if side in {"top", "bottom"}:
            limit = template_w - span - 1
        else:
            limit = template_h - span - 1
        if limit < 1:
            return None
        start = random.randint(1, limit)
        template_patch, notch_info = _carve_notch(template_patch, side, start, span, depth)
        notch_specs.append(notch_info)

    colors = random.sample(PIECE_COLORS, n_notches)
    pattern_templates = random.sample(PATTERN_TEMPLATES, n_notches)
    for notch, color, pattern_template in zip(notch_specs, colors, pattern_templates, strict=False):
        notch["color"] = int(color)
        notch["pattern"] = _pattern_from_template(int(color), pattern_template)

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    grid[template_r : template_r + template_h, template_c : template_c + template_w] = template_patch

    legend_order = list(range(n_notches))
    random.shuffle(legend_order)
    for row_offset, idx in enumerate(legend_order):
        color = int(notch_specs[idx]["color"])
        pattern = notch_specs[idx]["pattern"]
        r = legend_r + row_offset
        grid[r, legend_c] = color
        grid[r, legend_c + 1 : legend_c + 4] = np.array(pattern, dtype=int)

    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    _reserve_box(occupied, (template_r, template_c, template_r + template_h - 1, template_c + template_w - 1))
    _reserve_box(occupied, (legend_r, legend_c, grid_h - 1, grid_w - 1))

    placed_pieces: list[dict] = []
    for notch in notch_specs:
        base_mask = notch["mask"].copy()
        piece_mask = np.rot90(base_mask, random.randrange(4))
        ph, pw = map(int, piece_mask.shape)
        max_piece_c = legend_c - pw - 1
        if min_piece_c > max_piece_c:
            return None
        placed = False
        for _ in range(200):
            r0 = random.randint(1, grid_h - ph - 2)
            c0 = random.randint(min_piece_c, max_piece_c)
            if not _can_place_box(occupied, r0, c0, ph, pw):
                continue
            patch = np.full((ph, pw), BG, dtype=int)
            patch[piece_mask] = int(notch["color"])
            grid[r0 : r0 + ph, c0 : c0 + pw] = patch
            _reserve_box(occupied, (r0, c0, r0 + ph - 1, c0 + pw - 1))
            placed_pieces.append(
                {
                    "coords": np.argwhere(piece_mask) + np.array([r0, c0], dtype=int),
                    "color": int(notch["color"]),
                }
            )
            placed = True
            break
        if not placed:
            return None

    values, counts = np.unique(grid, return_counts=True)
    bg_count = int(counts[np.where(values == BG)[0][0]])
    if bg_count != int(counts.max()) or int(np.sum(counts == counts.max())) != 1:
        return None

    expected = grid.copy()
    for piece in placed_pieces:
        coords = piece["coords"]
        expected[coords[:, 0], coords[:, 1]] = BG
    for notch in notch_specs:
        abs_notch = {
            "side": notch["side"],
            "bbox": (
                template_r + notch["bbox"][0],
                template_c + notch["bbox"][1],
                template_r + notch["bbox"][2],
                template_c + notch["bbox"][3],
            ),
            "mask": notch["mask"],
        }
        _fill_notch(expected, abs_notch, notch["pattern"])

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    """Randomly generate a valid puzzle instance for this rule family."""
    for _ in range(MAX_GEN_ATTEMPTS):
        case = _sample_case()
        if case is not None:
            grid, _expected = case
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
