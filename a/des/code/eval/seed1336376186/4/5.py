from __future__ import annotations

from collections import Counter, defaultdict
import itertools
import random

import numpy as np


SIDES = ("top", "bottom", "left", "right")
PLACEMENT_TO_GAP = {"top": "down", "bottom": "up", "left": "right", "right": "left"}
PAIR_OPTIONS = [
    ("top", "bottom"),
    ("top", "left"),
    ("top", "right"),
    ("bottom", "left"),
    ("bottom", "right"),
    ("left", "right"),
]

GEN_BG = 0
GEN_TEMPLATE = 5
GEN_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_TEMPLATE_H = (6, 9)
GEN_TEMPLATE_W = (6, 10)
GEN_COLOR_COUNT = (2, 4)
GEN_OUTER_MARGIN = (1, 2)
GEN_SIDE_GAP = (1, 2)
GEN_COLOR_SPEC_ATTEMPTS = 160
GEN_ATTEMPTS = 400


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
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
        pts: list[tuple[int, int]] = []
        while stack:
            rr, cc = stack.pop()
            pts.append((rr, cc))
            for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(np.array(pts, dtype=int))
    return comps


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[counts.argmax()])


def _find_template(grid: np.ndarray, bg: int) -> tuple[int, tuple[int, int, int, int]]:
    h, w = grid.shape
    candidates: list[tuple[int, int, int, int, int, tuple[int, int, int, int]]] = []
    center_r = (h - 1) / 2.0
    center_c = (w - 1) / 2.0
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        for coords in _components_mask(grid == color):
            r0, c0, r1, c1 = _bbox(coords)
            area = int(coords.shape[0])
            box_area = int((r1 - r0 + 1) * (c1 - c0 + 1))
            if area != box_area:
                continue
            touches_border = int(r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1)
            dist = abs((r0 + r1) / 2.0 - center_r) + abs((c0 + c1) / 2.0 - center_c)
            candidates.append((area, -touches_border, -int(round(dist * 10)), color, len(coords), (r0, c0, r1, c1)))
    assert candidates, "expected a solid template rectangle"
    candidates.sort(reverse=True)
    _area, _tb, _dist, color, _n, bbox = candidates[0]
    return int(color), bbox


def _missing_side(mask: np.ndarray) -> tuple[str, tuple[int, int]]:
    missing = np.argwhere(~mask)
    assert missing.shape[0] == 1, "each object must be a rectangle with exactly one missing cell"
    r, c = map(int, missing[0])
    h, w = mask.shape
    sides = []
    if r == 0:
        sides.append("up")
    if r == h - 1:
        sides.append("down")
    if c == 0:
        sides.append("left")
    if c == w - 1:
        sides.append("right")
    assert len(sides) == 1, "missing cell must lie on exactly one edge"
    return sides[0], (r, c)


def _collect_blocks(
    grid: np.ndarray, bg: int, template_color: int, template_bbox: tuple[int, int, int, int]
) -> dict[int, list[dict]]:
    tr0, tc0, tr1, tc1 = template_bbox
    blocks: dict[int, list[dict]] = defaultdict(list)
    for color in np.unique(grid):
        color = int(color)
        if color in (bg, template_color):
            continue
        for coords in _components_mask(grid == color):
            r0, c0, r1, c1 = _bbox(coords)
            h = int(r1 - r0 + 1)
            w = int(c1 - c0 + 1)
            area = int(coords.shape[0])
            assert area == h * w - 1, "object bbox must be a rectangle with one missing cell"
            rel = coords - np.array([r0, c0], dtype=int)
            mask = np.zeros((h, w), dtype=bool)
            mask[rel[:, 0], rel[:, 1]] = True
            gap_side, missing = _missing_side(mask)
            blocks[color].append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "mask": mask,
                    "gap_side": gap_side,
                    "missing": missing,
                    "template_bbox": (tr0, tc0, tr1, tc1),
                }
            )
    return blocks


def _project_block(block: dict, template_bbox: tuple[int, int, int, int]) -> np.ndarray:
    tr0, tc0, tr1, tc1 = template_bbox
    th = int(tr1 - tr0 + 1)
    tw = int(tc1 - tc0 + 1)
    r0, c0, r1, c1 = block["bbox"]
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    gap_side = block["gap_side"]
    mask = np.asarray(block["mask"], dtype=bool)

    if gap_side == "up":
        dst_r0 = tr0
        dst_c0 = c0
    elif gap_side == "down":
        dst_r0 = tr1 - h + 1
        dst_c0 = c0
    elif gap_side == "left":
        dst_r0 = r0
        dst_c0 = tc0
    else:
        dst_r0 = r0
        dst_c0 = tc1 - w + 1

    out = np.zeros((th, tw), dtype=bool)
    for rr in range(h):
        for cc in range(w):
            if not mask[rr, cc]:
                continue
            ar = int(dst_r0 + rr)
            ac = int(dst_c0 + cc)
            if tr0 <= ar <= tr1 and tc0 <= ac <= tc1:
                out[ar - tr0, ac - tc0] = True
    return out


def _color_order(grid: np.ndarray, bg: int, template_color: int) -> list[int]:
    seen: set[int] = set()
    order: list[int] = []
    for value in grid.flat:
        color = int(value)
        if color in (bg, template_color) or color in seen:
            continue
        seen.add(color)
        order.append(color)
    return order


def main(grid: np.ndarray) -> np.ndarray:
    """
    Gap direction determines translation; same-color translated rectangles merge and stamp the template.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    template_color, template_bbox = _find_template(g, bg)
    tr0, tc0, tr1, tc1 = template_bbox
    th = int(tr1 - tr0 + 1)
    tw = int(tc1 - tc0 + 1)

    blocks = _collect_blocks(g, bg, template_color, template_bbox)
    out = np.full((th, tw), template_color, dtype=int)

    for color in _color_order(g, bg, template_color):
        if color not in blocks:
            continue
        merged = np.zeros((th, tw), dtype=bool)
        for block in blocks[color]:
            merged |= _project_block(block, template_bbox)
        out[merged] = int(color)

    return out


def _random_spec(side: str, th: int, tw: int, color: int) -> dict:
    gap_side = PLACEMENT_TO_GAP[side]
    if side in {"top", "bottom"}:
        height = random.randint(2, th)
        width = random.randint(3, tw)
        col0 = random.randint(0, tw - width)
        gap_index = random.randint(1, width - 2)
        return {
            "color": int(color),
            "side": side,
            "gap_side": gap_side,
            "height": int(height),
            "width": int(width),
            "col0": int(col0),
            "gap_index": int(gap_index),
        }
    height = random.randint(3, th)
    width = random.randint(2, tw)
    row0 = random.randint(0, th - height)
    gap_index = random.randint(1, height - 2)
    return {
        "color": int(color),
        "side": side,
        "gap_side": gap_side,
        "height": int(height),
        "width": int(width),
        "row0": int(row0),
        "gap_index": int(gap_index),
    }


def _spec_template_mask(spec: dict, th: int, tw: int) -> np.ndarray:
    h = int(spec["height"])
    w = int(spec["width"])
    mask = np.ones((h, w), dtype=bool)
    gap_side = spec["gap_side"]
    gap_index = int(spec["gap_index"])
    if gap_side == "up":
        mask[0, gap_index] = False
        dst_r0 = 0
        dst_c0 = int(spec["col0"])
    elif gap_side == "down":
        mask[-1, gap_index] = False
        dst_r0 = th - h
        dst_c0 = int(spec["col0"])
    elif gap_side == "left":
        mask[gap_index, 0] = False
        dst_r0 = int(spec["row0"])
        dst_c0 = 0
    else:
        mask[gap_index, -1] = False
        dst_r0 = int(spec["row0"])
        dst_c0 = tw - w

    out = np.zeros((th, tw), dtype=bool)
    out[dst_r0 : dst_r0 + h, dst_c0 : dst_c0 + w] = mask
    return out


def _sample_pair_specs(
    pair: tuple[str, str], color: int, th: int, tw: int, prev_union: np.ndarray, want_overlap: bool
) -> tuple[list[dict], np.ndarray]:
    best_specs: list[dict] | None = None
    best_union: np.ndarray | None = None
    if want_overlap:
        best_score = -1
    else:
        best_score = 10**9

    for _ in range(GEN_COLOR_SPEC_ATTEMPTS):
        specs = [_random_spec(side, th, tw, color) for side in pair]
        union = np.zeros((th, tw), dtype=bool)
        for spec in specs:
            union |= _spec_template_mask(spec, th, tw)
        score = int(np.sum(union & prev_union))
        if want_overlap:
            if score > 0:
                return specs, union
            if score > best_score:
                best_score = score
                best_specs = specs
                best_union = union
        else:
            if score < best_score:
                best_score = score
                best_specs = specs
                best_union = union
                if score == 0:
                    return specs, union

    assert best_specs is not None and best_union is not None
    return best_specs, best_union


def _assign_positions(specs: list[dict], th: int, tw: int) -> None:
    by_side: dict[str, list[dict]] = defaultdict(list)
    for spec in specs:
        by_side[spec["side"]].append(spec)

    for side in SIDES:
        lane_specs = by_side.get(side, [])
        random.shuffle(lane_specs)
        cursor = random.randint(*GEN_SIDE_GAP)
        for spec in lane_specs:
            h = int(spec["height"])
            w = int(spec["width"])
            if side == "top":
                spec["local_r"] = -cursor - h
                spec["local_c"] = int(spec["col0"])
                cursor += h + random.randint(*GEN_SIDE_GAP)
            elif side == "bottom":
                spec["local_r"] = th + cursor
                spec["local_c"] = int(spec["col0"])
                cursor += h + random.randint(*GEN_SIDE_GAP)
            elif side == "left":
                spec["local_r"] = int(spec["row0"])
                spec["local_c"] = -cursor - w
                cursor += w + random.randint(*GEN_SIDE_GAP)
            else:
                spec["local_r"] = int(spec["row0"])
                spec["local_c"] = tw + cursor
                cursor += w + random.randint(*GEN_SIDE_GAP)


def _render_patch(spec: dict, bg: int, color: int) -> np.ndarray:
    h = int(spec["height"])
    w = int(spec["width"])
    patch = np.full((h, w), int(color), dtype=int)
    gap_index = int(spec["gap_index"])
    gap_side = spec["gap_side"]
    if gap_side == "up":
        patch[0, gap_index] = bg
    elif gap_side == "down":
        patch[-1, gap_index] = bg
    elif gap_side == "left":
        patch[gap_index, 0] = bg
    else:
        patch[gap_index, -1] = bg
    return patch


def _sample_pairs(k: int) -> list[tuple[str, str]]:
    valid = []
    for subset in itertools.combinations(PAIR_OPTIONS, k):
        counts = Counter(side for pair in subset for side in pair)
        if max(counts.values(), default=0) <= 2:
            valid.append(subset)
    assert valid, "expected at least one valid side pairing"
    choice = list(random.choice(valid))
    random.shuffle(choice)
    return choice


def _build_expected(grid: np.ndarray, specs: list[dict], th: int, tw: int, template_color: int) -> np.ndarray:
    out = np.full((th, tw), int(template_color), dtype=int)
    bg = _get_bg(grid)
    order = _color_order(grid, bg, template_color)
    specs_by_color: dict[int, list[dict]] = defaultdict(list)
    for spec in specs:
        specs_by_color[int(spec["color"])].append(spec)
    for color in order:
        merged = np.zeros((th, tw), dtype=bool)
        for spec in specs_by_color.get(int(color), []):
            merged |= _spec_template_mask(spec, th, tw)
        out[merged] = int(color)
    return out


def _attempt_generate() -> np.ndarray | None:
    th = random.randint(*GEN_TEMPLATE_H)
    tw = random.randint(*GEN_TEMPLATE_W)
    color_count = random.randint(*GEN_COLOR_COUNT)
    colors = random.sample(GEN_COLORS, color_count)
    pairs = _sample_pairs(color_count)
    force_overlap = bool(random.getrandbits(1))

    specs: list[dict] = []
    prev_union = np.zeros((th, tw), dtype=bool)
    created_overlap = False
    for idx, (color, pair) in enumerate(zip(colors, pairs, strict=False)):
        want_overlap = force_overlap and idx > 0 and not created_overlap
        pair_specs, union = _sample_pair_specs(pair, color, th, tw, prev_union, want_overlap)
        created_overlap = created_overlap or bool(np.any(union & prev_union))
        specs.extend(pair_specs)
        prev_union |= union

    if force_overlap and color_count > 1 and not created_overlap:
        return None

    _assign_positions(specs, th, tw)

    min_r = min([0] + [int(spec["local_r"]) for spec in specs])
    min_c = min([0] + [int(spec["local_c"]) for spec in specs])
    max_r = max([th - 1] + [int(spec["local_r"]) + int(spec["height"]) - 1 for spec in specs])
    max_c = max([tw - 1] + [int(spec["local_c"]) + int(spec["width"]) - 1 for spec in specs])
    pad_r = random.randint(*GEN_OUTER_MARGIN)
    pad_c = random.randint(*GEN_OUTER_MARGIN)
    h = int(max_r - min_r + 1 + 2 * pad_r)
    w = int(max_c - min_c + 1 + 2 * pad_c)
    off_r = int(pad_r - min_r)
    off_c = int(pad_c - min_c)

    grid = np.full((h, w), GEN_BG, dtype=int)
    tr0 = off_r
    tc0 = off_c
    grid[tr0 : tr0 + th, tc0 : tc0 + tw] = GEN_TEMPLATE

    owner = np.full((h, w), -1, dtype=int)
    owner[tr0 : tr0 + th, tc0 : tc0 + tw] = -2
    for idx, spec in enumerate(specs):
        r0 = int(spec["local_r"]) + off_r
        c0 = int(spec["local_c"]) + off_c
        patch = _render_patch(spec, GEN_BG, int(spec["color"]))
        mask = patch != GEN_BG
        if np.any(owner[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]][mask] != -1):
            return None
        grid[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]][mask] = patch[mask]
        owner[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]][mask] = idx

    if _get_bg(grid) != GEN_BG:
        return None

    for color in colors:
        comps = _components_mask(grid == int(color))
        if len(comps) != 2:
            return None

    template_color, template_bbox = _find_template(grid, GEN_BG)
    if template_color != GEN_TEMPLATE:
        return None
    if template_bbox != (tr0, tc0, tr0 + th - 1, tc0 + tw - 1):
        return None

    expected = _build_expected(grid, specs, th, tw, GEN_TEMPLATE)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for task 5")
