from __future__ import annotations

import random
from collections import Counter, deque
from itertools import permutations

import numpy as np


DIR_ORDER = ["up", "right", "down", "left"]
DIR_TO_INDEX = {name: idx for idx, name in enumerate(DIR_ORDER)}
DIR_VECTORS = {
    "up": (-1, 0),
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
}
CORNER_POINTS = {
    "up": lambda h, w: (0, 0),
    "right": lambda h, w: (0, w - 1),
    "down": lambda h, w: (h - 1, w - 1),
    "left": lambda h, w: (h - 1, 0),
}
PATTERN_LIBRARY = [
    np.array(
        [
            [1, 0, 1, 0],
            [0, 1, 1, 0],
            [1, 0, 0, 0],
        ],
        dtype=int,
    ),
    np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
        ],
        dtype=int,
    ),
    np.array(
        [
            [1, 0, 1, 0, 0],
            [0, 1, 1, 0, 1],
            [1, 0, 0, 1, 0],
            [0, 0, 1, 0, 1],
        ],
        dtype=int,
    ),
    np.array(
        [
            [1, 0, 0, 1],
            [0, 1, 1, 0],
            [1, 0, 1, 0],
            [0, 0, 1, 1],
            [1, 0, 0, 0],
        ],
        dtype=int,
    ),
    np.array(
        [
            [1, 0, 1, 0],
            [0, 1, 0, 0],
            [1, 1, 0, 1],
            [0, 0, 1, 0],
        ],
        dtype=int,
    ),
]
QUADRANT_NAMES = ["tl", "tr", "bl", "br"]


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
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
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(np.min(coords[:, 0])),
        int(np.min(coords[:, 1])),
        int(np.max(coords[:, 0])),
        int(np.max(coords[:, 1])),
    )


def _flood_from_border(passable: np.ndarray) -> np.ndarray:
    h, w = passable.shape
    seen = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if passable[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if passable[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and passable[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
    return seen


def _rotate_patch_to_dir(patch: np.ndarray, native_dir: str, target_dir: str) -> np.ndarray:
    delta = (DIR_TO_INDEX[target_dir] - DIR_TO_INDEX[native_dir]) % 4
    return np.rot90(patch, -delta)


def _assign_corner_dirs(template_infos: list[dict], h: int, w: int) -> None:
    corner_dirs = list(CORNER_POINTS)
    points = {name: CORNER_POINTS[name](h, w) for name in corner_dirs}
    centers = []
    for info in template_infos:
        r0, c0, r1, c1 = info["bbox"]
        centers.append(((r0 + r1) / 2.0, (c0 + c1) / 2.0))

    best_perm = None
    best_score = None
    for perm in permutations(corner_dirs):
        score = 0.0
        for center, name in zip(centers, perm):
            cr, cc = center
            pr, pc = points[name]
            score += abs(cr - pr) + abs(cc - pc)
        if best_score is None or score < best_score:
            best_score = score
            best_perm = perm

    assert best_perm is not None
    for info, native_dir in zip(template_infos, best_perm):
        info["native_dir"] = native_dir


def _extract_templates(grid: np.ndarray, bg: int) -> tuple[list[dict], np.ndarray]:
    comps = _components(grid != bg)
    template_infos: list[dict] = []
    template_mask = np.zeros_like(grid, dtype=bool)

    for comp in comps:
        colors = set(int(grid[r, c]) for r, c in comp)
        if len(colors) <= 1:
            continue
        r0, c0, r1, c1 = _bbox(comp)
        patch = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
        template_infos.append(
            {
                "coords": comp,
                "bbox": (r0, c0, r1, c1),
                "patch": patch,
                "colors": colors,
            }
        )
        template_mask[comp[:, 0], comp[:, 1]] = True

    assert len(template_infos) == 4, "expected exactly four multicolor corner templates"
    _assign_corner_dirs(template_infos, *grid.shape)
    return template_infos, template_mask


def _outline_candidates(mask: np.ndarray) -> list[np.ndarray]:
    return _components(mask)


def _candidate_gap_positions(wall: np.ndarray, sub: np.ndarray, bg: int) -> list[tuple[int, int, str]]:
    h, w = wall.shape
    out: list[tuple[int, int, str]] = []
    for r in range(h):
        for c in range(w):
            if sub[r, c] != bg or wall[r, c]:
                continue
            up = r > 0 and wall[r - 1, c]
            down = r + 1 < h and wall[r + 1, c]
            left = c > 0 and wall[r, c - 1]
            right = c + 1 < w and wall[r, c + 1]
            if left and right and not up and not down:
                out.append((r, c, "vertical"))
            elif up and down and not left and not right:
                out.append((r, c, "horizontal"))
    return out


def _infer_region_from_outline(
    grid: np.ndarray,
    outline_comp: np.ndarray,
    outline_color: int,
    bg: int,
    seed_coords: np.ndarray,
) -> dict:
    r0, c0, r1, c1 = _bbox(outline_comp)
    pr0 = max(0, r0 - 1)
    pc0 = max(0, c0 - 1)
    pr1 = min(grid.shape[0] - 1, r1 + 1)
    pc1 = min(grid.shape[1] - 1, c1 + 1)

    sub = grid[pr0 : pr1 + 1, pc0 : pc1 + 1]
    wall = sub == outline_color
    local_seed = np.array([(int(r - pr0), int(c - pc0)) for r, c in seed_coords], dtype=int)

    valid: list[dict] = []
    for gr, gc, axis in _candidate_gap_positions(wall, sub, bg):
        sealed = wall.copy()
        sealed[gr, gc] = True
        exterior = _flood_from_border(~sealed)
        enclosed = (~sealed) & ~exterior
        if not np.all(enclosed[local_seed[:, 0], local_seed[:, 1]]):
            continue
        if axis == "vertical":
            up_out = gr > 0 and exterior[gr - 1, gc]
            down_out = gr + 1 < sealed.shape[0] and exterior[gr + 1, gc]
            up_in = gr > 0 and enclosed[gr - 1, gc]
            down_in = gr + 1 < sealed.shape[0] and enclosed[gr + 1, gc]
            if up_out and down_in and not down_out and not up_in:
                direction = "up"
            elif down_out and up_in and not up_out and not down_in:
                direction = "down"
            else:
                continue
        else:
            left_out = gc > 0 and exterior[gr, gc - 1]
            right_out = gc + 1 < sealed.shape[1] and exterior[gr, gc + 1]
            left_in = gc > 0 and enclosed[gr, gc - 1]
            right_in = gc + 1 < sealed.shape[1] and enclosed[gr, gc + 1]
            if left_out and right_in and not right_out and not left_in:
                direction = "left"
            elif right_out and left_in and not left_out and not right_in:
                direction = "right"
            else:
                continue

        interior_bg = enclosed & (sub == bg)
        if int(np.sum(interior_bg)) == 0:
            continue
        valid.append(
            {
                "direction": direction,
                "enclosed": enclosed,
                "fill_mask": interior_bg,
                "sub_bbox": (pr0, pc0, pr1, pc1),
            }
        )

    assert len(valid) == 1, "gap direction must be uniquely inferable"
    return valid[0]


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    templates, template_mask = _extract_templates(g, bg)

    remaining_mask = (g != bg) & ~template_mask
    color_counts = Counter(int(v) for v in g[remaining_mask])
    assert color_counts, "expected non-template foreground objects"
    outline_color = color_counts.most_common(1)[0][0]

    outline_comps = _outline_candidates(remaining_mask & (g == outline_color))
    seed_comps = _components(remaining_mask & (g != outline_color))
    assert len(outline_comps) == 4, "expected four outlined regions"
    assert len(seed_comps) == 4, "expected four seed components"

    seed_infos = []
    for comp in seed_comps:
        colors = set(int(g[r, c]) for r, c in comp)
        assert len(colors) == 1
        seed_infos.append({"coords": comp, "color": next(iter(colors)), "bbox": _bbox(comp)})

    seed_colors = {info["color"] for info in seed_infos}
    assert len(seed_colors) == 4, "seed colors must be unique"

    template_by_seed: dict[int, dict] = {}
    for info in templates:
        matches = list(info["colors"] & seed_colors)
        assert len(matches) == 1, "each template must encode exactly one seed color"
        fg = matches[0]
        info["seed_color"] = fg
        template_by_seed[fg] = info
    assert len(template_by_seed) == 4

    region_infos: list[dict] = []
    for outline_comp in outline_comps:
        r0, c0, r1, c1 = _bbox(outline_comp)
        containing = []
        for seed in seed_infos:
            sr0, sc0, sr1, sc1 = seed["bbox"]
            if r0 < sr0 <= sr1 < r1 and c0 < sc0 <= sc1 < c1:
                containing.append(seed)
        assert len(containing) == 1, "each outline must contain exactly one seed"
        seed = containing[0]
        inferred = _infer_region_from_outline(g, outline_comp, outline_color, bg, seed["coords"])
        region_infos.append({"seed": seed, **inferred})

    out = g.copy()
    for region in region_infos:
        seed_color = region["seed"]["color"]
        template = template_by_seed[seed_color]
        rotated = _rotate_patch_to_dir(template["patch"], template["native_dir"], region["direction"])
        pr0, pc0, _pr1, _pc1 = region["sub_bbox"]
        fill_mask = region["fill_mask"]
        enclosed = region["enclosed"]
        coords = np.argwhere(enclosed)
        rr0 = int(np.min(coords[:, 0]))
        cc0 = int(np.min(coords[:, 1]))
        th, tw = rotated.shape

        sub_out = out[pr0 : pr0 + fill_mask.shape[0], pc0 : pc0 + fill_mask.shape[1]]
        fill_coords = np.argwhere(fill_mask)
        for rr, cc in fill_coords:
            sub_out[rr, cc] = int(rotated[(rr - rr0) % th, (cc - cc0) % tw])

    return out


def main(grid: np.ndarray) -> np.ndarray:
    return _solve(grid)


def _boundary(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros_like(mask, dtype=bool)
    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if not (0 <= nr < h and 0 <= nc < w) or not mask[nr, nc]:
                    out[r, c] = True
                    break
    return out


def _is_connected(mask: np.ndarray) -> bool:
    if int(np.sum(mask)) == 0:
        return False
    return len(_components(mask)) == 1


def _interior(mask: np.ndarray) -> np.ndarray:
    return mask & ~_boundary(mask)


def _gap_candidates_for_direction(mask: np.ndarray, direction: str) -> list[tuple[int, int]]:
    h, w = mask.shape
    candidates: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            up = r > 0 and mask[r - 1, c]
            down = r + 1 < h and mask[r + 1, c]
            left = c > 0 and mask[r, c - 1]
            right = c + 1 < w and mask[r, c + 1]
            if direction == "up":
                if (r == 0 or not mask[r - 1, c]) and down and left and right:
                    candidates.append((r, c))
            elif direction == "down":
                if (r == h - 1 or not mask[r + 1, c]) and up and left and right:
                    candidates.append((r, c))
            elif direction == "left":
                if (c == 0 or not mask[r, c - 1]) and right and up and down:
                    candidates.append((r, c))
            else:
                if (c == w - 1 or not mask[r, c + 1]) and left and up and down:
                    candidates.append((r, c))
    return candidates


def _seed_patterns() -> list[list[tuple[int, int]]]:
    return [
        [(0, 0)],
        [(0, 0), (0, 1)],
        [(0, 0), (1, 0)],
        [(0, 0), (0, 1), (0, 2)],
        [(0, 0), (1, 0), (2, 0)],
        [(0, 0), (0, 1), (1, 0)],
        [(0, 0), (0, 1), (1, 1)],
        [(0, 0), (1, 0), (1, 1)],
        [(0, 1), (1, 0), (1, 1)],
    ]


def _make_seed_mask(interior_mask: np.ndarray) -> np.ndarray | None:
    coords = [(int(r), int(c)) for r, c in np.argwhere(interior_mask)]
    patterns = _seed_patterns()
    random.shuffle(coords)
    for anchor_r, anchor_c in coords:
        random.shuffle(patterns)
        for pattern in patterns:
            pts = []
            ok = True
            for dr, dc in pattern:
                nr, nc = anchor_r + dr, anchor_c + dc
                if not (0 <= nr < interior_mask.shape[0] and 0 <= nc < interior_mask.shape[1] and interior_mask[nr, nc]):
                    ok = False
                    break
                pts.append((nr, nc))
            if ok:
                seed = np.zeros_like(interior_mask, dtype=bool)
                for nr, nc in pts:
                    seed[nr, nc] = True
                return seed
    return None


def _make_shape_for_direction(direction: str) -> dict | None:
    for _ in range(300):
        h = random.randint(6, 8)
        w = random.randint(6, 8)
        mask = np.ones((h, w), dtype=bool)

        for _ in range(random.randint(1, 3)):
            side = random.choice(DIR_ORDER)
            if side in ("up", "down"):
                if w < 5 or h < 7:
                    continue
                span_max = max(1, w - 4)
                span = random.randint(1, span_max)
                start = random.randint(1, w - span - 2)
                depth = random.randint(1, min(2, h - 4))
                rows = slice(0, depth) if side == "up" else slice(h - depth, h)
                mask[rows, start : start + span] = False
            else:
                if h < 5 or w < 7:
                    continue
                span_max = max(1, h - 4)
                span = random.randint(1, span_max)
                start = random.randint(1, h - span - 2)
                depth = random.randint(1, min(2, w - 4))
                cols = slice(0, depth) if side == "left" else slice(w - depth, w)
                mask[start : start + span, cols] = False

        if not _is_connected(mask):
            continue

        interior = _interior(mask)
        if int(np.sum(interior)) < 6 or not _is_connected(interior):
            continue

        gaps = _gap_candidates_for_direction(mask, direction)
        if not gaps:
            continue

        seed_space = _interior(interior)
        if int(np.sum(seed_space)) == 0:
            continue

        seed_mask = _make_seed_mask(seed_space)
        if seed_mask is None:
            continue

        gap = random.choice(gaps)
        outline = _boundary(mask)
        outline[gap] = False
        if not _is_connected(outline):
            continue

        return {
            "filled": mask,
            "interior": interior,
            "outline": outline,
            "gap": gap,
            "seed_mask": seed_mask,
            "shape": mask.shape,
            "direction": direction,
        }
    return None


def _instantiate_template(binary: np.ndarray, fg: int, accent: int) -> np.ndarray:
    return np.where(binary == 1, fg, accent).astype(int)


def _paint_region(
    canvas: np.ndarray,
    top: int,
    left: int,
    region: dict,
    outline_color: int,
    seed_color: int,
) -> None:
    outline = region["outline"]
    seed_mask = region["seed_mask"]
    h, w = outline.shape
    sub = canvas[top : top + h, left : left + w]
    sub[outline] = outline_color
    sub[seed_mask] = seed_color


def _apply_expected_fill(
    canvas: np.ndarray,
    top: int,
    left: int,
    region: dict,
    template_patch: np.ndarray,
    native_dir: str,
    target_dir: str,
    bg: int,
) -> None:
    rotated = _rotate_patch_to_dir(template_patch, native_dir, target_dir)
    interior = region["interior"]
    seed_mask = region["seed_mask"]
    fill_mask = interior & ~seed_mask
    coords = np.argwhere(interior)
    r0 = int(np.min(coords[:, 0]))
    c0 = int(np.min(coords[:, 1]))
    th, tw = rotated.shape
    sub = canvas[top : top + interior.shape[0], left : left + interior.shape[1]]
    for rr, cc in np.argwhere(fill_mask):
        assert sub[rr, cc] == bg
        sub[rr, cc] = int(rotated[(rr - r0) % th, (cc - c0) % tw])


def _gen_attempt() -> np.ndarray | None:
    colors = list(range(10))
    bg = random.choice(colors)
    colors.remove(bg)
    outline_color = random.choice(colors)
    colors.remove(outline_color)
    accent_color = random.choice(colors)
    colors.remove(accent_color)
    seed_colors = random.sample(colors, 4)

    pattern_choices = random.sample(PATTERN_LIBRARY, 4)
    native_templates = {}
    for native_dir, seed_color, binary in zip(DIR_ORDER, seed_colors, pattern_choices):
        native_templates[seed_color] = {
            "native_dir": native_dir,
            "patch": _instantiate_template(binary, seed_color, accent_color),
        }

    regions = []
    for direction, seed_color in zip(DIR_ORDER, seed_colors):
        region = _make_shape_for_direction(direction)
        if region is None:
            return None
        region["seed_color"] = seed_color
        regions.append(region)

    quadrant_assignment = dict(zip(random.sample(QUADRANT_NAMES, 4), regions))

    top_strip = max(
        native_templates[seed_colors[0]]["patch"].shape[0],
        native_templates[seed_colors[1]]["patch"].shape[0],
    )
    bottom_strip = max(
        native_templates[seed_colors[2]]["patch"].shape[0],
        native_templates[seed_colors[3]]["patch"].shape[0],
    )
    left_strip = max(
        native_templates[seed_colors[0]]["patch"].shape[1],
        native_templates[seed_colors[3]]["patch"].shape[1],
    )
    right_strip = max(
        native_templates[seed_colors[1]]["patch"].shape[1],
        native_templates[seed_colors[2]]["patch"].shape[1],
    )

    row_heights = {
        "top": max(quadrant_assignment["tl"]["shape"][0], quadrant_assignment["tr"]["shape"][0]) + random.randint(0, 1),
        "bottom": max(quadrant_assignment["bl"]["shape"][0], quadrant_assignment["br"]["shape"][0]) + random.randint(0, 1),
    }
    col_widths = {
        "left": max(quadrant_assignment["tl"]["shape"][1], quadrant_assignment["bl"]["shape"][1]) + random.randint(0, 1),
        "right": max(quadrant_assignment["tr"]["shape"][1], quadrant_assignment["br"]["shape"][1]) + random.randint(0, 1),
    }

    top_pad = random.randint(1, 2)
    bottom_pad = random.randint(1, 2)
    left_pad = random.randint(1, 2)
    right_pad = random.randint(1, 2)
    row_gap = random.randint(1, 2)
    col_gap = random.randint(1, 2)

    h = top_strip + top_pad + row_heights["top"] + row_gap + row_heights["bottom"] + bottom_pad + bottom_strip
    w = left_strip + left_pad + col_widths["left"] + col_gap + col_widths["right"] + right_pad + right_strip
    if h > 34 or w > 34:
        return None
    if h < 20 or w < 20:
        return None

    grid = np.full((h, w), bg, dtype=int)

    corner_layout = {
        "up": (0, 0),
        "right": (0, w),
        "down": (h, w),
        "left": (h, 0),
    }
    for seed_color, info in native_templates.items():
        patch = info["patch"]
        native_dir = info["native_dir"]
        ph, pw = patch.shape
        if native_dir == "up":
            grid[:ph, :pw] = patch
        elif native_dir == "right":
            grid[:ph, w - pw :] = patch
        elif native_dir == "down":
            grid[h - ph :, w - pw :] = patch
        else:
            grid[h - ph :, :pw] = patch

    base_r = top_strip + top_pad
    base_c = left_strip + left_pad
    slot_boxes = {
        "tl": (base_r, base_c, row_heights["top"], col_widths["left"]),
        "tr": (base_r, base_c + col_widths["left"] + col_gap, row_heights["top"], col_widths["right"]),
        "bl": (base_r + row_heights["top"] + row_gap, base_c, row_heights["bottom"], col_widths["left"]),
        "br": (
            base_r + row_heights["top"] + row_gap,
            base_c + col_widths["left"] + col_gap,
            row_heights["bottom"],
            col_widths["right"],
        ),
    }

    placements: list[tuple[dict, int, int]] = []
    for name in QUADRANT_NAMES:
        region = quadrant_assignment[name]
        slot_r, slot_c, slot_h, slot_w = slot_boxes[name]
        rh, rw = region["shape"]
        max_off_r = slot_h - rh
        max_off_c = slot_w - rw
        if max_off_r < 0 or max_off_c < 0:
            return None
        top = slot_r + random.randint(0, max_off_r)
        left = slot_c + random.randint(0, max_off_c)
        placements.append((region, top, left))

    for region, top, left in placements:
        _paint_region(grid, top, left, region, outline_color, region["seed_color"])

    expected = grid.copy()
    for region, top, left in placements:
        template = native_templates[region["seed_color"]]
        _apply_expected_fill(
            expected,
            top,
            left,
            region,
            template["patch"],
            template["native_dir"],
            region["direction"],
            bg,
        )

    try:
        actual = _solve(grid)
    except AssertionError:
        return None

    if not np.array_equal(actual, expected):
        return None

    if np.array_equal(actual, grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
