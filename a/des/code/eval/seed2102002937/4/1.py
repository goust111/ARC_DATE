from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
GEN_RETRIES = 300
FRAME_COUNT = 4
FRAME_GAP = 1
TEMPLATE_GAP = 1


RIGHT_OPEN = frozenset({(0, 0), (1, 0), (1, 1)})  # X. / XX
LEFT_OPEN = frozenset({(0, 0), (0, 1), (1, 1)})   # XX / .X
UP_OPEN = frozenset({(0, 1), (1, 0), (1, 1)})     # .X / XX
DOWN_OPEN = frozenset({(0, 0), (0, 1), (1, 0)})   # XX / X.

OP_TO_ARROW = {
    "identity": RIGHT_OPEN,
    "flip_h": LEFT_OPEN,
    "flip_v": UP_OPEN,
    "flip_hv": DOWN_OPEN,
}
ARROW_TO_OP = {shape: op for op, shape in OP_TO_ARROW.items()}
ALL_OPS = ("identity", "flip_h", "flip_v", "flip_hv")


def _bg_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _touches_border(cells: Iterable[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in cells)


def _monochrome_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((int(rr), int(cc)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            coords = np.array(cells, dtype=int)
            r0, c0, r1, c1 = _bbox(coords)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "mask": mask,
                    "size": int(coords.shape[0]),
                    "shape": (int(mask.shape[0]), int(mask.shape[1])),
                }
            )
    return comps


def _is_frame_component(comp: dict) -> bool:
    h, w = comp["shape"]
    if h < 4 or w < 4:
        return False
    if comp["size"] != 2 * h + 2 * w - 4:
        return False
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return np.array_equal(comp["mask"], border)


def _arrow_op(comp: dict, grid_shape: tuple[int, int]) -> str | None:
    gh, gw = grid_shape
    if comp["size"] != 3 or comp["shape"] != (2, 2):
        return None
    cells = [tuple(map(int, xy)) for xy in comp["coords"]]
    if not _touches_border(cells, gh, gw):
        return None
    signature = frozenset((int(r), int(c)) for r, c in np.argwhere(comp["mask"]))
    return ARROW_TO_OP.get(signature)


def _transform_mask(mask: np.ndarray, op: str) -> np.ndarray:
    out = mask
    if "v" in op:
        out = np.flipud(out)
    if "h" in op:
        out = np.fliplr(out)
    return out.copy()


def _extract_scene(grid: np.ndarray) -> tuple[int, list[dict], dict[int, str], dict[int, np.ndarray]]:
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    comps = _monochrome_components(g, bg)

    frame_comps: list[dict] = []
    arrows: dict[int, str] = {}
    leftovers: list[dict] = []
    for comp in comps:
        if _is_frame_component(comp):
            frame_comps.append(comp)
            continue
        op = _arrow_op(comp, g.shape)
        if op is not None:
            color = int(comp["color"])
            assert color not in arrows, "each frame color must have exactly one legend arrow"
            arrows[color] = op
            continue
        leftovers.append(comp)

    frames: list[dict] = []
    for comp in frame_comps:
        r0, c0, r1, c1 = comp["bbox"]
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        seed_pts = np.argwhere(inner != bg)
        assert seed_pts.shape[0] == 1, "each frame interior must contain exactly one seed pixel"
        sr, sc = map(int, seed_pts[0])
        seed_r = int(r0 + 1 + sr)
        seed_c = int(c0 + 1 + sc)
        seed_color = int(inner[sr, sc])
        frames.append(
            {
                "border_color": int(comp["color"]),
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "inner_bbox": (int(r0 + 1), int(c0 + 1), int(r1 - 1), int(c1 - 1)),
                "seed": (seed_r, seed_c),
                "seed_color": seed_color,
            }
        )

    templates: dict[int, np.ndarray] = {}
    for comp in leftovers:
        if comp["size"] <= 1:
            continue
        r0, c0, r1, c1 = comp["bbox"]
        inside_frame = any(
            fr["inner_bbox"][0] <= r0
            and r1 <= fr["inner_bbox"][2]
            and fr["inner_bbox"][1] <= c0
            and c1 <= fr["inner_bbox"][3]
            for fr in frames
        )
        if inside_frame:
            continue
        h, w = comp["shape"]
        assert 3 <= h <= 5 and 3 <= w <= 5, "template bbox must be between 3x3 and 5x5"
        color = int(comp["color"])
        assert color not in templates, "each seed/template color must have exactly one template"
        templates[color] = comp["mask"].copy()

    return bg, frames, arrows, templates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow a seed-matched template inside each frame and flip it according to the frame-color legend.
    """
    g = np.asarray(grid, dtype=int)
    _bg, frames, arrows, templates = _extract_scene(g)
    out = g.copy()

    for frame in frames:
        border_color = int(frame["border_color"])
        seed_color = int(frame["seed_color"])
        assert border_color in arrows, "missing legend arrow for a frame color"
        assert seed_color in templates, "missing template for a seed color"

        mask = _transform_mask(templates[seed_color], arrows[border_color])
        ir0, ic0, ir1, ic1 = frame["inner_bbox"]
        seed_r, seed_c = frame["seed"]

        for dr, dc in np.argwhere(mask):
            rr = int(seed_r + dr)
            cc = int(seed_c + dc)
            if ir0 <= rr <= ir1 and ic0 <= cc <= ic1:
                out[rr, cc] = seed_color

    return out


def _pick_spaced_positions(rng: random.Random, limit: int, count: int, min_gap: int) -> list[int] | None:
    candidates = list(range(limit))
    rng.shuffle(candidates)
    picked: list[int] = []
    for x in candidates:
        if all(abs(x - y) >= min_gap for y in picked):
            picked.append(int(x))
            if len(picked) == count:
                return sorted(picked)
    return None


def _reserve(mask: np.ndarray, bbox: tuple[int, int, int, int], pad: int = FRAME_GAP) -> None:
    h, w = mask.shape
    r0, c0, r1, c1 = bbox
    pr0 = max(0, r0 - pad)
    pc0 = max(0, c0 - pad)
    pr1 = min(h - 1, r1 + pad)
    pc1 = min(w - 1, c1 + pad)
    mask[pr0 : pr1 + 1, pc0 : pc1 + 1] = True


def _bbox_is_free(mask: np.ndarray, bbox: tuple[int, int, int, int], pad: int = FRAME_GAP) -> bool:
    h, w = mask.shape
    r0, c0, r1, c1 = bbox
    if r0 < 0 or c0 < 0 or r1 >= h or c1 >= w:
        return False
    pr0 = max(0, r0 - pad)
    pc0 = max(0, c0 - pad)
    pr1 = min(h - 1, r1 + pad)
    pc1 = min(w - 1, c1 + pad)
    return not bool(mask[pr0 : pr1 + 1, pc0 : pc1 + 1].any())


def _make_template_mask(rng: random.Random) -> np.ndarray | None:
    for _ in range(200):
        h = rng.randint(3, 5)
        w = rng.randint(3, 5)

        path = [(0, 0)]
        r = c = 0
        steps = ["D"] * (h - 1) + ["R"] * (w - 1)
        rng.shuffle(steps)
        for step in steps:
            if step == "D":
                r += 1
            else:
                c += 1
            path.append((r, c))

        cells = set(path)
        min_area = len(cells)
        extra_max = max(2, (h * w) // 3)
        target = min(h * w - 2, min_area + rng.randint(1, extra_max))
        while len(cells) < target:
            frontier = set()
            for rr, cc in cells:
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.add((int(nr), int(nc)))
            if not frontier:
                break
            cells.add(rng.choice(sorted(frontier)))

        mask = np.zeros((h, w), dtype=bool)
        for rr, cc in cells:
            mask[rr, cc] = True

        if not mask[0, 0]:
            continue
        if not mask[-1, :].any() or not mask[:, -1].any():
            continue
        if np.array_equal(mask, np.fliplr(mask)):
            continue
        if np.array_equal(mask, np.flipud(mask)):
            continue
        if np.array_equal(mask, np.flipud(np.fliplr(mask))):
            continue
        return mask
    return None


def _place_templates(
    grid: np.ndarray,
    occ: np.ndarray,
    rng: random.Random,
    template_masks: dict[int, np.ndarray],
) -> list[tuple[int, tuple[int, int, int, int]]]:
    h, w = grid.shape
    colors = list(template_masks)
    rng.shuffle(colors)
    top_left = colors[:2]
    bottom_left = colors[2:]

    col0_w = max(template_masks[top_left[0]].shape[1], template_masks[bottom_left[0]].shape[1])
    col1_w = max(template_masks[top_left[1]].shape[1], template_masks[bottom_left[1]].shape[1])
    row0_h = max(template_masks[top_left[0]].shape[0], template_masks[top_left[1]].shape[0])
    row1_h = max(template_masks[bottom_left[0]].shape[0], template_masks[bottom_left[1]].shape[0])
    gap_r = rng.randint(1, 2)
    gap_c = rng.randint(1, 2)

    zone_h = row0_h + gap_r + row1_h
    zone_w = col0_w + gap_c + col1_w
    corner = rng.choice(("bl", "br"))
    zone_r0 = h - zone_h - 1
    zone_c0 = 1 if corner == "bl" else w - zone_w - 1
    if zone_r0 <= 3:
        raise AssertionError("template zone too tall for the canvas")

    cells = {
        top_left[0]: (0, 0, row0_h, col0_w),
        top_left[1]: (0, col0_w + gap_c, row0_h, col1_w),
        bottom_left[0]: (row0_h + gap_r, 0, row1_h, col0_w),
        bottom_left[1]: (row0_h + gap_r, col0_w + gap_c, row1_h, col1_w),
    }

    placed: list[tuple[int, tuple[int, int, int, int]]] = []
    zone_bbox = (zone_r0, zone_c0, zone_r0 + zone_h - 1, zone_c0 + zone_w - 1)
    _reserve(occ, zone_bbox, pad=1)

    for color, (off_r, off_c, slot_h, slot_w) in cells.items():
        mask = template_masks[color]
        th, tw = mask.shape
        jitter_r = 0 if slot_h == th else rng.randint(0, slot_h - th)
        jitter_c = 0 if slot_w == tw else rng.randint(0, slot_w - tw)
        r0 = zone_r0 + off_r + jitter_r
        c0 = zone_c0 + off_c + jitter_c
        for dr, dc in np.argwhere(mask):
            grid[r0 + int(dr), c0 + int(dc)] = int(color)
        placed.append((int(color), (int(r0), int(c0), int(r0 + th - 1), int(c0 + tw - 1))))
    return placed


def _frame_seed_candidates(inner_h: int, inner_w: int, templ_h: int, templ_w: int, clipped: bool) -> list[tuple[int, int]]:
    pts: list[tuple[int, int]] = []
    for r in range(inner_h):
        for c in range(inner_w):
            fully_fits = r + templ_h <= inner_h and c + templ_w <= inner_w
            if clipped and not fully_fits:
                pts.append((int(r), int(c)))
            if not clipped and fully_fits:
                pts.append((int(r), int(c)))
    return pts


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _make_case(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(22, 30)
    w = rng.randint(24, 34)
    g = np.full((h, w), BG, dtype=int)
    occ = np.zeros((h, w), dtype=bool)

    all_colors = rng.sample(list(range(1, 10)), 8)
    frame_colors = all_colors[:FRAME_COUNT]
    seed_colors = all_colors[FRAME_COUNT:]

    ops = list(ALL_OPS)
    rng.shuffle(ops)
    frame_to_op = {frame_colors[i]: ops[i] for i in range(FRAME_COUNT)}

    shuffled_seeds = seed_colors[:]
    rng.shuffle(shuffled_seeds)
    frame_to_seed = {frame_colors[i]: shuffled_seeds[i] for i in range(FRAME_COUNT)}

    template_masks: dict[int, np.ndarray] = {}
    for color in seed_colors:
        mask = _make_template_mask(rng)
        if mask is None:
            return None
        template_masks[color] = mask

    arrow_cols = _pick_spaced_positions(rng, w - 1, FRAME_COUNT, 3)
    if arrow_cols is None:
        return None
    rng.shuffle(frame_colors)
    for start_c, frame_color in zip(arrow_cols, frame_colors):
        shape = OP_TO_ARROW[frame_to_op[frame_color]]
        for dr, dc in shape:
            g[dr, start_c + dc] = frame_color
        _reserve(occ, (0, start_c, 1, start_c + 1), pad=1)

    _place_templates(g, occ, rng, template_masks)

    clip_flags = [True, True, False, False]
    rng.shuffle(clip_flags)
    specs: list[dict] = []
    for frame_color, clipped in zip(frame_colors, clip_flags):
        seed_color = frame_to_seed[frame_color]
        th, tw = template_masks[seed_color].shape
        if clipped:
            inner_h = rng.randint(max(3, th - 1), th + 1)
            inner_w = rng.randint(max(3, tw - 1), tw + 1)
            if inner_h >= th and inner_w >= tw:
                if rng.random() < 0.5:
                    inner_h = max(3, th - 1)
                else:
                    inner_w = max(3, tw - 1)
        else:
            inner_h = rng.randint(th, min(8, th + 3))
            inner_w = rng.randint(tw, min(8, tw + 3))
        seed_candidates = _frame_seed_candidates(inner_h, inner_w, th, tw, clipped)
        if not seed_candidates:
            return None
        specs.append(
            {
                "frame_color": int(frame_color),
                "seed_color": int(seed_color),
                "inner_h": int(inner_h),
                "inner_w": int(inner_w),
                "outer_h": int(inner_h + 2),
                "outer_w": int(inner_w + 2),
                "seed_candidates": seed_candidates,
            }
        )

    specs.sort(key=lambda item: item["outer_h"] * item["outer_w"], reverse=True)
    for spec in specs:
        placed = False
        outer_h = int(spec["outer_h"])
        outer_w = int(spec["outer_w"])
        for _ in range(300):
            r0 = rng.randint(3, h - outer_h - 1)
            c0 = rng.randint(1, w - outer_w - 1)
            bbox = (int(r0), int(c0), int(r0 + outer_h - 1), int(c0 + outer_w - 1))
            if not _bbox_is_free(occ, bbox, pad=1):
                continue
            _draw_frame(g, bbox, int(spec["frame_color"]))
            seed_r_local, seed_c_local = rng.choice(spec["seed_candidates"])
            g[r0 + 1 + seed_r_local, c0 + 1 + seed_c_local] = int(spec["seed_color"])
            _reserve(occ, bbox, pad=1)
            placed = True
            break
        if not placed:
            return None

    bg, frames, arrows, templates = _extract_scene(g)
    assert bg == BG
    assert len(frames) == FRAME_COUNT
    assert len(arrows) == FRAME_COUNT
    assert len(templates) == FRAME_COUNT
    assert set(arrows) == set(frame_to_op)
    assert set(templates) == set(template_masks)

    out = main(g)
    if not np.any(out != g):
        return None
    return g


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        try:
            grid = _make_case(rng)
        except Exception:
            grid = None
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a valid sample")
