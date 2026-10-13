from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BG = 0
FRAGMENT_COLORS = (3, 5, 7)
RECT_COLORS = (1, 2, 4, 6, 8, 9)
ARC_MAX = 30
MAX_GEN_ATTEMPTS = 500


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def mask_from_coords(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = bbox_coords(coords)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel = coords - np.array([r0, c0])
    out[rel[:, 0], rel[:, 1]] = True
    return out


def all_orientations(mask: np.ndarray) -> list[np.ndarray]:
    seen: set[tuple[tuple[int, int], bytes]] = set()
    out: list[np.ndarray] = []
    m0 = mask.astype(bool)
    for flip in (False, True):
        m1 = np.fliplr(m0) if flip else m0
        for k in range(4):
            m2 = np.rot90(m1, k)
            key = (m2.shape, m2.tobytes())
            if key not in seen:
                seen.add(key)
                out.append(m2.copy())
    return out


def paste_mask(dst: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    rows, cols = np.nonzero(mask)
    dst[rows + int(r0), cols + int(c0)] = int(color)


def extract_rectangles(grid: np.ndarray) -> list[dict]:
    rects = []
    for color in sorted(set(map(int, np.unique(grid)))):
        if color == BG or color in FRAGMENT_COLORS:
            continue
        coords = np.argwhere(grid == color)
        if coords.size == 0:
            continue
        rects.append(
            {
                "color": int(color),
                "coords": coords,
                "bbox": bbox_coords(coords),
            }
        )
    rects.sort(key=lambda obj: (obj["bbox"][0], obj["bbox"][1], obj["color"]))
    return rects


def extract_fragments(grid: np.ndarray) -> list[dict]:
    frags = []
    idx = 0
    for color in FRAGMENT_COLORS:
        for comp in components_mask(grid == color):
            bbox = bbox_coords(comp)
            frags.append(
                {
                    "id": idx,
                    "color": int(color),
                    "coords": comp,
                    "bbox": bbox,
                    "mask": mask_from_coords(comp),
                }
            )
            idx += 1
    frags.sort(key=lambda obj: (obj["bbox"][0], obj["bbox"][1], obj["color"], obj["id"]))
    return frags


def lr_symmetric(grid: np.ndarray) -> bool:
    return bool(np.array_equal(grid, grid[:, ::-1]))


def _touches_same_color(fill_coords: list[tuple[int, int]], prior: set[tuple[int, int]]) -> bool:
    if not prior:
        return False
    prior_set = set(prior)
    for r, c in fill_coords:
        for nr, nc in neighbors4(r, c):
            if (nr, nc) in prior_set:
                return True
    return False


def _rect_hidden_window(rect: dict, rect_colors: set[int], grid: np.ndarray) -> tuple[int, int, int, int] | None:
    r0, c0, r1, c1 = rect["bbox"]
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    mask = (sub != BG) & (sub != rect["color"]) & np.isin(sub, list(rect_colors))
    if not mask.any():
        return None
    coords = np.argwhere(mask)
    wr0, wc0, wr1, wc1 = bbox_coords(coords)
    return (r0 + wr0, c0 + wc0, r0 + wr1, c0 + wc1)


def _window_fill_coords(mask: np.ndarray, r0: int, c0: int) -> list[tuple[int, int]]:
    rows, cols = np.nonzero(mask)
    return [(int(r0 + rr), int(c0 + cc)) for rr, cc in zip(rows.tolist(), cols.tolist())]


def _simulate_candidate(
    out: np.ndarray,
    rect: dict,
    frag: dict,
    ori: np.ndarray,
    window: tuple[int, int, int, int],
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    wr0, wc0, wr1, wc1 = window
    trial = out.copy()
    trial[wr0 : wr1 + 1, wc0 : wc1 + 1] = int(rect["color"])
    paste_mask(trial, ori, wr0, wc0, int(frag["color"]))
    fill_coords = _window_fill_coords(ori, wr0, wc0)
    return trial, fill_coords


def _solve(grid: np.ndarray, return_trace: bool = False):
    g = np.array(grid, dtype=int)
    rects = extract_rectangles(g)
    frags = extract_fragments(g)
    rect_colors = {rect["color"] for rect in rects}

    out = g.copy()
    used_fragments: set[int] = set()
    filled_by_color: dict[int, set[tuple[int, int]]] = defaultdict(set)
    trace: list[dict] = []

    hidden = []
    for rect in rects:
        window = _rect_hidden_window(rect, rect_colors, g)
        if window is not None:
            hidden.append((rect, window))
    hidden.sort(key=lambda item: (item[0]["bbox"][0], item[0]["bbox"][1], item[0]["color"]))

    for rect, window in hidden:
        wr0, wc0, wr1, wc1 = window
        wh, ww = wr1 - wr0 + 1, wc1 - wc0 + 1
        raw_candidates = []
        symmetry_candidates = []
        final_candidates = []

        for frag in frags:
            if frag["id"] in used_fragments:
                continue
            for ori in all_orientations(frag["mask"]):
                if ori.shape != (wh, ww):
                    continue
                trial, fill_coords = _simulate_candidate(out, rect, frag, ori, window)
                raw_candidates.append((frag, ori, fill_coords, trial))
                rect_r0, rect_c0, rect_r1, rect_c1 = rect["bbox"]
                rect_crop = trial[rect_r0 : rect_r1 + 1, rect_c0 : rect_c1 + 1]
                if not lr_symmetric(rect_crop):
                    continue
                symmetry_candidates.append((frag, ori, fill_coords, trial))
                if _touches_same_color(fill_coords, filled_by_color[int(frag["color"])]):
                    continue
                final_candidates.append((frag, ori, fill_coords, trial))

        trace.append(
            {
                "rect_color": int(rect["color"]),
                "rect_bbox": rect["bbox"],
                "window_bbox": window,
                "raw_match_count": len(raw_candidates),
                "symmetry_match_count": len(symmetry_candidates),
                "final_match_count": len(final_candidates),
                "final_colors": [int(item[0]["color"]) for item in final_candidates],
            }
        )

        if len(final_candidates) != 1:
            raise ValueError("puzzle instance is not uniquely solvable under the stated rules")

        frag, ori, fill_coords, trial = final_candidates[0]
        out = trial
        used_fragments.add(int(frag["id"]))
        fr0, fc0, fr1, fc1 = frag["bbox"]
        out[fr0 : fr1 + 1, fc0 : fc1 + 1][frag["mask"]] = BG
        filled_by_color[int(frag["color"])].update(fill_coords)

    if return_trace:
        return out, trace
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repaint each occluded rectangle's overlap window by fitting one outside fragment and erasing the used fragment.

    Input structure:
    - Background is black.
    - Central colors form 2 or 3 solid rectangles; some later rectangles cover a centered rectangular window of an earlier rectangle.
    - Edge colors 3/5/7 form small 4-connected fragments scattered well away from the rectangle cluster.

    Rule:
    - For every rectangle whose bbox contains another rectangle's color, treat the covered sub-bbox as that rectangle's missing window.
    - Search unused edge fragments whose shape can be rotated or mirrored so its oriented bbox exactly matches that window size.
    - Trial-place each candidate by restoring the whole window to the hidden rectangle's own color, then painting the fragment color only on the candidate shape cells inside the window.
    - Keep the unique candidate that makes the completed rectangle left-right mirror symmetric; if several still remain, reject candidates whose new fill would 4-touch an earlier fill of the same fragment color.
    - Erase each used fragment from the edge and leave all other pixels unchanged.

    Key insight:
    - The overlap window comes from rectangle occlusion, but the replacement pattern is imported from an outside fragment; mirror symmetry of the completed rectangle is the decisive constraint.
    """
    return _solve(np.asarray(grid, dtype=int), return_trace=False)


SHAPES = {
    "bar2": np.array([[1, 1]], dtype=bool),
    "bar3": np.array([[1, 1, 1]], dtype=bool),
    "bar5": np.array([[1, 1, 1, 1, 1]], dtype=bool),
    "square2": np.array([[1, 1], [1, 1]], dtype=bool),
    "l3": np.array([[1, 0], [1, 1]], dtype=bool),
    "cross3": np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    "l5": np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]], dtype=bool),
}

SPECS = {
    "bar2": {"chosen": "bar2", "distractors": [], "ambiguous": False},
    "bar3": {"chosen": "bar3", "distractors": [], "ambiguous": False},
    "bar5": {"chosen": "bar5", "distractors": [], "ambiguous": False},
    "square2": {"chosen": "square2", "distractors": ["l3"], "ambiguous": True},
    "cross3": {"chosen": "cross3", "distractors": ["l5"], "ambiguous": True},
}


def _sample_hidden_size(window_shape: tuple[int, int]) -> tuple[int, int]:
    wh, ww = window_shape
    width = random.randint(max(ww + 4, 7), min(ww + 8, 13))
    if width % 2 != ww % 2:
        width += 1
    height = random.randint(max(wh + 3, 5), min(wh + 7, 11))
    return int(height), int(width)


def _place_single_layout(shape: tuple[int, int]):
    wh, ww = shape
    hh, hw = _sample_hidden_size(shape)
    H = random.randint(18, 24)
    W = random.randint(18, 24)

    left = random.randint(3, max(3, W - hw - 4))
    top = random.randint(4, max(4, H - hh - 5))
    win_top = random.randint(1, hh - wh - 1)
    win_left = (hw - ww) // 2

    ext_top = random.randint(1, 2)
    ext_bottom = random.randint(1, 3)
    front_top = top + win_top - ext_top
    front_left = left + win_left
    front_h = wh + ext_top + ext_bottom
    front_w = ww

    if front_top < 1 or front_top + front_h >= H - 1:
        return None

    hidden = (top, left, top + hh - 1, left + hw - 1)
    front = (front_top, front_left, front_top + front_h - 1, front_left + front_w - 1)
    return (H, W), [hidden], [front], [(top + win_top, left + win_left, top + win_top + wh - 1, left + win_left + ww - 1)]


def _place_double_layout(width: int, top_shape: tuple[int, int], bot_shape: tuple[int, int]):
    top_hole_h, _ = top_shape
    bot_hole_h, _ = bot_shape
    top_h, rect_w = _sample_hidden_size((top_hole_h, width))
    bot_h, _ = _sample_hidden_size((bot_hole_h, width))
    rect_w = max(rect_w, width + 4)
    if rect_w % 2 != width % 2:
        rect_w += 1

    H = random.randint(20, 26)
    W = random.randint(18, 24)
    left = random.randint(4, max(4, W - rect_w - 5))
    top1 = random.randint(2, 4)
    gap = random.randint(2, 4)
    top2 = top1 + top_h + gap
    if top2 + bot_h + 2 >= H:
        return None

    top_rect = (top1, left, top1 + top_h - 1, left + rect_w - 1)
    bot_rect = (top2, left, top2 + bot_h - 1, left + rect_w - 1)

    win_left = left + (rect_w - width) // 2
    top_window = (top_rect[2] - top_hole_h + 1, win_left, top_rect[2], win_left + width - 1)
    bot_window = (bot_rect[0], win_left, bot_rect[0] + bot_hole_h - 1, win_left + width - 1)

    front_top = top_window[0]
    front_bottom = bot_window[2]
    front = (front_top, win_left, front_bottom, win_left + width - 1)

    return (H, W), [top_rect, bot_rect], [front], [top_window, bot_window]


def _paint_rect(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)


def _expanded_bbox_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int], pad: int) -> np.ndarray:
    H, W = shape
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(H - 1, r1 + pad)
    cc1 = min(W - 1, c1 + pad)
    mask = np.zeros((H, W), dtype=bool)
    mask[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
    return mask


def _patch_from_mask(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = int(color)
    return patch


def _can_place_fragment(
    grid: np.ndarray,
    patch: np.ndarray,
    r0: int,
    c0: int,
    forbidden: np.ndarray,
    edge_band: int,
) -> bool:
    H, W = grid.shape
    h, w = patch.shape
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False
    mask = patch != BG
    if not mask.any():
        return False
    rows, cols = np.nonzero(mask)
    abs_rows = rows + r0
    abs_cols = cols + c0
    if forbidden[abs_rows, abs_cols].any():
        return False
    if (grid[abs_rows, abs_cols] != BG).any():
        return False
    if not (
        r0 <= edge_band
        or c0 <= edge_band
        or r0 + h - 1 >= H - 1 - edge_band
        or c0 + w - 1 >= W - 1 - edge_band
    ):
        return False
    for rr, cc in zip(abs_rows.tolist(), abs_cols.tolist()):
        for nr in range(rr - 1, rr + 2):
            for nc in range(cc - 1, cc + 2):
                if 0 <= nr < H and 0 <= nc < W and grid[nr, nc] != BG and (nr, nc) not in set(zip(abs_rows.tolist(), abs_cols.tolist())):
                    return False
    return True


def _place_fragment_edge(grid: np.ndarray, patch: np.ndarray, forbidden: np.ndarray) -> tuple[int, int] | None:
    H, W = grid.shape
    h, w = patch.shape
    candidates = []
    for r0 in range(H - h + 1):
        for c0 in range(W - w + 1):
            if _can_place_fragment(grid, patch, r0, c0, forbidden, edge_band=3):
                candidates.append((r0, c0))
    if not candidates:
        return None
    r0, c0 = random.choice(candidates)
    grid[r0 : r0 + h, c0 : c0 + w][patch != BG] = patch[patch != BG]
    return int(r0), int(c0)


def _choose_spec_set():
    if random.random() < 0.5:
        spec_name = random.choices(["square2", "cross3", "bar3", "bar5"], weights=[3, 3, 2, 1], k=1)[0]
        return "single", [spec_name]

    if random.random() < 0.5:
        return "double", ["square2", "bar2"]
    return "double", ["cross3", "bar3"]


def _build_expected(inp: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    return _solve(inp, return_trace=True)


def _semantic_check(inp: np.ndarray, trace: list[dict], ambiguous_expected: bool) -> bool:
    if max(inp.shape) > ARC_MAX:
        return False
    if not trace:
        return False
    saw_ambiguous = False
    for step in trace:
        if step["raw_match_count"] < 1 or step["final_match_count"] != 1:
            return False
        if step["raw_match_count"] > 1:
            saw_ambiguous = True
        if step["symmetry_match_count"] < step["final_match_count"]:
            return False
    if ambiguous_expected and not saw_ambiguous:
        return False

    out = main(inp)
    chosen_counts: dict[int, int] = defaultdict(int)
    for step in trace:
        chosen_counts[int(step["final_colors"][0])] += 1
    changed = (out != inp)
    for color, expected_components in chosen_counts.items():
        comps = components_mask(changed & (out == color))
        if len(comps) != expected_components:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    mode, spec_names = _choose_spec_set()
    ambiguous_expected = any(SPECS[name]["ambiguous"] for name in spec_names)

    if mode == "single":
        chosen_spec = SPECS[spec_names[0]]
        window_shape = SHAPES[chosen_spec["chosen"]].shape
        layout = _place_single_layout(window_shape)
        if layout is None:
            return None
        (H, W), hidden_rects, front_rects, windows = layout
    else:
        top_shape = SHAPES[SPECS[spec_names[0]]["chosen"]].shape
        bot_shape = SHAPES[SPECS[spec_names[1]]["chosen"]].shape
        width = max(top_shape[1], bot_shape[1])
        if top_shape[1] != width or bot_shape[1] != width:
            return None
        layout = _place_double_layout(width, top_shape, bot_shape)
        if layout is None:
            return None
        (H, W), hidden_rects, front_rects, windows = layout

    if max(H, W) > ARC_MAX:
        return None

    grid = np.zeros((H, W), dtype=int)
    central_colors = random.sample(list(RECT_COLORS), len(hidden_rects) + len(front_rects))

    hidden_specs = []
    for i, rect in enumerate(hidden_rects):
        hidden_specs.append(
            {
                "bbox": rect,
                "color": int(central_colors[i]),
                "window": windows[i],
                "spec": spec_names[i],
            }
        )
        _paint_rect(grid, rect, int(central_colors[i]))

    for i, rect in enumerate(front_rects):
        _paint_rect(grid, rect, int(central_colors[len(hidden_rects) + i]))

    cluster_bbox = (
        min(b[0] for b in [item["bbox"] for item in hidden_specs] + front_rects),
        min(b[1] for b in [item["bbox"] for item in hidden_specs] + front_rects),
        max(b[2] for b in [item["bbox"] for item in hidden_specs] + front_rects),
        max(b[3] for b in [item["bbox"] for item in hidden_specs] + front_rects),
    )
    forbidden = _expanded_bbox_mask(grid.shape, cluster_bbox, pad=1)

    fragment_pool = []
    for item in hidden_specs:
        spec = SPECS[item["spec"]]
        chosen_color = random.choice(FRAGMENT_COLORS)
        fragment_pool.append(
            {
                "role": "chosen",
                "mask": SHAPES[spec["chosen"]],
                "color": int(chosen_color),
                "for_rect": item["bbox"],
            }
        )
        if spec["ambiguous"]:
            for distractor_name in spec["distractors"]:
                distractor_color = random.choice(FRAGMENT_COLORS)
                fragment_pool.append(
                    {
                        "role": "distractor",
                        "mask": SHAPES[distractor_name],
                        "color": int(distractor_color),
                        "for_rect": item["bbox"],
                    }
                )

    if len(fragment_pool) < 2:
        extra_mask = random.choice([SHAPES["l3"], SHAPES["bar2"]])
        fragment_pool.append({"role": "extra", "mask": extra_mask, "color": int(random.choice(FRAGMENT_COLORS)), "for_rect": None})

    if len(fragment_pool) > 4:
        fragment_pool = fragment_pool[:4]

    random.shuffle(fragment_pool)
    placed_fragments = []
    for frag in fragment_pool:
        patch = _patch_from_mask(frag["mask"], frag["color"])
        loc = _place_fragment_edge(grid, patch, forbidden)
        if loc is None:
            return None
        r0, c0 = loc
        placed_fragments.append((frag, (r0, c0, r0 + patch.shape[0] - 1, c0 + patch.shape[1] - 1)))

    if len(extract_fragments(grid)) != len(placed_fragments):
        return None

    try:
        expected, trace = _build_expected(grid)
    except Exception:
        return None

    if not np.array_equal(expected, main(grid)):
        return None
    if not _semantic_check(grid, trace, ambiguous_expected=ambiguous_expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
