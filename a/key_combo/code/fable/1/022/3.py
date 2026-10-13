from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


SPIRAL = (
    (0, 0),
    (0, 1),
    (0, 2),
    (1, 2),
    (2, 2),
    (2, 1),
    (2, 0),
    (1, 0),
    (1, 1),
)

MAX_GENERATION_ATTEMPTS = 500
BG_COLOR = 0


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def colors_present(grid: np.ndarray, *, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, np.unique(grid)))
    if exclude is not None:
        colors -= set(map(int, exclude))
    return sorted(colors)


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    dst[r0 : r0 + h, c0 : c0 + w] = patch


def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    if connectivity == 4:
        directions = ((-1, 0), (1, 0), (0, -1), (0, 1))
    else:
        directions = tuple(
            (dr, dc)
            for dr in (-1, 0, 1)
            for dc in (-1, 0, 1)
            if not (dr == 0 and dc == 0)
        )

    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for dr, dc in directions:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _full_rect_rows(interior: np.ndarray, bg: int, fill_color: int) -> int:
    fill_mask = interior == fill_color
    assert np.isin(interior, [bg, fill_color]).all(), "visible tube interior must contain only background and liquid"
    row_all_fill = fill_mask.all(axis=1)
    row_all_bg = (interior == bg).all(axis=1)
    assert np.all(row_all_fill | row_all_bg), "liquid must form full-width rows"
    if not row_all_fill.any():
        return 0
    first_filled = int(np.argmax(row_all_fill))
    assert np.all(row_all_bg[:first_filled]), "empty headspace must stay above the liquid"
    assert np.all(row_all_fill[first_filled:]), "liquid rows must be bottom-aligned and contiguous"
    return int(interior.shape[0] - first_filled)


def _tube_info_from_coords(coords: np.ndarray) -> dict | None:
    r0, c0, r1, c1 = bbox_coords(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None

    local = np.zeros((h, w), dtype=bool)
    local[coords[:, 0] - r0, coords[:, 1] - c0] = True
    expected = np.zeros((h, w), dtype=bool)
    expected[:, 0] = True
    expected[:, -1] = True
    expected[-1, :] = True
    if not np.array_equal(local, expected):
        return None

    return {
        "bbox": (r0, c0, r1, c1),
        "inner_bbox": (r0, c0 + 1, r1 - 1, c1 - 1),
        "inner_h": h - 1,
        "inner_w": w - 2,
    }


def _is_spiral_prefix(coords: np.ndarray) -> bool:
    if coords.size == 0 or coords.shape[0] > len(SPIRAL):
        return False
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    rel = {(int(r - r0), int(c - c0)) for r, c in coords}
    return rel == set(SPIRAL[: coords.shape[0]])


def _slice_from_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1]


def _analyze_puzzle(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = get_bg(g)
    non_bg_colors = colors_present(g, exclude={bg})
    assert len(non_bg_colors) == 3, "expected wall, liquid, and cover colors"

    wall_candidates: list[tuple[int, list[dict]]] = []
    for color in non_bg_colors:
        comps = components_mask(g == color, connectivity=4)
        if len(comps) < 2:
            continue
        tubes: list[dict] = []
        ok = True
        for comp in comps:
            info = _tube_info_from_coords(comp)
            if info is None:
                ok = False
                break
            tubes.append(info)
        if ok:
            tubes.sort(key=lambda tube: tube["bbox"])
            wall_candidates.append((color, tubes))

    assert len(wall_candidates) == 1, "wall color must be uniquely identifiable by U-tubes"
    wall_color, tubes = wall_candidates[0]

    tube_bbox_mask = np.zeros_like(g, dtype=bool)
    for tube in tubes:
        r0, c0, r1, c1 = tube["bbox"]
        tube_bbox_mask[r0 : r1 + 1, c0 : c1 + 1] = True

    liquid_candidates: list[tuple[int, np.ndarray]] = []
    for color in non_bg_colors:
        if color == wall_color:
            continue
        gauge_coords = np.argwhere((g == color) & (~tube_bbox_mask))
        if gauge_coords.size == 0:
            continue
        if _is_spiral_prefix(gauge_coords):
            liquid_candidates.append((color, gauge_coords))

    assert len(liquid_candidates) == 1, "gauge must uniquely identify the liquid color"
    liquid_color, gauge_coords = liquid_candidates[0]
    gauge_count = int(gauge_coords.shape[0])

    visible_total = 0
    hidden_index: int | None = None
    cover_color: int | None = None
    tube_states: list[dict] = []
    for idx, tube in enumerate(tubes):
        inner = _slice_from_bbox(g, tube["inner_bbox"])
        if np.any(inner == liquid_color):
            liquid_height = _full_rect_rows(inner, bg, liquid_color)
            assert liquid_height > 0, "visible tubes must contain some liquid"
            visible_total += int(np.count_nonzero(inner == liquid_color))
            tube_states.append({"kind": "visible", "liquid_height": liquid_height})
            continue

        unique_inner = {int(color) for color in np.unique(inner)}
        assert bg not in unique_inner, "covered tube interior must be fully hidden"
        assert len(unique_inner) == 1, "covered tube must use one solid cover color"
        hidden_cover = unique_inner.pop()
        assert hidden_cover not in {wall_color, liquid_color}, "cover must use its own color"
        assert hidden_index is None, "expected exactly one covered tube"
        hidden_index = idx
        cover_color = hidden_cover
        tube_states.append({"kind": "covered"})

    assert hidden_index is not None, "expected one covered tube"
    hidden_tube = tubes[hidden_index]
    missing_cells = gauge_count - visible_total
    assert missing_cells >= 0, "gauge total cannot be smaller than visible liquid total"
    assert missing_cells <= hidden_tube["inner_h"] * hidden_tube["inner_w"], "hidden liquid exceeds tube capacity"
    assert missing_cells % hidden_tube["inner_w"] == 0, "hidden liquid must fill whole rows in its tube"
    hidden_height = missing_cells // hidden_tube["inner_w"]
    assert 0 <= hidden_height <= hidden_tube["inner_h"]

    return {
        "bg": bg,
        "wall_color": wall_color,
        "liquid_color": liquid_color,
        "cover_color": int(cover_color),
        "gauge_count": gauge_count,
        "visible_total": visible_total,
        "tubes": tubes,
        "tube_states": tube_states,
        "hidden_index": hidden_index,
        "hidden_height": hidden_height,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reveal the covered tube and reconstruct its missing liquid from the global spiral gauge total.

    Input structure:
    - The background is black, several disconnected single-cell-thick U-shaped vertical tubes share one wall color, and one 3x3 gauge sits elsewhere.
    - Visible tubes contain bottom-aligned full-width liquid rows in one shared liquid color.
    - Exactly one tube interior is fully hidden by a solid cover plate of a third non-background color.

    Rule:
    - Count the liquid-colored cells in the 3x3 clockwise spiral gauge.
    - Sum the visible liquid cells across every uncovered tube, subtract that sum from the gauge count, and treat the remainder as the hidden tube's liquid area.
    - Remove the cover plate and fill that many liquid cells from the covered tube's bottom upward, keeping all walls, visible liquid, gauge cells, and background unchanged.

    Key insight:
    - The spiral gauge does not measure any single tube; it encodes the conserved total liquid over the whole scene, so the covered tube must be recovered by global counting.
    """
    g = np.asarray(grid, dtype=int)
    info = _analyze_puzzle(g)

    out = g.copy()
    hidden_tube = info["tubes"][info["hidden_index"]]
    r0, c0, r1, c1 = hidden_tube["inner_bbox"]
    out[r0 : r1 + 1, c0 : c1 + 1] = info["bg"]
    if info["hidden_height"] > 0:
        fill_start = r1 - info["hidden_height"] + 1
        out[fill_start : r1 + 1, c0 : c1 + 1] = info["liquid_color"]
    return out


def _gauge_patch(total_cells: int, liquid_color: int) -> np.ndarray:
    patch = np.full((3, 3), BG_COLOR, dtype=int)
    for r, c in SPIRAL[:total_cells]:
        patch[r, c] = liquid_color
    return patch


def _tube_patch(inner_h: int, inner_w: int, wall_color: int, fill_color: int, liquid_height: int) -> np.ndarray:
    patch = np.full((inner_h + 1, inner_w + 2), BG_COLOR, dtype=int)
    patch[:, 0] = wall_color
    patch[:, -1] = wall_color
    patch[-1, :] = wall_color
    if liquid_height > 0:
        patch[inner_h - liquid_height : inner_h, 1 : inner_w + 1] = fill_color
    return patch


def _rect_conflict(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    return not (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _place_items(shapes: list[tuple[int, int]], height: int, width: int, gap: int = 1) -> list[tuple[int, int]] | None:
    order = sorted(range(len(shapes)), key=lambda i: shapes[i][0] * shapes[i][1], reverse=True)
    for _ in range(200):
        placed: dict[int, tuple[int, int]] = {}
        rects: dict[int, tuple[int, int, int, int]] = {}
        ok = True
        for idx in order:
            h, w = shapes[idx]
            candidates = [(r, c) for r in range(height - h + 1) for c in range(width - w + 1)]
            random.shuffle(candidates)
            chosen = None
            for r0, c0 in candidates:
                rect = (r0, c0, r0 + h - 1, c0 + w - 1)
                if any(_rect_conflict(rect, other, gap) for other in rects.values()):
                    continue
                chosen = (r0, c0)
                rects[idx] = rect
                placed[idx] = chosen
                break
            if chosen is None:
                ok = False
                break
        if ok:
            return [placed[i] for i in range(len(shapes))]
    return None


def _sample_tube_specs() -> list[dict]:
    for _ in range(400):
        n_tubes = 3 if random.random() < 0.65 else 2
        liquid_heights = random.sample([1, 2, 3, 4], n_tubes)
        inner_widths = [1 + int(random.random() < 0.45) for _ in range(n_tubes)]
        hidden_index = random.randrange(n_tubes)
        branch = random.choice(("any", "hidden_wide", "visible_wide"))
        if branch == "hidden_wide" and inner_widths[hidden_index] != 2:
            continue
        if branch == "visible_wide" and not any(w == 2 for i, w in enumerate(inner_widths) if i != hidden_index):
            continue

        total_cells = sum(h * w for h, w in zip(liquid_heights, inner_widths))
        if not (2 <= total_cells <= 9):
            continue

        specs: list[dict] = []
        for liquid_height, inner_width in zip(liquid_heights, inner_widths):
            max_extra = 2 if liquid_height < 5 else 1
            inner_h = liquid_height + random.randint(0, max_extra)
            specs.append(
                {
                    "inner_h": inner_h,
                    "inner_w": inner_width,
                    "liquid_height": liquid_height,
                }
            )

        if all(spec["inner_h"] == spec["liquid_height"] for spec in specs):
            idx = random.randrange(n_tubes)
            if specs[idx]["inner_h"] < 5:
                specs[idx]["inner_h"] += 1

        specs[hidden_index]["covered"] = True
        for idx, spec in enumerate(specs):
            spec["covered"] = idx == hidden_index
        return specs
    raise RuntimeError("failed to sample tube specs")


def _render_instance(specs: list[dict], wall_color: int, liquid_color: int, cover_color: int) -> tuple[np.ndarray, np.ndarray, dict] | None:
    total_cells = sum(spec["inner_w"] * spec["liquid_height"] for spec in specs)
    input_patches: list[np.ndarray] = []
    output_patches: list[np.ndarray] = []
    meta_tubes: list[dict] = []

    for spec in specs:
        if spec["covered"]:
            input_patch = _tube_patch(spec["inner_h"], spec["inner_w"], wall_color, cover_color, spec["inner_h"])
        else:
            input_patch = _tube_patch(spec["inner_h"], spec["inner_w"], wall_color, liquid_color, spec["liquid_height"])
        output_patch = _tube_patch(spec["inner_h"], spec["inner_w"], wall_color, liquid_color, spec["liquid_height"])
        input_patches.append(input_patch)
        output_patches.append(output_patch)
        meta_tubes.append(
            {
                "inner_h": spec["inner_h"],
                "inner_w": spec["inner_w"],
                "liquid_height": spec["liquid_height"],
                "covered": spec["covered"],
            }
        )

    gauge = _gauge_patch(total_cells, liquid_color)
    input_patches.append(gauge)
    output_patches.append(gauge)

    shapes = [patch.shape for patch in input_patches]
    min_h = max(shape[0] for shape in shapes) + 2
    min_w = sum(shape[1] for shape in shapes) + len(shapes)
    canvas_h = random.randint(max(10, min_h), min(24, max(10, min_h + 8)))
    canvas_w = random.randint(max(12, min_w), min(28, max(12, min_w + 8)))
    positions = _place_items(shapes, canvas_h, canvas_w, gap=1)
    if positions is None:
        return None

    inp = np.full((canvas_h, canvas_w), BG_COLOR, dtype=int)
    out = np.full((canvas_h, canvas_w), BG_COLOR, dtype=int)
    for patch, (r0, c0) in zip(input_patches, positions):
        paste(inp, patch, r0, c0)
    for patch, (r0, c0) in zip(output_patches, positions):
        paste(out, patch, r0, c0)

    meta = {
        "total_cells": total_cells,
        "gauge_pos": positions[-1],
        "tubes": meta_tubes,
        "wall_color": wall_color,
        "liquid_color": liquid_color,
        "cover_color": cover_color,
    }
    return inp, out, meta


def _semantic_checks(inp: np.ndarray, out: np.ndarray, meta: dict) -> bool:
    try:
        info = _analyze_puzzle(inp)
    except AssertionError:
        return False

    if info["gauge_count"] != meta["total_cells"]:
        return False
    if info["wall_color"] != meta["wall_color"]:
        return False
    if info["liquid_color"] != meta["liquid_color"]:
        return False
    if info["cover_color"] != meta["cover_color"]:
        return False

    input_levels = sorted(spec["liquid_height"] for spec in meta["tubes"])
    output_levels = sorted(
        state["liquid_height"] if state["kind"] == "visible" else info["hidden_height"]
        for state in info["tube_states"]
    )
    if sorted(set(input_levels)) != input_levels:
        return False
    if output_levels != input_levels:
        return False

    if np.any(out == meta["cover_color"]):
        return False
    if not np.array_equal(main(inp), out):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    wall_color, liquid_color, cover_color = random.sample(range(1, 10), 3)
    specs = _sample_tube_specs()
    rendered = _render_instance(specs, wall_color, liquid_color, cover_color)
    if rendered is None:
        return None
    inp, out, meta = rendered
    if not _semantic_checks(inp, out, meta):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid spiral-gauge tube puzzle")
