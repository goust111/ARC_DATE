from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import crop_bbox, find_components, get_bg, label_mask, paste


CORNERS = ("TL", "TR", "BR", "BL")
CORNER_INDEX = {name: i for i, name in enumerate(CORNERS)}
PALETTE = tuple(range(10))
GRID_H_RANGE = (18, 27)
GRID_W_RANGE = (18, 27)
CORNER_COUNT_CHOICES = (2, 3, 4)
CORNER_COUNT_WEIGHTS = (2, 3, 3)
CORNER_SIZE_RANGE = (5, 9)  # random.randrange high is exclusive
DISTRACTOR_SIZE_RANGE = (4, 7)
GEN_MAX_RETRIES = 400


def _bbox_corner(box: tuple[int, int, int, int], height: int, width: int) -> str | None:
    r0, c0, r1, c1 = map(int, box)
    hits: list[str] = []
    if r0 == 0 and c0 == 0:
        hits.append("TL")
    if r0 == 0 and c1 == width - 1:
        hits.append("TR")
    if r1 == height - 1 and c1 == width - 1:
        hits.append("BR")
    if r1 == height - 1 and c0 == 0:
        hits.append("BL")
    assert len(hits) <= 1, "a component bbox may touch at most one true corner"
    return hits[0] if hits else None


def _advance_corner(corner: str, steps: int) -> str:
    return CORNERS[(CORNER_INDEX[corner] + int(steps)) % 4]


def _corner_anchor(corner: str, height: int, width: int, obj_h: int, obj_w: int) -> tuple[int, int]:
    if corner == "TL":
        return 0, 0
    if corner == "TR":
        return 0, width - obj_w
    if corner == "BR":
        return height - obj_h, width - obj_w
    if corner == "BL":
        return height - obj_h, 0
    raise AssertionError(f"unknown corner {corner}")


def _hole_components_in_patch(patch: np.ndarray, color: int) -> list[np.ndarray]:
    fg = patch == int(color)
    void = ~fg
    labels, n_labels = label_mask(void, connectivity=4)
    holes: list[np.ndarray] = []
    h, w = patch.shape
    for label_id in range(1, n_labels + 1):
        coords = np.argwhere(labels == label_id)
        if coords.size == 0:
            continue
        rs = coords[:, 0]
        cs = coords[:, 1]
        if rs.min() == 0 or rs.max() == h - 1 or cs.min() == 0 or cs.max() == w - 1:
            continue
        holes.append(coords)
    return holes


def _count_holes_patch(patch: np.ndarray, color: int) -> int:
    return len(_hole_components_in_patch(patch, color))


def _hole_area_patch(patch: np.ndarray, color: int) -> int:
    return sum(int(coords.shape[0]) for coords in _hole_components_in_patch(patch, color))


def _fill_holes_patch(patch: np.ndarray, color: int) -> np.ndarray:
    out = patch.copy()
    for coords in _hole_components_in_patch(patch, color):
        out[coords[:, 0], coords[:, 1]] = int(color)
    return out


def _coords_from_patch(patch: np.ndarray, top: int, left: int, bg: int) -> set[tuple[int, int]]:
    coords = np.argwhere(patch != int(bg))
    return {(int(top + r), int(left + c)) for r, c in coords}


def _paint_cells(mask: np.ndarray, cells: set[tuple[int, int]]) -> None:
    for r, c in cells:
        mask[r, c] = True


def _dilate4(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _can_place(mask: np.ndarray, patch: np.ndarray, top: int, left: int, bg: int) -> bool:
    h, w = patch.shape
    if top < 0 or left < 0 or top + h > mask.shape[0] or left + w > mask.shape[1]:
        return False
    fg = patch != int(bg)
    return not bool((mask[top : top + h, left : left + w] & fg).any())


def _component_plan(
    grid: np.ndarray,
    *,
    step_mode: str = "count",
    fill_holes: bool = True,
) -> tuple[int, list[dict]]:
    g = np.array(grid, dtype=int, copy=True)
    assert g.ndim == 2, "expected a 2D grid"
    bg = int(get_bg(g))
    height, width = map(int, g.shape)
    plans: list[dict] = []
    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict"):
        corner = _bbox_corner(comp["bbox"], height, width)
        if corner is None:
            continue
        color = int(comp["color"])
        patch = crop_bbox(g, comp["bbox"])
        hole_count = _count_holes_patch(patch, color)
        hole_area = _hole_area_patch(patch, color)
        assert hole_count > 0, "corner objects must contain at least one hole"
        if step_mode == "count":
            steps = hole_count
        elif step_mode == "area":
            steps = hole_area
        elif step_mode == "fixed1":
            steps = 1
        else:
            raise ValueError(f"unknown step mode {step_mode}")
        target_corner = _advance_corner(corner, steps)
        dst_r0, dst_c0 = _corner_anchor(target_corner, height, width, patch.shape[0], patch.shape[1])
        stamp = _fill_holes_patch(patch, color) if fill_holes else patch.copy()
        plans.append(
            {
                "comp": comp,
                "color": color,
                "source_corner": corner,
                "target_corner": target_corner,
                "stamp": stamp,
                "dst": (int(dst_r0), int(dst_c0)),
            }
        )
    return bg, plans


def _apply_component_plan(
    grid: np.ndarray,
    bg: int,
    plans: list[dict],
    *,
    require_disjoint: bool = True,
) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    used: set[tuple[int, int]] = set()
    for plan in plans:
        for r, c in plan["comp"]["coords"]:
            out[int(r), int(c)] = bg
    for plan in plans:
        dst_r0, dst_c0 = plan["dst"]
        cells = _coords_from_patch(plan["stamp"], dst_r0, dst_c0, bg)
        if require_disjoint:
            assert not (used & cells), "constructed migration targets must be disjoint"
        used |= cells
        paste(out, plan["stamp"], dst_r0, dst_c0, transparent=bg)
    return out


def _simulate_corner_variant(
    grid: np.ndarray,
    *,
    step_mode: str,
    fill_holes: bool,
) -> np.ndarray:
    bg, plans = _component_plan(grid, step_mode=step_mode, fill_holes=fill_holes)
    return _apply_component_plan(grid, bg, plans, require_disjoint=False)


def _fill_all_holey_in_place(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int, copy=True)
    bg = int(get_bg(g))
    out = g.copy()
    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict"):
        patch = crop_bbox(g, comp["bbox"])
        color = int(comp["color"])
        if _count_holes_patch(patch, color) == 0:
            continue
        r0, c0, _r1, _c1 = map(int, comp["bbox"])
        paste(out, _fill_holes_patch(patch, color), r0, c0, transparent=bg)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each true corner object clockwise by its connected-hole count, then consume the holes.

    Input structure:
    - The dominant color is the background; foreground consists of monochromatic 4-connected objects.
    - Relevant objects are exactly those whose bounding boxes touch two adjacent grid borders at a true corner.
    - Corner objects contain one or more background-colored holes; side-only and interior objects are distractors.

    Rule:
    - For each corner object, count its 4-connected hole components.
    - Move that object clockwise by that many corners (TL->TR->BR->BL->TL), aligning its bounding box to the destination corner.
    - The move clears the source pixels to background, and at the destination every hole inside the moved object is filled with the object's own color.
    - Non-corner objects remain unchanged, including their holes.

    Key insight:
    - Hole count is the fuel for movement: it determines the clockwise corner offset, and those same holes are consumed by being filled after the move.
    """
    bg, plans = _component_plan(grid, step_mode="count", fill_holes=True)
    assert plans, "expected at least one corner object"
    return _apply_component_plan(grid, bg, plans)


def _random_distinct_colors(bg: int, count: int) -> list[int]:
    return random.sample([c for c in PALETTE if c != int(bg)], count)


def _seed_positions(h: int, w: int) -> list[tuple[int, int]]:
    return [(r, c) for r in range(1, h - 1, 2) for c in range(1, w - 1, 2)]


def _grow_hole_components(
    h: int,
    w: int,
    count: int,
    *,
    force_multi: bool,
) -> list[set[tuple[int, int]]] | None:
    seeds = _seed_positions(h, w)
    if len(seeds) < int(count):
        return None
    chosen = random.sample(seeds, int(count))
    comps: list[set[tuple[int, int]]] = [{seed} for seed in chosen]
    occupied = set(chosen)
    target_order = list(range(count))
    random.shuffle(target_order)
    for idx in target_order:
        grow_budget = random.randint(0, 2)
        for _ in range(grow_budget):
            frontier: list[tuple[int, int]] = []
            for r, c in comps[idx]:
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if not (1 <= nr < h - 1 and 1 <= nc < w - 1):
                        continue
                    if (nr, nc) in occupied:
                        continue
                    valid = True
                    for ar, ac in ((nr - 1, nc), (nr + 1, nc), (nr, nc - 1), (nr, nc + 1)):
                        if (ar, ac) in occupied and (ar, ac) not in comps[idx]:
                            valid = False
                            break
                    if valid:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cell = random.choice(frontier)
            comps[idx].add(cell)
            occupied.add(cell)
    if force_multi and not any(len(comp) > 1 for comp in comps):
        for idx in range(count):
            frontier = []
            for r, c in comps[idx]:
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if not (1 <= nr < h - 1 and 1 <= nc < w - 1):
                        continue
                    if (nr, nc) in occupied:
                        continue
                    valid = True
                    for ar, ac in ((nr - 1, nc), (nr + 1, nc), (nr, nc - 1), (nr, nc + 1)):
                        if (ar, ac) in occupied and (ar, ac) not in comps[idx]:
                            valid = False
                            break
                    if valid:
                        frontier.append((nr, nc))
            if frontier:
                cell = random.choice(frontier)
                comps[idx].add(cell)
                occupied.add(cell)
                break
    if force_multi and not any(len(comp) > 1 for comp in comps):
        return None
    return comps


def _make_holey_rect(
    h: int,
    w: int,
    color: int,
    bg: int,
    hole_count: int,
    *,
    force_multi: bool = False,
) -> np.ndarray | None:
    comps = _grow_hole_components(h, w, hole_count, force_multi=force_multi)
    if comps is None:
        return None
    patch = np.full((h, w), int(color), dtype=int)
    for comp in comps:
        for r, c in comp:
            patch[r, c] = int(bg)
    if _count_holes_patch(patch, color) != int(hole_count):
        return None
    if force_multi and _hole_area_patch(patch, color) == int(hole_count):
        return None
    return patch


def _make_blob_patch(color: int, bg: int) -> np.ndarray:
    h = random.randrange(2, 5)
    w = random.randrange(2, 5)
    target = random.randint(2, max(2, h * w - 1))
    mask = np.zeros((h, w), dtype=bool)
    start = (random.randrange(h), random.randrange(w))
    mask[start] = True
    frontier = {start}
    while int(mask.sum()) < target and frontier:
        cell = random.choice(tuple(frontier))
        frontier.discard(cell)
        r, c = cell
        neighbors = []
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                neighbors.append((nr, nc))
        random.shuffle(neighbors)
        for nr, nc in neighbors[: random.randint(1, len(neighbors)) if neighbors else 0]:
            mask[nr, nc] = True
            frontier.add((nr, nc))
            if int(mask.sum()) >= target:
                break
    coords = np.argwhere(mask)
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    cropped = mask[r0 : r1 + 1, c0 : c1 + 1]
    patch = np.full(cropped.shape, int(bg), dtype=int)
    patch[cropped] = int(color)
    return patch


def _choose_mapping(source_corners: list[str]) -> dict[str, str] | None:
    n = len(source_corners)
    for _ in range(200):
        targets = random.sample(list(CORNERS), n)
        random.shuffle(targets)
        mapping = {src: dst for src, dst in zip(source_corners, targets)}
        counts = [((CORNER_INDEX[mapping[src]] - CORNER_INDEX[src]) % 4) or 4 for src in source_corners]
        if len(set(counts)) < 2:
            continue
        if max(counts) < 2:
            continue
        if random.random() < 0.55 and 4 not in counts:
            continue
        return mapping
    return None


def _pairwise_separated(groups: list[set[tuple[int, int]]]) -> bool:
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            a = groups[i]
            b = groups[j]
            if a & b:
                return False
            for r, c in a:
                if (r - 1, c) in b or (r + 1, c) in b or (r, c - 1) in b or (r, c + 1) in b:
                    return False
    return True


def _candidate_positions(
    shape: tuple[int, int],
    patch_shape: tuple[int, int],
    mode: str,
) -> list[tuple[int, int]]:
    height, width = map(int, shape)
    h, w = map(int, patch_shape)
    positions: list[tuple[int, int]] = []
    if mode == "interior":
        for r0 in range(1, height - h):
            for c0 in range(1, width - w):
                positions.append((r0, c0))
    elif mode == "top":
        if h <= height and w < width - 1:
            for c0 in range(1, width - w):
                positions.append((0, c0))
    elif mode == "bottom":
        if h <= height and w < width - 1:
            for c0 in range(1, width - w):
                positions.append((height - h, c0))
    elif mode == "left":
        if w <= width and h < height - 1:
            for r0 in range(1, height - h):
                positions.append((r0, 0))
    elif mode == "right":
        if w <= width and h < height - 1:
            for r0 in range(1, height - h):
                positions.append((r0, width - w))
    else:
        raise ValueError(mode)
    random.shuffle(positions)
    return positions


def _place_distractor(
    grid_in: np.ndarray,
    grid_out: np.ndarray,
    forbidden: np.ndarray,
    patch: np.ndarray,
    *,
    bg: int,
    modes: list[str],
) -> tuple[int, int, str] | None:
    mask = _dilate4(forbidden)
    for mode in modes:
        for r0, c0 in _candidate_positions(grid_in.shape, patch.shape, mode):
            if not _can_place(mask, patch, r0, c0, bg):
                continue
            paste(grid_in, patch, r0, c0, transparent=bg)
            paste(grid_out, patch, r0, c0, transparent=bg)
            _paint_cells(forbidden, _coords_from_patch(patch, r0, c0, bg))
            return int(r0), int(c0), mode
    return None


def _gen_attempt() -> tuple[np.ndarray, np.ndarray, dict] | None:
    height = random.randrange(*GRID_H_RANGE)
    width = random.randrange(*GRID_W_RANGE)
    bg = random.choice(PALETTE)
    n_corners = random.choices(CORNER_COUNT_CHOICES, weights=CORNER_COUNT_WEIGHTS, k=1)[0]
    source_corners = random.sample(list(CORNERS), n_corners)
    mapping = _choose_mapping(source_corners)
    if mapping is None:
        return None

    corner_colors = _random_distinct_colors(bg, n_corners)
    corner_specs: list[dict] = []
    force_multi_idx = random.randrange(n_corners)
    for i, src in enumerate(source_corners):
        dst = mapping[src]
        count = ((CORNER_INDEX[dst] - CORNER_INDEX[src]) % 4) or 4
        patch = None
        for _ in range(60):
            h = random.randrange(*CORNER_SIZE_RANGE)
            w = random.randrange(*CORNER_SIZE_RANGE)
            if len(_seed_positions(h, w)) < count:
                continue
            patch = _make_holey_rect(
                h,
                w,
                corner_colors[i],
                bg,
                count,
                force_multi=(i == force_multi_idx),
            )
            if patch is not None:
                break
        if patch is None:
            return None
        filled = _fill_holes_patch(patch, corner_colors[i])
        src_r0, src_c0 = _corner_anchor(src, height, width, patch.shape[0], patch.shape[1])
        dst_r0, dst_c0 = _corner_anchor(dst, height, width, patch.shape[0], patch.shape[1])
        corner_specs.append(
            {
                "src_corner": src,
                "dst_corner": dst,
                "count": count,
                "color": corner_colors[i],
                "patch": patch,
                "filled": filled,
                "src_pos": (src_r0, src_c0),
                "dst_pos": (dst_r0, dst_c0),
                "hole_area": _hole_area_patch(patch, corner_colors[i]),
            }
        )

    if not any(spec["hole_area"] > spec["count"] for spec in corner_specs):
        return None

    source_groups = [_coords_from_patch(spec["patch"], *spec["src_pos"], bg) for spec in corner_specs]
    target_groups = [_coords_from_patch(spec["filled"], *spec["dst_pos"], bg) for spec in corner_specs]
    if not _pairwise_separated(source_groups):
        return None
    if not _pairwise_separated(target_groups):
        return None

    inp = np.full((height, width), int(bg), dtype=int)
    out = np.full((height, width), int(bg), dtype=int)
    forbidden = np.zeros((height, width), dtype=bool)

    for spec in corner_specs:
        paste(inp, spec["patch"], spec["src_pos"][0], spec["src_pos"][1], transparent=bg)
        paste(out, spec["filled"], spec["dst_pos"][0], spec["dst_pos"][1], transparent=bg)
        _paint_cells(forbidden, _coords_from_patch(spec["patch"], *spec["src_pos"], bg))
        _paint_cells(forbidden, _coords_from_patch(spec["filled"], *spec["dst_pos"], bg))

    distractor_palette = [c for c in PALETTE if c != bg]
    side_color = random.choice(distractor_palette if random.random() < 0.5 else corner_colors)
    side_patch = None
    for _ in range(40):
        dh = random.randrange(*DISTRACTOR_SIZE_RANGE)
        dw = random.randrange(*DISTRACTOR_SIZE_RANGE)
        holes = random.choice((1, 2))
        if len(_seed_positions(dh, dw)) < holes:
            continue
        side_patch = _make_holey_rect(dh, dw, side_color, bg, holes, force_multi=True)
        if side_patch is not None:
            break
    if side_patch is None:
        return None
    side_modes = random.sample(["top", "bottom", "left", "right"], 4)
    side_place = _place_distractor(inp, out, forbidden, side_patch, bg=bg, modes=side_modes)
    if side_place is None:
        return None

    inner_color = random.choice(distractor_palette if random.random() < 0.5 else corner_colors)
    inner_patch = None
    for _ in range(40):
        dh = random.randrange(*DISTRACTOR_SIZE_RANGE)
        dw = random.randrange(*DISTRACTOR_SIZE_RANGE)
        holes = random.choice((1, 2))
        if len(_seed_positions(dh, dw)) < holes:
            continue
        inner_patch = _make_holey_rect(dh, dw, inner_color, bg, holes, force_multi=True)
        if inner_patch is not None:
            break
    if inner_patch is None:
        return None
    inner_place = _place_distractor(inp, out, forbidden, inner_patch, bg=bg, modes=["interior"])
    if inner_place is None:
        return None

    extra_count = random.randint(0, 2)
    for _ in range(extra_count):
        color = random.choice(distractor_palette if random.random() < 0.6 else corner_colors)
        if random.random() < 0.5:
            patch = _make_blob_patch(color, bg)
        else:
            patch = None
            for _retry in range(30):
                dh = random.randrange(4, 6)
                dw = random.randrange(4, 6)
                holes = 1
                if len(_seed_positions(dh, dw)) < holes:
                    continue
                patch = _make_holey_rect(dh, dw, color, bg, holes, force_multi=random.random() < 0.5)
                if patch is not None:
                    break
            if patch is None:
                patch = _make_blob_patch(color, bg)
        modes = ["interior"]
        if random.random() < 0.6:
            modes = random.sample(["top", "bottom", "left", "right", "interior"], 5)
        if _place_distractor(inp, out, forbidden, patch, bg=bg, modes=modes) is None:
            return None

    if int(np.bincount(inp.ravel(), minlength=10).argmax()) != bg:
        return None

    meta = {
        "bg": bg,
        "corner_specs": corner_specs,
        "side_distractor": {
            "patch": side_patch,
            "pos": side_place[:2],
            "mode": side_place[2],
        },
        "inner_distractor": {
            "patch": inner_patch,
            "pos": inner_place[:2],
        },
    }
    return inp, out, meta


def _semantic_checks(inp: np.ndarray, expected: np.ndarray, meta: dict) -> bool:
    actual = main(inp)
    if not np.array_equal(actual, expected):
        return False

    wrong_no_fill = _simulate_corner_variant(inp, step_mode="count", fill_holes=False)
    if np.array_equal(wrong_no_fill, expected):
        return False

    wrong_one = _simulate_corner_variant(inp, step_mode="fixed1", fill_holes=True)
    if np.array_equal(wrong_one, expected):
        return False

    wrong_area = _simulate_corner_variant(inp, step_mode="area", fill_holes=True)
    if np.array_equal(wrong_area, expected):
        return False

    wrong_fill_all = _fill_all_holey_in_place(inp)
    if np.array_equal(wrong_fill_all, expected):
        return False

    side_patch = meta["side_distractor"]["patch"]
    sr0, sc0 = meta["side_distractor"]["pos"]
    sh, sw = side_patch.shape
    if not np.array_equal(inp[sr0 : sr0 + sh, sc0 : sc0 + sw], expected[sr0 : sr0 + sh, sc0 : sc0 + sw]):
        return False

    for spec in meta["corner_specs"]:
        src_r0, src_c0 = spec["src_pos"]
        h, w = spec["patch"].shape
        if _count_holes_patch(spec["patch"], spec["color"]) != spec["count"]:
            return False
        if np.array_equal(
            inp[src_r0 : src_r0 + h, src_c0 : src_c0 + w],
            expected[src_r0 : src_r0 + h, src_c0 : src_c0 + w],
        ) and spec["src_corner"] != spec["dst_corner"]:
            return False

    if not any(spec["hole_area"] > spec["count"] for spec in meta["corner_specs"]):
        return False
    if len({spec["count"] for spec in meta["corner_specs"]}) < 2:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_RETRIES):
        attempt = _gen_attempt()
        if attempt is None:
            continue
        inp, expected, meta = attempt
        if _semantic_checks(inp, expected, meta):
            return inp
    raise RuntimeError("failed to generate a valid input for puzzle 5")
