# Combo variant of fc7cae8d (index 29)
# combo_spec: D7/0+D8/2
# Name: 多对多+先排序再对齐
#
# Design analysis for the fused variant
# a) Concrete behavior:
#    Original fc7cae8d has one interior rectangle and one frame corner target.
#    Here the frame exposes 2-4 local L-controls, one per active grid corner:
#    e.g. TL=(2,7), TR=(4,7), BL=(2,9). An interior object whose two guide-edge
#    colors are {2,9} must align to BL, while {4,7} must align to TR. After the
#    correct dihedral transform, its guide edges are cropped off and only the
#    core remains. All objects are then ordered by core area from small to large
#    and stacked top-to-bottom into the output, so the processing order is visible.
# b) Complexity calibration:
#    2-4 objects is enough to force multi-object matching without making the grid
#    unreadable. Control colors are reused across corners, so one color alone is
#    insufficient; the pair identifies the target. Core areas are unique so the
#    sort rule is visible and unambiguous.
# c) Broken assumptions from the original solver:
#    - "there is exactly one interior object" is false now.
#    - "controls are whole frame sides" is false; controls are local corner Ls.
#    - "largest interior component is the target" is false; every interior
#      component is a target.
#    - "return the first aligned crop" is false; we must align all objects, then
#      sort by cropped-core area, then compose the final output.
# d) Placement legality:
#    The generator paints border controls first, then places each transformed
#    object with random_free_location(..., border=2, padding=2), preventing
#    overlap, border contact, and accidental touching between objects.
# e) Visual observability:
#    Humans can see several corner L-controls, several interior rectangles with
#    colored guide edges, and an output that is a size-ordered stack of the
#    cropped cores. The repeated control colors across different corners make the
#    many-to-many assignment visible without requiring hidden metadata.
# f) Visual cleanliness:
#    Control markers stay on the outer border only. Object cores keep a fill-color
#    border ring, so guide-edge detection is stable and the silhouettes remain
#    easy to read. Output uses a 1-cell background gap between stacked cores.
# g) Color inferability:
#    Only four control colors are used for the whole frame. They repeat across
#    corners in a structured way, so color reuse gives evidence for matching
#    without flooding the grid with unrelated colors.
# 9) New solver assumptions and generator guarantees:
#    - Every target object is a single 4-connected interior component.
#    - Each active corner contributes one unique pair of guide colors.
#    - Core areas are unique, so ascending-area order has no ties.
#    - Non-guide outer edges of each object are fill-colored, so only guide sides
#      look like near-solid non-fill edges after cropping to bbox.

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import crop_bbox, find_components, get_bg, paste, random_free_location


CORNER_SPECS = {
    "tl": {"hside": "top", "vside": "left"},
    "tr": {"hside": "top", "vside": "right"},
    "bl": {"hside": "bottom", "vside": "left"},
    "br": {"hside": "bottom", "vside": "right"},
}
CORNER_ORDER = ("tl", "tr", "bl", "br")


def _transform(arr: np.ndarray, k: int, flip: bool) -> np.ndarray:
    out = np.rot90(arr, int(k))
    return np.fliplr(out) if flip else out


def _inverse_transform(arr: np.ndarray, k: int, flip: bool) -> np.ndarray:
    out = np.fliplr(arr) if flip else arr
    return np.rot90(out, -int(k))


def _scan_run(values: np.ndarray, bg: int) -> tuple[int, int] | None:
    seq = [int(v) for v in values]
    if not seq or int(seq[0]) == int(bg):
        return None
    color = int(seq[0])
    length = 0
    for v in seq:
        if int(v) != int(color):
            break
        length += 1
    return (int(color), int(length)) if int(length) >= 2 else None


def _detect_controls(grid: np.ndarray, bg: int) -> dict[str, dict]:
    h, w = map(int, grid.shape)
    corners: dict[str, dict] = {}
    scans = {
        "tl": (grid[0, 1:], grid[1:, 0]),
        "tr": (grid[0, : w - 1][::-1], grid[1:, w - 1]),
        "bl": (grid[h - 1, 1:], grid[: h - 1, 0][::-1]),
        "br": (grid[h - 1, : w - 1][::-1], grid[: h - 1, w - 1][::-1]),
    }
    for key in CORNER_ORDER:
        hrun = _scan_run(scans[key][0], bg)
        vrun = _scan_run(scans[key][1], bg)
        if hrun is None or vrun is None:
            continue
        hcol, hlen = hrun
        vcol, vlen = vrun
        spec = CORNER_SPECS[key]
        corners[key] = {
            "corner": key,
            "guide_colors": {spec["hside"]: int(hcol), spec["vside"]: int(vcol)},
            "lengths": {"h": int(hlen), "v": int(vlen)},
        }
    return corners


def _extract_objects(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = map(int, grid.shape)
    comps = find_components(grid, bg=int(bg), connectivity=4, monochromatic=False, return_format="dict")
    objs: list[dict] = []
    for comp in comps:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        if bool(r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1):
            continue
        patch = crop_bbox(grid, (r0, c0, r1, c1))
        objs.append({"bbox": (r0, c0, r1, c1), "patch": patch})
    objs.sort(key=lambda o: (o["bbox"][0], o["bbox"][1]))
    return objs


def _guide_edges(arr: np.ndarray) -> dict[str, int]:
    fill = int(Counter(map(int, arr.flatten())).most_common(1)[0][0])
    edges = {
        "top": arr[0, :],
        "bottom": arr[-1, :],
        "left": arr[:, 0],
        "right": arr[:, -1],
    }
    out: dict[str, int] = {}
    for side, values in edges.items():
        seq = [int(v) for v in values]
        seq_nf = [v for v in seq if int(v) != int(fill)]
        if not seq_nf:
            continue
        color, count = Counter(seq_nf).most_common(1)[0]
        if int(color) != int(fill) and int(count) >= len(seq) - 1:
            out[side] = int(color)
    return out


def _crop_guides(arr: np.ndarray, guide_colors: dict[str, int]) -> np.ndarray:
    top = 1 if "top" in guide_colors else 0
    bottom = int(arr.shape[0]) - (1 if "bottom" in guide_colors else 0)
    left = 1 if "left" in guide_colors else 0
    right = int(arr.shape[1]) - (1 if "right" in guide_colors else 0)
    return arr[int(top) : int(bottom), int(left) : int(right)].copy()


def _match_object_to_control(obj: np.ndarray, guide_colors: dict[str, int]) -> np.ndarray | None:
    for k in range(4):
        base = np.rot90(obj, int(k))
        for flip in (False, True):
            t = np.fliplr(base) if flip else base
            edges = _guide_edges(t)
            if len(edges) != len(guide_colors):
                continue
            if all(side in edges and int(edges[side]) == int(color) for side, color in guide_colors.items()):
                return _crop_guides(t, guide_colors)
    return None


def _stack_cores(cores: list[np.ndarray], bg: int) -> np.ndarray:
    if not cores:
        return np.zeros((0, 0), dtype=int)
    gap = 1
    out_h = int(sum(int(core.shape[0]) for core in cores) + gap * (len(cores) - 1))
    out_w = int(max(int(core.shape[1]) for core in cores))
    out = np.full((out_h, out_w), int(bg), dtype=int)
    r0 = 0
    for idx, core in enumerate(cores):
        h, w = map(int, core.shape)
        c0 = int((out_w - w) // 2)
        paste(out, core, int(r0), int(c0))
        r0 += int(h)
        if idx != len(cores) - 1:
            r0 += gap
    return out


def _solve_objects(grid: np.ndarray) -> list[dict]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    controls = _detect_controls(g, bg)
    objects = _extract_objects(g, bg)
    assert 2 <= len(controls) <= 4
    assert len(objects) == len(controls)

    matched: list[dict] = []
    used_corners: set[str] = set()
    for obj in objects:
        candidates: list[tuple[str, np.ndarray]] = []
        for key, ctrl in controls.items():
            core = _match_object_to_control(obj["patch"], ctrl["guide_colors"])
            if core is not None:
                candidates.append((key, core))
        assert len(candidates) == 1
        corner, core = candidates[0]
        assert corner not in used_corners
        used_corners.add(corner)
        matched.append(
            {
                "corner": corner,
                "core": core,
                "area": int(core.shape[0] * core.shape[1]),
                "bbox": obj["bbox"],
            }
        )

    matched.sort(key=lambda m: (m["area"], CORNER_ORDER.index(m["corner"]), m["bbox"][0], m["bbox"][1]))
    return matched


def main(grid: np.ndarray) -> np.ndarray:
    """
    Multi-object version of fc7cae8d.

    1. Detect 2-4 local corner L-controls on the border.
    2. Detect all interior rectangle-with-guides objects.
    3. For each object, find the unique dihedral transform whose guide-edge colors
       match one control corner.
    4. Crop off the aligned guide edges to recover the core.
    5. Sort recovered cores by area ascending and stack them vertically.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    matched = _solve_objects(g)
    ordered_cores = [item["core"] for item in matched]
    return _stack_cores(ordered_cores, bg)


def _build_core(h: int, w: int, fill: int, pattern_colors: list[int]) -> np.ndarray:
    core = np.full((h, w), int(fill), dtype=int)
    inner_rows = list(range(1, h - 1))
    inner_cols = list(range(1, w - 1))
    assert inner_rows and inner_cols

    # Seed each pattern color at least once inside the border ring.
    available = [(r, c) for r in inner_rows for c in inner_cols]
    random.shuffle(available)
    for color in pattern_colors:
        r, c = available.pop()
        core[r, c] = int(color)

    extra_shapes = random.randint(2, 5)
    for _ in range(extra_shapes):
        color = int(random.choice(pattern_colors))
        kind = random.choice(["dot", "hline", "vline", "block"])
        if kind == "dot":
            cells = [(random.choice(inner_rows), random.choice(inner_cols))]
        elif kind == "hline":
            rr = random.choice(inner_rows)
            max_len = max(2, min(4, len(inner_cols)))
            line_len = random.randint(2, max_len)
            c0 = random.randint(1, w - 1 - line_len)
            cells = [(rr, c0 + dc) for dc in range(line_len)]
        elif kind == "vline":
            cc = random.choice(inner_cols)
            max_len = max(2, min(4, len(inner_rows)))
            line_len = random.randint(2, max_len)
            r0 = random.randint(1, h - 1 - line_len)
            cells = [(r0 + dr, cc) for dr in range(line_len)]
        else:
            bh = random.randint(1, min(3, h - 2))
            bw = random.randint(1, min(3, w - 2))
            r0 = random.randint(1, h - 1 - bh)
            c0 = random.randint(1, w - 1 - bw)
            cells = [(r0 + dr, c0 + dc) for dr in range(bh) for dc in range(bw)]
        for r, c in cells:
            core[int(r), int(c)] = int(color)
    return core


def _attach_guides(core: np.ndarray, guide_colors: dict[str, int]) -> np.ndarray:
    h, w = map(int, core.shape)
    has_top = "top" in guide_colors
    has_bottom = "bottom" in guide_colors
    has_left = "left" in guide_colors
    has_right = "right" in guide_colors
    out = np.zeros((h + has_top + has_bottom, w + has_left + has_right), dtype=int)
    r0 = 1 if has_top else 0
    c0 = 1 if has_left else 0
    paste(out, core, int(r0), int(c0))

    if has_top:
        out[0, c0 : c0 + w] = int(guide_colors["top"])
    if has_bottom:
        out[-1, c0 : c0 + w] = int(guide_colors["bottom"])
    if has_left:
        out[r0 : r0 + h, 0] = int(guide_colors["left"])
    if has_right:
        out[r0 : r0 + h, -1] = int(guide_colors["right"])

    if has_top and has_left:
        out[0, 0] = 0
    if has_top and has_right:
        out[0, -1] = 0
    if has_bottom and has_left:
        out[-1, 0] = 0
    if has_bottom and has_right:
        out[-1, -1] = 0
    return out


def _paint_controls(grid: np.ndarray, active_corners: list[str], control_colors: dict[str, tuple[int, int]]) -> None:
    h, w = map(int, grid.shape)
    for corner in active_corners:
        hcol, vcol = control_colors[corner]
        hlen = random.randint(3, 6)
        vlen = random.randint(3, 6)
        if corner == "tl":
            grid[0, 1 : 1 + hlen] = int(hcol)
            grid[1 : 1 + vlen, 0] = int(vcol)
        elif corner == "tr":
            grid[0, w - 1 - hlen : w - 1] = int(hcol)
            grid[1 : 1 + vlen, w - 1] = int(vcol)
        elif corner == "bl":
            grid[h - 1, 1 : 1 + hlen] = int(hcol)
            grid[h - 1 - vlen : h - 1, 0] = int(vcol)
        else:
            grid[h - 1, w - 1 - hlen : w - 1] = int(hcol)
            grid[h - 1 - vlen : h - 1, w - 1] = int(vcol)


def _control_palette() -> dict[str, tuple[int, int]]:
    h_left, h_right, v_top, v_bottom = random.sample(list(range(1, 10)), 4)
    return {
        "tl": (int(h_left), int(v_top)),
        "tr": (int(h_right), int(v_top)),
        "bl": (int(h_left), int(v_bottom)),
        "br": (int(h_right), int(v_bottom)),
    }


def _make_expected(cores: list[np.ndarray], bg: int) -> np.ndarray:
    ordered = sorted(cores, key=lambda a: int(a.shape[0] * a.shape[1]))
    return _stack_cores(ordered, bg)


def _choose_active_corners(n_obj: int) -> list[str]:
    if int(n_obj) == 2:
        adjacent_pairs = [
            ["tl", "tr"],
            ["tl", "bl"],
            ["tr", "br"],
            ["bl", "br"],
        ]
        return random.choice(adjacent_pairs)
    return random.sample(list(CORNER_ORDER), int(n_obj))


def _gen_attempt() -> np.ndarray | None:
    bg = 0
    n = random.randint(28, 34)
    n_obj = random.randint(2, 4)
    active_corners = _choose_active_corners(n_obj)
    control_colors = _control_palette()

    grid = np.full((n, n), int(bg), dtype=int)
    _paint_controls(grid, active_corners, control_colors)

    used_areas: set[int] = set()
    expected_cores: list[np.ndarray] = []

    for corner in random.sample(active_corners, len(active_corners)):
        dims = None
        for _ in range(200):
            h = random.randint(4, 8)
            w = random.randint(4, 8)
            area = int(h * w)
            if area in used_areas:
                continue
            used_areas.add(area)
            dims = (h, w)
            break
        if dims is None:
            return None

        used_control_colors = set(control_colors[corner])
        fill_choices = [c for c in range(1, 10) if c not in used_control_colors]
        fill = int(random.choice(fill_choices))
        pattern_pool = [c for c in fill_choices if c != int(fill)]
        n_patterns = random.randint(2, min(3, len(pattern_pool)))
        pattern_colors = random.sample(pattern_pool, n_patterns)

        core = _build_core(dims[0], dims[1], fill, pattern_colors)
        hside = CORNER_SPECS[corner]["hside"]
        vside = CORNER_SPECS[corner]["vside"]
        hcol, vcol = control_colors[corner]
        aligned = _attach_guides(core, {hside: int(hcol), vside: int(vcol)})

        k = random.randrange(4)
        flip = bool(random.randrange(2))
        patch = _inverse_transform(aligned, k, flip)

        try:
            r0, c0 = random_free_location(grid, patch, transparent=int(bg), border=2, padding=2)
        except ValueError:
            return None
        paste(grid, patch, int(r0), int(c0), transparent=int(bg))
        expected_cores.append(core)

    solved = _solve_objects(grid)
    raw_order = sorted(solved, key=lambda item: (item["bbox"][0], item["bbox"][1]))
    if [item["area"] for item in raw_order] == [item["area"] for item in solved]:
        return None

    expected = _make_expected(expected_cores, bg)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate combo input for fc7cae8d/29")
