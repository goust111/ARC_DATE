import random
from collections import deque

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]

PALETTE = list(range(10))
GEN_ATTEMPTS = 500
LOCAL_RETRIES = 400


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int) -> list[Coord]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _bbox(coords: list[Coord]) -> BBox:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_rect_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[Coord] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) != bg:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            box = _bbox(coords)
            r0, c0, r1, c1 = box
            patch = grid[r0 : r1 + 1, c0 : c1 + 1]
            assert np.all(patch != bg), "foreground component must be a dense rectangle"
            assert len(coords) == patch.size, "foreground component must fill its bbox"
            components.append(
                {
                    "bbox": box,
                    "patch": patch.copy(),
                    "colors": sorted(int(x) for x in np.unique(patch)),
                    "area": int(patch.size),
                }
            )
    return components


def _classify_rectangles(grid: np.ndarray) -> tuple[int, dict, dict]:
    bg = _dominant_color(grid)
    rects = _find_rect_components(grid, bg)
    assert len(rects) == 2, "expected exactly two foreground rectangles"
    plain = [rect for rect in rects if len(rect["colors"]) == 1]
    marked = [rect for rect in rects if len(rect["colors"]) > 1]
    assert len(plain) == 1 and len(marked) == 1, "expected one plain rectangle and one marked rectangle"
    plain_rect = plain[0]
    marked_rect = marked[0]
    base = plain_rect["colors"][0]
    assert base in marked_rect["colors"], "rectangles must share the same base color"
    assert plain_rect["area"] > marked_rect["area"], "plain rectangle must be larger"
    return bg, marked_rect, plain_rect


def _marker_specs(marked_rect: dict, base: int) -> list[dict]:
    r0, c0, r1, c1 = marked_rect["bbox"]
    patch = marked_rect["patch"]
    specs: list[dict] = []
    for lr in range(patch.shape[0]):
        for lc in range(patch.shape[1]):
            color = int(patch[lr, lc])
            if color == base:
                continue
            top = lr
            bottom = patch.shape[0] - 1 - lr
            left = lc
            right = patch.shape[1] - 1 - lc
            assert min(top, bottom, left, right) >= 1, "markers must be inside the rectangle"
            assert top != bottom and left != right, "nearest horizontal and vertical sides must be unique"
            distances = {
                "top": top,
                "bottom": bottom,
                "left": left,
                "right": right,
            }
            min_dist = min(distances.values())
            near_edges = [side for side, dist in distances.items() if dist == min_dist]
            assert len(near_edges) == 1, "each marker must have one uniquely nearest edge"
            hside = "top" if top < bottom else "bottom"
            vside = "left" if left < right else "right"
            specs.append(
                {
                    "color": color,
                    "grid_pos": (r0 + lr, c0 + lc),
                    "local_pos": (lr, lc),
                    "near_edge": near_edges[0],
                    "hside": hside,
                    "vside": vside,
                    "hdist": min(top, bottom),
                    "vdist": min(left, right),
                }
            )
    assert specs, "expected at least one marker"
    return specs


def _port_in_big(spec: dict, big_bbox: BBox) -> Coord:
    r0, c0, r1, c1 = big_bbox
    rr = r0 + spec["hdist"] if spec["hside"] == "top" else r1 - spec["hdist"]
    cc = c0 + spec["vdist"] if spec["vside"] == "left" else c1 - spec["vdist"]
    assert r0 < rr < r1 and c0 < cc < c1, "mapped port must remain inside the big rectangle"
    return int(rr), int(cc)


def _ray_cells(spec: dict, big_bbox: BBox) -> list[Coord]:
    r0, c0, r1, c1 = big_bbox
    pr, pc = _port_in_big(spec, big_bbox)
    if spec["near_edge"] == "top":
        return [(r, pc) for r in range(pr, r1)]
    if spec["near_edge"] == "bottom":
        return [(r, pc) for r in range(r0 + 1, pr + 1)]
    if spec["near_edge"] == "left":
        return [(pr, c) for c in range(pc, c1)]
    return [(pr, c) for c in range(c0 + 1, pc + 1)]


def _paint_big_rectangle(out: np.ndarray, big_bbox: BBox, base: int, specs: list[dict]) -> np.ndarray:
    r0, c0, r1, c1 = big_bbox
    out[r0 : r1 + 1, c0 : c1 + 1] = base

    vertical: dict[Coord, int] = {}
    horizontal: dict[Coord, int] = {}
    for spec in specs:
        cells = _ray_cells(spec, big_bbox)
        target = vertical if spec["near_edge"] in {"top", "bottom"} else horizontal
        for cell in cells:
            prev = target.get(cell)
            if prev is not None:
                assert prev == spec["color"], "same-direction overlapping rays are not supported"
            target[cell] = spec["color"]

    for cell, color in vertical.items():
        if cell not in horizontal:
            out[cell] = color
    for cell, color in horizontal.items():
        if cell not in vertical:
            out[cell] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transfer edge-relative marker offsets from the small rectangle onto the large rectangle as ray emitters, then erase horizontal-vertical crossings back to the base color.

    Input structure:
    - The background contains exactly two dense foreground rectangles separated by background.
    - One rectangle is small, uses one shared base color plus several interior marker pixels, and every marker has a uniquely nearest edge.
    - The other rectangle is larger, uses only that same base color, and acts as the target canvas.

    Rule:
    - Keep the small marked rectangle unchanged.
    - For each marker, preserve its nearest horizontal-side offset and nearest vertical-side offset to locate the corresponding emitter cell inside the large rectangle.
    - Color that emitter with the marker color and extend a same-color ray inward, perpendicular to the marker's uniquely nearest edge, stopping one cell before the opposite side.
    - If a horizontal ray cell and a vertical ray cell coincide, leave that cell in the large rectangle's base color instead of either ray color.

    Key insight:
    - The small rectangle is a blueprint rather than a sticker: its edge-relative marker coordinates choose emitter ports, and the visible pattern is formed by the transferred rays plus the base-colored annihilation gaps at their crossings.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg, marked_rect, plain_rect = _classify_rectangles(g)
    base = plain_rect["colors"][0]
    specs = _marker_specs(marked_rect, base)
    out = g.copy()
    _paint_big_rectangle(out, plain_rect["bbox"], base, specs)
    assert _dominant_color(out) == bg or bg in np.unique(out), "background must remain present"
    return out


def _marker_candidates(h: int, w: int) -> list[dict]:
    candidates: list[dict] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            top = r
            bottom = h - 1 - r
            left = c
            right = w - 1 - c
            if top == bottom or left == right:
                continue
            distances = {"top": top, "bottom": bottom, "left": left, "right": right}
            min_dist = min(distances.values())
            near_edges = [side for side, dist in distances.items() if dist == min_dist]
            if len(near_edges) != 1:
                continue
            candidates.append(
                {
                    "local_pos": (r, c),
                    "near_edge": near_edges[0],
                    "hside": "top" if top < bottom else "bottom",
                    "vside": "left" if left < right else "right",
                    "hdist": min(top, bottom),
                    "vdist": min(left, right),
                }
            )
    return candidates


def _choose_marker_layout(small_h: int, small_w: int, big_h: int, big_w: int, count: int) -> list[dict] | None:
    candidates = _marker_candidates(small_h, small_w)
    if len(candidates) < count:
        return None
    for _ in range(LOCAL_RETRIES):
        chosen = random.sample(candidates, count)
        unique_edges = {cand["near_edge"] for cand in chosen}
        if not (unique_edges & {"top", "bottom"} and unique_edges & {"left", "right"}):
            continue
        if len(unique_edges) < random.choice((3, 4)):
            continue
        if len({cand["hside"] for cand in chosen}) < 2 or len({cand["vside"] for cand in chosen}) < 2:
            continue

        port_positions = []
        vertical_cols: set[int] = set()
        vertical_rows: set[int] = set()
        horizontal_rows: set[int] = set()
        horizontal_cols: set[int] = set()
        ok = True
        for cand in chosen:
            spec = dict(cand)
            pr = spec["hdist"] if spec["hside"] == "top" else big_h - 1 - spec["hdist"]
            pc = spec["vdist"] if spec["vside"] == "left" else big_w - 1 - spec["vdist"]
            port = (pr, pc)
            if port in port_positions:
                ok = False
                break
            if spec["near_edge"] in {"top", "bottom"}:
                if pc in vertical_cols or pr in horizontal_rows or pc in horizontal_cols:
                    ok = False
                    break
                vertical_cols.add(pc)
                vertical_rows.add(pr)
            else:
                if pr in horizontal_rows or pr in vertical_rows or pc in vertical_cols:
                    ok = False
                    break
                horizontal_rows.add(pr)
                horizontal_cols.add(pc)
            port_positions.append(port)
        if not ok:
            continue
        if not vertical_cols or not horizontal_rows:
            continue
        return chosen
    return None


def _boxes_separated(a: BBox, b: BBox, gap: int = 2) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _place_rectangles(grid_h: int, grid_w: int, big_h: int, big_w: int, small_h: int, small_w: int) -> tuple[BBox, BBox] | None:
    for _ in range(LOCAL_RETRIES):
        br0 = random.randint(1, grid_h - big_h - 1)
        bc0 = random.randint(1, grid_w - big_w - 1)
        sr0 = random.randint(1, grid_h - small_h - 1)
        sc0 = random.randint(1, grid_w - small_w - 1)
        big_bbox = (br0, bc0, br0 + big_h - 1, bc0 + big_w - 1)
        small_bbox = (sr0, sc0, sr0 + small_h - 1, sc0 + small_w - 1)
        if _boxes_separated(big_bbox, small_bbox, gap=2):
            return big_bbox, small_bbox
    return None


def _render_input(
    grid_h: int,
    grid_w: int,
    bg: int,
    base: int,
    big_bbox: BBox,
    small_bbox: BBox,
    markers: list[dict],
) -> np.ndarray:
    grid = np.full((grid_h, grid_w), bg, dtype=int)
    br0, bc0, br1, bc1 = big_bbox
    sr0, sc0, sr1, sc1 = small_bbox
    grid[br0 : br1 + 1, bc0 : bc1 + 1] = base
    grid[sr0 : sr1 + 1, sc0 : sc1 + 1] = base
    for marker in markers:
        lr, lc = marker["local_pos"]
        grid[sr0 + lr, sc0 + lc] = marker["color"]
    return grid


def _expected_from_metadata(inp: np.ndarray, base: int, big_bbox: BBox, markers: list[dict]) -> np.ndarray:
    out = inp.copy()
    specs = []
    for marker in markers:
        spec = {
            "color": marker["color"],
            "near_edge": marker["near_edge"],
            "hside": marker["hside"],
            "vside": marker["vside"],
            "hdist": marker["hdist"],
            "vdist": marker["vdist"],
        }
        specs.append(spec)
    return _paint_big_rectangle(out, big_bbox, base, specs)


def _semantic_checks(inp: np.ndarray, out: np.ndarray, base: int, big_bbox: BBox, small_bbox: BBox, markers: list[dict]) -> bool:
    br0, bc0, br1, bc1 = big_bbox
    sr0, sc0, sr1, sc1 = small_bbox
    if not np.array_equal(inp[sr0 : sr1 + 1, sc0 : sc1 + 1], out[sr0 : sr1 + 1, sc0 : sc1 + 1]):
        return False
    if np.array_equal(inp[br0 : br1 + 1, bc0 : bc1 + 1], out[br0 : br1 + 1, bc0 : bc1 + 1]):
        return False

    vertical_cells: set[Coord] = set()
    horizontal_cells: set[Coord] = set()
    vertical_count = 0
    horizontal_count = 0
    for marker in markers:
        spec = {
            "color": marker["color"],
            "near_edge": marker["near_edge"],
            "hside": marker["hside"],
            "vside": marker["vside"],
            "hdist": marker["hdist"],
            "vdist": marker["vdist"],
        }
        cells = _ray_cells(spec, big_bbox)
        if marker["near_edge"] in {"top", "bottom"}:
            vertical_count += 1
            vertical_cells.update(cells)
        else:
            horizontal_count += 1
            horizontal_cells.update(cells)
    crossings = vertical_cells & horizontal_cells
    if vertical_count == 0 or horizontal_count == 0 or not crossings:
        return False
    for cell in crossings:
        if int(out[cell]) != base:
            return False
    if sum(int(out[cell] != base) for cell in vertical_cells | horizontal_cells) <= len(markers):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    small_h = random.randint(6, 8)
    small_w = random.randint(6, 8)
    big_h = random.randint(small_h + 2, min(15, small_h + 7))
    big_w = random.randint(small_w + 2, min(15, small_w + 7))
    marker_count = random.randint(4, min(6, len(PALETTE) - 2))
    layout = _choose_marker_layout(small_h, small_w, big_h, big_w, marker_count)
    if layout is None:
        return None

    fg_area = big_h * big_w + small_h * small_w
    min_h = max(big_h, small_h) + 4
    min_w = max(big_w, small_w) + 4
    if max(min_h, min_w) > 30:
        return None

    valid_shapes = []
    for grid_h in range(max(22, min_h), 31):
        for grid_w in range(max(22, min_w), 31):
            if grid_h * grid_w > 2 * fg_area + 10:
                valid_shapes.append((grid_h, grid_w))
    if not valid_shapes:
        return None
    grid_h, grid_w = random.choice(valid_shapes)

    placement = _place_rectangles(grid_h, grid_w, big_h, big_w, small_h, small_w)
    if placement is None:
        return None
    big_bbox, small_bbox = placement

    colors = random.sample(PALETTE, marker_count + 2)
    bg = colors[0]
    base = colors[1]
    marker_colors = colors[2:]
    if bg == base:
        return None

    markers = []
    for color, marker in zip(marker_colors, layout):
        item = dict(marker)
        item["color"] = color
        markers.append(item)

    inp = _render_input(grid_h, grid_w, bg, base, big_bbox, small_bbox, markers)
    expected = _expected_from_metadata(inp, base, big_bbox, markers)

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    if not _semantic_checks(inp, expected, base, big_bbox, small_bbox, markers):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
