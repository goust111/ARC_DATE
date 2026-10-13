from __future__ import annotations

import itertools
import random
from collections import Counter

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]

BACKGROUND = 0
PALETTE = tuple(range(1, 10))
CORNERS = ("ul", "ur", "ll", "lr")
CORNER_TO_INDEX = {"ul": (0, 0), "ur": (0, 1), "ll": (1, 0), "lr": (1, 1)}

GEN_RETRIES = 400
PLACEMENT_RETRIES = 300
CLUSTER_GAP = 2


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.reshape(-1))).most_common(1)[0][0])


def bbox_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, bbox: BBox) -> np.ndarray:
    r0, c0, r1, c1 = map(int, bbox)
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def paste(
    dst: np.ndarray,
    patch: np.ndarray,
    top: int,
    left: int,
    *,
    transparent: int | None = None,
) -> None:
    top = int(top)
    left = int(left)
    h, w = map(int, patch.shape)
    view = dst[top : top + h, left : left + w]
    if transparent is None:
        view[:, :] = patch
        return
    mask = patch != int(transparent)
    view[mask] = patch[mask]


def find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = map(int, grid.shape)
    seen = np.zeros((h, w), dtype=bool)
    out = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == int(bg):
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            coords_arr = np.array(coords, dtype=int)
            out.append(
                {
                    "color": color,
                    "coords": coords_arr,
                    "bbox": bbox_coords(coords_arr),
                    "area": int(coords_arr.shape[0]),
                }
            )
    return out


def _mask_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    stack = [start]
    seen = {start}
    h, w = map(int, mask.shape)
    while stack:
        r, c = stack.pop()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _corner_values(mask: np.ndarray) -> dict[str, bool]:
    h, w = map(int, mask.shape)
    return {
        "ul": bool(mask[0, 0]),
        "ur": bool(mask[0, w - 1]),
        "ll": bool(mask[h - 1, 0]),
        "lr": bool(mask[h - 1, w - 1]),
    }


def _detect_notch_corner(mask: np.ndarray) -> str:
    corners = _corner_values(mask)
    missing = [name for name, present in corners.items() if not present]
    present = [name for name, is_present in corners.items() if is_present]
    assert len(missing) == 1 and len(present) == 3, "expected exactly one missing bbox corner"
    return missing[0]


def _parse_marker_triple(positions: list[Coord]) -> tuple[BBox, str, Coord] | None:
    rows = sorted({int(r) for r, _ in positions})
    cols = sorted({int(c) for _, c in positions})
    if len(rows) != 2 or len(cols) != 2:
        return None

    full = {
        (rows[0], cols[0]): "ul",
        (rows[0], cols[1]): "ur",
        (rows[1], cols[0]): "ll",
        (rows[1], cols[1]): "lr",
    }
    given = {tuple(map(int, pos)) for pos in positions}
    if len(given) != 3 or not given.issubset(full):
        return None

    missing_pos = next(pos for pos in full if pos not in given)
    return (rows[0], cols[0], rows[1], cols[1]), full[missing_pos], missing_pos


def _aligned_top_left(corner_pos: Coord, shape: tuple[int, int], corner_name: str) -> Coord:
    r, c = map(int, corner_pos)
    h, w = map(int, shape)
    if corner_name == "ul":
        return r, c
    if corner_name == "ur":
        return r, c - w + 1
    if corner_name == "ll":
        return r - h + 1, c
    if corner_name == "lr":
        return r - h + 1, c - w + 1
    raise ValueError("invalid corner name")


def _centered_top_left(outer_shape: tuple[int, int], inner_shape: tuple[int, int]) -> Coord:
    outer_h, outer_w = map(int, outer_shape)
    inner_h, inner_w = map(int, inner_shape)
    return (outer_h - inner_h) // 2, (outer_w - inner_w) // 2


def _find_clusters(grid: np.ndarray, bg: int) -> list[dict]:
    components = find_components(grid, bg)
    markers = []
    objects = []
    for idx, comp in enumerate(sorted(components, key=lambda item: item["bbox"])):
        if int(comp["area"]) == 1:
            pos = tuple(map(int, comp["coords"][0]))
            markers.append({"id": idx, "color": int(comp["color"]), "pos": pos})
        else:
            objects.append({"id": idx, **comp})

    assert objects and markers, "expected objects and singleton corner markers"
    markers_by_color: dict[int, list[dict]] = {}
    for marker in markers:
        markers_by_color.setdefault(int(marker["color"]), []).append(marker)

    used_markers: set[int] = set()
    clusters = []
    for obj in objects:
        r0, c0, r1, c1 = map(int, obj["bbox"])
        center_r2 = r0 + r1
        center_c2 = c0 + c1
        candidates = []
        for marker_color, color_markers in markers_by_color.items():
            available = [marker for marker in color_markers if marker["id"] not in used_markers]
            if len(available) < 3:
                continue
            for triple in itertools.combinations(available, 3):
                parsed = _parse_marker_triple([marker["pos"] for marker in triple])
                if parsed is None:
                    continue
                outer_bbox, missing_corner, missing_pos = parsed
                mr0, mc0, mr1, mc1 = outer_bbox
                if mr0 + mr1 != center_r2 or mc0 + mc1 != center_c2:
                    continue
                if not (mr0 < r0 and r1 < mr1 and mc0 < c0 and c1 < mc1):
                    continue
                candidates.append(
                    {
                        "marker_color": int(marker_color),
                        "markers": triple,
                        "outer_bbox": outer_bbox,
                        "missing_corner": missing_corner,
                        "missing_pos": missing_pos,
                    }
                )
        assert len(candidates) == 1, "each object must match exactly one three-marker frame"
        cluster = {"object": obj, **candidates[0]}
        used_markers.update(marker["id"] for marker in cluster["markers"])
        clusters.append(cluster)

    assert len(used_markers) == len(markers), "all singleton foreground cells must be markers"
    return clusters


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each non-self-colored center object so its notch faces the missing corner, then slide it diagonally into that corner.

    Input structure
    - The background is dominant and the grid contains several separated clusters.
    - Each cluster has one monochrome 4-connected center object plus three same-color single-pixel corner markers forming three corners of a larger axis-aligned rectangle.
    - The center object's bbox shares the rectangle center, is strictly smaller, and has exactly one missing bbox corner cell that acts as a directional notch.

    Rule
    - For each cluster, infer the absent rectangle corner from the three markers.
    - If the center object color differs from the marker color, rotate the object by 0, 90, 180, or 270 degrees so its notch points toward that missing corner, then translate it along that diagonal until the rotated bbox corner of the same name lands on the missing corner cell.
    - If the center object color matches the marker color, leave the whole cluster unchanged and always keep the three markers fixed.

    Key insight
    - The empty marker corner supplies both the travel direction and the docking coordinate, while the object's own missing bbox corner supplies the rotation; matching colors invert the usual marker logic and freeze the object instead of selecting it.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = get_bg(g)
    clusters = _find_clusters(g, bg)
    out = g.copy()

    for cluster in clusters:
        if int(cluster["object"]["color"]) == int(cluster["marker_color"]):
            continue
        coords = cluster["object"]["coords"]
        out[coords[:, 0], coords[:, 1]] = int(bg)

    for cluster in clusters:
        obj = cluster["object"]
        if int(obj["color"]) == int(cluster["marker_color"]):
            continue

        patch = crop_bbox(g, obj["bbox"])
        options = []
        for k in range(4):
            rotated = np.rot90(patch, k)
            if _detect_notch_corner(rotated != bg) == cluster["missing_corner"]:
                options.append(rotated)
        assert len(options) == 1, "notch direction must determine a unique rotation"

        rotated = options[0]
        top, left = _aligned_top_left(cluster["missing_pos"], rotated.shape, cluster["missing_corner"])
        coords = np.argwhere(rotated != bg)
        coords = coords + np.array([top, left], dtype=int)
        assert np.all(coords[:, 0] >= 0) and np.all(coords[:, 0] < g.shape[0])
        assert np.all(coords[:, 1] >= 0) and np.all(coords[:, 1] < g.shape[1])
        assert np.all(out[coords[:, 0], coords[:, 1]] == bg), "moved object may only land on background"
        paste(out, rotated, top, left, transparent=bg)

    return out


def _make_object_mask(height: int, width: int, notch_corner: str) -> np.ndarray:
    mask = np.ones((height, width), dtype=bool)
    notch_r = 0 if notch_corner[0] == "u" else height - 1
    notch_c = 0 if notch_corner[1] == "l" else width - 1
    mask[notch_r, notch_c] = False

    interior = [(r, c) for r in range(1, height - 1) for c in range(1, width - 1)]
    random.shuffle(interior)
    max_remove = random.randint(0, max(0, len(interior) // 2))
    removed = 0
    for r, c in interior:
        if removed >= max_remove:
            break
        mask[r, c] = False
        if (
            not _mask_connected(mask)
            or np.any(mask.sum(axis=0) == 0)
            or np.any(mask.sum(axis=1) == 0)
            or _detect_notch_corner(mask) != notch_corner
        ):
            mask[r, c] = True
            continue
        removed += 1

    assert _mask_connected(mask)
    assert _detect_notch_corner(mask) == notch_corner
    return mask


def _render_mask(mask: np.ndarray, color: int, bg: int) -> np.ndarray:
    patch = np.full(mask.shape, int(bg), dtype=int)
    patch[mask] = int(color)
    return patch


def _render_cluster(
    object_mask: np.ndarray,
    object_color: int,
    marker_color: int,
    missing_corner: str,
    bg: int,
) -> tuple[np.ndarray, np.ndarray]:
    obj_h, obj_w = map(int, object_mask.shape)
    base = max(obj_h, obj_w)
    outer_h = base + 2 * random.randint(1, 3)
    outer_w = base + 2 * random.randint(1, 3)
    assert outer_h % 2 == obj_h % 2 and outer_w % 2 == obj_w % 2

    markers = np.full((outer_h, outer_w), int(bg), dtype=int)
    corner_cells = {
        "ul": (0, 0),
        "ur": (0, outer_w - 1),
        "ll": (outer_h - 1, 0),
        "lr": (outer_h - 1, outer_w - 1),
    }
    for corner_name, coord in corner_cells.items():
        if corner_name != missing_corner:
            markers[coord] = int(marker_color)

    input_patch = markers.copy()
    centered_top, centered_left = _centered_top_left((outer_h, outer_w), object_mask.shape)
    paste(input_patch, _render_mask(object_mask, object_color, bg), centered_top, centered_left, transparent=bg)

    if int(object_color) == int(marker_color):
        return input_patch, input_patch.copy()

    options = []
    for k in range(4):
        rotated_mask = np.rot90(object_mask, k)
        if _detect_notch_corner(rotated_mask) == missing_corner:
            options.append(rotated_mask)
    assert len(options) == 1
    rotated_mask = options[0]

    output_patch = markers.copy()
    if missing_corner == "ul":
        final_top, final_left = 0, 0
    elif missing_corner == "ur":
        final_top, final_left = 0, outer_w - rotated_mask.shape[1]
    elif missing_corner == "ll":
        final_top, final_left = outer_h - rotated_mask.shape[0], 0
    else:
        final_top, final_left = outer_h - rotated_mask.shape[0], outer_w - rotated_mask.shape[1]
    paste(output_patch, _render_mask(rotated_mask, object_color, bg), final_top, final_left, transparent=bg)
    return input_patch, output_patch


def _rect_separated(a: BBox, b: BBox, gap: int) -> bool:
    return a[2] + gap < b[0] or b[2] + gap < a[0] or a[3] + gap < b[1] or b[3] + gap < a[1]


def _place_patches(pairs: list[tuple[np.ndarray, np.ndarray]], bg: int) -> tuple[np.ndarray, np.ndarray] | None:
    max_h = max(int(inp.shape[0]) for inp, _ in pairs)
    max_w = max(int(inp.shape[1]) for inp, _ in pairs)
    height = random.randint(max(18, max_h + 4), 30)
    width = random.randint(max(18, max_w + 4), 30)

    inp = np.full((height, width), int(bg), dtype=int)
    out = np.full((height, width), int(bg), dtype=int)
    occupied: list[BBox] = []

    order = sorted(range(len(pairs)), key=lambda idx: pairs[idx][0].size, reverse=True)
    for idx in order:
        in_patch, out_patch = pairs[idx]
        ph, pw = map(int, in_patch.shape)
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            top = random.randint(0, height - ph)
            left = random.randint(0, width - pw)
            rect = (top, left, top + ph - 1, left + pw - 1)
            if any(not _rect_separated(rect, other, CLUSTER_GAP) for other in occupied):
                continue
            paste(inp, in_patch, top, left, transparent=bg)
            paste(out, out_patch, top, left, transparent=bg)
            occupied.append(rect)
            placed = True
            break
        if not placed:
            return None
    return inp, out


def _build_case() -> tuple[np.ndarray, np.ndarray] | None:
    cluster_count = random.randint(2, 3)
    if cluster_count == 2:
        anchored_flags = [True, False]
    else:
        anchored_flags = [True] + [False] * (cluster_count - 1)
        if random.random() < 0.4:
            anchored_flags[random.randrange(cluster_count)] = True
    random.shuffle(anchored_flags)
    assert any(anchored_flags) and not all(anchored_flags)

    size_choices = [(3, 3), (3, 5), (5, 3), (5, 5)]
    if cluster_count == 2:
        size_choices += [(3, 7), (7, 3), (5, 7), (7, 5)]

    moving_count = sum(not anchored for anchored in anchored_flags)
    colors = random.sample(PALETTE, cluster_count + moving_count)
    marker_colors = colors[:cluster_count]
    moving_object_colors = iter(colors[cluster_count:])

    cluster_pairs = []
    for marker_color, anchored in zip(marker_colors, anchored_flags):
        height, width = random.choice(size_choices)
        missing_corner = random.choice(CORNERS)
        notch_corner = random.choice(CORNERS if anchored else [corner for corner in CORNERS if corner != missing_corner])
        object_color = int(marker_color) if anchored else int(next(moving_object_colors))
        object_mask = _make_object_mask(height, width, notch_corner)
        in_patch, out_patch = _render_cluster(object_mask, object_color, int(marker_color), missing_corner, BACKGROUND)
        if not anchored:
            assert not np.array_equal(in_patch, out_patch), "moving clusters must visibly change"
        cluster_pairs.append((in_patch, out_patch))

    placed = _place_patches(cluster_pairs, BACKGROUND)
    if placed is None:
        return None
    inp, out = placed
    if np.array_equal(inp, out):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        case = _build_case()
        if case is None:
            continue
        inp, expected = case
        solved = main(inp)
        if np.array_equal(solved, expected):
            return inp
    raise RuntimeError("failed to generate a valid missing-corner slide-and-rotate puzzle")
