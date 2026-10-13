from __future__ import annotations

from collections import Counter, defaultdict
from itertools import product
import random

import numpy as np


ORTHOGONAL_8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)
ORIENTATIONS = ("nw", "ne", "sw", "se")
BG_COLOR = 0
COLOR_POOL = tuple(range(1, 10))
MARKER_SIZES = (2, 3)
DIMENSIONS = ((2, 2), (2, 3), (3, 2), (3, 3))
GAP_VALUES = (1, 2)
PADDING_VALUES = (1, 2)
GEN_RETRIES = 500

_MASK_CACHE: dict[tuple[int, int], list[np.ndarray]] = {}


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, *, transparent: int) -> None:
    h, w = patch.shape
    view = dst[r0 : r0 + h, c0 : c0 + w]
    mask = patch != transparent
    view[mask] = patch[mask]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in ORTHOGONAL_8:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _foreground_components(grid: np.ndarray, bg: int) -> list[dict]:
    out = []
    for idx, coords in enumerate(_components(grid != bg)):
        box = _bbox(coords)
        vals = [int(grid[r, c]) for r, c in coords]
        counts = Counter(vals)
        r0, c0, r1, c1 = box
        out.append(
            {
                "id": idx,
                "coords": coords,
                "bbox": box,
                "area": int(len(coords)),
                "color_counts": counts,
                "colors": tuple(sorted(counts)),
                "h": int(r1 - r0 + 1),
                "w": int(c1 - c0 + 1),
                "cr2": int(r0 + r1),
                "cc2": int(c0 + c1),
            }
        )
    return out


def _marker_orientation(comp: dict, grid: np.ndarray, bg: int) -> tuple[str, int, int] | None:
    if len(comp["colors"]) != 1:
        return None
    r0, c0, r1, c1 = comp["bbox"]
    h = comp["h"]
    if h != comp["w"] or h < 2 or comp["area"] != 2 * h - 1:
        return None
    patch = _crop(grid, comp["bbox"])
    color = comp["colors"][0]
    masks = {
        "nw": np.zeros((h, h), dtype=bool),
        "ne": np.zeros((h, h), dtype=bool),
        "sw": np.zeros((h, h), dtype=bool),
        "se": np.zeros((h, h), dtype=bool),
    }
    masks["nw"][0, :] = True
    masks["nw"][:, 0] = True
    masks["ne"][0, :] = True
    masks["ne"][:, -1] = True
    masks["sw"][-1, :] = True
    masks["sw"][:, 0] = True
    masks["se"][-1, :] = True
    masks["se"][:, -1] = True
    for orientation, mask in masks.items():
        good = np.full((h, h), bg, dtype=int)
        good[mask] = color
        if np.array_equal(patch, good):
            return orientation, h, color
    return None


def _aligned(markers: tuple[dict, dict, dict, dict]) -> bool:
    nw, ne, sw, se = [m["bbox"] for m in markers]
    return (
        (nw[0], nw[2]) == (ne[0], ne[2])
        and (sw[0], sw[2]) == (se[0], se[2])
        and (nw[1], nw[3]) == (sw[1], sw[3])
        and (ne[1], ne[3]) == (se[1], se[3])
        and nw[3] < ne[1]
        and sw[3] < se[1]
        and nw[2] < sw[0]
        and ne[2] < se[0]
    )


def _is_cluster(center: dict, markers: tuple[dict, dict, dict, dict]) -> bool:
    if not _aligned(markers):
        return False
    r0, c0, r1, c1 = center["bbox"]
    outer_r0 = min(marker["bbox"][0] for marker in markers)
    outer_c0 = min(marker["bbox"][1] for marker in markers)
    outer_r1 = max(marker["bbox"][2] for marker in markers)
    outer_c1 = max(marker["bbox"][3] for marker in markers)
    return (
        outer_r0 <= r0 <= r1 <= outer_r1
        and outer_c0 <= c0 <= c1 <= outer_c1
        and markers[0]["cr2"] < center["cr2"] < markers[2]["cr2"]
        and markers[0]["cc2"] < center["cc2"] < markers[1]["cc2"]
    )


def _scene(grid: np.ndarray) -> dict:
    bg = _get_bg(grid)
    comps = _foreground_components(grid, bg)
    markers = []
    centers = []
    for comp in comps:
        marker = _marker_orientation(comp, grid, bg)
        if marker is None:
            centers.append(comp)
            continue
        orientation, size, color = marker
        comp = dict(comp)
        comp["orientation"] = orientation
        comp["marker_size"] = size
        comp["marker_color"] = color
        markers.append(comp)

    markers_by_key: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for marker in markers:
        markers_by_key[(marker["marker_color"], marker["marker_size"])].append(marker)

    clusters = []
    for center in centers:
        for (marker_color, marker_size), same_markers in markers_by_key.items():
            oriented: dict[str, list[dict]] = {name: [] for name in ORIENTATIONS}
            for marker in same_markers:
                if marker["orientation"] == "nw" and marker["cr2"] < center["cr2"] and marker["cc2"] < center["cc2"]:
                    oriented["nw"].append(marker)
                elif marker["orientation"] == "ne" and marker["cr2"] < center["cr2"] and marker["cc2"] > center["cc2"]:
                    oriented["ne"].append(marker)
                elif marker["orientation"] == "sw" and marker["cr2"] > center["cr2"] and marker["cc2"] < center["cc2"]:
                    oriented["sw"].append(marker)
                elif marker["orientation"] == "se" and marker["cr2"] > center["cr2"] and marker["cc2"] > center["cc2"]:
                    oriented["se"].append(marker)
            if not all(oriented[name] for name in ORIENTATIONS):
                continue
            for combo in product(oriented["nw"], oriented["ne"], oriented["sw"], oriented["se"]):
                if len({marker["id"] for marker in combo}) != 4:
                    continue
                if _is_cluster(center, combo):
                    clusters.append(
                        {
                            "center": center,
                            "markers": combo,
                            "marker_color": marker_color,
                            "marker_size": marker_size,
                        }
                    )

    by_center = defaultdict(list)
    for cluster in clusters:
        by_center[cluster["center"]["id"]].append(cluster)
    assert all(len(v) == 1 for v in by_center.values()), "each center must belong to exactly one cluster"
    return {"bg": bg, "clusters": [v[0] for v in by_center.values()]}


def _dominant_color(comp: dict) -> int:
    counts = comp["color_counts"]
    [(color, count)] = counts.most_common(1)
    assert sum(v == count for v in counts.values()) == 1, "center must have a unique dominant color"
    return int(color)


def _index_info(grid: np.ndarray, cluster: dict) -> tuple[int, int, np.ndarray]:
    center = cluster["center"]
    patch = _crop(grid, center["bbox"])
    h, w = patch.shape
    assert center["area"] == h * w, "index block must be solid"
    counts = Counter(map(int, patch.ravel()))
    assert len(counts) == 2, "index block must have exactly two colors"
    major = cluster["marker_color"]
    assert counts[major] == h * w - 1, "marker color must dominate the index block"
    odd_coords = np.argwhere(patch != major)
    assert odd_coords.shape == (1, 2), "index block must have one odd cell"
    tr, tc = map(int, odd_coords[0])
    return tr, tc, patch


def main(grid: np.ndarray) -> np.ndarray:
    """Use the self-colored marker cluster as a miniature index, then crop the referenced cluster's center object.

    Input structure:
    - The background contains an R-by-C array of separate clusters.
    - Every cluster has four same-color corner markers surrounding one center object.
    - Exactly one cluster has marker color equal to the dominant color of its center object.

    Rule:
    - Find that self-colored cluster.
    - Its center object is a solid R-by-C block with exactly one odd-colored cell; read that odd cell's row and column inside the block.
    - Output the center object from the cluster at the same row and column of the whole cluster array, cropped to the center object's bounding box and keeping background pixels inside the crop.

    Key insight:
    - The self-colored cluster is not the answer; it is a miniature coordinate legend whose single odd cell points to the real target cluster.
    """
    g = np.asarray(grid, dtype=int)
    parsed = _scene(g)
    clusters = parsed["clusters"]
    assert clusters, "expected at least one cluster"

    self_clusters = []
    for cluster in clusters:
        center = cluster["center"]
        if _dominant_color(center) != cluster["marker_color"]:
            continue
        try:
            _index_info(g, cluster)
        except AssertionError:
            continue
        self_clusters.append(cluster)
    assert len(self_clusters) == 1, "expected exactly one self-colored index cluster"
    index_cluster = self_clusters[0]

    rows = sorted({cluster["center"]["cr2"] for cluster in clusters})
    cols = sorted({cluster["center"]["cc2"] for cluster in clusters})
    row_count, col_count = len(rows), len(cols)
    target_r, target_c, index_patch = _index_info(g, index_cluster)
    assert index_patch.shape == (row_count, col_count), "index block must encode the cluster lattice size"
    assert len(clusters) == row_count * col_count, "cluster array must be complete"

    by_position = {}
    for cluster in clusters:
        pos = (rows.index(cluster["center"]["cr2"]), cols.index(cluster["center"]["cc2"]))
        assert pos not in by_position, "cluster positions must be unique"
        by_position[pos] = cluster
    assert (target_r, target_c) in by_position, "odd cell must point to an existing cluster"

    target = by_position[(target_r, target_c)]
    return _crop(g, target["center"]["bbox"])


def _full_bbox_masks(h: int, w: int) -> list[np.ndarray]:
    key = (h, w)
    if key in _MASK_CACHE:
        return _MASK_CACHE[key]
    total = h * w
    masks = []
    for bits in range(1, 1 << total):
        flat = np.array([(bits >> i) & 1 for i in range(total)], dtype=bool)
        mask = flat.reshape(h, w)
        if not mask[0, :].any() or not mask[-1, :].any() or not mask[:, 0].any() or not mask[:, -1].any():
            continue
        if len(_components(mask)) != 1:
            continue
        masks.append(mask)
    _MASK_CACHE[key] = masks
    return masks


def _marker_patch(size: int, color: int, orientation: str, bg: int) -> np.ndarray:
    patch = np.full((size, size), bg, dtype=int)
    if orientation == "nw":
        patch[0, :] = color
        patch[:, 0] = color
    elif orientation == "ne":
        patch[0, :] = color
        patch[:, -1] = color
    elif orientation == "sw":
        patch[-1, :] = color
        patch[:, 0] = color
    else:
        patch[-1, :] = color
        patch[:, -1] = color
    return patch


def _cluster_patch(marker_size: int, marker_color: int, center_patch: np.ndarray, bg: int) -> np.ndarray:
    h, w = center_patch.shape
    overlap = marker_size - 2
    center_r0 = 2
    center_c0 = 2
    bottom_r0 = center_r0 + h - overlap
    right_c0 = center_c0 + w - overlap
    patch = np.full((bottom_r0 + marker_size, right_c0 + marker_size), bg, dtype=int)
    _paste(patch, _marker_patch(marker_size, marker_color, "nw", bg), 0, 0, transparent=bg)
    _paste(patch, _marker_patch(marker_size, marker_color, "ne", bg), 0, right_c0, transparent=bg)
    _paste(patch, _marker_patch(marker_size, marker_color, "sw", bg), bottom_r0, 0, transparent=bg)
    _paste(patch, _marker_patch(marker_size, marker_color, "se", bg), bottom_r0, right_c0, transparent=bg)
    _paste(patch, center_patch, center_r0, center_c0, transparent=bg)
    return patch


def _signature(patch: np.ndarray) -> tuple[int, ...]:
    return tuple(int(v) for v in patch.ravel())


def _choose_colors(exclude: set[int], count: int) -> list[int]:
    pool = [c for c in COLOR_POOL if c not in exclude]
    return random.sample(pool, count)


def _nonself_patch(
    h: int,
    w: int,
    marker_color: int,
    used_signatures: set[tuple[int, ...]],
    *,
    force_decoy: bool,
) -> np.ndarray:
    masks = _full_bbox_masks(h, w)
    nonfull_masks = [mask for mask in masks if int(mask.sum()) < h * w]
    for _ in range(200):
        main_color = random.choice([c for c in COLOR_POOL if c != marker_color])
        patch = np.full((h, w), BG_COLOR, dtype=int)
        make_decoy = force_decoy or random.random() < 0.4
        if make_decoy:
            patch[:, :] = main_color
            rr = random.randrange(h)
            cc = random.randrange(w)
            accent = random.choice([c for c in COLOR_POOL if c != main_color])
            patch[rr, cc] = accent
        else:
            mask = random.choice(nonfull_masks)
            patch[mask] = main_color
            cells = [tuple(map(int, coord)) for coord in np.argwhere(mask)]
            max_accents = 1 if len(cells) <= 3 else 2
            n_accents = random.randint(0, max_accents)
            for rr, cc in random.sample(cells, n_accents):
                accent = random.choice([c for c in COLOR_POOL if c not in {BG_COLOR, main_color}])
                patch[rr, cc] = accent

        if _signature(patch) in used_signatures:
            continue
        if len(_components(patch != BG_COLOR)) != 1:
            continue
        counts = Counter(map(int, patch[patch != BG_COLOR]))
        best = counts.most_common(1)[0][1]
        if sum(v == best for v in counts.values()) != 1:
            continue
        if counts.most_common(1)[0][0] == marker_color:
            continue
        used_signatures.add(_signature(patch))
        return patch
    raise RuntimeError("failed to build a non-self center patch")


def _index_patch(h: int, w: int, marker_color: int, target_pos: tuple[int, int]) -> np.ndarray:
    patch = np.full((h, w), marker_color, dtype=int)
    odd_color = random.choice([c for c in COLOR_POOL if c != marker_color])
    patch[target_pos] = odd_color
    return patch


def _render_instance(
    dims: tuple[int, int],
    marker_size: int,
    marker_colors: dict[tuple[int, int], int],
    center_patches: dict[tuple[int, int], np.ndarray],
    pads: tuple[int, int, int, int],
    row_gaps: list[int],
    col_gaps: list[int],
) -> np.ndarray:
    rows, cols = dims
    outer_h = rows + 4
    outer_w = cols + 4
    top, bottom, left, right = pads
    height = top + bottom + rows * outer_h + sum(row_gaps)
    width = left + right + cols * outer_w + sum(col_gaps)
    grid = np.full((height, width), BG_COLOR, dtype=int)

    row_starts = []
    cur = top
    for r in range(rows):
        row_starts.append(cur)
        cur += outer_h + (row_gaps[r] if r < rows - 1 else 0)

    col_starts = []
    cur = left
    for c in range(cols):
        col_starts.append(cur)
        cur += outer_w + (col_gaps[c] if c < cols - 1 else 0)

    for r in range(rows):
        for c in range(cols):
            cluster = _cluster_patch(marker_size, marker_colors[(r, c)], center_patches[(r, c)], BG_COLOR)
            _paste(grid, cluster, row_starts[r], col_starts[c], transparent=BG_COLOR)
    return grid


def _semantic_checks(
    grid: np.ndarray,
    expected: np.ndarray,
    self_pos: tuple[int, int],
    target_pos: tuple[int, int],
    center_patches: dict[tuple[int, int], np.ndarray],
) -> bool:
    if self_pos == target_pos:
        return False
    if _signature(center_patches[self_pos]) == _signature(center_patches[target_pos]):
        return False
    if len({_signature(patch) for patch in center_patches.values()}) != len(center_patches):
        return False

    parsed = _scene(grid)
    clusters = parsed["clusters"]
    self_clusters = []
    for cluster in clusters:
        if _dominant_color(cluster["center"]) != cluster["marker_color"]:
            continue
        try:
            _index_info(grid, cluster)
        except AssertionError:
            continue
        self_clusters.append(cluster)
    if len(self_clusters) != 1:
        return False

    rows = sorted({cluster["center"]["cr2"] for cluster in clusters})
    cols = sorted({cluster["center"]["cc2"] for cluster in clusters})
    if len(rows) * len(cols) != len(clusters):
        return False
    ir, ic, patch = _index_info(grid, self_clusters[0])
    if patch.shape != (len(rows), len(cols)):
        return False
    if (ir, ic) != target_pos:
        return False

    try:
        actual = main(grid)
    except AssertionError:
        return False
    return np.array_equal(actual, expected)


def _gen_attempt() -> np.ndarray | None:
    rows, cols = random.choice(DIMENSIONS)
    marker_size = random.choice(MARKER_SIZES)
    positions = [(r, c) for r in range(rows) for c in range(cols)]
    self_pos = random.choice(positions)
    target_pos = random.choice([pos for pos in positions if pos != self_pos])

    marker_palette = random.sample(list(COLOR_POOL), len(positions))
    marker_colors = {pos: marker_palette[i] for i, pos in enumerate(positions)}

    used_signatures: set[tuple[int, ...]] = set()
    center_patches: dict[tuple[int, int], np.ndarray] = {}
    decoy_positions = [pos for pos in positions if pos not in {self_pos, target_pos}]
    forced_decoy = random.choice(decoy_positions) if decoy_positions else None

    for pos in positions:
        marker_color = marker_colors[pos]
        if pos == self_pos:
            patch = _index_patch(rows, cols, marker_color, target_pos)
            used_signatures.add(_signature(patch))
        else:
            patch = _nonself_patch(rows, cols, marker_color, used_signatures, force_decoy=(pos == forced_decoy))
        center_patches[pos] = patch

    row_gaps = [random.choice(GAP_VALUES) for _ in range(max(0, rows - 1))]
    col_gaps = [random.choice(GAP_VALUES) for _ in range(max(0, cols - 1))]
    pads = tuple(random.choice(PADDING_VALUES) for _ in range(4))
    grid = _render_instance((rows, cols), marker_size, marker_colors, center_patches, pads, row_gaps, col_gaps)
    if max(grid.shape) > 30:
        return None

    expected = center_patches[target_pos].copy()
    if not _semantic_checks(grid, expected, self_pos, target_pos, center_patches):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid self-colored-index cluster grid")
