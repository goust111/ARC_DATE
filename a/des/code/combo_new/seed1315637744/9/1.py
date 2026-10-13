import random
from collections import Counter

import numpy as np


N4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

BG = 0
GREY = 5
GEN_RETRIES = 400
LEGEND_CELL = 5
LEGEND_GAP = 1
RIGHT_GAP = 2

SHAPE_LIBRARY = [
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0]], dtype=bool),
]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in N4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _tight_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return np.zeros((0, 0), dtype=bool)
    r0, c0, r1, c1 = _bbox_coords(coords)
    return np.asarray(mask[r0 : r1 + 1, c0 : c1 + 1], dtype=bool)


def _component_info(coords: np.ndarray) -> dict:
    r0, c0, r1, c1 = _bbox_coords(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return {
        "coords": coords,
        "bbox": (r0, c0, r1, c1),
        "mask": mask,
        "area": int(mask.sum()),
    }


def _mask_key(mask: np.ndarray) -> tuple[str, ...]:
    tight = _tight_mask(mask)
    return tuple("".join("1" if cell else "0" for cell in row) for row in tight)


def _shape_signature(mask: np.ndarray) -> tuple[str, ...]:
    base = _tight_mask(mask)
    variants: list[tuple[str, ...]] = []
    for k in range(4):
        rot = np.rot90(base, k)
        variants.append(_mask_key(rot))
        variants.append(_mask_key(np.fliplr(rot)))
    return min(variants)


def _unique_variants(mask: np.ndarray) -> list[np.ndarray]:
    base = _tight_mask(mask)
    out: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for k in range(4):
        rot = np.rot90(base, k)
        for cand in (rot, np.fliplr(rot)):
            tight = _tight_mask(cand)
            key = (tight.shape, tight.tobytes())
            if key in seen:
                continue
            seen.add(key)
            out.append(tight)
    return out


def _outline_from_shape(shape: np.ndarray) -> np.ndarray:
    shape = np.asarray(shape, dtype=bool)
    pad = np.zeros((shape.shape[0] + 2, shape.shape[1] + 2), dtype=bool)
    pad[1:-1, 1:-1] = shape
    outline = np.zeros_like(pad, dtype=bool)
    coords = np.argwhere(pad)
    for r, c in coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = int(r + dr), int(c + dc)
                if 0 <= nr < pad.shape[0] and 0 <= nc < pad.shape[1] and not pad[nr, nc]:
                    outline[nr, nc] = True
    return outline


def _enclosed_regions(patch: np.ndarray, bg: int) -> list[dict]:
    bg_mask = np.asarray(patch == bg, dtype=bool)
    h, w = bg_mask.shape
    regions: list[dict] = []
    for coords in _components(bg_mask):
        if np.any(coords[:, 0] == 0) or np.any(coords[:, 0] == h - 1):
            continue
        if np.any(coords[:, 1] == 0) or np.any(coords[:, 1] == w - 1):
            continue
        info = _component_info(coords)
        regions.append(info)
    return regions


def _background_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _grey_color(grid: np.ndarray, bg: int) -> int:
    counts = Counter(int(v) for v in grid.ravel() if int(v) != bg)
    assert counts, "grid must contain non-background objects"
    return int(max(counts, key=counts.get))


def _is_rect_frame(mask: np.ndarray) -> bool:
    mask = _tight_mask(mask)
    h, w = mask.shape
    if h < 4 or w < 4:
        return False
    if not (mask[0, :].all() and mask[-1, :].all() and mask[:, 0].all() and mask[:, -1].all()):
        return False
    return not bool(mask[1:-1, 1:-1].any())


def _inside_bbox(inner: tuple[int, int, int, int], outer: tuple[int, int, int, int]) -> bool:
    ir0, ic0, ir1, ic1 = inner
    or0, oc0, or1, oc1 = outer
    return or0 < ir0 and oc0 < ic0 and ir1 < or1 and ic1 < oc1


def _frame_indices(grey_infos: list[dict]) -> set[int]:
    rect_ids = [idx for idx, info in enumerate(grey_infos) if _is_rect_frame(info["mask"])]
    frames: set[int] = set()
    for idx in rect_ids:
        outer = grey_infos[idx]["bbox"]
        if any(j != idx and _inside_bbox(grey_infos[j]["bbox"], outer) for j in range(len(grey_infos))):
            frames.add(idx)
    return frames


def _legend_mapping(
    grid: np.ndarray, bg: int, grey: int, grey_infos: list[dict]
) -> tuple[dict[tuple[str, ...], int], set[int]]:
    h, w = grid.shape
    labels = [(r, c, int(grid[r, c])) for r in range(h) for c in range(w) if int(grid[r, c]) not in (bg, grey)]
    assert labels, "legend labels must exist"

    small_ids = [
        idx
        for idx, info in enumerate(grey_infos)
        if info["mask"].shape[0] <= 3 and info["mask"].shape[1] <= 3
    ]
    used: set[int] = set()
    legend_ids: set[int] = set()
    mapping: dict[tuple[str, ...], int] = {}

    for r, c, color in sorted(labels):
        best_idx = None
        best_key = None
        for idx in small_ids:
            if idx in used:
                continue
            cr0, cc0, cr1, cc1 = grey_infos[idx]["bbox"]
            if cr0 < r or cc0 < c or cr1 > r + 4 or cc1 > c + 4:
                continue
            key = (cr0 - r, cc0 - c, cr1 - cr0, cc1 - cc0)
            if best_idx is None or key < best_key:
                best_idx = idx
                best_key = key
        assert best_idx is not None, "every legend label must own one small grey shape"
        used.add(best_idx)
        legend_ids.add(best_idx)
        mapping[_shape_signature(grey_infos[best_idx]["mask"])] = color
    return mapping, legend_ids


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read each framed negative-space shape, map it through the legend, and color the matching outer outline.

    Input structure:
    - The background is the dominant color.
    - A legend sits on the left: each entry has one colored label pixel and one small grey shape drawn in the same local cell.
    - On the right there are large one-pixel-thick grey rectangular frames, each containing one enclosed background hole shape inside the frame interior.
    - Also on the right there are separate grey outline objects whose enclosed interior shape matches one of the framed holes up to rotation or reflection.

    Rule:
    - Parse the legend into shape-to-color mappings using the small grey template next to each colored label.
    - For every large rectangular frame, inspect background components inside the frame interior and extract the enclosed hole shape.
    - Match that hole shape to the legend under rotation/reflection to get its color, then find outer grey outlines with the same enclosed shape.
    - Return a grid of the same size that keeps only those outer outlines and fills their interiors with the matched colors; everything else becomes background.

    Key insight:
    - The important symbol is the enclosed negative space inside each frame, not the frame border itself; solving requires chaining hole-shape recognition, legend lookup, and outline recoloring.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    grey = _grey_color(g, bg)
    grey_infos = [_component_info(coords) for coords in _components(g == grey)]

    legend_map, legend_ids = _legend_mapping(g, bg, grey, grey_infos)
    frame_ids = _frame_indices(grey_infos)
    active_colors: dict[tuple[str, ...], int] = {}
    frame_boxes: list[tuple[int, int, int, int]] = []

    for idx, info in enumerate(grey_infos):
        if idx not in frame_ids:
            continue
        r0, c0, r1, c1 = info["bbox"]
        frame_boxes.append(info["bbox"])
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        enclosed = _enclosed_regions(interior, bg)
        matches = []
        for region in enclosed:
            sig = _shape_signature(region["mask"])
            if sig in legend_map:
                matches.append(sig)
        assert matches, "each frame must contain a legend-recognizable hole"
        active_colors[matches[0]] = legend_map[matches[0]]

    out = np.full_like(g, bg)
    for idx, info in enumerate(grey_infos):
        if idx in legend_ids or idx in frame_ids:
            continue
        if any(_inside_bbox(info["bbox"], frame_box) for frame_box in frame_boxes):
            continue
        r0, c0, r1, c1 = info["bbox"]
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        enclosed = _enclosed_regions(patch, bg)
        if len(enclosed) != 1:
            continue
        sig = _shape_signature(enclosed[0]["mask"])
        if sig not in active_colors:
            continue
        out[info["coords"][:, 0], info["coords"][:, 1]] = grey
        hole_coords = enclosed[0]["coords"]
        out[r0 + hole_coords[:, 0], c0 + hole_coords[:, 1]] = active_colors[sig]

    return out


def _shape_pool() -> list[np.ndarray]:
    by_sig: dict[tuple[str, ...], np.ndarray] = {}
    for shape in SHAPE_LIBRARY:
        by_sig[_shape_signature(shape)] = shape
    return list(by_sig.values())


def _legend_positions(count: int) -> list[tuple[int, int]]:
    positions = []
    for idx in range(count):
        rr = idx // 2
        cc = idx % 2
        top = 1 + rr * (LEGEND_CELL + LEGEND_GAP)
        left = 1 + cc * (LEGEND_CELL + LEGEND_GAP)
        positions.append((top, left))
    return positions


def _legend_width(count: int) -> int:
    cols = 2 if count > 1 else 1
    return 1 + cols * LEGEND_CELL + (cols - 1) * LEGEND_GAP + 1


def _choose_variant(shape: np.ndarray) -> np.ndarray:
    return random.choice(_unique_variants(shape))


def _frame_patch(shape: np.ndarray) -> np.ndarray:
    hole = _choose_variant(shape)
    outline = _outline_from_shape(hole)
    oh, ow = outline.shape
    top_pad = random.randint(1, 2)
    bottom_pad = random.randint(1, 2)
    left_pad = random.randint(1, 2)
    right_pad = random.randint(1, 2)
    patch = np.full((oh + top_pad + bottom_pad + 2, ow + left_pad + right_pad + 2), BG, dtype=int)
    patch[0, :] = GREY
    patch[-1, :] = GREY
    patch[:, 0] = GREY
    patch[:, -1] = GREY
    r0 = 1 + top_pad
    c0 = 1 + left_pad
    patch[r0 : r0 + oh, c0 : c0 + ow][outline] = GREY
    return patch


def _outer_outline_patch(shape: np.ndarray) -> np.ndarray:
    outline = _outline_from_shape(_choose_variant(shape))
    patch = np.full(outline.shape, BG, dtype=int)
    patch[outline] = GREY
    return patch


def _can_place(blocked: np.ndarray, top: int, left: int, shape: tuple[int, int]) -> bool:
    h, w = blocked.shape
    ph, pw = shape
    r0 = max(0, top - 1)
    c0 = max(0, left - 1)
    r1 = min(h, top + ph + 1)
    c1 = min(w, left + pw + 1)
    return not bool(blocked[r0:r1, c0:c1].any())


def _place_patch(grid: np.ndarray, blocked: np.ndarray, patch: np.ndarray, min_col: int) -> bool:
    h, w = grid.shape
    ph, pw = patch.shape
    candidates = [
        (r, c)
        for r in range(0, h - ph + 1)
        for c in range(min_col, w - pw + 1)
        if _can_place(blocked, r, c, patch.shape)
    ]
    if not candidates:
        return False
    top, left = random.choice(candidates)
    mask = patch != BG
    grid[top : top + ph, left : left + pw][mask] = patch[mask]
    blocked[top : top + ph, left : left + pw] = True
    return True


def _draw_legend_entries(grid: np.ndarray, entries: list[tuple[int, np.ndarray]]) -> None:
    for (color, shape), (top, left) in zip(entries, _legend_positions(len(entries))):
        grid[top, left] = color
        variant = _choose_variant(shape)
        sh, sw = variant.shape
        sr = top + 1 + random.randint(0, 4 - sh)
        sc = left + 1 + random.randint(0, 4 - sw)
        grid[sr : sr + sh, sc : sc + sw][variant] = GREY


def _gen_attempt() -> np.ndarray | None:
    shape_pool = _shape_pool()
    active_count = random.choice((2, 3))
    random.shuffle(shape_pool)
    active_shapes = shape_pool[:active_count]

    colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], active_count)
    entries = list(zip(colors, active_shapes))
    random.shuffle(entries)

    width_min = 27 if active_count == 3 else 24
    h = random.randint(22, 30)
    w = random.randint(width_min, 30)
    right_min_col = _legend_width(active_count) + RIGHT_GAP
    if right_min_col >= w - 5:
        return None

    grid = np.full((h, w), BG, dtype=int)
    _draw_legend_entries(grid, entries)

    blocked = np.zeros((h, w), dtype=bool)
    blocked[:, :right_min_col] = True

    patches = []
    for shape in active_shapes:
        patches.append(_frame_patch(shape))
        patches.append(_outer_outline_patch(shape))
    patches.sort(key=lambda p: (p.shape[0] * p.shape[1], p.shape[0], p.shape[1]), reverse=True)

    for patch in patches:
        if not _place_patch(grid, blocked, patch, right_min_col):
            return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to sample a valid grid")
