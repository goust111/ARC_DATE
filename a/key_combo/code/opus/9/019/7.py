from __future__ import annotations

from collections import Counter
import importlib.util
from pathlib import Path

import numpy as np


def _load_arc_utils():
    path = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location("arc_utils_shared", path)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_ARC = _load_arc_utils()


if _ARC is not None:
    get_bg = _ARC.get_bg
    bbox = _ARC.bbox
    crop_bbox = _ARC.crop_bbox
    find_components = _ARC.find_components
else:
    def get_bg(grid: np.ndarray) -> int:
        vals, counts = np.unique(grid, return_counts=True)
        return int(vals[np.argmax(counts)])

    def bbox(mask: np.ndarray):
        coords = np.argwhere(mask)
        if coords.size == 0:
            return None
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        return int(r0), int(c0), int(r1), int(c1)

    def crop_bbox(grid: np.ndarray, box):
        r0, c0, r1, c1 = map(int, box)
        return grid[r0 : r1 + 1, c0 : c1 + 1].copy()

    def find_components(
        grid: np.ndarray,
        *,
        bg: int | None = None,
        connectivity: int = 4,
        monochromatic: bool = True,
        colors=None,
        min_area: int = 1,
        return_format: str = "dict",
    ):
        if bg is None:
            bg = get_bg(grid)
        bg = int(bg)
        h, w = grid.shape
        if connectivity == 4:
            steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        else:
            steps = [
                (-1, -1),
                (-1, 0),
                (-1, 1),
                (0, -1),
                (0, 1),
                (1, -1),
                (1, 0),
                (1, 1),
            ]

        visited = np.zeros((h, w), dtype=bool)
        out = []
        if colors is None:
            allowed = set(map(int, np.unique(grid))) - {bg}
        else:
            allowed = set(map(int, colors)) - {bg}

        for r in range(h):
            for c in range(w):
                color = int(grid[r, c])
                if visited[r, c] or color == bg or color not in allowed:
                    continue
                if monochromatic:
                    def ok(rr: int, cc: int) -> bool:
                        return int(grid[rr, cc]) == color
                else:
                    def ok(rr: int, cc: int) -> bool:
                        return int(grid[rr, cc]) != bg and int(grid[rr, cc]) in allowed

                stack = [(r, c)]
                visited[r, c] = True
                coords = []
                while stack:
                    rr, cc = stack.pop()
                    coords.append((rr, cc))
                    for dr, dc in steps:
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and ok(nr, nc):
                            visited[nr, nc] = True
                            stack.append((nr, nc))
                if len(coords) < min_area:
                    continue
                arr = np.array(coords, dtype=int)
                out.append(
                    {
                        "color": color if monochromatic else None,
                        "coords": arr,
                        "bbox": (
                            int(arr[:, 0].min()),
                            int(arr[:, 1].min()),
                            int(arr[:, 0].max()),
                            int(arr[:, 1].max()),
                        ),
                        "area": int(len(coords)),
                    }
                )
        return out


BG = 0
FRAME_SIZES = (5, 7, 9)
PALETTE = tuple(range(1, 10))
GEN_ATTEMPTS = 400
PLACE_ATTEMPTS = 300


def _center_in_3x3(patch: np.ndarray, bg: int) -> np.ndarray:
    h, w = map(int, patch.shape)
    out = np.full((3, 3), int(bg), dtype=int)
    r0 = (3 - h) // 2
    c0 = (3 - w) // 2
    out[r0 : r0 + h, c0 : c0 + w] = patch
    return out


def _resize_to_3x3(motif: np.ndarray, bg: int) -> np.ndarray:
    h, w = map(int, motif.shape)
    row_edges = np.linspace(0, h, 4, dtype=int)
    col_edges = np.linspace(0, w, 4, dtype=int)
    out = np.full((3, 3), int(bg), dtype=int)
    for rr in range(3):
        for cc in range(3):
            block = motif[row_edges[rr] : row_edges[rr + 1], col_edges[cc] : col_edges[cc + 1]]
            vals = block[block != bg]
            if vals.size == 0:
                continue
            counts = Counter(map(int, vals.tolist()))
            best = max(counts.values())
            for value in vals.flat:
                if counts[int(value)] == best:
                    out[rr, cc] = int(value)
                    break
    return out


def _normalize_inner(interior: np.ndarray, bg: int) -> np.ndarray:
    box = bbox(interior != bg)
    assert box is not None, "expected a non-empty interior motif"
    motif = crop_bbox(interior, box)
    h, w = map(int, motif.shape)
    if h <= 3 and w <= 3:
        return _center_in_3x3(motif, bg)
    return _resize_to_3x3(motif, bg)


def _make_icon(frame_color: int, inner: np.ndarray, bg: int) -> np.ndarray:
    icon = np.full((5, 5), int(bg), dtype=int)
    icon[0, 1:4] = int(frame_color)
    icon[4, 1:4] = int(frame_color)
    icon[1:4, 0] = int(frame_color)
    icon[1:4, 4] = int(frame_color)
    icon[1:4, 1:4] = inner
    return icon


def _is_rounded_frame(grid: np.ndarray, box, color: int, bg: int) -> bool:
    r0, c0, r1, c1 = map(int, box)
    if r1 - r0 + 1 < 5 or c1 - c0 + 1 < 5:
        return False
    if int(grid[r0, c0]) != bg or int(grid[r0, c1]) != bg:
        return False
    if int(grid[r1, c0]) != bg or int(grid[r1, c1]) != bg:
        return False
    if not np.all(grid[r0, c0 + 1 : c1] == int(color)):
        return False
    if not np.all(grid[r1, c0 + 1 : c1] == int(color)):
        return False
    if not np.all(grid[r0 + 1 : r1, c0] == int(color)):
        return False
    if not np.all(grid[r0 + 1 : r1, c1] == int(color)):
        return False
    return True


def _extract_sprites(grid: np.ndarray) -> list[tuple[tuple[int, int, int, int], int]]:
    bg = int(get_bg(grid))
    candidates = []
    seen = set()
    for comp in find_components(grid, bg=bg, connectivity=8, monochromatic=True, return_format="dict"):
        color = int(comp["color"])
        box = tuple(map(int, comp["bbox"]))
        if box in seen:
            continue
        if _is_rounded_frame(grid, box, color, bg):
            candidates.append((box, color))
            seen.add(box)
    candidates.sort(key=lambda item: (item[0][0], item[0][1], item[1]))
    return candidates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Normalize scattered rounded-frame sprites to 5x5 icons, sort them by normalized interior density, and stack them vertically.

    Input structure:
    - A black background contains 4 to 6 non-overlapping rounded rectangular frame sprites.
    - Each sprite has one border color: the four corners of its bbox are black, and every other outer-border pixel has that border color.
    - Inside each rounded frame there is a non-empty motif made of non-black pixels, possibly using the border color or other colors.

    Rule:
    - Detect every rounded-frame sprite and crop its bounding box.
    - Remove the outer border, crop the interior motif to its own bbox, and normalize it to a 3x3 interior: motifs already at most 3x3 are centered, larger motifs are reduced to 3x3 by 3-by-3 occupancy binning.
    - Rebuild each sprite as a normalized 5x5 rounded-frame icon with the original border color and the normalized 3x3 interior.
    - Count the non-black cells in each normalized 3x3 interior, sort the icons from highest count to lowest, and stack them vertically with no gaps.

    Key insight:
    - The original sprites can have different physical sizes and different raw pixel totals, so the sortable quantity is the non-black count after every interior motif is first normalized to the same 3x3 reference frame.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    sprites = _extract_sprites(g)
    assert 4 <= len(sprites) <= 6, "expected 4 to 6 rounded-frame sprites"

    items = []
    for box, frame_color in sprites:
        patch = crop_bbox(g, box)
        inner = _normalize_inner(patch[1:-1, 1:-1], bg)
        icon = _make_icon(frame_color, inner, bg)
        items.append((int(np.count_nonzero(inner != bg)), box[0], box[1], icon))

    items.sort(key=lambda item: (-item[0], item[1], item[2]))
    return np.vstack([icon for _count, _r, _c, icon in items])


def _all_masks_by_count() -> tuple[dict[int, list[np.ndarray]], dict[int, list[np.ndarray]]]:
    all_masks = {count: [] for count in range(1, 10)}
    full_span = {count: [] for count in range(1, 10)}
    for bits in range(1, 1 << 9):
        mask = np.array([(bits >> k) & 1 for k in range(9)], dtype=int).reshape(3, 3)
        count = int(mask.sum())
        all_masks[count].append(mask)
        coords = np.argwhere(mask == 1)
        if coords[:, 0].min() == 0 and coords[:, 0].max() == 2 and coords[:, 1].min() == 0 and coords[:, 1].max() == 2:
            full_span[count].append(mask)
    return all_masks, full_span


_MASKS_BY_COUNT, _FULL_SPAN_MASKS = _all_masks_by_count()


def _pick_mask(count: int, rng: np.random.Generator, *, full_span: bool) -> np.ndarray:
    pool = _FULL_SPAN_MASKS[count] if full_span else _MASKS_BY_COUNT[count]
    if not pool:
        pool = _MASKS_BY_COUNT[count]
    idx = int(rng.integers(0, len(pool)))
    return pool[idx].copy()


def _assign_colors(mask: np.ndarray, frame_color: int, rng: np.random.Generator) -> np.ndarray:
    norm = np.zeros((3, 3), dtype=int)
    palette = [frame_color]
    others = [color for color in PALETTE if color != frame_color]
    rng.shuffle(others)
    palette.extend(others[: int(rng.integers(1, 3))])
    cells = np.argwhere(mask == 1)
    for r, c in cells:
        norm[r, c] = int(palette[int(rng.integers(0, len(palette)))])
    return norm


def _render_small_inner(count: int, inner_size: int, rng: np.random.Generator, colors: tuple[int, ...]) -> np.ndarray:
    if count == 1:
        crop = np.array([[colors[0]]], dtype=int)
    elif count == 2:
        if int(rng.integers(0, 2)) == 0:
            crop = np.array([[colors[0], colors[1 % len(colors)]]], dtype=int)
        else:
            crop = np.array([[colors[0]], [colors[1 % len(colors)]]], dtype=int)
    elif count == 3:
        crop = np.array([[colors[0], colors[1 % len(colors)]], [colors[2 % len(colors)], 0]], dtype=int)
    else:
        crop = np.array(
            [[colors[0], colors[1 % len(colors)]], [colors[2 % len(colors)], colors[3 % len(colors)]]],
            dtype=int,
        )

    out = np.zeros((inner_size, inner_size), dtype=int)
    ch, cw = crop.shape
    offsets = [(r, c) for r in range(inner_size - ch + 1) for c in range(inner_size - cw + 1)]
    rng.shuffle(offsets)
    r0, c0 = offsets[0]
    out[r0 : r0 + ch, c0 : c0 + cw] = crop
    return out


def _render_scaled_inner(
    norm: np.ndarray, inner_size: int, density: str, rng: np.random.Generator
) -> np.ndarray:
    out = np.zeros((inner_size, inner_size), dtype=int)
    row_edges = np.linspace(0, inner_size, 4, dtype=int)
    col_edges = np.linspace(0, inner_size, 4, dtype=int)
    for rr in range(3):
        for cc in range(3):
            color = int(norm[rr, cc])
            if color == 0:
                continue
            r0, r1 = int(row_edges[rr]), int(row_edges[rr + 1])
            c0, c1 = int(col_edges[cc]), int(col_edges[cc + 1])
            if density == "dense":
                out[r0:r1, c0:c1] = color
            elif density == "sparse":
                pr = r0 + (r1 - r0 - 1) // 2
                pc = c0 + (c1 - c0 - 1) // 2
                out[pr, pc] = color
            else:
                height = max(1, (r1 - r0 + 1) // 2)
                width = max(1, (c1 - c0 + 1) // 2)
                if int(rng.integers(0, 2)) == 0:
                    out[r0 : r0 + height, c0:c1] = color
                else:
                    out[r0:r1, c0 : c0 + width] = color
    return out


def _build_sprite(
    frame_color: int,
    size: int,
    mode: str,
    count: int,
    rng: np.random.Generator,
    *,
    full_span: bool = False,
    density: str = "medium",
) -> tuple[np.ndarray, np.ndarray]:
    inner_size = int(size - 2)
    colors = tuple([frame_color] + [color for color in PALETTE if color != frame_color][:4])
    if mode == "small":
        inner = _render_small_inner(count, inner_size, rng, colors)
    else:
        mask = _pick_mask(count, rng, full_span=full_span)
        norm = _assign_colors(mask, frame_color, rng)
        inner = _render_scaled_inner(norm, inner_size, density, rng)
    patch = np.zeros((size, size), dtype=int)
    patch[0, 1:-1] = frame_color
    patch[-1, 1:-1] = frame_color
    patch[1:-1, 0] = frame_color
    patch[1:-1, -1] = frame_color
    patch[1:-1, 1:-1] = inner
    normalized = _make_icon(frame_color, _normalize_inner(inner, BG), BG)
    return patch, normalized


def _place_sprites(
    patches: list[np.ndarray], rng: np.random.Generator
) -> tuple[np.ndarray, list[tuple[int, int]]] | None:
    n = len(patches)
    min_side = 22 + (n - 4)
    height = int(rng.integers(min_side, 31))
    width = int(rng.integers(min_side, 31))
    grid = np.zeros((height, width), dtype=int)
    blocked = np.zeros((height, width), dtype=bool)
    positions = [(-1, -1)] * n

    order = sorted(range(n), key=lambda idx: patches[idx].shape[0], reverse=True)
    for idx in order:
        patch = patches[idx]
        ph, pw = patch.shape
        placed = False
        for _ in range(PLACE_ATTEMPTS):
            r0 = int(rng.integers(0, height - ph + 1))
            c0 = int(rng.integers(0, width - pw + 1))
            rr0 = max(0, r0 - 1)
            rr1 = min(height, r0 + ph + 1)
            cc0 = max(0, c0 - 1)
            cc1 = min(width, c0 + pw + 1)
            if blocked[rr0:rr1, cc0:cc1].any():
                continue
            grid[r0 : r0 + ph, c0 : c0 + pw] = patch
            blocked[r0 : r0 + ph, c0 : c0 + pw] = True
            positions[idx] = (r0, c0)
            placed = True
            break
        if not placed:
            return None
    return grid, positions


def _sprite_raw_inner_count(patch: np.ndarray) -> int:
    return int(np.count_nonzero(patch[1:-1, 1:-1] != BG))


def _expected_stack(norm_icons: list[np.ndarray]) -> np.ndarray:
    items = []
    for idx, icon in enumerate(norm_icons):
        count = int(np.count_nonzero(icon[1:4, 1:4] != BG))
        items.append((count, idx, icon))
    items.sort(key=lambda item: (-item[0], item[1]))
    return np.vstack([icon for _count, _idx, icon in items])


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    n = int(rng.integers(4, 7))
    count_small = int(rng.integers(1, 5))
    dense_candidates = [count for count in range(2, 9) if count != count_small]
    if not dense_candidates:
        return None
    count_dense = int(rng.choice(dense_candidates))
    sparse_candidates = [count for count in range(count_dense + 1, 10) if count != count_small]
    if not sparse_candidates:
        return None
    remaining = [count for count in range(1, 10) if count not in {count_small, count_dense}]
    count_sparse = int(rng.choice(sparse_candidates))
    remaining = [count for count in remaining if count != count_sparse]
    extra = list(map(int, rng.choice(np.array(remaining), size=n - 3, replace=False)))
    counts = [count_small, count_dense, count_sparse] + extra
    rng.shuffle(counts)

    frame_colors = list(map(int, rng.choice(np.array(PALETTE), size=n, replace=False)))
    rng.shuffle(frame_colors)

    small_idx = counts.index(count_small)
    dense_idx = counts.index(count_dense)
    sparse_idx = counts.index(count_sparse)

    patches = []
    norm_icons = []
    modes = []
    sizes = []
    for idx, (count, frame_color) in enumerate(zip(counts, frame_colors)):
        if idx == small_idx:
            size = 5 if int(rng.integers(0, 2)) == 0 else 7
            patch, norm = _build_sprite(frame_color, size, "small", count, rng)
            mode = "small"
        elif idx == dense_idx:
            size = 9
            patch, norm = _build_sprite(frame_color, size, "scaled", count, rng, full_span=True, density="dense")
            mode = "scaled-dense"
        elif idx == sparse_idx:
            size = 7 if int(rng.integers(0, 2)) == 0 else 9
            patch, norm = _build_sprite(frame_color, size, "scaled", count, rng, full_span=True, density="sparse")
            mode = "scaled-sparse"
        else:
            size = int(rng.choice(np.array(FRAME_SIZES)))
            if size == 5 and int(rng.integers(0, 100)) < 45 and count <= 4:
                patch, norm = _build_sprite(frame_color, size, "small", count, rng)
                mode = "small"
            else:
                full_span = size > 5 and int(rng.integers(0, 100)) < 70
                density = "medium" if int(rng.integers(0, 100)) < 60 else "dense"
                patch, norm = _build_sprite(frame_color, size, "scaled", count, rng, full_span=full_span, density=density)
                mode = "scaled"
        patches.append(patch)
        norm_icons.append(norm)
        modes.append(mode)
        sizes.append(size)

    if 5 not in sizes or not any(size > 5 for size in sizes):
        return None

    placed = _place_sprites(patches, rng)
    if placed is None:
        return None
    grid, _positions = placed

    expected = _expected_stack(norm_icons)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None

    norm_counts = [int(np.count_nonzero(icon[1:4, 1:4] != BG)) for icon in norm_icons]
    raw_counts = [_sprite_raw_inner_count(patch) for patch in patches]
    norm_order = tuple(np.argsort([-count for count in norm_counts], kind="stable").tolist())
    raw_order = tuple(np.argsort([-count for count in raw_counts], kind="stable").tolist())
    area_order = tuple(np.argsort([-((size - 2) * (size - 2)) for size in sizes], kind="stable").tolist())
    if len(set(norm_counts)) != n:
        return None
    if raw_order == norm_order:
        return None
    if area_order == norm_order:
        return None
    if "small" not in modes or not any(mode.startswith("scaled") for mode in modes):
        return None
    if max(grid.shape) > 30 or expected.shape[0] > 30 or expected.shape[1] > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid normalized rounded-sprite sorting input")
