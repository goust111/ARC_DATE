from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
COLORS = tuple(range(1, 10))

GEN_RETRIES = 400
PLACEMENT_RETRIES = 250
TEMPLATE_RETRIES = 120


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _color_components(grid: np.ndarray, bg: int = BG) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []
    for r0 in range(h):
        for c0 in range(w):
            color = int(grid[r0, c0])
            if color == bg or seen[r0, c0]:
                continue
            q = deque([(r0, c0)])
            seen[r0, c0] = True
            coords: list[tuple[int, int]] = []
            while q:
                r, c = q.popleft()
                coords.append((r, c))
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            arr = np.array(coords, dtype=int)
            rmin = int(arr[:, 0].min())
            cmin = int(arr[:, 1].min())
            rmax = int(arr[:, 0].max())
            cmax = int(arr[:, 1].max())
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (rmin, cmin, rmax, cmax),
                    "touch_top": bool(np.any(arr[:, 0] == 0)),
                }
            )
    return comps


def _is_connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return False
    h, w = mask.shape
    start = tuple(map(int, pts[0]))
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == int(mask.sum())


def _full_border_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _component_mask(comp: dict[str, object]) -> np.ndarray:
    r0, c0, r1, c1 = comp["bbox"]  # type: ignore[index]
    mask = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=bool)
    coords = comp["coords"]  # type: ignore[index]
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _is_full_rectangle_frame(comp: dict[str, object]) -> bool:
    r0, c0, r1, c1 = comp["bbox"]  # type: ignore[index]
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 3 or w < 3:
        return False
    coords = comp["coords"]  # type: ignore[index]
    if int(coords.shape[0]) != 2 * (h + w) - 4:
        return False
    for r, c in coords:
        if int(r) not in (r0, r1) and int(c) not in (c0, c1):
            return False
    return True


def _sample_edge(edge: np.ndarray, new_len: int) -> np.ndarray:
    old_len = int(edge.shape[0])
    if old_len == new_len:
        return edge.copy()
    if new_len == 1:
        return edge[[0]].copy()
    idx = np.round(np.linspace(0, old_len - 1, new_len)).astype(int)
    return edge[idx]


def _scale_template(mask: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    out = np.zeros((target_h, target_w), dtype=bool)
    top = _sample_edge(mask[0, :], target_w)
    bottom = _sample_edge(mask[-1, :], target_w)
    left = _sample_edge(mask[:, 0], target_h)
    right = _sample_edge(mask[:, -1], target_h)
    out[0, :] |= top
    out[-1, :] |= bottom
    out[:, 0] |= left
    out[:, -1] |= right
    return out


def _clear_bbox_boundary(grid: np.ndarray, bbox: tuple[int, int, int, int], bg: int = BG) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = bg
    grid[r1, c0 : c1 + 1] = bg
    grid[r0 : r1 + 1, c0] = bg
    grid[r0 : r1 + 1, c1] = bg


def _marker_kind_count(grid: np.ndarray, bbox: tuple[int, int, int, int], border_color: int) -> int:
    r0, c0, r1, c1 = bbox
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    vals = set(map(int, np.unique(interior)))
    vals.discard(BG)
    vals.discard(int(border_color))
    return len(vals)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each rectangular frame border with the top template selected by the number of interior marker-color kinds.

    Input structure:
    - The background is black, with several single-color rectangular frames of thickness 1 scattered below the top band.
    - The inside of each frame may contain zero or more single-pixel marker dots whose colors differ from both the frame color and the background.
    - The top row contains a left-to-right sequence of small single-color template borders, each giving a different boundary shape.

    Rule:
    - Read all template borders that touch the top row and sort them from left to right; the first template means “1 marker-color kind”, the second means “2 kinds”, and so on.
    - For every non-template full rectangular frame, count how many distinct marker colors appear strictly inside it.
    - If that count is 0, leave the frame unchanged. Otherwise, scale the selected template's boundary pattern to the frame's bbox and redraw the frame border in the frame's own color.
    - Keep all interior marker dots unchanged.

    Key insight:
    - The mapping depends on the count of distinct interior colors, while the actual replacement shape is encoded only by the left-to-right order of the top templates.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    comps = _color_components(g, bg=BG)
    templates = [comp for comp in comps if bool(comp["touch_top"])]
    templates.sort(key=lambda comp: (comp["bbox"][1], comp["bbox"][0]))  # type: ignore[index]
    template_masks = [_component_mask(comp) for comp in templates]

    out = g.copy()
    for comp in comps:
        if bool(comp["touch_top"]):
            continue
        if not _is_full_rectangle_frame(comp):
            continue

        color = int(comp["color"])
        bbox = comp["bbox"]  # type: ignore[index]
        kinds = _marker_kind_count(g, bbox, color)
        if kinds <= 0 or kinds > len(template_masks):
            continue

        r0, c0, r1, c1 = bbox
        scaled = _scale_template(template_masks[kinds - 1], int(r1 - r0 + 1), int(c1 - c0 + 1))
        _clear_bbox_boundary(out, bbox, bg=BG)
        patch = out[r0 : r1 + 1, c0 : c1 + 1]
        patch[scaled] = color

    return out


def _mask_signature(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(v) for v in row) for row in mask.astype(int))


def _candidate_defect_cells(h: int, w: int) -> list[tuple[int, int]]:
    cells: set[tuple[int, int]] = {
        (0, 0),
        (0, w - 1),
        (h - 1, 0),
        (h - 1, w - 1),
    }
    top_cols = {1, max(1, w // 2), max(1, w - 2)}
    side_rows = {1, max(1, h // 2), max(1, h - 2)}
    for c in top_cols:
        if 0 < c < w - 1:
            cells.add((0, c))
            cells.add((h - 1, c))
    for r in side_rows:
        if 0 < r < h - 1:
            cells.add((r, 0))
            cells.add((r, w - 1))
    return list(cells)


def _make_template_mask(
    rng: random.Random,
    used_signatures: set[tuple[tuple[int, ...], ...]],
    must_change: bool,
) -> np.ndarray | None:
    for _ in range(TEMPLATE_RETRIES):
        h = rng.randint(4, 6)
        w = rng.randint(4, 6)
        mask = _full_border_mask(h, w)
        candidates = _candidate_defect_cells(h, w)
        rng.shuffle(candidates)
        min_remove = 1 if must_change else 0
        max_remove = min(4, len(candidates))
        target_remove = rng.randint(min_remove, max_remove)
        removed = 0
        for r, c in candidates:
            if removed >= target_remove:
                break
            trial = mask.copy()
            trial[r, c] = False
            if _is_connected(trial):
                mask = trial
                removed += 1
        if removed < min_remove:
            continue
        sig = _mask_signature(mask)
        if sig in used_signatures:
            continue
        used_signatures.add(sig)
        return mask
    return None


def _draw_mask(grid: np.ndarray, top: int, left: int, mask: np.ndarray, color: int) -> None:
    h, w = mask.shape
    patch = grid[top : top + h, left : left + w]
    patch[mask] = color


def _place_templates(
    grid: np.ndarray,
    rng: random.Random,
    template_masks: list[np.ndarray],
    template_colors: list[int],
) -> int:
    gaps = [rng.randint(1, 2) for _ in range(max(0, len(template_masks) - 1))]
    total_span = sum(mask.shape[1] for mask in template_masks) + sum(gaps)
    start_col = rng.randint(0, grid.shape[1] - total_span)
    c = start_col
    for i, mask in enumerate(template_masks):
        _draw_mask(grid, 0, c, mask, template_colors[i])
        c += mask.shape[1]
        if i < len(gaps):
            c += gaps[i]
    return max(mask.shape[0] for mask in template_masks)


def _frame_specs(rng: random.Random, template_count: int) -> list[dict[str, int]]:
    total_frames = template_count + 1 + rng.randint(0, 1)
    counts = list(range(template_count + 1))
    while len(counts) < total_frames:
        counts.append(rng.randint(0, template_count))
    rng.shuffle(counts)

    border_colors = rng.sample(COLORS, total_frames)
    specs: list[dict[str, int]] = []
    for border_color, kind_count in zip(border_colors, counts):
        extra_repeats = rng.randint(0, 2) if kind_count > 0 and rng.random() < 0.75 else 0
        min_markers = kind_count + extra_repeats
        for _ in range(50):
            h = rng.randint(4, 8)
            w = rng.randint(4, 9)
            if (h - 2) * (w - 2) >= max(1, min_markers):
                specs.append(
                    {
                        "h": h,
                        "w": w,
                        "border_color": border_color,
                        "kind_count": kind_count,
                        "extra_repeats": extra_repeats,
                        "area": h * w,
                    }
                )
                break
        else:
            return []
    return specs


def _paint_frame(
    grid: np.ndarray,
    top: int,
    left: int,
    h: int,
    w: int,
    color: int,
) -> None:
    grid[top, left : left + w] = color
    grid[top + h - 1, left : left + w] = color
    grid[top : top + h, left] = color
    grid[top : top + h, left + w - 1] = color


def _place_markers(
    grid: np.ndarray,
    rng: random.Random,
    top: int,
    left: int,
    h: int,
    w: int,
    border_color: int,
    kind_count: int,
    extra_repeats: int,
) -> None:
    if kind_count <= 0:
        return
    interior = [(r, c) for r in range(top + 1, top + h - 1) for c in range(left + 1, left + w - 1)]
    rng.shuffle(interior)
    marker_colors = rng.sample([c for c in COLORS if c != border_color], kind_count)
    idx = 0
    for color in marker_colors:
        r, c = interior[idx]
        grid[r, c] = color
        idx += 1
    for _ in range(extra_repeats):
        r, c = interior[idx]
        grid[r, c] = rng.choice(marker_colors)
        idx += 1


def _place_frames(
    grid: np.ndarray,
    rng: random.Random,
    specs: list[dict[str, int]],
    object_top: int,
) -> bool:
    h, w = grid.shape
    occupied = np.zeros((h, w), dtype=bool)
    occupied[:object_top, :] = True
    for spec in sorted(specs, key=lambda item: int(item["area"]), reverse=True):
        fh = int(spec["h"])
        fw = int(spec["w"])
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            r0 = rng.randint(object_top, h - fh)
            c0 = rng.randint(0, w - fw)
            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(h, r0 + fh + 1)
            cc1 = min(w, c0 + fw + 1)
            if occupied[rr0:rr1, cc0:cc1].any():
                continue
            _paint_frame(grid, r0, c0, fh, fw, int(spec["border_color"]))
            _place_markers(
                grid,
                rng,
                r0,
                c0,
                fh,
                fw,
                int(spec["border_color"]),
                int(spec["kind_count"]),
                int(spec["extra_repeats"]),
            )
            occupied[rr0:rr1, cc0:cc1] = True
            placed = True
            break
        if not placed:
            return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    template_count = rng.randint(2, 4)
    used_signatures: set[tuple[tuple[int, ...], ...]] = set()
    changed_index = rng.randrange(template_count)
    template_masks: list[np.ndarray] = []
    for i in range(template_count):
        mask = _make_template_mask(rng, used_signatures, must_change=(i == changed_index))
        if mask is None:
            return None
        template_masks.append(mask)

    gaps = [2] * max(0, template_count - 1)
    min_width = sum(mask.shape[1] for mask in template_masks) + sum(gaps)
    min_height = max(mask.shape[0] for mask in template_masks) + 10
    if min_width > 30 or min_height > 30:
        return None

    width = rng.randint(max(18, min_width), 30)
    height = rng.randint(max(18, min_height), 30)
    grid = np.zeros((height, width), dtype=int)

    template_colors = rng.sample(COLORS, template_count)
    band_h = _place_templates(grid, rng, template_masks, template_colors)
    object_top = band_h + 2

    specs = _frame_specs(rng, template_count)
    if not specs:
        return None
    if not _place_frames(grid, rng, specs, object_top):
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    if int(np.count_nonzero(grid == BG)) <= int(np.count_nonzero(grid != BG)):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
