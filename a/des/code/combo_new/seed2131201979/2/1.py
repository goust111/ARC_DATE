from __future__ import annotations

import math

import numpy as np


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]] | np.ndarray) -> tuple[int, int, int, int]:
    arr = np.asarray(points, dtype=int)
    return (
        int(arr[:, 0].min()),
        int(arr[:, 1].min()),
        int(arr[:, 0].max()),
        int(arr[:, 1].max()),
    )


def _scale_mask(mask: np.ndarray, scale: int) -> np.ndarray:
    return np.repeat(np.repeat(mask, scale, axis=0), scale, axis=1)


def _find_frames(grid: np.ndarray, bg: int) -> list[dict[str, int | tuple[int, int, int, int]]]:
    frames: list[dict[str, int | tuple[int, int, int, int]]] = []
    for color0 in np.unique(grid):
        color = int(color0)
        if color == bg:
            continue
        for pts in _components(grid == color):
            r0, c0, r1, c1 = _bbox(pts)
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            if h < 3 or w < 3:
                continue

            border = np.zeros((h, w), dtype=bool)
            border[0, :] = True
            border[-1, :] = True
            border[:, 0] = True
            border[:, -1] = True

            sub = grid[r0 : r1 + 1, c0 : c1 + 1]
            if len(pts) != int(border.sum()):
                continue
            if not np.array_equal(sub == color, border):
                continue

            interior = sub[1:-1, 1:-1]
            non_bg = np.argwhere(interior != bg)
            if len(non_bg) != 1:
                continue
            if tuple(map(int, non_bg[0])) != (0, 0):
                continue
            seed_color = int(interior[0, 0])
            if seed_color in (bg, color):
                continue

            frames.append(
                {
                    "frame_color": color,
                    "seed_color": seed_color,
                    "bbox": (r0, c0, r1, c1),
                    "inner_h": h - 2,
                    "inner_w": w - 2,
                }
            )

    assert frames, "expected at least one seed-bearing rectangular frame"
    return sorted(frames, key=lambda item: tuple(item["bbox"]))  # type: ignore[arg-type]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each fenced interior with a scaled copy of the same-colored outside template.

    Input structure:
    - The background is black or another dominant single color.
    - Several monochrome hollow rectangular fences appear on the grid, each with exactly one 1x1 seed cell in the interior's top-left corner.
    - Outside the fences, each seed color also appears once as a small monochrome template patch cropped inside a 3x3 or 4x4 bounding box.

    Rule:
    - Detect every rectangular fence and read the interior seed color.
    - Find the outside template that uses that same color, crop its bounding box, and nearest-neighbor scale its mask by an integer factor so it exactly matches the fence interior size.
    - Clear the fence interior back to background and paint the scaled template there, leaving fence borders, outside templates, and background unchanged.

    Key insight:
    - The interior seed is only a color key: it selects an external template whose shape must be enlarged to the frame's interior dimensions.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    colors, counts = np.unique(g, return_counts=True)
    bg = int(colors[np.argmax(counts)])

    frames = _find_frames(g, bg)
    inside_any_frame = np.zeros(g.shape, dtype=bool)
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]  # type: ignore[misc]
        inside_any_frame[r0 : r1 + 1, c0 : c1 + 1] = True

    template_masks: dict[int, np.ndarray] = {}
    for seed_color in {int(frame["seed_color"]) for frame in frames}:
        coords = np.argwhere((g == seed_color) & ~inside_any_frame)
        assert len(coords) > 0, f"missing outside template for seed color {seed_color}"
        tr0, tc0, tr1, tc1 = _bbox(coords)
        template_masks[seed_color] = g[tr0 : tr1 + 1, tc0 : tc1 + 1] == seed_color

    out = g.copy()
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]  # type: ignore[misc]
        seed_color = int(frame["seed_color"])
        mask = template_masks[seed_color]
        inner_h = int(frame["inner_h"])
        inner_w = int(frame["inner_w"])
        th, tw = mask.shape
        assert inner_h % th == 0 and inner_w % tw == 0, "frame interior must be an integer scale of the template"
        scale_h = inner_h // th
        scale_w = inner_w // tw
        assert scale_h == scale_w and scale_h >= 1, "template must scale uniformly"
        scaled = _scale_mask(mask, scale_h)
        assert scaled.shape == (inner_h, inner_w)

        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        interior[:, :] = bg
        interior[scaled] = seed_color

    return out


GEN_BG = 0
GEN_FRAME_COUNT_CHOICES = (2, 3)
GEN_TEMPLATE_SIZES = (3, 4)
GEN_SCALE_OPTIONS = {3: (2, 3, 4), 4: (2, 3)}
GEN_EDGE_MARGIN = 1
GEN_CLEARANCE = 1
GEN_RETRIES = 400


def _random_template_mask(size: int, rng: np.random.Generator, seen: set[bytes]) -> np.ndarray | None:
    area = size * size
    lo = max(size + 1, math.ceil(area * 0.35))
    hi = min(area - 2, math.floor(area * 0.7))
    for _ in range(200):
        p = float(rng.uniform(0.35, 0.65))
        mask = rng.random((size, size)) < p
        count = int(mask.sum())
        if count < lo or count > hi:
            continue
        if not mask[0].any() or not mask[-1].any() or not mask[:, 0].any() or not mask[:, -1].any():
            continue
        sig = bytes([size]) + np.packbits(mask.astype(np.uint8)).tobytes()
        if sig in seen:
            continue
        seen.add(sig)
        return mask
    return None


def _template_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.full(mask.shape, GEN_BG, dtype=int)
    patch[mask] = color
    return patch


def _frame_input_patch(inner_h: int, inner_w: int, frame_color: int, seed_color: int) -> np.ndarray:
    patch = np.full((inner_h + 2, inner_w + 2), GEN_BG, dtype=int)
    patch[0, :] = frame_color
    patch[-1, :] = frame_color
    patch[:, 0] = frame_color
    patch[:, -1] = frame_color
    patch[1, 1] = seed_color
    return patch


def _frame_output_patch(mask: np.ndarray, scale: int, frame_color: int, seed_color: int) -> np.ndarray:
    inner = _scale_mask(mask, scale)
    inner_h, inner_w = inner.shape
    patch = np.full((inner_h + 2, inner_w + 2), GEN_BG, dtype=int)
    patch[0, :] = frame_color
    patch[-1, :] = frame_color
    patch[:, 0] = frame_color
    patch[:, -1] = frame_color
    patch[1:-1, 1:-1][inner] = seed_color
    return patch


def _can_place(used: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int) -> bool:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(used.shape[0], r0 + h + pad)
    cc1 = min(used.shape[1], c0 + w + pad)
    return not bool(used[rr0:rr1, cc0:cc1].any())


def _mark_used(used: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int) -> None:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(used.shape[0], r0 + h + pad)
    cc1 = min(used.shape[1], c0 + w + pad)
    used[rr0:rr1, cc0:cc1] = True


def _paste(canvas: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    sub = canvas[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]]
    mask = patch != GEN_BG
    sub[mask] = patch[mask]


def _pick_positions(used: np.ndarray, h: int, w: int, rng: np.random.Generator) -> tuple[int, int] | None:
    max_r = used.shape[0] - h - GEN_EDGE_MARGIN
    max_c = used.shape[1] - w - GEN_EDGE_MARGIN
    min_r = GEN_EDGE_MARGIN
    min_c = GEN_EDGE_MARGIN
    if min_r > max_r or min_c > max_c:
        return None
    coords = [(r, c) for r in range(min_r, max_r + 1) for c in range(min_c, max_c + 1)]
    order = rng.permutation(len(coords))
    for idx in order:
        r0, c0 = coords[int(idx)]
        if _can_place(used, r0, c0, h, w, GEN_CLEARANCE):
            return r0, c0
    return None


def _template_sizes(frame_count: int, rng: np.random.Generator) -> list[int]:
    sizes = [3, 4]
    while len(sizes) < frame_count:
        sizes.append(int(rng.choice(GEN_TEMPLATE_SIZES)))
    rng.shuffle(sizes)
    return sizes[:frame_count]


def _choose_canvas(
    object_shapes: list[tuple[int, int]],
    rng: np.random.Generator,
) -> tuple[int, int] | None:
    max_h = max(h for h, _ in object_shapes) + 2 * GEN_EDGE_MARGIN
    max_w = max(w for _, w in object_shapes) + 2 * GEN_EDGE_MARGIN
    reserved = sum((h + 2 * GEN_CLEARANCE) * (w + 2 * GEN_CLEARANCE) for h, w in object_shapes)
    target_area = int(math.ceil(reserved * float(rng.uniform(1.2, 1.45))))
    min_h = max(18, max_h)
    min_w = max(18, max_w)
    candidates = []
    for h in range(min_h, 31):
        need_w = max(min_w, (target_area + h - 1) // h)
        if need_w <= 30:
            candidates.append((h, need_w))
    if not candidates:
        return None
    base_h, need_w = candidates[int(rng.integers(0, len(candidates)))]
    grid_h = int(rng.integers(base_h, 31))
    low_w = max(need_w, min_w)
    if low_w > 30:
        return None
    grid_w = int(rng.integers(low_w, 31))
    return grid_h, grid_w


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    frame_count = int(rng.choice(GEN_FRAME_COUNT_CHOICES))
    palette = list(map(int, rng.choice(np.arange(1, 10), size=frame_count + 1, replace=False)))
    frame_color = palette[0]
    seed_colors = palette[1:]

    specs: list[dict[str, np.ndarray | int]] = []
    seen_masks: set[bytes] = set()
    for size, seed_color in zip(_template_sizes(frame_count, rng), seed_colors):
        mask = _random_template_mask(size, rng, seen_masks)
        if mask is None:
            return None
        scale = int(rng.choice(GEN_SCALE_OPTIONS[size]))
        template = _template_patch(mask, seed_color)
        frame_in = _frame_input_patch(size * scale, size * scale, frame_color, seed_color)
        frame_out = _frame_output_patch(mask, scale, frame_color, seed_color)
        specs.append(
            {
                "template_in": template,
                "template_out": template.copy(),
                "frame_in": frame_in,
                "frame_out": frame_out,
            }
        )

    object_shapes = []
    for spec in specs:
        template = spec["template_in"]
        frame = spec["frame_in"]
        assert isinstance(template, np.ndarray) and isinstance(frame, np.ndarray)
        object_shapes.append(tuple(map(int, frame.shape)))
        object_shapes.append(tuple(map(int, template.shape)))

    canvas = _choose_canvas(object_shapes, rng)
    if canvas is None:
        return None
    grid_h, grid_w = canvas
    inp = np.full((grid_h, grid_w), GEN_BG, dtype=int)
    out = np.full((grid_h, grid_w), GEN_BG, dtype=int)
    used = np.zeros((grid_h, grid_w), dtype=bool)

    objects: list[tuple[np.ndarray, np.ndarray, int, int]] = []
    for spec in specs:
        frame_in = spec["frame_in"]
        frame_out = spec["frame_out"]
        template_in = spec["template_in"]
        template_out = spec["template_out"]
        assert isinstance(frame_in, np.ndarray) and isinstance(frame_out, np.ndarray)
        assert isinstance(template_in, np.ndarray) and isinstance(template_out, np.ndarray)
        objects.append((frame_in, frame_out, int(frame_in.shape[0]), int(frame_in.shape[1])))
        objects.append((template_in, template_out, int(template_in.shape[0]), int(template_in.shape[1])))

    objects.sort(key=lambda item: -(item[2] * item[3]))
    for patch_in, patch_out, h, w in objects:
        pos = _pick_positions(used, h, w, rng)
        if pos is None:
            return None
        r0, c0 = pos
        _paste(inp, patch_in, r0, c0)
        _paste(out, patch_out, r0, c0)
        _mark_used(used, r0, c0, h, w, GEN_CLEARANCE)

    try:
        actual = main(inp)
    except Exception:
        return None
    if np.array_equal(actual, inp):
        return None
    if not np.array_equal(actual, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
