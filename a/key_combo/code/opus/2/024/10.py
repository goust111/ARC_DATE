from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

try:
    from arc_utils import Color, bbox, crop_bbox, find_components, paste
except Exception:
    class Color:
        BLACK = 0
        GREY = 5

    def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
        coords = np.argwhere(mask)
        if coords.size == 0:
            return None
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        return int(r0), int(c0), int(r1), int(c1)

    def crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
        r0, c0, r1, c1 = map(int, box)
        return np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int).copy()

    def paste(
        dst: np.ndarray,
        patch: np.ndarray,
        r0: int,
        c0: int,
        *,
        transparent: int | None = None,
        mask: np.ndarray | None = None,
        overwrite: bool = True,
        clip: bool = False,
    ) -> np.ndarray:
        r0 = int(r0)
        c0 = int(c0)
        h, w = map(int, patch.shape)
        if not clip:
            assert 0 <= r0 and 0 <= c0 and r0 + h <= dst.shape[0] and c0 + w <= dst.shape[1]
            rr0, cc0, rr1, cc1 = r0, c0, r0 + h, c0 + w
            pr0, pc0 = 0, 0
        else:
            rr0 = max(0, r0)
            cc0 = max(0, c0)
            rr1 = min(dst.shape[0], r0 + h)
            cc1 = min(dst.shape[1], c0 + w)
            if rr0 >= rr1 or cc0 >= cc1:
                return dst
            pr0 = rr0 - r0
            pc0 = cc0 - c0
        pr1 = pr0 + (rr1 - rr0)
        pc1 = pc0 + (cc1 - cc0)
        dst_view = dst[rr0:rr1, cc0:cc1]
        patch_view = patch[pr0:pr1, pc0:pc1]
        active = np.ones_like(patch_view, dtype=bool)
        if mask is not None:
            active &= mask[pr0:pr1, pc0:pc1].astype(bool)
        if transparent is not None:
            active &= patch_view != int(transparent)
            if not overwrite:
                active &= dst_view == int(transparent)
        dst_view[active] = patch_view[active]
        return dst

    def find_components(
        grid: np.ndarray,
        *,
        bg: int | None = None,
        connectivity: int = 4,
        monochromatic: bool = True,
        colors: tuple[int, ...] | list[int] | None = None,
        min_area: int = 1,
        return_format: str = "dict",
    ) -> list[dict]:
        assert monochromatic
        g = np.asarray(grid, dtype=int)
        if bg is None:
            values, counts = np.unique(g, return_counts=True)
            bg = int(values[np.argmax(counts)])
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if int(connectivity) == 8:
            steps += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        if colors is None:
            colors = [int(v) for v in np.unique(g) if int(v) != int(bg)]
        out: list[dict] = []
        for color in colors:
            seen = np.zeros_like(g, dtype=bool)
            for r in range(g.shape[0]):
                for c in range(g.shape[1]):
                    if seen[r, c] or int(g[r, c]) != int(color):
                        continue
                    stack = [(int(r), int(c))]
                    seen[r, c] = True
                    coords: list[tuple[int, int]] = []
                    while stack:
                        rr, cc = stack.pop()
                        coords.append((rr, cc))
                        for dr, dc in steps:
                            nr = rr + dr
                            nc = cc + dc
                            if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1] and not seen[nr, nc] and int(g[nr, nc]) == int(color):
                                seen[nr, nc] = True
                                stack.append((nr, nc))
                    if len(coords) < int(min_area):
                        continue
                    arr = np.array(coords, dtype=int)
                    r0, c0 = arr.min(axis=0)
                    r1, c1 = arr.max(axis=0)
                    out.append(
                        {
                            "color": int(color),
                            "coords": arr,
                            "bbox": (int(r0), int(c0), int(r1), int(c1)),
                            "area": int(arr.shape[0]),
                        }
                    )
        return out


BG = Color.BLACK
FRAME = Color.GREY
ICON_SIZE = 5
NORM_SIZE = 3
MAX_GENERATION_ATTEMPTS = 400
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
FRAME_SIZE_RANGE = (5, 9)
OBJECT_COUNT_RANGE = (3, 5)
FRAME_GAP = 2
ALLOWED_RAW_SHAPES = (
    (1, 2),
    (2, 1),
    (1, 3),
    (3, 1),
    (2, 2),
    (2, 3),
    (3, 2),
    (3, 3),
)
SHAPE_PROFILES = (
    ((3, 3), (1, 2), (2, 2)),
    ((3, 3), (2, 1), (2, 3)),
    ((3, 3), (3, 1), (2, 2)),
    ((3, 3), (1, 3), (3, 2)),
)
GEN_CALL_COUNT = 0


def _rounded_frame_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((int(h), int(w)), dtype=bool)
    mask[0, 1:-1] = True
    mask[-1, 1:-1] = True
    mask[1:-1, 0] = True
    mask[1:-1, -1] = True
    return mask


def _normalize_patch(patch: np.ndarray) -> np.ndarray:
    src = np.asarray(patch, dtype=int)
    if src.shape[0] > NORM_SIZE:
        start = (src.shape[0] - NORM_SIZE) // 2
        src = src[start : start + NORM_SIZE, :]
    if src.shape[1] > NORM_SIZE:
        start = (src.shape[1] - NORM_SIZE) // 2
        src = src[:, start : start + NORM_SIZE]
    out = np.zeros((NORM_SIZE, NORM_SIZE), dtype=int)
    r0 = (NORM_SIZE - src.shape[0]) // 2
    c0 = (NORM_SIZE - src.shape[1]) // 2
    out[r0 : r0 + src.shape[0], c0 : c0 + src.shape[1]] = src
    return out


def _icon_from_normalized(norm: np.ndarray) -> np.ndarray:
    icon = np.zeros((ICON_SIZE, ICON_SIZE), dtype=int)
    icon[_rounded_frame_mask(ICON_SIZE, ICON_SIZE)] = FRAME
    icon[1:-1, 1:-1] = np.asarray(norm, dtype=int)
    return icon


def _extract_frames(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    grey_components = find_components(
        g,
        bg=BG,
        connectivity=8,
        monochromatic=True,
        colors=[FRAME],
        return_format="dict",
    )
    frames: list[dict] = []
    for comp in grey_components:
        box = tuple(map(int, comp["bbox"]))
        patch = crop_bbox(g, box)
        h, w = map(int, patch.shape)
        ring = _rounded_frame_mask(h, w)
        if h < FRAME_SIZE_RANGE[0] or w < FRAME_SIZE_RANGE[0]:
            continue
        if int(comp["area"]) != int(ring.sum()):
            continue
        if not np.array_equal(patch == FRAME, ring):
            continue
        if not np.all(patch[~ring] != FRAME):
            continue
        corners = (int(patch[0, 0]), int(patch[0, -1]), int(patch[-1, 0]), int(patch[-1, -1]))
        if corners != (BG, BG, BG, BG):
            continue

        inner = patch[1:-1, 1:-1]
        pattern_box = bbox(inner != BG)
        assert pattern_box is not None, "each rounded frame must contain an interior pattern"
        raw_patch = crop_bbox(inner, pattern_box)
        normalized = _normalize_patch(raw_patch)
        frames.append(
            {
                "bbox": box,
                "outer_size": (h, w),
                "raw_patch": raw_patch,
                "normalized": normalized,
                "icon": _icon_from_normalized(normalized),
                "pixel_count": int(np.count_nonzero(raw_patch)),
            }
        )

    assert len(frames) == len(grey_components), "every grey component must be a rounded frame"
    assert OBJECT_COUNT_RANGE[0] <= len(frames) <= OBJECT_COUNT_RANGE[1], "expected 3 to 5 rounded frames"
    frames.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))
    top_lefts = [(item["bbox"][0], item["bbox"][1]) for item in frames]
    assert len(top_lefts) == len(set(top_lefts)), "frame top-left coordinates must be unique"
    return frames


def _assemble_gallery(records: list[dict]) -> np.ndarray:
    out = np.zeros((ICON_SIZE, ICON_SIZE * len(records)), dtype=int)
    for i, record in enumerate(records):
        paste(out, record["icon"], 0, ICON_SIZE * i)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract each rounded-frame sprite, normalize it, then reorder the icons by original position into a horizontal gallery.

    Input structure:
    - The background is black.
    - The foreground consists of 3 to 5 disjoint grey rounded rectangular frames whose corners stay black.
    - Each frame encloses a small non-black interior pattern whose cropped bbox is at most 3x3.

    Rule:
    - For each grey rounded frame, read only the non-black pixels strictly inside the border and crop them to their own bbox.
    - Normalize that crop to 3x3, wrap it in a new 5x5 rounded grey frame, and remember the donor frame's original top-left coordinate.
    - Sort the standardized icons by donor frame row first and column second, then place them left-to-right with no gaps starting at output cell (0, 0).

    Key insight:
    - The visible content comes from the cropped interior sprite, but the placement order comes from the pre-crop container location, so the solver must preserve positional metadata while normalizing away size differences.
    """
    frames = _extract_frames(np.asarray(grid, dtype=int))
    return _assemble_gallery(frames)


def _boxes_have_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = FRAME_GAP) -> bool:
    ar0, ac0, ar1, ac1 = map(int, a)
    br0, bc0, br1, bc1 = map(int, b)
    return (
        ar1 + gap < br0
        or br1 + gap < ar0
        or ac1 + gap < bc0
        or bc1 + gap < ac0
    )


def _sample_outer_sizes(n: int) -> list[tuple[int, int]]:
    max_dim = 7 if n >= 5 else 8 if n == 4 else 9
    pool = [
        (h, w)
        for h in range(FRAME_SIZE_RANGE[0], max_dim + 1)
        for w in range(FRAME_SIZE_RANGE[0], max_dim + 1)
    ]
    for _ in range(200):
        sizes = random.sample(pool, n)
        if len({size for size in sizes}) < 2:
            continue
        if not any(h != w for h, w in sizes):
            continue
        return sizes
    raise RuntimeError("failed to sample diverse outer sizes")


def _sample_canvas_size(n: int, sizes: list[tuple[int, int]]) -> tuple[int, int]:
    base = 20 + 2 * (n - 3)
    min_h = max(base, max(h for h, _w in sizes) + 3)
    min_w = max(base, max(w for _h, w in sizes) + 3)
    return random.randint(min_h, 30), random.randint(min_w, 30)


def _place_boxes(height: int, width: int, sizes: list[tuple[int, int]]) -> list[tuple[int, int, int, int]] | None:
    order = sorted(range(len(sizes)), key=lambda idx: sizes[idx][0] * sizes[idx][1], reverse=True)
    boxes: list[tuple[int, int, int, int] | None] = [None] * len(sizes)
    for idx in order:
        h, w = sizes[idx]
        placed = False
        for _ in range(300):
            r0 = random.randint(0, height - h)
            c0 = random.randint(0, width - w)
            candidate = (r0, c0, r0 + h - 1, c0 + w - 1)
            if all(other is None or _boxes_have_gap(candidate, other) for other in boxes):
                boxes[idx] = candidate
                placed = True
                break
        if not placed:
            return None
    return [box for box in boxes if box is not None]


def _sample_raw_shapes(n: int, call_index: int) -> list[tuple[int, int]]:
    profile = list(SHAPE_PROFILES[call_index % len(SHAPE_PROFILES)])
    shapes = profile[: min(len(profile), n)]
    while len(shapes) < n:
        shapes.append(random.choice(ALLOWED_RAW_SHAPES))
    random.shuffle(shapes)
    return shapes


def _sample_pattern(shape: tuple[int, int], used: set[bytes], *, min_colors: int = 1) -> tuple[np.ndarray, np.ndarray, int]:
    h, w = map(int, shape)
    cells = [(r, c) for r in range(h) for c in range(w)]
    mandatory = {(0, 0), (h - 1, w - 1)}
    free = [cell for cell in cells if cell not in mandatory]
    max_pixels = min(7, h * w)
    for _ in range(300):
        count = random.randint(2, max_pixels)
        chosen = set(mandatory)
        extra = count - len(chosen)
        if extra > 0:
            chosen.update(random.sample(free, extra))
        color_count = random.randint(min(min_colors, count), min(3, count))
        colors = random.sample(PALETTE, color_count)
        assignments = [colors[i % color_count] for i in range(count)]
        random.shuffle(assignments)
        patch = np.zeros((h, w), dtype=int)
        for (r, c), color in zip(sorted(chosen), assignments):
            patch[r, c] = int(color)
        normalized = _normalize_patch(patch)
        sig = normalized.tobytes()
        if sig in used:
            continue
        used.add(sig)
        return patch, normalized, color_count
    return None, None, 0


def _center_offset(inner_h: int, inner_w: int, raw_h: int, raw_w: int) -> tuple[int, int]:
    return ((inner_h - raw_h) // 2, (inner_w - raw_w) // 2)


def _sample_offset(
    inner_h: int,
    inner_w: int,
    raw_h: int,
    raw_w: int,
    *,
    prefer_off_center: bool,
) -> tuple[int, int]:
    rows = list(range(inner_h - raw_h + 1))
    cols = list(range(inner_w - raw_w + 1))
    choices = [(r, c) for r in rows for c in cols]
    center = _center_offset(inner_h, inner_w, raw_h, raw_w)
    if prefer_off_center:
        off_center = [choice for choice in choices if choice != center]
        if off_center:
            return random.choice(off_center)
    return random.choice(choices)


def _draw_frame(grid: np.ndarray, box: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = map(int, box)
    grid[r0, c0 + 1 : c1] = FRAME
    grid[r1, c0 + 1 : c1] = FRAME
    grid[r0 + 1 : r1, c0] = FRAME
    grid[r0 + 1 : r1, c1] = FRAME


def _render_scene(height: int, width: int, specs: list[dict]) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    for spec in specs:
        _draw_frame(grid, spec["bbox"])
        r0, c0, _r1, _c1 = spec["bbox"]
        off_r, off_c = spec["offset"]
        paste(
            grid,
            spec["raw_patch"],
            r0 + 1 + off_r,
            c0 + 1 + off_c,
            transparent=BG,
        )
    return grid


def _gallery_from_specs(specs: list[dict], key) -> np.ndarray:
    ordered = sorted(specs, key=key)
    icons = [{"icon": spec["icon"]} for spec in ordered]
    return _assemble_gallery(icons)


def _validate_scene(grid: np.ndarray, specs: list[dict]) -> bool:
    expected = _gallery_from_specs(specs, key=lambda spec: (spec["bbox"][0], spec["bbox"][1]))
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False

    try:
        detected = _extract_frames(grid)
    except AssertionError:
        return False

    ordered_specs = sorted(specs, key=lambda spec: (spec["bbox"][0], spec["bbox"][1]))
    if len(detected) != len(ordered_specs):
        return False
    for detected_frame, spec in zip(detected, ordered_specs):
        if tuple(detected_frame["bbox"]) != tuple(spec["bbox"]):
            return False
        if not np.array_equal(detected_frame["raw_patch"], spec["raw_patch"]):
            return False
        if not np.array_equal(detected_frame["normalized"], spec["normalized"]):
            return False

    raw_shapes = [tuple(spec["raw_patch"].shape) for spec in specs]
    if (3, 3) not in raw_shapes:
        return False
    if not any(shape != (3, 3) for shape in raw_shapes):
        return False

    if len({spec["normalized"].tobytes() for spec in specs}) != len(specs):
        return False
    if len({spec["outer_size"] for spec in specs}) < 2:
        return False
    if not any(h != w for h, w in (spec["outer_size"] for spec in specs)):
        return False
    if not any(spec["color_count"] >= 2 for spec in specs):
        return False
    if not any(spec["offset"] != spec["center_offset"] for spec in specs if spec["can_shift"]):
        return False

    correct = expected
    if np.array_equal(correct, _gallery_from_specs(specs, key=lambda spec: (spec["bbox"][1], spec["bbox"][0]))):
        return False
    if np.array_equal(
        correct,
        _gallery_from_specs(
            specs,
            key=lambda spec: (
                spec["outer_size"][0] * spec["outer_size"][1],
                spec["bbox"][0],
                spec["bbox"][1],
            ),
        ),
    ):
        return False
    if np.array_equal(
        correct,
        _gallery_from_specs(
            specs,
            key=lambda spec: (
                spec["pixel_count"],
                spec["bbox"][0],
                spec["bbox"][1],
            ),
        ),
    ):
        return False
    return True


def _gen_attempt(call_index: int) -> np.ndarray | None:
    n = random.randint(*OBJECT_COUNT_RANGE)
    sizes = _sample_outer_sizes(n)
    canvas_h, canvas_w = _sample_canvas_size(n, sizes)
    boxes = _place_boxes(canvas_h, canvas_w, sizes)
    if boxes is None:
        return None

    shapes = _sample_raw_shapes(n, call_index)
    used_norms: set[bytes] = set()
    specs: list[dict] = []

    shiftable = [
        idx
        for idx, ((h, w), shape) in enumerate(zip(sizes, shapes))
        if (h - 2 > shape[0]) or (w - 2 > shape[1])
    ]
    if not shiftable:
        return None
    forced_off_center = random.choice(shiftable)
    forced_multicolor = random.randrange(n)

    for idx, (box, outer_size, shape) in enumerate(zip(boxes, sizes, shapes)):
        min_colors = 2 if idx == forced_multicolor else 1
        raw_patch, normalized, color_count = _sample_pattern(shape, used_norms, min_colors=min_colors)
        if raw_patch is None:
            return None

        outer_h, outer_w = outer_size
        inner_h = outer_h - 2
        inner_w = outer_w - 2
        raw_h, raw_w = map(int, raw_patch.shape)
        can_shift = (inner_h - raw_h) > 0 or (inner_w - raw_w) > 0
        offset = _sample_offset(
            inner_h,
            inner_w,
            raw_h,
            raw_w,
            prefer_off_center=idx == forced_off_center,
        )
        center = _center_offset(inner_h, inner_w, raw_h, raw_w)

        specs.append(
            {
                "bbox": box,
                "outer_size": outer_size,
                "raw_patch": raw_patch,
                "normalized": normalized,
                "icon": _icon_from_normalized(normalized),
                "pixel_count": int(np.count_nonzero(raw_patch)),
                "color_count": int(color_count),
                "offset": offset,
                "center_offset": center,
                "can_shift": bool(can_shift),
            }
        )

    grid = _render_scene(canvas_h, canvas_w, specs)
    if not _validate_scene(grid, specs):
        return None
    return grid


def generate_input() -> np.ndarray:
    global GEN_CALL_COUNT
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(GEN_CALL_COUNT)
        GEN_CALL_COUNT += 1
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid rounded-gallery ARC input")
