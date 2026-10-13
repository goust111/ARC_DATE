from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
FRAME = 5
SEED = 7
SOURCE_COLORS = (1, 2, 3, 4)

BOTTOM_GAP_KINDS = (
    {"name": "single", "orient": "h", "length": 1},
    {"name": "h2", "orient": "h", "length": 2},
    {"name": "h3", "orient": "h", "length": 3},
)
VERTICAL_GAP_KINDS = (
    {"name": "v2", "orient": "v", "length": 2},
    {"name": "v3", "orient": "v", "length": 3},
)

MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(np.asarray(cells, dtype=int))
    components.sort(key=lambda coords: (int(coords[:, 0].min()), int(coords[:, 1].min())))
    return components


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _normalize_coords(coords: np.ndarray | list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    arr = np.asarray(coords, dtype=int)
    r0, c0 = arr.min(axis=0)
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in arr))


def _signature_size(signature: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    h = 1 + max(r for r, _ in signature)
    w = 1 + max(c for _, c in signature)
    return h, w


def _rect_border_coords(r0: int, c0: int, h: int, w: int) -> set[tuple[int, int]]:
    border = set()
    for c in range(c0, c0 + w):
        border.add((r0, c))
        border.add((r0 + h - 1, c))
    for r in range(r0, r0 + h):
        border.add((r, c0))
        border.add((r, c0 + w - 1))
    return border


def _frame_from_component(grid: np.ndarray, coords: np.ndarray) -> dict:
    r0, c0, r1, c1 = _bbox_coords(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    assert h >= 3 and w >= 4, "frames must have rectangular bboxes"

    border = _rect_border_coords(r0, c0, h, w)
    comp_set = {tuple(map(int, rc)) for rc in coords}
    assert comp_set <= border, "grey component must lie on a rectangular border"

    gap = sorted(border - comp_set)
    assert gap, "each frame must have a gap"
    gap_mask = np.zeros((h, w), dtype=bool)
    for rr, cc in gap:
        gap_mask[rr - r0, cc - c0] = True
    assert len(_components_mask(gap_mask)) == 1, "gap must be 4-connected"
    assert comp_set == border - set(gap), "frame must be exactly a border minus the gap"

    gap_arr = np.asarray(gap, dtype=int)
    gap_signature = _normalize_coords(gap_arr)
    return {
        "bbox": (r0, c0, r1, c1),
        "h": h,
        "w": w,
        "gap_coords": gap,
        "gap_signature": gap_signature,
        "gap_size": len(gap),
        "gap_rows": sorted({int(r) for r, _ in gap}),
        "gap_cols": sorted({int(c) for _, c in gap}),
        "is_bottom_gap": all(r == r1 for r, _ in gap),
    }


def _extract_frames(grid: np.ndarray) -> list[dict]:
    frames = [_frame_from_component(grid, coords) for coords in _components_mask(grid == FRAME)]
    assert 3 <= len(frames) <= 4, "expected 3-4 grey frames"
    return frames


def _extract_sources(grid: np.ndarray) -> list[dict]:
    source_mask = ~(np.isin(grid, np.asarray([BG, FRAME, SEED], dtype=int)))
    sources: list[dict] = []
    for coords in _components_mask(source_mask):
        colors = {int(grid[r, c]) for r, c in coords}
        assert len(colors) == 1, "each source shape must be monochrome"
        color = next(iter(colors))
        sources.append(
            {
                "color": color,
                "coords": [tuple(map(int, rc)) for rc in coords],
                "signature": _normalize_coords(coords),
            }
        )
    assert 3 <= len(sources) <= 4, "expected 3-4 colored source shapes"
    return sources


def _seed_columns(grid: np.ndarray) -> list[int]:
    h, _ = grid.shape
    orange = np.argwhere(grid == SEED)
    assert orange.size > 0, "expected bottom orange seed pixels"
    assert all(int(r) == h - 1 for r, _ in orange), "input orange pixels must all be bottom seeds"
    cols = sorted(int(c) for c in np.argwhere(grid[h - 1] == SEED).ravel())
    assert len(cols) == len(set(cols)), "seed columns must be unique"
    return cols


def _apply_fills_and_erase(
    grid: np.ndarray,
    frames: list[dict],
    sources: list[dict],
    fill_frame_ids: set[int],
    erase_sources: bool,
) -> np.ndarray:
    out = np.array(grid, copy=True)
    if erase_sources:
        for source in sources:
            for r, c in source["coords"]:
                out[r, c] = BG
    for frame_id in fill_frame_ids:
        for r, c in frames[frame_id]["gap_coords"]:
            out[r, c] = FRAME
    return out


def _shoot_rays(grid: np.ndarray) -> np.ndarray:
    out = np.array(grid, copy=True)
    h, _ = out.shape
    for c in _seed_columns(out):
        stop_row = None
        for r in range(h - 2, -1, -1):
            if int(out[r, c]) != BG:
                stop_row = r
                break
        start = 0 if stop_row is None else stop_row + 1
        out[start:h, c] = SEED
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the uniquely matched frame gaps, erase the source pieces, then cast upward orange rays that stop at the first remaining obstacle.

    Input structure:
    - The background is black, with 3-4 gray rectangular border frames in the upper half.
    - Each frame is missing one 4-connected gap on its border, and 3-4 small colored source shapes appear elsewhere.
    - Orange seed pixels sit on the bottom row; some seed columns align with frame gaps above.

    Rule:
    - Detect every gray frame and compute the exact gap shape as the missing cells of its rectangular border.
    - For each colored source shape, find the unique frame gap with the same shape under translation only; erase the source and fill that frame gap with gray.
    - After all fills, shoot an orange vertical ray upward from each bottom seed column until the first non-black pixel; stop just below that obstacle, or fill to the top edge if no obstacle is encountered.

    Key insight:
    - The colored pieces are not copied into the output directly: they only decide which frame gaps become solid, and that hidden fill changes where the later rays are blocked.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frames = _extract_frames(g)
    sources = _extract_sources(g)
    _seed_columns(g)

    frame_by_signature: dict[tuple[tuple[int, int], ...], int] = {}
    for frame_id, frame in enumerate(frames):
        sig = frame["gap_signature"]
        assert sig not in frame_by_signature, "frame gap shapes must be uniquely identifiable"
        frame_by_signature[sig] = frame_id

    fill_frame_ids: set[int] = set()
    for source in sources:
        sig = source["signature"]
        assert sig in frame_by_signature, "every source shape must match a frame gap"
        frame_id = frame_by_signature[sig]
        assert frame_id not in fill_frame_ids, "each frame gap can be matched at most once"
        fill_frame_ids.add(frame_id)

    prepared = _apply_fills_and_erase(g, frames, sources, fill_frame_ids, erase_sources=True)
    return _shoot_rays(prepared)


def _choose_frame_dims(kind: dict) -> tuple[int, int]:
    length = int(kind["length"])
    if kind["orient"] == "h":
        h = random.randint(3, 4 if random.random() < 0.75 else 5)
        min_w = max(4, length + 2)
        w = min_w + random.randint(0, 1 if min_w < 5 else 0)
    else:
        min_h = max(3, length + 2)
        h = min_h + random.randint(0, 1 if min_h < 4 else 0)
        w = 4 + random.randint(0, 1)
    return h, w


def _interior_gap_starts(span: int, length: int) -> list[int]:
    if span - length >= 2:
        return list(range(1, span - length))
    return list(range(0, span - length + 1))


def _assign_gap(frame: dict, kind: dict) -> None:
    r0 = int(frame["r0"])
    c0 = int(frame["c0"])
    h = int(frame["h"])
    w = int(frame["w"])
    length = int(kind["length"])
    if kind["orient"] == "h":
        start = random.choice(_interior_gap_starts(w, length))
        gap = [(r0 + h - 1, c0 + start + k) for k in range(length)]
    else:
        start = random.choice(_interior_gap_starts(h, length))
        edge_c = c0 if random.random() < 0.5 else c0 + w - 1
        gap = [(r0 + start + k, edge_c) for k in range(length)]
    frame["gap_coords"] = gap
    frame["gap_signature"] = _normalize_coords(gap)
    frame["gap_cols"] = sorted({c for _, c in gap})
    frame["is_bottom_gap"] = kind["orient"] == "h"
    frame["kind"] = kind["name"]


def _render_input_grid(scene: dict) -> np.ndarray:
    grid = np.zeros((scene["height"], scene["width"]), dtype=int)
    for frame in scene["frames"]:
        border = _rect_border_coords(frame["r0"], frame["c0"], frame["h"], frame["w"])
        for r, c in sorted(border - set(frame["gap_coords"])):
            grid[r, c] = FRAME
    for c in scene["seed_cols"]:
        grid[scene["height"] - 1, c] = SEED
    for source in scene["sources"]:
        for r, c in source["coords"]:
            grid[r, c] = source["color"]
    return grid


def _can_place_shape(grid: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    h, w = grid.shape
    cell_set = set(cells)
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if int(grid[r, c]) != BG:
            return False
    for r, c in cells:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if not (0 <= rr < h and 0 <= cc < w):
                    continue
                if (rr, cc) in cell_set:
                    continue
                if int(grid[rr, cc]) != BG:
                    return False
    return True


def _shape_cells(signature: tuple[tuple[int, int], ...], top: int, left: int) -> list[tuple[int, int]]:
    return [(top + r, left + c) for r, c in signature]


def _place_source(
    grid: np.ndarray,
    signature: tuple[tuple[int, int], ...],
    row_min: int,
    row_max: int,
    col_min: int,
    col_max: int,
    *,
    preferred_col: int | None = None,
) -> list[tuple[int, int]] | None:
    if row_min > row_max or col_min > col_max:
        return None
    shape_h, shape_w = _signature_size(signature)
    top_values = list(range(row_min, row_max - shape_h + 2))
    if not top_values:
        return None

    left_values = list(range(col_min, col_max - shape_w + 2))
    if preferred_col is None:
        random.shuffle(top_values)
        random.shuffle(left_values)
        for top in top_values:
            for left in left_values:
                cells = _shape_cells(signature, top, left)
                if _can_place_shape(grid, cells):
                    return cells
        return None

    anchor_cols = sorted({c for _, c in signature})
    top_pairs = [(top, anchor_c) for top in top_values for anchor_c in anchor_cols]
    random.shuffle(top_pairs)
    for top, anchor_c in top_pairs:
        left = preferred_col - anchor_c
        if left < col_min or left + shape_w - 1 > col_max:
            continue
        cells = _shape_cells(signature, top, left)
        if _can_place_shape(grid, cells):
            return cells
    return None


def _build_scene() -> dict | None:
    n_frames = 4 if random.random() < 0.78 else 3
    frame_kinds = list(BOTTOM_GAP_KINDS)
    if n_frames == 4:
        frame_kinds.append(random.choice(VERTICAL_GAP_KINDS))
    random.shuffle(frame_kinds)

    frames: list[dict] = []
    widths: list[int] = []
    for kind in frame_kinds:
        h, w = _choose_frame_dims(kind)
        if n_frames == 4:
            if kind["orient"] == "h":
                w = max(4, int(kind["length"]) + 2)
                h = max(3, min(h, 4))
            else:
                w = 4
                h = max(3, int(kind["length"]) + 2)
        frames.append({"h": h, "w": w})
        widths.append(w)

    left_margin = 1 if n_frames == 4 else random.randint(1, 2)
    gaps = [2 if n_frames == 4 else 2 + (1 if random.random() < 0.35 else 0) for _ in range(n_frames - 1)]
    reserve_right = 4 if n_frames == 4 else random.randint(5, 7)
    width_floor = left_margin + sum(widths) + sum(gaps) + reserve_right
    if width_floor > 29:
        return None
    width = width_floor + random.randint(0, 29 - width_floor)

    c = left_margin
    for i, frame in enumerate(frames):
        frame["c0"] = c
        c += frame["w"]
        if i < n_frames - 1:
            c += gaps[i]
    right_strip_start = c + 1

    for frame in frames:
        frame["r0"] = random.randint(1, 6)
    max_bottom = max(frame["r0"] + frame["h"] - 1 for frame in frames)
    height_min = max(19, max_bottom + 9)
    if height_min > 26:
        return None
    height = random.randint(height_min, 26)

    for frame, kind in zip(frames, frame_kinds):
        _assign_gap(frame, kind)

    seed_frame_ids = [i for i, kind in enumerate(frame_kinds) if kind["orient"] == "h"]
    assert len(seed_frame_ids) == 3
    seed_cols = []
    for frame_id in seed_frame_ids:
        seed_cols.append(random.choice(frames[frame_id]["gap_cols"]))

    if n_frames == 4 and random.random() < 0.68:
        unmatched_id = random.choice(seed_frame_ids)
        matched_frame_ids = [i for i in range(n_frames) if i != unmatched_id]
    else:
        unmatched_id = None
        matched_frame_ids = list(range(n_frames))
    n_sources = len(matched_frame_ids)

    colors = list(SOURCE_COLORS)
    random.shuffle(colors)
    frame_to_source: dict[int, dict] = {}
    sources: list[dict] = []
    for color, frame_id in zip(colors, matched_frame_ids):
        source = {
            "frame_id": frame_id,
            "signature": frames[frame_id]["gap_signature"],
            "color": color,
        }
        frame_to_source[frame_id] = source
        sources.append(source)

    scene = {
        "height": height,
        "width": width,
        "frames": frames,
        "seed_cols": seed_cols,
        "unmatched_id": unmatched_id,
        "sources": [],
    }
    base_grid = _render_input_grid(scene)
    source_row_min = max(frame["r0"] + frame["h"] for frame in frames) + 2
    source_row_max = height - 3
    if source_row_min > source_row_max:
        return None

    preferred_seed_col = max(seed_cols)
    special_source = next(
        (src for src in sources if frame_kinds[src["frame_id"]]["orient"] == "v"),
        max(sources, key=lambda src: len(src["signature"])),
    )

    placements: list[dict] = []
    special_cells = _place_source(
        base_grid,
        special_source["signature"],
        source_row_min,
        source_row_max,
        width // 2,
        width - 1,
        preferred_col=preferred_seed_col,
    )
    if special_cells is None:
        return None
    placements.append({**special_source, "coords": special_cells})
    for r, c in special_cells:
        base_grid[r, c] = special_source["color"]

    other_sources = [src for src in sources if src is not special_source]
    other_sources.sort(key=lambda src: (-len(src["signature"]), src["frame_id"]))
    for source in other_sources:
        cells = _place_source(
            base_grid,
            source["signature"],
            source_row_min,
            source_row_max,
            max(width // 2, right_strip_start),
            width - 1,
        )
        if cells is None:
            return None
        placements.append({**source, "coords": cells})
        for r, c in cells:
            base_grid[r, c] = source["color"]

    scene["sources"] = placements
    scene["input"] = _render_input_grid(scene)
    scene["matched_frame_ids"] = set(matched_frame_ids)
    scene["special_frame_id"] = special_source["frame_id"]
    scene["special_seed_col"] = preferred_seed_col
    return scene


def _variant_output(scene: dict, *, fill_frame_ids: set[int], erase_sources: bool) -> np.ndarray:
    prepared = _apply_fills_and_erase(
        scene["input"],
        scene["frames"],
        scene["sources"],
        fill_frame_ids,
        erase_sources=erase_sources,
    )
    return _shoot_rays(prepared)


def _validate_scene(scene: dict) -> bool:
    inp = scene["input"]
    frames = _extract_frames(inp)
    sources = _extract_sources(inp)
    assert len(frames) == len(scene["frames"])
    assert len(sources) == len(scene["sources"])

    correct = main(inp)
    direct = _variant_output(scene, fill_frame_ids=set(scene["matched_frame_ids"]), erase_sources=True)
    assert np.array_equal(correct, direct), "main() must match the generator's intended semantics"

    keep_sources = _variant_output(scene, fill_frame_ids=set(scene["matched_frame_ids"]), erase_sources=False)
    assert not np.array_equal(correct, keep_sources), "erasing source shapes must matter"

    no_fill = _variant_output(scene, fill_frame_ids=set(), erase_sources=True)
    assert not np.array_equal(correct, no_fill), "filling at least one matched gap must matter"

    if scene["unmatched_id"] is not None:
        fill_all = _variant_output(scene, fill_frame_ids=set(range(len(scene["frames"]))), erase_sources=True)
        assert not np.array_equal(correct, fill_all), "unmatched frames prevent a fill-all shortcut"

    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        scene = _build_scene()
        if scene is None:
            continue
        try:
            if _validate_scene(scene):
                return scene["input"]
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid complementary-fill ray puzzle")
