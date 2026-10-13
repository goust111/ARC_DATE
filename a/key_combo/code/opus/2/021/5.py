from __future__ import annotations

import itertools
import random

import numpy as np


BBox = tuple[int, int, int, int]


def _bbox_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: BBox) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int).copy()


def _find_components(
    grid: np.ndarray,
    *,
    bg: int = 0,
    monochromatic: bool,
) -> list[dict]:
    h, w = map(int, grid.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue

            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if nr < 0 or nr >= h or nc < 0 or nc >= w or seen[nr, nc]:
                        continue
                    if int(grid[nr, nc]) == bg:
                        continue
                    if monochromatic and int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))

            arr = np.array(coords, dtype=int)
            comps.append(
                {
                    "color": color if monochromatic else None,
                    "coords": arr,
                    "bbox": _bbox_coords(arr),
                    "area": int(arr.shape[0]),
                }
            )

    return comps


def _component_crop(grid: np.ndarray, comp: dict, *, bg: int = 0) -> np.ndarray:
    r0, c0, r1, c1 = comp["bbox"]
    out = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
    coords = np.asarray(comp["coords"], dtype=int)
    local = coords - np.array([r0, c0], dtype=int)
    out[local[:, 0], local[:, 1]] = grid[coords[:, 0], coords[:, 1]]
    return out


def _steps_from_starts(starts: list[int], total: int) -> list[int]:
    steps: list[int] = []
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else total
        steps.append(int(end - start))
    return steps


def _compress_sprite(sprite: np.ndarray, *, bg: int = 0) -> tuple[np.ndarray, list[int], list[int]]:
    parts = _find_components(sprite, bg=bg, monochromatic=True)
    assert parts, "sprite must contain monochrome partitions"

    row_starts = sorted({int(part["bbox"][0]) for part in parts})
    col_starts = sorted({int(part["bbox"][1]) for part in parts})
    row_heights = _steps_from_starts(row_starts, int(sprite.shape[0]))
    col_widths = _steps_from_starts(col_starts, int(sprite.shape[1]))
    assert 3 <= len(row_starts) <= 5 and 3 <= len(col_starts) <= 5, "compressed skeleton must be 3x3..5x5"

    part_lookup = {(int(part["bbox"][0]), int(part["bbox"][1])): part for part in parts}
    assert len(part_lookup) == len(parts), "partition top-lefts must be unique"
    assert len(parts) == len(row_starts) * len(col_starts), "partitions must tile a full row/column grid"

    skeleton = np.full((len(row_starts), len(col_starts)), int(bg), dtype=int)
    for i, r0 in enumerate(row_starts):
        for j, c0 in enumerate(col_starts):
            assert (r0, c0) in part_lookup, "every partition-grid cell needs one monochrome rectangle"
            part = part_lookup[(r0, c0)]
            exp_bbox = (
                int(r0),
                int(c0),
                int(r0 + row_heights[i] - 1),
                int(c0 + col_widths[j] - 1),
            )
            assert tuple(map(int, part["bbox"])) == exp_bbox, "partition bboxes must align to the inferred row/column grid"
            assert int(part["area"]) == row_heights[i] * col_widths[j], "each partition must be a filled rectangle"
            skeleton[i, j] = int(sprite[r0, c0])
            assert skeleton[i, j] != bg, "partition samples must be foreground"

    return skeleton, row_heights, col_widths


def _analyze_grid(grid: np.ndarray, *, bg: int = 0) -> list[dict]:
    infos: list[dict] = []
    for comp in _find_components(grid, bg=bg, monochromatic=False):
        crop = _component_crop(grid, comp, bg=bg)
        skeleton, row_heights, col_widths = _compress_sprite(crop, bg=bg)
        infos.append(
            {
                "bbox": tuple(map(int, comp["bbox"])),
                "coords": np.asarray(comp["coords"], dtype=int),
                "crop": crop,
                "skeleton": skeleton,
                "row_heights": row_heights,
                "col_widths": col_widths,
                "symmetric": bool(np.array_equal(skeleton, np.fliplr(skeleton))),
                "color_count": len(set(map(int, skeleton.ravel()))),
            }
        )
    assert 3 <= len(infos) <= 4, "expected 3 or 4 partition sprites"
    return infos


def _selected_index(infos: list[dict]) -> int:
    winners = [i for i, info in enumerate(infos) if info["symmetric"]]
    assert len(winners) == 1, "exactly one compressed skeleton must be left-right symmetric"
    return int(winners[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the original sprite whose partition-sampled skeleton is uniquely left-right symmetric.

    Input structure:
    - Black background contains 3 or 4 separated multicolor sprites.
    - Each sprite is a full rectangular tiling of single-color rectangular partitions with unequal cell sizes.
    - Adjacent partitions differ in color, so each partition is one 4-connected monochrome component.

    Rule:
    - For each sprite, collect every monochrome partition, sort their top-left row/column coordinates, and treat those starts as the partition grid.
    - Sample one color at each partition top-left to build the compressed skeleton, then test whether that small grid equals its left-right mirror.
    - Find the unique symmetric skeleton and return the original-scale sprite cropped to its bounding box, with background kept black.

    Key insight:
    - The hidden property is not visible in raw pixels because unequal partition widths distort the picture; symmetry only appears after every partition is reduced to one sampled pixel.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    infos = _analyze_grid(g, bg=0)
    return infos[_selected_index(infos)]["crop"].copy()


_COLOR_POOL = tuple(range(1, 10))
_AXIS_CANDIDATES = {
    count: [
        seq
        for seq in itertools.product((1, 2, 3), repeat=count)
        if 6 <= sum(seq) <= 10 and len(set(seq)) > 1
    ]
    for count in range(3, 6)
}
_CANVAS_SIDE_RANGE = (24, 30)
_MAX_GENERATION_ATTEMPTS = 400


def _sample_axis_sizes(count: int, *, non_palindrome: bool = False) -> list[int]:
    candidates = _AXIS_CANDIDATES[int(count)]
    if non_palindrome:
        candidates = [seq for seq in candidates if list(seq) != list(seq)[::-1]]
    assert candidates, "axis candidates must exist"
    return list(random.choice(candidates))


def _usable_colors(k: int) -> list[int]:
    return random.sample(list(_COLOR_POOL), int(k))


def _is_proper_coloring(skeleton: np.ndarray) -> bool:
    rows, cols = map(int, skeleton.shape)
    for r in range(rows):
        for c in range(cols):
            if r + 1 < rows and int(skeleton[r, c]) == int(skeleton[r + 1, c]):
                return False
            if c + 1 < cols and int(skeleton[r, c]) == int(skeleton[r, c + 1]):
                return False
    return True


def _generate_symmetric_skeleton(rows: int, cols: int, palette: list[int]) -> np.ndarray:
    assert cols % 2 == 1, "left-right symmetry with distinct adjacent cells needs an odd column count"
    grid = np.full((rows, cols), -1, dtype=int)
    cells = [(r, c) for r in range(rows) for c in range(cols // 2 + 1)]

    def can_paint(positions: list[tuple[int, int]], color: int) -> bool:
        for rr, cc in positions:
            for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                if nr < 0 or nr >= rows or nc < 0 or nc >= cols or (nr, nc) in positions:
                    continue
                if int(grid[nr, nc]) == color:
                    return False
        return True

    def backtrack(idx: int) -> bool:
        if idx == len(cells):
            return True
        r, c = cells[idx]
        mc = cols - 1 - c
        positions = [(r, c)] if mc == c else [(r, c), (r, mc)]
        options = palette[:]
        random.shuffle(options)
        for color in options:
            if not can_paint(positions, int(color)):
                continue
            for rr, cc in positions:
                grid[rr, cc] = int(color)
            if backtrack(idx + 1):
                return True
            for rr, cc in positions:
                grid[rr, cc] = -1
        return False

    for _ in range(120):
        grid.fill(-1)
        if backtrack(0) and len(set(map(int, grid.ravel()))) == len(palette):
            return grid.copy()
    raise RuntimeError("failed to generate a symmetric partition skeleton")


def _generate_asymmetric_skeleton(rows: int, cols: int, palette: list[int]) -> np.ndarray:
    grid = np.full((rows, cols), -1, dtype=int)
    cells = [(r, c) for r in range(rows) for c in range(cols)]

    def backtrack(idx: int) -> bool:
        if idx == len(cells):
            return True
        r, c = cells[idx]
        blocked = set()
        if r > 0 and grid[r - 1, c] != -1:
            blocked.add(int(grid[r - 1, c]))
        if c > 0 and grid[r, c - 1] != -1:
            blocked.add(int(grid[r, c - 1]))
        options = [color for color in palette if int(color) not in blocked]
        random.shuffle(options)
        for color in options:
            grid[r, c] = int(color)
            if backtrack(idx + 1):
                return True
        grid[r, c] = -1
        return False

    for _ in range(120):
        grid.fill(-1)
        if not backtrack(0):
            continue
        if len(set(map(int, grid.ravel()))) != len(palette):
            continue
        if np.array_equal(grid, np.fliplr(grid)):
            continue
        return grid.copy()
    raise RuntimeError("failed to generate an asymmetric partition skeleton")


def _break_lr_symmetry(skeleton: np.ndarray, palette: list[int]) -> np.ndarray:
    rows, cols = map(int, skeleton.shape)
    candidates = [(r, c) for r in range(rows) for c in range(cols // 2)]
    random.shuffle(candidates)
    for r, c in candidates:
        blocked = {
            int(skeleton[r, c]),
            int(skeleton[r, cols - 1 - c]),
        }
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < rows and 0 <= nc < cols:
                blocked.add(int(skeleton[nr, nc]))
        options = [color for color in palette if int(color) not in blocked]
        random.shuffle(options)
        for color in options:
            out = skeleton.copy()
            out[r, c] = int(color)
            if _is_proper_coloring(out) and not np.array_equal(out, np.fliplr(out)):
                if len(set(map(int, out.ravel()))) == len(palette):
                    return out
    raise RuntimeError("failed to perturb a symmetric skeleton into a near-symmetric distractor")


def _render_sprite(skeleton: np.ndarray, row_heights: list[int], col_widths: list[int]) -> np.ndarray:
    h = sum(row_heights)
    w = sum(col_widths)
    sprite = np.zeros((h, w), dtype=int)
    r0 = 0
    for i, rh in enumerate(row_heights):
        c0 = 0
        for j, cw in enumerate(col_widths):
            sprite[r0 : r0 + rh, c0 : c0 + cw] = int(skeleton[i, j])
            c0 += cw
        r0 += rh
    return sprite


def _mirror_mismatch_count(skeleton: np.ndarray) -> int:
    return int(np.count_nonzero(skeleton != np.fliplr(skeleton)))


def _make_sprite_record(
    skeleton: np.ndarray,
    row_heights: list[int],
    col_widths: list[int],
    *,
    kind: str,
) -> dict:
    sprite = _render_sprite(skeleton, row_heights, col_widths)
    return {
        "kind": kind,
        "skeleton": np.asarray(skeleton, dtype=int),
        "row_heights": list(map(int, row_heights)),
        "col_widths": list(map(int, col_widths)),
        "sprite": sprite,
        "shape": sprite.shape,
        "color_count": len(set(map(int, skeleton.ravel()))),
        "mismatch_count": _mirror_mismatch_count(skeleton),
    }


def _build_records() -> list[dict]:
    target_rows = random.randint(3, 5)
    target_cols = random.choice((3, 5))
    row_heights = _sample_axis_sizes(target_rows)
    col_widths = _sample_axis_sizes(target_cols, non_palindrome=True)
    palette_size = random.randint(3, 5)

    for _ in range(160):
        target_palette = _usable_colors(palette_size)
        target_skeleton = _generate_symmetric_skeleton(target_rows, target_cols, target_palette)
        target = _make_sprite_record(target_skeleton, row_heights, col_widths, kind="target")
        if not np.array_equal(target["sprite"], np.fliplr(target["sprite"])):
            break
    else:
        raise RuntimeError("failed to make a compressed-only symmetric target sprite")

    foil_palette = _usable_colors(palette_size)
    foil_skeleton = _generate_asymmetric_skeleton(target_rows, target_cols, foil_palette)
    foil = _make_sprite_record(foil_skeleton, row_heights, col_widths, kind="foil")

    near_rows = random.randint(3, 5)
    near_cols = random.choice((3, 5))
    near_row_heights = _sample_axis_sizes(near_rows)
    near_col_widths = _sample_axis_sizes(near_cols)
    near_palette = _usable_colors(random.randint(3, 5))
    near_base = _generate_symmetric_skeleton(near_rows, near_cols, near_palette)
    near_skeleton = _break_lr_symmetry(near_base, near_palette)
    near = _make_sprite_record(near_skeleton, near_row_heights, near_col_widths, kind="near")

    records = [target, foil, near]
    if random.random() < 0.55:
        extra_rows = random.randint(3, 5)
        extra_cols = random.randint(3, 5)
        extra_row_heights = _sample_axis_sizes(extra_rows)
        extra_col_widths = _sample_axis_sizes(extra_cols)
        extra_palette = _usable_colors(random.randint(3, 5))
        extra_skeleton = _generate_asymmetric_skeleton(extra_rows, extra_cols, extra_palette)
        records.append(_make_sprite_record(extra_skeleton, extra_row_heights, extra_col_widths, kind="extra"))

    random.shuffle(records)
    return records


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = map(int, patch.shape)
    dst[r0 : r0 + h, c0 : c0 + w] = patch


def _place_records(records: list[dict]) -> np.ndarray:
    h = random.randint(*_CANVAS_SIDE_RANGE)
    w = random.randint(*_CANVAS_SIDE_RANGE)
    grid = np.zeros((h, w), dtype=int)

    quadrants = [
        (1, 1, h // 2 - 2, w // 2 - 2),
        (1, w // 2 + 1, h // 2 - 2, w - 2),
        (h // 2 + 1, 1, h - 2, w // 2 - 2),
        (h // 2 + 1, w // 2 + 1, h - 2, w - 2),
    ]
    slots = random.sample(range(4), len(records))

    for record, slot in zip(records, slots):
        r0, c0, r1, c1 = quadrants[slot]
        patch = record["sprite"]
        ph, pw = map(int, patch.shape)
        space_h = int(r1 - r0 + 1)
        space_w = int(c1 - c0 + 1)
        if ph > space_h or pw > space_w:
            raise RuntimeError("quadrant too small for sprite placement")
        pr = random.randint(r0, r1 - ph + 1)
        pc = random.randint(c0, c1 - pw + 1)
        _paste(grid, patch, pr, pc)
        record["placed_bbox"] = (int(pr), int(pc), int(pr + ph - 1), int(pc + pw - 1))

    return grid


def _validate_instance(grid: np.ndarray, records: list[dict]) -> None:
    infos = _analyze_grid(grid, bg=0)
    winner = _selected_index(infos)
    out = main(grid)

    target_records = [record for record in records if record["kind"] == "target"]
    assert len(target_records) == 1, "generation must produce exactly one intended target"
    target_sprite = target_records[0]["sprite"]

    assert len(infos) == len(records), "placement must preserve sprite count"
    assert sum(info["symmetric"] for info in infos) == 1, "exactly one sprite must stay symmetric after compression"
    assert np.array_equal(out, infos[winner]["crop"]), "main() must crop the selected original sprite"
    assert np.array_equal(out, target_sprite), "semantic target must match the intended compressed-symmetry sprite"
    assert not np.array_equal(out, np.fliplr(out)), "target must not be directly left-right symmetric in raw pixels"
    assert any(
        info["crop"].shape == out.shape and info["skeleton"].shape == infos[winner]["skeleton"].shape
        for idx, info in enumerate(infos)
        if idx != winner
    ), "need a same-size foil so crop size and partition count cannot identify the answer"
    assert any(
        info["color_count"] == infos[winner]["color_count"]
        for idx, info in enumerate(infos)
        if idx != winner
    ), "need a distractor with the same number of colors as the target"
    assert any(
        _mirror_mismatch_count(info["skeleton"]) == 2
        for idx, info in enumerate(infos)
        if idx != winner
    ), "need a near-symmetric distractor to rule out loose symmetry heuristics"
    assert all(
        len(set(info["row_heights"])) > 1 or len(set(info["col_widths"])) > 1
        for info in infos
    ), "every sprite must use unequal partition sizes"
    assert all(
        not np.array_equal(out, info["crop"])
        for idx, info in enumerate(infos)
        if idx != winner
    ), "target crop must be unique among sprites in the same instance"


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        try:
            records = _build_records()
            grid = _place_records(records)
            _validate_instance(grid, records)
            return grid
        except Exception:
            continue
    raise RuntimeError("failed to generate a valid compressed-skeleton symmetry puzzle")
