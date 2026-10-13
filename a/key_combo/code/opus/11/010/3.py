from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
GREEN = 3
CONTROL_COLORS = (1, 2, 4, 5, 6, 7, 8)
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRECTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # clockwise

GEN_OBJECT_COUNT_RANGE = (6, 9)
GEN_HEIGHT_RANGE = (16, 24)
GEN_WIDTH_RANGE = (17, 26)
GEN_MAX_ATTEMPTS = 600


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_components(mask: np.ndarray) -> list[np.ndarray]:
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
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _count_bends(mask: np.ndarray) -> int:
    h, w = mask.shape
    bends = 0
    for r in range(h - 1):
        for c in range(w - 1):
            if int(mask[r : r + 2, c : c + 2].sum()) == 3:
                bends += 1
    return bends


def _detect_control_matrix(grid: np.ndarray) -> tuple[tuple[int, int, int, int], np.ndarray]:
    control_mask = (grid != BG) & (grid != GREEN)
    assert control_mask.any(), "expected a non-green control matrix"
    coords = np.argwhere(control_mask)
    box = _bbox(coords)
    r0, c0, r1, c1 = box
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    assert patch.shape[0] == 3, "control matrix must have three rows"
    assert np.all(patch != BG) and np.all(patch != GREEN), "control matrix must be dense"
    assert int(control_mask.sum()) == int(patch.size), "all non-green colors must belong to the control matrix"
    h, w = grid.shape
    assert (r0 == 0 or r1 == h - 1) and (c0 == 0 or c1 == w - 1), "control matrix must sit in a corner"
    return box, patch.copy()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each green bent path by indexing a corner control matrix with bend count and within-class horizontal rank.

    Input structure:
    - The grid has black background, several green 4-connected one-pixel-wide bent path objects, and one dense non-green control matrix in a corner.
    - Every green object has 1, 2, or 3 L-shaped bends, and different objects have distinct bounding-box left edges.
    - The control matrix has exactly 3 rows and only non-black, non-green colors.

    Rule:
    - Detect each green object and count its bends by scanning all 2x2 windows for L-shaped three-pixel patterns.
    - Map bend counts 1, 2, 3 to control-matrix rows 0, 1, 2.
    - Within each bend-count class, sort objects by bounding-box left edge; the 0-based rank is the control-matrix column.
    - Recolor the whole object with the indexed control color, then erase the control matrix to background.

    Key insight:
    - The target color is determined jointly by topology and position: bend count chooses the row, but only the horizontal rank within the same bend class chooses the column.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    control_box, control = _detect_control_matrix(g)
    out = g.copy()

    objects = []
    for coords in _find_components(g == GREEN):
        box = _bbox(coords)
        r0, c0, r1, c1 = box
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        local = coords - np.array([r0, c0], dtype=int)
        mask[local[:, 0], local[:, 1]] = True
        bends = _count_bends(mask)
        assert bends in (1, 2, 3), f"unexpected bend count {bends}"
        objects.append({"coords": coords, "bbox": box, "bends": bends})

    left_edges = [obj["bbox"][1] for obj in objects]
    assert len(set(left_edges)) == len(left_edges), "object left edges must be unique"

    groups = {1: [], 2: [], 3: []}
    for obj in objects:
        groups[int(obj["bends"])].append(obj)

    for bends in (1, 2, 3):
        groups[bends].sort(key=lambda obj: (obj["bbox"][1], obj["bbox"][0]))
        assert len(groups[bends]) <= control.shape[1], "control matrix is too narrow for this bend class"
        for rank, obj in enumerate(groups[bends]):
            color = int(control[bends - 1, rank])
            coords = obj["coords"]
            out[coords[:, 0], coords[:, 1]] = color

    r0, c0, r1, c1 = control_box
    out[r0 : r1 + 1, c0 : c1 + 1] = BG
    return out


def _turn90(direction: tuple[int, int], delta: int) -> tuple[int, int]:
    return DIRECTIONS[(DIRECTIONS.index(direction) + delta) % 4]


def _random_composition(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(1, total), parts - 1))
    pieces = []
    prev = 0
    for cut in cuts + [total]:
        pieces.append(cut - prev)
        prev = cut
    return pieces


def _path_patch_for_bends(bends: int) -> np.ndarray:
    for _ in range(300):
        total_steps = random.randint(bends + 1, 5)
        segment_lengths = _random_composition(total_steps, bends + 1)
        direction = random.choice(DIRECTIONS)
        directions = [direction]
        for _ in range(bends):
            directions.append(_turn90(directions[-1], random.choice((-1, 1))))

        path = [(0, 0)]
        r = c = 0
        for seg_len, (dr, dc) in zip(segment_lengths, directions):
            for _ in range(seg_len):
                r += dr
                c += dc
                path.append((r, c))

        pts = set(path)
        if len(pts) != len(path):
            continue

        degrees = []
        for rr, cc in pts:
            degree = sum((nr, nc) in pts for nr, nc in _neighbors4(rr, cc))
            degrees.append(degree)
        if degrees.count(1) != 2 or any(deg not in (1, 2) for deg in degrees):
            continue

        coords = np.array(sorted(pts), dtype=int)
        r0, c0, r1, c1 = _bbox(coords)
        local = coords - np.array([r0, c0], dtype=int)
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        mask[local[:, 0], local[:, 1]] = True
        if _count_bends(mask) != bends:
            continue
        if not (3 <= int(mask.sum()) <= 6):
            continue

        patch = np.zeros(mask.shape, dtype=int)
        patch[mask] = GREEN
        return patch
    raise RuntimeError(f"failed to generate a {bends}-bend path")


def _sample_counts(total_objects: int) -> dict[int, int]:
    for _ in range(200):
        counts = [1, 1, 1]
        for _ in range(total_objects - 3):
            counts[random.randrange(3)] += 1
        if max(counts) <= 4 and sum(count >= 2 for count in counts) >= 2:
            return {1: counts[0], 2: counts[1], 3: counts[2]}
    raise RuntimeError("failed to sample bend counts")


def _is_interleaved(sequence: list[int]) -> bool:
    for bend in (1, 2, 3):
        positions = [i for i, value in enumerate(sequence) if value == bend]
        if positions and positions[-1] - positions[0] + 1 > len(positions):
            return True
    return False


def _sample_sequence(counts: dict[int, int]) -> list[int]:
    sequence = [bend for bend, count in counts.items() for _ in range(count)]
    for _ in range(300):
        random.shuffle(sequence)
        if _is_interleaved(sequence):
            return sequence.copy()
    raise RuntimeError("failed to interleave bend classes")


def _build_control_matrix(counts: dict[int, int]) -> np.ndarray:
    width = max(counts.values())
    for _ in range(300):
        palette_size = random.randint(3, min(6, len(CONTROL_COLORS)))
        palette = random.sample(CONTROL_COLORS, palette_size)
        control = np.array(
            [[random.choice(palette) for _ in range(width)] for _ in range(3)],
            dtype=int,
        )
        if any(count >= 2 and len(set(map(int, control[bend - 1, :count]))) < 2 for bend, count in counts.items()):
            continue
        shared_cols = [col for col in range(width) if sum(count > col for count in counts.values()) >= 2]
        if not shared_cols:
            continue
        if not any(
            len({int(control[bend - 1, col]) for bend, count in counts.items() if count > col}) >= 2
            for col in shared_cols
        ):
            continue
        used_colors = [int(control[bend - 1, col]) for bend, count in counts.items() for col in range(count)]
        if len(set(used_colors)) < 3:
            continue
        return control
    raise RuntimeError("failed to build a non-degenerate control matrix")


def _corner_bbox(grid_shape: tuple[int, int], patch_shape: tuple[int, int], corner: str) -> tuple[int, int, int, int]:
    h, w = map(int, grid_shape)
    ph, pw = map(int, patch_shape)
    if corner == "tl":
        r0, c0 = 0, 0
    elif corner == "tr":
        r0, c0 = 0, w - pw
    elif corner == "bl":
        r0, c0 = h - ph, 0
    else:
        r0, c0 = h - ph, w - pw
    return r0, c0, r0 + ph - 1, c0 + pw - 1


def _mark_reserved(mask: np.ndarray, box: tuple[int, int, int, int], pad: int = 1) -> None:
    h, w = mask.shape
    r0, c0, r1, c1 = box
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    mask[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _can_place(reserved: np.ndarray, r0: int, c0: int, patch: np.ndarray) -> bool:
    h, w = patch.shape
    return not reserved[r0 : r0 + h, c0 : c0 + w].any()


def _expected_output(
    grid: np.ndarray,
    objects: list[dict],
    control_box: tuple[int, int, int, int],
    control: np.ndarray,
) -> np.ndarray:
    out = grid.copy()
    for obj in objects:
        rr, cc = obj["coords"][:, 0], obj["coords"][:, 1]
        out[rr, cc] = int(control[obj["bends"] - 1, obj["rank"]])
    r0, c0, r1, c1 = control_box
    out[r0 : r1 + 1, c0 : c1 + 1] = BG
    return out


def _validate_scene(
    grid: np.ndarray,
    expected: np.ndarray,
    objects: list[dict],
    control_box: tuple[int, int, int, int],
    control: np.ndarray,
    counts: dict[int, int],
    sequence: list[int],
) -> bool:
    if grid.shape != expected.shape or max(grid.shape) > 30:
        return False
    if not _is_interleaved(sequence):
        return False
    if len(objects) != sum(counts.values()) or not (GEN_OBJECT_COUNT_RANGE[0] <= len(objects) <= GEN_OBJECT_COUNT_RANGE[1]):
        return False
    if len({obj["bbox"][1] for obj in objects}) != len(objects):
        return False

    used_rows = sum(len({int(control[bend - 1, col]) for col in range(count)}) >= 2 for bend, count in counts.items() if count >= 2)
    if used_rows == 0:
        return False
    shared_cols = [col for col in range(control.shape[1]) if sum(count > col for count in counts.values()) >= 2]
    if not shared_cols:
        return False
    if not any(len({int(control[bend - 1, col]) for bend, count in counts.items() if count > col}) >= 2 for col in shared_cols):
        return False

    box, detected = _detect_control_matrix(grid)
    if box != control_box or not np.array_equal(detected, control):
        return False

    for obj in objects:
        r0, c0, r1, c1 = obj["bbox"]
        patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        local = obj["coords"] - np.array([r0, c0], dtype=int)
        patch[local[:, 0], local[:, 1]] = True
        if _count_bends(patch) != obj["bends"]:
            return False

    try:
        solved = main(grid)
    except Exception:
        return False
    return np.array_equal(solved, expected)


def _generate_scene() -> tuple[np.ndarray, np.ndarray]:
    total_objects = random.randint(*GEN_OBJECT_COUNT_RANGE)
    counts = _sample_counts(total_objects)
    sequence = _sample_sequence(counts)
    control = _build_control_matrix(counts)

    objects_meta = []
    seen_within_class = {1: 0, 2: 0, 3: 0}
    for bend in sequence:
        patch = _path_patch_for_bends(bend)
        objects_meta.append(
            {
                "bends": bend,
                "rank": seen_within_class[bend],
                "patch": patch,
                "h": int(patch.shape[0]),
                "w": int(patch.shape[1]),
            }
        )
        seen_within_class[bend] += 1

    max_object_w = max(obj["w"] for obj in objects_meta)
    for _ in range(300):
        h = random.randint(*GEN_HEIGHT_RANGE)
        w = random.randint(max(GEN_WIDTH_RANGE[0], max_object_w + len(objects_meta)), GEN_WIDTH_RANGE[1])
        corner = random.choice(["tl", "tr", "bl", "br"])
        control_box = _corner_bbox((h, w), control.shape, corner)

        if w - max_object_w + 1 < len(objects_meta):
            continue
        left_cols = sorted(random.sample(range(w - max_object_w + 1), len(objects_meta)))

        grid = np.zeros((h, w), dtype=int)
        reserved = np.zeros((h, w), dtype=bool)
        cr0, cc0, cr1, cc1 = control_box
        grid[cr0 : cr1 + 1, cc0 : cc1 + 1] = control
        _mark_reserved(reserved, control_box, pad=1)

        placed_objects: list[dict] = []
        used_top_rows: set[int] = set()
        ok = True
        for obj, c0 in zip(objects_meta, left_cols):
            patch = obj["patch"]
            hh, ww = patch.shape
            row_options = list(range(h - hh + 1))
            random.shuffle(row_options)
            chosen_row = None
            for r0 in row_options:
                if _can_place(reserved, r0, c0, patch):
                    chosen_row = r0
                    break
            if chosen_row is None:
                ok = False
                break

            r0 = chosen_row
            grid[r0 : r0 + hh, c0 : c0 + ww][patch == GREEN] = GREEN
            coords = np.argwhere(patch == GREEN) + np.array([r0, c0], dtype=int)
            box = (r0, c0, r0 + hh - 1, c0 + ww - 1)
            _mark_reserved(reserved, box, pad=1)
            used_top_rows.add(r0)
            placed_objects.append(
                {
                    "coords": coords,
                    "bbox": box,
                    "bends": obj["bends"],
                    "rank": obj["rank"],
                }
            )

        if not ok or len(used_top_rows) < 2:
            continue

        expected = _expected_output(grid, placed_objects, control_box, control)
        if _validate_scene(grid, expected, placed_objects, control_box, control, counts, sequence):
            return grid, expected

    raise RuntimeError("failed to place a valid scene")


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        try:
            grid, _expected = _generate_scene()
            return grid
        except RuntimeError:
            continue
    raise RuntimeError("failed to generate a valid input grid")
