from __future__ import annotations

from collections import Counter
import numpy as np


BG = 0
SPRITE = 1
MARKER = 2
SHARED_COLOR = 4
UNIQUE_COLOR = 7

_RNG = np.random.default_rng()


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


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


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _tight_mask_from_coords(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel = coords - np.array([[r0, c0]])
    mask[rel[:, 0], rel[:, 1]] = True
    return mask, (r0, c0, r1, c1)


def _tight_crop(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _dihedral_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    cur = np.asarray(mask, dtype=bool)
    for _ in range(4):
        variants.append(cur.copy())
        variants.append(np.fliplr(cur).copy())
        cur = np.rot90(cur)
    return variants


def _canonical_tight_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    best: tuple[tuple[int, int], bytes] | None = None
    for variant in _dihedral_variants(mask):
        tight = _tight_crop(variant)
        key = (tuple(map(int, tight.shape)), tight.astype(np.uint8).tobytes())
        if best is None or key < best:
            best = key
    assert best is not None
    return best


def _canonical_cell_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    best: tuple[tuple[int, int], bytes] | None = None
    for variant in _dihedral_variants(mask):
        key = (tuple(map(int, variant.shape)), variant.astype(np.uint8).tobytes())
        if best is None or key < best:
            best = key
    assert best is not None
    return best


def _paste_mask(dst: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    h, w = mask.shape
    dst_slice = dst[r0 : r0 + h, c0 : c0 + w]
    dst_slice[mask] = int(color)


def _regular_lattice(grid: np.ndarray, bg: int) -> tuple[int, list[int], list[int]]:
    candidates: list[tuple[int, list[int], list[int]]] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != bg):
        coords = np.argwhere(grid == color)
        if coords.size == 0:
            continue
        rows = sorted({int(r) for r, _ in coords})
        cols = sorted({int(c) for _, c in coords})
        if len(rows) < 2 or len(cols) < 2:
            continue
        if int(coords.shape[0]) != len(rows) * len(cols):
            continue
        if any(int(grid[r, c]) != color for r in rows for c in cols):
            continue
        row_diffs = np.diff(rows)
        col_diffs = np.diff(cols)
        if len(set(map(int, row_diffs))) != 1 or len(set(map(int, col_diffs))) != 1:
            continue
        candidates.append((color, rows, cols))
    assert len(candidates) == 1, "expected exactly one regular marker lattice"
    return candidates[0]


def _cell_index(bounds: list[int], low: int, high: int) -> int:
    matches = [i for i in range(len(bounds) - 1) if bounds[i] < low and high < bounds[i + 1]]
    assert len(matches) == 1, "sprite must lie strictly inside one lattice cell"
    return int(matches[0])


def _centered_cell_mask(mask: np.ndarray, cell_shape: tuple[int, int]) -> tuple[np.ndarray, tuple[int, int]]:
    cell_h, cell_w = map(int, cell_shape)
    h, w = map(int, mask.shape)
    assert h <= cell_h and w <= cell_w
    local_r0 = (cell_h - h) // 2
    local_c0 = (cell_w - w) // 2
    cell_mask = np.zeros((cell_h, cell_w), dtype=bool)
    cell_mask[local_r0 : local_r0 + h, local_c0 : local_c0 + w][mask] = True
    return cell_mask, (local_r0, local_c0)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recenter each sprite inside its marker-lattice cell, then recolor by centered shape frequency.

    Input structure:
    - The background is black.
    - Red single-pixel markers form a complete regular Cartesian lattice that partitions the canvas into equal cells.
    - Some cells contain exactly one small blue 4-connected sprite placed off-center inside the cell.

    Rule:
    - Keep the red marker lattice fixed and clear the original blue sprite pixels.
    - For each sprite, identify its lattice cell and translate the whole sprite so its bounding box is centered inside that cell.
    - Compare the centered sprites by binary shape, treating rotations and mirror flips as equivalent, and count how often each centered shape occurs.
    - Repaint sprites whose centered shape appears at least twice in yellow; repaint shapes that appear exactly once in orange.

    Key insight:
    - The lattice provides a shared coordinate frame: the sprite must be normalized to the cell center before shape-frequency grouping is reliable.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _most_common_color(g)
    marker_color, marker_rows, marker_cols = _regular_lattice(g, bg)

    non_bg = sorted(int(v) for v in np.unique(g) if int(v) != bg)
    sprite_colors = [color for color in non_bg if color != marker_color]
    assert len(sprite_colors) == 1, "expected exactly one sprite color"
    sprite_color = int(sprite_colors[0])

    comps = _find_components(g == sprite_color)
    assert comps, "expected at least one sprite"

    out = np.full_like(g, bg)
    out[np.ix_(marker_rows, marker_cols)] = marker_color

    entries = []
    used_cells: set[tuple[int, int]] = set()
    for coords in comps:
        mask, (r0, c0, _r1, _c1) = _tight_mask_from_coords(coords)
        cell_r = _cell_index(marker_rows, r0, r0 + mask.shape[0] - 1)
        cell_c = _cell_index(marker_cols, c0, c0 + mask.shape[1] - 1)
        cell = (cell_r, cell_c)
        assert cell not in used_cells, "expected at most one sprite per cell"
        used_cells.add(cell)

        cell_h = marker_rows[cell_r + 1] - marker_rows[cell_r] - 1
        cell_w = marker_cols[cell_c + 1] - marker_cols[cell_c] - 1
        centered_mask, (local_r0, local_c0) = _centered_cell_mask(mask, (cell_h, cell_w))
        entries.append(
            {
                "shape_key": _canonical_cell_key(centered_mask),
                "mask": mask,
                "dest_r0": marker_rows[cell_r] + 1 + local_r0,
                "dest_c0": marker_cols[cell_c] + 1 + local_c0,
            }
        )

    freq = Counter(entry["shape_key"] for entry in entries)
    for entry in entries:
        color = SHARED_COLOR if int(freq[entry["shape_key"]]) >= 2 else UNIQUE_COLOR
        _paste_mask(out, entry["mask"], int(entry["dest_r0"]), int(entry["dest_c0"]), color)
    return out


def _shape_library() -> list[np.ndarray]:
    if hasattr(_shape_library, "_cache"):
        return getattr(_shape_library, "_cache")

    reps: dict[tuple[tuple[int, int], bytes], np.ndarray] = {}
    for h in range(1, 4):
        for w in range(1, 4):
            for bits in range(1, 1 << (h * w)):
                area = bits.bit_count()
                if not 3 <= area <= 5:
                    continue
                mask = np.zeros((h, w), dtype=bool)
                for idx in range(h * w):
                    if bits & (1 << idx):
                        mask[idx // w, idx % w] = True
                if not np.any(mask):
                    continue
                if _tight_crop(mask).shape != mask.shape:
                    continue
                if len(_find_components(mask)) != 1:
                    continue
                reps.setdefault(_canonical_tight_key(mask), mask.copy())

    library = list(reps.values())
    setattr(_shape_library, "_cache", library)
    return library


def _random_dihedral(mask: np.ndarray) -> np.ndarray:
    rotated = np.rot90(mask, int(_RNG.integers(0, 4)))
    if bool(_RNG.integers(0, 2)):
        rotated = np.fliplr(rotated)
    return rotated.copy()


def _grid_spec() -> tuple[int, int, int, int, int]:
    cell_rows, cell_cols = ((3, 3), (3, 4), (4, 3))[int(_RNG.integers(0, 3))]
    pitch = 6
    first_r = int(_RNG.integers(0, 3))
    first_c = int(_RNG.integers(0, 3))
    pad_r = int(_RNG.integers(0, 3))
    pad_c = int(_RNG.integers(0, 3))
    height = first_r + cell_rows * pitch + 1 + pad_r
    width = first_c + cell_cols * pitch + 1 + pad_c
    assert max(height, width) <= 30
    return cell_rows, cell_cols, first_r, first_c, pitch


def _choose_shared_sizes(total: int) -> list[int]:
    shared_groups = 2 + int(total >= 9 and _RNG.random() < 0.45)
    remaining = total
    sizes: list[int] = []
    for idx in range(shared_groups):
        min_left = 1 + 2 * (shared_groups - idx - 1)
        hi = min(4, remaining - min_left)
        size = int(_RNG.integers(2, hi + 1))
        sizes.append(size)
        remaining -= size
    return sizes


def _valid_offsets(mask_shape: tuple[int, int], cell_shape: tuple[int, int]) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    h, w = map(int, mask_shape)
    cell_h, cell_w = map(int, cell_shape)
    center = ((cell_h - h) // 2, (cell_w - w) // 2)
    offsets = [
        (rr, cc)
        for rr in range(cell_h - h + 1)
        for cc in range(cell_w - w + 1)
        if (rr, cc) != center
    ]
    assert offsets, "generator requires at least one non-center offset"
    return offsets, center


def _raw_cell_key(mask: np.ndarray, cell_shape: tuple[int, int], offset: tuple[int, int]) -> tuple[tuple[int, int], bytes]:
    cell = np.zeros(cell_shape, dtype=bool)
    rr, cc = offset
    h, w = mask.shape
    cell[rr : rr + h, cc : cc + w][mask] = True
    return _canonical_cell_key(cell)


def _build_example() -> tuple[np.ndarray, np.ndarray]:
    cell_rows, cell_cols, first_r, first_c, pitch = _grid_spec()
    marker_rows = [first_r + i * pitch for i in range(cell_rows + 1)]
    marker_cols = [first_c + i * pitch for i in range(cell_cols + 1)]
    inner_shape = (pitch - 1, pitch - 1)
    total_cells = cell_rows * cell_cols
    occupied_count = int(_RNG.integers(8, total_cells + 1))

    shared_sizes = _choose_shared_sizes(occupied_count)
    unique_count = occupied_count - sum(shared_sizes)
    assert unique_count >= 1

    library = _shape_library()
    need_templates = len(shared_sizes) + unique_count
    assert len(library) >= need_templates
    template_ids = _RNG.choice(len(library), size=need_templates, replace=False)
    shared_templates = [library[int(i)] for i in template_ids[: len(shared_sizes)]]
    unique_templates = [library[int(i)] for i in template_ids[len(shared_sizes) :]]

    cells = [(r, c) for r in range(cell_rows) for c in range(cell_cols)]
    chosen_cells = [cells[int(i)] for i in _RNG.choice(len(cells), size=occupied_count, replace=False)]
    _RNG.shuffle(chosen_cells)

    objects: list[dict] = []
    cell_idx = 0
    for template, group_size in zip(shared_templates, shared_sizes):
        offsets, _center = _valid_offsets(template.shape, inner_shape)
        if group_size > 1 and len(offsets) == 1:
            raise AssertionError("shared groups require offset diversity")
        chosen_offsets = [offsets[int(i)] for i in _RNG.choice(len(offsets), size=group_size, replace=False)]
        for k in range(group_size):
            oriented = _random_dihedral(template)
            local_offsets, _ = _valid_offsets(oriented.shape, inner_shape)
            preferred = [off for off in local_offsets if off == chosen_offsets[k]]
            offset = preferred[0] if preferred else local_offsets[int(_RNG.integers(0, len(local_offsets)))]
            objects.append(
                {
                    "group": f"shared-{cell_idx}",
                    "template_key": _canonical_tight_key(template),
                    "mask": oriented,
                    "cell": chosen_cells[cell_idx],
                    "input_offset": offset,
                }
            )
            cell_idx += 1

    for template in unique_templates:
        oriented = _random_dihedral(template)
        offsets, _ = _valid_offsets(oriented.shape, inner_shape)
        offset = offsets[int(_RNG.integers(0, len(offsets)))]
        objects.append(
            {
                "group": f"unique-{cell_idx}",
                "template_key": _canonical_tight_key(template),
                "mask": oriented,
                "cell": chosen_cells[cell_idx],
                "input_offset": offset,
            }
        )
        cell_idx += 1

    height = marker_rows[-1] + 1 + int(_RNG.integers(0, 3))
    width = marker_cols[-1] + 1 + int(_RNG.integers(0, 3))
    assert max(height, width) <= 30
    inp = np.zeros((height, width), dtype=int)
    out = np.zeros_like(inp)
    inp[np.ix_(marker_rows, marker_cols)] = MARKER
    out[np.ix_(marker_rows, marker_cols)] = MARKER

    centered_freq: Counter[tuple[tuple[int, int], bytes]] = Counter()
    shared_raw_diversity: dict[tuple[tuple[int, int], bytes], set[tuple[tuple[int, int], bytes]]] = {}

    for obj in objects:
        cell_r, cell_c = obj["cell"]
        base_r = marker_rows[cell_r] + 1
        base_c = marker_cols[cell_c] + 1
        mask = obj["mask"]
        h, w = mask.shape
        centered_cell, (center_r, center_c) = _centered_cell_mask(mask, inner_shape)
        centered_key = _canonical_cell_key(centered_cell)
        centered_freq[centered_key] += 1
        raw_key = _raw_cell_key(mask, inner_shape, obj["input_offset"])
        shared_raw_diversity.setdefault(obj["template_key"], set()).add(raw_key)

        in_r = base_r + int(obj["input_offset"][0])
        in_c = base_c + int(obj["input_offset"][1])
        out_r = base_r + center_r
        out_c = base_c + center_c
        _paste_mask(inp, mask, in_r, in_c, SPRITE)
        obj["centered_key"] = centered_key
        obj["out_r"] = out_r
        obj["out_c"] = out_c

    by_template = Counter(obj["template_key"] for obj in objects)
    shared_templates_present = [key for key, count in by_template.items() if count >= 2]
    assert len(shared_templates_present) >= 2
    assert any(count == 1 for count in by_template.values())
    assert any(len(shared_raw_diversity[key]) >= 2 for key in shared_templates_present)

    for obj in objects:
        color = SHARED_COLOR if centered_freq[obj["centered_key"]] >= 2 else UNIQUE_COLOR
        _paste_mask(out, obj["mask"], int(obj["out_r"]), int(obj["out_c"]), color)

    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(400):
        try:
            inp, expected = _build_example()
        except Exception:
            continue
        try:
            actual = main(inp)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return inp
    raise RuntimeError("failed to generate a valid centered-frequency ARC input")
