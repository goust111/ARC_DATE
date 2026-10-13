import numpy as np


BG = 0
COLOR_POOL = tuple(range(1, 10))
SIDES = ("top", "right", "bottom", "left")
ROTATE_TO_UP = {"top": 0, "right": 1, "bottom": 2, "left": 3}
MAX_GENERATION_ATTEMPTS = 300
MAX_USED_COLORS = 8

_RNG = np.random.default_rng()
_GEN_COUNTER = 0


def _paste(dst: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    region = dst[top : top + h, left : left + w]
    mask = patch != BG
    region[mask] = patch[mask]


def _boundary_mask(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _boundary_values(arr: np.ndarray) -> np.ndarray:
    return arr[_boundary_mask(arr.shape)]


def _read_palette(grid: np.ndarray) -> tuple[list[int], dict[int, int]]:
    cols = np.flatnonzero(grid[-1] != BG)
    assert cols.size >= 2
    assert np.all(np.diff(cols) == 1)
    palette = [int(grid[-1, c]) for c in cols]
    assert len(set(palette)) == len(palette)
    return palette, {color: idx for idx, color in enumerate(palette)}


def _find_components(mask: np.ndarray) -> list[tuple[tuple[int, int, int, int], np.ndarray]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[tuple[tuple[int, int, int, int], np.ndarray]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            coords = np.array(cells, dtype=int)
            bbox = (
                int(coords[:, 0].min()),
                int(coords[:, 1].min()),
                int(coords[:, 0].max()),
                int(coords[:, 1].max()),
            )
            comps.append((bbox, coords))
    return comps


def _extract_layer_patch(crop: np.ndarray, depth: int) -> np.ndarray | None:
    h = int(crop.shape[0] - 2 * depth)
    w = int(crop.shape[1] - 2 * depth)
    if h <= 0 or w <= 0:
        return None
    view = crop[depth : depth + h, depth : depth + w]
    patch = np.full((h, w), BG, dtype=int)
    patch[0, :] = view[0, :]
    patch[-1, :] = view[-1, :]
    patch[:, 0] = view[:, 0]
    patch[:, -1] = view[:, -1]
    return patch


def _detect_gap_side(outer_patch: np.ndarray) -> str:
    edge_bg = {
        "top": int(np.count_nonzero(outer_patch[0, :] == BG)),
        "right": int(np.count_nonzero(outer_patch[:, -1] == BG)),
        "bottom": int(np.count_nonzero(outer_patch[-1, :] == BG)),
        "left": int(np.count_nonzero(outer_patch[:, 0] == BG)),
    }
    sides = [side for side, count in edge_bg.items() if count == 1]
    assert len(sides) == 1
    assert sum(edge_bg.values()) == 1
    return sides[0]


def _parse_target(crop: np.ndarray, palette_index: dict[int, int]) -> dict:
    layers = []
    gap_side = None
    depth = 0
    while True:
        patch = _extract_layer_patch(crop, depth)
        if patch is None:
            break
        colors = sorted(set(map(int, patch[patch != BG])))
        if not colors:
            break
        assert len(colors) == 1
        if depth == 0:
            gap_side = _detect_gap_side(patch)
        else:
            assert int(np.count_nonzero(_boundary_values(patch) == BG)) == 0
        color = int(colors[0])
        assert color in palette_index
        layers.append({"color": color, "patch": patch})
        depth += 1

    assert layers and gap_side is not None
    k = ROTATE_TO_UP[gap_side]
    rotated_layers = []
    for layer in layers:
        rotated_layers.append(
            {
                "color": int(layer["color"]),
                "index": int(palette_index[int(layer["color"])]),
                "patch": np.rot90(layer["patch"], k=k),
            }
        )
    rotated_layers.sort(key=lambda item: int(item["index"]))
    return {
        "outer_index": int(palette_index[int(layers[0]["color"])]),
        "layers": rotated_layers,
    }


def _assemble_output(targets: list[dict]) -> np.ndarray:
    rows = []
    max_width = 0
    for target in targets:
        layers = target["layers"]
        row_h = int(max(layer["patch"].shape[0] for layer in layers))
        row_w = int(sum(layer["patch"].shape[1] for layer in layers) + max(0, len(layers) - 1))
        row = np.full((row_h, row_w), BG, dtype=int)
        col = 0
        for layer in layers:
            patch = layer["patch"]
            _paste(row, patch, 0, col)
            col += int(patch.shape[1]) + 1
        rows.append(row)
        max_width = max(max_width, row_w)

    out_h = int(sum(row.shape[0] for row in rows) + max(0, len(rows) - 1))
    out = np.full((out_h, max_width), BG, dtype=int)
    top = 0
    for row in rows:
        _paste(out, row, top, 0)
        top += int(row.shape[0]) + 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decompose each nested rectangular target into separately arranged rings normalized by the outer gap direction.

    Input structure:
    - bg = black. Above the last row are several disconnected target objects made of concentric rectangular color rings.
    - Within each target, ring colors change from outer to inner, and only the outermost ring has a single background gap on one side.
    - The bottom row contains one contiguous palette strip of distinct colors whose left-to-right order defines color indices.

    Rule:
    - For every target, peel off the rectangular rings from outer to inner by reading successive bounding-box borders.
    - Use the outer-gap side to rotate the whole target's ring set so that the gap points upward.
    - Draw the extracted rings as separate rectangular-ring patches with their original sizes preserved, sort them left-to-right by palette index, and leave one background column between neighbors.
    - Sort whole targets top-to-bottom by the palette index of their outermost color, leaving one background row between target rows.

    Key insight:
    - The palette is a visible ordering key for both the vertical order of targets and the horizontal order of layers, while the single missing outer pixel supplies the rotation needed to normalize orientation before rearrangement.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 2 and g.shape[1] >= 2

    _, palette_index = _read_palette(g)
    body = g[:-1, :]
    comps = _find_components(body != BG)
    assert comps

    targets = []
    for bbox, _coords in comps:
        r0, c0, r1, c1 = bbox
        crop = body[r0 : r1 + 1, c0 : c1 + 1]
        targets.append(_parse_target(crop, palette_index))

    targets.sort(key=lambda item: int(item["outer_index"]))
    return _assemble_output(targets)


def _sample_layer_counts(target_count: int) -> list[int]:
    min_total = 2 * target_count
    max_total = min(MAX_USED_COLORS, 4 * target_count, len(COLOR_POOL))
    total = int(_RNG.integers(min_total, max_total + 1))
    counts = [2] * target_count
    for _ in range(total - min_total):
        choices = [i for i, count in enumerate(counts) if count < 4]
        pick = int(_RNG.choice(choices))
        counts[pick] += 1
    _RNG.shuffle(counts)
    return counts


def _center_cap(layer_count: int) -> int:
    limit = (31 - layer_count * layer_count - 2 * layer_count) // layer_count
    return int(max(0, min(3, limit)))


def _build_target_spec(colors: list[int], side: str, center_h: int, center_w: int) -> dict:
    layer_count = len(colors)
    h = int(center_h + 2 * layer_count)
    w = int(center_w + 2 * layer_count)

    full = np.full((h, w), BG, dtype=int)
    layers = []
    for depth, color in enumerate(colors):
        lh = int(h - 2 * depth)
        lw = int(w - 2 * depth)
        patch = np.full((lh, lw), BG, dtype=int)
        patch[0, :] = color
        patch[-1, :] = color
        patch[:, 0] = color
        patch[:, -1] = color
        if depth == 0:
            if side in ("top", "bottom"):
                gap_col = int(_RNG.integers(1, lw - 1))
                gap_row = 0 if side == "top" else lh - 1
            else:
                gap_row = int(_RNG.integers(1, lh - 1))
                gap_col = 0 if side == "left" else lw - 1
            patch[gap_row, gap_col] = BG
        layers.append({"color": int(color), "patch": patch})
        _paste(full, patch, depth, depth)

    return {"full": full, "layers": layers, "side": side}


def _expected_output_from_specs(target_specs: list[dict], palette_index: dict[int, int]) -> np.ndarray:
    targets = []
    for spec in target_specs:
        k = ROTATE_TO_UP[spec["side"]]
        layers = []
        for layer in spec["layers"]:
            color = int(layer["color"])
            layers.append(
                {
                    "color": color,
                    "index": int(palette_index[color]),
                    "patch": np.rot90(layer["patch"], k=k),
                }
            )
        layers.sort(key=lambda item: int(item["index"]))
        targets.append(
            {
                "outer_index": int(palette_index[int(spec["layers"][0]["color"])]),
                "layers": layers,
            }
        )
    targets.sort(key=lambda item: int(item["outer_index"]))
    return _assemble_output(targets)


def _candidate_groupings(target_count: int) -> list[list[int]]:
    if target_count == 1:
        return [[1]]
    if target_count == 2:
        return [[2], [1, 1]]
    return [[3], [2, 1], [1, 2], [1, 1, 1]]


def _compose_input(target_specs: list[dict], palette: list[int]) -> np.ndarray | None:
    order = list(map(int, _RNG.permutation(len(target_specs))))
    targets = [target_specs[i]["full"] for i in order]
    groupings = _candidate_groupings(len(targets))
    _RNG.shuffle(groupings)

    for grouping in groupings:
        rows = []
        idx = 0
        for size in grouping:
            row_targets = targets[idx : idx + size]
            idx += size
            gaps = [int(_RNG.integers(2, 5)) for _ in range(max(0, size - 1))]
            row_width = int(sum(t.shape[1] for t in row_targets) + sum(gaps))
            band_h = int(max(t.shape[0] for t in row_targets) + int(_RNG.integers(0, 3)))
            x = 0
            items = []
            for i, target in enumerate(row_targets):
                y_offset = int(_RNG.integers(0, band_h - target.shape[0] + 1))
                items.append((y_offset, x, target))
                x += int(target.shape[1])
                if i < len(gaps):
                    x += int(gaps[i])
            rows.append({"band_h": band_h, "row_w": row_width, "items": items})

        v_gaps = [int(_RNG.integers(2, 5)) for _ in range(max(0, len(rows) - 1))]
        body_h = int(sum(row["band_h"] for row in rows) + sum(v_gaps))
        body_w = int(max(row["row_w"] for row in rows))
        palette_len = len(palette)

        gap_to_palette = int(_RNG.integers(1, 3))
        min_h = body_h + gap_to_palette + 2
        min_w = max(body_w, palette_len) + 2
        if min_h > 30 or min_w > 30:
            continue

        h = int(_RNG.integers(min_h, 31))
        w = int(_RNG.integers(min_w, 31))
        grid = np.full((h, w), BG, dtype=int)

        max_body_top = h - (body_h + gap_to_palette + 1)
        body_top = int(_RNG.integers(1, max_body_top + 1))
        y = body_top
        for row_index, row in enumerate(rows):
            max_row_left = w - row["row_w"] - 1
            if max_row_left < 1:
                break
            row_left = int(_RNG.integers(1, max_row_left + 1))
            for y_offset, x_offset, target in row["items"]:
                _paste(grid, target, y + y_offset, row_left + x_offset)
            y += row["band_h"]
            if row_index < len(v_gaps):
                y += v_gaps[row_index]
        else:
            palette_left = int(_RNG.integers(1, w - palette_len))
            grid[-1, palette_left : palette_left + palette_len] = np.array(palette, dtype=int)
            return grid

    return None


def _gen_attempt() -> np.ndarray | None:
    global _GEN_COUNTER

    target_count = int(_RNG.integers(1, 4))
    layer_counts = _sample_layer_counts(target_count)
    total_colors = int(sum(layer_counts))
    used_colors = list(map(int, _RNG.choice(COLOR_POOL, size=total_colors, replace=False)))
    palette = list(map(int, _RNG.permutation(used_colors)))
    palette_index = {color: idx for idx, color in enumerate(palette)}

    forced_side = SIDES[_GEN_COUNTER % 4]
    remaining_sides = [side for side in SIDES if side != forced_side]
    permuted = list(map(str, _RNG.permutation(np.array(remaining_sides, dtype=object))))
    sides = [forced_side] + permuted[: max(0, target_count - 1)]
    _RNG.shuffle(sides)

    color_order = list(map(int, _RNG.permutation(used_colors)))
    target_specs = []
    cursor = 0
    for target_index, layer_count in enumerate(layer_counts):
        colors = color_order[cursor : cursor + layer_count]
        cursor += layer_count
        cap = _center_cap(layer_count)
        center_h = int(_RNG.integers(0, cap + 1))
        center_w = int(_RNG.integers(0, cap + 1))
        target_specs.append(_build_target_spec(colors, sides[target_index], center_h, center_w))

    expected = _expected_output_from_specs(target_specs, palette_index)
    if max(expected.shape) > 30:
        return None

    grid = _compose_input(target_specs, palette)
    if grid is None:
        return None

    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None

    _GEN_COUNTER += 1
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid target-ring puzzle")
