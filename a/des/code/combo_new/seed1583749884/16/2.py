import itertools
import random
from collections import Counter

import numpy as np


BG = 8
SEP = 9
PALETTE = [0, 1, 2, 3, 4, 5, 6, 7]
GEN_RETRIES = 400


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_separator_row(grid: np.ndarray, bg: int) -> int:
    candidates = []
    for r in range(grid.shape[0]):
        row = grid[r]
        if row[0] != bg and np.all(row == row[0]):
            candidates.append(r)
    assert len(candidates) == 1, "expected exactly one full non-background separator row"
    return int(candidates[0])


def _component_mask(panel: np.ndarray, coords: list[tuple[int, int]]) -> np.ndarray:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask


def _find_objects(panel: np.ndarray, bg: int, *, row_offset: int = 0) -> list[dict]:
    h, w = panel.shape
    seen = np.zeros((h, w), dtype=bool)
    objects = []
    for r in range(h):
        for c in range(w):
            color = int(panel[r, c])
            if color == bg or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(panel[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            objects.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                    "mask": _component_mask(panel, coords),
                    "global_col": min(cols),
                    "global_row": row_offset + min(rows),
                }
            )
    return objects


def _mask_coords(mask: np.ndarray) -> np.ndarray:
    return np.argwhere(mask).astype(int)


def _normalize_coords(coords: np.ndarray) -> tuple[np.ndarray, int, int]:
    min_r = int(coords[:, 0].min())
    min_c = int(coords[:, 1].min())
    return coords - np.array([min_r, min_c], dtype=int), min_r, min_c


def _assembly_signature(top_coords: np.ndarray, bottom_coords: np.ndarray, shape: tuple[int, int], flipped: bool) -> tuple:
    top_mask = np.zeros(shape, dtype=np.uint8)
    bottom_mask = np.zeros(shape, dtype=np.uint8)
    top_mask[top_coords[:, 0], top_coords[:, 1]] = 1
    bottom_mask[bottom_coords[:, 0], bottom_coords[:, 1]] = 1
    return (shape, int(flipped), top_mask.tobytes(), bottom_mask.tobytes())


def _find_assemblies(top_mask: np.ndarray, bottom_input_mask: np.ndarray) -> list[dict]:
    top_coords = _mask_coords(top_mask)
    results = {}
    for flipped, oriented_bottom in ((False, bottom_input_mask), (True, np.flipud(bottom_input_mask))):
        bottom_coords = _mask_coords(oriented_bottom)
        for dr in range(-bottom_input_mask.shape[0], top_mask.shape[0] + 1):
            for dc in range(-bottom_input_mask.shape[1], top_mask.shape[1] + 1):
                shifted_bottom = bottom_coords + np.array([dr, dc], dtype=int)
                top_set = {tuple(x) for x in top_coords.tolist()}
                bottom_set = {tuple(x) for x in shifted_bottom.tolist()}
                if top_set & bottom_set:
                    continue
                union = np.vstack([top_coords, shifted_bottom])
                normalized, _, _ = _normalize_coords(union)
                rect_h = int(normalized[:, 0].max()) + 1
                rect_w = int(normalized[:, 1].max()) + 1
                if rect_h * rect_w != len(top_coords) + len(bottom_coords):
                    continue
                top_norm, _, _ = _normalize_coords(top_coords)
                bottom_norm, _, _ = _normalize_coords(shifted_bottom)
                top_norm = top_coords - np.array([union[:, 0].min(), union[:, 1].min()], dtype=int)
                bottom_norm = shifted_bottom - np.array([union[:, 0].min(), union[:, 1].min()], dtype=int)
                if float(top_norm[:, 0].mean()) >= float(bottom_norm[:, 0].mean()):
                    continue
                if not np.any(top_norm[:, 0] == 0):
                    continue
                if not np.any(bottom_norm[:, 0] == rect_h - 1):
                    continue
                sig = _assembly_signature(top_norm, bottom_norm, (rect_h, rect_w), flipped)
                results[sig] = {
                    "shape": (rect_h, rect_w),
                    "flipped": flipped,
                    "top_coords": top_norm,
                    "bottom_coords": bottom_norm,
                }
    return list(results.values())


def _pick_assembly(assemblies: list[dict]) -> dict:
    assert assemblies, "expected at least one valid assembly"
    return min(
        assemblies,
        key=lambda item: (
            item["shape"][0] * item["shape"][1],
            item["shape"][0],
            item["shape"][1],
            int(item["flipped"]),
            _assembly_signature(item["top_coords"], item["bottom_coords"], item["shape"], item["flipped"]),
        ),
    )


def _solve_matching(top_objects: list[dict], bottom_objects: list[dict]) -> list[tuple[int, int, dict]]:
    assert len(top_objects) == len(bottom_objects) and 2 <= len(top_objects) <= 3
    pair_options: dict[tuple[int, int], list[dict]] = {}
    for ti, top in enumerate(top_objects):
        row_options = 0
        for bi, bottom in enumerate(bottom_objects):
            assemblies = _find_assemblies(top["mask"], bottom["mask"])
            if assemblies:
                pair_options[(ti, bi)] = assemblies
                row_options += 1
        assert row_options > 0, "every top object must have a matching bottom object"

    valid_matchings = []
    for perm in itertools.permutations(range(len(bottom_objects))):
        if all((ti, bi) in pair_options for ti, bi in enumerate(perm)):
            valid_matchings.append(perm)
    assert len(valid_matchings) == 1, "expected a unique bijection between top and bottom objects"

    chosen = []
    for ti, bi in enumerate(valid_matchings[0]):
        chosen.append((ti, bi, _pick_assembly(pair_options[(ti, bi)])))
    return chosen


def _render_patch(top_color: int, bottom_color: int, assembly: dict) -> np.ndarray:
    h, w = assembly["shape"]
    patch = np.full((h, w), BG, dtype=int)
    patch[assembly["top_coords"][:, 0], assembly["top_coords"][:, 1]] = int(top_color)
    patch[assembly["bottom_coords"][:, 0], assembly["bottom_coords"][:, 1]] = int(bottom_color)
    return patch


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _dominant_color(g)
    sep_row = _find_separator_row(g, bg)

    top_panel = g[:sep_row, :]
    bottom_panel = g[sep_row + 1 :, :]
    top_objects = sorted(_find_objects(top_panel, bg), key=lambda obj: (obj["global_col"], obj["global_row"]))
    bottom_objects = sorted(_find_objects(bottom_panel, bg, row_offset=sep_row + 1), key=lambda obj: (obj["global_col"], obj["global_row"]))
    assert len(top_objects) == len(bottom_objects) and 2 <= len(top_objects) <= 3

    matches = _solve_matching(top_objects, bottom_objects)
    patches = [
        _render_patch(top_objects[ti]["color"], bottom_objects[bi]["color"], assembly)
        for ti, bi, assembly in matches
    ]

    out_h = max(patch.shape[0] for patch in patches)
    out_w = sum(patch.shape[1] for patch in patches) + len(patches) - 1
    out = np.full((out_h, out_w), bg, dtype=int)
    c = 0
    for patch in patches:
        h, w = patch.shape
        out[:h, c : c + w] = patch
        c += w + 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble complementary objects from two panels into packed rectangles.

    Input structure:
    - The background is Azure, and a full Maroon horizontal line splits the grid into upper and lower panels.
    - Each panel contains 2 or 3 separated 4-connected monochrome foreground objects.
    - Every upper-panel object has exactly one lower-panel partner whose shape can complement it to fill a rectangle.

    Rule:
    - For each upper/lower object pair, test whether the lower object, optionally after a vertical flip, can be translated next to the upper object so their union is one filled rectangle.
    - Keep the upper object's color on its cells and the lower object's color on its cells in the reconstructed rectangle.
    - Discard the separator and place all reconstructed rectangles left to right, with one background column between them, ordered by the original column positions of the upper-panel objects.

    Key insight:
    - Pairing depends on geometric complementarity under vertical reflection and translation, not on color, size ranking, or original panel position.
    """
    return _solve(grid)


def _random_profile(rng: random.Random, rect_h: int, rect_w: int) -> list[int]:
    while True:
        profile = [rng.randint(1, rect_h - 1)]
        for _ in range(rect_w - 1):
            delta = rng.choice([-1, 0, 1])
            nxt = profile[-1] + delta
            nxt = max(1, min(rect_h - 1, nxt))
            profile.append(nxt)
        if len(set(profile)) == 1:
            continue
        if max(profile) == 1 or min(profile) == rect_h - 1:
            continue
        return profile


def _build_pair_masks(profile: list[int], rect_h: int) -> tuple[np.ndarray, np.ndarray]:
    rect_w = len(profile)
    top = np.zeros((rect_h, rect_w), dtype=bool)
    bottom = np.zeros((rect_h, rect_w), dtype=bool)
    for c, split in enumerate(profile):
        top[:split, c] = True
        bottom[split:, c] = True
    top_rows = np.where(top.any(axis=1))[0]
    bottom_rows = np.where(bottom.any(axis=1))[0]
    top = top[top_rows[0] : top_rows[-1] + 1, :]
    bottom = bottom[bottom_rows[0] : bottom_rows[-1] + 1, :]
    return top, bottom


def _split_extra(rng: random.Random, total: int, slots: int) -> list[int]:
    parts = [0] * slots
    for _ in range(total):
        parts[rng.randrange(slots)] += 1
    return parts


def _sample_positions(rng: random.Random, widths: list[int], canvas_w: int, *, margin_min: int = 1, gap_min: int = 2) -> list[int]:
    n = len(widths)
    required = sum(widths) + gap_min * (n - 1) + 2 * margin_min
    assert required <= canvas_w
    extras = _split_extra(rng, canvas_w - required, n + 1)
    left_margin = margin_min + extras[0]
    gaps = [gap_min + extras[i + 1] for i in range(n - 1)]
    x = left_margin
    positions = []
    for i, width in enumerate(widths):
        positions.append(x)
        x += width
        if i < n - 1:
            x += gaps[i]
    return positions


def _place_masks(
    rng: random.Random,
    masks: list[np.ndarray],
    colors: list[int],
    canvas_h: int,
    canvas_w: int,
    order: list[int],
) -> np.ndarray:
    widths = [int(masks[idx].shape[1]) for idx in order]
    xs = _sample_positions(rng, widths, canvas_w)
    panel = np.full((canvas_h, canvas_w), BG, dtype=int)
    for slot, idx in enumerate(order):
        mask = masks[idx]
        color = colors[idx]
        max_y = canvas_h - mask.shape[0] - 1
        y = rng.randint(1, max_y) if max_y >= 1 else 0
        x = xs[slot]
        region = panel[y : y + mask.shape[0], x : x + mask.shape[1]]
        assert np.all(region == BG)
        region[mask] = int(color)
    return panel


def _target_output(pair_specs: list[dict], top_order: list[int]) -> np.ndarray:
    patches = []
    for idx in top_order:
        spec = pair_specs[idx]
        patch = np.full((spec["rect_h"], spec["rect_w"]), BG, dtype=int)
        patch[spec["top_coords"][:, 0], spec["top_coords"][:, 1]] = int(spec["top_color"])
        patch[spec["bottom_coords"][:, 0], spec["bottom_coords"][:, 1]] = int(spec["bottom_color"])
        patches.append(patch)
    out_h = max(p.shape[0] for p in patches)
    out_w = sum(p.shape[1] for p in patches) + len(patches) - 1
    out = np.full((out_h, out_w), BG, dtype=int)
    c = 0
    for patch in patches:
        out[: patch.shape[0], c : c + patch.shape[1]] = patch
        c += patch.shape[1] + 1
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_pairs = rng.choice([2, 3])
    available_colors = rng.sample(PALETTE, 2 * n_pairs)
    top_colors = available_colors[:n_pairs]
    bottom_colors = available_colors[n_pairs:]

    pair_specs = []
    total_width = 0
    max_top_h = 0
    max_bottom_h = 0
    for i in range(n_pairs):
        rect_w = rng.randint(4, 7 if n_pairs == 2 else 6)
        rect_h = rng.randint(4, 7)
        profile = _random_profile(rng, rect_h, rect_w)
        top_mask, bottom_mask = _build_pair_masks(profile, rect_h)
        input_bottom_mask = np.flipud(bottom_mask) if rng.random() < 0.5 else bottom_mask.copy()
        assemblies = _find_assemblies(top_mask, input_bottom_mask)
        if len(assemblies) != 1:
            return None
        assembly = assemblies[0]
        pair_specs.append(
            {
                "top_mask": top_mask,
                "bottom_input_mask": input_bottom_mask,
                "top_color": top_colors[i],
                "bottom_color": bottom_colors[i],
                "top_coords": assembly["top_coords"],
                "bottom_coords": assembly["bottom_coords"],
                "rect_h": assembly["shape"][0],
                "rect_w": assembly["shape"][1],
            }
        )
        total_width += rect_w
        max_top_h = max(max_top_h, top_mask.shape[0])
        max_bottom_h = max(max_bottom_h, input_bottom_mask.shape[0])

    if total_width + 2 * (n_pairs - 1) + 4 > 30:
        return None

    top_panel_h = max_top_h + rng.randint(3, 6)
    bottom_panel_h = max_bottom_h + rng.randint(3, 6)
    min_canvas_w = max(
        total_width + 2 * (n_pairs - 1) + 2,
        max(spec["rect_w"] for spec in pair_specs) + 2,
    )
    max_canvas_w = min(30, min_canvas_w + rng.randint(0, 4))
    if min_canvas_w > max_canvas_w:
        return None
    canvas_w = rng.randint(min_canvas_w, max_canvas_w)
    total_h = top_panel_h + 1 + bottom_panel_h
    if total_h > 30:
        return None

    top_order = list(range(n_pairs))
    rng.shuffle(top_order)
    bottom_order = list(range(n_pairs))
    while True:
        rng.shuffle(bottom_order)
        if bottom_order != top_order or n_pairs == 2:
            break

    top_panel = _place_masks(
        rng,
        [spec["top_mask"] for spec in pair_specs],
        [spec["top_color"] for spec in pair_specs],
        top_panel_h,
        canvas_w,
        top_order,
    )
    bottom_panel = _place_masks(
        rng,
        [spec["bottom_input_mask"] for spec in pair_specs],
        [spec["bottom_color"] for spec in pair_specs],
        bottom_panel_h,
        canvas_w,
        bottom_order,
    )

    grid = np.full((total_h, canvas_w), BG, dtype=int)
    grid[:top_panel_h, :] = top_panel
    grid[top_panel_h, :] = SEP
    grid[top_panel_h + 1 :, :] = bottom_panel

    try:
        solved = _solve(grid)
    except AssertionError:
        return None

    expected = _target_output(pair_specs, top_order)
    if not np.array_equal(solved, expected):
        return None

    top_objects = _find_objects(grid[:top_panel_h, :], BG)
    bottom_objects = _find_objects(grid[top_panel_h + 1 :, :], BG)
    if len(top_objects) != n_pairs or len(bottom_objects) != n_pairs:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(random.Random())
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
