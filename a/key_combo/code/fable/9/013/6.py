import random
from collections import Counter

import numpy as np


BG = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
MARKER_CORNERS = ("tl", "tr", "bl", "br")
GEN_RETRIES = 500


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _corner_rc(shape: tuple[int, int], corner: str) -> tuple[int, int]:
    h, w = map(int, shape)
    if corner == "tl":
        return 0, 0
    if corner == "tr":
        return 0, w - 1
    if corner == "bl":
        return h - 1, 0
    if corner == "br":
        return h - 1, w - 1
    raise ValueError(f"unknown corner {corner}")


def _swap_patch_colors(patch: np.ndarray, marker_color: int) -> np.ndarray:
    out = patch.copy()
    pattern_colors = [int(v) for v in np.unique(patch) if int(v) != int(marker_color)]
    assert len(pattern_colors) == 2
    a, b = pattern_colors
    mask_a = out == a
    mask_b = out == b
    out[mask_a] = b
    out[mask_b] = a
    return out


def _canonical_shape_signature(patch: np.ndarray, marker_corner: str) -> tuple[tuple[int, ...], ...]:
    rr, cc = _corner_rc(patch.shape, marker_corner)
    marker_color = int(patch[rr, cc])
    pattern_colors = [int(v) for v in np.unique(patch) if int(v) != marker_color]
    assert len(pattern_colors) == 2
    a, b = pattern_colors
    enc = np.full(patch.shape, 2, dtype=int)
    enc[patch == a] = 0
    enc[patch == b] = 1
    inv = enc.copy()
    inv[enc == 0] = 1
    inv[enc == 1] = 0
    sig1 = tuple(tuple(int(v) for v in row) for row in enc)
    sig2 = tuple(tuple(int(v) for v in row) for row in inv)
    return min(sig1, sig2)


def _extract_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _most_common_color(g)
    comps = _components(g != bg)
    assert comps, "expected at least one foreground component"

    blocks = []
    sizes = set()
    for coords in comps:
        box = _bbox(coords)
        patch = _crop(g, box)
        h, w = patch.shape
        sizes.add((h, w))
        assert len(coords) == h * w, "each component must fill its rectangular bbox"
        blocks.append({"bbox": box, "patch": patch, "area": h * w})

    assert len(sizes) == 1, "all blocks must have the same size"

    visible = []
    placeholders = []
    for block in blocks:
        patch = block["patch"]
        values = [int(v) for v in np.unique(patch)]
        if values == [GREY]:
            placeholders.append(block)
            continue

        counts = Counter(map(int, patch.flatten()))
        marker_candidates = []
        for corner in MARKER_CORNERS:
            rr, cc = _corner_rc(patch.shape, corner)
            col = int(patch[rr, cc])
            if counts[col] == 1:
                marker_candidates.append((corner, col))

        assert len(marker_candidates) == 1, "each visible block must have one unique corner marker"
        marker_corner, marker_color = marker_candidates[0]
        pattern_colors = [int(v) for v in values if int(v) != marker_color]
        assert len(pattern_colors) == 2, "each visible block must contain exactly two pattern colors"
        assert min(counts[c] for c in pattern_colors) >= 2, "pattern colors must remain visible after marking"

        visible.append(
            {
                **block,
                "marker_corner": marker_corner,
                "marker_color": marker_color,
                "pattern_colors": tuple(sorted(pattern_colors)),
                "shape_sig": _canonical_shape_signature(patch, marker_corner),
            }
        )

    assert len(placeholders) == 1, "expected exactly one grey placeholder block"
    assert visible, "expected visible pattern blocks"
    return {"bg": bg, "visible": visible, "placeholder": placeholders[0]}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the lone marker color to find the grey placeholder's partner, then restore its swapped-color twin.

    Input structure:
    - A background contains several separated solid rectangular blocks of the same size.
    - Every visible block is fully filled by exactly two pattern colors plus one single-pixel marker at a corner.
    - Blocks come in marker-color pairs: two blocks with the same marker color share the same cellwise pattern, but their two pattern colors are swapped.
    - Exactly one block is replaced by a solid grey rectangle, so one marker color appears only once among the visible blocks.

    Rule:
    - Find the visible block whose marker color is unpaired.
    - Copy that block and swap its two pattern colors while keeping the marker pixel unchanged.
    - Write that swapped copy into the grey placeholder location; leave every other cell unchanged.

    Key insight:
    - Pairing is determined by the corner marker color, not by position and not by shape alone; the missing block is the swapped-color twin of the lone visible marker block.
    """
    scene = _extract_scene(np.asarray(grid, dtype=int))
    visible = scene["visible"]
    placeholder = scene["placeholder"]

    marker_counts = Counter(block["marker_color"] for block in visible)
    lone_markers = [color for color, count in marker_counts.items() if count == 1]
    assert len(lone_markers) == 1, "expected exactly one unpaired marker color"
    lone_color = lone_markers[0]

    source = next(block for block in visible if block["marker_color"] == lone_color)
    restored = _swap_patch_colors(source["patch"], source["marker_color"])

    out = np.asarray(grid, dtype=int).copy()
    r0, c0, r1, c1 = placeholder["bbox"]
    out[r0 : r1 + 1, c0 : c1 + 1] = restored
    return out


def _sample_pattern_mask(h: int, w: int, marker_corner: str, rng: random.Random) -> np.ndarray:
    rr_mark, cc_mark = _corner_rc((h, w), marker_corner)
    for _ in range(80):
        style = rng.choice(("row", "col", "diag", "corner"))
        mask = np.zeros((h, w), dtype=int)

        if style == "row":
            base = rng.randint(1, h - 2)
            for c in range(w):
                cut = min(h - 1, max(1, base + rng.randint(-1, 1)))
                mask[cut:, c] = 1
        elif style == "col":
            base = rng.randint(1, w - 2)
            for r in range(h):
                cut = min(w - 1, max(1, base + rng.randint(-1, 1)))
                mask[r, cut:] = 1
        elif style == "diag":
            bias = rng.randint(-(min(h, w) // 2), min(h, w) // 2)
            for r in range(h):
                for c in range(w):
                    mask[r, c] = int(r - c >= bias + rng.randint(-1, 1))
        else:
            pivot_r = rng.randint(1, h - 2)
            pivot_c = rng.randint(1, w - 2)
            for r in range(h):
                for c in range(w):
                    mask[r, c] = int((r >= pivot_r) ^ (c >= pivot_c))

        counts = Counter(mask.flatten())
        if len(counts) < 2:
            continue
        visible_counts = Counter(mask.flatten())
        visible_counts[int(mask[rr_mark, cc_mark])] -= 1
        if min(visible_counts.values()) < 2:
            continue
        sig = tuple(tuple(int(v) for v in row) for row in mask)
        comp = tuple(tuple(1 - int(v) for v in row) for row in mask)
        if sig == comp:
            continue
        return mask
    raise RuntimeError("failed to sample a non-degenerate pattern mask")


def _make_patch(mask: np.ndarray, colors: tuple[int, int], marker_color: int, marker_corner: str) -> np.ndarray:
    a, b = colors
    patch = np.where(mask == 0, a, b).astype(int)
    rr, cc = _corner_rc(mask.shape, marker_corner)
    patch[rr, cc] = marker_color
    return patch


def _can_place(
    positions: list[tuple[int, int]],
    candidate: tuple[int, int],
    h: int,
    w: int,
    gap: int,
) -> bool:
    r, c = candidate
    for rr, cc in positions:
        if not (r + h - 1 + gap < rr or rr + h - 1 + gap < r or c + w - 1 + gap < cc or cc + w - 1 + gap < c):
            return False
    return True


def _sample_positions(
    rows: int,
    cols: int,
    h: int,
    w: int,
    count: int,
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    candidates = [(r, c) for r in range(1, rows - h) for c in range(1, cols - w)]
    rng.shuffle(candidates)
    positions: list[tuple[int, int]] = []
    for cand in candidates:
        if _can_place(positions, cand, h, w, gap=1):
            positions.append(cand)
            if len(positions) == count:
                return positions
    return None


def _distance(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _validate_instance(inp: np.ndarray, expected_out: np.ndarray) -> bool:
    if inp.ndim != 2 or expected_out.ndim != 2 or max(inp.shape) > 30 or max(expected_out.shape) > 30:
        return False

    predicted = main(inp)
    if not np.array_equal(predicted, expected_out):
        return False

    scene = _extract_scene(inp)
    visible = scene["visible"]
    placeholder = scene["placeholder"]
    marker_counts = Counter(block["marker_color"] for block in visible)
    if Counter(marker_counts.values()) != Counter({2: len(marker_counts) - 1, 1: 1}):
        return False

    lone_block = next(block for block in visible if marker_counts[block["marker_color"]] == 1)
    shape_counts = Counter(block["shape_sig"] for block in visible)
    if shape_counts[lone_block["shape_sig"]] < 3:
        return False
    if len(shape_counts) < 2:
        return False

    ph_r, ph_c, _, _ = placeholder["bbox"]
    buddy_d = _distance((ph_r, ph_c), lone_block["bbox"][:2])
    decoys = [block for block in visible if block["shape_sig"] == lone_block["shape_sig"] and block["marker_color"] != lone_block["marker_color"]]
    if len(decoys) < 2 or not any(_distance((ph_r, ph_c), block["bbox"][:2]) <= buddy_d for block in decoys):
        return False

    diff = inp != expected_out
    r0, c0, r1, c1 = placeholder["bbox"]
    outside = diff.copy()
    outside[r0 : r1 + 1, c0 : c1 + 1] = False
    if outside.any() or not diff[r0 : r1 + 1, c0 : c1 + 1].any():
        return False

    restored = _swap_patch_colors(lone_block["patch"], lone_block["marker_color"])
    if not np.array_equal(expected_out[r0 : r1 + 1, c0 : c1 + 1], restored):
        return False

    return True


def _gen_attempt(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    pair_count = rng.randint(3, 4)
    block_h = rng.randint(4, 6)
    block_w = rng.randint(4, 6)
    marker_corner = rng.choice(MARKER_CORNERS)
    rows = rng.randint(max(18, block_h * 3 + 2), 29)
    cols = rng.randint(max(18, block_w * 3 + 2), 29)
    total_blocks = pair_count * 2

    positions = _sample_positions(rows, cols, block_h, block_w, total_blocks, rng)
    if positions is None:
        return None

    hidden_pair = rng.randrange(pair_count)
    repeated_partner = rng.choice([idx for idx in range(pair_count) if idx != hidden_pair])
    hidden_variant = rng.randrange(2)

    shape_masks: list[np.ndarray] = [None] * pair_count  # type: ignore[assignment]
    shared_mask = _sample_pattern_mask(block_h, block_w, marker_corner, rng)
    shape_masks[hidden_pair] = shared_mask
    shape_masks[repeated_partner] = shared_mask
    for idx in range(pair_count):
        if shape_masks[idx] is not None:
            continue
        for _ in range(40):
            candidate = _sample_pattern_mask(block_h, block_w, marker_corner, rng)
            if tuple(tuple(int(v) for v in row) for row in candidate) != tuple(tuple(int(v) for v in row) for row in shared_mask):
                shape_masks[idx] = candidate
                break
        if shape_masks[idx] is None:
            return None

    marker_colors = rng.sample(PALETTE, pair_count)
    all_pairs = [(a, b) for i, a in enumerate(PALETTE) for b in PALETTE[i + 1 :]]
    rng.shuffle(all_pairs)
    color_pairs: list[tuple[int, int]] = []
    for marker in marker_colors:
        picked = None
        for pair in all_pairs:
            if pair in color_pairs:
                continue
            if marker not in pair:
                picked = pair
                break
        if picked is None:
            return None
        color_pairs.append(picked)

    pair_patches = []
    for idx in range(pair_count):
        mask = shape_masks[idx]
        colors = color_pairs[idx]
        marker = marker_colors[idx]
        patch_a = _make_patch(mask, colors, marker, marker_corner)
        patch_b = _make_patch(mask, (colors[1], colors[0]), marker, marker_corner)
        pair_patches.append((patch_a, patch_b))

    hidden_slot = rng.randrange(total_blocks)
    others = [i for i in range(total_blocks) if i != hidden_slot]
    hidden_anchor = positions[hidden_slot]
    others.sort(key=lambda idx: (_distance(hidden_anchor, positions[idx]), idx))
    if len(others) < 3:
        return None
    decoy_slots = others[:2]
    buddy_slot = rng.choice(others[2:])

    slot_to_patch: dict[int, np.ndarray] = {}
    hidden_actual = pair_patches[hidden_pair][hidden_variant]
    buddy_patch = pair_patches[hidden_pair][1 - hidden_variant]
    slot_to_patch[hidden_slot] = np.full((block_h, block_w), GREY, dtype=int)
    slot_to_patch[buddy_slot] = buddy_patch
    slot_to_patch[decoy_slots[0]] = pair_patches[repeated_partner][0]
    slot_to_patch[decoy_slots[1]] = pair_patches[repeated_partner][1]

    remaining_slots = [i for i in range(total_blocks) if i not in slot_to_patch]
    remaining_patches = []
    for idx in range(pair_count):
        if idx == hidden_pair:
            continue
        if idx == repeated_partner:
            continue
        remaining_patches.extend(pair_patches[idx])
    rng.shuffle(remaining_slots)
    rng.shuffle(remaining_patches)
    for slot, patch in zip(remaining_slots, remaining_patches):
        slot_to_patch[slot] = patch

    inp = np.full((rows, cols), BG, dtype=int)
    out = np.full((rows, cols), BG, dtype=int)
    for idx, (r, c) in enumerate(positions):
        inp[r : r + block_h, c : c + block_w] = slot_to_patch[idx]
        out[r : r + block_h, c : c + block_w] = hidden_actual if idx == hidden_slot else slot_to_patch[idx]

    if not _validate_instance(inp, out):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        sample = _gen_attempt(rng)
        if sample is not None:
            return sample[0]
    raise RuntimeError("failed to generate a valid input for puzzle 6")
