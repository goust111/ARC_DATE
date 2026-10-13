from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
ORANGE = 7
ACCENT_COLORS = (RED, GREEN, ORANGE)
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GRID_H_RANGE = (16, 23)
GRID_W_RANGE = (16, 23)
COPY_COUNT_OPTIONS = (3, 4)
TEMPLATE_SHAPES = ((4, 5), (5, 4))
PLACEMENT_GAP = 1
MAX_TEMPLATE_ATTEMPTS = 300
MAX_LAYOUT_ATTEMPTS = 200
MAX_GENERATION_ATTEMPTS = 600


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _border_mask(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _border_coords(shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    coords: list[tuple[int, int]] = []
    for c in range(w):
        coords.append((0, c))
    for r in range(1, h - 1):
        coords.append((r, w - 1))
    for c in range(w - 1, -1, -1):
        coords.append((h - 1, c))
    for r in range(h - 2, 0, -1):
        coords.append((r, 0))
    return coords


def _noncorner_border_coords(shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    corners = {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}
    return [pos for pos in _border_coords(shape) if pos not in corners]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                comp.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(comp)
    return comps


def _find_frames(grid: np.ndarray) -> list[dict]:
    frames: list[dict] = []
    for coords in _components(grid != BLACK):
        r0, c0, r1, c1 = _bbox_from_coords(coords)
        patch = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
        h, w = patch.shape
        if h < 3 or w < 3:
            continue
        border = _border_mask((h, w))
        if not np.array_equal(patch != BLACK, border):
            continue
        frames.append({"bbox": (r0, c0, r1, c1), "patch": patch})
    frames.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))
    return frames


def _dihedral_variants(patch: np.ndarray) -> list[dict]:
    variants: list[dict] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for flipped in (False, True):
        base = np.fliplr(patch) if flipped else patch
        for rot in range(4):
            transformed = np.rot90(base, rot)
            key = (tuple(map(int, transformed.shape)), transformed.tobytes())
            if key in seen:
                continue
            seen.add(key)
            variants.append(
                {
                    "name": f"{'F' if flipped else 'N'}{rot}",
                    "flipped": flipped,
                    "rot": rot,
                    "patch": transformed,
                }
            )
    return variants


def _match_copy(copy_patch: np.ndarray, variants: list[dict]) -> dict:
    matches: list[dict] = []
    for variant in variants:
        candidate = variant["patch"]
        if candidate.shape != copy_patch.shape:
            continue
        mismatch = np.argwhere(candidate != copy_patch)
        if mismatch.shape[0] != 1:
            continue
        rr, cc = map(int, mismatch[0])
        expected = int(candidate[rr, cc])
        actual = int(copy_patch[rr, cc])
        if actual != YELLOW or expected not in ACCENT_COLORS:
            continue
        matches.append(
            {
                "variant": variant,
                "local_pos": (rr, cc),
                "correct_color": expected,
            }
        )
    assert len(matches) == 1, "each copy must match exactly one transformed template with one yellow error"
    return matches[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect each transformed hollow frame's single yellow border error and use the recovered true color to fill that frame's cavity.

    Input structure:
    - The grid has a black background and several disjoint one-pixel-thick hollow rectangular frames.
    - Exactly one frame is the multicolor template: its border uses blue plus red/green/orange accents and contains no yellow.
    - Every other frame is a distinct dihedral transform of the template, except that exactly one non-corner border pixel has been changed to yellow.

    Rule:
    - Find the unique frame whose border contains no yellow; that frame is the template.
    - Enumerate all 8 dihedral transforms of the template and, for each yellow-containing frame, find the unique transform with exactly one pixel mismatch.
    - The mismatch cell is the tampered yellow pixel; replace it with the transformed template's color there, then fill the frame's strict interior with that recovered color.
    - Leave the template frame unchanged and keep all background pixels black.

    Key insight:
    - Yellow marks where a copy disagrees with the template, but the fill color is not fixed: it is the hidden border color revealed only after aligning the copy to the correct dihedral template transform.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    frames = _find_frames(g)
    assert len(frames) >= 4

    template_candidates = [frame for frame in frames if YELLOW not in frame["patch"][_border_mask(frame["patch"].shape)]]
    assert len(template_candidates) == 1, "expected exactly one template frame without yellow"
    template = template_candidates[0]
    variants = _dihedral_variants(template["patch"])

    out = g.copy()
    for frame in frames:
        if frame is template:
            continue
        match = _match_copy(frame["patch"], variants)
        r0, c0, r1, c1 = frame["bbox"]
        rr, cc = match["local_pos"]
        color = match["correct_color"]
        out[r0 + rr, c0 + cc] = color
        out[r0 + 1 : r1, c0 + 1 : c1] = color

    return out


def _random_template(copy_count: int) -> tuple[np.ndarray, list[int], list[dict]]:
    for _ in range(MAX_TEMPLATE_ATTEMPTS):
        h, w = random.choice(TEMPLATE_SHAPES)
        accent_palette = random.sample(list(ACCENT_COLORS), k=random.choice((2, 3)))
        patch = np.zeros((h, w), dtype=int)
        border = _border_mask((h, w))
        patch[border] = BLUE

        noncorner = _noncorner_border_coords((h, w))
        accent_count = random.randint(max(copy_count + 1, 4), min(len(noncorner), 8))
        accent_positions = random.sample(noncorner, accent_count)
        color_bag = [accent_palette[i % len(accent_palette)] for i in range(accent_count)]
        random.shuffle(color_bag)
        for (r, c), color in zip(accent_positions, color_bag):
            patch[r, c] = color

        touched_sides = set()
        for r, c in accent_positions:
            if r == 0:
                touched_sides.add("top")
            if r == h - 1:
                touched_sides.add("bottom")
            if c == 0:
                touched_sides.add("left")
            if c == w - 1:
                touched_sides.add("right")
        if len(touched_sides) < 3:
            continue

        variants = _dihedral_variants(patch)
        if len(variants) != 8:
            continue
        return patch, accent_palette, variants
    raise RuntimeError("failed to build an asymmetric template")


def _choose_copy_variants(template: np.ndarray, variants: list[dict], copy_count: int) -> list[dict] | None:
    template_shape = tuple(map(int, template.shape))
    swapped_shape = (template_shape[1], template_shape[0])
    for _ in range(200):
        chosen = random.sample(variants, copy_count)
        shapes = {tuple(map(int, item["patch"].shape)) for item in chosen}
        if template_shape not in shapes or swapped_shape not in shapes:
            continue
        if not any(item["flipped"] for item in chosen):
            continue
        if not any(not item["flipped"] for item in chosen):
            continue
        return chosen
    return None


def _choose_mutations(chosen_variants: list[dict], accent_palette: list[int]) -> list[dict] | None:
    preferred_colors = accent_palette[:]
    while len(preferred_colors) < len(chosen_variants):
        preferred_colors.append(random.choice(accent_palette))
    random.shuffle(preferred_colors)

    mutations: list[dict] = []
    used_colors: list[int] = []
    for variant, preferred in zip(chosen_variants, preferred_colors):
        candidates = [
            (r, c, int(variant["patch"][r, c]))
            for r, c in _noncorner_border_coords(variant["patch"].shape)
            if int(variant["patch"][r, c]) in ACCENT_COLORS
        ]
        if not candidates:
            return None
        exact = [cand for cand in candidates if cand[2] == preferred]
        r, c, color = random.choice(exact if exact else candidates)
        bad_patch = variant["patch"].copy()
        bad_patch[r, c] = YELLOW
        mutations.append(
            {
                "variant": variant,
                "bad_patch": bad_patch,
                "local_pos": (r, c),
                "correct_color": color,
            }
        )
        used_colors.append(color)
    if len(set(used_colors)) < 2:
        return None
    return mutations


def _boxes_clear(box: tuple[int, int, int, int], placed: list[tuple[int, int, int, int]], gap: int) -> bool:
    r0, c0, r1, c1 = box
    for pr0, pc0, pr1, pc1 in placed:
        if not (r1 + gap < pr0 or pr1 + gap < r0 or c1 + gap < pc0 or pc1 + gap < c0):
            return False
    return True


def _place_patches(shapes: list[tuple[int, int]], h: int, w: int) -> list[tuple[int, int]] | None:
    order = list(range(len(shapes)))
    order.sort(key=lambda idx: shapes[idx][0] * shapes[idx][1], reverse=True)
    placed_boxes: list[tuple[int, int, int, int]] = []
    positions: list[tuple[int, int] | None] = [None] * len(shapes)

    def search(i: int) -> bool:
        if i == len(order):
            return True
        idx = order[i]
        ph, pw = shapes[idx]
        candidates = [(r, c) for r in range(h - ph + 1) for c in range(w - pw + 1)]
        random.shuffle(candidates)
        for r, c in candidates:
            box = (r, c, r + ph - 1, c + pw - 1)
            if not _boxes_clear(box, placed_boxes, PLACEMENT_GAP):
                continue
            placed_boxes.append(box)
            positions[idx] = (r, c)
            if search(i + 1):
                return True
            positions[idx] = None
            placed_boxes.pop()
        return False

    return [pos for pos in positions if pos is not None] if search(0) else None


def _build_instance() -> np.ndarray | None:
    copy_count = random.choice(COPY_COUNT_OPTIONS)
    template, accent_palette, variants = _random_template(copy_count)
    chosen_variants = _choose_copy_variants(template, variants, copy_count)
    if chosen_variants is None:
        return None
    mutations = _choose_mutations(chosen_variants, accent_palette)
    if mutations is None:
        return None

    patches = [template] + [item["bad_patch"] for item in mutations]
    shapes = [tuple(map(int, patch.shape)) for patch in patches]

    for _ in range(MAX_LAYOUT_ATTEMPTS):
        h = random.randint(*GRID_H_RANGE)
        w = random.randint(*GRID_W_RANGE)
        positions = _place_patches(shapes, h, w)
        if positions is None:
            continue
        order = list(range(len(patches)))
        random.shuffle(order)
        grid = np.zeros((h, w), dtype=int)
        for idx in order:
            patch = patches[idx]
            r0, c0 = positions[idx]
            ph, pw = patch.shape
            if np.any(grid[r0 : r0 + ph, c0 : c0 + pw] != BLACK):
                break
            grid[r0 : r0 + ph, c0 : c0 + pw] = patch
        else:
            if _validate_instance(grid):
                return grid
    return None


def _validate_instance(grid: np.ndarray) -> bool:
    try:
        frames = _find_frames(grid)
        if len(frames) not in (4, 5):
            return False

        template_candidates = [frame for frame in frames if YELLOW not in frame["patch"][_border_mask(frame["patch"].shape)]]
        if len(template_candidates) != 1:
            return False
        template = template_candidates[0]
        variants = _dihedral_variants(template["patch"])
        if len(variants) != 8:
            return False

        matches = []
        fill_colors = []
        shapes = set()
        flips = set()
        for frame in frames:
            if frame is template:
                continue
            match = _match_copy(frame["patch"], variants)
            matches.append(match["variant"]["name"])
            fill_colors.append(match["correct_color"])
            shapes.add(tuple(map(int, frame["patch"].shape)))
            flips.add(bool(match["variant"]["flipped"]))

            rr, cc = match["local_pos"]
            h, w = frame["patch"].shape
            if (rr, cc) in {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}:
                return False

        if len(matches) != len(set(matches)):
            return False
        if len(set(fill_colors)) < 2:
            return False
        if len(shapes) < 2:
            return False
        if flips != {False, True}:
            return False

        out = main(grid)
        if out.shape != grid.shape:
            return False

        changed = 0
        for frame in frames:
            r0, c0, r1, c1 = frame["bbox"]
            out_patch = out[r0 : r1 + 1, c0 : c1 + 1]
            if frame is template:
                if not np.array_equal(out_patch, frame["patch"]):
                    return False
                continue
            match = _match_copy(frame["patch"], variants)
            color = match["correct_color"]
            rr, cc = match["local_pos"]
            if int(out_patch[rr, cc]) != color:
                return False
            if not np.all(out_patch[1:-1, 1:-1] == color):
                return False
            changed += int(np.count_nonzero(out_patch != frame["patch"]))

        return changed > 0
    except AssertionError:
        return False


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_instance()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
