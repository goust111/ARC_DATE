from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import types

import numpy as np


def _load_arc_utils():
    path = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
    module = types.ModuleType("arc_utils_embedded")
    code = compile(path.read_text(), str(path), "exec")
    exec(code, module.__dict__)
    return module


_ARC = _load_arc_utils()
components_mask = _ARC.components_mask
crop_bbox = _ARC.crop_bbox
find_components = _ARC.find_components
get_bg = _ARC.get_bg
paste = _ARC.paste


BG = 0
PALETTE = tuple(range(1, 10))
GEN_RETRIES = 400
PLACEMENT_TRIES = 250


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], tuple[int, ...]]:
    arr = mask.astype(np.uint8)
    a = tuple(int(v) for v in arr.flatten())
    b = tuple(int(v) for v in np.fliplr(arr).flatten())
    return tuple(mask.shape), min(a, b)


def _orientation_missing(full_mask: np.ndarray, observed_mask: np.ndarray) -> int | None:
    if full_mask.shape != observed_mask.shape:
        return None
    if np.any(observed_mask & ~full_mask):
        return None
    return int(np.sum(full_mask & ~observed_mask))


def _component_records(grid: np.ndarray) -> tuple[int, list[dict]]:
    bg = get_bg(grid)
    records = []
    for comp in find_components(grid, bg=bg, connectivity=8, monochromatic=False):
        r0, c0, r1, c1 = map(int, comp["bbox"])
        crop = crop_bbox(grid, comp["bbox"])
        mask = crop != bg
        colors = {int(v) for v in crop[mask]}
        assert len(colors) == 1
        records.append(
            {
                "bbox": (r0, c0, r1, c1),
                "crop": crop,
                "mask": mask,
                "color": colors.pop(),
                "canonical": _mask_key(mask),
                "area": int(mask.sum()),
            }
        )
    return int(bg), records


def main(grid: np.ndarray) -> np.ndarray:
    """
    Distinguish the legitimate mirrored sprite copy from the truly damaged copy and repair only the latter.

    Input structure:
    - A background canvas contains several non-touching translated copies of one monochrome sprite, each in a different color.
    - The sprite mask is left-right asymmetric inside its tight bbox.
    - Exactly one copy is a complete left-right mirror, and exactly one different copy is missing some sprite pixels.

    Rule:
    - Extract every copy as a tight-bbox mask and compare the copies under left-right mirror equivalence.
    - Use the unique majority mirror-equivalence class as the consensus full sprite shape.
    - Copies exactly matching the consensus or its mirror stay unchanged; the lone remaining copy is damaged.
    - Choose the damaged copy's orientation between consensus and mirror by the unique minimum number of missing pixels, then fill those missing cells at the damaged copy's original location using its own color.

    Key insight:
    - There are two apparent outliers under exact comparison, but only one is defective: mirror symmetry is a legal variant, so consensus must be formed modulo left-right reflection.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg, records = _component_records(g)
    assert len(records) >= 4

    canonical_counts = Counter(record["canonical"] for record in records)
    best_freq = max(canonical_counts.values())
    winners = [key for key, count in canonical_counts.items() if count == best_freq]
    assert len(winners) == 1
    consensus_key = winners[0]

    consensus_records = [record for record in records if record["canonical"] == consensus_key]
    representative = max(consensus_records, key=lambda record: record["area"])
    consensus_mask = representative["mask"]
    mirrored_mask = np.fliplr(consensus_mask)

    damaged = []
    for record in records:
        if np.array_equal(record["mask"], consensus_mask) or np.array_equal(record["mask"], mirrored_mask):
            continue
        else:
            damaged.append(record)

    assert len(damaged) == 1
    damaged_record = damaged[0]

    scores = []
    for candidate in (consensus_mask, mirrored_mask):
        missing = _orientation_missing(candidate, damaged_record["mask"])
        if missing is not None:
            scores.append((missing, candidate))

    assert scores
    best_missing = min(score for score, _candidate in scores)
    best_candidates = [candidate for score, candidate in scores if score == best_missing]
    assert len(best_candidates) == 1
    repaired_mask = best_candidates[0]
    assert best_missing > 0

    out = g.copy()
    r0, c0, r1, c1 = damaged_record["bbox"]
    patch = np.full(repaired_mask.shape, bg, dtype=int)
    patch[repaired_mask] = int(damaged_record["color"])
    existing = out[r0 : r1 + 1, c0 : c1 + 1]
    fill_mask = repaired_mask & (existing == bg)
    patch[~fill_mask] = bg
    paste(out, patch, r0, c0, transparent=bg)
    return out


def _is_connected(mask: np.ndarray) -> bool:
    comps = components_mask(mask.astype(bool), connectivity=4)
    return len(comps) == 1


def _frontier(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    out = set()
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                out.add((nr, nc))
    return list(out)


def _carve_path(mask: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> None:
    r, c = start
    tr, tc = end
    mask[r, c] = True
    while (r, c) != (tr, tc):
        choices = []
        if r < tr:
            choices.append((r + 1, c))
        elif r > tr:
            choices.append((r - 1, c))
        if c < tc:
            choices.append((r, c + 1))
        elif c > tc:
            choices.append((r, c - 1))
        r, c = random.choice(choices)
        mask[r, c] = True


def _make_base_mask() -> np.ndarray:
    for _ in range(250):
        h = random.randint(5, 8)
        w = random.randint(6, 9)
        target = random.randint(max(12, (h * w) // 3), min(h * w - 3, (h * w * 2) // 3))
        mask = np.zeros((h, w), dtype=bool)

        anchors = [
            (0, random.randrange(w)),
            (h - 1, random.randrange(w)),
            (random.randrange(h), 0),
            (random.randrange(h), w - 1),
        ]
        center = (random.randrange(h), random.randrange(w))
        for anchor in anchors:
            _carve_path(mask, center, anchor)

        while int(mask.sum()) < target:
            candidates = _frontier(mask)
            if not candidates:
                break
            random.shuffle(candidates)
            mask[candidates[0]] = True

        if not _is_connected(mask):
            continue
        if not mask[0].any() or not mask[-1].any() or not mask[:, 0].any() or not mask[:, -1].any():
            continue
        if np.array_equal(mask, np.fliplr(mask)):
            continue
        if int(np.sum(mask ^ np.fliplr(mask))) < 4:
            continue
        return mask
    raise RuntimeError("failed to generate an asymmetric sprite")


def _protected_edge_cells(mask: np.ndarray) -> set[tuple[int, int]]:
    h, w = mask.shape
    protected = {
        (0, random.choice(np.flatnonzero(mask[0]))),
        (h - 1, random.choice(np.flatnonzero(mask[-1]))),
        (random.choice(np.flatnonzero(mask[:, 0])), 0),
        (random.choice(np.flatnonzero(mask[:, -1])), w - 1),
    }
    return {(int(r), int(c)) for r, c in protected}


def _make_damaged_mask(full_mask: np.ndarray) -> np.ndarray:
    other = np.fliplr(full_mask)
    distinguishing = {
        (int(r), int(c))
        for r, c in np.argwhere(full_mask & ~other)
    }
    for _ in range(400):
        protected = _protected_edge_cells(full_mask)
        removable = [(int(r), int(c)) for r, c in np.argwhere(full_mask) if (int(r), int(c)) not in protected]
        if not removable:
            continue
        preferred = [cell for cell in removable if cell in distinguishing]
        others = [cell for cell in removable if cell not in distinguishing]
        max_remove = min(5, len(removable), max(2, int(full_mask.sum()) // 4))
        remove_count = random.randint(1, max_remove)
        removed = set()
        if preferred:
            removed.add(random.choice(preferred))
        pool = [cell for cell in others + preferred if cell not in removed]
        need = remove_count - len(removed)
        if need > len(pool):
            continue
        removed.update(random.sample(pool, need))
        damaged = full_mask.copy()
        for r, c in removed:
            damaged[r, c] = False
        if not _is_connected(damaged):
            continue
        if not damaged[0].any() or not damaged[-1].any() or not damaged[:, 0].any() or not damaged[:, -1].any():
            continue
        same_missing = _orientation_missing(full_mask, damaged)
        mirror_missing = _orientation_missing(other, damaged)
        assert same_missing is not None
        if mirror_missing is None or same_missing < mirror_missing:
            return damaged
    raise RuntimeError("failed to generate a uniquely oriented damaged sprite")


def _render_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.full(mask.shape, BG, dtype=int)
    patch[mask] = int(color)
    return patch


def _can_place(occupied: np.ndarray, mask: np.ndarray, r0: int, c0: int) -> bool:
    h, w = mask.shape
    H, W = occupied.shape
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(H, r0 + h + 1)
    cc1 = min(W, c0 + w + 1)
    return not occupied[rr0:rr1, cc0:cc1].any()


def _mark_forbidden(occupied: np.ndarray, mask: np.ndarray, r0: int, c0: int) -> None:
    H, W = occupied.shape
    h, w = mask.shape
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(H, r0 + h + 1)
    cc1 = min(W, c0 + w + 1)
    occupied[rr0:rr1, cc0:cc1] = True


def _build_case() -> tuple[np.ndarray, np.ndarray]:
    for _case_attempt in range(120):
        base = _make_base_mask()
        mirror = np.fliplr(base)
        damaged_is_mirror = random.random() < 0.5
        damaged_full = mirror if damaged_is_mirror else base
        try:
            damaged = _make_damaged_mask(damaged_full)
        except RuntimeError:
            continue

        copy_count = random.randint(4, 6)
        colors = random.sample(PALETTE, copy_count)

        patches_in: list[np.ndarray] = []
        patches_out: list[np.ndarray] = []

        normal_full_count = copy_count - 2
        for idx in range(normal_full_count):
            color = colors[idx]
            patches_in.append(_render_patch(base, color))
            patches_out.append(_render_patch(base, color))

        mirror_color = colors[normal_full_count]
        patches_in.append(_render_patch(mirror, mirror_color))
        patches_out.append(_render_patch(mirror, mirror_color))

        damaged_color = colors[-1]
        patches_in.append(_render_patch(damaged, damaged_color))
        patches_out.append(_render_patch(damaged_full, damaged_color))

        area_budget = sum(int(np.count_nonzero(patch)) for patch in patches_out)
        min_side = max(max(base.shape) + 4, int(np.sqrt(area_budget * 3)) + 4)
        min_side = min(min_side, 28)

        for _ in range(250):
            H = random.randint(min_side, 30)
            W = random.randint(min_side, 30)
            canvas = np.full((H, W), BG, dtype=int)
            target = np.full((H, W), BG, dtype=int)
            occupied = np.zeros((H, W), dtype=bool)

            order = list(range(copy_count))
            random.shuffle(order)
            placements = []
            ok = True
            for idx in order:
                patch = patches_out[idx]
                h, w = patch.shape
                coords = [(r, c) for r in range(H - h + 1) for c in range(W - w + 1)]
                random.shuffle(coords)
                placed = None
                for r0, c0 in coords[:PLACEMENT_TRIES]:
                    if _can_place(occupied, patch != BG, r0, c0):
                        placed = (r0, c0)
                        break
                if placed is None:
                    ok = False
                    break
                r0, c0 = placed
                placements.append((idx, r0, c0))
                _mark_forbidden(occupied, patch != BG, r0, c0)
            if not ok:
                continue

            for idx, r0, c0 in placements:
                paste(canvas, patches_in[idx], r0, c0, transparent=BG)
                paste(target, patches_out[idx], r0, c0, transparent=BG)

            if np.array_equal(canvas, target):
                continue
            solved = main(canvas)
            if not np.array_equal(solved, target):
                continue
            changed = canvas != solved
            if int(changed.sum()) == 0:
                continue
            return canvas, target

    raise RuntimeError("failed to place sprite copies")


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        inp, out = _build_case()
        if np.array_equal(main(inp), out):
            return inp
    raise RuntimeError("failed to generate a valid input")
