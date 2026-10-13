from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
ALERT = 2
GREY = 5
TEMPLATE_COLORS = (1, 3, 4, 6, 7, 8, 9)

GRID_SIDE_RANGE = (18, 24)
OBJECT_COUNTS = (3, 4)
SIZE_OPTIONS = (
    (3, 3),
    (3, 4),
    (4, 3),
    (3, 5),
    (5, 3),
    (4, 4),
    (4, 5),
    (5, 4),
    (3, 6),
    (6, 3),
    (5, 5),
)
PLACEMENT_GAP = 1
GEN_RETRIES = 500


def _neighbors(r: int, c: int, connectivity: int) -> tuple[tuple[int, int], ...]:
    if connectivity == 4:
        return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
    return (
        (r - 1, c - 1),
        (r - 1, c),
        (r - 1, c + 1),
        (r, c - 1),
        (r, c + 1),
        (r + 1, c - 1),
        (r + 1, c),
        (r + 1, c + 1),
    )


def _components(mask: np.ndarray, connectivity: int) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors(rr, cc, connectivity):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _qualifies(mask: np.ndarray) -> tuple[bool, int, int]:
    comps = _components(mask, connectivity=4)
    total = int(mask.sum())
    largest = max(len(comp) for comp in comps)
    return largest * 2 > total, int(largest), int(total)


def _extract_templates(grid: np.ndarray) -> list[dict]:
    mask = (grid != BG) & (grid != GREY)
    templates = []
    for coords in _components(mask, connectivity=8):
        box = _bbox(coords)
        patch = _crop(grid, box)
        colors = {int(v) for v in np.unique(patch) if int(v) not in (BG, GREY)}
        assert 2 <= len(colors) <= 3, "each template must be multicolor"
        local_mask = patch != BG
        assert local_mask[0].any() and local_mask[-1].any()
        assert local_mask[:, 0].any() and local_mask[:, -1].any()
        templates.append(
            {
                "bbox": box,
                "patch": patch,
                "mask": local_mask,
                "size": tuple(map(int, patch.shape)),
                "colors": colors,
            }
        )
    return templates


def _extract_targets(grid: np.ndarray) -> list[dict]:
    targets = []
    for coords in _components(grid == GREY, connectivity=4):
        box = _bbox(coords)
        patch = _crop(grid, box)
        h, w = patch.shape
        assert len(coords) == h * w, "grey targets must be solid rectangles"
        assert np.all(patch == GREY), "target rectangle must be fully grey"
        targets.append({"bbox": box, "size": (int(h), int(w))})
    return targets


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy only size-matched templates whose non-black pixels have a dominant 4-connected component, and paint failed matches red.

    Input structure:
    - A black background contains several multicolor template regions and several solid grey target rectangles.
    - Each template is an 8-connected multicolor object whose tight bbox defines its size, and each target is a solid grey rectangle.
    - Every template size appears exactly once and matches exactly one grey target of the same height and width.

    Rule:
    - Detect all templates and grey targets, then pair them by identical bbox size.
    - For each template, examine 4-connected components of all non-black pixels inside its bbox.
    - If the largest such component has area strictly greater than half of the template's total non-black area, copy the full template bbox onto the matched grey rectangle.
    - Otherwise replace the matched grey rectangle with a solid red warning rectangle, while leaving all original templates and background unchanged.

    Key insight:
    - Size matching identifies which template belongs to which target, but connectivity quality decides whether that matched template is accepted or rejected.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    templates = _extract_templates(g)
    targets = _extract_targets(g)
    assert len(templates) == len(targets) >= 1

    template_by_size: dict[tuple[int, int], dict] = {}
    for template in templates:
        size = template["size"]
        assert size not in template_by_size, "template sizes must be unique"
        template_by_size[size] = template

    target_by_size: dict[tuple[int, int], dict] = {}
    for target in targets:
        size = target["size"]
        assert size not in target_by_size, "target sizes must be unique"
        target_by_size[size] = target

    assert set(template_by_size) == set(target_by_size), "template/target sizes must match exactly"

    out = g.copy()
    for size, template in template_by_size.items():
        target = target_by_size[size]
        r0, c0, r1, c1 = target["bbox"]
        ok, _largest, _total = _qualifies(template["mask"])
        if ok:
            out[r0 : r1 + 1, c0 : c1 + 1] = template["patch"]
        else:
            out[r0 : r1 + 1, c0 : c1 + 1] = ALERT
    return out


def _checkerboard_fail_mask(h: int, w: int) -> np.ndarray:
    for phase in (random.randrange(2), 1 - random.randrange(2)):
        rr, cc = np.indices((h, w))
        mask = ((rr + cc + phase) % 2 == 0)
        if len(_components(mask, connectivity=8)) == 1:
            ok, largest, total = _qualifies(mask)
            if not ok and largest * 3 <= total:
                return mask
    raise AssertionError("failed to build fragmented template")


def _pass_corner_fragment(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    variant = random.choice(("tl", "tr", "bl", "br"))
    if variant == "tl":
        mask[1:, 1:] = True
        mask[0, 0] = True
    elif variant == "tr":
        mask[1:, : w - 1] = True
        mask[0, w - 1] = True
    elif variant == "bl":
        mask[: h - 1, 1:] = True
        mask[h - 1, 0] = True
    else:
        mask[: h - 1, : w - 1] = True
        mask[h - 1, w - 1] = True
    return mask


def _pass_holed_blob(h: int, w: int) -> np.ndarray:
    mask = np.ones((h, w), dtype=bool)
    hole_count = 1 if min(h, w) == 3 else random.choice((1, 2))
    candidates = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
    random.shuffle(candidates)
    for r, c in candidates[:hole_count]:
        mask[r, c] = False
    return mask


def _pass_notched_blob(h: int, w: int) -> np.ndarray:
    mask = np.ones((h, w), dtype=bool)
    variant = random.choice(("tl", "tr", "bl", "br"))
    if variant == "tl":
        mask[0, 0] = False
    elif variant == "tr":
        mask[0, w - 1] = False
    elif variant == "bl":
        mask[h - 1, 0] = False
    else:
        mask[h - 1, w - 1] = False
    return mask


def _make_template_mask(h: int, w: int, should_pass: bool) -> np.ndarray:
    builders = (
        (_pass_corner_fragment, _pass_holed_blob, _pass_notched_blob)
        if should_pass
        else (_checkerboard_fail_mask,)
    )
    for builder in random.sample(builders, k=len(builders)):
        mask = builder(h, w)
        assert mask.shape == (h, w)
        if not mask[0].any() or not mask[-1].any():
            continue
        if not mask[:, 0].any() or not mask[:, -1].any():
            continue
        if len(_components(mask, connectivity=8)) != 1:
            continue
        ok, largest, total = _qualifies(mask)
        if should_pass and ok and largest * 5 >= total * 3:
            return mask
        if not should_pass and not ok and largest * 3 <= total:
            return mask
    raise AssertionError("failed to build template mask")


def _colorize(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    color_count = random.choice((2, 2, 3))
    colors = random.sample(TEMPLATE_COLORS, color_count)
    patch = np.zeros((h, w), dtype=int)
    rr, cc = np.indices((h, w))
    mode = random.choice(("bands", "diagonal", "random"))
    if mode == "bands":
        values = (rr + 2 * cc + random.randrange(color_count)) % color_count
    elif mode == "diagonal":
        values = (2 * rr + cc + random.randrange(color_count)) % color_count
    else:
        values = np.random.randint(0, color_count, size=(h, w))
    patch[mask] = np.take(colors, values[mask])

    coords = np.argwhere(mask)
    for idx, color in enumerate(colors):
        r, c = coords[idx % len(coords)]
        patch[r, c] = color

    used = {int(v) for v in patch[mask]}
    assert len(used) == color_count
    return patch


def _rects_clear(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
    gap: int,
) -> bool:
    r0, c0, h0, w0 = first
    r1, c1, h1, w1 = second
    return (
        r0 + h0 + gap <= r1
        or r1 + h1 + gap <= r0
        or c0 + w0 + gap <= c1
        or c1 + w1 + gap <= c0
    )


def _place_rectangles(
    grid_shape: tuple[int, int],
    objects: list[dict],
) -> list[tuple[int, int]] | None:
    h, w = grid_shape
    order = sorted(range(len(objects)), key=lambda i: objects[i]["size"][0] * objects[i]["size"][1], reverse=True)

    for _ in range(120):
        placements: dict[int, tuple[int, int]] = {}
        for idx in order:
            oh, ow = objects[idx]["size"]
            candidates = []
            for r in range(h - oh + 1):
                for c in range(w - ow + 1):
                    rect = (r, c, oh, ow)
                    if all(
                        _rects_clear(rect, (*placements[j], *objects[j]["size"]), PLACEMENT_GAP)
                        for j in placements
                    ):
                        candidates.append((r, c))
            if not candidates:
                break
            placements[idx] = random.choice(candidates)
        if len(placements) == len(objects):
            return [placements[i] for i in range(len(objects))]
    return None


def _semantic_signature(grid: np.ndarray) -> tuple[int, int, int]:
    templates = _extract_templates(grid)
    targets = _extract_targets(grid)
    assert len(templates) == len(targets) in OBJECT_COUNTS

    sizes = [template["size"] for template in templates]
    assert len(set(sizes)) == len(sizes)
    assert set(sizes) == {target["size"] for target in targets}

    passed = failed = 0
    for template in templates:
        ok, largest, total = _qualifies(template["mask"])
        assert largest != 0 and total != 0
        passed += int(ok)
        failed += int(not ok)
    assert passed >= 1 and failed >= 1
    return len(templates), passed, failed


def _build_case() -> tuple[np.ndarray, np.ndarray]:
    grid_h = random.randint(*GRID_SIDE_RANGE)
    grid_w = random.randint(*GRID_SIDE_RANGE)
    object_count = random.choice(OBJECT_COUNTS)
    sizes = random.sample(SIZE_OPTIONS, object_count)

    outcome_flags = [True, False]
    while len(outcome_flags) < object_count:
        outcome_flags.append(bool(random.getrandbits(1)))
    random.shuffle(outcome_flags)

    templates = []
    for size, should_pass in zip(sizes, outcome_flags):
        h, w = size
        mask = _make_template_mask(h, w, should_pass=should_pass)
        patch = _colorize(mask)
        templates.append({"size": size, "patch": patch, "passes": should_pass})

    objects: list[dict] = []
    for index, template in enumerate(templates):
        objects.append({"kind": "template", "pair": index, "size": template["size"], "patch": template["patch"]})
        objects.append(
            {
                "kind": "target",
                "pair": index,
                "size": template["size"],
                "patch": np.full(template["size"], GREY, dtype=int),
            }
        )

    placements = _place_rectangles((grid_h, grid_w), objects)
    if placements is None:
        raise RuntimeError("placement failed")

    inp = np.zeros((grid_h, grid_w), dtype=int)
    out = np.zeros((grid_h, grid_w), dtype=int)

    for obj, (r0, c0) in zip(objects, placements):
        h, w = obj["size"]
        if obj["kind"] == "template":
            patch = obj["patch"]
            inp[r0 : r0 + h, c0 : c0 + w] = patch
            out[r0 : r0 + h, c0 : c0 + w] = patch
        else:
            pair = templates[obj["pair"]]
            inp[r0 : r0 + h, c0 : c0 + w] = GREY
            out[r0 : r0 + h, c0 : c0 + w] = pair["patch"] if pair["passes"] else ALERT

    _semantic_signature(inp)
    assert np.array_equal(main(inp), out)
    assert np.any(out == ALERT), "must include the rejection branch"
    assert np.any((inp == GREY) & (out != GREY)), "targets must change"
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        try:
            grid, _ = _build_case()
        except (AssertionError, RuntimeError):
            continue
        return grid
    raise RuntimeError("failed to generate a valid input")
